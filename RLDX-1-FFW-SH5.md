# RLDX-1 파인튜닝 — FFW-SH5 (레퍼런스)

[RLDX-1 Technical Report](https://arxiv.org/abs/2605.03269) (arXiv 2605.03269, 55p)를
우리 데이터·설정에 대조해서 정리한 문서. 대상은
[`learner1119/rlwrld_demo_0826_merge`](https://huggingface.co/datasets/learner1119/rlwrld_demo_0826_merge)
— 81 에피소드 / 33,018 프레임 / 10 fps, `_1539` + `_1652` + `_1748` 병합본.

스모크용이던 `ffw_sh5_0812`(7 ep)와 달리 이건 **실제로 성능을 노리는 첫 학습**이다.
실행 절차 요약은 [CLAUDE.md](CLAUDE.md)에, 발표용 축약본은
[FFW-SH5-0826-briefing.md](FFW-SH5-0826-briefing.md)에 있다.

**이 문서는 `RLDX-tuning.md`를 흡수한 단일 레퍼런스다.** 본문(§0~§16)은 현재
데이터셋과 학습 구성을, [부록 A](#부록-a-운영-지식--ffw_sh5_0812-시절에-얻은-것)는
데이터셋과 무관하게 유효한 운영 지식(모달리티 계약 / 평가 함정 / 배포 / 라이선스)을 담는다.

이 문서의 수치는 대부분 실측이다. 논문에서 인용한 값에는 절(section) 번호를 붙였고,
우리 데이터에서 계산한 값에는 재현용 명령을 같이 적었다.

---

## 0. 세 줄 요약

1. **`RLDX-1-PT`에는 motion / memory / physics 모듈이 아예 없다.** 이 세 기능은
   논문의 2단계(mid-training, 25K steps × batch 1024 = 2,560만 샘플)에서 주입되는데,
   PT 체크포인트는 1단계 산출물이라 해당 텐서가 0개다. 우리 데이터는 3.2만 샘플이라
   mid-training을 대체할 수 없다. → **1차 학습은 세 모듈 모두 끄고 간다.**
2. **`modality.json`의 슬라이스는 맞다.** ABSOLUTE/NON_EEF, horizon 16, head/lift
   제외까지 실측으로 확인했다. 다만 **action 54차원 중 28개가 상수 0**이라 실질 제어
   공간은 26차원이다(§4.2). → **action을 26차원으로 줄인 config를 만들어 검증까지
   마쳤다**(`ffw_sh5_rev1_config_used26.py`, §12 Q2). 1차 학습은 이걸 쓴다.
3. **1차 학습 결과: `checkpoint-10000`이 최적이다** (held-out MSE 0.0238, 9/9 우세).
   30K까지 갔지만 10K 이후로는 held-out이 개선되지 않았다 — 다음 레버는 스텝이 아니라
   데이터다(§17).
4. **fingertip(tactile) 90차원은 지금 상태로는 학습에 못 쓴다.** 90차원 중 실제
   신호가 있는 건 11개뿐이고, 그마저 전체 프레임의 0.69%에서만 0이 아니다.
   양손 5개 센서 중 3개는 33,018 프레임 내내 완전 무응답이다(§9).

---

## 1. 논문에서 우리에게 필요한 부분

### 1.1 아키텍처 (§2)

```
video(4 frames) ─→ [Vision Encoder + Motion Module] ─┐
language ────────→ [RLDX-1-VLM = Qwen3-VL 8B]  ──────┼─→ cognition h_t (64 tokens)
                                                      │        │
                                            [Memory Module] ←──┘  (past cognition queue)
                                                      │
    state s_t, physics p_t, noisy action ─→ [MSAT: flow-matching DiT] ─→ action chunk
```

우리가 알아야 할 숫자들:

| 항목 | 값 | 근거 |
|---|---|---|
| VLM backbone | Qwen3-VL 8B (robot-VQA로 적응 → `RLDX-1-VLM`) | §2.1 |
| action decoder가 쓰는 LLM layer | **18** (8이면 51.1%, 28이면 56.3%로 하락) | Table 2a |
| cognition token 수 | 64 | §2.1 |
| MSAT 구조 | double-stream(C,A) → single-stream(merged). physics 켜면 triple(C,A,P) → double(CA,P) | §2.2 |
| flow-matching | `u_θ(a^τ, τ, c)` 가 `a - ε` 를 예측, Euler 적분 | Eq.(1)(2) |
| video 프레임 오프셋 | `{-6, -4, -2, 0}` (= `video_length 4` × `video_stride 2`) | §4.1 |
| 정규화 | per-dataset **q01/q99 → [-1,1]** | §4.1 |
| 이미지 전처리 | aspect 보존 resize, area ≤ 256² → **프레임당 최대 64 vision token** | §B.1 |

우리 레포는 이걸 구현하고 있다 — `rldx/data/utils.py:253 normalize_values_q99`,
`TrainConfig.video_stride=2`, `backbone_select_layer` 기본 18.

한 가지 차이는 짚고 가야 한다. 논문 §B.1은 Qwen3-VL의 `smart_resize`를 그대로 쓴다고
하는데, 레포의 `resize_preserve_aspect_area_then_crop`
(`rldx/data/augmentations.py:42`)은 **짧은 변을 m배수로 맞춰 리사이즈한 뒤 긴 변을
m배수로 잘라낸다**. 즉 리사이즈만 하는 게 아니라 **크롭이 들어간다**.
정사각형(256×256) 입력에서는 무해하지만 우리처럼 가로로 긴 카메라에서는
가로가 실제로 잘려나간다. §5.2에서 다룬다.

### 1.2 3단계 학습 (§4)

| 단계 | steps | global batch | lr | 스케줄 | 하드웨어 | 무엇을 하는가 |
|---|---|---|---|---|---|---|
| Pre-training | 100K | 8192 | 1e-4 | constant + warmup 5% | 64×H200, 195h | 1.5M 에피소드 멀티임바디먼트 action prior |
| **Mid-training** | 25K | 1024 | 5e-5 | constant + warmup 5% | 64×H200, 15h | **motion / memory / physics 주입** + 임바디먼트 특화 |
| Post-training | 30K | 64~128 | 1e-4 | cosine + warmup 5% | — | task별 적응 (우리가 하려는 것) |

Mid-training의 디테일이 중요하다 (§4.2):
- 새로 추가된 modality 입력마다 **dropout 0.3**
- 앞 **2K step은 alignment warmup** — 사전학습 파라미터 전부 동결, 새 modality 파라미터만 학습
- physics stream은 **출력 가중치를 near-zero로 초기화**
- ALLEX는 in-house:synthetic = 5:5, FR3는 DROID:in-house = 8:2로 섞음

### 1.3 실로봇 post-training 레시피 (§G.2~G.5) — 우리가 베낄 표

| 플랫폼 | 데모 수 | steps | batch | lr | state dropout | horizon | 동결 |
|---|---|---|---|---|---|---|---|
| OpenArm humanoid | 348 (4개 task family 합계) | 30K | 64 | 1e-4 cosine | 0.3 | — | vision encoder + LLM(top-4 제외) |
| ALLEX (task별) | 40 / 90 / 72 / 62 | 30K | 128 | 1e-4 cosine | 0.8 | 40 | 동일 |
| FR3 (task별) | 50~100 | 30K | 64 | 1e-4 cosine | 0.3 | 16 | 동일 |
| ALLEX Pot-to-Cup (G.5) | 62 | 30K | 128 | 1e-4 cosine | 0.4 | 40 | 동일 |

**공통점: 실로봇 task 하나당 데모 40~100개, 30K step, batch 64~128, lr 1e-4 cosine, warmup 5%.**
우리 81 에피소드는 이 범위 안에 정확히 들어간다.

### 1.4 배치 크기는 클수록 좋다 (Table 10)

| batch | LIBERO | RoboCasa Kitchen | GR-1 Tabletop |
|---|---|---|---|
| 64 | 97.4 | 66.9 | 36.8 |
| 256 | 97.8 | 69.6 | 53.2 |
| 1024 | — | **70.6** | **58.7** |

휴머노이드 벤치(GR-1 Tabletop)에서 64 → 1024가 **+21.9점**이다. 단일 팔 벤치(LIBERO)는
거의 차이가 없다. 우리는 양팔 고DoF이므로 GR-1 쪽 경향에 가깝다고 보는 게 맞다.
다만 저 실험은 데이터가 큰 벤치에서 한 것이고, 실로봇 소규모 데이터에서는 논문 자신도
64~128을 썼다. B200 4장이면 256까지 무리 없이 올라간다(§7).

### 1.5 PEFT — backbone을 얼리면 안 된다 (Table 6, §D)

H200 1장, RoboCasa Kitchen, 24 tasks × 50 ep, 3 view × video 4 (12프레임) 기준:

| backbone | action model | 성공률 | 학습 파라미터 | VRAM(bs32) | VRAM(bs1) |
|---|---|---|---|---|---|
| **Full FT (top-4)** | **Full FT** | **62.67%** | 2,376M | 87.1 GiB | 56.8 GiB |
| Frozen | LoRA r=64 | 36.42% | 378M | 27.5 GiB | 23.3 GiB |
| Full FT (top-4) | LoRA r=64 | 62.67% | 1,150M | 76.6 GiB | 37.2 GiB |
| LoRA r=64 | LoRA r=64 | 55.33% | 398M | 35.9 GiB | 23.7 GiB |

- **backbone을 완전히 얼리면 26점을 잃는다.** `tune_top_llm_layers 4`는 반드시 켠 채로 둔다.
- action model만 LoRA로 바꾸면 성능 손실 없이 파라미터가 절반. VRAM이 빠듯할 때의 카드지만
  **B200 4장에는 필요 없다** (§7 실측).

---

## 2. 결정적 발견 — `RLDX-1-PT`에는 기능 모듈이 없다

Hub의 체크포인트 3종을 텐서 단위로 열어봤다.

```bash
pixi run -e rldx-sm100 python -c "
from huggingface_hub import hf_hub_download; import json
idx=json.load(open(hf_hub_download('RLWRLD/RLDX-1-PT','model.safetensors.index.json')))['weight_map']
for p in ['motion','stss','memory','physics']:
    print(p, len([k for k in idx if p in k.lower()]))"
```

| | `RLDX-1-PT` | `RLDX-1-MT-ALLEX` | `RLDX-1-MT-DROID` |
|---|---|---|---|
| 텐서 수 | 722 | 882 | 882 |
| motion(STSS) 텐서 | **0** | 28 | 28 |
| memory 텐서 | **0** | 19 | 19 |
| physics 텐서 | **0** | 21 | 21 |
| `double_blocks` / `single_blocks` | 80 / 64 | 120 / 128 | 120 / 128 |
| `physics_keys` | — | `['torque']` (48) | `['tactile','torque']` (15+7) |
| `action_horizon` | — | 40 | **16** |
| `state_dropout_prob` | 0.0 | 0.3 | 0.3 |

블록 수가 80/64 → 120/128로 늘어난 게 physics stream(triple/double-stream 확장)이다.
즉 **PT 위에서 `--use-physics`를 켜면 MSAT 블록의 P-stream QKV·projection까지
전부 무작위 초기화**되고, 그걸 우리 3.2만 샘플로 처음부터 학습시키는 게 된다.
논문은 같은 일을 **2,560만 샘플(25K × 1024)** 로 했다. 800배 차이다.

`--use-motion`, `--use-memory`도 사정은 같다.

> **결론: 1차 학습은 `--use-motion` / `--use-memory` / `--use-physics` 전부 끈다.**
> 현재 `run_scripts/train/examples/finetune_rldx1_ffw_sh5.sh`가 이미 그 상태이므로
> 스크립트의 기본 구성은 옳다.

기능 모듈을 꼭 쓰고 싶다면 base를 `RLWRLD/RLDX-1-MT-DROID`로 바꾸는 게 유일하게
합리적인 경로다 — 세 모듈이 학습된 상태이고 `action_horizon 16`도 우리와 같다.
다만 임바디먼트(단일 팔 FR3, state 8차원, tactile 15차원)가 우리와 달라서
입출력 projection은 어차피 새로 만들어진다. **1차 학습에서 시도할 일은 아니다.**

### 2.1 그러면 MT 체크포인트를 base로 쓰되 모듈만 끄면 어떤가

기술적으로는 된다. `assembly.py`의 `_OFFLOADABLE_FEATURES`가 정확히 그 경우를 지원한다 —
체크포인트에 memory/motion/physics가 있고 CLI에서 안 켜면 해당 스트림을 빌드에서 빼고
가중치를 버린다(로그로 알려준다). 반대 방향(ckpt에 없고 CLI로 켜기)도 허용된다.

그래도 **`RLDX-1-PT`를 쓰는 게 맞다.** mid-training의 목적 자체가
"embodiment specialization"(§4.2)이라 `MT-ALLEX`는 ALLEX에, `MT-DROID`는 FR3에
특화된 가중치다. PT 카드가 직접 말한다 — *"Use it as your starting point for new
embodiments and tasks."* 우리는 새 임바디먼트다.

---

## 3. 우리 데이터가 논문 기준에서 어디쯤인가

| 항목 | 논문 실로봇 기준 | 우리 (`0826_merge`) | 판정 |
|---|---|---|---|
| task당 데모 수 | 40~100 | **81** | 적정 |
| 프레임 수 | (ALLEX Object-in-Box 90 ep ≈ 3.1만) | **33,018** | 적정 |
| 로봇 시간 | — | 55분 | — |
| 카메라 | ALLEX는 head 스테레오 1종만 | `cam_head` 1종 | 동일 |
| fps | 10~16 | 10 | OK |
| task 종류 | atomic 1개 | **compositional 1개** | ⚠ §6 |
| 임바디먼트 | ALLEX 48-DoF | FFW-SH5 57-DoF 중 54 사용 | 유사 |

프레임 수는 문제없다. 문제는 **task의 성격**이다(§6).

---

## 4. `modality.json` / modality config 검증

### 4.1 슬라이스는 정확하다

`meta/info.json`의 `names` 배열과 대조한 결과 — 재현:

```bash
pixi run -e rldx-sm100 python -c "
import json; i=json.load(open('dataset/rlwrld_demo_0826_merge/meta/info.json'))
n=i['features']['observation.state']['names']
for a,b,l in [(0,7,'left_arm'),(7,14,'right_arm'),(14,34,'left_hand'),(34,54,'right_hand')]:
    print(l, n[a], '...', n[b-1])"
```

| modality key | 슬라이스 | 실제 관절명 | 판정 |
|---|---|---|---|
| `left_arm` | 0:7 | `arm_l_joint1` … `arm_l_joint7` | ✅ |
| `right_arm` | 7:14 | `arm_r_joint1` … `arm_r_joint7` | ✅ |
| `left_hand` | 14:34 | `finger_l_joint1` … `finger_l_joint20` | ✅ |
| `right_hand` | 34:54 | `finger_r_joint1` … `finger_r_joint20` | ✅ |
| (제외) | 54:57 | `head_joint1/2`, `lift_joint` | ✅ 제외가 맞다 |

`action`의 `names`는 `state`와 완전히 동일하다(실측 `==` True). head/lift를 뺀 것도 옳다 —
33,018 프레임 전체에서 `head_joint1` std = 0.00013, `head_joint2` std = 0.00009,
`lift_joint`는 −0.13 고정이다. 녹화 로그 `rilab_episodes.jsonl`의
`fallback_frames: {head_cmd: N, lift_cmd: N}`와도 일치한다 — 이 두 축은 명령이
들어온 적이 없다.

### 4.2 ⚠ action 54차원 중 28개가 상수 0이다

가장 중요한 발견. 손가락 20관절 중 **실제로 명령이 들어가는 건 손당 6개뿐**이다.

| 관절 | state | action | |
|---|---|---|---|
| `finger_*_joint1`~`13` | 움직임 (±0.02~0.3 rad) | **0.000 고정** | 명령 없음 |
| `finger_*_joint14,15,16` | 움직임 | 움직임 (0 ~ 1.29) | 명령 있음 |
| `finger_*_joint17` | 움직임 | **0.000 고정** | 명령 없음 |
| `finger_*_joint18,19,20` | 움직임 | 움직임 | 명령 있음 |

4관절/손가락 배열로 보면 `joint13-16` = 4번 손가락, `joint17-20` = 5번 손가락이다.
즉 **4·5번 손가락의 굽힘 3관절씩만 제어**하고, 벌림(joint13, 17)과 1~3번 손가락은
전부 0이다. **이건 의도된 운용이다** — 조이스틱 버튼 하나로 프리셋이 걸려
4·5번 손가락(사람으로 치면 약지·새끼)만으로 상자를 파지한다. 즉 이 데이터에서
FFW-SH5의 손은 **설계상 2지 그리퍼**다.

- **실질 action 차원 = 팔 14 + 손 12 = 26** (선언은 54)
- `meta/stats.json`에서 action 30/57 차원의 `q01 == q99`이고, 그 중 28개가 학습 대상
  0:54 범위 안에 있다.
- 기능적으로는 **문제없다.** `normalize_values_q99`가 `q01==q99` 차원을 마스킹해서
  0으로 보내고, 역정규화는 상수를 그대로 돌려준다. 모델은 그 차원에서 상수를 뱉게 되고,
  배포 시에도 데이터와 동일하게 0을 명령한다.
- **다만 손실이 희석된다.** flow-matching loss가 54차원 위에서 계산되는데 그 중 28개가
  학습 신호가 없는 상수다. 그리고 나중에 손 전체를 쓰는 데이터를 새로 모으면 이 체크포인트는
  그대로는 못 쓴다.
- 이건 `modality.json`의 오류가 **아니다.** raw 레이아웃을 정확히 반영한 것이다.

> **결정 (2026-08-26): action은 26차원으로 줄인다.** 위 "손실 희석" 때문이다.
> `modality.json`에 좁은 그룹 4개를 **추가**하고(기존 20차원 그룹은 유지 →
> 하위호환), `rldx/configs/data/ffw_sh5_rev1_config_used26.py`가 그걸 쓴다.
> `state`는 54차원 그대로 둔다 — 관측은 공짜고, 명령이 안 가는 손가락도 수동으로
> 움직이므로 접촉 정보를 담을 수 있다. 상세는 **§12 Q2**.

### 4.3 ABSOLUTE / NON_EEF는 맞다

```
mean|action[t] - state[t+0]| = 0.02600
mean|action[t] - state[t+1]| = 0.02244
mean|action[t] - state[t+2]| = 0.02040   ← 최소
mean|action[t] - state[t+5]| = 0.02868
mean|action[t] - action[t-1]| = 0.00482
mean|action[t]| = 0.1565                  ← delta라면 ~0.001
```

action의 절대값 크기가 state와 같은 스케일이고, state를 약 2 step(0.2초) 선행한다.
전형적인 텔레옵 지연이다. `ActionRepresentation.ABSOLUTE`, `ActionType.NON_EEF` 유지.

### 4.4 horizon 16 유지

`delta_indices=list(range(16))` = `TrainConfig.action_horizon` 16. 사전학습 혼합이 16이고,
`MT-DROID`도 16이다. ALLEX만 40인데 그건 mid-training부터 40으로 간 케이스다.
**PT에서 출발하는 우리는 16이 맞다.** 바꾸면 사전학습된 action encoder/decoder와 어긋난다.

### 4.5 `annotation.human.task_description` → `task_index` 매핑

`modality.json`의 `"annotation": {"human.task_description": {"original_key": "task_index"}}`는
정상 동작한다. 로더가 `task_index` → `tasks.jsonl` 조회로 실제 문장을 넣는다. 확인:

```
language sample: Receive the package from the conveyor, check the shipping la...
```

`ffw_sh5_0812`의 `"1611"` 같은 플레이스홀더 문제는 이 데이터셋엔 없다.

---

## 5. 데이터 전처리 — 실제로 해야 하는 것

### 5.1 이미 끝난 것

| 항목 | 상태 |
|---|---|
| `meta/modality.json` | ✅ 작성·검증 완료 (§4) |
| `meta/stats.json` / `relative_stats.json` | ✅ 생성 완료 (`{}` — 전부 ABSOLUTE이므로 정상) |
| LeRobot v2.1 정합성 | ✅ 81 ep 전수 감사 통과 (dead episode·NaN·영상 desync·인덱스 오류 0건) |
| 로더 통과 | ✅ `LeRobotEpisodeLoader`로 81 ep 로드, 차원 7/7/20/20 확인 |

### 5.2 ⚠ 이미지가 논문보다 작게 들어간다 — 조정 권장

`cam_head`는 376×672 (aspect 1.787)인데, 기본값 `image_max_area=65536`(=256²)에서는:

| `--image-max-area` | resize | crop | vision token/frame | 가로 손실 |
|---|---|---|---|---|
| **65536 (기본)** | 160×285 | **160×256** | **40** | **10.2%** |
| 67584 | 192×343 | 192×320 | 60 | 6.7% |
| 98304 | 224×400 | 224×384 | 84 | 4.0% |
| 131072 | 256×457 | 256×448 | 112 | 2.0% |

기본값으로 가면 논문의 64 token 예산 중 **40개만** 쓰고, 게다가 가로를 10% 센터크롭해서
버린다. 컨베이어에서 상자가 **옆에서 들어오는** 태스크라 가장자리 손실이 뼈아플 수 있다.

> **권장: `--image-max-area 98304`** (224×384, 84 token). B200 VRAM에 여유가 크므로
> (§7) 토큰을 2배 써도 부담이 없고, 가로 손실이 10.2% → 4.0%로 줄어든다.
>
> 보수적으로 가려면 `67584`(60 token, 논문의 64에 근접)도 합리적이다.
> **기본값 65536은 우리 종횡비에서 최악의 조합**이라는 점만 기억하면 된다.

재현:

```bash
pixi run -e rldx-sm100 python -c "
import sys; sys.path.insert(0,'.')
from rldx.data.augmentations import resize_preserve_aspect_area_then_crop as R
for ma in [65536,67584,98304,131072]:
    rs,cr=R(376,672,max_area=ma,m=32); print(ma,rs,cr,(cr[0]//32)*(cr[1]//32))"
```

### 5.3 ⚠ 트레이너에 validation이 없다 — held-out을 직접 만들어야 한다

`rldx/data/dataset/factory.py:68`:

```python
assert self.config.training.eval_strategy == "no", \
    "Sharded dataset does not support evaluation sets"
```

`eval_dataset`은 항상 `None`이다. **학습 중 검증 곡선이 안 나온다.** train loss만 보고
과적합을 판단해야 하는데, 100 epoch를 넘게 도는 구성(§8)에서 그건 위험하다.

> **권장: 학습 전에 9개 에피소드를 떼어 별도 데이터셋으로 만들고, 체크포인트마다
> `open_loop_eval`을 돌려 MSE/MAE를 기록한다.**
>
> 세션 편향(§5.5)이 있으므로 세 세션에서 **각 3개씩** 뽑는다:
> `1539` → merged ep **3, 10, 17** / `1652` → **26, 35, 44** / `1748` → **56, 65, 74**.

이 조합의 실제 통계 (검증함):

| | 에피소드 | 프레임 | 길이 min/med/max |
|---|---|---|---|
| held-out | 9 | 3,357 (**10.2%**) | 244 / 344 / 557 |
| 학습 | 72 | 29,661 | 215 / — / 916 |

전체 데이터의 길이 중앙값이 370이고 held-out 중앙값이 344라 편향이 크지 않다.
tactile이 완전히 죽은 ep 42(§9)는 held-out에 들어가 있지 않다.

### 5.4 코덱 혼재 — 그대로 둬도 된다

merged의 ep 0~20은 av1, 21~80은 mpeg4다. `info.json`은 다수인 `mpeg4`를 선언하고 있어
ep 0~20에 대해서는 그 필드가 사실과 다르다. 하지만 **학습 경로는 이 필드를 읽지 않는다** —
`video_backend="torchcodec"`이 컨테이너에서 직접 판별한다.
81개 전부 torchcodec 디코딩·정지프레임·블랙프레임 검사를 통과했다.
재인코딩은 2세대 손실만 만든다. **조치 불필요.**

### 5.5 ⚠ 세션 편향 — 알고는 있어야 한다

| 세션 | ep | 평균 길이 | 전체 프레임 비중 |
|---|---|---|---|
| `1539` | 21 | 320 | 20.3% |
| `1652` | 30 | 380 | 34.6% |
| `1748` | 30 | 497 | **45.1%** |

같은 task인데 세션이 갈수록 시연이 길어진다(작업자 피로 / 난이도 조정 / 상자 상태 변화).
길이 분포는 215~916으로 연속적이고 이봉(bimodal)이 아니라서, "상자를 뒤집은 회차 vs
아닌 회차"를 길이만으로는 못 가른다.

프레임 단위로 샘플링하므로 `1748`의 느린 시연이 학습 분포의 45%를 차지한다.
당장 조치할 필요는 없지만, **평가는 반드시 세 세션에 걸쳐서** 해야 한다(§5.3).

---

## 6. 데이터 양 — 81개면 충분한가

**프레임 수는 충분하다. 태스크 성격이 위험 요소다.**

논문의 실로봇 per-task 데모 수는 40~100이고 우리는 81이다. 프레임도 ALLEX
Object-in-Box(90 ep ≈ 3.1만 프레임)와 거의 같은 3.3만이다.

문제는 우리 task instruction이 이것이라는 점이다:

> "Receive the package from the conveyor, **check the shipping label**,
> **flip the box face up if necessary**, and forward it to the next station."

논문 기준으로 이건 atomic task가 아니라 **composite task**다. RoboCasa365에서
RLDX-1의 성적이 이 구분을 잘 보여준다 (Table 1b):

| | Atomic-Seen | Composite-Seen | Composite-Unseen |
|---|---|---|---|
| RLDX-1 | **67.3** | **19.0** | **5.6** |
| GR00T N1.6 | 61.1 | 12.6 | 2.6 |

composite로 가면 SOTA도 19%다. 게다가 우리 task에는 `if necessary`라는 **조건 분기**가
들어 있다 — 라벨을 보고 뒤집을지 말지 결정해야 한다. 이건 논문이 memory module로 푸는
종류의 문제인데, 우리는 그 모듈을 못 쓴다(§2).

> **판단: 81개로 "컨베이어에서 상자를 받아 다음 공정으로 넘기는" 주 동작은 학습될 것으로
> 본다. "라벨 확인 후 조건부 뒤집기"는 81개로는 부족할 가능성이 높다.**

### 6.1 분기 비율은 데이터에서 자동으로 알아낼 수 없다 (시도해 봄)

"뒤집은 회차"를 자동 판별해 보려고 두 가지 대리 지표를 재봤다. 둘 다 실패했다:

| 지표 | 결과 | 해석 |
|---|---|---|
| 에피소드 길이 | 215~916 연속 분포, 이봉 아님 | 길이로는 못 가른다 |
| 양손 파지 여부 | **81/81 전부 양손 사용** | 변별력 0 |
| 파지 이벤트 횟수 vs 길이 상관 | **0.01** | 재파지 횟수도 길이와 무관 |

즉 긴 에피소드가 뒤집기를 한 회차라는 근거가 없다. 세션별 길이 증가(§5.5)는
분기가 아니라 세션 효과로 보는 게 맞다.

> **결론: 분기 비율은 사람이 라벨링하거나 영상을 확인해야 한다.**
> 재현: 위 표의 계산은 commanded 차원(팔 0:14, 손 굽힘 27–33 / 47–53)만 써서
> 파지 임계 0.15로 상승 에지를 센 것이다.

### 6.2 ✅ 분기 신호는 화면에 있고, 학습 해상도에서도 살아남는다 (영상 확인)

§6.1에서 "관절 통계로는 못 가른다"고 했는데, **그럼 화면에는 있는가**를 실제로 확인했다.
`run_scripts/data/peek_frames.py`로 세 회차의 프레임을 원본과 학습 해상도 양쪽으로
뽑아 봤다 (결과: `outputs/frames/`).

작업 내용부터 정정한다. 이건 단순히 "상자를 뒤집는" 게 아니라
**택배 송장(흰 라벨)이 위를 향하도록 자세를 맞추는** 작업이다. 즉 분기를 결정하는
잠재변수는 "상자가 어떻게 놓여 들어왔는가 = 라벨이 어느 면에 있는가"다.

세 회차가 서로 다른 초기 상태를 보여준다:

| merged ep | 초기 라벨 위치 | 필요한 조작 |
|---|---|---|
| 5 | 앞쪽 수직면 | 세워서 위로 오도록 재배치 |
| 30 | **보이지 않음** (아랫면/뒷면) | 크게 뒤집어야 함 |
| 60 | 이미 윗면 | 뒤집기 불필요, 이동만 |

**중요한 결론 — 학습 입력에서 이 정보가 살아 있다.** 흰 송장 라벨은 갈색 상자 위의
큰 고대비 사각형이라, 376×672 → 224×384로 줄여도 **뚜렷하게 남는다.**
글자는 못 읽지만 **"어느 면에 붙어 있는가"는 읽을 필요가 없다.**

이게 왜 중요한가: §6이 걱정한 "같은 관측에 정답이 여러 개" 문제(§12 Q6-b의 3번)는
**관측이 실제로 같을 때만** 성립한다. 라벨 위치가 화면에 보인다면 관측은 서로 다르고,
정책이 원리적으로 분기를 학습할 수 있다. **입력에 정보가 없어서 못 배우는 상황은 아니다.**

남는 위험은 (a) 오차 누적과 (b) "라벨을 이미 확인했는가"라는 진행 상태 기억이지,
관측 부족이 아니다.

### 6.3 그래서 라벨링은 이렇게 한다 (하게 된다면)

§6.1이 제안한 "뒤집었는가 예/아니오"는 **부족하다.** 방향이 중요하기 때문이다.
최소한 이 정도는 있어야 분기 균형을 판단할 수 있다:

```json
{"episode_index": 5,
 "label_face_initial": "front",     // front | top | hidden | side
 "label_face_final":   "top",
 "reorientations": 1}
```

`label_face_initial`의 분포가 곧 분기 분포다. `top`(조작 불필요)이 대부분이면
어려운 케이스가 학습되지 않는다.

에피소드에 상자가 여러 개면(ep70 같은 경우, §14 Q9) **상자 단위로** 세야 한다:

```json
{"episode_index": 70, "n_boxes": 2,
 "boxes": [{"label_face_initial": "top",    "reorientations": 0},
           {"label_face_initial": "hidden", "reorientations": 1}]}
```

**자동화 시 주의 — 프레임을 균일 간격으로 뽑으면 안 된다.** 분기를 결정하는 정보는
**에피소드 앞부분(상자가 놓인 직후)** 에 있다. 앞쪽을 촘촘히, 마지막 프레임 하나로
목표 상태를 확인하는 식이 맞다.

### 6.4 `peek_frames.py` — 무엇을 하고 무엇을 하지 않는가

> **⚠ 이 스크립트는 아무것도 인식하지 않는다. VLM을 호출하지 않는다.**
> mp4를 디코딩해서 프레임 몇 장을 PNG 한 장으로 타일링해 저장할 뿐이다.
> 검출기도 없고 모델 호출도 없다.
>
> §6.2에서 "ep30은 라벨이 안 보이고 ep70은 상자가 둘"이라고 판단한 주체는
> **스크립트가 아니라 그 PNG를 직접 본 사람(나)** 이다. 자동 판별 기능이 있다고
> 오해하면 안 된다. 자동화하려면 이 PNG를 **VLM에 입력으로 넣는 단계를 따로**
> 붙여야 한다 — 그 부분은 아직 만들지 않았다.

**하는 일**은 셋이다:

1. torchcodec으로 지정 에피소드의 `cam_head` mp4에서 프레임을 뽑는다.
2. 두 벌을 만든다 — `ep0NN_raw.png`(원본 376×672)와
   `ep0NN_train.png`(**학습 시 리사이즈+크롭을 그대로 재현한 것**, 기본 224×384).
   둘을 비교하면 *내가 보려는 디테일이 다운스케일을 견디는가*를 바로 알 수 있다.
   §6.2의 "송장이 학습 해상도에서도 보인다"는 결론이 이 비교에서 나왔다.
3. 각 타일 아래에 **프레임 번호와 초**를 찍고, 콘솔에 샘플 시각과 **간격**을 출력한다.

**사용법:**

```bash
# 기본: 앞쪽에 몰린 8장 (분기 판정용)
pixi run -e rldx-sm100 python run_scripts/data/peek_frames.py outputs/frames 5,30,70

# 균일 타임라인: 2초마다 한 장
pixi run -e rldx-sm100 python run_scripts/data/peek_frames.py outputs/frames 30 --every-sec 2

# 다른 이미지 예산으로 학습할 때의 화질 확인
pixi run -e rldx-sm100 python run_scripts/data/peek_frames.py outputs/frames 30 --max-area 65536
```

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `out_dir` | (필수) | PNG 저장 경로. `outputs/` 아래를 쓸 것 — `/tmp`는 세션마다 날아간다 |
| `episodes` | (필수) | merged 에피소드 번호, 콤마 구분 (`5,30,70`) |
| `--dataset` | `dataset/rlwrld_demo_0826_merge` | 데이터셋 루트 |
| `--max-area` | `98304` | 학습 `--image-max-area`와 같은 값을 주면 `_train.png`가 실제 입력과 일치 |
| `--every-sec` | (없음) | 주면 **균일 간격** 샘플링으로 바뀐다 |
| `--video-key` | `observation.images.cam_head` | 다른 카메라를 볼 때 |

**출력 예 (`--every-sec` 없이):**

```
ep  30: 359 frames = 35.9s @ 10fps | 8 samples
        times(s): 0.0, 2.9, 5.9, 8.9, 11.9, 17.9, 23.9, 35.8
        gaps(s) : 2.9, 3.0, 3.0, 3.0, 6.0, 6.0, 11.9
```

**간격이 0.8초가 아닌 이유** — 8장은 "0.8초씩"이 아니라 **에피소드 전체를 비균등
비율(0, 1/12, 1/6, 1/4, 1/3, 1/2, 2/3, 1)로 나눈** 것이다. 그래서 앞쪽은 3초,
뒤쪽은 12초 간격이 된다. 에피소드 길이에 따라 간격도 달라진다(35.9초짜리는 위처럼,
54초짜리 ep70은 0/4.5/9/13.5/18/27/36/53.9초).

0.8초 간격이 필요하면 `--every-sec 0.8`을 주면 되는데, 35.9초 에피소드는
**45장**이 나온다. 분기 판정에는 과하고, 오히려 §6.3에서 말한 "앞쪽 촘촘히"를
쓰는 편이 장수 대비 정보량이 낫다.

추가 수집을 한다면 우선순위:

1. **뒤집기가 필요한 회차 / 아닌 회차의 비율을 세고 균형을 맞춘다.** 지금 그 비율을
   모른다는 게 가장 큰 공백이고, §6.1대로 데이터만으로는 알 수 없다.
   소수 분기가 20% 미만이면 그 분기는 학습되지 않는다.
   → `meta/episodes.jsonl`에 회차별 라벨(예: `"flipped": true/false`)을 붙이는 것부터.
2. 논문의 adaptive data collection(§4.3): 1차 학습 → 실패 조건 관찰 → **실패 케이스에
   집중해서 refinement 데이터 수집**. 무작정 늘리는 것보다 효율이 높다고 보고한다.
3. 상자 위치·자세의 variance를 넓힌다 (논문의 consistency factor / variance factor 구분).

목표치를 굳이 잡자면 **분기당 40~60개, 총 120~160 에피소드**가 논문 레시피와 맞는 규모다.

---

## 7. 하이퍼파라미터 — B200 4장 기준

### 7.1 실측

추측이 아니라 실제로 돌려서 쟀다. `rlwrld_demo_0826_merge` 81 ep + `RLDX-1-PT` +
`--video-length 4`, 4×B200, `torchrun --nproc_per_node=4`, 각 구성 30 step
(처리량은 warmup 이후 정상 구간, VRAM은 3초 간격 샘플링의 피크):

| global batch | GPU당 micro-batch | `image_max_area` | vision token/frame | **처리량** | **peak VRAM/GPU** | 30K step 소요 |
|---|---|---|---|---|---|---|
| 64 | 16 | 65536 | 40 | 3.15 it/s (202 sample/s) | **37.4 GiB** | 2.6 h |
| **128** | 32 | 65536 | 40 | 2.06 it/s (264 sample/s) | **47.4 GiB** | 4.0 h |
| **128** | 32 | **98304** | **84** | 1.60 it/s (205 sample/s) | **54.3 GiB** | 5.2 h ⚠ |
| 256 | 64 | 65536 | 40 | 1.33 it/s (340 sample/s) | **66.9 GiB** | 6.3 h |

읽어낼 것:

- **VRAM이 전혀 문제가 아니다.** B200은 183 GB인데 가장 무거운 구성이 67 GiB다.
  논문 Table 6의 H200 기준(batch 32에 87 GiB)보다 가벼운데, 우리는 카메라가
  1대이기 때문이다(논문 ablation은 3 view × video 4 = 12프레임).
  **LoRA를 쓸 이유가 없다.** batch 512도 물리적으로는 들어간다.
- **배치를 키울수록 샘플 처리량이 올라간다**: 202 → 264 → 340 sample/s.
  작은 배치는 B200을 못 채운다. 즉 batch 128은 64보다 30% 빠르고,
  256은 64보다 68% 빠르다 — **큰 배치가 시간을 더 쓰는 게 아니다.**
  batch 256을 안 고르는 이유는 속도나 메모리가 아니라 epoch 수(§8)다.
- 이미지를 40 → 84 token으로 키우는 비용은 **처리량 −22%, VRAM +7 GiB**다.
  30K step 기준 4.0 h → 5.2 h (단일 측정, §7.1.1 참고). §5.2의 화질 이득을 생각하면 **살 만한 값**이다.
- 30 step만에 loss가 0.35 → 0.27로 떨어졌다. 파이프라인이 merged 데이터셋에서
  끝까지 동작한다는 뜻이다(체크포인트 기록까지 확인).

### 7.1.1 ⚠ 처리량은 실행마다 크게 흔들린다 (I/O 변동)

같은 구성(batch 128 + `image_max_area 98304`)을 시간대를 달리해 네 번 재보니
**1.07 ~ 1.60 it/s**로 흩어졌다. 모델·설정이 동일한데 50% 차이가 난다.

| 실행 | modality config | 측정 |
|---|---|---|
| 1차 프로브 | wholebody (54차원) | 1.60 it/s |
| 스모크 | used26 (26차원) | 1.07 it/s |
| **A/B 연속 실행** | wholebody (54차원) | **1.28 it/s** |
| **A/B 연속 실행** | used26 (26차원) | **1.41 it/s** |

같은 config가 1.07~1.60 사이에 걸치므로 **차이의 원인은 config가 아니라 I/O다** —
`/NHNHOME`은 공유 Lustre이고 mp4 디코딩이 페이지 캐시 상태와 다른 사용자 부하를 탄다.

같은 세션에서 연속으로 돌린 A/B(아래 두 줄)만이 공정한 비교인데, 거기서는
**26차원이 오히려 10% 빠르다**(1.41 vs 1.28). action 그룹이 4개 → 6개로 늘어
프로세서 오버헤드가 커질까 걱정했지만 그 효과는 관측되지 않는다.

> **계획용 수치: 30K step에 5~8시간.** 5.2시간은 가장 좋았을 때의 값이지
> 기대값이 아니다. 밤에 걸어두면 아침에 끝나 있는 규모라고 보면 된다.

`loss` 값도 두 config 사이에 비교하면 안 된다 (26차원 0.12~0.14 vs 54차원 0.26).
상수 28차원의 flow-matching 타깃이 `−ε`(단위분산 노이즈)이라 학습 초기 loss를
크게 잡아먹을 뿐, 수렴하면 사라지는 항이다. **차원 수가 다르면 loss 절대값은
서로 다른 척도다.**

### 7.2 권장 설정

그대로 붙여넣어 쓸 수 있는 형태:

```bash
cd /NHNHOME/doyoung/RLDX-1
export PATH=/NHNHOME/doyoung/.pixi/bin:$PATH
export HF_HOME=/NHNHOME/doyoung/.cache/huggingface
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_PROJECT=rldx-finetune

pixi run -e rldx-sm100 torchrun --nproc_per_node=4 --master_port=$(shuf -i 20000-30000 -n 1) \
  rldx/experiment/launch_train.py \
  --base-model-path RLWRLD/RLDX-1-PT \
  --video-length 4 \
  --n-cog-tokens 64 \
  --dataset-path "$PWD/dataset/rlwrld_demo_0826_merge_train" \
  --embodiment-tag GENERAL_EMBODIMENT \
  --modality-config-path "$PWD/rldx/configs/data/ffw_sh5_rev1_config_used26.py" \
  --output-dir "$PWD/ckpt/rldx1/finetuned/rldx1_ft_ffw_sh5_0826" \
  --experiment-name rldx1_ft_ffw_sh5_0826 \
  --num-gpus 4 \
  --global-batch-size 128 \
  --learning-rate 1e-4 \
  --lr-scheduler-type cosine \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --max-grad-norm 1.0 \
  --tune-top-llm-layers 4 \
  --state-dropout-prob 0.3 \
  --image-max-area 98304 \
  --max-steps 30000 \
  --save-steps 2500 \
  --save-total-limit 6 \
  --dataloader-num-workers 8 \
  --use-wandb --wandb-project "$WANDB_PROJECT"
```

주의할 점 셋:

- `--base-model-path`는 `RLDX-1-PT`(video)이고 `--video-length 4`와 짝이다.
  `RLDX-1-PT-IMG`를 쓰려면 `--video-length 1`로 같이 바꿔야 한다 (CLAUDE.md).
- `--output-dir`에 체크포인트가 남아 있으면 `experiment.py`가
  `resume_from_checkpoint=True`로 **조용히 이어서 학습한다**. 새로 시작하려면
  `--experiment-name`과 디렉터리를 새로 준다 (부록 A.3).
- ⚠ **체크포인트 하나가 61 GB다** (실측. 8B 가중치 + 옵티마이저 상태).
  `--save-total-limit`을 12로 두면 **732 GB**를 먹는다. `/NHNHOME`에 51 TB가
  남아 있어 물리적으로는 들어가지만 `ckpt/`가 이미 443 GB다.
  **6으로 두고, 굴러떨어지기 전에 평가하고 필요 없는 건 지우는 운영**을 권한다.
  더 가볍게 가려면 `--save-steps 5000 --save-total-limit 6` (= 366 GB, 6개 전부 보존).

각 값의 근거:

| 파라미터 | 값 | 왜 |
|---|---|---|
| `global_batch_size` | **128** | 논문 실로봇 레시피(64~128)의 상단. Table 10에서 휴머노이드는 배치가 클수록 유리하고, VRAM(47~54 GiB)도 처리량도 128 쪽이 64보다 낫다. 256은 메모리·속도상 문제가 없는데도 안 고르는 이유가 하나뿐이다 — 30K step이면 269 epoch가 되어 과적합 위험이 더 커진다(§8). |
| `learning_rate` | **1e-4** | 논문의 모든 post-training이 1e-4. mid-training만 5e-5. |
| `lr_scheduler_type` | **cosine** | §G.2~G.5 전부 cosine. (pre-training만 constant) |
| `warmup_ratio` | **0.05** | 논문 전 단계 공통. |
| `state_dropout_prob` | **0.3** | OpenArm·FR3 값. ALLEX는 0.8까지 쓰는데 그건 horizon 40 + torque가 있는 구성이다. 우리는 0.3에서 시작. |

> **`state_dropout_prob` 주의:** `RLDX-1-PT`의 `config.json`에는 `state_dropout_prob: 0.0`으로
> 박혀 있지만, `assembly.py:166`이 CLI 값을 **무조건 덮어쓴다**
> (`run_config.model.state_dropout_prob = cli.state_dropout_prob`).
> 즉 아무것도 안 주면 `TrainConfig` 기본값 **0.3**이 적용된다 — 체크포인트의 0.0이 아니다.
> 헷갈리기 쉬우니 스크립트에 명시해 두는 편이 낫다.
| `tune_llm` / `tune_visual` | **False** | 논문: vision encoder와 LLM은 동결하되 **top-4 layer만** 학습. `tune_top_llm_layers=4`가 그 역할. |
| `action_model_use_lora` | **False** | Table 6에서 VRAM 절감용 카드인데 B200에선 불필요. |
| `use_motion/memory/physics` | **False** | §2. |
| `--num-gpus 4` / `--nproc_per_node=4` | 일치 필수 | CLAUDE.md |

### 7.3 건드리지 말 것

- `--video-stride`: 기본 2. 논문의 `{-6,-4,-2,0}`이 여기서 나온다.
- `--action-horizon`: 16. §4.4.
- `--backbone-select-layer`: 기본 18. Table 2a에서 8/28은 각각 −9.8 / −4.6점.
- `--episode-sampling-rate`: 기본 0.1. 이름이 오해를 부르는데 **데이터를 버리는 게 아니다** —
  에피소드를 10개 서브시퀀스로 쪼개 shard에 분산시켜 shard 내 다양성을 확보하는 장치다
  (`sharded_single_step_dataset.py:220`). 전 스텝이 다 쓰인다.

---

## 8. 몇 step까지 학습할 것인가

유효 학습 샘플 수 = `Σ(len - action_horizon + 1)`
(`sharded_single_step_dataset.py:260 get_effective_episode_length`).

- 전체 81 ep 기준: 33,018 − 81×15 = **31,803**
- held-out 9개를 뺀 72 ep 기준: 29,661 − 72×15 = **28,581** ← 실제 학습에 쓰이는 값

| batch | steps | epoch (72 ep 기준) | 비고 |
|---|---|---|---|
| 64 | 10,000 | 22.4 | 현재 스크립트 설정 — 논문 대비 짧다 |
| 64 | 30,000 | 67.2 | 논문 OpenArm과 동일 레시피 |
| **128** | **30,000** | **134.4** | 논문 ALLEX/FR3와 동일 레시피 |
| 256 | 30,000 | 268.8 | 과적합 위험 |

> **권장: batch 128 × 30,000 step.** 논문의 실로봇 post-training이 예외 없이 30K이고,
> 데모 수(62~90)도 우리와 같은 규모에서 그렇게 했다. epoch 수(120)가 커 보이지만
> 논문 ALLEX Pot-to-Cup(62 데모, batch 128, 30K)도 비슷한 영역이다.

**단, 30K를 끝까지 다 쓰라는 뜻은 아니다.** validation이 없으므로(§5.3):

- `--save-steps 2500` → 12개 체크포인트
- 각 체크포인트에서 held-out 9 에피소드로 `open_loop_eval` MSE/MAE를 기록
- **MSE가 바닥을 치고 올라가기 시작하는 지점**을 실제 배포 후보로 삼는다

소규모 실로봇 데이터에서는 보통 10K~20K 사이에서 held-out MSE가 최저를 찍고
그 뒤로는 train loss만 내려간다. 30K는 상한이지 목표가 아니다.
권장 구성에서 30K 완주가 5~8시간이므로(§7.1.1), 끝까지 돌려 놓고
사후에 최적 체크포인트를 고르는 편이 비용상 합리적이다.

평가 명령 (CLAUDE.md의 주의사항 그대로 — `_verbose` 래퍼를 쓰고 플롯은 `outputs/`에):

```bash
pixi run -e rldx-sm100 python run_scripts/eval/open_loop_eval_verbose.py \
    --model-path ckpt/rldx1/finetuned/<name>/<name>/checkpoint-10000 \
    --dataset-path dataset/rlwrld_demo_0826_holdout \
    --embodiment-tag GENERAL_EMBODIMENT \
    --traj-ids 0 1 2 3 4 5 6 7 8 --action-horizon 16 --steps 128 \
    --save-plot-path outputs/eval/holdout_10k.png
```

---

## 9. fingertip(tactile) 센서를 쓸 수 있는가

**결론: 지금 데이터로는 못 쓴다.** 이유가 세 겹이다.

### 9.1 모델 쪽 — physics stream 자체가 없다

§2. `RLDX-1-PT`의 physics 텐서는 0개다. 켜는 순간 P-stream 전체가 무작위 초기화되고,
논문이 2,560만 샘플로 학습시킨 것을 3.2만 샘플로 학습시키게 된다.

### 9.2 데이터 쪽 — 90차원 중 신호가 있는 건 11개뿐

```bash
pixi run -e rldx-sm100 python -c "
import json,numpy as np
st=json.load(open('dataset/rlwrld_demo_0826_merge/meta/stats.json'))['observation.tactile']
q01,q99=np.array(st['q01']),np.array(st['q99'])
print('degenerate dims (q01==q99):', int(np.isclose(q99,q01).sum()), '/90')"
```

| 지표 | 값 |
|---|---|
| `q01 == q99`인 차원 (정규화하면 항상 0 → 정보량 0) | **79 / 90** |
| 살아있는 차원 | 2, 27, 28, 38, 72, 73, 74, 77, 81, 82, 83 |
| 전체 (프레임×차원) 중 0이 아닌 비율 | **0.69%** |
| 하나라도 0이 아닌 프레임 | 38.4% |
| 3개 이상 0이 아닌 프레임 | **5.0%** |

게다가 **dim 2는 신호가 아니라 노이즈**다. `1539`에서는 33,018 프레임 내내 1~3 사이
값이 박혀 있고(항상 nonzero), `1652`에서는 항상 0, `1748`에서는 27.7%만 0이 아니고
값은 0 아니면 1이다. 세션마다 다른 상수 오프셋 — 접촉 신호가 아니다.

### 9.3 손이 2지로 운용되기 때문이다 (하드웨어 결함이 아님 — 2026-08-27 정정)

> **정정.** 이 절은 원래 "양손 5개 센서 중 3개가 전 구간 무응답"이라고 적고
> 배선 문제를 의심했다. **틀렸다.** 두 가지가 겹쳐 잘못 봤다:
> (1) 손가락 1~3을 안 쓰는 것은 **의도된 운용 방식**이다 — 조이스틱 버튼 하나로
> 프리셋이 걸려 **4·5번 손가락(사람으로 치면 약지·새끼)만으로 상자를 파지**한다.
> (2) 내가 "0이 아닌 프레임 비율 > 0.5%"로 걸러서 **드문 접촉 이벤트를 통째로
> 놓쳤다.**

전체 33,018 프레임을 다시 세면 이렇다:

| 센서 | 최대값 | 0 아닌 비율 | 접촉 이벤트 (>10) |
|---|---|---|---|
| 왼손 1 | 3 | 3.65% | **0** |
| 왼손 2 | **231** | 0.055% | **34** |
| 왼손 3 | **233** | 0.010% | **8** |
| 왼손 4 (파지) | 255 | 0.451% | 491 |
| 왼손 5 (파지) | 255 | 0.346% | 450 |
| 오른손 1 | 5 | 0.001% | **0** |
| 오른손 2 | **70** | 0.065% | **39** |
| 오른손 3 | **139** | 0.055% | **34** |
| 오른손 4 (파지) | 233 | 1.359% | 907 |
| 오른손 5 (파지) | 173 | 0.930% | 732 |

**센서 2·3은 멀쩡히 동작한다.** 값 크기(70~233)가 파지 손가락과 같은 수준이고
접촉 이벤트도 수십 번씩 잡힌다 — 안 쓰는 손가락이 상자에 가끔 스치는 것이다.
사실상 조용한 건 **양손 1번 센서뿐**이고(최대 3·5, 접촉 이벤트 0), 그 손가락이
아예 물체에 닿지 않는 자세라고 보면 일관된다.

즉 **고칠 하드웨어가 없다.** tactile이 희소한 것은 결함이 아니라
**2지 파지라는 설계의 결과**이고, 데이터를 다시 모아도 같은 모양이 나온다.

이건 §4.2의 발견과 정확히 맞아떨어진다 — 명령이 들어가는 손가락도 4·5번,
접촉이 집중되는 센서도 4·5번이다. **둘 다 같은 설계에서 나온 결과다.**

### 9.4 코드 쪽 — all-zero 샘플은 조용히 버려진다

`rldx/data/state_action/state_action_processor.py:565`:

```python
if modality in ("torque", "tactile") and np.all(values == 0):
    continue   # 센서 부재로 간주하고 이 modality를 통째로 스킵
```

우리 데이터는 프레임의 61.6%가 전 차원 0이다. 즉 physics를 켜도 **학습 샘플의 60% 이상에서
physics stream이 마스킹**되고, 나머지 40%도 11개 차원의 희소 신호만 들어간다.
`--allow-missing-physics` 없이 켜면 그 샘플들에서 문제가 생길 수 있다.

### 9.5 그래서 언제 쓸 수 있나

논문에서 tactile이 효과를 본 태스크는 **Plug Insertion**(20.8/16.7 → 33.3)과
**Egg Pick-and-Place**(45.8/37.5 → 61.1)다. 둘 다 "시각으로는 접촉 여부를 알 수 없는"
정밀 접촉 태스크다. 상자를 집어 옮기는 우리 태스크는 그 범주가 아니다 —
tactile이 있어도 이득이 크지 않을 가능성이 높다.

쓰고 싶다면 순서는 이렇다:

1. **90차원 중 파지에 실제로 쓰이는 차원만 남긴다** — 왼손 27–44 / 오른손 72–89
   = **36차원**(4·5번 손가락). 이 36차원에 접촉 이벤트 2,580건이 다 들어 있다.
   나머지 54차원은 넣어봐야 손해다.
   (§9.3대로 "고칠 하드웨어"는 없으므로 재수집을 기다릴 이유도 없다.)
3. base를 `RLWRLD/RLDX-1-MT-DROID`로 교체 (physics stream이 학습된 유일한 tactile 체크포인트,
   `action_horizon 16`도 일치).
4. `--use-physics --physics-keys tactile --physics-dims 36 --allow-missing-physics
   --new-param-warmup-steps 2000` — warmup은 논문 mid-training의 2K alignment warmup을 따른다.
5. 접촉이 실제로 성능을 가르는 태스크(삽입/파지력 조절)에서 A/B로 검증.

**1차 학습에서는 하지 않는다.**

---

## 10. 실행 체크리스트

```bash
cd /NHNHOME/doyoung/RLDX-1
git checkout ffw-sh5-b200          # rldx-sm100 환경 정의가 있는 브랜치
bash /NHNHOME/doyoung/setup_rldx.sh all
```

- [x] **26차원 action config 작성·검증** — `ffw_sh5_rev1_config_used26.py`,
      `modality.json`에 그룹 추가 (§12 Q2)
- [ ] **held-out 데이터셋 생성** — merged ep 3,10,17,26,35,44,56,65,74 (§5.3)
- [ ] 학습 스크립트 갱신: `DATASET_PATH`를 held-out 제외 학습셋으로,
      `MODALITY_CONFIG_PATH`를 **`ffw_sh5_rev1_config_used26.py`** 로,
      `MAX_STEPS=30000`, `GLOBAL_BATCH_SIZE=128`, `SAVE_STEPS=2500`,
      `--image-max-area 98304`, `--state-dropout-prob 0.3` 추가
- [ ] 배포 클라이언트의 26 → 57 관절 매핑 갱신 (§12 Q2 "배포 시 주의")
- [ ] ⚠ **Hub의 `meta/modality.json` 갱신** — 로컬에만 새 그룹이 있고 Hub 사본에는
      없다. 지금 상태로 데이터셋을 새로 내려받으면 `used26` config가 실패한다.
      `hf upload learner1119/rlwrld_demo_0826_merge dataset/rlwrld_demo_0826_merge/meta/modality.json meta/modality.json --repo-type dataset`
- [ ] `SMOKE=1`로 20 step 먼저 (loss 유한 / 체크포인트 기록 확인)
- [ ] 본 학습. `--experiment-name`을 새로 줄 것 —
      기존 `ckpt/rldx1/finetuned/rldx1_ft_ffw_sh5_wholebody`에 체크포인트가 있으면
      `resume_from_checkpoint=True`라 조용히 이어서 학습된다 (부록 A.3)
- [ ] 체크포인트마다 held-out `open_loop_eval` MSE 기록 → 최저점 선택
- [ ] 데이터팀에 확인: **초기 송장 라벨 위치의 분포** (front/top/hidden/side) — 단순 "뒤집었는가"가 아니다 (§6.3)

---

## 11. 미해결 / 확인 필요

| 항목 | 왜 중요한가 |
|---|---|
| ~~tactile 센서 1~3번 무응답 원인~~ | **해소됨** — 무응답이 아니었고, 2지 파지 설계의 결과다 (§9.3) |
| ~~손가락 1~3번이 제어되지 않는 이유~~ | **해소됨** — 조이스틱 프리셋으로 4·5번 손가락만 쓰는 의도된 운용이다 |
| 초기 송장 라벨 위치의 분포 (front/top/hidden/side) | 분기 균형을 좌우한다. 신호가 화면에 있다는 건 확인됨(§6.2), **분포는 미확인**(§6.3) |
| 세션별 시연 길이 증가(320→497)의 원인 | 분포 이동이면 평가 설계에 반영해야 한다 (§5.5) |

---

## 12. Q&A (2026-08-26 리뷰)

### Q1. `RLDX-1-PT`가 뭐야? pretraining 맞아? 우리는 파인튜닝만 할 건데.

맞다. 논문 3단계(§1.2) 중 **1단계 산출물**이다.

| Hub 이름 | 논문 단계 | 규모 | 우리와의 관계 |
|---|---|---|---|
| `RLDX-1-VLM` | VLM 적응 (§2.1) | Qwen3-VL 8B + robot-VQA | 백본. 직접 안 씀 |
| **`RLDX-1-PT`** | **1단계 Pre-training** | 1.5M 에피소드, 100K step, batch 8192, 64×H200 195h | **우리 출발점** |
| `RLDX-1-MT-*` | 2단계 Mid-training | 25K step, batch 1024, 64×H200 15h | ALLEX/FR3 전용. 안 씀(§2.1) |
| `RLDX-1-FT-*` | 3단계 Post-training | 30K step, batch 64~128 | **우리가 만들 것** |

즉 우리 작업은 정확히 **3단계(post-training = task-specific fine-tuning)** 하나다.
1·2단계는 RLWRLD가 이미 해놨고 우리가 재현할 규모가 아니다 —
1단계만 64×H200 × 195시간, 우리 4×B200 환산으로 수천 시간이다.

이 위치 때문에 §2의 결론이 나온다: PT는 1단계라 motion/memory/physics가 없고,
그건 2단계에서 붙는 것이라 우리가 3단계에서 대신 붙일 수 없다.

### Q2. 57차원 말고 실제 쓰는 26차원만 action expert 출력으로 쓰고 싶다

**좋은 결정이고, 구현해서 검증까지 마쳤다.**

먼저 사실관계 확인 — 네가 말한 preset(손당 `joint14,15,16,18,19,20`)이 데이터와 정확히
일치한다. 57차원 벡터에서의 인덱스는 이렇다:

| 그룹 | 인덱스 | 원 관절명 | 차원 |
|---|---|---|---|
| `left_arm` | 0:7 | `arm_l_joint1..7` | 7 |
| `right_arm` | 7:14 | `arm_r_joint1..7` | 7 |
| `left_hand_j14_16` | 27:30 | `finger_l_joint14..16` | 3 |
| `left_hand_j18_20` | 31:34 | `finger_l_joint18..20` | 3 |
| `right_hand_j14_16` | 47:50 | `finger_r_joint14..16` | 3 |
| `right_hand_j18_20` | 51:54 | `finger_r_joint18..20` | 3 |
| | | **합계** | **26** |

`joint17`(인덱스 30, 50)은 두 굽힘 그룹 사이에 끼어 있어서 그룹이 손당 2개로 쪼개진다.
`joint17`까지 넣어 `27:34` / `47:54`로 하면 그룹이 4개로 단순해지지만 상수 2차원이
따라 들어온다(28차원). **26을 정확히 맞추는 6그룹 구성을 택했다.**

**왜 54보다 나은가.** 54로 학습해도 틀리지는 않는다 — `q99` 정규화가 상수 차원을
0으로 보내고, flow-matching 타깃 `a − ε`는 `a=0`일 때 `−ε`인데 이건 노이즈 입력
`a^τ=(1−τ)ε`에서 정확히 복원 가능하다. 즉 학습은 된다. 다만 **디코더 폭과 그래디언트의
절반 이상을 신호 없는 28차원에 쓰는 셈**이다. 26으로 줄이면 그 낭비가 사라진다.

**만들어 둔 것:**

1. `dataset/rlwrld_demo_0826_merge/meta/modality.json` — 새 그룹 4개를 **추가**했다.
   기존 `left_hand`(14:34) / `right_hand`(34:54)는 **그대로 뒀다**. 로더가 그룹을
   독립적으로 슬라이스하므로 겹치는 그룹이 공존해도 무해하고, 덕분에 기존
   `ffw_sh5_rev1_config_wholebody.py`(54차원)도 계속 동작한다. 되돌릴 필요 없다.
2. `rldx/configs/data/ffw_sh5_rev1_config_used26.py` — 새 modality config.

**`stats.json`은 재생성할 필요가 없다.** 통계는 전체 57차원 벡터로 저장돼 있고
로더가 그룹 경계 `[start:end]`로 잘라 쓴다(`lerobot_episode_loader.py:463`).
그룹을 바꿔도 통계는 그대로 유효하다.

**검증 결과** (`LeRobotEpisodeLoader`로 실제 로드):

```
state dims : {'left_arm': 7, 'right_arm': 7, 'left_hand': 20, 'right_hand': 20} = 54
action dims: {'left_arm': 7, 'right_arm': 7, 'left_hand_j14_16': 3,
              'left_hand_j18_20': 3, 'right_hand_j14_16': 3, 'right_hand_j18_20': 3} = 26

action 그룹별 q99-q01:
   left_arm            0.686 ~ 1.579   dead_dims=0/7
   right_arm           0.845 ~ 2.377   dead_dims=0/7
   left_hand_j14_16    0.318 ~ 0.596   dead_dims=0/3
   left_hand_j18_20    0.080 ~ 0.556   dead_dims=0/3
   right_hand_j14_16   0.465 ~ 0.871   dead_dims=0/3
   right_hand_j18_20   0.116 ~ 0.813   dead_dims=0/3
```

**26차원 전부가 실제로 움직인다 — 죽은 차원 0개.**

학습도 끝까지 돈다 (4×B200, batch 128, `image_max_area 98304`, 40 step 스모크,
체크포인트 기록까지 확인). 처리량은 같은 조건의 54차원 config보다 **10% 빠르다**
(1.41 vs 1.28 it/s) — 다만 이 차이는 I/O 변동(§7.1.1)에 묻힐 정도라
**속도는 26차원을 고를 이유도 피할 이유도 아니다.**

**`state`는 54로 그대로 뒀다.** 이유:
- 관측은 공짜다. 명령이 안 가는 손가락도 실제로는 ±0.02~0.3 rad 움직이는데,
  그건 수동 변형이라 접촉 정보를 담고 있을 수 있다.
- `StateActionProcessor`가 state와 action의 `modality_keys`를 **독립적으로** 순회한다.
  둘이 같을 필요가 없다. 게다가 전부 `ABSOLUTE`라 `action_config.state_key`로
  묶이는 관계도 없다.
- 줄이고 싶으면 `state`의 `modality_keys`도 같은 6그룹으로 바꾸면 된다. 다만
  얻는 게 없다(입력 폭은 병목이 아니다).

**배포 시 주의.** 이제 모델 출력이 26차원이라, 로봇에 보낼 57차원 명령은 클라이언트가
조립해야 한다 — 26개를 제자리에 넣고 나머지(손가락 1~13·17, head, lift)는 0으로 채운다.
54차원으로 학습했을 때도 head/lift 3차원은 어차피 클라이언트가 채워야 했으니
부담이 늘어난 건 아니지만, **매핑 코드를 반드시 새 그룹 순서에 맞춰 다시 확인**해야 한다.
서버 관측/출력 키가 `action.left_hand`에서 `action.left_hand_j14_16` 등으로 바뀐다.

**학습 명령에서 바뀌는 건 한 줄이다:**

```diff
- --modality-config-path "$PWD/rldx/configs/data/ffw_sh5_rev1_config_wholebody.py"
+ --modality-config-path "$PWD/rldx/configs/data/ffw_sh5_rev1_config_used26.py"
```

### Q3. `--use-physics`는 tactile 쓸 때 켜는 거고, 우리는 안 쓰니까 이 옵션을 안 쓴다는 거지?

**맞다.** 한 가지만 정확히 하자면 `--use-physics`는 *tactile 전용 스위치가 아니라*
**physics stream 전체의 on/off**다. 어떤 신호를 넣을지는 `--physics-keys`가 정한다
(논문에서 ALLEX는 `torque` 하나, FR3는 `tactile` + `torque` 둘 — §2 표).

우리는 tactile도 torque도 안 쓰므로 **`--use-physics`를 아예 주지 않는다**
(기본값 `False`). 그러면 `--physics-keys` / `--physics-dims` /
`--physics-loss-weight` / `--allow-missing-physics`는 전부 무의미해진다.

`observation.tactile` 컬럼은 데이터에 남아 있어도 **로더가 아예 읽지 않는다** —
modality config에 `tactile` 항목이 없으면 그 컬럼은 순회 대상이 아니다
(`lerobot_episode_loader.py:363`). **데이터에서 지울 필요 없고, 별도 조치도 없다.**

다만 §11의 확인 항목(왼·오른손 센서 1~3번 무응답)은 **살려두는 걸 권한다.**
지금 학습에는 영향이 없지만, 나중에 접촉이 중요한 태스크를 하게 되면
그때 데이터를 다시 모아야 하고, 원인을 그때 알아내면 한 사이클을 버린다.

### Q4. 하이퍼파라미터는 "batch 128, lr 1e-4 cosine, warmup 0.05, state_dropout 0.3, top-4 LLM layer, 30K step" 이대로면 되나?

**맞다.** 그게 §7.2 명령의 내용이고, 논문 §G.3/G.4(ALLEX·FR3 실로봇 post-training)와
동일한 레시피다. 딱 두 가지만 같이 챙기면 된다:

1. **`--modality-config-path`를 `..._used26.py`로** 바꿀 것 (Q2).
2. **`--dataset-path`를 held-out을 뺀 학습셋으로** 줄 것 (§5.3). 전체 81개를 그대로
   넣으면 나중에 최적 체크포인트를 고를 근거가 없어진다.

나머지(`weight_decay 1e-5`, `max_grad_norm 1.0`, `video_stride 2`, `action_horizon 16`,
`backbone_select_layer 18`)는 전부 기본값이 이미 논문 값이라 건드릴 필요 없다(§7.3).

### Q5. `--image-max-area 98304`를 쓰겠다

그렇게 하면 된다. 376×672 → **224×384, 프레임당 84 vision token**, 가로 손실
10.2% → 4.0%. 실측 비용은 **처리량 −22%, VRAM +7 GiB**(2.06 → 1.60 it/s,
47.4 → 54.3 GiB), 30K step 기준 4.0 h → 5.2 h다 — 다만 이 수치는 실행마다 크게 흔들린다(§7.1.1).

고정된 head 캠 한 대뿐이고 상자가 컨베이어를 따라 **가로로** 들어오는 구성이라,
가로 화각을 지키는 쪽이 맞다. B200 VRAM은 183 GB라 여전히 30%도 안 쓴다.

단, 저 −22%는 한 번 측정한 값이다. 같은 구성을 여러 번 재보면 I/O 변동만으로
±25%가 흔들리므로(§7.1.1) **토큰 증가의 실제 비용은 저 숫자보다 불확실하다.**
방향(느려진다)은 맞고 크기는 대략 그 정도로만 보면 된다.

**눈으로도 확인했다** (`outputs/frames/cmp_area.png`, 왼쪽 65536 / 오른쪽 98304).
이 작업의 분기를 결정하는 흰 송장 라벨(§6.2)은 **양쪽 모두에서 보이지만 98304가
확실히 선명하다.** 65536은 라벨 덩어리가 뭉개지기 시작하고 가로 화각도 더 잘린다.
분기 판단이 이 라벨에 달려 있으므로 **98304를 고를 근거가 하나 더 생겼다.**

> 참고: 네가 링크한 `terminals/2.txt:1016-1017`은 그 파일이 160줄이라 확인하지 못했다.
> 위 내용이 가리키려던 것과 다르면 알려줘.

### Q6-a. ALLEX Object-in-Box 데이터셋을 볼 수 있나?

**없다.** RLWRLD의 Hugging Face 공개 **데이터셋은 0개**다 (모델은 11개 공개).

```bash
pixi run -e rldx-sm100 python -c "
from huggingface_hub import HfApi
print(list(HfApi().list_datasets(author='RLWRLD')))"   # -> []
```

논문 §3.2가 명시한 대로 ALLEX·FR3 데이터는 **in-house**다. 공개된 건 모델 가중치뿐이다.

대신 **atomic vs composite 구분을 직접 보고 싶으면 RoboCasa365를 보면 된다** —
논문이 그 구분을 평가한 벤치마크이고 공개돼 있다(§G.1: 300-task 사전학습 세트,
task당 100 데모, 50-task 평가 벤치).

### Q6-b. composite task가 뭐고 왜 위험한가? 논문 어디에 있나?

**논문 위치:**

| 내용 | 위치 |
|---|---|
| Atomic / Composite-Seen / Composite-Unseen 성적 | **Table 1b** (§6.1) |
| 벤치마크 정의 (300-task 학습, 50-task 평가, 3개 split) | **§G.1** RoboCasa365 항목 |
| task별 전체 목록 — atomic과 composite가 뭔지 이름으로 확인 가능 | **Table 15** (a)(b)(c) |
| "atomic → long-horizon으로 갈수록 격차가 커진다"는 해석 | §6.1 Result 문단 |
| 데이터 수집 시 full task를 primitive로 분해하는 절차 | **§4.3** Adaptive Data Collection |

**정의는 Table 15의 이름을 보면 바로 잡힌다:**

| | 예시 | 성격 |
|---|---|---|
| Atomic | `Close Fridge`, `Open Drawer`, `Turn On Stove`, `PnP Counter to Cabinet` | 동작 1개. 시작-끝이 명확 |
| Composite | `Prepare Coffee`, `Load Dishwasher`, `Pack Identical Lunches`, `Set Up Cutting Station` | 여러 primitive를 **순서대로** 엮어야 완료 |

**성적 (Table 1b):**

| | Atomic-Seen | Composite-Seen | Composite-Unseen |
|---|---|---|---|
| RLDX-1 | **67.3** | **19.0** | **5.6** |
| GR00T N1.6 | 61.1 | 12.6 | 2.6 |
| π₀ | 34.6 | 6.1 | 1.1 |

**왜 위험한가 — 세 가지 메커니즘:**

1. **오차 누적.** 단계마다 성공률 0.9여도 4단계면 0.9⁴ = 0.66이다. 단계가 길수록
   지수적으로 떨어진다. Table 1b의 67.3 → 19.0이 딱 이 모양이다.
2. **중간 상태 기억이 필요하다.** "라벨을 이미 확인했는가"는 현재 프레임에 안 보인다.
   논문이 memory module로 푸는 문제인데(§2.1 Functionality 2), 우리는 PT 기반이라
   그 모듈이 없다(§2).
3. **같은 관측에 정답이 여러 개가 된다 (이게 제일 실질적이다).** 상자를 막 잡은
   시점의 화면은 "이제 뒤집는다"와 "이제 넘긴다"가 거의 같아 보이는데, 언어 조건은
   전 구간 동일한 한 문장이다. 학습이 이 둘을 평균 내면 **둘 다 아닌 행동**이 나온다.
   flow-matching은 multimodal 분포를 표현할 수 있지만, 어느 모드를 고를지 알려주는
   신호가 입력에 없으면 조건부 분포가 진짜로 이봉이 되어버린다.

우리 instruction은 4단계 + 조건 분기 하나다:

> "Receive the package from the conveyor / check the shipping label /
> flip the box face up **if necessary** / forward it to the next station."

즉 **composite + 조건 분기**로, Table 1b의 Composite 열에 해당한다.

### Q6-c. subtask로 나눠서 진행해야 하나? 내가 라벨링해야 하나? 방법은?

**먼저: 맞다, 내가 분기 자동 검출에 실패한 이유가 바로 이거다.** 분기를 결정하는
정보는 **화면 안(라벨의 방향)** 에 있지 관절 궤적 통계에 있지 않다. 그래서 길이·파지
횟수·양손 사용 어느 것으로도 안 갈렸다(§6.1). 사람이나 VLM이 **영상을 봐야** 한다.

**좋은 소식: 코드베이스가 subtask 조건화를 이미 지원한다.** 새로 만들 게 없다.

`rldx/data/dataset/lerobot_episode_loader.py:69`:

```python
LANG_KEYS = ["task", "sub_task"]
```

language modality 키를 무엇으로 주느냐에 따라 경로가 셋으로 갈린다:

| `modality_keys` | 출처 | 우리 현재 |
|---|---|---|
| `["annotation.human.task_description"]` | `modality.json` → `task_index` → `tasks.jsonl` | ✅ 지금 이것 |
| `["task"]` | `episodes.jsonl`의 `tasks` 배열에서 랜덤 선택 | |
| **`["sub_task"]`** | **`episodes.jsonl`의 `sub_tasks` 구간에서 프레임별로** | 전환하려면 |

**`sub_task`로 가는 방법은 두 단계뿐이다.**

**(1) `meta/episodes.jsonl`에 `sub_tasks` 구간을 추가한다:**

```json
{"episode_index": 0,
 "tasks": ["Receive the package from the conveyor, check the shipping label, ..."],
 "length": 437,
 "sub_tasks": [
   {"start":   0, "end": 130, "text": "receive the package from the conveyor"},
   {"start": 130, "end": 210, "text": "check the shipping label"},
   {"start": 210, "end": 330, "text": "flip the box face up"},
   {"start": 330, "end": 437, "text": "forward the box to the next station"}]}
```

뒤집기가 없는 회차는 세 번째 구간을 그냥 빼면 된다. **이게 곧 분기 라벨이기도 하다** —
`"flip the box face up"` 구간이 있는 에피소드 수를 세면 §6이 못 구한 분기 비율이 나온다.

**(2) config의 language 한 줄을 바꾼다:**

```diff
- "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
+ "language": ModalityConfig(delta_indices=[0], modality_keys=["sub_task"]),
```

로더 동작(`create_language_from_meta`, 같은 파일 :484)에서 알아둘 것:

- 각 subtask는 `start - action_horizon//2` = **`start - 8` 프레임부터** 적용된다.
  다음 동작을 아직 끝내지 않은 시점에 이미 다음 지시가 들어오게 만드는 장치다.
- 그 8프레임 선행 때문에 **인접한 두 subtask가 겹치는 구간이 생기고, 겹친 프레임은
  둘 중 하나를 `random.choice`로 고른다.** 즉 전환이 딱 끊기지 않고 확률적으로
  섞인다 — 의도된 동작이니 그대로 두면 된다.
- 어느 구간에도 안 걸린 프레임은 빈 문자열 `""`이 된다. **구간이 0부터 `length`까지
  빈틈없이 덮도록 붙여서 쓰는 게 좋다.**

**⚠ 그런데 이건 배포 계약을 바꾼다.** `policy_loader.py:250 _resolve_language_key`가
체크포인트의 modality config에서 language 키를 읽어 그대로 추론 입력으로 요구한다.
`sub_task`로 학습하면 **실행 중에 "지금 어느 subtask인가"를 누군가 정해서 넣어줘야 한다** —
상위 플래너(VLM 등)가 필요해진다는 뜻이다. end-to-end 한 문장 조건화와는 다른 시스템이 된다.

**그래서 내 권장 순서는 이렇다:**

1. **1차 학습은 지금 그대로(단일 instruction) 간다.** subtask 라벨링은 81개 영상을
   사람이 훑어야 하는 비용이고, **1차 결과 없이는 어디를 쪼개야 하는지도 모른다.**
2. 1차 결과를 held-out open-loop eval + 실기 롤아웃으로 본다. 실패가
   (a) 전 구간 어설픔인지 (b) **특정 전환 지점에서 멈추거나 잘못된 분기를 타는 것**인지 본다.
3. (b)라면 그때 subtask를 도입한다. 이때도 전수 라벨링부터 하지 말고,
   **`"flip the box face up"` 구간이 있는지 여부만 먼저 라벨링**해서 분기 비율을 구하는 게
   비용 대비 효과가 가장 크다. 비율이 심하게 치우쳐 있으면(소수 분기 < 20%)
   subtask보다 **소수 분기 데이터를 더 모으는 게** 먼저다(§6).
4. 그래도 부족하면 논문 §4.3의 adaptive data collection — 실패 조건을 특정해서
   그 케이스만 집중 수집 — 으로 간다.

**라벨링을 반자동화하고 싶다면**: §6.3을 따른다 — 앞쪽에 몰아서 프레임을 뽑고
"흰 송장 라벨이 어느 면에 있는가"를 묻는다. 이미 `run_scripts/data/peek_frames.py`가
그 배치로 프레임을 뽑아 준다.

### Q6-d. "atomic으로 해보자"는 뜻인가? — 아니다

용어를 정리하고 가야 한다. **atomic / composite는 태스크 자체의 성질이지 라벨링 방식이
아니다.** 우리 작업은 4단계 + 조건 분기이므로 라벨을 어떻게 붙이든 **composite다.**
로봇이 해야 하는 일이 줄어들지 않는다.

내가 권한 건 "atomic으로 하자"가 아니라 **"1차는 전체 문장 하나로 end-to-end로 가자"** 다.
선택지를 정리하면 셋이다:

| | 방식 | 태스크 성질 | 필요한 것 | 배포 |
|---|---|---|---|---|
| **A** | **전체 문장 1개로 end-to-end** | composite | 없음 (지금 그대로) | 지시문 1개 | 
| B | `sub_task` 조건화 단일 정책 | composite | 구간 라벨 + 런타임 플래너 | 매 시점 subtask 지정 필요 |
| C | 구간별 정책 분리 + 오케스트레이터 | atomic 문제들로 분해 | 구간 라벨 + 정책 N개 + 전환 로직 | 시스템 전체 재설계 |

**1차 학습 권장은 A다.** B가 "composite를 atomic으로 바꾸는" 것도 아니다 —
실행 시점의 태스크는 여전히 composite이고, 정책이 보는 **언어 신호만** 잘게 쪼개질 뿐이다.
진짜로 atomic 문제들로 쪼개는 건 C인데, 그건 지금 할 일이 아니다.

데이터 처리 부담이 큰 상황이면 A가 더욱 맞다 — **A는 추가 라벨링이 0이다.**

### Q6-e. "뒤집었는가" 이분법으로는 부족하다 (맞다)

맞는 지적이라 §6.2 / §6.3을 새로 썼다. 이건 상자를 뒤집는 게 아니라
**송장 테이프가 위로 오도록 방향을 맞추는** 작업이고, 그러면 라벨은 이분법이 아니라
**"초기에 라벨이 어느 면에 있었는가"** 가 되어야 한다
(`front` / `top` / `hidden` / `side`).

그리고 영상을 실제로 확인해 보니 **좋은 소식이 하나 나왔다**: 그 흰 라벨이
학습 해상도(224×384)에서도 뚜렷하게 보인다(§6.2). 즉 분기를 결정하는 정보가
정책 입력에 실제로 들어 있다 — **정보가 없어서 못 배우는 상황은 아니다.**

### Q7. "Hub 사본에 새 그룹이 없다"는 게 무슨 말인가

오늘 낮에 merge 결과를
[`learner1119/rlwrld_demo_0826_merge`](https://huggingface.co/datasets/learner1119/rlwrld_demo_0826_merge)로
push했다. 그 뒤 26차원 작업(Q2)을 하면서 **로컬의**
`dataset/rlwrld_demo_0826_merge/meta/modality.json`에 그룹 4개를 추가했다.
**Hub에 올라가 있는 사본은 그 전 버전이다.**

지금 상태에서 이렇게 하면 깨진다:

```bash
rm -rf dataset/rlwrld_demo_0826_merge
hf download learner1119/rlwrld_demo_0826_merge --repo-type dataset \
   --local-dir dataset/rlwrld_demo_0826_merge      # 옛 modality.json이 내려옴
# -> used26 config가 left_hand_j14_16 그룹을 못 찾아 실패
```

다른 장비에서 받아 학습을 돌리려는 경우도 같다. 지금 이 서버의 로컬 파일만 최신이다.

고치는 건 파일 하나 올리는 것뿐이다:

```bash
hf upload learner1119/rlwrld_demo_0826_merge \
   dataset/rlwrld_demo_0826_merge/meta/modality.json meta/modality.json \
   --repo-type dataset
```

공개 repo를 갱신하는 것이라 승인 후에 진행한다. **올리지 않아도 이 서버에서
지금 학습하는 데는 아무 문제가 없다** — 재다운로드/타 장비 이관 시에만 문제가 된다.

---

## 13. fps와 chunk / execution horizon — 10 fps로 모은 게 문제인가

### 13.1 논문은 자체 실로봇의 수집 fps를 밝히지 않는다

논문 전문을 텍스트로 뽑아 검색한 결과, fps/Hz 언급은 딱 둘뿐이다:

| 위치 | 내용 | 무엇에 대한 값인가 |
|---|---|---|
| §1 / Table 4 | `> 22Hz`, per-step latency 43.7 ms | **추론 속도**(RTX 5090), 로봇 제어 주기가 아님 |
| §B.3 | `93 consecutive frames ... at 16 FPS` | **합성 영상 생성용** 비디오 모델 학습 클립 |

ALLEX·FR3·OpenArm을 **몇 Hz로 텔레옵·기록했는지는 논문 어디에도 없다.** §G.2~G.5의
하드웨어 설명에도, §4.2 구현 상세에도 없다. 그러니 "논문은 몇 fps인가"에 대한 정직한
답은 **"안 나와 있다"** 이다.

### 13.2 그런데 코드가 fps에 무관하다 — 이게 더 중요한 사실

```
rldx/data/dataset/lerobot_episode_loader.py:203
    self.fps = self.info_meta.get("fps", 30)
```

**읽기만 하고 데이터 파이프라인 어디서도 쓰지 않는다.** `self.fps`의 다른 사용처는
`rldx/eval/sim/wrapper/video_recording_wrapper.py`(시뮬 녹화용)뿐이다.

`delta_indices`, `video_stride`, `memory_stride`는 **전부 프레임 단위**다. 즉:

> **fps는 학습 코드의 동작을 전혀 바꾸지 않는다. 설정값의 "초 단위 의미"만 바꾼다.**

`info.json`의 `fps: 10`을 30으로 고쳐도 학습은 한 글자도 안 달라진다. 그러니
"10 fps라서 뭔가 잘못 학습된다"는 걱정은 성립하지 않는다. 대신 **우리가 고른
프레임 수가 실제로 몇 초를 뜻하는지**를 계산해서 판단해야 한다.

### 13.3 우리 10 fps에서 설정값의 실제 의미

| 설정 | 프레임 | **10 fps 기준 초** |
|---|---|---|
| video context `{-6,-4,-2,0}` | 6프레임 폭 | **0.6 s** 시각 이력 |
| video 프레임 간격 (`video_stride 2`) | 2 | 0.2 s |
| **action chunk (`action_horizon 16`)** | 16 | **1.6 s** 미래 행동 |
| memory 1슬롯 (`memory_stride 16`) | 16 | 1.6 s |
| memory 전체 (3슬롯) | 48 | 4.8 s |
| 에피소드 길이 | 215~916 | 21.5 ~ 91.6 s |

참고로 우리 녹화는 **명목 10 Hz가 아니라 실제로 정확히 10 Hz**다 —
`rilab_episodes.jsonl`의 `period_target_ms: 100.0`, `period_std_ms ≤ 3.6`,
dropped/stale 프레임 0(§5.1 감사). 타이밍 자체는 흠잡을 데 없다.

### 13.4 논문/레포가 실제로 쓴 chunk — 레포에 config가 들어 있다

논문 본문에 없는 것도 레포에 임바디먼트별 config로 남아 있다:

| 임바디먼트 | chunk (`action_horizon`) | 근거 |
|---|---|---|
| ALLEX (48-DoF 휴머노이드) | 40 | `configs/data/midtrain_allex_data_config.py:58` + §4.2 |
| FR3 (단일 팔) | 16 | §4.2 |
| **OpenArm (양팔 7-DoF + 6-DoF 손)** | **16** | `configs/data/openarm_inspire_config.py:45` |
| **우리 (FFW-SH5)** | **16** | ✅ 일치 |

**우리와 임바디먼트가 가장 가까운 OpenArm이 16이다.** ALLEX만 40인데 그건
mid-training부터 40으로 간 별도 계열이고, PT에서 출발하는 우리에겐 해당 없다(§4.4).

fps를 모르니 "논문의 1.6초와 우리의 1.6초"를 직접 비교할 수는 없다. 확실한 건
**프레임 수 기준으로 우리가 논문의 실로봇 설정과 같다**는 것이다.

### 13.5 ⚠ 10 fps에서 진짜 중요한 knob은 execution horizon이다

이게 이 절의 결론이다. **chunk 16을 뽑아 놓고 그중 몇 개를 실제로 실행할지**가
배포 성능을 가른다. 그리고 10 fps에서는 이 선택의 무게가 커진다.

- 16개 전부 실행 = **1.6초 동안 관측 없이 open-loop**
- 8개 실행 = 0.8초
- 상자가 컨베이어를 타고 **움직이는** 작업에서 1.6초는 길다

논문의 선택 (Table 8 / Appendix C):

| 벤치 / 로봇 | chunk | 실행 | 비율 |
|---|---|---|---|
| SIMPLER | 16 | 2 | 1/8 |
| LIBERO, GR-1 Tabletop | 16 | 8 | 1/2 |
| RoboCasa Kitchen, RoboCasa365 | 16 | 16 | 1/1 |
| **ALLEX (실로봇)** | 40 | **20** | **1/2** |

실로봇인 ALLEX가 **절반**을 쓴다. 그리고 ALLEX는 여기에 더해
**RTC(Real-Time Chunking)를 기본 추론 전략으로 쓴다**(§G.3, Appendix C).

> **권장: 배포 시 16개 중 8개만 실행(0.8초).** 논문 실로봇의 1/2 비율과 같고,
> 컨베이어의 동적 요소에도 대응할 여지가 생긴다.
> 여유가 되면 `--rtc-inference-mode guided`로 RTC를 얹는다 — 어떤 flow-matching
> 체크포인트에도 적용되므로 재학습이 필요 없다(`TrainConfig.rtc_inference_mode`).

**구현상 주의**: `run_rldx_server.py`의 `--execution-horizon`은 **`ReplayPolicy` 전용**이다
(`run_rldx_server.py:185`). 실제 정책(`RLDXPolicy`)은 **16개를 전부 반환**하고,
몇 개를 쓸지는 **클라이언트가 정한다.** 즉 이 값은 서버 플래그가 아니라
**로봇 제어 루프 쪽 코드에서 정해야 한다.** RTC를 쓸 때만
`--rtc-inference-exec-horizon`이 서버 쪽 knob이 된다.

나중에 memory 모듈을 켜게 되면 `memory_stride`를 이 실행 호라이즌과 맞춰야 한다
(`TrainConfig.memory_stride` docstring: *"Should equal the intended execution_horizon
at inference time"*).

### 13.6 그래서 10 fps 자체가 문제인가 — 정리

| | 판단 |
|---|---|
| 학습 코드가 깨지나 | **아니다.** 파이프라인이 fps에 무관하다(§13.2) |
| chunk 16을 바꿔야 하나 | **아니다.** OpenArm과 같고 사전학습 혼합도 16이다 |
| 위험한 지점 | **배포 시 실행 호라이즌**(§13.5). 16개 전부 실행하면 1.6초 open-loop |
| 접촉 해상도 | 1스텝 = 100 ms. 상자 운반엔 충분, **정밀 삽입엔 부족**할 수 있다 |
| 에피소드 스텝 수 | 215~916 스텝. 길다 → composite 우려를 키운다(§6) |
| 유리한 점 | video context 0.6초로 넉넉 → 컨베이어 **모션 추정에는 오히려 유리** |
| 되돌릴 수 있나 | **fps는 재수집 없이 못 올린다.** 내리는 건 가능하지만 이득이 없다 |

**결론: 지금 데이터로 그대로 학습하면 된다. fps 때문에 바꿀 학습 설정은 없다.**
대신 **배포 단계에서 "16개 중 8개 실행"을 반드시 챙길 것.**

다음 데이터를 새로 모을 일이 생기고 정밀 접촉 작업이 포함된다면, 그때는
20~30 Hz를 고려할 만하다. 지금 작업(상자 운반)에는 10 Hz가 부족하지 않다.

---

## 14. Q&A 2차 (2026-08-26 후속)

### Q8. ep30에서 17초까지 송장이 안 보인다는 걸 어떻게 알았나?

**정확히는 몰랐다. 구간으로만 좁혔다.** `peek_frames.py`가 뽑는 8프레임은
고정 비율(`0, n/12, n/6, n/4, n/3, n/2, 2n/3, n-1`)이고, ep30은 359프레임 = 35.9초라
샘플 시각이 이렇게 된다:

```
0.0s  2.9s  5.9s  8.9s  11.9s  17.9s  23.9s  35.8s
 ✗     ✗     ✗     ✗      ✗      ✓      ✓      ✓      ← 흰 송장 라벨 보임?
```

즉 내가 관측한 건 **"11.9초까지는 안 보이고 17.9초에는 보인다"** — 전환 시점은
`(11.9, 17.9]` 구간으로만 알 수 있었다. "17초"라는 정확한 값은 네가 알려준 것이고,
내 관측과 모순되지 않는다.

이건 §6.3의 "앞쪽을 촘촘히 뽑으라"는 조언이 왜 필요한지 보여주는 사례이기도 하다.
지금 배치는 뒤쪽(2n/3, n-1)에 2장을 쓰는데, 분기 판정에는 그보다 앞쪽 해상도가 더 필요하다.

### Q9. ep70은 상자가 두 개고 하나는 송장이 보이고 하나는 안 보이는데, 그것도 알 수 있나?

**보인다.** ep70(540프레임 = 54초)의 첫 프레임에 상자가 둘이다:

| | 위치 | 초기 송장 |
|---|---|---|
| 상자 A | 왼쪽 뒤 | **윗면에 흰 라벨 보임** |
| 상자 B | 오른쪽 앞 (주황 띠) | **안 보임** |

이후 프레임에서 로봇이 상자 B를 조작하고, 9초쯤(3번째 샘플)에 **B의 윗면에 흰 라벨이
나타난다** — 방향을 맞춘 것이다. 뒤쪽 프레임에는 주황 EMS 띠가 있는 더 큰 상자가
세워진 채로 다뤄지는 장면이 나온다.

이건 §6.3의 라벨 스키마가 **에피소드당 1개로는 부족할 수 있다**는 뜻이기도 하다.
한 회차에 상자가 여러 개면 상자 단위로 세야 분기 분포가 정확해진다:

```json
{"episode_index": 70, "n_boxes": 2,
 "boxes": [{"label_face_initial": "top",    "reorientations": 0},
           {"label_face_initial": "hidden", "reorientations": 1}]}
```

### Q10. modality.json을 Hub에 push해야 하나? 다른 데서 inference 하려면?

**학습은 여기서만 할 거면 push 불필요하다.** 그리고 **순수 inference에도 불필요하다** —
이건 확인했다.

체크포인트가 자기 완결적이기 때문이다. 학습이 끝나면 체크포인트 안에
`experiment_cfg/`가 통째로 복사되고, 거기에 두 가지가 다 들어 있다:

```
ckpt/.../checkpoint-N/experiment_cfg/
    conf.yaml                 ← modality_configs 전체
                                 (video.delta_indices: [-6,-4,-2,0] 까지 전개된 형태)
    dataset_statistics.json   ← q01/q99 정규화 파라미터
```

추론 시 `policy_loader.py`가 여기서 modality config를 읽는다
(CLAUDE.md의 "`--modality-config-path`가 없는 게 오산이 아닌 이유"가 이것).

정리하면:

| 상황 | 데이터셋 필요? | modality.json push 필요? |
|---|---|---|
| 이 서버에서 학습 | 로컬에 있음 | ❌ |
| **실로봇 서빙** (`run_rldx_server` + 체크포인트) | **불필요** | **❌** |
| 다른 장비에서 `open_loop_eval` | 필요 (데이터셋 로드) | ✅ |
| 다른 장비/시점에 재학습 | 필요 | ✅ |
| 데이터셋을 지웠다가 재다운로드 | 필요 | ✅ |

즉 **push는 "나중에 이 데이터셋으로 뭔가 다시 할 때"를 위한 보험**이지 추론 조건이
아니다. 지금 안 올려도 되고, 재학습·재평가 계획이 생기면 그때 한 줄이면 된다.

### Q11. 우리는 10 fps로 모았는데 RLDX-1은 몇 fps인가? chunk / n_action_step은?

**§13 전체가 이 답이다.** 요약하면:

1. **논문은 자체 실로봇 fps를 밝히지 않는다** (§13.1). ">22Hz"는 추론 속도,
   "16 FPS"는 합성 영상 생성용이다.
2. **코드가 fps에 무관하다** (§13.2). `self.fps`는 읽히기만 하고 안 쓰인다.
   `delta_indices`는 프레임 단위라 **fps는 설정의 "초 단위 의미"만 정한다.**
3. **chunk는 16 그대로 간다** (§13.4). 레포 config를 보면 우리와 임바디먼트가
   가장 비슷한 OpenArm(양팔 + 손)이 정확히 16이다.
4. **진짜 챙길 건 execution horizon** (§13.5). 10 fps에서 16개 전부 실행은
   1.6초 open-loop다. 논문 실로봇(ALLEX)은 40 중 20 = **절반**만 쓰고 RTC를 얹는다.
   → **우리도 16 중 8(0.8초)** 를 쓰고, 여유가 되면 RTC를 켠다.
5. 이건 **서버 플래그가 아니라 로봇 제어 루프 쪽 결정**이다 — `RLDXPolicy`는 16개를
   전부 반환하고 클라이언트가 몇 개 쓸지 정한다(§13.5).

### 13.7 chunk를 16 → 50으로 바꿀 수 있나 (실측: 된다)

**바꿀 수 있고, 실제로 돌려서 확인했다.** 다만 **두 곳을 같이** 바꿔야 한다:

```python
# rldx/configs/data/ffw_sh5_rev1_config_used26_h50.py
"action": ModalityConfig(delta_indices=list(range(50)), ...)
```
```bash
--action-horizon 50            # CLI 쪽도 같이
```

둘이 어긋나면 `assembly.py:304`가 학습 시작 전에 막는다 —
`action_horizon=16 disagrees with modality_config ... delta_indices has length 50`.
조용히 잘못 돌아가지 않으니 안심해도 된다.

**실측 (4×B200, batch 128, `image_max_area 98304`, 40 step):**

| chunk | 초 (@10fps) | 처리량 | peak VRAM/GPU | 30K step | 유효 학습 샘플 |
|---|---|---|---|---|---|
| 16 | 1.6 s | 1.41 it/s | 54.3 GiB | ~5.9 h | 28,581 |
| **50** | **5.0 s** | **1.20 it/s** | **55.9 GiB** | **~6.9 h** | **26,133 (−8.6%)** |

**비용이 거의 없다.** VRAM +1.6 GiB, 속도 −15%(이것도 I/O 변동 ±25% 안쪽이라
확정적이지 않다). loss도 정상적으로 떨어진다(40 step에 0.128).

#### 그런데 하나 짚고 갈 게 있다 — 사전학습은 16이었다

논문 §2.2: *"we apply rotary positional embeddings (RoPE) to the action (A) stream
to capture the relative temporal structure within the action chunk."*

즉 action stream의 시간 구조가 **RoPE로 인코딩**되는데, `RLDX-1-PT`는 chunk 16으로만
학습됐다. 50은 그 3배가 넘는 길이라 **RoPE가 미학습 영역으로 들어간다.**
ALLEX가 40을 쓰긴 하지만 그건 **mid-training(25K × 1024 = 2,560만 샘플)부터** 40으로
간 경우다. PT(16)에서 곧장 50으로 가는 우리와는 조건이 다르다 — 우리가 그 간극을
메울 데이터는 2.6만 샘플이다.

RoPE 외삽은 보통 되지만 "된다"와 "16만큼 잘 된다"는 다르다. **작동은 확인됐고,
성능 영향은 열려 있다.**

#### 대안: chunk 40이면 목표를 그대로 달성하면서 근거가 더 강하다

의도가 **"2초 미래를 실행"** 이라면 그건 chunk 크기가 아니라 실행 개수가 정한다.
20 스텝 = 2.0초는 chunk 40에서도 똑같다:

| | chunk | 실행 | 실행 시간 | 실행 비율 | 유효 샘플 | 근거 |
|---|---|---|---|---|---|---|
| 지금 계획 | 50 | 20 | 2.0 s | 0.40 | 26,133 | — |
| **대안** | **40** | **20** | **2.0 s** | **0.50** | **26,853** | **ALLEX와 동일 (§4.2)** |

chunk 40의 이점:
- **논문이 실제로 학습시켜 본 길이**다. 50은 아무도 안 해본 값이다.
- 실행 비율 0.50이 **ALLEX 실로봇 설정(20/40)과 정확히 일치**한다.
- 유효 샘플이 조금 더 많고(−6.0% vs −8.6%), 연산도 덜 든다.
- 뒤쪽 30개(50-20)가 아니라 20개(40-20)만 "예측하지만 실행 안 하는" 구간이 된다.
  안 쓸 구간에 loss 용량을 덜 쓴다는 뜻이다.

> **권장: 40.** 원하는 2초 실행을 그대로 얻으면서 논문 근거가 붙는다.
> **50으로 가도 막을 이유는 없다** — 실측상 돌아가고 비용도 거의 같다.
> 어느 쪽을 고르든 `--action-horizon`과 `delta_indices`를 함께 바꾸는 것만 지키면 된다.
>
> 진짜로 5초 예측이 필요한 게 아니라면 40이 낫다는 정도의 차이다.

만들어 둔 것: `rldx/configs/data/ffw_sh5_rev1_config_used26_h50.py` (chunk 50).
40으로 갈 거면 `range(50)` → `range(40)` 한 곳만 고치면 된다.

#### n_action_step = execution horizon, 그리고 학습과 무관하다

**맞다, 같은 것이다.** 그리고 **학습에는 전혀 안 들어간다.**

학습은 언제나 **전체 chunk를 예측하도록** 학습한다. "그중 몇 개를 실제로 실행할지"는
배포 시점의 클라이언트 결정이고, 체크포인트에 기록되지도 않는다(§13.5).
그래서 **같은 체크포인트로 실행 개수를 20으로도 10으로도 바꿔 가며 시험할 수 있다.**
재학습 없이.

예외가 딱 둘이다:

| 설정 | 학습 시 알아야 하나 | 왜 |
|---|---|---|
| `memory_stride` | ✅ | docstring: *"Should equal the intended execution_horizon at inference time"* — memory 슬롯 경계를 실행 경계에 맞추기 위함. **memory를 안 쓰면 무관**(우리는 안 씀, §2) |
| `rtc_training_max_delay` | ✅ | RTC를 `trained` 모드로 쓸 거면 학습 때 켜야 함 (§13.8) |

둘 다 우리 1차 학습에서는 해당 없다. 즉 **실행 개수는 학습 후에 자유롭게 정하면 된다.**

### 13.8 RTC (Real-Time Chunking) — 무엇이고, 학습부터 바꿔야 하나

**이해한 대로 맞다.** 청크 경계에서 궤적이 튀는 걸 막는 장치다. 정확히는 두 문제를
같이 다룬다:

1. **불연속** — 새 청크를 이전 청크와 무관하게 생성하면 경계에서 행동이 점프한다.
2. **추론 지연** — 관측 → 추론 → 실행 사이에 로봇은 계속 움직인다. 그 사이에 실행될
   앞부분 `d`개는 이미 확정(frozen)된 상태다.

RTC는 **"앞 `d`개를 고정하고 나머지를 거기에 이어지도록 생성"** 한다.

레포 구현(`rldx/model/modules/action_model/rtc.py`)은 학습용 1개 + 추론용 2개다:

| 플래그 | 단계 | 재학습 | 출처 |
|---|---|---|---|
| `--rtc-training-max-delay N` | **학습** | — | arXiv 2512.05964 Alg.1 |
| `--rtc-inference-mode trained` | 추론 | ✅ 위 학습 필수 | 2512.05964 |
| `--rtc-inference-mode guided` | 추론 | ❌ **어떤 flow-matching 체크포인트에도 동작** | arXiv 2506.07339 Eq.2 |

- **training RTC**: 샘플마다 `d ~ U[0, max_delay]`를 뽑아 앞 `d`개를 clean ground
  truth(τ=1)로 두고 그 구간 loss를 마스킹한다. "clean prefix가 주어졌을 때 뒤를 잇는"
  능력을 명시적으로 학습시킨다.
- **guided**: 예측된 clean trajectory의 vector-Jacobian product로 velocity field를
  수정한다. 학습에 손대지 않는다.
- 안전장치: `trained`를 `training_max_delay=0`인 체크포인트에 쓰면 **명시적 에러**가
  난다(`rtc.py:87`). 조용히 이상해지지 않는다.

제약(`rtc.py:100`): `0 <= d <= s <= H - d`. `d`=frozen prefix, `s`=execution horizon,
`H`=chunk. 그리고 `enabled_inference()`는 `inference_delay > 0`일 때만 참이다 —
**`d=0`이면 RTC는 아예 안 걸린다.**

`--rtc-inference-exec-horizon`은 **RTC가 켜졌을 때의 `s`**다. RTC를 안 쓰면 이 값은
무의미하고, 실행 개수는 순전히 클라이언트 결정이다(§13.5).

#### 논문은 어느 쪽을 썼나 — 재학습 필요 없는 쪽이다

논문이 RTC로 인용하는 문헌은 **하나뿐**이다:

> [15] Kevin Black, Manuel Y Galliker, and Sergey Levine.
> *Real-time execution of action chunking flow policies.* NeurIPS 2025c.

이게 arXiv 2506.07339 = **`guided`(추론 전용)** 쪽이다. 학습형 RTC 논문(2512.05964)은
**논문 어디에도 인용되지 않는다** — 레포가 논문 범위를 넘어 추가로 구현해 둔 기능이다.

> **결론: 1차 학습에서 RTC를 켤 필요 없다.** `--rtc-training-max-delay`는 기본값 0으로
> 두고, 배포 때 `--rtc-inference-mode guided`부터 시도한다. 이게 논문이 한 것과 같다.
>
> 나중에 `trained` RTC를 쓰고 싶어지면 **그때는 재학습이 필요**하다. 우리 규모는
> 30K step에 5~8시간이라 재학습 부담 자체는 크지 않다.

`guided`의 유일한 비용: 야코비안 autograd가 추론 메모리를 몇 배로 늘린다. 그래서
`rtc_jacobian_steps_only` 기본값이 3이다(앞 3 denoising step에만 적용). OOM이 나면
1로 줄이면 된다.

---

## 15. 학습 직전 점검 (pre-flight) — 실제로 다 돌려봤다

40 step짜리 스모크만으로는 확인되지 않는 것들이 있어서, 학습을 걸기 전에
**끝까지 한 바퀴** 돌려 봤다.

### 15.1 ⚠ 앞선 권장 하나가 틀렸다 — `--save-total-limit`

§7.2에서 `--save-steps 2500 --save-total-limit 6`을 권했는데, 계산해 보면
**우리가 찾으려는 구간을 정확히 지운다:**

```
저장 시점 : 2500, 5000, ..., 30000        (12개)
limit 6   → 잔존: 17500 20000 22500 25000 27500 30000
            삭제: 2500 5000 7500 10000 12500 15000   ← §8이 "최적 예상"이라 한 구간
```

§8은 "10K~20K에서 held-out MSE가 최저를 찍는 경우가 많다"고 써 놓고, §7.2는 그 구간을
지우는 설정을 권했다. **모순이었다.** 저장 간격과 limit은 반드시 같이 계산해야 한다:

| 설정 | 저장 횟수 | 잔존 | 용량 | 비고 |
|---|---|---|---|---|
| 2500 / 6 | 12 | 17.5K~30K | 366 GB | ❌ 최적 예상 구간이 삭제됨 |
| 2500 / 12 | 12 | 전부 | 732 GB | 해상도 최고, 용량 큼 |
| 5000 / 5 | 6 | 10K~30K | 305 GB | 5K 하나만 버림 |
| **10000 / 5** | **3** | **10K, 20K, 30K** | **183 GB** | **채택. 삭제 없음, 대신 해상도 3점** |

**채택: `--save-steps 10000 --save-total-limit 5`** — 삭제가 일어나지 않는다.
다만 최적점을 3개 지점으로만 찾게 되므로, 10K와 20K의 held-out MSE 차이가 애매하면
`SAVE_STEPS=5000`으로 다시 돌리는 걸 고려한다(환경변수 하나면 된다).

스크립트가 시작 시 이 계산을 출력하고, 삭제가 일어나는 조합이면 경고한다.

### 15.2 검증한 것들

| 항목 | 방법 | 결과 |
|---|---|---|
| held-out 분할 | `split_holdout.py` | train 72 ep / 29,661 frames, holdout 9 ep / 3,357 frames (10.2%). **양쪽 감사 통과, 겹침 0** |
| 중간 저장 | `--save-steps 20 --max-steps 45` | 20 / 40 / 45에 저장됨 ✅ |
| 저장 회전 | `--save-total-limit 2` | checkpoint-20 삭제, 40·45 잔존 ✅ |
| **저장 비용** | 위 런의 타임라인 | **1회당 약 40초.** 12회여도 8분 — 6~8시간 대비 무시 가능 |
| **`open_loop_eval` (26차원)** | holdout 3 traj | ✅ `gt_action_joints (64, 26)`, `pred (64, 26)`. **26차원이 끝까지 관통** |
| 체크포인트 자립성 | 위 eval | modality config를 체크포인트에서 읽음 — `--modality-config-path` 불필요 ✅ |
| RTC 기본 상태 | 위 eval 로그 | `[RTC] mode=none` ✅ |
| eval VRAM | 위 eval | 15.87 GiB — 학습과 동시 실행도 가능 |
| 호스트 RAM | `free -g` | 2,263 GB 중 2,048 GB 여유. worker 8×4=32 문제없음 |
| 디스크 | `df -h` | 61 TB 여유. 183 GB는 0.3% |

### 15.3 아직 확인하지 않은 것 (알고 시작하자)

| 항목 | 왜 남겨두나 |
|---|---|
| **30K step 완주** | 최장 45 step까지만 돌려봤다. 장시간 런에서만 나오는 문제(누수, Lustre 정체)는 열려 있다 |
| 중단 후 resume | `resume_from_checkpoint=True`가 항상 켜져 있으니 원리적으론 되지만 실제로 안 끊어봤다 |
| 실로봇 서빙 | `run_rldx_server`를 26차원 체크포인트로 띄워본 적 없다. **관측/출력 키가 `action.left_hand` → `action.left_hand_j14_16` 등으로 바뀐다**(§12 Q2) |
| chunk 40 장기 거동 | 40 step 스모크만 통과. RoPE 외삽의 성능 영향은 §13.7대로 열려 있다 |

### 15.4 실행

```bash
cd /NHNHOME/doyoung/RLDX-1
git checkout ffw-sh5-b200
bash /NHNHOME/doyoung/setup_rldx.sh all

# 파이프라인 확인 (20 step, 별도 디렉터리)
SMOKE=1 bash run_scripts/train/examples/finetune_rldx1_ffw_sh5_0826.sh

# 본 학습
bash run_scripts/train/examples/finetune_rldx1_ffw_sh5_0826.sh
```

체크포인트마다 평가:

```bash
for STEP in 10000 20000 30000; do
  pixi run -e rldx-sm100 python run_scripts/eval/open_loop_eval_verbose.py \
    --model-path ckpt/rldx1/finetuned/rldx1_ft_ffw_sh5_0826/rldx1_ft_ffw_sh5_0826/checkpoint-$STEP \
    --dataset-path dataset/rlwrld_demo_0826_holdout \
    --embodiment-tag GENERAL_EMBODIMENT \
    --traj-ids 0 1 2 3 4 5 6 7 8 --action-horizon 40 --steps 128 \
    --save-plot-path outputs/eval/holdout_${STEP}.png
done
```

> **`--traj-ids`는 0~8이다.** holdout 데이터셋에서 에피소드가 0부터 재번호되므로
> 원본 번호(3, 10, 17, …)를 주면 안 된다.
> **`--action-horizon 40`** — 학습과 맞춰야 한다. chunk 40이면 실행은 20(2.0초)이 기본 계획이다(§13.7).

---

## 16. Q&A 3차 (2026-08-26, 학습 직전)

### Q12. `peek_frames.py`는 무슨 원리로 박스나 송장을 관측하나? VLM을 부르나?

**아무것도 관측하지 않는다. VLM을 부르지 않는다.** §6.4에 자세히 썼다.
디코딩 → 타일링 → PNG 저장이 전부고, 검출기도 모델 호출도 없다.
"라벨이 어느 면에 있다"는 판단은 **그 PNG를 본 사람**이 한 것이다.

자동화하려면 이 PNG를 VLM에 입력으로 넣는 단계를 따로 붙여야 하고, 그건 아직 없다.

사용법·옵션은 §6.4의 표를 볼 것. 이번에 스크립트를 고쳐서 **타일마다 프레임 번호와
초를 찍고**, 콘솔에 샘플 시각과 간격을 출력하고, `--every-sec`로 균등 샘플링을
할 수 있게 했다.

### Q13. 8장 뽑으면 0.8초씩 증가해야 하는 것 아닌가? (10 fps니까)

**아니다. 8장은 등간격이 아니다.** 기본 샘플링은 에피소드 전체를 **비균등 비율**
(`0, 1/12, 1/6, 1/4, 1/3, 1/2, 2/3, 1`)로 나눈다. 그래서 ep30(35.9초)이면:

```
times(s): 0.0  2.9  5.9  8.9  11.9  17.9  23.9  35.8
gaps(s) :   2.9  3.0  3.0  3.0   6.0   6.0  11.9
```

앞쪽 3초, 뒤쪽 12초 간격이다. 에피소드 길이에 따라 간격도 달라진다.
**"8장"과 "0.8초"는 무관한 두 숫자다** — 후자는 execution horizon 8 스텝을 초로
환산한 값이고(§13.5), 전자는 콘택트 시트의 타일 수다.

0.8초 등간격을 원하면 `--every-sec 0.8`인데, 35.9초 에피소드에 **45장**이 나온다.
분기 판정에는 과하다.

**참고로 `--every-sec 2`로 다시 뽑아 보니 ep30의 전환 시점이 좁혀졌다:
16.0초에는 송장이 없고 18.0초에 나타난다.** (§14 Q8에서 `(11.9, 17.9]`로만
좁혔던 것을 2초 해상도로 다시 확인한 것.)

### Q14. chunk를 50으로 바꿔도 되나? n_action_step 20으로 2초 실행하려는데.

§13.7 전체가 답이다. 요약:

- **바꿀 수 있다.** `delta_indices=list(range(N))`과 `--action-horizon N`을 **같이**
  바꿔야 하고, 어긋나면 `assembly.py:304`가 시작 전에 막는다.
- **50도 실측상 돈다** — VRAM +1.6 GiB, 처리량 −15%.
- **다만 40을 권했고, 40으로 결정했다.** 2초 실행이라는 목표는 chunk가 아니라
  실행 개수가 정하므로 40에서도 그대로 달성되고, 40은 **논문이 실제로 학습시켜 본
  길이**(ALLEX)라 근거가 붙는다. 실행 비율도 20/40 = 0.50으로 ALLEX와 같아진다.
- **n_action_step = execution horizon 맞다. 학습에는 안 들어간다.** 학습은 늘 전체
  chunk를 예측하도록 하고, 몇 개 실행할지는 배포 시 클라이언트 결정이라 체크포인트에
  기록되지도 않는다. **같은 체크포인트로 20/10을 바꿔 시험할 수 있다.**
  예외는 `memory_stride`와 `rtc_training_max_delay` 둘뿐인데 우리는 둘 다 안 쓴다.

### Q15. RTC가 뭔가? 청크 궤적이 튀는 걸 막는 건가? 학습부터 바꿔야 하나?

**맞다.** 청크 경계 불연속 + 추론 지연을 같이 다룬다. §13.8에 자세히 썼다.

**학습부터 바꿀 필요는 없다.** 결정적인 근거: **논문이 인용한 RTC 문헌은 하나뿐이고,
그게 재학습이 필요 없는 쪽**이다.

> [15] Black, Galliker, Levine. *Real-time execution of action chunking flow
> policies.* NeurIPS 2025c → arXiv 2506.07339 = `guided`

학습형 RTC(2512.05964)는 **논문에 인용조차 되지 않는다** — 레포가 논문 범위를 넘어
추가로 구현해 둔 기능이다. 그러니 `--rtc-training-max-delay`는 0(기본)으로 두고,
배포에서 `--rtc-inference-mode guided`부터 시도하면 논문과 같은 구성이 된다.

`--rtc-inference-exec-horizon`은 **RTC가 켜졌을 때의 실행 호라이즌 `s`** 다.
RTC를 안 쓰면 무의미하고, 실행 개수는 클라이언트가 정한다.

### Q16. 그래서 subtask는 나누나? 첫 학습이니 그냥 해볼까?

**그냥 간다. subtask는 다음에.** §12 Q6-d의 선택지 A다.

이유를 한 번 더 정리하면:

1. **A는 추가 라벨링이 0이다.** subtask를 하려면 81개 영상을 사람이 훑어 구간을
   찍어야 한다.
2. **1차 결과 없이는 어디를 쪼개야 할지 모른다.** 실패가 "전 구간 어설픔"인지
   "특정 전환에서 멈춤"인지 봐야 쪼갤 지점이 정해진다.
3. **subtask는 배포 계약을 바꾼다.** `sub_task`로 학습하면 실행 중에 "지금 어느
   subtask인가"를 정해줄 상위 플래너가 필요해진다(§12 Q6-c). 1차에서 떠안을 복잡도가
   아니다.
4. 되돌리기 쉽다 — subtask 도입은 `episodes.jsonl`에 `sub_tasks`를 넣고 config
   한 줄을 바꾸는 일이라, **나중에 해도 지금 학습이 낭비가 되지 않는다.**

한 가지만 기억하면 된다: **1차 실패를 볼 때 "전환 지점에서 멈추는가"를 특히 보라.**
그게 subtask 도입 여부를 가르는 신호다.

### Q17. 학습 걸기 전에 마지막으로 논의할 게 뭔가?

§15가 그 답이다. 요약하면 **하나는 실제 오류였고, 하나는 미완이었다:**

1. **`--save-total-limit` 조합이 틀렸다** (§15.1). 지금은 `10000 / 5`로 삭제가
   일어나지 않는다.
2. **held-out 분할이 없었다** (§15.2). 만들었고 감사 통과했다.
3. `open_loop_eval`이 26차원에서 도는지 미검증이었다 → 확인했다.
4. wandb 키가 없다 → `USE_WANDB=0`(스크립트 기본값).

**남은 미확인은 §15.3에 적어 뒀다** — 30K 완주와 실로봇 서빙은 해보지 않았다.

### 15.5 ⚠ 정규화 통계가 held-out을 포함하고 있었다 (고침)

스모크 로그에 `Generating stats for .../rlwrld_demo_0826_train`이 찍혀서 통계가
새로 계산된 줄 알았는데, **아니었다.**

```python
# rldx/data/stats.py
def generate_stats(dataset_path):
    print(f"Generating stats for {dataset_path}")   # :118  ← 먼저 찍고
    ...
    if check_stats_validity(dataset_path, lowdim_features):
        return                                       # :127  ← 그다음 조기 반환
```

**로그를 먼저 찍고 나중에 건너뛴다.** 즉 저 줄은 "계산했다"는 증거가 아니다.
`split_holdout.py`가 merged의 `stats.json`을 그대로 복사해 뒀기 때문에, 학습 분할의
q01/q99가 **held-out 9개를 포함한 81 에피소드로 fit된 값**이었다.

정규화 상수가 held-out을 본 것이므로 누수다. 라벨이 새는 건 아니라 영향은 작지만
고치지 않을 이유도 없다.

**고친 내용:**
- `dataset/rlwrld_demo_0826_train/meta/stats.json`을 **72 에피소드 기준으로 재생성**했다.
  57차원 중 **30개가 바뀌었고 q99 최대 차이는 0.043 rad**다. 작지만 실재한다.
- `split_holdout.py`가 이제 `stats.json`을 **복사하지 않는다**(`modality.json`만 복사).
  없으면 `DatasetFactory._ensure_stats`가 분할별로 만든다.
- holdout에도 자기 9개 기준 `stats.json`을 만들어 뒀다. `open_loop_eval`은
  **데이터셋 통계를 쓰지 않고 체크포인트 통계로 정규화**하므로 이 값은 로더의
  assertion을 만족시키는 용도일 뿐 평가 수치에 영향이 없다.

### 15.6 스크립트 스모크 통과 — 그리고 논문과 정확히 일치하는 숫자 하나

`SMOKE=1 bash run_scripts/train/examples/finetune_rldx1_ffw_sh5_0826.sh` 결과:

```
[i] Trainable parameters: 2,375,979,600 (34.37%)
loss 0.2931 -> 0.1842 (20 step)
checkpoint-20 기록
🚀 Training completed!
```

**학습 파라미터 2,375,979,600 = 2,376.0M** 인데, 이건 논문 Table 6의
*"Full FT (backbone top-4) + Full FT (action model) → 2,376.0M"* 과 **정확히 같다.**
동결 설정(`--tune-top-llm-layers 4`, `tune_llm/tune_visual=False`)이 논문의
full-fine-tuning 구성과 일치한다는 독립적인 확인이다.

---

## 17. 1차 학습 결과 (2026-08-27)

30K를 완주했고 held-out으로 채점했다. **결론: `checkpoint-10000`이 최적이고,
그 뒤 20,000 스텝은 held-out을 개선하지 못했다.**

| step | held-out MSE | MAE | 최적 대비 | train loss |
|---|---|---|---|---|
| **10,000** | **0.02379** | **0.09208** | — | 0.0129 |
| 20,000 | 0.02653 | 0.09858 | +11.6% | 0.0034 |
| 30,000 | 0.02663 | 0.09985 | +11.9% | 0.0031 |

**train loss는 4배 좋아지는 동안 held-out은 나빠졌다.** 교과서적인 과적합이다.
그림: `outputs/eval/holdout_curve.png`.

### 17.1 ±15% 변동을 어떻게 넘어섰나 — 에피소드별로 짝지었다

부록 A.4대로 같은 체크포인트도 실행마다 MSE가 ±15% 흔들린다. 평균 차이 11.6%는
그 폭 안이라 **평균만으로는 단정할 수 없다.**

그래서 **9개 에피소드를 짝지어** 봤더니 **step 10,000이 9/9 전부에서 이긴다.**
샘플링 노이즈는 에피소드마다 독립적으로 흔들리므로, 한쪽이 9개 전부를 이길 확률은
동전 던지기라면 1/512다. **순서는 노이즈가 아니다.**

체크포인트를 비교할 때는 앞으로도 평균이 아니라 **에피소드별 짝비교**를 볼 것.

### 17.2 ⚠ 내 요약 스크립트에 버그가 있었다

`eval_holdout_all.sh`가 로그의 `Average MSE across all trajs` 줄을 읽었는데,
**그 줄은 궤적마다 한 번씩 찍히고 각 출력이 그 호출에서 본 궤적까지만 반영**한다.
`tail -1`로 읽으면 **평균이 아니라 마지막 궤적의 값**이 나온다.

처음 보고한 0.0368 / 0.0400 / 0.0408은 전부 **궤적 8의 값**이었다.
지금은 궤적별 줄을 직접 평균한다. 추세는 같았지만 절대값이 틀렸었다.

### 17.3 50K 연장은 중단했다

체크포인트가 남아 있어 `MAX_STEPS=50000`으로 resume을 걸었다가, 위 결과를 보고 껐다.
근거:

- **10K → 30K가 held-out을 하나도 개선하지 못했다.** 20K를 더 얹을 근거가 없다.
- resume은 **웜 리스타트**다 — `lr_lambdas`가 `[{}]`로만 저장돼 람다가 복원되지 않고
  새로 만든 50K 모양 코사인이 적용된다. step 30,000에서 LR이 **~0 → 3.77e-5로 되올라간다**
  (실측). 과적합 구간에서 LR을 다시 올리는 셈이다.

### 17.4 그래서 다음 레버는 데이터다

모델 용량도 스텝 수도 병목이 아니다. 81 에피소드에서 **10,000 스텝(45 epoch)이면
이 데이터로 배울 건 다 배웠다**는 뜻이다.

§6.3의 순서대로 간다 — 초기 송장 위치 분포를 세고, 부족한 분기를 집중 수집한다.

---

# 부록 A. 운영 지식 — `ffw_sh5_0812` 시절에 얻은 것

여기부터는 원래 `RLDX-tuning.md`에 있던 내용이다. 대상 데이터셋은 이전
스모크용 `ffw_sh5_0812`(7 에피소드)지만, **아래 내용은 데이터셋과 무관하게
지금도 그대로 유효**해서 이 문서로 옮겼다. `RLDX-tuning.md`는 삭제했다.

옮기지 않은 것: `ffw_sh5_0812` 자체의 데이터 특성(죽은 에피소드 0,
플레이스홀더 `"1611"`), 54차원 modality 구성, chunk 16 결정 —
전부 본문 §4·§13이 대체한다.

## A.1 ModalityConfig — 설정이 아니라 계약이다

`rldx/configs/data/ffw_sh5_rev1_config_wholebody.py`를 짜면서 든 생각.
이건 하이퍼파라미터가 아니라 **세 곳을 묶는 계약**이다: `meta/modality.json` ↔ Python
`ModalityConfig` ↔ 체크포인트에 저장된 프로세서.

**순서가 곧 의미다.** `modality_keys`의 순서가 모델 입력 벡터의 concat 순서이고
(`processing_rldx.py:469`), 출력에서 다시 관절 그룹으로 잘라내는 순서이기도 하다
(`processing_rldx.py:352-357`). 한 번 정한 순서는 학습·배포·실기가 전부 같아야 한다.
바꾸면 **에러 없이** 로봇 팔에 손가락 명령이 들어간다.

**체크포인트에 구워진다.** 배포 시 `run_rldx_server.py --model-path`는 modality config를
`.py`에서 읽지 않고 체크포인트에 저장된 프로세서에서 가져온다. `--modality-config-path`로
JSON을 넘기는 분기는 `ReplayPolicy`(데이터 재생) 전용이다.
즉 **잘못 정하고 학습하면 추론 시점에 못 고치고 재학습해야 한다.**
lr이나 batch size와 성격이 완전히 다른 이유다.

**조용히 틀리는 지점.** `action_configs`는 `modality_keys`와 위치로 `zip`되는데,
길이가 다르면 에러 대신 짧은 쪽에서 잘린다. 초안의 6개를 4개로 맞춘 게 이 이유다.
그대로 뒀으면 앞 4개만 쓰이면서 학습이 그냥 돌아갔을 것이다.

한 줄 요약: **modality config는 로봇의 관절 순서를 모델 가중치에 새기는 행위다.**

---

---

## A.2 ABSOLUTE vs RELATIVE

### "보통 다 absolute"는 사실이 아니다

`rldx/configs/data/` 17개 파일 전수 조사:

| rep | 쓰이는 곳 |
|---|---|
| **DELTA** | robocasa, libero, simpler, calvin, droid — 전부 **EEF(엔드이펙터) 제어** 데이터셋 |
| **ABSOLUTE** | 관절 공간 타겟, 그리고 손/그리퍼 |
| **RELATIVE** | `embodiment_configs.py`의 unitree_g1 팔뿐 (레포 전체 5곳) |

### DELTA는 변환이 아니다

코드를 따라가면 설정 파일 밖에서 `ActionRepresentation.DELTA`가 등장하는 곳은
`sharded_single_step_dataset.py:77` 하나뿐이고, 거기서도 RELATIVE와 묶여서
"범위 밖 인덱스를 clamp 대신 0으로 패딩"하는 데만 쓰인다.
실제 변환은 `state_action_processor.py:365`에서 **RELATIVE에 대해서만** 일어난다.

즉 DELTA는 *"이 데이터셋은 이미 델타로 저장돼 있다"*는 **선언**이지
*"델타로 바꿔달라"*는 요청이 아니다. 그래서 실질적 선택지는 **ABSOLUTE vs RELATIVE 둘뿐**이다.

### 우리 데이터는 실제로 absolute다 (실측, ffw_sh5_0812 기준)

`episode_000001.parquet`을 직접 뜯어서 확인:

```
left_arm   action range [-2.450, 0.743] rad     ← 관절 전체 범위
left_hand  action range [-1.571, 1.571]         ← 정확히 ±π/2, 관절 리밋
           |action[t] - state[t]|   0.0249
left_arm   |action[t] - state[t+1]| 0.0151      ← 더 가까움
```

action이 state를 **1프레임 앞서 끌고 가는 타겟**이다.
델타였다면 range가 0 근처여야 하는데 전혀 아니다.

### 결정: ABSOLUTE 유지

RELATIVE는 데이터를 바꾸는 게 아니라 **학습 시 재파라미터화**(`action_chunk - 현재 state`)라서
우리도 선택 가능하긴 하다. 그래도 ABSOLUTE로 가는 이유:

1. **가장 가까운 선례가 ALLEX다.** `midtrain_allex_data_config.py` — RLWRLD 자체 휴머노이드,
   손가락 포함 관절 공간 제어, **6개 그룹 전부 ABSOLUTE**. 우리와 문제 형태가 같다.
2. **unitree_g1의 논리는 우리에게 안 맞는다.** g1은 팔 RELATIVE / 손 ABSOLUTE인데,
   손을 ABSOLUTE로 둔 근거가 주석에 있다 — *"G1 hand is controlled by binary signals like a gripper"*.
   우리 손은 한쪽 20 DOF 연속 관절이라 해당 없다.
3. **움직이는 부품이 는다.** RELATIVE는 `relative_stats.json`을 다시 생성해야 하고,
   추론 시 `decode_action`이 절대 타겟 복원을 위해 현재 state를 요구한다.
   지금은 "학습이 걸리나 / 추론이 되나"를 보는 단계라 변수를 늘리지 않는다.

참고로 `use_relative_action` 기본값이 `True`(`rldx/configs/model/rldx.py:76`)라,
RELATIVE로 선언하면 실제로 작동은 한다. 나중에 비교 실험은 가능하다.

---

---

## A.3 같은 output_dir에 두 번 실행하면 조용히 이어서 학습한다

`experiment.py`가 **항상** `trainer.train(resume_from_checkpoint=True)`로 부른다.
즉 output_dir에 체크포인트가 남아 있으면 새 학습이 아니라 **재개**다. 그것 자체는 의도된
동작이지만, 재개 시 **CLI 인자 일부가 무시된다**는 부작용이 있다.

실제로 겪은 사고: `SMOKE=1`로 20스텝을 돌린 뒤 같은 디렉터리에 `--save-steps 5000`으로
10000스텝을 걸었더니, 체크포인트가 **20스텝마다** 저장되면서
`--save-total-limit 5` 때문에 9920~10000만 남고 5000은 사라졌다.

원인 사슬:

1. `DefaultFlowCallback`은 저장 시점을 `args.save_steps`가 아니라 **`state.save_steps`**로
   판단한다 (`transformers/trainer_callback.py`: `state.global_step % state.save_steps`).
2. 새 학습에서는 HF가 `state.compute_steps(args, max_steps)`로 args에서 유도한다.
3. **재개할 때는** `self.state = TrainerState.load_from_json(...)`로 상태를 통째로
   교체하고 다시 유도하지 않는다 (`transformers/trainer.py`). 그래서 스모크 런의
   `save_steps=20`이 그대로 살아남는다. `compare_trainer_and_checkpoint_args`는 경고만 한다.
4. `logging_steps` / `eval_steps`도 같은 경로로 이전 값을 물려받는다.

부수적으로 log_history와 step 카운터도 이어져서, 10000스텝 런의 첫 두 로그가
스모크 런의 loss(0.8992 / 0.7331)로 남아 있었다.

**대응 (둘 다 적용됨):**

- `rldx/experiment/trainer.py`의 `RestoreStepCadenceCallback` — `on_train_begin`은
  재개 시 상태 로드 **이후**에 실행되므로, 거기서 `save_steps`/`logging_steps`/`eval_steps`를
  CLI 값으로 되돌리고 무엇을 덮어썼는지 출력한다.
- 학습 스크립트에서 `SMOKE=1`은 `<CKPT_NAME>_smoke`라는 **별도 디렉터리**를 쓴다.
  스모크 런이 실전 런을 오염시킬 경로 자체를 없앴다.
- 그리고 재개가 일어날 상황이면 스크립트가 시작 시 경고한다.

새 학습을 원하면 `CKPT_NAME`을 바꾸거나 기존 디렉터리를 옮길 것.

---

## A.4 추론이 되는지 — open-loop eval

`rldx/eval/open_loop_eval.py`가 서버 없이 체크포인트를 직접 로드해서, 데이터셋의 실제 관측을
넣고 정답 action과 비교한다(MSE/MAE + 관절별 플롯).

```bash
pixi run -e rldx-sm100 python run_scripts/eval/open_loop_eval_verbose.py \
    --model-path ckpt/rldx1/finetuned/rldx1_ft_ffw_sh5_wholebody/rldx1_ft_ffw_sh5_wholebody/checkpoint-10000 \
    --dataset-path dataset/ffw_sh5_0812 \
    --embodiment-tag GENERAL_EMBODIMENT \
    --traj-ids 1 2 3 4 5 6 --action-horizon 16 --steps 64 \
    --save-plot-path outputs/eval/openloop_ckpt10000.png
```

궤적 0은 뺀다 — 죽은 녹화라 정답이 평평하고 평균 MSE만 낮춘다(함정 1, §6).
1~6 중에서도 **3과 6만 손가락이 움직인다**(42/57 관절). 나머지는 팔만 움직이므로,
손 예측을 보려면 3·6의 플롯을 볼 것.

| ep | frames | 움직인 관절 | 최대 변화폭 |
|---|---|---|---|
| 0 | 49 | **0/57** ← 죽은 녹화 | 0.0000 |
| 1 | 200 | 14/57 (팔만) | 0.8309 |
| 2 | 201 | 14/57 (팔만) | 1.3612 |
| **3** | 201 | **42/57 (팔+손)** | 1.1874 |
| 4 | 200 | 14/57 (팔만) | 0.8326 |
| 5 | 201 | 14/57 (팔만) | 0.4671 |
| **6** | 201 | **42/57 (팔+손)** | 1.1455 |

**`--save-plot-path`는 반드시 레포 안 상대경로로 줄 것.** 생략하면 상위 스크립트가
`/tmp/open_loop_eval/traj_{traj_id}.jpeg`에 쓰는데(`open_loop_eval.py:238`),
`/tmp`는 이 서버에서 세션이 바뀌면 통째로 사라진다(§1).
`outputs/`는 이미 `.gitignore`에 있어 산출물을 두기 맞고, 영구 볼륨 안이라 살아남는다.

래퍼(`run_scripts/eval/open_loop_eval_verbose.py`)가 상위 스크립트를 그대로 호출하되
네 가지를 더 한다:

| | 왜 |
|---|---|
| INFO 로깅 강제 | 안 그러면 MSE/MAE가 통째로 삼켜진다 (아래 함정 2) |
| 궤적별 파일명 | 상위 스크립트는 여러 궤적에 **같은 경로**를 재사용해 덮어쓴다 (함정 3) |
| `gc.collect()` 후 VRAM 측정 | 궤적마다 모델을 새로 로드해서, 안 지우면 피크가 2배로 보인다 |
| 출력 디렉터리 자동 생성 | |

**로컬에 없는 `--model-path`는 Hub repo id로 간주된다.** `policy_loader.py:152`의
`_resolve_model_dir`이 경로가 없으면 `snapshot_download`로 넘어간다. 그래서 공개된
체크포인트는 그대로 쓸 수 있다:

```bash
--model-path learner1119/RLDX-1-FT-ffw-sh5
```

반대로 **로컬 경로가 사라진 경우**(대개 `--save-total-limit`이 지웠거나 오타) 그 경로가
repo id로 넘어가서 `HFValidationError: Repo id must be in the form ...`라는 엉뚱한 에러가
난다. 래퍼가 둘을 구분해서, 후자면 그 디렉터리에 실제로 남아 있는 체크포인트를 보여준다.

**`--modality-config-path`가 없는 게 정상이다.** `open_loop_eval.py:317`의
`policy.get_modality_config()`가 **체크포인트에 저장된 프로세서**에서 꺼내온다.
§2에서 말한 "체크포인트에 구워진다"의 실제 증거다.

실측 (궤적 1, 2 / 64스텝):

| 체크포인트 | 궤적 1 MSE / MAE | 궤적 2 MSE / MAE |
|---|---|---|
| 20스텝 (스모크) | 0.04499 / 0.14648 | 0.04576 / 0.14857 |
| **10000스텝** | **0.00718 / 0.03328** | **0.00727 / 0.03133** |

`[VRAM] peak allocated (per trajectory): 15.83 GiB` (둘 다 동일).

MSE가 6배 넘게 줄었고 MAE는 0.146 rad(≈8°) → 0.033 rad(≈**1.9°**)로 떨어졌다.
아래에 적은 실행 간 변동폭(±15%)보다 훨씬 큰 차이라 노이즈가 아니다.
플롯에서도 예측이 정답 곡선을 따라간다 — 20스텝 때 제멋대로 방황하던 것과 대비된다.

다만 10000스텝은 1253프레임에 대해 약 510 에폭이라 과적합 구간이다.
"학습·추론 경로가 도는가"를 확인하는 데는 충분하지만 정책 성능의 근거는 못 된다.

**같은 체크포인트라도 실행마다 MSE가 달라진다.** 궤적 1을 다섯 번 돌린 값이
0.0363 / 0.0387 / 0.0412 / 0.0427 / 0.0450 로 ±15% 흔들린다. flow-matching 샘플링에
시드가 고정돼 있지 않아서다. 체크포인트끼리 비교할 때는 이 폭보다 큰 차이인지 볼 것.

함정 넷:

1. **`--traj-ids 0`을 쓰지 말 것.** 에피소드 0은 상수 데이터라(§6) 정답이 평평한 직선으로
   그려지고, 모델이 고장난 것처럼 보인다. 1~6을 쓸 것.
2. **상위 스크립트를 직접 쓰면 MSE/MAE가 안 찍힌다.** `main()`이
   `logging.basicConfig(level=INFO)`를 부르지만 그 시점엔 transformers가 이미 루트 로거를
   설정해둬서 `basicConfig`가 no-op이 된다. 결과가 전부 `logging.info()`라 통째로 삼켜지고,
   플롯만 나오고 아무 일도 안 한 것처럼 보인다. `force=True`로 덮어써야 하며
   `open_loop_eval_verbose.py`가 그걸 한다.
   직접 쓴다면 반드시 **파일로** 만들 것 — `pixi run ... python -c "..." --model-path ...`
   형태는 인자가 tyro까지 전달되지 않는다.
3. **`--save-plot-path`를 주면 궤적끼리 덮어쓴다.** 기본 경로는
   `traj_{traj_id}.jpeg`로 id가 들어가지만, 명시하면 그 경로 하나가 루프의 모든 궤적에
   재사용된다(`open_loop_eval.py:348` → `:238`). `--traj-ids 1 2`면 궤적 2가 궤적 1을
   덮어써서 결과가 조용히 하나만 남는다. 래퍼는 궤적별로 `_traj{id}`를 붙여 저장한다.
4. **플롯이 800×21600 픽셀이다** (관절 54개 세로 스택). 뷰어에 따라 안 열리니
   `PIL`로 잘라서 보는 게 낫다.

플롯 읽는 법: 파란선 = 정답, 주황선 = 예측, 빨간 점 = 추론 시점(`--action-horizon` 간격).
빨간 점 사이 구간이 한 번의 action chunk를 open-loop로 실행한 결과다.

---

## A.5 서빙 루프

실기 배포는 `rldx/eval/run_rldx_server.py --model-path ...`. msgpack으로
`ping` / `get_action` / `reset` 엔드포인트를 연다. 관측 키는 `{modality}.{key}` 규약이라
이 로봇은 `video.cam_head`, `state.left_arm`, `state.right_arm`, `state.left_hand`,
`state.right_hand`, `annotation.human.task_description`. 자세한 건 `docs/inference_server.md`.

---

## A.6 배포 대상 GPU

| GPU | arch | VRAM | 학습 | 추론 | 환경 |
|---|---|---|---|---|---|
| B200 | `sm_100` | 183 GiB | ✅ | ✅ | `pixi run -e rldx-sm100` (§1) |
| RTX 5090 | `sm_120` | 32 GiB | ⚠️ LoRA만 | ✅ | `pixi run -e rldx` (업스트림 기본) |
| A100 / H100 | `sm_80/90` | 40–80 GiB | ✅ | ✅ | `uv run` (업스트림 기본) |

**추론은 5090에서 여유롭다 — 추정이 아니라 실측이다.** `open_loop_eval`을 배치 1로 돌리면서
`torch.cuda.max_memory_allocated()`로 잰 값:

```
체크포인트 (bf16, 6.9B)      12.88 GiB
추론 피크 (배치 1, 실측)      15.83 GiB
RTX 5090 VRAM                32 GiB      → 약 2배 여유
```

가중치 12.88 GiB에 활성화가 3 GiB 남짓 붙는다. `video_length=4`(376×672 → 160×256으로
리사이즈), `n_cog_tokens=64`, denoising 4스텝이라 활성화가 작다.
(B200에서 잰 값이지만 bf16 경로가 동일하므로 5090도 비슷하다.)

**학습은 5090에서 빠듯하다.** full fine-tune은 가중치 13 GiB + AdamW optimizer state
(fp32 모멘텀 2개 = 파라미터당 8바이트 ≈ 55 GiB)라 32 GiB에 안 들어간다.
5090에서 학습하려면 LoRA를 써야 한다:

```bash
--action-model-use-lora --backbone-use-lora --backbone-lora-num-layers 4
```

**5090에서 학습 스크립트를 돌릴 때는 런처를 명시할 것.** 자동 감지는 `.pixi/envs/rldx-sm100`
유무만 보므로 5090에서는 `uv run`(cu126, sm_90까지)으로 떨어져서 실패한다:

```bash
RUN_PREFIX="pixi run -e rldx" bash run_scripts/train/examples/finetune_rldx1_ffw_sh5.sh
```

`rldx/inference/`의 가속 경로(torch.compile / CUDAGraph / Triton)를 쓸 때만
`pixi run -e inference`(cu130)가 따로 필요하다.

---

## A.7 배포한 체크포인트와 라이선스

[`learner1119/RLDX-1-FT-ffw-sh5`](https://huggingface.co/learner1119/RLDX-1-FT-ffw-sh5) (public).
10000스텝 체크포인트에서 **추론용 산출물만** 올렸다 — 가중치 3샤드, `processor/`,
`experiment_cfg/`, LICENSE, 모델 카드. 총 13G.

DeepSpeed 옵티마이저 상태(`global_step10000/`, 48G)는 **뺐다.** 원본 학습을 재개할 때만
쓰이고 추론이나 추가 파인튜닝에는 필요 없다. 체크포인트 61G 중 대부분이 이것이다.

**라이선스는 선택의 여지가 없다.** `RLDX-1-PT`가 RLWRLD Model License v1.0이고 파인튜닝
결과물은 명시적으로 "Derivative Model"이라, §3.3 share-alike에 따라 **같은 라이선스로만**
배포할 수 있다. §3.4는 라이선스 전문 사본 동봉을, §3.2는 출처 표시를 요구한다.
MIT/Apache로 올리면 위반이다.

모델 카드에는 한계를 명시했다 — 7 에피소드에 ~510 에폭이라 과적합, 언어 조건이
placeholder `"1611"`이라 지시 이행 능력 없음, head 카메라만 사용, 손가락은 2개
에피소드에서만 움직임. 받아 쓰는 쪽이 성능 지표로 오해하면 안 되는 물건이다.
