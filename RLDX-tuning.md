# RLDX-1 튜닝 노트 — FFW-SH5

FFW-SH5(rev1) 실로봇 데이터로 RLDX-1을 파인튜닝하면서 내린 결정들과 그 근거.
"어떻게 하는가"는 [CLAUDE.md](CLAUDE.md)에, "왜 그렇게 정했는가"는 이 문서에 둔다.

대상 데이터셋: `learner1119/ffw_sh5_0812` (LeRobot v2.1, 7 에피소드 / 1253 프레임 / 10 fps).
VLA 성능을 노린 데이터가 아니라 **학습이 걸리는지 / 추론이 되는지 확인하려고 만든 스모크용**이다.

---

## 1. 하드웨어: B200은 `sm_100`이라 uv 환경으로는 학습이 안 된다

가장 먼저 부딪힌 벽. `uv sync`가 설치하는 `torch==2.7.0+cu126`은 CUBIN이 `sm_90`(Hopper)에서 끝난다.
B200은 compute capability **(10, 0) = `sm_100`**이라 커널이 아예 없다.

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

증상이 헷갈리는 이유는 **에러가 나는 지점이 모델이나 데이터와 무관하다**는 것이다.
실제로는 modality config 로드, 체크포인트 다운로드, 프로세서 생성, 데이터셋 팩토리까지
전부 정상 통과한 뒤 **첫 CUDA 커널 호출**(`rldx/data/dataset/factory.py`의 NCCL `barrier()`)에서
죽는다. 그래서 config 문제처럼 보이지만 아니다.

확인 방법:

```bash
uv run python -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.get_device_capability(0))"
# ['sm_50', ..., 'sm_90']   (10, 0)   ← 불일치
```

### 왜 레포의 기존 pixi 환경으로는 안 되는가

`pixi.toml`에 Blackwell 경로가 이미 있지만 그건 **`sm_120`(RTX 5090, 컨슈머 Blackwell)** 전용이다.
B200의 `sm_100`(데이터센터 Blackwell)과는 **별개의 CUBIN 타겟**이라, 한쪽으로 빌드한 커널은
다른 쪽에서 실행되지 않는다. 이름이 둘 다 "Blackwell"이라 헷갈리기 쉬운 지점.

### 해결: `rldx-sm100` 환경 추가

`pixi.toml`에 기존 `rldx` feature를 미러링한 `rldx-sm100`을 추가했다. 기존 `rldx`(sm_120)
환경은 건드리지 않았다. 실질적 차이는 두 줄뿐이다:

```toml
[feature.rldx-sm100.tasks.install-flash-attn]
env = { FLASH_ATTN_CUDA_ARCHS = "100", ... }   # rldx는 "120"

[feature.rldx-sm100.activation]
env = { TORCH_CUDA_ARCH_LIST = "sm_100" }      # rldx는 "sm_120"
```

**torch는 재빌드가 필요 없다.** `torch==2.8.0+cu128` 휠에 sm_100 CUBIN이 이미 들어 있다.
소스 빌드가 필요한 건 flash-attn뿐이고, flash-attn 2.8.3의 기본 arch 목록이
`"80;90;100;120"`이라 `100`은 정식 지원 타겟이다 — 패치 없이 arch 선택만 하면 된다.
하나만 빌드하면 4-arch 기본값 대비 빌드 시간이 약 1/4로 준다.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi install -e rldx-sm100
pixi run -e rldx-sm100 postinstall          # flash-attn 소스 빌드, ~10분
pixi run -e rldx-sm100 python -c "import torch; print(torch.cuda.get_arch_list())"
```

학습 스크립트는 `RUN_PREFIX`로 런처를 뺐고 기본값이 `pixi run -e rldx-sm100`이다.
sm_80/sm_90 장비에서는 `RUN_PREFIX="uv run"`.

> 탈출구: `RLDX_ATTN_IMPL=sdpa` (`rldx/model/modules/backbone/adapter.py:18`)로 flash-attn을
> 우회할 수 있다. 어떤 실패가 flash-attn 탓인지 가릴 때 쓴다.

---

## 2. ModalityConfig — 설정이 아니라 계약이다

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

## 3. 사용할 modality: cam_head + 양팔/양손

`observation.state`와 `action` 둘 다 57차원이고 관절 순서가 동일하다:

| 슬라이스 | 원본 이름 | modality key | 차원 |
|---|---|---|---|
| `0:7` | `arm_l_joint1..7` | `left_arm` | 7 |
| `7:14` | `arm_r_joint1..7` | `right_arm` | 7 |
| `14:34` | `finger_l_joint1..20` | `left_hand` | 20 |
| `34:54` | `finger_r_joint1..20` | `right_hand` | 20 |
| `54:56` | `head_joint1..2` | — 제외 | 2 |
| `56:57` | `lift_joint` | — 제외 | 1 |

`head`/`lift`는 `modality.json` 슬라이스에서 빼는 것만으로 로더 단계에서 버려진다.
Python config에서도 같이 빼야 한다 (양쪽이 안 맞으면 stats 슬라이싱이 어긋난다).

카메라는 `cam_head` 하나만 쓴다. `cam_wrist_left` / `cam_wrist_right`도 데이터에 있지만
head-only 성능을 먼저 확인한 뒤에 붙인다.
`observation.tactile`(90차원 = 양손 × 5센서 × 9탁셀)도 있으나 이번엔 미사용 (§6).

---

## 4. chunk size는 16 — 사전학습을 존중

`delta_indices`는 시간축 요청이다. state의 `[0]`은 "현재 프레임만",
action의 `range(16)`은 "t부터 t+15까지".

레포의 다른 로봇 config들도 대체로 state `[0]` / action `range(16)`이고,
사전학습 mix인 `pt_data_config.py`도 16이다. **RLDX-1-PT가 16으로 학습됐을 가능성이 높으니
맞춰 간다.** 40을 쓰는 `midtrain_allex_data_config.py`는 문서에 `--action-horizon 40`을
명시적으로 요구한다고 적혀 있다.

### 굳이 바꾼다면

`delta_indices=list(range(30))`만 바꾸면 **안 된다.** 이 에러로 죽는다:

```
action_horizon=16 disagrees with modality_config for embodiment 'general_embodiment':
action.delta_indices has length 30. Fix either --action-horizon or the modality config so they match.
```

`rldx/experiment/assembly.py:289`의 `_validate_action_horizon_matches_modality`가 fail-fast로
잡는다. **두 군데를 같이** 바꿔야 한다 — config의 `delta_indices`와 CLI의 `--action-horizon 30`.

바꿔도 **파라미터 shape은 하나도 안 변한다.** 액션 인코더는 학습된 positional embedding이
아니라 sinusoidal PE를 쓰고(`embodiment_conditioned_mlp.py:192`), MSAT는 RoPE를
`max_seq_len`까지 미리 만들어 둔다(`action_model/ops.py:85`). horizon은 런타임 시퀀스
길이일 뿐이라 사전학습 가중치가 그대로 전이된다. 프로세서의 `max_action_horizon`도
override whitelist에 있어(`processing_rldx.py:759`) CLI 값이 전달된다.

그럼에도 고려할 점:

- 이 데이터는 **10 fps**라 30 스텝 = **3.0초 open-loop** (16 스텝은 1.6초). 손가락 20 DOF까지
  3초를 open-loop로 미는 건 길다.
- **예측 chunk를 늘리는 것과 실행 chunk를 늘리는 건 다른 문제다.** 보통 30개 예측하고 앞의
  N개만 실행한다 (`--rtc-inference-exec-horizon`).
- 사전학습이 horizon 16으로 돌았다면 뒤쪽 14 스텝은 처음 보는 구간이다. 7 에피소드로는 못 메운다.

---

## 5. ABSOLUTE vs RELATIVE

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

### 우리 데이터는 실제로 absolute다 (실측)

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

## 6. 데이터 자체에서 눈에 걸린 것

**언어 annotation이 `"1611"`이다.** `meta/tasks.jsonl`이
`{"task_index": 0, "task": "1611"}` 하나뿐이라, VLA가 받는 언어 조건이 자연어 지시가 아니라
ID 문자열이다. 학습은 기계적으로 돌아가지만 언어 grounding은 의미가 없다.
`meta/tasks.jsonl`의 `task`와 `meta/episodes.jsonl`의 `tasks` 필드를 고치면 되는 한 줄짜리 수정.

**손가락이 명령을 못 따라간다.** `left_hand`의 `|state[t+1] - state[t]|`는 0.00084인데
`|action - state|`는 0.0168 — 약 1도의 오프셋이 계속 남아 있다. 손가락이 뻣뻣하거나 느린 건지,
명령/측정 캘리브레이션 오프셋인지는 데이터만으론 구분이 안 된다.
스모크 테스트에는 영향 없지만, 나중에 손 동작이 "학습은 됐는데 실제로 안 움직인다"로
나오면 여기가 출발점이다.

**에피소드가 짧고 적다.** 7개, 총 1253 프레임, 그중 하나는 49 프레임.
horizon이 길어질수록 유효 시작 인덱스가 줄어든다.

---

## 7. TODO

- **tactile을 physics 모달리티로 학습.** `modality.json`에
  `"tactile": {"left_hand": {"start": 0, "end": 45}, "right_hand": {"start": 45, "end": 90}}`을 넣고
  대응하는 `ModalityConfig`를 추가한 뒤
  `--use-physics --physics-keys tactile --physics-dims 90 --new-param-warmup-steps 500`.
  플래그는 학습 스크립트에 주석으로 이미 stub 해뒀다.
  새 파라미터가 추가되므로 `--new-param-warmup-steps`가 필요하다.
- **손목 카메라 추가** — head-only 성능이 자리잡은 뒤에.
- **placeholder task 문자열 교체** (§6).
- **ABSOLUTE vs RELATIVE 비교 실험** — 데이터가 충분히 모이면.
