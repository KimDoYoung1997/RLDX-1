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

**에피소드 0은 죽은 녹화다.** 49프레임인데 **57차원 전부가 완전한 상수**(변화폭 0.0000)다.
`open_loop_eval`을 `--traj-ids 0`으로 돌리면 정답이 평평한 직선으로 나와서 모델이 망가진
것처럼 보인다. 평가는 반드시 움직이는 궤적(1~6)으로 할 것.

```bash
uv run python -c "
import pandas as pd, numpy as np
for ep in range(7):
    a = np.stack(pd.read_parquet(f'dataset/ffw_sh5_0812/data/chunk-000/episode_{ep:06d}.parquet')['action'].values)
    r = a.max(0) - a.min(0)
    print(f'ep{ep}: frames={len(a):4d} 움직인 관절={int((r>1e-4).sum()):2d}/57 최대변화폭={r.max():.4f}')
"
```

**`head`와 `lift`는 한 번도 안 움직인다.** 7개 에피소드 전체에서 변화폭 0.0000이다.
§3에서 제외한 결정이 데이터로 사후 검증된 셈이다 — 넣었더라도 상수를 학습했을 것이다.

| 그룹 | dims | 움직인 관절 | 최대 변화폭 |
|---|---|---|---|
| left_arm | 7 | 7 | 1.3612 |
| right_arm | 7 | 7 | 1.1252 |
| left_hand | 20 | 14 | 0.9425 |
| right_hand | 20 | 14 | 0.9425 |
| head | 2 | **0** | **0.0000** |
| lift | 1 | **0** | **0.0000** |

손도 20개 중 14개만 움직인다. 나머지 6개는 연동 관절이거나 이 태스크에서 안 쓰인 것으로 보인다.

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

## 7. 평가 (eval) 하는 법

### 7.1 학습이 걸리는지 — 스모크 런

`SMOKE=1`이 20스텝/batch 8로 줄여준다. `USE_WANDB=0`은 wandb 로그인 없이 돌린다.

```bash
SMOKE=1 USE_WANDB=0 bash run_scripts/train/examples/finetune_rldx1_ffw_sh5.sh
```

봐야 할 것 — loss가 **유한하고**, `grad_norm`이 폭주하지 않고, 체크포인트가 써지는가.
20스텝짜리 loss 값 자체에는 의미가 없다.

```
{'loss': 0.8992, 'grad_norm': 3.9186, 'learning_rate': 6.227e-05}
{'loss': 0.7331, 'grad_norm': 2.7253, 'learning_rate': 6.819e-07}
{'train_runtime': 82.2, 'train_loss': 0.8162}
→ ckpt/.../checkpoint-20/  (safetensors 3샤드 + processor + experiment_cfg)
```

체크포인트 경로가 `output_dir/run_name/` 으로 **한 겹 더 들어간다**:
`ckpt/rldx1/finetuned/<CKPT_NAME>/<RUN_NAME>/checkpoint-<step>`.

### 7.1.1 같은 output_dir에 두 번 실행하면 조용히 이어서 학습한다

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

### 7.2 추론이 되는지 — open-loop eval

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

### 7.3 서빙 루프

실기 배포는 `rldx/eval/run_rldx_server.py --model-path ...`. msgpack으로
`ping` / `get_action` / `reset` 엔드포인트를 연다. 관측 키는 `{modality}.{key}` 규약이라
이 로봇은 `video.cam_head`, `state.left_arm`, `state.right_arm`, `state.left_hand`,
`state.right_hand`, `annotation.human.task_description`. 자세한 건 `docs/inference_server.md`.

---

## 8. 배포 대상 GPU

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

## 9. 배포한 체크포인트

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

---

## 10. TODO

- **tactile을 physics 모달리티로 학습.** `modality.json`에
  `"tactile": {"left_hand": {"start": 0, "end": 45}, "right_hand": {"start": 45, "end": 90}}`을 넣고
  대응하는 `ModalityConfig`를 추가한 뒤
  `--use-physics --physics-keys tactile --physics-dims 90 --new-param-warmup-steps 500`.
  플래그는 학습 스크립트에 주석으로 이미 stub 해뒀다.
  새 파라미터가 추가되므로 `--new-param-warmup-steps`가 필요하다.
- **손목 카메라 추가** — head-only 성능이 자리잡은 뒤에.
- **placeholder task 문자열 교체** (§6).
- **ABSOLUTE vs RELATIVE 비교 실험** — 데이터가 충분히 모이면.
