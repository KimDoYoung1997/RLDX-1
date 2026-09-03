# Subtask 조건화 학습 — 무엇이 바뀌었나

`rldx1_ft_ffw_sh5_0826`(whole-sentence 베이스라인)과
`rldx1_ft_ffw_sh5_0826_subtask`(subtask 조건) 사이의 정확한 차이를 기록한다.

## 한 줄 요약

**학습 코드 변화는 0줄이다.** 바뀐 것은 데이터(parquet의 `task_index` 컬럼)와
실행 스크립트의 기본값 2줄뿐이다.

## 왜 코드가 안 바뀌나 — 메커니즘

로더가 원래부터 언어 조건을 **프레임 단위**로 만든다:

1. `meta/tasks.jsonl`을 `{task_index: 문장}` 맵으로 읽고
   (`rldx/data/dataset/lerobot_episode_loader.py:178`),
2. 에피소드 parquet의 `task_index` 컬럼을 **행(프레임)마다** 문장으로 변환한다
   (`lerobot_episode_loader.py:350`).

베이스라인 데이터는 `task_index`가 전 프레임 0(전체 문장 하나)이라 조건이
상수였을 뿐이다. 컬럼 값이 프레임 중간에 바뀌면 각 학습 윈도우는 자기 시작
시점의 subtask 문장을 조건으로 받는다. 즉 **조건화 방식의 변경이 순수 데이터
변경으로 표현된다.** modality config(`ffw_sh5_rev1_config_used26_h40.py`)와
`meta/modality.json`(`human.task_description → task_index`)도 그대로다.

## 정확한 diff 목록

| 항목 | 베이스라인 | subtask | 바꾼 주체 |
|---|---|---|---|
| 학습 진입점/트레이너/로더 | `rldx/experiment/launch_train.py` 외 | **동일 (무수정)** | — |
| modality config | `ffw_sh5_rev1_config_used26_h40.py` | **동일 (무수정)** | — |
| 하이퍼파라미터 | batch 128 · lr 1e-4 cosine · warmup 5% · wd 1e-5 · state_dropout 0.3 · LLM 상위 4층 · video 4프레임 · horizon 40 · 30k steps · 10k 저장 | **동일** | — |
| 데이터셋 | `dataset/rlwrld_demo_0826_train` | `dataset/rlwrld_demo_0826_train_subtask` (복사본) | `apply_subtasks.py` |
| parquet `task_index` | 전 프레임 0 | 프레임별 1(bring)/2(flip)/3(push) | 〃 |
| `meta/tasks.jsonl` | 문장 1개 | 4개 (0=기존 전체 문장 보존, 1~3=subtask) | 〃 |
| `meta/episodes.jsonl` `tasks` | 전체 문장 | 해당 ep의 subtask 문장 목록 | 〃 |
| `meta/info.json` `total_tasks` | 1 | 4 | 〃 |
| 학습 스크립트 | `finetune_rldx1_ffw_sh5_0826.sh` | `finetune_rldx1_ffw_sh5_0826_subtask.sh` (신규 사본, 원본 무수정) — 실질 diff는 `DATASET_PATH`/`CKPT_NAME` 기본값 2줄 | 신규 파일 |

## 새로 만든 파일

| 파일 | 역할 |
|---|---|
| `run_scripts/data/apply_subtasks.py` | 라벨(jsonl 구간)을 프레임별 `task_index`로 스탬핑한 데이터셋 복사본 생성. parquet은 **Arrow 레벨**로 `task_index` 컬럼만 교체 — pandas 왕복은 `fixed_size_list[57]`을 강등시키고 HF 스키마 메타데이터를 지우기 때문(merge 때 실측). 스키마·타 컬럼 비트 동일을 assert로 검증 |
| `run_scripts/train/examples/finetune_rldx1_ffw_sh5_0826_subtask.sh` | 학습 실행 (베이스라인 스크립트 사본, 기본값 2줄만 다름) |
| `dataset/rlwrld_demo_0826_merge/meta/subtask_labels.jsonl` | 사람이 작성한 원본 구간 라벨 (merge ep 0~80, 구간 {start,end,text}) 보존 |
| `dataset/rlwrld_demo_0826_{train,holdout}_subtask/` | 변환된 데이터셋 (원본 불변) |

## 라벨 통계

- 81 에피소드 전수, 전 프레임 커버(갭·중복 0), 에피소드당 2~8구간(반복 포함)
- 문장 3종: `bring the box from the left to the workspace` /
  `flip the box so the shipping label faces upward` / `push the box to the right`
- train 29,661f = bring 13,137 + flip 6,964 + push 9,560 ·
  holdout 3,357f = 1,530 + 712 + 1,115
- 라벨은 merge 인덱스 기준 → split 인덱스 매핑은 `meta/source_episodes.jsonl`의
  (source_repo, source_episode_index) 일치 + length 일치로 이중 검증

## 재현 절차

```bash
# 1. 변환 (원본 무변경, 복사본 생성)
uv run python run_scripts/data/apply_subtasks.py \
    --labels dataset/rlwrld_demo_0826_merge/meta/subtask_labels.jsonl \
    --dataset-dir dataset/rlwrld_demo_0826_train \
    --merge-meta dataset/rlwrld_demo_0826_merge/meta \
    --out-dir dataset/rlwrld_demo_0826_train_subtask
# (holdout도 동일하게)

# 2. 로더 검증 — 경계에서 문장이 바뀌는지
#    ep0: frame 135→136 bring→flip, 350→351 flip→push (검증 완료)

# 3. 스모크 → 본 학습
SMOKE=1 bash run_scripts/train/examples/finetune_rldx1_ffw_sh5_0826_subtask.sh
bash run_scripts/train/examples/finetune_rldx1_ffw_sh5_0826_subtask.sh

# 4. 평가 — 반드시 _subtask holdout으로 (전체 궤적이 기본)
CKPT_NAME=rldx1_ft_ffw_sh5_0826_subtask \
DATASET=$PWD/dataset/rlwrld_demo_0826_holdout_subtask \
OUT_DIR=$PWD/outputs/eval/holdout_subtask \
bash run_scripts/eval/eval_holdout_all.sh
```

## 해석 시 주의

- holdout 평가의 subtask 조건은 GT 구간 라벨 = **oracle 시퀀서 가정의 상한**이다.
  실기에서는 문장을 언제 바꿀지 정하는 시퀀서(FSM/검출기)가 별도로 필요하다.
- chunk 40(4초)이 구간 경계에 걸치면 라벨 A + 후반 동작 B가 되지만, 배포는
  앞 20 step(2초)만 실행하고 재질의하므로 경계 노출은 최대 2초로 제한된다.
- 비교 기준: 베이스라인 ckpt-10000, held-out MSE 0.0239 (전체 궤적, horizon 40).
  판정은 평균이 아니라 에피소드별 짝비교(±15% 샘플링 노이즈).

## 결과 (26.09.03 실측)

세 run 모두 open-loop MSE 동률: RLDX 0.0239→0.0241, GR00T ABS 0.0297→0.0308,
GR00T relarm 0.0236→0.0239 (각 최적 ckpt, 짝비교 5/9·3/9·5/9).

언어 민감도 프로브(remap_task_index.py로 만든 오답/전체문장 holdout 변형):
GR00T ABS는 오답 라벨에 +11%(9/9 악화)로 반응, relarm +7%, **RLDX는 ±0% —
subtask 문장을 완전히 무시한다.** 문장이 시각 상태와 완벽 상관이라 시각이
지름길이 된 것. 시퀀서 제어가 필요하면 "같은 장면·다른 명령" 데이터가 선행
조건이고, 그 전까지 배포는 whole-sentence로 충분하다.
자세한 요약: FFW-SH5-0826-briefing.md의 26.09.03 섹션.
