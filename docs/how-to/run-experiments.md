# How-to: `experiments/` 서브셋 실험 실행

> **범위:** 소규모 HF 서브셋을 준비하고 학습→checkpoint별 FID/IS→plot까지 실행하는 절차. 역사적 수치 해석은 [experiments/RESULTS.md](../../experiments/RESULTS.md), 정식 학습·평가·추론은 [how-to/train-eval-infer.md](train-eval-infer.md).
> **대상:** correctness audit 이후 정량 실험을 재현·확장하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-07-23.

현재 pipeline은 `environment.yml`의 `msclipgan` 환경 하나를 사용한다. shell 파일의
실행 bit에 의존하지 않도록 항상 `bash`로 실행하고, Python 진입점을 직접 부를 때는
repository root에서 `PYTHONPATH=.`를 붙인다.

## 1. 준비

```bash
conda env create -f environment.yml
conda activate msclipgan
python -c "import torch, clip, datasets, torchmetrics; print(torch.__version__)"
```

`datasets`는 `experiments/dl_real_scaled.py`가 Hugging Face dataset을 streaming으로
읽을 때 사용한다. 과거 로컬 전용 `msclipgan-smoke` 환경은 현재 계약이 아니다.
다른 Conda env 이름을 써야 하면 runner 앞에 `MSCLIPGAN_ENV=<name>`을 둔다.

## 2. 스크립트 역할

| 스크립트 | 역할 |
|---|---|
| [data_prep.sh](../../experiments/data_prep.sh) | HF에서 N장 다운로드·무결성 검사 → seed split → 결정적·atomic CLIP 전처리 → `data/{trainset,testset}_<TAG>.zip` |
| [dl_real_scaled.py](../../experiments/dl_real_scaled.py) | `ixw/celebahq-caption-10k`를 streaming download해 source `image.zip`/`text.zip` 생성 |
| [train.sh](../../experiments/train.sh) | 위치 인자로 run/data/epoch/save/batch를 받아 학습하고 GPU log·train log·요약 metadata 기록 |
| [eval_curve.py](../../experiments/eval_curve.py) | 선택한 generator checkpoint마다 표준 2048-d FID/IS 계산 + provenance sidecar 기록 |
| [plot_curves.py](../../experiments/plot_curves.py) | epoch loss와 FID/IS curve plot |
| [plot_compare.py](../../experiments/plot_compare.py) | 여러 run의 FID curve overlay; provenance/config 비교와 DiffAugment baseline→treatment 방향 검증 가능 |
| [run_all.sh](../../experiments/run_all.sh) | data prep → baseline train → eval → plot |
| [run_stable.sh](../../experiments/run_stable.sh) | EMA + TTUR + one-sided label smoothing 변형 |
| [run_sweep.sh](../../experiments/run_sweep.sh) | 새 prefix 아래 `swA_aggr`, `swB_mild` D-update 변형 probe |
| [run_diffaug.sh](../../experiments/run_diffaug.sh) | DiffAugment train/eval/plot; 현재-protocol baseline을 주면 검증 후 overlay |
| [fid_feature_space_demo.py](../../experiments/fid_feature_space_demo.py) | ignite 1000-d logits FID와 torchmetrics 2048-d pool3 FID 차이를 보여주는 역사적 demo |

모든 runner는 자신의 위치를 기준으로 repository root를 찾으므로 clone 경로에
`/home/yuhan/...`를 하드코딩하지 않는다. 단계가 실패하면 non-zero를 전파해 뒤 단계를
성공으로 표시하지 않는다.

자동 runner의 새 결과는 Git에서 제외된 `experiments/results/generated/<NAME>/`에만
쓴다. 기존 `experiments/results/{sub25,diffaug100,...}/eval.json`은 역사적 원자료이므로
runner가 덮어쓰지 않는다. 같은 generated name에 평가 JSON이 이미 있으면 즉시
실패하므로 새 `NAME`/`PREFIX`를 선택한다. 이름을 생략하면 timestamp가 붙은 현재
protocol 이름을 만든다.

## 3. End-to-end baseline

```bash
# N=3000, train ratio=0.83, dataset tag=sub,
# run name=sub25_new, 100 epochs, save every 10 epochs
bash experiments/run_all.sh 3000 0.83 sub sub25_new 100 10

# source revision까지 고정
HF_DATASET_REVISION=<tag-or-commit> \
  bash experiments/run_all.sh 3000 0.83 sub sub25_new 100 10
```

순서는 `data_prep.sh` → `experiments/train.sh` → `eval_curve.py` →
`plot_curves.py`다. `experiments/train.sh`의 기본 batch는 4이며 RTX 4060 Ti 8 GB에서
약 5.8~6.2 GB를 사용했다. 장비가 다르면 짧은 run으로 먼저 확인한다.

현재 training loop는 `save_freq`와 별개로 마지막 epoch를 강제 저장한다. 예를 들어
100 epoch는 0~99를 학습하고 epoch 99도 저장·평가 대상이 된다. 반면 2026-06의
역사적 JSON은 이 수정 전 결과라 100-epoch run의 마지막 저장·평가가 epoch 90,
50-epoch probe의 마지막 저장·평가가 epoch 45였다.

download를 실제 수행하면
`data/mm-celeba-hq-dataset/download_provenance.json`에 requested/resolved revision,
sample 수, source archive SHA-256, package version을 기록한다. 기존 `image.zip`에 N장
이상이더라도 요청 ref를 **현재 commit SHA로 다시 resolve**하고, 그 SHA가 provenance와
같으며 두 archive의 실제 SHA-256·정확한 count·중복 없는 paired stem set이 모두 맞고
N장 이상일 때만 download를 건너뛴다. 다운로드 직후에도 같은 검사를 반복한다. 기본 HF revision은 감사된 commit
`50e4e9dc81ca8f974af5a5e088ad82d830a7f5d0`이며, 4번째 `data_prep.sh` 인자나
`HF_DATASET_REVISION`으로 override한다.

train/test 전처리는 seed 42와 고정 ZIP metadata를 쓰고 모든 선택 sample이 성공한 뒤에만
기존 output을 atomic replace한다. processed ZIP의 `dataset.json.preprocess`는 seed/config를
기록하지만 raw revision·split-list hash를 내장하지 않으므로 raw provenance와 pickle을
결과와 함께 보존한다.

## 4. 변형 실행

```bash
# EMA + TTUR + label smoothing
bash experiments/run_stable.sh sub25_stable_new sub 100 10 1e-4 0.999 0.9

# 두 D-update 설정; 결과 이름은 sweep_current_{swA_aggr,swB_mild}
bash experiments/run_sweep.sh sweep_current sub 40 5

# 먼저 같은 현재 protocol의 no-augmentation baseline 생성
bash experiments/run_all.sh 3000 0.83 sub baseline_current 100 10

# DiffAugment 100 epoch + provenance가 맞는 baseline overlay
bash experiments/run_diffaug.sh \
  diffaug_current sub 100 10 color,translation,cutout \
  experiments/results/generated/baseline_current/eval.json

# baseline 인자를 생략하면 안전하게 개별 curve만 만들고 overlay는 건너뜀
bash experiments/run_diffaug.sh diffaug_probe_current sub 50 5
```

관련 학습 flag는 [§2 학습 옵션](../reference/configuration.md#2-학습-옵션-trainoptions)에 있다. 새 run은 기본 `linear` conditioning,
`image_only` alignment, mismatched real-text negative를 사용한다. 2026-06의 legacy
run과 동일한 architecture semantics가 아니므로 결과를 한 curve에 놓을 때 설정 차이를
명시한다.

conditioning 붕괴 진단 결과([explanation/correctness-and-fixes.md §2.4](../explanation/correctness-and-fixes.md#24-2026-07-23-conditioning-붕괴-진단복구))를
반영해 prompt-swap sensitivity를 회복시키는 recipe는 다음과 같다.

```bash
PYTHONPATH=. python scripts/train.py \
  --name cond_recovery \
  --data_path ./data/trainset_sub.zip \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --num_epochs 30 \
  --save_freq 5 \
  --use_uncond_loss --use_contrastive_loss --use_mixed_loss \
  --use_diffaugment \
  --conditioning_activation linear \
  --alignment_mode legacy_conditioned \
  --no_mismatched_condition \
  --gamma 1 --lam 2 \
  --kl_weight 0 --deterministic_cond \
  --gpu_ids 0
```

`--kl_weight 0`과 `--deterministic_cond`가 근본 원인 두 가지를 끄고, `--gamma 1 --lam 2`는
conditioning 관련 항들이 D 학습을 압도하지 않도록 낮춘 값이다. conditioning 효과는
epoch 10~30 부근에서 정점을 찍은 뒤 모든 run에서 감소하므로 **early stopping이
필요**하다 — `--num_epochs`를 크게 잡고 이 구간의 checkpoint들을
[§7 Prompt-sensitivity 측정](#7-prompt-sensitivity-측정-1차-conditioning-지표)으로 직접 비교한다.
회복된 응답은 여전히 한 자릿수~낮은 두 자릿수 `/255` 수준이고 FID는 350~410대에
머문다는 한계는 [explanation/correctness-and-fixes.md §2.4](../explanation/correctness-and-fixes.md#24-2026-07-23-conditioning-붕괴-진단복구)에 정리돼 있다.

## 5. 개별 checkpoint 평가

```bash
PYTHONPATH=. python experiments/eval_curve.py \
  data/testset_sub.zip \
  checkpoints/<run-name>/ckpt \
  auto \
  experiments/results/generated/<run-name>/eval.json \
  --seed 42
```

| 계약 | 동작 |
|---|---|
| checkpoint 선택 | `auto`는 존재하는 `epoch_*_Gen.pt` 전체. `0,10,20`처럼 명시하면 하나라도 없을 때 누락 목록과 함께 실패 |
| caption | test 이미지마다 저장된 **첫 번째 caption embedding 1개** |
| RNG | checkpoint load 뒤 매 epoch 같은 seed로 재설정; 목록·순서에 무관 |
| model config | checkpoint v2 `model_config` 사용; metadata 없는 checkpoint는 legacy 기본으로 해석 |
| metric | torchmetrics FID pool3 2048-d + IS, uint8 RGB |
| 빈 선택 | checkpoint가 하나도 없으면 non-zero 종료, 빈 `{}`를 성공 결과로 쓰지 않음 |

출력은 두 파일이다.

| 파일 | 내용 |
|---|---|
| `eval.json` | epoch별 `fid`, `is_mean`, `is_std`에 더해 `clip_score`, `clip_diversity`(생성 이미지 CLIP feature의 평균 pairwise cosine distance) |
| `eval.json.provenance.json` | eval seed·caption policy·sample 수·dataset/checkpoint/result JSON SHA-256·model/training/schedule config·학습 provenance·Git/runtime/hardware·CLIP 모델/가중치 fingerprint·fake sample 수 |

> ⚠️ **오염 경고:** 모든 학습 run은 `--use_contrastive_loss`로 G를 CLIP similarity에
> 대해 직접 학습시킨다. 따라서 `clip_score`/`clip_diversity`는 학습 목적함수와 부분적으로
> 겹치는 지표이며, conditioning이 실제로 작동한다는 독립 증거로 인용하지 않는다. 어떤
> loss도 최적화하지 않는 1차 conditioning 지표는 [§7 Prompt-sensitivity 측정](#7-prompt-sensitivity-측정-1차-conditioning-지표)의
> `experiments/prompt_sensitivity.py`다.

## 6. 결과 해석

- 새 결과: `experiments/results/generated/<NAME>/eval.json`, `.provenance.json`, `curves.png`.
- DiffAugment runner에 baseline을 주면 같은 디렉터리에 `compare_baseline.png`를 쓴다.
- provenance와 원자료를 함께 보존한다. plot만으로 수치를 인용하지 않는다.
- `plot_compare.py --require-comparable`은 evaluation data/seed/epoch/runtime와 v2
  model·training·schedule·학습 source/data fingerprint를 확인한다. 두 평가 모두
  **clean Git checkout**에서 실행되어 `git_dirty=false`여야 하며, 각 `eval.json`의
  실제 SHA-256도 sidecar 기록과 같아야 한다. 의도한 차이는
  `--allow-training-difference`로 열거한다. `run_diffaug.sh`는 추가로 baseline이
  no-augmentation, treatment가 요청 policy의 DiffAugment인지 방향까지 검증한다.
  sidecar 없는 역사적 JSON은 이 엄격 비교에 넣지 않는다.
- 510장 test set과 단일 training seed는 탐색용이다. validation으로 checkpoint를
  선택하고 held-out test와 반복 seed로 확인하기 전에는 우월성·붕괴·architecture
  ceiling을 단정하지 않는다.
- 역사적 결과는 [experiments/RESULTS.md](../../experiments/RESULTS.md)가 정본이다.
  커밋된 과거 JSON은 evaluation seed가 미기록이고, 그 뒤 loop-level seed 버전도
  checkpoint 순서에 의존하므로 현재 evaluator와 exact match를 보장하지 않는다.

## 7. Prompt-sensitivity 측정 (1차 conditioning 지표)

`clip_score`/`clip_diversity`는 학습 loss와 겹쳐 오염된 지표이므로(§5 경고 참고),
conditioning이 실제로 caption에 반응하는지 확인하는 1차 지표는
[experiments/prompt_sensitivity.py](../../experiments/prompt_sensitivity.py)다. noise
`z`와 conditioning-augmentation epsilon을 고정한 채 caption만 바꿔 256px 출력의 평균
절대 pixel 이동(0~255 스케일)과 CLIP matched-vs-shuffled gap을 측정한다 — 어떤 loss도
이 pixel-sensitivity 수치를 최적화하지 않으므로 gaming이 불가능하다.

```bash
PYTHONPATH=. python experiments/prompt_sensitivity.py \
  checkpoints/<run-name>/ckpt \
  20 \
  data/testset_sub.zip \
  --num-captions 16 \
  --seeds 5 \
  --device cuda
```

| 인자/옵션 | 의미 |
|---|---|
| `checkpoint_dir` | `epoch_<E>_Gen.pt`를 담은 디렉터리(위치 인자) |
| `epoch` | 측정할 checkpoint epoch(위치 인자) |
| `dataset_zip` | `dataset.json.clip_txt_features`를 가진 전처리 zip(위치 인자) |
| `--num-captions` | 서로 바꿔볼 caption 개수(기본 16, 최소 2) |
| `--seeds` | 독립적인 `(z, CA-epsilon)` draw 반복 수(기본 3). 단일 seed는 std=0으로 오해를 주므로 피한다 |
| `--seed` | base RNG seed(기본 42) |
| `--device` | 기본은 CUDA 가용 시 `cuda`, 아니면 `cpu` |
| `--output` | 결과를 JSON으로도 저장할 경로(선택) |

참조 스케일(이미 측정, 매 실행마다 함께 출력): 실제 이미지-자기 caption CLIP 유사도
0.2722, 셔플 caption과의 유사도 0.2025(usable range 0.0697); 붕괴된 모델의
prompt-swap sensitivity ~2/255; 서로 다른 checkpoint가 같은 caption에서 보이는 차이
~75.7/255. `--kl_weight 0 --deterministic_cond` recipe([§4 변형 실행](#4-변형-실행))를
적용한 checkpoint는 이 sensitivity가 ~5~11/255로 회복된다([explanation/correctness-and-fixes.md §2.4](../explanation/correctness-and-fixes.md#24-2026-07-23-conditioning-붕괴-진단복구)).
"% of range" 수치는 참조 range를 측정한 `data/testset_sub.zip`(510 caption)이 아닌 다른
dataset에서는 `nan`으로 억제된다.

## 관련 문서

- [experiments/RESULTS.md](../../experiments/RESULTS.md) — 역사적 결과·source date·한계
- [reference/configuration.md](../reference/configuration.md) — 실험 flag 정의
- [how-to/train-eval-infer.md](train-eval-infer.md) — 정식 학습/평가/추론
- [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md) — 평가·checkpoint 수정
