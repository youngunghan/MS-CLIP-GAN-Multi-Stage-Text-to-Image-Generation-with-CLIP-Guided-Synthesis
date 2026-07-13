# CLI / 스크립트 레퍼런스

> **범위:** Python 진입점과 shell wrapper의 인자·출력·실패 계약. 옵션 의미와 기본값은 [reference/configuration.md](configuration.md).
> **대상:** 개발자·실험 자동화 작성자.
> **상태:** 구현 반영 — 기준일 2026-07-10.

## 1. 공통 실행 계약

- working directory는 repository root다.
- shell wrapper는 `bash path/to/script.sh`로 실행한다. 실행 bit 유무에 의존하지 않는다.
- repository module을 import하는 Python은 `PYTHONPATH=. python ...`로 실행한다.
- 모든 Python parser는 알 수 없는 option을 거부한다. 실제 설치 코드의 정본은
  `PYTHONPATH=. python <script> --help`다.
- root와 `experiments/` shell은 script-relative로 repository root를 찾아 clone의
  절대 경로에 의존하지 않는다.
- option snapshot과 run directory는 train parser만 만든다. eval/infer가 받은
  checkpoint directory는 read-only 입력이며 parser가 그 안에 파일을 만들지 않는다.

## 2. Python 진입점

| 스크립트 | 함수 | 필수/핵심 인자 | 출력 |
|---|---|---|---|
| [scripts/train.py](../../scripts/train.py) | `__main__` | `TrainOptions`; data/model/loss/resume | checkpoint v2, sample, TensorBoard, `opt.txt` |
| [scripts/trainer.py](../../scripts/trainer.py) | `train_step()` | 내부 호출 | D/G 2-phase update |
| [scripts/infer.py](../../scripts/infer.py) | `main()` | `--checkpoint_path`, `--load_epoch`, `--eval_data_path None`, `--prompt` | `result_<size>.png` |
| [scripts/eval.py](../../scripts/eval.py) | `evaluate()`·`main()` | checkpoint epoch + eval zip + seed | console + provenance-aware `metrics.csv` + hash-prefixed sample images |
| [experiments/eval_curve.py](../../experiments/eval_curve.py) | `main()` | `TEST_ZIP CKPT_DIR EPOCHS OUT [--seed N]` | result JSON + `<OUT>.provenance.json` |
| [experiments/plot_curves.py](../../experiments/plot_curves.py) | module entry | `TRAIN_LOG EVAL_JSON OUT_DIR` | loss/FID/IS `curves.png` |
| [experiments/plot_compare.py](../../experiments/plot_compare.py) | `main()` | `OUT LABEL=EVAL...`; `--require-comparable`, allowed difference keys, optional `--require-diffaugment-pair POLICY` | neutral FID overlay; strict mode validates clean-Git sidecars and optional no-aug→aug direction |
| [experiments/dl_real_scaled.py](../../experiments/dl_real_scaled.py) | `main()` | `COUNT [--revision REV] [--output-dir DIR]` | source zips + `download_provenance.json` |
| [preprocessing/split_dataset.py](../../preprocessing/split_dataset.py) | `split_dataset()` | source, ratio, seed, optional max | train/test filename pickle |
| [preprocessing/preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) | `convert_dataset()` | source/list/dest/size, `--seed 42`, **`--emb_dim 512` 필수**, `--max-failure-frac 0.0` | deterministic RGB PNG + embedding ZIP/directory; 기본은 전부 성공 후 atomic replace, `--max-failure-frac`으로 일부 실패 허용 가능 |

## 3. Root shell wrapper

| 스크립트 | 인자 | 계약 |
|---|---|---|
| [train.sh](../../train.sh) | positional 없음 | 파일 안의 `BS=64`(기본; 8 GB급 GPU는 파일의 `BS`를 4로 낮춤), lr/epoch/save/GPU를 사용한 fresh run. resume positional을 지원하지 않음 |
| [eval.sh](../../eval.sh) | `<CKPT_DIR> [EPOCH]` | directory 필수, epoch 기본 149. wrapper의 prompt는 평가에서 무시 |
| [infer.sh](../../infer.sh) | `<CKPT_DIR> [EPOCH]` | directory 필수, epoch 기본 149. prompt는 wrapper 안에 고정; 세 번째 positional 없음 |
| [preprocessing/split_dataset.sh](../../preprocessing/split_dataset.sh) | positional 없음 | `image.zip`/`text.zip` filename split |
| [preprocessing/preprocess_train.sh](../../preprocessing/preprocess_train.sh) | positional 없음 | `data/trainset.zip` 생성 |
| [preprocessing/preprocess_test.sh](../../preprocessing/preprocess_test.sh) | positional 없음 | `data/testset.zip` 생성 |

`eval.sh`/`infer.sh`는 checkpoint directory를 생략하면 usage를 stderr에 출력하고
non-zero 종료한다. prompt를 동적으로 주려면
[§5 Prompt 추론](../how-to/train-eval-infer.md#5-prompt-추론)처럼
`scripts/infer.py`를 직접 호출한다.

## 4. Experiment shell wrapper

| 스크립트 | 위치 인자 | 기본/비고 |
|---|---|---|
| [data_prep.sh](../../experiments/data_prep.sh) | `[N] [RATIO] [TAG] [HF_REVISION]` | `3000 0.83 sub` + 감사된 HF commit 기본값. 환경 변수 `HF_DATASET_REVISION`, `MSCLIPGAN_ENV`로 override |
| [train.sh](../../experiments/train.sh) | `[NAME] [DATA_ZIP] [EPOCHS] [SAVE_FREQ] [BATCH]` | `sub25`, subset zip, 100, 10, 4 |
| [run_all.sh](../../experiments/run_all.sh) | `[N] [RATIO] [TAG] [NAME] [EPOCHS] [SAVE]` | prep→train→eval→plot. revision은 `HF_DATASET_REVISION` 환경 변수로 전달 |
| [run_stable.sh](../../experiments/run_stable.sh) | `[NAME] [TAG] [EPOCHS] [SAVE] [D_LR] [EMA] [SMOOTH]` | 안정화 변형 |
| [run_sweep.sh](../../experiments/run_sweep.sh) | `[PREFIX] [TAG] [EPOCHS] [SAVE]` | prefix 아래 script 내부 `CONFIGS` 두 개 순차 실행 |
| [run_diffaug.sh](../../experiments/run_diffaug.sh) | `[NAME] [TAG] [EPOCHS] [SAVE] [POLICY] [BASELINE_EVAL]` | DiffAugment 변형. baseline은 현재 v2 provenance가 있을 때만 검증 후 overlay |

`data_prep.sh`의 기본 HF revision은
`50e4e9dc81ca8f974af5a5e088ad82d830a7f5d0`이다. requested revision을 실제 commit
SHA로 resolve한 뒤 그 SHA로 download하고, `download_provenance.json`에 dataset
id/split, requested/resolved revision, count, archive SHA-256, package version을
기록한다. mutable ref는 매번 commit SHA로 다시 resolve한다. 기존 archive는 resolved
revision, manifest/실제 SHA-256, 정확한 image/text count, 중복 없는 paired stem set을
모두 다시 검증하고 요청 N 이상일 때만 재사용한다.

자동 runner는 새 결과를 Git-ignore된 `experiments/results/generated/<NAME>/`에 쓰며,
평가 JSON이 이미 있는 이름은 덮어쓰지 않는다. 이름을 생략하면 timestamp 기반 현재
protocol 이름을 쓴다. 커밋된 `experiments/results/<historical-run>/eval.json`은
역사적 원자료로 보존한다.

## 5. 출력·metadata 계약

| 출력 | 계약 |
|---|---|
| `checkpoints/<name-timestamp>/opt.txt` | parser option snapshot |
| `epoch_<E>_Gen.pt` | checkpoint v2: model/optimizer/scheduler, model·training·schedule config, training provenance, RNG, generator weight kind |
| `epoch_<E>_Gen_raw.pt` | EMA run의 live generator; exact resume에 필요 |
| `epoch_<E>_Dis_<stage>.pt` | stage D + optimizer/scheduler + v2 metadata |
| `metrics.csv` | 단일 checkpoint eval append 결과, checkpoint path/SHA·model semantics·seed·processed sample 수 |
| `<OUT>.provenance.json` | curve eval의 seed·data/checkpoint/result JSON hash·model/training/schedule config·학습 provenance·Git/runtime/hardware 정보 |
| `download_provenance.json` | HF source revision과 archive fingerprint |

checkpoint는 DataParallel prefix를 제거해 저장한다. eval/infer와 `eval_curve.py`는
checkpoint metadata를 model 생성 전에 읽어 저장된 architecture를 구성한다. metadata
없는 checkpoint는 parameter shape에서 기본 차원을 추론하고 `relu` conditioning +
`legacy_conditioned` alignment로 호환한다. eval/infer에는 generator만 필요하고,
exact training resume에는 같은 epoch의 G/D/optimizer/scheduler 및 EMA raw companion이
필요하다.

## 6. 실패 계약

- shell runner는 child exit code를 전파하며 GPU sampler는 `EXIT` cleanup한다.
- ZIP/directory 전처리는 duplicate/missing stem, decode/빈 caption 등 sample 실패 비율이
  `--max-failure-frac`(기본 0.0, 즉 한 sample 실패도 불허)을 넘으면 전체 non-zero로 처리하고
  staged output(ZIP 파일 또는 directory)을 버려 기존 destination을 보존한다.
  `--max-failure-frac`을 0보다 크게 주면 그 비율 이내 실패는 허용하고 성공한 sample만으로
  destination을 atomic하게 교체한다. OOM/interrupt도 전파한다.
- curve eval은 빈 checkpoint 선택, 잘못된 dataset/metadata, non-finite metric에서
  non-zero로 끝나며 빈 `{}`를 성공 결과로 쓰지 않는다.
- CLI range/architecture/resume 조합 오류는 학습 시작 전에 usage와 함께 실패한다.

## 관련 문서

- [reference/configuration.md](configuration.md) — option 기본값·검증 범위
- [how-to/train-eval-infer.md](../how-to/train-eval-infer.md) — 실행 예시
- [how-to/run-experiments.md](../how-to/run-experiments.md) — subset pipeline
