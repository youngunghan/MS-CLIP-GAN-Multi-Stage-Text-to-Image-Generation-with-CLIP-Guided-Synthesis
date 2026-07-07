# CLI / 스크립트 레퍼런스

> **범위:** Python 진입점과 셸 래퍼 스크립트, 인자, 입출력. 옵션 의미는 [reference/configuration.md](configuration.md).
> **대상:** 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-16.

## 1. Python 진입점

| 스크립트 | 함수 | 역할 | 출력 |
|---|---|---|---|
| [scripts/train.py](../../scripts/train.py) | `__main__` | 학습 루프 구동(모델·옵티마이저·스케줄러·resume·샘플·체크포인트) | `checkpoints/<name>/ckpt/`, `res/`, `runs/<name>` |
| [scripts/trainer.py](../../scripts/trainer.py) | `train_step()` | 한 에폭의 D/G 2단계 업데이트 | (train.py가 호출) |
| [scripts/infer.py](../../scripts/infer.py) | `main()` | 텍스트 프롬프트 → 단계별 이미지 | `result_{64,128,256}.png` |
| [scripts/eval.py](../../scripts/eval.py) | `evaluate()`·`main()` | FID/IS/CLIP score | 콘솔 + `metrics.csv` |
| [preprocessing/split_dataset.py](../../preprocessing/split_dataset.py) | `split_dataset()` | train/test 파일명 분할 | `data/celeba_filenames_{train,test}.pickle` |
| [preprocessing/preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) | `convert_dataset()` | crop+CLIP 피처 → zip | `data/{trainset,testset}.zip` |

`scripts/*.py`는 `PYTHONPATH=$(pwd)`(셸 스크립트가 설정)에서 `python scripts/train.py`로 실행한다(`from trainer import ...`가 스크립트 디렉터리 기준 해석).

## 2. 셸 래퍼

| 스크립트 | 인자 | 설명 |
|---|---|---|
| [train.sh](../../train.sh) | (내부 변수) | `GPUS`·`BS`·`LR`·`EPOCH`·`SAVE_FREQ` 편집. resume는 하단 주석 |
| [infer.sh](../../infer.sh) | `<CKPT_DIR> [EPOCH]` | `CKPT_DIR`은 **필수**(생략 시 usage 출력 후 `exit 1`)이며 `epoch_<E>_Gen.pt`가 있어야 함. `EPOCH` 기본값 149. `--prompt`로 프롬프트 |
| [eval.sh](../../eval.sh) | `<CKPT_DIR> [EPOCH]` | `CKPT_DIR`은 **필수**(생략 시 usage 출력 후 `exit 1`). `EPOCH` 기본값 149. `--eval_data_path ./data/testset.zip` |
| [preprocessing/split_dataset.sh](../../preprocessing/split_dataset.sh) | (내부) | `--source_path`·`--train_ratio`·`--seed` |
| [preprocessing/preprocess_train.sh](../../preprocessing/preprocess_train.sh) | (내부) | train zip 생성 |
| [preprocessing/preprocess_test.sh](../../preprocessing/preprocess_test.sh) | (내부) | test zip 생성 |

> 🟢 모든 셸 스크립트는 상대 경로를 쓴다(과거의 하드코딩 절대 경로 제거). `train.sh`는 `--gpu_ids`를 실제 디바이스로 사용하며 `CUDA_VISIBLE_DEVICES`를 별도로 설정하지 않는다.

## 3. 출력 경로 규약

- 학습: `checkpoints/<name-timestamp>/ckpt/epoch_<E>_Gen.pt`·`epoch_<E>_Dis_{0,1,2}.pt`, `checkpoints/<name-timestamp>/res/<name>_epoch_<E>.png`.
- `--use_ema`가 설정되면 `epoch_<E>_Gen.pt`에는 EMA 가중치가, 함께 저장되는 동반 파일 `epoch_<E>_Gen_raw.pt`에는 학습 중이던(raw) 생성기 가중치가 들어간다. eval/infer는 기본적으로 `Gen.pt`(EMA)를 로드한다([utils/utils.py](../../utils/utils.py) `save_checkpoint()`).
- 추론/평가: 기본 `./output/`(`infer`/`eval`은 `parse(print_options=False)`로 실험 디렉터리/`opt.txt`를 만들지 않음).

## 관련 문서

- [reference/configuration.md](configuration.md) — 옵션 표
- [how-to/train-eval-infer.md](../how-to/train-eval-infer.md) — 실행 절차
