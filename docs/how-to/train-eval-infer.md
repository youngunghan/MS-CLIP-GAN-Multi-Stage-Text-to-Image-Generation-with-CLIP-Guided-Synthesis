# How-to: 학습 · 평가 · 추론

> **범위:** 학습(단일/다중 GPU·resume), 평가(FID/IS/CLIP), 추론 실행 절차. 옵션 전체는 [reference/configuration.md](../reference/configuration.md), 스크립트는 [reference/cli.md](../reference/cli.md).
> **대상:** 학습/실험을 돌리는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-16.

## 1. 학습 (단일 GPU)

```bash
bash train.sh
```

[train.sh](../../train.sh)는 `--gpu_ids 0`으로 [scripts/train.py](../../scripts/train.py)를 실행한다. 핵심 동작:

- 데이터: `--data_path ./data/trainset.zip`을 [dataset/dataloader.py](../../dataset/dataloader.py) `MM_CelebA`로 로드(지연 디코딩, 256→각 단계 다운샘플).
- 모델: [networks/generator.py](../../networks/generator.py) `Generator` 1개 + [networks/discriminator.py](../../networks/discriminator.py) `Discriminator` 단계별 3개.
- 손실: BCE 적대 + (옵션) uncond·contrastive·mixed. CLIP은 동결. 상세 [explanation/architecture.md](../explanation/architecture.md).
- 저장: epoch마다 스케줄러 step 후 `save_freq` 주기로 `checkpoints/<name>/ckpt/`에 G·D 체크포인트(+optimizer·scheduler 상태), `res/`에 샘플 그리드.

| 자주 바꾸는 옵션 | 기본 | 의미 |
|---|---|---|
| `--batch_size` | 64 | 배치 크기 |
| `--learning_rate` | 1e-4 | Adam lr(β=0.5,0.999) |
| `--num_epochs` | 150 | 에폭 수 |
| `--save_freq` | 5 | 저장 주기(에폭) |
| `--use_uncond_loss` / `--use_contrastive_loss` / `--use_mixed_loss` | argparse 기본은 off, `train.sh`는 3개 모두 on | 보조 손실 토글 |
| `--seed` | 42 | 재현성 시드 |

## 2. 학습 (다중 GPU)

`train.sh`의 `GPUS`를 콤마로 지정한다(예: `GPUS="0,1"`). `--gpu_ids`가 **실제 CUDA 디바이스 인덱스**로 쓰이고, 2개 이상이면 `nn.DataParallel(device_ids=...)`로 감싼다([scripts/train.py](../../scripts/train.py)).

> 🟢 체크포인트는 항상 DataParallel을 벗겨 저장하고, 로드 시 `module.` 접두사를 자동 제거한다 → 다중 GPU로 학습한 체크포인트를 단일 GPU 추론에서 그대로 로드 가능([utils/utils.py](../../utils/utils.py) `save_checkpoint()`·`load_checkpoint()`).

## 3. Resume

[train.sh](../../train.sh) 하단 주석을 활성화한다:

```bash
python scripts/train.py ... \
    --resume_checkpoint_path ./checkpoints/<run_name>/ckpt \
    --resume_epoch <epoch>
```

- optimizer **및 LR 스케줄러** 상태까지 복구한다(스케줄러를 resume 이전에 생성해 전달, [scripts/train.py](../../scripts/train.py)).
- `--new_optim`을 주면 가중치만 로드하고 optimizer/scheduler는 새로 시작한다.

## 4. 평가 (FID / IS / CLIP score)

```bash
# ./eval.sh <CKPT_DIR> [EPOCH]   — CKPT_DIR은 필수(생략 시 usage 출력 후 exit 1)
bash eval.sh ./checkpoints/<run_name>/ckpt 149
```

[scripts/eval.py](../../scripts/eval.py) `evaluate()`는:

- `testset.zip`에서 텍스트 임베딩으로 이미지를 생성하고,
- **단일** `FrechetInceptionDistance`·`InceptionScore`에 전 배치를 누적한 뒤 1회 `compute()`(배치별 평균 아님), CLIP score는 표본 가중 평균.
- 결과를 콘솔 + `./output/metrics.csv`에 기록.

> ⚠️ FID/IS 수치를 인용하기 전 [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md)의 평가 타당성 절을 읽는다(real 이미지 출처·표본 수).

## 5. 추론

```bash
# ./infer.sh <CKPT_DIR> [EPOCH]   — CKPT_DIR은 필수(생략 시 usage 출력 후 exit 1)
bash infer.sh ./checkpoints/<run_name>/ckpt 149
```

[scripts/infer.py](../../scripts/infer.py)는 프롬프트를 CLIP `encode_text`로 임베딩 → 정규화 → `Generator`에 z와 함께 통과 → 단계별 이미지를 `./output/result_{64,128,256}.png`로 저장. `G.eval()` 적용, 생성기 체크포인트만 필요.

## 관련 문서

- [reference/configuration.md](../reference/configuration.md) — 옵션 전체
- [reference/cli.md](../reference/cli.md) — 스크립트 진입점
- [how-to/troubleshooting.md](troubleshooting.md) — 문제 해결
