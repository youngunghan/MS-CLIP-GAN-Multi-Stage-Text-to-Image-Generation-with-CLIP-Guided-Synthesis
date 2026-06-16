# Quickstart — 설치부터 첫 학습·추론까지

> **범위:** 환경 설치 → 데이터 전처리 → 학습 → 텍스트 프롬프트 추론까지 happy path. 세부 옵션은 [how-to/](../how-to/)·[reference/](../reference/).
> **대상:** 처음 셋업하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-16.

## 1. 전제 조건

| 항목 | 요구 | 비고 |
|---|---|---|
| GPU | NVIDIA, VRAM ≥ 8GB 권장 | CPU도 동작하나 매우 느림(자동 fallback) |
| Conda | 권장 | `environment.yml` 제공 |
| 데이터 | MM-CelebA-HQ | 이미지+캡션, [how-to/prepare-dataset.md](../how-to/prepare-dataset.md) |

## 2. 환경 설치

```bash
conda env create -f environment.yml
conda activate msclipgan
# Linux에서 libGL.so.1 오류 시
sudo apt-get install -y libgl1-mesa-glx
```

CLIP은 `environment.yml`의 pip 절에서 `git+https://github.com/openai/CLIP.git`로 설치된다. 설치 확인:

```bash
python -c "import torch, clip, torchmetrics; print(torch.__version__, torch.cuda.is_available())"
```

## 3. 데이터 전처리 (요약)

상세는 [how-to/prepare-dataset.md](../how-to/prepare-dataset.md). 핵심 3단계:

```bash
bash preprocessing/split_dataset.sh        # train/test 파일명 분할(pickle)
bash preprocessing/preprocess_train.sh     # 256x256 crop + CLIP 피처 → data/trainset.zip
bash preprocessing/preprocess_test.sh      # → data/testset.zip
```

> 🟢 전처리는 256×256 PNG와 이미지·텍스트 CLIP 임베딩을 zip에 함께 저장한다. 데이터로더는 이 256 해상도를 보존해 각 단계로 다운샘플한다([reference/dataset-format.md](../reference/dataset-format.md)).

## 4. 학습

```bash
bash train.sh
```

- 기본값: 3단계(64/128/256), batch 64, lr 1e-4, 150 epoch, `--gpu_ids 0`.
- 정상 신호: D/G 손실이 발산하지 않고, `runs/<name>`에 TensorBoard 로그, `checkpoints/<name>/ckpt/`에 `epoch_*_Gen.pt`·`epoch_*_Dis_{0,1,2}.pt`, `checkpoints/<name>/res/`에 샘플 그리드 저장.
- 모니터링: `tensorboard --logdir=runs`.

> ⚠️ `train.sh`는 기본으로 **처음부터 학습**한다(resume는 주석으로 안내). 다중 GPU·resume는 [how-to/train-eval-infer.md](../how-to/train-eval-infer.md).

## 5. 추론 (텍스트 → 이미지)

```bash
# ./infer.sh [CKPT_DIR] [EPOCH]   — CKPT_DIR은 epoch_<E>_Gen.pt 가 있는 디렉터리
bash infer.sh ./checkpoints/<run_name>/ckpt 99
```

- 프롬프트는 `infer.sh`의 `--prompt`로 지정. 결과는 `./output/result_64.png`·`result_128.png`·`result_256.png`.
- 추론은 **생성기 체크포인트만** 필요하다(판별기 불필요).

## 6. 다음 단계

1. [how-to/prepare-dataset.md](../how-to/prepare-dataset.md) — 데이터 준비 상세
2. [how-to/train-eval-infer.md](../how-to/train-eval-infer.md) — 평가(FID/IS/CLIP)·resume·다중 GPU
3. [explanation/architecture.md](../explanation/architecture.md) — 동작 원리
4. [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md) — ⭐ 수치 인용 전 필독
