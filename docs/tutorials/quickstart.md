# Quickstart — 설치부터 첫 학습·추론까지

> **범위:** 환경 설치 → 데이터 전처리 → 학습 → 텍스트 프롬프트 추론까지 happy path. 세부 옵션은 [how-to/](../how-to/)·[reference/](../reference/).
> **대상:** 처음 셋업하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-07-10.

## 1. 전제 조건

| 항목 | 요구 | 비고 |
|---|---|---|
| GPU | NVIDIA, 기본 BS=64는 대형 GPU 필요 | 8 GB급 카드(예: RTX 4060 Ti)는 `train.sh`의 `BS`를 4로 낮춘다 — 약 5.8~6.2 GB 실측. CPU fallback은 매우 느림 |
| Conda | 권장 | `environment.yml` 제공 |
| 데이터 | MM-CelebA-HQ | 이미지+캡션, [how-to/prepare-dataset.md](../how-to/prepare-dataset.md) |

## 2. 환경 설치

```bash
conda env create -f environment.yml
conda activate msclipgan
# Linux에서 libGL.so.1 오류 시
sudo apt-get install -y libgl1-mesa-glx
```

CLIP은 `environment.yml`의 pip 절에서 commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`로 고정 설치된다. 설치 확인:

```bash
python -c "import torch, clip, torchmetrics, datasets; print(torch.__version__, torch.cuda.is_available())"
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

- 기본값: 3단계(64/128/256), **batch 64**, lr 1e-4, 150 epoch, `--gpu_ids 0`. 8 GB급 GPU는 `train.sh`의 `BS`를 4로 낮춘다(약 5.8~6.2 GB 실측, RTX 4060 Ti). GPU가 다르면 먼저 1-epoch smoke run으로 조정한다.
- 정상 신호: D/G 손실이 발산하지 않고, `runs/<name>`에 TensorBoard 로그, `checkpoints/<name>/ckpt/`에 `epoch_*_Gen.pt`·`epoch_*_Dis_{0,1,2}.pt`, `checkpoints/<name>/res/`에 샘플 그리드 저장. 저장 주기와 무관하게 마지막 epoch도 강제 저장된다.
- 모니터링: `tensorboard --logdir=runs`.

> ⚠️ `train.sh`는 기본으로 **처음부터 학습**한다. 파일 끝의 resume 주석은 flag 조각일 뿐 별도 shell 명령이 아니다. 전체 resume 명령과 다중 GPU 절차는 [how-to/train-eval-infer.md](../how-to/train-eval-infer.md)를 따른다.

> ✅ 2026-07-10 이후 새 학습은 수정된 conditioning/alignment 기본값을 쓴다. 기존 metadata 없는 checkpoint는 호환을 위해 legacy 동작으로 로드되며, 수정 효과를 얻으려면 처음부터 다시 학습해야 한다([explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md)).

## 5. 추론 (텍스트 → 이미지)

```bash
# wrapper는 파일 안의 고정 prompt 사용
bash infer.sh ./checkpoints/<run_name>/ckpt 149

# 원하는 prompt를 직접 지정
PYTHONPATH=. python scripts/infer.py \
  --checkpoint_path ./checkpoints/<run_name>/ckpt \
  --load_epoch 149 \
  --eval_data_path None \
  --prompt "a portrait of a person with blond hair"
```

- `infer.sh` positional은 `<CKPT_DIR> [EPOCH]`뿐이며 세 번째 prompt 인자는 받지 않는다. 결과는 `./output/result_64.png`·`result_128.png`·`result_256.png`.
- 추론은 **생성기 체크포인트만** 필요하다(판별기 불필요).

## 6. 다음 단계

1. [how-to/prepare-dataset.md](../how-to/prepare-dataset.md) — 데이터 준비 상세
2. [how-to/train-eval-infer.md](../how-to/train-eval-infer.md) — 평가(FID/IS/CLIP)·resume·다중 GPU
3. [explanation/architecture.md](../explanation/architecture.md) — 동작 원리
4. [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md) — ⭐ 수치 인용 전 필독
