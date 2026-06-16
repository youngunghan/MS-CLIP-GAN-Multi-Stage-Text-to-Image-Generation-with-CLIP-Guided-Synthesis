# How-to: 데이터셋 준비 (MM-CelebA-HQ)

> **범위:** 원본 MM-CelebA-HQ에서 train/test 분할 → CLIP 피처 추출 전처리 → 학습용 zip 생성. 출력 zip 내부 스키마는 [reference/dataset-format.md](../reference/dataset-format.md).
> **대상:** 데이터를 준비하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-16.

## 1. 원본 데이터 배치

`data/` 아래에 MM-CelebA-HQ를 둔다. 전처리 스크립트는 원본 디렉터리 안의 `image.zip`·`text.zip`(`celeba-caption/*.txt`)을 기대한다([preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `open_image_zip()`).

```
data/
└── mm-celeba-hq-dataset/
    ├── image.zip          # images/*.jpg
    └── text.zip           # celeba-caption/*.txt
```

> 🟢 전처리 입력 `--source`는 **image.zip·text.zip을 담은 디렉터리**여야 한다(단일 zip 파일을 주면 명확한 에러로 거부, [preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `open_dataset()`).

## 2. train/test 분할

```bash
bash preprocessing/split_dataset.sh   # 기본 train_ratio 0.85, seed 42
```

- [split_dataset.py](../../preprocessing/split_dataset.py) `split_dataset()`가 zip 내 이미지(확장자 기준, 경로 prefix 무관)를 시드 셔플 후 분할 → `data/celeba_filenames_train.pickle`·`data/celeba_filenames_test.pickle`.
- `data/` 디렉터리는 자동 생성된다.

## 3. CLIP 피처 추출 전처리

```bash
bash preprocessing/preprocess_train.sh   # → data/trainset.zip
bash preprocessing/preprocess_test.sh    # → data/testset.zip
```

각 스크립트는 [preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `convert_dataset()`(StyleGAN2-ADA 파생)를 호출해 단계별로:

1. center-crop + resize → **256×256** PNG로 저장.
2. 이미지마다 CLIP(ViT-B/32) `encode_image` 피처(512차원) 1개.
3. 캡션마다(최대 10개) CLIP `encode_text` 피처. 너무 길면 문장/구로 쪼개 평균.
4. `dataset.json`에 `clip_img_features`·`clip_txt_features`로 직렬화.

| 옵션 | 값 | 의미 |
|---|---|---|
| `--transform` | `center-crop` | 정사각 크롭 |
| `--width`/`--height` | 256 | 출력 해상도(2의 거듭제곱) |
| `--emb_dim` | 512 | CLIP ViT-B/32 피처 차원 |

> 🟢 캡션 피처가 0개인 샘플은 **건너뛴다**(이미지·피처 모두 미저장) — 빈 임베딩이 데이터로더에서 IndexError를 내지 않도록([preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `convert_dataset()`).

## 4. 결과 확인

```
data/
├── trainset.zip      # 00000/img00000000.png ... + dataset.json
└── testset.zip
```

zip 내부 스키마와 데이터로더가 이를 어떻게 읽는지는 [reference/dataset-format.md](../reference/dataset-format.md). 학습은 [how-to/train-eval-infer.md](train-eval-infer.md).

## 관련 문서

- [reference/dataset-format.md](../reference/dataset-format.md) — zip/`dataset.json` 스키마
- [reference/cli.md](../reference/cli.md) — 전처리 스크립트 인자
