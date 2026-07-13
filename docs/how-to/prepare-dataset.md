# How-to: 데이터셋 준비 (MM-CelebA-HQ)

> **범위:** 원본 MM-CelebA-HQ에서 train/test 분할 → CLIP 피처 추출 전처리 → 학습용 zip 생성. 출력 zip 내부 스키마는 [reference/dataset-format.md](../reference/dataset-format.md).
> **대상:** 데이터를 준비하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-07-10.

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
- image↔caption identity는 `Path.stem`이다. 서로 다른 경로·확장자가 같은 stem을
  쓰면 어느 caption과 짝지을지 모호하므로 충돌 경로 전체를 출력하고 분할 전에 실패한다.

## 3. CLIP 피처 추출 전처리

```bash
bash preprocessing/preprocess_train.sh   # → data/trainset.zip
bash preprocessing/preprocess_test.sh    # → data/testset.zip
```

각 스크립트는 [preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `convert_dataset()`(StyleGAN2-ADA 파생)를 호출해 단계별로:

1. center-crop + resize → **256×256** PNG로 저장.
2. 이미지마다 CLIP(ViT-B/32) `encode_image` 피처(512차원) 1개.
3. 비어 있지 않은 캡션마다(최대 10개) `clip.tokenize(..., truncate=True)` 후 CLIP `encode_text` 피처.
4. `dataset.json`에 `clip_img_features`·`clip_txt_features`와 전처리 seed/config를 직렬화.

| 옵션 | 값 | 의미 |
|---|---|---|
| `--transform` | `center-crop` | 정사각 크롭 |
| `--width`/`--height` | 256 | 출력 해상도(2의 거듭제곱) |
| `--seed` | 42 | image-feature crop용 Python/NumPy/Torch/CUDA RNG; deterministic cuDNN 설정 |
| `--emb_dim` | 512 | CLIP ViT-B/32 피처 차원 |

`--emb_dim`은 필수이며 `512`만 허용한다. source image는 mode가 grayscale/RGBA여도
RGB로 정규화한 뒤 저장한다.

> ✅ 정상 pipeline의 ZIP 출력은 sibling temporary archive에 먼저 완성한다. image/text
> decode, 빈 caption, 선택 stem 누락 등 **단 한 sample이라도 실패하면 전체 run을
> non-zero로 끝내고 기존 destination ZIP을 보존**한다. 모든 sample과 metadata가 맞을
> 때만 fsync 후 `os.replace()`한다. `KeyboardInterrupt`, `SystemExit`, CUDA OOM도
> 삼키지 않는다. directory destination은 이 atomic-replace 계약 대상이 아니다.
>
> ✅ ZIP entry timestamp·permission과 JSON 순서를 고정하고 model load 뒤 `--seed`를
> 적용한다. 같은 input·code·package·device 환경에서는 byte-stable ZIP을 목표로 한다.
> `dataset.json.preprocess`가 seed, CLIP model/dimension, crop/resize config를 기록한다.
> 단, raw HF revision과 split pickle hash는 processed ZIP 안에 내장하지 않으므로
> `download_provenance.json`과 두 pickle도 함께 보존한다.

## 4. 결과 확인

```
data/
├── trainset.zip      # 00000/img00000000.png ... + dataset.json
└── testset.zip
```

zip 내부 스키마와 데이터로더가 이를 어떻게 읽는지는 [reference/dataset-format.md](../reference/dataset-format.md). 학습은 [how-to/train-eval-infer.md](train-eval-infer.md).

```bash
# image 수와 metadata 존재 확인
python - <<'PY'
import json, zipfile
for path in ("data/trainset.zip", "data/testset.zip"):
    with zipfile.ZipFile(path) as archive:
        pngs = [name for name in archive.namelist() if name.endswith(".png")]
        metadata = json.loads(archive.read("dataset.json"))
        print(path, len(pngs), len(metadata["clip_txt_features"]))
PY
```

두 수가 같고 0보다 커야 한다. HF 서브셋 download까지 포함한 실험 pipeline은
`environment.yml`의 같은 `msclipgan` 환경에서 `datasets`를 사용한다
([how-to/run-experiments.md](run-experiments.md)). 별도 `msclipgan-smoke` 환경은 필요 없다.

## 관련 문서

- [reference/dataset-format.md](../reference/dataset-format.md) — zip/`dataset.json` 스키마
- [reference/cli.md](../reference/cli.md) — 전처리 스크립트 인자
