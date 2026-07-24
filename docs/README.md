# MS-CLIP-GAN docs — CLIP 가이드 다단계 텍스트→이미지 생성

CLIP 텍스트 임베딩을 조건으로 64→128→256 해상도를 단계적으로 생성하는 다단계 GAN(StackGAN++ 계열 + LAFITE/AttnGAN 아이디어). MM-CelebA-HQ 얼굴-캡션 데이터로 학습한다. 코드 기준으로 작성한 개발자 문서다.

## 문서 목록 (Diátaxis)

### tutorials — 처음 따라하기

| 문서 | 설명 |
|---|---|
| [tutorials/quickstart.md](tutorials/quickstart.md) | 환경 설치 → 전처리 → 학습 → 텍스트 프롬프트로 추론까지 happy path |

### how-to — 과업 가이드

| 문서 | 설명 |
|---|---|
| [how-to/prepare-dataset.md](how-to/prepare-dataset.md) | MM-CelebA-HQ 분할 → CLIP 피처 추출 전처리 → `trainset.zip`/`testset.zip` |
| [how-to/train-eval-infer.md](how-to/train-eval-infer.md) | fresh 학습·exact resume/기간 연장·평가(FID/IS/CLIP)·prompt 추론 |
| [how-to/troubleshooting.md](how-to/troubleshooting.md) | 증상별 문제 해결(OOM·체크포인트 로드·CLIP·데이터 경로) |
| [how-to/run-experiments.md](how-to/run-experiments.md) | 서브셋 데이터 provenance→학습→checkpoint별 결정적 FID/IS/CLIP+provenance→plot; conditioning 복구 recipe·prompt-sensitivity 측정; 역사적 결과는 `experiments/RESULTS.md` |

### reference — 조회용 명세

| 문서 | 설명 |
|---|---|
| [reference/configuration.md](reference/configuration.md) | CLI 옵션(base/train/test) · 버전 핀 · 하이퍼파라미터 표 |
| [reference/cli.md](reference/cli.md) | 스크립트·셸 스크립트 진입점과 인자 |
| [reference/dataset-format.md](reference/dataset-format.md) | 전처리 출력 zip 구조 · `dataset.json` 임베딩 스키마 · 데이터로더 계약 |

### explanation — 깊은 설명

| 문서 | 설명 |
|---|---|
| [explanation/architecture.md](explanation/architecture.md) | 다단계 G/D·linear/legacy CA(deterministic 옵션 포함)·image-only 정렬·DiffAugment·손실/업데이트 흐름 |
| [explanation/correctness-and-fixes.md](explanation/correctness-and-fixes.md) | ⭐ 정확성 감사 수정·checkpoint 호환·conditioning 붕괴 진단/복구·실행 검증·결과 인용 한계 |

## 읽기 순서

처음이면 tutorials → how-to → reference. 동작 원리·설계 근거는 explanation/architecture, **결과·수치를 인용하기 전에는 explanation/correctness-and-fixes 필독**(데이터 해상도·평가 타당성 한계 포함).
