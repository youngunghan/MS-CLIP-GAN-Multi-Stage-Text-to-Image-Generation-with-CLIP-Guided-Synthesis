# How-to: experiments/ 서브셋 실험 워크플로

> **범위:** `experiments/` 아래 스크립트로 소규모 서브셋 데이터를 준비 → 학습 → FID/IS 곡선 평가 → 플롯까지 돌리는 절차. 정식 학습/평가/추론은 [how-to/train-eval-infer.md](train-eval-infer.md).
> **대상:** correctness audit 이후 정량 실험을 재현·확장하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-07-07.

`experiments/`는 루트의 `train.sh`/`eval.sh`와 별도로, 작은 서브셋(HF `celebahq-caption-10k`)에서 빠르게 학습→FID/IS 곡선을 뽑기 위한 스크립트 모음이다. **결과 수치와 해석은 여기서 다루지 않는다** — 전부 [experiments/RESULTS.md](../../experiments/RESULTS.md)를 본다. 이 문서는 어떤 스크립트가 무엇을 하는지와 실행 순서만 정리한다.

## 1. 스크립트 한눈에 보기

| 스크립트 | 역할 |
|---|---|
| [data_prep.sh](../../experiments/data_prep.sh) | HF에서 N장 다운로드(`dl_real_scaled.py`) → train/test 분할 → CLIP 전처리 → `data/{trainset,testset}_<TAG>.zip` |
| [dl_real_scaled.py](../../experiments/dl_real_scaled.py) | `data_prep.sh` 1단계가 호출. `ixw/celebahq-caption-10k`에서 N장을 스트리밍 다운로드해 `data/mm-celeba-hq-dataset/{image,text}.zip` 생성(결정적 순서라 부분집합이 항상 상위 집합의 앞부분) |
| [train.sh](../../experiments/train.sh) | 루트 `train.sh`와 동명이지만 다른 스크립트: 위치 인자로 이름/데이터/에폭/save_freq/배치를 받고, GPU 메모리·사용률 샘플링 + 학습 로그 + 메타 요약(wall time 등)까지 남긴다 |
| [eval_curve.py](../../experiments/eval_curve.py) | 저장된 `epoch_*_Gen.pt`마다 표준(torchmetrics, 2048-d) FID/IS를 계산 — 고정 프롬프트가 아니라 **각 테스트 캡션**으로 조건화 |
| [plot_curves.py](../../experiments/plot_curves.py) | 학습 로그의 epoch별 d_loss/g_loss + `eval_curve.py` 결과(FID/IS vs epoch)를 나란히 플롯 |
| [plot_compare.py](../../experiments/plot_compare.py) | 여러 런의 FID-vs-epoch 곡선을 한 그래프에 겹쳐 비교 |
| [run_all.sh](../../experiments/run_all.sh) | data_prep → train → eval_curve → plot_curves를 한 번에 실행하는 end-to-end 러너(기본 베이스라인 설정) |
| [run_stable.sh](../../experiments/run_stable.sh) | 이미 준비된 서브셋으로 EMA + TTUR(낮은 D LR) + 단측 라벨 스무딩 안정화 실험 → train → eval → plot |
| [run_sweep.sh](../../experiments/run_sweep.sh) | D를 더 세게/약하게 누르는 설정 2종(`swA_aggr`, `swB_mild`)을 40 epoch 짧은 프로브로 순차 실행 |
| [run_diffaug.sh](../../experiments/run_diffaug.sh) | 베이스라인 설정 + DiffAugment만 추가한 실험 → train → eval → plot + `plot_compare.py`로 베이스라인과 오버레이 |
| [fid_feature_space_demo.py](../../experiments/fid_feature_space_demo.py) | 같은 체크포인트/이미지에 대해 ignite 기본 FID(1000-d logits, 버그가 있는 스케일)와 torchmetrics FID(2048-d, 표준)를 나란히 계산해 두 스케일의 괴리를 보여주는 데모 |

각 스크립트는 절대경로(`REPO="/home/yuhan/repo/MS-CLIP-GAN-..."`)로 `cd`한 뒤 실행되므로, 이 저장소 경로가 다르면 스크립트 상단의 `REPO` 값을 맞춰야 한다.

## 2. End-to-end 예시

```bash
# N=3000장 다운로드, 0.83 분할, tag=sub, 런 이름=sub25, 100 epoch, save_freq=10
experiments/run_all.sh 3000 0.83 sub sub25 100 10
```

내부적으로 `data_prep.sh` → `experiments/train.sh` → `eval_curve.py` → `plot_curves.py` 순으로 실행되며, 각 단계 실패 시 이후 단계로 넘어가지 않고 종료한다. 결과는 `experiments/results/<NAME>/`(`eval.json`, `curves.png`)에 쌓인다.

## 3. 변형 실험

- **안정화 시도(EMA/TTUR/라벨 스무딩):** `experiments/run_stable.sh [NAME] [TAG] [EPOCHS] [SAVE] [D_LR] [EMA_DECAY] [SMOOTH]`
- **D 약화 스윕:** `experiments/run_sweep.sh` (인자 없음, 스크립트 내부 `CONFIGS` 배열 편집)
- **DiffAugment:** `experiments/run_diffaug.sh [NAME] [TAG] [EPOCHS] [SAVE] [POLICY]` — 기본값이 100-epoch 재현 실행, 짧은 프로브는 `experiments/run_diffaug.sh diffaug sub 50 5`처럼 인자를 준다.

이 스크립트들이 켜는 학습 플래그(`--use_ema`, `--d_lr`, `--real_label_smooth`, `--d_update_every`, `--use_diffaugment`, `--diffaugment_policy`)의 의미는 [reference/configuration.md](../reference/configuration.md)에 있다.

## 4. 결과 확인

- 런별 원자료: `experiments/results/<NAME>/eval.json`(에폭별 FID/IS), `curves.png`(손실+FID/IS 곡선).
- 비교 오버레이: `experiments/results/compare_fid.png`(D-스윕 비교), `experiments/results/compare_diffaug.png`(DiffAugment vs 베이스라인).
- **수치 해석·결론(베스트 에폭, 밴드 비교, 검증된/기각된 가설 등)은 전부 [experiments/RESULTS.md](../../experiments/RESULTS.md)를 본다** — 이 문서는 실행 방법만 다룬다.

## 관련 문서

- [experiments/RESULTS.md](../../experiments/RESULTS.md) — 실험 결과·해석
- [reference/configuration.md](../reference/configuration.md) — `--use_ema` 등 실험용 플래그 정의
- [how-to/train-eval-infer.md](train-eval-infer.md) — 정식 학습/평가/추론
- [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md) — FID 측정 타당성(스케일 문제 등)
