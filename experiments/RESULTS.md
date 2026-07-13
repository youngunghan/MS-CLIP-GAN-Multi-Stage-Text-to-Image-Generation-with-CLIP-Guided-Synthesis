# MS-CLIP-GAN 서브셋 실험 결과와 FID 조사

> **범위:** 2026-06-18~19에 수행한 25% 서브셋 실험의 역사적 결과, 측정 조건, 해석 한계. 현재 코드로 새 실험을 실행하는 절차는 [docs/how-to/run-experiments.md](../docs/how-to/run-experiments.md).
> **대상:** 실험 결과를 재현·비교·인용하는 개발자.
> **상태:** 역사적 결과 보존 + 2026-07-10 정확성 감사 반영. 아래 JSON 수치는 다시 쓰지 않았으며, 새 기본 동작의 재학습 결과가 아니다.

이 문서의 FID/IS 값은 커밋된 `experiments/results/<run>/eval.json`을 그대로
기록한다. 모두 **torchmetrics Inception pool3 2048-d FID**이지만, 이 JSON을 만든
평가에는 seed가 기록되지 않아 정확한 evaluation seed를 알 수 없다. 이후 commit
`5d0de43`이 evaluator loop 앞에서 seed를 한 번 설정했으나, 그 방식도 앞서 평가한
checkpoint 목록·순서에 따라 생성 noise가 달라졌다. 현재 evaluator는 checkpoint별
난수를 독립 고정하고 실행 provenance를 남기지만 역사적 JSON을 소급 수정하지 않는다.

## 1. 요약

- 원본 notebook의 `FID 0.2423`은 ignite 기본 1000-d logits feature에서 계산한
  비표준 스케일이다. 같은 모델·데이터의 표준 2048-d FID는 약 205였고, 두
  feature space 사이에는 고정 변환식이 없다.
- `sub25` baseline의 역사적 최솟값은 epoch 20의 **FID 163.3**이다. 이
  checkpoint가 `models/best_sub25_ep20/`에 승격된 유일한 artifact이지만,
  **수치상 전체 최저는 아니다**. D-약화 probe `swB_mild`는 159.3,
  `diffaug100`은 118.5를 기록했다.
- D-약화 4개 run의 최솟값은 159~184 범위였다. 이는 **이 단일-seed probe에서
  D 약화가 명확한 개선을 보이지 않았다는 관찰**이지, discriminator dominance를
  일반적으로 반증한 결과가 아니다.
- DiffAugment를 켠 `diffaug100`의 역사적 최솟값은 epoch 90의 **FID 118.5**로
  더 낮았다. 다만 1개 학습 seed, 510장 test set, 같은 test set을 사용한
  checkpoint 선택이라는 한계 때문에 효과 크기나 통계적 유의성을 확정할 수 없다.
- 이 실험의 checkpoint는 현재 수정 전의 conditioning/alignment 동작으로
  학습됐다. loader는 이 legacy 동작을 유지해 기존 가중치를 읽지만, 수정된
  기본 동작의 효과를 얻으려면 **처음부터 재학습**해야 한다.

## 2. 설정과 provenance

| 항목 | 역사적 실험 조건 |
|---|---|
| 데이터 | Hugging Face `ixw/celebahq-caption-10k`에서 3,000장 다운로드 → seed 42로 0.83 분할. 당시 HF revision은 미기록 |
| 표본 수 | train 2,490 / test 510, 교집합 0 |
| 학습 seed | 각 run 42, 반복 seed 없음 |
| 평가 seed | 역사적 JSON에 미기록; 정확한 값 불명 |
| 평가 조건 | test 이미지마다 저장된 **첫 번째 캡션 임베딩 1개**로 fake 1장 생성 |
| metric | `FrechetInceptionDistance(normalize=False)` 2048-d pool3 + `InceptionScore`, uint8 입력 |
| 장비 | RTX 4060 Ti 8 GB, batch 4, 약 5.8~6.2 GB peak, full-D run 약 204초/epoch |
| 결과 원자료 | `experiments/results/<run>/eval.json`; baseline/D sweep 기록 커밋 `5c9d75d` (2026-06-18), DiffAugment 기록 커밋 `f64515f` (2026-06-19) |

🟠 역사적 run에는 실행 당시의 정확한 Git commit, package lock, CLIP Git revision,
HF dataset revision, GPU/driver 정보가 machine-readable metadata로 남아 있지 않다.
따라서 위 커밋은 **결과 파일이 저장소에 기록된 시점**이지 실행 binary의 완전한
provenance를 보장하지 않는다.

🟠 checkpoint/설정 선택과 최종 보고에 같은 510장 test set을 썼다. 별도 validation
split이 없으므로 선택 편향이 있으며, 작은 표본 FID의 유한표본 편향과 생성 noise도
크다. 비교를 확정하려면 validation으로 checkpoint를 선택하고, 보류한 test set에서
여러 학습·평가 seed의 평균과 분산(또는 KID)을 보고해야 한다.

🟠 2026-07-10 감사에서 evaluation seed 42를 checkpoint별로 독립 고정해 단독 spot-check한
값은 baseline epoch 20이 약 164.3, DiffAugment epoch 90이 약 117.5였다. 큰 개선
방향은 유지됐지만, 역사적 JSON의 163.3/118.5와 정확히 같지 않다. 이 spot-check는
커밋된 원자료를 덮어쓰지 않는다.

## 3. D-balance run

epoch 번호는 0부터 시작한다. 당시 학습 코드는 마지막 epoch를 강제 저장하지 않아,
`num_epochs=100`은 epoch 0~99를 학습했어도 `save_freq=10`이면 마지막 저장·평가는
epoch 90이었다. 아래 `last eval`은 **마지막 학습 epoch가 아니라 마지막 저장·평가
checkpoint**다.

| run | 설정 | 학습 범위 | best FID | @epoch | last eval |
|---|---|---:|---:|---:|---:|
| `sub25` | baseline, G/D lr 2e-4 | 0~99 | 163.3 | 20 | 269.6 @90 |
| `sub25_stable` | EMA 0.999 + D lr 1e-4 + real label 0.9 | 0~99 | 178.7 | 60 | 264.0 @90 |
| `swA_aggr` | EMA + D lr 2e-5 + D 매 3 step | 0~39 | 183.6 | 20 | 184.4 @35 |
| `swB_mild` | EMA + D lr 5e-5 + D 매 2 step | 0~39 | **159.3** | 35 | 159.3 @35 |

원자료 곡선은 `experiments/results/<run>/curves.png`, 비교 이미지는
[`experiments/results/compare_fid.png`](results/compare_fid.png)다. 변동 예시는
`swB_mild`의 252.4→164.7→159.3(epoch 25→30→35), `sub25_stable`의
194.5→369.8→178.7(epoch 30→50→60)이다.

당시 사전 기준은 FID `<150`만 D-balance 가설을 다시 여는 것이었다. 네 run 중 그
기준을 통과한 값은 없었다. 그러므로 이 범위에서 내릴 수 있는 결론은
“시험한 D-약화 설정이 robust한 개선 증거를 만들지 못했다”까지다. `swB_mild`의
159.3은 baseline 163.3보다 낮지만 단 한 seed·마지막 평가 checkpoint이고, 평가
noise와 선택 편향을 분리할 반복이 없다. “D-dominance가 반증됐다”, “~160이
architecture ceiling이다”, “mode collapse가 확인됐다”는 표현은 증거보다 강하다.

## 4. DiffAugment run

두 run은 baseline에 `color,translation,cutout` DiffAugment를 켜는 것을 의도한
비교다. real/fake가 D에 들어갈 때만 증강하며 CLIP/VGG 보조 손실은 raw fake를 본다.
다만 무작위 학습을 한 번씩만 실행했으므로 “유일한 원인”이라는 인과 결론 대신
후속 반복이 필요한 유망한 차이로 해석한다.

| run | 요청한 epoch 수 | 실제 학습 범위 | 마지막 저장·평가 | best FID | @epoch |
|---|---:|---:|---:|---:|---:|
| `diffaug` (probe) | **50** | 0~49 | 45 | 173.3 | 40 |
| `diffaug100` | 100 | 0~99 | 90 | **118.5** | 90 |

`diffaug100` 역사적 곡선은 163.0@50 → 119.6@60 → 141.1@70 → 121.9@80 →
118.5@90이다. epoch 80과 90이 모두 pre-DiffAugment 최솟값보다 낮아 후속 검증의
우선순위는 높지만, epoch 90은 **마지막 학습 epoch 99가 아니라 마지막 저장·평가
checkpoint**다. 비교 이미지는
[`experiments/results/compare_diffaug.png`](results/compare_diffaug.png)다.

## 5. Artifact 상태와 호환성

- `models/best_sub25_ep20/`: baseline epoch 20. 사람이 승격·잠근 유일한 artifact다.
  “best”는 **promoted selection**이라는 뜻이며 저장소 전체의 수치 최저를 뜻하지 않는다.
- `diffaug100` epoch 90: 더 낮은 역사적 FID지만 `models/`에 승격하지 않았다. raw
  checkpoint는 git-ignored 로컬 경로
  `checkpoints/diffaug100-2026_06_19_08_51_13/ckpt/`에만 있다.
- 두 artifact를 포함한 모든 위 run은 legacy conditioning activation과
  text-conditioned alignment head로 학습됐다. checkpoint metadata가 없는 기존
  파일은 loader가 legacy 동작으로 읽는다. 새 기본 conditioning/alignment를
  사용하도록 강제해 옛 가중치를 해석하거나, 새 기본값의 성능으로 이 수치를
  인용하면 안 된다.

샘플이 흐리거나 왜곡된 것은 수치와 함께 관찰된 정성적 한계다. 수정 전 원본 코드의
64px target 업샘플 학습은 detail이 적어 더 깨끗해 보일 수 있었지만, 현재 데이터
경로는 실제 256px target을 보존하므로 직접 시각 비교가 공정하지 않다.

## 6. 다음 실험

1. 수정된 conditioning/alignment 기본값으로 처음부터 재학습한다. legacy checkpoint
   resume으로는 수정 효과를 얻을 수 없다.
2. 학습 seed를 최소 3개 이상 반복하고, checkpoint 선택용 validation과 최종 test를
   분리한다. FID와 함께 KID 또는 bootstrap 불확실성을 기록한다.
3. DiffAugment on/off를 동일 protocol에서 재실행한다. 그 뒤 full accessible data에서
   반복해 data-scale 효과를 분리한다.
4. 실행 metadata가 기록한 Git commit, dirty state, Python/PyTorch/CUDA/CLIP 버전,
   dataset 및 split fingerprint, seed, checkpoint 목록을 결과와 함께 보존한다.

## 7. 현재 코드로 실행

아래 명령은 파일 실행 bit에 의존하지 않는다. shell은 `bash`로, repository module을
import하는 Python 진입점은 `PYTHONPATH=.`로 실행한다.

```bash
# data prep (environment.yml의 msclipgan 환경 하나를 사용)
bash experiments/data_prep.sh 3000 0.83 sub

# baseline end-to-end
bash experiments/run_all.sh 3000 0.83 sub sub25_new 100 10

# stability / D-balance / DiffAugment
bash experiments/run_stable.sh sub25_stable_new sub 100 10 1e-4 0.999 0.9
bash experiments/run_sweep.sh sweep_new sub 40 5
bash experiments/run_diffaug.sh diffaug100_new sub 100 10

# 이미 존재하는 checkpoint 곡선 평가
PYTHONPATH=. python experiments/eval_curve.py \
  data/testset_sub.zip checkpoints/<run-name>/ckpt auto \
  experiments/results/generated/<run-name>/eval.json
```

현재 코드는 checkpoint별 평가 seed, 빈 checkpoint 집합 실패, 마지막 epoch 강제 저장,
provenance 기록 및 새 conditioning/alignment 기본값을 적용한다. 따라서 새 run은 위
역사적 JSON의 byte-for-byte 재현이 아니라 **수정된 protocol의 새 결과**다.
자동 runner는 커밋된 역사적 JSON을 덮어쓰지 않고 Git-ignore된
`experiments/results/generated/`에만 새 결과를 쓴다. DiffAugment overlay는 같은 현재
protocol baseline JSON을 여섯 번째 인자로 줄 때 provenance를 검증한 뒤에만 만든다.

## 관련 문서

- [docs/how-to/run-experiments.md](../docs/how-to/run-experiments.md) — 실행 절차
- [docs/explanation/correctness-and-fixes.md](../docs/explanation/correctness-and-fixes.md) — 수정과 결과 인용 한계
- [models/best_sub25_ep20/README.md](../models/best_sub25_ep20/README.md) — 승격 artifact 계약
