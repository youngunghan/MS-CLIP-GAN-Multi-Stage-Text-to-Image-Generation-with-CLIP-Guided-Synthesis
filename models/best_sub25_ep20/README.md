# 승격 모델 — `sub25` baseline epoch 20

> **범위:** `models/best_sub25_ep20/`의 로컬 checkpoint가 무엇인지, 어떤 조건에서 측정됐는지, 평가·추론하는 방법. 전체 실험 해석은 [experiments/RESULTS.md](../../experiments/RESULTS.md).
> **대상:** 이 artifact를 평가·추론·비교하는 개발자.
> **상태:** legacy artifact 계약 확인 — 기준일 2026-07-10.

이 디렉터리는 `sub25` baseline에서 사람이 선택해 승격한 epoch 20 artifact다.
이름의 “best”는 **유일하게 promoted/locked된 selection**이라는 뜻이며, 저장소에서
측정된 FID의 수치상 최저라는 뜻이 아니다.

## 1. 학습·평가 기록

| 항목 | 값 |
|---|---|
| run | `sub25`, epoch 0~99 학습, epoch 20 checkpoint |
| 데이터 | train 2,490 / test 510, 학습 seed 42 |
| 역사적 평가 seed | 기록되지 않아 불명 |
| 설정 | Adam G/D 2e-4, CosineAnnealing, EMA/TTUR/label smoothing/DiffAugment 없음 |
| 역사적 FID | 163.2953, torchmetrics 2048-d pool3, test 이미지마다 첫 caption 1개 |
| 2026-07-10 단독 spot-check | 약 164.3, checkpoint별 seed 독립 고정 |
| 선택 상태 | promoted artifact; 수치상 최저 아님 |

`swB_mild`는 역사적 FID 159.3, 승격하지 않은 `diffaug100` epoch 90은 118.5를
기록했다. 모두 1개 학습 seed와 같은 510장 test set으로 checkpoint를 고르고
보고했으므로 근소한 순위나 통계적 우월성을 확정할 수 없다. 자세한 provenance와
평가 RNG 한계는 [§2 설정과 provenance](../../experiments/RESULTS.md#2-설정과-provenance)를 본다.

## 2. Legacy 동작 계약

이 checkpoint는 2026-07-10 수정 이전에 학습되어 다음 legacy 동작을 사용한다.

- conditioning augmentation의 `mu`와 `log_sigma` 앞에 ReLU 적용
- alignment head 입력에 이미지 특징과 text-derived `mu`를 함께 사용

기존 `.pt`에는 이 동작을 설명하는 format metadata가 없다. 현재 loader는 metadata가
없는 artifact를 legacy 모드로 자동 해석하므로 평가·추론은 가능하다. 그러나 새
학습 기본값의 linear conditioning과 image-only alignment 효과는 기존 가중치에
소급 적용되지 않는다. 수정 효과를 검증하려면 새 run을 처음부터 학습해야 한다.

## 3. 파일

| 파일 | 용도 |
|---|---|
| `epoch_20_Gen.pt` | 평가·추론용 generator |
| `epoch_20_Dis_{0,1,2}.pt` | 단계별 discriminator; legacy resume용 |
| `sample_epoch20.png` | 이 checkpoint의 sample grid |

`*.pt`는 `.gitignore` 대상이라 이 작업공간에만 있고 fresh clone에 weight가 있다고
가정하면 안 된다. `sample_epoch20.png`, 이 README,
[experiments/RESULTS.md](../../experiments/RESULTS.md)는 커밋된 정성·계약 기록이다.

## 4. 평가와 추론

repository root에서 실행한다. `PYTHONPATH=.`를 붙여 root module import를 보장한다.

```bash
# 역사적 protocol과 달리 현재 evaluator는 checkpoint별 seed와 provenance를 기록한다.
PYTHONPATH=. python experiments/eval_curve.py \
  data/testset_sub.zip models/best_sub25_ep20 20 /tmp/best_eval.json

# 직접 prompt를 지정한 추론
PYTHONPATH=. python scripts/infer.py \
  --checkpoint_path models/best_sub25_ep20 \
  --load_epoch 20 \
  --eval_data_path None \
  --prompt "a photo of a person"
```

평가에는 generator만 필요하다. 이 artifact에는 같은 epoch의 G/D와
optimizer/scheduler가 있어 best-effort legacy resume은 가능하지만, training config,
data/source/runtime provenance와 RNG metadata가 없어 검증된 exact resume은 아니다.
연구 결과 보존용이며 새 기본 동작 학습의 출발점으로 권장하지 않는다.

## 관련 문서

- [experiments/RESULTS.md](../../experiments/RESULTS.md) — 전체 역사적 결과와 한계
- [docs/explanation/correctness-and-fixes.md](../../docs/explanation/correctness-and-fixes.md) — 현재 수정 사항
