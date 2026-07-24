# 정확성 감사 · 수정 · 남은 주의점

> **범위:** 코드 정확성 감사에서 확인·수정한 핵심 결함, legacy checkpoint 호환 계약, 결과를 인용하기 전 알아야 할 한계. 구조 설명은 [explanation/architecture.md](architecture.md).
> **대상:** 개발자·실험 결과를 인용하는 사람.
> **상태:** 구현·런타임 검증 반영 — 기준일 2026-07-23. conditioning 붕괴 진단·복구([§2.4](#24-2026-07-23-conditioning-붕괴-진단복구))까지 반영했으며, 다중 seed·held-out 통계 검증은 아직 없음.

## 1. 결론과 적용 범위

감사에서 데이터 해상도, 학습 조건, 메트릭 누적, checkpoint, 전처리, 조건 증강,
alignment shortcut, 평가 난수, resume 연속성을 수정했다. 새 학습은
`conditioning_activation=linear`, `alignment_mode=image_only`를 기본으로 사용한다.
metadata가 없는 기존 checkpoint는 자동으로 legacy `relu` +
`legacy_conditioned` 동작을 유지하므로 평가·추론은 가능하지만, **수정 효과를 얻으려면
처음부터 재학습**해야 한다.

기존 `experiments/results/**/eval.json`과 `models/best_sub25_ep20/`은 수정 전
conditioning/alignment와 과거 evaluator로 만든 역사적 기록이다. 현재 기본값의 성능
근거로 인용하지 않는다([experiments/RESULTS.md](../../experiments/RESULTS.md)).

새 기본값(`linear`+`image_only`)을 켠 채로 실제 학습해 보면 conditioning은 여전히
거의 응답하지 않았다. 2026-07-20~23 세 커밋은 그 증상을 드러낼 측정 도구를 먼저
만들고, 명백해 보이는 knob이 red herring임을 확인한 뒤, 근본 원인을 찾아 고쳤다 —
자세한 진단·수정·측정된 한계는 [§2.4](#24-2026-07-23-conditioning-붕괴-진단복구)와
[§5.2 남은 제한](#52-남은-제한)에 정리한다.

## 2. 적용된 수정

### 2.1 데이터·학습·평가 기반 수정

| 상태 | 영역 | 수정 전 | 현재 계약 | 위치 |
|---|---|---|---|---|
| ✅ | 데이터 해상도 | 64px로 줄인 뒤 128/256으로 재확대 | 256px를 보존해 64/128로 area downsample | [dataset/dataloader.py](../../dataset/dataloader.py) `MM_CelebA` |
| ✅ | 학습 조건 | D는 교란된 이미지 embedding, G는 text embedding | D/G 모두 같은 `txt_feature` 사용 | [scripts/trainer.py](../../scripts/trainer.py) `train_step()` |
| ✅ | D용 fake | G graph가 불필요하게 연결되고 train-mode BN buffer가 D-only forward에서도 갱신 | train-mode batch-stat 출력은 유지하되 `torch.no_grad()` + buffer snapshot/restore로 생성 | [scripts/trainer.py](../../scripts/trainer.py) `preserved_module_buffers()`·`train_step()` |
| ✅ | 평가 누적 | batch별 FID를 평균하고 uint8/normalize 계약 불일치 | 전체 표본을 한 metric에 누적, uint8 + `normalize=False`, 마지막에 1회 `compute()` | [scripts/eval.py](../../scripts/eval.py) `evaluate()` |
| ✅ | checkpoint | DataParallel `module.` 키와 경로 중첩으로 load 실패 | unwrap 저장, prefix 호환 load, infer는 G만 요구 | [utils/utils.py](../../utils/utils.py) `save_checkpoint()`·`load_checkpoint()` |
| ✅ | 추론 | 저장 경로 결합 오류, train mode, 3단계 하드코딩 | `os.path.join`, `G.eval()`, 단계 수 기반 출력 | [scripts/infer.py](../../scripts/infer.py) `main()` |
| ✅ | objective | `Sigmoid`+BCE와 WGAN-GP 혼용 | BCE GAN + discriminator spectral normalization | [criteria/loss.py](../../criteria/loss.py) `D_loss()` |

**해상도 수정 전후 (Figure 1)**

```text
수정 전: 256² PNG ──Resize──▶ 64² ──upsample──▶ 128²/256² 흐린 target
현재:    256² PNG ──────────▶ 256² 보존 ──area downsample──▶ 64²/128²
```

### 2.2 2026-07-10 조건화·학습 경로 수정

| 상태 | 영역 | 확인된 문제 | 현재 계약 | 위치 |
|---|---|---|---|---|
| ✅ [검증 반영 #4] (2026-07-10) | conditioning augmentation | ReLU 뒤에서 `mu`/`log_sigma`를 split해 음수를 표현하지 못하고 `sigma>=1`로 제한. `best_sub25_ep20`에서 `mu` 76.0%, `log_sigma` 99.85%가 정확히 0이고 `sigma` 범위가 1.000~1.081 | 이 repository의 포화를 피하도록 새 학습은 unconstrained linear. `--conditioning_activation {linear,relu}`; 기본 `linear`, metadata 없는 checkpoint는 `relu` | [networks/generator.py](../../networks/generator.py) `ConditioningAugmention` |
| ✅ [검증 반영 #5] (2026-07-10) | alignment | align head가 image feature와 text-derived `mu`를 함께 받아 text→text shortcut 가능 | `--alignment_mode {image_only,legacy_conditioned}`; 기본은 condition channel을 0으로 채우는 `image_only`, parameter shape는 유지 | [networks/discriminator.py](../../networks/discriminator.py) `AlignCondDiscriminator` |
| ✅ [검증 반영 #6] (2026-07-10) | mismatched condition | conditional D에 같은 batch의 틀린 real-text negative가 없음 | batch>1이면 mismatched BCE를 기본 추가하되 generated fake와 기존 negative mass를 절반씩 공유해 D 총 scale 보존. `--no_mismatched_condition`으로 비활성 | [criteria/loss.py](../../criteria/loss.py) `D_loss()` |
| ✅ [검증 반영 #7] (2026-07-10) | G update 중 D 상태 | 쓰지 않을 D gradient가 쌓이고 BatchNorm/spectral-normalization state가 변함 | G phase 동안 모든 D를 eval + freeze하고 gradient를 비운 뒤 원래 train/`requires_grad` 상태를 정확히 복원 | [scripts/trainer.py](../../scripts/trainer.py) `train_step()` |
| ✅ | D shared trunk | matched/wrong/unconditional head마다 같은 image feature extractor를 재실행해 BN/SN state와 계산량이 중복 | image당 trunk 1회, tensor-only detailed output으로 모든 요청 head가 feature 재사용(DataParallel 호환) | [networks/discriminator.py](../../networks/discriminator.py) `forward()` |
| ✅ | contrastive batch | batch 1이면 InfoNCE가 loss 0, gradient 0 | contrastive 사용 시 `batch_size>=2`를 요구하고 마지막 singleton remainder만 drop | [options/train_options.py](../../options/train_options.py) `TrainOptions.validate()`·[dataset/dataloader.py](../../dataset/dataloader.py) `get_dataloader()` |

linear CA는 `mu`와 `log_sigma`에 부호 제약을 두지 않는다. KL 항은 계속
`N(0,I)`를 향해 정규화한다. `image_only` alignment는 checkpoint tensor shape를
바꾸지 않고 condition channel만 0으로 채우므로 v1 weight를 읽을 수 있지만, loader가
metadata 없는 weight에 legacy mode를 선택해 과거 의미도 보존한다.

### 2.3 평가·resume·파이프라인 수정

| 상태 | 영역 | 현재 계약 | 위치 |
|---|---|---|---|
| ✅ [검증 반영 #8] (2026-07-10) | curve RNG | checkpoint마다 seed를 독립 재설정해 평가 목록·순서에 무관한 noise 사용. checkpoint가 0개면 non-zero 종료 | [experiments/eval_curve.py](../../experiments/eval_curve.py) |
| ✅ | 평가 provenance | 결과와 함께 seed, dataset/checkpoint/result JSON fingerprint, model/training/schedule config, 학습 provenance, Git/runtime/hardware를 machine-readable sidecar로 기록 | [experiments/eval_curve.py](../../experiments/eval_curve.py) |
| ✅ | eval/infer model config | checkpoint metadata를 model 생성 전에 읽어 v2 architecture를 적용하고, legacy 기본 차원을 parameter shape에서 추론 | [scripts/checkpoint_config.py](../../scripts/checkpoint_config.py) `apply_checkpoint_model_config()` |
| ✅ [검증 반영 #9] (2026-07-10) | checkpoint v2 | `format_version=2`, model/training/schedule config, 학습 data/source/runtime/hardware provenance, RNG, generator weight kind와 EMA raw marker 저장 | [utils/utils.py](../../utils/utils.py) `save_checkpoint()` |
| ✅ | exact resume | 저장 phase 끝·`T_max`·scheduler 진행 위치, loss/EMA/batch/save cadence/base LR와 실제 data/source/runtime/hardware fingerprint를 load 전 검증 | [utils/utils.py](../../utils/utils.py) `load_checkpoint()` |
| ✅ | 기간 연장 | `--new_optim`은 weight만 로드하고 남은 epoch에 새 cosine phase를 생성; 그 phase의 start/end/`T_max`도 저장되어 이후 중간 exact resume 가능 | [scripts/train.py](../../scripts/train.py) |
| ✅ | EMA resume | EMA run에서 raw companion과 metadata가 맞아야 resume; EMA와 raw optimizer를 조용히 혼합하지 않음 | [utils/utils.py](../../utils/utils.py) `load_checkpoint()` |
| ✅ | 마지막 저장 | `save_freq` 배수가 아니어도 `num_epochs-1` checkpoint와 sample을 강제 저장 | [scripts/train.py](../../scripts/train.py) |
| ✅ | 전처리 무결성 | 필수 embedding 차원·unique/missing stem 검증, 단 한 sample 실패도 전체 abort, 기존 ZIP 보존, seed·고정 ZIP metadata·preprocess config 기록 | [preprocessing/preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `convert_dataset()` |
| ✅ | CLI 계약 | RGB/단일 D 출력/단계 수/양수 범위와 loss·batch 조합을 parse 직후 검증; `--help`에 기본값·설명 노출 | [options/base_options.py](../../options/base_options.py) `BaseOptions.validate()`·[options/train_options.py](../../options/train_options.py) `TrainOptions.validate()` |

### 2.4 2026-07-23 conditioning 붕괴 진단·복구

| 상태 | 영역 | 확인된 문제 | 현재 계약 | 위치 |
|---|---|---|---|---|
| ✅ (2026-07-20) | 평가 가시성 | FID/IS는 caption을 무시해 conditioning 완전 붕괴가 드러나지 않았다 — caption을 바꿔도 256px 출력이 ~2/255만 움직였고, 서로 다른 checkpoint는 같은 caption에서 75.7/255 차이가 났다 | `eval_curve.py`가 매 epoch `clip_score`/`clip_diversity`를 추가로 기록(FID/IS와 같은 forward pass 재사용, 추가 forward 없음). 신규 `experiments/prompt_sensitivity.py`가 noise `z`와 CA epsilon을 고정하고 caption만 바꿔 pixel sensitivity를 직접 측정 | [experiments/eval_curve.py](../../experiments/eval_curve.py), [experiments/prompt_sensitivity.py](../../experiments/prompt_sensitivity.py) |
| ✅ (2026-07-22) | 명백해 보인 knob은 red herring | `--gamma`/`--lam`/warm-up 스케줄을 조정해도 conditioning 자체는 계속 ~0으로 죽어 있었다 — 실제로 움직인 것은 2차 증상인 FID 발산/붕괴뿐이었다. controlled 비교에서 conditioning 항(linear+image_only+mismatched)을 켜면 FID가 epoch 10/20/30에서 325.78→367.30→358.96로 발산하고(legacy는 281.98→266.76→249.44로 하강) CLIP diversity가 0.0111로 붕괴했다. 원인은 D-trunk를 공유하는 gamma alignment InfoNCE 항과 mismatched-condition negative가 D의 real/fake 분리 학습을 방해하는 것이었다 | `--gamma`(기본 5.0)·`--lam`(기본 10.0)·`--cond_warmup_epochs`(기본 0)·`--cond_ramp_epochs`(기본 0)로 이 항들의 크기와 도입 시점을 조절 가능. 기본값은 기존 하드코딩 동작을 그대로 재현 | [criteria/loss.py](../../criteria/loss.py), [scripts/trainer.py](../../scripts/trainer.py) |
| ✅ (2026-07-23) | 근본 원인 (측정됨, 두 요인 결합) | (1) conditioning-augmentation KL 정규화 항의 gradient가 CLIP-reward gradient의 7~50배라 `mu`의 caption-종속 변이를 붕괴시키고 `sigma=exp(log_sigma)`를 이론적 최적값 1.0에 고정한다. (2) `sigma`가 1로 고정된 채 `c_hat = mu + sigma·ε` reparameterization이 분산 1짜리 노이즈를 주입하는데, caption-종속 신호는 차원당 표준편차 0.001~0.005 수준이라 SNR이 ~0.001에 불과해 생성기가 conditioning을 우회하도록 학습된다 | `--kl_weight`(기본 1.0, `0`이면 KL 항 완전 제거)와 `--deterministic_cond`(기본 off, 켜면 `c_hat=mu`로 epsilon 없이 사용)로 두 원인을 모두 끌 수 있다. `--deterministic_cond`는 forward pass를 바꾸므로 v2 `model_config`에 저장되어 eval/infer/`eval_curve.py`/`prompt_sensitivity.py`가 재적용한다; metadata 없는 checkpoint는 `False`(기존 확률적 동작)로 해석 | [networks/generator.py](../../networks/generator.py) `ConditioningAugmention`, [scripts/trainer.py](../../scripts/trainer.py) |

이 두 flag를 함께 켜면(`--kl_weight 0 --deterministic_cond`) prompt-swap sensitivity가
~0.1/255에서 ~5~11/255로 회복된다. 검증에 쓴 전체 recipe와 재현 명령은
[how-to/run-experiments.md §4 변형 실행](../how-to/run-experiments.md#4-변형-실행)에 있다.

효과는 뚜렷하지만 작다. 회복된 응답은 여전히 한 자릿수~낮은 두 자릿수 /255
수준이고(서로 다른 checkpoint 간 75.7/255 규모와 비교), FID는 350~410대에 머문다.
conditioning 강도는 최근 실험에서 데이터양보다 batch size에 더 민감하게
반응했다(같은 데이터에서 batch 16이 batch 4 대비 prompt 응답을 대략 2배로 늘림).
학습 데이터를 3.6배(source `ixw/celebahq-caption-10k`의 표본 상한인 약 9,000장까지)
늘려도 FID 품질 상한은 오르지 않았다. 남은 지렛대는 데이터가 아니라 구조 쪽으로
보인다 — StackGAN 계열 conditioning augmentation 대신 상시 활성 cross-attention text
경로, 그리고 판별기의 alignment head가 real/fake head와 image trunk를 공유하지 않는
구조 등이다. conditioning 효과는 epoch 10~30 부근에서 정점을 찍은 뒤 매 run
감소하므로, recipe를 그대로 재현할 때도 early stopping이 필요하다.

## 3. Checkpoint 호환·재학습 계약

| checkpoint | load 시 동작 | 용도 | 주의 |
|---|---|---|---|
| v2, 전체 metadata 있음 | eval/infer는 저장 config로 model 생성; resume은 model/training/schedule/provenance와 RNG·optimizer state를 검증·복구 | 정확한 eval/infer, 검증된 exact resume | 기록 밖 외부 입력과 nondeterministic kernel까지 보장하지는 않음 |
| metadata 없는 legacy | parameter shape의 기본 차원을 추론하고 `relu` + `legacy_conditioned` 자동 선택 | 기존 weight eval/infer, optimizer/scheduler가 있을 때 best-effort resume | config/provenance/RNG가 없고 scheduler가 없으면 `--new_optim` 필요 |
| legacy weight를 새 기본값으로 강제 | 지원 계약 아님 | 사용하지 않음 | tensor가 load돼도 학습된 함수 의미가 달라짐 |
| 새 기본값 fresh run | `linear` + `image_only` | 수정 효과 평가 | 기존 FID와 직접 같은 모델로 간주하지 않음 |

`--new_optim`은 기간 연장이나 optimizer를 버리고 재시작하려는 명시적 선택이다.
이전 phase를 exact continuation하는 표기가 아니지만, 새 phase가 v2로 저장된 뒤에는
그 start/end/`T_max`를 이용해 해당 phase 중간을 exact resume할 수 있다. 자세한 명령은
[§3 Resume](../how-to/train-eval-infer.md#3-resume)에서 구분한다.

## 4. 결과 인용 한계

- **역사적 평가 RNG:** 커밋된 curve JSON은 evaluation seed 도입 전에 생성되어
  정확한 seed가 기록돼 있지 않다. 이후 commit `5d0de43`은 loop 앞에서 seed를 한 번
  설정했지만 이 방식도 checkpoint 목록·순서에 의존했다. 2026-07-10 evaluation seed
  42의 checkpoint별 독립 spot-check에서 baseline epoch 20은 약 164.3,
  DiffAugment epoch 90은 약 117.5로 큰 방향은 유지됐지만 역사적 163.3/118.5와
  일치하지 않았다.
- **표본과 선택:** 2,490 train / 510 test, 학습 seed 1개이며 같은 test set으로
  checkpoint와 설정을 고르고 최종 수치를 보고했다. 평균·분산이나 일반화 성능을
  주장할 근거가 아니다.
- **caption 계약:** curve 평가는 이미지마다 저장된 여러 caption 중 **첫 번째 1개**만
  쓴다. “각 test caption을 모두 평가”한 것이 아니다.
- **metric 해석:** FID는 표본 수와 feature extractor에 의존한다. ignite 1000-d
  logits FID와 torchmetrics pool3 2048-d FID는 변환할 수 없다. 표본 수·seed·feature
  space를 함께 보고한다.
- **CLIP score/diversity:** 모든 학습 run이 `--use_contrastive_loss`로 같은 CLIP에
  대해 G를 직접 학습시키므로, `eval_curve.py`가 기록하는 `clip_score`/`clip_diversity`는
  학습 목적함수와 부분적으로 겹치는 보조 지표다 — conditioning이 실제로 작동한다는
  독립 증거로 읽지 않는다. 어떤 loss도 최적화하지 않는 오염되지 않은 지표는
  `experiments/prompt_sensitivity.py`의 prompt-swap pixel sensitivity다([§2.4](#24-2026-07-23-conditioning-붕괴-진단복구)).
- **방법론 위치:** 이 구현은 StackGAN++/LAFITE/AttnGAN 등에서 가져온 구성의 연구용
  조합이다. 독창성·고품질·외부 benchmark 우월성은 현재 실험으로 입증되지 않았다.

역사적 수치의 출처·날짜·last saved checkpoint 구분은
[experiments/RESULTS.md](../../experiments/RESULTS.md)를 정본으로 삼는다.

## 5. 검증 상태와 남은 제한

### 5.1 실행 검증

✅ [검증 반영 #10] (2026-07-10) 다음을 실제 실행했다.

- Python `compileall`, shell `bash -n`, 결과 JSON parse
- 3-stage G/D forward와 CLIP·VGG·DiffAugment·EMA를 포함한 GPU 1-step backward
- EMA/raw checkpoint save/load round-trip, training provenance와 확장-phase resume contract
- standalone inference, 32-sample `scripts/eval.py` smoke
- baseline epoch 20과 DiffAugment epoch 90의 독립-seed FID spot-check
- 2,490/510 split 교집합 0 확인

회귀 test suite는 CA 부호 범위, alignment input, D freeze/state 복원, checkpoint v2,
resume phase/config/provenance, option validation, 전처리 실패, evaluator seed/명시
checkpoint 누락 계약을 검사한다. 이는 full training의 품질 검증을 대신하지 않는다.

### 5.2 남은 제한

- 🟠 **장기 학습 품질 미확정:** linear CA + image-only alignment 조합의 conditioning
  붕괴는 진단·복구했고 그 과정에서 짧은 통제 실험(epoch 10/20/30, FID/CLIP diversity)을
  실제로 돌렸지만([§2.4](#24-2026-07-23-conditioning-붕괴-진단복구)), 장기 학습 품질은
  여전히 legacy 결과와 나란히 놓을 만큼 검증되지 않았다. legacy 결과가 수정 성공의
  성능 증거는 아니다.
- 🟠 **통계 검증 미실행:** 여러 training/eval seed, 분리 validation, held-out test,
  confidence interval이 없다.
- 🟠 **환경 재현성:** Python/PyTorch/TorchVision/핵심 metric과 CLIP revision은
  고정했지만 data-download package와 transitive dependency에는 호환 범위가 남는다.
  provenance는 실제 실행 버전을 기록하지만 완전한 lockfile은 아니다.
- 🟠 **다중 파일 transaction:** processed ZIP 하나는 staged atomic replace지만 raw
  `image.zip`/`text.zip`/manifest와 train/test split pickle은 여러 파일을 순차 교체하며
  cross-process lock이 없다. 중단 뒤 다음 검증은 SHA/count/stem 불일치를 탐지하지만,
  동시에 data prep을 두 개 실행하지 않는다. directory destination도 atomic 대상이 아니다.
- 🟠 **processed-data lineage:** `dataset.json.preprocess`는 seed와 transform/CLIP 설정을
  기록하지만 raw HF revision과 split-list hash를 내장하지 않는다. raw
  `download_provenance.json`과 filename pickle을 함께 보존한다.
- 🟢 **보조 손실:** uncond/contrastive/mixed/DiffAugment는 선택 설정이다. 데이터와
  batch에 따라 안정성이 달라지므로 설정·seed를 결과와 함께 기록한다.
- 🟠 **conditioning 회복 효과가 제한적:** `--kl_weight 0 --deterministic_cond` recipe로
  prompt-swap sensitivity를 살려도 응답은 한 자릿수~낮은 두 자릿수 /255 수준([§2.4](#24-2026-07-23-conditioning-붕괴-진단복구))이며,
  epoch 10~30 부근에서 정점을 찍은 뒤 모든 run이 감소해 early stopping이 필요하다.
- 🟠 **FID·데이터 확장 한계:** 위 recipe로도 FID는 350~410대에 머물고, 학습 데이터를
  3.6배(~9,000장, `ixw/celebahq-caption-10k` 표본 상한)로 늘려도 FID 품질 상한이
  오르지 않았다. conditioning 강도는 데이터양보다 batch size에 더 민감했다(batch 16이
  batch 4 대비 응답 약 2배). clip_score/clip_diversity는 `--use_contrastive_loss`가
  항상 켜져 있어 학습 목적함수 자체와 겹치므로 conditioning이 실제로 작동하는지의
  독립 증거로 쓰지 않는다 — 오염되지 않은 지표는 `prompt_sensitivity.py`다.

## 관련 문서

- [explanation/architecture.md](architecture.md) — 수정 후 구조·loss 흐름
- [how-to/train-eval-infer.md](../how-to/train-eval-infer.md) — fresh/resume/extension 실행
- [how-to/troubleshooting.md](../how-to/troubleshooting.md) — 증상별 대응
- [experiments/RESULTS.md](../../experiments/RESULTS.md) — 역사적 실험 결과 정본
