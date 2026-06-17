# Configuration 레퍼런스

> **범위:** CLI 옵션(base/train/test), 의존성(environment.yml 기준), 기본 하이퍼파라미터. 옵션 정의는 [options/](../../options/).
> **대상:** 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. 공통 옵션 (BaseOptions)

[options/base_options.py](../../options/base_options.py) `BaseOptions.initialize()`.

| 옵션 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `--seed` | int | 42 | 재현성 시드(파싱 시 `seed_fix` 적용) |
| `--name` | str | `experiment_name` | 실험명(타임스탬프 자동 부착) |
| `--gpu_ids` | str | `0` | CUDA 디바이스 인덱스(콤마, `-1`=CPU) |
| `--num_workers` | int | 4 | DataLoader 워커 수 |
| `--data_path` | Path | `./data/sample_train.zip` | 학습 데이터 zip |
| `--checkpoint_path` | Path | `./checkpoints` | 체크포인트 루트 |
| `--result_path` | Path | `./output` | 샘플/결과 경로 |
| `--resume_checkpoint_path` | str | None | resume용 ckpt 디렉터리 |
| `--resume_epoch` | int | -1 | resume 에폭(-1=비활성) |
| `--report_interval` | int | 100 | 로그 주기(iter) |
| `--noise_dim` | int | 100 | 생성기 입력 노이즈 z 차원 |
| `--condition_dim` | int | 128 | 조건 증강 후 차원(c_hat) |
| `--clip_embedding_dim` | int | 512 | CLIP 텍스트 임베딩 차원. **512 고정**(ViT-B/32 전용 — 512가 아니면 `parse()`에서 에러) |
| `--g_in_chans` | int | 1024 | 생성기 base 채널(Ng) |
| `--g_out_chans` | int | 3 | 출력 채널(RGB) |
| `--d_in_chans` | int | 64 | 판별기 base 채널(Nd) |
| `--d_out_chans` | int | 1 | 판별기 출력 채널 |
| `--num_stage` | int | 3 | 단계 수(해상도 64·128·256) |
| `--clip_model` | str | `ViT-B/32` | CLIP 모델. **`ViT-B/32` 고정**(choices 제한). 전처리([preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `convert_dataset`)가 ViT-B/32를 **하드코딩**하고 `clip_embedding_dim=512`라 다른 모델은 차원 불일치. 변경은 CLI 옵션만으로 불가 — 전처리 코드 수정 + 재전처리 + `clip_embedding_dim` 동기화(=코드 변경)가 필요 |

> 🟢 `--gpu_ids`는 실제 디바이스 인덱스로 쓰인다(과거의 죽은 재매핑 분기 제거). 디바이스 선택은 스크립트가 담당하고 `BaseOptions`는 `set_device`를 호출하지 않는다.
> 🟢 `--seed` 변경 시에도 SSA 블록 차원이 어긋나지 않는다(생성기가 `cond_dim`을 명시적으로 전달, [networks/generator.py](../../networks/generator.py) `Generator_type_1`).

## 2. 학습 옵션 (TrainOptions)

[options/train_options.py](../../options/train_options.py).

| 옵션 | 타입 | 기본값 | 비고 |
|---|---|---|---|
| `--batch_size` | int | 1 | `train.sh`는 64 |
| `--num_epochs` | int | 50 | `train.sh`는 150 |
| `--learning_rate` | float | 2e-4 | `train.sh`는 1e-4 (Adam β=0.5,0.999) |
| `--save_freq` | int | 1 | 저장 주기(에폭) |
| `--use_uncond_loss` | flag | off | 무조건 판별 손실 |
| `--use_contrastive_loss` | flag | off | 정렬/CLIP 대조 손실 |
| `--use_mixed_loss` | flag | off | L1+VGG perceptual 혼합 손실 |
| `--new_optim` | flag | off | resume 시 optimizer/scheduler 새로 시작 |

## 3. 테스트/추론 옵션 (TestOptions)

[options/test_options.py](../../options/test_options.py).

| 옵션 | 타입 | 기본값 | 비고 |
|---|---|---|---|
| `--prompt` | str | `"a photo of a person"` | infer에서 사용(eval은 무시, optional) |
| `--load_epoch` | int | (필수) | 로드할 에폭 |
| `--eval_data_path` | str | (필수) | 평가 zip(infer는 더미 `None` 전달) |
| `--batch_size` | int | 16 | 평가 배치 |
| `--print_freq` | int | 10 | 평가 로그 주기 |

## 4. 의존성 (environment.yml 기준)

[environment.yml](../../environment.yml)에 선언된 의존성이다(별도 런타임 스모크 테스트로 검증한 조합은 아님).

| 패키지 | 버전 | 비고 |
|---|---|---|
| python | 3.8 | conda |
| pytorch | ≥1.7.1 | cudatoolkit 11.3 |
| torchvision | ≥0.8.2 | |
| torchmetrics | (+`[image]` pip) | FID/IS |
| numpy / pillow / opencv | ≥1.19.2 / ≥8.0.0 / ≥4.5.0 | |
| tensorboard | 2.11.0 | |
| CLIP | git(openai/CLIP) | pip 절 |
| ftfy · regex | pip | CLIP 토크나이저 의존 |
| torch-fidelity · pytorch-ignite | pip | (현재 메트릭은 torchmetrics 사용) |

> ⚠️ 버전 하한만 지정된 항목이 많아 완전 재현성은 약하다. 정확한 재현이 필요하면 실제 설치 버전을 별도로 고정한다.

## 관련 문서

- [reference/cli.md](cli.md) — 스크립트 진입점
- [explanation/architecture.md](../explanation/architecture.md) — 차원이 구조에서 어떻게 쓰이는지
