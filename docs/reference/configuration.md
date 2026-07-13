# Configuration 레퍼런스

> **범위:** CLI 옵션(base/train/test), 의존성(environment.yml 기준), 기본 하이퍼파라미터. 옵션 정의는 [options/](../../options/).
> **대상:** 개발자.
> **상태:** 구현 반영 — 기준일 2026-07-10.

## 1. 공통 옵션 (BaseOptions)

[options/base_options.py](../../options/base_options.py) `BaseOptions.initialize()`.

| 옵션 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `--seed` | int | 42 | 재현성 시드(파싱 시 `seed_fix` 적용) |
| `--name` | str | `experiment_name` | 실험명(타임스탬프 자동 부착) |
| `--gpu_ids` | str | `0` | CUDA 디바이스 인덱스(콤마, `-1`=CPU) |
| `--num_workers` | int | 4 | DataLoader 워커 수 |
| `--data_path` | Path | `./data/sample_train.zip` | 학습 데이터 zip |
| `--checkpoint_path` | Path | `./checkpoints` | train에서는 run namespace를 만들 root; eval/infer에서는 읽을 `ckpt` directory 그대로(쓰기 없음) |
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
| `--conditioning_activation` | choice | `linear` | CA projection: 새 학습은 `linear`, legacy 호환은 `relu`. checkpoint metadata가 load 시 의미를 결정 |
| `--alignment_mode` | choice | `image_only` | 정렬 head: 새 학습은 `image_only`, legacy 호환은 `legacy_conditioned`. checkpoint metadata가 load 시 의미를 결정 |
| `--clip_model` | str | `ViT-B/32` | CLIP 모델. **`ViT-B/32` 고정**(choices 제한). 전처리([preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `convert_dataset`)가 ViT-B/32를 **하드코딩**하고 `clip_embedding_dim=512`라 다른 모델은 차원 불일치. 변경은 CLI 옵션만으로 불가 — 전처리 코드 수정 + 재전처리 + `clip_embedding_dim` 동기화(=코드 변경)가 필요 |

> 🟢 `preprocess_dataset.py`의 `--max-failure-frac`(기본 0.0, 즉 sample 실패 0건까지만 허용)은
> 학습 옵션이 아니라 전처리 전용 CLI 옵션이다. 상세는 [reference/cli.md](cli.md)와
> [how-to/prepare-dataset.md](../how-to/prepare-dataset.md) 참고.

> 🟢 `--gpu_ids`는 실제 디바이스 인덱스로 쓰인다(과거의 죽은 재매핑 분기 제거). 디바이스 선택은 스크립트가 담당하고 `BaseOptions`는 `set_device`를 호출하지 않는다.
> 🟢 `--noise_dim` 변경 시에도 SSA 블록 차원이 어긋나지 않는다(생성기가 `cond_dim`을 명시적으로 전달, [networks/generator.py](../../networks/generator.py) `Generator_type_1`).
>
> 🟠 metadata 없는 기존 checkpoint는 CLI 기본값을 그대로 쓰지 않고 `conditioning_activation=relu`, `alignment_mode=legacy_conditioned`로 자동 전환한다. 새 기본값의 효과는 fresh training이 필요하다([§3 Checkpoint 호환·재학습 계약](../explanation/correctness-and-fixes.md#3-checkpoint-호환재학습-계약)).

## 2. 학습 옵션 (TrainOptions)

[options/train_options.py](../../options/train_options.py).

| 옵션 | 타입 | 기본값 | 비고 |
|---|---|---|---|
| `--batch_size` | int | 1 | `train.sh`는 4. contrastive 사용 시 2 이상 필수 |
| `--num_epochs` | int | 50 | `train.sh`는 150 |
| `--learning_rate` | float | 2e-4 | `train.sh`는 1e-4 (Adam β=0.5,0.999) |
| `--save_freq` | int | 1 | 저장 주기(에폭) |
| `--use_uncond_loss` | flag | argparse 기본 off — **`train.sh`는 on** | 무조건 판별 손실 |
| `--use_contrastive_loss` | flag | argparse 기본 off — **`train.sh`는 on** | 정렬/CLIP 대조 손실. batch 1은 parse error; 마지막 singleton remainder만 drop |
| `--use_mixed_loss` | flag | argparse 기본 off — **`train.sh`는 on** | L1+VGG perceptual 혼합 손실 |
| `--new_optim` | flag | off | resume 시 weight만 로드하고 남은 epoch 구간에 새 optimizer/cosine schedule 시작 |
| `--no_mismatched_condition` | flag | off | 기본으로 켜지는 real image + 틀린 text conditional BCE negative를 비활성 |
| `--d_lr` | float | -1.0 | 판별기 LR(TTUR). `-1`이면 `--learning_rate`를 D에도 그대로 사용하고, 그 밖에는 양수만 허용 |
| `--use_ema` | flag | off | 생성기 가중치의 EMA를 추적하고, 샘플링/체크포인트에 EMA 모델을 사용 |
| `--ema_decay` | float | 0.999 | 생성기 EMA decay |
| `--real_label_smooth` | float | 1.0 | 판별기의 real 라벨 타깃(예: 0.9 = 단측 라벨 스무딩) |
| `--d_update_every` | int | 1 | G 스텝 N번마다 D를 1번 업데이트(N>1이면 D를 약화, 즉 n_critic<1) |
| `--use_diffaugment` | flag | off | 판별기 입력의 real/fake 양쪽에 DiffAugment(미분 가능 증강) 적용 |
| `--diffaugment_policy` | str | `color,translation,cutout` | DiffAugment 정책(콤마 구분, `color`/`translation`/`cutout`의 부분집합) |
| `--is_train` | bool | `true` | 내부 mode 표식. `TrainOptions`에서는 `true`만 허용 |

> `--new_optim`이 없으면 v2 exact resume 계약이다. `--num_epochs`는 저장된 scheduler
> phase의 끝과 같아야 하며 저장된 phase `T_max`를 자동 재구성한다. 예를 들어
> `[150, 200)` 확장 phase의 `T_max`는 50이지만 resume 명령은 `--num_epochs 200`이다.
> training config와 데이터·소스·runtime·hardware provenance도 같아야 한다. 기간 연장이나
> optimizer 설정 변경이 의도라면 `--new_optim`으로 새 phase를 명시한다. metadata 없는
> legacy checkpoint는 이 검증 자료가 없어 best-effort resume만 가능하다.

## 3. 테스트/추론 옵션 (TestOptions)

[options/test_options.py](../../options/test_options.py).

| 옵션 | 타입 | 기본값 | 비고 |
|---|---|---|---|
| `--prompt` | str | `"a photo of a person"` | infer에서 사용(eval은 무시, optional) |
| `--load_epoch` | int | (필수) | 로드할 에폭 |
| `--eval_data_path` | str | (필수) | 평가 zip(infer는 더미 `None` 전달) |
| `--batch_size` | int | 16 | 평가 배치 |
| `--print_freq` | int | 10 | 평가 로그 주기 |
| `--max_batches` | int | -1 | 평가 배치 수 상한. `-1` = 전체 평가셋 사용(FID는 표본이 많이 필요) |
| `--is_train` | bool | `false` | 내부 mode 표식. `TestOptions`에서는 `false`만 허용 |

`TestOptions` parsing은 입력 checkpoint directory 아래에 `opt.txt`나 timestamp 폴더를
만들지 않는다. 학습만 `<checkpoint_root>/<name>-<timestamp_ns>-p<PID>/` namespace를
원자적으로 만들며, 이미 존재하는 namespace에 이어 쓰지 않고 실패한다.

## 4. 입력 검증 계약

| 검증 | 허용 계약 |
|---|---|
| 출력 채널 | `g_out_chans=3`, `d_out_chans=1` 고정 |
| 채널/단계 | `num_stage>0`, `g_in_chans`는 16으로 나누어지며 마지막 refinement 입력 채널이 4 이상 |
| 일반 양수 범위 | `noise_dim`, `condition_dim`, `g_in_chans`, `d_in_chans`, `report_interval` > 0; `num_workers>=0` |
| 학습 범위 | `batch_size`, `num_epochs`, `learning_rate`, `save_freq`, `d_update_every` > 0 |
| contrastive | `batch_size>=2` 및 dataset sample>=2 |
| EMA/label | `0<=ema_decay<1`, `0<real_label_smooth<=1` |
| resume | path/epoch를 함께 지정, `0<=resume_epoch<num_epochs-1`; `--new_optim`은 resume과 함께만 사용 |
| DiffAugment | 사용 시 policy는 `color`, `translation`, `cutout`의 비어 있지 않은 콤마 목록 |
| 평가 | `load_epoch>=0`, `batch_size>0`, `print_freq>0`, `max_batches=-1` 또는 양수 |

검증 실패는 argparse usage와 non-zero exit로 끝난다. 실제 parser가 정본이므로
`PYTHONPATH=. python scripts/train.py --help` 또는
`PYTHONPATH=. python scripts/eval.py --help`로 설치된 코드의 값을 확인한다.

## 5. 의존성 (`environment.yml`, 2026-07-10 기준)

[environment.yml](../../environment.yml)에 선언된 의존성이다. 핵심 학습·평가 조합은
이 작업공간에서 GPU train step, legacy infer, FID smoke로 실행 확인했다.

| 패키지 | 버전 | 비고 |
|---|---|---|
| python | 3.8.20 | conda |
| pytorch / torchvision | 2.4.0 / 0.19.0 | `pytorch-cuda=12.4`; 코드가 요구하는 `meshgrid(indexing=...)`·`InterpolationMode`와 호환 |
| torchmetrics | 1.5.2 (+`[image]` pip) | FID/IS |
| numpy / pillow / opencv | 1.24.4 / 10.4.0 / 4.10.0 | 검증 환경 고정값 |
| tensorboard | 2.11.0 | |
| CLIP | Git commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | pip 절에 immutable revision pin |
| ftfy · regex | pip | CLIP 토크나이저 의존 |
| datasets | ≥2.14, <3 | Python 3.8 호환 line; Hugging Face streaming download |
| huggingface_hub | ≥0.17, <1 | dataset revision을 immutable commit SHA로 resolve |
| torch-fidelity · pytorch-ignite | pip | (현재 메트릭은 torchmetrics 사용) |

> ⚠️ 학습 핵심 stack과 CLIP revision은 고정했지만 data-download용 두 package와 일부 pip
> 도구는 호환 범위다. `environment.yml`은 완전한 transitive lockfile이 아니며 curve
> evaluator의 `*.provenance.json`은 실제 runtime을 기록할 뿐 환경 복원을 대신하지 않는다.

## 관련 문서

- [reference/cli.md](cli.md) — 스크립트 진입점
- [explanation/architecture.md](../explanation/architecture.md) — 차원이 구조에서 어떻게 쓰이는지
