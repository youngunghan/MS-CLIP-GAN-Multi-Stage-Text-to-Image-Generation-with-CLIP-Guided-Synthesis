# How-to: 트러블슈팅

> **범위:** 설치·전처리·학습·resume·평가·추론의 증상별 원인과 조치. 구조적 한계는 [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md).
> **대상:** 실행 중 막힌 개발자.
> **상태:** 구현 반영 — 기준일 2026-07-10.

## 1. 설치·shell

**`ImportError: libGL.so.1: cannot open shared object file`**

- 원인: OpenCV가 요구하는 Linux GL library 부재.
- 조치: `sudo apt-get install -y libgl1-mesa-glx`.
- 위치: [environment.yml](../../environment.yml).

**`ModuleNotFoundError: networks` 또는 `utils`**

- 원인: repository module을 import하는 Python script를 다른 working directory나
  `PYTHONPATH` 없이 직접 실행.
- 조치: repository root에서 `PYTHONPATH=. python ...`로 실행하거나 제공 shell을
  `bash path/to/script.sh`로 호출.
- 위치: [§1 공통 실행 계약](../reference/cli.md#1-공통-실행-계약).

**HF download에서 `No module named datasets`**

- 원인: 과거 environment 또는 다른 Conda env를 사용.
- 조치: `conda env update -f environment.yml` 후 `conda activate msclipgan`.
  별도 `msclipgan-smoke` env는 현재 계약이 아니다.
- 위치: [environment.yml](../../environment.yml), [experiments/data_prep.sh](../../experiments/data_prep.sh).

**`Permission denied`로 `.sh` 실행 실패**

- 원인: checkout의 executable mode에 의존해 `./script.sh`로 실행.
- 조치: `bash train.sh`, `bash experiments/run_all.sh ...`처럼 shell을 명시.
- 위치: [§1 공통 실행 계약](../reference/cli.md#1-공통-실행-계약).

## 2. 데이터·전처리

**`--source ... not a single file`**

- 원인: `--source`에 단일 zip을 전달.
- 조치: `image.zip`과 `text.zip`을 담은 directory를 전달.
- 위치: [§1 원본 데이터 배치](prepare-dataset.md#1-원본-데이터-배치), [preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `open_dataset()`.

**`Missing option '--emb_dim'` 또는 512 범위 오류**

- 원인: preprocessing CLI가 CLIP embedding dimension을 필수로 검증.
- 조치: ViT-B/32 계약인 `--emb_dim 512`를 사용. 다른 CLIP model은 코드 변경과
  전체 재전처리가 필요.
- 위치: [preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `convert_dataset()`.

**`No samples were successfully preprocessed` / `Preprocessing was incomplete`**

- 원인: source/list 불일치, image decode 실패, caption 0개 등으로 하나 이상의 선택
  sample을 완성하지 못함. 현재 ZIP 계약은 부분 성공을 허용하지 않는다.
- 조치: 출력 직전의 sample별 오류를 확인하고 pickle filename, zip prefix,
  `celeba-caption/<id>.txt`를 대조. staged ZIP은 폐기되고 기존 destination은 보존된다.
- 위치: [§3 CLIP 피처 추출 전처리](prepare-dataset.md#3-clip-피처-추출-전처리), [preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `convert_dataset()`.

**`duplicate image stems` / `selected image stems missing`**

- 원인: 서로 다른 image 경로가 같은 `Path.stem`을 쓰거나 split pickle의 stem이 source에
  없음. text archive가 flat `celeba-caption/<stem>.txt`라 중복 stem은 모호하다.
- 조치: 오류에 나열된 충돌/누락 경로를 정리하고 split을 다시 만든 뒤 전처리한다.
- 위치: [split_dataset.py](../../preprocessing/split_dataset.py) `_unique_stems()`,
  [preprocess_dataset.py](../../preprocessing/preprocess_dataset.py) `_require_unique_stems()`.

**학습 시작 시 `dropped N image(s) without matching img/txt embeddings`**

- 원인: PNG와 image/text embedding key의 교집합만 loader가 인덱싱.
- 조치: 소수면 안전장치다. 많으면 `dataset.json`의 두 key 목록과 zip PNG를 비교해
  preprocessing 오류를 수정.
- 위치: [dataset/dataloader.py](../../dataset/dataloader.py) `MM_CelebA._load_metadata()`.

## 3. 학습

**CUDA out of memory** 🟠

- 원인: 256px G + stage D 3개 + CLIP/VGG 보조 loss의 메모리 사용.
- 조치: batch를 낮추고 1-epoch smoke run. RTX 4060 Ti 8 GB 실측 기준은 batch 4,
  약 5.8~6.2 GB다. 64→16만으로 충분하다고 가정하지 않는다.
- 위치: [§1 Fresh 학습](train-eval-infer.md#1-fresh-학습).

**`--use_contrastive_loss requires --batch_size >= 2`**

- 원인: batch 1 InfoNCE에는 negative가 없어 loss와 gradient가 0.
- 조치: batch를 2 이상으로 설정. loader는 마지막 remainder가 정확히 1일 때만
  해당 singleton batch를 drop한다.
- 위치: [options/train_options.py](../../options/train_options.py) `TrainOptions.validate()`, [dataset/dataloader.py](../../dataset/dataloader.py) `get_dataloader()`.

**loss가 NaN 또는 발산**

- 원인: data/learning rate/보조 loss 조합, 잘못된 source, 수치 불안정 등.
- 조치: 같은 seed로 BCE conditional baseline부터 확인한 뒤 uncond, contrastive,
  mixed, DiffAugment를 하나씩 추가. batch와 input range도 확인.
- 위치: [§5 손실](../explanation/architecture.md#5-손실).

## 4. Resume·checkpoint

**`--resume_checkpoint_path and --resume_epoch must be given together`**

- 원인: resume path/epoch 중 하나만 지정.
- 조치: 둘 다 주거나 둘 다 제거.
- 위치: [options/train_options.py](../../options/train_options.py) `TrainOptions.validate()`.

**scheduler phase / `--num_epochs` mismatch 오류**

- 원인: exact optimizer resume에서 저장된 phase 끝과 다른 `--num_epochs`를 지정했거나,
  checkpoint의 `schedule_config`와 scheduler state가 모순.
- 조치: exact resume은 저장된 phase 끝을 `--num_epochs`로 유지한다. `[150, 200)`
  확장 phase라면 저장된 `T_max=50`은 자동 사용되므로 `--num_epochs 200`을 준다. 기존
  phase 뒤로 기간을 더 늘리는 것이 목적이면 `--new_optim`으로 새 cosine phase를 시작한다.
- 위치: [§3 Resume](train-eval-infer.md#3-resume), [utils/utils.py](../../utils/utils.py) `load_checkpoint()`.

**`training_config` / `training_provenance` mismatch 오류**

- 원인: v2 exact resume에서 loss·batch·worker·save cadence·EMA·base LR 중 하나가
  달라졌거나, dataset/source/runtime/hardware fingerprint가 저장 당시와 다름.
- 조치: 저장된 설정과 환경을 복구한다. 의도적으로 조건을 바꾸는 실험이면
  `--new_optim`을 주고 새 optimization phase로 기록한다. dataset hash 계산은 시작 시
  실제 바이트를 읽으므로 대용량 archive에서는 잠시 기다린다.
- 위치: [§3.1 Exact optimizer resume](train-eval-infer.md#31-exact-optimizer-resume).

**EMA checkpoint resume에서 `Gen_raw.pt` 오류**

- 원인: EMA `Gen.pt`와 live generator/optimizer를 잇는 raw companion이 없거나
  metadata가 모순.
- 조치: 같은 epoch의 `Gen.pt`, `Gen_raw.pt`, 모든 D를 복구. v2 EMA primary만으로는
  `--new_optim`을 포함해 training resume을 지원하지 않는다. primary EMA weight는
  eval/infer에는 계속 사용할 수 있다.
- 위치: [utils/utils.py](../../utils/utils.py) `load_checkpoint()`.

**metadata 없는 checkpoint가 legacy mode로 로드됐다는 안내**

- 원인: 2026-07-10 이전 v1 weight에는 `model_config`가 없음.
- 조치: 평가·추론은 `relu` CA + `legacy_conditioned` alignment로 호환되는 것이
  정상이다. 새 linear/image-only 효과를 원하면 resume이 아니라 fresh training.
- 위치: [§3 Checkpoint 호환·재학습 계약](../explanation/correctness-and-fixes.md#3-checkpoint-호환재학습-계약).

**`No generator checkpoint found at ...`**

- 원인: directory 또는 epoch가 `epoch_<E>_Gen.pt`와 불일치.
- 조치: `find <CKPT_DIR> -maxdepth 1 -name 'epoch_*_Gen.pt'`로 실제 저장 epoch 확인.
- 위치: [utils/utils.py](../../utils/utils.py) `load_checkpoint()`.

## 5. 평가·추론

**curve evaluator가 `no generator checkpoints selected`로 종료**

- 원인: `auto` directory가 비었거나 명시 epoch가 없음.
- 조치: checkpoint directory를 확인. 빈 `{}`는 성공 결과가 아니며 non-zero가 정상.
- 위치: [experiments/eval_curve.py](../../experiments/eval_curve.py) `existing_checkpoints()`.

**같은 checkpoint의 새 FID가 역사적 JSON과 조금 다름**

- 원인: 커밋된 역사적 JSON은 evaluation seed가 미기록이고, 이후 `5d0de43` 버전도
  checkpoint loop 앞에서만 seed를 설정해 목록·순서에 의존. 현재는 checkpoint load
  뒤 seed를 매번 재설정.
- 조치: 새 `eval.json.provenance.json`의 seed·hash·config를 함께 비교하고 역사적 JSON을
  덮어쓰지 않음.
- 위치: [§5 개별 checkpoint 평가](run-experiments.md#5-개별-checkpoint-평가), [experiments/RESULTS.md](../../experiments/RESULTS.md).

**`metrics.csv uses an older/incompatible schema`**

- 원인: 과거 identity 없는 CSV가 `result_path`에 남아 있어 새 provenance-aware 행과
  섞으면 checkpoint를 구분할 수 없음.
- 조치: 기존 CSV를 역사 기록으로 다른 이름에 보존하거나 제거한 뒤 다시 평가한다.
  checkpoint curve 비교에는 sidecar까지 묶는 `experiments/eval_curve.py`를 권장한다.
- 위치: [utils/utils.py](../../utils/utils.py) `save_metrics_to_csv()`.

**`bash infer.sh ... "prompt"`의 prompt가 반영되지 않음**

- 원인: wrapper positional은 `<CKPT_DIR> [EPOCH]`뿐이고 prompt는 파일 안에 고정.
- 조치: `scripts/infer.py --prompt "..."`를 직접 실행.
- 위치: [§5 Prompt 추론](train-eval-infer.md#5-prompt-추론).

**추론 결과가 흐리거나 왜곡됨**

- 원인: 역사적 promoted artifact 자체의 품질 한계, out-of-domain prompt, legacy
  conditioning 동작 가능.
- 조치: checkpoint provenance/legacy mode와 prompt domain을 확인. 새 기본값의 품질은
  fresh retraining 없이 개선되지 않는다.
- 위치: [§5 Artifact 상태와 호환성](../../experiments/RESULTS.md#5-artifact-상태와-호환성).

## 관련 문서

- [reference/configuration.md](../reference/configuration.md) — option·검증 범위
- [reference/cli.md](../reference/cli.md) — 실행·실패 계약
- [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md) — 알려진 한계
