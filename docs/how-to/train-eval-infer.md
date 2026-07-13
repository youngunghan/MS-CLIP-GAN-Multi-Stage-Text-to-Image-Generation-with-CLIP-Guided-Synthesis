# How-to: 학습 · 평가 · 추론

> **범위:** fresh 학습, exact resume와 기간 연장, FID/IS/CLIP 평가, prompt 추론 절차. 옵션 전체는 [reference/configuration.md](../reference/configuration.md), 진입점 계약은 [reference/cli.md](../reference/cli.md).
> **대상:** 학습·평가·추론을 실행하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-07-10.

모든 명령은 repository root에서 실행한다. shell 파일은 실행 bit에 의존하지 않도록
`bash`로 부르고, Python을 직접 부를 때는 `PYTHONPATH=.`를 붙인다.

## 1. Fresh 학습

```bash
bash train.sh
```

[train.sh](../../train.sh)는 batch 64(기본), 3단계 64/128/256, lr 1e-4,
150 epoch, GPU 0으로 실행한다. 핵심 기본 동작은 다음과 같다.

| 항목 | `train.sh` 계약 |
|---|---|
| 데이터 | `./data/trainset.zip`; 256px 보존 후 단계별 area downsample |
| generator/discriminator | G 1개 + stage별 D 3개 |
| 새 architecture semantics | `conditioning_activation=linear`, `alignment_mode=image_only` |
| 조건 negative | real image + batch에서 한 칸 이동한 틀린 text BCE를 기본 사용 |
| 보조 loss | unconditional + contrastive + mixed를 wrapper가 켬 |
| checkpoint | `checkpoints/<name-timestamp>/ckpt/`; v2 model/training/schedule config·학습 provenance·RNG·optimizer 포함 |
| sample/log | `res/`, `runs/<name-timestamp>/`, `opt.txt` |
| 저장 시점 | `save_freq` 주기 + 마지막 epoch 강제 저장 |

기본 batch 64는 8 GB급 GPU에는 맞지 않는다. RTX 4060 Ti 8 GB에서는 `train.sh`의
`BS`를 4로 낮추면 약 5.8~6.2 GB를 사용했다(다른 모델·driver·loss 조합에서는 달라질
수 있다). 먼저 epoch 수를 줄인 smoke run으로 확인한다. `batch_size=1`은 argparse
기본일 뿐, contrastive를 켠 학습에는 허용되지 않는다.

직접 옵션을 조합할 때의 최소 예시는 다음과 같다.

```bash
PYTHONPATH=. python scripts/train.py \
  --name fresh \
  --data_path ./data/trainset.zip \
  --batch_size 64 \
  --num_epochs 150 \
  --learning_rate 1e-4 \
  --save_freq 5 \
  --use_uncond_loss \
  --use_contrastive_loss \
  --use_mixed_loss \
  --gpu_ids 0
```

새 기본 mismatched-condition 항만 끄려는 경우에만 `--no_mismatched_condition`을
추가한다. 보조 loss와 정확한 가중치는
[§5 손실](../explanation/architecture.md#5-손실)을 본다.

## 2. 다중 GPU

`train.sh`의 `GPUS`를 `GPUS="0,1"`처럼 바꾸거나 Python에
`--gpu_ids 0,1`을 준다. 값은 실제 CUDA device index이며 2개 이상이면
`nn.DataParallel(device_ids=...)`을 쓴다.

checkpoint는 DataParallel wrapper를 벗겨 저장하고 load 시 legacy `module.` prefix도
제거한다([utils/utils.py](../../utils/utils.py) `save_checkpoint()`·`load_checkpoint()`).
따라서 다중 GPU weight를 단일 GPU eval/infer에서 읽을 수 있다. CPU는
`--gpu_ids -1`이며 CUDA id와 혼합할 수 없다.

## 3. Resume

`train.sh`의 주석 두 줄을 shell 밖에서 활성화하는 방식은 유효한 명령이 아니다.
resume에는 원래 run의 `opt.txt`를 참고해 **전체 학습 명령**을 다시 쓰고 path/epoch를
함께 추가한다.

### 3.1 Exact optimizer resume

```bash
PYTHONPATH=. python scripts/train.py \
  --name resumed-exact \
  --data_path ./data/trainset.zip \
  --batch_size 64 \
  --num_epochs 150 \
  --learning_rate 1e-4 \
  --save_freq 5 \
  --use_uncond_loss --use_contrastive_loss --use_mixed_loss \
  --gpu_ids 0 \
  --resume_checkpoint_path ./checkpoints/<old-run>/ckpt \
  --resume_epoch 49
```

- checkpoint v2는 Python/NumPy/Torch/CUDA RNG, model, optimizer, scheduler를 복구한다.
  loss·batch·worker·save cadence·EMA와 base LR를 담은 `training_config`, cosine 구간을
  담은 `schedule_config`, 학습 데이터/소스/runtime/hardware fingerprint도 비교한다.
- `--num_epochs`는 저장된 **scheduler phase의 끝(exclusive)** 과 같아야 한다. fresh
  0~149 구간은 end 150·`T_max=150`이고, `--new_optim`으로 만든 150~199 확장 구간은
  end 200·`T_max=50`이다. 후자를 epoch 160에서 exact resume할 때는
  `--num_epochs 200`을 주면 저장된 50-epoch scheduler를 자동 재구성한다.
- 실행 시작 때 실제 dataset 바이트와 학습 관련 Python 소스를 SHA-256으로 한 번 읽으므로
  큰 dataset에서는 학습 시작 전에 시간이 걸릴 수 있다.
- recorded contract 일치는 강하게 검사하지만, bitwise 연속성에는 기록 밖 외부 입력이
  없어야 하고 사용 kernel 자체도 deterministic해야 한다.
- EMA run은 `Gen.pt`의 EMA와 `Gen_raw.pt`의 live weight가 metadata와 함께 모두
  있어야 한다. raw 파일이 없는데 EMA optimizer와 섞어 이어가지 않는다.
- metadata 없는 legacy checkpoint는 `relu` + `legacy_conditioned`로 로드한다. 저장된
  optimizer/scheduler가 있으면 best-effort로 복구하지만 training config·provenance·RNG가
  없어 **검증된 exact resume은 아니다**. scheduler가 없으면 `--new_optim`이 필요하다.

### 3.2 기간 연장 또는 optimizer 재시작

```bash
PYTHONPATH=. python scripts/train.py \
  --name extended \
  --data_path ./data/trainset.zip \
  --batch_size 64 \
  --num_epochs 200 \
  --learning_rate 1e-4 \
  --save_freq 5 \
  --use_uncond_loss --use_contrastive_loss --use_mixed_loss \
  --gpu_ids 0 \
  --resume_checkpoint_path ./checkpoints/<old-run>/ckpt \
  --resume_epoch 149 \
  --new_optim
```

`--new_optim`은 weight만 가져오고 optimizer/scheduler를 버리며, `resume_epoch+1`부터
`num_epochs-1`까지의 **남은 epoch 수**로 새 cosine phase를 만든다. 그 phase의 시작·끝과
`T_max`는 이후 v2 checkpoint에 저장되므로 중간에 다시 exact resume할 수 있다.
`--new_optim` 자체는 이전 trajectory의 exact continuation이 아니며 v2의 RNG만 복구한다.
legacy checkpoint를 이 방식으로 불러도 architecture semantics는 legacy로 유지되므로 새
linear/image-only 수정 효과를 얻지 못한다.

## 4. 단일 checkpoint 평가

```bash
# bash eval.sh <CKPT_DIR> [EPOCH]
bash eval.sh ./checkpoints/<run-name>/ckpt 149
```

[scripts/eval.py](../../scripts/eval.py) `evaluate()`는 test loader의 text embedding으로
fake를 만들고 전체 표본을 단일 FID/IS metric에 누적한다. CLIP score는 표본 가중
평균이며 checkpoint load 뒤 `--seed`를 다시 적용한다. 결과는 console과
`./output/metrics.csv`에 append하며 각 행에 checkpoint 절대경로·SHA-256·format/legacy·
conditioning/alignment·epoch·seed를 함께 기록한다. sample filename에도 epoch와 hash
prefix가 붙는다. metric이 non-finite면 CSV에 쓰지 않는다. `--max_batches -1`이 전체
dataset 계약이다.

`eval.sh`의 `--prompt` 값은 shared `TestOptions` 호환 인자일 뿐 **평가에는 사용되지
않는다**. 평가 caption은 dataset loader가 저장된 caption 중 선택한다. checkpoint curve,
첫-caption 고정, SHA-256 provenance가 필요하면
[§5 개별 checkpoint 평가](run-experiments.md#5-개별-checkpoint-평가)의
`experiments/eval_curve.py`를 사용한다.

510장 역사적 subset FID는 탐색 값이다. 표본 수·training/eval seed·checkpoint 선택
protocol 없이 수치만 인용하지 않는다([§4 결과 인용 한계](../explanation/correctness-and-fixes.md#4-결과-인용-한계)).

## 5. Prompt 추론

`infer.sh`의 positional 계약은 **checkpoint directory와 epoch 두 개뿐**이다.
prompt는 shell 파일 안에 고정돼 있으므로 다음 명령의 세 번째 인자로 넣어도 전달되지
않는다.

```bash
# bash infer.sh <CKPT_DIR> [EPOCH]
bash infer.sh ./checkpoints/<run-name>/ckpt 149
```

prompt를 지정하려면 Python 진입점을 직접 호출한다. `--eval_data_path None`은 shared
`TestOptions`의 필수 placeholder이며 infer가 dataset으로 열지는 않는다.

```bash
PYTHONPATH=. python scripts/infer.py \
  --checkpoint_path ./checkpoints/<run-name>/ckpt \
  --load_epoch 149 \
  --eval_data_path None \
  --prompt "The woman is young and has blond hair."
```

prompt를 CLIP `encode_text`로 encode·정규화하고 seed 42의 z와 함께 G에 넣는다.
generator checkpoint만 필요하며 결과는 기본 `./output/result_{64,128,256}.png`다.
checkpoint v2 metadata는 model 생성 전에 저장된 architecture와 conditioning/alignment
의미를 복구한다. metadata 없는 weight는 parameter shape에서 기본 차원을 추론하고
legacy CA 의미를 사용한다.

## 관련 문서

- [reference/configuration.md](../reference/configuration.md) — 옵션·검증 계약
- [reference/cli.md](../reference/cli.md) — 진입점·입출력
- [how-to/troubleshooting.md](troubleshooting.md) — 증상별 대응
- [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md) — legacy·평가 한계
