# How-to: 트러블슈팅

> **범위:** 설치·학습·추론·평가에서 자주 만나는 증상과 처방. 설계 배경은 [explanation/](../explanation/).
> **대상:** 실행 중 막힌 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-16.

## 1. 설치 / 환경

**`ImportError: libGL.so.1: cannot open shared object file`**
- 원인: OpenCV가 요구하는 시스템 GL 라이브러리 부재(Linux).
- 처방: `sudo apt-get install -y libgl1-mesa-glx`.

**`conda env create` 가 느리거나 실패**
- 원인: 버전 핀 해석. `environment.yml`의 conda 절에는 pip 전용 extras를 두지 않았다(`torchmetrics[image]`는 pip 절에만).
- 처방: 그래도 막히면 [reference/configuration.md](../reference/configuration.md)의 버전 표를 참고해 수동 설치.

## 2. 데이터 / 전처리

**전처리에서 `--source ... not a single file` 에러**
- 원인: `--source`에 단일 zip 파일을 줌.
- 처방: image.zip·text.zip을 담은 **디렉터리**를 준다([§1 원본 데이터 배치](prepare-dataset.md#1-원본-데이터-배치)).

**학습 시작 시 "dropped N image(s) without matching img/txt embeddings" 경고**
- 원인: 캡션 임베딩이 없는 이미지. 데이터로더가 img·txt 임베딩을 모두 가진 샘플만 인덱싱한다([dataset/dataloader.py](../../dataset/dataloader.py) `_load_metadata()`).
- 처방: 정상 동작(드롭은 안전장치). 드롭이 과도하면 전처리에서 캡션이 비었는지 확인.

## 3. 학습

**CUDA out of memory** 🟠
- 처방: `--batch_size`를 낮춘다(예: 64→16). 3단계 256 해상도 + 단계별 D 3개라 메모리 요구가 크다.
- 다중 GPU면 `train.sh`의 `GPUS="0,1"`로 분산([§2 학습 (다중 GPU)](train-eval-infer.md#2-학습-다중-gpu)).

**손실이 NaN/발산**
- 점검: `--use_contrastive_loss` 사용 시 CLIP 대조 손실은 stage 2(≥256)에서만 동작하며 float32로 계산한다([criteria/loss.py](../../criteria/loss.py) `contrastive_loss_G()`). 그래도 불안정하면 보조 손실을 끄고 BCE만으로 베이스라인을 확인.
- 시드: `--seed`로 재현 후 비교.

**`./train.sh` 가 resume를 시도하며 실패**
- 원인: resume 인자를 켰는데 경로가 없음.
- 처방: 처음 학습이면 [train.sh](../../train.sh) 하단 resume 주석을 비활성 상태로 둔다(기본값).

## 4. 체크포인트 / 추론 / 평가

**추론에서 `No generator checkpoint found at ...`**
- 원인: `CKPT_DIR`에 `epoch_<E>_Gen.pt`가 없음.
- 처방: 경로/에폭을 확인. 추론은 G 체크포인트만 필요(D 불필요, [utils/utils.py](../../utils/utils.py) `load_checkpoint()`).

**다중 GPU로 학습한 체크포인트가 추론에서 로드 실패(`module.` 키)**
- 처방: 현재 코드는 저장 시 DataParallel을 벗기고 로드 시 `module.` 접두사를 자동 제거하므로 정상 동작해야 한다. 외부에서 받은 구버전 체크포인트라면 키 접두사를 확인.

**추론 결과가 이상/저품질**
- 점검: `infer.py`는 `G.eval()`을 적용한다(BatchNorm 러닝 통계 사용). 학습이 충분히 진행됐는지, 프롬프트가 학습 도메인(얼굴 속성 캡션)과 맞는지 확인.

**FID/IS 가 비현실적**
- ⭐ [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md)의 평가 타당성 절 필독(real 이미지 해상도 출처·표본 수가 수치 해석을 좌우).

## 관련 문서

- [explanation/correctness-and-fixes.md](../explanation/correctness-and-fixes.md) — 알려진 한계
- [reference/configuration.md](../reference/configuration.md) — 옵션·버전
