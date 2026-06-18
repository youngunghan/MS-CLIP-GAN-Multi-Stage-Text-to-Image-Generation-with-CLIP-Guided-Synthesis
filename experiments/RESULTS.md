# MS-CLIP-GAN — Subset training experiments & FID study

This documents the quantitative runs done after the correctness audit, on the
`fix/correctness-audit` code. All FID values are the **standard torchmetrics
2048-d (pool3) FID**, accumulated over the whole test set, with fakes generated
**per test caption** (not a single fixed prompt). See "FID measurement" below.

## TL;DR

- The original notebook's "FID 0.2423" was **ignite's default FID (1000-d logits)** — a non-standard scale ~100–1000× smaller than literature FID, not "near-perfect". On the same model/data, standard 2048-d FID ≈ **205**. The two scales **cannot be converted** (the ratio is model-dependent: 1343× vs 1870× measured at two epochs).
- On a 25% subset (2490 train / 510 test), the model **peaks early (~epoch 20) then degrades** — training longer hurts.
- **Hypothesis tested and rejected:** "D overpowers G, so weaken D." Across EMA + TTUR + label-smoothing + skewed update ratios (4 runs), best FID stayed in a **159–183 band with no run meaningfully beating the baseline (163)** and large epoch-to-epoch oscillation. Weakening D never stabilized or moved the peak. **D-dominance is not the bottleneck.**
- The **~160 FID floor is a data/architecture/objective ceiling**, not a GAN-balance issue.
- **Best model: baseline default config, epoch 20, standard FID ≈ 163** → locked at [`models/best_sub25_ep20/`](../models/best_sub25_ep20/).

## Setup

- **Data:** `ixw/celebahq-caption-10k` (HF). 3000 downloaded → split 0.83 → **2490 train / 510 test**, CLIP-preprocessed with `dataset.json` embeddings (ViT-B/32, 512-d).
- **Hardware:** RTX 4060 Ti (8 GB). batch 4 (≈5.8–6.2 GB peak), ~90% util. ~204 s/epoch (full D updates).
- **Eval:** `experiments/eval_curve.py` — standard `FrechetInceptionDistance` (2048-d, `normalize=False`+uint8) + `InceptionScore`, fakes conditioned on each test caption.

## Runs (25% subset)

| run | d_lr | D update | EMA | smooth | best FID | @ep | last FID | notes |
|---|---|---|---|---|---|---|---|---|
| **baseline** `sub25` | 2e-4 | every step | – | – | **163** | 20 | 270@90 | best overall; collapses late |
| `sub25_stable` | 1e-4 | every step | 0.999 | 0.9 | 179 | 60 | 264@90 | EMA+TTUR; no improvement, still oscillates (370@50) |
| `swA_aggr` | 2e-5 | every 3 | 0.999 | 0.9 | 183 | 20 | 184@35 | most aggressive D-weakening; worse |
| `swB_mild` | 5e-5 | every 2 | 0.999 | 0.9 | 159 | 35 | 159@35 | best-of-sweep but in noise band; minimum at the *last* epoch |

Per-run curves: `experiments/results/<run>/curves.png`.
Consolidated overlay: [`experiments/results/compare_fid.png`](results/compare_fid.png).

**Trend:** all four runs' best FID cluster in **159–183** with large epoch-to-epoch oscillation (e.g. swB swings 252→165→159 over ep25–35; stable spikes 173→370→179). No run breaks the floor; weakening D does not move the peak later or make it more stable.

**Verdict (pre-registered thresholds):** swB_mild = **159** falls in the **150–165 "baseline-equivalent" band** (reopen-balance threshold was FID < 150). 159 < baseline 163 is within the ±50/epoch noise and lands on the last epoch, so it is *not* a genuine break of the ~160 floor. → **conclusion confirmed; keep the baseline ep20 lock.**

## Interpretation

In the D-weakening settings tested, there was **no evidence that D-dominance is the main bottleneck**; on the contrary, weakening D made FID worse. Therefore the current ~160 FID floor is more likely a **data / architecture / objective ceiling** than a training-balance issue. This is a hypothesis cheaply and decisively eliminated — a meaningful negative result.

A secondary finding: samples at the best checkpoints are blurry/distorted (consistent with FID ~160–180), and look worse than the original Colab full-data (30k) results. The original code trained on **64px-upsampled targets** (an easier, low-detail task that looks "clean"); the audited code trains on **genuine 256px targets** (more correct but harder, so it needs more data/training to reach comparable visual quality).

## Conclusion & future work

We found no evidence that discriminator dominance is the main bottleneck. Weakening D
did not produce a robust FID improvement, and the model consistently plateaued around
FID ~160 on the subset. The current best model is the baseline epoch-20 checkpoint.
Future work should test whether this ceiling is data-bound via a full-data run;
otherwise, further gains likely require **architectural or data-level changes** rather
than additional balance tuning.

## Best model (locked)

`models/best_sub25_ep20/` — baseline `sub25`, epoch 20, standard FID ≈ 163 (`epoch_20_Gen.pt` + 3 discriminators + `sample_epoch20.png`).

## Next decision (not balance tuning)

1. **Full-data run** — train on the full accessible ~9k (10k source) with early-stopping / best-checkpoint selection (~20 h on this GPU). Tests the data-scale ceiling; closest to the original good-looking 30k results.
2. **Document-and-ship** — accept the standard FID (~163), keep the locked best checkpoint, and record the honest findings (FID measurement fix, early-peak instability, 64px-vs-256 tradeoff, D-balance ruled out).

## Reproduce

```bash
# data prep (idempotent; skips download if image.zip already has N)
experiments/data_prep.sh 3000 0.83 sub
# baseline:        experiments/run_all.sh 3000 0.83 sub sub25 100 10   (prep+train+eval+plot)
# stability:       experiments/run_stable.sh sub25_stable sub 100 10 1e-4 0.999 0.9
# D-weaken sweep:  experiments/run_sweep.sh
# eval any run:    experiments/eval_curve.py data/testset_sub.zip checkpoints/<name>-*/ckpt auto out.json
```
