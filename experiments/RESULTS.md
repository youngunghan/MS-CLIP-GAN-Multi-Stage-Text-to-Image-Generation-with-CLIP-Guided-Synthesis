# MS-CLIP-GAN — Subset training experiments & FID study

This documents the quantitative runs done after the correctness audit, on the
`fix/correctness-audit` code. All FID values are the **standard torchmetrics
2048-d (pool3) FID**, accumulated over the whole test set, with fakes generated
**per test caption** (not a single fixed prompt). See "FID measurement" below.

## TL;DR

- The original notebook's "FID 0.2423" was **ignite's default FID (1000-d logits)** — a non-standard scale ~100–1000× smaller than literature FID, not "near-perfect". On the same model/data, standard 2048-d FID ≈ **205**. The two scales **cannot be converted** (the ratio is model-dependent: 1343× vs 1870× measured at two epochs).
- On a 25% subset (2490 train / 510 test), the model **peaks early (~epoch 20) then degrades** — training longer hurts.
- **Hypothesis tested and rejected:** "D overpowers G, so weaken D." Across EMA + TTUR + label-smoothing + skewed update ratios (4 runs), best FID stayed in a **159–183 band with no run meaningfully beating the baseline (163)** and large epoch-to-epoch oscillation. Weakening D never stabilized or moved the peak. **D-dominance is not the bottleneck.**
- The **~160 FID floor held across the whole D-balance sweep** (EMA/TTUR/skewed update ratios) — not a GAN-balance issue. It is **not an absolute ceiling** for the project, though: a later **DiffAugment** run (`diffaug100`) broke through it, reaching **FID ≈ 118.5** at epoch 90 (see "DiffAugment run" below).
- **Best locked model: baseline default config, epoch 20, standard FID ≈ 163** → locked at [`models/best_sub25_ep20/`](../models/best_sub25_ep20/) — this is the best **promoted** artifact from the pre-DiffAugment sweep. A later, not-yet-promoted `diffaug100` checkpoint (epoch 90) scores lower (FID ≈ 118.5); see "DiffAugment run" below.

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

**Trend:** all four runs' best FID cluster in **159–183** with large epoch-to-epoch oscillation (e.g. swB swings 252→165→159 over ep25–35; stable spikes 194→370→179). No run breaks the floor; weakening D does not move the peak later or make it more stable.

**Verdict (pre-registered thresholds):** swB_mild = **159** falls in the **150–165 "baseline-equivalent" band** (reopen-balance threshold was FID < 150). 159 < baseline 163 is within the ±50/epoch noise and lands on the last epoch, so it is *not* a genuine break of the ~160 floor. → **conclusion confirmed; keep the baseline ep20 lock.**

## DiffAugment run (breaks the pre-DiffAugment floor)

A follow-up run added **DiffAugment** (`color,translation,cutout`) on top of the plain
`sub25` baseline (same subset, same config, no EMA/TTUR/label-smoothing), isolating
augmentation as the only variable:

| run | epochs | best FID | @ep | notes |
|---|---|---|---|---|
| `diffaug` (probe) | 45 | 173 | 40 | short sanity-check run |
| `diffaug100` | 100 | **118.5** | 90 | full run; breaks the ~160 floor |

`experiments/results/diffaug100/eval.json`: FID drops sharply after ep50 (163.0) →
ep60 (**119.6**) → ep70 (141.1, a late-training bounce back up) → ep80 (121.9) → ep90
(**118.5**, the run's best). The improvement isn't perfectly monotonic late in
training (the ep70 bounce), but ep80/ep90 confirm it settles below the
pre-DiffAugment floor rather than a one-epoch fluke.

Consolidated overlay: [`experiments/results/compare_diffaug.png`](results/compare_diffaug.png).

**Implication:** the ~160 FID floor found in the D-balance sweep above is **not an
absolute architecture/objective ceiling** — it was a floor specific to runs *without*
strong augmentation. This doesn't overturn the D-balance conclusion (D-dominance still
isn't the bottleneck); it identifies augmentation, not balance tuning, as the axis that
actually moved FID.

**Checkpoint status:** unlike `sub25` ep20, the `diffaug100` epoch-90 weights have
**not** been promoted/locked into `models/`. They exist only as a raw, git-ignored
training checkpoint at
`checkpoints/diffaug100-2026_06_19_08_51_13/ckpt/epoch_90_{Gen,Dis_0,Dis_1,Dis_2}.pt`.
`models/best_sub25_ep20/` remains the only currently-locked artifact; `diffaug100` is
the better-scoring but not-yet-promoted result.

## Interpretation

In the D-weakening settings tested, there was **no evidence that D-dominance is the main bottleneck**; on the contrary, weakening D made FID worse. Therefore, within the D-balance sweep, the ~160 FID floor is more likely a **data / architecture / objective ceiling for that specific set of runs** than a training-balance issue — though it is not an absolute ceiling for the project overall; see "DiffAugment run" above, where augmentation alone broke past it. This is a hypothesis cheaply and decisively eliminated — a meaningful negative result.

A secondary finding: samples at the best checkpoints are blurry/distorted (consistent with FID ~160–180), and look worse than the original Colab full-data (30k) results. The original code trained on **64px-upsampled targets** (an easier, low-detail task that looks "clean"); the audited code trains on **genuine 256px targets** (more correct but harder, so it needs more data/training to reach comparable visual quality).

## Conclusion & future work

We found no evidence that discriminator dominance is the main bottleneck. Weakening D
did not produce a robust FID improvement, and within that D-balance sweep the model
consistently plateaued around FID ~160 on the subset. A follow-up **DiffAugment** run
(`diffaug100`, see above) shows this floor is not architecture-bound: it reached FID ≈
118.5 at epoch 90 using augmentation alone, with no balance changes. The current
**locked** best model is still the baseline epoch-20 checkpoint (the best *promoted*
artifact); `diffaug100` scores better but its checkpoint has not been promoted (see
"DiffAugment run" above). Future work should test whether the *remaining* gap (118.5 →
literature-competitive FID) is data-bound via a full-data run; further gains likely
require a combination of **more data and continued augmentation** rather than
additional D-balance tuning.

## Best model (locked)

`models/best_sub25_ep20/` — baseline `sub25`, epoch 20, standard FID ≈ 163 (`epoch_20_Gen.pt` + 3 discriminators + `sample_epoch20.png`). This is the best **promoted/locked** artifact from the pre-DiffAugment sweep (baseline vs. stability vs. D-weakening variants). It is **not** the best FID measured in this repo overall: `diffaug100` later reached FID ≈ 118.5 at epoch 90 (see "DiffAugment run" above), but that checkpoint exists only as a raw, git-ignored training checkpoint (`checkpoints/diffaug100-2026_06_19_08_51_13/ckpt/epoch_90_*.pt`) and has not been promoted to `models/`.

## Next decision (not balance tuning)

1. **DiffAugment (done)** — see "DiffAugment run" above: `diffaug100` reached FID ≈ 118.5 at epoch 90, well past the ~163 pre-DiffAugment floor. Promoting that checkpoint into `models/` (mirroring `best_sub25_ep20/`) is still open.
2. **Full-data run** — train on the full accessible ~9k (10k source) with early-stopping / best-checkpoint selection (~20 h on this GPU). Tests the data-scale ceiling; closest to the original good-looking 30k results.
3. **Document-and-ship** — accept the current locked FID (~163) or promote `diffaug100` (~118.5) as the new locked checkpoint, and record the honest findings (FID measurement fix, early-peak instability, 64px-vs-256 tradeoff, D-balance ruled out, DiffAugment breakthrough).

## Reproduce

```bash
# data prep (idempotent; skips download if image.zip already has N)
experiments/data_prep.sh 3000 0.83 sub
# baseline:        experiments/run_all.sh 3000 0.83 sub sub25 100 10   (prep+train+eval+plot)
# stability:       experiments/run_stable.sh sub25_stable sub 100 10 1e-4 0.999 0.9
# D-weaken sweep:  experiments/run_sweep.sh
# eval any run:    experiments/eval_curve.py data/testset_sub.zip checkpoints/<name>-*/ckpt auto out.json
```
