# Best model — `sub25` epoch 20

Locked best checkpoint from the post-audit subset experiments.

- **Run:** `sub25` (baseline default config), 100 epochs, 25% subset (2490 train / 510 test).
- **Epoch:** 20 (the FID minimum; training degrades after ~ep20–30).
- **Standard FID (2048-d pool3):** ≈ **163** — the best across the four **pre-DiffAugment
  sweep** runs (baseline 163, stability 179, swA_aggr 183, swB_mild 159 — see
  `experiments/RESULTS.md`). It is **not** the best FID in the repo overall: a later
  `diffaug100` run (DiffAugment on top of this same baseline config) reached FID ≈
  118.5 at epoch 90 — see "DiffAugment run" in `experiments/RESULTS.md`. That
  checkpoint has not been promoted here or elsewhere under `models/`; this directory
  remains the epoch-20 baseline artifact, best *within the pre-DiffAugment sweep*.
- **Config:** Adam 2e-4 (G and D), CosineAnnealing, no EMA / no TTUR / no label smoothing.
  The D-weakening variants did **not** improve on this — see RESULTS.md.

## Files
- `epoch_20_Gen.pt` — generator (the one to use for inference/eval)
- `epoch_20_Dis_{0,1,2}.pt` — per-stage discriminators (for resume only)
- `sample_epoch20.png` — sample grid at this checkpoint

> Note: `*.pt` is git-ignored, so these weights live **locally only**; this README + RESULTS.md
> are the committed record of what the lock is and how to regenerate it.

## Use
```bash
# FID/IS on the test set
python experiments/eval_curve.py data/testset_sub.zip models/best_sub25_ep20 20 /tmp/best_eval.json
# inference: point infer.sh / scripts/infer.py at this checkpoint dir, epoch 20
```
