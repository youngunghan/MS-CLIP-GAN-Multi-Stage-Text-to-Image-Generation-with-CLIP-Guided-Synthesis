#!/usr/bin/env python3
"""Plot training curves: (left) per-epoch d_loss/g_loss parsed from the train log,
(right) standard FID + IS vs epoch from eval_curve.py's JSON.

Usage:
  python experiments/plot_curves.py <train.log> <eval.json> <out_dir>
"""
import os, re, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG, EVAL, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTDIR, exist_ok=True)

# ---- parse loss log ----
ep, dl, gl = [], [], []
pat = re.compile(r"Epoch:\s*(\d+)\s*\t?\s*d_loss:\s*([\d.eE+-]+)\s*\t?\s*g_loss:\s*([\d.eE+-]+)")
with open(LOG, errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            ep.append(int(m.group(1))); dl.append(float(m.group(2))); gl.append(float(m.group(3)))

# ---- read FID/IS ----
if os.path.exists(EVAL):
    with open(EVAL) as f:
        res = json.load(f)
else:
    res = {}
# Mirror plot_compare.load_run's requirement that every entry have "fid";
# unlike that stricter loader, "is_mean" is optional here (e.g. an FID-only
# eval run) and missing values are plotted as gaps (NaN) instead of raising
# a bare KeyError.
es = sorted(int(k) for k in res if isinstance(res[k], dict) and "fid" in res[k])
fids = [res[str(e)]["fid"] for e in es]
iss = [res[str(e)].get("is_mean", float("nan")) for e in es]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

ax[0].plot(ep, dl, label="d_loss", color="tab:orange", lw=1.5)
ax[0].plot(ep, gl, label="g_loss", color="tab:green", lw=1.5)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Training loss")
ax[0].legend(); ax[0].grid(alpha=0.3)

a = ax[1]
if es:
    a.plot(es, fids, "o-", color="tab:red", label="FID (2048, ↓)")
    a.set_ylabel("FID", color="tab:red"); a.tick_params(axis="y", labelcolor="tab:red")
    b = a.twinx()
    b.plot(es, iss, "s--", color="tab:blue", label="IS (↑)")
    b.set_ylabel("IS", color="tab:blue"); b.tick_params(axis="y", labelcolor="tab:blue")
    for x, y in zip(es, fids):
        a.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8, color="tab:red")
a.set_xlabel("epoch"); a.set_title("Standard FID / IS vs epoch")
a.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(OUTDIR, "curves.png")
plt.savefig(out, dpi=120)
print("saved", out)
if es:
    best = min(es, key=lambda e: res[str(e)]["fid"])
    print(f"best FID = {res[str(best)]['fid']:.2f} @ epoch {best}")
    print(f"last FID = {res[str(es[-1])]['fid']:.2f} @ epoch {es[-1]}")
