"""Overlay the standard-FID-vs-epoch curves of several runs on one axis.

Usage:
  python experiments/plot_compare.py <out.png> "label1=path/to/eval.json" "label2=..." ...
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
specs = sys.argv[2:]
colors = plt.cm.tab10.colors

plt.figure(figsize=(9.5, 5.5))
for i, spec in enumerate(specs):
    label, path = spec.split("=", 1)
    res = json.load(open(path))
    es = sorted(int(k) for k in res)
    fids = [res[str(e)]["fid"] for e in es]
    best = min(es, key=lambda e: res[str(e)]["fid"])
    plt.plot(es, fids, "o-", color=colors[i % 10],
             label=f"{label}  (best {res[str(best)]['fid']:.0f} @ ep{best})")

plt.xlabel("epoch"); plt.ylabel("standard FID (2048-d pool3, lower = better)")
out_name = Path(out).name.lower()
if "diffaug" in out_name:
    title = "MS-CLIP-GAN — DiffAugment vs baseline (25% subset, 2490 imgs)"
else:
    title = "MS-CLIP-GAN — D-weakening sweep vs baseline (25% subset, 2490 imgs)"
plt.title(title)
plt.legend(fontsize=9); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(out, dpi=130)
print("saved", out)
