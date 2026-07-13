#!/usr/bin/env python3
"""Reproduce the FID feature-space pitfall on the shipped best checkpoint.

Same generator, same real/fake images, two extractors:
  - ignite default FID  -> num_features = 1000 (InceptionV3 logits)  [the bug]
  - torchmetrics FID    -> 2048-d pool3 features                     [the standard]

Prints both values + their ratio (and a real-vs-real ignite sanity check), and
saves a log-scale bar chart for the blog post. Fakes are conditioned on each test
caption (not a single fixed prompt).

Usage:
  PYTHONPATH=. conda run -n msclipgan python experiments/fid_feature_space_demo.py \
      models/best_sub25_ep20 20 data/testset_sub.zip experiments/results/ignite-vs-2048.png
"""
import io, sys, json, zipfile, types
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ignite.metrics import FID as IgniteFID
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.utils import normalize, load_checkpoint
from networks.generator import Generator

CKPT, EPOCH, TEST_ZIP, OUT_PNG = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS = 32

# ---- test set: real 256px images + one caption embedding per image ----
zf = zipfile.ZipFile(TEST_ZIP)
meta = json.loads(zf.read("dataset.json"))
txt = {}
for fname, emb in meta["clip_txt_features"]:
    if not emb:
        continue
    t = torch.tensor(emb, dtype=torch.float32)
    txt[fname] = (normalize(t, dim=1) if isinstance(emb[0], list)
                  else normalize(t, dim=0).unsqueeze(0))
    # end if
# end for
pngs = sorted(f for f in zf.namelist() if f.lower().endswith(".png") and f in txt)
to_t = T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
real = torch.stack([to_t(Image.open(io.BytesIO(zf.read(f))).convert("RGB")) for f in pngs])  # [N,3,256,256] in [0,1]
cap = torch.stack([txt[f][0] for f in pngs])  # [N,512] normalized
print(f"test set: {len(pngs)} images / captions", flush=True)

# ---- load generator + make one fake per test caption ----
G = Generator(1024, 3, 100, 128, 512, 3, DEV).to(DEV)
args = types.SimpleNamespace(is_train=False, new_optim=False)
load_checkpoint(args, G, [None] * 3, optim_g=None, optim_d_lst=[None] * 3, checkpoint_path=CKPT, epoch=EPOCH)
G.eval()
fakes = []
with torch.no_grad():
    for i in range(0, len(cap), BS):
        c = cap[i:i + BS].to(DEV).float()
        imgs, _, _ = G(c, torch.randn(c.size(0), 100, device=DEV))
        fakes.append(((imgs[-1].clamp(-1, 1) + 1) / 2).cpu())  # [0,1], 256px
    # end for
fake = torch.cat(fakes)

# ---- two feature spaces on the SAME images ----
fi = IgniteFID(device=DEV)
print("ignite default num_features =", fi._num_features,
      "| extractor =", type(fi._feature_extractor).__name__, flush=True)
fi.update((fake.to(DEV), real.to(DEV)))
ignite_fid = float(fi.compute())

u8 = lambda x: (x.clamp(0, 1) * 255).to(torch.uint8)
ft = FrechetInceptionDistance(normalize=False).to(DEV)
ft.update(u8(real.to(DEV)), real=True)
ft.update(u8(fake.to(DEV)), real=False)
tm_fid = float(ft.compute())

# real-vs-real sanity (ignite): should be ~0
fid_id = IgniteFID(device=DEV)
ridx = torch.randperm(len(real))
fid_id.update((real.to(DEV), real[ridx].to(DEV)))
ignite_realreal = float(fid_id.compute())

print(f"\nignite FID (1000-d logits)   = {ignite_fid:.4f}")
print(f"standard FID (2048-d pool3)  = {tm_fid:.2f}")
print(f"ratio (2048d / 1000d)        = {tm_fid / max(ignite_fid, 1e-9):.0f}x")
print(f"ignite real-vs-real (sanity) = {ignite_realreal:.4f}")

# ---- bar chart (log scale: the two values differ ~1000x) ----
fig, ax = plt.subplots(figsize=(6.4, 4.6))
labels = ["ignite default\n(1000-d logits)", "standard\n(2048-d pool3)"]
vals = [ignite_fid, tm_fid]
bars = ax.bar(labels, vals, color=["#d62728", "#1f77b4"], width=0.55)
ax.set_yscale("log")
ax.set_ylabel("FID (log scale)")
ax.set_title("Same model, same images — two FID feature spaces")
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.3f}" if v < 10 else f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    # end annotate
ax.margins(y=0.18)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=130)
print("saved", OUT_PNG)
