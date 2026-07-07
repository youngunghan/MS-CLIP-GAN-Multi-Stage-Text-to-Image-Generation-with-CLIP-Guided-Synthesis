"""Compute the STANDARD (torchmetrics, 2048-d pool3) FID + Inception Score for a
series of saved generator checkpoints, conditioning the fakes on the *actual test
captions* (one fake per test image) — not a single fixed prompt.

Usage:
  python experiments/eval_curve.py <testset.zip> <ckpt_dir> <epochs|auto> <out.json>

<ckpt_dir>  e.g. checkpoints/<name>/ckpt   (contains epoch_<E>_Gen.pt)
<epochs>    comma list "0,5,10" or "auto" to glob every saved epoch_*_Gen.pt
"""
import io, os, sys, json, glob, types, zipfile
import torch
import torchvision.transforms as T
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from utils.utils import normalize, load_checkpoint
from networks.generator import Generator

TEST_ZIP, CKPT_DIR, EPOCH_ARG, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS = 32
# Fixed seed so the noise draws (and thus FID/IS on a ~500-image test set) are
# reproducible and comparable across runs. Curves committed before this seed existed
# carry a small run-to-run noise component.
torch.manual_seed(42)

if EPOCH_ARG == "auto":
    eps = sorted(int(os.path.basename(p).split("_")[1])
                 for p in glob.glob(os.path.join(CKPT_DIR, "epoch_*_Gen.pt")))
else:
    eps = [int(x) for x in EPOCH_ARG.split(",")]

# ---- load test set: real 256px images + one caption embedding per image ----
zf = zipfile.ZipFile(TEST_ZIP)
meta = json.loads(zf.read("dataset.json"))
txt = {}
for fname, emb in meta["clip_txt_features"]:
    if not emb:
        continue
    t = torch.tensor(emb, dtype=torch.float32)
    txt[fname] = (normalize(t, dim=1) if isinstance(emb[0], list)
                  else normalize(t, dim=0).unsqueeze(0))
pngs = sorted(f for f in zf.namelist() if f.lower().endswith(".png") and f in txt)
to_t = T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
real = torch.stack([to_t(Image.open(io.BytesIO(zf.read(f))).convert("RGB")) for f in pngs])  # [N,3,256,256] in [0,1]
cap = torch.stack([txt[f][0] for f in pngs])  # [N,512] normalized, first caption
print(f"test set: {len(pngs)} images / captions", flush=True)

u8 = lambda x: (x.clamp(0, 1) * 255).to(torch.uint8)
args = types.SimpleNamespace(is_train=False, new_optim=False)
results = {}

for ep in eps:
    if not os.path.exists(os.path.join(CKPT_DIR, f"epoch_{ep}_Gen.pt")):
        print(f"epoch {ep}: (no checkpoint, skipped)", flush=True)
        continue
    G = Generator(1024, 3, 100, 128, 512, 3, DEV).to(DEV)
    load_checkpoint(args, G, [None] * 3, optim_g=None, optim_d_lst=[None] * 3,
                    checkpoint_path=CKPT_DIR, epoch=ep)
    G.eval()
    fid = FrechetInceptionDistance(normalize=False).to(DEV)
    isc = InceptionScore(normalize=False).to(DEV)
    with torch.no_grad():
        for i in range(0, len(real), BS):
            fid.update(u8(real[i:i + BS].to(DEV)), real=True)
        for i in range(0, len(cap), BS):
            c = cap[i:i + BS].to(DEV).float()
            imgs, _, _ = G(c, torch.randn(c.size(0), 100, device=DEV))
            f8 = u8((imgs[-1].clamp(-1, 1) + 1) / 2)
            fid.update(f8, real=False)
            isc.update(f8)
    fv = float(fid.compute()); im, isd = (float(x) for x in isc.compute())
    results[str(ep)] = {"fid": fv, "is_mean": im, "is_std": isd}
    print(f"epoch {ep:3d}:  FID(2048) = {fv:8.2f}   IS = {im:.3f} ± {isd:.3f}", flush=True)
    del G; torch.cuda.empty_cache()

json.dump(results, open(OUT, "w"), indent=2)
print("wrote", OUT)
