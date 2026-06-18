"""Download N real (image, caption) pairs from HuggingFace celebahq-caption-10k
into data/mm-celeba-hq-dataset/{image.zip,text.zip} — the raw input expected by
the repo preprocessing pipeline. N is given on the command line.

Streaming order is deterministic, so the first K of any larger download are a
stable superset: a 2500-image subset is exactly the first 2500 of a 10000 pull.
"""
import io, os, sys, zipfile
from PIL import Image
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
root = "data/mm-celeba-hq-dataset"
os.makedirs(root, exist_ok=True)
ds = load_dataset("ixw/celebahq-caption-10k", split="train", streaming=True)

iz = zipfile.ZipFile(os.path.join(root, "image.zip"), "w", zipfile.ZIP_STORED)
tz = zipfile.ZipFile(os.path.join(root, "text.zip"), "w", zipfile.ZIP_DEFLATED)
n = 0
for row in ds:
    if n >= N:
        break
    im = row["image"]
    if isinstance(im, dict):
        img = Image.open(io.BytesIO(im["bytes"])) if im.get("bytes") else Image.open(im["path"])
    else:
        img = im
    img = img.convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256))
    b = io.BytesIO(); img.save(b, "JPEG", quality=95)
    n += 1
    iz.writestr(f"images/{n:06d}.jpg", b.getvalue())
    cap = (row.get("text") or "").strip() or "a photography of a person"
    tz.writestr(f"celeba-caption/{n:06d}.txt", cap)
    if n % 500 == 0:
        print(f"  ... {n}/{N}", flush=True)
iz.close(); tz.close()
print(f"downloaded {n} REAL samples -> {root}/image.zip + text.zip")
