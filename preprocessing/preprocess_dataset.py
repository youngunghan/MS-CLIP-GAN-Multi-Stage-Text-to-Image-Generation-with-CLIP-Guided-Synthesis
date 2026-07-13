#!/usr/bin/env python3
"""Build a CLIP-embedded, RGB PNG dataset for MS-CLIP-GAN."""

import functools
import io
import json
import os
import pickle
import random
import shutil
import stat
import sys
import tarfile
import tempfile
import gzip
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import clip
import click
import numpy as np
import PIL.Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import cv2
from collections import OrderedDict
import os.path as op


CLIP_MODEL_NAME = "ViT-B/32"
CLIP_EMBEDDING_DIM = 512
ZIP_ENTRY_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _rgb_array(image: PIL.Image.Image) -> np.ndarray:
    """Return a detached uint8 RGB array regardless of the source image mode."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _require_unique_stems(paths, source_label: str) -> None:
    """Reject ambiguous image-to-caption identities before selecting samples."""
    by_stem = {}
    for path in paths:
        by_stem.setdefault(Path(path).stem, []).append(str(path))
    collisions = {stem: names for stem, names in by_stem.items() if len(names) > 1}
    if collisions:
        details = "; ".join(
            f"{stem}: {', '.join(names)}" for stem, names in sorted(collisions.items())
        )
        raise ValueError(f"duplicate image stems in {source_label}: {details}")


def _selected_stem_set(data_list, source_label: str):
    if not isinstance(data_list, (list, tuple, set)) or any(
        not isinstance(stem, str) or not stem for stem in data_list
    ):
        raise ValueError(f"{source_label} must contain non-empty string stems")
    if len(set(data_list)) != len(data_list):
        duplicates = sorted(
            stem for stem in set(data_list) if data_list.count(stem) > 1
        )
        raise ValueError(f"duplicate selected stems in {source_label}: {duplicates}")
    return set(data_list)


def _require_selected_stems(selected_stems, image_paths, source_label: str) -> None:
    available_stems = {Path(path).stem for path in image_paths}
    missing = sorted(selected_stems - available_stems)
    if missing:
        preview = ', '.join(missing[:20])
        suffix = '' if len(missing) <= 20 else f' ... ({len(missing)} total)'
        raise ValueError(f"selected image stems missing from {source_label}: {preview}{suffix}")


def seed_preprocessing(seed: int) -> None:
    """Seed every RNG used by preprocessing and select deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_reshape(img, mode='bicubic', ratio=0.99):   # more to be implemented here
    full_size = img.shape[-2]

    if full_size < 224:
        pad_1 = torch.randint(0, 224-full_size, ())
        pad_2 = torch.randint(0, 224-full_size, ())
        m = torch.nn.ConstantPad2d((pad_1, 224-full_size-pad_1, pad_2, 224-full_size-pad_2), 1.)
        reshaped_img = m(img)
    else:
        cut_size = torch.randint(int(ratio*full_size), full_size, ())
        left = torch.randint(0, full_size-cut_size, ())
        top = torch.randint(0, full_size-cut_size, ())
        cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
        reshaped_img = F.interpolate(cropped_img , (224, 224), mode=mode, align_corners=False)
    return  reshaped_img


def clip_preprocess():
    return T.Compose([
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------


# ####### Custom Implementation ######
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def get_img_id_to_img_path(source_dir, annotations):
    img_id_to_img_path = {}
    for img_id in annotations.keys():
        img_path = os.path.join(source_dir, img_id)
        img_id_to_img_path[img_id] = img_path

    return img_id_to_img_path


def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for img_id in annotations.keys():
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        caption = annotations[img_id]["overall_caption"]
        img_id_to_captions[img_id].append(caption)

    return img_id_to_captions

# ####### Custom Implementation ######

def open_image_folder(source_dir, *, max_images: Optional[int], src_data_list: str):
    print("Checking folder dataset...")
    print(f"Source directory: {source_dir}")
    print(f"Data list path: {src_data_list}")

    with open(src_data_list, 'rb') as f:
        data_list = pickle.load(f)
        print("Data list from pickle (first 5):", data_list[:5])

    image_dir = Path(op.join(source_dir, 'images'))
    print(f"Looking for images in: {image_dir}")

    all_images = [
        str(path) for path in sorted(image_dir.rglob('*'))
        if is_image_ext(path) and path.is_file()
    ]
    _require_unique_stems(all_images, str(image_dir))
    selected_stems = _selected_stem_set(data_list, src_data_list)
    _require_selected_stems(selected_stems, all_images, str(image_dir))
    input_images = [path for path in all_images if Path(path).stem in selected_stems]

    print(f"Found image files (first 5): {input_images[:5]}")
    print(f'Total images found: {len(input_images)}')

    max_idx = maybe_min(len(input_images), max_images)
    def iterate_images():
        for fname in input_images[:max_idx]:
            img_path = fname
            txt_path = op.join(source_dir, f"celeba-caption/{Path(img_path).stem}.txt")
            try:
                with PIL.Image.open(img_path) as source_image:
                    img = _rgb_array(source_image)
            except (OSError, ValueError) as exc:
                print(f'{img_path} failed: {exc}')
                continue

            try:
                with open(txt_path, 'r', encoding='utf-8') as file:
                    txt = file.read().splitlines()
            except (OSError, UnicodeError) as exc:
                print(f'Text file not found/readable: {txt_path}, Error: {exc}')
                txt = []

            yield dict(img=img, txt=txt)
    return max_idx, iterate_images()


#----------------------------------------------------------------------------
def open_image_zip(source_dir, max_images: Optional[int], src_data_list: str):
    print('Using zip file as dataset')

    with open(src_data_list, 'rb') as f:
        data_list = pickle.load(f)
        print("Data list from pickle (first 5):", data_list[:5])

    image_zip = os.path.join(source_dir, 'image.zip')
    text_zip = os.path.join(source_dir, 'text.zip')

    with zipfile.ZipFile(image_zip, mode='r') as z:
        all_files = z.namelist()
        print("Files in image.zip (first 5):", all_files[:5])

        all_images = [
            str(path) for path in sorted(all_files)
            if is_image_ext(path) and not path.endswith('/')
        ]
        _require_unique_stems(all_images, image_zip)
        selected_stems = _selected_stem_set(data_list, src_data_list)
        _require_selected_stems(selected_stems, all_images, image_zip)
        input_images = [path for path in all_images if Path(path).stem in selected_stems]
        print("Matched image files (first 5):", input_images[:5])

    print(f'Total matched images: {len(input_images)}')
    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(image_zip, mode='r') as img_z, \
             zipfile.ZipFile(text_zip, mode='r') as txt_z:
            for fname in input_images[:max_idx]:
                img_name = fname
                base_name = Path(fname).stem
                txt_name = f"celeba-caption/{base_name}.txt"

                try:
                    with img_z.open(img_name, 'r') as file:
                        with PIL.Image.open(file) as source_image:
                            img = _rgb_array(source_image)
                except (OSError, ValueError, KeyError) as exc:
                    print(f'Failed to process {fname}: {exc}')
                    continue

                try:
                    with txt_z.open(txt_name) as txt_file:
                        txt = txt_file.read().decode('utf-8').splitlines()
                except (OSError, UnicodeError, KeyError) as exc:
                    print(f'Text file not found/readable: {txt_name}, Error: {exc}')
                    txt = []

                yield dict(img=img, txt=txt)
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                if idx >= max_idx:
                    break
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except (IOError, cv2.error):
                        with PIL.Image.open(io.BytesIO(value)) as source_image:
                            img = _rgb_array(source_image)
                    if img.ndim != 3 or img.shape[2] != 3:
                        img = _rgb_array(PIL.Image.fromarray(img))
                    yield dict(img=img, label=None)
                except (OSError, ValueError, cv2.error) as exc:
                    print(f'LMDB sample {_key!r} failed: {exc}')

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    resample = { 'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS }[resize_filter]
    def scale(width, height, img):
        img = _rgb_array(PIL.Image.fromarray(img))
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img).convert('RGB')
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return _rgb_array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img).convert('RGB')
        img = img.resize((width, height), resample)
        return _rgb_array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img).convert('RGB')
        img = img.resize((width, height), resample)
        img = _rgb_array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int], src_data_list: str):
    if os.path.isdir(source):
        # 폴더 내에 zip 파일이 있는지 확인
        image_zip = os.path.join(source, 'image.zip')
        text_zip = os.path.join(source, 'text.zip')
        if os.path.isfile(image_zip) and os.path.isfile(text_zip):
            print(f"Found image.zip and text.zip in {source}")
            return open_image_zip(source, max_images=max_images, src_data_list=src_data_list)
        else:
            print(f"No zip files found in {source}, trying as regular folder")
            return open_image_folder(source, max_images=max_images, src_data_list=src_data_list)
    elif os.path.isfile(source):
        # open_image_zip expects a *directory* that contains image.zip and text.zip,
        # not a single archive file, so reject single-file sources with a clear message.
        error(f'--source must be a directory containing image.zip and text.zip '
              f'(or a folder of images), not a single file: {source}')
    else:
        error(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(
    dest: str,
) -> Tuple[
    str,
    Callable[[str, Union[bytes, str]], None],
    Callable[[], None],
    Callable[[], None],
]:
    """Open an output target and return ``root, write, commit, abort``.

    ZIP output is staged in a sibling temporary file.  ``commit`` closes and
    fsyncs the complete archive before atomically replacing ``dest``; ``abort``
    removes the staged archive and leaves any existing destination untouched.

    Folder output mirrors the same atomic pattern: files are written under a
    sibling temporary directory and ``commit`` atomically renames it onto
    ``dest``; ``abort`` removes the staged directory and leaves any existing
    ``dest`` untouched (so a hard failure never leaves partial PNGs behind or
    blocks a re-run).
    """
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        parent = os.path.dirname(os.path.abspath(dest))
        os.makedirs(parent, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f'.{os.path.basename(dest)}.', suffix='.tmp', dir=parent
        )
        try:
            os.close(fd)
            zf = zipfile.ZipFile(
                file=temporary, mode='w', compression=zipfile.ZIP_STORED
            )
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        closed = False
        committed = False

        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            if isinstance(data, str):
                data = data.encode('utf8')
            info = zipfile.ZipInfo(filename=fname, date_time=ZIP_ENTRY_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, data)

        def close_zip():
            nonlocal closed
            if not closed:
                zf.close()
                closed = True

        def commit_zip():
            nonlocal committed
            close_zip()
            with open(temporary, 'rb') as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, dest)
            committed = True

        def abort_zip():
            try:
                close_zip()
            finally:
                if not committed:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass

        return '', zip_write_bytes, commit_zip, abort_zip
    else:
        # If the output folder already exists, check that it is empty. This
        # is checked eagerly (rather than only at commit time) so a bad
        # --dest is reported before any work is staged.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')

        parent = os.path.dirname(os.path.abspath(dest))
        os.makedirs(parent, exist_ok=True)
        temporary = tempfile.mkdtemp(
            prefix=f'.{os.path.basename(dest)}.', suffix='.tmp', dir=parent
        )
        committed = False

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)

        def commit_folder():
            nonlocal committed
            # dest is guaranteed absent-or-empty by the eager guard above, and
            # os.replace() atomically replaces an existing empty directory (same
            # filesystem) in a single syscall, so no separate rmdir is needed.
            os.replace(temporary, dest)
            committed = True

        def abort_folder():
            if not committed:
                shutil.rmtree(temporary, ignore_errors=True)

        return temporary, folder_write_bytes, commit_folder, abort_folder

#----------------------------------------------------------------------------

def encode_text_features(clip_model, captions, device):
    """Encode up to ten non-empty captions, truncating safely at CLIP's limit."""
    if isinstance(captions, str):
        captions = captions.splitlines()
    if captions is None:
        captions = []

    valid_captions = [
        caption.strip()
        for caption in captions
        if isinstance(caption, str) and caption.strip()
    ]

    features = []
    for caption in valid_captions[:10]:
        tokenized = clip.tokenize([caption], truncate=True).to(device)
        with torch.no_grad():
            text_feature = clip_model.encode_text(tokenized)
        if text_feature.ndim != 2 or text_feature.shape != (1, CLIP_EMBEDDING_DIM):
            raise RuntimeError(
                f'{CLIP_MODEL_NAME} returned unexpected text feature shape '
                f'{tuple(text_feature.shape)}'
            )
        features.append(text_feature.reshape(-1).cpu().numpy().tolist())
    return features

#----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option(
    '--source',
    help='Directory containing image.zip/text.zip or images/ and celeba-caption/',
    required=True,
    metavar='PATH',
)
@click.option('--src_data_list', help='train or test dataset file name list', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option(
    '--max-images', help='Output only up to `max-images` images',
    type=click.IntRange(min=1), default=None
)
@click.option('--resize-filter', help='Filter to use when resizing images for output resolution', type=click.Choice(['box', 'lanczos']), default='lanczos', show_default=True)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)
@click.option(
    '--seed', help='Preprocessing RNG seed',
    type=click.IntRange(0, 2 ** 32 - 1), default=42, show_default=True
)
@click.option(
    '--emb_dim',
    help=f'CLIP embedding dimension (fixed by {CLIP_MODEL_NAME})',
    type=click.IntRange(CLIP_EMBEDDING_DIM, CLIP_EMBEDDING_DIM),
    required=True,
)
@click.option(
    '--max-failure-frac',
    help=(
        'Tolerate up to this fraction of per-sample failures (corrupt image, '
        'missing/empty caption, dropped transform) and still emit the dataset. '
        'Default 0.0 keeps the strict all-or-nothing behavior: any failure aborts.'
    ),
    type=click.FloatRange(0.0, 1.0), default=0.0, show_default=True,
)
def convert_dataset(
    ctx: click.Context,
    source: str,
    src_data_list: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int],
    seed: int,
    emb_dim: int,
    max_failure_frac: float,
):
    """Build the RGB PNG + CLIP-feature archive consumed by this repository.

    ``--source`` must be a directory. It may contain ``image.zip`` and
    ``text.zip``, or unpacked ``images/`` and ``celeba-caption/`` directories.
    ``--src_data_list`` selects image stems from a pickle produced by
    ``split_dataset.py``. The destination may be a directory or an uncompressed
    ZIP; ``dataset.json`` stores matching ``clip_img_features`` and
    ``clip_txt_features`` rows for every written PNG.

    Output images must have uniform, square, power-of-two dimensions. Use
    ``--transform=center-crop --width=256 --height=256`` for the normal pipeline.

    By default (``--max-failure-frac=0.0``) any per-sample failure aborts the
    whole run and leaves the destination untouched. Pass ``--max-failure-frac``
    above 0 to tolerate up to that fraction of skipped samples; the run still
    emits the dataset (with only the successful samples) and logs a summary of
    how many samples were skipped and why.
    """
    if emb_dim != CLIP_EMBEDDING_DIM:
        # Click validates this first; retain a direct-call guard for library users.
        ctx.fail(f'--emb_dim must be {CLIP_EMBEDDING_DIM} for {CLIP_MODEL_NAME}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    PIL.Image.init() # type: ignore
    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=device)
    clip_model.eval()
    # Model construction can consume Torch RNG state. Seed after loading so the
    # sample-to-crop mapping depends only on --seed, not model initialization.
    seed_preprocessing(seed)
    print('start')
    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')
    num_files, input_iter = open_dataset(source, max_images=max_images, src_data_list=src_data_list)
    print('source ready')
    # Validate transform arguments before creating a staged destination.
    transform_image = make_transform(transform, width, height, resize_filter)
    dataset_attrs = None

    clip_img_features = []
    clip_txt_features = []
    success_count = 0
    failure_count = 0
    processed_count = 0
    written_names = []

    archive_root_dir, save_bytes, commit_dest, abort_dest = open_dest(dest)
    try:
        print('target ready')
        for idx, image in tqdm(enumerate(input_iter), total=num_files):
            processed_count += 1
            idx_str = f'{idx:08d}'
            archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

            try:
                source_img = image['img']
                if not isinstance(source_img, np.ndarray):
                    raise TypeError(f"image must be a numpy array, got {type(source_img).__name__}")
                source_img = _rgb_array(PIL.Image.fromarray(source_img))
                img_array = transform_image(source_img)
                if img_array is None:
                    print(f'{archive_fname}: transform dropped sample')
                    failure_count += 1
                    continue
                img_array = _rgb_array(PIL.Image.fromarray(img_array))

                cur_image_attrs = {
                    'width': int(img_array.shape[1]),
                    'height': int(img_array.shape[0]),
                    'channels': int(img_array.shape[2]),
                }
                if dataset_attrs is None:
                    dataset_attrs = cur_image_attrs
                    image_width = dataset_attrs['width']
                    image_height = dataset_attrs['height']
                    if image_width != image_height:
                        raise click.ClickException(
                            'Image dimensions after scale and crop must be square. '
                            f'Got {image_width}x{image_height}'
                        )
                    if dataset_attrs['channels'] != 3:
                        raise click.ClickException('Preprocessed images must be RGB')
                    if image_width <= 0 or image_width & (image_width - 1):
                        raise click.ClickException(
                            'Image width/height after scale and crop must be a power of two'
                        )
                elif dataset_attrs != cur_image_attrs:
                    raise click.ClickException(
                        'All output images must have uniform attributes; expected '
                        f'{dataset_attrs}, got {cur_image_attrs} at {archive_fname}'
                    )

                pil_image = PIL.Image.fromarray(img_array, 'RGB')
                feature = torch.zeros(1, CLIP_EMBEDDING_DIM, device=device)
                reshaped_img = custom_reshape(T.ToTensor()(pil_image).unsqueeze(0))
                normed_img = clip_preprocess()(reshaped_img).to(device)
                with torch.no_grad():
                    encoded_image = clip_model.encode_image(normed_img)
                if encoded_image.shape != feature.shape:
                    raise RuntimeError(
                        f'{CLIP_MODEL_NAME} returned unexpected image feature shape '
                        f'{tuple(encoded_image.shape)}'
                    )
                feature.add_(encoded_image.to(feature.dtype))

                text_feature_list = encode_text_features(
                    clip_model, image.get('txt', []), device
                )
                if not text_feature_list:
                    print(f'{archive_fname}: no valid caption found, skipping sample')
                    failure_count += 1
                    continue
            except (SystemExit, KeyboardInterrupt, click.ClickException, RuntimeError):
                # RuntimeError includes torch.cuda.OutOfMemoryError.  These failures
                # are process-level/fatal and must never be mislabeled as a bad sample.
                raise
            except (KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
                print(f'{archive_fname} failed: {exc}')
                failure_count += 1
                continue

            image_bits = io.BytesIO()
            pil_image.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
            clip_img_features.append([
                archive_fname,
                feature.reshape(-1).cpu().numpy().tolist(),
            ])
            clip_txt_features.append([archive_fname, text_feature_list])
            written_names.append(archive_fname)
            success_count += 1

        if processed_count > num_files:
            raise RuntimeError(
                f'input iterator yielded {processed_count} samples, more than declared {num_files}'
            )
        # Source readers skip corrupt files before yielding. Account for those so the
        # declared input count remains reconcilable with successes and failures.
        failure_count += num_files - processed_count

        if success_count == 0:
            raise click.ClickException(
                f'No samples were successfully preprocessed ({failure_count} failed)'
            )
        # num_files > 0 is guaranteed here: success_count > 0 implies
        # success_count <= num_files.
        failure_frac = failure_count / num_files
        if failure_count and failure_frac > max_failure_frac:
            raise click.ClickException(
                f'Preprocessing was incomplete: {success_count} succeeded, '
                f'{failure_count} failed ({failure_frac:.1%} of {num_files}, '
                f'allowed {max_failure_frac:.1%} via --max-failure-frac); '
                f'destination was not replaced'
            )
        if failure_count:
            print(
                f'Tolerating {failure_count} failed/skipped sample(s) out of {num_files} '
                f'({failure_frac:.1%} <= --max-failure-frac={max_failure_frac:.1%}); '
                f'emitting dataset with the remaining {success_count} sample(s)'
            )
        if not (
            success_count == len(written_names)
            == len(clip_img_features)
            == len(clip_txt_features)
        ):
            raise RuntimeError('output image/feature counts are inconsistent')
        if success_count + failure_count != num_files:
            raise RuntimeError(
                f'input count mismatch: {success_count} succeeded + {failure_count} failed '
                f'!= {num_files} selected'
            )
        if [row[0] for row in clip_img_features] != written_names or [
            row[0] for row in clip_txt_features
        ] != written_names:
            raise RuntimeError('output image and feature filenames are inconsistent')

        metadata = {
            'clip_img_features': clip_img_features,
            'clip_txt_features': clip_txt_features,
            'preprocess': {
                'schema_version': 1,
                'seed': seed,
                'clip_model': CLIP_MODEL_NAME,
                'clip_embedding_dim': CLIP_EMBEDDING_DIM,
                'transform': transform,
                'resize_filter': resize_filter,
                'width': width,
                'height': height,
                'max_images': max_images,
                'image_feature_crop_ratio': 0.99,
            },
        }
        save_bytes(
            os.path.join(archive_root_dir, 'dataset.json'),
            json.dumps(metadata, sort_keys=True, separators=(',', ':')),
        )
        commit_dest()
        print(f'{success_count} succeeded, {failure_count} failed')
    finally:
        abort_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
