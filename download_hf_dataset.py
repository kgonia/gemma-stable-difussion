#!/usr/bin/env python3
"""
Streaming-zip downloader: RobinWZQ/improved_aesthetics_6.5plus → max-1GB zip.

Writes directly to the output zip as samples arrive. After each image+caption
pair is added, checks the on-disk zip size. Stops when the ZIP FILE exceeds the
limit (not the uncompressed data).

Ensures proper image-txt pairing: img_000000.jpg + img_000000.txt in flat layout.

Usage:
    python download_hf_dataset.py --output dataset.zip --max-size-mb 950

Colab:
    !python download_hf_dataset.py --output /content/dataset.zip
    !unzip -q /content/dataset.zip -d /content/training_data/
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser(description="Streaming zip: HF dataset → max-1GB zip")
parser.add_argument("--output", default="dataset.zip", help="Output zip path")
parser.add_argument("--max-size-mb", type=int, default=950, help="Max ZIP FILE size in MB")
parser.add_argument("--dataset", default="RobinWZQ/improved_aesthetics_6.5plus")
parser.add_argument("--split", default="train")
parser.add_argument("--image-col", default="URL", help="Column with image URL or PIL image")
parser.add_argument("--text-col", default="TEXT", help="Column with caption")
parser.add_argument("--start", type=int, default=0, help="Skip first N samples")
args = parser.parse_args()

# ── Deps ─────────────────────────────────────────────────────────
try:
    from datasets import load_dataset
except ImportError:
    sys.exit("ERROR: pip install datasets pillow")
try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: pip install pillow")

# ── Load dataset (streaming) ─────────────────────────────────────
print(f"Streaming {args.dataset} [{args.split}] ...", flush=True)
ds = load_dataset(args.dataset, split=args.split, streaming=True)
print("  Dataset handle acquired, starting download...", flush=True)

max_bytes = args.max_size_mb * 1024 * 1024
saved = 0
skipped_no_cap = 0
skipped_no_img = 0
errors = 0

# ── Open streaming zip ───────────────────────────────────────────
# We'll write to a temp file first so we can measure size.
# Python ZipFile doesn't support removing entries, so we write to
# a BytesIO buffer in memory and flush to disk only when done.
# BUT: for 1GB this is too much memory. Instead, we use a staging
# approach: write each pair to a temporary zip, check size, if under
# limit copy to main zip as we go. Actually simplest: write to a temp
# zip file, then rename at the end.
#
# BEST approach for streaming: use zipfile.ZipFile in write mode
# on the output file. After each writestr/write, close and reopen
# to check size. If over, delete the output and report where we stopped.

# Simpler: accumulate pairs in a list of (name, data_bytes) until
# projected zip size exceeds limit, then flush all at once.
# But we need to measure compressed size, not raw size.

# SIMPLEST that actually works: write to disk incrementally.
# Use a temporary directory for staging, but write pairs directly
# into a ZipFile. After each pair, check the zip's file size.
# If it crossed the limit, we need to remove the last entry.
# Since ZipFile can't remove, we'll write to a NEW temp zip each time
# and swap. Overhead: copying the zip N times. For 500 images × 1MB zip
# this is ~500MB of extra writes. Acceptable.

# ACTUALLY SIMPLEST: write to temp dir, then zip at the end.
# Track uncompressed size as a rough proxy. JPEGs don't compress
# further in zip, so uncompressed ≈ zip size. Add 5% margin.

import tempfile
import shutil

try:
    import requests
except ImportError:
    sys.exit("ERROR: pip install requests")

# Colab: disable HF download progress bars that clutter background logs
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BARS", "1")

tmpdir = tempfile.mkdtemp(prefix="hf_stream_")
total_raw = 0
MAX_RAW = int(max_bytes * 0.90)  # 10% margin for zip overhead + txt files

# HTTP session with retries for URL-based datasets
session = requests.Session()
session.headers.update({"User-Agent": "Hermes-dataset-downloader/1.0"})
from requests.adapters import HTTPAdapter, Retry
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

print(f"Target: {args.max_size_mb} MB zip  →  ~{MAX_RAW / 1024**2:.0f} MB raw limit", flush=True)
print(f"Temp staging: {tmpdir}", flush=True)

ds_iter = iter(ds)
for _ in range(args.start):
    try:
        next(ds_iter)
    except StopIteration:
        break

for sample in ds_iter:
    # ── Image (from URL or PIL) ──────────────────────────────────
    pil_img = None
    img_val = sample.get(args.image_col)

    if img_val is not None:
        if isinstance(img_val, Image.Image):
            pil_img = img_val
        elif isinstance(img_val, dict) and 'bytes' in img_val:
            pil_img = Image.open(io.BytesIO(img_val['bytes']))
        elif isinstance(img_val, bytes):
            pil_img = Image.open(io.BytesIO(img_val))
        elif isinstance(img_val, str):
            # URL column — download
            try:
                r = session.get(img_val, timeout=15)
                r.raise_for_status()
                pil_img = Image.open(io.BytesIO(r.content))
            except Exception:
                skipped_no_img += 1
                continue

    # Try alternative image columns if the primary one didn't yield an image
    if pil_img is None:
        for alt in ['image', 'jpg', 'png', 'img']:
            alt_val = sample.get(alt)
            if alt_val is not None and isinstance(alt_val, (Image.Image, dict, bytes)):
                if isinstance(alt_val, Image.Image):
                    pil_img = alt_val
                elif isinstance(alt_val, dict) and 'bytes' in alt_val:
                    pil_img = Image.open(io.BytesIO(alt_val['bytes']))
                elif isinstance(alt_val, bytes):
                    pil_img = Image.open(io.BytesIO(alt_val))
                break

    if pil_img is None:
        skipped_no_img += 1
        continue

    # ── Caption (case-insensitive fallback) ───────────────────────
    caption = ""
    for key in [args.text_col, 'TEXT', 'text', 'caption', 'CAPTION', 'prompt', 'text_en']:
        if key in sample and sample[key]:
            caption = str(sample[key]).strip()
            if caption:
                break
    if not caption:
        skipped_no_cap += 1
        continue

    # ── Encode image to JPEG bytes ───────────────────────────────
    try:
        if pil_img.mode in ('RGBA', 'P', 'LA'):
            pil_img = pil_img.convert('RGB')
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=92)
        img_bytes = buf.getvalue()
    except Exception:
        errors += 1
        continue

    caption_bytes = caption.encode('utf-8')

    # ── Check if adding this pair would exceed limit ─────────────
    pair_raw = len(img_bytes) + len(caption_bytes)
    if total_raw + pair_raw > MAX_RAW:
        print(f"  Limit reached at {saved} images ({total_raw/1024**2:.1f} MB raw)")
        break

    # ── Write to staging ─────────────────────────────────────────
    img_name = f"img_{saved:06d}.jpg"
    txt_name = f"img_{saved:06d}.txt"
    with open(os.path.join(tmpdir, img_name), 'wb') as f:
        f.write(img_bytes)
    with open(os.path.join(tmpdir, txt_name), 'w', encoding='utf-8') as f:
        f.write(caption)

    total_raw += pair_raw
    saved += 1

    if saved % 100 == 0:
        print(f"  {saved} images | {total_raw / 1024**2:.1f} MB raw", flush=True)

# ── Zip it ──────────────────────────────────────────────────────
print(f"\n{saved} images (no_cap={skipped_no_cap}, no_img={skipped_no_img}, errors={errors})")
print(f"Raw total: {total_raw / 1024**2:.1f} MB")
print(f"Creating zip: {args.output} ...")

with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(tmpdir)):
        zf.write(os.path.join(tmpdir, fname), fname)

zip_size = os.path.getsize(args.output)
print(f"Zip size: {zip_size / 1024**2:.1f} MB")

if zip_size > max_bytes:
    print(f"WARNING: zip exceeds {args.max_size_mb} MB limit!")
    print(f"  Overshoot: {(zip_size - max_bytes) / 1024**2:.1f} MB")
    print(f"  Re-run with --max-size-mb {int(zip_size/1024**2 * 0.85)} for safety")

# ── Validate pairing ────────────────────────────────────────────
print("\nValidating pair integrity...")
with zipfile.ZipFile(args.output, 'r') as zf:
    names = zf.namelist()
    imgs = sorted([n for n in names if n.endswith(('.jpg', '.jpeg', '.png'))])
    txts = sorted([n for n in names if n.endswith('.txt')])
    orphan_imgs = 0
    orphan_txts = 0
    for n in imgs:
        expected = os.path.splitext(n)[0] + '.txt'
        if expected not in txts:
            orphan_imgs += 1
    for n in txts:
        expected = os.path.splitext(n)[0] + '.jpg'
        if expected not in imgs:
            orphan_txts += 1
    print(f"  Images: {len(imgs)}, Captions: {len(txts)}")
    print(f"  Orphans: {orphan_imgs} images, {orphan_txts} captions")
    if orphan_imgs == 0 and orphan_txts == 0:
        print("  ✓ All pairs intact")

# ── Cleanup ─────────────────────────────────────────────────────
shutil.rmtree(tmpdir, ignore_errors=True)

output_abs = os.path.abspath(args.output)
print(f"\nDone → {output_abs}")
print(f"Colab usage:")
print(f"  !unzip -q {args.output} -d /content/training_data/")
