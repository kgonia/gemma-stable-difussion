"""Streaming, full-frame image-caption data loading for SD training."""
from __future__ import annotations
import glob
import io
import json
import math
import os
import random
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

from pure_ella.camera import (
    CAMERA_CONDITION_DIM,
    extract_camera_metadata,
    camera_metadata_to_tensor,
)


BUCKETS = (
    (1024, 1024), (1024, 896), (1024, 768), (1024, 640), (1024, 512),
    (896, 1024), (768, 1024), (640, 1024), (512, 1024),
)


def _get_bucket(w: int, h: int, buckets=BUCKETS):
    target = w / h
    return min(buckets, key=lambda x: abs(math.log((x[0] / x[1]) / target)))


def resize_long_edge(img: Image.Image, max_dimension: int = 1024):
    """Downscale without cropping so neither dimension exceeds the limit."""
    max_dimension = int(max_dimension)
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    width, height = img.size
    longest = max(width, height)
    if longest <= max_dimension:
        return img
    scale = max_dimension / longest
    size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return img.resize(size, Image.Resampling.LANCZOS)


def resize_full_frame(img: Image.Image, bucket):
    """Fit an entire image into a bucket and return a padding content mask."""
    bw, bh = (int(bucket[0]), int(bucket[1]))
    resized = ImageOps.contain(img, (bw, bh), method=Image.Resampling.LANCZOS)
    left = (bw - resized.width) // 2
    top = (bh - resized.height) // 2
    canvas = Image.new("RGB", (bw, bh), color=(127, 127, 127))
    canvas.paste(resized, (left, top))
    mask = Image.new("L", (bw, bh), color=0)
    mask.paste(255, (left, top, left + resized.width, top + resized.height))
    return canvas, mask


def _local_parquet_files(source: str) -> list[str]:
    """Resolve a local Parquet file, directory, or glob; return [] for HF IDs."""
    expanded = os.path.expandvars(os.path.expanduser(source)).replace("\\", "/")
    if len(expanded) >= 3 and expanded[1:3] == ":/":
        expanded = f"/mnt/{expanded[0].lower()}/{expanded[3:]}"
    elif expanded.startswith("mnt/"):
        expanded = f"/{expanded}"
    path = Path(expanded)
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Unsupported local data file: {source}")
        return [str(path.resolve())]
    if path.is_dir():
        files = sorted(str(item.resolve()) for item in path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files found under {source}")
        return files
    if glob.has_magic(expanded):
        files = sorted(
            str(Path(item).resolve()) for item in glob.glob(expanded, recursive=True)
            if Path(item).is_file() and Path(item).suffix.lower() == ".parquet"
        )
        if not files:
            raise FileNotFoundError(f"No Parquet files match {source}")
        return files
    if (expanded.startswith(("/", "./", "../"))
            or expanded.lower().endswith(".parquet")):
        raise FileNotFoundError(f"Local data source not found: {source}")
    return []


class RoundRobinSources:
    """Yield each configured source once without requiring matching schemas."""

    def __init__(self, sources, seed: int):
        self.sources = list(sources)
        self.seed = int(seed)

    def __iter__(self):
        sources = list(self.sources)
        random.Random(self.seed).shuffle(sources)
        active = [
            [iter(dataset), source_root]
            for dataset, source_root in sources
        ]
        while active:
            remaining = []
            for iterator, source_root in active:
                try:
                    sample = dict(next(iterator))
                except StopIteration:
                    continue
                sample.setdefault("_source_root", source_root)
                yield sample
                remaining.append([iterator, source_root])
            active = remaining


class StreamingSDDataset(IterableDataset):
    """Normalize heterogeneous image-caption samples into bucketed tensors."""

    def __init__(self, ds_iter, max_samples: int = 2000, buckets=BUCKETS,
                 max_image_dimension: int = 1024):
        self.ds_iter = ds_iter
        self.max_samples = int(max_samples)
        self.buckets = tuple((int(w), int(h)) for w, h in buckets)
        self.max_image_dimension = int(max_image_dimension)

    def __iter__(self):
        def _get_caption_variants(sample: dict) -> dict:
            meta = sample.get("json", {})
            if isinstance(meta, bytes):
                meta = meta.decode("utf-8", errors="ignore")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}

            def value(*keys):
                for source in (sample, meta):
                    for key in keys:
                        candidate = source.get(key)
                        if candidate is None:
                            continue
                        if isinstance(candidate, float) and math.isnan(candidate):
                            continue
                        text = str(candidate).strip()
                        if text and text.lower() not in {"nan", "none", "null"}:
                            return text
                return ""

            long_caption = value(
                "training_caption", "caption_detailed", "caption_long",
                "long_caption", "prompt", "caption", "text")
            medium_caption = value("caption_medium", "medium_caption")
            short_caption = value(
                "caption_short", "short_caption", "original_caption")
            if not long_caption:
                long_caption = medium_caption or short_caption
            return {
                "caption": long_caption,
                "caption_medium": medium_caption,
                "caption_short": short_caption,
            }

        def _get_image(sample: dict):
            for key in ["jpg", "jpeg", "png", "webp", "image"]:
                img = sample.get(key)
                if img is not None:
                    return img
            for key in ["local_image_path", "image_path", "path", "filepath"]:
                path = sample.get(key)
                if path:
                    path = Path(os.path.expanduser(str(path)))
                    if not path.is_absolute():
                        path = Path(sample.get("_source_root", ".")) / path
                    return str(path)
            return None

        def _open_image(image_value):
            if isinstance(image_value, Image.Image):
                return image_value
            if isinstance(image_value, dict):
                if image_value.get("bytes") is not None:
                    return Image.open(io.BytesIO(image_value["bytes"]))
                image_value = image_value.get("path")
            if isinstance(image_value, (bytes, bytearray, memoryview)):
                return Image.open(io.BytesIO(bytes(image_value)))
            if isinstance(image_value, (str, os.PathLike)):
                return Image.open(image_value)
            raise TypeError(f"Unsupported image value: {type(image_value).__name__}")

        count = 0
        for sample in self.ds_iter:
            if count >= self.max_samples:
                break
            captions = _get_caption_variants(sample)
            if not captions["caption"]:
                continue
            image_value = _get_image(sample)
            if image_value is None:
                continue
            try:
                opened = _open_image(image_value)
                try:
                    oriented = ImageOps.exif_transpose(opened)
                    try:
                        img = oriented.convert("RGB")
                    finally:
                        if oriented is not opened:
                            oriented.close()
                finally:
                    opened.close()
                img = resize_long_edge(img, self.max_image_dimension)
            except (OSError, TypeError, ValueError) as exc:
                print(f"WARNING: skipping unreadable image {image_value!r}: {exc}")
                continue
            bw, bh = _get_bucket(img.width, img.height, self.buckets)
            img, content_mask = resize_full_frame(img, (bw, bh))
            img_tensor = transforms.ToTensor()(img) * 2 - 1
            mask_tensor = transforms.ToTensor()(content_mask)
            yield {
                "image": img_tensor,
                "image_mask": mask_tensor,
                "camera_condition": camera_metadata_to_tensor(
                    extract_camera_metadata(sample)),
                "bucket": (bw, bh),
                **captions,
            }
            count += 1


class BucketBatchDataset(IterableDataset):
    """Collect streaming samples into shape-compatible aspect-ratio batches."""

    def __init__(self, dataset: IterableDataset, batch_size: int,
                 drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    @staticmethod
    def _collate(samples):
        return {
            "image": torch.stack([sample["image"] for sample in samples]),
            "image_mask": torch.stack([sample["image_mask"] for sample in samples]),
            "caption": [sample["caption"] for sample in samples],
            "caption_medium": [sample.get("caption_medium", "") for sample in samples],
            "caption_short": [sample.get("caption_short", "") for sample in samples],
            "camera_condition": torch.stack([
                sample.get(
                    "camera_condition",
                    torch.zeros(CAMERA_CONDITION_DIM, dtype=torch.float32),
                )
                for sample in samples
            ]),
            "bucket": samples[0]["bucket"],
        }

    def __iter__(self):
        buffers = {}
        for sample in self.dataset:
            bucket = sample["bucket"]
            buffer = buffers.setdefault(bucket, [])
            buffer.append(sample)
            if len(buffer) == self.batch_size:
                yield self._collate(buffer)
                buffers[bucket] = []
        if not self.drop_last:
            for buffer in buffers.values():
                if buffer:
                    yield self._collate(buffer)


def make_streaming_dataloader(
    data_sources: Sequence[str] | str,
    phase: int,
    epoch: int,
    max_samples: int,
    batch_size: int = 4,
    shuffle: bool = True,
    shuffle_buffer: int = 10_000,
    base_seed: int = 1234,
    buckets=BUCKETS,
    drop_last: bool = True,
    max_image_dimension: int = 1024,
) -> DataLoader:
    """Create a loader from local Parquet locations and/or HF repositories."""
    from datasets import load_dataset

    if isinstance(data_sources, str):
        data_sources = [data_sources]
    data_sources = list(data_sources)
    if not data_sources:
        raise ValueError("data_sources must not be empty")

    shuffle_seed = base_seed + 1000 * int(phase) + int(epoch)
    loaded_sources = []
    for source in data_sources:
        parquet_files = _local_parquet_files(source)
        if parquet_files:
            source_specs = [
                ("parquet", {"train": [parquet_file]},
                 str(Path(parquet_file).parent))
                for parquet_file in parquet_files
            ]
        else:
            source_specs = [(source, None, "")]
        for dataset_name, data_files, source_root in source_specs:
            kwargs = {"split": "train", "streaming": True}
            if data_files is not None:
                kwargs["data_files"] = data_files
            dataset = load_dataset(dataset_name, **kwargs)
            if shuffle:
                dataset = dataset.shuffle(
                    buffer_size=shuffle_buffer,
                    seed=shuffle_seed + len(loaded_sources),
                )
            loaded_sources.append((dataset, source_root))

    ds_full = RoundRobinSources(loaded_sources, seed=shuffle_seed)
    samples = StreamingSDDataset(
        ds_full, max_samples=max_samples, buckets=buckets,
        max_image_dimension=max_image_dimension)
    ds = BucketBatchDataset(
        samples, batch_size=batch_size, drop_last=drop_last)
    dl = DataLoader(ds, batch_size=None, num_workers=0)
    print(
        f"DataLoader phase={phase} epoch={epoch} max_samples={max_samples} "
        f"batch={batch_size} buckets={list(map(tuple, buckets))} "
        f"max_image_dimension={max_image_dimension} sources={data_sources} "
        f"drop_last={drop_last} shuffle={shuffle} seed={shuffle_seed}"
    )
    return dl
