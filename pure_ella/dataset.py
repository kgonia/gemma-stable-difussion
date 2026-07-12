"""
Streaming dataset for SD training via HuggingFace datasets.

Uses streaming mode to avoid downloading the entire dataset to disk.
"""
from __future__ import annotations
import io
import json
import math
from typing import List, Optional

from PIL import Image, ImageOps
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms


BUCKETS = (
    (512, 512), (576, 448), (640, 384),
    (448, 576), (384, 640),
)


def _get_bucket(w: int, h: int, buckets=BUCKETS):
    target = w / h
    return min(buckets, key=lambda x: abs(math.log((x[0] / x[1]) / target)))


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


class StreamingSDDataset(IterableDataset):
    """Streaming SD dataset from a HuggingFace dataset repository.

    Yields dicts with ``image`` (torch float32 in [-1, 1]) and ``caption`` (str).
    """
    def __init__(self, ds_iter, max_samples: int = 2000, buckets=BUCKETS):
        self.ds_iter = ds_iter
        self.max_samples = int(max_samples)
        self.buckets = tuple((int(w), int(h)) for w, h in buckets)

    def __iter__(self):
        def _get_caption(sample: dict) -> str:
            meta = sample.get("json", {})
            if isinstance(meta, bytes):
                meta = meta.decode("utf-8", errors="ignore")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    return ""
            if isinstance(meta, dict):
                return meta.get("prompt") or meta.get("caption") or meta.get("text") or ""
            return ""

        def _get_image(sample: dict):
            for key in ["jpg", "jpeg", "png", "webp", "image"]:
                img = sample.get(key)
                if img is not None:
                    return img
            return None

        count = 0
        for sample in self.ds_iter:
            if count >= self.max_samples:
                break
            caption = _get_caption(sample)
            if not caption:
                continue
            img = _get_image(sample)
            if img is None:
                continue
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
            bw, bh = _get_bucket(img.width, img.height, self.buckets)
            img, content_mask = resize_full_frame(img, (bw, bh))
            img_tensor = transforms.ToTensor()(img) * 2 - 1
            mask_tensor = transforms.ToTensor()(content_mask)
            yield {
                "image": img_tensor,
                "image_mask": mask_tensor,
                "caption": caption,
                "bucket": (bw, bh),
            }
            count += 1


class BucketBatchDataset(IterableDataset):
    """Collect streaming samples into shape-compatible aspect-ratio batches."""

    def __init__(self, dataset: IterableDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)

    @staticmethod
    def _collate(samples):
        return {
            "image": torch.stack([sample["image"] for sample in samples]),
            "image_mask": torch.stack([sample["image_mask"] for sample in samples]),
            "caption": [sample["caption"] for sample in samples],
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
        for buffer in buffers.values():
            if buffer:
                yield self._collate(buffer)


def make_streaming_dataloader(
    repo: str,
    phase: int,
    epoch: int,
    max_samples: int,
    batch_size: int = 4,
    shuffle: bool = True,
    shuffle_buffer: int = 10_000,
    base_seed: int = 1234,
    buckets=BUCKETS,
) -> DataLoader:
    """Create a streaming DataLoader with epoch-aware shuffle seed."""
    from datasets import load_dataset

    ds_full = load_dataset(repo, split="train", streaming=True)
    shuffle_seed = base_seed + 1000 * int(phase) + int(epoch)
    if shuffle:
        ds_full = ds_full.shuffle(buffer_size=shuffle_buffer, seed=shuffle_seed)
    samples = StreamingSDDataset(
        ds_full, max_samples=max_samples, buckets=buckets)
    ds = BucketBatchDataset(samples, batch_size=batch_size)
    dl = DataLoader(ds, batch_size=None, num_workers=0)
    print(
        f"DataLoader phase={phase} epoch={epoch} max_samples={max_samples} "
        f"batch={batch_size} buckets={list(map(tuple, buckets))} "
        f"shuffle={shuffle} seed={shuffle_seed}"
    )
    return dl
