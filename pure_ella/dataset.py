"""
Streaming dataset for SD training via HuggingFace datasets.

Uses streaming mode to avoid downloading the entire dataset to disk.
"""
from __future__ import annotations
import io
import json
import math
from typing import List, Optional

from PIL import Image
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


BUCKETS = [(512, 512)]  # single-bucket default


def _get_bucket(w: int, h: int):
    target = w / h
    return min(BUCKETS, key=lambda x: abs(x[0] / x[1] - target))


class StreamingSDDataset(IterableDataset):
    """Streaming SD dataset from a HuggingFace dataset repository.

    Yields dicts with ``image`` (torch float32 in [-1, 1]) and ``caption`` (str).
    """
    def __init__(self, ds_iter, max_samples: int = 2000):
        self.ds_iter = ds_iter
        self.max_samples = int(max_samples)

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
            bw, bh = _get_bucket(img.width, img.height)
            b_ratio = bw / bh
            w, h = img.size
            if w / h > b_ratio:
                new_w = int(h * b_ratio)
                img = img.crop(((w - new_w) // 2, 0, (w + new_w) // 2, h))
            else:
                new_h = int(w / b_ratio)
                img = img.crop((0, (h - new_h) // 2, w, (h + new_h) // 2))
            img = img.resize((bw, bh), Image.LANCZOS)
            img_tensor = transforms.ToTensor()(img) * 2 - 1
            yield {"image": img_tensor, "caption": caption}
            count += 1


def make_streaming_dataloader(
    repo: str,
    phase: int,
    epoch: int,
    max_samples: int,
    batch_size: int = 4,
    shuffle: bool = True,
    shuffle_buffer: int = 10_000,
    base_seed: int = 1234,
) -> DataLoader:
    """Create a streaming DataLoader with epoch-aware shuffle seed."""
    ds_full = load_dataset(repo, split="train", streaming=True)
    shuffle_seed = base_seed + 1000 * int(phase) + int(epoch)
    if shuffle:
        ds_full = ds_full.shuffle(buffer_size=shuffle_buffer, seed=shuffle_seed)
    ds = StreamingSDDataset(ds_full, max_samples=max_samples)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
    print(
        f"DataLoader phase={phase} epoch={epoch} max_samples={max_samples} "
        f"batch={batch_size} shuffle={shuffle} seed={shuffle_seed}"
    )
    return dl
