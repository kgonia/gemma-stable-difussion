"""Streaming, full-frame image-caption data loading for SD training."""
from __future__ import annotations
import glob
import io
import json
import math
import os
import random
import tarfile
from pathlib import Path
from typing import Any, Sequence

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

DEFAULT_PROMPT_SOURCE_FIELDS = (
    "training_caption", "caption_detailed", "caption_long", "long_caption",
    "caption_florence-2-large", "caption_internvl-3-8b",
    "caption_sharegpt4v-7b", "caption_gemini-2.5-flash-lite",
    "caption_gemini_2_5_flash_lite", "caption_original",
    "prompt", "caption", "text",
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


def _local_data_files(source: str) -> list[tuple[str, str]]:
    """Resolve local parquet/tar files, directories, or globs; [] for HF IDs."""
    suffix_to_kind = {
        ".parquet": "parquet",
        ".tar": "tar",
    }

    def supported(path: Path) -> bool:
        return path.suffix.lower() in suffix_to_kind

    def as_pair(path: Path) -> tuple[str, str]:
        return suffix_to_kind[path.suffix.lower()], str(path.resolve())

    expanded = os.path.expandvars(os.path.expanduser(source)).replace("\\", "/")
    if len(expanded) >= 3 and expanded[1:3] == ":/":
        expanded = f"/mnt/{expanded[0].lower()}/{expanded[3:]}"
    elif expanded.startswith("mnt/"):
        expanded = f"/{expanded}"
    path = Path(expanded)
    if path.is_file():
        if not supported(path):
            raise ValueError(f"Unsupported local data file: {source}")
        return [as_pair(path)]
    if path.is_dir():
        files = sorted(as_pair(item) for item in path.rglob("*") if item.is_file() and supported(item))
        if not files:
            raise FileNotFoundError(f"No supported parquet/tar files found under {source}")
        return files
    if glob.has_magic(expanded):
        files = sorted(as_pair(Path(item)) for item in glob.glob(expanded, recursive=True)
                       if Path(item).is_file() and supported(Path(item)))
        if not files:
            raise FileNotFoundError(f"No Parquet files match {source}")
        return files
    if (expanded.startswith(("/", "./", "../"))
            or expanded.lower().endswith((".parquet", ".tar"))):
        raise FileNotFoundError(f"Local data source not found: {source}")
    return []


def _local_parquet_files(source: str) -> list[str]:
    """Backward-compatible helper for callers/tests that only want parquet."""
    return [path for kind, path in _local_data_files(source) if kind == "parquet"]


class LocalTarImageJsonDataset:
    """Stream WebDataset-style local tar shards while ignoring heavy embeddings."""

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, tar_files: Sequence[str], shuffle: bool = False,
                 seed: int = 0):
        self.tar_files = list(tar_files)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    @staticmethod
    def _sample_key(name: str) -> str:
        return name.rsplit(".", 1)[0]

    def __iter__(self):
        tar_files = list(self.tar_files)
        if self.shuffle and len(tar_files) > 1:
            random.Random(self.seed).shuffle(tar_files)
        for tar_path in tar_files:
            pending = {}
            with tarfile.open(tar_path) as handle:
                for member in handle:
                    if not member.isfile():
                        continue
                    suffix = Path(member.name).suffix.lower()
                    if suffix != ".json" and suffix not in self.IMAGE_SUFFIXES:
                        continue
                    key = self._sample_key(member.name)
                    extracted = handle.extractfile(member)
                    if extracted is None:
                        continue
                    payload = extracted.read()
                    sample = pending.setdefault(key, {"__key__": key, "__url__": tar_path})
                    if suffix == ".json":
                        try:
                            sample["json"] = json.loads(payload.decode("utf-8"))
                        except json.JSONDecodeError as exc:
                            print(f"WARNING: skipping malformed json {member.name}: {exc}")
                            pending.pop(key, None)
                            continue
                    elif suffix in self.IMAGE_SUFFIXES:
                        sample[suffix.lstrip(".")] = payload
                    if sample.get("json") is not None and any(
                            image_key in sample for image_key in ["jpg", "jpeg", "png", "webp"]):
                        yield sample
                        pending.pop(key, None)


class RoundRobinSources:
    """Yield each configured source once without requiring matching schemas."""

    def __init__(self, sources, seed: int):
        self.sources = list(sources)
        self.seed = int(seed)

    def __iter__(self):
        sources = list(self.sources)
        random.Random(self.seed).shuffle(sources)
        active = []
        for item in sources:
            if len(item) == 2:
                dataset, source_root = item
                prompt_source_fields = None
                prompt_source_mode = None
            elif len(item) == 4:
                dataset, source_root, prompt_source_fields, prompt_source_mode = item
            else:
                raise ValueError(
                    "RoundRobinSources entries must be (dataset, source_root) "
                    "or (dataset, source_root, prompt_source_fields, "
                    "prompt_source_mode)")
            active.append([
                iter(dataset), source_root, prompt_source_fields,
                prompt_source_mode])
        while active:
            remaining = []
            for iterator, source_root, prompt_source_fields, prompt_source_mode in active:
                try:
                    sample = dict(next(iterator))
                except StopIteration:
                    continue
                sample.setdefault("_source_root", source_root)
                if prompt_source_fields is not None:
                    sample.setdefault("_prompt_source_fields", prompt_source_fields)
                if prompt_source_mode is not None:
                    sample.setdefault("_prompt_source_mode", prompt_source_mode)
                yield sample
                remaining.append([
                    iterator, source_root, prompt_source_fields, prompt_source_mode])
            active = remaining


class StreamingSDDataset(IterableDataset):
    """Normalize heterogeneous image-caption samples into bucketed tensors."""

    def __init__(self, ds_iter, max_samples: int = 2000, buckets=BUCKETS,
                 max_image_dimension: int = 1024,
                 prompt_source_fields: Sequence[str] | None = None,
                 prompt_source_mode: str = "random",
                 prompt_source_seed: int = 0):
        self.ds_iter = ds_iter
        self.max_samples = int(max_samples)
        self.buckets = tuple((int(w), int(h)) for w, h in buckets)
        self.max_image_dimension = int(max_image_dimension)
        self.prompt_source_fields = tuple(
            prompt_source_fields or DEFAULT_PROMPT_SOURCE_FIELDS)
        self.prompt_source_mode = str(prompt_source_mode)
        self.prompt_source_seed = int(prompt_source_seed)
        if self.prompt_source_mode not in {"random", "first"}:
            raise ValueError("prompt_source_mode must be 'random' or 'first'")

    def __iter__(self):
        prompt_rng = random.Random(self.prompt_source_seed)

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

            def values(keys):
                found = []
                seen = set()
                for source in (sample, meta):
                    for key in keys:
                        candidate = source.get(key)
                        if candidate is None:
                            continue
                        if isinstance(candidate, float) and math.isnan(candidate):
                            continue
                        if isinstance(candidate, (list, tuple)):
                            raw_values = candidate
                        else:
                            raw_values = [candidate]
                        for raw in raw_values:
                            text = str(raw).strip()
                            if not text or text.lower() in {"nan", "none", "null"}:
                                continue
                            if text not in seen:
                                found.append(text)
                                seen.add(text)
                return found

            source_prompt_fields = tuple(
                sample.get("_prompt_source_fields") or self.prompt_source_fields)
            source_prompt_mode = str(
                sample.get("_prompt_source_mode") or self.prompt_source_mode)
            if source_prompt_mode not in {"random", "first"}:
                raise ValueError(
                    "per-source prompt_source_mode must be 'random' or 'first'")

            prompt_candidates = values(source_prompt_fields)
            if prompt_candidates:
                if source_prompt_mode == "random":
                    long_caption = prompt_rng.choice(prompt_candidates)
                else:
                    long_caption = prompt_candidates[0]
            else:
                long_caption = ""

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


def _bucket_key(bucket) -> str:
    return f"{int(bucket[0])}x{int(bucket[1])}"


def _format_bucket_counts(counts: dict, buckets) -> str:
    ordered = {_bucket_key(bucket): int(counts.get(tuple(bucket), 0)) for bucket in buckets}
    extras = sorted(
        (tuple(bucket), int(value)) for bucket, value in counts.items()
        if tuple(bucket) not in {tuple(configured) for configured in buckets}
    )
    for bucket, value in extras:
        ordered[_bucket_key(bucket)] = value
    return json.dumps(ordered, sort_keys=False)


class BucketBatchDataset(IterableDataset):
    """Collect streaming samples into shape-compatible aspect-ratio batches."""

    def __init__(self, dataset: IterableDataset, batch_size: int,
                 drop_last: bool = True, buckets=BUCKETS,
                 log_prefix: str = "DataLoader"):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.buckets = tuple((int(w), int(h)) for w, h in buckets)
        self.log_prefix = str(log_prefix)

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
        seen_counts = {bucket: 0 for bucket in self.buckets}
        emitted_batches = {bucket: 0 for bucket in self.buckets}
        emitted_samples = {bucket: 0 for bucket in self.buckets}
        completed = False
        try:
            for sample in self.dataset:
                bucket = tuple(sample["bucket"])
                seen_counts[bucket] = seen_counts.get(bucket, 0) + 1
                buffer = buffers.setdefault(bucket, [])
                buffer.append(sample)
                if len(buffer) == self.batch_size:
                    emitted_batches[bucket] = emitted_batches.get(bucket, 0) + 1
                    emitted_samples[bucket] = emitted_samples.get(bucket, 0) + len(buffer)
                    batch = self._collate(buffer)
                    buffers[bucket] = []
                    yield batch
            completed = True
            if not self.drop_last:
                for bucket, buffer in buffers.items():
                    if buffer:
                        emitted_batches[bucket] = emitted_batches.get(bucket, 0) + 1
                        emitted_samples[bucket] = emitted_samples.get(bucket, 0) + len(buffer)
                        yield self._collate(buffer)
        finally:
            dropped_tail = {
                tuple(bucket): len(buffer)
                for bucket, buffer in buffers.items()
                if buffer and self.drop_last
            }
            print(
                f"{self.log_prefix} bucket_sample_counts="
                f"{_format_bucket_counts(seen_counts, self.buckets)} "
                f"bucket_emitted_batches="
                f"{_format_bucket_counts(emitted_batches, self.buckets)} "
                f"bucket_emitted_samples="
                f"{_format_bucket_counts(emitted_samples, self.buckets)} "
                f"bucket_dropped_tail_samples="
                f"{_format_bucket_counts(dropped_tail, self.buckets)} "
                f"total_seen_samples={sum(seen_counts.values())} "
                f"total_emitted_samples={sum(emitted_samples.values())} "
                f"total_dropped_tail_samples={sum(dropped_tail.values())} "
                f"completed={completed}"
            )


def _normalize_data_source(
    source: Any,
    default_prompt_source_fields: Sequence[str] | None,
    default_prompt_source_mode: str,
) -> tuple[str, Sequence[str] | None, str]:
    if isinstance(source, dict):
        path = source.get("path") or source.get("source") or source.get("uri")
        if not path:
            raise ValueError(
                "data source dict must contain 'path', 'source', or 'uri'")
        prompt_source_fields = source.get(
            "prompt_source_fields", default_prompt_source_fields)
        prompt_source_mode = source.get(
            "prompt_source_mode", default_prompt_source_mode)
        if prompt_source_fields is not None:
            prompt_source_fields = tuple(prompt_source_fields)
        prompt_source_mode = str(prompt_source_mode)
        if prompt_source_mode not in {"random", "first"}:
            raise ValueError(
                "data source prompt_source_mode must be 'random' or 'first'")
        return str(path), prompt_source_fields, prompt_source_mode
    return str(source), default_prompt_source_fields, default_prompt_source_mode


def make_streaming_dataloader(
    data_sources: Sequence[Any] | str,
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
    prompt_source_fields: Sequence[str] | None = None,
    prompt_source_mode: str = "random",
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
    normalized_sources = []
    for source in data_sources:
        source_path, source_prompt_fields, source_prompt_mode = _normalize_data_source(
            source, prompt_source_fields, prompt_source_mode)
        normalized_sources.append({
            "path": source_path,
            "prompt_source_mode": source_prompt_mode,
            "prompt_source_fields": list(
                source_prompt_fields or DEFAULT_PROMPT_SOURCE_FIELDS),
        })
        local_files = _local_data_files(source_path)
        if local_files:
            source_specs = []
            for kind, local_file in local_files:
                if kind == "parquet":
                    source_specs.append((
                        "parquet", {"train": [local_file]},
                        str(Path(local_file).parent), None,
                        source_prompt_fields, source_prompt_mode))
                elif kind == "tar":
                    source_specs.append((
                        "local_tar", None, str(Path(local_file).parent),
                        local_file, source_prompt_fields, source_prompt_mode))
                else:
                    raise ValueError(f"Unsupported local dataset kind: {kind}")
        else:
            source_specs = [(
                source_path, None, "", None,
                source_prompt_fields, source_prompt_mode)]
        for (dataset_name, data_files, source_root, local_file,
             source_prompt_fields, source_prompt_mode) in source_specs:
            if dataset_name == "local_tar":
                if local_file is None:
                    raise RuntimeError("local_tar dataset spec is missing a tar file")
                dataset = LocalTarImageJsonDataset(
                    [local_file], shuffle=shuffle,
                    seed=shuffle_seed + len(loaded_sources))
            else:
                kwargs = {"split": "train", "streaming": True}
                if data_files is not None:
                    kwargs["data_files"] = data_files
                dataset = load_dataset(dataset_name, **kwargs)
                if shuffle:
                    dataset = dataset.shuffle(
                        buffer_size=shuffle_buffer,
                        seed=shuffle_seed + len(loaded_sources),
                    )
            loaded_sources.append((
                dataset, source_root, source_prompt_fields, source_prompt_mode))

    ds_full = RoundRobinSources(loaded_sources, seed=shuffle_seed)
    samples = StreamingSDDataset(
        ds_full, max_samples=max_samples, buckets=buckets,
        max_image_dimension=max_image_dimension,
        prompt_source_fields=prompt_source_fields,
        prompt_source_mode=prompt_source_mode,
        prompt_source_seed=shuffle_seed)
    log_prefix = (
        f"DataLoader phase={phase} epoch={epoch}"
    )
    ds = BucketBatchDataset(
        samples, batch_size=batch_size, drop_last=drop_last, buckets=buckets,
        log_prefix=log_prefix)
    dl = DataLoader(ds, batch_size=None, num_workers=0)
    print(
        f"DataLoader phase={phase} epoch={epoch} max_samples={max_samples} "
        f"batch={batch_size} buckets={list(map(tuple, buckets))} "
        f"max_image_dimension={max_image_dimension} sources={normalized_sources} "
        f"drop_last={drop_last} shuffle={shuffle} seed={shuffle_seed} "
        f"prompt_source_mode={prompt_source_mode} "
        f"prompt_source_fields={list(prompt_source_fields or DEFAULT_PROMPT_SOURCE_FIELDS)}"
    )
    return dl
