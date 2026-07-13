"""Deterministic prompt-only loading for CLIP-alignment pretraining."""
from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import DataLoader, IterableDataset


def _expand_prompt_sources(sources: Sequence[str] | str) -> list[Path]:
    if isinstance(sources, str):
        sources = [sources]
    paths: list[Path] = []
    for source in sources:
        expanded = os.path.expandvars(os.path.expanduser(source)).replace(
            "\\", "/")
        if len(expanded) >= 3 and expanded[1:3] == ":/":
            expanded = f"/mnt/{expanded[0].lower()}/{expanded[3:]}"
        elif expanded.startswith("mnt/"):
            expanded = f"/{expanded}"
        candidate = Path(expanded)
        if candidate.is_file():
            matches = [candidate]
        elif candidate.is_dir():
            matches = sorted(candidate.rglob("*.txt"))
        elif glob.has_magic(expanded):
            matches = sorted(Path(item) for item in glob.glob(expanded))
        else:
            raise FileNotFoundError(f"Prompt source not found: {source}")
        if not matches:
            raise FileNotFoundError(f"No prompt text files found for: {source}")
        for match in matches:
            if not match.is_file() or match.suffix.lower() != ".txt":
                raise ValueError(f"Prompt sources must be .txt files: {match}")
            paths.append(match.resolve())
    if not paths:
        raise ValueError("At least one prompt source is required")
    return paths


def _nonempty_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            prompt = line.strip()
            if prompt:
                yield prompt


def _round_robin_sources(paths: list[Path], seed: int) -> Iterable[str]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    active = [iter(_nonempty_lines(path)) for path in shuffled]
    while active:
        remaining = []
        for iterator in active:
            try:
                yield next(iterator)
            except StopIteration:
                continue
            remaining.append(iterator)
        active = remaining


def _buffered_shuffle(
    prompts: Iterable[str], buffer_size: int, seed: int,
) -> Iterable[str]:
    rng = random.Random(seed)
    iterator = iter(prompts)
    buffer = []
    for _ in range(max(1, int(buffer_size))):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    for prompt in iterator:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = prompt
    rng.shuffle(buffer)
    yield from buffer


class PromptTextDataset(IterableDataset):
    """Stream nonempty lines with deterministic bounded-memory shuffling."""

    def __init__(
        self,
        sources: Sequence[str] | str,
        max_samples: int,
        shuffle: bool,
        shuffle_buffer: int,
        seed: int,
    ):
        self.paths = _expand_prompt_sources(sources)
        self.max_samples = int(max_samples)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)

    def __iter__(self):
        prompts = _round_robin_sources(self.paths, self.seed)
        if self.shuffle:
            prompts = _buffered_shuffle(
                prompts, self.shuffle_buffer, self.seed)
        for index, prompt in enumerate(prompts):
            if index >= self.max_samples:
                break
            yield prompt


def make_prompt_dataloader(
    sources: Sequence[str] | str,
    epoch: int,
    max_samples: int,
    batch_size: int,
    shuffle: bool,
    shuffle_buffer: int,
    base_seed: int,
) -> DataLoader:
    seed = int(base_seed) + 10_000 + int(epoch)
    dataset = PromptTextDataset(
        sources=sources,
        max_samples=max_samples,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        drop_last=False,
        num_workers=0,
    )
    print(
        f"PromptDataLoader epoch={epoch} max_samples={max_samples} "
        f"batch={batch_size} sources={list(map(str, dataset.paths))} "
        f"shuffle={shuffle} buffer={shuffle_buffer} seed={seed}"
    )
    return loader


def sample_validation_prompts(
    sources: Sequence[str] | str,
    samples_per_source: int,
    seed: int,
) -> dict[str, list[str]]:
    """Reservoir-sample a stable validation set independently per file."""
    limit = int(samples_per_source)
    if limit <= 0:
        raise ValueError("samples_per_source must be positive")
    result: dict[str, list[str]] = {}
    for source_index, path in enumerate(_expand_prompt_sources(sources)):
        rng = random.Random(int(seed) + source_index)
        reservoir: list[str] = []
        for index, prompt in enumerate(_nonempty_lines(path)):
            if index < limit:
                reservoir.append(prompt)
                continue
            replacement = rng.randint(0, index)
            if replacement < limit:
                reservoir[replacement] = prompt
        if not reservoir:
            raise ValueError(f"Validation prompt source is empty: {path}")
        label = path.stem
        if label in result:
            label = f"{label}_{source_index}"
        result[label] = reservoir
    return result
