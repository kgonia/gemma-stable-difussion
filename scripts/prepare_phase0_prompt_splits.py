#!/usr/bin/env python3
"""Create deterministic train/validation splits for small Phase 0 prompt sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_prompts(path: Path, prompts: list[str]) -> None:
    path.write_text("".join(f"{prompt}\n" for prompt in prompts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=21234)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "validation_size_per_source": args.validation_size,
        "sources": [],
    }
    for source_index, source in enumerate(args.sources):
        prompts = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(prompts) <= args.validation_size:
            raise ValueError(
                f"{source} has {len(prompts)} prompts; need more than "
                f"validation-size={args.validation_size}"
            )
        validation_indices = set(
            random.Random(args.seed + source_index).sample(
                range(len(prompts)), args.validation_size
            )
        )
        train = [
            prompt for index, prompt in enumerate(prompts)
            if index not in validation_indices
        ]
        validation = [
            prompt for index, prompt in enumerate(prompts)
            if index in validation_indices
        ]
        train_path = args.output_dir / f"{source.stem}.train.txt"
        validation_path = args.output_dir / f"{source.stem}.validation.txt"
        _write_prompts(train_path, train)
        _write_prompts(validation_path, validation)
        manifest["sources"].append({
            "source": str(source),
            "source_sha256": _sha256(source),
            "total": len(prompts),
            "train": len(train),
            "validation": len(validation),
            "train_path": str(train_path),
            "train_sha256": _sha256(train_path),
            "validation_path": str(validation_path),
            "validation_sha256": _sha256(validation_path),
        })

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
