#!/usr/bin/env python3
"""Create a deterministic, group-safe Parquet holdout for residual P1.

Rows connected by photo ID, exact content hash, perceptual hash, or an
explicit source identity stay in the same split.  Run this before generating
any caption variants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


IDENTITY_COLUMNS = (
    "photo_id", "id", "image_id", "source_id", "source_identity",
    "sha256", "image_sha256", "exact_hash", "phash", "image_phash",
)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be in (0, 1)")

    import pyarrow.parquet as pq
    table = pq.read_table(args.source)
    columns = set(table.column_names)
    usable = [name for name in IDENTITY_COLUMNS if name in columns]
    if not usable:
        raise ValueError(
            "No identity columns found. Refusing an unsafe row-level split; "
            f"expected one of {IDENTITY_COLUMNS}, got {sorted(columns)}")
    rows = table.num_rows
    values = {name: table[name].to_pylist() for name in usable}
    groups, uf = {}, UnionFind(rows)
    for name in usable:
        for index, value in enumerate(values[name]):
            if value is None or not str(value).strip():
                continue
            key = (name, str(value).strip())
            if key in groups:
                uf.union(index, groups[key])
            else:
                groups[key] = index
    components = defaultdict(list)
    for index in range(rows):
        components[uf.find(index)].append(index)

    validation = set()
    for root, members in components.items():
        digest = hashlib.sha256(f"{args.seed}:{root}".encode()).digest()
        if int.from_bytes(digest[:8], "big") / 2**64 < args.validation_fraction:
            validation.update(members)
    if not validation or len(validation) == rows:
        raise RuntimeError("Degenerate split; choose a different seed/fraction")
    train_indices = [index for index in range(rows) if index not in validation]
    validation_indices = sorted(validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    validation_path = args.output_dir / "validation.parquet"
    pq.write_table(table.take(train_indices), train_path)
    pq.write_table(table.take(validation_indices), validation_path)
    manifest = {
        "version": 1, "source": str(args.source.resolve()), "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "identity_columns": usable, "rows": rows,
        "identity_groups": len(components), "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "train": str(train_path.resolve()),
        "validation": str(validation_path.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
