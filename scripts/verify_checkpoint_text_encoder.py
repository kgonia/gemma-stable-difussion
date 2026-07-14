#!/usr/bin/env python3
"""Offline provenance check for a single-file SD checkpoint text encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import CLIPTextModel


def digest_state_dict(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--clip-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--output", default="checkpoint_text_encoder_provenance.json")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    prefixes = (
        "cond_stage_model.transformer.",
        "conditioner.embedders.0.transformer.",
        "text_encoder.",
    )
    # local_files_only makes this a provenance check, not an accidental download.
    reference_encoder = CLIPTextModel.from_pretrained(args.clip_id, local_files_only=True)
    reference_state = reference_encoder.state_dict()

    def compatible_clip_key(raw_name: str) -> str:
        candidates = [raw_name]
        if raw_name.startswith("text_model."):
            candidates.append(raw_name.removeprefix("text_model."))
        else:
            candidates.append(f"text_model.{raw_name}")
        return next((name for name in candidates if name in reference_state), raw_name)

    checkpoint_state = {}
    ignored_checkpoint_buffers = []
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            prefix = next((value for value in prefixes if key.startswith(value)), None)
            if prefix is None:
                continue
            name = compatible_clip_key(key[len(prefix):])
            # SD checkpoints can carry this generated-position buffer while
            # the reference model omits it. It is not a learned tensor and
            # must not make stock CLIP look modified.
            if name.endswith("embeddings.position_ids") and name not in reference_state:
                ignored_checkpoint_buffers.append(name)
                continue
            checkpoint_state[name] = handle.get_tensor(key)
    if not checkpoint_state:
        raise RuntimeError("Checkpoint contains no recognized CLIP text-encoder tensors")
    common = sorted(set(checkpoint_state) & set(reference_state))
    unmatched = sorted(set(checkpoint_state) - set(reference_state))
    changed = [name for name in common if not torch.equal(checkpoint_state[name].cpu(), reference_state[name].cpu())]
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "clip_id": args.clip_id,
        "checkpoint_tensor_count": len(checkpoint_state), "reference_tensor_count": len(reference_state),
        "common_tensor_count": len(common), "different_tensor_count": len(changed),
        "ignored_checkpoint_buffers": sorted(ignored_checkpoint_buffers),
        "unmatched_checkpoint_tensors": unmatched[:20],
        "identical_to_openai_clip": len(common) == len(checkpoint_state) == len(reference_state) and not changed,
        "checkpoint_state_sha256": digest_state_dict(checkpoint_state),
        "reference_state_sha256": digest_state_dict(reference_state),
        "first_different_tensors": changed[:20],
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
