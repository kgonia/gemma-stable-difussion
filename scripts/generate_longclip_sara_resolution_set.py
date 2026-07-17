#!/usr/bin/env python3
"""Generate fixed LongCLIP+SaRA samples at requested resolutions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from pure_ella.config import TrainConfig, seed_everything
from pure_ella.diagnostics import generate_clip_teacher, save_validation_grid
from pure_ella.longclip_sara import (
    LongClipEncoder,
    LongClipSaraConfig,
    load_longclip_sara_checkpoint,
)
from scripts.generate_requested_ella_samples import SAMPLES, make_scheduler
from train import (
    TrainingState,
    load_stylejourney,
    resolve_autocast_dtype,
    resolve_model_weight_dtype,
)

EXTRA_COMPLEX = {
    "name": "red_motorcycle_cottage_owl",
    "prompt": (
        "a cinematic photo of a red vintage motorcycle parked beside a stone "
        "cottage, with a brass telescope on the seat, blue wildflowers in the "
        "basket, and a tiny owl perched on the handlebar at sunrise"
    ),
    "negative_prompt": "",
    "steps": 30,
    "guidance": 5.5,
    "sampler": "dpmpp_sde_karras",
    "seed": 777,
}

RESOLUTIONS = ((1024, 1024), (768, 1024), (768, 768))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default="config_longclip_sara_3x_3epoch_from_5pct.json")
    parser.add_argument("--native-config", default="config_clip_gemma_residual_p1.json")
    parser.add_argument("--sara-patch", default="output_longclip_sara_3x_3epoch_from_5pct/longclip_sara_unet_attn2_kv_sparse.pt")
    parser.add_argument("--output-dir", default="output_longclip_sara_3x_3epoch_from_5pct_resolution_samples")
    args = parser.parse_args()

    patch_path = Path(args.sara_patch)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    native_cfg = TrainConfig.from_json(args.native_config)
    train_cfg = LongClipSaraConfig.from_json(args.train_config)
    seed_everything(train_cfg.base_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = TrainingState(
        cfg=native_cfg,
        device=device,
        unet_dtype=resolve_model_weight_dtype(native_cfg),
        autocast_dtype=resolve_autocast_dtype(native_cfg, device),
    )
    load_stylejourney(state)
    encoder = LongClipEncoder(
        train_cfg.longclip_repo,
        train_cfg.longclip_checkpoint,
        device,
        state.unet_dtype,
        fail_on_truncation=True,
    )
    state.encode_clip = lambda prompts: encoder.encode(prompts, track_truncation=False)

    patch = torch.load(patch_path, map_location="cpu", weights_only=True)
    patch_cfg = LongClipSaraConfig(**patch["run_config"])
    loaded = load_longclip_sara_checkpoint(
        state.unet, patch, patch_cfg, encoder, str(patch_path))
    print(
        "Applied LongCLIP SaRA patch: "
        f"{loaded['sparse_values']:,} sparse values, "
        f"sidecars={loaded['sidecars']}")

    base_cases = [
        {
            "name": sample.name,
            "prompt": sample.prompt,
            "negative_prompt": sample.negative_prompt,
            "steps": sample.steps,
            "guidance": sample.guidance,
            "sampler": sample.sampler,
            "seed": sample.seed,
        }
        for sample in SAMPLES
    ]
    cases = base_cases + [EXTRA_COMPLEX]
    if len(cases) != 5:
        raise RuntimeError(f"Expected 5 cases, got {len(cases)}")

    records = []
    for width, height in RESOLUTIONS:
        images = []
        labels = []
        for index, case in enumerate(cases, 1):
            state.inf_scheduler = make_scheduler(case["sampler"], state.scheduler)
            image = generate_clip_teacher(
                case["prompt"],
                state,
                steps=case["steps"],
                guidance=case["guidance"],
                seed=case["seed"],
                negative_prompt=case["negative_prompt"],
                height=height,
                width=width,
            )
            filename = f"{width}x{height}_{index:02d}_{case['name']}.png"
            path = output_dir / filename
            image.save(path)
            record = {
                "path": str(path),
                "width": width,
                "height": height,
                "name": case["name"],
                "prompt": case["prompt"],
                "negative_prompt": case["negative_prompt"],
                "steps": case["steps"],
                "guidance": case["guidance"],
                "sampler": case["sampler"],
                "seed": case["seed"],
                "bytes": path.stat().st_size,
            }
            records.append(record)
            images.append(image)
            labels.append(f"{case['name']} | {width}x{height}")
            print(f"Saved {path}")
        grid_path = output_dir / f"grid_{width}x{height}.png"
        save_validation_grid(
            images,
            labels,
            str(grid_path),
            f"LongCLIP+SaRA 15pct/3epoch complex prompts {width}x{height}",
            cols=1,
        )
        print(f"Saved grid {grid_path}")

    manifest = {
        "train_config": args.train_config,
        "native_config": args.native_config,
        "sara_patch": str(patch_path),
        "sara_patch_bytes": patch_path.stat().st_size,
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved manifest {manifest_path}")


if __name__ == "__main__":
    main()
