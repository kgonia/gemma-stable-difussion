#!/usr/bin/env python3
"""Compare P0/P1/CLIP using the full user-supplied generation settings."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train import (
    TrainingState,
    load_clip,
    load_gemma,
    load_stylejourney,
    make_encode_clip,
    make_encode_gemma,
    resolve_autocast_dtype,
    resolve_model_weight_dtype,
)
from pure_ella.config import TrainConfig, seed_everything
from pure_ella.diagnostics import (
    generate_clip_teacher,
    generate_ella,
    save_validation_grid,
)
from scripts.compare_phase0_connectors import load_connector
from scripts.generate_requested_ella_samples import SAMPLES, make_scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_unsplash_all_p1.json")
    parser.add_argument(
        "--phase0-config", default="config_phase0_ella_tsc_all_prompts.json")
    parser.add_argument(
        "--phase0-checkpoint",
        default=("output_phase0_ella_tsc_all_prompts/"
                 "pure_ella_connector_L77.pt"),
    )
    parser.add_argument(
        "--phase1-checkpoint",
        default="output_unsplash_all_p1/ella_connector_frozen_unet.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="output_p0_p1_extended_settings_comparison",
    )
    args = parser.parse_args()

    for checkpoint in (args.phase0_checkpoint, args.phase1_checkpoint):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Connector checkpoint not found: {checkpoint}")

    cfg = TrainConfig.from_json(args.config)
    cfg.enable_unet_gradient_checkpointing = False
    cfg.camera_conditioning_enabled = False
    seed_everything(cfg.val_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = TrainingState(
        cfg=cfg,
        device=device,
        unet_dtype=resolve_model_weight_dtype(cfg),
        autocast_dtype=resolve_autocast_dtype(cfg, device),
    )
    load_gemma(state)
    load_clip(state)
    load_stylejourney(state)
    state.encode_gemma = make_encode_gemma(state)
    state.encode_clip = make_encode_clip(state)
    connectors = {
        "Phase 0": load_connector(
            args.phase0_config, args.phase0_checkpoint, state),
        "Phase 1": load_connector(
            args.config, args.phase1_checkpoint, state),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {**{label: [] for label in connectors}, "CLIP": []}
    for sample in SAMPLES:
        for label, connector in connectors.items():
            state.connector = connector
            state.inf_scheduler = make_scheduler(sample.sampler, state.scheduler)
            image = generate_ella(
                sample.prompt,
                state,
                steps=sample.steps,
                guidance=sample.guidance,
                seed=sample.seed,
                negative_prompt=sample.negative_prompt,
                width=sample.width,
                height=sample.height,
            )
            image.save(
                output_dir
                / f"{sample.name}_{label.lower().replace(' ', '_')}.png"
            )
            generated[label].append(image)

        state.inf_scheduler = make_scheduler(sample.sampler, state.scheduler)
        clip_image = generate_clip_teacher(
            sample.prompt,
            state,
            steps=sample.steps,
            guidance=sample.guidance,
            seed=sample.seed,
            negative_prompt=sample.negative_prompt,
            width=sample.width,
            height=sample.height,
        )
        clip_image.save(output_dir / f"{sample.name}_clip.png")
        generated["CLIP"].append(clip_image)

    images = []
    labels = []
    columns = (*connectors.keys(), "CLIP")
    for sample_index, sample in enumerate(SAMPLES):
        for label in columns:
            images.append(generated[label][sample_index])
            labels.append(
                f"{label} | {sample.name} | {sample.sampler} | "
                f"{sample.steps} steps | CFG {sample.guidance:g} | "
                f"seed {sample.seed} | {sample.width}x{sample.height}"
            )
    grid_path = output_dir / "extended_settings_p0_p1_clip.png"
    save_validation_grid(
        images,
        labels,
        str(grid_path),
        "Full generation settings: Phase 0 vs Phase 1 vs CLIP",
        cols=len(columns),
    )
    print(f"Completed {grid_path.resolve()}")


if __name__ == "__main__":
    main()
