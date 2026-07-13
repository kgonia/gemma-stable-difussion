#!/usr/bin/env python3
"""Generate fixed-seed Phase 0, Phase 1, and CLIP comparison grids."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    seed: int
    steps: int = 30
    guidance: float = 5.5
    width: int = 512
    height: int = 512


def _standard_cases(cfg: TrainConfig) -> list[Case]:
    return [
        Case(f"standard_{index + 1}", prompt, cfg.val_seed + index)
        for index, prompt in enumerate(cfg.val_prompts)
    ]


def _extended_cases(cfg: TrainConfig) -> list[Case]:
    cases = [
        Case(
            str(item["name"]),
            str(item["prompt"]),
            int(item.get("seed", cfg.complex_prompt_seed)),
            steps=int(item.get("steps", cfg.complex_prompt_steps)),
            guidance=float(item.get("guidance", cfg.complex_prompt_guidance)),
            width=int(item.get("width", cfg.complex_prompt_width)),
            height=int(item.get("height", cfg.complex_prompt_height)),
        )
        for item in cfg.complex_generation_cases
    ]
    cases.extend(
        Case(f"long_{index + 1}", prompt, cfg.val_seed + 100 + index)
        for index, prompt in enumerate(cfg.long_eval_prompts)
    )
    return cases


def _generate_suite(
    name: str,
    cases: list[Case],
    connectors: dict[str, torch.nn.Module],
    state: TrainingState,
    output_dir: Path,
) -> None:
    generated: dict[str, list] = {
        **{label: [] for label in connectors},
        "CLIP": [],
    }
    suite_dir = output_dir / name
    suite_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for label, connector in connectors.items():
            state.connector = connector
            image = generate_ella(
                case.prompt,
                state,
                steps=case.steps,
                guidance=case.guidance,
                seed=case.seed,
                height=case.height,
                width=case.width,
            )
            image.save(suite_dir / f"{case.name}_{label.lower().replace(' ', '_')}.png")
            generated[label].append(image)
        clip_image = generate_clip_teacher(
            case.prompt,
            state,
            steps=case.steps,
            guidance=case.guidance,
            seed=case.seed,
            height=case.height,
            width=case.width,
        )
        clip_image.save(suite_dir / f"{case.name}_clip.png")
        generated["CLIP"].append(clip_image)

    images = []
    labels = []
    column_order = (*connectors.keys(), "CLIP")
    for case_index, case in enumerate(cases):
        for label in column_order:
            images.append(generated[label][case_index])
            labels.append(f"{label} | {case.name} | {case.prompt}")
    grid_path = output_dir / f"{name}_p0_p1_clip.png"
    save_validation_grid(
        images,
        labels,
        str(grid_path),
        f"{name.title()} prompts: Phase 0 vs Phase 1 vs CLIP",
        cols=len(column_order),
    )
    print(f"Completed {grid_path.resolve()}")


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
        "--output-dir", default="output_p0_p1_post_training_comparison")
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
    _generate_suite(
        "standard", _standard_cases(cfg), connectors, state, output_dir)
    _generate_suite(
        "extended", _extended_cases(cfg), connectors, state, output_dir)


if __name__ == "__main__":
    main()
