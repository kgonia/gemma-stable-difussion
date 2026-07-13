#!/usr/bin/env python3
"""Generate a fixed-seed ELLA/TRM/CLIP comparison grid."""
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
from pure_ella.connector import build_connector
from pure_ella.diagnostics import (
    generate_clip_teacher,
    generate_ella,
    save_validation_grid,
)


PROMPTS = (
    "a cat sitting on a windowsill looking outside",
    "a watercolor painting of a mountain lake",
    "a neon-lit cyberpunk alleyway at night",
    "a rainy city storefront with a bright sign reading LAB CABIN X9",
    "a product photograph of hiking boots with orange laces, a folded trail "
    "map, a silver compass, and raindrops on the leather",
    "a cinematic photo of a red vintage motorcycle beside a stone cottage, "
    "with a brass telescope on the seat and blue wildflowers in the basket",
)


def load_connector(config_path: str, checkpoint_path: str, state):
    cfg = TrainConfig.from_json(config_path)
    connector = build_connector(
        cfg.connector_type,
        gemma_dim=state.gemma_hidden_size,
        width=cfg.connector_width,
        context_tokens=cfg.context_tokens,
        anchor_tokens=cfg.clip_anchor_tokens,
        layers=cfg.connector_layers,
        heads=cfg.connector_heads,
        ff_mult=cfg.connector_ff_mult,
        dropout=cfg.connector_dropout,
        time_embed_dim=cfg.connector_time_embed_dim,
        extra_gate_init=cfg.connector_extra_gate_init,
        gemma_layer_mix_count=cfg.gemma_layer_mix_count,
        recursive_y_steps=cfg.recursive_y_steps,
        recursive_y_gate_init=cfg.recursive_y_gate_init,
        trm_outer_steps=cfg.trm_outer_steps,
        trm_inner_steps=cfg.trm_inner_steps,
        trm_scratch_tokens=cfg.trm_scratch_tokens,
        trm_y_gate_init=cfg.trm_y_gate_init,
        trm_z_gate_init=cfg.trm_z_gate_init,
    ).to(device=state.device, dtype=state.unet_dtype).eval()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    connector.load_state_dict(
        checkpoint["connector_state_dict"], strict=True)
    print(
        f"Loaded {cfg.connector_type} checkpoint strictly: {checkpoint_path}")
    return connector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config", default="config_unsplash_short_p1.json")
    parser.add_argument(
        "--ella-config",
        default="config_phase0_scaled_ella_tsc_b64_200k.json")
    parser.add_argument(
        "--ella-checkpoint",
        default=("output_phase0_scaled_ella_tsc_b64_200k/"
                 "pure_ella_connector_L77.pt"))
    parser.add_argument(
        "--trm-config",
        default="config_phase0_scaled_trm_yz_b64_200k.json")
    parser.add_argument(
        "--trm-checkpoint",
        default=("output_phase0_scaled_trm_yz_b64_200k/"
                 "pure_ella_connector_L77.pt"))
    parser.add_argument("--ella-label", default="ELLA-TSC")
    parser.add_argument("--trm-label", default="TRM-YZ")
    parser.add_argument("--title", default="Phase 0 connector comparison")
    parser.add_argument(
        "--output",
        default=("output_phase0_scaled_comparison/"
                 "ella_trm_clip_comparison.png"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.base_config)
    cfg.run_long_context_diagnostics = False
    cfg.run_suffix_counterfactual_grids = False
    cfg.camera_conditioning_enabled = False
    seed_everything(args.seed)
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
        args.ella_label: load_connector(
            args.ella_config, args.ella_checkpoint, state),
        args.trm_label: load_connector(
            args.trm_config, args.trm_checkpoint, state),
    }
    generated = {name: [] for name in connectors}
    generated["CLIP"] = []
    for prompt_index, prompt in enumerate(PROMPTS):
        seed = args.seed + prompt_index
        for name, connector in connectors.items():
            state.connector = connector
            generated[name].append(generate_ella(
                prompt, state, steps=args.steps, guidance=args.guidance,
                seed=seed,
            ))
        generated["CLIP"].append(generate_clip_teacher(
            prompt, state, steps=args.steps, guidance=args.guidance,
            seed=seed,
        ))

    images = []
    labels = []
    for prompt, row_index in zip(PROMPTS, range(len(PROMPTS))):
        for name in (args.ella_label, args.trm_label, "CLIP"):
            images.append(generated[name][row_index])
            labels.append(f"{name}: {prompt}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_validation_grid(
        images, labels, str(output),
        args.title,
    )
    print(f"Comparison grid complete: {output.resolve()}")


if __name__ == "__main__":
    main()
