#!/usr/bin/env python3
"""Strictly validate and package a Phase 0 best-validation checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pure_ella.config import TrainConfig
from pure_ella.connector import build_connector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    connector = build_connector(
        cfg.connector_type,
        gemma_dim=640,
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
    )
    connector.load_state_dict(
        checkpoint["connector_state_dict"], strict=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": cfg.connector_type,
        "connector_type": cfg.connector_type,
        "connector_state_dict": checkpoint["connector_state_dict"],
        "camera_conditioner_state_dict": None,
        "camera_condition_schema": None,
        "gemma_model_id": cfg.gemma_id,
        "sd_checkpoint": cfg.sd_checkpoint,
        "run_config": cfg.to_dict(),
        "source_stage": checkpoint.get("stage"),
        "validation_step": checkpoint.get("validation_step"),
        "validation_metrics": checkpoint.get("validation_metrics"),
    }, output)
    print(
        f"Strict connector validation PASS: {cfg.connector_type} -> {output}")


if __name__ == "__main__":
    main()
