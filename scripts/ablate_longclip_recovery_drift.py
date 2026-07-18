#!/usr/bin/env python3
"""Ablate short-prompt drift between sparse SaRA and LongCLIP sidecars."""
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

from pure_ella.config import seed_everything
from pure_ella.longclip_sara import (
    LongClipEncoder,
    LongClipSaraConfig,
    install_longclip_sara_sidecars,
    validate_longclip_sara_checkpoint,
)
from pure_ella.sara import (
    build_sara_attn2_kv_sparse_masks,
    collect_sara_sparse_values,
    load_sara_sparse_values,
)
from scripts.train_longclip_sara import (
    force_include_initial_sara_masks,
    load_initial_sara_patch,
    relative_prediction_rms,
    short_prompt_prediction_signature,
)
from train import (
    TrainingState,
    load_stylejourney,
    resolve_autocast_dtype,
    resolve_model_weight_dtype,
)


def cloned_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("candidate")
    parser.add_argument("output_json")
    args = parser.parse_args()

    cfg = LongClipSaraConfig.from_json(args.config)
    seed_everything(cfg.base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = TrainingState(
        cfg=cfg,
        device=device,
        unet_dtype=resolve_model_weight_dtype(cfg),
        autocast_dtype=resolve_autocast_dtype(cfg, device),
    )
    load_stylejourney(state)
    encoder = LongClipEncoder(
        cfg.longclip_repo,
        cfg.longclip_checkpoint,
        device,
        state.unet_dtype,
        cfg.fail_on_prompt_truncation,
    )
    initial_sparse_values = load_initial_sara_patch(state, cfg, encoder)
    summary = build_sara_attn2_kv_sparse_masks(
        state.unet,
        selection_mode=cfg.sara_selection_mode,
        target_fraction=cfg.sara_target_fraction,
        threshold=cfg.sara_threshold,
        min_sparse_fraction=cfg.sara_min_sparse_fraction,
        max_sparse_fraction_warn=cfg.sara_max_sparse_fraction_warn,
        max_sparse_fraction_abort=cfg.sara_max_sparse_fraction_abort,
        target_substrings=cfg.sara_target_substrings,
    )
    force_include_initial_sara_masks(state.unet, summary, initial_sparse_values)
    baseline_sparse = collect_sara_sparse_values(state.unet)

    install_longclip_sara_sidecars(state.unet, cfg)
    baseline_resolution = cloned_state_dict(state.unet.class_embedding)
    baseline_p4 = cloned_state_dict(state.unet.p4_blocks)

    candidate_path = Path(args.candidate)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    validate_longclip_sara_checkpoint(candidate, cfg, encoder, str(candidate_path))

    def restore_baseline() -> None:
        load_sara_sparse_values(state.unet, baseline_sparse)
        state.unet.class_embedding.load_state_dict(baseline_resolution, strict=True)
        state.unet.p4_blocks.load_state_dict(baseline_p4, strict=True)

    def load_candidate_sparse() -> None:
        load_sara_sparse_values(state.unet, candidate["sparse_values"])

    def load_candidate_resolution() -> None:
        state.unet.class_embedding.load_state_dict(
            candidate["resolution_conditioner_state_dict"], strict=True)

    def load_candidate_p4() -> None:
        state.unet.p4_blocks.load_state_dict(candidate["p4_state_dict"], strict=True)

    state.unet.eval()
    restore_baseline()
    baseline = short_prompt_prediction_signature(state, cfg, encoder)
    signatures: dict[str, torch.Tensor] = {"baseline": baseline}

    restore_baseline()
    load_candidate_sparse()
    signatures["sparse_only"] = short_prompt_prediction_signature(state, cfg, encoder)

    restore_baseline()
    load_candidate_resolution()
    signatures["resolution_only"] = short_prompt_prediction_signature(state, cfg, encoder)

    restore_baseline()
    load_candidate_p4()
    signatures["p4_only"] = short_prompt_prediction_signature(state, cfg, encoder)

    restore_baseline()
    load_candidate_resolution()
    load_candidate_p4()
    signatures["sidecars_only"] = short_prompt_prediction_signature(state, cfg, encoder)

    restore_baseline()
    load_candidate_sparse()
    load_candidate_resolution()
    load_candidate_p4()
    signatures["combined"] = short_prompt_prediction_signature(state, cfg, encoder)

    drifts = {
        name: relative_prediction_rms(signature, baseline)
        for name, signature in signatures.items()
        if name != "baseline"
    }
    result = {
        "config": str(Path(args.config)),
        "candidate": str(candidate_path),
        "optimizer_steps": candidate["completion"]["optimizer_steps"],
        "short_prompt_limit": cfg.short_prompt_max_relative_rms,
        "relative_rms_vs_inherited_baseline": drifts,
        "candidate_recorded_combined_relative_rms": candidate["validation_gate"][
            "short_prompt_relative_rms"
        ],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
