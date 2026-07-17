#!/usr/bin/env python3
"""Fixed-seed visual comparison: StyleJourney native CLIP vs LongCLIP-L.

LongCLIP is used as the paper intends: a direct 248-token replacement for
CLIP's per-token context.  The StyleJourney U-Net and VAE remain unchanged.
The script deliberately does not touch connector training or checkpoints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow direct invocation from the repository root without PYTHONPATH=.
if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from pure_ella.config import TrainConfig, seed_everything
from pure_ella.diagnostics import generate_clip_teacher, save_validation_grid
from pure_ella.longclip_sara import (
    LongClipEncoder, LongClipSaraConfig, load_longclip_sara_checkpoint,
)
from train import (
    TrainingState,
    load_clip,
    load_stylejourney,
    make_encode_clip,
    resolve_autocast_dtype,
    resolve_model_weight_dtype,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_clip_gemma_residual_p1.json")
    parser.add_argument("--longclip-repo", type=Path, required=True)
    parser.add_argument("--longclip-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sara-patch", type=Path,
        help="Optional completed longclip_sara_unet_attn2_kv_sparse.pt patch. "
             "Its exact LongCLIP/StyleJourney provenance is verified before use.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output_stylejourney_longclip"))
    parser.add_argument(
        "--requested-four", action="store_true",
        help="Use the four established complex visual-judgment prompts, "
             "including their original seeds, negative prompts, samplers, "
             "and resolutions.",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config)
    if not args.requested_four and not cfg.long_eval_prompts:
        raise ValueError("Config has no long_eval_prompts")
    seed_everything(cfg.base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = TrainingState(
        cfg=cfg,
        device=device,
        unet_dtype=resolve_model_weight_dtype(cfg),
        autocast_dtype=resolve_autocast_dtype(cfg, device),
    )
    load_stylejourney(state)
    # The native comparison must use StyleJourney's own text encoder, not
    # stock OpenAI CLIP. load_clip applies the verified checkpoint loader.
    load_clip(state)
    native_encode = make_encode_clip(state)
    longclip = LongClipEncoder(
        args.longclip_repo, args.longclip_checkpoint, device, state.unet_dtype,
        fail_on_truncation=True)
    longclip_encode = longclip.encode
    if args.sara_patch is not None:
        patch = torch.load(args.sara_patch, map_location="cpu", weights_only=True)
        patch_cfg = LongClipSaraConfig(**patch["run_config"])
        if Path(cfg.sd_checkpoint).expanduser().resolve() != Path(
                patch_cfg.sd_checkpoint).expanduser().resolve():
            raise RuntimeError(
                "LongCLIP SaRA patch was trained against a different "
                "StyleJourney checkpoint")
        loaded = load_longclip_sara_checkpoint(
            state.unet, patch, patch_cfg, longclip, str(args.sara_patch))
        print(
            "Applied LongCLIP SaRA patch: "
            f"{loaded['sparse_values']:,} sparse values, "
            f"sidecars={loaded['sidecars']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.requested_four:
        from scripts.generate_requested_ella_samples import SAMPLES, make_scheduler
        cases = [
            {
                "name": sample.name,
                "prompt": sample.prompt,
                "negative_prompt": sample.negative_prompt,
                "seed": sample.seed,
                "steps": sample.steps,
                "guidance": sample.guidance,
                "height": sample.height,
                "width": sample.width,
                "scheduler": sample.sampler,
            }
            for sample in SAMPLES
        ]
    else:
        base_seed = cfg.val_seed if args.seed is None else args.seed
        cases = [
            {
                "name": f"long_{index + 1:02d}",
                "prompt": prompt,
                "negative_prompt": "",
                "seed": base_seed + index,
                "steps": args.steps,
                "guidance": args.guidance,
                "height": args.height,
                "width": args.width,
                "scheduler": None,
            }
            for index, prompt in enumerate(cfg.long_eval_prompts)
        ]

    images, labels = [], []
    for case in cases:
        if case["scheduler"] is not None:
            state.inf_scheduler = make_scheduler(case["scheduler"], state.scheduler)
        state.encode_clip = native_encode
        native = generate_clip_teacher(
            case["prompt"], state, steps=case["steps"], guidance=case["guidance"],
            seed=case["seed"], negative_prompt=case["negative_prompt"],
            height=case["height"], width=case["width"])
        native.save(args.output_dir / f"{case['name']}_stylejourney_native.png")
        if case["scheduler"] is not None:
            state.inf_scheduler = make_scheduler(case["scheduler"], state.scheduler)
        state.encode_clip = longclip_encode
        longclip = generate_clip_teacher(
            case["prompt"], state, steps=case["steps"], guidance=case["guidance"],
            seed=case["seed"], negative_prompt=case["negative_prompt"],
            height=case["height"], width=case["width"])
        longclip.save(args.output_dir / f"{case['name']}_longclip.png")
        images.extend([native, longclip])
        labels.extend(["StyleJourney native CLIP (77)", "LongCLIP-L (248)"])

    grid_name = (
        "stylejourney_native_vs_longclip_requested_four.png"
        if args.requested_four else "stylejourney_native_vs_longclip.png"
    )
    if args.sara_patch is not None:
        grid_name = grid_name.removesuffix(".png") + "_sara.png"
    grid_path = args.output_dir / grid_name
    save_validation_grid(
        images, labels, str(grid_path),
        "StyleJourney: native CLIP vs LongCLIP-L (matched seeds)", cols=2)
    print(f"Saved comparison grid: {grid_path}")


if __name__ == "__main__":
    main()
