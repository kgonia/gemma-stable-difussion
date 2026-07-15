#!/usr/bin/env python3
"""Train sparse SD U-Net SaRA weights under frozen direct LongCLIP-L context.

This is intentionally separate from the Gemma connector pipeline.  There is
no connector, no Gemma model, and no 77-token adapter: the U-Net receives the
native 248x768 LongCLIP context directly.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pure_ella.config import seed_everything
from pure_ella.dataset import make_streaming_dataloader
from pure_ella.diagnostics import (
    generate_clip_teacher, save_validation_grid, scheduler_for_generation_case,
)
from pure_ella.longclip_sara import (
    LongClipEncoder, LongClipSaraConfig, longclip_sara_schema,
)
from pure_ella.sara import (
    build_sara_attn2_kv_sparse_masks, capture_sara_selected_values,
    collect_sara_sparse_values, install_sara_gradient_masks,
    remove_sara_gradient_masks, sara_selected_delta_metrics,
)
from train import (
    TrainingState, _as_prompt_list, camera_conditioned_unet, masked_mse,
    model_autocast, resolve_autocast_dtype, resolve_model_weight_dtype,
    load_stylejourney, select_training_captions,
)


def lr_scheduler(optimizer, cfg):
    if not cfg.lr_decay_steps:
        return None
    def scale(step):
        if step < cfg.lr_warmup_steps:
            return float(step + 1) / max(1, cfg.lr_warmup_steps)
        return max(0.0, 1.0 - (step - cfg.lr_warmup_steps) / max(
            1, cfg.lr_decay_steps - cfg.lr_warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def save_visual_check(state, cfg):
    """Save the four canonical cases under the trained LongCLIP+SaRA model."""
    if not cfg.run_final_visual_check or not cfg.complex_generation_cases:
        return
    images, labels = [], []
    old_scheduler = state.inf_scheduler
    try:
        for case in cfg.complex_generation_cases:
            requested = scheduler_for_generation_case(case, state.scheduler)
            if requested is not None:
                state.inf_scheduler = requested
            image = generate_clip_teacher(
                case["prompt"], state,
                steps=case.get("steps", cfg.val_steps),
                guidance=case.get("guidance", cfg.val_guidance),
                seed=case.get("seed", cfg.val_seed),
                negative_prompt=case.get("negative_prompt", ""),
                height=case.get("height", 512), width=case.get("width", 512),
            )
            images.append(image)
            labels.append(
                f"LongCLIP+SaRA | {case['name']} | "
                f"{case.get('sampler', 'default')} | "
                f"{case.get('steps', cfg.val_steps)} steps")
    finally:
        state.inf_scheduler = old_scheduler
    save_validation_grid(
        images, labels, f"{cfg.output_dir}/longclip_sara_complex_prompts.png",
        "LongCLIP-L direct conditioning + sparse SaRA", cols=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="LongCLIP SaRA JSON config")
    args = parser.parse_args()
    cfg = LongClipSaraConfig.from_json(args.config)
    seed_everything(cfg.base_seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = TrainingState(
        cfg=cfg, device=device,
        unet_dtype=resolve_model_weight_dtype(cfg),
        autocast_dtype=resolve_autocast_dtype(cfg, device),
    )
    load_stylejourney(state)
    encoder = LongClipEncoder(
        cfg.longclip_repo, cfg.longclip_checkpoint, device, state.unet_dtype,
        cfg.fail_on_prompt_truncation)
    state.encode_clip = encoder.encode

    # Smoke-test the exact condition shape accepted by SD cross-attention.
    with torch.no_grad():
        context, _ = encoder.encode(["a detailed red bicycle beside a blue vase"])
        latent = torch.randn(1, 4, 8, 8, device=device, dtype=state.unet_dtype)
        timestep = torch.tensor([500], device=device, dtype=torch.long)
        with model_autocast(state):
            output = state.unet(latent, timestep, encoder_hidden_states=context).sample
        if not torch.isfinite(output).all():
            raise RuntimeError("LongCLIP U-Net smoke forward emitted NaN/Inf")
    print(f"LongCLIP direct U-Net smoke PASS: context={tuple(context.shape)}")

    sara_summary = build_sara_attn2_kv_sparse_masks(
        state.unet, selection_mode=cfg.sara_selection_mode,
        target_fraction=cfg.sara_target_fraction, threshold=cfg.sara_threshold,
        min_sparse_fraction=cfg.sara_min_sparse_fraction,
        max_sparse_fraction_warn=cfg.sara_max_sparse_fraction_warn,
        max_sparse_fraction_abort=cfg.sara_max_sparse_fraction_abort,
        target_substrings=cfg.sara_target_substrings)
    baseline = capture_sara_selected_values(state.unet)
    hooks = install_sara_gradient_masks(state.unet)
    trainable = [p for p in state.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.sara_lr,
                                  weight_decay=0.0, eps=1e-6)
    scheduler = lr_scheduler(optimizer, cfg)
    state.unet.train()
    state.vae.eval()

    step = 0
    accumulation = 0
    started = time.time()
    try:
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(cfg.epochs):
            loader = make_streaming_dataloader(
                cfg.data_sources, phase=40, epoch=epoch, max_samples=cfg.max_samples,
                batch_size=cfg.train_batch_size, shuffle=cfg.shuffle_streaming,
                shuffle_buffer=cfg.shuffle_buffer, base_seed=cfg.base_seed,
                buckets=cfg.aspect_ratio_buckets,
                drop_last=cfg.drop_last_bucket_batches,
                max_image_dimension=cfg.max_image_dimension)
            for batch in tqdm(loader, desc=f"LongCLIP+SaRA {epoch + 1}/{cfg.epochs}"):
                captions = select_training_captions(batch, state)
                image = batch["image"].to(device=device, dtype=state.unet_dtype)
                image_mask = batch.get("image_mask")
                if image_mask is not None:
                    image_mask = image_mask.to(device=device)
                with torch.no_grad():
                    with model_autocast(state):
                        latent = state.vae.encode(image).latent_dist.sample()
                        latent = latent * state.vae.config.scaling_factor
                    noise = torch.randn_like(latent)
                    timestep = torch.randint(
                        0, state.scheduler.config.num_train_timesteps,
                        (len(captions),), device=device).long()
                    noisy = state.scheduler.add_noise(latent, noise, timestep)
                    context, _ = encoder.encode(captions)
                with model_autocast(state):
                    prediction = camera_conditioned_unet(
                        state.unet, noisy, timestep,
                        encoder_hidden_states=context.to(dtype=state.unet_dtype),
                    ).sample
                    loss = masked_mse(prediction, noise, image_mask)
                if not torch.isfinite(loss):
                    raise RuntimeError("LongCLIP SaRA loss is NaN/Inf")
                (loss / cfg.gradient_accumulation_steps).backward()
                accumulation += 1
                if accumulation < cfg.gradient_accumulation_steps:
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
                step += 1
                if step % 25 == 0 or step == 1:
                    print(f"longclip_sara step {step}: loss={loss.item():.6f} "
                          f"lr={optimizer.param_groups[0]['lr']:.3g}")
                if step >= cfg.max_opt_steps:
                    break
            if step >= cfg.max_opt_steps:
                break
    finally:
        remove_sara_gradient_masks(hooks)

    if step <= 0:
        raise RuntimeError("LongCLIP SaRA completed zero optimizer steps")
    delta = sara_selected_delta_metrics(state.unet, baseline)
    artifact = {
        "longclip_sara_schema": longclip_sara_schema(cfg, encoder),
        "sparse_values": collect_sara_sparse_values(state.unet),
        "sara_sparse_summary": sara_summary,
        "sara_selected_delta": delta,
        "completion": {"completed": True, "optimizer_steps": step},
        "run_config": cfg.to_dict(),
    }
    path = Path(cfg.output_dir) / "longclip_sara_unet_attn2_kv_sparse.pt"
    torch.save(artifact, path)
    print(f"LongCLIP SaRA patch saved: {path}")
    print(f"SaRA selected-weight delta: {delta}")
    print(f"Completed {step} updates in {(time.time() - started) / 60:.1f} min; "
          f"max LongCLIP tokens observed={encoder.max_observed_tokens}; "
          f"truncated training prompts={encoder.truncated_prompt_count}")
    save_visual_check(state, cfg)


if __name__ == "__main__":
    main()
