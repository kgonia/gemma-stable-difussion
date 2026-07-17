#!/usr/bin/env python3
"""Train sparse SD U-Net SaRA weights under frozen direct LongCLIP-L context.

This is intentionally separate from the Gemma connector pipeline.  There is
no connector, no Gemma model, and no 77-token adapter: the U-Net receives the
native 248x768 LongCLIP context directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# ``python scripts/train_longclip_sara.py ...`` sets sys.path[0] to scripts/.
# Make the repository package importable without requiring a PYTHONPATH tweak.
if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from pure_ella.config import seed_everything
from pure_ella.dataset import make_streaming_dataloader
from pure_ella.diagnostics import (
    generate_clip_teacher, save_validation_grid, scheduler_for_generation_case,
)
from pure_ella.longclip_sara import (
    LongClipEncoder, LongClipSaraConfig, install_longclip_sara_sidecars,
    load_longclip_sara_checkpoint, longclip_sara_schema,
)
from pure_ella.p4 import p4_state_dict, p4_telemetry
from pure_ella.resolution import make_resolution_condition_from_bucket
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


def resolution_condition_for_batch(
    batch: dict, latent: torch.Tensor, cfg, *, training: bool = True,
) -> torch.Tensor | None:
    if not cfg.resolution_conditioning_enabled:
        return None
    return make_resolution_condition_from_bucket(
        batch["bucket"], batch_size=latent.shape[0], device=latent.device)


def install_longclip_sidecars(state, cfg) -> dict:
    """Install optional resolution/P4 sidecars after SaRA freezes base params."""
    summary = install_longclip_sara_sidecars(state.unet, cfg)
    if summary.get("resolution_conditioning") is not None:
        state.resolution_conditioner = state.unet.class_embedding
        print(
            "Resolution conditioner PASS: "
            f"output={state.unet.class_embedding.output_dim} "
            f"hidden={state.unet.class_embedding.hidden_dim} "
            "zero-output identity=PASS")
    if summary.get("p4") is not None:
        print(
            "P4 blocks installed: "
            f"variant={cfg.p4_variant} sites={cfg.p4_insertions} "
            "external_gate_init=0")
    return summary


def sidecar_identity_smoke(
    state, cfg, baseline_output: torch.Tensor, context: torch.Tensor,
    latent: torch.Tensor, timestep: torch.Tensor,
) -> dict:
    """Verify sidecars preserve the exact U-Net output at step zero."""
    if (not cfg.p4_smoke_identity_check
            or (not cfg.p4_enabled and not cfg.resolution_conditioning_enabled)):
        return {}
    with torch.no_grad():
        with model_autocast(state):
            output = camera_conditioned_unet(
                state.unet, latent, timestep,
                encoder_hidden_states=context.to(dtype=state.unet_dtype),
            ).sample
    max_abs = float((output - baseline_output).detach().abs().max().cpu())
    if max_abs != 0.0:
        raise RuntimeError(
            "Sidecar step-0 identity smoke failed: "
            f"max_abs_diff={max_abs:.8g}")
    print("Sidecar step-0 identity smoke PASS: max_abs_diff=0")
    return {"step0_identity_max_abs_diff": max_abs}


def load_initial_sara_patch(state, cfg, encoder) -> dict | None:
    """Load a completed previous LongCLIP SaRA patch before continuing."""
    if not cfg.initial_sara_patch:
        return None
    path = Path(cfg.initial_sara_patch).expanduser()
    patch = torch.load(path, map_location="cpu", weights_only=True)
    patch_cfg = LongClipSaraConfig(**patch["run_config"])
    expected = longclip_sara_schema(cfg, encoder)
    actual = patch.get("longclip_sara_schema")
    comparable_keys = (
        "version", "conditioning_backend", "longclip", "sd_checkpoint",
        "sd_checkpoint_sha256", "sara_target_substrings",
        "sara_selection_mode",
    )
    if not isinstance(actual, dict) or any(
            actual.get(key) != expected.get(key) for key in comparable_keys):
        raise RuntimeError(
            f"Initial LongCLIP SaRA patch provenance mismatch: {path}")
    if patch_cfg.resolution_conditioning_enabled or patch_cfg.p4_enabled:
        expected_sidecars = longclip_sara_schema(cfg, encoder)
        patch_sidecars = longclip_sara_schema(patch_cfg, encoder)
        for key in ("resolution_conditioning", "p4"):
            if patch_sidecars.get(key) != expected_sidecars.get(key):
                raise RuntimeError(
                    f"Initial LongCLIP SaRA sidecar schema mismatch for {key}: {path}")
    loaded = load_longclip_sara_checkpoint(
        state.unet, patch, patch_cfg, encoder, str(path),
        load_sparse=True, load_sidecars=False)
    sparse_values = patch["sparse_values"]
    state.initial_longclip_sara_checkpoint = patch
    state.initial_longclip_sara_patch_cfg = patch_cfg
    state.initial_longclip_sara_patch_path = str(path)
    print(
        f"Initial LongCLIP SaRA endpoint loaded from {path}: "
        f"{loaded['sparse_values']:,} sparse values")
    return sparse_values


def load_initial_sidecar_state_after_install(state, encoder) -> dict[str, int] | None:
    patch = getattr(state, "initial_longclip_sara_checkpoint", None)
    patch_cfg = getattr(state, "initial_longclip_sara_patch_cfg", None)
    source = getattr(state, "initial_longclip_sara_patch_path", "initial_sara_patch")
    if patch is None or patch_cfg is None:
        return None
    if not (patch_cfg.resolution_conditioning_enabled or patch_cfg.p4_enabled):
        return None
    loaded = load_longclip_sara_checkpoint(
        state.unet, patch, patch_cfg, encoder, source,
        load_sparse=False, load_sidecars=True)
    print(f"Initial LongCLIP SaRA sidecars loaded from {source}: {loaded['sidecars']}")
    return loaded["sidecars"]


def force_include_initial_sara_masks(unet, summary: dict, initial_sparse_values: dict | None) -> dict:
    """Keep previous endpoint values in the new sparse mask without growing it."""
    if not initial_sparse_values:
        return summary
    named = dict(unet.named_parameters())
    missing_refs = []
    removable_refs = []
    for name, pack in initial_sparse_values.items():
        p = named[name]
        current = getattr(p, "_sara_sparse_mask", None)
        if current is None:
            raise RuntimeError(f"No new SaRA mask for initial patch parameter {name}")
        previous = pack["mask"].to(device=p.device, dtype=torch.bool)
        missing = previous & ~current
        if missing.any():
            missing_refs.append((name, missing))
        removable = current & ~previous
        if removable.any():
            magnitudes = p.detach().abs()[removable].float().cpu()
            removable_refs.extend(
                (float(value), name, int(index))
                for value, index in zip(magnitudes, torch.nonzero(removable.reshape(-1), as_tuple=False)[:, 0].cpu())
            )
    missing_count = sum(int(mask.sum().item()) for _, mask in missing_refs)
    if not missing_count:
        print("Initial SaRA mask already fully contained in new sparse selection")
        return summary
    if len(removable_refs) < missing_count:
        raise RuntimeError(
            "Cannot force-include initial SaRA mask while preserving sparse count")
    removable_refs.sort(reverse=True)
    by_name = {name: getattr(p, "_sara_sparse_mask", None) for name, p in named.items()}
    for _, name, flat_index in removable_refs[:missing_count]:
        mask = by_name[name]
        if mask is None:
            raise RuntimeError(f"No new SaRA mask for removable entry {name}")
        mask.reshape(-1)[flat_index] = False
    for name, missing in missing_refs:
        mask = by_name[name]
        if mask is None:
            raise RuntimeError(f"No new SaRA mask for initial entry {name}")
        mask.logical_or_(missing)

    rows = []
    selected = 0
    total_target = 0
    total_unet = sum(p.numel() for p in unet.parameters())
    for row in summary["rows"]:
        p = named[row["name"]]
        mask = getattr(p, "_sara_sparse_mask")
        count = int(mask.sum().item())
        total = int(p.numel())
        selected += count
        total_target += total
        p.requires_grad_(count > 0)
        rows.append({**row, "selected": count, "fraction": count / max(total, 1)})
    updated = {
        **summary,
        "selected": selected,
        "total_target": total_target,
        "fraction": selected / max(total_target, 1),
        "total_unet": total_unet,
        "target_scope_fraction": total_target / max(total_unet, 1),
        "whole_unet_fraction": selected / max(total_unet, 1),
        "rows": rows,
        "initial_sara_forced_included": missing_count,
    }
    print(
        "Initial SaRA endpoint mask forced into expanded selection: "
        f"{missing_count:,} entries swapped; selected={selected:,} "
        f"({updated['fraction']:.4%})")
    return updated


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


def validation_skip_reason(cfg) -> str | None:
    if not cfg.validation_data_sources:
        return "no validation_data_sources"
    if cfg.validation_max_samples == 0:
        return "validation_max_samples=0"
    return None


def deterministic_validation_captions(batch: dict) -> list[str]:
    """Use the same long caption on every validation pass."""
    return _as_prompt_list(batch["caption"])


def fixed_validation_timesteps(total: int, num_train_timesteps: int,
                               device: torch.device) -> torch.Tensor:
    """Cover the full diffusion trajectory with a stable sample assignment."""
    if total <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.linspace(
        0, num_train_timesteps - 1, steps=total,
        device=device, dtype=torch.float64).round().long()


def masked_per_sample_mse(prediction: torch.Tensor, target: torch.Tensor,
                          image_mask: torch.Tensor | None) -> torch.Tensor:
    """Return one padding-aware MSE per image, independent of batch grouping."""
    error = (prediction.float() - target.float()).square()
    if image_mask is None:
        return error.flatten(1).mean(1)
    mask = torch.nn.functional.interpolate(
        image_mask.float(), size=prediction.shape[-2:], mode="nearest"
    ).to(device=prediction.device, dtype=error.dtype)
    return (error * mask).flatten(1).sum(1) / (
        mask.flatten(1).sum(1).clamp_min(1.0) * prediction.shape[1])


def bucket_keys_for_batch(batch: dict, batch_size: int) -> list[str]:
    bucket = batch.get("bucket")
    if bucket is None:
        return ["unknown"] * batch_size
    if isinstance(bucket, torch.Tensor):
        value = bucket.detach().cpu()
        if value.ndim == 1 and value.numel() >= 2:
            width, height = int(value[0]), int(value[1])
            return [f"{width}x{height}"] * batch_size
        rows = value.reshape(-1, value.shape[-1])
        keys = [f"{int(row[0])}x{int(row[1])}" for row in rows[:batch_size]]
        return keys + [keys[-1] if keys else "unknown"] * (batch_size - len(keys))
    values = list(bucket)
    if len(values) >= 2 and all(isinstance(v, (int, float)) for v in values[:2]):
        return [f"{int(values[0])}x{int(values[1])}"] * batch_size
    keys = [f"{int(pair[0])}x{int(pair[1])}" for pair in values[:batch_size]]
    return keys + [keys[-1] if keys else "unknown"] * (batch_size - len(keys))


@torch.no_grad()
def heldout_diffusion_loss(state, cfg, encoder) -> float | None:
    """Compute masked denoising MSE on the configured held-out parquet split."""
    if validation_skip_reason(cfg) is not None:
        return None
    was_training = state.unet.training
    state.unet.eval()
    per_sample_losses = []
    bucket_sums: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    sample_count = 0
    generator = torch.Generator(device=state.device).manual_seed(cfg.val_seed)
    timestep_grid = fixed_validation_timesteps(
        cfg.validation_max_samples,
        state.scheduler.config.num_train_timesteps,
        state.device)
    cuda_devices = []
    if state.device.type == "cuda":
        cuda_devices = [
            state.device.index
            if state.device.index is not None else torch.cuda.current_device()
        ]
    try:
        # Restore CPU/CUDA global RNG even if a future validation component
        # accidentally draws from it.  All intended randomness uses the local
        # generator above.
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            loader = make_streaming_dataloader(
                cfg.validation_data_sources, phase=41, epoch=0,
                max_samples=cfg.validation_max_samples,
                batch_size=cfg.train_batch_size, shuffle=False, shuffle_buffer=1,
                base_seed=cfg.base_seed,
                buckets=cfg.aspect_ratio_buckets,
                drop_last=False, max_image_dimension=cfg.max_image_dimension,
                prompt_source_fields=(
                    cfg.validation_prompt_source_fields or cfg.prompt_source_fields),
                prompt_source_mode=cfg.validation_prompt_source_mode)
            for batch in loader:
                captions = deterministic_validation_captions(batch)
                batch_size = len(captions)
                image = batch["image"].to(
                    device=state.device, dtype=state.unet_dtype)
                image_mask = batch.get("image_mask")
                if image_mask is not None:
                    image_mask = image_mask.to(device=state.device)
                with model_autocast(state):
                    latent = state.vae.encode(image).latent_dist.mode()
                    latent = latent * state.vae.config.scaling_factor
                noise = torch.randn(
                    latent.shape, generator=generator, device=state.device,
                    dtype=latent.dtype)
                timestep = timestep_grid[
                    sample_count:sample_count + batch_size]
                if len(timestep) != batch_size:
                    raise RuntimeError(
                        "Held-out loader exceeded validation_max_samples")
                noisy = state.scheduler.add_noise(latent, noise, timestep)
                resolution_condition = resolution_condition_for_batch(
                    batch, latent, cfg, training=False)
                context, _ = encoder.encode(
                    captions, track_truncation=False)
                with model_autocast(state):
                    prediction = camera_conditioned_unet(
                        state.unet, noisy, timestep,
                        encoder_hidden_states=context.to(dtype=state.unet_dtype),
                        resolution_condition=resolution_condition,
                    ).sample
                    losses = masked_per_sample_mse(
                        prediction, noise, image_mask).detach().cpu().tolist()
                    per_sample_losses.extend(losses)
                for key, loss_value in zip(
                        bucket_keys_for_batch(batch, batch_size), losses):
                    bucket_sums[key] = bucket_sums.get(key, 0.0) + float(loss_value)
                    bucket_counts[key] = bucket_counts.get(key, 0) + 1
                sample_count += batch_size
    finally:
        state.unet.train(was_training)
    if not sample_count:
        raise RuntimeError("LongCLIP held-out validation received zero batches")
    if sample_count != cfg.validation_max_samples:
        raise RuntimeError(
            "LongCLIP held-out validation expected "
            f"{cfg.validation_max_samples} samples but received {sample_count}; "
            "refusing a timestep-skewed metric")
    global_loss = float(sum(per_sample_losses) / sample_count)
    bucket_summary = {
        key: {
            "count": bucket_counts[key],
            "mean_loss": bucket_sums[key] / max(bucket_counts[key], 1),
        }
        for key in sorted(bucket_counts)
    }
    state.longclip_validation_bucket_summary = {
        "global_loss": global_loss,
        "sample_count": sample_count,
        "buckets": bucket_summary,
    }
    print("LongCLIP held-out per-bucket losses: "
          f"{json.dumps(bucket_summary, sort_keys=True)}")
    return global_loss


@torch.no_grad()
def short_prompt_prediction_signature(state, cfg, encoder) -> torch.Tensor:
    """Deterministic U-Net predictions used to gate short-prompt drift."""
    was_training = state.unet.training
    state.unet.eval()
    generator = torch.Generator(device=state.device).manual_seed(cfg.val_seed + 1)
    prompts = list(cfg.val_prompts)
    timesteps = fixed_validation_timesteps(
        len(prompts), state.scheduler.config.num_train_timesteps, state.device)
    latents = torch.randn(
        (len(prompts), 4, 64, 64), generator=generator,
        device=state.device, dtype=state.unet_dtype)
    try:
        context, _ = encoder.encode(prompts, track_truncation=False)
        with model_autocast(state):
            prediction = camera_conditioned_unet(
                state.unet, latents, timesteps,
                encoder_hidden_states=context.to(dtype=state.unet_dtype),
            ).sample
        return prediction.detach().float().cpu()
    finally:
        state.unet.train(was_training)


def relative_prediction_rms(current: torch.Tensor,
                            baseline: torch.Tensor) -> float:
    delta_rms = (current - baseline).square().mean().sqrt()
    baseline_rms = baseline.square().mean().sqrt().clamp_min(1e-8)
    return float((delta_rms / baseline_rms).item())


@torch.no_grad()
def save_periodic_prompt_grid(state, cfg, opt_step: int):
    """Render fixed validation prompts at configured training intervals."""
    was_training = state.unet.training
    state.unet.eval()
    try:
        images = [generate_clip_teacher(
            prompt, state, steps=cfg.val_steps, guidance=cfg.val_guidance,
            seed=cfg.val_seed) for prompt in cfg.val_prompts]
    finally:
        state.unet.train(was_training)
    path = f"{cfg.output_dir}/longclip_sara_step_{opt_step:06d}.png"
    save_validation_grid(images, ["LongCLIP+SaRA"] * len(images), path,
                         f"LongCLIP+SaRA step {opt_step}")
    print(f"LongCLIP SaRA validation grid saved: {path}")


def initialize_wandb(cfg):
    if not cfg.wandb_enabled:
        return None
    import wandb
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        name=cfg.wandb_run_name or "longclip-l-stylejourney-sara",
        config=cfg.to_dict())
    return wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="LongCLIP SaRA JSON config")
    args = parser.parse_args()
    cfg = LongClipSaraConfig.from_json(args.config)
    seed_everything(cfg.base_seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    wandb = initialize_wandb(cfg)
    if cfg.validation_every_opt_steps:
        reason = validation_skip_reason(cfg)
        if reason is not None:
            print(f"LongCLIP held-out validation disabled: {reason}")
            if cfg.require_validation_gate:
                raise ValueError(
                    f"LongCLIP validation gate is required but disabled: {reason}")

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
    initial_sparse_values = load_initial_sara_patch(state, cfg, encoder)
    # Diagnostic/visual calls are evaluation, so they must never mutate the
    # training-only truncation counter.
    state.encode_clip = lambda prompts: encoder.encode(
        prompts, track_truncation=False)

    # Smoke-test the exact condition shape accepted by SD cross-attention.
    with torch.no_grad():
        context, _ = encoder.encode(["a detailed red bicycle beside a blue vase"])
        latent = torch.randn(1, 4, 8, 8, device=device, dtype=state.unet_dtype)
        timestep = torch.tensor([500], device=device, dtype=torch.long)
        with model_autocast(state):
            output = state.unet(latent, timestep, encoder_hidden_states=context).sample
        if not torch.isfinite(output).all():
            raise RuntimeError("LongCLIP U-Net smoke forward emitted NaN/Inf")
        smoke_context = context
        smoke_latent = latent
        smoke_timestep = timestep
        smoke_output = output.detach().clone()
    print(f"LongCLIP direct U-Net smoke PASS: context={tuple(context.shape)}")

    sara_summary = build_sara_attn2_kv_sparse_masks(
        state.unet, selection_mode=cfg.sara_selection_mode,
        target_fraction=cfg.sara_target_fraction, threshold=cfg.sara_threshold,
        min_sparse_fraction=cfg.sara_min_sparse_fraction,
        max_sparse_fraction_warn=cfg.sara_max_sparse_fraction_warn,
        max_sparse_fraction_abort=cfg.sara_max_sparse_fraction_abort,
        target_substrings=cfg.sara_target_substrings)
    sara_summary = force_include_initial_sara_masks(
        state.unet, sara_summary, initial_sparse_values)
    baseline = capture_sara_selected_values(state.unet)
    sidecar_summary = install_longclip_sidecars(state, cfg)
    initial_sidecar_loaded = load_initial_sidecar_state_after_install(state, encoder)
    if initial_sidecar_loaded is None:
        sidecar_smoke = sidecar_identity_smoke(
            state, cfg, smoke_output, smoke_context, smoke_latent, smoke_timestep)
    else:
        sidecar_smoke = {"skipped": "initial_sidecar_state_loaded"}
    hooks = install_sara_gradient_masks(state.unet)
    trainable = [p for p in state.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.sara_lr,
                                  weight_decay=0.0, eps=1e-6)
    scheduler = lr_scheduler(optimizer, cfg)
    state.unet.train()
    state.vae.eval()

    baseline_validation_loss = heldout_diffusion_loss(state, cfg, encoder)
    baseline_bucket_summary = (
        getattr(state, "longclip_validation_bucket_summary", None)
        if baseline_validation_loss is not None else None)
    if baseline_validation_loss is not None:
        print(
            "LongCLIP frozen-U-Net held-out baseline: "
            f"loss={baseline_validation_loss:.6f}")
        if wandb is not None:
            wandb.log({"longclip_sara_val/baseline_loss": baseline_validation_loss},
                      step=0)
    short_prompt_baseline = (
        short_prompt_prediction_signature(state, cfg, encoder)
        if cfg.require_short_prompt_regression_gate else None)

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
                max_image_dimension=cfg.max_image_dimension,
                prompt_source_fields=cfg.prompt_source_fields,
                prompt_source_mode=cfg.prompt_source_mode)
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
                    resolution_condition = resolution_condition_for_batch(
                        batch, latent, cfg, training=True)
                    context, _ = encoder.encode(captions)
                with model_autocast(state):
                    prediction = camera_conditioned_unet(
                        state.unet, noisy, timestep,
                        encoder_hidden_states=context.to(dtype=state.unet_dtype),
                        resolution_condition=resolution_condition,
                    ).sample
                    loss = masked_mse(prediction, noise, image_mask)
                if not torch.isfinite(loss):
                    raise RuntimeError("LongCLIP SaRA loss is NaN/Inf")
                (loss / cfg.gradient_accumulation_steps).backward()
                accumulation += 1
                if accumulation < cfg.gradient_accumulation_steps:
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
                p4_metrics_before_step = p4_telemetry(state.unet)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
                step += 1
                if step % 25 == 0 or step == 1:
                    print(f"longclip_sara step {step}: loss={loss.item():.6f} "
                          f"lr={optimizer.param_groups[0]['lr']:.3g}")
                    if p4_metrics_before_step:
                        print(f"longclip_sara p4 telemetry step {step}: {p4_metrics_before_step}")
                if wandb is not None:
                    payload = {
                        "longclip_sara/step": step,
                        "longclip_sara/loss": float(loss.item()),
                        "longclip_sara/lr": optimizer.param_groups[0]["lr"],
                    }
                    payload.update(p4_metrics_before_step)
                    wandb.log(payload, step=step)
                if (cfg.validation_every_opt_steps
                        and step % cfg.validation_every_opt_steps == 0):
                    val_loss = heldout_diffusion_loss(state, cfg, encoder)
                    if val_loss is not None:
                        print(f"longclip_sara validation step {step}: loss={val_loss:.6f}")
                        if wandb is not None:
                            wandb.log({"longclip_sara_val/loss": val_loss}, step=step)
                if (cfg.generation_grid_every_opt_steps
                        and step % cfg.generation_grid_every_opt_steps == 0):
                    save_periodic_prompt_grid(state, cfg, step)
                if step >= cfg.max_opt_steps:
                    break
            if step >= cfg.max_opt_steps:
                break
        if accumulation > 0 and step < cfg.max_opt_steps:
            scale = cfg.gradient_accumulation_steps / accumulation
            if scale != 1.0:
                for param in trainable:
                    if param.grad is not None:
                        param.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
            p4_metrics_before_step = p4_telemetry(state.unet)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(
                "longclip_sara final partial accumulation step "
                f"{step}: flushed {accumulation}/"
                f"{cfg.gradient_accumulation_steps} accumulated mini-batches")
            accumulation = 0
    finally:
        remove_sara_gradient_masks(hooks)

    if step <= 0:
        raise RuntimeError("LongCLIP SaRA completed zero optimizer steps")
    final_validation_loss = heldout_diffusion_loss(state, cfg, encoder)
    final_bucket_summary = (
        getattr(state, "longclip_validation_bucket_summary", None)
        if final_validation_loss is not None else None)
    validation_relative_change = None
    if baseline_validation_loss is not None:
        validation_relative_change = (
            final_validation_loss - baseline_validation_loss
        ) / max(baseline_validation_loss, 1e-12)
        allowed = baseline_validation_loss * (
            1.0 + cfg.validation_max_relative_regression)
        print(
            f"LongCLIP final held-out loss={final_validation_loss:.6f} "
            f"baseline={baseline_validation_loss:.6f} "
            f"relative_change={validation_relative_change:+.4%} "
            f"allowed_max={allowed:.6f}")
        if final_validation_loss > allowed:
            raise RuntimeError(
                "LongCLIP final validation gate failed: "
                f"loss {final_validation_loss:.6f} exceeds {allowed:.6f}")
    elif cfg.require_validation_gate:
        raise RuntimeError("Required LongCLIP final validation did not run")

    short_prompt_relative_rms = None
    if short_prompt_baseline is not None:
        short_prompt_final = short_prompt_prediction_signature(state, cfg, encoder)
        short_prompt_relative_rms = relative_prediction_rms(
            short_prompt_final, short_prompt_baseline)
        print(
            "LongCLIP short-prompt prediction drift: "
            f"relative_rms={short_prompt_relative_rms:.6f} "
            f"limit={cfg.short_prompt_max_relative_rms:.6f}")
        if short_prompt_relative_rms > cfg.short_prompt_max_relative_rms:
            raise RuntimeError(
                "LongCLIP short-prompt regression gate failed: "
                f"relative RMS {short_prompt_relative_rms:.6f} exceeds "
                f"{cfg.short_prompt_max_relative_rms:.6f}")

    delta = sara_selected_delta_metrics(state.unet, baseline)
    validation_summary = {
        "baseline_loss": baseline_validation_loss,
        "final_loss": final_validation_loss,
        "baseline_bucket_summary": baseline_bucket_summary,
        "final_bucket_summary": final_bucket_summary,
        "relative_change": validation_relative_change,
        "max_relative_regression": cfg.validation_max_relative_regression,
        "short_prompt_relative_rms": short_prompt_relative_rms,
        "short_prompt_max_relative_rms": cfg.short_prompt_max_relative_rms,
        "passed": True,
    }
    artifact = {
        "longclip_sara_schema": longclip_sara_schema(cfg, encoder),
        "sparse_values": collect_sara_sparse_values(state.unet),
        "resolution_conditioner_state_dict": (
            {k: v.detach().cpu() for k, v in state.unet.class_embedding.state_dict().items()}
            if cfg.resolution_conditioning_enabled else None),
        "p4_state_dict": p4_state_dict(state.unet) if cfg.p4_enabled else None,
        "sidecar_summary": sidecar_summary,
        "initial_sidecar_loaded": initial_sidecar_loaded,
        "sidecar_smoke": sidecar_smoke,
        "sara_sparse_summary": sara_summary,
        "sara_selected_delta": delta,
        "validation_gate": validation_summary,
        "training_summary": {
            "truncated_training_prompts": encoder.truncated_prompt_count,
            "p4_final_telemetry": p4_telemetry(state.unet),
        },
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
    if wandb is not None:
        final_metrics = {
            "final/optimizer_steps": step,
            "final/selected_delta_relative_l2": delta["selected_delta_relative_l2"],
            "final/truncated_training_prompts": encoder.truncated_prompt_count,
            "final/validation_loss": final_validation_loss,
            "final/validation_relative_change": validation_relative_change,
            "final/short_prompt_relative_rms": short_prompt_relative_rms,
        }
        wandb.log({key: value for key, value in final_metrics.items()
                   if value is not None})
        wandb.finish()


if __name__ == "__main__":
    main()
