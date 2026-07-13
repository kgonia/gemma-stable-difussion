#!/usr/bin/env python3
"""Paired held-out residual-vs-native-CLIP noise-prediction evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import (
    TrainingState, build_ella_connector, load_clip, load_gemma,
    load_stylejourney, make_encode_clip, make_encode_gemma, model_autocast,
    residual_context_for_captions, resolve_autocast_dtype,
    resolve_model_weight_dtype, validate_connector_checkpoint_schema,
)
from pure_ella.camera import camera_conditioned_unet
from pure_ella.config import TrainConfig, seed_everything
from pure_ella.dataset import make_streaming_dataloader


def bootstrap_mean(values, rng, samples=10_000):
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, q)) for q in (0.025, 0.975)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_clip_gemma_residual_p1.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--output", default="residual_paired_eval.json")
    args = parser.parse_args()
    cfg = TrainConfig.from_json(args.config)
    if cfg.connector_type != "clip_gemma_residual_tsc":
        raise ValueError("paired evaluator requires clip_gemma_residual_tsc")
    if not cfg.validation_data_sources:
        raise ValueError("validation_data_sources must point at the group-safe holdout")
    seed_everything(cfg.base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = TrainingState(cfg=cfg, device=device,
                          unet_dtype=resolve_model_weight_dtype(cfg),
                          autocast_dtype=resolve_autocast_dtype(cfg, device))
    load_gemma(state); load_clip(state); load_stylejourney(state)
    state.encode_gemma = make_encode_gemma(state)
    state.encode_clip = make_encode_clip(state)
    build_ella_connector(state)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    validate_connector_checkpoint_schema(checkpoint, state, args.checkpoint)
    state.connector.load_state_dict(checkpoint["connector_state_dict"], strict=True)
    state.connector.eval(); state.unet.eval(); state.vae.eval()

    loader = make_streaming_dataloader(
        cfg.validation_data_sources, phase=991, epoch=0,
        max_samples=args.max_samples, batch_size=cfg.train_batch_size,
        shuffle=False, shuffle_buffer=1, base_seed=cfg.base_seed,
        buckets=cfg.aspect_ratio_buckets, drop_last=False,
        max_image_dimension=cfg.max_image_dimension)
    differences = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=state.unet_dtype)
            with model_autocast(state):
                latent = state.vae.encode(image).latent_dist.sample() * state.vae.config.scaling_factor
            noise = torch.randn_like(latent)
            timestep = torch.randint(0, state.scheduler.config.num_train_timesteps,
                                     (latent.shape[0],), device=device).long()
            noisy = state.scheduler.add_noise(latent, noise, timestep)
            captions = batch["caption"]
            clip_context, _ = state.encode_clip(captions)
            residual_context, _, _, _ = residual_context_for_captions(state, captions, timestep)
            clip_pred = camera_conditioned_unet(
                state.unet, noisy, timestep,
                encoder_hidden_states=clip_context.to(dtype=state.unet_dtype)).sample
            residual_pred = camera_conditioned_unet(
                state.unet, noisy, timestep,
                encoder_hidden_states=residual_context.to(dtype=state.unet_dtype)).sample
            clip_loss = (clip_pred.float() - noise.float()).square().flatten(1).mean(1)
            residual_loss = (residual_pred.float() - noise.float()).square().flatten(1).mean(1)
            differences.extend((clip_loss - residual_loss).cpu().tolist())
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(cfg.base_seed)
    report = {
        "samples": int(len(values)), "metric": "clip_loss_minus_residual_loss",
        "win_rate": float((values > 0).mean()), "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "bootstrap_95_ci": bootstrap_mean(values, rng),
        "sign_test_two_sided_p": float(2 * min(
            (values > 0).mean(), (values < 0).mean())),
        "positive_means_residual_has_lower_noise_prediction_loss": True,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
