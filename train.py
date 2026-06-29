#!/usr/bin/env python3
"""
pure-ella: Gemma -> Stable Diffusion training with ELLA-style timestep-aware connector.

Usage:
    python train.py config.json
    python train.py --config config.json
    python train.py config.json --phases ella,sara --resume-ckpt /workspace/output/ella_connector_clip_pretrain.pt
    python train.py --help
"""

from __future__ import annotations
import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

# Local modules
from pure_ella.config import TrainConfig, resolve_sd_checkpoint, seed_everything
from pure_ella.connector import build_connector
from pure_ella.dataset import make_streaming_dataloader
from pure_ella.sara import (
    build_sara_attn2_kv_sparse_masks,
    install_sara_gradient_masks,
    remove_sara_gradient_masks,
    collect_sara_sparse_values,
    load_sara_sparse_values,
)
from pure_ella.diagnostics import (
    FINAL_SUMMARIES,
    ClipGeometryLoss,
    safe_wandb_log,
    remember_final_summary,
    print_final_summary,
    generate_ella,
    generate_clip_teacher,
    connector_prompt_sensitivity,
    teacher_student_delta_alignment,
    fixed_overfit_loss,
    extra_token_ablation_metrics,
    save_validation_grid,
    _image_collapse_stats,
    _pil_to_uint8_tensor,
    _log_metrics,
    _rel_diff,
    _to_float_maybe,
    connector_extra_grad_stats,
    suffix_counterfactual_sensitivity,
    suffix_counterfactual_contrastive_loss,
    summarize_context_tokens,
    zero_extra_tokens,
    generate_case_image,
    save_complex_case_grid,
    save_suffix_counterfactual_grids,
)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
@dataclass
class TrainingState:
    """Mutable training state shared across all phases."""
    cfg: TrainConfig
    device: torch.device
    unet_dtype: torch.dtype

    # Models
    gemma_model: Any = None
    gemma_tokenizer: Any = None
    gemma_hidden_size: int = 640
    clip_model: Any = None
    clip_tokenizer: Any = None
    unet: Any = None
    vae: Any = None
    scheduler: Any = None
    inf_scheduler: Any = None
    connector: nn.Module = None

    # Dataset
    overfit_eval_batch: dict = None
    ref_quality_cache: dict = None

    # Wandb
    wandb: Any = None

    # Helpers
    encode_gemma = None
    encode_clip = None

    # Quality metrics
    quality_ref_cache: dict = None

    # SaRA internal state
    _sara_grad_handles: list = None
    _sara_summary: dict = None


@dataclass
class StagePlan:
    """Precomputed step budget for a training stage."""
    name: str
    steps_per_epoch: int
    uncapped: int
    effective: int
    image_exposures: int

    def print(self):
        print(f"[{self.name}] steps_per_epoch={self.steps_per_epoch:,} "
              f"uncapped={self.uncapped:,} effective={self.effective:,} "
              f"image_exposures={self.image_exposures:,}")


def estimate_steps(max_samples: int, batch_size: int, epochs: int,
                   cap: Optional[int]) -> StagePlan:
    steps_per_epoch = max(1, math.ceil(int(max_samples) / int(batch_size)))
    uncapped = steps_per_epoch * int(epochs)
    effective = min(uncapped, int(cap)) if cap is not None else uncapped
    return StagePlan(
        name="stage",
        steps_per_epoch=steps_per_epoch,
        uncapped=uncapped,
        effective=effective,
        image_exposures=effective * int(batch_size),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_gemma(state: TrainingState):
    """Load frozen Gemma 3 model and tokenizer."""
    cfg = state.cfg
    print(f"Loading Gemma: {cfg.gemma_id}")
    state.gemma_tokenizer = AutoTokenizer.from_pretrained(
        cfg.gemma_id, token=os.environ.get("HF_TOKEN"))
    state.gemma_model = AutoModel.from_pretrained(
        cfg.gemma_id, torch_dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    for p in state.gemma_model.parameters():
        p.requires_grad_(False)
    state.gemma_hidden_size = state.gemma_model.config.hidden_size
    print(f"Gemma hidden_size: {state.gemma_hidden_size}")


def load_clip(state: TrainingState):
    """Load frozen CLIP text model as teacher/diagnostic."""
    cfg = state.cfg
    print(f"Loading CLIP teacher: {cfg.clip_id}")
    state.clip_tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_id)
    state.clip_model = CLIPTextModel.from_pretrained(
        cfg.clip_id, torch_dtype=state.unet_dtype,
        token=os.environ.get("HF_TOKEN"),
    ).to(state.device).eval()
    for p in state.clip_model.parameters():
        p.requires_grad_(False)
    print("CLIP loaded as teacher/diagnostic only")


def load_stylejourney(state: TrainingState):
    """Load StyleJourney/runwayml SD checkpoint: UNet, VAE, scheduler."""
    cfg = state.cfg
    sd_path = resolve_sd_checkpoint(cfg)
    print(f"Loading SD checkpoint: {sd_path}")
    if sd_path.endswith(".safetensors"):
        pipe = StableDiffusionPipeline.from_single_file(
            sd_path, torch_dtype=state.unet_dtype,
            token=os.environ.get("HF_TOKEN"),
        )
        state.unet = pipe.unet.to(state.device)
        state.vae = pipe.vae.to(state.device)
        state.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        pipe = None
    else:
        state.unet = UNet2DConditionModel.from_pretrained(
            sd_path, subfolder="unet", torch_dtype=state.unet_dtype,
            token=os.environ.get("HF_TOKEN"),
        ).to(state.device)
        state.vae = AutoencoderKL.from_pretrained(
            sd_path, subfolder="vae", torch_dtype=state.unet_dtype,
            token=os.environ.get("HF_TOKEN"),
        ).to(state.device)
        state.scheduler = DDPMScheduler.from_pretrained(
            sd_path, subfolder="scheduler")

    state.inf_scheduler = DPMSolverMultistepScheduler.from_config(
        state.scheduler.config)

    if cfg.enable_unet_gradient_checkpointing:
        state.unet.enable_gradient_checkpointing()
        print("UNet gradient checkpointing: ON")
    print(f"UNet dtype: {state.unet_dtype}")
    print(f"UNet cross_attention_dim: {state.unet.config.cross_attention_dim}")
    print("No UNet surgery. Original cross-attn stays intact.")

    for p in state.unet.parameters():
        p.requires_grad_(False)
    for p in state.vae.parameters():
        p.requires_grad_(False)


def build_ella_connector(state: TrainingState):
    """Build the ELLA connector and smoke-test."""
    cfg = state.cfg
    params = {
        "gemma_dim": state.gemma_hidden_size,
        "width": cfg.connector_width,
        "context_tokens": cfg.context_tokens,
        "anchor_tokens": cfg.clip_anchor_tokens,
        "layers": cfg.connector_layers,
        "heads": cfg.connector_heads,
        "ff_mult": cfg.connector_ff_mult,
        "dropout": cfg.connector_dropout,
        "time_embed_dim": cfg.connector_time_embed_dim,
        "extra_gate_init": cfg.connector_extra_gate_init,
        "recursive_y_steps": cfg.recursive_y_steps,
        "recursive_y_gate_init": cfg.recursive_y_gate_init,
        "trm_outer_steps": cfg.trm_outer_steps,
        "trm_inner_steps": cfg.trm_inner_steps,
        "trm_scratch_tokens": cfg.trm_scratch_tokens,
        "trm_y_gate_init": cfg.trm_y_gate_init,
        "trm_z_gate_init": cfg.trm_z_gate_init,
    }
    state.connector = build_connector(cfg.connector_type, **params).to(
        device=state.device, dtype=state.unet_dtype)

    with torch.no_grad():
        g = torch.randn(2, 32, state.gemma_hidden_size,
                        device=state.device, dtype=state.unet_dtype)
        m = torch.ones(2, 32, device=state.device, dtype=torch.long)
        t = torch.tensor([10, 500], device=state.device).long()
        y = state.connector(g, t, m)
        assert y.shape == (2, cfg.context_tokens, 768), y.shape
        assert torch.isfinite(y).all()

    n_params = sum(p.numel() for p in state.connector.parameters())
    print(f"Connector PASS: output={tuple(y.shape)}, params={n_params:,}")
    eg = state.connector.extra_gate_logit
    if eg is not None:
        print("extra_token_gate:", torch.sigmoid(eg).item())


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def _as_prompt_list(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(v) for v in list(x)]


def make_encode_gemma(state: TrainingState):
    @torch.no_grad()
    def encode_gemma(prompts, max_length=None):
        max_length = max_length or state.cfg.max_gemma_len
        toks = state.gemma_tokenizer(
            _as_prompt_list(prompts),
            padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(state.gemma_model.device)
        out = state.gemma_model(**toks, output_hidden_states=True,
                                use_cache=False)
        layers = out.hidden_states
        idx = state.cfg.gemma_layer_index
        idx = idx if idx >= 0 else len(layers) + idx
        if idx < 0 or idx >= len(layers):
            raise IndexError(
                f"GEMMA_LAYER_INDEX={state.cfg.gemma_layer_index} "
                f"invalid for {len(layers)} hidden states")
        h = layers[idx].to(device=state.device)
        m = toks.attention_mask.to(device=state.device)
        return h, m
    return encode_gemma


def make_encode_clip(state: TrainingState):
    @torch.no_grad()
    def encode_clip(prompts):
        toks = state.clip_tokenizer(
            _as_prompt_list(prompts),
            padding="max_length", truncation=True,
            max_length=state.cfg.clip_anchor_tokens, return_tensors="pt",
        ).to(state.device)
        out = state.clip_model(**toks).last_hidden_state
        return out, toks.attention_mask
    return encode_clip


# ---------------------------------------------------------------------------
# VRAM logging
# ---------------------------------------------------------------------------
def log_vram(label: str, step: int, state: TrainingState):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[vram:{label}] step={step} alloc={alloc:.2f}GB "
          f"reserved={reserved:.2f}GB peak={peak:.2f}GB")
    safe_wandb_log({
        f"vram/{label}_alloc_gb": alloc,
        f"vram/{label}_reserved_gb": reserved,
        f"vram/{label}_peak_gb": peak,
        "global_step": step,
    }, wandb=state.wandb)


# ---------------------------------------------------------------------------
# Phase 0: CLIP alignment pretrain
# ---------------------------------------------------------------------------
def run_clip_pretrain(state: TrainingState):
    cfg = state.cfg
    if (not cfg.run_training or not cfg.run_clip_alignment_pretrain
            or cfg.pretrain_epochs <= 0):
        print("CLIP alignment pretrain skipped")
        return

    assert state.clip_model is not None, "CLIP required for pretrain"
    clip_geom = ClipGeometryLoss()

    for p in state.unet.parameters(): p.requires_grad_(False)
    for p in state.vae.parameters(): p.requires_grad_(False)
    for p in state.gemma_model.parameters(): p.requires_grad_(False)
    for p in state.clip_model.parameters(): p.requires_grad_(False)
    for p in state.connector.parameters(): p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        state.connector.parameters(), lr=cfg.pretrain_lr,
        weight_decay=0.01, eps=1e-6)
    state.connector.train()
    state.gemma_model.eval()
    state.clip_model.eval()
    state.unet.eval()
    state.vae.eval()

    plan = estimate_steps(cfg.max_samples_pretrain, cfg.train_batch_size,
                          cfg.pretrain_epochs, cfg.pretrain_max_opt_steps)
    plan.name = "clip_pretrain"
    plan.print()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_vram("clip_pretrain_start", 0, state)

    opt_step = 0
    best = None
    last_pretrain = {}
    stop = False
    t0 = time.time()

    for epoch in range(cfg.pretrain_epochs):
        dl = make_streaming_dataloader(
            cfg.stream_repo, phase=10, epoch=epoch,
            max_samples=cfg.max_samples_pretrain,
            batch_size=cfg.train_batch_size,
            shuffle=cfg.shuffle_streaming,
            shuffle_buffer=cfg.shuffle_buffer, base_seed=cfg.base_seed)
        progress = tqdm(dl, desc=f"CLIP-pretrain {epoch+1}/{cfg.pretrain_epochs}")
        for batch in progress:
            captions = _as_prompt_list(batch["caption"])
            with torch.no_grad():
                gh, gm = state.encode_gemma(captions)
                ch, cm = state.encode_clip(captions)
            t = torch.zeros(len(captions), device=state.device, dtype=torch.long)
            pred = state.connector(
                gh.to(dtype=state.unet_dtype), t, gm,
                context_tokens=cfg.clip_anchor_tokens)
            ld = clip_geom(pred, ch, cm)
            loss = ld["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("CLIP alignment loss NaN/Inf")

            last_pretrain = {
                "step": opt_step + 1,
                "loss": loss.item(),
                "mse": ld["mse"].item(),
                "cos": ld["cos"].item(),
                "pooled_cos": ld["pooled_cos"].item(),
                "norm_ratio": ld["norm_ratio"].item(),
            }
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(state.connector.parameters(),
                                      cfg.grad_clip_norm)
            optimizer.step()
            opt_step += 1
            best = (float(loss.item()) if best is None
                    else min(best, float(loss.item())))

            if opt_step % 50 == 0:
                print(f"pretrain step {opt_step}: total={loss.item():.5f} "
                      f"mse={ld['mse'].item():.5f} "
                      f"cos={ld['cos'].item():.5f} "
                      f"pooled_cos={ld['pooled_cos'].item():.5f} "
                      f"norm_ratio={ld['norm_ratio'].item():.3f}")
                safe_wandb_log({
                    "pretrain/step": opt_step,
                    "pretrain/loss": loss.item(),
                    "pretrain/mse": ld["mse"].item(),
                    "pretrain/cos": ld["cos"].item(),
                    "pretrain/pooled_cos": ld["pooled_cos"].item(),
                    "pretrain/norm_ratio": ld["norm_ratio"].item(),
                }, wandb=state.wandb)

            progress.set_postfix(
                {"loss": f"{loss.item():.4f}", "best": f"{best:.4f}"})
            if (cfg.pretrain_max_opt_steps
                    and opt_step >= cfg.pretrain_max_opt_steps):
                stop = True
                print("Stopping CLIP pretrain at step cap", opt_step)
                break
        if stop:
            break

    elapsed_min = (time.time() - t0) / 60
    log_vram("clip_pretrain_end", opt_step, state)
    pretrain_summary = {
        "steps": opt_step,
        "best_loss": best or float("nan"),
        "elapsed_min": elapsed_min,
        "final_loss": last_pretrain.get("loss", float("nan")),
        "final_mse": last_pretrain.get("mse", float("nan")),
        "final_cos": last_pretrain.get("cos", float("nan")),
        "final_pooled_cos": last_pretrain.get("pooled_cos", float("nan")),
        "final_norm_ratio": last_pretrain.get("norm_ratio", float("nan")),
    }
    remember_final_summary(
        "clip_pretrain", pretrain_summary,
        wandb_prefix="final/clip_pretrain", wandb=state.wandb)
    print(f"CLIP pretrain done: steps={opt_step} best={best} "
          f"elapsed={elapsed_min:.1f}m")
    print_final_summary("clip_pretrain")

    ckpt_path = f"{cfg.output_dir}/ella_connector_clip_pretrain.pt"
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save({
        "connector_state_dict": {k: v.detach().cpu()
                                 for k, v in state.connector.state_dict().items()},
        "config": cfg.to_dict(),
        "stage": "clip_alignment_pretrain",
    }, ckpt_path)

    post_sens = connector_prompt_sensitivity(
        cfg.val_prompts, state, label="post_clip_pretrain", wandb=state.wandb)
    post_delta = teacher_student_delta_alignment(
        cfg.val_prompts[0], state, label="post_clip_pretrain",
        wandb=state.wandb)
    pretrain_summary.update({
        "post_clip_prompt_sensitivity": post_sens,
        "post_clip_delta_cos": (post_delta.get("delta_cos", float("nan"))
                                if isinstance(post_delta, dict) else float("nan")),
        "post_clip_norm_ratio": (post_delta.get("norm_ratio", float("nan"))
                                 if isinstance(post_delta, dict) else float("nan")),
    })
    remember_final_summary(
        "clip_pretrain", pretrain_summary,
        wandb_prefix="final/clip_pretrain", wandb=state.wandb)
    print_final_summary("clip_pretrain")


# ---------------------------------------------------------------------------
# Phase 1: ELLA connector diffusion + teacher-delta training
# ---------------------------------------------------------------------------
def run_ella_training(state: TrainingState):
    cfg = state.cfg
    if not cfg.run_training or cfg.ella_epochs <= 0:
        print("ELLA connector phase skipped")
        return

    for p in state.unet.parameters(): p.requires_grad_(False)
    for p in state.vae.parameters(): p.requires_grad_(False)
    for p in state.gemma_model.parameters(): p.requires_grad_(False)
    if state.clip_model is not None:
        for p in state.clip_model.parameters(): p.requires_grad_(False)
    for p in state.connector.parameters(): p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        state.connector.parameters(), lr=cfg.ella_lr,
        weight_decay=0.01, eps=1e-6)
    use_teacher_delta = bool(
        cfg.use_clip_teacher_delta and state.clip_model is not None)
    print("Phase 1 forward mode:",
          "paired_cfg_teacher" if use_teacher_delta else "conditional_only")

    state.connector.train()
    state.unet.eval()
    state.vae.eval()
    state.gemma_model.eval()
    if state.clip_model is not None:
        state.clip_model.eval()

    plan = estimate_steps(cfg.max_samples_ella, cfg.train_batch_size,
                          cfg.ella_epochs, cfg.ella_max_opt_steps)
    plan.name = "ella_frozen_unet"
    plan.print()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_vram("ella_start", 0, state)

    opt_step = 0
    best = None
    last_ella = {}
    stop = False
    t0 = time.time()

    for epoch in range(cfg.ella_epochs):
        dl = make_streaming_dataloader(
            cfg.stream_repo, phase=20, epoch=epoch,
            max_samples=cfg.max_samples_ella,
            batch_size=cfg.train_batch_size,
            shuffle=cfg.shuffle_streaming,
            shuffle_buffer=cfg.shuffle_buffer, base_seed=cfg.base_seed)
        progress = tqdm(dl, desc=f"ELLA {epoch+1}/{cfg.ella_epochs}")
        for batch in progress:
            captions = _as_prompt_list(batch["caption"])
            img = batch["image"].to(device=state.device,
                                     dtype=state.unet_dtype)

            with torch.no_grad():
                latent = (state.vae.encode(img).latent_dist.sample()
                          * state.vae.config.scaling_factor)
                noise = torch.randn_like(latent)
                t = torch.randint(
                    0, state.scheduler.config.num_train_timesteps,
                    (latent.shape[0],), device=state.device).long()
                noisy = state.scheduler.add_noise(latent, noise, t)
                gh, gm = state.encode_gemma(captions)

            loss_teacher = noise.new_tensor(0.0)
            loss_delta = noise.new_tensor(0.0)
            loss_anchor = noise.new_tensor(0.0)

            if use_teacher_delta:
                empty = [""] * len(captions)
                with torch.no_grad():
                    ugh, ugm = state.encode_gemma(empty)
                noisy_pair = torch.cat([noisy, noisy], dim=0)
                t_pair = torch.cat([t, t], dim=0)
                g_pair = torch.cat([gh, ugh], dim=0)
                m_pair = torch.cat([gm, ugm], dim=0)
                ctx = state.connector(
                    g_pair.to(dtype=state.unet_dtype), t_pair, m_pair,
                    context_tokens=cfg.context_tokens)
                student_pair = state.unet(
                    noisy_pair, t_pair, encoder_hidden_states=ctx).sample
                student_cond, student_uncond = student_pair.chunk(2)
                loss_diff = F.mse_loss(student_cond.float(), noise.float())

                with torch.no_grad():
                    ch, cm = state.encode_clip(captions)
                    uch, ucm = state.encode_clip(empty)
                    clip_pair = torch.cat([ch, uch], dim=0)
                    clip_m_pair = torch.cat([cm, ucm], dim=0)
                    teacher_pair = state.unet(
                        noisy_pair, t_pair,
                        encoder_hidden_states=clip_pair.to(dtype=state.unet_dtype),
                        encoder_attention_mask=clip_m_pair).sample.detach()
                    teacher_cond, teacher_uncond = teacher_pair.chunk(2)
                    teacher_delta = teacher_cond - teacher_uncond
                student_delta = student_cond - student_uncond
                loss_teacher = F.mse_loss(
                    student_cond.float(), teacher_cond.float())
                loss_delta = F.mse_loss(
                    student_delta.float(), teacher_delta.float())
            else:
                ctx = state.connector(
                    gh.to(dtype=state.unet_dtype), t, gm,
                    context_tokens=cfg.context_tokens)
                student_cond = state.unet(
                    noisy, t, encoder_hidden_states=ctx).sample
                loss_diff = F.mse_loss(student_cond.float(), noise.float())

            if (cfg.phase1_semantic_anchor_weight > 0
                    and state.clip_model is not None):
                clip_geom = ClipGeometryLoss()
                with torch.no_grad():
                    ch, cm = state.encode_clip(captions)
                pred77 = ctx[:len(captions), :cfg.clip_anchor_tokens, :]
                loss_anchor = clip_geom(pred77, ch, cm)["total"]

            loss = (cfg.lambda_diffusion * loss_diff
                    + cfg.lambda_teacher * loss_teacher
                    + cfg.lambda_text_delta * loss_delta
                    + cfg.phase1_semantic_anchor_weight * loss_anchor)

            # ── suffix counterfactual contrastive loss (training pressure) ──
            loss_suffix_cf = loss.new_tensor(0.0)
            if (cfg.suffix_counterfactual_loss_weight > 0
                    and cfg.suffix_counterfactual_loss_every > 0
                    and (opt_step + 1) % cfg.suffix_counterfactual_loss_every == 0
                    and cfg.context_tokens > cfg.clip_anchor_tokens):
                loss_suffix_cf = suffix_counterfactual_contrastive_loss(
                    state, margin=cfg.suffix_counterfactual_loss_margin)
                if not torch.isfinite(loss_suffix_cf):
                    raise RuntimeError("suffix_counterfactual_contrastive_loss NaN/Inf")
                loss = loss + cfg.suffix_counterfactual_loss_weight * loss_suffix_cf

            if not torch.isfinite(loss):
                raise RuntimeError("ELLA loss NaN/Inf")

            last_ella = {
                "step": opt_step + 1,
                "loss": loss.item(),
                "loss_diff": loss_diff.item(),
                "loss_teacher": float(loss_teacher.item()),
                "loss_delta": float(loss_delta.item()),
                "loss_anchor": float(loss_anchor.item()),
                "loss_suffix_cf": float(loss_suffix_cf.item()),
            }
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_stats = connector_extra_grad_stats(
                state.connector, cfg.clip_anchor_tokens, state)
            nn.utils.clip_grad_norm_(state.connector.parameters(),
                                      cfg.grad_clip_norm)
            optimizer.step()
            opt_step += 1
            best = (float(loss.item()) if best is None
                    else min(best, float(loss.item())))

            if opt_step % 25 == 0:
                safe_wandb_log({
                    "ella/step": opt_step,
                    "ella/loss": loss.item(),
                    "ella/loss_diff": loss_diff.item(),
                    "ella/loss_teacher": float(loss_teacher.item()),
                    "ella/loss_delta": float(loss_delta.item()),
                    "ella/loss_anchor": float(loss_anchor.item()),
                    "ella/loss_suffix_cf": float(loss_suffix_cf.item()),
                    "ella/extra_gate_grad_norm":
                        grad_stats["extra_gate_grad_norm"]
                        if grad_stats["extra_gate_grad_norm"] is not None
                        else 0.0,
                    "ella/extra_query_grad_norm":
                        grad_stats["extra_query_grad_norm"]
                        if grad_stats["extra_query_grad_norm"] is not None
                        else 0.0,
                    "ella/extra_pos_grad_norm":
                        grad_stats["extra_pos_grad_norm"]
                        if grad_stats["extra_pos_grad_norm"] is not None
                        else 0.0,
                }, wandb=state.wandb)

            if (cfg.validation_every_opt_steps
                    and opt_step % cfg.validation_every_opt_steps == 0):
                print(f"ella step {opt_step}: loss={loss.item():.5f} "
                      f"diff={loss_diff.item():.5f} "
                      f"teacher={loss_teacher.item():.5f} "
                      f"delta={loss_delta.item():.5f} "
                      f"anchor={loss_anchor.item():.5f}")
                fixed_overfit_loss(
                    state, label=f"ella_step_{opt_step:06d}",
                    wandb=state.wandb)
                teacher_student_delta_alignment(
                    cfg.val_prompts[0], state,
                    label=f"ella_step_{opt_step:06d}", wandb=state.wandb)
                if (cfg.run_long_context_diagnostics
                        and cfg.context_tokens > cfg.clip_anchor_tokens):
                    ab = extra_token_ablation_metrics(
                        cfg.val_prompts[0], state,
                        label=f"ella_step_{opt_step:06d}",
                        wandb=state.wandb)
                    zd = ab.get("rel_diff_full_vs_zeroextra_delta")
                    if zd is not None:
                        safe_wandb_log({"ella/suffix_zero_delta": zd}, wandb=state.wandb)
                        if zd < 0.03:
                            print(f"⚠  LOW suffix sensitivity: "
                                  f"rel_diff_full_vs_zeroextra_delta={zd:.5f} - "
                                  f"extra tokens have almost no effect on CFG delta")
                    suffix_counterfactual_sensitivity(
                        state, label=f"ella_step_{opt_step:06d}",
                        wandb=state.wandb)
                if (cfg.generation_grid_every_opt_steps > 0
                        and opt_step % cfg.generation_grid_every_opt_steps == 0):
                    save_checkpoint_grid(
                        f"ella_step_{opt_step:06d}", state)

            progress.set_postfix(
                {"loss": f"{loss.item():.4f}", "best": f"{best:.4f}"})
            if cfg.ella_max_opt_steps and opt_step >= cfg.ella_max_opt_steps:
                stop = True
                print("Stopping ELLA at step cap", opt_step)
                break
        if stop:
            break

    elapsed_min = (time.time() - t0) / 60
    log_vram("ella_end", opt_step, state)
    ella_summary = {
        "steps": opt_step,
        "best_loss": best or float("nan"),
        "elapsed_min": elapsed_min,
        "final_loss": last_ella.get("loss", float("nan")),
        "final_loss_diff": last_ella.get("loss_diff", float("nan")),
        "final_loss_teacher": last_ella.get("loss_teacher", float("nan")),
        "final_loss_delta": last_ella.get("loss_delta", float("nan")),
        "final_loss_anchor": last_ella.get("loss_anchor", float("nan")),
    }
    remember_final_summary(
        "ella_frozen_unet", ella_summary,
        wandb_prefix="final/ella_frozen_unet", wandb=state.wandb)
    print(f"ELLA frozen-UNet done: steps={opt_step} best={best} "
          f"elapsed={elapsed_min:.1f}m")
    print_final_summary("ella_frozen_unet")

    ckpt_path = f"{cfg.output_dir}/ella_connector_frozen_unet.pt"
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save({
        "connector_state_dict": {k: v.detach().cpu()
                                 for k, v in state.connector.state_dict().items()},
        "config": cfg.to_dict(),
        "stage": "ella_frozen_unet",
    }, ckpt_path)
    ella_summary["checkpoint_saved"] = ckpt_path
    remember_final_summary(
        "ella_frozen_unet", ella_summary,
        wandb_prefix="final/ella_frozen_unet", wandb=state.wandb)
    print_final_summary("ella_frozen_unet")


# ---------------------------------------------------------------------------
# Phase 2: Sparse SaRA attn2 K/V adaptation
# ---------------------------------------------------------------------------
def run_sara_training(state: TrainingState):
    cfg = state.cfg
    if not cfg.run_sara_phase or cfg.sara_epochs <= 0:
        print("SaRA phase disabled")
        return

    # Build sparse masks on attn2 K/V weights
    for p in state.unet.parameters():
        p.requires_grad_(False)
    sara_summary = build_sara_attn2_kv_sparse_masks(
        state.unet,
        threshold=cfg.sara_threshold,
        max_sparse_fraction_warn=cfg.sara_max_sparse_fraction_warn,
        max_sparse_fraction_abort=cfg.sara_max_sparse_fraction_abort,
        target_substrings=cfg.sara_target_substrings,
    )
    grad_handles = install_sara_gradient_masks(state.unet)

    # Connector stays trainable
    for p in state.connector.parameters():
        p.requires_grad_(True)
    trainable_connector = [p for p in state.connector.parameters() if p.requires_grad]
    trainable_unet = [p for p in state.unet.parameters() if p.requires_grad]
    trainable_params = trainable_connector + trainable_unet
    print(f"Trainable params in ELLA+SaRA: {sum(p.numel() for p in trainable_params):,}")
    print(f"  connector: {sum(p.numel() for p in trainable_connector):,}")
    print(f"  sparse UNet: {sum(p.numel() for p in trainable_unet):,}")

    optimizer = torch.optim.AdamW([
        {"params": trainable_connector, "weight_decay": 0.01},
        {"params": trainable_unet, "weight_decay": 0.0},
    ], lr=cfg.sara_lr, eps=1e-6)

    use_teacher_delta_phase2 = bool(
        cfg.use_clip_teacher_delta and cfg.use_clip_teacher_delta_phase2
        and state.clip_model is not None,
    )
    if cfg.use_clip_teacher_delta and not use_teacher_delta_phase2:
        print("Phase 2 CLIP teacher delta disabled to avoid moving-teacher distillation.")
    print("Phase 2 forward mode:",
          "paired_cfg_teacher" if use_teacher_delta_phase2 else "conditional_only")

    state.connector.train()
    state.unet.train()
    state.vae.eval()
    state.gemma_model.eval()
    if state.clip_model is not None:
        state.clip_model.eval()

    plan = estimate_steps(cfg.max_samples_ella, cfg.train_batch_size,
                          cfg.sara_epochs, cfg.sara_max_opt_steps)
    plan.name = "ella_sara_attn2_kv"
    plan.print()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_vram("sara_start", 0, state)

    opt_step = 0
    best = None
    stop = False
    t0 = time.time()

    for epoch in range(cfg.sara_epochs):
        dl = make_streaming_dataloader(
            cfg.stream_repo, phase=30, epoch=epoch,
            max_samples=cfg.max_samples_ella,
            batch_size=cfg.train_batch_size,
            shuffle=cfg.shuffle_streaming,
            shuffle_buffer=cfg.shuffle_buffer, base_seed=cfg.base_seed)
        progress = tqdm(dl, desc=f"ELLA+SaRA {epoch+1}/{cfg.sara_epochs}")
        for batch in progress:
            captions = _as_prompt_list(batch["caption"])
            img = batch["image"].to(device=state.device, dtype=state.unet_dtype)

            with torch.no_grad():
                latent = (state.vae.encode(img).latent_dist.sample()
                          * state.vae.config.scaling_factor)
                noise = torch.randn_like(latent)
                t = torch.randint(
                    0, state.scheduler.config.num_train_timesteps,
                    (latent.shape[0],), device=state.device).long()
                noisy = state.scheduler.add_noise(latent, noise, t)
                gh, gm = state.encode_gemma(captions)

            loss_teacher = noise.new_tensor(0.0)
            loss_delta = noise.new_tensor(0.0)
            loss_anchor = noise.new_tensor(0.0)

            if use_teacher_delta_phase2:
                empty = [""] * len(captions)
                with torch.no_grad():
                    ugh, ugm = state.encode_gemma(empty)
                noisy_pair = torch.cat([noisy, noisy], dim=0)
                t_pair = torch.cat([t, t], dim=0)
                g_pair = torch.cat([gh, ugh], dim=0)
                m_pair = torch.cat([gm, ugm], dim=0)
                ctx = state.connector(
                    g_pair.to(dtype=state.unet_dtype), t_pair, m_pair,
                    context_tokens=cfg.context_tokens)
                student_pair = state.unet(
                    noisy_pair, t_pair, encoder_hidden_states=ctx).sample
                student_cond, student_uncond = student_pair.chunk(2)
                loss_diff = F.mse_loss(student_cond.float(), noise.float())

                with torch.no_grad():
                    ch, cm = state.encode_clip(captions)
                    uch, ucm = state.encode_clip(empty)
                    clip_pair = torch.cat([ch, uch], dim=0)
                    clip_m_pair = torch.cat([cm, ucm], dim=0)
                    teacher_pair = state.unet(
                        noisy_pair, t_pair,
                        encoder_hidden_states=clip_pair.to(dtype=state.unet_dtype),
                        encoder_attention_mask=clip_m_pair).sample.detach()
                    teacher_cond, teacher_uncond = teacher_pair.chunk(2)
                    teacher_delta = teacher_cond - teacher_uncond
                student_delta = student_cond - student_uncond
                loss_teacher = F.mse_loss(
                    student_cond.float(), teacher_cond.float())
                loss_delta = F.mse_loss(
                    student_delta.float(), teacher_delta.float())
            else:
                ctx = state.connector(
                    gh.to(dtype=state.unet_dtype), t, gm,
                    context_tokens=cfg.context_tokens)
                student_cond = state.unet(
                    noisy, t, encoder_hidden_states=ctx).sample
                loss_diff = F.mse_loss(student_cond.float(), noise.float())

            if (cfg.phase2_semantic_anchor_weight > 0
                    and state.clip_model is not None):
                clip_geom = ClipGeometryLoss()
                with torch.no_grad():
                    ch, cm = state.encode_clip(captions)
                pred77 = ctx[:len(captions), :cfg.clip_anchor_tokens, :]
                loss_anchor = clip_geom(pred77, ch, cm)["total"]

            loss = (cfg.lambda_diffusion * loss_diff
                    + cfg.lambda_teacher * loss_teacher
                    + cfg.lambda_text_delta * loss_delta
                    + cfg.phase2_semantic_anchor_weight * loss_anchor)
            if not torch.isfinite(loss):
                raise RuntimeError("ELLA+SaRA loss NaN/Inf")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_stats = connector_extra_grad_stats(state.connector, cfg.clip_anchor_tokens, state)
            nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
            optimizer.step()
            opt_step += 1
            best = (float(loss.item()) if best is None
                    else min(best, float(loss.item())))

            if opt_step % 25 == 0:
                safe_wandb_log({
                    "sara/step": opt_step,
                    "sara/loss": loss.item(),
                    "sara/loss_diff": loss_diff.item(),
                    "sara/loss_teacher": float(loss_teacher.item()),
                    "sara/loss_delta": float(loss_delta.item()),
                    "sara/loss_anchor": float(loss_anchor.item()),
                    "sara/extra_gate_grad_norm":
                        grad_stats["extra_gate_grad_norm"]
                        if grad_stats["extra_gate_grad_norm"] is not None
                        else 0.0,
                    "sara/extra_query_grad_norm":
                        grad_stats["extra_query_grad_norm"]
                        if grad_stats["extra_query_grad_norm"] is not None
                        else 0.0,
                    "sara/extra_pos_grad_norm":
                        grad_stats["extra_pos_grad_norm"]
                        if grad_stats["extra_pos_grad_norm"] is not None
                        else 0.0,
                }, wandb=state.wandb)

            if (cfg.validation_every_opt_steps
                    and opt_step % cfg.validation_every_opt_steps == 0):
                print(f"sara step {opt_step}: loss={loss.item():.5f} "
                      f"diff={loss_diff.item():.5f} "
                      f"teacher={loss_teacher.item():.5f} "
                      f"delta={loss_delta.item():.5f} "
                      f"anchor={loss_anchor.item():.5f}")
                fixed_overfit_loss(
                    state, label=f"sara_step_{opt_step:06d}",
                    wandb=state.wandb)
                teacher_student_delta_alignment(
                    cfg.val_prompts[0], state,
                    label=f"sara_step_{opt_step:06d}", wandb=state.wandb)
                if (cfg.run_long_context_diagnostics
                        and cfg.context_tokens > cfg.clip_anchor_tokens):
                    ab = extra_token_ablation_metrics(
                        cfg.val_prompts[0], state,
                        label=f"sara_step_{opt_step:06d}",
                        wandb=state.wandb)
                    zd = ab.get("rel_diff_full_vs_zeroextra_delta")
                    if zd is not None:
                        safe_wandb_log({"sara/suffix_zero_delta": zd}, wandb=state.wandb)
                        if zd < 0.03:
                            print(f"⚠  LOW suffix sensitivity: "
                                  f"rel_diff_full_vs_zeroextra_delta={zd:.5f} - "
                                  f"extra tokens have almost no effect on CFG delta")
                    suffix_counterfactual_sensitivity(
                        state, label=f"sara_step_{opt_step:06d}",
                        wandb=state.wandb)
                if (cfg.generation_grid_every_opt_steps > 0
                        and opt_step % cfg.generation_grid_every_opt_steps == 0):
                    save_checkpoint_grid(
                        f"sara_step_{opt_step:06d}", state)

            progress.set_postfix(
                {"loss": f"{loss.item():.4f}", "best": f"{best:.4f}"})
            if (cfg.sara_max_opt_steps
                    and opt_step >= cfg.sara_max_opt_steps):
                stop = True
                print("Stopping ELLA+SaRA at step cap", opt_step)
                break
        if stop:
            break

    elapsed_min = (time.time() - t0) / 60
    log_vram("sara_end", opt_step, state)

    sara_summary_dict = {
        "steps": opt_step,
        "best_loss": best or float("nan"),
        "elapsed_min": elapsed_min,
        "sparse_selected": sara_summary["selected"],
        "sparse_total": sara_summary["total_target"],
        "sparse_fraction": sara_summary["fraction"],
    }
    remember_final_summary(
        "sara_attn2_kv", sara_summary_dict,
        wandb_prefix="final/sara_attn2_kv", wandb=state.wandb)

    # Store handles + summary on state for save_artifacts
    state._sara_grad_handles = grad_handles
    state._sara_summary = sara_summary

    print(f"ELLA+SaRA done: steps={opt_step} best={best} "
          f"elapsed={elapsed_min:.1f}m")
    print_final_summary("sara_attn2_kv")


# ---------------------------------------------------------------------------
# Validation & Final proof
# ---------------------------------------------------------------------------
def save_checkpoint_grid(label: str, state: TrainingState):
    """Generate + save a validation image grid at a training checkpoint."""
    cfg = state.cfg
    imgs, labels = [], []
    for ptxt in cfg.val_prompts:
        imgs.append(generate_ella(ptxt, state, steps=cfg.val_steps,
                                   guidance=cfg.val_guidance,
                                   seed=cfg.val_seed))
        labels.append(f"ELLA L{cfg.context_tokens}: {ptxt[:40]}")
    grid_path = f"{cfg.output_dir}/{label}_ella_L{cfg.context_tokens}.png"
    save_validation_grid(imgs, labels, grid_path,
                         f"ELLA L{cfg.context_tokens} [{label}]")
    print(f"Checkpoint grid saved: {grid_path}")

    if state.clip_model is not None:
        clip_imgs, clip_labels = [], []
        for ptxt in cfg.val_prompts:
            clip_imgs.append(generate_clip_teacher(
                ptxt, state, steps=cfg.val_steps,
                guidance=cfg.val_guidance, seed=cfg.val_seed))
            clip_labels.append("CLIP teacher")
        clip_path = f"{cfg.output_dir}/{label}_clip_teacher.png"
        save_validation_grid(clip_imgs, clip_labels, clip_path,
                             f"CLIP teacher [{label}]")


def run_validation_grids(state: TrainingState):
    cfg = state.cfg
    print("Prompt sensitivity:")
    final_sens = connector_prompt_sensitivity(
        cfg.val_prompts, state, label="final_ella", wandb=state.wandb)
    final_delta = teacher_student_delta_alignment(
        cfg.val_prompts[0], state, label="final_ella", wandb=state.wandb)
    final_overfit = fixed_overfit_loss(
        state, label="final_overfit", wandb=state.wandb)

    fev_summary = {
        "prompt_sensitivity": final_sens,
        "delta_cos": (final_delta.get("delta_cos", float("nan"))
                      if isinstance(final_delta, dict) else float("nan")),
        "norm_ratio": (final_delta.get("norm_ratio", float("nan"))
                       if isinstance(final_delta, dict) else float("nan")),
        "fixed_overfit_mse": final_overfit,
    }
    remember_final_summary(
        "final_eval", fev_summary,
        wandb_prefix="final/eval", wandb=state.wandb)
    print_final_summary("final_eval")

    if (cfg.run_long_context_diagnostics
            and cfg.context_tokens > cfg.clip_anchor_tokens):
        print("Long-context numeric suite skipped in script mode")
    else:
        print("Long-context numeric suite skipped at 77-token stage")

    if cfg.run_fixed_validation_grids:
        ella_imgs, labels = [], []
        for ptxt in cfg.val_prompts:
            print("Generating ELLA:", ptxt)
            ella_imgs.append(generate_ella(
                ptxt, state, steps=cfg.val_steps,
                guidance=cfg.val_guidance, seed=cfg.val_seed))
            labels.append(f"ELLA L{cfg.context_tokens}: {ptxt[:40]}")
        grid_path = (f"{cfg.output_dir}/"
                     f"validation_ella_L{cfg.context_tokens}.png")
        save_validation_grid(
            ella_imgs, labels, grid_path,
            f"ELLA L{cfg.context_tokens}")
        fev_summary["validation_grid_saved"] = grid_path
        remember_final_summary(
            "final_eval", fev_summary,
            wandb_prefix="final/eval", wandb=state.wandb)
        print_final_summary("final_eval")

        if state.clip_model is not None:
            clip_imgs, clip_labels = [], []
            for ptxt in cfg.val_prompts:
                clip_imgs.append(generate_clip_teacher(
                    ptxt, state, steps=cfg.val_steps,
                    guidance=cfg.val_guidance, seed=cfg.val_seed))
                clip_labels.append("CLIP teacher")
            save_validation_grid(
                clip_imgs, clip_labels,
                f"{cfg.output_dir}/validation_clip_teacher.png",
                "CLIP teacher baseline")
    else:
        print("Validation grids skipped")


def run_save_artifacts(state: TrainingState):
    cfg = state.cfg
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = f"{cfg.output_dir}/pure_ella_connector_L{cfg.context_tokens}.pt"
    torch.save({
        "architecture": cfg.connector_type,
        "connector_type": cfg.connector_type,
        "connector_state_dict": {k: v.detach().cpu()
                                 for k, v in state.connector.state_dict().items()},
        "gemma_model_id": cfg.gemma_id,
        "sd_checkpoint": cfg.sd_checkpoint,
        "run_config": cfg.to_dict(),
    }, path)
    print("Connector saved:", path)
    summary = {"connector_saved": path, "context_tokens": cfg.context_tokens}

    # Save SaRA sparse UNet patch if active
    if cfg.run_sara_phase and state._sara_summary is not None:
        sparse_values = collect_sara_sparse_values(state.unet)
        sparse_path = f"{cfg.output_dir}/pure_ella_unet_attn2_kv_sparse_L{cfg.context_tokens}.pt"
        torch.save({
            "architecture": "SD UNet original graph + sparse attn2.to_k/to_v value patch",
            "sparse_values": sparse_values,
            "sara_sparse_summary": state._sara_summary,
            "sd_checkpoint": cfg.sd_checkpoint,
            "run_config": cfg.to_dict(),
        }, sparse_path)
        print("Sparse UNet patch saved:", sparse_path)
        summary["unet_sparse_saved"] = sparse_path
    else:
        print("Sparse UNet patch skipped")

    remember_final_summary(
        "save_artifacts", summary,
        wandb_prefix="final/save_artifacts", wandb=state.wandb)
    print_final_summary("save_artifacts")


def run_reload_proof(state: TrainingState):
    cfg = state.cfg
    if not cfg.run_final_proof:
        print("Final proof skipped")
        return

    path = f"{cfg.output_dir}/pure_ella_connector_L{cfg.context_tokens}.pt"
    print("Reloading connector from:", path)
    ckpt = torch.load(path, map_location="cpu")
    reloaded = build_connector(
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
        recursive_y_steps=cfg.recursive_y_steps,
        recursive_y_gate_init=cfg.recursive_y_gate_init,
        trm_outer_steps=cfg.trm_outer_steps,
        trm_inner_steps=cfg.trm_inner_steps,
        trm_scratch_tokens=cfg.trm_scratch_tokens,
        trm_y_gate_init=cfg.trm_y_gate_init,
        trm_z_gate_init=cfg.trm_z_gate_init,
    ).to(device=state.device, dtype=state.unet_dtype).eval()
    missing, unexpected = reloaded.load_state_dict(
        ckpt["connector_state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    print("Connector strict reload PASS")

    print("Reloading fresh SD checkpoint")
    sd_path = resolve_sd_checkpoint(cfg)
    if sd_path.endswith(".safetensors"):
        pipe = StableDiffusionPipeline.from_single_file(
            sd_path, torch_dtype=state.unet_dtype,
            token=os.environ.get("HF_TOKEN"))
        reloaded_unet = pipe.unet.to(state.device).eval()
        pipe = None
    else:
        reloaded_unet = UNet2DConditionModel.from_pretrained(
            sd_path, subfolder="unet", torch_dtype=state.unet_dtype,
            token=os.environ.get("HF_TOKEN"),
        ).to(state.device).eval()

    # Apply SaRA sparse UNet patch if saved
    sparse_path = f"{cfg.output_dir}/pure_ella_unet_attn2_kv_sparse_L{cfg.context_tokens}.pt"
    if cfg.run_sara_phase and os.path.exists(sparse_path):
        sparse_ckpt = torch.load(sparse_path, map_location="cpu")
        loaded = load_sara_sparse_values(reloaded_unet, sparse_ckpt["sparse_values"])
        print(f"Sparse UNet patch reload PASS: {loaded:,} values")

    with torch.no_grad():
        gh, gm = state.encode_gemma(["a small red car", ""])
        t = torch.tensor([500, 500], device=state.device).long()
        ctx = reloaded(gh.to(dtype=state.unet_dtype), t, gm,
                        context_tokens=cfg.context_tokens)
        test_latent = torch.randn(
            2, 4, 64, 64, device=state.device, dtype=state.unet_dtype)
        pred = reloaded_unet(test_latent, t, encoder_hidden_states=ctx).sample
        assert torch.isfinite(pred).all()
    print("Fresh reload finite forward PASS")
    reload_summary = {
        "strict_connector_reload": 1.0,
        "finite_forward": 1.0,
        "context_tokens": cfg.context_tokens,
    }
    remember_final_summary(
        "reload_proof", reload_summary,
        wandb_prefix="final/reload_proof", wandb=state.wandb)
    print_final_summary("reload_proof")

    old_connector = state.connector
    old_unet = state.unet
    state.connector = reloaded
    state.unet = reloaded_unet
    if state.clip_model is not None:
        del state.clip_model
        del state.clip_tokenizer
        state.clip_model = None
        state.clip_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        print("CLIP available during proof generation:",
              state.clip_model is not None)
        proof_imgs, proof_labels = [], []
        for ptxt in cfg.val_prompts:
            proof_imgs.append(generate_ella(
                ptxt, state, steps=cfg.val_steps,
                guidance=cfg.val_guidance, seed=cfg.val_seed))
            proof_labels.append(f"reloaded ELLA L{cfg.context_tokens}")
        proof_grid = (f"{cfg.output_dir}/"
                      f"proof_reloaded_pure_ella_L{cfg.context_tokens}.png")
        save_validation_grid(
            proof_imgs, proof_labels, proof_grid,
            "Reloaded pure ELLA Gemma-only proof")
        print("Final proof grid:", proof_grid)
        reload_summary["proof_grid_saved"] = proof_grid
        remember_final_summary(
            "reload_proof", reload_summary,
            wandb_prefix="final/reload_proof", wandb=state.wandb)
        print_final_summary("reload_proof")
    finally:
        state.connector = old_connector
        state.unet = old_unet
        if 'reloaded' in locals():
            del reloaded
        if 'reloaded_unet' in locals():
            del reloaded_unet
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Complex prompt checks (post-training)
# ---------------------------------------------------------------------------
def run_complex_prompt_checks(state: TrainingState):
    """Generate complex prompt grids after training, same as colab notebooks."""
    cfg = state.cfg
    if not cfg.run_complex_prompt_grids:
        print("Complex prompt grids skipped (run_complex_prompt_grids=False)")
        return

    ctx = cfg.context_tokens
    out = cfg.output_dir

    # Complex generation cases grid
    cmplx_path = f"{out}/validation_complex_prompts_L{ctx}.png"
    save_complex_case_grid(
        cfg.complex_generation_cases, cmplx_path,
        f"Complex natural-language prompts L{ctx}", state,
        context_tokens=ctx,
    )
    print(f"Complex prompt grid saved: {cmplx_path}")

    # Long-context eval: short vs long prompts paired
    long_imgs, long_labels = [], []
    for short, long in zip(cfg.long_eval_short_controls, cfg.long_eval_prompts):
        long_imgs.append(generate_ella(
            short, state,
            steps=cfg.val_steps, guidance=cfg.val_guidance,
            seed=cfg.val_seed, context_tokens=cfg.clip_anchor_tokens,
        ))
        long_labels.append(f"short L{cfg.clip_anchor_tokens}")
        long_imgs.append(generate_ella(
            long, state,
            steps=cfg.val_steps, guidance=cfg.val_guidance,
            seed=cfg.val_seed, context_tokens=ctx,
        ))
        long_labels.append(f"long L{ctx}")
    long_path = f"{out}/validation_long_context_L{ctx}.png"
    save_validation_grid(long_imgs, long_labels, long_path,
                         f"Long-context ELLA L{ctx}")
    print(f"Long-context grid saved: {long_path}")

    # Suffix counterfactual grids
    if cfg.run_suffix_counterfactual_grids and ctx > cfg.clip_anchor_tokens:
        sfx_out = save_suffix_counterfactual_grids(
            cfg.suffix_counterfactual_cases,
            f"{out}/validation_suffix_counterfactual_L{ctx}",
            "Suffix counterfactuals", state, context_tokens=ctx,
        )
        for p in sfx_out:
            print(f"Suffix counterfactual grid saved: {p}")
    elif ctx <= cfg.clip_anchor_tokens:
        print("Suffix counterfactual grids skipped at 77-token stage")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_connector_checkpoint(state: TrainingState, ckpt_path: str):
    """Load a connector checkpoint saved by run_clip_pretrain."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("connector_state_dict")
    if sd is None:
        print("Checkpoint has no connector_state_dict")
        return False
    missing, unexpected = state.connector.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARNING: missing keys ({len(missing)}): {missing[:5]}…")
    if unexpected:
        print(f"WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}…")
    stage = ckpt.get("stage", "unknown")
    print(f"Loaded connector from {ckpt_path} (stage={stage})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="pure-ella: Gemma -> SD training")
    parser.add_argument(
        "config", nargs="?", default="config.json",
        help="Path to JSON config file")
    parser.add_argument(
        "--config", dest="config_flag",
        help="Path to JSON config file (alternate flag)")
    parser.add_argument(
        "--phases", type=str, default="pretrain,ella,sara",
        help="Comma-separated phases to run: pretrain,ella,sara (default: all)")
    parser.add_argument(
        "--resume-ckpt", type=str, default="",
        help="Path to connector checkpoint to load before first phase")
    args = parser.parse_args()
    config_path = args.config_flag or args.config

    phases_requested = set(p.strip().lower() for p in args.phases.split(",") if p.strip())
    valid_phases = {"pretrain", "ella", "sara"}
    unknown = phases_requested - valid_phases
    if unknown:
        print(f"ERROR: unknown phases: {unknown}. Valid: {valid_phases}")
        sys.exit(1)
    print(f"Phases: {sorted(phases_requested)}")

    cfg = TrainConfig.from_json(config_path)
    seed_everything(cfg.base_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_dtype = torch.float32

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.wandb_enabled:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=f"pure-ella-{cfg.experiment_stage}-{cfg.run_mode}-L{cfg.context_tokens}",
            config=cfg.to_dict(),
        )
        wandb.define_metric("pretrain/step")
        wandb.define_metric("pretrain/*", step_metric="pretrain/step")
        wandb.define_metric("ella/step")
        wandb.define_metric("ella/*", step_metric="ella/step")
        wandb.define_metric("sara/step")
        wandb.define_metric("sara/*", step_metric="sara/step")
        wandb.define_metric("diagnostics/*")
        wandb.define_metric("quality/*")
        wandb.define_metric("final/*")
    else:
        class NoWandb:
            summary = {}
            def log(self, *a, **kw): pass
            def define_metric(self, *a, **kw): pass
            def finish(self): pass
        wandb = NoWandb()

    state = TrainingState(
        cfg=cfg, device=device, unet_dtype=unet_dtype, wandb=wandb)

    cfg.print_plan()

    load_gemma(state)
    if cfg.run_clip_alignment_pretrain or cfg.use_clip_teacher_delta:
        load_clip(state)
    load_stylejourney(state)
    state.encode_gemma = make_encode_gemma(state)
    if state.clip_model is not None:
        state.encode_clip = make_encode_clip(state)
    build_ella_connector(state)

    if cfg.run_mode == "overfit_train":
        dl = make_streaming_dataloader(
            cfg.stream_repo, phase=1, epoch=0,
            max_samples=max(cfg.max_samples_ella, 4), batch_size=4,
            shuffle=cfg.shuffle_streaming,
            shuffle_buffer=cfg.shuffle_buffer, base_seed=cfg.base_seed)
        batch = next(iter(dl))
        state.overfit_eval_batch = {
            "image": batch["image"],
            "caption": _as_prompt_list(batch["caption"]),
        }
        print("Exact overfit captions:")
        for i, p in enumerate(state.overfit_eval_batch["caption"]):
            print(f"  {i}: {p[:160]}")
    else:
        print("Non-overfit run: generic validation prompts active")

    # ── resume / phase selection ──
    resume_path = args.resume_ckpt or cfg.init_connector_ckpt_path
    if "pretrain" not in phases_requested:
        if resume_path:
            if not _load_connector_checkpoint(state, resume_path):
                print("ERROR: failed to load resume checkpoint, aborting")
                sys.exit(1)
        else:
            print("WARNING: skipping pretrain with no resume checkpoint - "
                  "connector weights are freshly initialized")
    else:
        if resume_path:
            print("NOTE: --resume-ckpt ignored because pretrain is in the phase list")

    run_clip_pretrain(state) if "pretrain" in phases_requested else print("pretrain SKIPPED")
    print_final_summary("clip_pretrain") if "pretrain" in phases_requested else None

    run_ella_training(state) if "ella" in phases_requested else print("ella SKIPPED")
    print_final_summary("ella_frozen_unet") if "ella" in phases_requested else None

    run_sara_training(state) if "sara" in phases_requested else print("sara SKIPPED")
    print_final_summary("sara_attn2_kv") if "sara" in phases_requested else None

    run_validation_grids(state)
    run_complex_prompt_checks(state)
    run_save_artifacts(state)
    run_reload_proof(state)

    if cfg.wandb_enabled:
        wandb.finish()
    print("Done. Training complete.")


if __name__ == "__main__":
    main()
