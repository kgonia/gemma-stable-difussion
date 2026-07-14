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
import os
import sys
import time
from contextlib import contextmanager
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
from pure_ella.camera import (
    CameraConditioner,
    apply_camera_dropout,
    camera_condition_schema,
    camera_conditioned_unet,
    install_camera_conditioner,
    make_camera_condition,
)
from pure_ella.dataset import make_streaming_dataloader
from pure_ella.prompts import (
    make_prompt_dataloader,
    sample_validation_prompts,
)
from pure_ella.sara import (
    build_sara_attn2_kv_sparse_masks,
    install_sara_gradient_masks,
    remove_sara_gradient_masks,
    collect_sara_sparse_values,
    capture_sara_selected_values,
    load_sara_sparse_values,
    sara_selected_delta_metrics,
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
    save_validation_grid,
    _image_collapse_stats,
    _pil_to_uint8_tensor,
    _log_metrics,
    _rel_diff,
    _to_float_maybe,
    suffix_counterfactual_sensitivity,
    camera_counterfactual_sensitivity,
    validate_suffix_counterfactual_token_boundaries,
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
    autocast_dtype: Optional[torch.dtype] = None

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
    camera_conditioner: nn.Module = None

    # Dataset
    overfit_eval_batch: dict = None
    ref_quality_cache: dict = None
    caption_availability_logged: bool = False
    camera_metadata_logged: bool = False
    camera_metadata_samples: int = 0
    camera_presence_counts: list = field(
        default_factory=lambda: [0, 0, 0, 0])
    gemma_prompt_max_observed: int = -1
    gemma_truncated_prompt_count: int = 0
    suffix_token_boundary_signature: tuple = None
    suffix_token_boundaries: dict = None

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


@dataclass(frozen=True)
class DiffusionPhaseSpec:
    """Phase-specific controls for the shared diffusion training loop."""
    name: str
    display_name: str
    plan_name: str
    data_phase: int
    epochs: int
    max_samples: int
    max_opt_steps: int
    semantic_anchor_weight: float
    use_teacher_delta: bool
    camera_dropout_probability: float


@dataclass
class DiffusionStepOutput:
    """Differentiable loss tensors returned by one shared training step."""
    loss: torch.Tensor
    loss_diff: torch.Tensor
    loss_teacher: torch.Tensor
    loss_delta: torch.Tensor
    loss_anchor: torch.Tensor
    loss_prefix: torch.Tensor
    loss_norm: torch.Tensor
    clip_scaffold_scale: float
    residual_norm_ratio: Optional[torch.Tensor] = None
    timesteps: Optional[torch.Tensor] = None

    def scalars(self, step: int) -> dict:
        return {
            "step": step,
            "loss": float(self.loss.detach().item()),
            "loss_diff": float(self.loss_diff.detach().item()),
            "loss_teacher": float(self.loss_teacher.detach().item()),
            "loss_delta": float(self.loss_delta.detach().item()),
            "loss_anchor": float(self.loss_anchor.detach().item()),
            "loss_prefix": float(self.loss_prefix.detach().item()),
            "loss_norm": float(self.loss_norm.detach().item()),
            "clip_scaffold_scale": self.clip_scaffold_scale,
        }


def estimate_steps(
    max_samples: int,
    batch_size: int,
    epochs: int,
    cap: Optional[int],
    drop_last: bool = False,
    bucket_count: int = 1,
) -> StagePlan:
    """Return an upper-bound plan that accounts for per-bucket tail batches."""
    if drop_last:
        steps_per_epoch = int(max_samples) // int(batch_size)
    else:
        active_buckets = min(
            max(int(bucket_count), 1),
            max(int(max_samples), 0),
        )
        remaining = max(int(max_samples) - active_buckets, 0)
        steps_per_epoch = (
            active_buckets + remaining // int(batch_size)
        )
    uncapped = steps_per_epoch * int(epochs)
    effective = min(uncapped, int(cap)) if cap is not None else uncapped
    return StagePlan(
        name="stage",
        steps_per_epoch=steps_per_epoch,
        uncapped=uncapped,
        effective=effective,
        image_exposures=effective * int(batch_size),
    )


def resolve_model_weight_dtype(cfg: TrainConfig) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[cfg.model_weight_dtype]


def resolve_autocast_dtype(cfg: TrainConfig, device: torch.device):
    if cfg.mixed_precision == "no":
        return None
    if device.type != "cuda":
        print(
            f"WARNING: mixed_precision={cfg.mixed_precision} requires CUDA; "
            "autocast disabled"
        )
        return None
    if cfg.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "mixed_precision='bf16' requested but this CUDA device does not "
            "support bfloat16; set mixed_precision='no'"
        )
    return torch.bfloat16


@contextmanager
def model_autocast(state: TrainingState):
    """Autocast forward operations without lowering parameter precision."""
    if state.autocast_dtype is None:
        yield
        return
    with torch.autocast(
        device_type=state.device.type,
        dtype=state.autocast_dtype,
    ):
        yield


def masked_mse(prediction: torch.Tensor, target: torch.Tensor,
               image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE over real image content, excluding full-frame bucket padding."""
    error = (prediction.float() - target.float()).pow(2)
    if image_mask is None:
        return error.mean()
    mask = F.interpolate(
        image_mask.float(), size=prediction.shape[-2:], mode="nearest")
    mask = mask.to(device=prediction.device, dtype=error.dtype)
    denominator = mask.sum().clamp_min(1.0) * prediction.shape[1]
    return (error * mask).sum() / denominator


def apply_conditioning_dropout(captions, probability: float):
    captions = _as_prompt_list(captions)
    if probability <= 0:
        return captions
    dropped = torch.rand(len(captions)) < probability
    return ["" if bool(drop) else caption
            for caption, drop in zip(captions, dropped)]


def select_training_captions(batch: dict, state: TrainingState):
    """Sample caption lengths with nearest-length fallback."""
    cfg = state.cfg
    long_captions = _as_prompt_list(batch["caption"])
    medium_captions = _as_prompt_list(
        batch.get("caption_medium", [""] * len(long_captions)))
    short_captions = _as_prompt_list(
        batch.get("caption_short", [""] * len(long_captions)))
    if not state.caption_availability_logged:
        short_count = sum(bool(caption) for caption in short_captions)
        medium_count = sum(bool(caption) for caption in medium_captions)
        print(
            f"Caption variants in first batch: short={short_count}/"
            f"{len(long_captions)} medium={medium_count}/{len(long_captions)} "
            f"long={len(long_captions)}/{len(long_captions)}"
        )
        if short_count == 0 and medium_count == 0:
            print("WARNING: caption curriculum unavailable; using long captions only")
        state.caption_availability_logged = True
    choices = torch.rand(len(long_captions))
    short_cutoff = cfg.caption_mix_short
    medium_cutoff = short_cutoff + cfg.caption_mix_medium
    selected = []
    for value, short_caption, medium_caption, long_caption in zip(
            choices, short_captions, medium_captions, long_captions):
        if value < short_cutoff:
            candidates = (short_caption, medium_caption, long_caption)
        elif value < medium_cutoff:
            candidates = (medium_caption, short_caption, long_caption)
        else:
            candidates = (long_caption, medium_caption, short_caption)
        selected.append(next(caption for caption in candidates if caption))
    return selected


def clip_visible_prompts(state: TrainingState, captions):
    """Return the exact textual prefix visible to CLIP's 77-token encoder."""
    tokens = state.clip_tokenizer(
        _as_prompt_list(captions),
        padding=False,
        truncation=True,
        max_length=state.cfg.clip_anchor_tokens,
    )
    return state.clip_tokenizer.batch_decode(
        tokens["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def clip_scaffold_scale(cfg: TrainConfig, step: int) -> float:
    if cfg.clip_teacher_decay_steps <= 0:
        return 1.0
    return max(0.0, 1.0 - step / cfg.clip_teacher_decay_steps)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_gemma(state: TrainingState):
    """Load frozen Gemma 3 model and tokenizer."""
    cfg = state.cfg
    print(f"Loading Gemma: {cfg.gemma_id}")
    state.gemma_tokenizer = AutoTokenizer.from_pretrained(
        cfg.gemma_id, token=os.environ.get("HF_TOKEN"))
    if (cfg.max_gemma_len > cfg.clip_anchor_tokens
            and (cfg.run_long_context_diagnostics
                 or cfg.run_suffix_counterfactual_grids)):
        validate_suffix_counterfactual_token_boundaries(state)
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
    if cfg.use_sd_checkpoint_text_encoder:
        checkpoint = resolve_sd_checkpoint(cfg)
        if not checkpoint.endswith(".safetensors"):
            raise ValueError("use_sd_checkpoint_text_encoder requires a single-file safetensors checkpoint")
        from safetensors import safe_open
        prefixes = ("cond_stage_model.transformer.",
                    "conditioner.embedders.0.transformer.", "text_encoder.")
        expected = state.clip_model.state_dict()

        def compatible_clip_key(raw_name: str) -> str:
            """Map SD/Transformers 4 and 5 CLIP namespaces to this model."""
            candidates = [raw_name]
            if raw_name.startswith("text_model."):
                candidates.append(raw_name.removeprefix("text_model."))
            else:
                candidates.append(f"text_model.{raw_name}")
            return next((name for name in candidates if name in expected), raw_name)

        checkpoint_state = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                prefix = next((item for item in prefixes if key.startswith(item)), None)
                if prefix is not None:
                    name = compatible_clip_key(key[len(prefix):])
                    checkpoint_state[name] = handle.get_tensor(key)
        if not checkpoint_state:
            raise RuntimeError(f"No CLIP text encoder found in {checkpoint}")
        incompatible = [name for name, value in checkpoint_state.items()
                        if name in expected and expected[name].shape != value.shape]
        if incompatible:
            raise RuntimeError(f"Checkpoint CLIP tensor shape mismatch: {incompatible[:5]}")
        missing, unexpected = state.clip_model.load_state_dict(
            checkpoint_state, strict=False)
        is_position_id = lambda name: name.endswith("embeddings.position_ids")
        missing = [name for name in missing if not is_position_id(name)]
        unexpected = [
            name for name in unexpected
            if name != "text_projection.weight" and not is_position_id(name)
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint CLIP load incomplete; missing={missing[:5]} unexpected={unexpected[:5]}")
        print("Loaded CLIP text encoder weights from SD checkpoint")
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
            text_encoder=None,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
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
        "gemma_layer_mix_count": cfg.gemma_layer_mix_count,
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
        g = torch.randn(
            2, cfg.gemma_layer_mix_count, 32, state.gemma_hidden_size,
            device=state.device, dtype=state.unet_dtype)
        m = torch.ones(2, 32, device=state.device, dtype=torch.long)
        t = torch.tensor([10, 500], device=state.device).long()
        if is_residual_connector(cfg):
            clip = torch.randn(2, cfg.context_tokens, 768,
                               device=state.device, dtype=state.unet_dtype)
            y = state.connector(clip, g, t, m)
            assert torch.equal(y, torch.zeros_like(y)), "residual must start at zero"
        else:
            y = state.connector(g, t, m)
        assert y.shape == (2, cfg.context_tokens, 768), y.shape
        assert torch.isfinite(y).all()

    n_params = sum(p.numel() for p in state.connector.parameters())
    print(f"Connector PASS: output={tuple(y.shape)}, params={n_params:,}")
    eg = state.connector.extra_gate_logit
    if eg is not None:
        print("extra_token_gate:", torch.sigmoid(eg).item())


def build_camera_conditioner(state: TrainingState):
    """Install P3 as an additive SD timestep-embedding condition."""
    if not state.cfg.camera_conditioning_enabled:
        return
    time_embed_dim = state.unet.time_embedding.linear_2.out_features
    state.camera_conditioner = CameraConditioner(
        output_dim=time_embed_dim,
        hidden_dim=state.cfg.camera_hidden_dim,
        fourier_bands=state.cfg.camera_fourier_bands,
        enable_geometry_head=state.cfg.camera_enable_geometry_head,
        enable_raw_focal_head=state.cfg.camera_enable_raw_focal_head,
        enable_exposure_head=state.cfg.camera_enable_exposure_head,
        use_capture_type=state.cfg.camera_use_capture_type,
    ).to(device=state.device, dtype=state.unet_dtype)
    install_camera_conditioner(state.unet, state.camera_conditioner)
    with torch.no_grad():
        unknown = state.camera_conditioner.unknown(2, state.device)
        output = state.camera_conditioner(unknown)
        assert torch.count_nonzero(output).item() == 0
    count = sum(p.numel() for p in state.camera_conditioner.parameters())
    print(
        f"P3 camera conditioner PASS: output={time_embed_dim} params={count:,} "
        "permanent unknown identity=PASS"
    )


def _camera_schema(state: TrainingState) -> dict | None:
    if state.camera_conditioner is None:
        return None
    return camera_condition_schema(
        enable_geometry_head=state.cfg.camera_enable_geometry_head,
        enable_raw_focal_head=state.cfg.camera_enable_raw_focal_head,
        enable_exposure_head=state.cfg.camera_enable_exposure_head,
        use_capture_type=state.cfg.camera_use_capture_type)


def _validate_camera_checkpoint_schema(
    checkpoint: dict, state: TrainingState, source: str,
) -> None:
    expected = _camera_schema(state)
    actual = checkpoint.get("camera_condition_schema")
    if actual != expected:
        raise RuntimeError(
            f"Camera condition schema mismatch in {source}: "
            f"expected {expected!r}, got {actual!r}"
        )


def connector_checkpoint_schema(state: TrainingState) -> dict:
    """Versioned architecture identity; prevents silent pure/residual mixes."""
    cfg = state.cfg
    return {
        # The residual is meaningful only in the CLIP coordinate system and at
        # the composition scale it was trained with.  Keep these provenance
        # fields in the strict schema so a P2 resume cannot silently swap the
        # StyleJourney text encoder for stock OpenAI CLIP (or alter delta gain).
        "version": 2,
        "connector_type": cfg.connector_type,
        "context_tokens": cfg.context_tokens,
        "clip_anchor_tokens": cfg.clip_anchor_tokens,
        "connector_width": cfg.connector_width,
        "connector_layers": cfg.connector_layers,
        "connector_heads": cfg.connector_heads,
        "connector_ff_mult": cfg.connector_ff_mult,
        "connector_time_embed_dim": cfg.connector_time_embed_dim,
        "connector_dropout": cfg.connector_dropout,
        "gemma_dim": state.gemma_hidden_size,
        "gemma_id": cfg.gemma_id,
        "gemma_layer_index": cfg.gemma_layer_index,
        "gemma_layer_mix_count": cfg.gemma_layer_mix_count,
        "clip_id": cfg.clip_id,
        "use_sd_checkpoint_text_encoder": cfg.use_sd_checkpoint_text_encoder,
        "sd_checkpoint": os.path.abspath(os.path.expanduser(cfg.sd_checkpoint)),
        "residual": is_residual_connector(cfg),
        "residual_strength": cfg.residual_strength,
    }


def validate_connector_checkpoint_schema(checkpoint: dict, state: TrainingState,
                                         source: str) -> None:
    actual = checkpoint.get("connector_schema")
    expected = connector_checkpoint_schema(state)
    if actual is None:
        if is_residual_connector(state.cfg):
            raise RuntimeError(f"Residual checkpoint {source} has no connector_schema")
        return  # Legacy pure checkpoints remain loadable.
    # Schema v1 predates residual CLIP-basis/scale provenance.  Continue to
    # accept it only for pure connectors, whose old contract it fully records.
    # Residual v1 checkpoints must be rejected because their CLIP basis and
    # residual strength cannot be reconstructed safely for a P2 resume.
    if (
        actual.get("version") == 1
        and not is_residual_connector(state.cfg)
        and not actual.get("residual", False)
    ):
        legacy_expected = expected.copy()
        legacy_expected.update({"version": 1})
        for key in (
            "use_sd_checkpoint_text_encoder", "sd_checkpoint",
            "residual_strength",
        ):
            legacy_expected.pop(key)
        if actual == legacy_expected:
            return
    if actual != expected:
        raise RuntimeError(
            f"Connector checkpoint schema mismatch for {source}: "
            f"expected {expected}, got {actual}")


def save_residual_step_checkpoint(state: TrainingState, step: int) -> str:
    """Save frequent, schema-tagged residual weights for drift recovery."""
    path = os.path.join(state.cfg.output_dir, f"residual_connector_step_{step:06d}.pt")
    torch.save({
        "connector_schema": connector_checkpoint_schema(state),
        "connector_state_dict": {k: v.detach().cpu()
                                  for k, v in state.connector.state_dict().items()},
        "config": state.cfg.to_dict(),
        "stage": "clip_gemma_residual_frozen_unet",
        "optimizer_step": step,
    }, path)
    return path


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
    def encode_gemma(prompts, max_length=None, pad_to_max_length=True):
        max_length = max_length or state.cfg.max_gemma_len
        prompts = _as_prompt_list(prompts)
        encoded = state.gemma_tokenizer(
            prompts, padding=False, truncation=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        longest = max(lengths, default=0)
        if longest > state.gemma_prompt_max_observed:
            print(
                f"Gemma prompt tokens: batch_min={min(lengths, default=0)} "
                f"batch_max={longest} observed_max={longest} limit={max_length}"
            )
            state.gemma_prompt_max_observed = longest
        overflow_count = sum(length > max_length for length in lengths)
        if overflow_count and state.cfg.fail_on_prompt_truncation:
            index = lengths.index(longest)
            raise ValueError(
                f"Gemma prompt has {longest} tokens but max_gemma_len={max_length}; "
                f"refusing silent truncation: {prompts[index][:160]!r}"
            )
        if overflow_count:
            state.gemma_truncated_prompt_count += overflow_count
            if state.gemma_truncated_prompt_count == overflow_count:
                print(
                    "WARNING: permissive Gemma prompt truncation is enabled; "
                    f"truncating {overflow_count} prompt(s) to {max_length} tokens"
                )
            truncate = (
                (lambda sequence: sequence[-max_length:])
                if state.gemma_tokenizer.truncation_side == "left"
                else (lambda sequence: sequence[:max_length])
            )
            encoded = {
                name: [truncate(sequence) for sequence in sequences]
                for name, sequences in encoded.items()
            }
        toks = state.gemma_tokenizer.pad(
            encoded,
            padding=("max_length" if pad_to_max_length else "longest"),
            max_length=max_length,
            return_tensors="pt",
        ).to(state.gemma_model.device)
        out = state.gemma_model(**toks, output_hidden_states=True,
                                use_cache=False)
        layers = out.hidden_states
        end = state.cfg.gemma_layer_index
        end = end if end >= 0 else len(layers) + end
        if end < 0 or end >= len(layers):
            raise IndexError(
                f"GEMMA_LAYER_INDEX={state.cfg.gemma_layer_index} "
                f"invalid for {len(layers)} hidden states")
        start = end - state.cfg.gemma_layer_mix_count + 1
        if start < 0:
            raise IndexError(
                f"Cannot mix {state.cfg.gemma_layer_mix_count} layers ending "
                f"at hidden-state index {end}"
            )
        h = torch.stack(layers[start:end + 1], dim=1).to(device=state.device)
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


def is_residual_connector(cfg: TrainConfig) -> bool:
    return cfg.connector_type == "clip_gemma_residual_tsc"


def prompt_exceeds_clip_window(state: TrainingState, prompts) -> torch.Tensor:
    """Return the non-truncating CLIP length predicate used by the hard bypass."""
    prompts = _as_prompt_list(prompts)
    encoded = state.clip_tokenizer(prompts, padding=False, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    return torch.tensor(lengths, device=state.device) > state.cfg.clip_anchor_tokens


def _clip_visible_prefixes(state: TrainingState, prompts) -> list[str]:
    """Decode the same CLIP-visible token prefix used for the prefix-null loss."""
    # Let CLIP's tokenizer reserve its EOS slot.  Manual ``ids[:77]`` keeps
    # BOS plus 76 content tokens and omits EOS, while the real CLIP encoder
    # sees BOS + 75 content tokens + EOS.
    encoded = state.clip_tokenizer(
        _as_prompt_list(prompts), padding="max_length", truncation=True,
        max_length=state.cfg.clip_anchor_tokens)
    return [state.clip_tokenizer.decode(ids,
                                        skip_special_tokens=True)
            for ids in encoded["input_ids"]]


def residual_context_for_captions(state: TrainingState, captions, timesteps):
    """Build residual conditioning with a control-flow (not masked) short bypass.

    Returns context plus the full/prefix deltas for regularization.  Gemma and
    the connector are never invoked for an all-short batch.
    """
    clip_context, _ = state.encode_clip(captions)
    context = clip_context.clone()
    long_mask = prompt_exceeds_clip_window(state, captions)
    zero = context.new_zeros(())
    if not bool(long_mask.any()):
        return context, zero, zero, long_mask
    idx = long_mask.nonzero(as_tuple=True)[0]
    long_captions = [captions[i] for i in idx.tolist()]
    gemma_h, gemma_mask = state.encode_gemma(long_captions)
    delta_full = state.connector(
        clip_context[idx].to(dtype=state.unet_dtype),
        gemma_h.to(dtype=state.unet_dtype), timesteps[idx], gemma_mask,
        context_tokens=state.cfg.context_tokens)
    context[idx] = clip_context[idx] + state.cfg.residual_strength * delta_full

    # The prefix control is only needed for loss construction.  It intentionally
    # has the same CLIP queries and timestep as the full prompt.
    prefix_prompts = _clip_visible_prefixes(state, long_captions)
    prefix_h, prefix_mask = state.encode_gemma(prefix_prompts)
    delta_prefix = state.connector(
        clip_context[idx].to(dtype=state.unet_dtype),
        prefix_h.to(dtype=state.unet_dtype), timesteps[idx], prefix_mask,
        context_tokens=state.cfg.context_tokens)
    return context, delta_full, delta_prefix, long_mask


def normalized_residual_square(delta: torch.Tensor, clip_context: torch.Tensor) -> torch.Tensor:
    """Residual energy normalized to the native CLIP RMS scale."""
    return delta.float().square().mean() / clip_context.float().square().mean().clamp_min(1e-8)


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


@torch.no_grad()
def evaluate_clip_pretrain(
    state: TrainingState,
    clip_geom: ClipGeometryLoss,
    validation_sets: dict[str, list[str]],
    step: int,
) -> dict[str, float]:
    """Evaluate CLIP geometry on fixed prompt sets without image decoding."""
    if not validation_sets:
        return {}
    was_training = state.connector.training
    state.connector.eval()
    batch_size = state.cfg.pretrain_batch_size or state.cfg.train_batch_size
    aggregate = {
        name: 0.0
        for name in ("loss", "mse", "cos", "norm", "ctr", "pooled_cos",
                     "norm_ratio")
    }
    aggregate_count = 0
    result: dict[str, float] = {}
    validation_timesteps = state.cfg.pretrain_validation_timesteps
    timestep_sums = {
        timestep: {name: 0.0 for name in aggregate}
        for timestep in validation_timesteps
    }
    timestep_counts = {timestep: 0 for timestep in validation_timesteps}
    gemma_max_length = (
        state.cfg.pretrain_max_gemma_len or state.cfg.max_gemma_len)
    for source_name, raw_prompts in validation_sets.items():
        source_sums = {name: 0.0 for name in aggregate}
        source_count = 0
        for offset in range(0, len(raw_prompts), batch_size):
            prompt_batch = clip_visible_prompts(
                state, raw_prompts[offset:offset + batch_size])
            gh, gm = state.encode_gemma(
                prompt_batch, max_length=gemma_max_length,
                pad_to_max_length=False)
            with model_autocast(state):
                ch, cm = state.encode_clip(prompt_batch)
            token_mask = torch.ones_like(cm)
            for timestep in validation_timesteps:
                timesteps = torch.full(
                    (len(prompt_batch),), timestep,
                    device=state.device, dtype=torch.long)
                with model_autocast(state):
                    prediction = state.connector(
                        gh.to(dtype=state.unet_dtype), timesteps, gm,
                        context_tokens=state.cfg.clip_anchor_tokens,
                    )
                loss_values = clip_geom(
                    prediction, ch, token_mask, pool_mask=cm)
                count = len(prompt_batch)
                values = {
                    "loss": loss_values["total"].item(),
                    "mse": loss_values["mse"].item(),
                    "cos": loss_values["cos"].item(),
                    "norm": loss_values["norm"].item(),
                    "ctr": loss_values["ctr"].item(),
                    "pooled_cos": loss_values["pooled_cos"].item(),
                    "norm_ratio": loss_values["norm_ratio"].item(),
                }
                for name, value in values.items():
                    weighted = value * count
                    source_sums[name] += weighted
                    aggregate[name] += weighted
                    timestep_sums[timestep][name] += weighted
                source_count += count
                aggregate_count += count
                timestep_counts[timestep] += count
        for name, value in source_sums.items():
            result[f"{source_name}_{name}"] = value / source_count

    for name, value in aggregate.items():
        result[name] = value / aggregate_count
    for timestep, sums in timestep_sums.items():
        for name, value in sums.items():
            result[f"t{timestep:04d}_{name}"] = (
                value / timestep_counts[timestep])
    payload = {
        "pretrain_val/step": step,
        **{f"pretrain_val/{name}": value for name, value in result.items()},
    }
    safe_wandb_log(payload, wandb=state.wandb)
    print(
        f"pretrain validation step {step}: loss={result['loss']:.5f} "
        f"mse={result['mse']:.5f} cos={result['cos']:.5f} "
        f"pooled_cos={result['pooled_cos']:.5f} "
        f"norm_ratio={result['norm_ratio']:.3f}"
    )
    state.connector.train(was_training)
    return result


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

    if state.unet is not None:
        for p in state.unet.parameters():
            p.requires_grad_(False)
    if state.vae is not None:
        for p in state.vae.parameters():
            p.requires_grad_(False)
    for p in state.gemma_model.parameters(): p.requires_grad_(False)
    for p in state.clip_model.parameters(): p.requires_grad_(False)
    for p in state.connector.parameters(): p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        state.connector.parameters(), lr=cfg.pretrain_lr,
        weight_decay=0.01, eps=1e-6)
    state.connector.train()
    state.gemma_model.eval()
    state.clip_model.eval()
    if state.unet is not None:
        state.unet.eval()
    if state.vae is not None:
        state.vae.eval()

    pretrain_batch_size = cfg.pretrain_batch_size or cfg.train_batch_size
    prompt_only = bool(cfg.pretrain_prompt_sources)
    gemma_max_length = cfg.pretrain_max_gemma_len or cfg.max_gemma_len
    plan = estimate_steps(
        cfg.max_samples_pretrain,
        pretrain_batch_size,
        cfg.pretrain_epochs,
        cfg.pretrain_max_opt_steps,
        drop_last=(False if prompt_only else cfg.drop_last_bucket_batches),
        bucket_count=(1 if prompt_only else len(cfg.aspect_ratio_buckets)),
    )
    plan.name = "clip_pretrain"
    plan.print()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_vram("clip_pretrain_start", 0, state)

    opt_step = 0
    best = None
    last_pretrain = {}
    validation_sets = (
        sample_validation_prompts(
            cfg.pretrain_validation_prompt_sources,
            cfg.pretrain_validation_samples_per_source,
            cfg.base_seed + 20_000,
        )
        if cfg.pretrain_validation_prompt_sources else {}
    )
    if validation_sets:
        print(
            "Phase 0 held-out prompt sets: "
            + ", ".join(
                f"{name}={len(prompts)}"
                for name, prompts in validation_sets.items()
            )
        )
        evaluate_clip_pretrain(state, clip_geom, validation_sets, step=0)
    best_validation_loss = float("inf")
    best_validation_step = 0
    best_validation_metrics: dict[str, float] = {}
    best_validation_state = None
    last_validation_step = 0
    stop = False
    t0 = time.time()

    for epoch in range(cfg.pretrain_epochs):
        if prompt_only:
            dl = make_prompt_dataloader(
                cfg.pretrain_prompt_sources,
                epoch=epoch,
                max_samples=cfg.max_samples_pretrain,
                batch_size=pretrain_batch_size,
                shuffle=cfg.shuffle_streaming,
                shuffle_buffer=cfg.shuffle_buffer,
                base_seed=cfg.base_seed,
            )
        else:
            dl = make_streaming_dataloader(
                cfg.data_sources, phase=10, epoch=epoch,
                max_samples=cfg.max_samples_pretrain,
                batch_size=pretrain_batch_size,
                shuffle=cfg.shuffle_streaming,
                shuffle_buffer=cfg.shuffle_buffer, base_seed=cfg.base_seed,
                buckets=cfg.aspect_ratio_buckets,
                drop_last=cfg.drop_last_bucket_batches,
                max_image_dimension=cfg.max_image_dimension)
        progress = tqdm(dl, desc=f"CLIP-pretrain {epoch+1}/{cfg.pretrain_epochs}")
        for batch in progress:
            if prompt_only:
                phase0_sources = _as_prompt_list(batch)
                supplied_short = phase0_sources
            else:
                long_captions = _as_prompt_list(batch["caption"])
                supplied_short = _as_prompt_list(batch.get(
                    "caption_short", [""] * len(long_captions)))
                phase0_sources = [short or long for short, long in zip(
                    supplied_short, long_captions)]
            captions = clip_visible_prompts(state, phase0_sources)
            if epoch == 0 and opt_step == 0:
                if prompt_only:
                    print(
                        f"Phase 0 prompt-only batch={len(captions)}; all inputs "
                        "are reduced to the exact CLIP-visible decoded prefix"
                    )
                else:
                    supplied = sum(bool(caption) for caption in supplied_short)
                    print(
                        f"Phase 0 short captions supplied={supplied}/{len(captions)}; "
                        "all inputs are reduced to the exact CLIP-visible decoded prefix"
                    )
            with torch.no_grad():
                gh, gm = state.encode_gemma(
                    captions, max_length=gemma_max_length,
                    pad_to_max_length=False)
                with model_autocast(state):
                    ch, cm = state.encode_clip(captions)
            if cfg.pretrain_timestep_sampling == "uniform":
                t = torch.randint(
                    cfg.pretrain_num_train_timesteps,
                    (len(captions),), device=state.device, dtype=torch.long)
            else:
                t = torch.zeros(
                    len(captions), device=state.device, dtype=torch.long)
            with model_autocast(state):
                pred = state.connector(
                    gh.to(dtype=state.unet_dtype), t, gm,
                    context_tokens=cfg.clip_anchor_tokens)
            ld = clip_geom(pred, ch, torch.ones_like(cm), pool_mask=cm)
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
                "timestep_mean": t.float().mean().item(),
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
                    "pretrain/timestep_mean": t.float().mean().item(),
                }, wandb=state.wandb)

            validation_interval = cfg.pretrain_validation_every_opt_steps
            if (validation_sets and validation_interval > 0
                    and opt_step % validation_interval == 0):
                validation_metrics = evaluate_clip_pretrain(
                    state, clip_geom, validation_sets, step=opt_step)
                last_validation_step = opt_step
                if validation_metrics["loss"] < best_validation_loss:
                    best_validation_loss = validation_metrics["loss"]
                    best_validation_step = opt_step
                    best_validation_metrics = validation_metrics
                    best_validation_state = {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in state.connector.state_dict().items()
                    }

            progress.set_postfix(
                {"loss": f"{loss.item():.4f}", "best": f"{best:.4f}"})
            if (cfg.pretrain_max_opt_steps
                    and opt_step >= cfg.pretrain_max_opt_steps):
                stop = True
                print("Stopping CLIP pretrain at step cap", opt_step)
                break
        if stop:
            break

    if validation_sets and last_validation_step != opt_step:
        validation_metrics = evaluate_clip_pretrain(
            state, clip_geom, validation_sets, step=opt_step)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_validation_step = opt_step
            best_validation_metrics = validation_metrics
            best_validation_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in state.connector.state_dict().items()
            }

    elapsed_min = (time.time() - t0) / 60
    prompt_exposures = opt_step * pretrain_batch_size
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
        "connector_params": sum(
            parameter.numel() for parameter in state.connector.parameters()),
        "prompt_only": int(prompt_only),
        "prompt_exposures": prompt_exposures,
        "prompts_per_second": (
            prompt_exposures / max(elapsed_min * 60.0, 1e-6)),
    }
    if best_validation_metrics:
        pretrain_summary.update({
            "best_validation_step": best_validation_step,
            **{
                f"best_validation_{name}": value
                for name, value in best_validation_metrics.items()
            },
        })
    remember_final_summary(
        "clip_pretrain", pretrain_summary,
        wandb_prefix="final/clip_pretrain", wandb=state.wandb)
    print(f"CLIP pretrain done: steps={opt_step} best={best} "
          f"elapsed={elapsed_min:.1f}m")
    print_final_summary("clip_pretrain")

    ckpt_path = f"{cfg.output_dir}/ella_connector_clip_pretrain.pt"
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save({
        "connector_schema": connector_checkpoint_schema(state),
        "connector_state_dict": {k: v.detach().cpu()
                                 for k, v in state.connector.state_dict().items()},
        "camera_conditioner_state_dict": (
            {k: v.detach().cpu() for k, v in
             state.camera_conditioner.state_dict().items()}
            if state.camera_conditioner is not None else None
        ),
        "camera_condition_schema": _camera_schema(state),
        "config": cfg.to_dict(),
        "stage": "clip_alignment_pretrain",
    }, ckpt_path)
    if best_validation_state is not None:
        best_ckpt_path = (
            f"{cfg.output_dir}/ella_connector_clip_pretrain_best.pt")
        torch.save({
            "connector_state_dict": best_validation_state,
            "camera_conditioner_state_dict": None,
            "camera_condition_schema": _camera_schema(state),
            "config": cfg.to_dict(),
            "stage": "clip_alignment_pretrain_best_validation",
            "validation_step": best_validation_step,
            "validation_metrics": best_validation_metrics,
        }, best_ckpt_path)
        print(
            f"Best held-out Phase 0 checkpoint: {best_ckpt_path} "
            f"step={best_validation_step} "
            f"loss={best_validation_loss:.5f}"
        )

    if state.unet is None:
        print("Fast prompt-only Phase 0: U-Net post-diagnostics skipped")
        post_sens = float("nan")
        post_delta = {}
    else:
        post_sens = connector_prompt_sensitivity(
            cfg.val_prompts, state, label="post_clip_pretrain",
            wandb=state.wandb)
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
# Shared diffusion training step and loop
# ---------------------------------------------------------------------------
def diffusion_training_step(
    state: TrainingState,
    batch: dict,
    opt_step: int,
    *,
    use_teacher_delta: bool,
    semantic_anchor_weight: float,
    camera_dropout_probability: float = 0.0,
    clip_geom: Optional[ClipGeometryLoss] = None,
) -> DiffusionStepOutput:
    """Run one Phase 1/2 forward pass and assemble the common loss."""
    cfg = state.cfg
    captions = apply_conditioning_dropout(
        select_training_captions(batch, state),
        cfg.conditioning_dropout_prob,
    )
    image = batch["image"].to(
        device=state.device,
        dtype=state.unet_dtype,
    )
    image_mask = batch.get("image_mask")
    if image_mask is not None:
        image_mask = image_mask.to(device=state.device)
    camera_condition = None
    if cfg.camera_conditioning_enabled:
        raw_camera_condition = batch["camera_condition"]
        present = raw_camera_condition[:, 4:8].sum(dim=0).tolist()
        state.camera_metadata_samples += raw_camera_condition.shape[0]
        state.camera_presence_counts = [
            old + int(new)
            for old, new in zip(state.camera_presence_counts, present)
        ]
        if not state.camera_metadata_logged:
            print(
                "Camera metadata in first batch: "
                f"fov={int(present[0])}/{len(captions)} "
                f"focal={int(present[1])}/{len(captions)} "
                f"aperture={int(present[2])}/{len(captions)} "
                f"iso={int(present[3])}/{len(captions)}"
            )
            state.camera_metadata_logged = True
        camera_condition = raw_camera_condition.to(
            device=state.device, dtype=torch.float32)
        camera_condition = apply_camera_dropout(
            camera_condition, camera_dropout_probability)

    with torch.no_grad():
        with model_autocast(state):
            latent = (
                state.vae.encode(image).latent_dist.sample()
                * state.vae.config.scaling_factor
            )
        noise = torch.randn_like(latent)
        timestep = torch.randint(
            0,
            state.scheduler.config.num_train_timesteps,
            (latent.shape[0],),
            device=state.device,
        ).long()
        noisy = state.scheduler.add_noise(latent, noise, timestep)
        # Residual conditioning obtains Gemma states only for long prompts
        # below.  This is essential: a zero mask would still run Gemma and can
        # turn 0 * NaN into a non-identity short-prompt result.
        if not is_residual_connector(cfg):
            gemma_h, gemma_mask = state.encode_gemma(captions)

    loss_teacher = noise.new_tensor(0.0)
    loss_delta = noise.new_tensor(0.0)
    loss_anchor = noise.new_tensor(0.0)
    loss_prefix = noise.new_tensor(0.0)
    loss_norm = noise.new_tensor(0.0)
    residual_norm_ratio = None

    with model_autocast(state):
        if is_residual_connector(cfg):
            context, delta_full, delta_prefix, long_mask = residual_context_for_captions(
                state, captions, timestep)
            student_cond = camera_conditioned_unet(
                state.unet, noisy, timestep,
                encoder_hidden_states=context.to(dtype=state.unet_dtype),
                camera_condition=camera_condition).sample
            loss_diff = masked_mse(student_cond, noise, image_mask)
            if bool(long_mask.any()):
                clip_long = context[long_mask].detach() - (
                    cfg.residual_strength * delta_full).detach()
                loss_prefix = normalized_residual_square(delta_prefix, clip_long)
                loss_norm = normalized_residual_square(delta_full, clip_long)
                residual_norm_ratio = (
                    delta_full.float().square().mean(dim=(1, 2)).sqrt()
                    / clip_long.float().square().mean(dim=(1, 2)).sqrt().clamp_min(1e-8))
            scaffold_scale = 1.0
        elif use_teacher_delta:
            empty = [""] * len(captions)
            with torch.no_grad():
                uncond_h, uncond_mask = state.encode_gemma(empty)
            noisy_pair = torch.cat([noisy, noisy], dim=0)
            timestep_pair = torch.cat([timestep, timestep], dim=0)
            gemma_pair = torch.cat([gemma_h, uncond_h], dim=0)
            mask_pair = torch.cat([gemma_mask, uncond_mask], dim=0)
            context = state.connector(
                gemma_pair.to(dtype=state.unet_dtype),
                timestep_pair,
                mask_pair,
                context_tokens=cfg.context_tokens,
            )
            camera_pair = torch.cat(
                [camera_condition, camera_condition], dim=0
            ) if camera_condition is not None else None
            student_pair = camera_conditioned_unet(
                state.unet,
                noisy_pair,
                timestep_pair,
                encoder_hidden_states=context,
                camera_condition=camera_pair,
            ).sample
            student_cond, student_uncond = student_pair.chunk(2)
            loss_diff = masked_mse(student_cond, noise, image_mask)

            with torch.no_grad():
                clip_h, _ = state.encode_clip(captions)
                uncond_clip_h, _ = state.encode_clip(empty)
                clip_pair = torch.cat([clip_h, uncond_clip_h], dim=0)
                teacher_pair = camera_conditioned_unet(
                    state.unet,
                    noisy_pair,
                    timestep_pair,
                    encoder_hidden_states=clip_pair.to(
                        dtype=state.unet_dtype
                    ),
                    camera_condition=camera_pair,
                ).sample.detach()
                teacher_cond, teacher_uncond = teacher_pair.chunk(2)
                teacher_delta = teacher_cond - teacher_uncond
            student_delta = student_cond - student_uncond
            loss_teacher = masked_mse(
                student_cond, teacher_cond, image_mask
            )
            loss_delta = masked_mse(
                student_delta, teacher_delta, image_mask
            )
        else:
            context = state.connector(
                gemma_h.to(dtype=state.unet_dtype),
                timestep,
                gemma_mask,
                context_tokens=cfg.context_tokens,
            )
            student_cond = camera_conditioned_unet(
                state.unet,
                noisy,
                timestep,
                encoder_hidden_states=context,
                camera_condition=camera_condition,
            ).sample
            loss_diff = masked_mse(student_cond, noise, image_mask)

        if semantic_anchor_weight > 0 and state.clip_model is not None:
            if clip_geom is None:
                raise ValueError(
                    "clip_geom is required when semantic anchor weight is positive"
                )
            with torch.no_grad():
                clip_h, clip_mask = state.encode_clip(captions)
            pred77 = context[:len(captions), :cfg.clip_anchor_tokens, :]
            loss_anchor = clip_geom(pred77, clip_h, clip_mask)["total"]

        scaffold_scale = (1.0 if is_residual_connector(cfg)
                          else clip_scaffold_scale(cfg, opt_step))
        loss = (
            cfg.lambda_diffusion * loss_diff
            + scaffold_scale * cfg.lambda_teacher * loss_teacher
            + scaffold_scale * cfg.lambda_text_delta * loss_delta
            + scaffold_scale * semantic_anchor_weight * loss_anchor
            + max(cfg.residual_penalty_floor, cfg.residual_prefix_weight) * loss_prefix
            + max(cfg.residual_penalty_floor, cfg.residual_norm_weight) * loss_norm
        )

    if not torch.isfinite(loss):
        raise RuntimeError("Diffusion training loss is NaN/Inf")
    return DiffusionStepOutput(
        loss=loss,
        loss_diff=loss_diff,
        loss_teacher=loss_teacher,
        loss_delta=loss_delta,
        loss_anchor=loss_anchor,
        loss_prefix=loss_prefix,
        loss_norm=loss_norm,
        clip_scaffold_scale=scaffold_scale,
        residual_norm_ratio=residual_norm_ratio,
        # residual_norm_ratio has one value per long prompt only; retain the
        # matching timesteps so mixed short/long batches cannot mis-bucket it.
        timesteps=timestep[long_mask] if residual_norm_ratio is not None else None,
    )


def _run_periodic_training_validation(
    state: TrainingState,
    phase_name: str,
    opt_step: int,
    metrics: dict,
):
    cfg = state.cfg
    if not cfg.validation_every_opt_steps:
        return
    if opt_step % cfg.validation_every_opt_steps:
        return

    print(
        f"{phase_name} step {opt_step}: loss={metrics['loss']:.5f} "
        f"diff={metrics['loss_diff']:.5f} "
        f"teacher={metrics['loss_teacher']:.5f} "
        f"delta={metrics['loss_delta']:.5f} "
        f"anchor={metrics['loss_anchor']:.5f} "
        f"prefix={metrics['loss_prefix']:.5f} norm={metrics['loss_norm']:.5f}"
    )
    label = f"{phase_name}_step_{opt_step:06d}"
    fixed_overfit_loss(state, label=label, wandb=state.wandb)
    if not is_residual_connector(cfg):
        teacher_student_delta_alignment(
            cfg.val_prompts[0], state, label=label, wandb=state.wandb
        )
    if (
        cfg.run_long_context_diagnostics
        and cfg.max_gemma_len > cfg.clip_anchor_tokens
    ):
        suffix_counterfactual_sensitivity(
            state, label=label, wandb=state.wandb
        )
    if (cfg.camera_conditioning_enabled
            and cfg.run_camera_counterfactual_diagnostics):
        camera_counterfactual_sensitivity(
            state, label=label, wandb=state.wandb)
    if (
        cfg.generation_grid_every_opt_steps > 0
        and opt_step % cfg.generation_grid_every_opt_steps == 0
    ):
        save_checkpoint_grid(label, state)


def run_diffusion_training_loop(
    state: TrainingState,
    spec: DiffusionPhaseSpec,
    optimizer: torch.optim.Optimizer,
    trainable_params: List[nn.Parameter],
) -> dict:
    """Shared data/optimization loop for connector-only and SaRA phases."""
    cfg = state.cfg
    clip_geom = (
        ClipGeometryLoss() if spec.semantic_anchor_weight > 0 else None
    )
    plan = estimate_steps(
        spec.max_samples,
        cfg.train_batch_size,
        spec.epochs,
        spec.max_opt_steps,
        drop_last=cfg.drop_last_bucket_batches,
        bucket_count=len(cfg.aspect_ratio_buckets),
    )
    plan.name = spec.plan_name
    plan.print()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_vram(f"{spec.name}_start", 0, state)

    opt_step = 0
    accumulation = cfg.gradient_accumulation_steps
    accumulation_count = 0
    microbatch_metrics = []
    # These counters deliberately cover one logging window, not the entire
    # run.  A cumulative average would hide a late residual blow-up behind its
    # zero-initialized early updates.
    drift_sums = {f"t{start:03d}_{start + 249:03d}": 0.0
                  for start in range(0, 1000, 250)}
    drift_counts = {key: 0 for key in drift_sums}
    lr_scheduler = None
    if cfg.lr_decay_steps:
        def lr_scale(step):
            if step < cfg.lr_warmup_steps:
                return float(step + 1) / max(1, cfg.lr_warmup_steps)
            progress = (step - cfg.lr_warmup_steps) / max(
                1, cfg.lr_decay_steps - cfg.lr_warmup_steps)
            return max(0.0, 1.0 - progress)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    best = None
    last_metrics = {}
    stop = False
    started = time.time()
    if cfg.camera_conditioning_enabled:
        state.camera_metadata_samples = 0
        state.camera_presence_counts = [0, 0, 0, 0]

    for epoch in range(spec.epochs):
        dataloader = make_streaming_dataloader(
            cfg.data_sources,
            phase=spec.data_phase,
            epoch=epoch,
            max_samples=spec.max_samples,
            batch_size=cfg.train_batch_size,
            shuffle=cfg.shuffle_streaming,
            shuffle_buffer=cfg.shuffle_buffer,
            base_seed=cfg.base_seed,
            buckets=cfg.aspect_ratio_buckets,
            drop_last=cfg.drop_last_bucket_batches,
            max_image_dimension=cfg.max_image_dimension,
        )
        progress = tqdm(
            dataloader,
            desc=f"{spec.display_name} {epoch + 1}/{spec.epochs}",
        )
        optimizer.zero_grad(set_to_none=True)
        for batch in progress:
            output = diffusion_training_step(
                state,
                batch,
                opt_step,
                use_teacher_delta=spec.use_teacher_delta,
                semantic_anchor_weight=spec.semantic_anchor_weight,
                camera_dropout_probability=spec.camera_dropout_probability,
                clip_geom=clip_geom,
            )
            (output.loss / accumulation).backward()
            microbatch_metrics.append(output.scalars(opt_step + 1))
            if output.residual_norm_ratio is not None:
                for ratio, timestep_value in zip(
                        output.residual_norm_ratio.detach().cpu().tolist(),
                        output.timesteps.detach().cpu().tolist()):
                    start = min(750, int(timestep_value) // 250 * 250)
                    bucket = f"t{start:03d}_{start + 249:03d}"
                    drift_sums[bucket] += ratio
                    drift_counts[bucket] += 1
            accumulation_count += 1
            if accumulation_count < accumulation:
                continue
            nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0

            opt_step += 1
            last_metrics = {
                key: float(np.mean([metric[key] for metric in microbatch_metrics]))
                for key in microbatch_metrics[0]
            }
            last_metrics["step"] = opt_step
            microbatch_metrics.clear()
            last_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if (is_residual_connector(cfg)
                    and cfg.residual_checkpoint_every_opt_steps
                    and opt_step % cfg.residual_checkpoint_every_opt_steps == 0):
                checkpoint_path = save_residual_step_checkpoint(state, opt_step)
                print(f"Residual checkpoint saved: {checkpoint_path}")
            best = (
                last_metrics["loss"]
                if best is None
                else min(best, last_metrics["loss"])
            )

            if opt_step % 25 == 0:
                drift_metrics = {
                    f"{spec.name}/residual_rms_ratio_{bucket}": drift_sums[bucket] / drift_counts[bucket]
                    for bucket in drift_sums if drift_counts[bucket]
                }
                if drift_metrics:
                    maximum = max(drift_metrics.values())
                    if maximum > cfg.residual_drift_warn_ratio:
                        print(f"WARNING: residual RMS/CLIP RMS={maximum:.3f} exceeds "
                              f"warn={cfg.residual_drift_warn_ratio:.3f}")
                    if maximum > cfg.residual_drift_stop_ratio:
                        # Do not let an unattended run continue after the
                        # explicit safety gate.  Periodic residual checkpoints
                        # before this point remain the rollback candidates;
                        # the current in-memory state is deliberately not
                        # promoted to one.
                        stop_message = (
                            "Residual drift stop: window RMS(delta)/RMS(CLIP)="
                            f"{maximum:.3f} exceeds "
                            f"threshold={cfg.residual_drift_stop_ratio:.3f}"
                        )
                safe_wandb_log({
                    f"{spec.name}/step": opt_step,
                    f"{spec.name}/loss": last_metrics["loss"],
                    f"{spec.name}/loss_diff": last_metrics["loss_diff"],
                    f"{spec.name}/loss_teacher": last_metrics["loss_teacher"],
                    f"{spec.name}/loss_delta": last_metrics["loss_delta"],
                    f"{spec.name}/loss_anchor": last_metrics["loss_anchor"],
                    f"{spec.name}/loss_prefix": last_metrics["loss_prefix"],
                    f"{spec.name}/loss_norm": last_metrics["loss_norm"],
                    f"{spec.name}/clip_scaffold_scale": last_metrics[
                        "clip_scaffold_scale"
                    ],
                    f"{spec.name}/lr": last_metrics["lr"],
                    **drift_metrics,
                }, wandb=state.wandb)
                # Reset after logging so the next threshold applies to fresh
                # optimizer updates rather than a diluted whole-run average.
                for bucket in drift_sums:
                    drift_sums[bucket] = 0.0
                    drift_counts[bucket] = 0
                if drift_metrics and maximum > cfg.residual_drift_stop_ratio:
                    raise RuntimeError(stop_message)

            _run_periodic_training_validation(
                state, spec.name, opt_step, last_metrics
            )
            progress.set_postfix({
                "loss": f"{last_metrics['loss']:.4f}",
                "best": f"{best:.4f}",
            })
            if spec.max_opt_steps and opt_step >= spec.max_opt_steps:
                stop = True
                print(
                    f"Stopping {spec.display_name} at step cap {opt_step}"
                )
                break
        if stop:
            break
        if accumulation_count:
            # Keep every optimizer update at the declared effective batch size.
            # A partial bucket tail is intentionally not promoted to an update.
            print(f"Dropping {accumulation_count} tail microbatch(es) below accumulation={accumulation}")
            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0
            microbatch_metrics.clear()

    elapsed_min = (time.time() - started) / 60
    log_vram(f"{spec.name}_end", opt_step, state)
    summary = {
        "steps": opt_step,
        "best_loss": best if best is not None else float("nan"),
        "elapsed_min": elapsed_min,
        "final_loss": last_metrics.get("loss", float("nan")),
        "final_loss_diff": last_metrics.get("loss_diff", float("nan")),
        "final_loss_teacher": last_metrics.get(
            "loss_teacher", float("nan")
        ),
        "final_loss_delta": last_metrics.get("loss_delta", float("nan")),
        "final_loss_anchor": last_metrics.get("loss_anchor", float("nan")),
        "final_loss_prefix": last_metrics.get("loss_prefix", float("nan")),
        "final_loss_norm": last_metrics.get("loss_norm", float("nan")),
    }
    if cfg.camera_conditioning_enabled and state.camera_metadata_samples:
        for name, count in zip(
            ("fov", "focal", "aperture", "iso"),
            state.camera_presence_counts,
        ):
            summary[f"camera_{name}_coverage"] = (
                count / state.camera_metadata_samples)
    return summary


# ---------------------------------------------------------------------------
# Phase 1: ELLA connector diffusion + teacher-delta training
# ---------------------------------------------------------------------------
def run_ella_training(state: TrainingState):
    cfg = state.cfg
    if not cfg.run_training or cfg.ella_epochs <= 0:
        print("ELLA connector phase skipped")
        return

    for p in state.unet.parameters():
        p.requires_grad_(False)
    for p in state.vae.parameters():
        p.requires_grad_(False)
    for p in state.gemma_model.parameters():
        p.requires_grad_(False)
    if state.clip_model is not None:
        for p in state.clip_model.parameters():
            p.requires_grad_(False)
    for p in state.connector.parameters():
        p.requires_grad_(
            not cfg.camera_conditioning_enabled or cfg.camera_train_connector)
    if state.camera_conditioner is not None:
        for p in state.camera_conditioner.parameters():
            p.requires_grad_(True)

    trainable_connector = [
        parameter
        for parameter in state.connector.parameters()
        if parameter.requires_grad
    ]
    trainable_camera = (
        list(state.camera_conditioner.parameters())
        if state.camera_conditioner is not None else []
    )
    trainable_params = trainable_connector + trainable_camera
    optimizer_groups = []
    if trainable_connector:
        optimizer_groups.append({
            "params": trainable_connector,
            "lr": cfg.ella_lr,
            "weight_decay": 0.01,
        })
    if trainable_camera:
        optimizer_groups.append({
            "params": trainable_camera,
            "lr": cfg.camera_lr,
            "weight_decay": 0.01,
        })
    optimizer = torch.optim.AdamW(optimizer_groups, eps=1e-6)
    print(
        f"Phase 1 trainable params: connector="
        f"{sum(p.numel() for p in trainable_connector):,} camera="
        f"{sum(p.numel() for p in trainable_camera):,}"
    )
    use_teacher_delta = bool(
        cfg.use_clip_teacher_delta
        and (cfg.lambda_teacher > 0 or cfg.lambda_text_delta > 0)
        and state.clip_model is not None
    )
    print(
        "Phase 1 forward mode:",
        "paired_cfg_teacher" if use_teacher_delta else "conditional_only",
    )

    state.unet.eval()
    state.connector.train()
    if state.camera_conditioner is not None:
        state.camera_conditioner.train()
    state.vae.eval()
    state.gemma_model.eval()
    if state.clip_model is not None:
        state.clip_model.eval()
    ella_summary = run_diffusion_training_loop(
        state,
        DiffusionPhaseSpec(
            name="ella",
            display_name="ELLA",
            plan_name="ella_frozen_unet",
            data_phase=20,
            epochs=cfg.ella_epochs,
            max_samples=cfg.max_samples_ella,
            max_opt_steps=cfg.ella_max_opt_steps,
            semantic_anchor_weight=cfg.phase1_semantic_anchor_weight,
            use_teacher_delta=use_teacher_delta,
            camera_dropout_probability=cfg.camera_metadata_dropout_prob_ella,
        ),
        optimizer,
        trainable_params,
    )
    remember_final_summary(
        "ella_frozen_unet", ella_summary,
        wandb_prefix="final/ella_frozen_unet",
        wandb=state.wandb,
    )
    print(
        f"ELLA frozen-UNet done: steps={ella_summary['steps']} "
        f"best={ella_summary['best_loss']} "
        f"elapsed={ella_summary['elapsed_min']:.1f}m"
    )
    print_final_summary("ella_frozen_unet")

    ckpt_path = f"{cfg.output_dir}/ella_connector_frozen_unet.pt"
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save({
        "connector_schema": connector_checkpoint_schema(state),
        "connector_state_dict": {k: v.detach().cpu()
                                 for k, v in state.connector.state_dict().items()},
        "camera_conditioner_state_dict": (
            {k: v.detach().cpu() for k, v in
             state.camera_conditioner.state_dict().items()}
            if state.camera_conditioner is not None else None
        ),
        "camera_condition_schema": _camera_schema(state),
        "config": cfg.to_dict(),
        "stage": "ella_frozen_unet",
        "completion": {
            "completed": True,
            "phase": "ella",
            "optimizer_steps": ella_summary["steps"],
            "best_loss": ella_summary["best_loss"],
        },
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
    if not cfg.run_training or not cfg.run_sara_phase or cfg.sara_epochs <= 0:
        print("SaRA phase disabled")
        return

    # Build sparse masks on attn2 K/V weights
    for p in state.unet.parameters():
        p.requires_grad_(False)
    sara_summary = build_sara_attn2_kv_sparse_masks(
        state.unet,
        selection_mode=cfg.sara_selection_mode,
        target_fraction=cfg.sara_target_fraction,
        threshold=cfg.sara_threshold,
        min_sparse_fraction=cfg.sara_min_sparse_fraction,
        max_sparse_fraction_warn=cfg.sara_max_sparse_fraction_warn,
        max_sparse_fraction_abort=cfg.sara_max_sparse_fraction_abort,
        target_substrings=cfg.sara_target_substrings,
    )
    initial_sparse_values = capture_sara_selected_values(state.unet)
    grad_handles = install_sara_gradient_masks(state.unet)

    # Keep P3 additive when configured: camera trains, connector can stay frozen.
    for p in state.connector.parameters():
        p.requires_grad_(
            not cfg.camera_conditioning_enabled or cfg.camera_train_connector)
    if state.camera_conditioner is not None:
        for p in state.camera_conditioner.parameters():
            p.requires_grad_(True)
    trainable_connector = [p for p in state.connector.parameters() if p.requires_grad]
    camera_ids = {
        id(p) for p in state.camera_conditioner.parameters()
    } if state.camera_conditioner is not None else set()
    trainable_camera = (
        list(state.camera_conditioner.parameters())
        if state.camera_conditioner is not None else []
    )
    trainable_unet = [
        p for p in state.unet.parameters()
        if p.requires_grad and id(p) not in camera_ids
    ]
    trainable_params = trainable_connector + trainable_camera + trainable_unet
    print(f"Trainable params in ELLA+SaRA: {sum(p.numel() for p in trainable_params):,}")
    print(f"  connector: {sum(p.numel() for p in trainable_connector):,}")
    print(f"  camera: {sum(p.numel() for p in trainable_camera):,}")
    print(
        f"  masked UNet tensors: {sum(p.numel() for p in trainable_unet):,} "
        f"({sara_summary['selected']:,} selected entries)"
    )

    optimizer_groups = []
    if trainable_connector:
        optimizer_groups.append({
            "params": trainable_connector, "lr": cfg.sara_lr,
            "weight_decay": 0.01,
        })
    if trainable_camera:
        optimizer_groups.append({
            "params": trainable_camera, "lr": cfg.camera_lr,
            "weight_decay": 0.01,
        })
    optimizer_groups.append({
        "params": trainable_unet, "lr": cfg.sara_lr,
        "weight_decay": 0.0,
    })
    optimizer = torch.optim.AdamW(optimizer_groups, eps=1e-6)

    use_teacher_delta_phase2 = bool(
        cfg.use_clip_teacher_delta and cfg.use_clip_teacher_delta_phase2
        and (cfg.lambda_teacher > 0 or cfg.lambda_text_delta > 0)
        and state.clip_model is not None
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
    sara_summary_dict = run_diffusion_training_loop(
        state,
        DiffusionPhaseSpec(
            name="sara",
            display_name="ELLA+SaRA",
            plan_name="ella_sara_attn2_kv",
            data_phase=30,
            epochs=cfg.sara_epochs,
            max_samples=cfg.max_samples_sara,
            max_opt_steps=cfg.sara_max_opt_steps,
            semantic_anchor_weight=cfg.phase2_semantic_anchor_weight,
            use_teacher_delta=use_teacher_delta_phase2,
            camera_dropout_probability=cfg.camera_metadata_dropout_prob_sara,
        ),
        optimizer,
        trainable_params,
    )
    delta_metrics = sara_selected_delta_metrics(
        state.unet, initial_sparse_values
    )
    sara_summary_dict.update({
        "sparse_selected": sara_summary["selected"],
        "sparse_total": sara_summary["total_target"],
        "sparse_fraction": sara_summary["fraction"],
        "sparse_target_scope_fraction": sara_summary[
            "target_scope_fraction"
        ],
        "sparse_whole_unet_fraction": sara_summary[
            "whole_unet_fraction"
        ],
        **delta_metrics,
    })
    print(
        "SaRA selected-weight delta: "
        f"L2={delta_metrics['selected_delta_l2']:.6g} "
        f"relative={delta_metrics['selected_delta_relative_l2']:.6g} "
        f"RMS={delta_metrics['selected_delta_rms']:.6g}"
    )
    remember_final_summary(
        "sara_attn2_kv", sara_summary_dict,
        wandb_prefix="final/sara_attn2_kv", wandb=state.wandb)

    # Store handles + summary on state for save_artifacts
    state._sara_grad_handles = grad_handles
    state._sara_summary = sara_summary

    print(
        f"ELLA+SaRA done: steps={sara_summary_dict['steps']} "
        f"best={sara_summary_dict['best_loss']} "
        f"elapsed={sara_summary_dict['elapsed_min']:.1f}m"
    )
    print_final_summary("sara_attn2_kv")


# ---------------------------------------------------------------------------
# Validation & Final proof
# ---------------------------------------------------------------------------
def save_checkpoint_grid(label: str, state: TrainingState):
    """Generate + save a validation image grid at a training checkpoint."""
    cfg = state.cfg
    if is_residual_connector(cfg):
        cases = cfg.long_eval_prompts or cfg.val_prompts
        imgs, labels = [], []
        for ptxt in cases:
            prefix = _clip_visible_prefixes(state, [ptxt])[0]
            common = dict(steps=cfg.val_steps, guidance=cfg.val_guidance,
                          seed=cfg.val_seed)
            imgs.extend([
                generate_clip_teacher(ptxt, state, **common),
                generate_ella(ptxt, state, residual_gemma_prompt=prefix,
                              force_residual=True, **common),
                generate_ella(ptxt, state, **common),
            ])
            labels.extend(["native CLIP", "prefix-Gemma residual", "full-Gemma residual"])
        grid_path = f"{cfg.output_dir}/{label}_residual_controls.png"
        save_validation_grid(imgs, labels, grid_path,
                             f"Residual controls [{label}]", cols=3)
        print(f"Residual checkpoint grid saved: {grid_path}")
        return
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
    if (cfg.camera_conditioning_enabled
            and cfg.run_camera_counterfactual_diagnostics):
        camera_metrics = camera_counterfactual_sensitivity(
            state, label="final_camera", wandb=state.wandb)
        fev_summary.update({
            f"camera_{key}": value
            for key, value in camera_metrics.items()
        })
    remember_final_summary(
        "final_eval", fev_summary,
        wandb_prefix="final/eval", wandb=state.wandb)
    print_final_summary("final_eval")

    if (cfg.run_long_context_diagnostics
            and cfg.max_gemma_len > cfg.clip_anchor_tokens):
        suffix_metrics = suffix_counterfactual_sensitivity(
            state, label="final_suffix", wandb=state.wandb)
        fev_summary["suffix_sensitivity_mean"] = suffix_metrics.get(
            "suffix_sensitivity_mean")
        remember_final_summary(
            "final_eval", fev_summary,
            wandb_prefix="final/eval", wandb=state.wandb)
    else:
        print("Long-input diagnostics disabled")

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

        if cfg.camera_conditioning_enabled:
            camera_images = []
            camera_labels = []
            camera_cases = []
            observed_fov = bool(state.camera_presence_counts[0])
            if cfg.camera_run_fov_counterfactual or observed_fov:
                camera_cases.append((
                    "vertical FOV", "vertical_fov_deg",
                    cfg.camera_counterfactual_fov_a,
                    cfg.camera_counterfactual_fov_b,
                ))
            camera_cases.append((
                "focal length", "focal_length_mm",
                cfg.camera_counterfactual_focal_a,
                cfg.camera_counterfactual_focal_b,
            ))
            for case_name, field_name, value_a, value_b in camera_cases:
                for value in (value_a, value_b):
                    condition = make_camera_condition(
                        **{field_name: value}, capture_type="photo")
                    camera_images.append(generate_ella(
                        cfg.val_prompts[0], state, steps=cfg.val_steps,
                        guidance=cfg.val_guidance, seed=cfg.val_seed,
                        camera_condition=condition,
                    ))
                    camera_labels.append(f"{case_name} {value:g}")
            camera_grid_path = (
                f"{cfg.output_dir}/validation_camera_counterfactual.png")
            save_validation_grid(
                camera_images, camera_labels, camera_grid_path,
                "P3 camera counterfactuals")
            fev_summary["camera_grid_saved"] = camera_grid_path
            remember_final_summary(
                "final_eval", fev_summary,
                wandb_prefix="final/eval", wandb=state.wandb)

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
        "connector_schema": connector_checkpoint_schema(state),
        "connector_state_dict": {k: v.detach().cpu()
                                 for k, v in state.connector.state_dict().items()},
        "camera_conditioner_state_dict": (
            {k: v.detach().cpu() for k, v in
             state.camera_conditioner.state_dict().items()}
            if state.camera_conditioner is not None else None
        ),
        "camera_condition_schema": _camera_schema(state),
        "gemma_model_id": cfg.gemma_id,
        "sd_checkpoint": cfg.sd_checkpoint,
        "run_config": cfg.to_dict(),
    }, path)
    print("Connector saved:", path)
    summary = {"connector_saved": path, "context_tokens": cfg.context_tokens}

    if state.camera_conditioner is not None:
        camera_path = f"{cfg.output_dir}/pure_ella_camera_conditioner.pt"
        torch.save({
            "architecture": "centered camera timestep conditioning",
            "camera_conditioner_state_dict": {
                k: v.detach().cpu()
                for k, v in state.camera_conditioner.state_dict().items()
            },
            "camera_condition_schema": _camera_schema(state),
            "run_config": cfg.to_dict(),
        }, camera_path)
        print("Camera conditioner saved:", camera_path)
        summary["camera_conditioner_saved"] = camera_path

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
    validate_connector_checkpoint_schema(ckpt, state, path)
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
        gemma_layer_mix_count=cfg.gemma_layer_mix_count,
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
            token=os.environ.get("HF_TOKEN"),
            text_encoder=None,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        reloaded_unet = pipe.unet.to(state.device).eval()
        pipe = None
    else:
        reloaded_unet = UNet2DConditionModel.from_pretrained(
            sd_path, subfolder="unet", torch_dtype=state.unet_dtype,
            token=os.environ.get("HF_TOKEN"),
        ).to(state.device).eval()

    reloaded_camera = None
    if cfg.camera_conditioning_enabled:
        camera_path = f"{cfg.output_dir}/pure_ella_camera_conditioner.pt"
        camera_ckpt = torch.load(
            camera_path, map_location="cpu", weights_only=True)
        _validate_camera_checkpoint_schema(camera_ckpt, state, camera_path)
        reloaded_camera = CameraConditioner(
            output_dim=reloaded_unet.time_embedding.linear_2.out_features,
            hidden_dim=cfg.camera_hidden_dim,
            fourier_bands=cfg.camera_fourier_bands,
            enable_geometry_head=cfg.camera_enable_geometry_head,
            enable_raw_focal_head=cfg.camera_enable_raw_focal_head,
            enable_exposure_head=cfg.camera_enable_exposure_head,
            use_capture_type=cfg.camera_use_capture_type,
        ).to(device=state.device, dtype=state.unet_dtype)
        reloaded_camera.load_state_dict(
            camera_ckpt["camera_conditioner_state_dict"], strict=True)
        install_camera_conditioner(reloaded_unet, reloaded_camera)
        reloaded_camera.eval()
        proof_conditions = torch.stack((
            make_camera_condition(capture_type="unknown"),
            make_camera_condition(
                focal_length_mm=cfg.camera_counterfactual_focal_a,
                capture_type="photo"),
        )).to(state.device)
        with torch.no_grad():
            live_camera_output = state.camera_conditioner(proof_conditions)
            reloaded_camera_output = reloaded_camera(proof_conditions)
        if not torch.equal(live_camera_output, reloaded_camera_output):
            raise RuntimeError("Reloaded camera conditioner output mismatch")
        print("Camera conditioner strict reload PASS")

    # Apply SaRA sparse UNet patch if saved
    sparse_path = f"{cfg.output_dir}/pure_ella_unet_attn2_kv_sparse_L{cfg.context_tokens}.pt"
    if cfg.run_sara_phase and os.path.exists(sparse_path):
        sparse_ckpt = torch.load(sparse_path, map_location="cpu")
        loaded = load_sara_sparse_values(reloaded_unet, sparse_ckpt["sparse_values"])
        print(f"Sparse UNet patch reload PASS: {loaded:,} values")

    with torch.no_grad():
        t = torch.tensor([500, 500], device=state.device).long()
        if is_residual_connector(cfg):
            clip_context, _ = state.encode_clip(["a small red car", ""])
            gemma_h, gemma_mask = state.encode_gemma(["a small red car"])
            ctx = clip_context.clone()
            ctx[:1] += cfg.residual_strength * reloaded(
                clip_context[:1].to(dtype=state.unet_dtype),
                gemma_h.to(dtype=state.unet_dtype), t[:1], gemma_mask,
                context_tokens=cfg.context_tokens)
        else:
            gh, gm = state.encode_gemma(["a small red car", ""])
            ctx = reloaded(gh.to(dtype=state.unet_dtype), t, gm,
                           context_tokens=cfg.context_tokens)
        test_latent = torch.randn(
            2, 4, 64, 64, device=state.device, dtype=state.unet_dtype)
        pred = camera_conditioned_unet(
            reloaded_unet, test_latent, t, encoder_hidden_states=ctx).sample
        assert torch.isfinite(pred).all()
    print("Fresh reload finite forward PASS")
    reload_summary = {
        "strict_connector_reload": 1.0,
        "strict_camera_reload": (
            1.0 if cfg.camera_conditioning_enabled else None),
        "finite_forward": 1.0,
        "context_tokens": cfg.context_tokens,
    }
    remember_final_summary(
        "reload_proof", reload_summary,
        wandb_prefix="final/reload_proof", wandb=state.wandb)
    print_final_summary("reload_proof")

    old_connector = state.connector
    old_unet = state.unet
    old_camera = state.camera_conditioner
    state.connector = reloaded
    state.unet = reloaded_unet
    state.camera_conditioner = reloaded_camera
    if state.clip_model is not None and not is_residual_connector(cfg):
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
        state.camera_conditioner = old_camera
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
        long_labels.append(f"short -> C{cfg.context_tokens}")
        long_imgs.append(generate_ella(
            long, state,
            steps=cfg.val_steps, guidance=cfg.val_guidance,
            seed=cfg.val_seed, context_tokens=ctx,
        ))
        long_labels.append(f"long G{cfg.max_gemma_len} -> C{ctx}")
    long_path = f"{out}/validation_long_input_G{cfg.max_gemma_len}_C{ctx}.png"
    save_validation_grid(long_imgs, long_labels, long_path,
                         f"Long-input ELLA G{cfg.max_gemma_len} -> C{ctx}")
    print(f"Long-context grid saved: {long_path}")

    # Suffix counterfactual grids
    if (cfg.run_suffix_counterfactual_grids
            and cfg.max_gemma_len > cfg.clip_anchor_tokens):
        sfx_out = save_suffix_counterfactual_grids(
            cfg.suffix_counterfactual_cases,
            f"{out}/validation_suffix_counterfactual_G{cfg.max_gemma_len}_C{ctx}",
            "Suffix counterfactuals", state, context_tokens=ctx,
        )
        for p in sfx_out:
            print(f"Suffix counterfactual grid saved: {p}")
    else:
        print("Suffix counterfactual grids disabled")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_connector_checkpoint(state: TrainingState, ckpt_path: str):
    """Load connector and optional P3 state from a training checkpoint."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("connector_state_dict")
    if sd is None:
        print("Checkpoint has no connector_state_dict")
        return False
    strict = not state.cfg.init_connector_partial_warmstart
    try:
        validate_connector_checkpoint_schema(ckpt, state, ckpt_path)
    except RuntimeError as error:
        print(error)
        return False
    try:
        missing, unexpected = state.connector.load_state_dict(sd, strict=strict)
    except RuntimeError as error:
        print(f"Checkpoint is incompatible with the configured connector: {error}")
        return False
    if missing:
        print(f"WARNING: missing keys ({len(missing)}): {missing[:5]}…")
    if unexpected:
        print(f"WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}…")
    stage = ckpt.get("stage", "unknown")
    print(f"Loaded connector from {ckpt_path} (stage={stage}, strict={strict})")
    camera_sd = ckpt.get("camera_conditioner_state_dict")
    if state.camera_conditioner is not None and camera_sd:
        try:
            _validate_camera_checkpoint_schema(ckpt, state, ckpt_path)
        except RuntimeError as error:
            print(error)
            return False
        state.camera_conditioner.load_state_dict(camera_sd, strict=True)
        print("Loaded camera conditioner from the same checkpoint")
    return True


def validate_residual_sara_resume(
    cfg: TrainConfig, phases_requested: set[str], resume_path: str,
) -> None:
    """Require the proven P1 connector before residual sparse-U-Net P2.

    SaRA can technically optimize a freshly initialized residual connector,
    but that conflates P1's suffix-conditioning falsifier with U-Net capacity
    growth and makes its provenance unusable.
    """
    requires_handoff = (
        cfg.run_training
        and cfg.run_sara_phase
        and "sara" in phases_requested
        and is_residual_connector(cfg)
    )
    if not requires_handoff:
        return
    if not resume_path:
        raise ValueError(
            "stage3 residual SaRA requires --resume-ckpt or "
            "init_connector_ckpt_path from a completed P1 connector run"
        )
    if not os.path.isfile(resume_path):
        raise ValueError(f"stage3 residual SaRA P1 checkpoint not found: {resume_path}")
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
    completion = checkpoint.get("completion")
    if (
        checkpoint.get("stage") != "ella_frozen_unet"
        or not isinstance(completion, dict)
        or completion.get("completed") is not True
        or completion.get("phase") != "ella"
    ):
        raise ValueError(
            "stage3 residual SaRA requires the completed P1 "
            "ella_connector_frozen_unet.pt checkpoint, not an intermediate "
            "or later-stage artifact"
        )


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
    try:
        cfg.validate_requested_phases(phases_requested)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    seed_everything(cfg.base_seed)

    resume_path = args.resume_ckpt or cfg.init_connector_ckpt_path
    try:
        validate_residual_sara_resume(cfg, phases_requested, resume_path)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    pretrain_will_run = (
        "pretrain" in phases_requested
        and cfg.run_clip_alignment_pretrain
        and cfg.run_training
        and cfg.pretrain_epochs > 0
    )
    fast_text_only_pretrain = bool(
        cfg.pretrain_text_only_fast
        and pretrain_will_run
        and phases_requested == {"pretrain"}
    )
    p3_frozen_connector_training = (
        cfg.camera_conditioning_enabled
        and not cfg.camera_train_connector
        and bool({"ella", "sara"} & phases_requested)
    )
    if p3_frozen_connector_training and pretrain_will_run:
        print(
            "ERROR: P3 with camera_train_connector=false cannot run connector "
            "pretraining; disable run_clip_alignment_pretrain and resume P1"
        )
        sys.exit(1)
    if p3_frozen_connector_training and not resume_path:
        print(
            "ERROR: P3 freezes the text connector by default and requires "
            "--resume-ckpt or init_connector_ckpt_path from a completed P1 run"
        )
        sys.exit(1)
    if resume_path and not pretrain_will_run and not os.path.isfile(resume_path):
        print(f"ERROR: connector checkpoint not found: {resume_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_dtype = resolve_model_weight_dtype(cfg)
    autocast_dtype = resolve_autocast_dtype(cfg, device)

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.wandb_enabled:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=(cfg.wandb_run_name or
                  f"pure-ella-{cfg.experiment_stage}-{cfg.run_mode}-L{cfg.context_tokens}"),
            group=cfg.wandb_group or None,
            config=cfg.to_dict(),
        )
        wandb.define_metric("pretrain/step")
        wandb.define_metric("pretrain/*", step_metric="pretrain/step")
        wandb.define_metric("pretrain_val/step")
        wandb.define_metric("pretrain_val/*", step_metric="pretrain_val/step")
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
        cfg=cfg,
        device=device,
        unet_dtype=unet_dtype,
        autocast_dtype=autocast_dtype,
        wandb=wandb,
    )

    cfg.print_plan()

    load_gemma(state)
    if (cfg.run_clip_alignment_pretrain or is_residual_connector(cfg)
            or cfg.use_clip_teacher_delta
            or cfg.phase1_semantic_anchor_weight > 0
            or cfg.phase2_semantic_anchor_weight > 0):
        load_clip(state)
    if fast_text_only_pretrain:
        print("Fast prompt-only Phase 0: U-Net and VAE loading skipped")
    else:
        load_stylejourney(state)
        build_camera_conditioner(state)
    state.encode_gemma = make_encode_gemma(state)
    if state.clip_model is not None:
        state.encode_clip = make_encode_clip(state)
    build_ella_connector(state)

    if cfg.run_mode == "overfit_train":
        overfit_batch_size = min(4, cfg.train_batch_size)
        dl = make_streaming_dataloader(
            cfg.data_sources, phase=1, epoch=0,
            max_samples=max(cfg.max_samples_ella, overfit_batch_size),
            batch_size=overfit_batch_size,
            shuffle=cfg.shuffle_streaming,
            shuffle_buffer=cfg.shuffle_buffer, base_seed=cfg.base_seed,
            buckets=cfg.aspect_ratio_buckets,
            drop_last=False,
            max_image_dimension=cfg.max_image_dimension)
        batch = next(iter(dl))
        state.overfit_eval_batch = {
            "image": batch["image"],
            "image_mask": batch["image_mask"],
            "caption": _as_prompt_list(batch["caption"]),
            "camera_condition": batch["camera_condition"],
        }
        print("Exact overfit captions:")
        for i, p in enumerate(state.overfit_eval_batch["caption"]):
            print(f"  {i}: {p[:160]}")
    else:
        print("Non-overfit run: generic validation prompts active")

    # ── resume / phase selection ──
    if not pretrain_will_run:
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

    if fast_text_only_pretrain:
        run_save_artifacts(state)
        print("Fast prompt-only Phase 0: diffusion grids and reload proof skipped")
    else:
        run_validation_grids(state)
        run_complex_prompt_checks(state)
        run_save_artifacts(state)
        run_reload_proof(state)

    if cfg.wandb_enabled:
        wandb.finish()
    print("Done. Training complete.")


if __name__ == "__main__":
    main()
