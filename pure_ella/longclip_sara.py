"""LongCLIP-L conditioning and provenance helpers for sparse U-Net tuning.

LongCLIP is intentionally frozen here.  Its value is that ``encode_text_full``
already emits 248 CLIP-compatible, 768-wide states; training it alongside a
U-Net would destroy the direct-replacement baseline we are trying to measure.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from pure_ella.config import TrainConfig


LONGCLIP_L_CONTEXT_TOKENS = 248
LONGCLIP_L_WIDTH = 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longclip_attention_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Mask through EOT without assuming token id 0 is always padding.

    CLIP's EOT id is the largest vocabulary id, so ``argmax`` is the official
    pooling convention and remains correct when a real content token has id 0.
    """
    if tokens.ndim != 2:
        raise ValueError(f"LongCLIP tokens must be rank 2, got {tokens.shape}")
    eot_positions = tokens.argmax(dim=-1)
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    return (positions.unsqueeze(0) <= eot_positions.unsqueeze(1)).long()


def is_longclip_context_overflow(error: RuntimeError) -> bool:
    return f"too long for context length {LONGCLIP_L_CONTEXT_TOKENS}" in str(error)


@dataclass
class LongClipSaraConfig:
    """Configuration for direct LongCLIP-L + sparse-SaRA adaptation."""

    sd_checkpoint: str
    longclip_repo: str
    longclip_checkpoint: str
    output_dir: str
    data_sources: list[Any]
    validation_data_sources: list[Any] = field(default_factory=list)
    run_mode: Literal["short_train", "full_train"] = "full_train"
    context_tokens: int = LONGCLIP_L_CONTEXT_TOKENS
    longclip_width: int = LONGCLIP_L_WIDTH
    fail_on_prompt_truncation: bool = True
    model_weight_dtype: Literal["float32", "bfloat16"] = "float32"
    mixed_precision: Literal["no", "bf16"] = "bf16"
    enable_unet_gradient_checkpointing: bool = True
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_samples: int = 11708
    epochs: int = 1
    max_opt_steps: int = 1463
    sara_lr: float = 1e-5
    sidecar_lr: float | None = None
    lr_warmup_steps: int = 100
    lr_decay_steps: int = 1463
    grad_clip_norm: float = 0.5
    shuffle_streaming: bool = True
    shuffle_buffer: int = 10000
    prompt_source_fields: list[str] = field(
        default_factory=lambda: TrainConfig().prompt_source_fields)
    prompt_source_mode: Literal["random", "first"] = "random"
    validation_prompt_source_fields: list[str] = field(default_factory=list)
    validation_prompt_source_mode: Literal["random", "first"] = "first"
    aspect_ratio_buckets: list[tuple[int, int]] = field(default_factory=lambda: [
        (1024, 1024), (1024, 896), (1024, 768), (1024, 640), (1024, 512),
        (896, 1024), (768, 1024), (640, 1024), (512, 1024),
    ])
    drop_last_bucket_batches: bool = True
    max_image_dimension: int = 1024
    caption_mix_short: float = 0.25
    caption_mix_medium: float = 0.25
    caption_mix_long: float = 0.50
    sara_selection_mode: Literal["magnitude_threshold", "target_fraction"] = "target_fraction"
    sara_target_fraction: float = 0.05
    sara_threshold: float = 1e-3
    sara_min_sparse_fraction: float = 0.01
    sara_max_sparse_fraction_warn: float = 0.10
    sara_max_sparse_fraction_abort: float = 0.25
    sara_target_substrings: tuple[str, ...] = ("attn2.to_k", "attn2.to_v")
    validation_every_opt_steps: int = 250
    validation_max_samples: int = 64
    require_validation_gate: bool = False
    validation_max_relative_regression: float = 0.02
    generation_grid_every_opt_steps: int = 0
    base_seed: int = 1234
    val_steps: int = 30
    val_guidance: float = 5.5
    val_seed: int = 777
    val_prompts: list[str] = field(default_factory=lambda: [
        "a cat sitting on a windowsill looking outside",
        "a watercolor painting of a mountain lake",
        "a neon-lit cyberpunk alleyway at night",
    ])
    require_short_prompt_regression_gate: bool = True
    short_prompt_max_relative_rms: float = 0.05
    short_prompt_validation_every_opt_steps: int = 0
    complex_generation_cases: list[dict[str, Any]] = field(
        default_factory=lambda: TrainConfig().complex_generation_cases)
    run_final_visual_check: bool = True
    wandb_enabled: bool = True
    wandb_project: str = "gemma3-sd-pure-ella"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    initial_sara_patch: str = ""
    resolution_conditioning_enabled: bool = False
    resolution_conditioning_hidden_dim: int = 256
    resolution_conditioning_dropout_prob: float = 0.0
    p4_enabled: bool = False
    p4_variant: Literal["no_pe", "rope"] = "no_pe"
    p4_insertions: list[str] = field(default_factory=lambda: ["pre_mid"])
    p4_hidden_dim: int = 0  # 0 = deepest U-Net channel count
    p4_heads: int = 8
    p4_ff_mult: float = 2.0
    p4_rope_base: float = 10000.0
    p4_timestep_adaln: bool = True
    p4_smoke_identity_check: bool = True

    def __post_init__(self):
        if self.context_tokens != LONGCLIP_L_CONTEXT_TOKENS:
            raise ValueError(
                f"LongCLIP-L requires context_tokens={LONGCLIP_L_CONTEXT_TOKENS}")
        if self.longclip_width != LONGCLIP_L_WIDTH:
            raise ValueError(
                f"LongCLIP-L requires longclip_width={LONGCLIP_L_WIDTH}")
        if not self.data_sources:
            raise ValueError("LongCLIP SaRA requires data_sources")
        if self.train_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch size and gradient accumulation must be positive")
        if self.epochs < 1 or self.max_samples < 1 or self.max_opt_steps < 1:
            raise ValueError("LongCLIP SaRA needs positive data and step budgets")
        if self.sara_lr <= 0:
            raise ValueError("sara_lr must be positive")
        if self.sidecar_lr is not None and self.sidecar_lr <= 0:
            raise ValueError("sidecar_lr must be positive when configured")
        if self.short_prompt_validation_every_opt_steps < 0:
            raise ValueError(
                "short_prompt_validation_every_opt_steps must be non-negative")
        if self.validation_max_samples < 0:
            raise ValueError("validation_max_samples must be non-negative")
        if self.validation_max_relative_regression < 0:
            raise ValueError("validation_max_relative_regression must be non-negative")
        if self.short_prompt_max_relative_rms < 0:
            raise ValueError("short_prompt_max_relative_rms must be non-negative")
        if self.require_short_prompt_regression_gate and not self.val_prompts:
            raise ValueError("short-prompt regression gate requires val_prompts")
        if self.mixed_precision not in {"no", "bf16"}:
            raise ValueError("mixed_precision must be 'no' or 'bf16'")
        if self.prompt_source_mode not in {"random", "first"}:
            raise ValueError("prompt_source_mode must be 'random' or 'first'")
        if self.validation_prompt_source_mode not in {"random", "first"}:
            raise ValueError("validation_prompt_source_mode must be 'random' or 'first'")
        if not self.prompt_source_fields:
            raise ValueError("prompt_source_fields must not be empty")
        if self.model_weight_dtype != "float32":
            raise ValueError("SaRA training requires float32 master weights")
        if self.sara_selection_mode not in {"magnitude_threshold", "target_fraction"}:
            raise ValueError("unsupported SaRA selection mode")
        if not 0 < self.sara_target_fraction <= 1:
            raise ValueError("sara_target_fraction must be in (0, 1]")
        if not (0 <= self.sara_min_sparse_fraction <= self.sara_max_sparse_fraction_warn
                <= self.sara_max_sparse_fraction_abort <= 1):
            raise ValueError("SaRA fraction gates are inconsistent")
        if self.lr_decay_steps and self.lr_decay_steps < self.lr_warmup_steps:
            raise ValueError("lr_decay_steps must be >= lr_warmup_steps")
        if self.resolution_conditioning_hidden_dim < 1:
            raise ValueError("resolution_conditioning_hidden_dim must be positive")
        if self.resolution_conditioning_dropout_prob != 0.0:
            raise ValueError(
                "resolution_conditioning_dropout_prob must be 0.0; "
                "resolution is always known and dropout/unknown alias real 1024x1024")
        if self.p4_variant not in {"no_pe", "rope"}:
            raise ValueError("p4_variant must be 'no_pe' or 'rope'")
        if self.p4_enabled:
            if not self.p4_insertions:
                raise ValueError("p4_enabled requires at least one insertion site")
            allowed_sites = {"pre_mid", "post_mid"}
            unknown_sites = sorted(set(self.p4_insertions) - allowed_sites)
            if unknown_sites:
                raise ValueError(f"unsupported P4 insertion sites: {unknown_sites}")
            if self.p4_heads < 1:
                raise ValueError("p4_heads must be positive")
            if self.p4_hidden_dim < 0:
                raise ValueError("p4_hidden_dim must be non-negative")
            if self.p4_hidden_dim and self.p4_hidden_dim % self.p4_heads:
                raise ValueError("p4_hidden_dim must be divisible by p4_heads")
            if self.p4_variant == "rope":
                deepest_dim = self.p4_hidden_dim or 1280
                if (deepest_dim // self.p4_heads) % 4:
                    raise ValueError("P4 RoPE requires per-head dim divisible by 4")
            if self.p4_ff_mult <= 0:
                raise ValueError("p4_ff_mult must be positive")
            if self.p4_rope_base <= 0:
                raise ValueError("p4_rope_base must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "LongClipSaraConfig":
        with Path(path).open() as handle:
            raw = json.load(handle)
        if "sara_target_substrings" in raw:
            raw["sara_target_substrings"] = tuple(raw["sara_target_substrings"])
        if "aspect_ratio_buckets" in raw:
            raw["aspect_ratio_buckets"] = [tuple(item) for item in raw["aspect_ratio_buckets"]]
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


class LongClipEncoder:
    """Frozen official LongCLIP-L adapter exposing SD conditioning states."""

    def __init__(self, repo: str | Path, checkpoint: str | Path,
                 device: torch.device, dtype: torch.dtype,
                 fail_on_truncation: bool = True):
        self.repo = Path(repo).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not (self.repo / "model" / "longclip.py").is_file():
            raise FileNotFoundError(
                f"Official LongCLIP checkout missing model/longclip.py: {self.repo}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"LongCLIP checkpoint not found: {self.checkpoint}")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        longclip = importlib.import_module("model.longclip")
        self._longclip = longclip
        self.model, _ = longclip.load(str(self.checkpoint), device=device)
        self.model = self.model.eval().to(dtype=dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.fail_on_truncation = fail_on_truncation
        self.max_observed_tokens = 0
        self.truncated_prompt_count = 0

    @torch.no_grad()
    def encode(self, prompts, *, track_truncation: bool = True,
               ) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        # First attempt the strict path.  For corpus training, callers may
        # explicitly permit clipping; in that case count it instead of hiding
        # it behind LongCLIP's ``truncate=True`` convenience switch.
        try:
            tokens = self._longclip.tokenize(
                prompts, context_length=LONGCLIP_L_CONTEXT_TOKENS,
                truncate=False)
        except RuntimeError as error:
            if not is_longclip_context_overflow(error):
                raise
            if self.fail_on_truncation:
                raise
            overlong = 0
            for prompt in prompts:
                try:
                    self._longclip.tokenize(
                        [prompt], context_length=LONGCLIP_L_CONTEXT_TOKENS,
                        truncate=False)
                except RuntimeError as prompt_error:
                    if not is_longclip_context_overflow(prompt_error):
                        raise
                    overlong += 1
            if track_truncation:
                self.truncated_prompt_count += overlong
            if track_truncation and self.truncated_prompt_count == overlong:
                print(
                    "WARNING: LongCLIP training truncation enabled; clipped "
                    f"{overlong}/{len(prompts)} prompt(s) to "
                    f"{LONGCLIP_L_CONTEXT_TOKENS} tokens")
            tokens = self._longclip.tokenize(
                prompts, context_length=LONGCLIP_L_CONTEXT_TOKENS,
                truncate=True)
        tokens = tokens.to(self.device)
        attention_mask = longclip_attention_mask(tokens)
        lengths = attention_mask.sum(dim=1)
        self.max_observed_tokens = max(self.max_observed_tokens, int(lengths.max()))
        states = self.model.encode_text_full(tokens)
        expected = (len(prompts), LONGCLIP_L_CONTEXT_TOKENS, LONGCLIP_L_WIDTH)
        if tuple(states.shape) != expected:
            raise RuntimeError(
                f"LongCLIP conditioning shape mismatch: expected {expected}, "
                f"got {tuple(states.shape)}")
        if not torch.isfinite(states).all():
            raise RuntimeError("LongCLIP emitted NaN/Inf conditioning")
        return states, attention_mask

    def provenance(self) -> dict:
        return {
            "encoder": "BeichenZhang/LongCLIP-L",
            "context_tokens": LONGCLIP_L_CONTEXT_TOKENS,
            "width": LONGCLIP_L_WIDTH,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
        }


def longclip_sara_schema(cfg: LongClipSaraConfig, encoder: LongClipEncoder) -> dict:
    sd_checkpoint = Path(cfg.sd_checkpoint).expanduser().resolve()
    if not sd_checkpoint.is_file():
        raise FileNotFoundError(f"StyleJourney checkpoint not found: {sd_checkpoint}")
    schema = {
        "version": 2,
        "conditioning_backend": "longclip_l_direct",
        "longclip": encoder.provenance(),
        "sd_checkpoint": str(sd_checkpoint),
        "sd_checkpoint_sha256": sha256_file(sd_checkpoint),
        "sara_target_substrings": list(cfg.sara_target_substrings),
        "sara_selection_mode": cfg.sara_selection_mode,
        "sara_target_fraction": cfg.sara_target_fraction,
        "sara_threshold": cfg.sara_threshold,
    }
    if cfg.resolution_conditioning_enabled:
        from pure_ella.resolution import resolution_condition_schema
        schema["resolution_conditioning"] = resolution_condition_schema(
            hidden_dim=cfg.resolution_conditioning_hidden_dim)
    if cfg.p4_enabled:
        from pure_ella.p4 import p4_schema
        schema["p4"] = p4_schema(
            enabled=True,
            variant=cfg.p4_variant,
            sites=cfg.p4_insertions,
            hidden_dim=(cfg.p4_hidden_dim or 1280),
            heads=cfg.p4_heads,
            ff_mult=cfg.p4_ff_mult,
            rope_base=cfg.p4_rope_base,
            timestep_adaln=cfg.p4_timestep_adaln)
    return schema


def validate_longclip_sara_checkpoint(
    checkpoint: dict, cfg: LongClipSaraConfig, encoder: LongClipEncoder,
    source: str,
    *,
    require_validation_gate: bool | None = None,
    require_short_prompt_regression_gate: bool | None = None,
) -> None:
    """Reject a sparse patch if its LongCLIP or SD provenance differs."""
    expected = longclip_sara_schema(cfg, encoder)
    actual = checkpoint.get("longclip_sara_schema")
    if actual != expected:
        raise RuntimeError(
            f"LongCLIP SaRA schema mismatch in {source}: "
            f"expected {expected!r}, got {actual!r}")
    completion = checkpoint.get("completion")
    if (not isinstance(completion, dict)
            or completion.get("completed") is not True
            or not isinstance(completion.get("optimizer_steps"), int)
            or completion["optimizer_steps"] <= 0):
        raise RuntimeError(
            f"LongCLIP SaRA checkpoint {source} is incomplete or has zero updates")
    if not checkpoint.get("sparse_values"):
        raise RuntimeError(f"LongCLIP SaRA checkpoint {source} has no sparse values")
    if cfg.resolution_conditioning_enabled:
        state_dict = checkpoint.get("resolution_conditioner_state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(
                f"LongCLIP SaRA checkpoint {source} is missing "
                "resolution_conditioner_state_dict")
    elif checkpoint.get("resolution_conditioner_state_dict") is not None:
        raise RuntimeError(
            f"LongCLIP SaRA checkpoint {source} has unexpected "
            "resolution_conditioner_state_dict")
    if cfg.p4_enabled:
        state_dict = checkpoint.get("p4_state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(
                f"LongCLIP SaRA checkpoint {source} is missing p4_state_dict")
    elif checkpoint.get("p4_state_dict") is not None:
        raise RuntimeError(
            f"LongCLIP SaRA checkpoint {source} has unexpected p4_state_dict")
    gate = checkpoint.get("validation_gate")
    validation_gate_required = (
        cfg.require_validation_gate
        if require_validation_gate is None else require_validation_gate)
    short_prompt_gate_required = (
        cfg.require_short_prompt_regression_gate
        if require_short_prompt_regression_gate is None
        else require_short_prompt_regression_gate)
    if validation_gate_required or short_prompt_gate_required:
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            raise RuntimeError(
                f"LongCLIP SaRA checkpoint {source} did not pass validation gates")
        if validation_gate_required and not _checkpoint_heldout_gate_passed(gate):
            raise RuntimeError(
                f"LongCLIP SaRA checkpoint {source} did not run/pass held-out validation")
        if short_prompt_gate_required and not _checkpoint_short_prompt_gate_passed(gate):
            raise RuntimeError(
                f"LongCLIP SaRA checkpoint {source} did not run/pass short-prompt validation")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _checkpoint_heldout_gate_passed(gate: dict) -> bool:
    """Return True when held-out validation independently ran and passed.

    New checkpoints record explicit ``heldout_ran``/``heldout_passed`` fields.
    Legacy checkpoints are accepted only if the old aggregate ``passed`` flag is
    backed by finite held-out metrics, which keeps existing real endpoints
    reloadable while rejecting synthetic ``passed=True`` shells.
    """
    if "heldout_ran" in gate or "heldout_passed" in gate:
        return (gate.get("heldout_ran") is True
                and gate.get("heldout_passed") is True
                and _finite_number(gate.get("baseline_loss"))
                and _finite_number(gate.get("final_loss"))
                and _finite_number(gate.get("relative_change")))
    return (gate.get("passed") is True
            and _finite_number(gate.get("baseline_loss"))
            and _finite_number(gate.get("final_loss"))
            and _finite_number(gate.get("relative_change")))


def _checkpoint_short_prompt_gate_passed(gate: dict) -> bool:
    """Return True when short-prompt regression validation ran and passed."""
    if "short_prompt_ran" in gate or "short_prompt_passed" in gate:
        return (gate.get("short_prompt_ran") is True
                and gate.get("short_prompt_passed") is True
                and _finite_number(gate.get("short_prompt_relative_rms")))
    return (gate.get("passed") is True
            and _finite_number(gate.get("short_prompt_relative_rms")))


def install_longclip_sara_sidecars(unet, cfg: LongClipSaraConfig) -> dict[str, dict | None]:
    """Install configured resolution/P4 sidecars on a loaded SD U-Net.

    This is the canonical installer used by training, inference, generation, and
    checkpoint reload tests.  Resolution and P4 modules are trainable by default;
    callers that build SaRA masks must install them after mask construction.
    """
    summary: dict[str, dict | None] = {"resolution_conditioning": None, "p4": None}
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    if cfg.resolution_conditioning_enabled:
        from pure_ella.resolution import (
            ResolutionConditioner,
            install_resolution_conditioner,
            resolution_condition_schema,
        )
        existing = getattr(unet, "class_embedding", None)
        if existing is None:
            conditioner = ResolutionConditioner(
                output_dim=unet.time_embedding.linear_2.out_features,
                hidden_dim=cfg.resolution_conditioning_hidden_dim,
            ).to(device=device, dtype=dtype)
            install_resolution_conditioner(unet, conditioner)
        elif not isinstance(existing, ResolutionConditioner):
            raise RuntimeError(
                "Cannot install LongCLIP resolution sidecar: U-Net already has "
                f"class_embedding={type(existing).__name__}")
        summary["resolution_conditioning"] = resolution_condition_schema(
            hidden_dim=cfg.resolution_conditioning_hidden_dim)
    if cfg.p4_enabled:
        from pure_ella.p4 import install_p4_blocks, p4_schema
        if not hasattr(unet, "p4_blocks"):
            install_p4_blocks(
                unet,
                sites=cfg.p4_insertions,
                variant=cfg.p4_variant,
                hidden_dim=cfg.p4_hidden_dim,
                heads=cfg.p4_heads,
                ff_mult=cfg.p4_ff_mult,
                rope_base=cfg.p4_rope_base,
                timestep_adaln=cfg.p4_timestep_adaln,
            )
        summary["p4"] = p4_schema(
            enabled=True,
            variant=cfg.p4_variant,
            sites=cfg.p4_insertions,
            hidden_dim=(cfg.p4_hidden_dim or unet.config.block_out_channels[-1]),
            heads=cfg.p4_heads,
            ff_mult=cfg.p4_ff_mult,
            rope_base=cfg.p4_rope_base,
            timestep_adaln=cfg.p4_timestep_adaln)
    return summary


def load_longclip_sara_sidecar_state(unet, checkpoint: dict, cfg: LongClipSaraConfig,
                                     source: str) -> dict[str, int]:
    """Strictly load resolution/P4 sidecar state into already-installed modules."""
    loaded = {"resolution_tensors": 0, "p4_tensors": 0}
    install_longclip_sara_sidecars(unet, cfg)
    if cfg.resolution_conditioning_enabled:
        from pure_ella.resolution import ResolutionConditioner
        conditioner = getattr(unet, "class_embedding", None)
        if not isinstance(conditioner, ResolutionConditioner):
            raise RuntimeError(f"Resolution sidecar was not installed for {source}")
        state_dict = checkpoint["resolution_conditioner_state_dict"]
        conditioner.load_state_dict(state_dict, strict=True)
        loaded["resolution_tensors"] = len(state_dict)
    if cfg.p4_enabled:
        blocks = getattr(unet, "p4_blocks", None)
        if blocks is None:
            raise RuntimeError(f"P4 sidecars were not installed for {source}")
        state_dict = checkpoint["p4_state_dict"]
        blocks.load_state_dict(state_dict, strict=True)
        loaded["p4_tensors"] = len(state_dict)
    return loaded


def load_longclip_sara_checkpoint(
    unet,
    checkpoint: dict,
    cfg: LongClipSaraConfig,
    encoder: LongClipEncoder,
    source: str,
    *,
    load_sparse: bool = True,
    load_sidecars: bool = True,
    require_validation_gate: bool | None = None,
    require_short_prompt_regression_gate: bool | None = None,
) -> dict[str, Any]:
    """Validate and load a LongCLIP SaRA checkpoint without dropping sidecars."""
    validate_longclip_sara_checkpoint(
        checkpoint, cfg, encoder, source,
        require_validation_gate=require_validation_gate,
        require_short_prompt_regression_gate=require_short_prompt_regression_gate)
    result: dict[str, Any] = {"sparse_values": 0, "sidecars": {}}
    if load_sidecars:
        result["sidecars"] = load_longclip_sara_sidecar_state(
            unet, checkpoint, cfg, source)
    if load_sparse:
        from pure_ella.sara import load_sara_sparse_values
        result["sparse_values"] = load_sara_sparse_values(
            unet, checkpoint["sparse_values"])
    return result
