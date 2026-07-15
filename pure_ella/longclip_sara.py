"""LongCLIP-L conditioning and provenance helpers for sparse U-Net tuning.

LongCLIP is intentionally frozen here.  Its value is that ``encode_text_full``
already emits 248 CLIP-compatible, 768-wide states; training it alongside a
U-Net would destroy the direct-replacement baseline we are trying to measure.
"""
from __future__ import annotations

import hashlib
import importlib
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


@dataclass
class LongClipSaraConfig:
    """Configuration for direct LongCLIP-L + sparse-SaRA adaptation."""

    sd_checkpoint: str
    longclip_repo: str
    longclip_checkpoint: str
    output_dir: str
    data_sources: list[str]
    validation_data_sources: list[str] = field(default_factory=list)
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
    lr_warmup_steps: int = 100
    lr_decay_steps: int = 1463
    grad_clip_norm: float = 0.5
    shuffle_streaming: bool = True
    shuffle_buffer: int = 10000
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
    complex_generation_cases: list[dict[str, Any]] = field(
        default_factory=lambda: TrainConfig().complex_generation_cases)
    run_final_visual_check: bool = True
    wandb_enabled: bool = True
    wandb_project: str = "gemma3-sd-pure-ella"
    wandb_entity: str = ""
    wandb_run_name: str = ""

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
        if self.mixed_precision not in {"no", "bf16"}:
            raise ValueError("mixed_precision must be 'no' or 'bf16'")
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
    def encode(self, prompts) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        # First attempt the strict path.  For corpus training, callers may
        # explicitly permit clipping; in that case count it instead of hiding
        # it behind LongCLIP's ``truncate=True`` convenience switch.
        try:
            tokens = self._longclip.tokenize(
                prompts, context_length=LONGCLIP_L_CONTEXT_TOKENS,
                truncate=False)
        except RuntimeError:
            if self.fail_on_truncation:
                raise
            overlong = 0
            for prompt in prompts:
                try:
                    self._longclip.tokenize(
                        [prompt], context_length=LONGCLIP_L_CONTEXT_TOKENS,
                        truncate=False)
                except RuntimeError:
                    overlong += 1
            self.truncated_prompt_count += overlong
            if self.truncated_prompt_count == overlong:
                print(
                    "WARNING: LongCLIP training truncation enabled; clipped "
                    f"{overlong}/{len(prompts)} prompt(s) to "
                    f"{LONGCLIP_L_CONTEXT_TOKENS} tokens")
            tokens = self._longclip.tokenize(
                prompts, context_length=LONGCLIP_L_CONTEXT_TOKENS,
                truncate=True)
        tokens = tokens.to(self.device)
        lengths = tokens.ne(0).sum(dim=1)
        self.max_observed_tokens = max(self.max_observed_tokens, int(lengths.max()))
        states = self.model.encode_text_full(tokens)
        expected = (len(prompts), LONGCLIP_L_CONTEXT_TOKENS, LONGCLIP_L_WIDTH)
        if tuple(states.shape) != expected:
            raise RuntimeError(
                f"LongCLIP conditioning shape mismatch: expected {expected}, "
                f"got {tuple(states.shape)}")
        if not torch.isfinite(states).all():
            raise RuntimeError("LongCLIP emitted NaN/Inf conditioning")
        return states, tokens.ne(0).long()

    def provenance(self) -> dict:
        return {
            "encoder": "BeichenZhang/LongCLIP-L",
            "context_tokens": LONGCLIP_L_CONTEXT_TOKENS,
            "width": LONGCLIP_L_WIDTH,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
        }


def longclip_sara_schema(cfg: LongClipSaraConfig, encoder: LongClipEncoder) -> dict:
    return {
        "version": 1,
        "conditioning_backend": "longclip_l_direct",
        "longclip": encoder.provenance(),
        "sd_checkpoint": str(Path(cfg.sd_checkpoint).expanduser().resolve()),
        "sara_target_substrings": list(cfg.sara_target_substrings),
        "sara_selection_mode": cfg.sara_selection_mode,
        "sara_target_fraction": cfg.sara_target_fraction,
    }


def validate_longclip_sara_checkpoint(
    checkpoint: dict, cfg: LongClipSaraConfig, encoder: LongClipEncoder,
    source: str,
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
