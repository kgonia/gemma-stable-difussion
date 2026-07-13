"""
Configuration system for gemma3-sd-pure-ella training.

Loads a JSON config file and returns a typed config dataclass.
All uppercase keys from the JSON become module-level constants.
"""
from __future__ import annotations
import json
import math
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal
import random
import numpy as np
import torch


@dataclass
class TrainConfig:
    """All configuration fields with defaults matching the notebook."""

    # --- Experiment stage ---
    experiment_stage: str = "stage2_long_context_no_sara"  # stage1_77_pretrain_then_ella, stage2_long_context_no_sara, stage3_long_context_with_sara
    run_mode: str = "short_train"  # diagnostic, overfit_train, short_train, full_train
    run_training: bool = True
    run_final_proof: bool = True
    run_fixed_validation_grids: bool = True
    run_long_context_diagnostics: bool = True
    run_complex_prompt_grids: bool = True
    run_suffix_counterfactual_grids: bool = True
    run_unet_attention_proof: bool = True

    # --- CLIP scaffold ---
    run_clip_alignment_pretrain: bool = True
    use_clip_teacher_delta: bool = False
    use_clip_teacher_delta_phase2: bool = False
    enable_unet_gradient_checkpointing: bool = True
    phase1_semantic_anchor_weight: float = 0.0
    phase2_semantic_anchor_weight: float = 0.0

    # --- Text input and U-Net conditioning contracts ---
    # Gemma may read a long prompt while the connector preserves SD1.5's
    # fixed 77-token cross-attention contract.
    context_tokens: int = 77
    clip_anchor_tokens: int = 77  # always 77 for SD1.5

    # --- SaRA phase ---
    run_sara_phase: bool = False

    # --- Data budget ---
    max_samples_pretrain: int = 6000
    max_samples_ella: int = 6000
    max_samples_sara: int = 6000
    pretrain_epochs: int = 1
    ella_epochs: int = 1
    sara_epochs: int = 1
    pretrain_max_opt_steps: int = 1500
    ella_max_opt_steps: int = 1500
    sara_max_opt_steps: int = 500
    validation_every_opt_steps: int = 250
    generation_grid_every_opt_steps: int = 500  # 0 = off; save image grid during training
    shuffle_streaming: bool = True

    # --- Model IDs ---
    gemma_id: str = "google/gemma-3-270m-it"
    gemma_layer_index: int = -1
    gemma_layer_mix_count: int = 4
    max_gemma_len: int = 336
    fail_on_prompt_truncation: bool = True
    clip_id: str = "openai/clip-vit-large-patch14"
    sd_checkpoint: str = ""  # path to .safetensors SD checkpoint, empty = use runwayml/stable-diffusion-v1-5

    # --- Connector ---
    connector_type: str = "ella_tsc"  # ella_tsc is the supported baseline
    connector_width: int = 768
    connector_layers: int = 6
    connector_heads: int = 8
    connector_ff_mult: int = 4
    connector_dropout: float = 0.0
    connector_time_embed_dim: int = 768
    connector_extra_gate_init: float = -5.0
    recursive_y_steps: int = 2
    recursive_y_gate_init: float = -2.0
    trm_outer_steps: int = 3
    trm_inner_steps: int = 2
    trm_scratch_tokens: int = 32
    trm_y_gate_init: float = -2.0
    trm_z_gate_init: float = -1.0
    init_connector_ckpt_path: str = ""
    init_connector_partial_warmstart: bool = False
    require_stage2_warmstart: bool = False

    # --- Learning rates ---
    pretrain_lr: float = 1e-4
    ella_lr: float = 1e-4
    sara_lr: float = 1e-5
    lambda_diffusion: float = 1.0
    lambda_teacher: float = 0.5
    lambda_text_delta: float = 1.0
    clip_teacher_decay_steps: int = 5000  # optional short-caption scaffold only

    # --- SaRA ---
    sara_scope: str = "attn2_kv_sparse"
    sara_target_substrings: tuple = ("attn2.to_k", "attn2.to_v")
    sara_threshold: float = 1e-3
    sara_max_sparse_fraction_warn: float = 0.02
    sara_max_sparse_fraction_abort: float = 0.05

    # --- Dataset ---
    train_batch_size: int = 4
    shuffle_buffer: int = 10000
    grad_clip_norm: float = 0.5
    data_sources: List[str] = field(default_factory=lambda: [
        "jackyhate/text-to-image-2M",
    ])
    max_image_dimension: int = 1024
    aspect_ratio_buckets: List[List[int]] = field(default_factory=lambda: [
        [1024, 1024], [1024, 896], [1024, 768], [1024, 640], [1024, 512],
        [896, 1024], [768, 1024], [640, 1024], [512, 1024],
    ])
    drop_last_bucket_batches: bool = True
    caption_mix_short: float = 0.25
    caption_mix_medium: float = 0.25
    caption_mix_long: float = 0.50
    conditioning_dropout_prob: float = 0.1

    # --- Quality metrics ---
    run_image_quality_metrics: bool = True
    run_fid: bool = True
    run_kid: bool = True
    run_clip_score: bool = False
    quality_every_opt_steps: int = 250
    fid_every_opt_steps: int = 1000
    fid_reference_size: int = 512
    fid_generation_size: int = 64
    kid_reference_size: int = 256
    kid_generation_size: int = 64
    quality_generation_steps: int = 15
    quality_guidance: float = 5.5
    quality_seed: int = 4242

    # --- Validation ---
    base_seed: int = 1234
    val_steps: int = 30
    val_guidance: float = 5.5
    val_seed: int = 777
    val_prompts: List[str] = field(default_factory=lambda: [
        "a cat sitting on a windowsill looking outside",
        "a watercolor painting of a mountain lake",
        "a neon-lit cyberpunk alleyway at night",
    ])
    # --- Complex / long-context prompt grids (from colab notebook) ---
    complex_prompt_steps: int = 30
    complex_prompt_guidance: float = 5.5
    complex_prompt_seed: int = 777
    complex_prompt_width: int = 512
    complex_prompt_height: int = 512
    complex_generation_cases: List[dict] = field(default_factory=lambda: [
        {
            "name": "retrofuturist_magazine_cars",
            "prompt": "A highly detailed retrofuturist magazine infographic about future cars, laid out like a beautifully preserved issue of Popular Mechanics, with elegant diagram panels, mechanical callouts, polished concept-art rendering, vivid color, and the feeling of an award-winning poster-sized editorial spread.",
            "steps": 30, "guidance": 7.0, "seed": 3197632166, "width": 512, "height": 512,
        },
        {
            "name": "warrior_princess_poster",
            "prompt": "A dramatic poster of a warrior princess standing centered on a hill as the main cinematic key visual, with intricate linework, vibrant colors, panoramic scale, breathtaking fantasy atmosphere, and the finish of a carefully painted illustrated poster.",
            "steps": 30, "guidance": 7.0, "seed": 4267154965, "width": 512, "height": 512,
        },
    ])
    long_eval_prompts: List[str] = field(default_factory=lambda: [
        "a cinematic photo of a red vintage motorcycle parked beside a stone cottage, with a brass telescope on the seat, blue wildflowers in the basket, and a tiny owl perched on the handlebar at sunrise",
        "a detailed product photograph of hiking boots on a wooden table, with orange laces, a folded trail map, a silver compass, and raindrops on the leather",
    ])
    long_eval_short_controls: List[str] = field(default_factory=lambda: [
        "a cinematic photo of a red vintage motorcycle parked beside a stone cottage",
        "a detailed product photograph of hiking boots on a wooden table",
    ])
    suffix_counterfactual_prefix: str = (
        "Describe a single coherent high quality image with natural lighting and realistic materials. "
        "The composition is centered and calm, with one main subject in the foreground, a readable background, "
        "gentle shadows, balanced colors, clear edges, and no text overlays. Keep the camera angle slightly low, "
        "the lens natural, the scene uncluttered, the mood quiet, and the details consistent. "
        "The setting includes a wooden table, linen cloth, ceramic bowl, glass vase, folded paper, small candle, "
        "soft window light, distant plants, and warm reflections. "
        "Use this exact shared setup before the final object instruction:"
    )
    suffix_counterfactual_cases: List[dict] = field(default_factory=lambda: [
        {
            "name": "object_after_anchor_counterfactual",
            "short_prompt": "A calm realistic still life scene on a wooden table in soft window light.",
            "prompt_suffix_a": "Final instruction: make the main object a red ceramic teapot with white flowers painted on it.",
            "prompt_suffix_b": "Final instruction: make the main object a blue glass violin with silver strings resting beside it.",
            "seed": 1201, "steps": 30, "guidance": 5.5, "width": 512, "height": 512,
        },
        {
            "name": "action_after_anchor_counterfactual",
            "short_prompt": "A cinematic outdoor scene with a person standing in a quiet garden at golden hour.",
            "prompt_suffix_a": "Final instruction: show a woman in a yellow raincoat feeding a small black raven from her hand.",
            "prompt_suffix_b": "Final instruction: show a man in a purple velvet jacket tuning a brass telescope on a tripod.",
            "seed": 2202, "steps": 30, "guidance": 5.5, "width": 512, "height": 512,
        },
    ])
    suffix_diagnostic_timestep: int = 500

    # --- Wandb ---
    wandb_enabled: bool = True
    wandb_project: str = "gemma3-sd-pure-ella"
    wandb_entity: str = ""

    # --- Output ---
    output_dir: str = "./output"
    resume_checkpoint: str = ""

    def __post_init__(self):
        """Derive computed fields."""
        self.run_training = self.run_mode in {"overfit_train", "short_train", "full_train"}
        self.run_sara_phase = self.experiment_stage == "stage3_long_context_with_sara"
        if self.experiment_stage not in {
            "stage1_77_pretrain_then_ella",
            "stage2_long_context_no_sara",
            "stage3_long_context_with_sara",
        }:
            raise ValueError(f"Unknown experiment_stage: {self.experiment_stage}")
        if self.context_tokens != self.clip_anchor_tokens:
            raise ValueError(
                "The supported connector preserves SD1.5's 77-token conditioning "
                "contract. Set context_tokens == clip_anchor_tokens; use "
                "max_gemma_len for long prompts."
            )
        if self.max_gemma_len < self.clip_anchor_tokens:
            raise ValueError("max_gemma_len must be at least clip_anchor_tokens")
        if self.gemma_layer_mix_count < 1:
            raise ValueError("gemma_layer_mix_count must be positive")
        if not 0.0 <= self.conditioning_dropout_prob < 1.0:
            raise ValueError("conditioning_dropout_prob must be in [0, 1)")
        caption_mix = (
            self.caption_mix_short,
            self.caption_mix_medium,
            self.caption_mix_long,
        )
        if any(weight < 0 for weight in caption_mix):
            raise ValueError("caption mix weights must be non-negative")
        if not math.isclose(sum(caption_mix), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("caption mix weights must sum to 1.0")
        if not self.aspect_ratio_buckets:
            raise ValueError("aspect_ratio_buckets must not be empty")
        if not self.data_sources or any(
                not isinstance(source, str) or not source.strip()
                for source in self.data_sources):
            raise ValueError("data_sources must contain at least one location")
        if self.max_image_dimension <= 0:
            raise ValueError("max_image_dimension must be positive")
        for bucket in self.aspect_ratio_buckets:
            if len(bucket) != 2 or any(int(v) <= 0 or int(v) % 8 for v in bucket):
                raise ValueError(
                    f"Invalid aspect-ratio bucket {bucket}; dimensions must be positive multiples of 8"
                )
            if max(map(int, bucket)) > self.max_image_dimension:
                raise ValueError(
                    f"Invalid aspect-ratio bucket {bucket}; dimensions exceed "
                    f"max_image_dimension={self.max_image_dimension}"
                )

        # Resolve SD checkpoint
        if not self.sd_checkpoint:
            self.sd_checkpoint = "runwayml/stable-diffusion-v1-5"

    def to_dict(self) -> dict:
        """Export non-default config as dict."""
        return {k: v for k, v in asdict(self).items()}

    def print_plan(self):
        """Print a readable plan summary."""
        print(f"Experiment: {self.experiment_stage} run_mode={self.run_mode}")
        print(f"Gemma input tokens: {self.max_gemma_len}")
        print(f"UNet conditioning tokens: {self.context_tokens} anchor={self.clip_anchor_tokens}")
        print(f"Connector: {self.connector_type}")
        print(f"CLIP pretrain: {self.run_clip_alignment_pretrain} (max_steps={self.pretrain_max_opt_steps})")
        print(f"ELLA training: steps≤{self.ella_max_opt_steps}")
        print(f"SaRA: {self.run_sara_phase}")
        print(f"Gemma: {self.gemma_id} layer={self.gemma_layer_index}")
        print(f"SD checkpoint: {self.sd_checkpoint}")
        print(
            f"Datasets: {self.data_sources} batch={self.train_batch_size} "
            f"max_image_dimension={self.max_image_dimension}"
        )
        print(f"Output: {self.output_dir}")

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        """Load config from JSON file, allows partial overrides."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        valid = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        return cls(**data)

    def save_json(self, path: str):
        """Save current config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def resolve_sd_checkpoint(cfg: TrainConfig) -> str:
    """Resolve SD checkpoint path."""
    candidates = [
        cfg.sd_checkpoint,
        os.path.expanduser("~/models/stylejourney_v10.safetensors"),
        "/content/drive/MyDrive/model/stylejourney_v10.safetensors",
    ]
    for p in candidates:
        expanded = os.path.expanduser(p) if p else p
        if expanded and os.path.exists(expanded):
            return expanded

    configured = cfg.sd_checkpoint.strip()
    looks_local = (
        configured.startswith(("/", ".", "~"))
        or configured.endswith((".safetensors", ".ckpt", ".pt", ".bin"))
    )
    if not looks_local and configured.count("/") == 1:
        return configured
    raise FileNotFoundError(
        "Stable Diffusion checkpoint not found. Checked: "
        + ", ".join(os.path.expanduser(p) for p in candidates if p)
    )


def seed_everything(seed: int):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
