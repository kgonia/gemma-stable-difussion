"""
Configuration system for gemma3-sd-pure-ella training.

Loads a JSON config file and returns a typed config dataclass.
All uppercase keys from the JSON become module-level constants.
"""
from __future__ import annotations
import json
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
    experiment_stage: str = "stage1_77_pretrain_then_ella"  # stage1_77_pretrain_then_ella, stage2_long_context_no_sara, stage3_long_context_with_sara
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
    use_clip_teacher_delta: bool = True
    use_clip_teacher_delta_phase2: bool = False
    enable_unet_gradient_checkpointing: bool = True
    phase1_semantic_anchor_weight: float = 0.05
    phase2_semantic_anchor_weight: float = 0.0

    # --- Context tokens ---
    context_tokens: int = 77  # 77, 128, 192, or 256
    clip_anchor_tokens: int = 77  # always 77 for SD1.5
    long_context_target: int = 128  # used only when experiment_stage is stage2/stage3

    # --- SaRA phase ---
    run_sara_phase: bool = False
    run_reloaded_long_proof: bool = False

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
    max_gemma_len: int = 256
    clip_id: str = "openai/clip-vit-large-patch14"
    sd_checkpoint: str = ""  # path to .safetensors SD checkpoint, empty = use runwayml/stable-diffusion-v1-5

    # --- Connector ---
    connector_type: str = "trm_yz"  # ella_tsc, recursive_y, trm_yz
    connector_width: int = 768
    connector_layers: int = 4
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
    init_connector_partial_warmstart: bool = True
    require_stage2_warmstart: bool = False

    # --- Learning rates ---
    pretrain_lr: float = 1e-4
    ella_lr: float = 1e-4
    sara_lr: float = 1e-5
    lambda_diffusion: float = 1.0
    lambda_teacher: float = 0.5
    lambda_text_delta: float = 1.0

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
    stream_repo: str = "jackyhate/text-to-image-2M"

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
    run_complex_prompt_grids: bool = True
    run_suffix_counterfactual_grids: bool = True
    suffix_counterfactual_loss_weight: float = 0.0   # set >0 to add training-time suffix pressure
    suffix_counterfactual_loss_every: int = 0         # apply every N steps (0=disabled)
    suffix_counterfactual_loss_margin: float = 0.05   # hinge margin for rel_diff
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
    extra_token_diagnostic_timestep: int = 500
    extra_token_diagnostic_seed: int = 777

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
        self.run_reloaded_long_proof = self.context_tokens > self.clip_anchor_tokens

        # Resolve context_tokens from experiment_stage
        if self.experiment_stage == "stage1_77_pretrain_then_ella":
            self.context_tokens = 77
        elif self.experiment_stage in {"stage2_long_context_no_sara", "stage3_long_context_with_sara"}:
            self.context_tokens = self.long_context_target
        else:
            raise ValueError(f"Unknown experiment_stage: {self.experiment_stage}")

        assert self.context_tokens in {77, 128, 192, 256}
        assert self.context_tokens >= self.clip_anchor_tokens

        # Run mode overrides
        if self.run_mode == "overfit_train":
            self.max_samples_pretrain = 64
            self.max_samples_ella = 64
            self.pretrain_epochs = 20
            self.ella_epochs = 40
            self.sara_epochs = 20
            self.pretrain_max_opt_steps = 300
            self.ella_max_opt_steps = 600
            self.sara_max_opt_steps = 400
            self.validation_every_opt_steps = 50
            self.shuffle_streaming = False
        elif self.run_mode == "diagnostic":
            self.max_samples_pretrain = 128
            self.max_samples_ella = 128
            self.pretrain_epochs = 0
            self.ella_epochs = 0
            self.sara_epochs = 0
            self.pretrain_max_opt_steps = 0
            self.ella_max_opt_steps = 0
            self.sara_max_opt_steps = 0
            self.validation_every_opt_steps = 0
            self.shuffle_streaming = True

        self.quality_every_opt_steps = self.validation_every_opt_steps

        # Resolve SD checkpoint
        if not self.sd_checkpoint:
            self.sd_checkpoint = "runwayml/stable-diffusion-v1-5"

    def to_dict(self) -> dict:
        """Export non-default config as dict."""
        return {k: v for k, v in asdict(self).items()}

    def print_plan(self):
        """Print a readable plan summary."""
        print(f"Experiment: {self.experiment_stage} run_mode={self.run_mode}")
        print(f"Context tokens: {self.context_tokens} anchor={self.clip_anchor_tokens}")
        print(f"Connector: {self.connector_type}")
        print(f"CLIP pretrain: {self.run_clip_alignment_pretrain} (max_steps={self.pretrain_max_opt_steps})")
        print(f"ELLA training: steps≤{self.ella_max_opt_steps}")
        print(f"SaRA: {self.run_sara_phase}")
        print(f"Gemma: {self.gemma_id} layer={self.gemma_layer_index}")
        print(f"SD checkpoint: {self.sd_checkpoint}")
        print(f"Dataset: {self.stream_repo} batch={self.train_batch_size}")
        print(f"Output: {self.output_dir}")

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        """Load config from JSON file, allows partial overrides."""
        cfg = cls()  # defaults
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

        # Apply overrides
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        # Re-derive computed fields
        cfg.__post_init__()
        return cfg

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
        if p and os.path.exists(p):
            return p
        # Also check if it's a HuggingFace model ID (no slash prefix needed)
        if p and not os.path.exists(p) and "/" in p:
            return p  # assume it's a valid HF model ID
    # Fall back to the configured checkpoint
    return cfg.sd_checkpoint


def seed_everything(seed: int):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
