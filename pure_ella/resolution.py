"""Canvas-resolution conditioning sidecar for SD U-Net timestep embeddings."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


RESOLUTION_CONDITION_DIM = 2
RESOLUTION_SCHEMA_VERSION = 2


def resolution_condition_schema(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "version": RESOLUTION_SCHEMA_VERSION,
        "condition_dim": RESOLUTION_CONDITION_DIM,
        "feature_names": ["log2_width_over_1024", "log2_height_over_1024"],
        "normalization": {
            "width": "log2(width / 1024)",
            "height": "log2(height / 1024)",
        },
        "source": "emitted_canvas_bucket_or_latent_shape",
        "hidden_dim": int(hidden_dim),
        "resolution_always_known": True,
    }


def make_resolution_condition(
    width: int | float,
    height: int | float,
    *,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    width = float(width)
    height = float(height)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("resolution width must be positive and finite")
    if not math.isfinite(height) or height <= 0:
        raise ValueError("resolution height must be positive and finite")
    values = torch.tensor(
        [math.log2(width / 1024.0), math.log2(height / 1024.0)],
        dtype=torch.float32,
        device=device,
    )
    return values.reshape(1, RESOLUTION_CONDITION_DIM).expand(int(batch_size), -1)


def make_resolution_condition_from_latents(
    sample: torch.Tensor,
    *,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    if sample.ndim != 4:
        raise ValueError(f"latent sample must have shape [B,C,H,W], got {tuple(sample.shape)}")
    height = int(sample.shape[-2]) * int(vae_scale_factor)
    width = int(sample.shape[-1]) * int(vae_scale_factor)
    return make_resolution_condition(
        width, height, batch_size=sample.shape[0], device=sample.device)


def make_resolution_condition_from_bucket(
    bucket: Any,
    *,
    batch_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(bucket, torch.Tensor):
        flat = bucket.detach().cpu().reshape(-1).tolist()
    else:
        flat = list(bucket)
    if len(flat) < 2:
        raise ValueError(f"bucket must contain width and height, got {bucket!r}")
    width, height = flat[:2]
    return make_resolution_condition(
        width, height, batch_size=batch_size, device=device)


class ResolutionConditioner(nn.Module):
    """Zero-output MLP for emitted-canvas width/height conditioning.

    The final projection is zero-initialized so installing the sidecar is exactly
    identity at step zero.  Gradients still flow into the final projection on the
    first step; earlier layers train once it moves away from zero.
    """

    def __init__(self, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.input = nn.Linear(RESOLUTION_CONDITION_DIM, self.hidden_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != RESOLUTION_CONDITION_DIM:
            raise ValueError(
                "resolution condition must have shape "
                f"[batch, {RESOLUTION_CONDITION_DIM}]"
            )
        condition = condition.to(dtype=self.input.weight.dtype)
        return self.output(self.activation(self.input(condition)))


def install_resolution_conditioner(unet, conditioner: ResolutionConditioner) -> None:
    if getattr(unet, "class_embedding", None) is not None:
        raise ValueError("UNet already has class conditioning; resolution branch conflicts")
    unet.class_embedding = conditioner
