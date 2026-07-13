"""Camera metadata extraction and centered SD timestep conditioning."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

import torch
import torch.nn as nn


CAMERA_VALUE_COUNT = 4
CAMERA_CONDITION_DIM = 9
CAMERA_CAPTURE_TYPES = ("unknown", "photo", "render", "artwork")
CAMERA_SCHEMA_VERSION = 2
CAMERA_FEATURE_NAMES = (
    "vertical_fov_deg_normalized",
    "focal_length_mm_normalized",
    "aperture_f_number_normalized",
    "iso_normalized",
    "vertical_fov_present",
    "focal_length_present",
    "aperture_present",
    "iso_present",
    "capture_type_id",
)
_CAPTURE_TO_ID = {name: index for index, name in enumerate(CAMERA_CAPTURE_TYPES)}


def camera_condition_schema(*, use_capture_type: bool) -> dict:
    """Return the semantic schema that must match exactly across checkpoints."""
    return {
        "version": CAMERA_SCHEMA_VERSION,
        "condition_dim": CAMERA_CONDITION_DIM,
        "feature_names": list(CAMERA_FEATURE_NAMES),
        "normalization": {
            "vertical_fov_deg": "clip[5,150]; (x-75)/70",
            "focal_length_mm": "clip[2,1200]; log(x/50)/log(10)",
            "aperture_f_number": "clip[0.5,64]; log(x/4)/log(4)",
            "iso": "clip[12.5,409600]; log(x/100)/log(64)",
        },
        "unknown_centered": True,
        "use_capture_type": bool(use_capture_type),
    }


def _mapping(value: Any) -> dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return value if isinstance(value, dict) else {}


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _first_float(sources: list[Mapping[str, Any]], *keys: str) -> float | None:
    for source in sources:
        for key in keys:
            value = _finite_float(source.get(key))
            if value is not None:
                return value
    return None


def _capture_type(sample: Mapping[str, Any], metadata: Mapping[str, Any],
                  exif: Mapping[str, Any]) -> str:
    explicit = " ".join(
        str(value).lower()
        for source in (sample, metadata)
        for key in ("capture_type", "media_type", "image_type")
        if (value := source.get(key)) is not None
    )
    text = " ".join(
        str(sample.get(key, "")).lower()
        for key in ("caption", "caption_detailed", "upstream_keywords")
    )
    combined = f"{explicit} {text}"
    if re.search(r"\b(render|cgi|3d render|digital render)\b", combined):
        return "render"
    if re.search(r"\b(illustration|painting|drawing|artwork)\b", combined):
        return "artwork"
    if "photo" in explicit or exif or metadata:
        return "photo"
    return "unknown"


def extract_camera_metadata(sample: Mapping[str, Any]) -> dict:
    """Extract physical values without fabricating missing sensor information."""
    metadata = {}
    exif = {}
    for key in ("upstream_json", "json", "metadata"):
        candidate = _mapping(sample.get(key))
        candidate_exif = _mapping(candidate.get("exif"))
        metadata.update({
            nested_key: value
            for nested_key, value in candidate.items()
            if nested_key != "exif"
        })
        exif.update(candidate_exif)
    sources = [sample, exif, metadata]

    vertical_fov = _first_float(
        sources, "vertical_fov", "vertical_fov_deg", "fov_vertical_deg")
    focal_35mm = _first_float(
        sources, "focal_length_35mm", "focal_length_in_35mm_film",
        "focal_length_35mm_equivalent")
    if vertical_fov is None and focal_35mm is not None:
        vertical_fov = math.degrees(2.0 * math.atan(24.0 / (2.0 * focal_35mm)))

    return {
        "vertical_fov_deg": vertical_fov,
        "focal_length_mm": _first_float(
            sources, "focal_length", "focal_length_mm"),
        "aperture_f_number": _first_float(
            sources, "aperture_value", "aperture", "f_number", "f_stop"),
        "iso": _first_float(sources, "iso", "iso_speed", "iso_speed_ratings"),
        "capture_type": _capture_type(sample, metadata, exif),
    }


def camera_metadata_to_tensor(metadata: Mapping[str, Any]) -> torch.Tensor:
    """Return normalized values, value-presence bits, and a capture-type ID."""
    raw = tuple(_finite_float(metadata.get(key)) for key in (
        "vertical_fov_deg",
        "focal_length_mm",
        "aperture_f_number",
        "iso",
    ))
    present = [value is not None for value in raw]

    def clipped(value, low, high):
        return min(max(float(value), low), high)

    values = [0.0] * CAMERA_VALUE_COUNT
    if present[0]:
        values[0] = (clipped(raw[0], 5.0, 150.0) - 75.0) / 70.0
    if present[1]:
        values[1] = math.log(clipped(raw[1], 2.0, 1200.0) / 50.0) / math.log(10.0)
    if present[2]:
        values[2] = math.log(clipped(raw[2], 0.5, 64.0) / 4.0) / math.log(4.0)
    if present[3]:
        values[3] = math.log(clipped(raw[3], 12.5, 409600.0) / 100.0) / math.log(64.0)
    capture_id = _CAPTURE_TO_ID.get(
        str(metadata.get("capture_type", "unknown")).lower(), 0)
    return torch.tensor(
        values + [float(value) for value in present] + [float(capture_id)],
        dtype=torch.float32,
    )


def make_camera_condition(
    *, vertical_fov_deg: float | None = None,
    focal_length_mm: float | None = None,
    aperture_f_number: float | None = None,
    iso: float | None = None,
    capture_type: str = "unknown",
) -> torch.Tensor:
    return camera_metadata_to_tensor({
        "vertical_fov_deg": vertical_fov_deg,
        "focal_length_mm": focal_length_mm,
        "aperture_f_number": aperture_f_number,
        "iso": iso,
        "capture_type": capture_type,
    })


def apply_camera_dropout(condition: torch.Tensor, probability: float) -> torch.Tensor:
    """Replace complete per-sample metadata records with the unknown condition."""
    if probability <= 0 or condition.numel() == 0:
        return condition
    dropped = condition.clone()
    mask = torch.rand(condition.shape[0], device=condition.device) < probability
    dropped[mask] = 0
    return dropped


class CameraConditioner(nn.Module):
    """Map camera scalars into SD's timestep embedding space.

    Every output is centered on the learned unknown output. Consequently an
    unknown record remains exactly zero after every optimizer update.
    """

    def __init__(self, output_dim: int, hidden_dim: int = 512,
                 fourier_bands: int = 6, capture_embed_dim: int = 32,
                 use_capture_type: bool = False):
        super().__init__()
        if output_dim <= 0 or hidden_dim <= 0 or fourier_bands <= 0:
            raise ValueError("Camera conditioner dimensions must be positive")
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.fourier_bands = int(fourier_bands)
        self.use_capture_type = bool(use_capture_type)
        self.capture_embed_dim = (
            int(capture_embed_dim) if self.use_capture_type else 0)
        self.register_buffer(
            "frequencies",
            torch.pi * (2.0 ** torch.arange(fourier_bands, dtype=torch.float32)),
        )
        self.capture_embedding = (
            nn.Embedding(len(CAMERA_CAPTURE_TYPES), self.capture_embed_dim)
            if self.use_capture_type else None
        )
        input_dim = (
            CAMERA_VALUE_COUNT * (1 + 2 * fourier_bands)
            + CAMERA_VALUE_COUNT
            + self.capture_embed_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _features(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != CAMERA_CONDITION_DIM:
            raise ValueError(
                f"camera condition must have shape [batch, {CAMERA_CONDITION_DIM}]"
            )
        values = condition[:, :CAMERA_VALUE_COUNT].float()
        present = condition[:, CAMERA_VALUE_COUNT:2 * CAMERA_VALUE_COUNT].float()
        values = values * present
        angles = values.unsqueeze(-1) * self.frequencies
        fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)
        parts = [values, fourier, present]
        if self.capture_embedding is not None:
            capture_ids = condition[:, -1].long().clamp(
                0, len(CAMERA_CAPTURE_TYPES) - 1)
            parts.append(self.capture_embedding(capture_ids))
        return torch.cat(parts, dim=1).to(dtype=self.mlp[0].weight.dtype)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        features = self._features(condition)
        unknown_features = self._features(torch.zeros_like(condition))
        residual = self.mlp(features) - self.mlp(unknown_features)
        unknown = condition.eq(0).all(dim=1, keepdim=True)
        return residual.masked_fill(unknown, 0)

    def unknown(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            int(batch_size), CAMERA_CONDITION_DIM,
            device=device, dtype=torch.float32)


def install_camera_conditioner(unet, conditioner: CameraConditioner) -> None:
    if getattr(unet, "class_embedding", None) is not None:
        raise ValueError("UNet already has class conditioning; camera branch conflicts")
    unet.class_embedding = conditioner


def camera_conditioned_unet(
    unet, sample: torch.Tensor, timestep: torch.Tensor,
    *, encoder_hidden_states: torch.Tensor,
    camera_condition: torch.Tensor | None = None, **kwargs,
):
    """Call a UNet with camera labels when the P3 branch is installed."""
    conditioner = getattr(unet, "class_embedding", None)
    if isinstance(conditioner, CameraConditioner):
        if camera_condition is None:
            camera_condition = conditioner.unknown(sample.shape[0], sample.device)
        else:
            camera_condition = camera_condition.to(
                device=sample.device, dtype=torch.float32)
        kwargs["class_labels"] = camera_condition
    return unet(
        sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
