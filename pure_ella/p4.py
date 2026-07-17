"""P4 zero-gated residual-delta transformer blocks for SD U-Net features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


P4_SCHEMA_VERSION = 1
P4_SITE_PRE_MID = "pre_mid"
P4_SITE_POST_MID = "post_mid"
P4_ALLOWED_SITES = (P4_SITE_PRE_MID, P4_SITE_POST_MID)


@dataclass
class P4HookBundle:
    handles: list[Any]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def p4_schema(
    *,
    enabled: bool,
    variant: str,
    sites: list[str] | tuple[str, ...],
    hidden_dim: int,
    heads: int,
    ff_mult: float,
    rope_base: float,
    timestep_adaln: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    return {
        "version": P4_SCHEMA_VERSION,
        "identity": "external_tanh_gate_exact_zero",
        "branch": "residual_delta_attention_plus_ffn",
        "variant": variant,
        "sites": list(sites),
        "first_site": "after down_blocks[-1], before mid_block",
        "hidden_dim": int(hidden_dim),
        "heads": int(heads),
        "ff_mult": float(ff_mult),
        "timestep_adaln": bool(timestep_adaln),
        "positional_encoding": (
            {
                "type": "axial_2d_rope",
                "rope_base": float(rope_base),
                "row_major": True,
                "origin": "upper_left",
                "rotate": "qk_only",
                "head_dim_requires_multiple_of": 4,
            }
            if variant == "rope" else {"type": "none"}
        ),
        "attention_mask": "none",
    }


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    x_even = x_fp32[..., 0::2]
    x_odd = x_fp32[..., 1::2]
    rotated = torch.stack((
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos,
    ), dim=-1).flatten(-2)
    return rotated.to(dtype=x.dtype)


def _axial_rope_sincos(
    h: int,
    w: int,
    axis_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis_dim % 2 != 0:
        raise ValueError("axis rotary dimension must be even")
    inv_freq = 1.0 / (
        float(base) ** (
            torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32)
            / max(axis_dim, 1)
        )
    )
    y = torch.arange(h, device=device, dtype=torch.float32).repeat_interleave(w)
    x = torch.arange(w, device=device, dtype=torch.float32).repeat(h)
    y_freq = torch.outer(y, inv_freq)
    x_freq = torch.outer(x, inv_freq)
    return (
        y_freq.cos().to(dtype=dtype),
        y_freq.sin().to(dtype=dtype),
        x_freq.cos().to(dtype=dtype),
        x_freq.sin().to(dtype=dtype),
    )


def apply_axial_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    h: int,
    w: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply axial 2D RoPE to Q/K shaped [B, heads, H*W, head_dim]."""
    if q.shape != k.shape:
        raise ValueError("q and k must have matching shapes for RoPE")
    if q.ndim != 4:
        raise ValueError(f"q/k must be [B, heads, N, head_dim], got {tuple(q.shape)}")
    if q.shape[-2] != int(h) * int(w):
        raise ValueError("RoPE token count does not match H*W")
    head_dim = q.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError("axial 2D RoPE requires head_dim % 4 == 0")
    axis_dim = head_dim // 2
    y_cos, y_sin, x_cos, x_sin = _axial_rope_sincos(
        int(h), int(w), axis_dim, base, q.device, q.dtype)
    # Broadcast over batch and heads: [1, 1, N, pairs].
    y_cos = y_cos.unsqueeze(0).unsqueeze(0)
    y_sin = y_sin.unsqueeze(0).unsqueeze(0)
    x_cos = x_cos.unsqueeze(0).unsqueeze(0)
    x_sin = x_sin.unsqueeze(0).unsqueeze(0)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        y_part = _rotate_pairs(tensor[..., :axis_dim], y_cos, y_sin)
        x_part = _rotate_pairs(tensor[..., axis_dim:], x_cos, x_sin)
        return torch.cat((y_part, x_part), dim=-1)

    return rotate(q), rotate(k)


class P4SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, *, use_rope: bool, rope_base: float):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        if self.hidden_dim % self.heads != 0:
            raise ValueError("P4 hidden_dim must be divisible by heads")
        self.head_dim = self.hidden_dim // self.heads
        if use_rope and self.head_dim % 4 != 0:
            raise ValueError("P4 axial RoPE requires per-head dim divisible by 4")
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, tokens: torch.Tensor, *, h: int, w: int) -> torch.Tensor:
        batch, tokens_count, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, tokens_count, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_rope:
            q, k = apply_axial_2d_rope(q, k, h=h, w=w, base=self.rope_base)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch, tokens_count, self.hidden_dim)
        return self.out(out)


class P4DeltaBlock(nn.Module):
    """Exact-zero external-gated residual-delta transformer for [B,C,H,W]."""

    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        *,
        hidden_dim: int | None = None,
        heads: int = 8,
        ff_mult: float = 2.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        timestep_adaln: bool = True,
        site: str = P4_SITE_PRE_MID,
    ):
        super().__init__()
        if site not in P4_ALLOWED_SITES:
            raise ValueError(f"unsupported P4 insertion site: {site}")
        self.channels = int(channels)
        self.time_embed_dim = int(time_embed_dim)
        self.hidden_dim = int(hidden_dim or channels)
        self.heads = int(heads)
        self.ff_mult = float(ff_mult)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.timestep_adaln = bool(timestep_adaln)
        self.site = str(site)
        self.input_projection = nn.Conv2d(self.channels, self.hidden_dim, kernel_size=1)
        self.output_projection = nn.Conv2d(self.hidden_dim, self.channels, kernel_size=1)
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.attn = P4SelfAttention(
            self.hidden_dim, self.heads,
            use_rope=self.use_rope, rope_base=self.rope_base)
        ffn_hidden = max(1, int(round(self.hidden_dim * self.ff_mult)))
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, self.hidden_dim),
        )
        self.time_modulation = (
            nn.Linear(self.time_embed_dim, self.hidden_dim * 4)
            if self.timestep_adaln else None
        )
        if self.time_modulation is not None:
            # Zero scale/shift means ordinary normalized transformer branch:
            # norm(x) * (1 + 0) + 0.  It does not zero F(x).
            nn.init.zeros_(self.time_modulation.weight)
            nn.init.zeros_(self.time_modulation.bias)
        self.gate = nn.Parameter(torch.zeros(()))
        self.last_branch_rms: float = 0.0
        self.last_output_delta_rms: float = 0.0

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, *, h: int, w: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], h, w)

    @staticmethod
    def _map_to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2)

    @staticmethod
    def _modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return tokens * (1 + scale[:, None, :]) + shift[:, None, :]

    def _norm_with_time(
        self,
        norm: nn.LayerNorm,
        tokens: torch.Tensor,
        temb: torch.Tensor | None,
        index: int,
    ) -> torch.Tensor:
        normalized = norm(tokens)
        if self.time_modulation is None:
            return normalized
        if temb is None:
            raise RuntimeError("P4 timestep AdaLN requires SD timestep embedding")
        mods = self.time_modulation(temb).chunk(4, dim=-1)
        shift, scale = mods[index], mods[index + 1]
        return self._modulate(normalized, shift, scale)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"P4 block expects [B,C,H,W], got {tuple(x.shape)}")
        h, w = int(x.shape[-2]), int(x.shape[-1])
        z0 = self._map_to_tokens(self.input_projection(x))
        attn_in = self._norm_with_time(self.attn_norm, z0, temb, 0)
        delta_attn = self.attn(attn_in, h=h, w=w)
        z1 = z0 + delta_attn
        ffn_in = self._norm_with_time(self.ffn_norm, z1, temb, 2)
        delta_ffn = self.ffn(ffn_in)
        branch_tokens = delta_attn + delta_ffn
        branch = self.output_projection(self._tokens_to_map(branch_tokens, h=h, w=w))
        gate = torch.tanh(self.gate).to(dtype=branch.dtype)
        with torch.no_grad():
            self.last_branch_rms = float(branch.detach().float().square().mean().sqrt().cpu())
            self.last_output_delta_rms = float(
                (gate.detach().float() * branch.detach().float()).square().mean().sqrt().cpu())
        return x + gate * branch


def _extract_temb(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
    temb = kwargs.get("temb")
    if temb is None and len(args) >= 2:
        candidate = args[1]
        if isinstance(candidate, torch.Tensor):
            temb = candidate
    return temb


def _replace_first_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def install_p4_blocks(
    unet: nn.Module,
    *,
    sites: list[str] | tuple[str, ...],
    variant: str,
    hidden_dim: int = 0,
    heads: int = 8,
    ff_mult: float = 2.0,
    rope_base: float = 10000.0,
    timestep_adaln: bool = True,
) -> P4HookBundle:
    if variant not in {"no_pe", "rope"}:
        raise ValueError("P4 variant must be 'no_pe' or 'rope'")
    if not sites:
        raise ValueError("P4 requires at least one insertion site")
    for site in sites:
        if site not in P4_ALLOWED_SITES:
            raise ValueError(f"unsupported P4 insertion site: {site}")
    if not hasattr(unet, "down_blocks") or not hasattr(unet, "mid_block"):
        raise ValueError("P4 installation requires a diffusers-style UNet")
    config = getattr(unet, "config")
    channels = int(getattr(config, "block_out_channels")[-1])
    time_embedding = cast(Any, getattr(unet, "time_embedding"))
    time_embed_dim = int(time_embedding.linear_2.out_features)
    module_dict = nn.ModuleDict()
    handles = []
    use_rope = variant == "rope"
    for site in sites:
        block = P4DeltaBlock(
            channels, time_embed_dim,
            hidden_dim=(hidden_dim or channels), heads=heads, ff_mult=ff_mult,
            use_rope=use_rope, rope_base=rope_base,
            timestep_adaln=timestep_adaln, site=site)
        module_dict[site] = block.to(
            device=next(unet.parameters()).device,
            dtype=next(unet.parameters()).dtype)
    setattr(unet, "p4_blocks", module_dict)
    down_blocks = cast(Any, getattr(unet, "down_blocks"))
    mid_block = cast(Any, getattr(unet, "mid_block"))

    if P4_SITE_PRE_MID in module_dict:
        def pre_mid_hook(module, args, kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = module_dict[P4_SITE_PRE_MID](hidden, _extract_temb(args, kwargs))
            return _replace_first_output(output, hidden)
        handles.append(down_blocks[-1].register_forward_hook(
            pre_mid_hook, with_kwargs=True))

    if P4_SITE_POST_MID in module_dict:
        def post_mid_hook(module, args, kwargs, output):
            return module_dict[P4_SITE_POST_MID](output, _extract_temb(args, kwargs))
        handles.append(mid_block.register_forward_hook(
            post_mid_hook, with_kwargs=True))

    bundle = P4HookBundle(handles)
    object.__setattr__(unet, "_p4_hook_bundle", bundle)
    return bundle


def remove_p4_blocks(unet: nn.Module) -> None:
    bundle = getattr(unet, "_p4_hook_bundle", None)
    if bundle is not None:
        bundle.remove()
    if hasattr(unet, "_p4_hook_bundle"):
        delattr(unet, "_p4_hook_bundle")


def p4_telemetry(unet: nn.Module) -> dict[str, float]:
    blocks = getattr(unet, "p4_blocks", None)
    if not blocks:
        return {}
    metrics: dict[str, float] = {}
    for name, block in blocks.items():
        metrics[f"p4/{name}_gate"] = float(block.gate.detach().float().cpu())
        metrics[f"p4/{name}_tanh_gate"] = float(torch.tanh(block.gate.detach()).float().cpu())
        metrics[f"p4/{name}_branch_rms"] = float(block.last_branch_rms)
        metrics[f"p4/{name}_output_delta_rms"] = float(block.last_output_delta_rms)
        if block.gate.grad is not None:
            metrics[f"p4/{name}_gate_grad"] = float(block.gate.grad.detach().float().cpu())
    return metrics


def p4_state_dict(unet: nn.Module) -> dict[str, torch.Tensor]:
    blocks = getattr(unet, "p4_blocks", None)
    if not blocks:
        return {}
    return {key: value.detach().cpu() for key, value in blocks.state_dict().items()}
