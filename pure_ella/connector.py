"""
Pure ELLA timestep-aware connector architectures.

Supports three connector types:
- ella_tsc: PureELLALongConnector (timestep-aware semantic connector)
- recursive_y: RecursiveYConnector (y-only recursive refinement)
- trm_yz: TRMYZConnector (y/z scratchpad recursive refinement)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_layer_mix_logits(layer_count: int) -> nn.Parameter:
    layer_count = int(layer_count)
    if layer_count < 1:
        raise ValueError("gemma_layer_mix_count must be positive")
    logits = torch.full((layer_count,), -4.0)
    logits[-1] = 0.0
    return nn.Parameter(logits)


def _mix_gemma_layers(gemma_h, layer_mix_logits, connector_name: str):
    layer_count = layer_mix_logits.shape[0]
    if gemma_h.ndim == 3:
        if layer_count != 1:
            raise ValueError(
                f"{connector_name} expects stacked Gemma layers with shape "
                "[B, K, L, D]"
            )
        return gemma_h
    if gemma_h.ndim != 4:
        raise ValueError(
            "Gemma states must have shape [B, L, D] or [B, K, L, D]")
    if gemma_h.shape[1] != layer_count:
        raise ValueError(
            f"{connector_name} expected {layer_count} Gemma layers, "
            f"got {gemma_h.shape[1]}"
        )
    weights = torch.softmax(layer_mix_logits, dim=0).to(
        device=gemma_h.device, dtype=gemma_h.dtype)
    return (gemma_h * weights[None, :, None, None]).sum(dim=1)


class ELLAFeedForward(nn.Module):
    """Simple feed-forward block with LayerNorm, expansion, GELU, dropout, projection."""
    def __init__(self, width: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mult, width),
        )

    def forward(self, x):
        return self.net(x)


class ELLAConnectorBlock(nn.Module):
    """One transformer-style block: cross-attn from q to kv, then self-attn, then FF."""
    def __init__(self, width: int = 768, heads: int = 8, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(width)
        self.kv_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = ELLAFeedForward(width, ff_mult, dropout)

    def forward(self, q, kv, key_padding_mask=None):
        x = q
        y, _ = self.cross(
            self.q_norm(x), self.kv_norm(kv), self.kv_norm(kv),
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        x = x + y
        y, _ = self.self_attn(
            self.self_norm(x), self.self_norm(x), self.self_norm(x),
            need_weights=False,
        )
        x = x + y
        x = x + self.ff(x)
        return x


class TimestepAwareConnectorBlock(nn.Module):
    """TSC block with timestep-conditioned normalization."""

    def __init__(self, width: int = 768, heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cross_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.kv_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(width, width * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * ff_mult, width),
        )
        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, width * 6),
        )

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, q, kv, time_embedding, key_padding_mask=None):
        cross_shift, cross_scale, self_shift, self_scale, ff_shift, ff_scale = (
            self.time_modulation(time_embedding).chunk(6, dim=-1)
        )
        cross_q = self._modulate(
            self.cross_norm(q), cross_shift, cross_scale)
        normalized_kv = self.kv_norm(kv)
        cross_out, _ = self.cross(
            cross_q, normalized_kv, normalized_kv,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        q = q + cross_out
        self_q = self._modulate(
            self.self_norm(q), self_shift, self_scale)
        self_out, _ = self.self_attn(
            self_q, self_q, self_q, need_weights=False)
        q = q + self_out
        q = q + self.ff(self._modulate(
            self.ff_norm(q), ff_shift, ff_scale))
        return q


class PureELLALongConnector(nn.Module):
    """Gemma -> fixed-length, timestep-aware SD1.5 conditioning.

    Long context belongs on the Gemma input side. The connector deliberately
    preserves SD1.5's output token count so untrained tokens cannot perturb the
    U-Net cross-attention softmax.
    """
    def __init__(
        self,
        gemma_dim: int = 640,
        width: int = 768,
        context_tokens: int = 77,
        anchor_tokens: int = 77,
        layers: int = 6,
        heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        time_embed_dim: int = 768,
        extra_gate_init: float = -5.0,
        gemma_layer_mix_count: int = 1,
    ):
        super().__init__()
        if context_tokens != anchor_tokens:
            raise ValueError(
                "ella_tsc preserves the pretrained U-Net token contract; "
                "context_tokens must equal anchor_tokens"
            )
        self.context_tokens = int(context_tokens)
        self.anchor_tokens = int(anchor_tokens)
        self.gemma_layer_mix_count = int(gemma_layer_mix_count)
        self.layer_mix_logits = _make_layer_mix_logits(
            self.gemma_layer_mix_count)
        self.input_proj = nn.Linear(gemma_dim, width)
        self.input_norm = nn.LayerNorm(width)
        self.query_tokens = nn.Parameter(torch.randn(1, context_tokens, width) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, context_tokens, width) * 0.01)
        self.time_mlp = nn.Sequential(
            nn.Linear(320, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, width),
        )
        self.blocks = nn.ModuleList([
            TimestepAwareConnectorBlock(width, heads, ff_mult, dropout)
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, width)
        self.extra_gate_logit = None

    @staticmethod
    def timestep_embedding(timesteps, dim: int = 320, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def _mix_gemma_layers(self, gemma_h):
        return _mix_gemma_layers(
            gemma_h, self.layer_mix_logits, type(self).__name__)

    def forward(self, gemma_h, timesteps, gemma_mask=None, context_tokens=None):
        context_tokens = int(context_tokens or self.context_tokens)
        if context_tokens != self.context_tokens:
            raise ValueError(
                f"ella_tsc has a fixed output length of {self.context_tokens}, "
                f"got {context_tokens}"
            )
        gemma_h = self._mix_gemma_layers(gemma_h)
        kv = self.input_norm(self.input_proj(gemma_h.to(dtype=self.input_proj.weight.dtype)))
        q = self.query_tokens[:, :context_tokens, :] + self.pos_emb[:, :context_tokens, :]
        q = q.expand(gemma_h.shape[0], -1, -1)
        temb = self.time_mlp(
            self.timestep_embedding(timesteps, self.time_mlp[0].in_features).to(device=q.device, dtype=q.dtype)
        )
        key_padding_mask = None if gemma_mask is None else ~gemma_mask.to(device=q.device, dtype=torch.bool)
        x = q
        for block in self.blocks:
            x = block(x, kv, temb, key_padding_mask=key_padding_mask)
        x = self.out(self.final_norm(x))
        return x


class RecursiveYConnector(PureELLALongConnector):
    """Minimal y-only recursive refinement on top of the ELLA/TSC connector.

    Cheap TRM ablation: no scratchpad z. Tests iterative refinement of y.
    """
    def __init__(self, *args, recursive_y_steps: int = 2, recursive_y_gate_init: float = -2.0,
                 block_width: int = 768, block_heads: int = 8, block_ff_mult: int = 4,
                 block_dropout: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.recursive_y_steps = int(recursive_y_steps)
        self.recursive_y_gate_logit = nn.Parameter(torch.tensor(float(recursive_y_gate_init)))
        self.recursive_y_block = ELLAConnectorBlock(
            width=block_width, heads=block_heads, ff_mult=block_ff_mult, dropout=block_dropout,
        )
        self.recursive_y_norm = nn.LayerNorm(block_width)

    def _apply_long_extra_gate(self, x, context_tokens):
        if self.extra_gate_logit is not None and int(context_tokens) > self.anchor_tokens:
            base = x[:, :self.anchor_tokens, :]
            extra = x[:, self.anchor_tokens:, :] * torch.sigmoid(self.extra_gate_logit).to(dtype=x.dtype)
            return torch.cat([base, extra], dim=1)
        return x

    def forward(self, gemma_h, timesteps, gemma_mask=None, context_tokens=None):
        context_tokens = int(context_tokens or self.context_tokens)
        y = super().forward(gemma_h, timesteps, gemma_mask=gemma_mask, context_tokens=context_tokens)
        if self.recursive_y_steps <= 0:
            return y
        gemma_h = self._mix_gemma_layers(gemma_h)
        kv = self.input_norm(self.input_proj(gemma_h.to(dtype=self.input_proj.weight.dtype)))
        key_padding_mask = None if gemma_mask is None else ~gemma_mask.to(device=y.device, dtype=torch.bool)
        gate = torch.sigmoid(self.recursive_y_gate_logit).to(dtype=y.dtype)
        for _ in range(self.recursive_y_steps):
            y_next = self.recursive_y_block(y, kv, key_padding_mask=key_padding_mask)
            y = y + gate * (y_next - y)
            y = self.recursive_y_norm(y)
            y = self._apply_long_extra_gate(y, context_tokens)
        return y


class TRMYZConnector(nn.Module):
    """TRM-style y/z recursive connector.

    y = current SD conditioning tokens.
    z = latent semantic scratchpad tokens.
    A single shared transition block updates z repeatedly, then y.
    """
    def __init__(
        self,
        gemma_dim: int = 640,
        width: int = 768,
        context_tokens: int = 128,
        anchor_tokens: int = 77,
        layers: int = 4,
        heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        time_embed_dim: int = 768,
        extra_gate_init: float = -5.0,
        trm_outer_steps: int = 3,
        trm_inner_steps: int = 2,
        trm_scratch_tokens: int = 32,
        trm_y_gate_init: float = -2.0,
        trm_z_gate_init: float = -1.0,
        gemma_layer_mix_count: int = 1,
    ):
        super().__init__()
        assert context_tokens >= anchor_tokens
        self.context_tokens = int(context_tokens)
        self.anchor_tokens = int(anchor_tokens)
        self.gemma_layer_mix_count = int(gemma_layer_mix_count)
        self.layer_mix_logits = _make_layer_mix_logits(
            self.gemma_layer_mix_count)
        self.trm_outer_steps = int(trm_outer_steps)
        self.trm_inner_steps = int(trm_inner_steps)
        self.trm_scratch_tokens = int(trm_scratch_tokens)
        self.input_proj = nn.Linear(gemma_dim, width)
        self.input_norm = nn.LayerNorm(width)
        self.query_tokens = nn.Parameter(torch.randn(1, context_tokens, width) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, context_tokens, width) * 0.01)
        self.z_tokens = nn.Parameter(torch.randn(1, trm_scratch_tokens, width) * 0.02)
        self.z_pos_emb = nn.Parameter(torch.randn(1, trm_scratch_tokens, width) * 0.01)
        self.role_y = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.role_z = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.step_emb = nn.Embedding(128, width)
        self.time_mlp = nn.Sequential(
            nn.Linear(320, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, width),
        )
        self.transition = ELLAConnectorBlock(width, heads, ff_mult, dropout)
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, width)
        self.y_gate_logit = nn.Parameter(torch.tensor(float(trm_y_gate_init)))
        self.z_gate_logit = nn.Parameter(torch.tensor(float(trm_z_gate_init)))
        if context_tokens > anchor_tokens:
            self.extra_gate_logit = nn.Parameter(torch.tensor(float(extra_gate_init)))
        else:
            self.extra_gate_logit = None

    @staticmethod
    def timestep_embedding(timesteps, dim: int = 320, max_period: int = 10000):
        return PureELLALongConnector.timestep_embedding(timesteps, dim=dim, max_period=max_period)

    def _masked_mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1, keepdim=True)
        m = mask.to(device=x.device, dtype=x.dtype)[:, :, None]
        return (x * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _source_mask(self, gemma_mask, extra_len, device):
        if gemma_mask is None:
            return None
        src_pad = ~gemma_mask.to(device=device, dtype=torch.bool)
        extra_pad = torch.zeros(src_pad.shape[0], int(extra_len), device=device, dtype=torch.bool)
        return torch.cat([src_pad, extra_pad], dim=1)

    def _apply_extra_gate(self, y, context_tokens):
        if self.extra_gate_logit is not None and int(context_tokens) > self.anchor_tokens:
            base = y[:, :self.anchor_tokens, :]
            extra = y[:, self.anchor_tokens:, :] * torch.sigmoid(self.extra_gate_logit).to(dtype=y.dtype)
            return torch.cat([base, extra], dim=1)
        return y

    def _mix_gemma_layers(self, gemma_h):
        return _mix_gemma_layers(
            gemma_h, self.layer_mix_logits, type(self).__name__)

    def forward(self, gemma_h, timesteps, gemma_mask=None, context_tokens=None):
        context_tokens = int(context_tokens or self.context_tokens)
        if context_tokens > self.context_tokens:
            raise ValueError(f"Requested context_tokens={context_tokens}, max is {self.context_tokens}")
        gemma_h = self._mix_gemma_layers(gemma_h)
        x = self.input_norm(self.input_proj(gemma_h.to(dtype=self.input_proj.weight.dtype)))
        pooled = self._masked_mean(x, gemma_mask)
        temb = self.time_mlp(
            self.timestep_embedding(timesteps, self.time_mlp[0].in_features).to(device=x.device, dtype=x.dtype)
        )[:, None, :]
        y = self.query_tokens[:, :context_tokens, :] + self.pos_emb[:, :context_tokens, :]
        y = y.expand(x.shape[0], -1, -1) + pooled + temb
        z = self.z_tokens + self.z_pos_emb
        z = z.expand(x.shape[0], -1, -1) + pooled + temb
        y_gate = torch.sigmoid(self.y_gate_logit).to(dtype=x.dtype)
        z_gate = torch.sigmoid(self.z_gate_logit).to(dtype=x.dtype)
        step_id = 0
        for outer in range(self.trm_outer_steps):
            for inner in range(self.trm_inner_steps):
                step = self.step_emb(
                    torch.tensor([min(step_id, self.step_emb.num_embeddings - 1)], device=x.device)
                )[:, None, :].to(dtype=x.dtype)
                src = torch.cat([x, y], dim=1)
                src_mask = self._source_mask(gemma_mask, y.shape[1], x.device)
                z_next = self.transition(z + self.role_z + step, src, key_padding_mask=src_mask)
                z = z + z_gate * (z_next - z)
                step_id += 1
            step = self.step_emb(
                torch.tensor([min(step_id, self.step_emb.num_embeddings - 1)], device=x.device)
            )[:, None, :].to(dtype=x.dtype)
            src = torch.cat([x, z], dim=1)
            src_mask = self._source_mask(gemma_mask, z.shape[1], x.device)
            y_next = self.transition(y + self.role_y + step, src, key_padding_mask=src_mask)
            y = y + y_gate * (y_next - y)
            step_id += 1
        y = self.out(self.out_norm(y))
        y = self._apply_extra_gate(y, context_tokens)
        return y


# Connector factory
_BASE_CONNECTOR_KEYS = {
    "gemma_dim", "width", "context_tokens", "anchor_tokens",
    "layers", "heads", "ff_mult", "dropout", "time_embed_dim", "extra_gate_init",
    "gemma_layer_mix_count",
}


def build_connector(connector_type: str, **cfg) -> nn.Module:
    """Build a connector from type and config dict."""
    connector_type = str(connector_type)
    base_cfg = {k: v for k, v in cfg.items() if k in _BASE_CONNECTOR_KEYS}
    if connector_type == "ella_tsc":
        return PureELLALongConnector(**base_cfg)
    if connector_type == "recursive_y":
        return RecursiveYConnector(
            **base_cfg,
            recursive_y_steps=cfg.get("recursive_y_steps", 2),
            recursive_y_gate_init=cfg.get("recursive_y_gate_init", -2.0),
            block_width=cfg.get("width", 768),
            block_heads=cfg.get("heads", 8),
            block_ff_mult=cfg.get("ff_mult", 4),
            block_dropout=cfg.get("dropout", 0.0),
        )
    if connector_type == "trm_yz":
        return TRMYZConnector(
            **base_cfg,
            trm_outer_steps=cfg.get("trm_outer_steps", 3),
            trm_inner_steps=cfg.get("trm_inner_steps", 2),
            trm_scratch_tokens=cfg.get("trm_scratch_tokens", 32),
            trm_y_gate_init=cfg.get("trm_y_gate_init", -2.0),
            trm_z_gate_init=cfg.get("trm_z_gate_init", -1.0),
        )
    raise ValueError(f"Unknown CONNECTOR_TYPE={connector_type}")
