"""
Diagnostics, validation, and generation helpers for the pure-ELLA training pipeline.

Includes:
- CLIP geometry loss for pretrain
- Prompt sensitivity
- Teacher/student delta alignment
- Extra-token ablation metrics
- Suffix counterfactual metrics
- Long-context numeric suite
- Image quality metrics (FID, KID, collapse stats)
- Fixed overfit loss
- Generation helpers (ELLA, CLIP teacher)
- Validation grid saving
"""
from __future__ import annotations
import gc
import time
from typing import List, Optional, Dict, Any, Union
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Final summaries — global dict for unhidden printing
# ---------------------------------------------------------------------------
FINAL_SUMMARIES: Dict[str, dict] = {}


def _to_float_maybe(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    try:
        if hasattr(v, "detach"):
            return float(v.detach().float().cpu().item())
    except Exception:
        pass
    return None


def safe_wandb_log(payload: dict, summary_prefix: Optional[str] = None, wandb=None) -> dict:
    """Log to wandb safely, converting tensors to floats."""
    if wandb is None:
        return {}
    numeric = {}
    for k, v in payload.items():
        fv = _to_float_maybe(v)
        if fv is not None and np.isfinite(fv):
            numeric[k] = fv
    if not numeric:
        return {}
    try:
        wandb.log(numeric, commit=True)
    except Exception as e:
        print(f"wandb.log failed: {type(e).__name__}: {e}")
    if summary_prefix is not None:
        try:
            summary = getattr(wandb, "summary", None)
            if summary is None and getattr(wandb, "run", None) is not None:
                summary = wandb.run.summary
            if summary is not None:
                for k, v in numeric.items():
                    summary[f"{summary_prefix}/{k.split('/')[-1]}"] = v
        except Exception as e:
            print(f"wandb summary update failed: {type(e).__name__}: {e}")
    return numeric


def _log_metrics(prefix: str, metrics: dict, summary_prefix: str = None, wandb=None) -> dict:
    payload = {}
    for k, v in metrics.items():
        fv = _to_float_maybe(v)
        if fv is not None:
            payload[f"{prefix}/{k}"] = fv
    return safe_wandb_log(payload, summary_prefix=summary_prefix, wandb=wandb)


def remember_final_summary(name: str, summary: dict, wandb_prefix: str = None, wandb=None):
    clean = {}
    for k, v in summary.items():
        fv = _to_float_maybe(v)
        clean[k] = fv if fv is not None else v
    FINAL_SUMMARIES[name] = clean
    if wandb_prefix and wandb:
        safe_wandb_log(
            {f"{wandb_prefix}/{k}": v for k, v in clean.items()},
            summary_prefix=wandb_prefix, wandb=wandb,
        )
    return clean


def print_final_summary(name: str, summary: dict = None):
    summary = FINAL_SUMMARIES.get(name, summary or {})
    print(f"\n===== UNHIDDEN FINAL SUMMARY: {name} =====")
    if not summary:
        print("No summary recorded.")
        return
    for k in sorted(summary):
        v = summary[k]
        if isinstance(v, float):
            print(f"{k}: {v:.6g}")
        else:
            print(f"{k}: {v}")
    print(f"===== END SUMMARY: {name} =====\n")


# ---------------------------------------------------------------------------
# CLIP Geometry Loss
# ---------------------------------------------------------------------------
class ClipGeometryLoss(nn.Module):
    """Multi-component loss aligning predicted hidden states to CLIP teacher."""
    def __init__(self, w_mse=1.0, w_cos=0.5, w_norm=0.25, w_ctr=0.2, temp=0.07):
        super().__init__()
        self.w_mse = w_mse
        self.w_cos = w_cos
        self.w_norm = w_norm
        self.w_ctr = w_ctr
        self.temp = temp

    @staticmethod
    def _mask(mask, x):
        return mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1)

    @staticmethod
    def _pooled(x, mask):
        m = ClipGeometryLoss._mask(mask, x).to(dtype=x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def forward(self, pred, target, mask):
        pred = pred[:, :target.shape[1], :].float()
        target = target.float()
        m = self._mask(mask, pred)
        pred_ln = F.layer_norm(pred, pred.shape[-1:])
        target_ln = F.layer_norm(target, target.shape[-1:])
        mse = ((pred_ln - target_ln).pow(2) * m).sum() / m.sum().clamp_min(1.0) / pred.shape[-1]
        cos = 1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)
        cos = (cos * m.squeeze(-1).float()).sum() / m.squeeze(-1).float().sum().clamp_min(1.0)
        pred_norm = pred.norm(dim=-1).clamp_min(1e-6)
        target_norm = target.norm(dim=-1).clamp_min(1e-6)
        norm = (torch.log(pred_norm / target_norm).abs() * m.squeeze(-1).float()).sum() / m.squeeze(-1).float().sum().clamp_min(1.0)
        pp = F.normalize(self._pooled(pred, mask), dim=-1)
        tt = F.normalize(self._pooled(target, mask), dim=-1)
        logits = pp @ tt.t() / self.temp
        labels = torch.arange(pred.shape[0], device=pred.device)
        ctr = F.cross_entropy(logits, labels) if pred.shape[0] > 1 else pred.new_tensor(0.0)
        pooled_cos = (pp * tt).sum(dim=-1).mean()
        total = self.w_mse * mse + self.w_cos * cos + self.w_norm * norm + self.w_ctr * ctr
        return {
            "total": total,
            "mse": mse,
            "cos": cos,
            "norm": norm,
            "ctr": ctr,
            "pooled_cos": pooled_cos,
            "norm_ratio": pred_norm.mean() / target_norm.mean().clamp_min(1e-6),
        }


# ---------------------------------------------------------------------------
# Simple relative difference helper
# ---------------------------------------------------------------------------
def _rel_diff(a, b, eps: float = 1e-8) -> float:
    a = a.float(); b = b.float()
    return float((a - b).pow(2).mean().sqrt().item() / (b.pow(2).mean().sqrt().item() + eps))


def _maybe_float(x):
    return None if x is None else float(x)


# ---------------------------------------------------------------------------
# Extra-token utilities
# ---------------------------------------------------------------------------
def zero_extra_tokens(ctx, anchor_tokens: int):
    if ctx.shape[1] <= anchor_tokens:
        return ctx
    out = ctx.clone()
    out[:, anchor_tokens:, :] = 0
    return out


def summarize_context_tokens(ctx, anchor_tokens: int, extra_gate=None) -> dict:
    ctx = ctx.float()
    base = ctx[:, :anchor_tokens, :]
    extra = ctx[:, anchor_tokens:, :]
    base_rms = base.pow(2).mean().sqrt().item()
    out = {"base_rms": base_rms, "extra_gate": _maybe_float(extra_gate)}
    if extra.numel() == 0:
        out.update({"extra_rms": 0.0, "extra_abs_mean": 0.0, "extra_to_base_ratio": 0.0})
    else:
        extra_rms = extra.pow(2).mean().sqrt().item()
        out.update({
            "extra_rms": extra_rms,
            "extra_abs_mean": extra.abs().mean().item(),
            "extra_to_base_ratio": extra_rms / max(base_rms, 1e-8),
        })
    return out


def connector_extra_grad_stats(model, anchor_tokens: int, state):
    if getattr(model, "extra_gate_logit", None) is None:
        return {"extra_gate_grad_norm": None, "extra_query_grad_norm": None, "extra_pos_grad_norm": None}
    stats = {}
    gate_grad = model.extra_gate_logit.grad
    stats["extra_gate_grad_norm"] = float(gate_grad.detach().float().norm().item()) if gate_grad is not None else None
    q_grad = getattr(model, "query_tokens", None)
    p_grad = getattr(model, "pos_emb", None)
    if q_grad is not None and q_grad.grad is not None and q_grad.shape[1] > anchor_tokens:
        stats["extra_query_grad_norm"] = float(q_grad.grad[:, anchor_tokens:, :].detach().float().norm().item())
    else:
        stats["extra_query_grad_norm"] = None
    if p_grad is not None and p_grad.grad is not None and p_grad.shape[1] > anchor_tokens:
        stats["extra_pos_grad_norm"] = float(p_grad.grad[:, anchor_tokens:, :].detach().float().norm().item())
    else:
        stats["extra_pos_grad_norm"] = None
    return stats


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_ella(
    prompt: str,
    state,
    steps: int = 30,
    guidance: float = 5.5,
    seed: int = 777,
    context_tokens: int = None,
) -> Image.Image:
    """Generate an image from the ELLA connector + frozen UNet."""
    connector = state.connector
    unet = state.unet
    vae = state.vae
    scheduler = state.scheduler
    device = state.device
    unet_dtype = state.unet_dtype
    encode_gemma = state.encode_gemma
    tokenizer = state.gemma_tokenizer
    gemma_model = state.gemma_model
    ctx = context_tokens or state.cfg.context_tokens

    gen = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps.to(device)

    gh, gm = encode_gemma([prompt])
    ugh, ugm = encode_gemma([""])

    for t in tqdm(timesteps, desc=f"gen:{prompt[:40]}"):
        t_batch = t.expand(2)
        latent_input = scheduler.scale_model_input(torch.cat([latent, latent], dim=0), t)
        h_pair = torch.cat([gh, ugh], dim=0)
        m_pair = torch.cat([gm, ugm], dim=0)
        context = connector(h_pair.to(dtype=unet_dtype), t_batch.to(device), m_pair, context_tokens=ctx)
        noise_pred = unet(latent_input, t_batch, encoder_hidden_states=context).sample
        noise_cond, noise_uncond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
        latent = scheduler.step(noise_pred, t, latent).prev_sample

    latent = latent / vae.config.scaling_factor
    img = vae.decode(latent.to(dtype=vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


@torch.no_grad()
def generate_clip_teacher(prompt: str, state, steps: int = 30, guidance: float = 5.5, seed: int = 777) -> Image.Image:
    """Generate an image using the CLIP teacher (for diagnostics)."""
    unet = state.unet
    vae = state.vae
    scheduler = state.scheduler
    device = state.device
    unet_dtype = state.unet_dtype
    encode_clip = state.encode_clip

    gen = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps.to(device)

    ch, cm = encode_clip([prompt])
    uch, ucm = encode_clip([""])

    for t in tqdm(timesteps, desc=f"clip:{prompt[:40]}"):
        t_batch = t.expand(2)
        latent_input = scheduler.scale_model_input(torch.cat([latent, latent], dim=0), t)
        h_pair = torch.cat([ch, uch], dim=0)
        m_pair = torch.cat([cm, ucm], dim=0)
        noise_pred = unet(latent_input, t_batch, encoder_hidden_states=h_pair.to(dtype=unet_dtype), encoder_attention_mask=m_pair).sample
        noise_cond, noise_uncond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
        latent = scheduler.step(noise_pred, t, latent).prev_sample

    latent = latent / vae.config.scaling_factor
    img = vae.decode(latent.to(dtype=vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


# ---------------------------------------------------------------------------
# Prompt sensitivity
# ---------------------------------------------------------------------------
@torch.no_grad()
def connector_prompt_sensitivity(prompts: List[str], state, timestep: int = 500, label: str = "connector", wandb=None) -> float:
    connector = state.connector
    unet = state.unet
    device = state.device
    unet_dtype = state.unet_dtype
    encode_gemma = state.encode_gemma
    cfg = state.cfg

    connector.eval(); unet.eval()
    gen = torch.Generator(device=device).manual_seed(123)
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    t = torch.tensor([timestep], device=device).long()
    preds = []
    for ptxt in prompts:
        gh, gm = encode_gemma([ptxt])
        ctx = connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=cfg.context_tokens)
        pred = unet(latent, t, encoder_hidden_states=ctx).sample.float()
        preds.append(pred)
    base = preds[0].pow(2).mean().sqrt().item() + 1e-8
    vals = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            diff = (preds[i] - preds[j]).pow(2).mean().sqrt().item() / base
            vals.append(diff)
            print(f"[{label}] {i} vs {j}: rel_diff={diff:.6f}")
    mean_val = float(np.mean(vals)) if vals else 0.0
    safe_wandb_log({f"diagnostics/{label}_mean_relative_diff": mean_val}, summary_prefix=f"diagnostics/{label}", wandb=wandb)
    return mean_val


# ---------------------------------------------------------------------------
# Teacher-student delta alignment
# ---------------------------------------------------------------------------
@torch.no_grad()
def teacher_student_delta_alignment(prompt: str, state, timestep: int = 500, label: str = "delta", wandb=None) -> dict:
    if state.clip_model is None:
        print(f"[{label}] CLIP unavailable; skipped")
        return {}
    connector = state.connector
    unet = state.unet
    scheduler = state.scheduler
    device = state.device
    unet_dtype = state.unet_dtype
    encode_gemma = state.encode_gemma
    encode_clip = state.encode_clip
    cfg = state.cfg

    connector.eval(); unet.eval()
    gen = torch.Generator(device=device).manual_seed(777)
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    noise = torch.randn_like(latent)
    t = torch.tensor([timestep], device=device).long()
    noisy = scheduler.add_noise(latent, noise, t)
    noisy_pair = torch.cat([noisy, noisy], dim=0)
    t_pair = torch.cat([t, t], dim=0)

    ch, cm = encode_clip([prompt])
    uch, ucm = encode_clip([""])
    clip_h = torch.cat([ch, uch], dim=0)
    clip_m = torch.cat([cm, ucm], dim=0)
    teacher = unet(noisy_pair, t_pair, encoder_hidden_states=clip_h.to(dtype=unet_dtype), encoder_attention_mask=clip_m).sample.float()
    teacher_cond, teacher_uncond = teacher.chunk(2)

    gh, gm = encode_gemma([prompt])
    ugh, ugm = encode_gemma([""])
    gemma_h = torch.cat([gh, ugh], dim=0)
    gemma_m = torch.cat([gm, ugm], dim=0)
    ctx = connector(gemma_h.to(dtype=unet_dtype), t_pair, gemma_m, context_tokens=cfg.context_tokens)
    student = unet(noisy_pair, t_pair, encoder_hidden_states=ctx).sample.float()
    student_cond, student_uncond = student.chunk(2)

    td = (teacher_cond - teacher_uncond).flatten()
    sd = (student_cond - student_uncond).flatten()
    cos = float(F.cosine_similarity(sd[None], td[None]).item())
    ratio = float(sd.norm().item() / max(td.norm().item(), 1e-8))
    print(f"[{label}] delta_cos={cos:.6f} norm_ratio={ratio:.6f}")
    safe_wandb_log(
        {f"diagnostics/{label}_delta_cos": cos, f"diagnostics/{label}_norm_ratio": ratio},
        summary_prefix=f"diagnostics/{label}", wandb=wandb,
    )
    return {"delta_cos": cos, "norm_ratio": ratio}


# ---------------------------------------------------------------------------
# Fixed overfit loss
# ---------------------------------------------------------------------------
@torch.no_grad()
def fixed_overfit_loss(state, label: str = "fixed_overfit", timestep: int = 500, seed: int = 777, wandb=None) -> float:
    """Compute deterministic noise-prediction MSE on a fixed eval batch."""
    if not state.overfit_eval_batch:
        return float("nan")
    connector = state.connector
    unet = state.unet
    vae = state.vae
    scheduler = state.scheduler
    device = state.device
    unet_dtype = state.unet_dtype
    encode_gemma = state.encode_gemma
    cfg = state.cfg

    batch = state.overfit_eval_batch
    captions = batch["caption"]
    img = batch["image"].to(device=device, dtype=unet_dtype)
    gen = torch.Generator(device=device).manual_seed(seed)
    latent = vae.encode(img).latent_dist.sample(generator=gen) * vae.config.scaling_factor
    noise = torch.randn_like(latent)
    t_val = min(timestep, scheduler.config.num_train_timesteps - 1)
    t = torch.full((latent.shape[0],), t_val, device=device).long()
    noisy = scheduler.add_noise(latent, noise, t)
    gh, gm = encode_gemma(captions)
    ctx = connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=cfg.context_tokens)
    pred = unet(noisy, t, encoder_hidden_states=ctx).sample
    loss = F.mse_loss(pred.float(), noise.float()).item()
    print(f"[{label}] fixed overfit MSE @t={t_val}: {loss:.6f}")
    safe_wandb_log({f"validation/{label}_mse": loss}, wandb=wandb)
    return loss


# ---------------------------------------------------------------------------
# Validation grid saving
# ---------------------------------------------------------------------------
def _pil_grid(images: List[Image.Image], labels: List[str], cols: int = 3) -> Image.Image:
    """Create a grid image from a list of PIL images with labels."""
    from PIL import ImageDraw, ImageFont
    n = len(images)
    rows = math.ceil(n / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (w * cols, h * rows + 24 * rows), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * (h + 24) + 24))
        draw.text((c * w + 4, r * (h + 24) + 4), label[:80], fill=(0, 0, 0))
    return grid


def save_validation_grid(images: List[Image.Image], labels: List[str], path: str, title: str = ""):
    """Save a grid of generated images to disk."""
    grid = _pil_grid(images, labels)
    grid.save(path)
    print(f"Validation grid saved: {path}")


# ---------------------------------------------------------------------------
# Extra-token ablation metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def extra_token_ablation_metrics(prompt: str, state, timestep: int = None, seed: int = None,
                                  context_tokens: int = None, label: str = "extra_tokens",
                                  negative_prompt: str = "", wandb=None) -> dict:
    cfg = state.cfg
    timestep = cfg.extra_token_diagnostic_timestep if timestep is None else int(timestep)
    seed = cfg.extra_token_diagnostic_seed if seed is None else int(seed)
    context_tokens = cfg.context_tokens if context_tokens is None else int(context_tokens)
    if context_tokens <= cfg.clip_anchor_tokens:
        print(f"[{label}] context_tokens={context_tokens}; extra-token skipped at 77-token")
        return {"skipped": 1.0}

    connector = state.connector
    unet = state.unet
    scheduler = state.scheduler
    device = state.device
    unet_dtype = state.unet_dtype
    encode_gemma = state.encode_gemma

    connector.eval(); unet.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    noise = torch.randn_like(latent)
    t = torch.tensor([int(timestep)], device=device).long()
    noisy = scheduler.add_noise(latent, noise, t)

    gh, gm = encode_gemma([prompt])
    ugh, ugm = encode_gemma([negative_prompt or ""])
    ctx77 = connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=cfg.clip_anchor_tokens)
    ctx_full = connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=context_tokens)
    ctx_zero = zero_extra_tokens(ctx_full, cfg.clip_anchor_tokens)
    uctx77 = connector(ugh.to(dtype=unet_dtype), t, ugm, context_tokens=cfg.clip_anchor_tokens)
    uctx_full = connector(ugh.to(dtype=unet_dtype), t, ugm, context_tokens=context_tokens)
    uctx_zero = zero_extra_tokens(uctx_full, cfg.clip_anchor_tokens)

    extra_gate = torch.sigmoid(connector.extra_gate_logit).item() if connector.extra_gate_logit is not None else None
    metrics = summarize_context_tokens(ctx_full, cfg.clip_anchor_tokens, extra_gate=extra_gate)
    metrics["prefix_drift_rel"] = _rel_diff(ctx_full[:, :cfg.clip_anchor_tokens, :], ctx77)

    pred77 = unet(noisy, t, encoder_hidden_states=ctx77).sample.float()
    predfull = unet(noisy, t, encoder_hidden_states=ctx_full).sample.float()
    predzero = unet(noisy, t, encoder_hidden_states=ctx_zero).sample.float()
    upred77 = unet(noisy, t, encoder_hidden_states=uctx77).sample.float()
    upredfull = unet(noisy, t, encoder_hidden_states=uctx_full).sample.float()
    upredzero = unet(noisy, t, encoder_hidden_states=uctx_zero).sample.float()

    delta77 = pred77 - upred77
    deltafull = predfull - upredfull
    deltazero = predzero - upredzero

    metrics.update({
        "rel_diff_77_vs_full_cond": _rel_diff(predfull, pred77),
        "rel_diff_full_vs_zeroextra_cond": _rel_diff(predfull, predzero),
        "rel_diff_77_vs_full_delta": _rel_diff(deltafull, delta77),
        "rel_diff_full_vs_zeroextra_delta": _rel_diff(deltafull, deltazero),
    })
    print(f"[{label}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items() if isinstance(v, (int, float))))
    _log_metrics(f"extra_tokens/{label}", metrics, wandb=wandb)
    return metrics


# ---------------------------------------------------------------------------
# Image quality metrics (FID, KID, collapse)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _pil_to_uint8_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.no_grad()
def _image_collapse_stats(uint8_images: torch.Tensor) -> dict:
    x = uint8_images.float() / 255.0
    gray = x.mean(dim=1, keepdim=True)
    gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    edge = (gx.abs().mean() + gy.abs().mean()).item()
    flat = x.flatten(2)
    pixel_std = float(flat.std(dim=2).mean().item())
    channel_std = float(x.mean(dim=(2, 3)).std(dim=1).mean().item())
    hist_vals = []
    for im in gray:
        h = torch.histc(im.flatten(), bins=64, min=0.0, max=1.0)
        p = h / h.sum().clamp_min(1)
        hist_vals.append(float(-(p[p > 0] * p[p > 0].log2()).sum().item()))
    return {
        "quality/pixel_std": pixel_std,
        "quality/channel_mean_std": channel_std,
        "quality/edge_density": float(edge),
        "quality/luma_entropy": float(np.mean(hist_vals)),
    }


# ---------------------------------------------------------------------------
# Complex / long-context prompt grids (from colab notebook)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_case_image(case: dict, state, context_tokens: int = None):
    """Generate an image from a complex case dict using ELLA connector."""
    return generate_ella(
        case["prompt"],
        state,
        steps=case.get("steps", state.cfg.val_steps),
        guidance=case.get("guidance", state.cfg.val_guidance),
        seed=case.get("seed", state.cfg.val_seed),
        context_tokens=context_tokens,
    )


@torch.no_grad()
def save_complex_case_grid(
    cases: list, path: str, title: str, state,
    context_tokens: int = None,
):
    """Generate a grid from complex prompt cases."""
    ctx = context_tokens or state.cfg.context_tokens
    imgs, labels = [], []
    for case in cases:
        print(f"Generating complex case: {case['name']}")
        imgs.append(generate_case_image(case, state, context_tokens=ctx))
        labels.append(f"L{ctx}: {case['name']}")
    save_validation_grid(imgs, labels, path, title)


@torch.no_grad()
def save_suffix_counterfactual_grids(
    cases: list, path_prefix: str, title_prefix: str,
    state, context_tokens: int = None,
):
    """Generate suffix counterfactual grids (short vs long prompt comparisons)."""
    cfg = state.cfg
    ctx = context_tokens or cfg.context_tokens
    if ctx <= cfg.clip_anchor_tokens:
        print("Suffix counterfactual grids skipped at 77-token stage")
        return []
    prefix = cfg.suffix_counterfactual_prefix
    out_paths = []
    for case in cases:
        imgs, labels = [], []
        print(f"Generating suffix counterfactual grid: {case['name']}")
        # Short prompt at anchor tokens
        imgs.append(generate_ella(
            case["short_prompt"], state,
            steps=case.get("steps", cfg.val_steps),
            guidance=case.get("guidance", cfg.val_guidance),
            seed=case.get("seed", cfg.val_seed),
            context_tokens=cfg.clip_anchor_tokens,
        ))
        labels.append(f"short@77 {case['name']}")
        # Full prompt A at context_tokens
        imgs.append(generate_ella(
            prefix + " " + case["prompt_suffix_a"], state,
            steps=case.get("steps", cfg.val_steps),
            guidance=case.get("guidance", cfg.val_guidance),
            seed=case.get("seed", cfg.val_seed),
            context_tokens=ctx,
        ))
        labels.append(f"fullA@L{ctx} {case['name']}")
        # Full prompt B at context_tokens
        imgs.append(generate_ella(
            prefix + " " + case["prompt_suffix_b"], state,
            steps=case.get("steps", cfg.val_steps),
            guidance=case.get("guidance", cfg.val_guidance),
            seed=case.get("seed", cfg.val_seed),
            context_tokens=ctx,
        ))
        labels.append(f"fullB@L{ctx} {case['name']}")
        out_path = f"{path_prefix}_{case['name']}.png"
        save_validation_grid(imgs, labels, out_path, f"{title_prefix}: {case['name']}")
        out_paths.append(out_path)
    return out_paths
