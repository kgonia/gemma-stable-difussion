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

from pure_ella.camera import camera_conditioned_unet, make_camera_condition


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

    def forward(self, pred, target, mask, pool_mask=None):
        """Align token geometry while optionally pooling over another mask.

        Phase 0 supervises all 77 CLIP states because SD1.5 consumes all of
        them, but semantic pooling should still exclude padding positions.
        Other callers retain the historical single-mask behavior.
        """
        pred = pred[:, :target.shape[1], :].float()
        target = target.float()
        pool_mask = mask if pool_mask is None else pool_mask
        m = self._mask(mask, pred)
        token_weights = m.squeeze(-1).float()
        token_count = token_weights.sum().clamp_min(1.0)
        pred_ln = F.layer_norm(pred, pred.shape[-1:])
        target_ln = F.layer_norm(target, target.shape[-1:])
        mse = ((pred_ln - target_ln).pow(2) * m).sum() / m.sum().clamp_min(1.0) / pred.shape[-1]
        cos = 1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)
        cos = (cos * token_weights).sum() / token_count
        pred_norm = pred.norm(dim=-1).clamp_min(1e-6)
        target_norm = target.norm(dim=-1).clamp_min(1e-6)
        norm = (torch.log(pred_norm / target_norm).abs() * token_weights).sum() / token_count
        pp = F.normalize(self._pooled(pred, pool_mask), dim=-1)
        tt = F.normalize(self._pooled(target, pool_mask), dim=-1)
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
            "norm_ratio": (
                (pred_norm * token_weights).sum() / token_count
            ) / (
                (target_norm * token_weights).sum() / token_count
            ).clamp_min(1e-6),
        }


# ---------------------------------------------------------------------------
# Simple relative difference helper
# ---------------------------------------------------------------------------
def _rel_diff(a, b, eps: float = 1e-8) -> float:
    a = a.float(); b = b.float()
    return float((a - b).pow(2).mean().sqrt().item() / (b.pow(2).mean().sqrt().item() + eps))


def guided_prediction(
    conditional: torch.Tensor, unconditional: torch.Tensor, guidance: float,
) -> torch.Tensor:
    """Return the classifier-free guided prediction."""
    return unconditional + float(guidance) * (conditional - unconditional)


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
    negative_prompt: str = "",
    height: int = 512,
    width: int = 512,
    context_tokens: int = None,
    camera_condition: torch.Tensor = None,
) -> Image.Image:
    """Generate an image from the ELLA connector + frozen UNet."""
    connector = state.connector
    unet = state.unet
    vae = state.vae
    scheduler = state.inf_scheduler or state.scheduler
    device = state.device
    unet_dtype = state.unet_dtype
    encode_gemma = state.encode_gemma
    tokenizer = state.gemma_tokenizer
    gemma_model = state.gemma_model
    ctx = context_tokens or state.cfg.context_tokens

    gen = torch.Generator(device=device).manual_seed(seed)
    if height % 8 or width % 8:
        raise ValueError("Generation height and width must be divisible by 8")
    latent = torch.randn(
        1, 4, height // 8, width // 8,
        generator=gen, device=device, dtype=unet_dtype,
    )
    scheduler.set_timesteps(steps, device=device)
    latent = latent * scheduler.init_noise_sigma
    timesteps = scheduler.timesteps

    residual = state.cfg.connector_type == "clip_gemma_residual_tsc"
    if residual:
        clip_cond, _ = state.encode_clip([prompt])
        clip_uncond, _ = state.encode_clip([negative_prompt])
        cond_long = len(state.clip_tokenizer([prompt], truncation=False)["input_ids"][0]) > state.cfg.clip_anchor_tokens
        uncond_long = len(state.clip_tokenizer([negative_prompt], truncation=False)["input_ids"][0]) > state.cfg.clip_anchor_tokens
        if cond_long:
            gh, gm = encode_gemma([prompt])
        if uncond_long:
            ugh, ugm = encode_gemma([negative_prompt])
    else:
        gh, gm = encode_gemma([prompt])
        ugh, ugm = encode_gemma([negative_prompt])

    for t in tqdm(timesteps, desc=f"gen:{prompt[:40]}"):
        t_batch = t.expand(2)
        latent_input = scheduler.scale_model_input(torch.cat([latent, latent], dim=0), t)
        if residual:
            context = torch.cat([clip_cond, clip_uncond], dim=0).clone()
            if cond_long:
                context[:1] += state.cfg.residual_strength * connector(
                    clip_cond.to(dtype=unet_dtype), gh.to(dtype=unet_dtype),
                    t_batch[:1].to(device), gm, context_tokens=ctx)
            if uncond_long:
                context[1:] += state.cfg.residual_strength * connector(
                    clip_uncond.to(dtype=unet_dtype), ugh.to(dtype=unet_dtype),
                    t_batch[1:].to(device), ugm, context_tokens=ctx)
        else:
            h_pair = torch.cat([gh, ugh], dim=0)
            m_pair = torch.cat([gm, ugm], dim=0)
            context = connector(h_pair.to(dtype=unet_dtype), t_batch.to(device), m_pair, context_tokens=ctx)
        camera_pair = None
        if camera_condition is not None:
            camera_single = camera_condition.reshape(1, -1)
            camera_pair = camera_single.expand(2, -1)
        noise_pred = camera_conditioned_unet(
            unet, latent_input, t_batch, encoder_hidden_states=context,
            camera_condition=camera_pair,
        ).sample
        noise_cond, noise_uncond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
        latent = scheduler.step(
            noise_pred, t, latent, generator=gen).prev_sample

    latent = latent / vae.config.scaling_factor
    img = vae.decode(latent.to(dtype=vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


@torch.no_grad()
def generate_clip_teacher(
    prompt: str, state, steps: int = 30, guidance: float = 5.5,
    seed: int = 777, negative_prompt: str = "", height: int = 512,
    width: int = 512,
) -> Image.Image:
    """Generate an image using the CLIP teacher (for diagnostics)."""
    unet = state.unet
    vae = state.vae
    scheduler = state.inf_scheduler or state.scheduler
    device = state.device
    unet_dtype = state.unet_dtype
    encode_clip = state.encode_clip

    gen = torch.Generator(device=device).manual_seed(seed)
    if height % 8 or width % 8:
        raise ValueError("Generation height and width must be divisible by 8")
    latent = torch.randn(
        1, 4, height // 8, width // 8,
        generator=gen, device=device, dtype=unet_dtype,
    )
    scheduler.set_timesteps(steps, device=device)
    latent = latent * scheduler.init_noise_sigma
    timesteps = scheduler.timesteps

    ch, _ = encode_clip([prompt])
    uch, _ = encode_clip([negative_prompt])

    for t in tqdm(timesteps, desc=f"clip:{prompt[:40]}"):
        t_batch = t.expand(2)
        latent_input = scheduler.scale_model_input(torch.cat([latent, latent], dim=0), t)
        h_pair = torch.cat([ch, uch], dim=0)
        noise_pred = camera_conditioned_unet(
            unet, latent_input, t_batch,
            encoder_hidden_states=h_pair.to(dtype=unet_dtype),
        ).sample
        noise_cond, noise_uncond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
        latent = scheduler.step(
            noise_pred, t, latent, generator=gen).prev_sample

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

    connector_was_training = connector.training
    unet_was_training = unet.training
    connector.eval(); unet.eval()
    gen = torch.Generator(device=device).manual_seed(123)
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    t = torch.tensor([timestep], device=device).long()
    preds = []
    for ptxt in prompts:
        gh, gm = encode_gemma([ptxt])
        ctx = connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=cfg.context_tokens)
        pred = camera_conditioned_unet(
            unet, latent, t, encoder_hidden_states=ctx).sample.float()
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
    connector.train(connector_was_training)
    unet.train(unet_was_training)
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

    ch, _ = encode_clip([prompt])
    uch, _ = encode_clip([""])
    clip_h = torch.cat([ch, uch], dim=0)
    teacher = camera_conditioned_unet(
        unet, noisy_pair, t_pair,
        encoder_hidden_states=clip_h.to(dtype=unet_dtype),
    ).sample.float()
    teacher_cond, teacher_uncond = teacher.chunk(2)

    gh, gm = encode_gemma([prompt])
    ugh, ugm = encode_gemma([""])
    gemma_h = torch.cat([gh, ugh], dim=0)
    gemma_m = torch.cat([gm, ugm], dim=0)
    ctx = connector(gemma_h.to(dtype=unet_dtype), t_pair, gemma_m, context_tokens=cfg.context_tokens)
    student = camera_conditioned_unet(
        unet, noisy_pair, t_pair, encoder_hidden_states=ctx).sample.float()
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


@torch.no_grad()
def camera_counterfactual_sensitivity(
    state, prompt: str = None, label: str = "camera_cf", wandb=None,
) -> dict:
    """Measure camera A/B influence on guided and component predictions."""
    cfg = state.cfg
    if not cfg.camera_conditioning_enabled:
        return {}
    prompt = prompt or cfg.val_prompts[0]
    connector = state.connector
    unet = state.unet
    connector_was_training = connector.training
    unet_was_training = unet.training
    connector.eval()
    unet.eval()

    generator = torch.Generator(device=state.device).manual_seed(cfg.val_seed)
    latent = torch.randn(
        1, 4, 64, 64, generator=generator, device=state.device,
        dtype=state.unet_dtype)
    noise = torch.randn(
        latent.shape, generator=generator, device=state.device,
        dtype=state.unet_dtype)
    timestep_value = min(
        cfg.camera_diagnostic_timestep,
        state.scheduler.config.num_train_timesteps - 1,
    )
    timestep = torch.tensor([timestep_value], device=state.device).long()
    noisy = state.scheduler.add_noise(latent, noise, timestep)
    noisy_pair = noisy.expand(2, -1, -1, -1)
    timestep_pair = timestep.expand(2)

    cond_h, cond_mask = state.encode_gemma([prompt])
    uncond_h, uncond_mask = state.encode_gemma([""])
    gemma_h = torch.cat((cond_h, uncond_h), dim=0)
    gemma_mask = torch.cat((cond_mask, uncond_mask), dim=0)
    context = connector(
        gemma_h.to(dtype=state.unet_dtype), timestep_pair, gemma_mask,
        context_tokens=cfg.context_tokens)

    def evaluate(name: str, field_name: str,
                 value_a: float, value_b: float):
        conditions = []
        for value in (value_a, value_b):
            single = make_camera_condition(
                **{field_name: value}, capture_type="photo").to(state.device)
            conditions.append(single.reshape(1, -1).expand(2, -1))
        predictions = [
            camera_conditioned_unet(
                unet, noisy_pair, timestep_pair,
                encoder_hidden_states=context, camera_condition=condition,
            ).sample.float()
            for condition in conditions
        ]
        delta_a = predictions[0][0:1] - predictions[0][1:2]
        delta_b = predictions[1][0:1] - predictions[1][1:2]
        sensitivity = _rel_diff(delta_a, delta_b)
        guided_a = guided_prediction(
            predictions[0][0:1], predictions[0][1:2], cfg.val_guidance)
        guided_b = guided_prediction(
            predictions[1][0:1], predictions[1][1:2], cfg.val_guidance)
        guided_rel_diff = _rel_diff(guided_a, guided_b)
        prediction_rel_diff = _rel_diff(predictions[0], predictions[1])
        print(
            f"[{label}] {name} {value_a:g} vs {value_b:g}: "
            f"guided rel_diff={guided_rel_diff:.6f} "
            f"CFG-delta rel_diff={sensitivity:.6f}"
        )
        return {
            f"{name}_guided_relative_diff": guided_rel_diff,
            f"{name}_cfg_delta_relative_diff": sensitivity,
            f"{name}_prediction_relative_diff": prediction_rel_diff,
            f"{name}_a": value_a,
            f"{name}_b": value_b,
        }

    observed_fov = bool(
        getattr(state, "camera_presence_counts", [0])[0])
    metrics = {}
    if cfg.camera_run_fov_counterfactual or observed_fov:
        metrics.update(evaluate(
            "fov", "vertical_fov_deg", cfg.camera_counterfactual_fov_a,
            cfg.camera_counterfactual_fov_b))
    else:
        print(f"[{label}] FOV counterfactual skipped: no observed FOV labels")
    metrics.update(evaluate(
        "focal", "focal_length_mm", cfg.camera_counterfactual_focal_a,
        cfg.camera_counterfactual_focal_b))
    _log_metrics(f"camera_sensitivity/{label}", metrics, wandb=wandb)
    connector.train(connector_was_training)
    unet.train(unet_was_training)
    return metrics


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
    image_mask = batch.get("image_mask")
    gen = torch.Generator(device=device).manual_seed(seed)
    latent = vae.encode(img).latent_dist.sample(generator=gen) * vae.config.scaling_factor
    noise = torch.randn_like(latent)
    t_val = min(timestep, scheduler.config.num_train_timesteps - 1)
    t = torch.full((latent.shape[0],), t_val, device=device).long()
    noisy = scheduler.add_noise(latent, noise, t)
    gh, gm = encode_gemma(captions)
    ctx = connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=cfg.context_tokens)
    camera_condition = batch.get("camera_condition")
    pred = camera_conditioned_unet(
        unet, noisy, t, encoder_hidden_states=ctx,
        camera_condition=camera_condition,
    ).sample
    if image_mask is None:
        loss = F.mse_loss(pred.float(), noise.float()).item()
    else:
        mask = F.interpolate(
            image_mask.float(), size=pred.shape[-2:], mode="nearest").to(
                device=device, dtype=pred.dtype)
        error = (pred.float() - noise.float()).pow(2)
        loss = ((error * mask).sum() /
                (mask.sum().clamp_min(1.0) * pred.shape[1])).item()
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


def save_validation_grid(
    images: List[Image.Image], labels: List[str], path: str,
    title: str = "", cols: int = 3,
):
    """Save a grid of generated images to disk."""
    grid = _pil_grid(images, labels, cols=cols)
    grid.save(path)
    print(f"Validation grid saved: {path}")


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
    if cfg.max_gemma_len <= cfg.clip_anchor_tokens:
        print("Suffix counterfactual grids skipped without long Gemma input")
        return []
    validate_suffix_counterfactual_token_boundaries(state, cases)
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
        labels.append(f"fullA@G{cfg.max_gemma_len}/C{ctx} {case['name']}")
        # Full prompt B at context_tokens
        imgs.append(generate_ella(
            prefix + " " + case["prompt_suffix_b"], state,
            steps=case.get("steps", cfg.val_steps),
            guidance=case.get("guidance", cfg.val_guidance),
            seed=case.get("seed", cfg.val_seed),
            context_tokens=ctx,
        ))
        labels.append(f"fullB@G{cfg.max_gemma_len}/C{ctx} {case['name']}")
        out_path = f"{path_prefix}_{case['name']}.png"
        save_validation_grid(imgs, labels, out_path, f"{title_prefix}: {case['name']}")
        out_paths.append(out_path)
    return out_paths


# ---------------------------------------------------------------------------
# Suffix counterfactual sensitivity
# ---------------------------------------------------------------------------
def _gemma_token_count(state, text: str) -> int:
    return len(state.gemma_tokenizer.encode(
        text, add_special_tokens=True, truncation=False))


def validate_suffix_counterfactual_token_boundaries(state, cases=None) -> dict:
    """Prove that suffix cases test late, non-truncated Gemma input."""
    cfg = state.cfg
    cases = cfg.suffix_counterfactual_cases if cases is None else cases
    prefix = cfg.suffix_counterfactual_prefix
    signature = (
        prefix,
        cfg.clip_anchor_tokens,
        cfg.max_gemma_len,
        tuple(
            (case["name"], case["prompt_suffix_a"], case["prompt_suffix_b"])
            for case in cases
        ),
    )
    if (getattr(state, "suffix_token_boundary_signature", None) == signature
            and getattr(state, "suffix_token_boundaries", None) is not None):
        return state.suffix_token_boundaries

    prefix_tokens = _gemma_token_count(state, prefix + " ")
    if prefix_tokens <= cfg.clip_anchor_tokens:
        raise ValueError(
            f"Suffix starts at Gemma token {prefix_tokens}, not after the "
            f"{cfg.clip_anchor_tokens}-token anchor"
        )

    per_case = {}
    for case in cases:
        prompt_a = prefix + " " + case["prompt_suffix_a"]
        prompt_b = prefix + " " + case["prompt_suffix_b"]
        tokens_a = _gemma_token_count(state, prompt_a)
        tokens_b = _gemma_token_count(state, prompt_b)
        longest = max(tokens_a, tokens_b)
        if longest > cfg.max_gemma_len:
            raise ValueError(
                f"Suffix diagnostic {case['name']!r} needs {longest} Gemma "
                f"tokens but max_gemma_len={cfg.max_gemma_len}"
            )
        per_case[case["name"]] = {
            "prefix_tokens": prefix_tokens,
            "prompt_a_tokens": tokens_a,
            "prompt_b_tokens": tokens_b,
        }
        print(
            f"[suffix_tokens] {case['name']}: prefix={prefix_tokens} "
            f"full_a={tokens_a} full_b={tokens_b} limit={cfg.max_gemma_len}"
        )
    state.suffix_token_boundary_signature = signature
    state.suffix_token_boundaries = per_case
    return per_case


@torch.no_grad()
def suffix_counterfactual_sensitivity(
    state, label: str = "suffix_cf", wandb=None,
) -> dict:
    """Measure how much varying the suffix changes the CFG delta.
    
    Returns rel_diff between delta_A and delta_B for each counterfactual case.
    Higher = suffix is actually changing the UNet output.
    """
    cfg = state.cfg
    ctx = cfg.context_tokens
    if (cfg.max_gemma_len <= cfg.clip_anchor_tokens
            or not cfg.suffix_counterfactual_cases):
        return {"suffix_sensitivity_mean": None, "per_case": {}}
    token_boundaries = validate_suffix_counterfactual_token_boundaries(state)
    
    connector = state.connector
    unet = state.unet
    scheduler = state.scheduler
    device = state.device
    unet_dtype = state.unet_dtype
    encode_gemma = state.encode_gemma

    connector_was_training = connector.training
    unet_was_training = unet.training
    connector.eval(); unet.eval()
    prefix = cfg.suffix_counterfactual_prefix
    timestep = cfg.suffix_diagnostic_timestep
    
    sensitivities = {}
    for case in cfg.suffix_counterfactual_cases:
        name = case["name"]
        prompt_a = prefix + " " + case["prompt_suffix_a"]
        prompt_b = prefix + " " + case["prompt_suffix_b"]
        
        gen = torch.Generator(device=device).manual_seed(int(case.get("seed", 777)))
        latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
        noise = torch.randn(
            latent.shape, generator=gen, device=device, dtype=unet_dtype)
        t = torch.tensor([int(timestep)], device=device).long()
        noisy = scheduler.add_noise(latent, noise, t)
        
        # Encode prompts
        gh_a, gm_a = encode_gemma([prompt_a])
        gh_b, gm_b = encode_gemma([prompt_b])
        ugh, ugm = encode_gemma([""])
        
        # Get contexts (cond + uncond for each)
        ctx_a_c = connector(gh_a.to(dtype=unet_dtype), t, gm_a, context_tokens=ctx)
        ctx_u = connector(ugh.to(dtype=unet_dtype), t, ugm, context_tokens=ctx)
        ctx_b_c = connector(gh_b.to(dtype=unet_dtype), t, gm_b, context_tokens=ctx)
        
        # UNet predictions
        pred_a_c = camera_conditioned_unet(
            unet, noisy, t, encoder_hidden_states=ctx_a_c).sample.float()
        pred_u = camera_conditioned_unet(
            unet, noisy, t, encoder_hidden_states=ctx_u).sample.float()
        pred_b_c = camera_conditioned_unet(
            unet, noisy, t, encoder_hidden_states=ctx_b_c).sample.float()
        
        delta_a = pred_a_c - pred_u
        delta_b = pred_b_c - pred_u
        
        sensitivity = _rel_diff(delta_a, delta_b)
        sensitivities[name] = sensitivity
        print(f"[{label}] {name}: suffix_sensitivity={sensitivity:.5f}")
    
    mean_sens = float(np.mean(list(sensitivities.values()))) if sensitivities else 0.0
    _log_metrics(f"suffix_sensitivity/{label}", sensitivities, wandb=wandb)
    safe_wandb_log({f"suffix_sensitivity/{label}_mean": mean_sens}, wandb=wandb)
    connector.train(connector_was_training)
    unet.train(unet_was_training)
    return {
        "suffix_sensitivity_mean": mean_sens,
        "per_case": sensitivities,
        "token_boundaries": token_boundaries,
    }
