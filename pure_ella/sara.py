"""
Sparse SaRA-style attn2 K/V adaptation for SD UNet.

Creates sparse binary masks on attn2.to_k/attn2.to_v weight params,
registers gradient hooks to zero-out gradients for non-selected entries,
and provides save/load helpers for the sparse value patches.
"""

from __future__ import annotations
import math
from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn


def is_sara_target_param(name: str, target_substrings: Sequence[str]) -> bool:
    """Check if a parameter name matches SaRA target substrings."""
    return any(s in name for s in target_substrings) and name.endswith("weight")


def build_sara_attn2_kv_sparse_masks(
    unet: nn.Module,
    selection_mode: str = "magnitude_threshold",
    target_fraction: float = 0.10,
    threshold: float = 1e-3,
    min_sparse_fraction: float = 0.0,
    max_sparse_fraction_warn: float = 0.02,
    max_sparse_fraction_abort: float = 0.05,
    target_substrings: Tuple[str, ...] = ("attn2.to_k", "attn2.to_v"),
) -> dict:
    """
    Build sparse binary masks on attn2 K/V weight parameters.

    ``magnitude_threshold`` selects weights where ``|weight| < threshold``.
    ``target_fraction`` selects an exact global fraction of the smallest
    magnitude weights across the complete target scope.
    Parameters with zero selected entries get requires_grad=False.

    Returns a summary dict with selection statistics.
    """
    if selection_mode not in {"magnitude_threshold", "target_fraction"}:
        raise ValueError(f"Unknown SaRA selection mode: {selection_mode}")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("SaRA target_fraction must be in (0, 1]")
    if threshold <= 0:
        raise ValueError("SaRA threshold must be positive")
    if not (
        0.0 <= min_sparse_fraction <= max_sparse_fraction_warn
        <= max_sparse_fraction_abort <= 1.0
    ):
        raise ValueError(
            "SaRA gates must satisfy 0 <= min <= warn <= abort <= 1"
        )

    targets = []
    total_unet = sum(p.numel() for p in unet.parameters())

    for name, p in unet.named_parameters():
        if hasattr(p, "_sara_sparse_mask"):
            del p._sara_sparse_mask
        if not is_sara_target_param(name, target_substrings):
            p.requires_grad_(False)
            continue
        if not p.is_leaf:
            print("Skipping non-leaf target:", name)
            p.requires_grad_(False)
            continue
        targets.append((name, p))

    total_target = sum(p.numel() for _, p in targets)
    if total_target == 0:
        raise RuntimeError(
            "SaRA found no target parameters; adjust sara_target_substrings"
        )

    magnitude_cutoff = None
    tie_budget = 0
    if selection_mode == "target_fraction":
        requested = max(1, min(total_target, round(total_target * target_fraction)))
        magnitudes = torch.cat([
            p.detach().abs().reshape(-1).float().cpu() for _, p in targets
        ])
        magnitude_cutoff = float(torch.kthvalue(magnitudes, requested).values)
        strictly_lower = int((magnitudes < magnitude_cutoff).sum().item())
        tie_budget = requested - strictly_lower
        del magnitudes

    selected = 0
    rows = []
    for name, p in targets:
        magnitudes = p.detach().abs()
        if selection_mode == "target_fraction":
            mask = magnitudes < magnitude_cutoff
            if tie_budget:
                ties = magnitudes == magnitude_cutoff
                tie_count = int(ties.sum().item())
                take = min(tie_budget, tie_count)
                if take == tie_count:
                    mask = mask | ties
                elif take:
                    tie_indices = torch.nonzero(
                        ties.reshape(-1), as_tuple=False
                    )[:take, 0]
                    mask = mask.reshape(-1)
                    mask[tie_indices] = True
                    mask = mask.reshape_as(p)
                tie_budget -= take
        else:
            mask = magnitudes < threshold
        cnt = int(mask.sum().item())
        total = int(p.numel())
        selected += cnt
        p.requires_grad_(cnt > 0)
        rows.append({
            "name": name,
            "selected": cnt,
            "total": total,
            "fraction": cnt / max(total, 1),
        })
        p._sara_sparse_mask = mask.to(device=p.device, dtype=torch.bool)

    if tie_budget:
        raise RuntimeError(
            f"SaRA exact-rank selection left {tie_budget} unresolved ties"
        )
    frac = selected / max(total_target, 1)
    target_scope_fraction = total_target / max(total_unet, 1)
    whole_unet_fraction = selected / max(total_unet, 1)
    selector = (
        f"target_fraction={target_fraction:.4%} cutoff={magnitude_cutoff:.8g}"
        if selection_mode == "target_fraction"
        else f"threshold={threshold:g}"
    )
    print(
        f"SaRA-style attn2 K/V sparse selected: {selected:,} / {total_target:,} "
        f"target entries ({frac:.4%}); target scope={target_scope_fraction:.4%} "
        f"and selected={whole_unet_fraction:.4%} of the whole U-Net; "
        f"mode={selection_mode} {selector}"
    )
    for row in rows[:30]:
        print(
            f"  {row['name']}: {row['selected']:,}/{row['total']:,} "
            f"({row['fraction']:.4%})"
        )

    if selected == 0:
        raise RuntimeError(
            "SaRA selected zero attn2 K/V params; "
            "adjust threshold or target scope"
        )
    if frac < min_sparse_fraction:
        raise RuntimeError(
            f"SaRA sparse fraction {frac:.4%} is below minimum gate "
            f"{min_sparse_fraction:.4%}"
        )
    if frac > max_sparse_fraction_abort:
        raise RuntimeError(
            f"SaRA sparse fraction {frac:.4%} exceeds abort gate "
            f"{max_sparse_fraction_abort:.4%}"
        )
    if frac > max_sparse_fraction_warn:
        print(
            f"WARNING: sparse fraction {frac:.4%} exceeds warn gate "
            f"{max_sparse_fraction_warn:.4%}"
        )

    return {
        "selected": selected,
        "total_target": total_target,
        "fraction": frac,
        "total_unet": total_unet,
        "target_scope_fraction": target_scope_fraction,
        "whole_unet_fraction": whole_unet_fraction,
        "selection_mode": selection_mode,
        "target_fraction": target_fraction,
        "threshold": threshold,
        "magnitude_cutoff": magnitude_cutoff,
        "rows": rows,
    }


def capture_sara_selected_values(unet: nn.Module) -> Dict[str, torch.Tensor]:
    """Copy selected values to CPU for post-training update diagnostics."""
    values = {}
    for name, p in unet.named_parameters():
        mask = getattr(p, "_sara_sparse_mask", None)
        if mask is not None and mask.any():
            values[name] = p.detach()[mask].float().cpu()
    return values


def sara_selected_delta_metrics(
    unet: nn.Module,
    baseline: Dict[str, torch.Tensor],
) -> dict:
    """Measure how far the selected weights moved from their phase-2 start."""
    delta_sq = 0.0
    baseline_sq = 0.0
    max_abs = 0.0
    selected = 0
    named = dict(unet.named_parameters())
    for name, initial in baseline.items():
        p = named[name]
        mask = getattr(p, "_sara_sparse_mask", None)
        if mask is None:
            raise RuntimeError(f"SaRA mask disappeared for {name}")
        current = p.detach()[mask].float().cpu()
        if current.shape != initial.shape:
            raise RuntimeError(f"SaRA mask changed during training for {name}")
        delta = current - initial
        delta_sq += float(delta.square().sum().item())
        baseline_sq += float(initial.square().sum().item())
        if delta.numel():
            max_abs = max(max_abs, float(delta.abs().max().item()))
        selected += delta.numel()

    delta_l2 = math.sqrt(delta_sq)
    return {
        "selected_delta_l2": delta_l2,
        "selected_delta_relative_l2": delta_l2 / max(
            math.sqrt(baseline_sq), 1e-12
        ),
        "selected_delta_rms": math.sqrt(delta_sq / max(selected, 1)),
        "selected_delta_max_abs": max_abs,
    }


def install_sara_gradient_masks(unet: nn.Module) -> list:
    """
    Register backward hooks that zero gradients for non-selected entries.

    Returns a list of hook handles (call remove() on each to uninstall).
    """
    handles = []
    for name, p in unet.named_parameters():
        mask = getattr(p, "_sara_sparse_mask", None)
        if mask is None or not p.requires_grad:
            continue

        def make_hook(m: torch.Tensor):
            mm = m.to(device=p.device)
            return lambda grad: grad * mm

        handles.append(p.register_hook(make_hook(mask)))

    print(f"Installed sparse gradient hooks: {len(handles)}")
    return handles


def remove_sara_gradient_masks(handles: list):
    """Remove all SaRA gradient mask hooks."""
    for h in handles:
        h.remove()


def collect_sara_sparse_values(unet: nn.Module) -> dict:
    """
    Collect sparse mask + values from UNet for checkpoint save.

    Returns dict: {param_name: {mask, values, shape}}.
    Returns empty dict if no sparse masks found.
    """
    sparse_values = {}
    for name, p in unet.named_parameters():
        mask = getattr(p, "_sara_sparse_mask", None)
        if mask is not None:
            cpu_mask = mask.detach().cpu().bool()
            sparse_values[name] = {
                "mask": cpu_mask,
                "values": p.detach().cpu()[cpu_mask],
                "shape": tuple(p.shape),
            }
    return sparse_values


def load_sara_sparse_values(unet: nn.Module, sparse_values: dict) -> int:
    """
    Apply saved sparse values to UNet parameters in-place.

    Returns number of patched values.
    """
    named = dict(unet.named_parameters())
    patched = 0
    for name, pack in sparse_values.items():
        p = named[name]
        mask = pack["mask"].to(device=p.device)
        vals = pack["values"].to(device=p.device, dtype=p.dtype)
        p.data[mask] = vals
        patched += int(mask.sum().item())
    print(f"Sparse UNet patch loaded: {patched:,} values applied")
    return patched
