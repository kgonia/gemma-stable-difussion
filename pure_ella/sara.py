"""
Sparse SaRA-style attn2 K/V adaptation for SD UNet.

Creates sparse binary masks on attn2.to_k/attn2.to_v weight params,
registers gradient hooks to zero-out gradients for non-selected entries,
and provides save/load helpers for the sparse value patches.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn


def is_sara_target_param(name: str, target_substrings: Tuple[str, ...]) -> bool:
    """Check if a parameter name matches SaRA target substrings."""
    return any(s in name for s in target_substrings) and name.endswith("weight")


def build_sara_attn2_kv_sparse_masks(
    unet: nn.Module,
    threshold: float = 1e-3,
    max_sparse_fraction_warn: float = 0.02,
    max_sparse_fraction_abort: float = 0.05,
    target_substrings: Tuple[str, ...] = ("attn2.to_k", "attn2.to_v"),
) -> dict:
    """
    Build sparse binary masks on attn2 K/V weight parameters.

    Masks are True where |weight| < threshold (sparse selection).
    Parameters with zero selected entries get requires_grad=False.

    Returns a summary dict with selection statistics.
    """
    selected = 0
    total_target = 0
    rows = []

    for name, p in unet.named_parameters():
        if not is_sara_target_param(name, target_substrings):
            p.requires_grad_(False)
            continue
        if not p.is_leaf:
            print("Skipping non-leaf target:", name)
            p.requires_grad_(False)
            continue
        mask = (p.detach().abs() < threshold)
        cnt = int(mask.sum().item())
        total = int(p.numel())
        selected += cnt
        total_target += total
        p.requires_grad_(cnt > 0)
        rows.append((name, cnt, total, cnt / max(total, 1)))
        p._sara_sparse_mask = mask.to(device=p.device, dtype=torch.bool)

    frac = selected / max(total_target, 1)
    print(
        f"SaRA-style attn2 K/V sparse selected: {selected:,} / {total_target:,} "
        f"({100*frac:.4f}%) threshold={threshold:g}"
    )
    for name, cnt, total, f in rows[:30]:
        print(f"  {name}: {cnt:,}/{total:,} ({100*f:.4f}%)")

    if selected == 0:
        raise RuntimeError(
            "SaRA selected zero attn2 K/V params; "
            "adjust threshold or target scope"
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
        "threshold": threshold,
        "rows": rows,
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
