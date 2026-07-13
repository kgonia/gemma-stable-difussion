# Code review — train.py + pure_ella package

*Reviewed 2026-07-03 on branch `dual-conditioning`, alongside the creation of
[`PLAN.md`](PLAN.md). Scope: `train.py`, `pure_ella/config.py`,
`pure_ella/connector.py`, `pure_ella/sara.py`, `pure_ella/dataset.py`,
`config_long_256.json`. `pure_ella/diagnostics.py` (770 lines) was not deep-reviewed.*

Findings ordered by severity. Status column is for tracking fixes.

| # | Severity | Where | Finding | Status |
|---|----------|-------|---------|--------|
| 1 | **Bug** | `train.py:800`, `train.py:817` | `max_samples_sara` never used — SaRA phase runs on `max_samples_ella` | **fixed** |
| 2 | **Bug** | `pure_ella/config.py:294` | `resolve_sd_checkpoint` fallback chain unreachable for absolute paths | **fixed** |
| 3 | Refactor | shared diffusion step/loop | ~200 duplicated lines between ELLA and SaRA training loops | **fixed** |
| 4 | Perf | `train.py:602`, `train.py:882` | `ClipGeometryLoss()` constructed every step inside the loop | **fixed** |
| 5 | Memory | `pure_ella/sara.py:54` | Sparse masks stored at weight dtype (fp32) instead of bool | **fixed** |
| 6 | Config | `pure_ella/config.py`, `train.py` | Explicit weight dtype and BF16 autocast | **fixed** |
| 7 | Hazard | `pure_ella/config.py:279–282` | Unknown JSON keys silently dropped (typos vanish) | **fixed** |
| 8 | Hazard | `pure_ella/config.py:213–218` | `context_tokens` in JSON silently overridden by `experiment_stage` | **fixed** |
| 9 | Hazard | `pure_ella/config.py:224–245` | `run_mode` overfit/diagnostic overrides clobber JSON-provided budgets | **fixed** |
| 10 | Perf | `pure_ella/dataset.py:101` | `num_workers=0` on streaming loader — decode/resize on main thread | open |
| 11 | Docs | `README.md` | Called SaRA phase a "stub" — it is fully implemented | **fixed** |

---

## 1. `max_samples_sara` is dead config (bug)

`run_sara_training` uses the ELLA budget for both the step-plan estimate and the
dataloader:

```python
# train.py:800
plan = estimate_steps(cfg.max_samples_ella, cfg.train_batch_size,
                      cfg.sara_epochs, cfg.sara_max_opt_steps)
# train.py:815-820
dl = make_streaming_dataloader(
    cfg.stream_repo, phase=30, epoch=epoch,
    max_samples=cfg.max_samples_ella, ...)
```

`config_long_256.json` sets `max_samples_sara: 40000` and `max_samples_ella: 200000`;
the SaRA phase will stream up to 200k samples per epoch, not 40k. The
`sara_max_opt_steps: 5000` cap limits the damage in practice, but the epoch structure
and the printed `StagePlan` are wrong.

**Fix:** use `cfg.max_samples_sara` in both places.

## 2. `resolve_sd_checkpoint` fallback chain is unreachable (bug)

```python
# pure_ella/config.py:301-306
for p in candidates:
    if p and os.path.exists(p):
        return p
    if p and not os.path.exists(p) and "/" in p:
        return p  # assume it's a valid HF model ID
```

The first candidate is `cfg.sd_checkpoint`. If that's a missing local path like
`/workspace/stylejourney_v10.safetensors`, it contains `/`, so it is returned
immediately as a supposed HF model ID — the `~/models/...` and Drive fallbacks are
never tried, and the run later fails with a confusing HF 404 instead of a clear
"checkpoint not found".

**Fix:** only treat a candidate as an HF ID if it doesn't look like a filesystem path —
e.g. no leading `/`, `.`, or `~`, exactly one `/`, and no `.safetensors` suffix. Move
that check after the whole existence loop.

## 3. Shared ELLA and SaRA training step (fixed)

`run_ella_training` (train.py:490) and `run_sara_training` (train.py:751) share the
teacher-delta paired forward, loss assembly, NaN gate, grad-stat logging, validation
cadence, and step-cap logic — nearly line for line. They have already drifted (suffix
counterfactual loss exists only in the ELLA loop; wandb prefixes differ).

This is the main blocker for the PLAN.md P3 work: metadata conditioning modifies the
training step, and today every such change must be written twice and kept in sync by
hand.

`diffusion_training_step` now owns image/latent preparation, conditional and
teacher forwards, and loss assembly. `run_diffusion_training_loop` owns budgets,
optimization, logging, and validation cadence for both phases.

## 4. `ClipGeometryLoss()` instantiated per step (perf, minor)

`train.py:602` and `train.py:882` construct the loss object inside the inner loop
whenever the semantic anchor weight is > 0. Hoist to before the loop (as
`run_clip_pretrain` already does at train.py:345).

## 5. SaRA masks stored at weight dtype (memory, minor)

```python
# pure_ella/sara.py:54
p._sara_sparse_mask = mask.to(device=p.device, dtype=p.dtype)
```

Stores a full-size fp32 tensor per target weight — duplicating every `attn2.to_k/to_v`
tensor on the GPU. Store as `bool` (1 byte/elem instead of 4) and cast inside the
gradient hook, which already does `mm.to(dtype=grad.dtype)`.

Note the surrounding machinery is otherwise sound: hook closures bind the mask
correctly at registration time, non-selected entries get zero grad and zero AdamW
momentum, and the UNet param group uses `weight_decay=0.0` so decay cannot leak into
non-selected entries. The sparse save/reload path (`collect_sara_sparse_values` /
`load_sara_sparse_values`) round-trips correctly.

## 6. Configurable model dtype and BF16 autocast (fixed)

`model_weight_dtype` controls parameter storage and `mixed_precision` controls
forward autocast. Supplied configs keep trainable weights and Adam state in FP32,
use BF16 CUDA autocast, and retain FP32 loss reductions. SaRA rejects BF16 U-Net
weights because sparse updates require full-precision parameters.

## 7–9. Config-loading hazards (document or assert)

- **7.** `TrainConfig.from_json` (config.py:279) applies only keys that match existing
  attributes; a typo like `elle_max_opt_steps` is silently ignored. Fix: warn or raise
  on unknown keys.
- **8.** `__post_init__` derives `context_tokens` from `experiment_stage` /
  `long_context_target`, ignoring the JSON value — `config_long_256.json` says
  `"context_tokens": 77` but runs at 256. Intended, but the config file misleads.
  Fix: drop the key from configs, or assert JSON value matches the derived one.
- **9.** `from_json` re-runs `__post_init__` after applying overrides, so
  `run_mode: overfit_train`/`diagnostic` clobber any sample/step budgets given in the
  same JSON. Intended as a preset, but worth a printed notice when it happens.

## 10. Streaming dataloader on the main thread (perf)

`pure_ella/dataset.py:101` uses `num_workers=0`; JPEG decode, center-crop, and LANCZOS
resize run on the training process. With batch 8 this can starve the GPU. The
`_stream_prefetch.py` / `download_then_tokenize.py` experiments in the repo root look
like they're already probing this. A background-thread prefetcher or
`num_workers=2` with a picklable iterable would be the minimal fix.

Also noted: the dataset only extracts `prompt/caption/text` from the embedded JSON
(`dataset.py:37-48`). PLAN.md P3 (metadata conditioning) will need this hook extended
to emit a `metadata` dict — and `jackyhate/text-to-image-2M` is synthetic imagery with
no real camera EXIF, so P3 needs a data-source decision first (see PLAN.md).

## Positive observations

- The zero-effect-init discipline is applied consistently: `extra_gate_logit` init −5.0
  (−3.5 in the long config), recursive/TRM gates, SaRA near-zero weight selection. This
  is exactly the right invariant for the roadmap and should be kept mandatory for new
  pillars.
- `run_reload_proof` (train.py:1124) reloads the connector strictly, rebuilds a fresh
  UNet, applies the sparse patch, and generates CLIP-free — a genuine end-to-end
  deliverable proof, not a smoke test.
- Diagnostics culture is strong: suffix counterfactual sensitivity + contrastive loss,
  extra-token ablation with an explicit < 0.03 warning threshold, teacher–student delta
  alignment, FID/KID, VRAM logging.
- Loss NaN gates (`raise RuntimeError` on non-finite loss) in every phase — fail fast
  instead of training garbage.
