# gemma3-sd-pure-ella

Replace Stable Diffusion's CLIP text encoder with Gemma 3 via a pure ELLA-style
timestep-aware connector. Final inference is CLIP-free: `Gemma → connector → frozen UNet`.

## Architecture

```
Gemma 3 270M (frozen)
  → Timestep-aware ELLA connector (trainable)
  → [B, 77, 768] SD1.5-compatible conditioning tokens
  → Original StyleJourney SD 1.5 UNet cross-attention (frozen)
```

CLIP is used as an optional teacher/diagnostic during training but is not loaded
at inference time. `max_gemma_len`, rather than the U-Net conditioning length,
controls long-prompt support. Captions target roughly 256 Gemma tokens and the
default 320-token window provides headroom instead of silently truncating tails.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Train the long-Gemma-input / 77-SD-token baseline
python train.py config.json

# 3. Resume ELLA training after a completed compatibility pretrain
python train.py config.json --phases ella \
    --resume-ckpt output/ella_connector_clip_pretrain.pt
```

## Configuration

Edit `config.json` or pass a custom path. Key settings:

| Field | Default | Description |
|-------|---------|-------------|
| `experiment_stage` | `stage2_long_context_no_sara` | `stage3_long_context_with_sara` enables the optional SaRA phase |
| `run_mode` | `short_train` | `diagnostic`, `overfit_train`, `short_train`, `full_train` |
| `connector_type` | `ella_tsc` | ELLA-style fixed-query timestep-aware resampler |
| `max_gemma_len` | 320 | Input limit with headroom for roughly 256-token captions |
| `context_tokens` | 77 | Fixed SD1.5 cross-attention contract; output expansion is rejected |
| `gemma_layer_mix_count` | 4 | Learned mixture of upper Gemma hidden layers |
| `aspect_ratio_buckets` | five buckets | Full-frame resize-and-pad buckets; images are not cropped |
| `caption_mix_*` | 25/25/50% | Sample paired short/medium/long captions when supplied |
| `fail_on_prompt_truncation` | `true` | Fail instead of training on silently truncated descriptions |
| `gemma_id` | `google/gemma-3-270m-it` | Gated model — needs HF token |
| `sd_checkpoint` | `""` | Path to `.safetensors` file; empty = runwayml SD 1.5 |
| `wandb_enabled` | `false` | Requires `WANDB_API_KEY` when enabled |

`run_mode` controls whether training runs; sample and step budgets are always
explicit configuration values and are never silently overwritten by the mode.

## Training phases

1. **CLIP alignment pretrain** — optional Phase 0. Uses `caption_short` when
   supplied and decodes the exact CLIP-visible prefix before feeding both Gemma
   and CLIP. This prevents a suffix-blind teacher from supervising long text.

2. **ELLA connector diffusion training** — Phase 1. Trains the connector
   end-to-end with the standard diffusion noise-prediction loss and 10% CFG
   conditioning dropout. The U-Net is frozen. When paired variants exist,
   batches sample 25% short, 25% medium, and 50% long captions. Phase-1 CLIP
   teacher losses are disabled in the supplied long-caption configs.

3. **Sparse SaRA attn2 K/V adaptation** — Phase 2. Sparse gradient-masked
   fine-tuning of the undertrained (|w| < threshold) UNet cross-attention K/V
   weights, with warn/abort gates on the sparse fraction and a removable
   sparse-patch checkpoint format. Only activated at
   `stage3_long_context_with_sara`.

Phases can be selected and resumed from the CLI:

```bash
python train.py config_long_256.json --phases ella \
    --resume-ckpt output/ella_connector_clip_pretrain.pt
```

## Output

All artifacts land in `output_dir` (default `./output`):

- `ella_connector_clip_pretrain.pt` — after pretrain
- `ella_connector_frozen_unet.pt` — after ELLA training
- `pure_ella_connector_L77.pt` — final Gemma-long-input / SD1.5-compatible connector
- `pure_ella_unet_attn2_kv_sparse_L*.pt` — SaRA sparse UNet patch (stage 3 only)
- `validation_ella_L*.png` — validation grids
- `proof_reloaded_pure_ella_L*.png` — CLIP-free reloaded proof grids
- `validation_clip_teacher.png` — teacher baseline (CLIP available)
- `validation_suffix_counterfactual_G*_C77_*.png` — late-input utilization grids

## Dependencies

Python ≥ 3.10, managed by [uv](https://docs.astral.sh/uv/):

- `torch` ≥ 2.1, `diffusers` ≥ 0.28, `transformers` ≥ 4.45
- `datasets` (streaming), `wandb`, `torchvision`, `pillow`, `torchmetrics[image]`

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | Yes (Gemma is gated) | HuggingFace auth for model downloads |
| `WANDB_API_KEY` | Only if `wandb_enabled: true` | Weights & Biases logging |

## Roadmap and code review

- [`docs/PLAN.md`](docs/PLAN.md) — the four-pillar roadmap: long Gemma input,
  optional SaRA widening, camera-intrinsics / capture-metadata conditioning,
  and optional zero-gated DiT-style UNet capacity. The implemented baseline
  preserves 77 U-Net conditioning tokens while Gemma reads dense captions
  through a 320-token input window.
- [`docs/REVIEW.md`](docs/REVIEW.md) — code review findings backing the plan's
  Phase 0 fix list (known bugs, refactors, and config hazards with file:line
  references).

## Colab notebook

The repository also contains `gemma3_sd_pure_ella_colab.ipynb` — the full
notebook with extra-token validation suite, suffix counterfactuals, FID/KID
metrics, and interactive generation. Used as the reference implementation
from which `train.py` was derived.

## License

MIT
