# gemma3-sd-pure-ella

Replace Stable Diffusion's CLIP text encoder with Gemma 3 via a pure ELLA-style
timestep-aware connector. Final inference is CLIP-free: `Gemma → connector → frozen UNet`.

## Architecture

```
Gemma 3 270M (frozen)
  → Timestep-aware ELLA connector (trainable)
  → [B, L, 768] context tokens (L = 77, 128, 192, or 256)
  → Original StyleJourney SD 1.5 UNet cross-attention (frozen)
```

CLIP is used as an optional teacher/diagnostic during training but is not loaded
at inference time.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Run with default config (short 77-token train)
python train.py config.json

# 3. Full train on 50k samples
python train.py --config config.json   # set run_mode to "full_train" in config

# 4. Long-context 128-token train
python train.py --config config.json   # set experiment_stage to "stage2_long_context_no_sara"

# 5. Overfit falsifier (64 samples, deterministic order)
python train.py --config config.json   # set run_mode to "overfit_train"
```

## Configuration

Edit `config.json` or pass a custom path. Key settings:

| Field | Default | Description |
|-------|---------|-------------|
| `experiment_stage` | `stage1_77_pretrain_then_ella` | `stage2_long_context_no_sara`, `stage3_long_context_with_sara` |
| `run_mode` | `short_train` | `diagnostic`, `overfit_train`, `short_train`, `full_train` |
| `connector_type` | `trm_yz` | `ella_tsc`, `recursive_y`, `trm_yz` |
| `context_tokens` | 77 | 77 for phase 1; 128/192/256 for long-context stages |
| `gemma_id` | `google/gemma-3-270m-it` | Gated model — needs HF token |
| `sd_checkpoint` | `""` | Path to `.safetensors` file; empty = runwayml SD 1.5 |
| `wandb_enabled` | `true` | Requires `WANDB_API_KEY` env var |

## Training phases

1. **CLIP alignment pretrain** — optional Phase 0. Trains the first 77 connector
   output tokens to match CLIP hidden states via geometry loss (MSE, cosine,
   pool cosine, norm).

2. **ELLA connector diffusion training** — Phase 1. Trains the connector
   end-to-end with diffusion noise-prediction loss, optional CLIP teacher
   distillation, optional text-delta alignment, and optional semantic anchor
   loss. UNet is frozen.

3. **Sparse SaRA attn2 K/V adaptation** — Phase 2 (stub). Sparse gradient
   masked fine-tuning of UNet cross-attention weights. Only activated at
   `stage3_long_context_with_sara`.

## Output

All artifacts land in `output_dir` (default `./output`):

- `ella_connector_clip_pretrain.pt` — after pretrain
- `ella_connector_frozen_unet.pt` — after ELLA training
- `pure_ella_connector_L{77,128,192,256}.pt` — final deliverable
- `validation_ella_L*.png` — validation grids
- `proof_reloaded_pure_ella_L*.png` — CLIP-free reloaded proof grids
- `validation_clip_teacher.png` — teacher baseline (CLIP available)

## Dependencies

Python ≥ 3.10, managed by [uv](https://docs.astral.sh/uv/):

- `torch` ≥ 2.1, `diffusers` ≥ 0.28, `transformers` ≥ 4.45
- `datasets` (streaming), `wandb`, `torchvision`, `pillow`, `torchmetrics[image]`

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | Yes (Gemma is gated) | HuggingFace auth for model downloads |
| `WANDB_API_KEY` | Only if `wandb_enabled: true` | Weights & Biases logging |

## Colab notebook

The repository also contains `gemma3_sd_pure_ella_colab.ipynb` — the full
notebook with extra-token validation suite, suffix counterfactuals, FID/KID
metrics, and interactive generation. Used as the reference implementation
from which `train.py` was derived.

## License

MIT
