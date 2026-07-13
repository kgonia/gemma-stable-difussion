# Engineering handoff: Gemma conditioning for SD1.5

**Date:** 2026-07-13

**Branch:** `dual-conditioning`

**Last committed revision:** `69af082 Separate and gate P3 camera conditioning`

## Executive decision

Do not continue the current pure-Gemma connector as the near-term production
path. The completed runs show that prompt-only CLIP imitation is an inadequate
replacement objective at the available data and optimization budget, and that
the subsequent frozen-U-Net diffusion run can destroy semantics while still
producing nonzero suffix diagnostics.

The next experiment is a **CLIP-preserving Gemma residual connector**. Native
CLIP remains the conditioning baseline and Gemma learns only a correction for
information outside CLIP's visible prefix:

```python
clip_context = clip(short_or_truncated_prompt)  # [B, 77, 768]
delta = residual_tsc(
    clip_queries=clip_context,
    gemma_states=gemma(full_prompt),
    timestep=timestep,
)
context = clip_context + residual_strength * delta
```

This keeps the pretrained U-Net's exact 77-token cross-attention contract. Do
not concatenate extra tokens into the existing attention path: even zero-valued
extra keys and values change softmax normalization. A separate gated U-Net
attention branch is a later escalation only if the 77 residual slots prove to
be a bottleneck.

The pure connector remains useful as a negative baseline and possible later
distillation target. The completed run was insufficiently controlled and
undertrained, so it does not prove that pure replacement is impossible.

## Repository state

The initial residual architecture and training path are implemented in the
current worktree as `clip_gemma_residual_tsc`. This includes zero-delta
initialization, the control-flow short-prompt bypass, prefix/null and residual
penalties, checkpoint schema validation, focused unit tests, and
`config_clip_gemma_residual_p1.json`. It has **not** completed an end-to-end GPU
smoke run or the paired evaluation protocol, so it is not yet ready for a full
training run.

The worktree is intentionally dirty. It contains the completed prompt-only and
Unsplash experiment implementation, configs, scripts, logs, checkpoints, and
comparison images. At handoff time, tracked modifications include:

- `.gitignore`
- `docs/PLAN.md`
- `pure_ella/config.py`
- `pure_ella/diagnostics.py`
- `tests/test_connector_training.py`
- `train.py`

Important untracked source files include `pure_ella/prompts.py`, the
`config_phase0_*.json` and `config_unsplash_*.json` experiment configs, and the
scripts under `scripts/` used by the runs below. Large output and log directories
are also untracked. Inspect and separate source/config changes from generated
artifacts before committing. Do not discard the dirty worktree or commit the
large model outputs by default.

The project-owned connector suite was run on this dirty worktree at handoff and
passed 46 tests. Run it again after each integration increment because the
current dirty state is newer than the last commit:

```bash
PYTHONPATH=. uv run --frozen pytest -q tests/test_connector_training.py
```

## Completed experiments

### Prompt-only Phase 0: pure ELLA-TSC

- Sources: 1,000,581 training prompts and 512 held-out prompts.
- Connector: six-block, width-768 `ella_tsc`, approximately 80M parameters.
- Batch: 64.
- Steps: 15,635, one pass, about 28.4 minutes.
- Best checkpoint: step 15,500.
- Held-out loss: `1.09352`.
- Held-out pooled cosine: `0.853187`.
- Held-out token cosine loss: `0.35305`, or approximately `0.647` mean
  per-token cosine similarity.
- W&B: <https://wandb.ai/mistic-jedi/gemma3-sd-phase0-objective-fix/runs/sv79xv9p>

Artifacts:

- `output_phase0_ella_tsc_all_prompts/pure_ella_connector_L77.pt`
- `output_phase0_ella_tsc_all_prompts/ella_connector_clip_pretrain_best.pt`
- `logs/all_prompts_then_unsplash/phase0.log`

Interpretation: pooled similarity overstated success. SD1.5 cross-attention
consumes token-level geometry, where the match remained materially worse. The
connector produced some plausible generic images, but failed distinctive
entities and styles even when those concepts occurred in the prompt corpus.
Phase 0 is an initializer for pure replacement, not a sufficient final training
objective. It should be skipped for the first residual experiment because CLIP
already supplies the correct starting context.

### Frozen-U-Net Phase 1: pure ELLA-TSC

- Dataset: all 9,293 Unsplash records from the production Parquet.
- Caption mix: long caption only.
- Resolution: no crop; 1024-edge letterbox/aspect buckets with masked loss.
- Batch: 1.
- Steps: 9,293, one pass, about 133.4 minutes.
- Optimizer LR: constant `1e-4`.
- Trainable module: connector only; U-Net, Gemma, CLIP and VAE frozen.
- Final diffusion loss: `0.399157`.
- Final prompt sensitivity: `0.0626821`.
- Final suffix sensitivity mean: `0.283471`.
- Final delta cosine: `-0.275995`.
- Strict connector reload and fresh finite U-Net forward passed.
- W&B: <https://wandb.ai/mistic-jedi/gemma3-sd-pure-ella-unsplash-all-p1/runs/bkdasj0g>

Artifacts:

- `output_unsplash_all_p1/pure_ella_connector_L77.pt`
- `output_unsplash_all_p1/ella_connector_frozen_unet.pt`
- `output_unsplash_all_p1/ella_step_*.png`
- `logs/all_prompts_then_unsplash/phase1.log`

The minimum single-sample loss (`0.000881`) is not a useful checkpoint-selection
metric. No intermediate weights were saved, only intermediate image grids. A
good step-3000 cat grid followed by poor later grids demonstrates drift but
cannot be recovered as a model checkpoint.

Interpretation: this run was insufficiently controlled and undertrained, not a
fair proof that the architecture can never work. Constant `1e-4` alone is not
the defect; ELLA used it for SD1.5. The damaging combination was batch 1, only
9,293 updates, unrestricted replacement of CLIP conditioning, no preservation
constraint, and no useful intermediate checkpoints.

### Visual comparison

Primary comparisons:

- `output_p0_p1_post_training_comparison/standard_p0_p1_clip.png`
- `output_p0_p1_post_training_comparison/extended_p0_p1_clip.png`
- `output_p0_p1_extended_settings_comparison/extended_settings_p0_p1_clip.png`

The extended comparison uses StyleJourney v10 with the requested A1111-style
positive/negative prompts, seeds, samplers, CFG 7, and 768x960 output. Native
CLIP retained Darth Vader, goddess, and Milky Way semantics. Phase 0 lost those
specific concepts; Phase 1 drifted further, including unrelated horses,
silhouettes, text/page structure, and noise. Two successful generic/portrait
outputs do not offset these regressions.

## Data and model inputs

Base checkpoint:

```text
/mnt/e/stable-diffusion-webui/models/Stable-diffusion/stylejourney_v10.safetensors
```

Image-caption dataset currently approved for text-connector training:

```text
/mnt/e/data/unsplash-lite-gemma-captions/artifacts/production/unsplash_lite_gemma_captions_10000.parquet
```

It contains 9,293 records. The current training caption is present for all
records and spans roughly 113-321 Gemma tokens under the current audit; the
configured 336-token window is adequate. There are no useful short caption
variants in this artifact. `PicturesTraining` is excluded from connector
training because it does not have captions. Its EXIF-rich subset belongs to the
separate P3 camera-metadata work.

Prompt-only files are under `data/phase0_prompts/` and total about 96 MiB. They
are retained for reproducing the negative baseline, not required by the first
residual experiment.

## Residual connector specification

Build a new connector type rather than silently changing `ella_tsc` checkpoint
semantics. A name such as `clip_gemma_residual_tsc` makes artifact compatibility
explicit.

For the first falsifier:

- Width: 640.
- Blocks: 3 (2-4 is acceptable for an LR/capacity sweep).
- Heads: 8.
- Gemma input: learned mixture of the last four Gemma layers, up to 336 tokens.
- Queries: projected CLIP hidden states, not learned query tokens.
- Query projection: 768 -> 640.
- Gemma projection: 640 -> 640.
- Output projection: 640 -> 768, weight and bias exactly zero-initialized.
- Time conditioning: existing internal TSC AdaLN path.
- No learnable external gate or learnable `timestep_scale`.
- Inference residual strength: fixed scalar, default `1.0`.
- FP32 master weights/optimizer state with BF16 forward autocast.

Zero-initializing the output projection makes step zero exactly equivalent to
native CLIP while preserving normal gradients into that projection. Combining
it with a sigmoid gate initialized near zero would unnecessarily suppress
learning.

### True short-prompt bypass

The short-prompt guarantee must be control-flow identity, not multiplication by
a zero mask:

```python
context = clip_context.clone()
long_mask = prompt_exceeds_clip_window(...)
if long_mask.any():
    idx = long_mask.nonzero(as_tuple=True)[0]
    delta = connector(
        clip_context[idx], gemma_full[idx], timesteps[idx], gemma_mask[idx]
    )
    context[idx] = clip_context[idx] + residual_strength * delta
```

Do not run Gemma or the connector for an all-short batch. This avoids wasted
work and prevents `0 * NaN` from violating the guarantee. Add a unit test that
asserts `torch.equal(context, clip_context)` for a non-truncating batch. Exact
final-image equality additionally requires identical batching and deterministic
kernels.

The hard boundary is appropriate for the falsifier but creates a discontinuity
when one token makes a prompt exceed CLIP's window. Retain it for the first
experiment and design a smooth production policy only after Gemma adds measured
value.

### Prefix-null and residual-norm constraints

On long prompts, the connector can edit all 77 slots using both prefix and
suffix information. Architecture alone does not prevent it from overwriting a
named entity or style trigger that CLIP already handles. Train with the full
prompt and a CLIP-visible-prefix control:

```python
delta_full = connector(clip_context, gemma_full, timestep, full_mask)
delta_prefix = connector(clip_context, gemma_prefix, timestep, prefix_mask)

context_full = clip_context + delta_full
loss = (
    diffusion_loss(context_full)
    + lambda_prefix * normalized_square(delta_prefix, clip_context)
    + lambda_norm * normalized_square(delta_full, clip_context)
)
```

Normalize the penalties relative to CLIP RMS so their meaning is stable across
precision and timestep. Tune `lambda_prefix` and `lambda_norm`; they may decay
after the residual stabilizes but should retain a nonzero floor. Log each loss
term separately.

Conditioning dropout should be zero for this strict residual falsifier. Native
CLIP already supplies the unconditional branch, and short/unconditional prompts
must take the bypass. Revisit dropout only when relaxing that policy.

## First training protocol

1. Update `docs/PLAN.md` so the residual experiment is the current P1 path and
   pure replacement is historical/optional.
2. Implement the new connector, schema/versioned checkpoint metadata, true
   bypass, prefix extraction, and unit tests without modifying old checkpoint
   semantics.
3. Add paired residual-vs-CLIP diffusion evaluation using identical image,
   latent, noise, and timestep.
4. Split images before generating caption variants. Group by `photo_id`, exact
   hash, and perceptual hash/source identity to prevent resized duplicates from
   crossing train and validation.
5. Screen learning rates `1e-5`, `3e-5`, and `1e-4`. Treat `2e-5` only as another
   candidate, not a known optimum.
6. Use long captions for residual training. Short prompts contribute zero
   residual gradient under the hard bypass and belong in the regression suite.
7. Profile physical batch size, then use gradient accumulation for effective
   batch at least 8. Save weight checkpoints every 250-500 optimizer steps.
8. For fast architecture screening, use no-crop/full-frame 512-class buckets.
   Confirm the winner with the 1024-edge buckets before accepting it. The 512
   screen is a compute decision, not permission to crop or deform images.
9. Skip prompt-only Phase 0 and do not warm-start from the failed pure connector
   in the first residual run.
10. Freeze CLIP, Gemma, VAE, U-Net, camera conditioner, and SaRA weights. Train
    only the residual connector.

## Required evaluation

Held-out noise-prediction loss is useful but is not the primary quality gate.
The first experiment must report all of the following:

- Paired residual-vs-native-CLIP diffusion loss with identical latent, noise,
  and timestep.
- Per-sample win rate, paired mean/median difference, bootstrap confidence
  interval, and a sign or paired test rather than only aggregate means.
- Full-Gemma vs. CLIP-visible-prefix-Gemma residual comparison with identical
  CLIP context.
- Suffix counterfactual sensitivity for composition, attributes, spatial
  relations, counting, colors, and scene content.
- Prefix named-entity and community-model trigger-word preservation.
- Dense compositional image evaluation with multiple seeds.
- Native CLIP baseline at identical seed, sampler, CFG, and resolution.
- CFG 4, 5, and 7 to distinguish conditioning defects from guidance
  amplification.
- Blind human preference or a relation-capable VLM alignment score.

Do not require rare named entities introduced only in the suffix to pass the
first architecture gate. Gemma-3-270M may not supply CLIP-grade visual grounding
for them. Prefix entities remain a hard no-regression criterion because CLIP
already sees them.

Log `RMS(delta) / RMS(clip_context)` by timestep bucket from step zero. Use the
curve as an early drift alarm. Provisional review thresholds were `>0.25` for a
warning and `>0.5` for stop-and-inspect; calibrate and pre-register the actual
threshold before comparing runs rather than changing it after seeing results.

## Go/no-go decision

Continue the residual path only if it shows a paired held-out improvement and
measurable full-vs-prefix suffix use, improves dense compositional following,
and preserves short prompts plus prefix entities/style triggers against native
CLIP. Pixel differences or suffix sensitivity alone are not success.

If the residual is ignored, strengthen suffix-identifying data or constraints
before adding U-Net capacity. If useful suffix information appears but the 77
slots saturate, test a separately zero-gated cross-attention branch. If dual
conditioning succeeds and eliminating CLIP later becomes important, distill
the successful dual model into a pure Gemma connector using paired U-Net
outputs/diffusion targets rather than CLIP embedding imitation alone.

## Deferred work

- **P2 SaRA:** implemented but do not combine it with the first residual
  falsifier; establish connector value with a frozen U-Net first.
- **P3 camera metadata:** centered and split-head baseline code exists. Real P3
  training remains blocked on the versioned offline manifest, trusted FOV label
  provenance, null-label controls, and camera/lens holdouts. Do not use the
  quarantined raw-focal config as evidence of physical FOV control.
- **P4 U-Net/DiT capacity:** evidence-gated only after P1-P3.

## Reference points

ELLA trained its final connector through diffusion loss on about 34M image-text
pairs, with about 140k ablation steps and 280k main steps. Its tested text
encoders started around 1.1B/1.2B parameters, much larger than Gemma-3-270M.
Those numbers are context, not directly comparable sample-efficiency targets.

- Paper: <https://arxiv.org/abs/2403.05135>
- Official implementation and community-model CLIP note:
  <https://github.com/TencentQQGYLab/ELLA#3-ellaclip-for-community-models>
