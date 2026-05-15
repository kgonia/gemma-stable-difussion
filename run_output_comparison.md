# Gemma→SD semantic adapter run-output comparison

Compared notebooks:

- `gemma3_sd_colab (overfit).ipynb`
- `gemma3_sd_colab (diagnostic).ipynb`
- `gemma3_sd_colab (short_train).ipynb`
- `gemma3_sd_colab (full_train).ipynb`

Raw extracted notebook-output dump:

- `run_output_comparison_raw.txt`

Extracted embedded diagnostic images:

- `run_output_images/diagnostic_cell28_out12_19693cfb.png` — CLIP teacher baseline
- `run_output_images/diagnostic_cell28_out14_7f480244.png` — semantic adapter generic controls

## Executive comparison

| Run | Mode | Executed through | Error captured? | Main evidence |
|---|---|---:|---:|---|
| overfit | `overfit_train` | cell 31 / final reload proof | No | 64-sample geometry loss improved to `best_loss≈0.844`, checkpoint saved; but pretrain exact-overfit reference/fixed loss skipped before dataset existed and cell 5.1 output is hidden in saved notebook. |
| diagnostic | `diagnostic` | cell 31 / final reload proof | No | CLIP teacher grid follows prompts; untrained semantic adapter grid fails all prompts. Diagnostics show student/teacher delta cosine negative and huge delta norm ratio. |
| short_train | `short_train` | cell 31 / final reload proof | Yes, 5.3 OOM | 100 optimizer steps on 6k sample stream. Geometry improved only modestly (`best_loss≈1.413`); posttrain delta cosine `0.310`; sensitivity `0.0746`; 5.3 OOM was autograd leakage, not model failure. |
| full_train | `full_train` | cell 31 / final reload proof | Yes, 5.3 OOM | 3000 optimizer steps / 2 epochs over 6k sample stream. Geometry objective improved strongly (`best_loss≈0.848`), but semantic diagnostics are mixed: delta cosine lower than short (`0.260` vs `0.310`) and sensitivity lower (`0.0556` vs `0.0746`), while delta norm ratio improved (`4.21` vs `7.14`). |

## Common pre-training diagnostics

All runs used:

- `CONDITIONING_ARCH = gemma_clip_semantic_adapter`
- StyleJourney checkpoint from `/content/drive/MyDrive/model/stylejourney_v10.safetensors`
- CLIP teacher text-delta relative: `0.024562`
- CLIP teacher prompt sensitivity mean: about `0.02917`
- Gemma layer discriminability:
  - layer 4 mean pairwise cos `0.9827`, token_var `0.1499`
  - layer 8 mean pairwise cos `0.9965`, token_var `0.0308`
  - layer 12 mean pairwise cos `0.9908`, token_var `0.0276`
  - layer 16 mean pairwise cos `0.9637`, token_var `0.0759`
  - layer 18 mean pairwise cos `0.6935`, token_var `0.1755`

Layer 18 is the only swept layer with materially non-collapsed prompt separation. Earlier layers are very high-cosine/collapsed across prompts.

## Diagnostic run visual comparison

Diagnostic notebook embedded two grids.

### CLIP teacher baseline

File:

- `run_output_images/diagnostic_cell28_out12_19693cfb.png`

Vision inspection:

- Cat/windowsill prompt: clear cat sitting on a windowsill looking out.
- Watercolor mountain lake: clear watercolor-style mountain lake landscape.
- Cyberpunk alley: clear neon-lit urban alley at night.

Conclusion: teacher path is working and the validation prompts are reasonable.

### Semantic adapter generic controls

File:

- `run_output_images/diagnostic_cell28_out14_7f480244.png`

Vision inspection:

- Cat/windowsill prompt: no cat, no recognizable windowsill.
- Mountain lake prompt: generated a human portrait; no lake/mountains.
- Cyberpunk alley prompt: abstract glowing wall/panel; no alley/cyberpunk city structure.

Conclusion: semantic adapter student has prompt sensitivity but not semantic alignment before training. This is not a small quality issue; the object/scene nouns are absent.

## Short vs full training metrics

### Short train, 100 optimizer steps

Config/evidence:

```text
max_samples=6000
Semantic adapter optimizer-step limit: 100
Phase A semantic adapter complete: steps=100, best_loss=1.4127966165542603
semantic step 100: total=1.45649 mse=1.04943 cos=0.51629 pooled_cos=0.69842 retrieval=0.750 collapse(pred/clip)=0.782/0.470 norm_ratio=1.528
[post_semantic] student/teacher delta cosine = 0.310035
[post_semantic] student delta norm = 27.2755, teacher delta norm = 3.8207
[post_semantic] delta norm ratio = 7.1389
Semantic-adapter post-train sensitivity: 0.074582
```

### Full train, 3000 optimizer steps

Config/evidence:

```text
max_samples=6000
Semantic adapter optimizer-step limit: None
Phase A semantic adapter complete: steps=3000, best_loss=0.8476237654685974
semantic step 3000: total=1.05892 mse=0.82448 cos=0.40530 pooled_cos=0.72029 retrieval=1.000 collapse(pred/clip)=0.537/0.325 norm_ratio=1.304
[post_semantic] student/teacher delta cosine = 0.260398
[post_semantic] student delta norm = 11.2650, teacher delta norm = 2.6765
[post_semantic] delta norm ratio = 4.2088
Semantic-adapter post-train sensitivity: 0.055579
```

### Interpretation

Longer training clearly improves the supervised CLIP-state geometry objective:

- best loss: `1.413 → 0.848`
- final MSE: `1.049 → 0.824`
- retrieval: `0.75 → 1.0`
- collapse(pred/clip): `0.782/0.470 → 0.537/0.325`
- delta norm ratio: `7.14 → 4.21`

But it does **not** clearly improve the prompt-direction diagnostics:

- student/teacher delta cosine got worse: `0.310 → 0.260`
- semantic-adapter prompt sensitivity got lower: `0.0746 → 0.0556`
- pooled cosine at the final logged step is only modestly better than short (`0.698 → 0.720`), despite much longer training.

So the adapter behaves more CLIP-like geometrically with the longer run, but there is no strong evidence from the saved logs that it behaves more semantically useful for generation.

## 5.3 failure interpretation

Both `short_train` and `full_train` fail at the same final proof line:

```text
Generating reloaded semantic adapter: a cat sitting on a windowsill looking outside
Proof reloaded semantic: 37%|███▋| 11/30
OutOfMemoryError: CUDA out of memory
38.87 GiB allocated by PyTorch on a 39.49 GiB GPU
pred = unet(inp, t_step, encoder_hidden_states=ctx).sample
```

This is not evidence that the model needs more VRAM or that the full run is too large. It is a proof-loop bug: the final denoising loop was missing `torch.inference_mode()`, so autograd retained one UNet graph per scheduler step. Commit `0d24b2f` fixes this in the tracked notebook.

## Current decision

Based on the logs, longer training is **not a decisive win**.

What improved:

- representation/geometry loss;
- retrieval on small logged batches;
- overlarge student delta norm moved closer to teacher scale.

What did not improve enough:

- prompt-direction alignment;
- prompt sensitivity;
- visible final generated proof, because 5.3 OOMed before saving the reloaded proof grid.

I would not start whole-UNet finetuning solely from these logs. The next useful action is to rerun only patched 5.3 from the full checkpoint and inspect the saved reloaded semantic grid. If the full checkpoint grid still misses the nouns/scenes, more geometry-only adapter training is probably saturating the wrong objective; the next change should be objective/stage design, not whole-UNet full finetune.

## Implication for next decision

Do not run whole-UNet full finetune yet based on these outputs. The current bottleneck still appears to be the text-conditioning semantic bridge, not UNet capacity.

Better next check before broad UNet training:

- Rerun patched 5.3 only from the saved full-train adapter checkpoint.
- Save and inspect `samples_reloaded_semantic_adapter.png`.
- If possible, unhide/save cell 5.1 grids and exact overfit comparison outputs.
- Compare exact training-caption grid, CLIP teacher grid, semantic adapter grid, and posttrain delta alignment.

Only consider UNet adaptation after exact-caption semantic adapter generation beats the untrained diagnostic failure and the delta direction improves materially.
