# Overfit Train Mode + CLIP/Gemma Curriculum Implementation Plan

> **Superseded direction note:** after inspecting ELLA (`docs/2403.05135v1.pdf`) and TencentQQGYLab/ELLA, prioritize the newer plan `.hermes/plans/2026-05-14-ella-adjusted-gemma-connector-plan.md`. Keep `overfit_train` and delta diagnostics from this plan, but defer CLIP/Gemma token-count curriculum.

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add an `overfit_train` run mode to `gemma3_sd_colab.ipynb` and prepare a minimal, safe curriculum path for CLIP-anchor + Gemma-branch training without restructuring the whole notebook.

**Architecture:** Keep the current dual-branch design: original CLIP cross-attention branch plus native Gemma cross-attention branch. Add a config-only overfit mode first. Then add small helper functions for scheduled Gemma token budget and CLIP/Gemma scale curriculum, without raw CLIP/Gemma token concatenation.

**Tech Stack:** Jupyter/Colab `.ipynb`, PyTorch, Diffusers SD1.5 UNet, frozen CLIP teacher, frozen Gemma encoder, W&B logging.

---

## Socratic framing

### Q1. What is the smallest experiment that can falsify “the Gemma bridge can learn”?

A tiny repeated-subset overfit. If 32–64 repeated samples cannot make the Gemma path visibly more prompt-responsive, full-dataset or whole-UNet training is premature.

### Q2. Can this be done without changing notebook structure?

Mostly yes. Existing loops already recreate the streaming DataLoader per epoch. If `SHUFFLE_STREAMING = False`, each epoch should revisit the same first `MAX_SAMPLES_*` examples. So `overfit_train` can be introduced as config-only defaults first.

### Q3. What must `overfit_train` prove?

Not just lower loss. It must improve:

- Gemma-only prompt sensitivity,
- student/teacher delta cosine,
- qualitative validation grids,
- final reloaded-pruned Gemma-only samples.

### Q4. Should we add Gemma→CLIP adapter now?

Not as the next notebook change. It is a valid baseline, but the user wants to assess a native dual-branch handoff curriculum. The dual-branch curriculum preserves Gemma-native conditioning and avoids reducing the whole project to “Gemma pretending to be CLIP.”

### Q5. Is token-count curriculum alone enough?

No. Token count is not influence. CLIP tokens are already meaningful to SD1.5; Gemma tokens are initially alien. Use token budget plus explicit scale/gate schedule and teacher/student delta diagnostics.

### Q6. Should raw CLIP and Gemma tokens be concatenated into one stream?

No. Current notebook already has the better architecture: separate CLIP and Gemma branches whose outputs are combined by scale. Preserve that.

### Q7. What from the external reviews is correct?

Correct:

- CLIP baseline is strong, so base SD pipeline is fine.
- Gemma-only reloaded proof is technically strong.
- Gemma branch is alive but semantically wrong.
- Prompt sensitivity is only a liveness signal, not semantic alignment.
- Add student/teacher delta cosine and norm ratio.
- Current Phase B trainable count must be reported after `unfreeze_gemma_to_out`, not before LoRA wrapping only.

Needs caution:

- Per-block attention-output distillation is promising but is a larger structural change than `overfit_train`; defer until after the small falsifier.
- Token-ratio curriculum should not drop CLIP tokens early; keep 77 CLIP tokens and anneal scale first.

---

## Acceptance criteria

1. Notebook has `RUN_MODE = "overfit_train"` as a selectable option.
2. `overfit_train` defaults repeat a tiny stable subset:
   - `MAX_SAMPLES_WARMUP = 64`
   - `MAX_SAMPLES_PHASEB = 64`
   - `SHUFFLE_STREAMING = False`
3. Phase A overfit can run without Phase B:
   - `FULLRANK_EPOCHS = 40`
   - `PHASEB_EPOCHS = 0`
   - `PHASE_A_MAX_OPT_STEPS = 300`
4. W&B config logs `overfit_train`, `paired_unet_batch_size`, `shuffle_streaming=False`, effective target steps, and curriculum fields.
5. Validation logs include:
   - CLIP-only sensitivity,
   - mixed sensitivity if curriculum is enabled,
   - Gemma-only sensitivity,
   - student/teacher delta cosine,
   - student/teacher delta norm ratio.
6. No raw concatenation of `[CLIP tokens][Gemma tokens]` into a single context stream.
7. Final proof still uses freshly reloaded pruned Gemma-only checkpoint.

---

## Task 1: Add `overfit_train` run-mode defaults

**Objective:** Add an overfit mode using only config-cell changes.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, config cell containing `RUN_MODE = "short_train"`.

**Implementation details:**

Change selectable modes from:

```python
RUN_MODE = "short_train"  # @param ["diagnostic", "short_train", "full_train"]
RUN_TRAINING = RUN_MODE in {"short_train", "full_train"}
```

to:

```python
RUN_MODE = "overfit_train"  # @param ["diagnostic", "overfit_train", "short_train", "full_train"]
RUN_TRAINING = RUN_MODE in {"overfit_train", "short_train", "full_train"}
```

Replace scalar ternaries with explicit mode table:

```python
if RUN_MODE == "diagnostic":
    MAX_SAMPLES_WARMUP = 128
    MAX_SAMPLES_PHASEB = 128
    FULLRANK_EPOCHS = 0
    PHASEB_EPOCHS = 0
    PHASE_A_MAX_OPT_STEPS = 0
    PHASE_B_MAX_OPT_STEPS = 0
    TEACHER_DECAY_STEPS = 200
    VALIDATION_EVERY_OPT_STEPS = 0
    SHUFFLE_STREAMING = True
elif RUN_MODE == "overfit_train":
    MAX_SAMPLES_WARMUP = 64
    MAX_SAMPLES_PHASEB = 64
    FULLRANK_EPOCHS = 40
    PHASEB_EPOCHS = 0
    PHASE_A_MAX_OPT_STEPS = 300
    PHASE_B_MAX_OPT_STEPS = 0
    TEACHER_DECAY_STEPS = 300
    VALIDATION_EVERY_OPT_STEPS = 50
    SHUFFLE_STREAMING = False
elif RUN_MODE == "short_train":
    MAX_SAMPLES_WARMUP = 6_000
    MAX_SAMPLES_PHASEB = 6_000
    FULLRANK_EPOCHS = 1
    PHASEB_EPOCHS = 1
    PHASE_A_MAX_OPT_STEPS = 100
    PHASE_B_MAX_OPT_STEPS = 200
    TEACHER_DECAY_STEPS = 200
    VALIDATION_EVERY_OPT_STEPS = 50
    SHUFFLE_STREAMING = True
elif RUN_MODE == "full_train":
    MAX_SAMPLES_WARMUP = 6_000
    MAX_SAMPLES_PHASEB = 6_000
    FULLRANK_EPOCHS = 2
    PHASEB_EPOCHS = 1
    PHASE_A_MAX_OPT_STEPS = None
    PHASE_B_MAX_OPT_STEPS = None
    TEACHER_DECAY_STEPS = 1000
    VALIDATION_EVERY_OPT_STEPS = 250
    SHUFFLE_STREAMING = True
else:
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")
```

Keep:

```python
SHUFFLE_BUFFER = 10_000
```

but log that it is inactive when `SHUFFLE_STREAMING=False`.

**Verification:**

Run notebook verification script snippets:

- JSON loads.
- `RUN_MODE` option exists.
- W&B config includes overfit values.
- No stale `RUN_MODE in {"short_train", "full_train"}` remains except comments.

---

## Task 2: Add overfit-specific run notes and stop criteria

**Objective:** Make the notebook print clear overfit expectations so results are not misread as full training.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, config cell.

**Implementation details:**

After run config printing, add:

```python
if RUN_MODE == "overfit_train":
    print("OVERFIT MODE: repeating a tiny non-shuffled streaming subset.")
    print("Success criterion is not loss alone: Gemma-only semantic validation must improve.")
    print("If delta cosine stays <= 0 or final grids remain semantically wrong, do not scale.")
```

Add W&B config keys:

```python
"overfit_expected_repeated_subset": RUN_MODE == "overfit_train",
"overfit_success_gate": "loss down + delta cosine up + Gemma-only validation semantically improves",
```

**Verification:**

Static scan confirms these strings appear exactly once.

---

## Task 3: Add teacher/student delta alignment diagnostic

**Objective:** Measure whether Gemma text effect points in the same direction as CLIP teacher text effect.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, diagnostics cell near `compute_teacher_text_delta_baseline` and prompt sensitivity helpers.

**Implementation details:**

Add helper:

```python
@torch.no_grad()
def compute_teacher_student_delta_alignment(prompts, seed=321, timestep=500, clip_scale=0.0, gemma_scale=1.0, label="delta_alignment"):
    latents = torch.randn(len(prompts), unet.config.in_channels, 64, 64, device=device, dtype=unet_dtype, generator=torch.Generator(device=device).manual_seed(seed))
    t = torch.full((len(prompts),), int(timestep), device=device, dtype=torch.long)
    empty = [""] * len(prompts)

    cond_clip, cond_clip_mask = encode_clip_prompts(prompts)
    uncond_clip, uncond_clip_mask = encode_clip_prompts(empty)
    cond_gemma, cond_gemma_mask = encode_gemma_prompts(prompts)
    uncond_gemma, uncond_gemma_mask = encode_gemma_prompts(empty)

    noisy_pair = torch.cat([latents, latents], dim=0)
    t_pair = torch.cat([t, t], dim=0)
    clip_h_pair = torch.cat([cond_clip, uncond_clip], dim=0)
    clip_mask_pair = torch.cat([cond_clip_mask, uncond_clip_mask], dim=0)
    gemma_h_pair = torch.cat([cond_gemma, uncond_gemma], dim=0)
    gemma_mask_pair = torch.cat([cond_gemma_mask, uncond_gemma_mask], dim=0)

    set_dual_attention_context(unet, gemma_encoder_hidden_states=None, gemma_attention_mask=None, clip_scale=1.0, gemma_scale=0.0)
    teacher_pair = unet(noisy_pair, t_pair, encoder_hidden_states=clip_h_pair, encoder_attention_mask=clip_mask_pair).sample.float()
    teacher_cond, teacher_uncond = teacher_pair.chunk(2)
    teacher_delta = teacher_cond - teacher_uncond

    set_dual_attention_context(unet, gemma_encoder_hidden_states=gemma_h_pair, gemma_attention_mask=gemma_mask_pair, clip_scale=clip_scale, gemma_scale=gemma_scale)
    student_pair = unet(noisy_pair, t_pair, encoder_hidden_states=clip_h_pair, encoder_attention_mask=clip_mask_pair).sample.float()
    student_cond, student_uncond = student_pair.chunk(2)
    student_delta = student_cond - student_uncond

    td = teacher_delta.flatten(1)
    sd = student_delta.flatten(1)
    delta_cos = nn.functional.cosine_similarity(sd, td, dim=1).mean()
    teacher_norm = td.norm(dim=1).mean().clamp_min(1e-8)
    student_norm = sd.norm(dim=1).mean()
    norm_ratio = student_norm / teacher_norm
    print(f"[{label}] student/teacher delta cosine={delta_cos.item():.6f}, norm_ratio={norm_ratio.item():.6f}")
    wandb.log({f"{label}/delta_cosine": delta_cos.item(), f"{label}/delta_norm_ratio": norm_ratio.item()})
    return {"delta_cosine": float(delta_cos.item()), "delta_norm_ratio": float(norm_ratio.item())}
```

Call it:

- before training after mandatory diagnostics,
- after Phase A final validation,
- after Phase B final validation if Phase B runs,
- before final save/proof if training ran.

**Verification:**

Static scan confirms function exists and is called at least pre-training and post-Phase-A.

---

## Task 4: Add optional token budget to Gemma encoder

**Objective:** Support Gemma token-count curriculum while preserving separate CLIP/Gemma branches.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, `encode_gemma_prompts` helper and config cell.

**Implementation details:**

Add config:

```python
ENABLE_CONTEXT_CURRICULUM = False
GEMMA_TOKEN_BUDGET = 100
```

Change helper signature:

```python
def encode_gemma_prompts(prompts, layer_index=None, return_all_layers=False, token_budget=None):
```

After tokenization and hidden selection, apply:

```python
budget = GEMMA_TOKEN_BUDGET if token_budget is None else token_budget
if budget is not None:
    budget = int(budget)
    hidden = hidden[:, :budget, :]
    attention_mask = tok.attention_mask[:, :budget]
else:
    attention_mask = tok.attention_mask
return hidden, attention_mask
```

For `return_all_layers=True`, either do not slice initially or slice all returned layers consistently:

```python
if return_all_layers:
    attention_mask = tok.attention_mask[:, :budget] if budget is not None else tok.attention_mask
    hidden_states = tuple(h[:, :budget, :].to(device=device, dtype=unet_dtype) for h in out.hidden_states) if budget is not None else ...
    return hidden_states, attention_mask
```

**Verification:**

- `hidden.shape[1] == attention_mask.shape[1]` for budgets 16, 32, 64, 100.
- Existing diagnostics still run with default budget.

---

## Task 5: Add simple curriculum schedule helper, initially disabled

**Objective:** Encode the reviewed schedule without making it the default until overfit mode is proven.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, config or helper cell.

**Implementation details:**

Add:

```python
CONTEXT_CURRICULUM = [
    {"until_step": 100, "clip_tokens": 77, "gemma_tokens": 16,  "clip_scale": 1.0,  "gemma_scale": 0.05, "teacher_lambda": 1.0},
    {"until_step": 200, "clip_tokens": 77, "gemma_tokens": 32,  "clip_scale": 1.0,  "gemma_scale": 0.15, "teacher_lambda": 1.0},
    {"until_step": 400, "clip_tokens": 77, "gemma_tokens": 64,  "clip_scale": 0.75, "gemma_scale": 0.35, "teacher_lambda": 0.75},
    {"until_step": 700, "clip_tokens": 77, "gemma_tokens": 100, "clip_scale": 0.5,  "gemma_scale": 0.75, "teacher_lambda": 0.5},
    {"until_step": 1000,"clip_tokens": 77, "gemma_tokens": 100, "clip_scale": 0.2,  "gemma_scale": 1.0,  "teacher_lambda": 0.25},
    {"until_step": 1200,"clip_tokens": 77, "gemma_tokens": 100, "clip_scale": 0.05, "gemma_scale": 1.0,  "teacher_lambda": 0.1},
    {"until_step": None,"clip_tokens": 77, "gemma_tokens": 100, "clip_scale": 0.0,  "gemma_scale": 1.0,  "teacher_lambda": 0.0},
]

def get_context_curriculum(step):
    if not ENABLE_CONTEXT_CURRICULUM:
        return {"clip_tokens": 77, "gemma_tokens": GEMMA_TOKEN_BUDGET, "clip_scale": 0.0, "gemma_scale": 1.0, "teacher_lambda": 1.0}
    for stage in CONTEXT_CURRICULUM:
        if stage["until_step"] is None or step < stage["until_step"]:
            return stage
    return CONTEXT_CURRICULUM[-1]
```

Important: keep `clip_tokens=77`; do not implement CLIP token dropping in this pass. Scale annealing handles CLIP removal.

**Verification:**

- Schedule returns expected stage at steps 0, 150, 500, 1300.
- W&B logs current stage when enabled.

---

## Task 6: Wire curriculum into Phase A/B carefully

**Objective:** Allow scheduled Gemma token budget and scales without changing the default current behavior.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, Phase A and Phase B cells.

**Implementation details:**

Where Gemma prompts are encoded inside loops, replace:

```python
cond_gemma_h, cond_gemma_mask = encode_gemma_prompts(captions)
uncond_gemma_h, uncond_gemma_mask = encode_gemma_prompts(empty_captions)
```

with:

```python
stage = get_context_curriculum(phase_a_opt_step)  # or optimizer_step in Phase B
cond_gemma_h, cond_gemma_mask = encode_gemma_prompts(captions, token_budget=stage["gemma_tokens"])
uncond_gemma_h, uncond_gemma_mask = encode_gemma_prompts(empty_captions, token_budget=stage["gemma_tokens"])
```

For student context, replace fixed scales:

```python
clip_scale=WARMUP_CLIP_SCALE, gemma_scale=WARMUP_GEMMA_SCALE
```

with:

```python
clip_scale=stage["clip_scale"] if ENABLE_CONTEXT_CURRICULUM else WARMUP_CLIP_SCALE,
gemma_scale=stage["gemma_scale"] if ENABLE_CONTEXT_CURRICULUM else WARMUP_GEMMA_SCALE,
```

For loss, multiply teacher loss by `teacher_lambda` when curriculum is enabled:

```python
teacher_lambda = stage["teacher_lambda"] if ENABLE_CONTEXT_CURRICULUM else 1.0
loss = teacher_lambda * loss_teacher + LAMBDA_TEXT_DELTA * loss_text_delta + LAMBDA_DIFFUSION * loss_diffusion
```

For default `overfit_train`, leave `ENABLE_CONTEXT_CURRICULUM=False` initially so the first falsifier is not confounded.

**Verification:**

- With `ENABLE_CONTEXT_CURRICULUM=False`, source behavior matches previous scale constants.
- With `ENABLE_CONTEXT_CURRICULUM=True`, logs include stage values.

---

## Task 7: Correct Phase B trainable-count reporting

**Objective:** Avoid misleading “0.04% LoRA” interpretation after `to_out` is unfrozen.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, Phase B cell.

**Implementation details:**

Ensure output after:

```python
unfreeze_gemma_to_out(unet)
trainable_params = [p for p in unet.parameters() if p.requires_grad]
print_trainable_summary(unet, label="phase_b_after_unfreeze_to_out")
```

is visible and W&B-logged. If `print_trainable_summary` only prints, add it returns a dict or separately log:

```python
wandb.log({"phase_b/trainable_params_after_to_out": sum(p.numel() for p in trainable_params)})
```

**Verification:**

- Static scan finds `phase_b_after_unfreeze_to_out`.
- Output no longer relies only on `phase_b_after_lora_wrap_before_to_out`.

---

## Task 8: Notebook verification

**Objective:** Prove the edited notebook is structurally safe before Colab execution.

**Files:**
- Verify: `gemma3_sd_colab.ipynb`.

**Checks:**

Run deterministic Python checks:

- JSON parses.
- No embedded multi-newline source elements.
- Pure Python cells compile, skipping shell/magic cells.
- No stale model ID suffixes.
- Gemma mask propagation remains intact.
- Final proof remains guarded by `RUN_TRAINING`.
- `overfit_train` appears in mode list and config dict.
- No raw concatenation of CLIP/Gemma tokens into one context tensor beyond existing cond/uncond batching.

**Verification command idea:**

```bash
python verify_notebook.py gemma3_sd_colab.ipynb
```

or run the `colab-notebook-verification` snippets via Hermes.

---

## Execution order

1. Implement Tasks 1–3 only.
2. Verify notebook.
3. Commit.
4. Run `overfit_train` Phase A only in Colab.
5. Inspect:
   - loss curve,
   - delta cosine/norm ratio,
   - Gemma-only validation grids,
   - final reloaded-pruned proof.
6. Only if overfit shows movement, implement Tasks 4–7.
7. If overfit fails, do not add curriculum complexity yet; investigate representation/objective alignment first.

---

## Decision gate after overfit_train

Proceed to curriculum only if at least two of these improve:

```text
student/teacher delta cosine: moves meaningfully positive and upward
Gemma-only sensitivity ratio: rises toward CLIP baseline
Gemma-only validation images: visibly less arbitrary and more prompt-related
training loss: falls without NaN/Inf
final reloaded-pruned proof: matches live Gemma-only path
```

Do not proceed if:

```text
delta cosine remains near zero/negative
Gemma-only samples remain facade/grid/wall-like for unrelated prompts
loss falls but semantic validation does not move
reloaded-pruned proof diverges from live graph
```

---

## Commit plan

Commit 1:

```bash
git add gemma3_sd_colab.ipynb
git commit -m "Add overfit training mode and delta diagnostics"
```

Commit 2, only after overfit evidence supports it:

```bash
git add gemma3_sd_colab.ipynb
git commit -m "Add optional CLIP-Gemma context curriculum"
```
