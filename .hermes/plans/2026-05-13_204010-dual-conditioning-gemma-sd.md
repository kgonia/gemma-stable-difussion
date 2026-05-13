# Dual-conditioning Gemma-SD migration plan

Date: 2026-05-13
Repo: `/home/krz/PycharmProjects/gemma-stable-difussion`
Notebook: `gemma3_sd_colab.ipynb`

## Verdict

Plan is directionally right.

Current direct baked CLIP→Gemma init should stop as mainline. Runtime evidence says it collapsed:

- calibration train R²: `0.0000`
- calibration val R²: `-0.0233`
- validation cosine: `0.3555`
- `W norm: 0.0000`
- `b norm: 10.6670`
- Gemma hidden stats: `std=7.1325`, `max=43.25`
- warmup/training loss finite around `0.097–0.098`, but quality poor

Interpretation: model learned denoising prior, not useful text conditioning. Static linear CLIP/Gemma geometry is not enough.

Use CLIP as scaffold/teacher, not as embedding target. Add native Gemma attention branch beside original CLIP attention. Train Gemma branch while CLIP stabilizes. Force Gemma self-sufficiency with dropout/distillation. Anneal CLIP away. Prune CLIP. Continue Gemma-only.

## Non-negotiable architecture choice

Preferred implementation:

```text
hidden/image states
    ↓ shared to_q
CLIP hidden [B,77,768]  → original attn2 K/V/to_out path → clip_out
Gemma hidden [B,T,640]  → native Gemma K/V path          → gemma_out
    ↓
out = clip_weight * clip_out + gemma_weight * gemma_out
    ↓ residual add
```

Do not make main plan `Gemma → Linear(640→768) → original CLIP attn2`, except maybe as a disposable fast baseline. That is adapter-to-CLIP-space and contradicts target philosophy.

Native final target after pruning:

```text
Gemma hidden [B,T,640]
    ↓ LayerNorm/RMSNorm
native to_k/to_v: 640 → inner_dim
UNet cross-attention
```

## Key design details

### Gemma normalization

Required. Current Gemma last hidden has std ~7.13 and max ~43.25. Add per-block or shared:

```python
gemma_norm = nn.LayerNorm(640)
```

Apply before Gemma K/V. Also test middle Gemma layers, not only final layer.

Candidate hidden layers to sweep:

```text
last, middle-upper, maybe layer -2/-4
```

Metrics:

- hidden mean/std/max
- finite check
- prompt sensitivity in UNet prediction
- image quality after small training

### Gates / weights

Use explicit global schedule first, not learned-only gates.

Start:

```text
clip_weight = 1.0
gemma_weight = 0.0 or 0.05
```

Then train with scheduled weights. Per-block learned gates can exist, but global schedule controls phase.

### Prevent Gemma free-riding

If CLIP always present, Gemma branch may learn nothing. Must include:

1. CLIP dropout during dual training:

```text
randomly set clip_weight = 0 on some batches
```

2. Gemma-only teacher distillation:

```python
teacher_pred = unet_dual_or_clip_only(...).detach()
student_pred = unet_gemma_only(...)
loss_teacher = mse(student_pred, teacher_pred)
```

3. Evaluation always includes Gemma-only pass.

### Distillation losses

Main:

```python
loss_diffusion = mse(student_or_dual_pred, true_noise)
```

Transition:

```python
loss_teacher = mse(gemma_only_pred, clip_or_dual_teacher_pred.detach())
loss = loss_diffusion + lambda_teacher * loss_teacher
```

Optional later:

```python
loss_attn = KL/MSE(gemma_attn_map, clip_attn_map.detach())
```

Do not start with attention-map loss if it complicates implementation; it requires custom attention processor returning probabilities and may break flash attention. Start with epsilon prediction distillation.

### CFG

CFG must be defined for both branches:

```text
cond:   CLIP(prompt), Gemma(prompt)
uncond: CLIP(""),     Gemma("")
```

During anneal, guidance behavior changes. Track quality at CFG values:

```text
3.0, 5.0, 7.5
```

Do not assume SD1.5 `7.5` remains optimal for Gemma-only.

### Sequence length

Start with Gemma length 77 to isolate variable.

Only after Gemma-only works at 77, introduce longer prompts:

```text
Phase 5A: T=77
Phase 5B: T=128
Phase 5C: T=256 if useful
```

Longer prompt switch must be its own experiment because UNet attention softmax distribution changes with token count.

## sd-scripts stance

Use `kohya-ss/sd-scripts` for dependency/install/reference only at first.

Do not try to deeply fork sd-scripts immediately. It hardcodes CLIP strategy/tokenizer/text encoder assumptions. Dual encoder + custom UNet attention is custom Diffusers training loop territory.

Later, once architecture proven, decide whether to upstream/fork sd-scripts.

## Phase plan

### Phase A — freeze current baseline

Goal: preserve current working-but-poor notebook.

Actions:

1. Commit current runtime notebook as evidence branch/checkpoint.
2. Tag current state:
   - `v0.1-baked-gemma-direct-runtime`
3. Save generated bad examples if useful.
4. Record calibration-collapse evidence in markdown cell.

Gate:

- `git status` clean after commit/tag.

### Phase B — dual-attention smoke prototype, one block or minimal UNet patch

Goal: prove architecture compiles and forward/backward finite before rewriting whole notebook.

Implement small isolated cell/module:

- keep original SD1.5 `attn2`
- add parallel Gemma K/V branch
- use shared query and output projection where possible
- apply Gemma LayerNorm
- combine outputs with scalar weights

Minimum tests:

1. CLIP-only equivalence:

```text
clip_weight=1, gemma_weight=0 should match original output within tiny tolerance
```

2. Gemma branch finite:

```text
gemma_weight=1, clip_weight=0 forward/backward finite
```

3. gradients exist only for intended Gemma params.

Gate:

- no NaN/Inf
- CLIP-only path unchanged
- memory measured

### Phase C — full UNet dual branch

Goal: replace every cross-attn block with dual-capable cross-attn while preserving original CLIP path.

Trainable initially:

```text
Gemma norm
Gemma to_k/to_v
optional Gemma to_out if separate
global/per-block gates
```

Frozen:

```text
VAE
Gemma
CLIP
original CLIP attention path
most UNet
```

Gate:

- count modified cross-attn blocks = 16
- no original CLIP K/V destroyed
- CLIP-only generation reproduces SD1.5 baseline
- checkpoint reload smoke test passes

### Phase D — Phase 1 training: Gemma branch warmup under CLIP scaffold

Settings:

```text
clip_weight=1.0
gemma_weight=0.05→0.1
CLIP dropout maybe 10–25%
lambda_teacher=0.5 initially
LR around 1e-5 to 5e-5 for Gemma branch
fp32 trainable params, frozen encoders
```

Training loss:

```python
loss = loss_diffusion_dual + lambda_teacher * mse(gemma_only_pred, clip_teacher_pred.detach())
```

Diagnostics every checkpoint:

- dual loss
- Gemma-only loss
- CLIP-only teacher loss/reference
- prompt sensitivity relative diff
- generated grid: CLIP-only / dual / Gemma-only
- gate/weight schedule values
- Gemma K/V grad norms

Gate:

- Gemma-only prediction changes by prompt
- Gemma-only generated images not pure mush
- CLIP dropout batches improve over time

### Phase E — anneal CLIP away

Schedule conservative:

```text
0–20%:   clip=1.0,  gemma=0.1
20–40%:  clip=0.75, gemma=0.25
40–60%:  clip=0.5,  gemma=0.5
60–80%:  clip=0.25, gemma=0.75
80–100%: clip=0.0,  gemma=1.0
```

But do not advance schedule blindly. Require gate per stage:

- loss not >3x prior stage
- prompt sensitivity positive
- visual grid acceptable
- Gemma-only not worse than previous checkpoint

If cliff appears, pause anneal; more dropout/distill at previous level.

### Phase F — structural prune to Gemma-only

Goal: remove CLIP from model graph.

Actions:

1. Create pruning function:

```text
DualCrossAttention → GemmaOnlyCrossAttention
```

2. Delete CLIP modules from inference path.
3. Save Gemma-only checkpoint with architecture metadata.
4. Reload in fresh runtime with no CLIP model loaded.

Gate:

- fresh runtime can generate with Gemma tokenizer/model only
- `strict=True` reload or explicit custom state validation passes
- no CLIP tensors required in checkpoint/inference

### Phase G — Gemma-only longer fine-tuning

Start at length 77. Then extend.

Trainable staging:

```text
G1: Gemma K/V + norms + to_out
G2: add attn2 Q/out LoRA
G3: add attn1 + FF LoRA
G4: optional ResNet LoRA / partial full fine-tune
G5: optional top Gemma layers tiny LR
```

CFG native to Gemma:

- caption dropout 10%
- empty prompt baseline first
- learned null context later if needed

Gate:

- prompt sensitivity stable
- same seed different prompts -> different content
- same prompt different seeds -> diversity
- longer prompts improve detail, not degrade all images

## Notebook migration tasks

### Task 1: Evidence/cleanup plan cell

Rewrite top markdown to say:

- current Path C direct bake collapsed
- new path is dual native cross-attn
- CLIP is training scaffold, not final inference dependency

### Task 2: keep CLIP loader

Current cells 10–12 load CLIP temporarily and delete it. Rewrite:

- CLIP tokenizer/model persistent for phases B–E
- no ridge W/b solve
- no deletion until prune phase

Delete/bypass old calibration cells 11–12 from mainline.

### Task 3: dual attention module cell

Add:

- `DualCrossAttention` or custom processor/block wrapper
- Gemma LayerNorm
- separate Gemma K/V
- schedule weights
- mode switch: `clip_only`, `dual`, `gemma_only`

Need exact Diffusers integration chosen after small source inspection.

### Task 4: UNet surgery cell

Replace all 16 cross-attn modules with dual modules.

Verify:

- count = 16
- original CLIP path weights unchanged
- Gemma path initialized std=0.02 or safe Kaiming
- all params finite
- trainable param count printed

### Task 5: forward/backward verification cell

Add checks:

- CLIP hidden stats
- Gemma hidden stats before/after norm
- dual output finite
- CLIP-only equivalence vs original if feasible
- Gemma-only forward finite
- one backward step finite
- memory summary

### Task 6: dataset/training rewrite

Use existing streaming dataset but recreate per epoch.

Encoding per batch:

- CLIP tokenizer/model no grad
- Gemma tokenizer/model no grad
- caption dropout consistent for both branches

Training modes:

- warmup dual scaffold
- CLIP dropout/distill
- anneal schedule

### Task 7: diagnostics cell

Permanent diagnostics:

- `prompt_sensitivity()` for Gemma-only and dual
- eval grid rows prompts, cols seeds
- ablations: CLIP-only / dual / Gemma-only
- wandb images + scalar logs

### Task 8: save/reload cells

Artifacts:

```text
checkpoint_dual_phase.pt
checkpoint_gemma_only_pruned.pt
architecture_meta.json-like dict inside .pt
```

Reload smoke tests:

- dual checkpoint reload with CLIP+Gemma
- pruned checkpoint reload with Gemma only
- generation smoke test

## Autonomous agent/review process

Use small chunks. No whole-notebook edit by Claude/Gemini; notebooks are bad for string patch tools.

### Implementation loop per chunk

1. Hermes orchestrator creates/updates notebook JSON programmatically.
2. Python verifier checks syntax, ordering, stale refs, embedded-newline source integrity.
3. Claude Code read-only review on selected changed cells, max 200 lines.
4. Gemini adversarial review on architecture/risk for selected cells.
5. Fix only concrete issues.
6. Commit.

### Suggested chunks

1. Plan + preserve current runtime evidence.
2. CLIP persistent loader + delete calibration path.
3. Minimal dual attention prototype cell.
4. Full UNet surgery cell.
5. Forward/backward verification cell.
6. Training loop warmup + distill.
7. Anneal schedule + diagnostics.
8. Prune/save/reload.
9. Final notebook verification + clean outputs if shipping.

## Verification script requirements

Before every commit:

- JSON loads
- all code cells parse after masking `!`/`%`
- no embedded newlines inside source elements
- gated models use explicit `token=`
- no stale `W`, `b`, `clip_gemma_calib`, `old_k_weight @ W` in mainline cells
- no hardcoded fp16 training paths unless intentional
- streaming dataset recreated each epoch
- outputs either intentionally kept as runtime evidence or cleared before shipping

## Risks

### Highest risks

1. Diffusers attention internals: dual branch needs careful integration.
2. Gemma free-riding: CLIP scaffold can hide useless Gemma branch.
3. CFG scale shift: Gemma-only CFG may need lower scale.
4. Checkpoint reloadability: custom architecture can orphan checkpoints.
5. Longer prompts: do only after 77-token Gemma-only works.

### Mitigations

- Build one-block/minimal smoke first.
- Require Gemma-only eval from first training phase.
- CLIP dropout + teacher distill from beginning.
- Save architecture code/metadata with checkpoints.
- Stage longer context separately.

## Decision

Proceed with dual native Gemma cross-attention plan, not direct W bake.

Do not implement additive `Gemma→768→CLIP hidden` as main path unless user explicitly wants a fast disposable bridge. It is useful as a baseline but not aligned with final Gemma-native UNet.

Next concrete action after approval:

```text
Task 1: commit/tag current runtime notebook evidence, then create branch dual-conditioning.
Task 2: create minimal dual-attention prototype cell and verifier, no full training yet.
```
