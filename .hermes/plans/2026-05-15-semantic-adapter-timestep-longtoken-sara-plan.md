# Gemma→SD Semantic Adapter, Timestep Residual, Long-Token Conditioning, and SaRA Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task after review. Do not run expensive Colab training during implementation unless explicitly requested.

**Goal:** Replace the failed diffusion-loss-only connector with a staged conditioning contract: first make Gemma produce CLIP/SD-readable semantic conditioning, then add timestep adaptation, then extend token capacity beyond 77 and train only the under-adapted UNet weights with SaRA-style sparse low-rank updates.

**Architecture:** Phase 1 learns a Gemma→CLIP-state semantic adapter that emits the exact SD1.5 CLIP contract `[B,77,768]` and is trained with Qwen-distillation-style geometry losses. Phase 1b adds a timestep-aware residual on top of those CLIP-like tokens, not instead of them. Phase 2 extends conditioning length using an ELLA/Perceiver long-token path and only then introduces UNet-side adaptation for longer token use. Phase 3 uses SaRA on selected UNet cross-attention/text-interface weights after earlier gates prove semantic overfit.

**Tech Stack:** `gemma3_sd_colab.ipynb`, PyTorch, Diffusers SD1.5/StyleJourney, frozen Gemma 270M, frozen CLIP teacher during training diagnostics, optional SaRA code from `sjtuplayer/SaRA` (`optim/adamw.py`, `optim/adamw2.py`).

---

## Why this plan replaces the current connector-only overfit

The current `ella_gemma_connector` overfit failed visually: exact training prompts did not reconstruct or approach training images. Diagnostics showed prompt movement but poor semantic alignment: high connector sensitivity, low-ish teacher/student delta cosine, and oversized student delta norm. This is the same failure mode the Qwen→Z-Image notebook was designed to avoid: do not train only through generator loss before proving the adapter matches the teacher conditioning contract.

The Qwen notebook inspiration:

- one shared `condition()` path for train/eval/visuals;
- adapter geometry loss before/alongside generator consistency;
- mask propagation into cross-attention;
- collapse/retrieval diagnostics;
- image-side validation as a gate, not a replacement for representation diagnostics.

For SD1.5/StyleJourney, the teacher conditioning contract is initially CLIP-like `[B,77,768]`. We should prove Gemma can synthesize that contract before asking the frozen UNet to interpret arbitrary `[B,64,768]` Perceiver latents.

---

## Source findings to preserve

### Existing project state

Main notebook:

```text
gemma3_sd_colab.ipynb
```

Current default path:

```python
RUN_MODE = "overfit_train"
CONDITIONING_ARCH = "ella_gemma_connector"
```

Current failure gates added:

```python
TRAIN_VAL_PROMPTS
OVERFIT_EVAL_BATCH
save_overfit_reference_grid()
compute_fixed_overfit_eval_loss()
```

Keep these gates. Extend them; do not remove them.

### Qwen distillation notebook inspiration

Inspected:

```text
/mnt/c/Users/krzgo/AndroidStudioProjects/litert-samples/docs/Train_Qwen3_1-7B_To_Match_Qwen3_4B_States.DistillKit.ipynb
```

Local extract:

```text
docs/qwen_zimage_distill_inspiration_extract.md
```

Relevant adapter pattern:

```python
class Adapter(nn.Module):
    def forward(self, student_hs, mask):
        q = self.q_proj(student_hs)
        kv = self.kv_proj(student_hs)
        key_padding_mask = ~mask.bool()
        for block in self.blocks:
            q = block(q, kv, key_padding_mask=key_padding_mask)
        out = self.proj_out(q)
        out = out.masked_fill(~mask[..., None].bool(), 0)
        return out
```

Relevant loss pattern:

```python
mse = ((LayerNorm(pred) - LayerNorm(target)) ** 2 * mask).sum() / mask.sum() / dim
norm = relative/log norm matching
cos = masked 1 - cosine(pred, target)
contrastive = in-batch retrieval CE/MSE over pooled states
```

### SaRA source findings

Inspected shallow clone:

```text
/tmp/SaRA
commit ea21192 Update the evaluation code
```

Readme usage:

```python
from optim import adamw
optimizer = adamw(model, threshold=2e-3)
```

Example command:

```bash
python3 finetune.py \
  --config=configs/Barbie.json \
  --output_dir=$path_to_save \
  --sd_version=1.5 \
  --threshold=2e-3 \
  --lr_scheduler=cosine \
  --progressive_iter=2500 \
  --lambda_rank=0.0005
```

Implementation notes from `/tmp/SaRA/optim/adamw.py`:

- trainable sparse parameters are selected where `param.abs() < threshold`;
- trainable vectors are spliced into the model with a custom autograd `Sparse` function;
- `progressive_iter` re-selects trainable parameters at a later step;
- `lambda_rank` adds nuclear-norm regularization on randomly selected large 2D matrices;
- `optimizer.save_params()` saves sparse tuned values plus threshold.

Important caveat: SaRA mutates model parameter objects. In our notebook, use it only in a later phase and isolate it to the UNet submodule or selected copied UNet block, not the full pipeline object, until reload/save is proven.

---

## Phase outline and gates

### Phase 0 — Keep current failure diagnostics, add representation preflight

**Purpose:** Make the notebook fail early if the semantic adapter cannot match CLIP states.

Must pass before any image/diffusion training:

1. Gemma hidden states finite.
2. CLIP hidden states finite.
3. Adapter output shape equals `[B,77,768]`.
4. Adapter→CLIP geometry loss decreases on exact overfit captions.
5. Adapter→CLIP retrieval accuracy improves above random.
6. Adapter collapse cosine is not worse than CLIP teacher collapse.
7. Norm ratio approaches 1.0.

Stop if these fail. Do not train UNet or SaRA.

### Phase 1 — Semantic adapter to CLIP contract `[B,77,768]`

**Architecture:**

```text
Gemma hidden [B,S,640] + Gemma mask
→ SemanticClipStateAdapter
→ CLIP-like states [B,77,768]
→ frozen original SD UNet cross-attention
```

Recommended adapter:

```python
class GemmaToClipSemanticAdapter(nn.Module):
    def __init__(
        self,
        gemma_dim=640,
        clip_dim=768,
        width=768,
        num_clip_tokens=77,
        layers=4,
        heads=8,
        ff_mult=4,
        dropout=0.0,
    ):
        self.input_proj = nn.Linear(gemma_dim, width)
        self.clip_queries = nn.Parameter(torch.randn(1, num_clip_tokens, width) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, num_clip_tokens, width) * 0.01)
        self.blocks = nn.ModuleList([...pre-norm cross-attn + FF...])
        self.final_norm = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, clip_dim)

    def forward(self, gemma_h, gemma_mask):
        kv = self.input_proj(gemma_h)
        q = (self.clip_queries + self.pos_emb).expand(gemma_h.shape[0], -1, -1)
        key_padding_mask = ~gemma_mask.bool()
        for block in self.blocks:
            q = block(q, kv, key_padding_mask=key_padding_mask)
        return self.output_proj(self.final_norm(q))
```

Why 77 first: frozen SD1.5 UNet expects CLIP-like 77-token conditioning. Arbitrary 64 latent tokens are out-of-distribution. Long-token support comes later after we have a proven semantic contract.

**Training objective:**

Teacher:

```python
clip_h, clip_mask = encode_clip_prompts(captions)
```

Student:

```python
gemma_h, gemma_mask = encode_gemma_prompts(captions)
pred_clip_h = semantic_adapter(gemma_h, gemma_mask)
```

Loss:

```python
loss = (
    w_mse * layernorm_mse(pred_clip_h, clip_h, clip_mask)
  + w_cos * token_cosine_loss(pred_clip_h, clip_h, clip_mask)
  + w_norm * log_or_relative_norm_loss(pred_clip_h, clip_h, clip_mask)
  + w_ctr * pooled_contrastive_loss(pred_clip_h, clip_h, clip_mask)
)
```

Suggested initial weights:

```python
SEM_W_MSE = 1.0
SEM_W_COS = 0.5
SEM_W_NORM = 0.25
SEM_W_CONTRASTIVE = 0.2
SEM_CONTRASTIVE_TEMP = 0.07
```

Avoid raw norm MSE domination. Use log norm or relative norm.

**Primary overfit mode:**

Use exact `TRAIN_VAL_PROMPTS` and `OVERFIT_EVAL_BATCH`. For a 64-sample overfit, train semantic adapter for 300–1000 optimizer steps before any diffusion loss.

**Gate:**

Pass if all improve:

- semantic loss decreases substantially;
- adapter pooled cosine to CLIP increases;
- adapter→CLIP retrieval accuracy > random and trends upward;
- adapter collapse cosine is not near 1.0 for diverse prompts unless CLIP teacher itself is near-collapsed;
- frozen UNet generation from `pred_clip_h` starts to resemble CLIP baseline more than current connector output.

### Phase 1a — Frozen UNet generation using semantic adapter only

**Purpose:** Prove that SD can consume adapter output before timestep tricks.

Generation path:

```python
cond = semantic_adapter(cond_gemma_h, cond_gemma_mask)
uncond = semantic_adapter(uncond_gemma_h, uncond_gemma_mask)
encoder_hidden_states = torch.cat([uncond, cond], dim=0)
pred = unet(latents_pair, t, encoder_hidden_states=encoder_hidden_states).sample
```

No timestep adapter yet. This answers: “Can Gemma synthesize CLIP-like states that the frozen StyleJourney UNet understands?”

Compare grids:

1. CLIP teacher baseline.
2. Semantic adapter on exact overfit prompts.
3. Semantic adapter on generic controls.
4. Existing failed connector baseline if retained.

### Phase 1b — Timestep adapter/residual, not replacement

Only after Phase 1 passes.

**Architecture:**

```text
semantic_tokens = GemmaToClipSemanticAdapter(gemma_h, gemma_mask)  # [B,77,768]
residual = TimestepResidualAdapter(semantic_tokens, timestep, optional_long_tokens)
context = semantic_tokens + alpha * residual
```

Start with residual scale zero or tiny:

```python
self.res_scale = nn.Parameter(torch.tensor(-4.0))  # sigmoid or exp gives small residual
context = semantic + torch.sigmoid(self.res_scale) * residual
```

This prevents the timestep adapter from overwriting the CLIP-like contract at init.

Loss:

```text
semantic geometry loss remains active
+ teacher/student text-delta loss
+ diffusion loss
```

Teacher/student text delta:

```python
teacher_delta = teacher_clip_cond_pred - teacher_clip_uncond_pred
student_delta = student_cond_pred - student_uncond_pred
loss_delta = mse(student_delta, teacher_delta)
```

Do not let diffusion MSE be the only loss.

### Phase 2 — More tokens / longer prompts

The user wants more than 77 tokens. Treat this as a controlled architecture extension, not an immediate replacement.

#### Phase 2A — Long-token sidecar with frozen UNet, no UNet surgery

Goal: preserve CLIP-like first 77 tokens and append extra tokens in a way the frozen UNet can ignore at init.

Context format:

```text
context = concat([
    semantic_77_tokens,        # CLIP-compatible anchor
    long_extra_tokens          # Gemma-derived, initially gated near zero
], dim=1)
```

Example lengths:

```python
BASE_CLIP_TOKENS = 77
EXTRA_TOKENS = 51       # total 128 first
LONG_TOTAL_TOKENS = 128 # later 256
```

Long token adapter:

```text
Gemma hidden [B,S,640]
→ Perceiver/Resampler queries [B,EXTRA_TOKENS,768]
→ gated extra tokens
```

Gating:

```python
extra = extra * torch.sigmoid(extra_gate_logit)  # init ~0.01
context = torch.cat([semantic_77, extra], dim=1)
```

Rationale: Diffusers cross-attention accepts variable sequence length. The UNet was trained on 77 CLIP tokens, so extra tokens may be OOD. Gating near zero lets us test whether they help without destroying the anchor.

**Gate:** With frozen UNet, adding gated extra tokens must not degrade CLIP-like generation. If it degrades, the extra path is too strong or malformed.

#### Phase 2B — Train only UNet text-interface weights to use extra tokens

If Phase 2A is stable but extra tokens do not help, train small UNet subsets:

Preferred target order:

1. Cross-attention `attn2.to_k` and `attn2.to_v` only.
2. Then cross-attention `to_out[0]` if needed.
3. Then selected attention norms if present.
4. Avoid ResNet/self-attention early.

Do not whole-UNet fine-tune yet.

Losses:

```text
teacher CLIP baseline preservation on 77-token prompts
+ text-delta loss
+ diffusion/noise loss on training images
+ semantic adapter geometry loss remains active
```

Important preservation regularizer:

```python
loss_preserve = mse(unet_with_long_context_pred, clip_teacher_pred)
```

Use prompts both short and long. For long prompts, CLIP teacher is imperfect due to truncation, so also evaluate semantic/object checklist and exact overfit references.

#### Phase 2C — More tokens after proven 128

Only after 128-token run improves long-prompt behavior:

```text
128 → 192 → 256
```

Do not jump to 512. Each jump changes attention compute and distribution.

### Phase 3 — SaRA for undertrained UNet text-interface weights

Use SaRA only after Phase 1/2 gates pass. SaRA is not a bridge substitute; it is a parameter-efficient way to let under-adapted UNet weights learn to read the new/longer context.

#### SaRA integration strategy — whole UNet scan, sparse trainables only

The intended SaRA interpretation for this project is: **allow SaRA to scan the entire UNet**, because this is not dense whole-UNet fine-tuning. SaRA exposes only sparse low-magnitude / undertrained coordinates selected by threshold, then regularizes large 2D matrices with a low-rank/nuclear-norm term. This should preserve original StyleJourney knowledge better than dense AdamW over the whole UNet.

Use upstream-style construction on the whole UNet after adapter/long-token gates pass:

```python
from optim import adamw as sara_adamw

sara_optimizer = sara_adamw.AdamW(
    unet,
    lr=SARA_LR,
    threshold=SARA_THRESHOLD,
    progressive_iter=SARA_PROGRESSIVE_ITER,
    lambda_rank=SARA_LAMBDA_RANK,
    weight_decay=SARA_WEIGHT_DECAY,
)
```

Do **not** freeze most UNet params before SaRA construction in the first SaRA experiment; that would defeat the point of SaRA discovering undertrained coordinates across the model. Instead, verify the sparse mask statistics after construction.

Required SaRA instrumentation:

```python
def summarize_sara_masks(sara_optimizer, unet):
    # Print global sparse trainable count and grouped counts by module family:
    # down_blocks, mid_block, up_blocks, attn2.to_k, attn2.to_v, attn2.to_out,
    # attn1/self-attn, resnet/conv, norms, other.
    ...
```

The mask summary is the safety gate: if SaRA selects an unexpectedly huge fraction of UNet parameters or heavily targets fragile image-prior conv/resnet blocks, stop and adjust threshold before training.

Optional fallback only if whole-UNet SaRA proves unsafe: filtered SaRA for `attn2.to_k/to_v/to_out`. This is not the primary plan.

#### SaRA defaults from repo

Start with repo defaults:

```python
SARA_THRESHOLD = 2e-3
SARA_PROGRESSIVE_ITER = 2500
SARA_LAMBDA_RANK = 0.0005
SARA_LR = 1e-5  # use explicit LR; repo default depends exponentially on threshold
```

But for our overfit/diagnostic phase, use shorter:

```python
SARA_PROGRESSIVE_ITER = 100  # for 300-step overfit diagnostic
```

because the repo’s 2500 is for longer downstream fine-tuning.

#### What “undertrained weights” means here

SaRA may scan all UNet parameter tensors. The expectation is that the sparse threshold preferentially exposes undertrained/low-magnitude coordinates while leaving most original knowledge untouched.

Still monitor where SaRA places trainable coordinates. Desired/expected useful groups:

```text
UNet cross-attn to_k / to_v — text token reading
UNet cross-attn to_out[0] — injecting attended text into image hidden states
possibly transformer block norms around attn2
```

Allowed but high-risk groups:

```text
self-attention attn1
ResNet/conv blocks
large output/input convolutions
```

Do not manually exclude these in the first SaRA experiment, but treat heavy selection in these groups as a warning requiring review. The adapter itself should remain dense AdamW; SaRA is for UNet adaptation, not for the semantic adapter.

#### SaRA gates

Before SaRA:

- Phase 1 semantic adapter passes CLIP geometry gate.
- Frozen-UNet adapter generation is at least partially prompt-aligned.
- Long-token sidecar does not degrade 77-token generation.

During SaRA:

- Save sparse params separately with threshold metadata.
- Save merged/reload proof path.
- Run fixed short-prompt and long-prompt validation every interval.
- Stop if CLIP/77-token baseline preservation degrades while long-token gain is absent.

---

## Concrete implementation tasks

### Task 1: Add mode/architecture names and phase flags

**Objective:** Make notebook paths explicit and prevent accidental connector/diffusion-only training.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`
- Modify: `.hermes/plans/2026-05-15-semantic-adapter-timestep-longtoken-sara-plan.md` only if discoveries change the plan

**Changes:**

Add architecture options:

```python
CONDITIONING_ARCH = "gemma_clip_semantic_adapter"  # ["dual_native", "ella_gemma_connector", "gemma_clip_semantic_adapter"]
```

Add phase switches:

```python
RUN_SEMANTIC_ADAPTER_TRAIN = CONDITIONING_ARCH == "gemma_clip_semantic_adapter"
RUN_TIMESTEP_RESIDUAL = False
RUN_LONG_TOKEN_PHASE = False
RUN_SARA_PHASE = False
```

Add config:

```python
SEMANTIC_NUM_TOKENS = 77
SEMANTIC_WIDTH = 768
SEMANTIC_LAYERS = 4
SEMANTIC_HEADS = 8
SEMANTIC_LR = 1e-4
SEMANTIC_MAX_OPT_STEPS = 500
SEMANTIC_VALIDATE_EVERY = 50
```

**Verification:**

Run notebook static compile. Confirm old `ella_gemma_connector` path still exists but is not default.

### Task 2: Implement `GemmaToClipSemanticAdapter`

**Objective:** Add an adapter that emits `[B,77,768]` CLIP-like states.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`

**Implementation notes:**

Use pre-norm cross-attention blocks adapted from the Qwen notebook. Propagate Gemma mask to `key_padding_mask`.

Add smoke test:

```python
g = torch.randn(2, 128, 640, device=device, dtype=unet_dtype)
m = torch.ones(2, 128, device=device, dtype=torch.long)
out = semantic_adapter(g, m)
assert out.shape == (2, 77, 768)
assert torch.isfinite(out).all()
```

**Verification:**

- shape pass;
- finite pass;
- trainable param count printed.

### Task 3: Add `ClipGeometryLoss`

**Objective:** Port Qwen-style masked representation loss.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`

**Implementation:**

Implement:

```python
layernorm_mse(pred, target, mask)
token_cosine_loss(pred, target, mask)
log_norm_loss(pred, target, mask)
pooled_contrastive_loss(pred, target, mask)
```

Return dict:

```python
{
  "total": total,
  "mse": mse,
  "cos": cos_l,
  "norm": norm_l,
  "ctr": ctr,
  "pnorm": pnorm,
  "tnorm": tnorm,
  "norm_ratio": pnorm / tnorm,
  "pooled_cos": pooled_cos,
}
```

**Verification:**

Run on random tensors and real CLIP/Gemma batch. Confirm finite loss and nonzero gradients on adapter params.

### Task 4: Add semantic adapter training loop

**Objective:** Train Gemma adapter to match CLIP states before diffusion loss.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`

**Loop:**

Use the deterministic 64-sample overfit subset first.

```python
for step, batch in ...:
    captions = _as_prompt_list(batch["caption"])
    with torch.no_grad():
        clip_h, clip_mask = encode_clip_prompts(captions)
        gemma_h, gemma_mask = encode_gemma_prompts(captions)
    pred = semantic_adapter(gemma_h, gemma_mask)
    loss_dict = clip_geometry_loss(pred, clip_h, clip_mask)
    loss_dict["total"].backward()
```

Do not include UNet/diffusion loss in this task.

**Verification:**

Every 50 steps:

- log semantic loss components;
- log adapter→CLIP pooled cosine;
- log retrieval accuracy;
- run collapse diagnostic.

Gate before next task: semantic loss must decrease and retrieval/cosine must improve.

### Task 5: Add frozen-UNet semantic adapter generation

**Objective:** Prove the frozen UNet can consume adapter states.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`

**Function:**

```python
def generate_semantic_adapter(prompt, steps=VAL_STEPS, guidance=VAL_GUIDANCE, seed=VAL_SEED):
    cond_g, cond_m = encode_gemma_prompts([prompt])
    uncond_g, uncond_m = encode_gemma_prompts([""])
    cond = semantic_adapter(cond_g, cond_m)
    uncond = semantic_adapter(uncond_g, uncond_m)
    context = torch.cat([uncond, cond], dim=0)
    ... unet(..., encoder_hidden_states=context) ...
```

Save grids:

```text
validation_semantic_adapter_overfit_train_prompts.png
validation_semantic_adapter_generic_controls.png
validation_clip_teacher_baseline.png
```

**Verification:**

Compare visually and log fixed overfit eval loss using semantic adapter context.

### Task 6: Add timestep residual adapter

**Objective:** Add timestep awareness without breaking semantic CLIP-like anchor.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`

**Architecture:**

```python
semantic = semantic_adapter(gemma_h, gemma_mask)
residual = timestep_adapter(semantic, timesteps)
context = semantic + gate * residual
```

Initialize gate small.

**Training:**

Keep semantic geometry loss active:

```python
loss = sem_loss + W_DELTA * text_delta_loss + W_DIFFUSION * diffusion_loss
```

**Verification:**

- gate value logged;
- delta cosine improves without semantic geometry collapse;
- generated overfit grid improves over Phase 1.

### Task 7: Add long-token sidecar, gated near zero

**Objective:** Support longer prompts without immediately changing UNet weights.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`

**Config:**

```python
LONG_CONTEXT_TOTAL_TOKENS = 128
LONG_EXTRA_TOKENS = LONG_CONTEXT_TOTAL_TOKENS - 77
LONG_EXTRA_GATE_INIT = -5.0
```

**Context:**

```python
semantic77 = semantic_adapter(...)
extra = long_token_resampler(gemma_h, gemma_mask)  # [B,51,768]
extra = sigmoid(extra_gate) * extra
context = torch.cat([semantic77, extra], dim=1)
```

**Verification:**

- context shape `[B,128,768]`;
- frozen UNet forward finite;
- with gate near zero, outputs close to 77-token semantic path;
- no degradation on short prompts.

### Task 8: Add long-prompt dataset/eval prompts

**Objective:** Evaluate why extra tokens matter.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`

Add long prompts that differ after the first 77 CLIP tokens. For each long prompt, create a truncated/short control.

Metrics:

- CLIP-token truncated baseline image;
- semantic77 image;
- long128 image;
- visual checklist of late-prompt attributes.

**Verification:**

Long128 must preserve early prompt and improve late-prompt attributes before increasing to 192/256.

### Task 9: Integrate SaRA as optional UNet text-interface phase

**Objective:** Train sparse low-rank updates on selected UNet text-interface weights after long-token path is stable.

**Files:**

- Modify: `gemma3_sd_colab.ipynb`
- Optionally vendor minimal SaRA optimizer code into notebook cell or import from a local cloned path.

**Target:**

```text
entire UNet scanned by SaRA sparse masks
```

This is intentionally not dense whole-UNet fine-tuning. It is whole-UNet sparse undertrained-coordinate adaptation.

**Config:**

```python
RUN_SARA_PHASE = False
SARA_SCOPE = "whole_unet_sparse"
SARA_THRESHOLD = 2e-3
SARA_PROGRESSIVE_ITER = 100
SARA_LAMBDA_RANK = 0.0005
SARA_LR = 1e-5
SARA_WEIGHT_DECAY = 1e-2
SARA_MAX_SPARSE_FRACTION_WARN = 0.02  # warn if >2% of UNet params are selected
SARA_MAX_SPARSE_FRACTION_ABORT = 0.05 # abort if >5% unless explicitly overridden
```

For long runs, use repo-like `SARA_PROGRESSIVE_ITER=2500`; for 300-step diagnostics, use ~100.

**Verification:**

- print sparse trainable count globally and per module family;
- warn/abort if sparse selected fraction is too high;
- save sparse params with threshold, sparse fractions, and `SARA_SCOPE` metadata;
- reload sparse params into fresh UNet and run finite proof;
- validation grids for 77-token and long-token prompts;
- preservation check: short prompt CLIP/semantic baseline should not degrade without long-prompt gain.

### Task 10: Save/reload artifacts for each phase

**Objective:** Make results reproducible and avoid live-notebook-only proof.

Artifacts:

```text
gemma_semantic_adapter_phase1.pt
gemma_timestep_residual_phase1b.pt
gemma_long_context_adapter_phase2.pt
sara_unet_text_interface_sparse.pt
```

Each checkpoint must include:

```python
{
  "architecture": ...,
  "config": ...,
  "state_dict": ...,
  "conditioning_contract": "Gemma -> CLIP-like [B,77,768]" or "Gemma -> long [B,128,768]",
  "run_config": RUN_CONFIG,
}
```

Reload proof:

- rebuild fresh StyleJourney UNet;
- rebuild Gemma and adapters;
- load checkpoint strict;
- finite forward;
- generate sample grid from reloaded modules, not live modules.

---

## Stop/continue gates

### Stop immediately if

- semantic adapter cannot reduce CLIP geometry loss on 64-sample overfit;
- adapter outputs collapse to near-identical pooled embeddings across diverse prompts;
- frozen UNet generation from semantic adapter is worse than the current connector and does not improve after semantic loss improves;
- long-token sidecar degrades 77-token behavior with gate near zero;
- SaRA changes/degrades short-prompt outputs but gives no long-prompt gain;
- whole-UNet SaRA sparse mask selects too large a fraction of UNet parameters or unexpectedly concentrates in fragile image-prior blocks without clear benefit.

### Continue only if

- semantic adapter proves CLIP-state alignment;
- exact overfit training prompts visibly improve;
- fixed overfit eval loss falls;
- delta cosine improves and norm ratio approaches 1;
- long-token sidecar preserves short prompts and helps late-prompt attributes;
- SaRA sparse target count is small and reloadable.

---

## User-facing answer to “should UNet adjustments be done earlier?”

Do not adjust UNet before Phase 1 semantic adapter passes. If the adapter cannot synthesize the 77-token CLIP contract, UNet training will compensate for a broken bridge and likely learn image priors/collapse. However, for long-token support, UNet-side adaptation probably is necessary because the frozen SD1.5 UNet never learned to use >77 CLIP-position tokens. Therefore:

1. First prove 77-token semantic contract with frozen UNet.
2. Then append gated long tokens with frozen UNet to ensure no degradation.
3. Then train only cross-attention text-interface weights, preferably with SaRA.
4. Only later consider broader UNet tuning.

---

## Open questions for implementation review

1. Should Phase 1 use CLIP final hidden states or penultimate hidden states? Use exactly what the current StyleJourney pipeline text encoder feeds to UNet if available.
2. Should semantic adapter output all 77 tokens including special/pad tokens, or mask loss to CLIP attention mask only? Start masked but preserve output length.
3. Should timestep residual produce 77 residuals only or also residuals for long extra tokens? Start 77 only; extend after Phase 2.
4. Should SaRA use upstream optimizer or notebook-local implementation? Primary plan is upstream-style whole-UNet SaRA scan; notebook-local code is only for adding better mask statistics/reload metadata or if upstream optimizer breaks in Colab.
