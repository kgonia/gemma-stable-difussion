# ELLA-Adjusted Gemma Conditioning Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Adapt the current Gemma→SD experiment using ELLA’s strongest lesson: train a lightweight timestep-aware semantic connector from frozen Gemma features into the frozen SD/StyleJourney UNet conditioning interface, then run inference with Gemma + connector only, no CLIP.

**Architecture:** Replace the previous “CLIP/Gemma token-count handoff” as the main next direction. Keep `overfit_train` as the immediate falsifier, but the next architecture should be ELLA-like: frozen Gemma text encoder → trainable timestep-aware Perceiver/Resampler connector → original frozen SD1.5 cross-attention (`encoder_hidden_states` dim 768). CLIP remains a teacher/evaluator only during diagnostics/training, not part of final inference.

**Tech Stack:** Colab `.ipynb`, PyTorch, Diffusers SD1.5/StyleJourney, frozen Gemma 3 270M, optional frozen CLIP teacher, ELLA-style Perceiver Resampler with AdaLN timestep conditioning.

---

## Evidence from ELLA paper/repo

Local paper: `docs/2403.05135v1.pdf`.
Repo inspected: `https://github.com/TencentQQGYLab/ELLA`, commit `3c228f1`.

ELLA’s key facts:

1. ELLA freezes both the LLM text encoder and the diffusion model.
2. The only trainable component is a lightweight Timestep-Aware Semantic Connector (TSC).
3. TSC receives arbitrary-length LLM hidden states plus diffusion timestep embedding.
4. TSC outputs fixed-length semantic queries used as `encoder_hidden_states` for the frozen UNet cross-attention.
5. They compare MLP vs Perceiver Resampler vs timestep-aware Resampler.
6. Best reported connector is Resampler + timestep using AdaLN, not AdaLN-Zero.
7. ELLA uses dense captions; paper trained on 34M image-text pairs for main results, with ablations around 140k optimization steps.
8. ELLA’s SD1.5 trainable parameter count is ~0.07B, much smaller than UNet tuning.
9. ELLA repo implementation details:
   - `PerceiverResampler(width=768, layers=6, heads=8, num_latents=64, input_dim=2048)`
   - `ELLA.forward(text_encode_features, timesteps)` returns connector output.
   - `ELLAProxyUNet.forward(...)` computes `time_aware_encoder_hidden_states = ella(encoder_hidden_states, timestep)` and passes that to the original frozen UNet.
   - Fixed output is 64 tokens.

Important implication for this project:

The prior proposed token-ratio curriculum is less central now. ELLA suggests the bridge should not be “raw Gemma tokens slowly replacing CLIP tokens.” It should be a learned timestep-aware resampler that turns Gemma’s decoder hidden sequence into UNet-readable conditioning tokens.

---

## Socratic reassessment

### Q1. What is the bottleneck shown by our current notebook?

The Gemma branch is alive and reloadable, but semantically wrong. It responds to text in some direction, but not the CLIP/SD semantic direction.

### Q2. What does ELLA say about that bottleneck?

ELLA says a direct raw LLM-hidden-state cross-attention bridge is probably underpowered/misaligned. The successful bridge is not just K/V replacement; it is a trainable semantic connector/resampler with timestep awareness.

### Q3. Does ELLA violate the user’s “only Gemma conditioning” goal?

No, if implemented as:

```text
prompt → Gemma → GemmaTSC/Resampler → frozen StyleJourney UNet
```

There is no CLIP at inference. The connector outputs UNet-compatible conditioning, but runtime conditioning is entirely derived from Gemma.

### Q4. Is this just the Gemma→CLIP adapter idea under another name?

Not exactly.

A simple adapter is:

```text
Gemma tokens → per-token Linear/MLP → 768 CLIP-like tokens
```

ELLA-style connector is:

```text
Gemma arbitrary-length tokens + timestep → Perceiver latent queries → 64 dynamic semantic tokens → UNet
```

The output dimension is CLIP-compatible because the frozen SD UNet expects 768, but the method is not a static CLIP embedding predictor. It learns denoising-stage-specific conditioning.

### Q5. Why is this better for VRAM?

Final inference loads:

```text
StyleJourney UNet + VAE + scheduler + Gemma 270M + connector
```

It does not load CLIP. Gemma 270M is smaller than CLIP-L/14 text encoder in the current notebook outputs. Connector can be much smaller than ELLA’s 0.07B if needed.

### Q6. Should we still add `overfit_train`?

Yes. But the overfit target should shift:

Old overfit target:

```text
Can native Gemma K/V branch overfit?
```

New overfit target:

```text
Can an ELLA-style Gemma connector overfit 32–64 repeated images while UNet is frozen?
```

This is closer to ELLA and more directly tests the intended final architecture.

### Q7. Should we continue the CLIP+Gemma branch curriculum?

Deprioritize it. Keep it as fallback/diagnostic only. ELLA’s evidence says the stronger path is a timestep-aware connector, not token handoff.

---

## Updated recommendation

Do this next:

1. Keep the already planned `overfit_train` mode.
2. Add delta-cosine diagnostics regardless.
3. Add a new ELLA-style connector path behind a flag:

```python
CONDITIONING_ARCH = "ella_gemma_connector"  # ["dual_native", "ella_gemma_connector"]
```

4. First run `ella_overfit_train` / `overfit_train` with:

```text
frozen UNet
frozen VAE
frozen Gemma
train connector only
small repeated subset
no CLIP in student forward
CLIP only for teacher diagnostics/loss if enabled
```

5. If connector cannot overfit a tiny subset, do not scale.
6. If connector overfits, scale data before unfreezing UNet.

---

## Architecture target

### Final inference graph

```text
prompt
  ↓
Gemma tokenizer + frozen Gemma 3 270M
  ↓ last/mid hidden states [B, S, 640]
GemmaTimestepSemanticConnector
  - learned latent queries, e.g. 32 or 64
  - cross-attend to Gemma hidden states
  - AdaLN timestep conditioning
  - output dim 768
  ↓ [B, N_latents, 768]
frozen StyleJourney SD1.5 UNet original cross-attention
  ↓
VAE decode
```

No CLIP at inference.

### Training graph

Student:

```text
Gemma → connector(timestep) → frozen UNet → predicted noise
```

Optional teacher:

```text
CLIP → frozen original UNet → teacher predicted noise
```

Loss options, staged:

```python
loss = (
    lambda_diffusion * mse(student_pred, noise)
    + lambda_teacher * mse(student_pred, teacher_pred.detach())
    + lambda_delta_dir * delta_direction_loss(student_delta, teacher_delta)
)
```

For the first ELLA-like run, keep it simple:

```text
connector-only diffusion loss + optional teacher/delta diagnostics
```

Do not train UNet initially.

---

## Key design choices for Gemma 270M

### Connector size

ELLA SD1.5 uses ~0.07B params with 6 resampler blocks. For Colab/VRAM iteration, start smaller:

```python
GEMMA_CONNECTOR_WIDTH = 768
GEMMA_CONNECTOR_LAYERS = 2       # first overfit, then 4/6 if promising
GEMMA_CONNECTOR_HEADS = 8
GEMMA_CONNECTOR_NUM_LATENTS = 64 # maybe 32 for VRAM, 64 closer to ELLA
GEMMA_CONNECTOR_INPUT_DIM = 640
GEMMA_CONNECTOR_OUTPUT_DIM = 768
```

Rough direction:

- 1-block/2-block connector: faster falsifier.
- 6-block connector: closer to ELLA and likely better for real alignment.

### Gemma layer

Do not assume final Gemma layer is best. ELLA used encoder LMs; Gemma is decoder-only. Keep layer selection configurable:

```python
GEMMA_LAYER_INDEX = -1
GEMMA_LAYER_POOL = "single"  # future: "last4_mean"
```

If overfit fails, try:

```text
layer 12 / 16 / 18 / last4_mean
```

before unfreezing UNet.

### Token length

ELLA trains with 128 extracted text tokens. For Gemma 270M:

```python
MAX_GEMMA_LEN = 128
```

Keep connector output fixed at 64 latents. Do not map Gemma token count directly to CLIP token count.

### Timestep awareness

Must be in the first serious connector. ELLA’s ablation shows Resampler+AdaLN timestep beats plain resampler and AdaLN-Zero.

Implement:

```text
Timesteps(320) → TimestepEmbedding(768) → AdaLayerNorm in resampler blocks
```

Can copy/adapt ELLA repo model.py structure, but adjust `input_dim=640`.

---

## Revised acceptance criteria

1. `RUN_MODE` includes `overfit_train`.
2. Notebook includes `CONDITIONING_ARCH` with at least:

```python
"dual_native"
"ella_gemma_connector"
```

3. In `ella_gemma_connector` mode:
   - original UNet cross-attention is not surgically replaced,
   - UNet parameters are frozen,
   - Gemma parameters are frozen,
   - only connector parameters train.
4. Student forward does not require CLIP hidden states.
5. Final inference/proof for connector path reloads:

```text
StyleJourney UNet + Gemma + connector checkpoint
```

and generates without CLIP.
6. Diagnostics include:
   - CLIP baseline grid, for comparison only,
   - Gemma-connector student grid,
   - student/teacher delta cosine if CLIP teacher is loaded,
   - final fresh reload proof.
7. The prior CLIP/Gemma token-ratio curriculum is not implemented until connector overfit is tested.

---

## Task 1: Mark previous token-curriculum plan as superseded

**Objective:** Prevent implementation from following the older CLIP/Gemma token-ratio path first.

**Files:**
- Modify: `.hermes/plans/2026-05-14-overfit-train-and-curriculum-plan.md`

**Change:** Add at top:

```markdown
> Superseded direction note: after inspecting ELLA (`docs/2403.05135v1.pdf` and TencentQQGYLab/ELLA), prioritize an ELLA-style Gemma timestep-aware connector over CLIP/Gemma token-count curriculum. Keep `overfit_train` and delta diagnostics, but defer token curriculum.
```

**Verification:** Read first 10 lines and confirm note exists.

---

## Task 2: Add `overfit_train` mode first

**Objective:** Preserve the useful low-risk config change.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, config cell.

**Defaults:**

```python
RUN_MODE = "overfit_train"  # @param ["diagnostic", "overfit_train", "short_train", "full_train"]
RUN_TRAINING = RUN_MODE in {"overfit_train", "short_train", "full_train"}

if RUN_MODE == "overfit_train":
    MAX_SAMPLES_WARMUP = 64
    MAX_SAMPLES_PHASEB = 64
    FULLRANK_EPOCHS = 40
    PHASEB_EPOCHS = 0
    PHASE_A_MAX_OPT_STEPS = 300
    PHASE_B_MAX_OPT_STEPS = 0
    TEACHER_DECAY_STEPS = 300
    VALIDATION_EVERY_OPT_STEPS = 50
    SHUFFLE_STREAMING = False
```

**Verification:** Static notebook scan confirms mode exists and shuffle is false for overfit.

---

## Task 3: Add delta alignment diagnostic

**Objective:** Keep the earlier review’s most useful diagnostic.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, diagnostics cell.

**Add:** `compute_teacher_student_delta_alignment(...)` as described in the previous plan.

**Use for both architectures:**

- `dual_native`: student uses Gemma branch.
- `ella_gemma_connector`: student uses connector output.

**Verification:** Function prints/logs:

```text
student/teacher delta cosine
student/teacher delta norm ratio
```

---

## Task 4: Add ELLA-style connector classes

**Objective:** Add a small self-contained connector implementation adapted from ELLA.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, new cell after Gemma/UNet load helpers.

**Classes:**

```python
class AdaLayerNorm(nn.Module):
    ...

class SquaredReLU(nn.Module):
    ...

class PerceiverAttentionBlock(nn.Module):
    ...

class GemmaTimestepSemanticConnector(nn.Module):
    ...
```

**Implementation notes from ELLA:**

- Use `nn.MultiheadAttention(batch_first=True)`.
- Latents are learned query tokens.
- Cross-attention key/value is `torch.cat([normed_latents, normed_gemma_tokens], dim=1)` like ELLA.
- Use AdaLN, not AdaLN-Zero.
- Timestep embedding should match ELLA style:

```python
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
```

**Suggested constructor:**

```python
class GemmaTimestepSemanticConnector(nn.Module):
    def __init__(self, input_dim=640, width=768, output_dim=768, layers=2, heads=8, num_latents=64, time_channel=320, time_embed_dim=768):
        ...
```

**Forward:**

```python
def forward(self, gemma_hidden_states, timesteps, gemma_attention_mask=None):
    # returns [B, num_latents, 768]
```

Mask support is important. ELLA’s minimal repo code does not use a key-padding mask inside `MultiheadAttention`; for Gemma, add it if feasible. If mask support is too risky for first pass, log this limitation loudly and rely on max-length padding/truncation, but do not hide it.

**Verification:** Run a smoke forward:

```python
g = torch.randn(2, 128, 640, device=device, dtype=unet_dtype)
t = torch.randint(0, scheduler.config.num_train_timesteps, (2,), device=device)
out = gemma_connector(g, t)
assert out.shape == (2, GEMMA_CONNECTOR_NUM_LATENTS, 768)
assert torch.isfinite(out).all()
```

---

## Task 5: Add `CONDITIONING_ARCH` switch

**Objective:** Allow current dual-native path and new ELLA path to coexist without breaking existing results.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, config and forward helper cells.

**Config:**

```python
CONDITIONING_ARCH = "ella_gemma_connector"  # @param ["dual_native", "ella_gemma_connector"]
```

**Behavior:**

- `dual_native`: current surgery/wrappers path.
- `ella_gemma_connector`: do not replace UNet attention modules. Keep original UNet cross-attention and train connector only.

**Important:** In connector mode, skip cells that install `DualNativeAttention` or guard them:

```python
if CONDITIONING_ARCH == "dual_native":
    install_dual_native_attention(...)
else:
    print("Skipping DualNativeAttention surgery; using frozen original UNet + Gemma connector.")
```

**Verification:** Static scan confirms surgery is conditional.

---

## Task 6: Add connector training loop or branch existing Phase A

**Objective:** Train only connector in `ella_gemma_connector` mode.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, Phase A training cell.

**Connector training forward:**

For each batch:

```python
with torch.no_grad():
    latent = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
    cond_gemma_h, cond_gemma_mask = encode_gemma_prompts(captions)
    uncond_gemma_h, uncond_gemma_mask = encode_gemma_prompts(empty_captions)

noise = torch.randn_like(latent)
t = torch.randint(0, scheduler.config.num_train_timesteps, (latent.shape[0],), device=device).long()
noisy = scheduler.add_noise(latent, noise, t)

# CFG/text-delta paired batch
noisy_pair = torch.cat([noisy, noisy], dim=0)
t_pair = torch.cat([t, t], dim=0)
gemma_h_pair = torch.cat([cond_gemma_h, uncond_gemma_h], dim=0)
gemma_mask_pair = torch.cat([cond_gemma_mask, uncond_gemma_mask], dim=0)

student_context = gemma_connector(gemma_h_pair, t_pair, gemma_mask_pair)
student_pair = unet(noisy_pair, t_pair, encoder_hidden_states=student_context).sample
student_cond, student_uncond = student_pair.chunk(2)
```

Loss first pass:

```python
loss_diffusion = mse(student_cond.float(), noise.float())
```

Optional if CLIP teacher loaded:

```python
teacher_pair = clip_teacher_unet_forward(...)
loss_teacher = mse(student_cond.float(), teacher_cond.float())
loss_text_delta = delta_direction_or_mse(...)
```

For pure overfit falsifier, prefer simple:

```python
loss = loss_diffusion
```

Then add teacher/delta if diffusion-only overfit produces images but weak semantics.

**Trainables:**

```python
for p in unet.parameters(): p.requires_grad_(False)
for p in gemma_model.parameters(): p.requires_grad_(False)
for p in gemma_connector.parameters(): p.requires_grad_(True)
```

**Optimizer:**

ELLA uses 1e-4 for SD1.5. For small connector overfit:

```python
CONNECTOR_LR = 1e-4
optimizer = torch.optim.AdamW(gemma_connector.parameters(), lr=CONNECTOR_LR, weight_decay=0.01)
```

**Verification:** Trainable summary shows only connector params.

---

## Task 7: Add connector inference and fresh reload proof

**Objective:** Prove final path is Gemma-only and CLIP-free.

**Files:**
- Modify: `gemma3_sd_colab.ipynb`, validation/proof cells.

**Generation helper:**

```python
@torch.no_grad()
def generate_ella_gemma(prompt, steps=30, guidance=7.5, seed=42):
    cond_gemma, cond_mask = encode_gemma_prompts([prompt])
    uncond_gemma, uncond_mask = encode_gemma_prompts([""])
    ...
    context = gemma_connector(gemma_h, timestep_batch, gemma_mask)
    pred = unet(latent_model_input, t, encoder_hidden_states=context).sample
```

Because connector is timestep-aware, compute connector context at every denoising step, not once per prompt.

**Save artifact:**

```python
torch.save({
    "connector_state_dict": gemma_connector.state_dict(),
    "connector_config": {...},
    "gemma_model_id": GEMMA_MODEL_ID,
    "sd_checkpoint": SD_CHECKPOINT_PATH,
    "conditioning_arch": "ella_gemma_connector",
}, path)
```

**Reload proof:**

Freshly reconstruct:

```text
StyleJourney UNet from safetensors
Gemma frozen
GemmaTimestepSemanticConnector from config
strict load connector
generate validation grid
```

Do not load CLIP for this proof.

**Verification:** Print:

```text
CLIP loaded for final proof: False
Connector reload strict: PASS
Gemma-only connector finite forward: PASS
```

---

## Task 8: Defer CLIP/Gemma token curriculum

**Objective:** Prevent premature complexity.

**Files:**
- Modify: `.hermes/plans/2026-05-14-overfit-train-and-curriculum-plan.md` or leave as deprecated reference.

**Decision:** Do not implement token-ratio schedule until after connector overfit results.

Reason:

ELLA’s core mechanism is not CLIP token removal. It is LLM→fixed semantic-query connector with timestep awareness.

---

## Updated execution order

1. Add `overfit_train` and delta diagnostic.
2. Commit.
3. Add ELLA-style connector classes and `CONDITIONING_ARCH` switch.
4. Verify notebook statically.
5. Commit.
6. Run `overfit_train` with `CONDITIONING_ARCH="ella_gemma_connector"`.
7. Inspect tiny-overfit results.
8. If positive, scale connector training data.
9. Only after connector improves prompt semantics, consider:
   - larger connector (4/6 blocks),
   - denser captions,
   - teacher/delta loss,
   - limited UNet unfreeze.

---

## Decision gate after ELLA-style overfit

Proceed if:

```text
connector-only train loss falls
Gemma connector validation images become more semantically related to prompts
student/teacher delta cosine improves if teacher is used
final reload proof is CLIP-free and matches live connector path
```

Stop if:

```text
loss falls but images remain arbitrary grid/facade/wall priors
delta cosine stays near zero/negative
connector output collapses or NaNs
final generation accidentally requires CLIP
```

---

## Practical answer to “why not just full training?”

ELLA’s successful recipe used 34M dense captions and 140k–280k optimization steps while freezing UNet. That strongly argues against whole-UNet finetuning as the next fix. The missing piece is not global denoiser capacity; it is a learned semantic connector plus enough high-density caption data.

For this project, the closest low-cost approximation is:

```text
Gemma 270M frozen + small timestep-aware connector + frozen StyleJourney UNet + overfit falsifier
```

If that works, scale connector training. If that does not work, whole-UNet finetune is still premature.
