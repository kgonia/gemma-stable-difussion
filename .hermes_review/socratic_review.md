# Socratic Review: Gemma→SD Pure ELLA Long-Context — Color-Only Degradation

## EXECUTIVE VERDICT

The connector's extra tokens (positions 77-127) are **functionally dead**: gated to ~4.7% of signal power, contribute near-zero to conditioning output, and show zero learning progress over 1,500 steps. Combined with anchor tokens that drift uncontrollably (norm_ratio → 2.6x, delta_cos oscillates 0.1↔0.54), the UNet receives conditioning that is ~90% uninformative noise from a drifting subspace → color-blob collapse.

Three root causes, in order of lethality:
1. **Dead extra-token gate** — catch-22: gate initialized at sigmoid(-3.0)=0.047, gradients through dead tokens vanish, gate never rises, tokens never learn
2. **Alt-text dataset** (jackyhate/text-to-image-2M) — captions are ~10-15 tokens; 128-token context is 90% padding, giving extra query tokens nothing to attend to
3. **No anchor regularization** — PHASE1_SEMANTIC_ANCHOR_WEIGHT=0.0 after CLIP pretrain lets anchor outputs drift arbitrarily from semantic space the UNet can interpret

---

## HARD EVIDENCE TABLE

| Metric | Step 250 | Step 1500 | What It Means |
|--------|----------|-----------|---------------|
| extra_gate (sigmoid) | 0.04741 | 0.04685 | Gate FROZEN — extra tokens contribute ~4.7% signal power, zero learning |
| extra_to_base_ratio | 0.0468 | 0.0468 | Extra-token RMS is 4.7% of anchor RMS — effectively dead |
| rel_diff_full_vs_zeroextra_cond | 0.0030 | 0.0073 | Zeroing all extra tokens changes conditioning by <1% — they are irrelevant |
| rel_diff_77_vs_full_cond | 0.026 | 0.035 | Using 77 vs 128 tokens barely matters (~3% difference) |
| suffix_swap_rel_diff_full_cond | 0.0097 | 0.0174 | Swapping "teapot"↔"violin" suffix changes conditioning ~1-2% — connector is blind to content differences |
| delta_cos (vs CLIP teacher) | 0.097 | 0.541 | Wild oscillation — anchor space drifts then partially recovers, unstable |
| norm_ratio (output/teacher norm) | 1.11 | 2.61 | Output norm drifts 2.6x beyond CLIP teacher — UNet likely can't map this |
| best ELLA loss | 0.00462 | — | Training loss seems "good" but is misleading — model learned to predict noise from uninformative conditioning, not to generate images |
| image_exposures | — | 6,000 | ELLA paper uses 34M (5,700x more data) |
| dataset | — | jackyhate/text-to-image-2M | LAION alt-text (~10-15 tokens avg) vs ELLA's 50-word CogVLM dense captions |

### Attention Analysis (UNet cross-attention to connector outputs)

| Metric | Step 250 | Step 1500 | Interpretation |
|--------|----------|-----------|----------------|
| attn_prefix_share | 0.655 | 0.582 | UNet pays ~60% attention to anchor tokens |
| attn_extra_share | 0.345 | 0.418 | UNet pays ~40% attention to extra tokens |
| Key insight | | | UNet IS attending to extra positions, but those positions carry near-zero information (gate ~0.047) — the UNet is "looking at empty space" |

### Suffix Boundary Analysis

Prompt structure: LONG_CONTEXT_SHARED_PREFIX (125 tokens) + "Final instruction: ..." (differentiating)
- first_diff_index = 125: differentiation starts at token 125, inside the extra-token region
- BUT: only tokens 125-127 fall within the 128 context window — just 3 differentiating tokens
- The extra gate crushes these 3 tokens to ~4.7% → suffix_swap_rel_diff_full_cond = 0.01-0.02

---

## SOCRATIC PROBING QUESTIONS

Each question is designed to force a decision about the next experiment. Answer honestly — the answers directly determine what to fix.

### Q1: THE GATE DEATH SPIRAL

The extra gate is initialized at sigmoid(-3.0) = 0.047. It moves from 0.04743 → 0.04685 over 1,500 steps — effectively ZERO change. The gradient through sigmoid at x=-3.0 is sigmoid(-3)*(1-sigmoid(-3)) = 0.047*0.953 = 0.045, which is non-zero. So gradient IS flowing through the gate. Yet the gate doesn't move.

**Question**: Is the gate frozen because:
- (a) The loss signal through the frozen UNet's cross-attention is too weak to push meaningful gradient through 51 near-zero extra tokens → gate → connector?
- (b) Gradient clipping at 0.5 is crushing the gate gradient? (The gate is a single scalar parameter — its gradient norm equals its absolute gradient, which may be tiny compared to the connector's 40M other parameters)
- (c) The gate is initialized in a local minimum — reducing it further would make extra tokens even less useful, and increasing it would add noise that initially hurts the loss?

**Decisive experiment**: Run with CONNECTOR_EXTRA_GATE_INIT = 0.0 (sigmoid(0) = 0.5, giving extra tokens equal weight). Does the gate now move? Do extra tokens start learning?

### Q2: WHAT ARE EXTRA QUERY TOKENS ATTENDING TO?

The data is `jackyhate/text-to-image-2M` — captions extracted from `json.prompt` field. LAION/COYO alt-texts average 10-15 CLIP tokens. After Gemma tokenization, these are ~15-30 tokens. The Gemma context is truncated to MAX_GEMMA_LEN=256, and the connector has 128 query tokens.

**Question**: For a 20-token caption, tokens 20-255 of the Gemma output are padding. Query tokens 77-127 attend via cross-attention to Gemma hidden states at ALL positions, including padding positions 20-255. What are they learning from? Are extra query tokens essentially learning to attend to:
- (a) Meaningful content at positions 0-19 (redundant with anchor query tokens 0-76)?
- (b) Padding token embeddings that carry zero semantic information?
- (c) A mix — diluting any real signal below the noise floor?

**Decisive experiment**: Log the average cross-attention distribution of extra query tokens (77-127) to Gemma positions. Are they attending to content positions (0-19) or padding positions (20+)? If to padding, the extra tokens are learning nothing useful regardless of gate.

### Q3: WHY 6,000 SAMPLES INSTEAD OF 34M?

ELLA paper: 34M dense-caption pairs, 140K-280K optimization steps, 128 token length, AdamW lr=1e-4.
Our run: 6,000 alt-text pairs, 1,500 steps, 128 token length, same lr.

**Question**: Even ignoring data quality, is 6K samples / 1,500 steps enough for the connector to learn ANY non-trivial mapping from Gemma hidden states → UNet conditioning? What's the minimum viable scale? Could this be tested by:
- (a) Overfitting on 64 samples with 600 steps to see if the connector can ever produce coherent images?
- (b) Training a 77-token-only connector (CONTEXT_TOKENS=77) first to verify the basic architecture works?

**Decisive experiment**: Run `RUN_MODE=overfit_train` with CONTEXT_TOKENS=77. If 77-token overfitting produces coherent images but 128-token fails, the issue is the extra-token mechanism. If 77-token also fails, the core connector architecture or data is broken.

### Q4: IS THE UNET COMPATIBLE WITH DRIFTING CONDITIONING?

The UNet was trained on CLIP embeddings. CLIP embeddings live in a specific 768-dim subspace with characteristic norm (~1.0). Our connector output:
- norm_ratio: 1.11 → 2.61 (output norm 2.6x CLIP norm by step 1500)
- delta_cos oscillates: 0.10 → 0.54 → 0.39 → 0.19 → 0.35 → 0.54

**Question**: The UNet's cross-attention KV projections were trained on CLIP-norm vectors. When the connector outputs vectors with 2.6x the expected norm, are the UNet's attention logits saturated? Does this explain the color-blob output — the UNet cross-attention produces near-uniform attention weights because all key vectors have inflated norms, making every spatial position attend equally to all context tokens?

**Decisive experiment**: After training, normalize connector output to unit norm before passing to UNet cross-attention. If images improve, norm drift is a major factor.

### Q5: ADDITIVE TIMESTEP VS AdaLN — DOES IT MATTER?

ELLA paper: timestep injected via AdaLN (Adaptive Layer Normalization) in each resampler block. The layer norm scale/bias are functions of timestep.
Our implementation: `q = q + temb` — timestep embedding added ONCE to query tokens before the first block.

**Question**: The ELLA paper's Table 6 shows AdaLN significantly outperforms other designs. Our additive timestep is more like the "no timestep" variant because the single additive bias gets diluted through 4 blocks of residual connections. Is the current additive timestep providing any meaningful timestep conditioning? How would you verify — by checking if connector output for timestep 0 vs 999 differs in a structured way?

**Decisive experiment**: Log the cosine similarity between connector outputs at timestep 0, 250, 500, 750, 999 for the same prompt. If similarity > 0.99 across all timesteps, the timestep conditioning is not working.

### Q6: WHY IS THE CLIP PRETRAIN NOT HELPING?

CLIP pretrain achieves best total loss = 0.983, pooled_cos = 0.813 — reasonable alignment for 77-token outputs. But then:
- ELLA phase has PHASE1_SEMANTIC_ANCHOR_WEIGHT = 0.0 — no anchor regularization
- delta_cos degrades from 0.125 (post-pretrain) to 0.541 (step 1500 ELLA) — 4.3x worse

**Question**: The CLIP pretrain aligns anchor tokens to CLIP space, but ELLA training immediately undoes this because there's no regularization. The connector is rewarded for producing outputs that minimize diffusion loss, not for staying in CLIP space. If the diffusion loss landscape in UNet-conditioning space has a "shortcut" (e.g., outputting a constant vector that makes the UNet predict zero noise), would the connector find it? Is `best=0.0046` suspiciously low — could the connector have learned to output a near-constant conditioning vector?

**Decisive experiment**: Log the variance of connector outputs across different prompts. If variance is near zero (all prompts produce similar conditioning), the connector has collapsed to a constant.

---

## RECOMMENDED MINIMAL FIXES (Before Architecture Changes)

Ordered by cost/benefit. Each is directly testable in one Colab run.

### Fix 1 (CRITICAL — 1 line change): Unfreeze the extra gate
**Change**: `CONNECTOR_EXTRA_GATE_INIT = 0.0` (from -3.0)
**Why**: Gate at -3.0 is a self-fulfilling prophecy. Initialize at 0.0 so extra tokens contribute equally. If they're useless, the gate will naturally drift down. If they're useful, they'll contribute.
**Expected effect**: extra_to_base_ratio should rise from 0.047 → toward 1.0. suffix_swap_rel_diff should increase.

### Fix 2 (CRITICAL — config change): Enable anchor regularization
**Change**: `PHASE1_SEMANTIC_ANCHOR_WEIGHT = 0.05` (from 0.0)
**Why**: Prevents anchor tokens from drifting outside the semantic subspace the UNet can interpret. The norm_ratio drift to 2.6x is a red flag.
**Expected effect**: norm_ratio should stay near 1.0. delta_cos oscillations should damp.

### Fix 3 (IMPORTANT — data pipeline): Verify caption lengths
**Change**: Add logging: print average/median token length of captions after Gemma tokenization.
**Why**: If captions are ~15 tokens, 128 context tokens are 88% padding. Extra query tokens learn from padding → dead.
**Expected finding**: Likely confirms data is too short for 128-token context.

### Fix 4 (IMPORTANT — 1 line diagnostic): Log output variance across prompts
**Change**: Add diagnostic: for 16 different prompts, compute variance of connector output. Log ratio of between-prompt variance to within-prompt noise.
**Why**: Tests whether connector collapsed to near-constant output (explaining color-only images).
**Expected finding**: If variance_ratio < 2.0, connector has collapsed.

### Fix 5 (IF FIX 3 CONFIRMS SHORT CAPTIONS): Switch to 77-token context first
**Change**: `CONTEXT_TOKENS = 77` with warm-start from CLIP pretrain checkpoint.
**Why**: Isolate whether the core Gemma→SD bridge works before adding long-context complexity.
**Expected effect**: If 77-token works but 128 doesn't, the extra-token mechanism is the problem. If neither works, the core connector or data is broken.

### Fix 6 (NICE TO HAVE): Overfit diagnostic
**Change**: `RUN_MODE = overfit_train`, `MAX_SAMPLES_ELLA = 64`, `ELLA_MAX_OPT_STEPS = 600`
**Why**: If the connector can't overfit 64 images, no amount of data will help.
**Expected effect**: Should produce recognizable versions of training images. If not, fundamental architecture issue.

---

## SUMMARY: THE CATCH-22

```
Gate init = -3.0 → extra_tokens ≈ 0.047 * raw_output
                      ↓
            Extra token gradient ∝ gate ≈ 0.047  (20x smaller than anchor)
                      ↓
            Extra query tokens barely update → stay near random init
                      ↓
            Extra token contribution stays negligible → gate gradient stays tiny
                      ↓
            Gate never moves — self-reinforcing death spiral
                      ↓
            Connector = effectively 77-token with 51 dead positions
                      ↓
            UNet cross-attention: 40% of attention goes to dead tokens
                      ↓
            Conditioning is ~60% drifting anchor + ~40% near-zero noise
                      ↓
            Color-blob images
```

Break the cycle at step 1: `CONNECTOR_EXTRA_GATE_INIT = 0.0`. Then verify there's actual content for extra tokens to learn (step 3: check caption lengths). Then prevent anchor drift (step 2: enable anchor regularization). Then test at 77-token scale first (step 5).

---

## PAPER-VS-IMPLEMENTATION GAP TABLE

| Aspect | ELLA Paper | Our Implementation | Gap Severity |
|--------|-----------|-------------------|--------------|
| Data volume | 34M pairs | 6K pairs | **CRITICAL** (5,700x less) |
| Caption type | CogVLM dense (avg 50 words, 62 tokens) | LAION alt-text (avg ~10 words, ~12 tokens) | **CRITICAL** (content mismatch) |
| Timestep conditioning | AdaLN in every block | Additive bias to queries (once) | HIGH (paper shows AdaLN is key) |
| TSC blocks | 6 blocks | 4 blocks | MEDIUM |
| Training steps | 140K-280K | 1,500 | **CRITICAL** (100x fewer) |
| LLM | T5-XL (encoder, 1.2B) | Gemma-3-270M (decoder, 270M) | MEDIUM (different architecture) |
| Base model | SDv1.5 | StyleJourney v10 (SD1.5-based) | LOW |
| Anchor mechanism | None (TSC output directly conditions UNet) | First 77 tokens gated differently from 78-128 | MEDIUM (our extra complexity) |
| CLIP teacher | Not used | Optional pretrain; disabled during ELLA | N/A (our extension) |
