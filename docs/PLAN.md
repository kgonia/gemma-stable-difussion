# Roadmap: Gemma-SD from long prompts to metadata-conditioned, DiT-flavored UNet

*Created 2026-07-03, on branch `dual-conditioning`. Grounded in a code review of
`train.py`, `pure_ella/{connector,sara,config,dataset,diagnostics}.py` and
`config_long_256.json`.*

## Implementation update (2026-07-13)

### LongCLIP direct branch (2026-07-15)

LongCLIP-L is now a separate evidence path, not a Gemma-connector phase.
`scripts/train_longclip_sara.py` freezes the official 248-token, 768-wide
LongCLIP encoder and feeds its states directly to StyleJourney's unchanged
cross-attention interface; only sparse SaRA-selected `attn2.to_k/to_v` U-Net
entries train. This tests whether small U-Net adaptation improves an already
CLIP-compatible long-text encoder without introducing a translation connector.

The direct LongCLIP baseline must be compared against native StyleJourney CLIP
before applying SaRA. The full SaRA run is gated by deterministic held-out
masked diffusion loss (fixed rows, VAE modes, noise, and timestep grid), the
four canonical visual prompts, strict LongCLIP/StyleJourney content hashes, and
short-prompt regression checks. LongCLIP+Gemma is intentionally out of scope;
ordinary prompt lengths do not justify stacking both long-context mechanisms.

The current P1 path is **CLIP-preserving Gemma residual conditioning**. Native
CLIP remains the SD1.5 context and `clip_gemma_residual_tsc` learns a
timestep-aware correction to its fixed 77 slots from long Gemma input. Its
output projection starts exactly at zero, so the initial residual context is
identical to CLIP. Prompts within CLIP's window take a control-flow bypass:
Gemma and the connector are not run. Training uses prefix-null and
CLIP-RMS-normalized residual-norm penalties. The pure `ella_tsc` replacement
connector below is retained as a historical baseline, not the supported P1
production direction.

Before launching `config_clip_gemma_residual_p1.json`, create the immutable
group-safe split (the script refuses a row-level split when no suitable image
identity is present):

```bash
uv run python scripts/prepare_residual_holdout_split.py \
  /mnt/e/data/unsplash-lite-gemma-captions/artifacts/production/unsplash_lite_gemma_captions_10000.parquet \
  --output-dir /mnt/e/data/unsplash-lite-gemma-captions/artifacts/residual_p1_split
```

Evaluate a completed checkpoint with
`scripts/evaluate_residual_paired.py`; it uses identical held-out images,
latents, noise, and timesteps for residual and native CLIP predictions. Also
run `scripts/verify_checkpoint_text_encoder.py <stylejourney.safetensors>`
before treating OpenAI CLIP as the StyleJourney-native baseline.

The supported P1 baseline is now **approximately 256-token Gemma captions
(336-token safety window) -> 77 SD1.5 conditioning tokens**. `ella_tsc` is a
fixed-query, timestep-aware resampler
with a learned mixture of upper Gemma layers; it keeps the pretrained U-Net
cross-attention contract intact. Direct 77 -> 256 token concatenation is not
supported because even zero-valued extra K/V tokens perturb attention softmax
normalization. A future extension must use a separately gated U-Net attention
branch, not concatenation.

Implemented Phase 0 corrections: SaRA uses `max_samples_sara`; checkpoint
resolution checks all local fallbacks before accepting a Hugging Face ID;
unknown JSON keys fail fast; run mode no longer overwrites explicit budgets;
full-frame aspect-ratio buckets replace center cropping; sparse masks remain
bool through gradient hooks. Input-side suffix diagnostics now verify that the
counterfactual suffix begins after token 77 and is not truncated. Phase 0 uses
the exact CLIP-visible decoded prefix; Phase 1 samples paired short/medium/long
captions at 25/25/50 when the dataset supplies them. Gemma encoding performs one
tokenizer pass per batch, and suffix boundaries fail fast once at startup.
Phase 1 and SaRA now use one shared diffusion forward/loss function and loop
driver. Model weight dtype and BF16 forward autocast are explicit configuration
fields; supplied configs retain FP32 trainable weights and optimizer state while
using BF16 CUDA activations. Bucket dimensions are validated as multiples of 64,
and printed step plans include the upper bound from per-bucket tail batches.

Phase 0 now matches the actual SD1.5 conditioning contract: all 77 CLIP hidden
states, including the causally contextualized padding positions consumed by the
U-Net, receive token-geometry supervision. The real-token mask is retained only
for semantic pooling and contrastive alignment. ELLA-TSC is trained at uniformly
sampled diffusion timesteps, and best-checkpoint validation covers fixed
timesteps from 0 through 999. CLIP and connector U-Net diagnostics both use the
same unmasked 77-token cross-attention path. The earlier masked, `t=0`-only
objective left most output slots unconstrained and caused structured grain in
Phase-0-only generations; it is retained only in historical checkpoints, not as
a supported training mode.

P3 camera conditioning is implemented behind `camera_conditioning_enabled`.
It adds a Fourier/MLP class-embedding branch to the UNet timestep embedding,
with an algebraically centered unknown residual that remains exactly zero after
optimizer updates. The camera-only template uses zero record dropout, keeps the
text connector frozen, and disables the unsupported photo/render/artwork input.
All JSON metadata containers are merged; guided-prediction counterfactuals are
the primary diagnostic and CFG-delta sensitivity is secondary. Camera artifacts
now carry an exact versioned semantic schema and incompatible camera state fails
at load. The P3/CLIP moving-teacher combination is rejected by config validation.

Camera schema v3 separates trusted FOV, experimental raw focal, exposure, and
optional capture into independently centered heads. Runtime 35 mm-equivalent
conversion is removed; manifests must provide axis-specific `vertical_fov_deg`.
ELLA and SaRA have separate metadata-dropout settings validated after CLI phase
selection. `config_camera_p3.json` is diagnostic-only. The former 25k Unsplash
run is quarantined in `config_camera_raw_focal_experimental.json` and requires an
explicit experimental opt-in.

The committed local audit and sensor table prove a record-level conjunction of
288 strictly validated geometry records, not the earlier provisional 1,191.
The next implementation gate before P3a/P3b training is the versioned offline
manifest builder, including label-null controls and camera/lens holdouts. Controlled
paired or synthetic data remains required before any physical-control claim.

## Vision

Four capability pillars, in dependency order:

1. **P1 — Long Gemma prompts** via the ELLA timestep-aware connector
   (up to 336 input tokens -> 77 fixed SD1.5 conditioning tokens).
2. **P2 — Train only undertrained weights** (SaRA sparse masks on `attn2` K/V) so new
   capability lands without destroying pretrained knowledge.
3. **P3 — Second conditioning signal**: camera intrinsics / capture metadata alongside
   the text prompt (Metric3D v2 + FINO inspiration: the image is under-determined;
   metadata resolves the ambiguity).
4. **P4 — DiT-ification (optional)**: extra transformer capacity in the UNet, inserted
   as exact-zero-gated residual-delta blocks, with axial 2D RoPE as the preferred
   positional variant after a no-position control.

**Guiding invariant** (already the house style — connector `extra_gate_init=-5.0`,
recursive/TRM gates, SaRA near-zero weights): *every new capability enters through
parameters that start at or near zero effect.* At init the model must be numerically
identical (or ε-close) to the previous milestone. This is both the knowledge-preservation
guarantee and the ablation story: each pillar is exactly removable.

---

## Current state (what the code already does)

- `train.py` runs three phases: Phase 0 prompt-only CLIP-geometry pretrain of
  the connector across the full 77-token U-Net contract and diffusion timestep
  range (`run_clip_pretrain`), Phase 1 standard diffusion training with
  a frozen UNet and no suffix-divergence objective
  (`run_ella_training`), Phase 2 SaRA sparse `attn2.to_k/to_v` adaptation
  (`run_sara_training` — implemented, not a stub despite README wording).
- `ella_tsc` is the supported default. It compresses long Gemma input into 77
  fixed queries; `recursive_y` and `trm_yz` remain experimental variants.
- `pure_ella/sara.py` supports the legacy |w| threshold and exact global
  magnitude-rank selection. Shipped runs select 10% of the `attn2` K/V scope
  with 5% minimum, 15% warning, and 25% abort gates; reports also show the much
  smaller whole-U-Net fraction. Masks remain boolean through gradient hooks,
  selected-weight delta norms are recorded after training, and the reload proof
  applies the sparse patch to a freshly loaded U-Net.
- Diagnostics cover input-suffix sensitivity, short-prompt teacher-student delta
  alignment, FID/KID, and reload proofs. Output-token ablation is intentionally
  obsolete because the connector always emits exactly 77 tokens.
- Dataset: multiple local Parquet/Hugging Face locations, streamed round-robin
  into 1024-edge full-frame buckets with masked letterbox padding and optional
  `caption_short`/`caption_medium`/`caption_long` variants.
- `config_long_256.json` = connector-only long-input baseline, batch 1,
  StyleJourney v10; SaRA is deferred to a separate stage.

---

## Phase 0 — Fixes and refactors before adding anything (small, do first)

Findings from the code review (full write-up with file:line references and fix
suggestions in [`REVIEW.md`](REVIEW.md)), ordered by importance:

1. **Bug: `max_samples_sara` is never used.** `run_sara_training` uses
   `cfg.max_samples_ella` both for the step plan (`train.py:800`) and the dataloader
   (`train.py:817`). `config_long_256.json` sets `max_samples_sara: 40000` expecting it
   to cap the SaRA phase — it silently doesn't. Fix: thread `max_samples_sara` through.
2. **Bug: broken checkpoint fallback chain.** `resolve_sd_checkpoint`
   (`pure_ella/config.py:294`) returns any non-existent path containing `/` as if it
   were a HF model ID — so if `/workspace/stylejourney_v10.safetensors` is missing, the
   function returns that bogus "HF ID" instead of trying the `~/models/...` fallback,
   and the run dies later with a confusing HF 404. Fix: only treat as HF ID if it
   doesn't look like a filesystem path (no leading `/` or `~`, ≤1 slash).
3. **Refactor (fixed): `run_ella_training` and `run_sara_training` shared ~200 duplicated lines**
   (the whole teacher-delta forward, loss assembly, logging, validation cadence).
   P3 requires touching the training step; doing it twice in two divergent copies is
   how bugs get in. Both phases now call `diffusion_training_step` through a
   shared phase-configured loop driver.
4. **Minor (fixed):** `ClipGeometryLoss()` is instantiated once per phase. SaRA
   masks remain bool through their gradient hooks. Model weight dtype and BF16
   autocast are configured separately; all training modes require FP32 master
   weights so connector/SaRA updates and Adam state remain full precision.
5. **Silent-config hazards (fixed):** unknown JSON keys fail fast;
   `context_tokens` is validated as the fixed 77-token SD1.5 output contract; run
   modes no longer overwrite explicit sample or optimizer-step budgets.

**Exit criteria:** SaRA phase respects its own sample budget; one shared training-step
function; short_train smoke run reproduces current metrics within noise. CLIP-prefix
pretrain, connector-only diffusion, and SaRA have each passed real-image, one-step
GPU smoke runs; production-length metric validation remains part of P1/P2.

---

## P1 — Historical pure ELLA replacement baseline

This is the baseline every later pillar is measured against.

- Train `config_long_256.json` with standard diffusion MSE after the short-prefix
  CLIP-geometry pretrain. Keep Phase 1 CLIP teacher/delta losses disabled: CLIP cannot
  observe suffix tokens beyond its 77-token window and would reward suffix blindness.
- Use paired caption variants when available: 25% short, 25% medium, 50% long. Target
  approximately 256 Gemma tokens while retaining the configured 336-token safety
  window; fail instead of silently truncating longer samples.
- **Exit criteria:** suffix A/B counterfactual sensitivity is non-trivial and sustained;
  compositional long-prompt grids visibly bind suffix attributes; short-prompt FID/KID
  stays within tolerance of the CLIP baseline; connector reload proof passes.
- **Deliverable:** a Gemma-336-input/SD1.5-77-output connector checkpoint, frozen as
  the reference checkpoint ("B0") for subsequent SaRA and metadata ablations.

## P2 — Undertrained-weight training (SaRA), widened deliberately

Already implemented for `attn2` K/V. The initial capacity experiment uses an exact
global magnitude rank rather than treating the accidental 2.33% selected by
`|w| < 1e-3` as a safety boundary. The pillar-2 work is *disciplined widening*, not
new machinery:

1. Keep `attn2.to_k/to_v` only until P1 exit criteria are met and gains plateau.
2. Sweep 5%, 10%, and 20% of K/V first. Then widen the mask scope in controlled
   increments, re-using the same
   `sara_target_substrings` mechanism: `attn2.to_q` → `attn2.to_out` → `attn1` →
   feed-forward `ff.net`. One increment per experiment; the 25% target-scope
   abort gate prevents accidental broad updates while quality metrics remain the
   actual knowledge-preservation gate.
3. Track a **knowledge-preservation metric** per increment: FID/KID and CLIP-teacher
   grids on *short* (77-token) prompts must not regress vs. B0. If they do, the
   increment is rolled back — the sparse-patch format makes this a file deletion.
4. Optional (paper-faithful): re-select the mask from *current* weights at the start of
   each increment rather than reusing stale masks — the "undertrained" set shifts as
   training proceeds. A mid-run replacement must save the union of every updated
   mask so a reloaded sparse patch is identical to the live U-Net.

**Exit criteria:** long-prompt metrics improve monotonically per increment; short-prompt
FID/KID within a fixed tolerance band (suggest ≤5% relative) of B0.

## P3 — Camera intrinsics / metadata conditioning (the `dual-conditioning` pillar)

The measured two-source data and training strategy is specified in
[`P3_MIXED_DATA_PROPOSAL.md`](P3_MIXED_DATA_PROPOSAL.md).
That proposal supersedes the older P3 notes below wherever they conflict. In
particular, camera-only training uses an algebraically centered unknown residual
and therefore no whole-record metadata dropout.

### Design

Two injection points, in order of preference:

1. **Micro-conditioning through the timestep embedding** (SDXL pattern, primary).
   Fourier-embed each metadata scalar, concat, small MLP → add to the UNet time
   embedding. Focal/FOV is a *global projection* property, so global injection is
   semantically right. Every head is a permanently centered residual
   `head(value) - head(unknown)`, not merely zero-initialized. Metadata-free
   generation therefore remains exactly on the P1/P2 path after training.
   - Condition the trusted head on manifest-provided **vertical FOV**, not raw
     focal length in mm. Any 35 mm-equivalent conversion must be axis- and
     aspect-aware preprocessing, never a training-time 24 mm assumption.
   - Candidate signals, most valuable first: trusted FOV/35 mm equivalent,
     aperture, exposure time, ISO and flash. Disable photo/render/artwork until a
     source with meaningful render and artwork coverage is added.
2. **Metadata tokens through the connector** (secondary, only if global injection is
   insufficient): append k learned tokens conditioned on the metadata to the connector
   output. Costs nothing architecturally (cross-attn is length-agnostic) and reuses the
   existing `extra_gate_logit` pattern — but it entangles the ablation with P1's token
   budget, so keep it as a fallback.

Whole-record conditioning dropout is useful only when the connector or U-Net is
also trainable. In a camera-only phase, a centered unknown residual has zero loss
gradient for a fully dropped record, so dropout only wastes examples. Natural
per-field missingness still trains partial records. Use a small dropout only in
the optional joint-refinement phase to preserve metadata-free behavior in the
newly unfrozen path.

### Data status

The Unsplash smoke dataset has broad scenes and exposure metadata but no trusted
FOV. The local PicturesTraining set has about 1.2k focal-plane records but only
288 currently pass strict dimension and sensor provenance, with narrow camera
and photographer coverage. Neither observational set
can isolate physical FOV from camera distance, subject, and composition. Raw
focal length remains a correlation-only feature and must not share the trusted
geometry head. The reproducible audit, manifest provenance, neutral captions,
sampling arithmetic, and controls are specified in the mixed-data proposal.

### Plan of record

1. **P3a association/memorization falsifier:** local high-confidence geometry,
   camera-only head, real labels versus shuffled-label and constant-label controls.
2. **P3b observational cross-source generalization:** reproducible 80/20 offline
   manifests, camera/lens holdouts, trusted geometry plus exposure heads.
3. **P3c controlled physical validation:** paired same-scene focal sweeps or
   controlled synthetic renders, preferably with pose/depth. Only this stage can
   support a claim of physical FOV control; FOV changes projection/framing at a
   fixed pose, not camera-position perspective.
4. **P3d optional joint refinement:** lower-rate connector or separately gated
   SaRA update, with metadata dropout restored for the unfrozen path.
5. Only if global conditioning demonstrably saturates, evaluate separately gated
   metadata tokens.

**Exit criteria:** unknown residual is exactly zero after optimization; real
labels outperform shuffled and constant controls on held-out diffusion and
guided-control metrics; metadata-free quality stays within the B0/P1 tolerance.
Observational P3a/P3b results are reported as FOV-correlated control, not isolated
physical control.

## Resolution conditioning sidecar (separate from P3 and P4)

Width/height conditioning is a separate sidecar, not P3 camera metadata and not P4
capacity. It declares the requested canvas geometry globally; RoPE/relative position in
P4 instead describes relationships among spatial tokens. Keep these ablations separate
so improvements remain attributable.

Resolution-conditioner v1 uses only the two independent emitted-canvas features:

```text
log2(width / 1024)
log2(height / 1024)
```

Use the bucket/canvas size actually emitted to the U-Net, not raw source image size.
Do not include `log_aspect` or `log_area` in v1 because they are exact linear
combinations of the two log-dimension features and may encourage nine-bucket
memorization. Derived features are a later ablation only. The sidecar should have an
unknown/dropout path and its own schema, checkpoint, validation grids, and on/off
generation switch.

## P4 — DiT-ification via exact-zero-gated blocks (optional, last, ROI-gated)

Do **not** insert raw new transformer blocks into SD 1.5 — fresh blocks are maximally
undertrained and directly contradict P2. P4 is allowed only as an identity-safe delta
artifact when P1/P2/P3 or the resolution sidecar leave measurable long-prompt or layout
headroom.

### Identity and branch design

Use exactly one identity mechanism, at the output of the whole new branch:

```text
y = x + tanh(g) * F(x), with g initialized exactly to 0
```

The exact-zero gate gives bit-exact baseline behavior at step 0 and still receives a
gradient because `F(x)` is nonzero. Do **not** combine this with a zero-initialized
output projection, an internal zero residual gate, or any modulation that zeros the
branch output; that creates a gradient deadlock or an unnecessarily slow staged opening.
The output projection and internal attention/FFN residuals must be nonzero-capable at
initialization.

`F(x)` should be a residual delta, not a learned remapping of the whole input stream:

```text
z0 = input_projection(x)
delta_attn = self_attention(adaln_or_norm(z0))
z1 = z0 + delta_attn
delta_ffn = ffn(adaln_or_norm(z1))
F(x) = output_projection(delta_attn + delta_ffn)
```

Equivalently, project `z_final - z0`. This prevents P4 from opening as a simple learned
rescale/remix of `x`; the gated branch represents new computation.

P4 should be timestep-aware using the existing SD timestep embedding, preferably through
AdaLN/scale-shift inside the branch. Enforce the same one-zero rule there: use ordinary
nonzero-capable attention/FFN branches. If scale/shift projections are zero-initialized,
they must be formulated as standard transformer modulation (`normalized * (1 + scale) +
shift`), not as an internal gate that zeros the branch.

### Positional variants

P4a is the mandatory attribution control: identical block, no explicit positional
encoding. It tests capacity alone.

P4b is the preferred serious P4 design: the same block and initialization, but Q/K
self-attention uses axial 2D RoPE. This gives parameter-free relative spatial geometry
native to every bucket shape, with no learned position table, no interpolation, no extra
input channels, and no width/height-conditioning entanglement.

RoPE contract:

- require `head_dim % 4 == 0`;
- split each attention head's Q/K dimensions equally between vertical and horizontal
  axes, with rotary pairs inside each axis slice;
- rotate Q and K only, never V;
- generate integer feature coordinates `y = 0..H-1`, `x = 0..W-1` from the actual
  feature map shape;
- use row-major flattening with origin at the upper-left corner and record that
  convention in the schema;
- generate/cache sin/cos by `(H, W, device, dtype)`, computing frequencies in FP32 and
  casting for attention;
- record `rope_base` in the schema (`10000` is a reasonable initial control);
- no learned relative-bias table and no absolute DiT-style sin/cos grid in P4b.

Do not use the training `image_mask` as a P4 attention mask initially. It exists because
training samples are padded, while pure text-to-image inference has no content mask;
masking P4 attention would add another training/inference mismatch.

### Insertion ladder for SD1.5

Avoid ambiguous "lowest-resolution stage" language. In SD1.5, the deepest down-block and
the mid-block operate at the same spatial resolution (for a 1024x640 bucket: 128x80
latent -> 16x10 deepest grid). The sites differ architecturally, not by grid size.

1. **P4 block 1:** after `down_blocks[-1]`, before `mid_block`. This is adjacent to
   the existing mid-block transformer, improves the representation entering the existing
   mid-block text/image processing, and is the friendliest first insertion site. Record
   the exact module path in the schema and assert the observed feature shape at runtime.
2. **P4 block 2, only if justified:** after `mid_block`, before the first up block, for
   post-mid global refinement at the same spatial resolution.
3. **Later only:** the next 32x20 level if same-resolution P4 passes preservation and
   composition gates.

### Ablation and gates

The approved ladder is:

1. Monet4 baseline.
2. Resolution conditioner using the two independent log-dimension features.
3. P4a: one external-zero-gated delta transformer, no positional encoding.
4. P4b: identical non-RoPE initialization and training stream, axial 2D RoPE enabled.
5. Second P4 insertion only if the first passes preservation and composition gates.
6. P4b + resolution conditioner only after both individual effects are established.

The resolution-conditioner arm and P4b arm are independent and may run in parallel
against the same baseline if compute permits. P4a is an attribution control for P4b, not
a prerequisite gate that P4b must wait on. For a credible RoPE comparison, P4a and P4b
must use the same base checkpoint, same initialization seed for every non-RoPE parameter,
same samples/order/timesteps/noise seeds/optimizer/budget, identical parameter counts,
and the same frozen SaRA endpoint during the initial P4 comparison.

Before any full-epoch arm, run the existing smoke pattern (`smoke_10`/`smoke_100` style)
with telemetry on. Required smoke checks:

- step-0 outputs are bit-identical to the baseline with the P4/resolution sidecar loaded;
- P4 gate values move off zero;
- gate gradients are nonzero at step 0;
- branch parameter gradient norms become nonzero after the gate opens;
- P2 knowledge-preservation metrics and per-bucket validation/contact sheets are logged.

**Go/no-go:** only scale P4 if P1/P2/P3 and the resolution sidecar leave measurable
headroom on long-prompt compositional or layout metrics that connector/SaRA scaling
cannot close. If gates barely open after one epoch over the current 51k-image Monet4 mix,
that is a capacity-vs-data finding, not automatically an implementation failure. For
large-bucket coherence defects, also check the SD1.5 schedule/SNR confound before blaming
positional encoding.

---

## Sequencing and ablation matrix

```
Phase 0 (fixes)  →  P1 finish (B0 checkpoint)  →  P2 increments (B1..Bn)
                                   ↘
                                    P3 v0 falsifier → v1 → v2   (branches off B0/B1)
                                                        ↘
                                                         P4 (only if headroom proven)
```

Every milestone ships as a *delta artifact* (connector .pt, sparse patch, metadata-MLP
.pt, resolution-sidecar .pt, gated-block .pt) applied to the same frozen StyleJourney
UNet — so any combination can be composed or ablated by choosing which patches to load,
extending the existing `run_reload_proof` pattern to a compositional proof.

## Related directions considered

- **Semantic-First Diffusion (SeFi-Image / SFD)** — dual-latent generation: a DINOv2-derived
  semantic latent is channel-concatenated with the texture latent and denoised jointly, with
  the semantic stream leading by a timestep offset. **Rejected for this project**, for two
  reasons. (1) It changes the denoiser's *input contract* (new `conv_in`/`conv_out`, joint
  velocity field, dual-timestep embedding) — there is no zero-init/ε-close-to-identity way to
  bolt that onto the frozen StyleJourney UNet, so it violates the guiding invariant harder
  than any P4 block would. (2) The semantic latent is image-derived at training time but must
  be *generated* at inference; SFD does this via joint diffusion, which we can't have — the
  alternative is a separate text→semantic-latent generator (RCG pattern), a second model
  whose training cost exceeds P3+P4. What survives in spirit: a REPA-style DINOv2 alignment
  loss on UNet mid-block features during P1/P2 (training-time only, removable by
  construction), and — at most a P3-v3 bullet — a pooled semantic embedding injected through
  the same zero-gated P3 machinery, gated on the connector learning to predict it from text.

## Risks

| Risk | Pillar | Mitigation |
|------|--------|-----------|
| Suffix sensitivity stays < 0.03 at 256 tokens | P1 | contrastive loss weight ↑, gate init ↑ (already at −3.5), P2 widening |
| SaRA widening degrades short prompts | P2 | per-increment FID gate, instant rollback (delete patch) |
| Pseudo-labels too noisy -> FOV signal ignored | P3 | keep them as a separate ablation against real, shuffled and constant-label controls |
| Metadata leaks into text semantics (entanglement) | P3 | keep injection global (time-embed), not cross-attn, in v1 |
| Resolution conditioning memorizes buckets | resolution sidecar | v1 uses only log2(width/1024), log2(height/1024); derived features only in later ablations |
| P4 gates open and drag frozen knowledge | P4 | exact-zero external gate, step-0 identity smoke, knowledge-preservation hard gate, gate and branch-gradient logging |
| P4 positional result is training variance | P4 | paired P4a/P4b seeds, order, timesteps, noise, optimizer, budget, and frozen SaRA endpoint |
| Compute budget (single GPU) | all | FP32 trainable weights + BF16 autocast; gradient checkpointing already on |
