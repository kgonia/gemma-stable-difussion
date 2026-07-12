# Roadmap: Gemma-SD from long prompts to metadata-conditioned, DiT-flavored UNet

*Created 2026-07-03, on branch `dual-conditioning`. Grounded in a code review of
`train.py`, `pure_ella/{connector,sara,config,dataset,diagnostics}.py` and
`config_long_256.json`.*

## Implementation update (2026-07-12)

The supported P1 baseline is now **256 Gemma input tokens -> 77 SD1.5
conditioning tokens**. `ella_tsc` is a fixed-query, timestep-aware resampler
with a learned mixture of upper Gemma layers; it keeps the pretrained U-Net
cross-attention contract intact. Direct 77 -> 256 token concatenation is not
supported because even zero-valued extra K/V tokens perturb attention softmax
normalization. A future extension must use a separately gated U-Net attention
branch, not concatenation.

Implemented Phase 0 corrections: SaRA uses `max_samples_sara`; checkpoint
resolution checks all local fallbacks before accepting a Hugging Face ID;
unknown JSON keys fail fast; run mode no longer overwrites explicit budgets;
full-frame aspect-ratio buckets replace center cropping; sparse masks are bool.

## Vision

Four capability pillars, in dependency order:

1. **P1 — Long Gemma prompts** via the ELLA timestep-aware connector (77 → 256 tokens).
2. **P2 — Train only undertrained weights** (SaRA sparse masks on `attn2` K/V) so new
   capability lands without destroying pretrained knowledge.
3. **P3 — Second conditioning signal**: camera intrinsics / capture metadata alongside
   the text prompt (Metric3D v2 + FINO inspiration: the image is under-determined;
   metadata resolves the ambiguity).
4. **P4 — DiT-ification (optional)**: extra transformer capacity in the UNet, inserted
   as zero-gated blocks.

**Guiding invariant** (already the house style — connector `extra_gate_init=-5.0`,
recursive/TRM gates, SaRA near-zero weights): *every new capability enters through
parameters that start at or near zero effect.* At init the model must be numerically
identical (or ε-close) to the previous milestone. This is both the knowledge-preservation
guarantee and the ablation story: each pillar is exactly removable.

---

## Current state (what the code already does)

- `train.py` runs three phases: Phase 0 CLIP-geometry pretrain of the connector
  (`run_clip_pretrain`), Phase 1 ELLA diffusion training with frozen UNet + CLIP
  teacher/delta distillation + optional suffix-counterfactual contrastive loss
  (`run_ella_training`), Phase 2 SaRA sparse `attn2.to_k/to_v` adaptation
  (`run_sara_training` — implemented, not a stub despite README wording).
- Three connector variants (`ella_tsc`, `recursive_y`, `trm_yz`), all gated for length
  extension past the 77-token CLIP anchor. `trm_yz` is the current default.
- `pure_ella/sara.py` builds |w| < 1e-3 masks with warn/abort gates (2% / 5%),
  gradient hooks, and sparse save/load — the reload proof applies the sparse patch to a
  freshly loaded UNet, which is exactly the right "removable delta" design.
- Diagnostics are strong: suffix sensitivity, extra-token ablation, teacher-student
  delta alignment, FID/KID, reload proofs.
- Dataset: `jackyhate/text-to-image-2M` streamed; captions from embedded JSON;
  512×512 single bucket; `num_workers=0`.
- `config_long_256.json` = stage3 (256 tokens, SaRA on), batch 8, StyleJourney v10.

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
3. **Refactor: `run_ella_training` and `run_sara_training` share ~200 duplicated lines**
   (the whole teacher-delta forward, loss assembly, logging, validation cadence).
   P3 requires touching the training step; doing it twice in two divergent copies is
   how bugs get in. Extract a single `diffusion_training_step(state, batch, phase_cfg)`
   and a shared loop driver before starting P3.
4. **Minor:** `ClipGeometryLoss()` is re-instantiated every step inside both loops
   (`train.py:602`, `train.py:882`) — hoist it. SaRA masks are stored at weight dtype
   (`sara.py:54`), duplicating every target tensor in fp32; store `bool` and cast in the
   hook. `unet_dtype` is hardcoded `float32` in `main()` — consider a config field
   (bf16 autocast) before the SaRA phase grows.
5. **Silent-config hazards (document or assert):** JSON keys that don't match a
   `TrainConfig` field are silently dropped (typos vanish); `context_tokens` in JSON is
   ignored (derived from `experiment_stage`/`long_context_target` — `config_long_256.json`
   says 77 but runs at 256, which is intended but reads as a lie); `run_mode:
   overfit_train`/`diagnostic` clobber JSON-provided budgets in `__post_init__`.

**Exit criteria:** SaRA phase respects its own sample budget; one shared training-step
function; short_train smoke run reproduces current metrics within noise.

---

## P1 — Long-context ELLA at 256 tokens (in flight — finish first)

This is the baseline every later pillar is measured against.

- Run `config_long_256.json` to completion (pretrain → ELLA → SaRA) with the suffix
  counterfactual loss enabled if `suffix_zero_delta` stays < 0.03 without it.
- **Exit criteria:** `rel_diff_full_vs_zeroextra_delta` ≥ 0.03 sustained; suffix
  counterfactual A/B grids visibly differ; FID/KID within tolerance of the 77-token
  stage; reload proof passes.
- **Deliverable:** `pure_ella_connector_L256.pt` + sparse UNet patch — frozen as the
  reference checkpoint ("B0") for all subsequent ablations.

## P2 — Undertrained-weight training (SaRA), widened deliberately

Already implemented for `attn2` K/V. The pillar-2 work is *disciplined widening*, not
new machinery:

1. Keep `attn2.to_k/to_v` only until P1 exit criteria are met and gains plateau.
2. Then widen the mask scope in controlled increments, re-using the same
   `sara_target_substrings` mechanism: `attn2.to_q` → `attn2.to_out` → `attn1` →
   feed-forward `ff.net`. One increment per experiment; keep the 5% abort gate.
3. Track a **knowledge-preservation metric** per increment: FID/KID and CLIP-teacher
   grids on *short* (77-token) prompts must not regress vs. B0. If they do, the
   increment is rolled back — the sparse-patch format makes this a file deletion.
4. Optional (paper-faithful): re-select the mask from *current* weights at the start of
   each increment rather than reusing stale masks — the "undertrained" set shifts as
   training proceeds.

**Exit criteria:** long-prompt metrics improve monotonically per increment; short-prompt
FID/KID within a fixed tolerance band (suggest ≤5% relative) of B0.

## P3 — Camera intrinsics / metadata conditioning (the `dual-conditioning` pillar)

### Design

Two injection points, in order of preference:

1. **Micro-conditioning through the timestep embedding** (SDXL pattern, primary).
   Fourier-embed each metadata scalar, concat, small MLP → add to the UNet time
   embedding. Focal length is a *global* scene-geometry property (see
   `docs/HLQJQx1acAAkFtd.png`: Z = S·f/s — same pixels, different depth), so global
   injection is semantically right. **The final MLP layer is zero-initialized** →
   at step 0 the model is exactly the P1/P2 checkpoint.
   - Condition on **vertical FOV** (or log-focal normalized by sensor height), not raw
     focal length in mm — resolution/crop invariant.
   - Candidate signals, most valuable first: FOV, capture type (photo/render/artwork as
     a learned embedding), aperture (DoF proxy), ISO (noise/low-light proxy).
2. **Metadata tokens through the connector** (secondary, only if global injection is
   insufficient): append k learned tokens conditioned on the metadata to the connector
   output. Costs nothing architecturally (cross-attn is length-agnostic) and reuses the
   existing `extra_gate_logit` pattern — but it entangles the ablation with P1's token
   budget, so keep it as a fallback.

**Conditioning dropout is mandatory:** drop the metadata signal (replace with a learned
"unknown" embedding) for 30–50% of training samples, because (a) inference will usually
have no EXIF, and (b) it enables metadata-CFG later. This is the FINO framing: metadata
guides representation learning; it must not become a hard dependency.

### The data problem (blocking — resolve before writing model code)

`jackyhate/text-to-image-2M` is largely **synthetic imagery — there is no real camera
EXIF to harvest**. Options, not mutually exclusive:

- **Pseudo-labeling (recommended default):** run a camera-intrinsics estimator
  (Metric3D-style canonical-camera reasoning; e.g. a WildCamera/GeoCalib-class model)
  over the training images to produce FOV pseudo-labels. Works on the existing pipeline;
  noisy labels are acceptable given conditioning dropout.
- **EXIF-bearing photo dataset mixed in:** Unsplash-style or LAION subsets with EXIF
  (FocalLength, FNumber, ISO, Model). Requires extending
  `StreamingSDDataset._get_caption`'s JSON parsing to also emit a `metadata` dict —
  the hook point already exists (`dataset.py:37`).
- **Cheap prompt-derived tags** as a v0: "wide-angle", "telephoto", "85mm portrait",
  "macro" regexed out of captions → coarse FOV buckets. Zero new data needed; good for
  a falsifier run before investing in pseudo-labels.

### Plan of record

1. v0 falsifier: caption-derived FOV buckets, timestep-embedding injection, zero-init,
   dropout 0.4, short_train. **Success test:** same prompt + seed, sweep the FOV input →
   monotonic, visible perspective/framing change; short-prompt FID unchanged.
2. v1: pseudo-labeled continuous FOV on the full stream; add a `fov_counterfactual`
   diagnostic mirroring `suffix_counterfactual_sensitivity` (same prompt, FOV A vs B,
   rel-diff on CFG delta) so the existing diagnostics culture covers the new signal.
3. v2 (only if v1 works): add aperture/ISO channels; optionally metadata tokens.

**Exit criteria:** FOV counterfactual sensitivity above threshold; generation with
"unknown" metadata identical-quality to B0; sweep grids show controlled perspective.

## P4 — DiT-ification via zero-gated blocks (optional, last, ROI-gated)

Do **not** insert raw new transformer blocks into SD 1.5 — fresh blocks are maximally
undertrained and directly contradict P2. If pursued:

- Insert blocks (extra self-attn depth in mid-block and lowest-resolution stages first —
  cheapest, most semantic) wrapped in `tanh(α)·f(x)` with α=0 or a zero-init output
  projection (Flamingo/ControlNet pattern). Identity at init, capacity opens on demand.
- Train the gates + new blocks with the P2 knowledge-preservation metric as a hard gate.
- **Go/no-go:** only start P4 if P1–P3 leave measurable headroom on long-prompt
  compositional metrics that connector/SaRA scaling demonstrably cannot close. Otherwise
  spend the compute on P2 increments and longer P1 training.

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
.pt, gated-block .pt) applied to the same frozen StyleJourney UNet — so any combination
can be composed or ablated by choosing which patches to load, extending the existing
`run_reload_proof` pattern to a compositional proof.

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
| Pseudo-labels too noisy → FOV signal ignored | P3 | conditioning dropout + v0 caption-bucket falsifier decides cheaply |
| Metadata leaks into text semantics (entanglement) | P3 | keep injection global (time-embed), not cross-attn, in v1 |
| P4 gates open and drag frozen knowledge | P4 | knowledge-preservation hard gate; gate-value logging like `extra_gate_logit` |
| Compute budget (single-GPU, fp32 UNet) | all | Phase 0 dtype config; gradient checkpointing already on |
