# Socratic Review A: Dual CLIP/Gemma Transition Plan

## Status of Code Reviewed
Notebook: gemma3_sd_colab.ipynb (33 cells, verifier: 20/22 pass)
Code matches the dual-branch plan described in context. The notebook is IN SPEC.
The code-level bugs flagged by the other review have been addressed:
  - Gemma masks ARE passed (set_dual_attention_context → module-level, then bool-cast in forward)
  - Gemma Q/out ARE copied from CLIP, not random (clone_linear_shape + strict load_state_dict)
  - Warmup uses clip_scale=0.0, gemma_scale=1.0 (pure Gemma-only student)
  - Pruned reload path exists with RUN_FINAL_PROOF=True

The code is mechanically correct. The failure is at the signal level.

================================================================================
WHAT IS PROVEN
================================================================================

P1. DUAL FORWARD CORRECTNESS
    CLIP-only equivalence max delta = 0 (identical output when gemma_scale=0).
    Gemma-only forward produces finite values (no NaN/Inf).
    → The architecture does not corrupt the signal pipeline.

P2. TEACHER-STUDENT MSE CONVERGENCE
    Loss ~0.096/0.100 after 1 epoch full-rank warmup.
    → The student learns to approximate the teacher's noise predictions.

P3. CHECKPOINT HYGIENE
    strict=True reload passes for both dual and pruned checkpoints.
    kv_bias=True, merged_lora=True, reload_note all present.
    → The artifact chain is structurally sound.

WHAT IS NOT PROVEN (AND LIKELY FALSE)
================================================================================

U1. GEMMA-ONLY PROMPT CONDITIONING EXISTS
    Evidence: prompt_sensitivity ~0.006 (different prompts produce near-identical
    UNet outputs). Generated images are brown ornament textures regardless of prompt.
    → The model does NOT condition on Gemma hidden states in a useful way.

U2. LOW LOSS MEANS MEANINGFUL LEARNING
    Counter-hypothesis: the student learns a near-constant noise prediction
    (the mean of the teacher's output distribution) rather than conditioning
    on the prompt. Low MSE is compatible with this collapse — if the teacher's
    outputs have limited variance across different captions, predicting the
    mean gives deceptively low loss.
    → The loss is NOT a sufficient signal of conditioning fidelity.

U3. GEMMA_LAYER_INDEX = -1 PROVIDES USEFUL REPRESENTATIONS
    The final hidden layer of a 270M causal LM is optimized for next-token
    prediction. These representations may be:
    - Collapsed across token positions (all tokens converge to similar vectors)
    - Dominated by positional/nuisance variance rather than semantic content
    → Not yet tested.

================================================================================
WHAT ASSUMPTIONS DIFFER FROM THE PLAN
================================================================================

A1. "LayerNorm is sufficient to normalize Gemma states for cross-attention"
    REALITY: LayerNorm centers and scales per-token, but does NOT change the
    geometry or discriminability of the representation space. If Gemma last-layer
    hidden states have low pairwise variation across prompts, LayerNorm cannot fix it.

A2. "Teacher distillation MSE is a sufficient training signal"
    REALITY: The teacher produces noise predictions, not a representation target.
    If the noise predictions are intrinsically similar across captions (noise is
    noise), the student can achieve low loss by learning a constant function.
    The teacher loss term may reinforce mode collapse rather than conditioning.

A3. "1 epoch full-rank + 1 epoch LoRA is enough training"
    REALITY: Compared to the CLIP calibration approach (which failed because the
    linear map collapsed), this is a much harder problem — the student must learn
    a nonlinear mapping from a 640-dim alien representation space to UNet noise
    space. The CLIP path has millions of pretrained parameters. 1 epoch of
    student-only training on a tiny parameter subset is unlikely to succeed.

A4. "CLIP-scaffold training transfers to Gemma-only inference"
    REALITY: The scaffold is present only as a loss signal during training.
    At inference, the Gemma branch must fly solo. If the teacher loss merely
    taught the student to output the average teacher prediction (see U2),
    there is nothing to transfer.

================================================================================
ROOT CAUSE HYPOTHESES (ordered by probability × fixability)
================================================================================

H1. GEMMA LAST-LAYER REPRESENTATIONS ARE NOT DISCRIMINATIVE ENOUGH (HIGH)
    The final hidden layer of Gemma 3 270M is a 26-layer causal LM. Token
    representations at the final layer may be:
    - Near-deterministic given the prefix (low entropy)
    - Dominated by next-token prediction features, not semantic embeddings
    - Highly correlated across different prompts at the CLS-equivalent position
    FALSIFIER: Measure pairwise cosine similarity of Gemma hidden states
    across 10+ diverse prompts. If mean cos_sim > 0.85, representations
    lack the variation needed for conditioning. Try middle layers (6-12).

H2. K/V INITIALIZATION SCALE IS MISMATCHED (MEDIUM)
    K/V are random-init with std=0.02 (CLIP convention). But after Gemma
    LayerNorm, the hidden states may have different variance than CLIP
    hidden states. If the dot-product attention scores are too small or
    too uniform, the attention pattern carries no prompt-specific signal.
    FALSIFIER: Log mean/max/min of attention weights in gemma_attn during
    a forward pass. If the attention distribution is near-uniform (entropy
    close to log(seq_len)), increase K/V init std to 0.1 or use
    xavier_uniform_ with Gemma hidden states as a scaling reference.

H3. TEACHER LOSS COLLAPSED TO CONSTANT FUNCTION (MEDIUM)
    The student minimizes MSE(student, teacher). At early training, the
    student outputs near-zero (small random weights). The optimal strategy
    is to output the mean teacher prediction. Once converged to this
    mean, the gradient w.r.t. prompt variation is zero — the model never
    learns to condition.
    FALSIFIER: Before training, compute teacher_pred variance across 10
    different prompts at t=500. If std < 0.01, the teacher itself has low
    prompt sensitivity and the loss signal is genuinely weak. Add a
    contrastive or triplet loss term that explicitly penalizes the model
    for producing identical outputs for different prompts.

H4. INSUFFICIENT TRAINING CAPACITY/TIME (LOW, BUT EASY FIX)
    1 epoch × 1000 steps × batch_size=1 = ~1000 samples. The K/V projections
    have 640×in_features parameters each, which is ~400K params per dual
    block. With 16 dual blocks, that's ~6.4M trainable K/V params. 1000
    samples is not enough to learn a meaningful nonlinear mapping.
    FALSIFIER: Train for 5 epochs instead of 1. If loss decreases further
    but sensitivity stays at ~0.006, capacity is not the bottleneck.

================================================================================
FALSIFICATION GATES (in execution order — each is cheap)
================================================================================

GATE 0: TEACHER BASELINE DIAGNOSTIC
    Cost: 1 forward pass. Time: 30 seconds.
    Action: Run prompt_sensitivity() with clip_scale=1.0, gemma_scale=0.0
    (teacher-only) on 3+ diverse prompts at t=500.
    PASS if: teacher relative diff > 0.02 (teacher has meaningful sensitivity)
    FAIL if: teacher relative diff < 0.01 (the metric itself is broken;
             try t=250 or t=750, or add a noise-free teacher output comparison)
    CRITICAL: If this gate fails, all student sensitivity measurements are
    uninterpretable. Fix the diagnostic first.

GATE 1: GEMMA REPRESENTATION DISCRIMINABILITY
    Cost: 1 Gemma forward pass. Time: 5 seconds.
    Action: Encode 10 semantically diverse prompts (e.g., "a red car",
    "a blue ocean", "quantum physics equation", "a barking dog"). Compute
    mean-pooled hidden states, then pairwise cosine similarity matrix.
    PASS if: mean pairwise cos_sim < 0.7 (representations are discriminable)
    FAIL if: mean pairwise cos_sim > 0.85 (too similar → try GEMMA_LAYER_INDEX=6
             and GEMMA_LAYER_INDEX=12; if all layers fail, Gemma 3 270M last
             layer is simply not suitable and we need a different approach)
    ACTION on FAIL: Test layers 0, 6, 12, 18, 24. Also test concatenation
    of multiple layers. Also test token-position diversity (do different
    token positions carry different information?).

GATE 2: INITIALIZATION SCALE CHECK
    Cost: 1 forward pass with logging. Time: 10 seconds.
    Action: Add a hook to gemma_attn to log attention weight statistics
    (mean, std, min, max, entropy) during a Gemma-only forward pass with
    untrained K/V weights.
    PASS if: attention_entropy > 0.3 * log(seq_len) (meaningful soft assignment)
    FAIL if: attention weights are near-uniform (entropy ~log(seq_len), all
            tokens attend equally → no position-specific signal)
    ACTION on FAIL: Scale up K/V initialization (try std=0.1, 0.2) or use
    a learnable temperature in the attention.

GATE 3: TEACHER VARIANCE DIAGNOSTIC
    Cost: 10 teacher forward passes. Time: 2 minutes.
    Action: Compute teacher_pred (CLIP-only) for 10 different prompts at
    t=500. Report per-element variance and L2 norm of pairwise differences.
    PASS if: mean pairwise L2 diff / mean norm > 0.02
    FAIL if: teacher predictions are near-identical across prompts (teacher
            itself has low conditioning variance at this timestep)
    ACTION on FAIL: The teacher distillation signal is inherently weak.
    Switch to loss on decoded images (perceptual loss) or train at multiple
    timesteps including low-noise t where conditioning matters more.

GATE 4: STUDENT TRAINING DYNAMICS
    Cost: 30 minutes of training. Time: ~30 min.
    Action: Track prompt_sensitivity every 100 training steps during warmup.
    PASS if: sensitivity increases monotonically (at step 0 it's near 0;
             by step 1000 it should be > 0.01 and rising)
    FAIL if: sensitivity stays flat at ~0.006 from step 0 to step 1000
             (model is NOT learning conditioning at all)
    ACTION on FAIL: Stop. The current loss function is not teaching
    conditioning. Try:
    a) Replace teacher MSE with a contrastive loss (push different prompts apart)
    b) Add an auxiliary task: predict CLIP hidden states from Gemma hidden
       states (a small probe head) — gives a direct representation-learning signal
    c) Try a different Gemma layer
    d) Unfreeze to_q/to_out from the start (more capacity)

GATE 5: END-TO-END GENERATION QUALITY
    Cost: 3 generations. Time: 2 minutes.
    Action: Generate images after training with 3 prompts that differ
    semantically. Visually inspect.
    PASS if: images are recognizably different and correspond to prompts
    FAIL if: images are the same brown/texture pattern regardless of prompt
    NOTE: Gate 5 is the final integration test. Gates 0-4 should pass first.

================================================================================
RECOMMENDED NEXT EXPERIMENT (smallest falsifier, avoids full UNet training)
================================================================================

BEFORE ANY MORE TRAINING, run Gates 0-3 in a single Colab cell:

  # === GATE 0: Teacher sensitivity ===
  prompts = ["a red sports car", "a snowy mountain at dawn",
             "an underwater coral reef", "a rustic Italian kitchen",
             "abstract geometric shapes in neon colors"]
  prompt_sensitivity(prompts, clip_scale=1.0, gemma_scale=0.0)
  # EXPECT: relative diffs > 0.02 for most pairs

  # === GATE 1: Gemma discriminability ===
  import torch.nn.functional as F
  hidden_list = []
  for p in prompts:
      gh, _ = encode_gemma_prompts([p])
      hidden_list.append(gh.mean(dim=1))  # mean pool over tokens
  cos_sim_matrix = torch.zeros(len(prompts), len(prompts))
  for i in range(len(prompts)):
      for j in range(len(prompts)):
          cos_sim_matrix[i,j] = F.cosine_similarity(
              hidden_list[i], hidden_list[j], dim=-1)
  print("Mean pairwise cos_sim:", cos_sim_matrix[triu_indices].mean().item())
  # EXPECT: < 0.7 (if > 0.85, Gemma last layer is the bottleneck)

  # === GATE 2: Attention entropy ===
  # (requires a forward hook on gemma_attn — 10 lines of code)
  # EXPECT: attention not uniform

  # === GATE 3: Teacher output variance ===
  teacher_preds = []
  for p in prompts:
      clip_h, clip_mask = encode_clip_prompts([p])
      set_dual_attention_context(unet, clip_scale=1.0, gemma_scale=0.0)
      pred = unet(noisy, t, encoder_hidden_states=clip_h,
                  encoder_attention_mask=clip_mask).sample
      teacher_preds.append(pred)
  pairwise_diffs = []
  for i in range(len(teacher_preds)):
      for j in range(i+1, len(teacher_preds)):
          diff = (teacher_preds[i]-teacher_preds[j]).pow(2).mean().sqrt()
          pairwise_diffs.append(diff.item())
  print("Mean teacher pairwise diff:", np.mean(pairwise_diffs))
  print("Teacher pred std:", torch.stack(teacher_preds).std().item())
  # EXPECT: mean diff > 0.01

THE MOST LIKELY OUTCOME: Gate 1 fails (Gemma cos_sim > 0.85). The fix is to
try GEMMA_LAYER_INDEX in {6, 12, 18} — middle layers of the transformer which
typically encode richer semantic content. If all layers fail, the conditional
recommendation is to add a trainable projection head (small MLP: 640→512→640)
between Gemma hidden states and the cross-attention input, trained jointly
with the K/V weights. This gives the UNet more degrees of freedom to discover
useful structure in Gemma's representation space.

================================================================================
PLAN REFINEMENT: WHAT TO CHANGE IN THE NOTEBOOK
================================================================================

1. ADD: Gemma layer sweep diagnostic (Gate 1) as a mandatory pre-training check.
   If cos_sim > 0.85, abort and try other layers.

2. ADD: Teacher baseline sensitivity (Gate 0) before any training. If teacher
   sensitivity is also low, the prompt_sensitivity() diagnostic at t=500 is
   inadequate — switch to t=250 or full-generation comparison.

3. ADD: Attention entropy logging during training. If attention collapses to
   uniform, the model has found a trivial solution. Add entropy regularization.

4. CONSIDER: Replace pure teacher MSE with a hybrid loss:
   - Teacher MSE (λ=0.5, decaying)
   - Contrastive loss: push student predictions for different prompts apart
   - Diffusion noise MSE (λ=0.25, kept)
   The contrastive term explicitly prevents mode collapse.

5. CONSIDER: Increase training from 2 epochs to 5-10 epochs. The current
   regime (1 epoch full-rank + 1 epoch LoRA ≈ 2000 steps at batch_size=1)
   is probably insufficient for the representation gap between CLIP and
   Gemma hidden spaces.

6. DEFER: Pruned checkpoint generation until Gate 5 (end-to-end quality) passes.
   Currently the notebook runs save+prune+reload even when sensitivity is ~0.006.

================================================================================
BOTTOM LINE
================================================================================

The dual-branch code is correct. The failure is that the Gemma-only student learns
a constant function rather than conditioning on the prompt. The most likely root
cause is that Gemma 3 270M final-layer hidden states lack sufficient semantic
discriminability (they're optimized for next-token prediction, not semantic
embedding). The cheapest next step is to run Gates 0-3 (30 seconds of compute)
to isolate whether the bottleneck is in the Gemma representations, the teacher
signal, or the training dynamics. Do NOT re-train until these diagnostics pass.

The dual-branch plan itself is sound — we just need to confirm that the Gemma
representations contain enough information for the UNet to condition on. If they
don't, no amount of CLIP scaffolding or teacher distillation will help.
