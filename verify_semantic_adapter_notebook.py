#!/usr/bin/env python3
"""
Deterministic static verifier for the Gemma→CLIP semantic adapter / long-token / SaRA notebook path.

This does not run model training or import heavy ML dependencies. It verifies notebook JSON,
syntax for normal Python cells, and the specific code symbols/gates required by
.hermes/plans/2026-05-15-semantic-adapter-timestep-longtoken-sara-plan.md.
"""
import ast
import json
import re
import sys
from pathlib import Path

NB_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "gemma3_sd_colab.ipynb")


def load_nb(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def text_of(nb):
    return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))


def code_cells(nb):
    return [(i, "".join(c.get("source", []))) for i, c in enumerate(nb.get("cells", [])) if c.get("cell_type") == "code"]


def check(results, name, ok, detail=""):
    results.append((name, bool(ok), detail))


def strip_colab_shell(src):
    lines = []
    skip_cont = False
    for ln in src.splitlines():
        stripped = ln.lstrip()
        if skip_cont:
            if not stripped.endswith("\\"):
                skip_cont = False
            continue
        if stripped.startswith("!") or stripped.startswith("%"):
            skip_cont = stripped.endswith("\\")
            continue
        lines.append(ln)
    return "\n".join(lines)


def main():
    results = []
    nb = load_nb(NB_PATH)
    full = text_of(nb)
    check(results, "VALID_JSON_AND_CELLS", isinstance(nb.get("cells"), list), f"cells={len(nb.get('cells', []))}")

    syntax_errors = []
    for idx, src in code_cells(nb):
        stripped = strip_colab_shell(src)
        if not stripped.strip():
            continue
        try:
            ast.parse(stripped, filename=f"cell_{idx}")
        except SyntaxError as e:
            syntax_errors.append(f"cell {idx}: {e.msg} line {e.lineno}")
    check(results, "PYTHON_SYNTAX_EXCLUDING_COLAB_SHELL", not syntax_errors, "; ".join(syntax_errors[:5]))

    required = {
        "ARCH_OPTION_PRESENT": r'gemma_clip_semantic_adapter',
        "DEFAULT_ARCH_NEW_PATH": r'CONDITIONING_ARCH\s*=\s*"gemma_clip_semantic_adapter"',
        "PHASE_FLAGS_PRESENT": r'RUN_SEMANTIC_ADAPTER_TRAIN.*RUN_TIMESTEP_RESIDUAL.*RUN_LONG_TOKEN_PHASE.*RUN_SARA_PHASE',
        "SEMANTIC_CONFIG_PRESENT": r'SEMANTIC_NUM_TOKENS\s*=\s*77.*SEMANTIC_WIDTH\s*=\s*768.*SEMANTIC_LAYERS\s*=\s*4',
        "SEMANTIC_ADAPTER_CLASS": r'class\s+GemmaToClipSemanticAdapter',
        "PRENORM_CROSS_ATTN_BLOCK": r'class\s+SemanticCrossAttentionBlock.*nn\.MultiheadAttention',
        "MASK_TO_KEY_PADDING": r'key_padding_mask\s*=\s*~gemma_mask',
        "CLIP_GEOMETRY_LOSS": r'class\s+ClipGeometryLoss.*pooled_contrastive|class\s+ClipGeometryLoss.*cross_entropy',
        "GEOMETRY_COMPONENTS": r'"mse".*"cos".*"norm".*"ctr".*"pooled_cos"',
        "RETRIEVAL_DIAGNOSTIC": r'def\s+adapter_retrieval_accuracy',
        "COLLAPSE_DIAGNOSTIC": r'def\s+adapter_collapse_cosine',
        "SEMANTIC_PHASE_A_BRANCH": r'Phase A: Gemma→CLIP semantic adapter representation training',
        "NO_DIFFUSION_IN_SEMANTIC_PHASE_TEXT": r'No UNet/diffusion loss in this phase',
        "FROZEN_UNET_GENERATION": r'def\s+generate_semantic_adapter',
        "CLIP_TEACHER_BASELINE_GENERATION": r'def\s+generate_clip_teacher',
        "TIMESTEP_RESIDUAL_CLASS": r'class\s+TimestepResidualAdapter',
        "LONG_TOKEN_RESAMPLER_CLASS": r'class\s+LongTokenResampler',
        "LONG_CONTEXT_128_CONFIG": r'LONG_CONTEXT_TOTAL_TOKENS\s*=\s*128.*LONG_EXTRA_TOKENS',
        "WHOLE_UNET_SARA_SCOPE": r'SARA_SCOPE\s*=\s*"whole_unet_sparse"',
        "SARA_SUMMARY_FUNCTION": r'def\s+summarize_sara_masks',
        "SARA_WHOLE_UNET_CONSTRUCTION": r'sara_adamw\.AdamW\(\s*unet,',
        "SARA_WARN_ABORT_GATES": r'SARA_MAX_SPARSE_FRACTION_WARN.*SARA_MAX_SPARSE_FRACTION_ABORT',
        "SEMANTIC_CHECKPOINT": r'gemma_semantic_adapter_phase1\.pt',
        "TIMESTEP_CHECKPOINT": r'gemma_timestep_residual_phase1b\.pt',
        "LONG_CHECKPOINT": r'gemma_long_context_adapter_phase2\.pt',
        "SARA_CHECKPOINT": r'sara_unet_text_interface_sparse\.pt',
        "SEMANTIC_RELOAD_PROOF": r'Semantic adapter reload proof: PASS',
        "SEMANTIC_33_SMOKE_TEST": r'def\s+test_semantic_adapter_forward.*Semantic adapter frozen-UNet smoke test OK',
        "SEMANTIC_DIAGNOSTIC_BRANCH": r'elif\s+CONDITIONING_ARCH\s*==\s*"gemma_clip_semantic_adapter".*compute_semantic_adapter_prompt_sensitivity',
        "SEMANTIC_DELTA_ALIGNMENT_BRANCH": r'elif\s+CONDITIONING_ARCH\s*==\s*"gemma_clip_semantic_adapter"\s+and\s+semantic_adapter\s+is\s+not\s+None.*student_context\s*=\s*semantic_adapter',
        "SEMANTIC_OVERFIT_FIXED_LOSS_BRANCH": r'elif\s+CONDITIONING_ARCH\s*==\s*"gemma_clip_semantic_adapter".*context\s*=\s*semantic_adapter.*pred\s*=\s*unet\(noisy,\s*t,\s*encoder_hidden_states=context\)',
        "OVERFIT_PRETRAIN_GATE_AFTER_DATASET": r'=== Exact-overfit pre-training gate ===.*save_overfit_reference_grid\(\).*compute_fixed_overfit_eval_loss\(label="pretrain_overfit"\)',
        "NO_OVERFIT_GATE_IN_CELL_33_TEXT": r'Overfit reference/fixed-loss gates run at the start of Phase A',
        "SEMANTIC_POSTTRAIN_DELTA": r'=== Post-training semantic adapter diagnostics ===.*compute_teacher_student_delta_alignment\(DIAGNOSTIC_PROMPTS\[0\],\s*label="post_semantic"\)',
        "SEMANTIC_VALIDATION_DELTA": r'compute_teacher_student_delta_alignment\(VAL_PROMPTS\[0\],\s*label="validation_semantic"\)',
        "SEMANTIC_EXACT_OVERFIT_COMPARISON_GRID": r'validation_overfit_semantic_vs_clip_comparison\.png',
        "SEMANTIC_GENERIC_COMPARISON_GRID": r'validation_semantic_vs_clip_comparison\.png',
        "SEMANTIC_RELOADED_PROOF_GRID": r'samples_reloaded_semantic_adapter\.png',
        "SEMANTIC_CHECKPOINT_SELFTEST": r'Semantic adapter checkpoint self-test strict-reload: PASS',
        "SEMANTIC_RELOADED_PROOF_INFERENCE_MODE": r'with\s+torch\.inference_mode\(\):.*Proof reloaded semantic.*pred\s*=\s*unet\(inp,\s*t_step,\s*encoder_hidden_states=ctx\)',
        "CONNECTOR_RELOADED_PROOF_INFERENCE_MODE": r'with\s+torch\.inference_mode\(\):.*Proof reloaded connector.*pred\s*=\s*reloaded_unet\(inp,\s*t_step,\s*encoder_hidden_states=ctx\)',
    }
    for name, pattern in required.items():
        check(results, name, re.search(pattern, full, flags=re.S) is not None)

    # Regression check for the Colab crash where semantic mode fell through to connector smoke test.
    bad_fallthrough = re.search(
        r'if\s+CONDITIONING_ARCH\s*==\s*"dual_native".*?else:\s*\n\s*test_connector_forward\(\)',
        full,
        flags=re.S,
    )
    check(results, "NO_SEMANTIC_CONNECTOR_FALLTHROUGH", bad_fallthrough is None)

    # Ensure old architectures are still present, not deleted.
    check(results, "OLD_DUAL_NATIVE_PATH_RETAINED", 'CONDITIONING_ARCH == "dual_native"' in full)
    check(results, "OLD_CONNECTOR_PATH_RETAINED", 'CONDITIONING_ARCH == "ella_gemma_connector"' in full)

    # SaRA must not be enabled by default.
    check(results, "SARA_DISABLED_BY_DEFAULT", re.search(r'RUN_SARA_PHASE\s*=\s*False', full) is not None)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = [(n, d) for n, ok, d in results if not ok]
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"[{status}] {name}{suffix}")
    print(f"\nResults: {passed} pass, {len(failed)} fail, {len(results)} total")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
