# Notebook Migration Verifier — Check Reference

## File
`verify_notebook_migration.py` — run with: `python verify_notebook_migration.py <notebook.ipynb>`

## Check Summary

### CATEGORY 1: Syntax & Structural Integrity
| # | Check Name | What It Catches |
|---|-----------|----------------|
| 1 | `ALL_CELLS_VALID_STRUCTURE` | Malformed cells (missing cell_type/source, source not a list) |
| 2 | `NO_SOURCE_BARE_CR` | Bare `\r` characters in source (not `\r\n`) — corrupts rendering |
| 3 | `NO_SOURCE_NULL_BYTES` | Null bytes in notebook file — indicates binary corruption |
| 4 | `ALL_SOURCE_LINES_STRINGS` | Non-string elements in source arrays (JSON editing artifact) |
| 5 | `NO_EMBEDDED_SOURCE_NEWLINES` | One Jupyter `source` list element contains multiple physical lines — JSON editing artifact |
| 6 | `CELL_COUNT_REASONABLE` | Notebook too small (<15) or too large (>50) after migration |

### CATEGORY 2: Stale Calibration Path Detection
| # | Check Name | What It Catches |
|---|-----------|----------------|
| 7 | `NO_STALE_CALIBRATION_PATH` | Any reference to `clip_gemma_calib.pt` — the removed artifact |
| 8 | `NO_CALIBRATION_CAPTIONS_LIST` | The 60+ string `CALIBRATION_CAPTIONS` list — must be deleted |
| 9 | `NO_CLIP_CALIBRATION_DELETE_REFS` | Persistent CLIP scaffold is required. Forbids `del clip_model`, `del clip_tokenizer`, and old calibration-only/delete wording; requires CLIP refs to exist |
| 10 | `NO_W_BAKE_SOLVE_CODE` | Ridge regression: `GTG`, `GTC`, `torch.linalg.solve`, `gather_pair`, `calibration_metrics` |
| 11 | `NO_CALIBRATION_SECTION_HEADER` | Markdown headers mentioning "CLIP→Gemma baked initialization" or "Section 2B" |

### CATEGORY 3: Dual-Native Attention Presence
| # | Check Name | What It Catches |
|---|-----------|----------------|
| 12 | `DUAL_NATIVE_ATTENTION_SYMBOLS` | Must find ≥1 of: `GemmaNativeAttention`, `DualAttention`, `dual_attn`, `to_k_gemma`, `to_v_gemma`, `gemma_native_attn`, class definitions matching `Native.*Attention` |
| 13 | `NO_KAIMING_FALLBACK` | Surgery cell must NOT contain `kaiming_uniform_` or "Kaiming fallback" — dual-native replaces this |
| 14 | `DUAL_ATTN_KV_BIAS_TRUE` | K/V projections must have `bias=True` or `uses_kv_bias=True` — required for strict state_dict reload |

### CATEGORY 4: Cell Ordering
| # | Check Name | What It Catches |
|---|-----------|----------------|
| 15 | `CELL_ORDERING` | 15 expected phases in monotonic order: Drive Mount → Env Setup → Imports → HF Login → Load Gemma → UNet Surgery → Verify Forward → LoRA Training → Streaming Dataset → Full-Rank Warmup → Manual LoRA → Training Loop → Inference → Save Checkpoint → Smoke Test |

### CATEGORY 5: Checkpoint Reloadability Markers
| # | Check Name | What It Catches |
|---|-----------|----------------|
| 16 | `CHECKPOINT_HAS_RELOAD_NOTE` | `reload_note` key or "Rebuild SD1.5 UNet... then load_state_dict" instruction in checkpoint metadata |
| 17 | `CHECKPOINT_HAS_USES_KV_BIAS` | `uses_kv_bias=True` flag in at least one checkpoint save dict |
| 18 | `CHECKPOINT_HAS_MERGED_LORA_FLAG` | `merged_lora=True` or `merge_lora_linear` function — proves LoRA is folded into base weights |
| 19 | `SMOKE_TEST_HAS_STRICT_TRUE` | `load_state_dict(strict=True)` in the reload smoke test cell |
| 20 | `SMOKE_TEST_CELL_PRESENT` | Cell containing "Reload Merged Checkpoint Smoke Test", `RUN_RELOAD_SMOKE_TEST`, or `apply_gemma_kv_shape_for_reload` |

### CATEGORY 6: Additional Integrity
| # | Check Name | What It Catches |
|---|-----------|----------------|
| 21 | `NO_RUNTIME_OUTPUTS_ON_KEY_CELLS` | Surgery/checkpoint/smoke-test cells with stale execution outputs (warn only) |
| 22 | `GEMMA_MODEL_ID_CONSISTENT` | All `gemma_path` and `gemma_model_id` references use the same string |

## Current Status (pre-migration)

**`gemma3_sd_colab.ipynb`**: 13 pass / 8 fail / 1 warn
- Fails: 5-9 (calibration section intact), 12-13 (no dual-native attention yet)
- Passes: structure, ordering, checkpoint markers (the notebook already has good reload hygiene)

**`gemma3_sd_colab - Copy.ipynb`**: 12 pass / 10 fail
- Additional failures: 14-20 (no checkpoint reload markers at all, wrong ordering)

## Migration Targets

After migration, ALL checks must pass. The migration should:
1. Delete cells 9-12 (Section 2B: calibration + W solve + CLIP delete)
2. Rewrite cell 15 (UNet Surgery) to use dual-native attention instead of Kaiming fallback
3. Add at least one dual-native attention symbol (class definition or module import)
4. Ensure K/V bias=True for strict reload compatibility
5. Keep all checkpoint reload markers from the original (they're already good)
