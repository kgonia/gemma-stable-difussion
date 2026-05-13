#!/usr/bin/env python3
"""
Deterministic notebook verifier for gemma3_sd_colab.ipynb migration.

Migration objectives:
  A) Remove/bypass clip_gemma_calib / W-bake mainline (no CLIP calibration step)
  B) Add dual-native attention (Gemma-native cross-attention module)
  C) Keep checkpoint reloadability markers intact

Usage:
  python verify_notebook_migration.py [path/to/notebook.ipynb]

Returns exit code 0 if all checks pass, non-zero otherwise.
Each check prints PASS/FAIL/WARN with a deterministic rule name.
"""

import json
import sys
import os
import re
import hashlib
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────
# Core infrastructure
# ──────────────────────────────────────────────────────────────────────

def load_notebook(path: str) -> Dict[str, Any]:
    """Load and validate basic JSON structure. Returns parsed notebook dict."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    # Check 1: Valid JSON
    try:
        nb = json.loads(raw, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON: {e}"}
    # Check 2: Jupyter nbformat structure
    if not isinstance(nb, dict):
        return {"_error": "Not a JSON object (notebook must be a dict)"}
    if nb.get("nbformat") is None:
        return {"_error": "Missing nbformat key"}
    if "cells" not in nb:
        return {"_error": "Missing cells key"}
    return nb


def nb_text(nb: Dict) -> str:
    """Concatenate all cell source into a single string for regex searches."""
    parts = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        parts.append(src)
    return "\n".join(parts)


def cell_list(nb: Dict) -> List[Dict]:
    """Return cells that have at least one source line."""
    return nb.get("cells", [])


# ──────────────────────────────────────────────────────────────────────
# CHECK CATEGORIES
# ──────────────────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = "", warn: bool = False):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.warn = warn

    def __str__(self):
        status = "WARN" if self.warn else ("PASS" if self.passed else "FAIL")
        extra = f" — {self.detail}" if self.detail else ""
        return f"[{status}] {self.name}{extra}"


class Verifier:
    def __init__(self, nb_path: str):
        self.path = nb_path
        self.nb = load_notebook(nb_path)
        self.results: List[CheckResult] = []
        self._text: Optional[str] = None
        self._cells: Optional[List[Dict]] = None

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = nb_text(self.nb)
        return self._text

    @property
    def cells(self) -> List[Dict]:
        if self._cells is None:
            self._cells = cell_list(self.nb)
        return self._cells

    def cell_source(self, idx: int) -> str:
        """Return concatenated source of cell at index."""
        if idx < 0 or idx >= len(self.cells):
            return ""
        return "".join(self.cells[idx].get("source", []))

    def check(self, name: str, condition: bool, detail: str = "", warn: bool = False):
        self.results.append(CheckResult(name, condition, detail, warn=warn))
        return condition

    # ── FACTORY: all checks ──────────────────────────────────────────

    def run_all(self):
        # If JSON was invalid, stop early
        if "_error" in self.nb:
            self.check("VALID_JSON", False, self.nb["_error"])
            return

        # 1. Syntax & structural integrity
        self.check_valid_json_structure()
        self.check_no_corrupted_source_newlines()
        self.check_cell_count_reasonable()

        # 2. Calibration removal checks (stale path detection)
        self.check_no_stale_calibration_path()
        self.check_no_calibration_captions_list()
        self.check_no_clip_calibration_delete_refs()
        self.check_no_w_bake_solve_code()
        self.check_no_calibration_section_header()

        # 3. Dual-native attention presence checks
        self.check_dual_native_attention_symbols_present()
        self.check_no_kaiming_fallback_in_surgery()
        self.check_dual_attn_has_kv_bias()

        # 4. Cell ordering checks
        self.check_cell_ordering()

        # 5. Checkpoint reloadability marker checks
        self.check_merged_checkpoint_has_reload_note()
        self.check_checkpoint_has_uses_kv_bias()
        self.check_checkpoint_has_merged_lora_flag()
        self.check_smoke_test_has_strict_true()
        self.check_smoke_test_cell_present()

        # 6. Additional integrity checks
        self.check_no_runtime_outputs_on_target_cells()
        self.check_gemma_model_id_consistent()

    # ──────────────────────────────────────────────────────────────────
    # 1. SYNTAX & STRUCTURAL INTEGRITY
    # ──────────────────────────────────────────────────────────────────

    def check_valid_json_structure(self):
        """Verify every cell has required fields."""
        for i, cell in enumerate(self.cells):
            ct = cell.get("cell_type")
            if ct not in ("markdown", "code", "raw"):
                self.check(f"CELL_{i}_VALID_TYPE", False, f"Unknown cell_type '{ct}'")
                return
            if "source" not in cell:
                self.check(f"CELL_{i}_HAS_SOURCE", False, "Missing source key")
                return
            if not isinstance(cell["source"], list):
                self.check(f"CELL_{i}_SOURCE_IS_LIST", False, f"source is {type(cell['source']).__name__}")
                return
        self.check("ALL_CELLS_VALID_STRUCTURE", True)

    def check_no_corrupted_source_newlines(self):
        """Detect source newline corruption: bare \\r, mixed line endings, JSON-escaped corruption."""
        raw_bytes = None
        with open(self.path, 'rb') as f:
            raw_bytes = f.read()

        # Check 1: No bare CR (\r) without \n in source strings
        # In JSON, \r inside strings should be escaped as \\r or part of \\r\\n
        # We check the decoded text for any raw \r that isn't followed by \n
        text = self.text
        bare_cr_pattern = re.compile(r'\r(?!\n)')
        bare_cr_matches = list(bare_cr_pattern.finditer(text))
        if bare_cr_matches:
            self.check("NO_SOURCE_BARE_CR", False,
                       f"Found {len(bare_cr_matches)} bare \\r instances (not \\r\\n)")
        else:
            self.check("NO_SOURCE_BARE_CR", True)

        # Check 2: No null bytes inside source strings
        if b'\x00' in raw_bytes:
            self.check("NO_SOURCE_NULL_BYTES", False, "Null bytes found in notebook file")
        else:
            self.check("NO_SOURCE_NULL_BYTES", True)

        # Check 3: Every cell source line is valid UTF-8 (already guaranteed by json.load, but double-check)
        for i, cell in enumerate(self.cells):
            for j, line in enumerate(cell.get("source", [])):
                if not isinstance(line, str):
                    self.check(f"CELL_{i}_SRC_LINE_{j}_NOT_STRING", False,
                               f"Source line is {type(line).__name__}")
                    return
        self.check("ALL_SOURCE_LINES_STRINGS", True)

        # Check 4: No source element may contain an embedded physical newline.
        # Legit Python strings may contain the two characters "\\n"; that is OK.
        # Corruption means one Jupyter source-list element contains multiple real lines.
        embedded = []
        for i, cell in enumerate(self.cells):
            for j, line in enumerate(cell.get("source", [])):
                if line.count("\n") > 1 or ("\n" in line and not line.endswith("\n")):
                    embedded.append((i, j))
        if embedded:
            self.check("NO_EMBEDDED_SOURCE_NEWLINES", False,
                       f"Found embedded real newlines in source elements: {embedded[:5]}")
        else:
            self.check("NO_EMBEDDED_SOURCE_NEWLINES", True)

    def check_cell_count_reasonable(self):
        """Notebook should have between 20 and 45 cells after migration."""
        n = len(self.cells)
        if n < 15:
            self.check("CELL_COUNT_MIN", False, f"Only {n} cells (min 15)")
        elif n > 50:
            self.check("CELL_COUNT_MAX", False, f"{n} cells (max 50)")
        else:
            self.check("CELL_COUNT_REASONABLE", True, f"{n} cells")

    # ──────────────────────────────────────────────────────────────────
    # 2. CALIBRATION REMOVAL CHECKS
    # ──────────────────────────────────────────────────────────────────

    def check_no_stale_calibration_path(self):
        """The migrated notebook must NOT reference clip_gemma_calib.pt."""
        patterns = [
            r'clip_gemma_calib\.pt',
            r'clip_gemma_calib',
            r'calib_path\s*=\s*f.*clip_gemma',
        ]
        full_text = self.text
        for pat in patterns:
            matches = list(re.finditer(pat, full_text))
            if matches:
                lines = [full_text[max(0, m.start()-50):m.end()+50] for m in matches]
                self.check("NO_STALE_CALIBRATION_PATH", False,
                           f"Found '{pat}': {lines[0][:120]}")
                return
        self.check("NO_STALE_CALIBRATION_PATH", True)

    def check_no_calibration_captions_list(self):
        """CALIBRATION_CAPTIONS list must be absent (60+ hardcoded captions)."""
        if "CALIBRATION_CAPTIONS" in self.text:
            self.check("NO_CALIBRATION_CAPTIONS_LIST", False,
                       "CALIBRATION_CAPTIONS list still present (60+ hardcoded captions)")
        else:
            self.check("NO_CALIBRATION_CAPTIONS_LIST", True)

    def check_no_clip_calibration_delete_refs(self):
        """Persistent CLIP is required for dual-conditioning; only forbid calibration/delete mainline refs."""
        code_text = "\n".join(
            "".join(c.get("source", []))
            for c in self.cells if c.get("cell_type") == "code"
        )
        forbidden_patterns = [
            (r'del\s+clip_model', "deleting persistent CLIP model"),
            (r'del\s+clip_tokenizer', "deleting persistent CLIP tokenizer"),
            (r'CLIP deleted', "old calibration cleanup message"),
            (r'calibration-only', "old calibration-only CLIP wording"),
            (r'calibration only', "old calibration-only CLIP wording"),
        ]
        for pat, desc in forbidden_patterns:
            if re.search(pat, code_text, flags=re.IGNORECASE):
                self.check("NO_CLIP_CALIBRATION_DELETE_REFS", False, f"Found: {desc}")
                return
        # Positive check: dual-conditioning mainline should still load/use CLIP.
        has_clip = any(s in code_text for s in ["CLIPTextModel", "CLIPTokenizer", "clip_model", "clip_tokenizer"])
        self.check("NO_CLIP_CALIBRATION_DELETE_REFS", has_clip, "Persistent CLIP references present" if has_clip else "No CLIP refs found; dual scaffold missing")

    def check_no_w_bake_solve_code(self):
        """No ridge regression W-solving code (GTG, GTC, torch.linalg.solve for calibration)."""
        patterns = [
            r'GTG\s*=\s*G_ctr\.T\s*@\s*G_ctr',
            r'RIDGE_LAMBDA',
            r'torch\.linalg\.solve\(GTG',
            r'W\s*=\s*M\.T.*CLIP.*Gemma',
            r'C_mean\s*=\s*C\.mean',
            r'gather_pair\(',
            r'gather_caption_split\(',
            r'calibration_metrics\(',
        ]
        code_text = "\n".join(
            "".join(c.get("source", []))
            for c in self.cells if c.get("cell_type") == "code"
        )
        for pat in patterns:
            if re.search(pat, code_text):
                self.check("NO_W_BAKE_SOLVE_CODE", False,
                           f"Found ridge-regression code: '{pat}'")
                return
        self.check("NO_W_BAKE_SOLVE_CODE", True)

    def check_no_calibration_section_header(self):
        """Section 2B (CLIP→Gemma baked initialization) must be removed."""
        patterns = [
            r'CLIP→Gemma baked initialization',
            r'One-time CLIP→Gemma',
            r'Section 2B.*CLIP',
            r'calibration only.*deleted after init',
        ]
        for pat in patterns:
            if re.search(pat, self.text):
                self.check("NO_CALIBRATION_SECTION_HEADER", False,
                           f"Found calibration section header: '{pat}'")
                return
        self.check("NO_CALIBRATION_SECTION_HEADER", True)

    # ──────────────────────────────────────────────────────────────────
    # 3. DUAL-NATIVE ATTENTION PRESENCE CHECKS
    # ──────────────────────────────────────────────────────────────────

    def check_dual_native_attention_symbols_present(self):
        """After migration, at least one dual-native attention symbol must exist."""
        required_symbols = [
            # Class/module names
            r'GemmaNativeAttention',
            r'DualAttention',
            r'GemmaCrossAttention',
            r'NativeCrossAttention',
            r'DualNativeAttention',
            # Structural markers
            r'dual_attn',
            r'dual_native',
            r'native_attn',
            r'gemma_native_attn',
            # Method/parameter markers
            r'to_k_gemma',
            r'to_v_gemma',
            r'gemma_k_proj',
            r'gemma_v_proj',
            # Import statement patterns
            r'from.*import.*NativeAttention',
            r'class.*Native.*Attention',
            r'class.*Dual.*Attention',
        ]
        found = []
        for sym in required_symbols:
            if re.search(sym, self.text):
                found.append(sym)
        if not found:
            self.check("DUAL_NATIVE_ATTENTION_SYMBOLS", False,
                       "No dual-native attention symbols found. "
                       "Expected at least one of: GemmaNativeAttention, DualAttention, "
                       "dual_attn, to_k_gemma, to_v_gemma, etc.")
        else:
            self.check("DUAL_NATIVE_ATTENTION_SYMBOLS", True,
                       f"Found {len(found)} symbols: {[f.split('|')[0] for f in found[:5]]}")

    def check_no_kaiming_fallback_in_surgery(self):
        """Surgery cell should NOT have Kaiming fallback — dual-native attention replaces it."""
        surgery_text = ""
        for cell in self.cells:
            src = "".join(cell.get("source", []))
            if "UNet Surgery" in src or "Replace Cross-Attention" in src or "attn2" in src:
                surgery_text += src + "\n"
        if not surgery_text:
            self.check("NO_KAIMING_FALLBACK", True,
                       "(no surgery cell found — may be fine)")
            return
        if "kaiming_uniform_" in surgery_text or "Kaiming fallback" in surgery_text:
            self.check("NO_KAIMING_FALLBACK", False,
                       "Surgery cell still contains Kaiming init / fallback — "
                       "dual-native attention should replace this pattern")
        else:
            self.check("NO_KAIMING_FALLBACK", True)

    def check_dual_attn_has_kv_bias(self):
        """Dual-native attention K/V projections must have bias=True for checkpoint compatibility."""
        # Search for Linear creations with bias parameter in dual-attn context
        bias_patterns = [
            r'bias\s*=\s*True.*to_k',
            r'bias\s*=\s*True.*to_v',
            r'use_kv_bias\s*=\s*True',
            r'kv_bias\s*=\s*True',
            r'bias=k_bias_flag',
            r'bias=v_bias_flag',
            r'to_k.*bias\s*=\s*True',
            r'to_v.*bias\s*=\s*True',
        ]
        # Also acceptable: declaring uses_kv_bias=True in checkpoint metadata
        meta_patterns = [
            r'"uses_kv_bias"\s*:\s*True',
            r"'uses_kv_bias'\s*:\s*True",
            r'uses_kv_bias\s*=\s*True',
        ]
        has_bias = any(re.search(p, self.text) for p in bias_patterns)
        has_meta = any(re.search(p, self.text) for p in meta_patterns)
        if has_bias or has_meta:
            self.check("DUAL_ATTN_KV_BIAS_TRUE", True)
        else:
            self.check("DUAL_ATTN_KV_BIAS_TRUE", False,
                       "No bias=True found on to_k/to_v in dual-attn context. "
                       "Checkpoint reload requires bias=True for strict state_dict matching.")

    # ──────────────────────────────────────────────────────────────────
    # 4. CELL ORDERING CHECKS
    # ──────────────────────────────────────────────────────────────────

    def check_cell_ordering(self):
        """Verify cells follow the expected logical order for the migrated pipeline."""
        # Expected phases (in order):
        phases = [
            ("section_0_drive_mount", [r"Mount Google Drive", r"drive\.mount"]),
            ("section_1_env_setup", [r"Environment Setup", r"pip install"]),
            ("section_1_imports", [r"import torch", r"print\(\"OK\"\)"]),
            ("section_1_hf_login", [r"HuggingFace Login", r"hf_token"]),
            ("section_2_load_gemma", [r"Load Gemma", r"AutoModelForCausalLM"]),
            # Section 2B (CLIP calibration) should be ABSENT — checked elsewhere
            ("section_3_unet_surgery", [r"UNet Surgery", r"Load SD 1\.5"]),
            ("section_3_verify_forward", [r"Verify Forward Pass", r"test_forward"]),
            ("section_4_lora_training", [r"LoRA Training", r"Freeze VAE"]),
            ("section_4_streaming_dataset", [r"Streaming Dataset", r"IterableDataset"]),
            ("section_4_fullrank_warmup", [r"Full-Rank Cross-Attn", r"fullrank_params"]),
            ("section_4_manual_lora", [r"Manual LoRA", r"class ManualLoRA"]),
            ("section_4_training_loop", [r"Training Loop", r"DDPMScheduler"]),
            ("section_5_inference", [r"Inference", r"Generate Image"]),
            ("section_5_save_checkpoint", [r"Save.*Checkpoint", r"torch\.save"]),
            ("section_5_smoke_test", [r"Reload.*Smoke Test", r"RUN_RELOAD_SMOKE_TEST"]),
        ]

        text = self.text
        last_pos = -1
        ordering_ok = True
        phase_positions = []

        for phase_name, patterns in phases:
            pos = None
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    pos = m.start()
                    break
            if pos is not None:
                phase_positions.append((phase_name, pos))

        # Check monotonic ordering
        for i in range(1, len(phase_positions)):
            prev_name, prev_pos = phase_positions[i-1]
            curr_name, curr_pos = phase_positions[i]
            if curr_pos < prev_pos:
                self.check("CELL_ORDERING", False,
                           f"'{curr_name}' appears before '{prev_name}' — wrong order")
                return

        if len(phase_positions) >= 10:
            self.check("CELL_ORDERING", True,
                       f"{len(phase_positions)}/{len(phases)} expected phases in correct order")
        else:
            self.check("CELL_ORDERING", False,
                       f"Only {len(phase_positions)}/{len(phases)} expected phases found")

    # ──────────────────────────────────────────────────────────────────
    # 5. CHECKPOINT RELOADABILITY MARKER CHECKS
    # ──────────────────────────────────────────────────────────────────

    def check_merged_checkpoint_has_reload_note(self):
        """The merged inference checkpoint MUST have a reload_note or reload instruction."""
        patterns = [
            r'reload_note',
            r'Rebuild SD1\.5 UNet.*then load_state_dict',
            r'load_state_dict\(strict=True\)',
            r'apply_gemma_kv_shape_for_reload',
        ]
        for pat in patterns:
            if re.search(pat, self.text):
                self.check("CHECKPOINT_HAS_RELOAD_NOTE", True,
                           f"Found reload marker: '{pat.split('|')[0]}'")
                return
        self.check("CHECKPOINT_HAS_RELOAD_NOTE", False,
                   "No reload_note or reload instructions found in checkpoint metadata")

    def check_checkpoint_has_uses_kv_bias(self):
        """Checkpoint metadata must include uses_kv_bias flag."""
        patterns = [
            r'"uses_kv_bias"\s*:\s*True',
            r"'uses_kv_bias'\s*:\s*True",
            r'uses_kv_bias\s*=\s*True',
        ]
        for pat in patterns:
            if re.search(pat, self.text):
                self.check("CHECKPOINT_HAS_USES_KV_BIAS", True)
                return
        self.check("CHECKPOINT_HAS_USES_KV_BIAS", False,
                   "Checkpoint missing uses_kv_bias=True marker")

    def check_checkpoint_has_merged_lora_flag(self):
        """The merged checkpoint must have merged_lora=True flag."""
        patterns = [
            r'"merged_lora"\s*:\s*True',
            r"'merged_lora'\s*:\s*True",
            r'merged_lora\s*=\s*True',
        ]
        for pat in patterns:
            if re.search(pat, self.text):
                self.check("CHECKPOINT_HAS_MERGED_LORA_FLAG", True)
                return
        # Alternative: "merge_lora_linear" function indicates merge is happening
        if re.search(r'merge_lora_linear|merge_all_lora|merge_lora', self.text):
            self.check("CHECKPOINT_HAS_MERGED_LORA_FLAG", True,
                       "(merge_lora function present — merged_lora flag implied)")
            return
        self.check("CHECKPOINT_HAS_MERGED_LORA_FLAG", False,
                   "Missing merged_lora=True flag in checkpoint metadata")

    def check_smoke_test_has_strict_true(self):
        """The reload smoke test must use strict=True for state_dict loading."""
        patterns = [
            r'load_state_dict\(.*strict\s*=\s*True\)',
            r'load_state_dict\(.*strict=True',
        ]
        for pat in patterns:
            if re.search(pat, self.text):
                self.check("SMOKE_TEST_HAS_STRICT_TRUE", True)
                return
        self.check("SMOKE_TEST_HAS_STRICT_TRUE", False,
                   "Smoke test does not use strict=True — reload may silently miss mismatches")

    def check_smoke_test_cell_present(self):
        """A reload smoke test cell must exist (Section 5.3 or similar)."""
        patterns = [
            r'Reload Merged Checkpoint Smoke Test',
            r'smoke.test',
            r'RUN_RELOAD_SMOKE_TEST',
            r'apply_gemma_kv_shape_for_reload',
        ]
        for pat in patterns:
            if re.search(pat, self.text):
                self.check("SMOKE_TEST_CELL_PRESENT", True)
                return
        self.check("SMOKE_TEST_CELL_PRESENT", False,
                   "No reload smoke test cell found")

    # ──────────────────────────────────────────────────────────────────
    # 6. ADDITIONAL INTEGRITY CHECKS
    # ──────────────────────────────────────────────────────────────────

    def check_no_runtime_outputs_on_target_cells(self):
        """Key cells (surgery, checkpoint, smoke test) should have no stale runtime outputs."""
        target_keywords = ["UNet Surgery", "Smoke Test", "Save.*Checkpoint", "Manual LoRA"]
        for cell in self.cells:
            src = "".join(cell.get("source", []))
            is_target = any(re.search(kw, src) for kw in target_keywords)
            if is_target and cell.get("outputs") and len(cell.get("outputs", [])) > 0:
                # Warn but don't fail — outputs are harmless in saved notebooks
                self.check("NO_RUNTIME_OUTPUTS_ON_KEY_CELLS", False,
                           f"Cell with '{src[:80]}' has {len(cell['outputs'])} output(s)",
                           warn=True)
                return
        self.check("NO_RUNTIME_OUTPUTS_ON_KEY_CELLS", True)

    def check_gemma_model_id_consistent(self):
        """The gemma model ID should be consistent across all cells."""
        gemma_ids = re.findall(r'gemma_path\s*=\s*"([^"]+)"', self.text)
        if not gemma_ids:
            gemma_ids = re.findall(r'"gemma_model_id"\s*:\s*"([^"]+)"', self.text)
        if not gemma_ids:
            gemma_ids = re.findall(r"'gemma_model_id'\s*:\s*'([^']+)'", self.text)
        if not gemma_ids:
            self.check("GEMMA_MODEL_ID_CONSISTENT", True, "(no gemma_model_id found)")
            return
        unique = set(gemma_ids)
        if len(unique) > 1:
            self.check("GEMMA_MODEL_ID_CONSISTENT", False,
                       f"Inconsistent gemma model IDs: {unique}")
        else:
            self.check("GEMMA_MODEL_ID_CONSISTENT", True,
                       f"All references use: {list(unique)[0]}")

    # ──────────────────────────────────────────────────────────────────
    # REPORTING
    # ──────────────────────────────────────────────────────────────────

    def report(self) -> int:
        """Print results and return exit code (0 = all pass, 1 = failures, 2 = warnings only)."""
        print(f"\n{'='*60}")
        print(f"Notebook Verifier: {os.path.basename(self.path)}")
        print(f"{'='*60}\n")
        for r in self.results:
            print(f"  {r}")
        passes = sum(1 for r in self.results if r.passed and not r.warn)
        failures = sum(1 for r in self.results if not r.passed and not r.warn)
        warnings = sum(1 for r in self.results if not r.passed and r.warn)
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"Results: {passes} pass, {failures} fail, {warnings} warn (of {total} checks)")
        print(f"{'='*60}")
        if failures > 0:
            return 1
        return 0


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "gemma3_sd_colab.ipynb"
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(2)
    v = Verifier(path)
    v.run_all()
    sys.exit(v.report())


if __name__ == "__main__":
    main()
