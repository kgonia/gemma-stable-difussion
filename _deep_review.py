#!/usr/bin/env python3
"""Deep static review: focused checks for real issues, with false-positive filtering."""
import json
import ast
import sys
from collections import OrderedDict

NOTEBOOK = "/home/krz/PycharmProjects/gemma-stable-difussion/gemma3_sd_pure_ella_colab.ipynb"

with open(NOTEBOOK) as f:
    nb = json.load(f)

def get_code_cells(nb):
    cells = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and c.get('source'):
            src = ''.join(c['source'])
            cells.append((i, src))
    return cells

code_cells = get_code_cells(nb)

# Track top-level assignments per cell
defined_by_cell = {}  # cell_idx -> set of names assigned
cumulative = set()
import builtins
BUILTINS = set(dir(builtins))
COMMON = {'self', 'cls', 'True', 'False', 'None', 'Ellipsis', '_', 'get_ipython',
          'In', 'Out', '__name__', '__file__', '__doc__', '__package__'}

for ci, src in code_cells:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Likely IPython magic like !pip — skip
        defined_by_cell[ci] = set()
        continue
    
    assigned = set()
    # Only collect top-level assignments (not nested in func/class)
    for stmt in ast.iter_child_nodes(tree):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in (stmt.names if isinstance(stmt, ast.Import) else []):
                name = alias.asname or alias.name.split('.')[0]
                assigned.add(name)
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    name = alias.asname or alias.name
                    assigned.add(name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign) and stmt.target:
                targets = [stmt.target]
            elif isinstance(stmt, ast.AugAssign):
                targets = [stmt.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            assigned.add(elt.id)
        elif isinstance(stmt, ast.FunctionDef):
            assigned.add(stmt.name)
        elif isinstance(stmt, ast.ClassDef):
            assigned.add(stmt.name)
        elif isinstance(stmt, ast.For):
            if isinstance(stmt.target, ast.Name):
                assigned.add(stmt.target.id)
    
    defined_by_cell[ci] = assigned
    cumulative.update(assigned)

# Now check each cell for top-level name references that aren't in cumulative
print("=" * 70)
print("DEEP STATIC REVIEW: Real issues only")
print("=" * 70)

real_issues = []

for ci, src in code_cells:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    
    # Collect top-level name usages (not in func/class bodies)
    class TopLevelUseCollector(ast.NodeVisitor):
        def __init__(self):
            self.uses = OrderedDict()
            self.func_default_refs = []
        def visit_FunctionDef(self, node):
            # Only check defaults
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None:
                    for n in ast.walk(default):
                        if isinstance(n, ast.Name):
                            self.func_default_refs.append((node.name, n.id, node.lineno))
            # Don't recurse into body
        def visit_ClassDef(self, node):
            for base in node.bases:
                self.visit(base)
            # Don't recurse into body
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                self.uses.setdefault(node.id, []).append(node.lineno)
        def visit_Assign(self, node):
            # Visit RHS for name references
            self.visit(node.value)
            # Don't recurse into LHS or body
        def visit_AnnAssign(self, node):
            if node.value:
                self.visit(node.value)
        def visit_AugAssign(self, node):
            self.visit(node.value)
        def visit_Expr(self, node):
            self.visit(node.value)
        def visit_For(self, node):
            self.visit(node.iter)
            # Visit body
            for s in node.body:
                self.visit(s)
            for s in node.orelse:
                self.visit(s)
        def visit_If(self, node):
            self.visit(node.test)
            for s in node.body:
                self.visit(s)
            for s in node.orelse:
                self.visit(s)
        def visit_With(self, node):
            for item in node.items:
                self.visit(item.context_expr)
            for s in node.body:
                self.visit(s)
        def visit_While(self, node):
            self.visit(node.test)
            for s in node.body:
                self.visit(s)
        def visit_Try(self, node):
            for s in node.body:
                self.visit(s)
            for h in node.handlers:
                for s in h.body:
                    self.visit(s)
            for s in node.orelse:
                self.visit(s)
            for s in node.finalbody:
                self.visit(s)
        def visit_Return(self, node):
            if node.value:
                self.visit(node.value)
        def generic_visit(self, node):
            pass  # Don't recurse by default
    
    collector = TopLevelUseCollector()
    for stmt in ast.iter_child_nodes(tree):
        collector.visit(stmt)
    
    # Check uses
    prev_defined = set()
    for cj in range(ci):
        prev_defined.update(defined_by_cell.get(cj, set()))
    
    for name, linenos in collector.uses.items():
        if name in BUILTINS or name in COMMON:
            continue
        if name in prev_defined:
            continue
        if name in defined_by_cell.get(ci, set()):
            continue
        # Check if it's a dotted reference
        if '.' in name:
            base = name.split('.')[0]
            if base in prev_defined or base in defined_by_cell.get(ci, set()):
                continue
        # Check if it's a dict comp loop variable (k, v pattern)
        # We can't easily detect this from AST without more context, but we can check
        # if the name appears ONLY in a dict comp context
        real_issues.append(f"CELL[{ci}] UNDEFINED '{name}' at lines {list(set(linenos))}")
    
    # Check function default args
    for func_name, var_name, lineno in collector.func_default_refs:
        if var_name in BUILTINS or var_name in COMMON:
            continue
        if var_name in prev_defined:
            continue
        # Check if defined in this cell
        if var_name in defined_by_cell.get(ci, set()):
            # Might be defined after the function - check line order
            # Hard to determine precisely; flag it
            real_issues.append(f"CELL[{ci}] FUNC '{func_name}' default arg '{var_name}' at line {lineno} — not defined in prior cells (may be defined later in same cell)")

if real_issues:
    print(f"\nPOTENTIAL ISSUES ({len(real_issues)}):")
    for iss in real_issues:
        print(f"  [!] {iss}")
else:
    print("\nNo undefined-reference issues found.")

# ============================================================
# Specific pattern checks
# ============================================================
print("\n" + "=" * 70)
print("PATTERN-SPECIFIC CHECKS")
print("=" * 70)

# 1. Check QUALITY_EVERY_OPT_STEPS ordering in CELL[7]
print("\n--- QUALITY_EVERY_OPT_STEPS ordering (CELL[7]) ---")
c7_src = ''.join(nb['cells'][7]['source'])
c7_lines = c7_src.split('\n')

# Find where VALIDATION_EVERY_OPT_STEPS is first defined
val_def_line = None
quality_def_line = None
for li, line in enumerate(c7_lines, 1):
    if 'VALIDATION_EVERY_OPT_STEPS' in line and '=' in line and 'QUALITY' not in line:
        if val_def_line is None:
            val_def_line = li
    if 'QUALITY_EVERY_OPT_STEPS' in line and '=' in line:
        if quality_def_line is None:
            quality_def_line = li

print(f"  VALIDATION_EVERY_OPT_STEPS first defined at line: {val_def_line}")
print(f"  QUALITY_EVERY_OPT_STEPS defined at line: {quality_def_line}")
if val_def_line and quality_def_line and quality_def_line > val_def_line:
    print("  OK: QUALITY_EVERY_OPT_STEPS defined after VALIDATION_EVERY_OPT_STEPS")
else:
    print("  ISSUE: Ordering problem!")

# 2. Check TRM/recursive cells for proper config references
print("\n--- TRM/recursive/FID cell references ---")
for ci, src in code_cells:
    lower = src.lower()
    has_trm = 'trm_' in lower
    has_recursive = 'recursive_y' in lower
    has_fid = 'fid_score' in lower or ('fid' in lower and 'torchmetrics' in lower)
    
    if has_trm or has_recursive or has_fid:
        # Check if this cell is at top-level (not nested in if/for)
        tree = ast.parse(src)
        uses = set()
        class UseFinder(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    uses.add(node.id)
        for stmt in ast.iter_child_nodes(tree):
            # Only check non-nested
            if isinstance(stmt, (ast.Assign, ast.Expr, ast.If, ast.For, ast.With, ast.While, ast.Try)):
                UseFinder().visit(stmt)
        
        # Check which of TRM/RECURSIVE/FID config vars are used
        trm_vars = {'TRM_OUTER_STEPS', 'TRM_INNER_STEPS', 'TRM_SCRATCH_TOKENS', 'TRM_Y_GATE_INIT', 'TRM_Z_GATE_INIT'}
        rec_vars = {'RECURSIVE_Y_STEPS', 'RECURSIVE_Y_GATE_INIT'}
        fid_vars = {'RUN_FID', 'RUN_KID', 'FID_EVERY_OPT_STEPS', 'FID_REFERENCE_SIZE', 'FID_GENERATION_SIZE', 'QUALITY_EVERY_OPT_STEPS', 'RUN_IMAGE_QUALITY_METRICS'}
        
        all_config = trm_vars | rec_vars | fid_vars
        used_config = uses & all_config
        missing = used_config - cumulative
        if missing:
            print(f"  CELL[{ci}] uses config vars {missing} that may not be defined yet")

# 3. Check save/reload connector config consistency
print("\n--- Save/reload connector config consistency ---")
connector_saves = []
for ci, src in code_cells:
    if 'torch.save' in src and 'connector' in src.lower():
        connector_saves.append(ci)
        # Check what keys are saved
        saved_keys = set()
        for line in src.split('\n'):
            if 'connector_state_dict' in line:
                saved_keys.add('connector_state_dict')
            if '\"config\"' in line or "'config'" in line:
                saved_keys.add('config')
            if '\"run_config\"' in line or "'run_config'" in line:
                saved_keys.add('run_config')
            if '\"connector_config\"' in line or "'connector_config'" in line:
                saved_keys.add('connector_config')
            if '\"architecture\"' in line or "'architecture'" in line:
                saved_keys.add('architecture')
            if '\"connector_type\"' in line or "'connector_type'" in line:
                saved_keys.add('connector_type')
            if '\"stage\"' in line or "'stage'" in line:
                saved_keys.add('stage')
        print(f"  CELL[{ci}] saves keys: {sorted(saved_keys)}")

# Also check reloads
print("\n--- Connector reloads ---")
for ci, src in code_cells:
    if ('torch.load' in src or 'load_state_dict' in src) and 'connector' in src.lower():
        print(f"  CELL[{ci}] reloads connector")

print("\n" + "=" * 70)
if real_issues:
    print("VERDICT: RUN_WITH_FIXES")
else:
    print("VERDICT: PASS")
print("=" * 70)
