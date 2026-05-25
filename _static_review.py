#!/usr/bin/env python3
"""Static review v2: detect cell-order/name errors in notebook code cells.

Only tracks TOP-LEVEL (non-nested) assignments and name usages.
Also checks function default args for references to not-yet-defined globals.
"""

import json
import ast
import sys
from collections import OrderedDict

NOTEBOOK = "/home/krz/PycharmProjects/gemma-stable-difussion/gemma3_sd_pure_ella_colab.ipynb"

def load_nb(path):
    with open(path) as f:
        return json.load(f)

def get_code_cells(nb):
    cells = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and c.get('source'):
            src = ''.join(c['source'])
            cells.append((i, src))
    return cells

# ---------------------------------------------------------------------------
# Top-level only visitor
# ---------------------------------------------------------------------------

class TopLevelVisitor(ast.NodeVisitor):
    """Visits only top-level statements. Does NOT recurse into function/class bodies."""
    def __init__(self):
        self.assigned = OrderedDict()   # name -> lineno (top-level assignments)
        self.used = OrderedDict()       # name -> [linenos] (top-level usages)
        self.func_default_refs = []     # (func_name, var_in_default, lineno)
        self.imported = OrderedDict()   # name -> lineno

    def _add_assign(self, name, lineno):
        if name not in self.assigned:
            self.assigned[name] = lineno

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self._add_assign(name, node.lineno)
        # Don't recurse

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self._add_assign(name, node.lineno)
        # Don't recurse

    def visit_Assign(self, node):
        # Collect RHS names as top-level usages
        rhs_visitor = _NameCollector(self.used, node.lineno)
        rhs_visitor.visit(node.value)
        # Collect LHS as top-level assignments
        for target in node.targets:
            self._collect_assign_target(target, node.lineno)
        # Don't recurse into nested

    def visit_AnnAssign(self, node):
        if node.value:
            rhs_visitor = _NameCollector(self.used, node.lineno)
            rhs_visitor.visit(node.value)
        if node.target:
            self._collect_assign_target(node.target, node.lineno)
        # Don't recurse

    def visit_AugAssign(self, node):
        rhs_visitor = _NameCollector(self.used, node.lineno)
        rhs_visitor.visit(node.value)
        self._collect_assign_target(node.target, node.lineno)
        # Don't recurse

    def visit_For(self, node):
        # The iterable expression is top-level usage
        rhs_visitor = _NameCollector(self.used, node.lineno)
        rhs_visitor.visit(node.iter)
        # Loop variable is assigned
        if isinstance(node.target, ast.Name):
            self._add_assign(node.target.id, node.lineno)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self._add_assign(elt.id, node.lineno)
        # Visit body (function/class defs inside loops are weird but handle it)
        for stmt in node.body:
            self.visit(stmt)
        # Don't visit orelse

    def visit_FunctionDef(self, node):
        self._add_assign(node.name, node.lineno)
        # Check default args for references to globals
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None:
                for n in ast.walk(default):
                    if isinstance(n, ast.Name):
                        self.func_default_refs.append((node.name, n.id, node.lineno))
        # Do NOT recurse into function body

    def visit_ClassDef(self, node):
        self._add_assign(node.name, node.lineno)
        # Check bases for name references
        for base in node.bases:
            for n in ast.walk(base):
                if isinstance(n, ast.Name):
                    self.used.setdefault(n.id, []).append(node.lineno)
        # Do NOT recurse into class body (class-level assignments are handled differently in notebooks)

    def visit_Expr(self, node):
        # Top-level expression statements (e.g., bare function calls)
        rhs_visitor = _NameCollector(self.used, node.lineno)
        rhs_visitor.visit(node.value)

    def visit_If(self, node):
        # Test
        rhs_visitor = _NameCollector(self.used, node.lineno)
        rhs_visitor.visit(node.test)
        # Body
        for stmt in node.body:
            self.visit(stmt)
        # Orelse
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node):
        for item in node.items:
            rhs_visitor = _NameCollector(self.used, node.lineno)
            rhs_visitor.visit(item.context_expr)
            if item.optional_vars:
                self._collect_assign_target(item.optional_vars, node.lineno)
        for stmt in node.body:
            self.visit(stmt)

    def visit_While(self, node):
        rhs_visitor = _NameCollector(self.used, node.lineno)
        rhs_visitor.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Try(self, node):
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            for stmt in handler.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

    def visit_Return(self, node):
        if node.value:
            rhs_visitor = _NameCollector(self.used, node.lineno)
            rhs_visitor.visit(node.value)

    def visit_Delete(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.used.setdefault(target.id, []).append(node.lineno)

    def _collect_assign_target(self, target, lineno):
        if isinstance(target, ast.Name):
            self._add_assign(target.id, lineno)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_assign_target(elt, lineno)
        elif isinstance(target, ast.Subscript):
            pass
        elif isinstance(target, ast.Attribute):
            pass

    def generic_visit(self, node):
        # Don't recurse by default — we only handle specific top-level nodes
        pass


class _NameCollector(ast.NodeVisitor):
    """Collects all ast.Name nodes (Load context) in a subtree."""
    def __init__(self, used_dict, lineno):
        self.used = used_dict
        self.lineno = lineno

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.setdefault(node.id, []).append(self.lineno)

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.visit(child)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(nb_path):
    nb = load_nb(nb_path)
    code_cells = get_code_cells(nb)

    defined_so_far = set()
    import builtins
    BUILTINS = set(dir(builtins))
    # Additional common names that don't need explicit def
    COMMON = {'self', 'cls', 'True', 'False', 'None', 'NotImplemented', 'Ellipsis',
              '__name__', '__file__', '__doc__', '__builtins__', '__package__',
              '__loader__', '__spec__', '_', 'In', 'Out', 'get_ipython'}

    issues = []

    for ci, src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            issues.append(f"CELL[{ci}] SYNTAX ERROR: {e}")
            continue

        visitor = TopLevelVisitor()
        # Visit all top-level statements
        for stmt in ast.iter_child_nodes(tree):
            visitor.visit(stmt)

        # --- Check 1: top-level names used but not defined yet ---
        for name, linenos in visitor.used.items():
            if name in BUILTINS or name in COMMON:
                continue
            if name in defined_so_far:
                continue
            if name in visitor.assigned:
                # Defined in this cell, possibly later
                continue
            # Check if it's a dotted name like foo.bar where foo is imported
            if '.' in name:
                base = name.split('.')[0]
                if base in defined_so_far:
                    continue
            # Not defined anywhere — flag it
            issues.append(f"CELL[{ci}] UNDEFINED VAR '{name}' used at top-level (lines {list(set(linenos))}) — not defined in any prior cell nor this cell")

        # --- Check 2: function default args referencing undefined globals ---
        for func_name, var_name, lineno in visitor.func_default_refs:
            if var_name in BUILTINS or var_name in COMMON:
                continue
            if var_name in defined_so_far:
                continue
            if var_name in visitor.assigned:
                # Defined in same cell but possibly after function def
                # Check order
                if var_name in visitor.assigned:
                    def_line = visitor.assigned[var_name]
                    if def_line > lineno:
                        issues.append(f"CELL[{ci}] FUNC '{func_name}' default arg references '{var_name}' (line {lineno}) which is assigned at line {def_line} in same cell — order issue!")
            else:
                issues.append(f"CELL[{ci}] FUNC '{func_name}' default arg references UNDEFINED '{var_name}' (line {lineno})")

        # --- Check 3: names used at top-level before assigned in same cell ---
        assigned_in_cell = list(visitor.assigned.keys())
        for name, def_lineno in visitor.assigned.items():
            if name in visitor.used:
                use_lines = visitor.used[name]
                for ul in use_lines:
                    if ul < def_lineno and name not in defined_so_far:
                        issues.append(f"CELL[{ci}] VAR '{name}' used at line {ul} but assigned at line {def_lineno} (same cell, not previously defined)")

        # Update defined set
        defined_so_far.update(visitor.assigned)

    return issues


# ---------------------------------------------------------------------------
# Targeted checks for specific patterns
# ---------------------------------------------------------------------------

def targeted_checks(nb_path):
    nb = load_nb(nb_path)
    code_cells = get_code_cells(nb)
    extra = []

    for ci, src in code_cells:
        # Check for TRM/recursive/FID cell patterns
        lower = src.lower()
        if 'trm_' in lower or 'recursive_y' in lower or 'fid_score' in lower:
            # Check if key imports/config vars are defined
            if 'from' not in src and 'import' not in src:
                extra.append(f"CELL[{ci}] contains TRM/recursive/FID logic without visible imports — may reference undefined globals")

        # Check save/reload connector config mismatch
        if ('save' in lower and 'connector' in lower) or ('reload' in lower and 'connector' in lower):
            extra.append(f"CELL[{ci}] save/reload connector — review for config mismatch")

        # Check EVERY_OPT_STEPS ordering
        if 'EVERY_OPT_STEPS' in src:
            lines = src.split('\n')
            for j, line in enumerate(lines):
                if 'EVERY_OPT_STEPS' in line:
                    # Check if right-hand side references another EVERY_OPT_STEPS
                    if '=' in line:
                        rhs = line.split('=', 1)[1]
                        for other_line in lines:
                            if 'EVERY_OPT_STEPS' in other_line and other_line != line:
                                other_name = other_line.split('=')[0].strip()
                                if other_name in rhs:
                                    extra.append(f"CELL[{ci}] line {j+1}: {line.strip()[:80]} — references {other_name} which may be defined later in same cell")

    return extra


if __name__ == '__main__':
    print("=" * 70)
    print("STATIC REVIEW v2: gemma3_sd_pure_ella_colab.ipynb")
    print("(Top-level names only, no nested bodies)")
    print("=" * 70)

    issues = analyze(NOTEBOOK)

    if issues:
        print(f"\nFOUND {len(issues)} POTENTIAL ISSUES:\n")
        for iss in issues:
            print(f"  [!] {iss}")
    else:
        print("\nNo use-before-definition issues found.\n")

    extra = targeted_checks(NOTEBOOK)
    if extra:
        print(f"\nTARGETED CHECK HITS ({len(extra)}):\n")
        for e in extra:
            print(f"  [i] {e}")

    # Categorize severity
    critical = [i for i in issues if 'UNDEFINED' in i or 'default arg' in i]
    warnings = [i for i in issues if 'used at line' in i and 'assigned at line' in i]

    if critical:
        print("\n" + "=" * 70)
        print(f"VERDICT: FAIL — {len(critical)} undefined-reference issues")
        print("=" * 70)
        sys.exit(1)
    elif warnings:
        print("\n" + "=" * 70)
        print(f"VERDICT: RUN_WITH_FIXES — {len(warnings)} intra-cell ordering issues")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("VERDICT: PASS")
        print("=" * 70)
        sys.exit(0)
