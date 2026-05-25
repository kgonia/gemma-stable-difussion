#!/usr/bin/env python3
"""Inspect specific cells and lines from the notebook for detail."""
import json

NOTEBOOK = "/home/krz/PycharmProjects/gemma-stable-difussion/gemma3_sd_pure_ella_colab.ipynb"

with open(NOTEBOOK) as f:
    nb = json.load(f)

# Dump specific cells with line numbers
for ci in [3, 7, 20, 22, 24, 30]:
    c = nb['cells'][ci]
    src = ''.join(c['source'])
    lines = src.split('\n')
    print(f"{'='*60}")
    print(f"CELL[{ci}] ({c['cell_type']}) — {len(lines)} lines")
    print(f"{'='*60}")
    for li, line in enumerate(lines, 1):
        marker = ""
        if ci == 7 and li in [92, 220, 221]:
            marker = "  <<< ISSUE"
        if ci == 20 and li == 14:
            marker = "  <<< ISSUE"
        if ci == 22 and li == 63:
            marker = "  <<< ISSUE"
        if ci == 24 and li == 113:
            marker = "  <<< ISSUE"
        if ci == 30 and li == 5:
            marker = "  <<< ISSUE"
        print(f"  {li:4d}: {line}{marker}")
    print()
