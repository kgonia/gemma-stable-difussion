#!/usr/bin/env python3
"""Extract notebook cells into a readable text file."""
import json, sys

nb_path = "/home/krz/PycharmProjects/gemma-stable-difussion/gemma3_sd_pure_ella_colab.ipynb"
out_path = "/home/krz/PycharmProjects/gemma-stable-difussion/_nb_extracted.txt"

with open(nb_path) as f:
    nb = json.load(f)

lines = []
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    ct = cell['cell_type']
    lines.append(f"{'='*80}")
    lines.append(f"CELL {i} [{ct}] | {len(src)} chars")
    lines.append(f"{'='*80}")
    lines.append(src)
    lines.append("")

with open(out_path, 'w') as f:
    f.write('\n'.join(lines))

print(f"Extracted {len(nb['cells'])} cells to {out_path}")
