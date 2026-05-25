#!/usr/bin/env python3
import json

with open('/home/krz/PycharmProjects/gemma-stable-difussion/gemma3_sd_pure_ella_colab.ipynb') as f:
    nb = json.load(f)

cells = nb['cells']
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    ct = c['cell_type']
    first = src.split('\n')[0][:100]
    print(f"\n{'='*80}")
    print(f"CELL [{i}] {ct} | {first}")
    print(f"{'='*80}")
    if ct == 'code':
        # Print up to 200 lines
        lines = src.split('\n')
        for j, line in enumerate(lines[:200]):
            print(f"  {j+1:4d}| {line}")
        if len(lines) > 200:
            print(f"  ... ({len(lines) - 200} more lines)")
