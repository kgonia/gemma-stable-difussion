#!/usr/bin/env python3
import json, sys

with open('/home/krz/PycharmProjects/gemma-stable-difussion/gemma3_sd_pure_ella_colab.ipynb') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")
print()

for i, c in enumerate(cells):
    src = ''.join(c['source'])
    first = src.split('\n')[0][:120]
    ct = c['cell_type']
    ec = c.get('execution_count','None')
    print(f"[{i:3d}] {ct:8s} exec={str(ec):5s} | {first}")
