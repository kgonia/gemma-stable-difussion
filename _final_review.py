#!/usr/bin/env python3
"""Final focused review: only real cross-cell issues."""
import json, ast

NOTEBOOK = "/home/krz/PycharmProjects/gemma-stable-difussion/gemma3_sd_pure_ella_colab.ipynb"

with open(NOTEBOOK) as f:
    nb = json.load(f)

code_cells = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and c.get('source'):
        src = ''.join(c['source'])
        try:
            ast.parse(src)
            code_cells.append((i, src))
        except SyntaxError:
            pass  # IPython magic

# Manual scan for key config vars referenced across cells
# Check where WANDB_ENABLED is defined
print("=== 1. WANDB_ENABLED definition search ===")
for ci, src in code_cells:
    if 'WANDB_ENABLED' in src:
        print(f"  CELL[{ci}] references WANDB_ENABLED")
        for li, line in enumerate(src.split('\n'), 1):
            if 'WANDB_ENABLED' in line:
                print(f"    L{li}: {line.strip()[:100]}")

# Check USE_CLIP_TEACHER_DELTA_PHASE2
print("\n=== 2. USE_CLIP_TEACHER_DELTA_PHASE2 ===")
for ci, src in code_cells:
    if 'USE_CLIP_TEACHER_DELTA_PHASE2' in src:
        print(f"  CELL[{ci}] references USE_CLIP_TEACHER_DELTA_PHASE2")
        for li, line in enumerate(src.split('\n'), 1):
            if 'USE_CLIP_TEACHER_DELTA_PHASE2' in line:
                print(f"    L{li}: {line.strip()[:120]}")

# Check CELL[31] variables
print("\n=== 3. CELL[31] undefined vars ===")
c31 = nb['cells'][31]
src31 = ''.join(c31['source'])
print(f"  CELL[31] first 2000 chars:")
print(src31[:2000])

# Check connector save configs
print("\n=== 4. Connector save keys ===")
for ci, src in code_cells:
    if 'torch.save' in src and ('connector' in src.lower() or 'ella_connector' in src):
        print(f"\n  CELL[{ci}] save:")
        for li, line in enumerate(src.split('\n'), 1):
            if any(kw in line for kw in ['torch.save', 'state_dict', '"config"', "'config'", 
                                          '"run_config"', "'run_config'", 
                                          '"connector_config"', "'connector_config'",
                                          '"architecture"', "'architecture'",
                                          '"connector_type"', "'connector_type'",
                                          '"stage"', "'stage'",
                                          '"gemma_model_id"', "'gemma_model_id'",
                                          '"sd_checkpoint"', "'sd_checkpoint'"]):
                print(f"    L{li}: {line.strip()[:120]}")

# Check all connector loads
print("\n=== 5. Connector load/reload sites ===")
for ci, src in code_cells:
    if 'torch.load' in src and 'connector' in src.lower():
        print(f"\n  CELL[{ci}] load:")
        for li, line in enumerate(src.split('\n'), 1):
            if 'torch.load' in line or 'load_state_dict' in line or 'connector' in line.lower():
                print(f"    L{li}: {line.strip()[:150]}")

# Check what RUN_CONFIG is at CELL[7]
print("\n=== 6. RUN_CONFIG capture at CELL[7] ===")
c7_lines = ''.join(nb['cells'][7]['source']).split('\n')
for li in [220, 221, 222, 223, 224]:
    print(f"  L{li}: {c7_lines[li-1].strip()[:150]}")

# Check CONNECTOR_CONFIG and CONNECTOR_VARIANT_CONFIG
print("\n=== 7. CONNECTOR_CONFIG and CONNECTOR_VARIANT_CONFIG ===")
for ci, src in code_cells:
    for var in ['CONNECTOR_CONFIG', 'CONNECTOR_VARIANT_CONFIG']:
        if var in src:
            print(f"  CELL[{ci}] references {var}")
            for li, line in enumerate(src.split('\n'), 1):
                if var in line:
                    print(f"    L{li}: {line.strip()[:120]}")
