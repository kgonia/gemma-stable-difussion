# Gemma3-SD Colab v2 — Fixes + Streaming Dataset

> **For Hermes:** Use subagent-driven-development. Dispatch per task. Claude Code for code-gen, Gemini CLI for review. Orchestrator (me) does NOT code.

**Goal:** Fix 11 review findings + integrate jackyhate/text-to-image-2M streaming dataset (100k samples configurable) + Google Drive persistence.

**Architecture:** Modify existing notebook cells. No new notebook from scratch. Streaming via webdataset — tar shards downloaded on-the-fly, never fully in memory. Pinned deps, gradient accumulation, peft-based LoRA.

**Caveman summary of review findings → fixes:**

| # | Severity | What | Fix |
|---|----------|------|-----|
| B1 | BLOCKER | Device/dtype crash (cells 3.2/4.1) | `.to(device).to(torch.float16)` after new Linear/ManualLoRA |
| B2 | BLOCKER | xformers breaks torch pin (cell 1.1) | Pin `xformers==0.0.23.post1` compatible with torch 2.1 |
| B3 | BLOCKER | ZeroDivisionError probe training (cell 2.4) | Guard: `if gemma_flat.shape[0] < batch_size: raise` or auto-adjust |
| B4 | BLOCKER | batch_size=1 no grad accum (cell 4.4) | Add `GRADIENT_ACCUMULATION_STEPS` config, accumulate then step |
| W1 | WARNING | Empty captions → unconditional (cell 4.2) | Skip samples with empty captions, log warning count |
| W2 | WARNING | Bucket resize warps images (cell 4.2) | Center-crop to bucket aspect ratio BEFORE resize |
| W3 | WARNING | VAE noise frozen (cell 4.3) | Cache mean/logvar, re-sample each epoch |
| W4 | WARNING | LoRA state_dict incompatible (cell 4.1) | Replace ManualLoRA with peft LoraConfig targeting attn2.to_k/attn2.to_v |
| I1 | INFO | CLIP VRAM leak (cell 2.1) | `del clip_model; torch.cuda.empty_cache()` after Section 2 |
| I2 | INFO | Double I/O in Dataset.__init__ (cell 4.2) | Use streaming dataset, bucket determined from latent size not image |
| N1 | NOTE | probe_bias unused (cell 3.2) | Use probe_bias in warm-start: `new_bias = old_bias + probe_bias @ W_old^T` |

---

## New Requirements

### R1: Google Drive in Cell 1
```
from google.colab import drive
drive.mount('/content/drive')
```
Mount before all downloads. Save all artifacts (probe.pt, lora.pt, complete model, samples) to `/content/drive/MyDrive/gemma3-sd/`.

### R2: Streaming Dataset
Replace hardcoded CAPTIONS + demo noise images with:
```python
from datasets import load_dataset
base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
NUM_SHARDS = 46  # total: 46 tar files
urls = [base_url.format(i=i) for i in range(NUM_SHARDS)]
dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
```
Dataset yields rows with `image` (PIL) and `text` (caption). Streaming = no disk download.

### R3: Configurable Sample Count
```python
MAX_SAMPLES = 100_000  # Configurable. Change before running.
```
Take first N from streaming dataset. Shuffle buffer for randomness:
```python
dataset = dataset.shuffle(buffer_size=10_000).take(MAX_SAMPLES)
```

### R4: Non-Blocking Pipeline
Streaming dataset feeds directly into training. No pre-download. Use Colab's `/tmp/` for any temp files. Prefetch with `DataLoader(num_workers=2, prefetch_factor=2)`.

### R5: Results to Google Drive
- Probe: `/content/drive/MyDrive/gemma3-sd/probe.pt`
- LoRA: `/content/drive/MyDrive/gemma3-sd/gemma3_sd_lora.pt`
- Complete model: `/content/drive/MyDrive/gemma3-sd/gemma3_sd_complete.pt`
- Samples: `/content/drive/MyDrive/gemma3-sd/samples.png`
- wandb logs auto-sync

---

## Task Breakdown (bite-sized, 2-5 min each)

### Phase 0: Orchestration Setup

#### Task 0.1: Agent knowledge alignment (Socratic)
**Method:** Both agents read existing notebook + this plan. Orchestrator asks each agent to critique the other's understanding of the fix. Resolve disagreements before coding.

```
Orchestrator → Claude: "Explain the manual LoRA → peft migration. What could break?"
Orchestrator → Gemini: "Critique Claude's peft migration plan. Find gaps."
Orchestrator → both: "Agree on approach."
```

### Phase 1: Critical Bug Fixes (make notebook runnable)

#### Task 1.1: Fix device/dtype crash
**Files:** `gemma3_sd_colab.ipynb` cells 3.2, 4.1
**Agent:** claude-code (print mode, 3 turns)

- Cell 3.2: After `new_k = nn.Linear(new_dim, inner_dim, bias=has_bias)`, add:
  ```python
  new_k = new_k.to(device=device, dtype=torch.float16)
  new_v = new_v.to(device=device, dtype=torch.float16)
  ```
- Cell 4.1: In `ManualLoRA.__init__`, after creating lora_A/lora_B, add `.to(dtype=base_linear.weight.dtype, device=base_linear.weight.device)`
- Also fix: `ManualLoRA` layers created in cell 4.1 need explicit `.to(device)` after wrapping

**Verify:** Read the cell, confirm `.to()` calls exist at correct locations. No execution needed (can't run Colab locally).

#### Task 1.2: Pin xformers version
**Files:** `gemma3_sd_colab.ipynb` cell 1.1
**Agent:** claude-code (print mode, 1 turn)

- Change: `!pip install -q datasets==2.20.0 xformers --index-url https://download.pytorch.org/whl/cu118`
- To: `!pip install -q datasets==2.20.0 xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118`
- Note: 0.0.23.post1 is the last release built for torch 2.1 + cu118

**Verify:** Check version string exists.

#### Task 1.3: Fix ZeroDivisionError in probe training
**Files:** `gemma3_sd_colab.ipynb` cell 2.4
**Agent:** claude-code (print mode, 2 turns)

- Before training loop, add guard:
  ```python
  n_samples = gemma_flat.shape[0]
  if n_samples < batch_size:
      batch_size = max(1, n_samples // 4)
      print(f"Small dataset: reduced batch_size to {batch_size}")
  n_batches = n_samples // batch_size
  assert n_batches > 0, f"Not enough data for training: {n_samples} tokens"
  ```
- Change loss print: `loss = {total / n_batches:.6f}`

**Verify:** Check guard code exists, division uses `n_batches`.

### Phase 2: Dataset Migration (hardcoded → streaming)

#### Task 2.1: Add Google Drive mount cell
**Files:** New cell after title, before 1.1
**Agent:** claude-code (print mode, 2 turns)

- New markdown cell: `## Section 0: Google Drive`
- New code cell (cell 0.1):
  ```python
  # @title 0.1 Mount Google Drive
  from google.colab import drive
  drive.mount('/content/drive')
  
  import os
  DRIVE_OUT = '/content/drive/MyDrive/gemma3-sd'
  os.makedirs(DRIVE_OUT, exist_ok=True)
  print(f"Saving results to: {DRIVE_OUT}")
  ```
- Update cell 1.4 (wandb) to also log `drive_output_path`

**Verify:** New cells exist, DRIVE_OUT variable defined.

#### Task 2.2: Replace hardcoded captions with streaming dataset
**Files:** `gemma3_sd_colab.ipynb` — replace cell 2.2, modify cell 2.3
**Agent:** claude-code (print mode, 4 turns)

- New cell 2.2 (replaces hardcoded CAPTIONS):
  ```python
  # @title 2.2 Load Streaming Caption Dataset
  from datasets import load_dataset
  
  MAX_SAMPLES = 100_000  # CONFIGURABLE: change before running
  
  base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
  NUM_SHARDS = 46
  urls = [base_url.format(i=i) for i in range(NUM_SHARDS)]
  
  ds_stream = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
  ds_stream = ds_stream.shuffle(buffer_size=10_000).take(MAX_SAMPLES)
  
  # Extract captions only (images not needed for probe)
  captions = []
  for i, sample in enumerate(ds_stream):
      captions.append(sample["text"])
      if (i + 1) % 1000 == 0:
          print(f"  Loaded {i+1} captions...")
  
  print(f"Loaded {len(captions)} captions for probe training")
  wandb.log({"probe_captions": len(captions)})
  ```
- Remove: old CAPTIONS list + augmentation loop (augmentation adds noise for probe — captions only mapped, not augmented)
- Actually keep augmentation: after loading, create 2 variants per caption (with/without "high quality" suffix) for the probe

**Verify:** Check dataset loading code present, MAX_SAMPLES configurable, shuffle + take pattern.

#### Task 2.3: Update probe embedding collection for streaming
**Files:** `gemma3_sd_colab.ipynb` cell 2.3
**Agent:** claude-code (print mode, 2 turns)

- `get_clip_hidden` and `get_gemma_hidden` accept `captions` list — no change needed
- But: captions now from streaming dataset, may have > MAX_SAMPLES. Add progress bar.
- Keep `min_seq` alignment logic (CLIP 77, Gemma up to 128)

**Verify:** Functions unchanged, captions list source changed.

#### Task 2.4: Replace demo noise dataset with streaming images
**Files:** `gemma3_sd_colab.ipynb` — replace Section 4A (cells 4A.1), modify 4.2
**Agent:** claude-code (print mode, 5 turns)

- New cell 4A.1 replaces entire Google Drive / manual upload / demo noise approach:
  ```python
  # @title 4A.1 Streaming Image-Caption Dataset
  import io
  from PIL import Image
  
  # Re-create streaming dataset (or reuse from probe phase — but we need images now)
  ds_stream = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
  ds_stream = ds_stream.shuffle(buffer_size=10_000).take(MAX_SAMPLES)
  
  # Save to disk for DataLoader compatibility (streaming → disk bridge)
  # Use /tmp/ for Colab ephemeral storage
  TRAIN_DIR = "/tmp/training_data"
  os.makedirs(TRAIN_DIR, exist_ok=True)
  
  saved = 0
  for i, sample in enumerate(ds_stream):
      img = sample["image"]
      if isinstance(img, dict):  # some webdataset rows have nested dicts
          img = img.get("png") or img.get("jpg") or list(img.values())[0]
      if isinstance(img, bytes):
          img = Image.open(io.BytesIO(img))
      if not isinstance(img, Image.Image):
          continue
      # Resize to 512px short side for bucketing
      w, h = img.size
      scale = 512 / min(w, h)
      img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
      img.save(f"{TRAIN_DIR}/img_{saved:06d}.png")
      caption = sample.get("text", "")
      if not caption or not caption.strip():
          caption = "an image"  # fallback, don't train on empty
      with open(f"{TRAIN_DIR}/img_{saved:06d}.txt", "w") as f:
          f.write(caption.strip())
      saved += 1
      if saved % 500 == 0:
          print(f"  Saved {saved} images...")
      if saved >= MAX_SAMPLES:
          break
  
  print(f"Saved {saved} image-caption pairs to {TRAIN_DIR}")
  wandb.log({"training_samples": saved})
  ```
- Remove: old cell 4A.1 (Google Drive / manual / demo noise methods)
- Cell 4.2 (SDDataset) now reads from `/tmp/training_data/`
- Note: `num_workers=2` in DataLoader for prefetching

**Verify:** Streaming → disk bridge exists, caption fallback present, image count logged.

### Phase 3: Training Quality Fixes

#### Task 3.1: Add gradient accumulation
**Files:** `gemma3_sd_colab.ipynb` cell 4.4
**Agent:** claude-code (print mode, 3 turns)

- Add config at top of cell:
  ```python
  GRADIENT_ACCUMULATION_STEPS = 4  # effective batch_size = 4
  ```
- Wrap backward/step:
  ```python
  loss = loss / GRADIENT_ACCUMULATION_STEPS
  loss.backward()
  
  if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
      nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
      optimizer.step()
      optimizer.zero_grad()
  ```
- Update progress bar to show effective batch

**Verify:** Accumulation code present, div-by-accum-steps, conditional step/zero_grad.

#### Task 3.2: Fix bucket resizing (center-crop before resize)
**Files:** `gemma3_sd_colab.ipynb` cell 4.2 — `SDDataset.__getitem__`
**Agent:** claude-code (print mode, 2 turns)

- In `__getitem__`, before `img.resize((bw, bh), Image.LANCZOS)`:
  ```python
  # Center-crop to bucket aspect ratio first
  b_ratio = bw / bh
  w, h = img.size
  if w / h > b_ratio:
      new_w = int(h * b_ratio)
      img = img.crop(((w - new_w) // 2, 0, (w + new_w) // 2, h))
  else:
      new_h = int(w / b_ratio)
      img = img.crop((0, (h - new_h) // 2, w, (h + new_h) // 2))
  ```
- Then resize (now same-aspect-ratio, no warping)

**Verify:** Center-crop code before resize.

#### Task 3.3: Freeze VAE noise → re-sample each epoch
**Files:** `gemma3_sd_colab.ipynb` cell 4.3
**Agent:** claude-code (print mode, 3 turns)

- Cache mean and logvar instead of latent:
  ```python
  cache[i] = {
      "mean": latent_dist.mean.cpu(),
      "logvar": (2 * torch.log(latent_dist.std)).cpu(),  # log(var) for numerical stability
      "input_ids": batch["input_ids"],
      "attention_mask": batch["attention_mask"],
  }
  ```
- In training loop (cell 4.4), reconstruct:
  ```python
  latent_mean = c["mean"].to(device, dtype=torch.float16)
  latent_std = (0.5 * c["logvar"]).exp().to(device, dtype=torch.float16)
  latent = latent_mean + latent_std * torch.randn_like(latent_mean)
  ```

**Verify:** Cache stores mean+logvar, training loop re-samples.

#### Task 3.4: Replace ManualLoRA with peft
**Files:** `gemma3_sd_colab.ipynb` — replace cell 4.1, modify cell 5.1
**Agent:** claude-code (print mode, 4 turns)

- Remove entire ManualLoRA class
- Replace LoRA application with:
  ```python
  from peft import LoraConfig, get_peft_model, TaskType
  
  peft_config = LoraConfig(
      task_type=TaskType.FEATURE_EXTRACTION,
      r=8,
      lora_alpha=16,
      target_modules=[],  # peft can't auto-target renamed layers
      modules_to_save=[],
  )
  # Manually add LoRA to attn2.to_k and attn2.to_v
  # peft doesn't know about diffusers attention naming, so:
  # Use peft's LoraLayer internally but apply manually
  ```
- **PITFALL:** peft's `get_peft_model` doesn't understand diffusers' attention structure. Two options:
  - Option A: Use peft's `LoraModel._create_and_replace` at low level → complex, fragile
  - Option B: Keep ManualLoRA but save in peft-compatible format using `PeftModel.save_pretrained` → simpler

- **Decision needed from Socratic debate (Task 0.1):** peft vs ManualLoRA tradeoff.
- If ManualLoRA kept: document `--compat` note, save in HuggingFace format with `{"lora_A.weight": ..., "lora_B.weight": ...}` mapping to standard key names `"to_k.lora_A.default.weight"` etc.

**Verify:** Socratic consensus reached. Code consistent with decision.

### Phase 4: Cleanup + Drive Integration

#### Task 4.1: Fix CLIP VRAM leak
**Files:** `gemma3_sd_colab.ipynb` — after cell 2.4
**Agent:** claude-code (print mode, 1 turn)

- Add after probe training completes:
  ```python
  # Free CLIP — no longer needed
  del clip_model, clip_tokenizer
  gc.collect()
  torch.cuda.empty_cache()
  print(f"VRAM freed: {torch.cuda.memory_allocated() / 1e9:.1f} GB in use")
  ```

**Verify:** Del + empty_cache present after Section 2.

#### Task 4.2: Fix probe_bias warm-start
**Files:** `gemma3_sd_colab.ipynb` cell 3.2
**Agent:** claude-code (print mode, 2 turns)

- After bias copy:
  ```python
  if has_bias and probe_bias is not None:
      corrected_bias = module.to_k.bias.data + probe_bias @ module.to_k.weight.data.T  # wait, this is wrong
      # Correct: new bias should account for probe transform
      # b_new = b_old  (since probe is linear, the bias shift is handled by weight)
      # Actually: if y = W_old @ probe @ x + b_old, that's W_new @ x + b_old
      # So b_new = b_old is correct for forward pass.
      # But the INITIALIZATION should be b_new = b_old + old_linear(probe_bias)
      pass
  ```
- **Socratic debate needed:** Is probe_bias correction even meaningful? The probe was trained on token-level embeddings where the bias is a small offset. For initialization, zero-bias probe is fine. Document the decision.

**Verify:** Either probe_bias used correctly, or removed with comment explaining why.

#### Task 4.3: Save all artifacts to Google Drive
**Files:** `gemma3_sd_colab.ipynb` — update save paths in cells 2.4, 4.4, 5.2, 5.1
**Agent:** claude-code (print mode, 2 turns)

- Replace all `/content/` save paths with `DRIVE_OUT`:
  - Cell 2.4: `torch.save(..., f"{DRIVE_OUT}/probe.pt")`
  - Cell 4.4: `torch.save(lora_w, f"{DRIVE_OUT}/gemma3_sd_lora.pt")` and `torch.save(lora_w, f"{DRIVE_OUT}/lora_epoch{epoch+1}.pt")`
  - Cell 5.1: `plt.savefig(f"{DRIVE_OUT}/samples.png", dpi=100)`
  - Cell 5.2: `torch.save(save, f"{DRIVE_OUT}/gemma3_sd_complete.pt")`

**Verify:** All paths use `DRIVE_OUT`.

#### Task 4.4: Add wandb.finish() in try/finally
**Files:** `gemma3_sd_colab.ipynb` cell 4.4
**Agent:** claude-code (print mode, 1 turn)

- Wrap training loop:
  ```python
  try:
      # training loop here
  finally:
      wandb.finish()
  ```

**Verify:** try/finally exists, wandb.finish() in finally.

### Phase 5: Verification Pass (Both Agents)

#### Task 5.1: Gemini CLI adversarial review
**Agent:** gemini-cli (headless, plan mode)
- Read final notebook
- Challenge EVERY claim: "Does streaming dataset actually work in Colab? Does peft migration handle diffusers attention naming? Will gradient accumulation compile?"
- Return BLOCKER list

#### Task 5.2: Claude Code architecture review
**Agent:** claude-code (print mode, 3 turns)
- Read final notebook
- Verify: dimension consistency end-to-end, CFG math still correct after changes, VAE encode/decode scaling factors untouched, scheduler step matches latent shape
- Return PASS/BLOCKER per section

#### Task 5.3: Resolve disagreements (Socratic)
**Orchestrator:** Collect findings from both agents. If both flag same BLOCKER → fix. If they disagree → ask each to critique the other's finding, then decide.

---

## Execution Order

```
Phase 0 → Task 0.1 (Socratic alignment)
Phase 1 → Tasks 1.1, 1.2, 1.3 (can parallelize: 3 separate claude-code instances)
Phase 2 → Tasks 2.1, 2.2, 2.3, 2.4 (sequential: each builds on prior)
Phase 3 → Tasks 3.1, 3.2, 3.3, 3.4 (can parallelize 3.1-3.3, 3.4 needs 0.1 decision)
Phase 4 → Tasks 4.1, 4.2, 4.3, 4.4 (can parallelize)
Phase 5 → Tasks 5.1, 5.2 (parallel), then 5.3 (orchestrator synthesis)
```

---

## Agent Assignment

| Phase | Task | Agent | Mode | Turns |
|-------|------|-------|------|-------|
| 0 | 0.1 | Both + Orchestrator | PTY + chat | — |
| 1 | 1.1 | claude-code | print (-p) | 3 |
| 1 | 1.2 | claude-code | print (-p) | 1 |
| 1 | 1.3 | claude-code | print (-p) | 2 |
| 2 | 2.1 | claude-code | print (-p) | 2 |
| 2 | 2.2 | claude-code | print (-p) | 4 |
| 2 | 2.3 | claude-code | print (-p) | 2 |
| 2 | 2.4 | claude-code | print (-p) | 5 |
| 3 | 3.1 | claude-code | print (-p) | 3 |
| 3 | 3.2 | claude-code | print (-p) | 2 |
| 3 | 3.3 | claude-code | print (-p) | 3 |
| 3 | 3.4 | claude-code | print (-p) | 4 |
| 4 | 4.1 | claude-code | print (-p) | 1 |
| 4 | 4.2 | claude-code | print (-p) | 2 |
| 4 | 4.3 | claude-code | print (-p) | 2 |
| 4 | 4.4 | claude-code | print (-p) | 1 |
| 5 | 5.1 | gemini-cli | headless (-p) plan | — |
| 5 | 5.2 | claude-code | print (-p) | 3 |
| 5 | 5.3 | Orchestrator | manual synthesis | — |

---

## Undecided Items (Require Socratic Resolution in Task 0.1)

1. **peft vs ManualLoRA:** Keep ManualLoRA with export wrapper, or fight peft into diffusers compatibility? Tradeoff: 1 hour peft debugging vs incompatible checkpoints.
2. **probe_bias correctness:** Worth computing `b_new = b_old + W_old @ probe_bias`? Likely negligible for warm-start quality. Document and skip?
3. **Streaming → disk bridge:** Save 100k images to /tmp/ (uses ~10-20GB Colab disk) vs truly streaming DataLoader? Colab T4 has 78GB disk, 100k images at 512px ≈ 15GB. Fits. IterableDataset alternative avoids disk but complex with bucketing.
4. **Dataset split:** Training only, or hold out validation set? wandb tracks loss — validation images could go to Drive for visual inspection.
