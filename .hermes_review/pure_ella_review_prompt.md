You are doing a read-only adversarial Socratic review of a Colab notebook for Gemma 3 -> Stable Diffusion 1.5 pure ELLA-style conditioning.

Target: gemma3_sd_pure_ella_colab.ipynb
Evidence file: .hermes_review/gemma3_sd_pure_ella_review_evidence.txt
Full code dump: .hermes_review/gemma3_sd_pure_ella_colab_code_dump.py

Context/goal:
- Final inference must be CLIP-free: Gemma -> connector -> SD UNet.
- CLIP is allowed only as optional pretrain/teacher/diagnostic.
- Desired staged direction: first prove 77-token connector/adapter geometry, then long-context tokens, then sparse SaRA on UNet attn2.to_k/to_v only. No dense whole-UNet finetune.
- Use Socratic Method: frame findings as questions that force the design gate/assumption to be explicit.

Review requirements:
1. Review static notebook correctness and architecture safety. Do not modify files.
2. Identify blockers that would waste a Colab run or make evidence misleading.
3. Focus on: stage gating, long-token contract, CLIP-free proof, teacher/student losses, CFG delta diagnostics, sparse SaRA implementation, checkpoint/reload correctness, VRAM risk, and any hidden non-sparse parameter updates.
4. Return:
   - verdict: PASS / RUN_WITH_FIXES / DO_NOT_RUN
   - top Socratic questions
   - critical blockers
   - high/medium issues
   - concrete minimal fixes before running
   - what evidence/gates must be produced after execution.

Be concise but specific. Cite cell numbers or code snippets from the evidence when possible.