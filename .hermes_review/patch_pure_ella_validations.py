from pathlib import Path
import json, shutil, hashlib, textwrap

root = Path('/home/krz/PycharmProjects/gemma-stable-difussion')
nb_path = root / 'gemma3_sd_pure_ella_colab.ipynb'
raw = nb_path.read_text(encoding='utf-8')
sha_before = hashlib.sha256(raw.encode()).hexdigest()
backup = nb_path.with_name(f"gemma3_sd_pure_ella_colab.pre_extra_token_validation_{sha_before[:12]}.ipynb")
if not backup.exists():
    shutil.copy2(nb_path, backup)

nb = json.loads(raw)

def src(s: str):
    return [line + '\n' for line in textwrap.dedent(s).strip('\n').split('\n')]

nb['cells'][7]['source'] = src(r'''
# @title 1.1 Run config
EXPERIMENT_STAGE = "stage1_77_pretrain_then_ella"  # @param ["stage1_77_pretrain_then_ella", "stage2_long_context_no_sara", "stage3_long_context_with_sara"]
LONG_CONTEXT_TARGET = 128  # @param [128, 192, 256]
RUN_MODE = "short_train"  # @param ["diagnostic", "overfit_train", "short_train", "full_train"]
RUN_TRAINING = RUN_MODE in {"overfit_train", "short_train", "full_train"}
RUN_FINAL_PROOF = True
RUN_FIXED_VALIDATION_GRIDS = True
RUN_LONG_CONTEXT_DIAGNOSTICS = True
RUN_COMPLEX_PROMPT_GRIDS = True
RUN_SUFFIX_COUNTERFACTUAL_GRIDS = True
RUN_UNET_ATTENTION_PROOF = True

# Core experiment: pure ELLA. CLIP alignment optional pretrain/diagnostic, not runtime path.
RUN_CLIP_ALIGNMENT_PRETRAIN = True  # @param {type:"boolean"}
USE_CLIP_TEACHER_DELTA = True       # @param {type:"boolean"}
USE_CLIP_TEACHER_DELTA_PHASE2 = False  # @param {type:"boolean"}
ENABLE_UNET_GRADIENT_CHECKPOINTING = True  # @param {type:"boolean"}
PHASE1_SEMANTIC_ANCHOR_WEIGHT = 0.05
PHASE2_SEMANTIC_ANCHOR_WEIGHT = 0.05

assert LONG_CONTEXT_TARGET in {128, 192, 256}
if EXPERIMENT_STAGE == "stage1_77_pretrain_then_ella":
    CONTEXT_TOKENS = 77
elif EXPERIMENT_STAGE in {"stage2_long_context_no_sara", "stage3_long_context_with_sara"}:
    CONTEXT_TOKENS = LONG_CONTEXT_TARGET
else:
    raise ValueError(EXPERIMENT_STAGE)

CLIP_ANCHOR_TOKENS = 77
RUN_SARA_PHASE = EXPERIMENT_STAGE == "stage3_long_context_with_sara"
RUN_RELOADED_LONG_PROOF = CONTEXT_TOKENS > CLIP_ANCHOR_TOKENS
assert CONTEXT_TOKENS in {77, 128, 192, 256}
assert CONTEXT_TOKENS >= CLIP_ANCHOR_TOKENS
assert not RUN_SARA_PHASE or CONTEXT_TOKENS > CLIP_ANCHOR_TOKENS, "Sparse SaRA stage is only for post-77-token expansion runs"

BASE_SEED = 1234
random.seed(BASE_SEED); np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)

if RUN_MODE == "overfit_train":
    MAX_SAMPLES_PRETRAIN = 64
    MAX_SAMPLES_ELLA = 64
    PRETRAIN_EPOCHS = 20
    ELLA_EPOCHS = 40
    SARA_EPOCHS = 20
    PRETRAIN_MAX_OPT_STEPS = 300
    ELLA_MAX_OPT_STEPS = 600
    SARA_MAX_OPT_STEPS = 400
    VALIDATION_EVERY_OPT_STEPS = 50
    SHUFFLE_STREAMING = False
elif RUN_MODE == "short_train":
    MAX_SAMPLES_PRETRAIN = 6_000
    MAX_SAMPLES_ELLA = 6_000
    PRETRAIN_EPOCHS = 1
    ELLA_EPOCHS = 1
    SARA_EPOCHS = 1
    PRETRAIN_MAX_OPT_STEPS = 1_500
    ELLA_MAX_OPT_STEPS = 1_500
    SARA_MAX_OPT_STEPS = 500
    VALIDATION_EVERY_OPT_STEPS = 250
    SHUFFLE_STREAMING = True
elif RUN_MODE == "full_train":
    MAX_SAMPLES_PRETRAIN = 50_000
    MAX_SAMPLES_ELLA = 50_000
    PRETRAIN_EPOCHS = 1
    ELLA_EPOCHS = 1
    SARA_EPOCHS = 1
    PRETRAIN_MAX_OPT_STEPS = 12_500
    ELLA_MAX_OPT_STEPS = 12_500
    SARA_MAX_OPT_STEPS = 4_000
    VALIDATION_EVERY_OPT_STEPS = 500
    SHUFFLE_STREAMING = True
elif RUN_MODE == "diagnostic":
    MAX_SAMPLES_PRETRAIN = 128
    MAX_SAMPLES_ELLA = 128
    PRETRAIN_EPOCHS = 0
    ELLA_EPOCHS = 0
    SARA_EPOCHS = 0
    PRETRAIN_MAX_OPT_STEPS = 0
    ELLA_MAX_OPT_STEPS = 0
    SARA_MAX_OPT_STEPS = 0
    VALIDATION_EVERY_OPT_STEPS = 0
    SHUFFLE_STREAMING = True
else:
    raise ValueError(RUN_MODE)

TRAIN_BATCH_SIZE = 4
SHUFFLE_BUFFER = 10_000
GRAD_CLIP_NORM = 0.5

GEMMA_ID = "google/gemma-3-270m-it"
GEMMA_LAYER_INDEX = -1
MAX_GEMMA_LEN = 256
CLIP_ID = "openai/clip-vit-large-patch14"

CONNECTOR_WIDTH = 768
CONNECTOR_LAYERS = 4
CONNECTOR_HEADS = 8
CONNECTOR_FF_MULT = 4
CONNECTOR_DROPOUT = 0.0
CONNECTOR_TIME_EMBED_DIM = 768
CONNECTOR_EXTRA_GATE_INIT = -5.0

PRETRAIN_LR = 1e-4
ELLA_LR = 1e-4
SARA_LR = 1e-5
LAMBDA_DIFFUSION = 1.0
LAMBDA_TEACHER = 0.5
LAMBDA_TEXT_DELTA = 1.0

# SaRA-style sparse adaptation limited to SD text interface.
SARA_SCOPE = "attn2_kv_sparse"
SARA_TARGET_SUBSTRINGS = ("attn2.to_k", "attn2.to_v")
SARA_THRESHOLD = 1e-3
SARA_MAX_SPARSE_FRACTION_WARN = 0.02
SARA_MAX_SPARSE_FRACTION_ABORT = 0.05

EXTRA_TOKEN_DIAGNOSTIC_TIMESTEP = 500
EXTRA_TOKEN_DIAGNOSTIC_SEED = 777

VAL_PROMPTS = [
    "a cat sitting on a windowsill looking outside",
    "a watercolor painting of a mountain lake",
    "a neon-lit cyberpunk alleyway at night",
]

LONG_EVAL_SHORT_CONTROLS = [
    "a cinematic photo of a red vintage motorcycle parked beside a stone cottage",
    "a detailed product photograph of hiking boots on a wooden table",
]
LONG_EVAL_PROMPTS = [
    "a cinematic photo of a red vintage motorcycle parked beside a stone cottage, with a brass telescope on the seat, blue wildflowers in the basket, and a tiny owl perched on the handlebar at sunrise",
    "a detailed product photograph of hiking boots on a wooden table, with orange laces, a folded trail map, a silver compass, and raindrops on the leather",
]

COMPLEX_GENERATION_CASES = [
    {
        "name": "retrofuturist_magazine_cars",
        "prompt": "A highly detailed retrofuturist magazine infographic about future cars, laid out like a beautifully preserved issue of Popular Mechanics, with elegant diagram panels, mechanical callouts, polished concept-art rendering, vivid color, and the feeling of an award-winning poster-sized editorial spread.",
        "steps": 30,
        "guidance": 7.0,
        "seed": 3197632166,
        "width": 768,
        "height": 960,
    },
    {
        "name": "warrior_princess_poster",
        "prompt": "A dramatic poster of a warrior princess standing centered on a hill as the main cinematic key visual, with intricate linework, vibrant colors, panoramic scale, breathtaking fantasy atmosphere, and the finish of a carefully painted illustrated poster.",
        "steps": 30,
        "guidance": 7.0,
        "seed": 4267154965,
        "width": 768,
        "height": 960,
    },
]

SUFFIX_COUNTERFACTUAL_CASES = [
    {
        "name": "motorcycle_counterfactual",
        "short_prompt": "A cinematic photograph of a red vintage motorcycle parked beside a weathered stone cottage at sunrise.",
        "prompt_a": "A cinematic photograph of a red vintage motorcycle parked beside a weathered stone cottage at sunrise, with dew on the grass, warm light across the walls, ivy climbing the chimney, soft mist in the distance, worn shutters on the windows, a wicker basket near the front wheel, and a quiet meadow stretching beyond the cottage, with a brass telescope on the seat, blue wildflowers spilling from the basket, and a tiny owl perched on the handlebar.",
        "prompt_b": "A cinematic photograph of a red vintage motorcycle parked beside a weathered stone cottage at sunrise, with dew on the grass, warm light across the walls, ivy climbing the chimney, soft mist in the distance, worn shutters on the windows, a wicker basket near the front wheel, and a quiet meadow stretching beyond the cottage, with a folded paper map on the seat, yellow marigolds spilling from the basket, and a silver pocket watch hanging from the handlebar.",
        "seed": 1201,
        "steps": 30,
        "guidance": 5.5,
        "width": 768,
        "height": 960,
    },
    {
        "name": "boots_counterfactual",
        "short_prompt": "A detailed product photograph of hiking boots on a wooden table.",
        "prompt_a": "A detailed product photograph of hiking boots on a wooden table in soft window light, with visible leather grain, careful studio composition, a neutral workshop background, dust on the tabletop, shallow depth of field, subtle reflections on the eyelets, and the sense of a premium outdoor catalog page, with orange laces, a folded trail map, a silver compass, and raindrops on the leather.",
        "prompt_b": "A detailed product photograph of hiking boots on a wooden table in soft window light, with visible leather grain, careful studio composition, a neutral workshop background, dust on the tabletop, shallow depth of field, subtle reflections on the eyelets, and the sense of a premium outdoor catalog page, with green laces, a field notebook, a brass whistle, and dried mud on the leather.",
        "seed": 2202,
        "steps": 30,
        "guidance": 5.5,
        "width": 768,
        "height": 960,
    },
]

LONG_CONTEXT_DIAGNOSTIC_PROMPTS = [
    LONG_EVAL_PROMPTS[0],
    LONG_EVAL_PROMPTS[1],
    COMPLEX_GENERATION_CASES[0]["prompt"],
]

RUN_CONFIG = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (str, int, float, bool, tuple, list, dict, type(None)))}
print(json.dumps({k: RUN_CONFIG[k] for k in [
    "EXPERIMENT_STAGE", "RUN_MODE", "RUN_CLIP_ALIGNMENT_PRETRAIN", "USE_CLIP_TEACHER_DELTA",
    "CONTEXT_TOKENS", "CLIP_ANCHOR_TOKENS", "RUN_SARA_PHASE", "RUN_RELOADED_LONG_PROOF", "SARA_SCOPE"
]}, indent=2))
if EXPERIMENT_STAGE == "stage1_77_pretrain_then_ella" and not RUN_CLIP_ALIGNMENT_PRETRAIN:
    print("Stage 1 note: optional CLIP alignment pretrain disabled by user; notebook will run direct pure ELLA from random connector init.")

if WANDB_ENABLED:
    wandb.init(project="gemma3-sd-pure-ella", name=f"pure-ella-{EXPERIMENT_STAGE}-{RUN_MODE}-L{CONTEXT_TOKENS}", config=RUN_CONFIG)
else:
    class _NoWandb:
        def log(self, *args, **kwargs):
            pass
        def finish(self):
            pass
    wandb = _NoWandb()
''')

nb['cells'][15]['source'] = src(r'''
# @title 3.3 Geometry loss + diagnostics
import contextlib
from diffusers.models.attention_processor import AttnProcessor

class ClipGeometryLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_cos=0.5, w_norm=0.25, w_ctr=0.2, temp=0.07):
        super().__init__()
        self.w_mse = w_mse
        self.w_cos = w_cos
        self.w_norm = w_norm
        self.w_ctr = w_ctr
        self.temp = temp

    @staticmethod
    def _mask(mask, x):
        return mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1)

    @staticmethod
    def _pooled(x, mask):
        m = ClipGeometryLoss._mask(mask, x).to(dtype=x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def forward(self, pred, target, mask):
        pred = pred[:, :target.shape[1], :].float()
        target = target.float()
        m = self._mask(mask, pred)
        pred_ln = F.layer_norm(pred, pred.shape[-1:])
        target_ln = F.layer_norm(target, target.shape[-1:])
        mse = ((pred_ln - target_ln).pow(2) * m).sum() / m.sum().clamp_min(1.0) / pred.shape[-1]
        cos = 1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)
        cos = (cos * m.squeeze(-1).float()).sum() / m.squeeze(-1).float().sum().clamp_min(1.0)
        pred_norm = pred.norm(dim=-1).clamp_min(1e-6)
        target_norm = target.norm(dim=-1).clamp_min(1e-6)
        norm = (torch.log(pred_norm / target_norm).abs() * m.squeeze(-1).float()).sum() / m.squeeze(-1).float().sum().clamp_min(1.0)
        pp = F.normalize(self._pooled(pred, mask), dim=-1)
        tt = F.normalize(self._pooled(target, mask), dim=-1)
        logits = pp @ tt.t() / self.temp
        labels = torch.arange(pred.shape[0], device=pred.device)
        ctr = F.cross_entropy(logits, labels) if pred.shape[0] > 1 else pred.new_tensor(0.0)
        pooled_cos = (pp * tt).sum(dim=-1).mean()
        total = self.w_mse * mse + self.w_cos * cos + self.w_norm * norm + self.w_ctr * ctr
        return {
            "total": total,
            "mse": mse,
            "cos": cos,
            "norm": norm,
            "ctr": ctr,
            "pooled_cos": pooled_cos,
            "norm_ratio": pred_norm.mean() / target_norm.mean().clamp_min(1e-6),
        }

clip_geometry_loss = ClipGeometryLoss()


def _rel_diff(a, b, eps=1e-8):
    a = a.float(); b = b.float()
    return float((a - b).pow(2).mean().sqrt().item() / (b.pow(2).mean().sqrt().item() + eps))


def _maybe_float(x):
    return None if x is None else float(x)


def _log_metrics(prefix, metrics):
    payload = {}
    for k, v in metrics.items():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            payload[f"{prefix}/{k}"] = float(v)
    if payload:
        wandb.log(payload)


def zero_extra_tokens(ctx, anchor_tokens=CLIP_ANCHOR_TOKENS):
    if ctx.shape[1] <= anchor_tokens:
        return ctx
    out = ctx.clone()
    out[:, anchor_tokens:, :] = 0
    return out


def summarize_context_tokens(ctx, anchor_tokens=CLIP_ANCHOR_TOKENS, extra_gate=None):
    ctx = ctx.float()
    base = ctx[:, :anchor_tokens, :]
    extra = ctx[:, anchor_tokens:, :]
    base_rms = base.pow(2).mean().sqrt().item()
    out = {
        "base_rms": base_rms,
        "extra_gate": _maybe_float(extra_gate),
    }
    if extra.numel() == 0:
        out.update({
            "extra_rms": 0.0,
            "extra_abs_mean": 0.0,
            "extra_to_base_ratio": 0.0,
        })
    else:
        extra_rms = extra.pow(2).mean().sqrt().item()
        out.update({
            "extra_rms": extra_rms,
            "extra_abs_mean": extra.abs().mean().item(),
            "extra_to_base_ratio": extra_rms / max(base_rms, 1e-8),
        })
    return out


def connector_extra_grad_stats(model=ella_connector, anchor_tokens=CLIP_ANCHOR_TOKENS):
    if getattr(model, "extra_gate_logit", None) is None:
        return {
            "extra_gate_grad_norm": None,
            "extra_query_grad_norm": None,
            "extra_pos_grad_norm": None,
        }
    stats = {}
    gate_grad = model.extra_gate_logit.grad
    stats["extra_gate_grad_norm"] = float(gate_grad.detach().float().norm().item()) if gate_grad is not None else None
    q_grad = getattr(model.query_tokens, "grad", None)
    p_grad = getattr(model.pos_emb, "grad", None)
    if q_grad is not None and q_grad.shape[1] > anchor_tokens:
        stats["extra_query_grad_norm"] = float(q_grad[:, anchor_tokens:, :].detach().float().norm().item())
    else:
        stats["extra_query_grad_norm"] = None
    if p_grad is not None and p_grad.shape[1] > anchor_tokens:
        stats["extra_pos_grad_norm"] = float(p_grad[:, anchor_tokens:, :].detach().float().norm().item())
    else:
        stats["extra_pos_grad_norm"] = None
    return stats


def gemma_token_prefix_report(prompt_a, prompt_b, anchor_tokens=CLIP_ANCHOR_TOKENS):
    enc_a = gemma_tokenizer(prompt_a, truncation=True, max_length=MAX_GEMMA_LEN, add_special_tokens=True)
    enc_b = gemma_tokenizer(prompt_b, truncation=True, max_length=MAX_GEMMA_LEN, add_special_tokens=True)
    ids_a = list(enc_a["input_ids"])
    ids_b = list(enc_b["input_ids"])
    shared = 0
    for aa, bb in zip(ids_a, ids_b):
        if aa != bb:
            break
        shared += 1
    anchor = min(anchor_tokens, len(ids_a), len(ids_b))
    same_first_anchor = ids_a[:anchor] == ids_b[:anchor]
    first_diff = shared if shared < min(len(ids_a), len(ids_b)) else None
    suffix_a = gemma_tokenizer.decode(ids_a[anchor_tokens:anchor_tokens+24], skip_special_tokens=True)
    suffix_b = gemma_tokenizer.decode(ids_b[anchor_tokens:anchor_tokens+24], skip_special_tokens=True)
    report = {
        "len_a": len(ids_a),
        "len_b": len(ids_b),
        "shared_prefix_tokens": shared,
        "same_first_anchor_tokens": bool(same_first_anchor),
        "first_diff_index": -1 if first_diff is None else int(first_diff),
        "suffix_a_after_anchor": suffix_a,
        "suffix_b_after_anchor": suffix_b,
    }
    return report


class RecordingCrossAttnProcessor(AttnProcessor):
    def __init__(self, name, store, anchor_tokens=CLIP_ANCHOR_TOKENS):
        super().__init__()
        self.name = name
        self.store = store
        self.anchor_tokens = int(anchor_tokens)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        sequence_length = encoder_hidden_states.shape[1]
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        src_tokens = attention_probs.shape[-1]
        anchor = min(self.anchor_tokens, src_tokens)
        prefix_share = attention_probs[..., :anchor].sum(dim=-1).mean().item()
        extra_share = attention_probs[..., anchor:].sum(dim=-1).mean().item() if src_tokens > anchor else 0.0
        self.store.append({
            "name": self.name,
            "prefix_share": float(prefix_share),
            "extra_share": float(extra_share),
            "src_tokens": int(src_tokens),
        })

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


@torch.no_grad()
def forward_unet_with_optional_attn(noisy, t, ctx, attention_mask=None, record_attn=False, anchor_tokens=CLIP_ANCHOR_TOKENS):
    if not record_attn:
        pred = unet(noisy, t, encoder_hidden_states=ctx, encoder_attention_mask=attention_mask).sample
        return pred, {}
    if not hasattr(unet, "attn_processors"):
        pred = unet(noisy, t, encoder_hidden_states=ctx, encoder_attention_mask=attention_mask).sample
        return pred, {"attention_recording_supported": 0.0}
    old = dict(unet.attn_processors)
    store = []
    patched = {}
    for name, proc in old.items():
        if "attn2" in name:
            patched[name] = RecordingCrossAttnProcessor(name, store, anchor_tokens=anchor_tokens)
        else:
            patched[name] = proc
    unet.set_attn_processor(patched)
    try:
        pred = unet(noisy, t, encoder_hidden_states=ctx, encoder_attention_mask=attention_mask).sample
    finally:
        unet.set_attn_processor(old)
    if not store:
        return pred, {"attention_recording_supported": 0.0}
    prefix_share = float(np.mean([x["prefix_share"] for x in store]))
    extra_share = float(np.mean([x["extra_share"] for x in store]))
    return pred, {
        "attention_recording_supported": 1.0,
        "attn_prefix_share": prefix_share,
        "attn_extra_share": extra_share,
        "attn_extra_ratio": extra_share / max(prefix_share + extra_share, 1e-8),
        "attn_layers_recorded": float(len(store)),
    }


@torch.no_grad()
def connector_prompt_sensitivity(prompts, timestep=500, label="connector"):
    ella_connector.eval(); unet.eval()
    gen = torch.Generator(device=device).manual_seed(123)
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    t = torch.tensor([timestep], device=device).long()
    preds = []
    for ptxt in prompts:
        gh, gm = encode_gemma_prompts([ptxt])
        ctx = ella_connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=CONTEXT_TOKENS)
        pred = unet(latent, t, encoder_hidden_states=ctx).sample.float()
        preds.append(pred)
    base = preds[0].pow(2).mean().sqrt().item() + 1e-8
    vals = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            diff = (preds[i] - preds[j]).pow(2).mean().sqrt().item() / base
            vals.append(diff)
            print(f"[{label}] {i} vs {j}: rel_diff={diff:.6f}")
    mean_val = float(np.mean(vals)) if vals else 0.0
    wandb.log({f"diagnostics/{label}_mean_relative_diff": mean_val})
    return mean_val


@torch.no_grad()
def teacher_student_delta_alignment(prompt, timestep=500, label="delta"):
    if clip_model is None:
        print(f"[{label}] CLIP unavailable; skipped")
        return {}
    ella_connector.eval(); unet.eval()
    gen = torch.Generator(device=device).manual_seed(777)
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    noise = torch.randn_like(latent)
    t = torch.tensor([timestep], device=device).long()
    noisy = scheduler.add_noise(latent, noise, t)
    noisy_pair = torch.cat([noisy, noisy], dim=0)
    t_pair = torch.cat([t, t], dim=0)
    ch, cm = encode_clip_prompts([prompt])
    uch, ucm = encode_clip_prompts([""])
    clip_h = torch.cat([ch, uch], dim=0)
    clip_m = torch.cat([cm, ucm], dim=0)
    teacher = unet(noisy_pair, t_pair, encoder_hidden_states=clip_h.to(dtype=unet_dtype), encoder_attention_mask=clip_m).sample.float()
    teacher_cond, teacher_uncond = teacher.chunk(2)
    gh, gm = encode_gemma_prompts([prompt])
    ugh, ugm = encode_gemma_prompts([""])
    gemma_h = torch.cat([gh, ugh], dim=0)
    gemma_m = torch.cat([gm, ugm], dim=0)
    ctx = ella_connector(gemma_h.to(dtype=unet_dtype), t_pair, gemma_m, context_tokens=CONTEXT_TOKENS)
    student = unet(noisy_pair, t_pair, encoder_hidden_states=ctx).sample.float()
    student_cond, student_uncond = student.chunk(2)
    td = (teacher_cond - teacher_uncond).flatten()
    sd = (student_cond - student_uncond).flatten()
    cos = float(F.cosine_similarity(sd[None], td[None]).item())
    ratio = float(sd.norm().item() / max(td.norm().item(), 1e-8))
    print(f"[{label}] delta_cos={cos:.6f} norm_ratio={ratio:.6f}")
    wandb.log({f"diagnostics/{label}_delta_cos": cos, f"diagnostics/{label}_norm_ratio": ratio})
    return {"delta_cos": cos, "norm_ratio": ratio}


@torch.no_grad()
def extra_token_ablation_metrics(prompt, negative_prompt="", timestep=EXTRA_TOKEN_DIAGNOSTIC_TIMESTEP, seed=EXTRA_TOKEN_DIAGNOSTIC_SEED, context_tokens=CONTEXT_TOKENS, label="extra_tokens"):
    if context_tokens <= CLIP_ANCHOR_TOKENS:
        print(f"[{label}] context_tokens={context_tokens}; extra-token diagnostics skipped at 77-token stage")
        return {"skipped": 1.0}
    ella_connector.eval(); unet.eval(); gemma_model.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    noise = torch.randn_like(latent)
    t = torch.tensor([int(timestep)], device=device).long()
    noisy = scheduler.add_noise(latent, noise, t)

    gh, gm = encode_gemma_prompts([prompt])
    ugh, ugm = encode_gemma_prompts([negative_prompt or ""])
    ctx77 = ella_connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=CLIP_ANCHOR_TOKENS)
    ctx_full = ella_connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=context_tokens)
    ctx_zero = zero_extra_tokens(ctx_full, CLIP_ANCHOR_TOKENS)
    uctx77 = ella_connector(ugh.to(dtype=unet_dtype), t, ugm, context_tokens=CLIP_ANCHOR_TOKENS)
    uctx_full = ella_connector(ugh.to(dtype=unet_dtype), t, ugm, context_tokens=context_tokens)
    uctx_zero = zero_extra_tokens(uctx_full, CLIP_ANCHOR_TOKENS)

    extra_gate = torch.sigmoid(ella_connector.extra_gate_logit).item() if ella_connector.extra_gate_logit is not None else None
    metrics = summarize_context_tokens(ctx_full, extra_gate=extra_gate)
    metrics["prefix_drift_rel"] = _rel_diff(ctx_full[:, :CLIP_ANCHOR_TOKENS, :], ctx77)

    pred77, _ = forward_unet_with_optional_attn(noisy, t, ctx77, record_attn=False)
    predfull, attn_stats = forward_unet_with_optional_attn(noisy, t, ctx_full, record_attn=RUN_UNET_ATTENTION_PROOF)
    predzero, _ = forward_unet_with_optional_attn(noisy, t, ctx_zero, record_attn=False)
    upred77, _ = forward_unet_with_optional_attn(noisy, t, uctx77, record_attn=False)
    upredfull, _ = forward_unet_with_optional_attn(noisy, t, uctx_full, record_attn=False)
    upredzero, _ = forward_unet_with_optional_attn(noisy, t, uctx_zero, record_attn=False)

    delta77 = pred77 - upred77
    deltafull = predfull - upredfull
    deltazero = predzero - upredzero

    metrics.update(attn_stats)
    metrics.update({
        "rel_diff_77_vs_full_cond": _rel_diff(predfull, pred77),
        "rel_diff_full_vs_zeroextra_cond": _rel_diff(predfull, predzero),
        "rel_diff_77_vs_full_delta": _rel_diff(deltafull, delta77),
        "rel_diff_full_vs_zeroextra_delta": _rel_diff(deltafull, deltazero),
    })
    print(f"[{label}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items() if isinstance(v, (int, float))))
    _log_metrics(f"extra_tokens/{label}", metrics)
    return metrics


@torch.no_grad()
def suffix_counterfactual_metrics(case, timestep=EXTRA_TOKEN_DIAGNOSTIC_TIMESTEP, context_tokens=CONTEXT_TOKENS, label="suffix_case"):
    if context_tokens <= CLIP_ANCHOR_TOKENS:
        print(f"[{label}] context_tokens={context_tokens}; suffix-only diagnostics skipped at 77-token stage")
        return {"skipped": 1.0}
    prompt_a = case["prompt_a"]
    prompt_b = case["prompt_b"]
    short_prompt = case["short_prompt"]
    seed = int(case.get("seed", EXTRA_TOKEN_DIAGNOSTIC_SEED))
    report = gemma_token_prefix_report(prompt_a, prompt_b, anchor_tokens=CLIP_ANCHOR_TOKENS)
    print(f"[{label}] token boundary report: {report}")

    gen = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=unet_dtype)
    noise = torch.randn_like(latent)
    t = torch.tensor([int(timestep)], device=device).long()
    noisy = scheduler.add_noise(latent, noise, t)

    gh_a, gm_a = encode_gemma_prompts([prompt_a])
    gh_b, gm_b = encode_gemma_prompts([prompt_b])
    gh_s, gm_s = encode_gemma_prompts([short_prompt])
    ugh, ugm = encode_gemma_prompts([""])

    ctx_a_full = ella_connector(gh_a.to(dtype=unet_dtype), t, gm_a, context_tokens=context_tokens)
    ctx_b_full = ella_connector(gh_b.to(dtype=unet_dtype), t, gm_b, context_tokens=context_tokens)
    ctx_a_77 = ella_connector(gh_a.to(dtype=unet_dtype), t, gm_a, context_tokens=CLIP_ANCHOR_TOKENS)
    ctx_b_77 = ella_connector(gh_b.to(dtype=unet_dtype), t, gm_b, context_tokens=CLIP_ANCHOR_TOKENS)
    ctx_short_77 = ella_connector(gh_s.to(dtype=unet_dtype), t, gm_s, context_tokens=CLIP_ANCHOR_TOKENS)
    uctx_full = ella_connector(ugh.to(dtype=unet_dtype), t, ugm, context_tokens=context_tokens)
    uctx_77 = ella_connector(ugh.to(dtype=unet_dtype), t, ugm, context_tokens=CLIP_ANCHOR_TOKENS)

    pred_a_full, _ = forward_unet_with_optional_attn(noisy, t, ctx_a_full, record_attn=False)
    pred_b_full, _ = forward_unet_with_optional_attn(noisy, t, ctx_b_full, record_attn=False)
    pred_a_77, _ = forward_unet_with_optional_attn(noisy, t, ctx_a_77, record_attn=False)
    pred_b_77, _ = forward_unet_with_optional_attn(noisy, t, ctx_b_77, record_attn=False)
    pred_short_77, _ = forward_unet_with_optional_attn(noisy, t, ctx_short_77, record_attn=False)
    upred_full, _ = forward_unet_with_optional_attn(noisy, t, uctx_full, record_attn=False)
    upred_77, _ = forward_unet_with_optional_attn(noisy, t, uctx_77, record_attn=False)

    delta_a_full = pred_a_full - upred_full
    delta_b_full = pred_b_full - upred_full
    delta_a_77 = pred_a_77 - upred_77
    delta_b_77 = pred_b_77 - upred_77

    metrics = {
        "same_first_anchor_tokens": float(report["same_first_anchor_tokens"]),
        "first_diff_index": float(report["first_diff_index"]),
        "shared_prefix_tokens": float(report["shared_prefix_tokens"]),
        "suffix_swap_rel_diff_full_cond": _rel_diff(pred_a_full, pred_b_full),
        "suffix_swap_rel_diff_trunc77_cond": _rel_diff(pred_a_77, pred_b_77),
        "suffix_swap_rel_diff_full_delta": _rel_diff(delta_a_full, delta_b_full),
        "suffix_swap_rel_diff_trunc77_delta": _rel_diff(delta_a_77, delta_b_77),
        "short_vs_full_a_cond": _rel_diff(pred_a_full, pred_short_77),
        "short_vs_full_b_cond": _rel_diff(pred_b_full, pred_short_77),
    }
    print(f"[{label}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items() if isinstance(v, (int, float))))
    _log_metrics(f"suffix_counterfactual/{label}", metrics)
    return {"metrics": metrics, "token_report": report}


@torch.no_grad()
def run_long_context_numeric_suite(label_prefix="final_long_context"):
    if CONTEXT_TOKENS <= CLIP_ANCHOR_TOKENS:
        print(f"[{label_prefix}] 77-token stage; long-context numeric suite skipped")
        return {}
    results = {}
    results["ablation"] = extra_token_ablation_metrics(LONG_CONTEXT_DIAGNOSTIC_PROMPTS[0], label=f"{label_prefix}_ablation")
    results["suffix"] = suffix_counterfactual_metrics(SUFFIX_COUNTERFACTUAL_CASES[0], label=f"{label_prefix}_suffix0")
    return results
''')

nb['cells'][19]['source'] = src(r'''
# @title 5.1 Generation helpers
from diffusers import DPMSolverMultistepScheduler

def latent_shape_from_size(width=512, height=512):
    return height // 8, width // 8

@torch.no_grad()
def decode_latents_to_image(latents):
    vae_dtype = next(vae.parameters()).dtype
    x = (latents / vae.config.scaling_factor).to(dtype=vae_dtype)
    img = vae.decode(x).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    arr = img.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    return Image.fromarray((arr * 255).astype(np.uint8))

def save_validation_grid(images, labels, path, title):
    cols = min(len(images), 3)
    rows = math.ceil(len(images) / cols)
    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(labels[i][:80], fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    print("Saved:", path)

@torch.no_grad()
def generate_ella(prompt, steps=VAL_STEPS, guidance=VAL_GUIDANCE, seed=VAL_SEED, negative_prompt="", width=512, height=512, context_tokens=CONTEXT_TOKENS):
    ella_connector.eval(); unet.eval(); vae.eval(); gemma_model.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    infer_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    infer_scheduler.set_timesteps(int(steps), device=device)
    cond_g, cond_m = encode_gemma_prompts([prompt])
    uncond_g, uncond_m = encode_gemma_prompts([negative_prompt or ""])
    g_h = torch.cat([uncond_g, cond_g], dim=0)
    g_m = torch.cat([uncond_m, cond_m], dim=0)
    lh, lw = latent_shape_from_size(width=width, height=height)
    latents = torch.randn(1, 4, lh, lw, generator=gen, device=device, dtype=unet_dtype) * infer_scheduler.init_noise_sigma
    for t in tqdm(infer_scheduler.timesteps, desc=f"ELLA L{context_tokens}"):
        inp = torch.cat([latents] * 2, dim=0)
        inp = infer_scheduler.scale_model_input(inp, t)
        t_pair = t.unsqueeze(0).expand(2).to(device=device)
        ctx = ella_connector(g_h.to(dtype=unet_dtype), t_pair, g_m, context_tokens=context_tokens)
        pred = unet(inp, t, encoder_hidden_states=ctx).sample
        u, c = pred.chunk(2)
        pred = u + guidance * (c - u)
        latents = infer_scheduler.step(pred, t, latents).prev_sample
    return decode_latents_to_image(latents)

@torch.no_grad()
def generate_clip_teacher(prompt, steps=VAL_STEPS, guidance=VAL_GUIDANCE, seed=VAL_SEED, negative_prompt="", width=512, height=512):
    if clip_model is None:
        return None
    unet.eval(); vae.eval(); clip_model.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    infer_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    infer_scheduler.set_timesteps(int(steps), device=device)
    ch, cm = encode_clip_prompts([prompt])
    uh, um = encode_clip_prompts([negative_prompt or ""])
    clip_h = torch.cat([uh, ch], dim=0)
    clip_m = torch.cat([um, cm], dim=0)
    lh, lw = latent_shape_from_size(width=width, height=height)
    latents = torch.randn(1, 4, lh, lw, generator=gen, device=device, dtype=unet_dtype) * infer_scheduler.init_noise_sigma
    for t in tqdm(infer_scheduler.timesteps, desc="CLIP teacher"):
        inp = torch.cat([latents] * 2, dim=0)
        inp = infer_scheduler.scale_model_input(inp, t)
        pred = unet(inp, t, encoder_hidden_states=clip_h.to(dtype=unet_dtype), encoder_attention_mask=clip_m).sample
        u, c = pred.chunk(2)
        pred = u + guidance * (c - u)
        latents = infer_scheduler.step(pred, t, latents).prev_sample
    return decode_latents_to_image(latents)

@torch.no_grad()
def fixed_overfit_loss(label="fixed_overfit", timestep=500, seed=777):
    if not OVERFIT_EVAL_BATCH:
        print(f"[{label}] no OVERFIT_EVAL_BATCH; skipped")
        return None
    ella_connector.eval(); unet.eval(); vae.eval()
    imgs = OVERFIT_EVAL_BATCH["image"].to(device=device, dtype=unet_dtype)
    captions = OVERFIT_EVAL_BATCH["caption"]
    gen = torch.Generator(device=device).manual_seed(seed)
    latent = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
    noise = torch.randn(latent.shape, generator=gen, device=device, dtype=latent.dtype)
    t = torch.full((latent.shape[0],), int(timestep), device=device, dtype=torch.long)
    noisy = scheduler.add_noise(latent, noise, t)
    gh, gm = encode_gemma_prompts(captions)
    ctx = ella_connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=CONTEXT_TOKENS)
    pred = unet(noisy, t, encoder_hidden_states=ctx).sample
    loss = F.mse_loss(pred.float(), noise.float()).item()
    print(f"[{label}] fixed overfit MSE @t={timestep}: {loss:.6f}")
    wandb.log({f"validation/{label}_mse": loss})
    return loss

@torch.no_grad()
def generate_case_image(case, context_tokens=CONTEXT_TOKENS):
    return generate_ella(
        case["prompt"],
        steps=case.get("steps", VAL_STEPS),
        guidance=case.get("guidance", VAL_GUIDANCE),
        seed=case.get("seed", VAL_SEED),
        width=case.get("width", 512),
        height=case.get("height", 512),
        context_tokens=context_tokens,
    )

@torch.no_grad()
def save_complex_case_grid(cases, path, title, context_tokens=CONTEXT_TOKENS):
    imgs, labels = [], []
    for case in cases:
        print(f"Generating complex case: {case['name']}")
        imgs.append(generate_case_image(case, context_tokens=context_tokens))
        labels.append(f"L{context_tokens}: {case['name']}")
    save_validation_grid(imgs, labels, path, title)

@torch.no_grad()
def save_suffix_counterfactual_grids(cases, path_prefix, title_prefix, context_tokens=CONTEXT_TOKENS):
    if context_tokens <= CLIP_ANCHOR_TOKENS:
        print("Suffix counterfactual grids skipped at 77-token stage")
        return []
    out_paths = []
    for case in cases:
        imgs, labels = [], []
        print(f"Generating suffix counterfactual grid: {case['name']}")
        imgs.append(generate_ella(case["short_prompt"], steps=case.get("steps", VAL_STEPS), guidance=case.get("guidance", VAL_GUIDANCE), seed=case.get("seed", VAL_SEED), width=case.get("width", 512), height=case.get("height", 512), context_tokens=CLIP_ANCHOR_TOKENS))
        labels.append(f"short@77 {case['name']}")
        imgs.append(generate_ella(case["prompt_a"], steps=case.get("steps", VAL_STEPS), guidance=case.get("guidance", VAL_GUIDANCE), seed=case.get("seed", VAL_SEED), width=case.get("width", 512), height=case.get("height", 512), context_tokens=context_tokens))
        labels.append(f"fullA@L{context_tokens} {case['name']}")
        imgs.append(generate_ella(case["prompt_b"], steps=case.get("steps", VAL_STEPS), guidance=case.get("guidance", VAL_GUIDANCE), seed=case.get("seed", VAL_SEED), width=case.get("width", 512), height=case.get("height", 512), context_tokens=context_tokens))
        labels.append(f"fullB@L{context_tokens} {case['name']}")
        out_path = f"{path_prefix}_{case['name']}.png"
        save_validation_grid(imgs, labels, out_path, f"{title_prefix}: {case['name']}")
        out_paths.append(out_path)
    return out_paths
''')

nb['cells'][21]['source'] = src(r'''
# @title 6.1 Optional CLIP alignment pretrain for first 77 connector outputs
if not RUN_TRAINING or not RUN_CLIP_ALIGNMENT_PRETRAIN or PRETRAIN_EPOCHS <= 0:
    print("CLIP alignment pretrain skipped")
else:
    assert clip_model is not None, "CLIP pretrain requires CLIP loaded"
    for p in unet.parameters(): p.requires_grad_(False)
    for p in vae.parameters(): p.requires_grad_(False)
    for p in gemma_model.parameters(): p.requires_grad_(False)
    for p in clip_model.parameters(): p.requires_grad_(False)
    for p in ella_connector.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.AdamW(ella_connector.parameters(), lr=PRETRAIN_LR, weight_decay=0.01, eps=1e-6)
    ella_connector.train(); gemma_model.eval(); clip_model.eval(); unet.eval(); vae.eval()
    plan = print_stage_plan("clip_pretrain", MAX_SAMPLES_PRETRAIN, TRAIN_BATCH_SIZE, PRETRAIN_EPOCHS, PRETRAIN_MAX_OPT_STEPS)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    log_vram("clip_pretrain_start", 0)
    opt_step = 0
    best = None
    stop = False
    t0 = time.time()
    for epoch in range(PRETRAIN_EPOCHS):
        dl = make_streaming_dataloader(phase=10, epoch=epoch, max_samples=MAX_SAMPLES_PRETRAIN, batch_size=TRAIN_BATCH_SIZE)
        progress = tqdm(dl, desc=f"CLIP-pretrain {epoch+1}/{PRETRAIN_EPOCHS}")
        for batch in progress:
            captions = _as_prompt_list(batch["caption"])
            with torch.no_grad():
                gh, gm = encode_gemma_prompts(captions)
                ch, cm = encode_clip_prompts(captions)
            t = torch.zeros(len(captions), device=device, dtype=torch.long)
            pred = ella_connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=CLIP_ANCHOR_TOKENS)
            loss_dict = clip_geometry_loss(pred, ch, cm)
            loss = loss_dict["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("CLIP alignment loss NaN/Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_stats = connector_extra_grad_stats(ella_connector)
            nn.utils.clip_grad_norm_(ella_connector.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            opt_step += 1
            best = float(loss.item()) if best is None else min(best, float(loss.item()))
            if opt_step % 50 == 0:
                print(f"pretrain step {opt_step}: total={loss.item():.5f} mse={loss_dict['mse'].item():.5f} cos={loss_dict['cos'].item():.5f} pooled_cos={loss_dict['pooled_cos'].item():.5f} norm_ratio={loss_dict['norm_ratio'].item():.3f}")
                wandb.log({
                    "pretrain/step": opt_step,
                    "pretrain/loss": loss.item(),
                    "pretrain/mse": loss_dict["mse"].item(),
                    "pretrain/cos": loss_dict["cos"].item(),
                    "pretrain/pooled_cos": loss_dict["pooled_cos"].item(),
                    "pretrain/norm_ratio": loss_dict["norm_ratio"].item(),
                    "pretrain/extra_gate_grad_norm": grad_stats["extra_gate_grad_norm"] if grad_stats["extra_gate_grad_norm"] is not None else 0.0,
                    "pretrain/extra_query_grad_norm": grad_stats["extra_query_grad_norm"] if grad_stats["extra_query_grad_norm"] is not None else 0.0,
                    "pretrain/extra_pos_grad_norm": grad_stats["extra_pos_grad_norm"] if grad_stats["extra_pos_grad_norm"] is not None else 0.0,
                })
            progress.set_postfix({"loss": f"{loss.item():.4f}", "best": f"{best:.4f}"})
            if PRETRAIN_MAX_OPT_STEPS is not None and opt_step >= PRETRAIN_MAX_OPT_STEPS:
                stop = True
                print("Stopping CLIP pretrain at step cap", opt_step)
                break
        if stop:
            break
    log_vram("clip_pretrain_end", opt_step)
    print(f"CLIP pretrain done: steps={opt_step} best={best} elapsed={(time.time()-t0)/60:.1f}m")
    torch.save({
        "connector_state_dict": {k: v.detach().cpu() for k, v in ella_connector.state_dict().items()},
        "config": RUN_CONFIG,
        "stage": "clip_alignment_pretrain",
    }, f"{DRIVE_OUT}/ella_connector_clip_pretrain.pt")
    connector_prompt_sensitivity(VAL_PROMPTS, label="post_clip_pretrain")
    teacher_student_delta_alignment(VAL_PROMPTS[0], label="post_clip_pretrain")
''')

nb['cells'][23]['source'] = src(r'''
# @title 7.1 ELLA connector diffusion + teacher-delta training, frozen UNet
if not RUN_TRAINING or ELLA_EPOCHS <= 0:
    print("ELLA connector phase skipped")
else:
    for p in unet.parameters(): p.requires_grad_(False)
    for p in vae.parameters(): p.requires_grad_(False)
    for p in gemma_model.parameters(): p.requires_grad_(False)
    if clip_model is not None:
        for p in clip_model.parameters(): p.requires_grad_(False)
    for p in ella_connector.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.AdamW(ella_connector.parameters(), lr=ELLA_LR, weight_decay=0.01, eps=1e-6)
    use_clip_teacher_delta_phase1 = bool(USE_CLIP_TEACHER_DELTA and clip_model is not None)
    print("Phase 1 forward mode:", "paired_cfg_teacher" if use_clip_teacher_delta_phase1 else "conditional_only")
    ella_connector.train(); unet.eval(); vae.eval(); gemma_model.eval()
    if clip_model is not None: clip_model.eval()
    plan = print_stage_plan("ella_frozen_unet", MAX_SAMPLES_ELLA, TRAIN_BATCH_SIZE, ELLA_EPOCHS, ELLA_MAX_OPT_STEPS)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    log_vram("ella_start", 0)
    opt_step = 0
    best = None
    stop = False
    t0 = time.time()
    for epoch in range(ELLA_EPOCHS):
        dl = make_streaming_dataloader(phase=20, epoch=epoch, max_samples=MAX_SAMPLES_ELLA, batch_size=TRAIN_BATCH_SIZE)
        progress = tqdm(dl, desc=f"ELLA {epoch+1}/{ELLA_EPOCHS}")
        for batch in progress:
            captions = _as_prompt_list(batch["caption"])
            img = batch["image"].to(device=device, dtype=unet_dtype)
            with torch.no_grad():
                latent = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latent)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (latent.shape[0],), device=device).long()
                noisy = scheduler.add_noise(latent, noise, t)
                gh, gm = encode_gemma_prompts(captions)
            loss_teacher = noise.new_tensor(0.0)
            loss_delta = noise.new_tensor(0.0)
            loss_anchor = noise.new_tensor(0.0)
            if use_clip_teacher_delta_phase1:
                empty = [""] * len(captions)
                with torch.no_grad():
                    ugh, ugm = encode_gemma_prompts(empty)
                noisy_pair = torch.cat([noisy, noisy], dim=0)
                t_pair = torch.cat([t, t], dim=0)
                g_pair = torch.cat([gh, ugh], dim=0)
                m_pair = torch.cat([gm, ugm], dim=0)
                ctx = ella_connector(g_pair.to(dtype=unet_dtype), t_pair, m_pair, context_tokens=CONTEXT_TOKENS)
                student_pair = unet(noisy_pair, t_pair, encoder_hidden_states=ctx).sample
                student_cond, student_uncond = student_pair.chunk(2)
                loss_diff = F.mse_loss(student_cond.float(), noise.float())
                with torch.no_grad():
                    ch, cm = encode_clip_prompts(captions)
                    uch, ucm = encode_clip_prompts(empty)
                    clip_pair = torch.cat([ch, uch], dim=0)
                    clip_m_pair = torch.cat([cm, ucm], dim=0)
                    teacher_pair = unet(noisy_pair, t_pair, encoder_hidden_states=clip_pair.to(dtype=unet_dtype), encoder_attention_mask=clip_m_pair).sample.detach()
                    teacher_cond, teacher_uncond = teacher_pair.chunk(2)
                    teacher_delta = teacher_cond - teacher_uncond
                student_delta = student_cond - student_uncond
                loss_teacher = F.mse_loss(student_cond.float(), teacher_cond.float())
                loss_delta = F.mse_loss(student_delta.float(), teacher_delta.float())
            else:
                ctx = ella_connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=CONTEXT_TOKENS)
                student_cond = unet(noisy, t, encoder_hidden_states=ctx).sample
                loss_diff = F.mse_loss(student_cond.float(), noise.float())
            if PHASE1_SEMANTIC_ANCHOR_WEIGHT > 0 and clip_model is not None:
                with torch.no_grad():
                    ch, cm = encode_clip_prompts(captions)
                pred77 = ctx[:len(captions), :CLIP_ANCHOR_TOKENS, :]
                loss_anchor = clip_geometry_loss(pred77, ch, cm)["total"]
            loss = LAMBDA_DIFFUSION * loss_diff + LAMBDA_TEACHER * loss_teacher + LAMBDA_TEXT_DELTA * loss_delta + PHASE1_SEMANTIC_ANCHOR_WEIGHT * loss_anchor
            if not torch.isfinite(loss):
                raise RuntimeError("ELLA loss NaN/Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_stats = connector_extra_grad_stats(ella_connector)
            nn.utils.clip_grad_norm_(ella_connector.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            opt_step += 1
            best = float(loss.item()) if best is None else min(best, float(loss.item()))
            if opt_step % 25 == 0:
                wandb.log({
                    "ella/step": opt_step,
                    "ella/loss": loss.item(),
                    "ella/loss_diff": loss_diff.item(),
                    "ella/loss_teacher": float(loss_teacher.item()),
                    "ella/loss_delta": float(loss_delta.item()),
                    "ella/loss_anchor": float(loss_anchor.item()),
                    "ella/extra_gate_grad_norm": grad_stats["extra_gate_grad_norm"] if grad_stats["extra_gate_grad_norm"] is not None else 0.0,
                    "ella/extra_query_grad_norm": grad_stats["extra_query_grad_norm"] if grad_stats["extra_query_grad_norm"] is not None else 0.0,
                    "ella/extra_pos_grad_norm": grad_stats["extra_pos_grad_norm"] if grad_stats["extra_pos_grad_norm"] is not None else 0.0,
                })
            if VALIDATION_EVERY_OPT_STEPS and opt_step % VALIDATION_EVERY_OPT_STEPS == 0:
                print(f"ella step {opt_step}: loss={loss.item():.5f} diff={loss_diff.item():.5f} teacher={loss_teacher.item():.5f} delta={loss_delta.item():.5f} anchor={loss_anchor.item():.5f}")
                fixed_overfit_loss(label=f"ella_step_{opt_step:06d}")
                teacher_student_delta_alignment(VAL_PROMPTS[0], label=f"ella_step_{opt_step:06d}")
                if RUN_LONG_CONTEXT_DIAGNOSTICS and CONTEXT_TOKENS > CLIP_ANCHOR_TOKENS:
                    extra_token_ablation_metrics(LONG_CONTEXT_DIAGNOSTIC_PROMPTS[0], label=f"ella_step_{opt_step:06d}")
                    suffix_counterfactual_metrics(SUFFIX_COUNTERFACTUAL_CASES[0], label=f"ella_step_{opt_step:06d}_suffix")
            progress.set_postfix({"loss": f"{loss.item():.4f}", "best": f"{best:.4f}"})
            if ELLA_MAX_OPT_STEPS is not None and opt_step >= ELLA_MAX_OPT_STEPS:
                stop = True
                print("Stopping ELLA at step cap", opt_step)
                break
        if stop:
            break
    log_vram("ella_end", opt_step)
    print(f"ELLA frozen-UNet done: steps={opt_step} best={best} elapsed={(time.time()-t0)/60:.1f}m")
    torch.save({
        "connector_state_dict": {k: v.detach().cpu() for k, v in ella_connector.state_dict().items()},
        "config": RUN_CONFIG,
        "stage": "ella_frozen_unet",
    }, f"{DRIVE_OUT}/ella_connector_frozen_unet.pt")
''')

nb['cells'][26]['source'] = src(r'''
# @title 8.2 ELLA + sparse SaRA attn2 K/V training
if not RUN_TRAINING or not RUN_SARA_PHASE or SARA_EPOCHS <= 0:
    print("Sparse SaRA phase skipped")
else:
    for p in ella_connector.parameters(): p.requires_grad_(True)
    trainable_connector_params = [p for p in ella_connector.parameters() if p.requires_grad]
    trainable_unet_params = [p for p in unet.parameters() if p.requires_grad]
    trainable_params = trainable_connector_params + trainable_unet_params
    print(f"Trainable params in ELLA+SaRA: {sum(p.numel() for p in trainable_params):,}")
    print(f"  connector params: {sum(p.numel() for p in trainable_connector_params):,}")
    print(f"  sparse UNet params: {sum(p.numel() for p in trainable_unet_params):,}")
    optimizer = torch.optim.AdamW([
        {"params": trainable_connector_params, "weight_decay": 0.01},
        {"params": trainable_unet_params, "weight_decay": 0.0},
    ], lr=SARA_LR, eps=1e-6)
    use_clip_teacher_delta_phase2 = bool(USE_CLIP_TEACHER_DELTA and USE_CLIP_TEACHER_DELTA_PHASE2 and clip_model is not None)
    if USE_CLIP_TEACHER_DELTA and not use_clip_teacher_delta_phase2:
        print("Phase 2 CLIP teacher delta disabled to avoid moving-teacher distillation.")
    print("Phase 2 forward mode:", "paired_cfg_teacher" if use_clip_teacher_delta_phase2 else "conditional_only")
    ella_connector.train(); unet.train(); vae.eval(); gemma_model.eval()
    if clip_model is not None: clip_model.eval()
    plan = print_stage_plan("ella_sara_attn2_kv", MAX_SAMPLES_ELLA, TRAIN_BATCH_SIZE, SARA_EPOCHS, SARA_MAX_OPT_STEPS)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    log_vram("sara_start", 0)
    opt_step = 0
    best = None
    stop = False
    t0 = time.time()
    for epoch in range(SARA_EPOCHS):
        dl = make_streaming_dataloader(phase=30, epoch=epoch, max_samples=MAX_SAMPLES_ELLA, batch_size=TRAIN_BATCH_SIZE)
        progress = tqdm(dl, desc=f"ELLA+SaRA {epoch+1}/{SARA_EPOCHS}")
        for batch in progress:
            captions = _as_prompt_list(batch["caption"])
            img = batch["image"].to(device=device, dtype=unet_dtype)
            with torch.no_grad():
                latent = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latent)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (latent.shape[0],), device=device).long()
                noisy = scheduler.add_noise(latent, noise, t)
                gh, gm = encode_gemma_prompts(captions)
            loss_teacher = noise.new_tensor(0.0)
            loss_delta = noise.new_tensor(0.0)
            loss_anchor = noise.new_tensor(0.0)
            if use_clip_teacher_delta_phase2:
                empty = [""] * len(captions)
                with torch.no_grad():
                    ugh, ugm = encode_gemma_prompts(empty)
                noisy_pair = torch.cat([noisy, noisy], dim=0)
                t_pair = torch.cat([t, t], dim=0)
                g_pair = torch.cat([gh, ugh], dim=0)
                m_pair = torch.cat([gm, ugm], dim=0)
                ctx = ella_connector(g_pair.to(dtype=unet_dtype), t_pair, m_pair, context_tokens=CONTEXT_TOKENS)
                student_pair = unet(noisy_pair, t_pair, encoder_hidden_states=ctx).sample
                student_cond, student_uncond = student_pair.chunk(2)
                loss_diff = F.mse_loss(student_cond.float(), noise.float())
                with torch.no_grad():
                    ch, cm = encode_clip_prompts(captions)
                    uch, ucm = encode_clip_prompts(empty)
                    clip_pair = torch.cat([ch, uch], dim=0)
                    clip_m_pair = torch.cat([cm, ucm], dim=0)
                    teacher_pair = unet(noisy_pair, t_pair, encoder_hidden_states=clip_pair.to(dtype=unet_dtype), encoder_attention_mask=clip_m_pair).sample.detach()
                    teacher_cond, teacher_uncond = teacher_pair.chunk(2)
                    teacher_delta = teacher_cond - teacher_uncond
                student_delta = student_cond - student_uncond
                loss_teacher = F.mse_loss(student_cond.float(), teacher_cond.float())
                loss_delta = F.mse_loss(student_delta.float(), teacher_delta.float())
            else:
                ctx = ella_connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=CONTEXT_TOKENS)
                student_cond = unet(noisy, t, encoder_hidden_states=ctx).sample
                loss_diff = F.mse_loss(student_cond.float(), noise.float())
            if PHASE2_SEMANTIC_ANCHOR_WEIGHT > 0 and clip_model is not None:
                with torch.no_grad():
                    ch, cm = encode_clip_prompts(captions)
                pred77 = ctx[:len(captions), :CLIP_ANCHOR_TOKENS, :]
                loss_anchor = clip_geometry_loss(pred77, ch, cm)["total"]
            loss = LAMBDA_DIFFUSION * loss_diff + LAMBDA_TEACHER * loss_teacher + LAMBDA_TEXT_DELTA * loss_delta + PHASE2_SEMANTIC_ANCHOR_WEIGHT * loss_anchor
            if not torch.isfinite(loss):
                raise RuntimeError("ELLA+SaRA loss NaN/Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_stats = connector_extra_grad_stats(ella_connector)
            nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
            optimizer.step()
            opt_step += 1
            best = float(loss.item()) if best is None else min(best, float(loss.item()))
            if opt_step % 25 == 0:
                wandb.log({
                    "sara/step": opt_step,
                    "sara/loss": loss.item(),
                    "sara/loss_diff": loss_diff.item(),
                    "sara/loss_teacher": float(loss_teacher.item()),
                    "sara/loss_delta": float(loss_delta.item()),
                    "sara/loss_anchor": float(loss_anchor.item()),
                    "sara/extra_gate_grad_norm": grad_stats["extra_gate_grad_norm"] if grad_stats["extra_gate_grad_norm"] is not None else 0.0,
                    "sara/extra_query_grad_norm": grad_stats["extra_query_grad_norm"] if grad_stats["extra_query_grad_norm"] is not None else 0.0,
                    "sara/extra_pos_grad_norm": grad_stats["extra_pos_grad_norm"] if grad_stats["extra_pos_grad_norm"] is not None else 0.0,
                })
            if VALIDATION_EVERY_OPT_STEPS and opt_step % VALIDATION_EVERY_OPT_STEPS == 0:
                print(f"sara step {opt_step}: loss={loss.item():.5f} diff={loss_diff.item():.5f} teacher={loss_teacher.item():.5f} delta={loss_delta.item():.5f} anchor={loss_anchor.item():.5f}")
                fixed_overfit_loss(label=f"sara_step_{opt_step:06d}")
                teacher_student_delta_alignment(VAL_PROMPTS[0], label=f"sara_step_{opt_step:06d}")
                if RUN_LONG_CONTEXT_DIAGNOSTICS and CONTEXT_TOKENS > CLIP_ANCHOR_TOKENS:
                    extra_token_ablation_metrics(LONG_CONTEXT_DIAGNOSTIC_PROMPTS[0], label=f"sara_step_{opt_step:06d}")
                    suffix_counterfactual_metrics(SUFFIX_COUNTERFACTUAL_CASES[0], label=f"sara_step_{opt_step:06d}_suffix")
            progress.set_postfix({"loss": f"{loss.item():.4f}", "best": f"{best:.4f}"})
            if SARA_MAX_OPT_STEPS is not None and opt_step >= SARA_MAX_OPT_STEPS:
                stop = True
                print("Stopping ELLA+SaRA at step cap", opt_step)
                break
        if stop:
            break
    log_vram("sara_end", opt_step)
    print(f"ELLA+SaRA done: steps={opt_step} best={best} elapsed={(time.time()-t0)/60:.1f}m")
''')

nb['cells'][28]['source'] = src(r'''
# @title 9.1 Validation grids + extra-token evidence
print("Prompt sensitivity:")
connector_prompt_sensitivity(VAL_PROMPTS, label="final_ella")
teacher_student_delta_alignment(VAL_PROMPTS[0], label="final_ella")
fixed_overfit_loss(label="final_overfit")

if RUN_LONG_CONTEXT_DIAGNOSTICS and CONTEXT_TOKENS > CLIP_ANCHOR_TOKENS:
    run_long_context_numeric_suite(label_prefix="final_long_context")
else:
    print("Long-context numeric suite skipped at 77-token stage")

if RUN_FIXED_VALIDATION_GRIDS:
    ella_imgs = []
    labels = []
    for ptxt in VAL_PROMPTS:
        print("Generating ELLA:", ptxt)
        ella_imgs.append(generate_ella(ptxt, context_tokens=CONTEXT_TOKENS))
        labels.append(f"ELLA L{CONTEXT_TOKENS}: {ptxt[:40]}")
    save_validation_grid(ella_imgs, labels, f"{DRIVE_OUT}/validation_ella_L{CONTEXT_TOKENS}.png", f"ELLA L{CONTEXT_TOKENS}")

    if RUN_COMPLEX_PROMPT_GRIDS:
        save_complex_case_grid(COMPLEX_GENERATION_CASES, f"{DRIVE_OUT}/validation_complex_prompts_L{CONTEXT_TOKENS}.png", f"Complex natural-language prompts L{CONTEXT_TOKENS}", context_tokens=CONTEXT_TOKENS)

    if CONTEXT_TOKENS > CLIP_ANCHOR_TOKENS:
        long_imgs = []
        long_labels = []
        for short_prompt, long_prompt in zip(LONG_EVAL_SHORT_CONTROLS, LONG_EVAL_PROMPTS):
            long_imgs.append(generate_ella(short_prompt, context_tokens=CLIP_ANCHOR_TOKENS))
            long_labels.append(f"short L{CLIP_ANCHOR_TOKENS}")
            long_imgs.append(generate_ella(long_prompt, context_tokens=CONTEXT_TOKENS))
            long_labels.append(f"long L{CONTEXT_TOKENS}")
        save_validation_grid(long_imgs, long_labels, f"{DRIVE_OUT}/validation_long_context_L{CONTEXT_TOKENS}.png", f"Long-context ELLA L{CONTEXT_TOKENS}")
        if RUN_SUFFIX_COUNTERFACTUAL_GRIDS:
            save_suffix_counterfactual_grids(SUFFIX_COUNTERFACTUAL_CASES, f"{DRIVE_OUT}/validation_suffix_counterfactual_L{CONTEXT_TOKENS}", "Suffix counterfactuals", context_tokens=CONTEXT_TOKENS)
    else:
        print("Long-context validation grid skipped at 77-token stage")

    if clip_model is not None:
        clip_imgs = []
        clip_labels = []
        for ptxt in VAL_PROMPTS:
            clip_imgs.append(generate_clip_teacher(ptxt))
            clip_labels.append("CLIP teacher")
        save_validation_grid(clip_imgs, clip_labels, f"{DRIVE_OUT}/validation_clip_teacher.png", "CLIP teacher baseline")
else:
    print("Validation grids skipped")
''')

nb['cells'][30]['source'] = src(r'''
# @title 9.3 Fresh reload proof + CLIP-free generation
if RUN_FINAL_PROOF:
    print("Reloading connector strict from:", connector_final_path)
    ckpt = torch.load(connector_final_path, map_location="cpu")
    cfg = ckpt["connector_config"]
    reloaded_connector = PureELLALongConnector(**cfg).to(device=device, dtype=unet_dtype).eval()
    missing, unexpected = reloaded_connector.load_state_dict(ckpt["connector_state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    print("Connector strict reload PASS")

    print("Reloading fresh SD checkpoint")
    reloaded_unet, _, _ = load_stylejourney_components(torch_dtype=unet_dtype, target_device=device)
    reloaded_unet.eval()
    if RUN_SARA_PHASE and os.path.exists(unet_sparse_path):
        sparse_ckpt = torch.load(unet_sparse_path, map_location="cpu")
        named = dict(reloaded_unet.named_parameters())
        patched = 0
        with torch.no_grad():
            for name, pack in sparse_ckpt["sparse_values"].items():
                if name not in named:
                    raise KeyError(name)
                p = named[name]
                mask = pack["mask"].to(device=p.device)
                vals = pack["values"].to(device=p.device, dtype=p.dtype)
                p[mask] = vals
                patched += int(mask.sum().item())
        print(f"Sparse UNet patch reload PASS: patched={patched:,}")

    with torch.no_grad():
        gh, gm = encode_gemma_prompts(["a small red car", ""])
        t = torch.tensor([500, 500], device=device).long()
        ctx = reloaded_connector(gh.to(dtype=unet_dtype), t, gm, context_tokens=CONTEXT_TOKENS)
        test_latent = torch.randn(2, 4, 64, 64, device=device, dtype=unet_dtype)
        pred = reloaded_unet(test_latent, t, encoder_hidden_states=ctx).sample
        assert torch.isfinite(pred).all()
    print("Fresh reload finite forward PASS")

    live_connector = ella_connector
    live_unet = unet
    ella_connector = reloaded_connector
    unet = reloaded_unet
    del clip_model
    del clip_tokenizer
    clip_model = None
    clip_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        assert clip_model is None and clip_tokenizer is None, "CLIP must be unloaded for CLIP-free proof"
        proof_imgs = []
        proof_labels = []
        print("CLIP available during proof generation:", clip_model is not None)
        for ptxt in VAL_PROMPTS:
            proof_imgs.append(generate_ella(ptxt, context_tokens=CONTEXT_TOKENS))
            proof_labels.append(f"reloaded ELLA L{CONTEXT_TOKENS}")
        proof_grid = f"{DRIVE_OUT}/proof_reloaded_pure_ella_L{CONTEXT_TOKENS}.png"
        save_validation_grid(proof_imgs, proof_labels, proof_grid, "Reloaded pure ELLA Gemma-only proof")
        print("Final proof grid:", proof_grid)

        if RUN_LONG_CONTEXT_DIAGNOSTICS and CONTEXT_TOKENS > CLIP_ANCHOR_TOKENS:
            run_long_context_numeric_suite(label_prefix="reloaded_long_context")

        if RUN_COMPLEX_PROMPT_GRIDS:
            save_complex_case_grid(COMPLEX_GENERATION_CASES, f"{DRIVE_OUT}/proof_reloaded_complex_prompts_L{CONTEXT_TOKENS}.png", f"Reloaded complex natural-language prompts L{CONTEXT_TOKENS}", context_tokens=CONTEXT_TOKENS)

        if RUN_RELOADED_LONG_PROOF and CONTEXT_TOKENS > CLIP_ANCHOR_TOKENS:
            reloaded_long_imgs = []
            reloaded_long_labels = []
            for short_prompt, long_prompt in zip(LONG_EVAL_SHORT_CONTROLS, LONG_EVAL_PROMPTS):
                reloaded_long_imgs.append(generate_ella(short_prompt, context_tokens=CLIP_ANCHOR_TOKENS))
                reloaded_long_labels.append(f"reloaded short@L{CLIP_ANCHOR_TOKENS}")
                reloaded_long_imgs.append(generate_ella(long_prompt, context_tokens=CONTEXT_TOKENS))
                reloaded_long_labels.append(f"reloaded long@L{CONTEXT_TOKENS}")
            long_proof_grid = f"{DRIVE_OUT}/proof_reloaded_long_context_L{CONTEXT_TOKENS}.png"
            save_validation_grid(reloaded_long_imgs, reloaded_long_labels, long_proof_grid, f"Reloaded long-context ELLA L{CONTEXT_TOKENS}")
            print("Final long-context proof grid:", long_proof_grid)
            if RUN_SUFFIX_COUNTERFACTUAL_GRIDS:
                save_suffix_counterfactual_grids(SUFFIX_COUNTERFACTUAL_CASES, f"{DRIVE_OUT}/proof_reloaded_suffix_counterfactual_L{CONTEXT_TOKENS}", "Reloaded suffix counterfactuals", context_tokens=CONTEXT_TOKENS)
        else:
            print("Reloaded long-context proof skipped at 77-token stage")
    finally:
        ella_connector = live_connector
        unet = live_unet
        if 'reloaded_connector' in locals():
            del reloaded_connector
        if 'reloaded_unet' in locals():
            del reloaded_unet
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
else:
    print("Final proof skipped")
''')

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
sha_after = hashlib.sha256(nb_path.read_text(encoding='utf-8').encode()).hexdigest()
print(json.dumps({"sha_before": sha_before, "sha_after": sha_after, "backup": str(backup)}, indent=2))
