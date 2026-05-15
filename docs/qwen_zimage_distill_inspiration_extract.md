# Extracted relevant blocks from Qwen3 Z-Image distillation notebook

## XAttnBlock
```python
class XAttnBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, key_padding_mask=None):
        q = q + self.attn(
            self.norm_q(q),
            self.norm_kv(kv),
            self.norm_kv(kv),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        q = q + self.ff(self.norm_ff(q))
        return q
```

## Adapter
```python
class Adapter(nn.Module):
    def __init__(self, s_dim, t_dim, dim=1024, heads=8, blocks=2, ff_mult=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(s_dim, dim)
        self.kv_proj = nn.Linear(s_dim, dim)
        self.blocks = nn.ModuleList([
            XAttnBlock(dim, heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(blocks)
        ])
        self.proj_out = nn.Linear(dim, t_dim)

    def forward(self, student_hs, mask):
        q = self.q_proj(student_hs)
        kv = self.kv_proj(student_hs)
        key_padding_mask = ~mask.bool()
        for block in self.blocks:
            q = block(q, kv, key_padding_mask=key_padding_mask)
        out = self.proj_out(q)
        out = out.masked_fill(~mask[..., None].bool(), 0)
        return out


# ===== Teacher / student load =====

from huggingface_hub import snapshot_download as _hf_snapshot
```

## condition
```python
def condition(batch, student_model=None, adapter_module=None, teacher_model=None):
    core = student_model.module if (student_model is not None and hasattr(student_model, 'module')) else (student_model or student)
    adapter_mod = adapter_module or getattr(core, 'custom_distill_adapter')
    teacher_mod = teacher_model or teacher

    ids = batch['input_ids'].to(DEVICE)
    mask = batch['attention_mask'].to(DEVICE)

    content_mask = batch.get('content_mask')
    if content_mask is None:
        content_mask = mask
    else:
        content_mask = content_mask.to(DEVICE)
    content_mask = (content_mask.bool() & mask.bool()).long()

    empty_rows = content_mask.sum(dim=1) == 0
    if empty_rows.any():
        content_mask[empty_rows] = mask[empty_rows]

    t_hs_cached = batch.get('teacher_hs_precomputed')
    if t_hs_cached is not None:
        t_hs = t_hs_cached.to(DEVICE)
    else:
        with torch.no_grad():
            t_out = teacher_mod(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )
            t_hs = _extract_hs(t_out, HS_TAP_INDEX).detach()

    s_out = core(
        input_ids=ids,
        attention_mask=mask,
        output_hidden_states=True,
        return_dict=True,
    )
    s_hs = _extract_hs(s_out, HS_TAP_INDEX)

    # Keep adapter input dtype aligned with adapter params (bf16/fp16), avoid matmul dtype mismatch.
    adapter_dtype = next(adapter_mod.parameters()).dtype
    if s_hs.dtype != adapter_dtype:
        s_hs = s_hs.to(dtype=adapter_dtype)

    pred = adapter_mod(s_hs, mask)
    # Keep teacher activations numerically compatible with student path.
    if t_hs.dtype != pred.dtype:
        t_hs = t_hs.to(dtype=pred.dtype)

    return pred, t_hs, mask, content_mask, s_out


# ===== Phase 3 memory-safe precompute =====

_PHASE3_PRECOMP_STATE = {'root': None, 'splits': {}}
_PHASE3_MEMSAFE_DS_CACHE = {'key': None, 'train': None, 'eval': None}
```

## GeometryLoss
```python
class GeometryLoss(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def _last_token_pool(self, states, mask):
        idx = mask.long().sum(dim=1).sub(1).clamp(min=0)
        return states[torch.arange(states.size(0), device=states.device), idx]

    def _masked_mean_pool(self, states, pool_mask):
        m = pool_mask.unsqueeze(-1).float()
        return (states.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def _last_k_mean_pool(self, states, mask, k):
        k = max(1, int(k))
        pooled = []
        for i in range(states.size(0)):
            idx = torch.nonzero(mask[i].bool(), as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                pooled.append(states[i, 0].float())
                continue
            sel = idx[-k:]
            pooled.append(states[i, sel].float().mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _pool_for_contrastive(self, states, mask, content_mask=None):
        mode = str(CONTRASTIVE_POOLING).strip().lower()
        if mode == 'last_token':
            return self._last_token_pool(states, mask).float()
        if mode == 'last_k_mean':
            return self._last_k_mean_pool(states, mask, CONTRASTIVE_LAST_K)
        if mode == 'content_masked_mean':
            pool_mask = content_mask if content_mask is not None else mask
            return self._masked_mean_pool(states, pool_mask)
        raise ValueError(f'Unknown CONTRASTIVE_POOLING: {CONTRASTIVE_POOLING}')

    def _contrastive(self, pred, target, mask, content_mask=None):
        b = pred.shape[0]
        if b < CONTRASTIVE_MIN_BATCH:
            return pred.new_tensor(0.0)

        p_pool = F.normalize(self._pool_for_contrastive(pred, mask, content_mask), dim=-1)
        t_pool = F.normalize(self._pool_for_contrastive(target, mask, content_mask), dim=-1)
        logits = p_pool @ t_pool.T / CONTRASTIVE_TEMP
        labels = torch.arange(b, device=pred.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def forward(self, pred, target, mask, content_mask=None):
        m = mask.unsqueeze(-1).float()
        n = m.sum().clamp(min=1)
        d = pred.shape[-1]

        mse = ((self.ln(pred.float()) - self.ln(target.float())) ** 2 * m).sum() / n / d

        pn = pred.float().norm(dim=-1)
        tn = target.float().norm(dim=-1)
        rel = (pn - tn) / tn.clamp(min=1e-3)
        norm_l = (rel ** 2 * mask.float()).sum() / mask.float().sum().clamp(min=1)

        cos = F.cosine_similarity(pred.float(), target.float(), dim=-1)
        cos_l = ((1 - cos) * mask.float()).sum() / mask.float().sum().clamp(min=1)

        ctr = self._contrastive(pred, target, mask, content_mask)
        ctr_w_eff = LOSS_W_CONTRASTIVE if pred.shape[0] >= CONTRASTIVE_MIN_BATCH else 0.0

        total = mse + LOSS_W_NORM * norm_l + LOSS_W_COSINE * cos_l + ctr_w_eff * ctr

        # Diagnostics (not separate loss terms):
        with torch.no_grad():
            diff = (pred.float() - target.float())
            drift_l2 = torch.sqrt((diff.pow(2) * m).sum() / n)

            pool_c_pred = self._masked_mean_pool(pred, content_mask)
            pool_c_tgt = self._masked_mean_pool(target, content_mask)
            cos_content = F.cosine_similarity(pool_c_pred, pool_c_tgt, dim=-1).mean()
            norm_gap_content = (pool_c_pred.norm(dim=-1) - pool_c_tgt.norm(dim=-1)).abs().mean()

            pool_g_pred = self._masked_mean_pool(pred, mask)
            pool_g_tgt = self._masked_mean_pool(target, mask)
            cos_global = F.cosine_similarity(pool_g_pred, pool_g_tgt, dim=-1).mean()
            norm_gap_global = (pool_g_pred.norm(dim=-1) - pool_g_tgt.norm(dim=-1)).abs().mean()

        return {
            'total': total,
            'mse': mse,
            'norm': norm_l,
            'cos': cos_l,
            'ctr': ctr,
            'ctr_w_eff': pred.new_tensor(float(ctr_w_eff)),
            'drift_l2': drift_l2,
            'cos_content': cos_content,
            'norm_gap_content': norm_gap_content,
            'cos_global': cos_global,
            'norm_gap_global': norm_gap_global,
            'tnorm': tn.mean(),
            'pnorm': pn.mean(),
        }


geometry_loss = GeometryLoss(TEACHER_DIM).to(DEVICE)


@torch.no_grad()
```

## collapse_diagnostic
```python
def collapse_diagnostic(prompts, student_model=None, adapter_module=None):
    pools = []
    for text in prompts:
        b = collate_contract([{'text': text}])
        pred, _, mask, _, _ = condition(b, student_model=student_model, adapter_module=adapter_module)
        pooled = geometry_loss._masked_mean_pool(pred, mask)
        pools.append(pooled[0])

    print(f'Collapse diagnostic ({len(prompts)} prompts):')
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            cos = F.cosine_similarity(pools[i], pools[j], dim=0).item()
            tag = '*** COLLAPSED ***' if cos > 0.95 else ('WARNING' if cos > 0.85 else 'OK')
            print(f'  [{i}] vs [{j}]: cosine={cos:.4f}  {tag}')


_lpips_fn = None
```

## debug_retrieval
```python
def debug_retrieval(batch, student_model=None, adapter_module=None):
    pred, t_hs, mask, content_mask, _ = condition(batch, student_model=student_model, adapter_module=adapter_module)

    t_pool = geometry_loss._pool_for_contrastive(t_hs, mask, content_mask)
    p_pool = geometry_loss._pool_for_contrastive(pred, mask, content_mask)

    stats = _pairwise_pool_stats(t_pool)
    print(
        'teacher pairwise stats '
        f"(pool={CONTRASTIVE_POOLING}): "
        f"diag={stats['diag_mean']:.4f} offdiag={stats['offdiag_mean']:.4f} "
        f"best_offdiag={stats['best_offdiag_mean']:.4f} "
        f"margin={stats['diag_margin_mean']:.4f} rank1={stats['diag_rank1']:.4f}"
    )

    t_pool_n = F.normalize(t_pool.float(), dim=-1)
    p_pool_n = F.normalize(p_pool.float(), dim=-1)
    logits = p_pool_n @ t_pool_n.T / CONTRASTIVE_TEMP
    labels = torch.arange(logits.shape[0], device=logits.device)
    ce = F.cross_entropy(logits, labels).item()
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    print(f'adapter→teacher retrieval: CE={ce:.4f} acc={acc:.4f}')


@torch.no_grad()
```

## TeacherTargetedFMLoss
```python
class TeacherTargetedFMLoss(nn.Module):
    """Teacher-targeted flow distillation on frozen diffusion backbone outputs."""

    def __init__(self, diffusion_model: nn.Module, scheduler=None, kind: str = 'transformer'):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.kind = str(kind)
        cfg = getattr(diffusion_model, 'config', None)
        self.num_train_timesteps = int(getattr(getattr(scheduler, 'config', None), 'num_train_timesteps', 1000))
        self.prediction_type = str(getattr(getattr(scheduler, 'config', None), 'prediction_type', 'unknown'))
        self.in_channels = int(getattr(cfg, 'in_channels', getattr(diffusion_model, 'in_channels', 16)))

        self.forward_variant = None
        self._forward_variant_logged = False

        self.sampler_variant = 'unknown'
        self._sampler_variant_logged = False
        self._sample_from_trajectory = False
        self._error_streak = 0
        self._fm_disabled = False
        self._max_consec_errors = max(1, int(globals().get('PHASE3_FM_MAX_CONSEC_ERRORS', 3)))

    def _latent_hw(self):
        cfg = getattr(self.diffusion_model, 'config', None)
        ss = getattr(cfg, 'sample_size', None)
        if int(PHASE3_FM_LATENT_SIZE) > 0:
            h = w = int(PHASE3_FM_LATENT_SIZE)
        elif isinstance(ss, (tuple, list)) and len(ss) >= 2:
            h, w = int(ss[0]), int(ss[1])
        elif isinstance(ss, int):
            h = w = int(ss)
        else:
            h = w = 64
        return max(8, h), max(8, w)

    def _zero_fm_payload(self, ref_tensor, t, uses_content_mask, failed=False):
        z = ref_tensor.new_tensor(0.0)
        return {
            'fm_total': z,
            'fm_mse': z,
            'fm_cos': z,
            'fm_timestep_mean': t.float().mean() if torch.is_tensor(t) else z,
            'fm_pool_uses_content_mask': ref_tensor.new_tensor(1.0 if uses_content_mask else 0.0),
            'fm_uses_teacher_trajectory': ref_tensor.new_tensor(1.0 if self._sample_from_trajectory else 0.0),
            'fm_failed': ref_tensor.new_tensor(1.0 if failed else 0.0),
            'fm_error_streak': ref_tensor.new_tensor(float(self._error_streak)),
            'fm_disabled': ref_tensor.new_tensor(1.0 if self._fm_disabled else 0.0),
        }

    @staticmethod
    def _mask_pool(states, mask):
        if mask is None:
            return states.float().mean(dim=1)
        m = mask.unsqueeze(-1).float()
        return (states.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


    @staticmethod
    def _extract_model_tensor(out):
        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                raise RuntimeError('Empty diffusion output sequence')
            first = out[0]
            # Z-Image transformer often returns ([tensor, tensor, ...], ...)
            if isinstance(first, (list, tuple)) and first and all(torch.is_tensor(v) for v in first):
                out = torch.stack(list(first), dim=0)
            elif torch.is_tensor(first):
                out = first
            elif all(torch.is_tensor(v) for v in out):
                out = torch.stack(list(out), dim=0)

        if torch.is_tensor(out):
            return out

        for key in ('sample', 'pred', 'prediction', 'noise_pred'):
            if hasattr(out, key):
                v = getattr(out, key)
                if torch.is_tensor(v):
                    return v
                if isinstance(v, (list, tuple)) and v and all(torch.is_tensor(t) for t in v):
                    return torch.stack(list(v), dim=0)

        raise RuntimeError(f'Unable to extract tensor from diffusion output type={type(out)}')

    @staticmethod
    def _extract_prev_sample(step_out):
        if torch.is_tensor(step_out):
            return step_out
        if isinstance(step_out, (tuple, list)) and step_out:
            first = step_out[0]
            if torch.is_tensor(first):
                return first
        for key in ('prev_sample', 'sample'):
            if hasattr(step_out, key):
                v = getattr(step_out, key)
                if torch.is_tensor(v):
                    return v
        return None

    @staticmethod
    def _prepare_zimage_x(x_t):
        # Z-Image transformer expects x as a list of per-sample tensors [C, F, H, W].
        if isinstance(x_t, (list, tuple)):
            x_list = list(x_t)
        elif torch.is_tensor(x_t):
            if x_t.ndim == 4:
                x_t = x_t.unsqueeze(2)  # [B, C, 1, H, W]
            elif x_t.ndim != 5:
                raise RuntimeError(f'Unsupported x_t rank for Z-Image transformer: {tuple(x_t.shape)}')
            x_list = list(x_t.unbind(dim=0))
        else:
            raise RuntimeError(f'Unsupported x_t type for Z-Image transformer: {type(x_t)}')

        if not x_list or not torch.is_tensor(x_list[0]):
            raise RuntimeError('Prepared Z-Image x input is empty or non-tensor')
        return x_list

    @staticmethod
    def _to_cap_feats_list(cond, token_mask=None):
        # Z-Image expects cap_feats as List[Tensor(seq_i, dim)] (each entry rank-2).
        if isinstance(cond, (list, tuple)):
            feats = []
            for i, v in enumerate(cond):
                if not torch.is_tensor(v):
                    raise RuntimeError(f'cap_feats[{i}] is not a tensor: {type(v)}')
                if v.ndim == 3 and v.shape[0] == 1:
                    v = v.squeeze(0)
                if v.ndim == 1:
                    v = v.unsqueeze(0)
                if v.ndim != 2:
                    raise RuntimeError(f'cap_feats[{i}] must be rank-2, got shape={tuple(v.shape)}')
                feats.append(v)
            return feats

        if not torch.is_tensor(cond):
            raise RuntimeError(f'Unsupported cap_feats type: {type(cond)}')

        if cond.ndim == 2:
            return [cond]
        if cond.ndim != 3:
            raise RuntimeError(f'Unsupported cap_feats rank: {tuple(cond.shape)}')

        bsz, seqlen, _ = cond.shape
        if token_mask is None:
            token_mask = torch.ones((bsz, seqlen), device=cond.device, dtype=torch.bool)
        else:
            token_mask = token_mask.to(device=cond.device)
            if token_mask.ndim == 1:
                token_mask = token_mask.unsqueeze(0).expand(bsz, -1)
            if token_mask.ndim != 2 or token_mask.shape[0] != bsz or token_mask.shape[1] != seqlen:
                token_mask = torch.ones((bsz, seqlen), device=cond.device, dtype=torch.bool)
            else:
                token_mask = token_mask.bool()

        feats = []
        for i in range(bsz):
            m = token_mask[i]
            if int(m.sum().item()) == 0:
                m = torch.ones_like(m, dtype=torch.bool)
            v = cond[i][m]
            if v.ndim != 2:
                raise RuntimeError(f'masked cap_feats[{i}] must be rank-2, got shape={tuple(v.shape)}')
            feats.append(v)
        return feats

    @staticmethod
    def _postprocess_zimage_pred(tensor):
        # Match pipeline convention: stack(list_out) -> squeeze frame dim -> negate.
        if tensor.ndim == 5 and tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)
        return -tensor

    def _forward_backbone(self, x_t, t, cond, mask, content_mask=None):
        try:
            sig = inspect.signature(self.diffusion_model.forward)
            sig_params = sig.parameters
            sig_keys = list(sig_params.keys())
        except Exception:
            sig_params = {}
            sig_keys = []

        # Dedicated fast path for Z-Image transformer contract: forward(x, t, cap_feats, ...)
        if {'x', 't', 'cap_feats'}.issubset(set(sig_keys)):
            errors = []
            pool_name = 'content_mask' if (PHASE3_USE_CONTENT_MASK_FOR_POOL and content_mask is not None) else 'attention_mask'

            x_list = self._prepare_zimage_x(x_t)
            t_float = t.float()
            cond_dtype = (
                cond.dtype if torch.is_tensor(cond)
                else (cond[0].dtype if isinstance(cond, (list, tuple)) and len(cond) > 0 and torch.is_tensor(cond[0]) else t_float.dtype)
            )
            t_norm = (1000.0 - t_float) / 1000.0 if float(t_float.max().item()) > 1.5 else t_float
            t_variants = (
                ('norm', t_norm.to(dtype=cond_dtype)),
                ('float', t_float.to(dtype=cond_dtype)),
                ('long', t.long()),
            )

            # Strict Z-Image contract: cap_feats must be list of rank-2 tensors.
            if torch.is_tensor(cond) and cond.ndim == 3:
                use_content = bool(PHASE3_USE_CONTENT_MASK_FOR_POOL and content_mask is not None)
                token_mask = content_mask if use_content else mask
                cond_variants = [('list_masked', self._to_cap_feats_list(cond, token_mask))]
            elif torch.is_tensor(cond) and cond.ndim == 2:
                cond_variants = [('list_from_2d', self._to_cap_feats_list(cond, None))]
            elif isinstance(cond, (list, tuple)):
                cond_variants = [('list_as_is', self._to_cap_feats_list(cond, None))]
            else:
                raise RuntimeError(f'Unsupported cap_feats input for Z-Image fast path: type={type(cond)}')

            for cond_name, cond_val in cond_variants:
                for t_name, t_val in t_variants:
                    for call_style in ('keyword', 'positional'):
                        try:
                            if call_style == 'keyword':
                                trial = {'x': x_list, 't': t_val, 'cap_feats': cond_val}
                                if 'return_dict' in sig_params:
                                    trial['return_dict'] = False
                                out = self.diffusion_model(**trial)
                                used_kwargs = sorted(trial.keys())
                            else:
                                out = self.diffusion_model(x_list, t_val, cond_val)
                                used_kwargs = ['<positional:x,t,cap_feats>']

                            tensor = self._extract_model_tensor(out)
                            tensor = self._postprocess_zimage_pred(tensor)

                            if not self._forward_variant_logged:
                                self.forward_variant = {
                                    'sample_arg': 'x(list)',
                                    'cond_arg': f'cap_feats[{cond_name}]',
                                    'timestep_arg': 't',
                                    't_variant': t_name,
                                    't_dtype': str(t_val.dtype),
                                    'pool_mask': pool_name,
                                    'kwargs': used_kwargs,
                                }
                                print(
                                    'Phase 3 FM forward variant: '
                                    f"sample_arg={self.forward_variant['sample_arg']} cond_arg={self.forward_variant['cond_arg']} "
                                    f"timestep_arg=t t_variant={t_name} dtype={t_val.dtype} "
                                    f"pool_mask={pool_name} kwargs={used_kwargs}"
                                )
                                self._forward_variant_logged = True
                            return tensor
                        except Exception as exc:
                            errors.append(
                                f'call={call_style}/cond={cond_name}/t={t_name}/{t_val.dtype}: '
                                f'{type(exc).__name__}: {exc}'
                            )

            err_preview = '; '.join(errors[-4:]) if errors else 'no attempts'
            x0_shape = tuple(x_list[0].shape) if (isinstance(x_list, list) and len(x_list) > 0 and torch.is_tensor(x_list[0])) else None
            c0 = cond_variants[0][1][0] if (cond_variants and isinstance(cond_variants[0][1], list) and cond_vari
# ... truncated ...

```
