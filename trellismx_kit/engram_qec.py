"""Model-neutral Full-Stack Post-Quantization Error Correction (QEC).

Combines two orthogonal recovery layers on top of a frozen sub-3-bit TrellisMX
checkpoint without offloading expert weights or re-running the trellis encoder:

1. Mergeable non-expert QEC: rank-r Attention LoRA on attention projections plus
   RMSNorm scale weights. After distillation, AttentionLoRA folds algebraically
   back into the BF16/FP8 attention matrices with zero extra VRAM, zero new
   serving kernels, and zero inference latency.
2. Host-RAM Engram QEC sidecar: context-gated unigram plus multi-head
   prime-hashed bigram and trigram lookup tables attached at selected layers.
   Tables reside in host DDR5 and are deterministically prefetched from input
   token IDs prior to layer execution; only compact W_k and W_v projections sit
   on the GPU. W_v initializes to exact zero so distillation starts from the
   unmodified quantized baseline.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .profile import ModelProfile


def _resolve_attr(root: nn.Module, dotted_path: str) -> nn.Module:
    obj: object = root
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]  # type: ignore[index]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, nn.Module):
        raise TypeError(f"{dotted_path!r} did not resolve to an nn.Module")
    return obj


class EngramQECLayer(nn.Module):
    """Context-gated multi-head n-gram lookup residual layer."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        d_uni: int = 128,
        d_bi: int = 64,
        n_bi_heads: int = 4,
        bi_buckets: int = 262_139,
        d_tri: int = 64,
        n_tri_heads: int = 2,
        tri_buckets: int = 262_139,
        dropout: float = 0.05,
        seed: int = 7000,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_bi_heads = n_bi_heads
        self.n_tri_heads = n_tri_heads
        self.bi_buckets = bi_buckets
        self.tri_buckets = tri_buckets

        gen = torch.Generator().manual_seed(seed)
        if n_bi_heads > 0:
            self.register_buffer("bi_m1", torch.randint(100_003, 9_999_991, (n_bi_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("bi_m2", torch.randint(100_003, 9_999_991, (n_bi_heads,), generator=gen, dtype=torch.int64) | 1)
        if n_tri_heads > 0:
            self.register_buffer("tri_m1", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("tri_m2", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("tri_m3", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1)

        self.uni_emb = nn.Embedding(vocab_size, d_uni)
        self.bi_embs = nn.ModuleList([nn.Embedding(bi_buckets, d_bi) for _ in range(n_bi_heads)])
        self.tri_embs = nn.ModuleList([nn.Embedding(tri_buckets, d_tri) for _ in range(n_tri_heads)])

        d_cat = d_uni + n_bi_heads * d_bi + n_tri_heads * d_tri
        self.w_k = nn.Linear(d_cat, hidden_size, bias=False)
        self.w_v = nn.Linear(d_cat, hidden_size, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.uni_emb.weight, std=0.02)
        for emb in self.bi_embs:
            nn.init.normal_(emb.weight, std=0.02)
        for emb in self.tri_embs:
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.w_k.weight, std=0.02)
        nn.init.zeros_(self.w_v.weight)

    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute concatenated n-gram embeddings from token IDs [B, T]."""
        t0 = token_ids.clamp(0, self.vocab_size - 1)
        parts = [self.uni_emb(t0)]
        if self.n_bi_heads > 0:
            t1 = F.pad(t0[:, :-1], (1, 0), value=0)
            for h in range(self.n_bi_heads):
                idx = ((t1 * self.bi_m1[h]) ^ (t0 * self.bi_m2[h])).abs() % self.bi_buckets
                parts.append(self.bi_embs[h](idx))
        if self.n_tri_heads > 0:
            t1 = F.pad(t0[:, :-1], (1, 0), value=0)
            t2 = F.pad(t0[:, :-2], (2, 0), value=0)
            for h in range(self.n_tri_heads):
                idx = ((t2 * self.tri_m1[h]) ^ (t1 * self.tri_m2[h]) ^ (t0 * self.tri_m3[h])).abs() % self.tri_buckets
                parts.append(self.tri_embs[h](idx))
        return torch.cat(parts, dim=-1)

    def forward(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        e = self.dropout(self.lookup(token_ids))
        k = self.w_k(e)
        v = self.w_v(e)
        h_f = hidden.float()
        k_f = k.float()
        h_norm = h_f * torch.rsqrt(h_f.square().mean(dim=-1, keepdim=True) + 1e-6)
        k_norm = k_f * torch.rsqrt(k_f.square().mean(dim=-1, keepdim=True) + 1e-6)
        score = (h_norm * k_norm).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        alpha = torch.sigmoid(score + self.gate_bias.float())
        return (alpha * v.float()).to(hidden.dtype)


class AttentionLoRA(nn.Module):
    """Zero-initialized rank-r adapter that merges algebraically into a Linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int = 32, alpha: float = 64.0) -> None:
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.b(self.a(x.float())) * self.scale).to(x.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return (self.b.weight.float() @ self.a.weight.float()) * self.scale


class FullStackQEC(nn.Module):
    """Model-neutral wrapper attaching mergeable Attention LoRA + Engram QEC."""

    def __init__(
        self,
        model: nn.Module,
        profile: ModelProfile,
        *,
        engram_layers: Sequence[int] = (),
        lora_rank: int = 32,
        lora_alpha: float = 64.0,
        engram_kwargs: Mapping[str, object] | None = None,
        train_norms: bool = True,
    ) -> None:
        super().__init__()
        profile.validate()
        self.model = model
        self.profile = profile
        self.engram_layers = tuple(int(x) for x in engram_layers)
        self.cur_tokens: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for p in model.parameters():
            p.requires_grad_(False)

        self.engrams = nn.ModuleDict()
        ek = dict(engram_kwargs or {})
        for layer in self.engram_layers:
            mod = EngramQECLayer(
                hidden_size=profile.hidden_size,
                vocab_size=profile.vocab_size,
                seed=7000 + layer,
                **ek,
            )
            self.engrams[str(layer)] = mod

        self.loras = nn.ModuleDict()
        self._lora_targets: dict[str, nn.Linear] = {}
        if lora_rank > 0:
            for layer in range(profile.num_layers):
                block = _resolve_attr(model, profile.layer_path_template.format(layer=layer))
                attn = getattr(block, profile.attn_attr)
                for proj_name in profile.attn_projections:
                    proj = getattr(attn, proj_name)
                    if not isinstance(proj, nn.Linear):
                        continue
                    key = f"{layer}_{proj_name}"
                    self.loras[key] = AttentionLoRA(
                        proj.in_features, proj.out_features, rank=lora_rank, alpha=lora_alpha
                    )
                    self._lora_targets[key] = proj

        if train_norms:
            for name, module in model.named_modules():
                low = name.lower()
                if "indexer" in low or "router" in low:
                    continue
                if "norm" in low and hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                    module.weight.requires_grad_(True)

        self._install_hooks()

    def _install_hooks(self) -> None:
        for layer in self.engram_layers:
            block = _resolve_attr(self.model, self.profile.layer_path_template.format(layer=layer))
            mod = self.engrams[str(layer)]

            def make_engram_hook(m: EngramQECLayer):
                def hook(_module, _inp, out):
                    if self.cur_tokens is None:
                        raise RuntimeError("cur_tokens must be set before forward pass")
                    if isinstance(out, tuple):
                        h = out[0]
                        return (h + m(h, self.cur_tokens),) + out[1:]
                    return out + m(out, self.cur_tokens)
                return hook

            self._handles.append(block.register_forward_hook(make_engram_hook(mod)))

        for key, proj in self._lora_targets.items():
            adapter = self.loras[key]

            def make_lora_hook(m: AttentionLoRA):
                def hook(_module, inp, out):
                    return out + m(inp[0])
                return hook

            self._handles.append(proj.register_forward_hook(make_lora_hook(adapter)))

    def remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    @torch.no_grad()
    def merge_lora_into_model(self) -> int:
        """Fold all AttentionLoRA weights into the target Linear weights and detach LoRA hooks."""
        merged = 0
        for key, proj in self._lora_targets.items():
            adapter = self.loras[key]
            delta = adapter.delta_weight().to(device=proj.weight.device, dtype=proj.weight.dtype)
            proj.weight.data.add_(delta)
            adapter.b.weight.data.zero_()
            merged += 1
        return merged

    def parameter_groups(
        self,
        *,
        lr_emb: float = 1.2e-3,
        lr_proj: float = 3e-4,
        lr_lora: float = 2.5e-4,
        lr_norm: float = 3e-4,
        weight_decay: float = 1e-2,
    ) -> list[dict[str, object]]:
        emb, proj, lora, norm = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "engrams." in name and "emb" in name:
                emb.append(param)
            elif "engrams." in name:
                proj.append(param)
            elif "loras." in name:
                lora.append(param)
            else:
                norm.append(param)
        groups: list[dict[str, object]] = []
        if emb:
            groups.append({"params": emb, "lr": lr_emb, "weight_decay": weight_decay})
        if proj:
            groups.append({"params": proj, "lr": lr_proj, "weight_decay": weight_decay})
        if lora:
            groups.append({"params": lora, "lr": lr_lora, "weight_decay": weight_decay})
        if norm:
            groups.append({"params": norm, "lr": lr_norm, "weight_decay": 0.0})
        return groups


def build_context_curriculum(
    total_tokens: int,
    short_tokens_per_step: int,
    long_tokens_per_step: int,
    *,
    long_fraction: float = 0.20,
    mode: str = "interleaved",
) -> list[str]:
    """Return a step-by-step sequence of 'short' and 'long' labels for an 80/20 curriculum.

    Default ``long_fraction=0.20`` allocates 80% of distillation tokens to short
    2K-4K chunks (high-throughput n-gram saturation) and 20% to long 16K-32K+
    sequences (calibrating Attention LoRA and Engram context gates across sparse
    long-context retrieval). ``mode='interleaved'`` spreads long steps uniformly
    across training; ``mode='tail'`` places them in the closing phase.
    """
    if not 0.0 <= long_fraction <= 1.0:
        raise ValueError(f"long_fraction must be in [0, 1]; got {long_fraction}")
    if short_tokens_per_step <= 0 or long_tokens_per_step <= 0:
        raise ValueError("tokens per step must be positive")
    if mode not in ("interleaved", "tail"):
        raise ValueError(f"unsupported curriculum mode {mode!r}")

    long_target = int(round(total_tokens * long_fraction))
    short_target = max(0, total_tokens - long_target)
    n_long = long_target // long_tokens_per_step if long_fraction > 0 else 0
    n_short = short_target // short_tokens_per_step if long_fraction < 1.0 else 0
    if n_short == 0 and n_long == 0:
        return ["long" if long_fraction >= 0.5 else "short"]
    if mode == "tail" or n_long == 0 or n_short == 0:
        return ["short"] * n_short + ["long"] * n_long

    total_steps = n_short + n_long
    schedule: list[str] = []
    long_placed = 0
    for step in range(total_steps):
        expected_long = ((step + 1) * n_long) // total_steps
        if expected_long > long_placed:
            schedule.append("long")
            long_placed += 1
        else:
            schedule.append("short")
    return schedule


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_top_logp: torch.Tensor,
    teacher_top_idx: torch.Tensor,
    *,
    teacher_logsumexp: torch.Tensor | None = None,
    teacher_top_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute mean KL(teacher || student) from top-K teacher entries [B, T, K].

    When ``teacher_logsumexp`` (shape ``[B, T]``) and ``teacher_top_logits``
    (shape ``[B, T, K]``) are supplied, uses exact full-vocabulary teacher
    probabilities on the top-K tokens plus one exact residual tail bucket
    ``p_tail = 1 - sum_k p_k`` so student probability leakage outside the top-K
    support is penalized accurately.
    """
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    s_top_logp = s_logp.gather(-1, teacher_top_idx.long())

    if teacher_logsumexp is not None:
        t_vals = (
            teacher_top_logits.float()
            if teacher_top_logits is not None
            else teacher_top_logp.float()
        )
        t_top_logp = t_vals - teacher_logsumexp.float().unsqueeze(-1)
        t_top_p = t_top_logp.exp().clamp(0.0, 1.0)
        p_sum = t_top_p.sum(dim=-1)
        p_tail = (1.0 - p_sum).clamp_min(0.0)

        s_top_p = s_top_logp.exp().clamp(0.0, 1.0)
        q_tail = (1.0 - s_top_p.sum(dim=-1)).clamp_min(1e-12)

        kl_top = (t_top_p * (t_top_logp - s_top_logp)).sum(dim=-1)
        kl_tail = torch.where(
            p_tail > 1e-12,
            p_tail * (p_tail.clamp_min(1e-12).log() - q_tail.log()),
            torch.zeros_like(p_tail),
        )
        return (kl_top + kl_tail).clamp_min(0.0).mean()

    top_p = teacher_top_logp.float().exp()
    top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (top_p * (top_p.clamp_min(1e-12).log() - s_top_logp)).sum(dim=-1).mean()
