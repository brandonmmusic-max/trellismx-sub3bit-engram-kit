"""Model-neutral Dual-Trellis Full-Stack Post-Quantization Error Correction (QEC).

Combines orthogonal recovery layers on top of a frozen sub-3-bit TrellisMX
checkpoint without offloading expert weights or re-running the trellis encoder:

1. Zero-VRAM codebook recalibration: continuous per-neuron intermediate scales
   (``[L, E, 3, I]`` matching existing ``intermediate_scales_fp16`` sidecar
   slots) and optional tiny-VRAM per-expert output centroid bias
   (``[L, E, H]``).
2. Mergeable non-expert QEC (zero extra VRAM, zero extra inference latency):
   rank-r LoRA on attention projections (supporting hybrid topologies such as
   KDA + MLA via ``ModelProfile.projections_for_layer``), optional dense prefix
   MLP, shared-expert, and untied ``lm_head`` LoRA, plus FP32-promoted
   text-backbone RMSNorm scales. Once distillation finishes, LoRA deltas fold
   algebraically into the base weights and LoRA hooks detach cleanly.
3. Dual-Trellis Host-RAM Engram v2 sidecar (zero GPU expert VRAM, O(1)
   deterministic prefetch): shared depth banks (Early / Mid / Late) holding a
   raw unigram table plus zero-initialized residual multi-head prime-hashed
   bigram (half raw, half NFKC-canonicalized), trigram, and skip-gram tables,
   paired with per-layer GPU projections, per-order multi-group context gating,
   optional router-aware gate modulation, and a zero-initialized width-4 causal
   depthwise Conv1D shift register.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

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


def _distinct_primes_below(limit: int, count: int) -> list[int]:
    if count <= 0:
        return []

    def _is_prime(n: int) -> bool:
        if n < 2 or n % 2 == 0:
            return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True

    primes: list[int] = []
    cand = max(3, int(limit))
    if cand % 2 == 0:
        cand -= 1
    while len(primes) < count and cand >= 3:
        if _is_prime(cand):
            primes.append(cand)
        cand -= 2
    while len(primes) < count:
        primes.append(max(3, int(limit)))
    return primes


def build_nfkc_canon_map(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """Build deterministic NFKC + lowercase + space-collapsed token canonicalization LUT."""
    canon_to_id: dict[str, int] = {}
    mapping = torch.arange(vocab_size, dtype=torch.int64)
    for tid in range(vocab_size):
        try:
            piece = tokenizer.convert_ids_to_tokens(tid)
        except Exception:
            piece = None
        if not piece or not isinstance(piece, str):
            continue
        if (piece.startswith("<") and piece.endswith(">")) or (
            piece.startswith("[") and piece.endswith("]")
        ):
            continue
        s = piece.replace("\u0120", " ").replace("\u2581", " ")
        s = unicodedata.normalize("NFKC", s).lower().strip()
        if not s:
            continue
        cid = canon_to_id.get(s)
        if cid is None:
            canon_to_id[s] = tid
            cid = tid
        mapping[tid] = cid
    return mapping


def _quantize_fp8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D embedding table to FP8 E4M3 with per-row FP32 scales."""
    w = weight.detach().float()
    max_val = float(torch.finfo(torch.float8_e4m3fn).max)
    row_max = w.abs().amax(dim=-1).clamp_min(1e-12)
    scale = (row_max / max_val).to(torch.float32)
    q = (w / scale.unsqueeze(-1)).clamp(-max_val, max_val).to(torch.float8_e4m3fn)
    return q.cpu(), scale.cpu()


def _split_default_bank_groups(engram_layers: Sequence[int]) -> dict[str, list[int]]:
    layers = [int(x) for x in engram_layers]
    if not layers:
        return {}
    if len(layers) <= 2:
        return {"bank0": layers}
    n = len(layers)
    c1 = max(1, n // 3)
    c2 = max(c1 + 1, (2 * n) // 3)
    groups = {
        "early": layers[:c1],
        "mid": layers[c1:c2],
        "late": layers[c2:],
    }
    return {k: v for k, v in groups.items() if v}


class SharedHostNGramBank(nn.Module):
    """Shared cross-layer n-gram bank with zero-init residual bi/tri/skip tables."""

    def __init__(
        self,
        vocab_size: int,
        *,
        canon_map: torch.Tensor | None = None,
        d_uni: int = 128,
        d_bi: int = 64,
        n_bi_heads: int = 4,
        d_tri: int = 64,
        n_tri_heads: int = 2,
        d_skip: int = 64,
        n_skip_heads: int = 2,
        buckets: int = 262_139,
        table_device: str | torch.device | None = None,
        seed: int = 7000,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.sentinel = int(vocab_size)
        self.d_uni = int(d_uni)
        self.d_bi = int(d_bi)
        self.d_tri = int(d_tri)
        self.d_skip = int(d_skip)
        self.n_bi_heads = int(n_bi_heads) if d_bi > 0 else 0
        self.n_tri_heads = int(n_tri_heads) if d_tri > 0 else 0
        self.n_skip_heads = int(n_skip_heads) if d_skip > 0 else 0
        self.d_bi_tot = self.n_bi_heads * self.d_bi
        self.d_tri_tot = self.n_tri_heads * self.d_tri
        self.d_skip_tot = self.n_skip_heads * self.d_skip
        self.table_device = torch.device(table_device) if table_device is not None else None

        if canon_map is not None:
            if tuple(canon_map.shape) != (self.vocab_size,):
                raise ValueError(
                    f"canon_map must have shape ({self.vocab_size},), got {tuple(canon_map.shape)}"
                )
            self.register_buffer("canon_map", canon_map.to(torch.int64).clone())
        else:
            self.register_buffer("canon_map", None)

        total_heads = self.n_bi_heads + self.n_tri_heads + self.n_skip_heads
        primes = _distinct_primes_below(buckets, total_heads)
        self.bi_primes = primes[: self.n_bi_heads]
        self.tri_primes = primes[self.n_bi_heads : self.n_bi_heads + self.n_tri_heads]
        self.skip_primes = primes[self.n_bi_heads + self.n_tri_heads :]

        gen = torch.Generator().manual_seed(seed)
        if self.n_bi_heads > 0:
            self.register_buffer(
                "bi_m1",
                torch.randint(100_003, 9_999_991, (self.n_bi_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "bi_m2",
                torch.randint(100_003, 9_999_991, (self.n_bi_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "bi_salt",
                torch.randint(1, 9_999_991, (self.n_bi_heads,), generator=gen, dtype=torch.int64),
            )
        if self.n_tri_heads > 0:
            self.register_buffer(
                "tri_m1",
                torch.randint(100_003, 9_999_991, (self.n_tri_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "tri_m2",
                torch.randint(100_003, 9_999_991, (self.n_tri_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "tri_m3",
                torch.randint(100_003, 9_999_991, (self.n_tri_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "tri_salt",
                torch.randint(1, 9_999_991, (self.n_tri_heads,), generator=gen, dtype=torch.int64),
            )
        if self.n_skip_heads > 0:
            self.register_buffer(
                "skip_m1",
                torch.randint(100_003, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "skip_m2",
                torch.randint(100_003, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "skip_salt",
                torch.randint(1, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64),
            )

        self.uni_emb = nn.Embedding(self.vocab_size, self.d_uni)
        nn.init.normal_(self.uni_emb.weight, std=0.02)

        self.bi_embs = nn.ModuleList([nn.Embedding(p, self.d_bi) for p in self.bi_primes])
        self.tri_embs = nn.ModuleList([nn.Embedding(p, self.d_tri) for p in self.tri_primes])
        self.skip_embs = nn.ModuleList([nn.Embedding(p, self.d_skip) for p in self.skip_primes])
        for modlist in (self.bi_embs, self.tri_embs, self.skip_embs):
            for emb in modlist:
                nn.init.zeros_(emb.weight)

        if self.table_device is not None:
            self._pin_tables_to(self.table_device)

    def _pin_tables_to(self, dev: torch.device) -> None:
        self.uni_emb.to(dev)
        self.bi_embs.to(dev)
        self.tri_embs.to(dev)
        self.skip_embs.to(dev)

    def to(self, *args, **kwargs):  # type: ignore[override]
        res = super().to(*args, **kwargs)
        if self.table_device is not None:
            self._pin_tables_to(self.table_device)
        return res

    def lookup(
        self,
        token_ids: torch.Tensor,
        *,
        prev_tokens: torch.Tensor | None = None,
        out_device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Lookup per-order embeddings ``(e_uni, e_bi, e_tri, e_skip)`` safe for ``T >= 1``."""
        target_dev = out_device if out_device is not None else token_ids.device
        t_dev = self.uni_emb.weight.device
        t0 = token_ids.to(device=t_dev, dtype=torch.int64).clamp(0, self.vocab_size - 1)
        bsz, seq_len = t0.shape

        if prev_tokens is not None:
            if prev_tokens.shape != (bsz, 2):
                raise ValueError(
                    f"prev_tokens must have shape ({bsz}, 2), got {tuple(prev_tokens.shape)}"
                )
            prev = prev_tokens.to(device=t_dev, dtype=torch.int64)
            padded_raw = torch.cat((prev, t0), dim=1)
        else:
            sent = torch.full((bsz, 2), self.sentinel, device=t_dev, dtype=torch.int64)
            padded_raw = torch.cat((sent, t0), dim=1)

        t1_raw = padded_raw[:, 1 : 1 + seq_len]
        t2_raw = padded_raw[:, 0:seq_len]

        if self.canon_map is not None:
            cmap = self.canon_map.to(t_dev)

            def _map(x: torch.Tensor) -> torch.Tensor:
                valid = x < self.vocab_size
                return torch.where(valid, cmap[x.clamp(0, self.vocab_size - 1)], x)

            t0_c = _map(t0)
            t1_c = _map(t1_raw)
            t2_c = _map(t2_raw)
        else:
            t0_c, t1_c, t2_c = t0, t1_raw, t2_raw

        e_uni = self.uni_emb(t0)

        e_bi: torch.Tensor | None = None
        if self.n_bi_heads > 0:
            bi_parts = []
            for h in range(self.n_bi_heads):
                a, b = (t1_raw, t0) if (self.canon_map is None or h < self.n_bi_heads // 2) else (t1_c, t0_c)
                m1 = self.bi_m1[h].to(t_dev)
                m2 = self.bi_m2[h].to(t_dev)
                salt = self.bi_salt[h].to(t_dev)
                idx = (((a + 1) * m1) ^ ((b + 1) * m2) ^ salt).abs() % self.bi_primes[h]
                bi_parts.append(self.bi_embs[h](idx))
            e_bi = torch.cat(bi_parts, dim=-1)

        e_tri: torch.Tensor | None = None
        if self.n_tri_heads > 0:
            tri_parts = []
            for h in range(self.n_tri_heads):
                m1 = self.tri_m1[h].to(t_dev)
                m2 = self.tri_m2[h].to(t_dev)
                m3 = self.tri_m3[h].to(t_dev)
                salt = self.tri_salt[h].to(t_dev)
                idx = (
                    ((t2_c + 1) * m1)
                    ^ ((t1_c + 1) * m2)
                    ^ ((t0_c + 1) * m3)
                    ^ salt
                ).abs() % self.tri_primes[h]
                tri_parts.append(self.tri_embs[h](idx))
            e_tri = torch.cat(tri_parts, dim=-1)

        e_skip: torch.Tensor | None = None
        if self.n_skip_heads > 0:
            skip_parts = []
            for h in range(self.n_skip_heads):
                m1 = self.skip_m1[h].to(t_dev)
                m2 = self.skip_m2[h].to(t_dev)
                salt = self.skip_salt[h].to(t_dev)
                idx = (((t2_c + 1) * m1) ^ ((t0_c + 1) * m2) ^ salt).abs() % self.skip_primes[h]
                skip_parts.append(self.skip_embs[h](idx))
            e_skip = torch.cat(skip_parts, dim=-1)

        def _move(x: torch.Tensor | None) -> torch.Tensor | None:
            if x is None or x.device == target_dev:
                return x
            return x.to(device=target_dev, non_blocking=True)

        return _move(e_uni), _move(e_bi), _move(e_tri), _move(e_skip)  # type: ignore[return-value]


class DualTrellisEngramLayer(nn.Module):
    """Per-layer GPU projection + per-order multi-group gate + causal DWConv1D."""

    def __init__(
        self,
        hidden_size: int,
        bank: SharedHostNGramBank,
        *,
        gate_groups: int = 8,
        use_router_gate: bool = False,
        num_experts: int = 0,
        use_causal_conv: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        if gate_groups <= 0 or self.hidden_size % gate_groups != 0:
            gate_groups = 1
        self.gate_groups = gate_groups
        self.group_dim = self.hidden_size // gate_groups
        self.use_router_gate = bool(use_router_gate and num_experts > 0)
        self.use_causal_conv = bool(use_causal_conv)
        self.dropout = nn.Dropout(dropout)

        self.w_k_uni = nn.Linear(bank.d_uni, self.hidden_size, bias=False)
        self.w_v_uni = nn.Linear(bank.d_uni, self.hidden_size, bias=False)
        nn.init.normal_(self.w_k_uni.weight, std=0.02)
        nn.init.zeros_(self.w_v_uni.weight)

        self.w_k_bi = nn.Linear(bank.d_bi_tot, self.hidden_size, bias=False) if bank.d_bi_tot > 0 else None
        self.w_v_bi = nn.Linear(bank.d_bi_tot, self.hidden_size, bias=False) if bank.d_bi_tot > 0 else None
        self.w_k_tri = nn.Linear(bank.d_tri_tot, self.hidden_size, bias=False) if bank.d_tri_tot > 0 else None
        self.w_v_tri = nn.Linear(bank.d_tri_tot, self.hidden_size, bias=False) if bank.d_tri_tot > 0 else None
        self.w_k_skip = nn.Linear(bank.d_skip_tot, self.hidden_size, bias=False) if bank.d_skip_tot > 0 else None
        self.w_v_skip = nn.Linear(bank.d_skip_tot, self.hidden_size, bias=False) if bank.d_skip_tot > 0 else None

        for wk, wv in (
            (self.w_k_bi, self.w_v_bi),
            (self.w_k_tri, self.w_v_tri),
            (self.w_k_skip, self.w_v_skip),
        ):
            if wk is not None and wv is not None:
                nn.init.normal_(wk.weight, std=0.02)
                nn.init.orthogonal_(wv.weight, gain=0.05)

        self.h_scale = nn.Parameter(torch.ones(self.hidden_size))
        self.k_scales = nn.Parameter(torch.ones(4, self.hidden_size))
        self.gate_bias = nn.Parameter(torch.zeros(4, self.gate_groups))

        if self.use_router_gate:
            self.route_gate_w = nn.Parameter(torch.zeros(4, int(num_experts)))
        else:
            self.route_gate_w = None

        if self.use_causal_conv:
            self.conv_norm_scale = nn.Parameter(torch.ones(self.hidden_size))
            self.dw_conv = nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=4, groups=self.hidden_size, bias=False
            )
            nn.init.zeros_(self.dw_conv.weight)
        else:
            self.conv_norm_scale = None
            self.dw_conv = None

    def _gate_and_apply(
        self,
        h_norm_g: torch.Tensor,
        e: torch.Tensor | None,
        wk: nn.Linear | None,
        wv: nn.Linear | None,
        order_idx: int,
        route_bonus: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if e is None or wk is None or wv is None:
            return None
        e_f = self.dropout(e).to(device=wk.weight.device, dtype=torch.float32)
        k = F.linear(e_f, wk.weight.float())
        k_norm = (
            k
            * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + 1e-6)
            * self.k_scales[order_idx].float()
        )
        bsz, seq_len, hdim = k_norm.shape
        k_g = k_norm.view(bsz, seq_len, self.gate_groups, self.group_dim)
        score = (h_norm_g * k_g).sum(dim=-1) / math.sqrt(self.group_dim)
        bias = self.gate_bias[order_idx].float().view(1, 1, self.gate_groups)
        if route_bonus is not None:
            score = score + route_bonus[:, :, order_idx : order_idx + 1]
        alpha = torch.sigmoid(score + bias).unsqueeze(-1)
        v = F.linear(e_f, wv.weight.float()).view(bsz, seq_len, self.gate_groups, self.group_dim)
        return (alpha * v).reshape(bsz, seq_len, hdim)

    def forward(
        self,
        hidden: torch.Tensor,
        bank_out: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        route_weights_dense: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e_uni, e_bi, e_tri, e_skip = bank_out
        dev = self.h_scale.device
        h_f = hidden.to(device=dev, dtype=torch.float32)
        bsz, seq_len, hdim = h_f.shape
        h_norm = (
            h_f
            * torch.rsqrt(h_f.square().mean(dim=-1, keepdim=True) + 1e-6)
            * self.h_scale.float()
        )
        h_norm_g = h_norm.view(bsz, seq_len, self.gate_groups, self.group_dim)

        route_bonus = None
        if self.route_gate_w is not None and route_weights_dense is not None:
            rw = route_weights_dense.to(device=dev, dtype=torch.float32)
            route_bonus = torch.matmul(rw, self.route_gate_w.float().t())

        y = self._gate_and_apply(h_norm_g, e_uni, self.w_k_uni, self.w_v_uni, 0, route_bonus)
        assert y is not None
        for idx, (e, wk, wv) in enumerate(
            (
                (e_bi, self.w_k_bi, self.w_v_bi),
                (e_tri, self.w_k_tri, self.w_v_tri),
                (e_skip, self.w_k_skip, self.w_v_skip),
            ),
            start=1,
        ):
            term = self._gate_and_apply(h_norm_g, e, wk, wv, idx, route_bonus)
            if term is not None:
                y = y + term

        if self.use_causal_conv and self.dw_conv is not None and self.conv_norm_scale is not None:
            y_norm = (
                y
                * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + 1e-6)
                * self.conv_norm_scale.float()
            )
            y_pad = F.pad(y_norm.transpose(1, 2), (3, 0))
            conv_out = F.silu(
                F.conv1d(y_pad, self.dw_conv.weight.float(), groups=hdim).transpose(1, 2)
            )
            y = y + conv_out

        return y.to(device=hidden.device, dtype=hidden.dtype)


class EngramQECLayer(nn.Module):
    """Per-layer standalone context-gated multi-head n-gram residual layer (v1 layout)."""

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
        d_skip: int = 0,
        n_skip_heads: int = 0,
        skip_buckets: int = 262_139,
        gate_groups: int = 8,
        use_causal_conv: bool = True,
        zero_init_hash_tables: bool = True,
        canon_map: torch.Tensor | None = None,
        table_device: str | torch.device | None = None,
        dropout: float = 0.0,
        seed: int = 7000,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sentinel = vocab_size
        self.n_bi_heads = n_bi_heads
        self.n_tri_heads = n_tri_heads
        self.n_skip_heads = n_skip_heads if d_skip > 0 else 0
        self.use_causal_conv = bool(use_causal_conv)
        self.table_device = torch.device(table_device) if table_device is not None else None

        if gate_groups <= 0 or hidden_size % gate_groups != 0:
            gate_groups = 1
        self.gate_groups = gate_groups
        self.group_dim = hidden_size // gate_groups

        if canon_map is not None:
            if tuple(canon_map.shape) != (vocab_size,):
                raise ValueError(
                    f"canon_map must have shape ({vocab_size},), got {tuple(canon_map.shape)}"
                )
            self.register_buffer("canon_map", canon_map.to(torch.int64).clone())
        else:
            self.register_buffer("canon_map", None)

        self.bi_primes = _distinct_primes_below(bi_buckets, n_bi_heads)
        self.tri_primes = _distinct_primes_below(tri_buckets, n_tri_heads)
        self.skip_primes = _distinct_primes_below(skip_buckets, self.n_skip_heads)

        gen = torch.Generator().manual_seed(seed)
        if n_bi_heads > 0:
            self.register_buffer(
                "bi_m1", torch.randint(100_003, 9_999_991, (n_bi_heads,), generator=gen, dtype=torch.int64) | 1
            )
            self.register_buffer(
                "bi_m2", torch.randint(100_003, 9_999_991, (n_bi_heads,), generator=gen, dtype=torch.int64) | 1
            )
            self.register_buffer(
                "bi_salt", torch.randint(1, 9_999_991, (n_bi_heads,), generator=gen, dtype=torch.int64)
            )
        if n_tri_heads > 0:
            self.register_buffer(
                "tri_m1", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1
            )
            self.register_buffer(
                "tri_m2", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1
            )
            self.register_buffer(
                "tri_m3", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1
            )
            self.register_buffer(
                "tri_salt", torch.randint(1, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64)
            )
        if self.n_skip_heads > 0:
            self.register_buffer(
                "skip_m1",
                torch.randint(100_003, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "skip_m2",
                torch.randint(100_003, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64) | 1,
            )
            self.register_buffer(
                "skip_salt", torch.randint(1, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64)
            )

        self.uni_emb = nn.Embedding(vocab_size, d_uni)
        self.bi_embs = nn.ModuleList([nn.Embedding(p, d_bi) for p in self.bi_primes])
        self.tri_embs = nn.ModuleList([nn.Embedding(p, d_tri) for p in self.tri_primes])
        self.skip_embs = nn.ModuleList([nn.Embedding(p, d_skip) for p in self.skip_primes])

        nn.init.normal_(self.uni_emb.weight, std=0.02)
        for modlist in (self.bi_embs, self.tri_embs, self.skip_embs):
            for emb in modlist:
                if zero_init_hash_tables:
                    nn.init.zeros_(emb.weight)
                else:
                    nn.init.normal_(emb.weight, std=0.02)

        d_cat = d_uni + n_bi_heads * d_bi + n_tri_heads * d_tri + self.n_skip_heads * d_skip
        self.w_k = nn.Linear(d_cat, hidden_size, bias=False)
        self.w_v = nn.Linear(d_cat, hidden_size, bias=False)
        self.h_scale = nn.Parameter(torch.ones(hidden_size))
        self.k_scale = nn.Parameter(torch.ones(hidden_size))
        self.gate_bias = nn.Parameter(torch.zeros(gate_groups))
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.w_k.weight, std=0.02)
        nn.init.zeros_(self.w_v.weight)

        if self.use_causal_conv:
            self.conv_norm_scale = nn.Parameter(torch.ones(hidden_size))
            self.dw_conv = nn.Conv1d(
                hidden_size, hidden_size, kernel_size=4, groups=hidden_size, bias=False
            )
            nn.init.zeros_(self.dw_conv.weight)
        else:
            self.conv_norm_scale = None
            self.dw_conv = None

        if self.table_device is not None:
            self._pin_tables_to(self.table_device)

    def _pin_tables_to(self, dev: torch.device) -> None:
        self.uni_emb.to(dev)
        self.bi_embs.to(dev)
        self.tri_embs.to(dev)
        self.skip_embs.to(dev)

    def to(self, *args, **kwargs):  # type: ignore[override]
        res = super().to(*args, **kwargs)
        if self.table_device is not None:
            self._pin_tables_to(self.table_device)
        return res

    def lookup(
        self,
        token_ids: torch.Tensor,
        *,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute concatenated n-gram embeddings from token IDs ``[B, T]`` (safe for ``T >= 1``)."""
        out_dev = token_ids.device
        t_dev = self.uni_emb.weight.device
        t0 = token_ids.to(device=t_dev, dtype=torch.int64).clamp(0, self.vocab_size - 1)
        bsz, seq_len = t0.shape

        if prev_tokens is not None:
            if prev_tokens.shape != (bsz, 2):
                raise ValueError(
                    f"prev_tokens must have shape ({bsz}, 2), got {tuple(prev_tokens.shape)}"
                )
            prev = prev_tokens.to(device=t_dev, dtype=torch.int64)
            padded_raw = torch.cat((prev, t0), dim=1)
        else:
            sent = torch.full((bsz, 2), self.sentinel, device=t_dev, dtype=torch.int64)
            padded_raw = torch.cat((sent, t0), dim=1)

        t1_raw = padded_raw[:, 1 : 1 + seq_len]
        t2_raw = padded_raw[:, 0:seq_len]

        if self.canon_map is not None:
            cmap = self.canon_map.to(t_dev)

            def _map(x: torch.Tensor) -> torch.Tensor:
                valid = x < self.vocab_size
                return torch.where(valid, cmap[x.clamp(0, self.vocab_size - 1)], x)

            t0_c = _map(t0)
            t1_c = _map(t1_raw)
            t2_c = _map(t2_raw)
        else:
            t0_c, t1_c, t2_c = t0, t1_raw, t2_raw

        parts = [self.uni_emb(t0)]
        for h in range(self.n_bi_heads):
            a, b = (t1_raw, t0) if (self.canon_map is None or h < self.n_bi_heads // 2) else (t1_c, t0_c)
            m1 = self.bi_m1[h].to(t_dev)
            m2 = self.bi_m2[h].to(t_dev)
            salt = self.bi_salt[h].to(t_dev)
            idx = (((a + 1) * m1) ^ ((b + 1) * m2) ^ salt).abs() % self.bi_primes[h]
            parts.append(self.bi_embs[h](idx))
        for h in range(self.n_tri_heads):
            m1 = self.tri_m1[h].to(t_dev)
            m2 = self.tri_m2[h].to(t_dev)
            m3 = self.tri_m3[h].to(t_dev)
            salt = self.tri_salt[h].to(t_dev)
            idx = (
                ((t2_c + 1) * m1) ^ ((t1_c + 1) * m2) ^ ((t0_c + 1) * m3) ^ salt
            ).abs() % self.tri_primes[h]
            parts.append(self.tri_embs[h](idx))
        for h in range(self.n_skip_heads):
            m1 = self.skip_m1[h].to(t_dev)
            m2 = self.skip_m2[h].to(t_dev)
            salt = self.skip_salt[h].to(t_dev)
            idx = (((t2_c + 1) * m1) ^ ((t0_c + 1) * m2) ^ salt).abs() % self.skip_primes[h]
            parts.append(self.skip_embs[h](idx))

        out = torch.cat(parts, dim=-1)
        if out.device != out_dev:
            out = out.to(device=out_dev, non_blocking=True)
        return out

    def forward(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dev = self.w_k.weight.device
        e = self.dropout(self.lookup(token_ids, prev_tokens=prev_tokens))
        e_f = e.to(device=dev, dtype=torch.float32)
        k_f = F.linear(e_f, self.w_k.weight.float())
        v_f = F.linear(e_f, self.w_v.weight.float())

        h_f = hidden.to(device=dev, dtype=torch.float32)
        bsz, seq_len, hdim = h_f.shape
        h_norm = (
            h_f
            * torch.rsqrt(h_f.square().mean(dim=-1, keepdim=True) + 1e-6)
            * self.h_scale.float()
        )
        k_norm = (
            k_f
            * torch.rsqrt(k_f.square().mean(dim=-1, keepdim=True) + 1e-6)
            * self.k_scale.float()
        )

        h_g = h_norm.view(bsz, seq_len, self.gate_groups, self.group_dim)
        k_g = k_norm.view(bsz, seq_len, self.gate_groups, self.group_dim)
        score = (h_g * k_g).sum(dim=-1) / math.sqrt(self.group_dim)
        alpha = torch.sigmoid(
            score + self.gate_bias.float().view(1, 1, self.gate_groups)
        ).unsqueeze(-1)
        y = (alpha * v_f.view(bsz, seq_len, self.gate_groups, self.group_dim)).reshape(
            bsz, seq_len, hdim
        )

        if self.use_causal_conv and self.dw_conv is not None and self.conv_norm_scale is not None:
            y_norm = (
                y
                * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + 1e-6)
                * self.conv_norm_scale.float()
            )
            y_pad = F.pad(y_norm.transpose(1, 2), (3, 0))
            conv_out = F.silu(
                F.conv1d(y_pad, self.dw_conv.weight.float(), groups=hdim).transpose(1, 2)
            )
            y = y + conv_out

        return y.to(device=hidden.device, dtype=hidden.dtype)


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
        dev = self.a.weight.device
        x_f = x.to(device=dev, dtype=torch.float32)
        out = F.linear(F.linear(x_f, self.a.weight.float()), self.b.weight.float()) * self.scale
        return out.to(device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return (self.b.weight.float() @ self.a.weight.float()) * self.scale


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        if param.device.type != "meta":
            return param.device
    for buf in module.buffers():
        if buf.device.type != "meta":
            return buf.device
    return torch.device("cpu")


class FullStackQEC(nn.Module):
    """Model-neutral Dual-Trellis Full-Stack QEC wrapper.

    Supports single-GPU, multi-GPU pipeline (``device_map='auto'``), and
    expert-parallel (``DistributedConfig(..., enable_expert_parallel=True)``)
    frozen quantized MoE checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        profile: ModelProfile,
        *,
        engram_layers: Sequence[int] = (),
        engram_mode: str = "v1_per_layer",
        engram_bank_groups: Mapping[str, Sequence[int]] | None = None,
        engram_hook_target: str = "auto",
        lora_rank: int = 32,
        lora_alpha: float = 64.0,
        engram_kwargs: Mapping[str, object] | None = None,
        train_norms: bool = True,
        train_dense_lora: bool = False,
        train_shared_lora: bool = False,
        train_lm_head_lora: bool = False,
        train_inter_scales: bool = False,
        train_expert_bias: bool = False,
        use_router_gate: bool = False,
        ep_rank: int = 0,
        ep_world_size: int = 1,
    ) -> None:
        super().__init__()
        if engram_mode not in ("v1_per_layer", "v2_shared_banks"):
            raise ValueError(f"unsupported engram_mode {engram_mode!r}")
        if engram_hook_target not in ("auto", "block", "mlp"):
            raise ValueError(f"unsupported engram_hook_target {engram_hook_target!r}")
        profile.validate()
        self.model = model
        self.profile = profile
        self.engram_mode = engram_mode
        self.engram_hook_target = engram_hook_target
        self.engram_layers = tuple(int(x) for x in engram_layers)
        self.train_inter_scales = bool(train_inter_scales)
        self.train_expert_bias = bool(train_expert_bias)
        self.use_router_gate = bool(use_router_gate and engram_mode == "v2_shared_banks")
        self.ep_rank = int(ep_rank)
        self.ep_world_size = max(1, int(ep_world_size))
        self.cur_tokens: torch.Tensor | None = None
        self._bank_cache: dict[str, tuple[torch.Tensor, ...]] = {}
        self._route_cache: dict[int, torch.Tensor] = {}
        self._engram_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._lora_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._orig_moe_forwards: dict[int, tuple[nn.Module, Any]] = {}
        self._orig_experts_forwards: dict[int, tuple[nn.Module, Any]] = {}
        self._promoted_norm_keys: list[str] = []
        self._lora_target_param_keys: list[str] = []

        for p in model.parameters():
            p.requires_grad_(False)

        ek = dict(engram_kwargs or {})
        self.banks = nn.ModuleDict()
        self.bank_map: dict[int, str] = {}
        self.engrams = nn.ModuleDict()

        if self.engram_mode == "v2_shared_banks":
            groups_raw = (
                {str(k): [int(x) for x in v] for k, v in engram_bank_groups.items()}
                if engram_bank_groups is not None
                else _split_default_bank_groups(self.engram_layers)
            )
            if not self.engram_layers and groups_raw:
                flat = []
                for ls in groups_raw.values():
                    flat.extend(ls)
                self.engram_layers = tuple(sorted(set(flat)))
            bank_buckets = int(ek.pop("buckets", ek.pop("bi_buckets", 262_139)))
            ek.pop("tri_buckets", None)
            ek.pop("skip_buckets", None)
            ek.pop("zero_init_hash_tables", None)
            gate_groups = int(ek.pop("gate_groups", 8))
            use_causal_conv = bool(ek.pop("use_causal_conv", True))
            dropout = float(ek.pop("dropout", 0.0))
            table_dev = ek.pop("table_device", None)
            canon_map = ek.pop("canon_map", None)
            d_uni = int(ek.pop("d_uni", 128))
            d_bi = int(ek.pop("d_bi", 64))
            n_bi_heads = int(ek.pop("n_bi_heads", 4))
            d_tri = int(ek.pop("d_tri", 64))
            n_tri_heads = int(ek.pop("n_tri_heads", 2))
            d_skip = int(ek.pop("d_skip", 64))
            n_skip_heads = int(ek.pop("n_skip_heads", 2))

            for idx_b, (bname, layers_in_bank) in enumerate(groups_raw.items()):
                bank = SharedHostNGramBank(
                    vocab_size=profile.vocab_size,
                    canon_map=canon_map if isinstance(canon_map, torch.Tensor) else None,
                    d_uni=d_uni,
                    d_bi=d_bi,
                    n_bi_heads=n_bi_heads,
                    d_tri=d_tri,
                    n_tri_heads=n_tri_heads,
                    d_skip=d_skip,
                    n_skip_heads=n_skip_heads,
                    buckets=bank_buckets,
                    table_device=table_dev,
                    seed=7000 + idx_b * 100,
                )
                if table_dev is None and layers_in_bank:
                    first_block = _resolve_attr(
                        model, profile.layer_path_template.format(layer=layers_in_bank[0])
                    )
                    bank = bank.to(_module_device(first_block))
                self.banks[bname] = bank
                for l_idx in layers_in_bank:
                    self.bank_map[int(l_idx)] = bname

            for l_idx in self.engram_layers:
                bname = self.bank_map[l_idx]
                block = _resolve_attr(model, profile.layer_path_template.format(layer=l_idx))
                dev = _module_device(block)
                mod = DualTrellisEngramLayer(
                    hidden_size=profile.hidden_size,
                    bank=self.banks[bname],
                    gate_groups=gate_groups,
                    use_router_gate=self.use_router_gate,
                    num_experts=profile.num_experts,
                    use_causal_conv=use_causal_conv,
                    dropout=dropout,
                ).to(dev)
                self.engrams[str(l_idx)] = mod
        else:
            for layer in self.engram_layers:
                block = _resolve_attr(model, profile.layer_path_template.format(layer=layer))
                dev = _module_device(block)
                mod = EngramQECLayer(
                    hidden_size=profile.hidden_size,
                    vocab_size=profile.vocab_size,
                    seed=7000 + layer,
                    **ek,
                ).to(dev)
                self.engrams[str(layer)] = mod

        param_name_by_obj = {id(m): n for n, m in model.named_modules()}
        self.loras = nn.ModuleDict()
        self._lora_targets: dict[str, nn.Linear] = {}

        def _attach_lora(key: str, proj: nn.Linear) -> None:
            dev = proj.weight.device
            self.loras[key] = AttentionLoRA(
                proj.in_features, proj.out_features, rank=lora_rank, alpha=lora_alpha
            ).to(dev)
            self._lora_targets[key] = proj
            mod_path = param_name_by_obj.get(id(proj))
            if mod_path:
                self._lora_target_param_keys.append(f"{mod_path}.weight")

        if lora_rank > 0:
            for layer in range(profile.num_layers):
                block = _resolve_attr(model, profile.layer_path_template.format(layer=layer))
                attn = getattr(block, profile.attn_attr)
                for proj_name in profile.projections_for_layer(layer):
                    proj = getattr(attn, proj_name, None)
                    if isinstance(proj, nn.Linear):
                        _attach_lora(f"{layer}_attn_{proj_name}", proj)

                if train_dense_lora and layer in profile.dense_layers and hasattr(block, profile.moe_attr):
                    mlp = getattr(block, profile.moe_attr)
                    for proj_name in profile.dense_mlp_projections:
                        proj = getattr(mlp, proj_name, None)
                        if isinstance(proj, nn.Linear):
                            _attach_lora(f"{layer}_dense_{proj_name}", proj)

                if (
                    train_shared_lora
                    and profile.shared_expert_attr
                    and layer in profile.routed_layers
                    and hasattr(block, profile.moe_attr)
                ):
                    moe = getattr(block, profile.moe_attr)
                    shared = getattr(moe, profile.shared_expert_attr, None)
                    if isinstance(shared, nn.Module):
                        for proj_name in profile.shared_expert_projections:
                            proj = getattr(shared, proj_name, None)
                            if isinstance(proj, nn.Linear):
                                _attach_lora(f"{layer}_shared_{proj_name}", proj)

            if train_lm_head_lora and not profile.tie_word_embeddings:
                lm_head = getattr(model, "lm_head", None)
                if lm_head is None and hasattr(model, "get_output_embeddings"):
                    lm_head = model.get_output_embeddings()
                if isinstance(lm_head, nn.Linear):
                    _attach_lora("lm_head", lm_head)

        # Zero-VRAM intermediate codebook scales & tiny-VRAM per-expert output centroid bias
        self.inter_scales = nn.ParameterDict()
        self.expert_bias = nn.ParameterDict()
        if self.train_inter_scales or self.train_expert_bias:
            for layer in profile.routed_layers:
                block = _resolve_attr(model, profile.layer_path_template.format(layer=layer))
                moe = getattr(block, profile.moe_attr, None)
                if moe is None:
                    continue
                local_e = self._infer_local_experts(moe)
                if local_e <= 0:
                    continue
                dev = _module_device(moe)
                if self.train_inter_scales:
                    self.inter_scales[str(layer)] = nn.Parameter(
                        torch.ones(local_e, 3, profile.intermediate_size, dtype=torch.float32, device=dev)
                    )
                if self.train_expert_bias:
                    self.expert_bias[str(layer)] = nn.Parameter(
                        torch.zeros(local_e, profile.hidden_size, dtype=torch.float32, device=dev)
                    )

        if train_norms:
            backbone_prefixes = tuple(
                profile.layer_path_template.format(layer=l) + "."
                for l in range(profile.num_layers)
            )
            layer_parent = profile.layer_path_template.rsplit(".", 2)[0]
            for name, module in model.named_modules():
                low = name.lower()
                if "indexer" in low or "router" in low or "visual" in low or "vision" in low:
                    continue
                in_backbone = name.startswith(backbone_prefixes) or name in (
                    f"{layer_parent}.norm",
                    "model.norm",
                    "norm",
                )
                if not in_backbone:
                    continue
                if "norm" in low and hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                    self._promote_norm_fp32(module)
                    self._promoted_norm_keys.append(f"{name}.weight")

        self._install_hooks()

    def _infer_local_experts(self, moe: nn.Module) -> int:
        experts = getattr(moe, "experts", None)
        if experts is not None and hasattr(experts, "gate_up_proj"):
            p = experts.gate_up_proj
            local = p.to_local() if hasattr(p, "to_local") else p
            return int(local.shape[0])
        if hasattr(moe, "input_linear") and hasattr(moe.input_linear, "weight"):
            return int(moe.input_linear.weight.shape[0])
        return self.profile.num_experts // self.ep_world_size

    def _promote_norm_fp32(self, norm_mod: nn.Module) -> None:
        if norm_mod.weight.dtype != torch.float32:
            norm_mod.weight.data = norm_mod.weight.data.float()
            if getattr(norm_mod, "bias", None) is not None and isinstance(norm_mod.bias, nn.Parameter):
                norm_mod.bias.data = norm_mod.bias.data.float()
            orig_fwd = norm_mod.forward

            def _fp32_norm_forward(*args, **kwargs):
                orig_dtype = None
                for a in args:
                    if isinstance(a, torch.Tensor) and a.is_floating_point():
                        orig_dtype = a.dtype
                        break
                if orig_dtype is None:
                    for v in kwargs.values():
                        if isinstance(v, torch.Tensor) and v.is_floating_point():
                            orig_dtype = v.dtype
                            break
                if orig_dtype is None:
                    return orig_fwd(*args, **kwargs)
                new_args = tuple(
                    a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                    for a in args
                )
                new_kwargs = {
                    k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                    for k, v in kwargs.items()
                }
                return orig_fwd(*new_args, **new_kwargs).to(orig_dtype)

            norm_mod.forward = _fp32_norm_forward  # type: ignore[method-assign]
        norm_mod.weight.requires_grad_(True)

    def _install_hooks(self) -> None:
        # 1. MoE intermediate_scales, expert_bias, and router-weight capture
        need_moe_wrap = self.train_inter_scales or self.train_expert_bias or self.use_router_gate
        if need_moe_wrap:
            for layer in self.profile.routed_layers:
                block = _resolve_attr(self.model, self.profile.layer_path_template.format(layer=layer))
                moe = getattr(block, self.profile.moe_attr, None)
                if moe is None:
                    continue
                save_route = self.use_router_gate and (layer in self.engram_layers)
                has_layer_qec = (str(layer) in self.inter_scales) or (str(layer) in self.expert_bias)
                if not (save_route or has_layer_qec):
                    continue
                if hasattr(moe, "experts") and hasattr(moe.experts, "gate_up_proj") and hasattr(moe, "gate"):
                    self._wrap_3d_tensor_moe(layer, moe, save_route=save_route)
                elif hasattr(moe, "router") and hasattr(moe, "input_linear") and hasattr(moe, "output_linear"):
                    self._wrap_granite_moe(layer, moe, save_route=save_route)

        # 2. Engram residual hooks
        for layer in self.engram_layers:
            block = _resolve_attr(self.model, self.profile.layer_path_template.format(layer=layer))
            mod = self.engrams[str(layer)]
            use_mlp_target = (
                self.engram_hook_target == "mlp"
                or (
                    self.engram_hook_target == "auto"
                    and (hasattr(block, "ffn_hc") or self.engram_mode == "v2_shared_banks")
                )
            ) and hasattr(block, self.profile.moe_attr)
            target_mod = getattr(block, self.profile.moe_attr) if use_mlp_target else block

            def make_engram_hook(m: nn.Module, layer_idx: int):
                def hook(_module, _inp, out):
                    if self.cur_tokens is None:
                        raise RuntimeError("cur_tokens must be set before forward pass")
                    h = out[0] if isinstance(out, tuple) else out
                    if h.ndim != 3:
                        raise RuntimeError(
                            f"Engram hook at layer {layer_idx} expected 3D [B, T, H], got {tuple(h.shape)}; "
                            "set engram_hook_target='mlp' for multi-stream hyper-connection blocks"
                        )
                    if self.engram_mode == "v2_shared_banks":
                        bname = self.bank_map[layer_idx]
                        use_live_lookup = (
                            torch.is_grad_enabled()
                            and self.training
                            and bool(getattr(self.model, "is_gradient_checkpointing", False))
                        )
                        if use_live_lookup:
                            bank_out = self.banks[bname].lookup(self.cur_tokens, out_device=h.device)
                        else:
                            bank_out = self._bank_cache.get(bname)
                            if bank_out is None:
                                bank_out = self.banks[bname].lookup(self.cur_tokens, out_device=h.device)
                                self._bank_cache[bname] = bank_out
                        rw = self._route_cache.get(layer_idx)
                        delta = m(h, bank_out, rw)
                    else:
                        delta = m(h, self.cur_tokens)
                    if isinstance(out, tuple):
                        return (h + delta,) + out[1:]
                    return h + delta

                return hook

            self._engram_handles.append(target_mod.register_forward_hook(make_engram_hook(mod, layer)))

        # 3. Mergeable LoRA hooks
        for key, proj in self._lora_targets.items():
            adapter = self.loras[key]

            def make_lora_hook(m: AttentionLoRA):
                def hook(_module, inp, out):
                    return out + m(inp[0])

                return hook

            self._lora_handles.append(proj.register_forward_hook(make_lora_hook(adapter)))

    def _wrap_3d_tensor_moe(self, layer_idx: int, moe: nn.Module, *, save_route: bool) -> None:
        """Wrap GLM-5.3 / HF 3D-parameter MoE (``mlp.experts`` + ``mlp.gate``)."""
        experts = moe.experts
        orig_experts_fwd = experts.forward
        orig_moe_fwd = moe.forward
        self._orig_experts_forwards[layer_idx] = (experts, orig_experts_fwd)
        self._orig_moe_forwards[layer_idx] = (moe, orig_moe_fwd)

        swiglu_limit = getattr(experts, "swiglu_limit", None)
        if swiglu_limit is None and self.profile.activation == "silu-cap10":
            swiglu_limit = 10.0
        num_total_experts = self.profile.num_experts

        def experts_qec_fwd(
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            w_gu = experts.gate_up_proj
            w_dn = experts.down_proj
            w_gu_local = w_gu.to_local() if hasattr(w_gu, "to_local") else w_gu
            w_dn_local = w_dn.to_local() if hasattr(w_dn, "to_local") else w_dn
            local_e = int(w_gu_local.shape[0])
            ep_start = self.ep_rank * local_e if self.ep_world_size > 1 else 0

            sc_param = self.inter_scales[str(layer_idx)] if str(layer_idx) in self.inter_scales else None
            bias_param = self.expert_bias[str(layer_idx)] if str(layer_idx) in self.expert_bias else None

            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                local_idx = top_k_index - ep_start
                valid = (local_idx >= 0) & (local_idx < local_e)
                clamped = torch.where(valid, local_idx, torch.full_like(local_idx, local_e))
                mask = F.one_hot(clamped, num_classes=local_e + 1).permute(2, 1, 0)
                hit = torch.greater(mask[:local_e].sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

            sc_cast = sc_param.to(dtype=hidden_states.dtype) if sc_param is not None else None
            bias_cast = bias_param.to(dtype=hidden_states.dtype) if bias_param is not None else None

            for idx_row in hit:
                e_loc = int(idx_row[0].item())
                top_k_pos, token_idx = torch.where(mask[e_loc])
                gu = F.linear(hidden_states[token_idx], w_gu_local[e_loc])
                gate, up = gu.chunk(2, dim=-1)
                if sc_cast is not None:
                    gate = gate * sc_cast[e_loc, 0]
                    up = up * sc_cast[e_loc, 1]
                if swiglu_limit is not None:
                    gate = gate.clamp(min=None, max=swiglu_limit)
                    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
                act = F.silu(gate) * up
                if sc_cast is not None:
                    act = act * sc_cast[e_loc, 2]
                out = F.linear(act, w_dn_local[e_loc])
                if bias_cast is not None:
                    out = out + bias_cast[e_loc]
                out = out * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, out.to(final.dtype))
            return final

        if (str(layer_idx) in self.inter_scales) or (str(layer_idx) in self.expert_bias):
            experts.forward = experts_qec_fwd  # type: ignore[method-assign]

        if save_route:
            def _record_route(hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor) -> None:
                with torch.no_grad():
                    bsz, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
                    rw = torch.zeros(
                        (bsz * seq_len, num_total_experts),
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                    rw.scatter_add_(
                        1,
                        topk_indices.reshape(-1, topk_indices.shape[-1]).to(torch.long).clamp(0, num_total_experts - 1),
                        topk_weights.reshape(-1, topk_weights.shape[-1]).float(),
                    )
                    self._route_cache[layer_idx] = rw.reshape(bsz, seq_len, num_total_experts)

            if isinstance(moe.gate, nn.Module):
                def _gate_hook(_module, inp, out):
                    if isinstance(out, tuple) and len(out) >= 3:
                        _record_route(inp[0], out[1], out[2])

                self._engram_handles.append(moe.gate.register_forward_hook(_gate_hook))
            else:
                def moe_route_fwd(hidden_states: torch.Tensor) -> torch.Tensor:
                    residuals = hidden_states
                    orig_shape = hidden_states.shape
                    _, topk_weights, topk_indices = moe.gate(hidden_states)
                    _record_route(hidden_states, topk_weights, topk_indices)
                    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                    out = moe.experts(flat, topk_indices, topk_weights).reshape(*orig_shape)
                    if hasattr(moe, "shared_experts") and moe.shared_experts is not None:
                        out = out + moe.shared_experts(residuals)
                    return out

                moe.forward = moe_route_fwd  # type: ignore[method-assign]

    def _wrap_granite_moe(self, layer_idx: int, moe: nn.Module, *, save_route: bool) -> None:
        """Wrap Granite-style ``block_sparse_moe`` (``router`` + ``input_linear`` + ``output_linear``)."""
        orig_fwd = moe.forward
        self._orig_moe_forwards[layer_idx] = (moe, orig_fwd)
        num_e = self.profile.num_experts

        def fwd(layer_input: torch.Tensor):
            bsz, length, emb_size = layer_input.size()
            flat = layer_input.reshape(-1, emb_size)
            _, batch_index, batch_gates, expert_size, _router_logits = moe.router(flat)
            expert_inputs = flat[batch_index]
            hidden_states = moe.input_linear(expert_inputs, expert_size)
            g, u = hidden_states.chunk(2, dim=-1)

            counts = torch.tensor(expert_size, device=flat.device, dtype=torch.long)
            expert_ids = torch.repeat_interleave(
                torch.arange(num_e, device=flat.device, dtype=torch.long), counts
            )
            if str(layer_idx) in self.inter_scales:
                sc = self.inter_scales[str(layer_idx)].to(g.dtype)
                g = g * sc[expert_ids, 0]
                u = u * sc[expert_ids, 1]
                act = moe.activation(g) * u * sc[expert_ids, 2]
            else:
                act = moe.activation(g) * u

            expert_outputs = moe.output_linear(act, expert_size)
            if str(layer_idx) in self.expert_bias:
                expert_outputs = expert_outputs + self.expert_bias[str(layer_idx)].to(expert_outputs.dtype)[expert_ids]
            expert_outputs = expert_outputs * batch_gates[:, None]

            if save_route:
                with torch.no_grad():
                    rw = torch.zeros((bsz * length, num_e), device=flat.device, dtype=torch.float32)
                    rw.index_put_((batch_index, expert_ids), batch_gates.float(), accumulate=True)
                    self._route_cache[layer_idx] = rw.view(bsz, length, num_e)

            zeros = torch.zeros((bsz * length, emb_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
            return zeros.index_add(0, batch_index, expert_outputs).view(bsz, length, emb_size)

        moe.forward = fwd  # type: ignore[method-assign]

    def train(self, mode: bool = True) -> "FullStackQEC":
        """Toggle training mode on QEC modules while preserving gradient checkpointing on frozen student."""
        super().train(mode)
        if mode and bool(getattr(self.model, "is_gradient_checkpointing", False)):
            self.model.train(True)
            for mod in self.model.modules():
                if isinstance(mod, nn.Dropout):
                    mod.eval()
        else:
            self.model.eval()
        return self

    @torch.no_grad()
    def quantize_engram_tables_inplace(self, table_dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """Quantize Engram bank/v1 embedding tables in-place (BF16 or FP8 E4M3 dequantized) and return backups."""
        backups: dict[str, torch.Tensor] = {}

        def _qdq(w: torch.Tensor) -> torch.Tensor:
            if table_dtype == torch.float8_e4m3fn:
                q, sc = _quantize_fp8_per_row(w)
                return (q.to(device=w.device, dtype=torch.float32) * sc.to(device=w.device).unsqueeze(-1)).to(w.dtype)
            return w.detach().to(table_dtype).to(w.dtype)

        for bname, bank in self.banks.items():
            for name, param in bank.named_parameters():
                if name.endswith(".weight") and param.ndim == 2:
                    key = f"banks.{bname}.{name}"
                    backups[key] = param.data.detach().cpu().clone()
                    param.data.copy_(_qdq(param.data))
        for lkey, mod in self.engrams.items():
            for name, param in mod.named_parameters():
                if ("emb" in name) and name.endswith(".weight") and param.ndim == 2:
                    key = f"engrams.{lkey}.{name}"
                    backups[key] = param.data.detach().cpu().clone()
                    param.data.copy_(_qdq(param.data))
        return backups

    @torch.no_grad()
    def restore_engram_tables(self, backups: Mapping[str, torch.Tensor]) -> None:
        """Restore Engram embedding tables from a backup dict returned by quantize_engram_tables_inplace."""
        named = dict(self.named_parameters())
        for key, saved in backups.items():
            if key in named:
                named[key].data.copy_(saved.to(device=named[key].device, dtype=named[key].dtype))

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Standard forward entrypoint compatible with DistributedDataParallel and pipeline sharding."""
        self.cur_tokens = input_ids
        self._bank_cache.clear()
        self._route_cache.clear()
        return self.model(input_ids=input_ids, **kwargs)

    def remove_hooks(self) -> None:
        for handle in reversed(self._lora_handles):
            handle.remove()
        self._lora_handles.clear()
        for handle in reversed(self._engram_handles):
            handle.remove()
        self._engram_handles.clear()
        for mod, orig_fwd in self._orig_moe_forwards.values():
            mod.forward = orig_fwd  # type: ignore[method-assign]
        self._orig_moe_forwards.clear()
        for mod, orig_fwd in self._orig_experts_forwards.values():
            mod.forward = orig_fwd  # type: ignore[method-assign]
        self._orig_experts_forwards.clear()

    @torch.no_grad()
    def merge_lora_into_model(self) -> int:
        """Fold all LoRA weights into base Linear layers and detach LoRA hooks."""
        merged = 0
        for key, proj in self._lora_targets.items():
            adapter = self.loras[key]
            delta = adapter.delta_weight().to(device=proj.weight.device, dtype=proj.weight.dtype)
            proj.weight.data.add_(delta)
            adapter.b.weight.data.zero_()
            merged += 1
        for handle in reversed(self._lora_handles):
            handle.remove()
        self._lora_handles.clear()
        return merged

    def export_non_expert_state(self) -> dict[str, torch.Tensor]:
        """Export only the exact merged LoRA target weights and trained text-backbone norms."""
        keep = set(self._lora_target_param_keys) | set(self._promoted_norm_keys)
        return {
            k: v.detach().cpu()
            for k, v in self.model.state_dict().items()
            if k in keep
        }

    def export_expert_qec_state(self) -> dict[str, torch.Tensor]:
        """Export zero-VRAM intermediate_scales_fp16 and tiny-VRAM expert_bias per routed layer."""
        out: dict[str, torch.Tensor] = {}
        for layer, p in self.inter_scales.items():
            out[f"layer_{layer}.intermediate_scales_fp16"] = p.detach().to(torch.float16).cpu()
        for layer, p in self.expert_bias.items():
            out[f"layer_{layer}.expert_bias_bf16"] = p.detach().to(torch.bfloat16).cpu()
        return out

    def export_engram_sidecar(self, *, table_dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
        """Export Engram banks and per-layer projections in BF16 or FP8 E4M3."""
        banks_out: dict[str, Any] = {}
        for bname, bank in self.banks.items():
            b_state: dict[str, torch.Tensor] = {}
            for k, v in bank.state_dict().items():
                is_emb_weight = k.endswith(".weight") and v.ndim == 2
                if is_emb_weight and table_dtype == torch.float8_e4m3fn:
                    q, sc = _quantize_fp8_per_row(v)
                    b_state[k] = q
                    b_state[f"{k}.row_scale"] = sc
                elif is_emb_weight:
                    b_state[k] = v.detach().to(table_dtype).cpu()
                else:
                    b_state[k] = v.detach().cpu()
            banks_out[bname] = b_state

        engrams_out: dict[str, dict[str, torch.Tensor]] = {}
        for layer_key, mod in self.engrams.items():
            e_state: dict[str, torch.Tensor] = {}
            for k, v in mod.state_dict().items():
                is_v1_emb = ("emb" in k) and k.endswith(".weight") and v.ndim == 2
                if is_v1_emb and table_dtype == torch.float8_e4m3fn:
                    q, sc = _quantize_fp8_per_row(v)
                    e_state[k] = q
                    e_state[f"{k}.row_scale"] = sc
                elif is_v1_emb:
                    e_state[k] = v.detach().to(table_dtype).cpu()
                else:
                    e_state[k] = v.detach().to(torch.bfloat16 if v.is_floating_point() else v.dtype).cpu()
            engrams_out[layer_key] = e_state

        return {
            "mode": self.engram_mode,
            "table_dtype": str(table_dtype),
            "bank_map": dict(self.bank_map),
            "banks": banks_out,
            "engrams": engrams_out,
        }

    def parameter_groups(
        self,
        *,
        lr_emb: float = 1.5e-3,
        lr_proj: float = 4e-4,
        lr_scales: float = 6e-4,
        lr_bias: float = 3e-4,
        lr_lora: float = 2.5e-4,
        lr_norm: float = 3e-4,
        weight_decay: float = 1e-2,
    ) -> list[dict[str, object]]:
        emb, proj, scales, bias, lora, no_wd = [], [], [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ("banks." in name) or ("engrams." in name and "emb" in name):
                emb.append(param)
            elif "engrams." in name:
                if param.ndim == 1 or "bias" in name or "scale" in name or "route_gate" in name:
                    no_wd.append(param)
                else:
                    proj.append(param)
            elif "inter_scales." in name:
                scales.append(param)
            elif "expert_bias." in name:
                bias.append(param)
            elif "loras." in name:
                lora.append(param)
            else:
                no_wd.append(param)
        groups: list[dict[str, object]] = []
        if emb:
            groups.append({"params": emb, "lr": lr_emb, "weight_decay": 0.0})
        if proj:
            groups.append({"params": proj, "lr": lr_proj, "weight_decay": weight_decay})
        if scales:
            groups.append({"params": scales, "lr": lr_scales, "weight_decay": 0.0})
        if bias:
            groups.append({"params": bias, "lr": lr_bias, "weight_decay": 0.0})
        if lora:
            groups.append({"params": lora, "lr": lr_lora, "weight_decay": weight_decay})
        if no_wd:
            groups.append({"params": no_wd, "lr": lr_norm, "weight_decay": 0.0})
        return groups


def build_context_curriculum(
    total_tokens: int,
    short_tokens_per_step: int,
    long_tokens_per_step: int,
    *,
    long_fraction: float = 0.20,
    mode: str = "interleaved",
) -> list[str]:
    """Return a step-by-step sequence of 'short' and 'long' labels for an 80/20 curriculum."""
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive; got {total_tokens}")
    if not 0.0 <= long_fraction <= 1.0:
        raise ValueError(f"long_fraction must be in [0, 1]; got {long_fraction}")
    if short_tokens_per_step <= 0 or long_tokens_per_step <= 0:
        raise ValueError("tokens per step must be positive")
    if mode not in ("interleaved", "tail"):
        raise ValueError(f"unsupported curriculum mode {mode!r}")

    long_target = int(round(total_tokens * long_fraction))
    short_target = max(0, total_tokens - long_target)
    n_long = long_target // long_tokens_per_step if long_fraction > 0 else 0
    if long_fraction > 0 and n_long == 0 and total_tokens >= long_tokens_per_step:
        n_long = 1
        short_target = max(0, total_tokens - long_tokens_per_step)
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


def _topk_kl_chunk(
    s_chunk: torch.Tensor,
    idx_chunk: torch.Tensor,
    t_logp_chunk: torch.Tensor,
    t_lse_chunk: torch.Tensor | None,
    t_logits_chunk: torch.Tensor | None,
    t_ptail_chunk: torch.Tensor | None,
) -> torch.Tensor:
    idx_dev = idx_chunk.to(device=s_chunk.device, dtype=torch.long)
    s_logp = F.log_softmax(s_chunk.float(), dim=-1)
    s_top_logp = s_logp.gather(-1, idx_dev)

    use_tail = (
        t_ptail_chunk is not None
        or t_lse_chunk is not None
        or t_logits_chunk is not None
    )
    if use_tail:
        if t_logits_chunk is not None:
            if t_lse_chunk is None:
                raise ValueError("teacher_top_logits requires teacher_logsumexp")
            t_top_logp = (
                t_logits_chunk.to(device=s_chunk.device, dtype=torch.float32)
                - t_lse_chunk.to(device=s_chunk.device, dtype=torch.float32).unsqueeze(-1)
            )
        else:
            t_top_logp = t_logp_chunk.to(device=s_chunk.device, dtype=torch.float32)

        t_top_logp = t_top_logp.clamp_max(0.0)
        t_top_p = t_top_logp.exp()
        if t_ptail_chunk is not None:
            p_tail = t_ptail_chunk.to(device=s_chunk.device, dtype=torch.float32).clamp(0.0, 1.0)
        else:
            p_tail = (1.0 - t_top_p.sum(dim=-1)).clamp_min(0.0)

        mass = t_top_p.sum(dim=-1) + p_tail
        if bool((mass > 1.0).any()):
            scale = mass.clamp_min(1.0)
            t_top_p = t_top_p / scale.unsqueeze(-1)
            t_top_logp = t_top_p.clamp_min(1e-30).log()
            p_tail = p_tail / scale

        s_masked = s_logp.scatter(-1, idx_dev, float("-inf"))
        log_q_tail = torch.logsumexp(s_masked, dim=-1)
        has_tail = (p_tail > 1e-12) & torch.isfinite(log_q_tail)
        safe_log_q_tail = torch.where(has_tail, log_q_tail, torch.zeros_like(log_q_tail))

        kl_top = (t_top_p * (t_top_logp - s_top_logp)).sum(dim=-1)
        kl_tail = torch.where(
            has_tail,
            p_tail * (p_tail.clamp_min(1e-12).log() - safe_log_q_tail),
            torch.zeros_like(p_tail),
        )
        return (kl_top + kl_tail).sum()

    t_renorm = F.log_softmax(t_logp_chunk.to(device=s_chunk.device, dtype=torch.float32), dim=-1)
    t_p = t_renorm.exp()
    return (t_p * (t_renorm - s_top_logp)).sum(dim=-1).sum()


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_top_logp: torch.Tensor,
    teacher_top_idx: torch.Tensor,
    *,
    teacher_logsumexp: torch.Tensor | None = None,
    teacher_top_logits: torch.Tensor | None = None,
    teacher_p_tail: torch.Tensor | None = None,
    chunk_tokens: int = 512,
) -> torch.Tensor:
    """Chunked, checkpointed log-space exact tail-calibrated KL(teacher || student) over [B, T, K]."""
    bsz, seq_len, _vocab = student_logits.shape
    total_loss = student_logits.new_zeros((), dtype=torch.float32)
    use_ckpt = bool(student_logits.requires_grad and seq_len > chunk_tokens)

    for t0 in range(0, seq_len, chunk_tokens):
        t1 = min(seq_len, t0 + chunk_tokens)
        s_chunk = student_logits[:, t0:t1]
        idx_chunk = teacher_top_idx[:, t0:t1]
        lp_chunk = teacher_top_logp[:, t0:t1]
        lse_chunk = teacher_logsumexp[:, t0:t1] if teacher_logsumexp is not None else None
        tlog_chunk = teacher_top_logits[:, t0:t1] if teacher_top_logits is not None else None
        pt_chunk = teacher_p_tail[:, t0:t1] if teacher_p_tail is not None else None
        if use_ckpt:
            chunk_loss = _grad_checkpoint(
                _topk_kl_chunk,
                s_chunk,
                idx_chunk,
                lp_chunk,
                lse_chunk,
                tlog_chunk,
                pt_chunk,
                use_reentrant=False,
            )
        else:
            chunk_loss = _topk_kl_chunk(s_chunk, idx_chunk, lp_chunk, lse_chunk, tlog_chunk, pt_chunk)
        total_loss = total_loss + chunk_loss

    return total_loss / float(max(1, bsz * seq_len))
