"""Model-neutral Full-Stack Post-Quantization Error Correction (QEC).

Combines orthogonal recovery layers on top of a frozen sub-3-bit TrellisMX
checkpoint without offloading expert weights or re-running the trellis encoder:

1. Mergeable non-expert QEC (zero extra VRAM, zero extra inference latency):
   rank-r LoRA on attention projections (supporting hybrid topologies such as
   KDA + MLA via ModelProfile.projections_for_layer), optional dense prefix MLP
   and shared-expert LoRA, and FP32-promoted text-backbone RMSNorm scales. Once
   distillation finishes, LoRA deltas fold algebraically into the base weights
   and LoRA hooks detach cleanly.
2. Dual-Trellis Host-RAM Engram QEC sidecar (zero GPU expert VRAM, O(1)
   deterministic prefetch): context-gated unigram plus zero-initialized residual
   multi-head prime-hashed bigram, trigram, and optional skip-gram tables with
   optional NFKC canonicalization, multi-group channel gating, and width-4
   causal depthwise Conv1D shift-register memory. Tables can reside on CPU host
   RAM (table_device='cpu') to prevent GPU OOM on 300B-1T MoE runs.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

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


class EngramQECLayer(nn.Module):
    """Context-gated multi-head n-gram lookup residual layer with causal Conv1D."""

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
                raise ValueError(f"canon_map must have shape ({vocab_size},), got {tuple(canon_map.shape)}")
            self.register_buffer("canon_map", canon_map.to(torch.int64).clone())
        else:
            self.register_buffer("canon_map", None)

        self.bi_primes = _distinct_primes_below(bi_buckets, n_bi_heads)
        self.tri_primes = _distinct_primes_below(tri_buckets, n_tri_heads)
        self.skip_primes = _distinct_primes_below(skip_buckets, self.n_skip_heads)

        gen = torch.Generator().manual_seed(seed)
        if n_bi_heads > 0:
            self.register_buffer("bi_m1", torch.randint(100_003, 9_999_991, (n_bi_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("bi_m2", torch.randint(100_003, 9_999_991, (n_bi_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("bi_salt", torch.randint(1, 9_999_991, (n_bi_heads,), generator=gen, dtype=torch.int64))
        if n_tri_heads > 0:
            self.register_buffer("tri_m1", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("tri_m2", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("tri_m3", torch.randint(100_003, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("tri_salt", torch.randint(1, 9_999_991, (n_tri_heads,), generator=gen, dtype=torch.int64))
        if self.n_skip_heads > 0:
            self.register_buffer("skip_m1", torch.randint(100_003, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("skip_m2", torch.randint(100_003, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64) | 1)
            self.register_buffer("skip_salt", torch.randint(1, 9_999_991, (self.n_skip_heads,), generator=gen, dtype=torch.int64))

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
            self.dw_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, groups=hidden_size, bias=False)
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
        """Compute concatenated n-gram embeddings from token IDs [B, T] (safe for T >= 1)."""
        out_dev = token_ids.device
        t_dev = self.uni_emb.weight.device
        t0 = token_ids.to(device=t_dev, dtype=torch.int64).clamp(0, self.vocab_size - 1)
        bsz, seq_len = t0.shape

        if prev_tokens is not None:
            if prev_tokens.shape != (bsz, 2):
                raise ValueError(f"prev_tokens must have shape ({bsz}, 2), got {tuple(prev_tokens.shape)}")
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
            idx = (((t2_c + 1) * m1) ^ ((t1_c + 1) * m2) ^ ((t0_c + 1) * m3) ^ salt).abs() % self.tri_primes[h]
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
        e = self.dropout(self.lookup(token_ids, prev_tokens=prev_tokens))
        e_f = e.float()
        k_f = F.linear(e_f, self.w_k.weight.float())
        v_f = F.linear(e_f, self.w_v.weight.float())

        h_f = hidden.float()
        bsz, seq_len, hdim = h_f.shape
        h_norm = h_f * torch.rsqrt(h_f.square().mean(dim=-1, keepdim=True) + 1e-6) * self.h_scale.float()
        k_norm = k_f * torch.rsqrt(k_f.square().mean(dim=-1, keepdim=True) + 1e-6) * self.k_scale.float()

        h_g = h_norm.view(bsz, seq_len, self.gate_groups, self.group_dim)
        k_g = k_norm.view(bsz, seq_len, self.gate_groups, self.group_dim)
        score = (h_g * k_g).sum(dim=-1) / math.sqrt(self.group_dim)
        alpha = torch.sigmoid(score + self.gate_bias.float().view(1, 1, self.gate_groups)).unsqueeze(-1)
        y = (alpha * v_f.view(bsz, seq_len, self.gate_groups, self.group_dim)).reshape(bsz, seq_len, hdim)

        if self.use_causal_conv and self.dw_conv is not None and self.conv_norm_scale is not None:
            y_norm = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + 1e-6) * self.conv_norm_scale.float()
            y_pad = F.pad(y_norm.transpose(1, 2), (3, 0))
            conv_out = F.silu(
                F.conv1d(y_pad, self.dw_conv.weight.float(), groups=hdim).transpose(1, 2)
            )
            y = y + conv_out

        return y.to(hidden.dtype)


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
        return (
            F.linear(F.linear(x.float(), self.a.weight.float()), self.b.weight.float()) * self.scale
        ).to(x.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return (self.b.weight.float() @ self.a.weight.float()) * self.scale


class FullStackQEC(nn.Module):
    """Model-neutral wrapper attaching mergeable non-expert LoRA + FP32 norms + Engram QEC."""

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
        train_dense_lora: bool = False,
        train_shared_lora: bool = False,
    ) -> None:
        super().__init__()
        profile.validate()
        self.model = model
        self.profile = profile
        self.engram_layers = tuple(int(x) for x in engram_layers)
        self.cur_tokens: torch.Tensor | None = None
        self._engram_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._lora_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._promoted_norm_keys: list[str] = []
        self._lora_target_param_keys: list[str] = []

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

        param_name_by_obj = {id(m): n for n, m in model.named_modules()}
        self.loras = nn.ModuleDict()
        self._lora_targets: dict[str, nn.Linear] = {}

        if lora_rank > 0:
            for layer in range(profile.num_layers):
                block = _resolve_attr(model, profile.layer_path_template.format(layer=layer))
                attn = getattr(block, profile.attn_attr)
                for proj_name in profile.projections_for_layer(layer):
                    proj = getattr(attn, proj_name, None)
                    if not isinstance(proj, nn.Linear):
                        continue
                    key = f"{layer}_attn_{proj_name}"
                    self.loras[key] = AttentionLoRA(
                        proj.in_features, proj.out_features, rank=lora_rank, alpha=lora_alpha
                    )
                    self._lora_targets[key] = proj
                    mod_path = param_name_by_obj.get(id(proj))
                    if mod_path:
                        self._lora_target_param_keys.append(f"{mod_path}.weight")

                # Optional mergeable LoRA on dense prefix MLP layers (zero VRAM once merged)
                if train_dense_lora and layer in profile.dense_layers and hasattr(block, profile.moe_attr):
                    mlp = getattr(block, profile.moe_attr)
                    for proj_name in profile.dense_mlp_projections:
                        proj = getattr(mlp, proj_name, None)
                        if not isinstance(proj, nn.Linear):
                            continue
                        key = f"{layer}_dense_{proj_name}"
                        self.loras[key] = AttentionLoRA(
                            proj.in_features, proj.out_features, rank=lora_rank, alpha=lora_alpha
                        )
                        self._lora_targets[key] = proj
                        mod_path = param_name_by_obj.get(id(proj))
                        if mod_path:
                            self._lora_target_param_keys.append(f"{mod_path}.weight")

                # Optional mergeable LoRA on BF16/FP8 shared experts (zero VRAM once merged)
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
                            if not isinstance(proj, nn.Linear):
                                continue
                            key = f"{layer}_shared_{proj_name}"
                            self.loras[key] = AttentionLoRA(
                                proj.in_features, proj.out_features, rank=lora_rank, alpha=lora_alpha
                            )
                            self._lora_targets[key] = proj
                            mod_path = param_name_by_obj.get(id(proj))
                            if mod_path:
                                self._lora_target_param_keys.append(f"{mod_path}.weight")

        if train_norms:
            # Restrict norm training strictly to text-backbone layers and final norm,
            # promoting BF16 norm scales to FP32 so small AdamW steps never round to zero.
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

    def _promote_norm_fp32(self, norm_mod: nn.Module) -> None:
        if norm_mod.weight.dtype != torch.float32:
            norm_mod.weight.data = norm_mod.weight.data.float()
            if getattr(norm_mod, "bias", None) is not None and isinstance(norm_mod.bias, nn.Parameter):
                norm_mod.bias.data = norm_mod.bias.data.float()
            orig_fwd = norm_mod.forward

            def _fp32_norm_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                orig_dtype = hidden_states.dtype
                return orig_fwd(hidden_states.float()).to(orig_dtype)

            norm_mod.forward = _fp32_norm_forward  # type: ignore[method-assign]
        norm_mod.weight.requires_grad_(True)

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

            self._engram_handles.append(block.register_forward_hook(make_engram_hook(mod)))

        for key, proj in self._lora_targets.items():
            adapter = self.loras[key]

            def make_lora_hook(m: AttentionLoRA):
                def hook(_module, inp, out):
                    return out + m(inp[0])
                return hook

            self._lora_handles.append(proj.register_forward_hook(make_lora_hook(adapter)))

    def train(self, mode: bool = True) -> "FullStackQEC":
        """Toggle training mode on QEC modules while keeping the frozen student in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Standard forward entrypoint compatible with DistributedDataParallel."""
        self.cur_tokens = input_ids
        return self.model(input_ids=input_ids, **kwargs)

    def remove_hooks(self) -> None:
        for handle in reversed(self._lora_handles):
            handle.remove()
        self._lora_handles.clear()
        for handle in reversed(self._engram_handles):
            handle.remove()
        self._engram_handles.clear()

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

    def parameter_groups(
        self,
        *,
        lr_emb: float = 1.5e-3,
        lr_proj: float = 3e-4,
        lr_lora: float = 2.5e-4,
        lr_norm: float = 3e-4,
        weight_decay: float = 1e-2,
    ) -> list[dict[str, object]]:
        emb, proj, lora, no_wd = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "engrams." in name and "emb" in name:
                emb.append(param)
            elif "engrams." in name:
                if param.ndim == 1 or "bias" in name or "scale" in name:
                    no_wd.append(param)
                else:
                    proj.append(param)
            elif "loras." in name:
                lora.append(param)
            else:
                no_wd.append(param)
        groups: list[dict[str, object]] = []
        # Zero weight decay on sparse hash embeddings so untouched buckets are never decayed
        if emb:
            groups.append({"params": emb, "lr": lr_emb, "weight_decay": 0.0})
        if proj:
            groups.append({"params": proj, "lr": lr_proj, "weight_decay": weight_decay})
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
    """Chunked, log-space exact tail-calibrated KL(teacher || student) over [B, T, K].

    Accepts either:
    - Shifted teacher log-probabilities ``teacher_top_logp`` (with optional exact
      ``teacher_p_tail``), OR
    - Raw ``teacher_top_logits`` plus ``teacher_logsumexp`` (shape ``[B, T]``).

    Computes ``log_q_tail`` in log-space via masked ``logsumexp`` over chunks of
    ``chunk_tokens`` positions so 32K-64K contexts never OOM and tail gradients
    stay strictly bounded with zero FP32 cancellation.
    """
    bsz, seq_len, vocab = student_logits.shape
    total_loss = student_logits.new_zeros((), dtype=torch.float32)

    for t0 in range(0, seq_len, chunk_tokens):
        t1 = min(seq_len, t0 + chunk_tokens)
        s_chunk = student_logits[:, t0:t1]
        idx_chunk = teacher_top_idx[:, t0:t1].long()
        s_logp = F.log_softmax(s_chunk.float(), dim=-1)
        s_top_logp = s_logp.gather(-1, idx_chunk)

        use_tail = (
            teacher_p_tail is not None
            or teacher_logsumexp is not None
            or teacher_top_logits is not None
        )
        if use_tail:
            if teacher_top_logits is not None:
                if teacher_logsumexp is None:
                    raise ValueError("teacher_top_logits requires teacher_logsumexp")
                t_top_logp = (
                    teacher_top_logits[:, t0:t1].float()
                    - teacher_logsumexp[:, t0:t1].float().unsqueeze(-1)
                )
            else:
                t_top_logp = teacher_top_logp[:, t0:t1].float()

            # Clamp tiny FP16 rounding overshoot so top-K probs never exceed 1.0
            t_top_logp = t_top_logp.clamp_max(0.0)
            t_top_p = t_top_logp.exp()
            if teacher_p_tail is not None:
                p_tail = teacher_p_tail[:, t0:t1].float().clamp(0.0, 1.0)
            else:
                p_tail = (1.0 - t_top_p.sum(dim=-1)).clamp_min(0.0)

            mass = t_top_p.sum(dim=-1) + p_tail
            renorm = mass > 1.0
            if bool(renorm.any()):
                scale = mass.clamp_min(1.0)
                t_top_p = t_top_p / scale.unsqueeze(-1)
                t_top_logp = t_top_p.clamp_min(1e-30).log()
                p_tail = p_tail / scale

            s_masked = s_logp.scatter(-1, idx_chunk, float("-inf"))
            log_q_tail = torch.logsumexp(s_masked, dim=-1)
            has_tail = (p_tail > 1e-12) & torch.isfinite(log_q_tail)
            safe_log_q_tail = torch.where(has_tail, log_q_tail, torch.zeros_like(log_q_tail))

            kl_top = (t_top_p * (t_top_logp - s_top_logp)).sum(dim=-1)
            kl_tail = torch.where(
                has_tail,
                p_tail * (p_tail.clamp_min(1e-12).log() - safe_log_q_tail),
                torch.zeros_like(p_tail),
            )
            total_loss = total_loss + (kl_top + kl_tail).sum()
        else:
            t_renorm = F.log_softmax(teacher_top_logp[:, t0:t1].float(), dim=-1)
            t_p = t_renorm.exp()
            total_loss = total_loss + (t_p * (t_renorm - s_top_logp)).sum(dim=-1).sum()

    return total_loss / float(max(1, bsz * seq_len))
