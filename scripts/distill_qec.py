#!/usr/bin/env python3
"""Model-neutral Dual-Trellis QEC v2 distillation CLI with 80/20 context curriculum.

Supports:
- Dual-Trellis QEC v2 (3 shared depth banks, zero-init residual bi/tri/skip hash
  tables, NFKC canonicalization, per-order G=8 gating, router-aware gate scalar,
  width-4 causal DWConv1D, zero-VRAM ``intermediate_scales_fp16``, tiny-VRAM
  per-expert centroid bias, mergeable rank-32 LoRA on hybrid attention + dense
  prefix + shared experts + untied ``lm_head``, and FP32 text-backbone RMSNorms).
- Single-GPU / DDP, multi-GPU pipeline sharding (``--device-map-auto``), or
  Transformers Expert/Tensor Parallel (``--distributed-ep``).
- Optional on-the-fly installation of decoded per-expert TrellisMX rank sidecars
  into a BF16 student backbone (``--per-expert-release``).
- Exact untouched FP64 KLD evaluation at configurable token milestones plus
  post-merge verification and BF16 + FP8 Engram sidecar export.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None  # type: ignore[assignment]

from trellismx_kit.engram_qec import (
    FullStackQEC,
    build_context_curriculum,
    build_nfkc_canon_map,
    topk_kl_loss,
)
from trellismx_kit.glm5_patch import patch_glm5_long_context
from trellismx_kit.kld_score import (
    ALIGNMENT_BAND,
    summarize_windows,
    tokenwise_kld_and_alignment_fp64,
    tokenwise_kld_fp64,
)
from trellismx_kit.pe_format import decode_release_layer_experts, read_rank_sidecar, tensor_sha256
from trellismx_kit.profile import ModelProfile, load_profile


class _ShardStream:
    """Sequential reader over pre-dumped sparse top-K teacher logit shards."""

    def __init__(self, shard_dir: str | Path) -> None:
        self.paths = sorted(Path(shard_dir).glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no .pt teacher shards found under {shard_dir}")
        self._shard_idx = 0
        self._row_idx = 0
        self._cur = self._load(0)
        self.seq_len = int(self._cur["token_ids"].shape[1])
        self.total_rows = sum(
            int(torch.load(p, map_location="cpu", weights_only=True)["token_ids"].shape[0])
            for p in self.paths
        )

    def state_dict(self) -> dict[str, int]:
        return {"shard_idx": self._shard_idx, "row_idx": self._row_idx}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._shard_idx = int(state["shard_idx"])
        self._row_idx = int(state["row_idx"])
        self._cur = self._load(self._shard_idx)

    def _load(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index % len(self.paths)]
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "token_ids" not in data or "top_idx" not in data:
            raise KeyError(f"teacher shard {path} missing token_ids or top_idx")
        if "top_logp" not in data and "top_logits" not in data:
            raise KeyError(f"teacher shard {path} must contain top_logp or top_logits")
        return data

    def take(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        ids_parts, idx_parts, logp_parts, ptail_parts, lse_parts = [], [], [], [], []
        remaining = count
        while remaining > 0:
            n_rows = int(self._cur["token_ids"].shape[0])
            avail = n_rows - self._row_idx
            if avail <= 0:
                self._shard_idx += 1
                self._row_idx = 0
                self._cur = self._load(self._shard_idx)
                continue
            take_n = min(remaining, avail)
            sl = slice(self._row_idx, self._row_idx + take_n)
            ids_parts.append(self._cur["token_ids"][sl])
            idx_parts.append(self._cur["top_idx"][sl])
            if "top_logp" in self._cur:
                logp_parts.append(self._cur["top_logp"][sl])
            else:
                raw = self._cur["top_logits"][sl].float()
                lse = self._cur["logsumexp"][sl].float()
                logp_parts.append((raw - lse.unsqueeze(-1)).to(torch.float16))
            if "p_tail" in self._cur:
                ptail_parts.append(self._cur["p_tail"][sl])
            if "logsumexp" in self._cur:
                lse_parts.append(self._cur["logsumexp"][sl])
            self._row_idx += take_n
            remaining -= take_n
        return (
            torch.cat(ids_parts, dim=0),
            torch.cat(idx_parts, dim=0),
            torch.cat(logp_parts, dim=0),
            torch.cat(ptail_parts, dim=0) if ptail_parts else None,
            torch.cat(lse_parts, dim=0) if lse_parts else None,
        )


class _MultiLengthShardStream:
    """Round-robin reader across one or more homogeneous shard directories (e.g. 16K, 32K, 64K)."""

    def __init__(self, shard_dirs: list[str | Path]) -> None:
        self.streams = [_ShardStream(d) for d in shard_dirs]
        if not self.streams:
            raise ValueError("at least one shard directory is required")
        # Interleave shard directories proportionally by actual row count so every milestone sees all lengths
        counts = [max(1, st.total_rows) for st in self.streams]
        g = math.gcd(*counts)
        weights = [c // g for c in counts]
        total = sum(weights)
        placed = [0] * len(self.streams)
        self.schedule: list[int] = []
        for step in range(total):
            best_i = max(
                range(len(self.streams)),
                key=lambda i: ((step + 1) * weights[i]) / total - placed[i],
            )
            self.schedule.append(best_i)
            placed[best_i] += 1
        self._pos = 0
        self.seq_len = int(
            round(
                sum(st.seq_len * st.total_rows for st in self.streams)
                / max(1, sum(st.total_rows for st in self.streams))
            )
        )

    def state_dict(self) -> dict[str, Any]:
        return {"pos": self._pos, "streams": [st.state_dict() for st in self.streams]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._pos = int(state["pos"])
        for st, sub in zip(self.streams, state["streams"], strict=True):
            st.load_state_dict(sub)

    def take(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        idx = self.schedule[self._pos % len(self.schedule)]
        self._pos += 1
        return self.streams[idx].take(count)


def _input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight.device.type != "meta":
            w = emb.weight.to_local() if hasattr(emb.weight, "to_local") else emb.weight
            return w.device
    for param in model.parameters():
        if param.device.type != "meta":
            w = param.to_local() if hasattr(param, "to_local") else param
            return w.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(
    model_path: str,
    *,
    device: torch.device,
    device_map_auto: bool,
    distributed_ep: bool,
    world_size: int,
    trust_remote_code: bool,
    attn_implementation: str = "",
) -> torch.nn.Module:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    archs = list(getattr(cfg, "architectures", None) or ())
    use_vlm = AutoModelForImageTextToText is not None and any(
        "ConditionalGeneration" in a or "Vision" in a for a in archs
    )
    loader = AutoModelForImageTextToText if use_vlm else AutoModelForCausalLM

    kwargs: dict[str, object] = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    if distributed_ep and world_size > 1:
        from transformers.distributed.configuration_utils import DistributedConfig

        cfg.base_model_tp_plan = None
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg.text_config.base_model_tp_plan = None
        kwargs["config"] = cfg
        try:
            kwargs["distributed_config"] = DistributedConfig(
                ep_size=world_size,
                tp_size=world_size,
                fsdp_size=1,
                pp_size=1,
                ep_implementation="all_reduce",
            )
        except TypeError:
            kwargs["distributed_config"] = DistributedConfig(
                tp_size=world_size,
                fsdp_size=1,
                pp_size=1,
                enable_expert_parallel=True,
            )
        model = loader.from_pretrained(model_path, **kwargs)
        patch_glm5_long_context(model)
        return model
    if device_map_auto:
        kwargs["device_map"] = "auto"
        n_gpu = torch.cuda.device_count()
        if n_gpu >= 2:
            # Reserve extra headroom on GPU 0 (embeddings + io_same_device) and last GPU (lm_head)
            max_mem: dict[int, str] = {}
            for gi in range(n_gpu):
                total_gib = int(torch.cuda.get_device_properties(gi).total_memory // (1024**3))
                reserve = 60 if (gi == 0 or gi == n_gpu - 1) else 30
                max_mem[gi] = f"{max(24, total_gib - reserve)}GiB"
            kwargs["max_memory"] = max_mem
        model = loader.from_pretrained(model_path, **kwargs)
        patch_glm5_long_context(model)
        return model
    model = loader.from_pretrained(model_path, **kwargs).to(device)
    patch_glm5_long_context(model)
    return model


def _resolve_block(model: torch.nn.Module, path: str) -> torch.nn.Module:
    obj: object = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)  # type: ignore[index]
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"{path!r} did not resolve to an nn.Module")
    return obj


def _install_per_expert_release(
    model: torch.nn.Module,
    profile: ModelProfile,
    release_dir: Path,
    *,
    ep_rank: int = 0,
    ep_world_size: int = 1,
) -> None:
    """Decode four-rate CSF TP rank sidecars, invert coupled scales, and overwrite routed experts."""
    local_count = profile.num_experts // ep_world_size
    expert_start = ep_rank * local_count
    expert_range = range(expert_start, expert_start + local_count)
    sidecar_root = release_dir / "sidecars"
    t0 = time.time()
    for layer in profile.routed_layers:
        block = _resolve_block(model, profile.layer_path_template.format(layer=layer))
        moe = getattr(block, profile.moe_attr)
        experts = moe.experts
        gate_up_target = (
            experts.gate_up_proj.to_local()
            if hasattr(experts.gate_up_proj, "to_local")
            else experts.gate_up_proj
        )
        down_target = (
            experts.down_proj.to_local()
            if hasattr(experts.down_proj, "to_local")
            else experts.down_proj
        )
        sidecars = [
            read_rank_sidecar(
                sidecar_root / f"p8-layer-{layer:03d}-tp{profile.tp_size}-rank-{r}.safetensors",
                profile,
            )
            for r in range(profile.tp_size)
        ]
        decode_dev = gate_up_target.device if gate_up_target.device.type == "cuda" else torch.device("cpu")
        gu_err_sq = 0.0
        gu_ref_sq = 0.0
        dn_err_sq = 0.0
        dn_ref_sq = 0.0
        expert_list = list(expert_range)
        # Decode in 16-expert chunks so FP32 peak stays ~1.5 GiB instead of 27 GiB
        for c0 in range(0, len(expert_list), 16):
            chunk_es = expert_list[c0 : c0 + 16]
            decoded = decode_release_layer_experts(
                sidecars,
                profile,
                chunk_es,
                device=decode_dev,
                invert_coupled_scales=True,
            )
            with torch.no_grad():
                for global_e in chunk_es:
                    local_e = global_e - expert_start
                    gu_w, dn_w = decoded[global_e]
                    gu_cast = gu_w.to(device=gate_up_target.device, dtype=gate_up_target.dtype)
                    dn_cast = dn_w.to(device=down_target.device, dtype=down_target.dtype)
                    gu_ref = gate_up_target[local_e].float()
                    dn_ref = down_target[local_e].float()
                    gu_err_sq += float((gu_cast.float() - gu_ref).square().sum().item())
                    gu_ref_sq += float(gu_ref.square().sum().item())
                    dn_err_sq += float((dn_cast.float() - dn_ref).square().sum().item())
                    dn_ref_sq += float(dn_ref.square().sum().item())
                    gate_up_target[local_e].copy_(gu_cast)
                    down_target[local_e].copy_(dn_cast)
                    if not torch.equal(gate_up_target[local_e], gu_cast):
                        raise RuntimeError(f"gate_up_proj install check failed at layer {layer} expert {global_e}")
                    if not torch.equal(down_target[local_e], dn_cast):
                        raise RuntimeError(f"down_proj install check failed at layer {layer} expert {global_e}")
            decoded.clear()
        gu_nmse = gu_err_sq / max(1e-30, gu_ref_sq)
        dn_nmse = dn_err_sq / max(1e-30, dn_ref_sq)
        if gu_nmse > 0.25 or dn_nmse > 0.25:
            raise RuntimeError(
                f"installed expert NMSE sanity check failed at layer {layer}: "
                f"gate_up_nmse={gu_nmse:.4f}, down_nmse={dn_nmse:.4f} (expected < 0.25)"
            )
        if ep_rank == 0 and (layer in (profile.routed_layers[0], profile.routed_layers[len(profile.routed_layers) // 2], profile.routed_layers[-1])):
            print(f"[install] layer {layer:02d} verified vs BF16: gate_up_nmse={gu_nmse:.4f} down_nmse={dn_nmse:.4f}", flush=True)
        sidecars.clear()
    if ep_rank == 0:
        print(
            f"[install] Decoded and installed {len(profile.routed_layers)} routed layers "
            f"from {release_dir} in {time.time() - t0:.1f}s",
            flush=True,
        )


@torch.no_grad()
def evaluate_dense_dir(
    eval_model: torch.nn.Module,
    eval_dir: Path,
    *,
    vocab_limit: int | None = None,
    resamples: int = 5000,
    self_check: bool = True,
    strict_alignment: bool = False,
    is_writer: bool = True,
) -> dict[str, dict[str, object]]:
    """Evaluate untouched exact FP64 KLD grouped by split prefix in ``eval_dir``."""
    was_training = eval_model.training
    eval_model.eval()
    base_mod = eval_model.model if isinstance(eval_model, FullStackQEC) else eval_model
    dev = _input_device(base_mod)
    paths = sorted(eval_dir.glob("*.pt"))
    grouped: dict[str, list[Path]] = {}
    for p in paths:
        split = p.stem.split("-window-")[0]
        grouped.setdefault(split, []).append(p)

    results: dict[str, dict[str, object]] = {}
    for split, split_paths in grouped.items():
        klds: list[torch.Tensor] = []
        agrees: list[torch.Tensor] = []
        t_lps: list[torch.Tensor] = []
        s_lps: list[torch.Tensor] = []
        t_hits: list[torch.Tensor] = []
        self_checked = False
        for p in split_paths:
            rec = torch.load(p, map_location="cpu", weights_only=True)
            token_ids = rec["token_ids"]
            ids = token_ids.unsqueeze(0).to(device=dev, dtype=torch.long)
            s_logits = eval_model(input_ids=ids, use_cache=False).logits[0, :-1].cpu()
            if is_writer:
                t_logits = rec["logits"]
                if self_check and not self_checked:
                    self_kld, _ = tokenwise_kld_fp64(t_logits, t_logits, vocab_limit=vocab_limit)
                    if float(self_kld.max().item()) != 0.0:
                        raise RuntimeError(f"teacher self-KLD canary failed on {p.name}: {self_kld.max().item()}")
                    self_checked = True
                kld, agree, t_lp, s_lp, hit = tokenwise_kld_and_alignment_fp64(
                    t_logits, s_logits, token_ids[1:], vocab_limit=vocab_limit
                )
                klds.append(kld)
                agrees.append(agree)
                t_lps.append(t_lp)
                s_lps.append(s_lp)
                t_hits.append(hit)
            del ids, s_logits
        if is_writer and klds:
            summ = summarize_windows(
                klds,
                agrees,
                window_teacher_logp=t_lps,
                window_student_logp=s_lps,
                window_teacher_hits=t_hits,
                resamples=resamples,
            )
            align_ok = (
                True
                if summ.realized_alignment_rate is None
                else (ALIGNMENT_BAND[0] <= summ.realized_alignment_rate <= ALIGNMENT_BAND[1])
            )
            if not align_ok:
                msg = (
                    f"realized-token alignment {summ.realized_alignment_rate:.4f} outside "
                    f"allowed band {ALIGNMENT_BAND} on split {split}"
                )
                if strict_alignment:
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}", flush=True)
            results[split] = {
                "windows": summ.windows,
                "positions": summ.positions,
                "mean_kld": summ.mean_kld,
                "median_kld": summ.median_kld,
                "p90_kld": summ.p90_kld,
                "p99_kld": summ.p99_kld,
                "p99_9_kld": summ.p99_9_kld,
                "max_kld": summ.max_kld,
                "ci95_low": summ.ci95_low,
                "ci95_high": summ.ci95_high,
                "percentile_ci95_low": summ.percentile_ci95_low,
                "percentile_ci95_high": summ.percentile_ci95_high,
                "top1_agreement_pct": summ.top1_agreement_pct,
                "top1_wilson95_low_pct": summ.top1_wilson95_low_pct,
                "top1_wilson95_high_pct": summ.top1_wilson95_high_pct,
                "perplexity_ratio": summ.perplexity_ratio,
                "realized_alignment_rate": summ.realized_alignment_rate,
                "alignment_ok": align_ok,
            }
    if was_training:
        eval_model.train()
    return results


def _parse_bank_groups(spec: str) -> dict[str, list[int]] | None:
    if not spec.strip():
        return None
    groups: dict[str, list[int]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, layers_str = part.split(":", 1)
        groups[name.strip()] = [int(x) for x in layers_str.split(",") if x.strip()]
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill Dual-Trellis QEC v2 on a frozen quantized MoE with an 80/20 curriculum."
    )
    parser.add_argument("--profile", required=True, help="ModelProfile JSON path")
    parser.add_argument("--student-model", required=True, help="Quantized student HF checkpoint (or BF16 backbone with --per-expert-release)")
    parser.add_argument("--per-expert-release", default="", help="Optional per-expert TrellisMX release dir to decode into student experts")
    parser.add_argument("--teacher-model", default="", help="Live teacher HF checkpoint path (optional if --teacher-shards is set)")
    parser.add_argument("--corpus", default="", help="Short-context pre-tokenized int32 [N, seq_len] .pt file")
    parser.add_argument("--long-corpus", default="", help="Long-context pre-tokenized int32 [N, seq_len] .pt file")
    parser.add_argument("--teacher-shards", default="", help="Directory of pre-dumped short-context sparse top-K teacher shards")
    parser.add_argument(
        "--long-teacher-shards",
        nargs="*",
        default=[],
        help="One or more directories of pre-dumped long-context sparse top-K teacher shards (e.g. 16K, 32K, 64K)",
    )
    parser.add_argument("--eval-dense-dir", default="", help="Optional eval_dense/ directory for untouched FP64 KLD milestones")
    parser.add_argument("--eval-milestones-m", default="10,35,50,75,100", help="Comma-separated token milestones (millions) for eval + checkpointing")
    parser.add_argument("--long-fraction", type=float, default=0.20)
    parser.add_argument("--curriculum-mode", choices=("interleaved", "tail"), default="interleaved")
    parser.add_argument("--engram-mode", choices=("v2_shared_banks", "v1_per_layer"), default="v2_shared_banks")
    parser.add_argument("--engram-layers", default="", help="Comma-separated layer indices for Engram QEC")
    parser.add_argument("--engram-banks", default="", help="Explicit depth banks, e.g. 'early:3,5,7,9;mid:12,16,20,24,28,32;late:36,39,42,44'")
    parser.add_argument("--engram-hook-target", choices=("auto", "block", "mlp"), default="auto")
    parser.add_argument("--engram-table-device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--canon-tokenizer", default="", help="Tokenizer path for NFKC canonicalization LUT (defaults to --student-model)")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--train-dense-lora", action="store_true")
    parser.add_argument("--train-shared-lora", action="store_true")
    parser.add_argument("--train-lm-head-lora", action="store_true")
    parser.add_argument("--train-inter-scales", action="store_true")
    parser.add_argument("--train-expert-bias", action="store_true")
    parser.add_argument("--use-router-gate", action="store_true")
    parser.add_argument("--buckets", type=int, default=1_048_573)
    parser.add_argument("--skip-heads", type=int, default=2)
    parser.add_argument("--unigram-only", action="store_true")
    parser.add_argument("--tokens-m", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--long-batch-size", type=int, default=1)
    parser.add_argument("--long-grad-accum", type=int, default=4)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--device-map-auto", action="store_true", help="Shard frozen student across GPUs with device_map='auto'")
    parser.add_argument("--distributed-ep", action="store_true", help="Run torchrun Expert/Tensor Parallel student")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--loss-chunk-tokens", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    profile = load_profile(args.profile)

    use_shards = bool(args.teacher_shards)
    if not use_shards and (not args.teacher_model or not args.corpus):
        raise ValueError("Provide either --teacher-shards or both --teacher-model and --corpus")

    distributed = "RANK" in os.environ
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_writer = rank == 0
    dp_world = 1 if (args.distributed_ep or args.device_map_auto) else world_size
    dp_rank = 0 if dp_world == 1 else rank

    short_shards = _ShardStream(args.teacher_shards) if use_shards else None
    long_shard_dirs = [d for d in (args.long_teacher_shards or []) if d]
    long_shards = _MultiLengthShardStream(long_shard_dirs) if long_shard_dirs else None
    short_corpus = None if use_shards else torch.load(args.corpus, map_location="cpu", weights_only=True)
    long_corpus = (
        torch.load(args.long_corpus, map_location="cpu", weights_only=True)
        if (not use_shards and args.long_corpus)
        else None
    )

    short_seq_len = short_shards.seq_len if short_shards is not None else int(short_corpus.shape[1])
    has_long = (long_shards is not None) or (long_corpus is not None)
    long_seq_len = (
        long_shards.seq_len
        if long_shards is not None
        else (int(long_corpus.shape[1]) if long_corpus is not None else short_seq_len)
    )
    effective_long_fraction = args.long_fraction if has_long else 0.0

    teacher = None
    if not use_shards:
        teacher = _load_model(
            args.teacher_model,
            device=device,
            device_map_auto=args.device_map_auto,
            distributed_ep=args.distributed_ep,
            world_size=world_size,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        ).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    student = _load_model(
        args.student_model,
        device=device,
        device_map_auto=args.device_map_auto,
        distributed_ep=args.distributed_ep,
        world_size=world_size,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if args.per_expert_release:
        _install_per_expert_release(
            student,
            profile,
            Path(args.per_expert_release),
            ep_rank=rank if args.distributed_ep else 0,
            ep_world_size=world_size if args.distributed_ep else 1,
        )

    out_dir = Path(args.output)
    eval_dir = Path(args.eval_dense_dir) if args.eval_dense_dir else None
    stock_baseline_eval = None
    if eval_dir is not None and not args.resume:
        stock_baseline_eval = evaluate_dense_dir(
            student, eval_dir, vocab_limit=profile.vocab_size, is_writer=is_writer
        )
        if is_writer:
            print(f"[eval stock BF16 baseline @ 0.0M] {json.dumps(stock_baseline_eval)}", flush=True)

    if args.grad_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        try:
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.gradient_checkpointing_enable()
        if hasattr(student, "config"):
            student.config.use_cache = False
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

    bank_groups = _parse_bank_groups(args.engram_banks)
    engram_layers = [int(x) for x in args.engram_layers.split(",") if x.strip()]
    if not engram_layers and bank_groups:
        engram_layers = sorted({l for ls in bank_groups.values() for l in ls})

    canon_map = None
    canon_map_sha256 = ""
    tok_src = args.canon_tokenizer or args.student_model
    if not args.unigram_only and tok_src:
        tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=args.trust_remote_code)
        canon_map = build_nfkc_canon_map(tok, profile.vocab_size)
        canon_map_sha256 = tensor_sha256(canon_map)
        if is_writer:
            unique_canon = int(canon_map.unique().numel())
            print(f"[canon] built NFKC LUT ({profile.vocab_size} -> {unique_canon} canonical IDs, sha256={canon_map_sha256[:16]})", flush=True)

    table_dev = torch.device("cpu") if args.engram_table_device == "cpu" else None
    ek: dict[str, object] = (
        {"d_uni": 256, "n_bi_heads": 0, "n_tri_heads": 0, "n_skip_heads": 0, "table_device": table_dev}
        if args.unigram_only
        else {
            "d_uni": 128,
            "d_bi": 64,
            "n_bi_heads": 4,
            "buckets": args.buckets,
            "bi_buckets": args.buckets,
            "d_tri": 64,
            "n_tri_heads": 2,
            "tri_buckets": args.buckets,
            "d_skip": 64 if args.skip_heads > 0 else 0,
            "n_skip_heads": args.skip_heads,
            "skip_buckets": args.buckets,
            "canon_map": canon_map,
            "table_device": table_dev,
        }
    )

    qec = FullStackQEC(
        student,
        profile,
        engram_layers=engram_layers,
        engram_mode=args.engram_mode,
        engram_bank_groups=bank_groups,
        engram_hook_target=args.engram_hook_target,
        lora_rank=args.lora_rank,
        engram_kwargs=ek,
        train_norms=True,
        train_dense_lora=args.train_dense_lora,
        train_shared_lora=args.train_shared_lora,
        train_lm_head_lora=args.train_lm_head_lora,
        train_inter_scales=args.train_inter_scales,
        train_expert_bias=args.train_expert_bias,
        use_router_gate=args.use_router_gate,
        ep_rank=rank if args.distributed_ep else 0,
        ep_world_size=world_size if args.distributed_ep else 1,
    )
    if is_writer:
        n_attn_l = sum(1 for k in qec.loras if "_attn_" in k)
        n_dense_l = sum(1 for k in qec.loras if "_dense_" in k)
        n_shared_l = sum(1 for k in qec.loras if "_shared_" in k)
        n_lm_l = 1 if "lm_head" in qec.loras else 0
        print(
            f"[qec] attached LoRAs: attn={n_attn_l} dense={n_dense_l} shared={n_shared_l} lm_head={n_lm_l} "
            f"| promoted_norms={len(qec._promoted_norm_keys)} | engram_layers={len(qec.engrams)} | banks={list(qec.banks.keys())}",
            flush=True,
        )
    if args.lora_rank > 0 and len(qec.loras) == 0:
        raise RuntimeError("lora_rank > 0 but zero LoRA adapters attached; check ModelProfile projection names")
    if not (args.device_map_auto or args.distributed_ep):
        qec = qec.to(device)
    qec.train()

    if distributed and (not args.device_map_auto) and (not args.distributed_ep) and (table_dev is not None):
        raise ValueError("DDP training cannot combine --engram-table-device cpu with multi-rank torchrun")
    use_ddp = distributed and (not args.device_map_auto) and (not args.distributed_ep) and (table_dev is None)
    model_run: torch.nn.Module = DDP(qec, device_ids=[local_rank]) if use_ddp else qec
    param_groups = qec.parameter_groups()
    opt = torch.optim.AdamW(param_groups)

    short_chunks_per_step = dp_world * args.batch_size * args.grad_accum
    long_chunks_per_step = dp_world * args.long_batch_size * args.long_grad_accum
    short_tokens_per_step = short_chunks_per_step * (short_seq_len - 1)
    long_tokens_per_step = long_chunks_per_step * (long_seq_len - 1)

    schedule = build_context_curriculum(
        total_tokens=max(1, int(round(args.tokens_m * 1e6))),
        short_tokens_per_step=short_tokens_per_step,
        long_tokens_per_step=long_tokens_per_step,
        long_fraction=effective_long_fraction,
        mode=args.curriculum_mode,
    )
    total_steps = len(schedule)
    warmup = max(20, total_steps // 20)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_step = 0
    short_cursor = 0
    long_cursor = 0
    short_tokens_seen = 0
    long_tokens_seen = 0
    milestones_m = sorted(float(x) for x in args.eval_milestones_m.split(",") if x.strip())
    next_milestone_idx = 0
    eval_history: list[dict[str, Any]] = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        qec.banks.load_state_dict(ckpt["banks"])
        qec.engrams.load_state_dict(ckpt["engrams"])
        qec.loras.load_state_dict(ckpt["loras"])
        if "inter_scales" in ckpt:
            qec.inter_scales.load_state_dict(ckpt["inter_scales"])
        if "expert_bias" in ckpt:
            qec.expert_bias.load_state_dict(ckpt["expert_bias"])
        if "norms" in ckpt:
            named_p = dict(qec.model.named_parameters())
            with torch.no_grad():
                for k, v in ckpt["norms"].items():
                    if k in named_p:
                        named_p[k].data.copy_(v.to(device=named_p[k].device, dtype=named_p[k].dtype))
        if short_shards is not None and "short_shard_state" in ckpt:
            short_shards.load_state_dict(ckpt["short_shard_state"])
        if long_shards is not None and "long_shard_state" in ckpt:
            long_shards.load_state_dict(ckpt["long_shard_state"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"])
        short_cursor = int(ckpt["short_cursor"])
        long_cursor = int(ckpt["long_cursor"])
        short_tokens_seen = int(ckpt["short_tokens_seen"])
        long_tokens_seen = int(ckpt["long_tokens_seen"])
        if "eval_history" in ckpt:
            eval_history = list(ckpt["eval_history"])
        seen_m = (short_tokens_seen + long_tokens_seen) / 1e6
        while next_milestone_idx < len(milestones_m) and milestones_m[next_milestone_idx] <= seen_m + 1e-6:
            next_milestone_idx += 1

    if eval_dir is not None and start_step == 0 and stock_baseline_eval is not None:
        if is_writer:
            eval_history.append({"tokens_m": 0.0, "step": 0, "eval": stock_baseline_eval})

    in_dev = _input_device(qec.model)
    t0 = time.time()

    def _save_checkpoint(tag: str, cur_step: int) -> None:
        if not is_writer:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        norm_keys = set(qec._promoted_norm_keys)
        norms_state = {
            k: v.detach().cpu()
            for k, v in qec.model.named_parameters()
            if k in norm_keys
        }
        payload: dict[str, Any] = {
            "step": cur_step,
            "banks": qec.banks.state_dict(),
            "engrams": qec.engrams.state_dict(),
            "loras": qec.loras.state_dict(),
            "inter_scales": qec.inter_scales.state_dict(),
            "expert_bias": qec.expert_bias.state_dict(),
            "norms": norms_state,
            "eval_history": eval_history,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "short_cursor": short_cursor,
            "long_cursor": long_cursor,
            "short_tokens_seen": short_tokens_seen,
            "long_tokens_seen": long_tokens_seen,
        }
        if short_shards is not None:
            payload["short_shard_state"] = short_shards.state_dict()
        if long_shards is not None:
            payload["long_shard_state"] = long_shards.state_dict()
        ckpt_path = out_dir / f"checkpoint_{tag}.pt"
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)

    last_step_eval = None
    for step in range(start_step, total_steps):
        phase = schedule[step]
        is_long = phase == "long"
        micro_bs = args.long_batch_size if is_long else args.batch_size
        grad_accum = args.long_grad_accum if is_long else args.grad_accum
        chunks_this_step = dp_world * micro_bs * grad_accum

        if use_shards:
            stream = long_shards if is_long else short_shards
            assert stream is not None
            step_ids, step_idx, step_logp, step_ptail, _step_lse = stream.take(chunks_this_step)
        else:
            src_corpus = long_corpus if is_long else short_corpus
            assert src_corpus is not None
            cursor = long_cursor if is_long else short_cursor
            rows = [src_corpus[(cursor + i) % len(src_corpus)] for i in range(chunks_this_step)]
            step_ids = torch.stack(rows, dim=0)
            step_idx = step_logp = step_ptail = None
            if is_long:
                long_cursor += chunks_this_step
            else:
                short_cursor += chunks_this_step

        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for micro in range(grad_accum):
            offset = (micro * dp_world + dp_rank) * micro_bs
            ids = step_ids[offset : offset + micro_bs].to(device=in_dev, dtype=torch.long)
            if use_shards:
                assert step_idx is not None and step_logp is not None
                top_idx = step_idx[offset : offset + micro_bs]
                top_logp = step_logp[offset : offset + micro_bs]
                p_tail = step_ptail[offset : offset + micro_bs] if step_ptail is not None else None
            else:
                assert teacher is not None
                t_dev = _input_device(teacher)
                with torch.no_grad():
                    t_logits = teacher(input_ids=ids.to(t_dev), use_cache=False).logits[:, :-1].float()
                    t_lse = torch.logsumexp(t_logits, dim=-1)
                    top_vals, top_idx = torch.topk(t_logits, k=args.topk, dim=-1)
                    top_logp = (top_vals - t_lse.unsqueeze(-1)).clamp_max(0.0)
                    p_tail = (1.0 - top_logp.exp().sum(dim=-1)).clamp(0.0, 1.0)

            ctx = model_run.no_sync() if (use_ddp and micro + 1 < grad_accum) else torch.enable_grad()
            with ctx:
                s_logits = model_run(input_ids=ids, use_cache=False).logits[:, :-1]
                loss = (
                    topk_kl_loss(
                        s_logits,
                        top_logp,
                        top_idx,
                        teacher_p_tail=p_tail,
                        chunk_tokens=args.loss_chunk_tokens,
                    )
                    / grad_accum
                )
                loss.backward()
                acc_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        if is_long:
            long_tokens_seen += int(step_ids.shape[0] * (step_ids.shape[1] - 1))
        else:
            short_tokens_seen += int(step_ids.shape[0] * (step_ids.shape[1] - 1))

        seen = short_tokens_seen + long_tokens_seen
        seen_m = seen / 1e6
        if is_writer and ((step + 1) % 25 == 0 or step + 1 == total_steps):
            print(
                f"step={step+1}/{total_steps} split={phase} positions={seen_m:.2f}M "
                f"(short={short_tokens_seen/1e6:.2f}M long={long_tokens_seen/1e6:.2f}M) "
                f"loss={acc_loss:.4f} t={time.time()-t0:.1f}s",
                flush=True,
            )

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            _save_checkpoint("resumable", step + 1)

        hit_milestone = (
            next_milestone_idx < len(milestones_m)
            and (seen_m >= milestones_m[next_milestone_idx] or (step + 1 == total_steps))
        )
        if hit_milestone:
            m_tag = f"{milestones_m[next_milestone_idx]:.0f}m"
            _save_checkpoint(m_tag, step + 1)
            if eval_dir is not None:
                m_eval = evaluate_dense_dir(qec, eval_dir, vocab_limit=profile.vocab_size, is_writer=is_writer)
                if step + 1 == total_steps:
                    last_step_eval = m_eval
                if is_writer:
                    eval_history.append({"tokens_m": round(seen_m, 3), "step": step + 1, "eval": m_eval})
                    print(f"[eval @ {seen_m:.2f}M] {json.dumps(m_eval)}", flush=True)
            next_milestone_idx += 1

    if distributed:
        dist.barrier()

    pre_merge_eval = (
        last_step_eval
        if last_step_eval is not None
        else (
            evaluate_dense_dir(qec, eval_dir, vocab_limit=profile.vocab_size, is_writer=is_writer)
            if eval_dir is not None
            else None
        )
    )
    merged = qec.merge_lora_into_model()
    restored_norms = qec.finalize_norms_to_carrier_dtype()
    if is_writer:
        print(f"[finalize] merged {merged} LoRA projections and restored {restored_norms} norms to stock carrier BF16", flush=True)

    post_merge_carrier_only_eval = None
    post_merge_bf16_engram_eval = None
    post_merge_fp8_engram_eval = None
    if eval_dir is not None:
        # 1. Zero-sidecar merged carrier only (merged BF16 LoRA + stock BF16 norms, Engram disabled)
        qec.engram_enabled = False
        post_merge_carrier_only_eval = evaluate_dense_dir(
            qec, eval_dir, vocab_limit=profile.vocab_size, is_writer=is_writer
        )
        qec.engram_enabled = True
        # 2. Exact shipped BF16 artifact (stock BF16 norms + BF16 Engram projections + BF16 tables)
        bf16_bak = qec.quantize_engram_tables_inplace(torch.bfloat16, quantize_projections_bf16=True)
        post_merge_bf16_engram_eval = evaluate_dense_dir(
            qec, eval_dir, vocab_limit=profile.vocab_size, is_writer=is_writer
        )
        qec.restore_engram_tables(bf16_bak)
        # 3. Exact shipped FP8 artifact (stock BF16 norms + BF16 Engram projections + FP8 E4M3 tables)
        fp8_bak = qec.quantize_engram_tables_inplace(torch.float8_e4m3fn, quantize_projections_bf16=True)
        post_merge_fp8_engram_eval = evaluate_dense_dir(
            qec, eval_dir, vocab_limit=profile.vocab_size, is_writer=is_writer
        )
        qec.restore_engram_tables(fp8_bak)
        if is_writer:
            print(f"[eval post-merge carrier-only (0 sidecar)] {json.dumps(post_merge_carrier_only_eval)}", flush=True)
            print(f"[eval post-merge shipped BF16 engram]     {json.dumps(post_merge_bf16_engram_eval)}", flush=True)
            print(f"[eval post-merge shipped FP8 engram]      {json.dumps(post_merge_fp8_engram_eval)}", flush=True)

    if is_writer:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(qec.export_engram_sidecar(table_dtype=torch.bfloat16), out_dir / "engram_qec_sidecar_bf16.pt")
        torch.save(qec.export_engram_sidecar(table_dtype=torch.float8_e4m3fn), out_dir / "engram_qec_sidecar_fp8.pt")
        torch.save(qec.export_non_expert_state(), out_dir / "merged_non_expert_qec.pt")
        expert_qec = qec.export_expert_qec_state()
        if expert_qec:
            torch.save(expert_qec, out_dir / "expert_scales_and_bias_qec.pt")

        total_seen = short_tokens_seen + long_tokens_seen
        manifest = {
            "profile": profile.name,
            "engram_mode": args.engram_mode,
            "supervised_positions": total_seen,
            "short_positions": short_tokens_seen,
            "long_positions": long_tokens_seen,
            "short_seq_len": short_seq_len,
            "long_seq_len": long_seq_len if has_long else 0,
            "long_fraction_target": effective_long_fraction,
            "curriculum_mode": args.curriculum_mode,
            "merged_lora_projections": merged,
            "engram_layers": list(engram_layers),
            "engram_bank_map": dict(qec.bank_map),
            "buckets": args.buckets,
            "skip_heads": args.skip_heads,
            "train_inter_scales": args.train_inter_scales,
            "train_expert_bias": args.train_expert_bias,
            "train_lm_head_lora": args.train_lm_head_lora,
            "use_router_gate": args.use_router_gate,
            "canon_map_sha256": canon_map_sha256,
            "stock_baseline_eval": stock_baseline_eval,
            "pre_merge_eval": pre_merge_eval,
            "post_merge_carrier_only_eval": post_merge_carrier_only_eval,
            "post_merge_bf16_engram_eval": post_merge_bf16_engram_eval,
            "post_merge_fp8_engram_eval": post_merge_fp8_engram_eval,
            "eval_history": eval_history,
        }
        (out_dir / "qec_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Saved QEC artifacts to {out_dir}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
