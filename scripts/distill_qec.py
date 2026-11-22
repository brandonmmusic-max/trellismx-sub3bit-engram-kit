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
from trellismx_kit.kld_score import summarize_windows, tokenwise_kld_fp64
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
        # Interleave shard directories proportionally by shard count so every milestone sees all lengths
        counts = [max(1, len(st.paths)) for st in self.streams]
        total = sum(counts)
        placed = [0] * len(self.streams)
        self.schedule: list[int] = []
        for step in range(total):
            best_i = max(
                range(len(self.streams)),
                key=lambda i: ((step + 1) * counts[i]) / total - placed[i],
            )
            self.schedule.append(best_i)
            placed[best_i] += 1
        self._pos = 0
        self.seq_len = int(
            round(
                sum(st.seq_len * max(1, len(st.paths)) for st in self.streams)
                / max(1, len(self.schedule))
            )
        )

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

        try:
            kwargs["distributed_config"] = DistributedConfig(
                ep_size=world_size,
                tp_size=world_size,
                tp_plan={},
                fsdp_size=1,
                pp_size=1,
            )
        except TypeError:
            kwargs["distributed_config"] = DistributedConfig(
                tp_size=world_size,
                fsdp_size=1,
                pp_size=1,
                enable_expert_parallel=True,
            )
        return loader.from_pretrained(model_path, **kwargs)
    if device_map_auto:
        kwargs["device_map"] = "auto"
        return loader.from_pretrained(model_path, **kwargs)
    return loader.from_pretrained(model_path, **kwargs).to(device)


def _install_per_expert_release(
    model: torch.nn.Module,
    profile: ModelProfile,
    release_dir: Path,
    *,
    ep_rank: int = 0,
    ep_world_size: int = 1,
) -> None:
    """Decode per-expert TP rank sidecars and overwrite routed experts in-place."""
    from glm53kit.kit.kld.capture_student import install_expert_weights, load_release_experts

    local_count = profile.num_experts // ep_world_size
    expert_start = ep_rank * local_count
    expert_range = range(expert_start, expert_start + local_count)
    for layer in profile.routed_layers:
        decoded = load_release_experts(
            release_dir,
            layer,
            expert_range,
            hidden=profile.hidden_size,
            intermediate=profile.intermediate_size,
            tp_size=profile.tp_size,
        )
        weights = {(layer, e): decoded[e] for e in expert_range}
        install_expert_weights(
            model,
            weights,
            world_size=ep_world_size,
            rank=ep_rank,
            expert_count=profile.num_experts,
            hidden=profile.hidden_size,
            intermediate=profile.intermediate_size,
            layers=(layer,),
        )
        weights.clear()


@torch.no_grad()
def evaluate_dense_dir(
    qec: FullStackQEC,
    eval_dir: Path,
    *,
    is_writer: bool = True,
) -> dict[str, dict[str, object]]:
    """Evaluate untouched exact FP64 KLD grouped by split prefix in ``eval_dir``."""
    was_training = qec.training
    qec.eval()
    dev = _input_device(qec.model)
    paths = sorted(eval_dir.glob("*.pt"))
    grouped: dict[str, list[Path]] = {}
    for p in paths:
        split = p.stem.split("-window-")[0]
        grouped.setdefault(split, []).append(p)

    results: dict[str, dict[str, object]] = {}
    for split, split_paths in grouped.items():
        klds: list[torch.Tensor] = []
        agrees: list[torch.Tensor] = []
        for p in split_paths:
            rec = torch.load(p, map_location="cpu", weights_only=True)
            ids = rec["token_ids"].unsqueeze(0).to(device=dev, dtype=torch.long)
            s_logits = qec(ids, use_cache=False).logits[0, :-1].cpu()
            if is_writer:
                t_logits = rec["logits"]
                kld, agree = tokenwise_kld_fp64(t_logits, s_logits)
                klds.append(kld)
                agrees.append(agree)
            del ids, s_logits
        if is_writer and klds:
            summ = summarize_windows(klds, agrees, resamples=1000)
            results[split] = {
                "windows": summ.windows,
                "positions": summ.positions,
                "mean_kld": summ.mean_kld,
                "median_kld": summ.median_kld,
                "p90_kld": summ.p90_kld,
                "p99_kld": summ.p99_kld,
                "ci95_low": summ.ci95_low,
                "ci95_high": summ.ci95_high,
                "top1_agreement": summ.top1_agreement,
            }
    if was_training:
        qec.train()
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

    if args.grad_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

    bank_groups = _parse_bank_groups(args.engram_banks)
    engram_layers = [int(x) for x in args.engram_layers.split(",") if x.strip()]
    if not engram_layers and bank_groups:
        engram_layers = sorted({l for ls in bank_groups.values() for l in ls})

    canon_map = None
    tok_src = args.canon_tokenizer or args.student_model
    if not args.unigram_only and tok_src:
        try:
            tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=args.trust_remote_code)
            canon_map = build_nfkc_canon_map(tok, profile.vocab_size)
        except Exception:
            canon_map = None

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
    if not (args.device_map_auto or args.distributed_ep):
        qec = qec.to(device)
    qec.train()

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
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"])
        short_cursor = int(ckpt["short_cursor"])
        long_cursor = int(ckpt["long_cursor"])
        short_tokens_seen = int(ckpt["short_tokens_seen"])
        long_tokens_seen = int(ckpt["long_tokens_seen"])
        seen_m = (short_tokens_seen + long_tokens_seen) / 1e6
        while next_milestone_idx < len(milestones_m) and milestones_m[next_milestone_idx] <= seen_m + 1e-6:
            next_milestone_idx += 1

    out_dir = Path(args.output)
    eval_dir = Path(args.eval_dense_dir) if args.eval_dense_dir else None
    if eval_dir is not None and start_step == 0:
        base_eval = evaluate_dense_dir(qec, eval_dir, is_writer=is_writer)
        if is_writer:
            eval_history.append({"tokens_m": 0.0, "step": 0, "eval": base_eval})
            print(f"[eval @ 0.0M] {json.dumps(base_eval)}", flush=True)

    in_dev = _input_device(qec.model)
    t0 = time.time()

    def _save_checkpoint(tag: str, cur_step: int) -> None:
        if not is_writer:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": cur_step,
                "banks": qec.banks.state_dict(),
                "engrams": qec.engrams.state_dict(),
                "loras": qec.loras.state_dict(),
                "inter_scales": qec.inter_scales.state_dict(),
                "expert_bias": qec.expert_bias.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "short_cursor": short_cursor,
                "long_cursor": long_cursor,
                "short_tokens_seen": short_tokens_seen,
                "long_tokens_seen": long_tokens_seen,
            },
            out_dir / f"checkpoint_{tag}.pt",
        )

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

        if next_milestone_idx < len(milestones_m) and seen_m >= milestones_m[next_milestone_idx]:
            m_tag = f"{milestones_m[next_milestone_idx]:.0f}m"
            _save_checkpoint(m_tag, step + 1)
            if eval_dir is not None:
                m_eval = evaluate_dense_dir(qec, eval_dir, is_writer=is_writer)
                if is_writer:
                    eval_history.append({"tokens_m": round(seen_m, 3), "step": step + 1, "eval": m_eval})
                    print(f"[eval @ {seen_m:.2f}M] {json.dumps(m_eval)}", flush=True)
            next_milestone_idx += 1

    if distributed:
        dist.barrier()

    pre_merge_eval = evaluate_dense_dir(qec, eval_dir, is_writer=is_writer) if eval_dir is not None else None
    merged = qec.merge_lora_into_model()
    post_merge_eval = evaluate_dense_dir(qec, eval_dir, is_writer=is_writer) if eval_dir is not None else None

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
            "pre_merge_eval": pre_merge_eval,
            "post_merge_eval": post_merge_eval,
            "eval_history": eval_history,
        }
        (out_dir / "qec_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Saved QEC artifacts to {out_dir}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
