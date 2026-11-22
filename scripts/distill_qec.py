#!/usr/bin/env python3
"""Model-neutral DDP Full-Stack QEC distillation CLI with 80/20 context curriculum.

Freezes all routed MoE expert weights, routers, and sparse attention indexers of
a quantized Hugging Face checkpoint and distills mergeable Attention LoRA,
RMSNorm scales, and Host-RAM Engram QEC tables against either a live teacher
model or pre-dumped sparse top-K teacher shards (with exact full-vocabulary
logsumexp tail calibration).

Default long-context curriculum allocates 80% of tokens to short 2K-4K windows
for high-throughput n-gram saturation and 20% to long 16K-32K+ windows so
Attention LoRA and Engram context gates calibrate across sparse long-context
retrieval.

Usage (4-GPU DDP with live teacher):
    NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 scripts/distill_qec.py \
        --profile configs/granite-3.1-3b-a800m.json \
        --teacher-model /path/to/teacher \
        --student-model /path/to/quantized-student \
        --corpus /path/to/short_2048.pt \
        --long-corpus /path/to/long_16384.pt \
        --long-fraction 0.20 --long-batch-size 1 --grad-checkpointing \
        --engram-layers 0,1,2,4,8,16,24,31 \
        --tokens-m 50 --output /path/to/qec-out

Usage (4-GPU DDP with pre-dumped sparse top-K teacher shards):
    NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 scripts/distill_qec.py \
        --profile configs/glm-5.3-flash.json \
        --student-model /path/to/quantized-student \
        --teacher-shards /path/to/short_shards \
        --long-teacher-shards /path/to/long_shards \
        --long-fraction 0.20 --long-batch-size 1 --grad-checkpointing \
        --engram-layers 3,6,9,12,15,18,21,24,27,30,33,36,39,42,44 \
        --buckets 524287 --tokens-m 50 --output /path/to/qec-out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from trellismx_kit.engram_qec import (
    FullStackQEC,
    build_context_curriculum,
    topk_kl_loss,
)
from trellismx_kit.profile import load_profile


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
        data = torch.load(self.paths[index % len(self.paths)], map_location="cpu", weights_only=True)
        for key in ("token_ids", "top_idx", "top_logits", "logsumexp"):
            if key not in data:
                raise KeyError(f"teacher shard {self.paths[index]} missing required tensor {key!r}")
        return data

    def take(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids_parts, idx_parts, val_parts, lse_parts = [], [], [], []
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
            val_parts.append(self._cur["top_logits"][sl])
            lse_parts.append(self._cur["logsumexp"][sl])
            self._row_idx += take_n
            remaining -= take_n
        return (
            torch.cat(ids_parts, dim=0),
            torch.cat(idx_parts, dim=0),
            torch.cat(val_parts, dim=0),
            torch.cat(lse_parts, dim=0),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill Full-Stack QEC on a frozen quantized MoE with an 80/20 short/long curriculum."
    )
    parser.add_argument("--profile", required=True, help="ModelProfile JSON path")
    parser.add_argument("--student-model", required=True, help="Quantized student HF checkpoint path")
    parser.add_argument("--teacher-model", default="", help="Live teacher HF checkpoint path (optional if --teacher-shards is set)")
    parser.add_argument("--corpus", default="", help="Short-context pre-tokenized int32 [N, seq_len] .pt file (2K-4K)")
    parser.add_argument("--long-corpus", default="", help="Long-context pre-tokenized int32 [N, seq_len] .pt file (16K-32K+)")
    parser.add_argument("--teacher-shards", default="", help="Directory of pre-dumped short-context sparse top-K teacher shards")
    parser.add_argument("--long-teacher-shards", default="", help="Directory of pre-dumped long-context sparse top-K teacher shards")
    parser.add_argument("--long-fraction", type=float, default=0.20, help="Fraction of distillation tokens on long-context sequences (default 0.20 for 80/20 split)")
    parser.add_argument("--curriculum-mode", choices=("interleaved", "tail"), default="interleaved")
    parser.add_argument("--engram-layers", default="", help="Comma-separated layer indices for Engram QEC")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--buckets", type=int, default=262_139)
    parser.add_argument("--unigram-only", action="store_true")
    parser.add_argument("--tokens-m", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size for short-context steps")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps for short-context steps")
    parser.add_argument("--long-batch-size", type=int, default=1, help="Micro-batch size for long-context steps")
    parser.add_argument("--long-grad-accum", type=int, default=2, help="Gradient accumulation steps for long-context steps")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Enable gradient checkpointing on student backbone for long sequences")
    parser.add_argument("--topk", type=int, default=64)
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
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    short_shards = _ShardStream(args.teacher_shards) if use_shards else None
    long_shards = _ShardStream(args.long_teacher_shards) if args.long_teacher_shards else None
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
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device)
    if args.grad_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    engram_layers = [int(x) for x in args.engram_layers.split(",") if x.strip()]
    ek = (
        {"d_uni": 256, "n_bi_heads": 0, "n_tri_heads": 0}
        if args.unigram_only
        else {
            "d_uni": 128,
            "d_bi": 64,
            "n_bi_heads": 4,
            "bi_buckets": args.buckets,
            "d_tri": 64,
            "n_tri_heads": 2,
            "tri_buckets": args.buckets,
        }
    )
    qec = FullStackQEC(
        student,
        profile,
        engram_layers=engram_layers,
        lora_rank=args.lora_rank,
        engram_kwargs=ek,
        train_norms=True,
    ).to(device)

    model_run: torch.nn.Module = DDP(qec, device_ids=[local_rank]) if distributed else qec
    param_groups = qec.parameter_groups()
    opt = torch.optim.AdamW(param_groups)

    short_chunks_per_step = world_size * args.batch_size * args.grad_accum
    long_chunks_per_step = world_size * args.long_batch_size * args.long_grad_accum
    short_tokens_per_step = short_chunks_per_step * short_seq_len
    long_tokens_per_step = long_chunks_per_step * long_seq_len

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
    t0 = time.time()
    short_cursor = 0
    long_cursor = 0
    short_tokens_seen = 0
    long_tokens_seen = 0

    for step, phase in enumerate(schedule):
        is_long = phase == "long"
        micro_bs = args.long_batch_size if is_long else args.batch_size
        grad_accum = args.long_grad_accum if is_long else args.grad_accum
        chunks_this_step = world_size * micro_bs * grad_accum

        if use_shards:
            stream = long_shards if is_long else short_shards
            assert stream is not None
            step_ids, step_idx, step_vals, step_lse = stream.take(chunks_this_step)
        else:
            src_corpus = long_corpus if is_long else short_corpus
            assert src_corpus is not None
            cursor = long_cursor if is_long else short_cursor
            rows = [src_corpus[(cursor + i) % len(src_corpus)] for i in range(chunks_this_step)]
            step_ids = torch.stack(rows, dim=0)
            step_idx = step_vals = step_lse = None
            if is_long:
                long_cursor += chunks_this_step
            else:
                short_cursor += chunks_this_step

        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for micro in range(grad_accum):
            offset = (micro * world_size + rank) * micro_bs
            ids = step_ids[offset : offset + micro_bs].to(device=device, dtype=torch.long)
            if use_shards:
                assert step_idx is not None and step_vals is not None and step_lse is not None
                top_idx = step_idx[offset : offset + micro_bs].to(device=device, dtype=torch.long)
                top_vals = step_vals[offset : offset + micro_bs].to(device=device, dtype=torch.float32)
                t_lse = step_lse[offset : offset + micro_bs].to(device=device, dtype=torch.float32)
                top_logp = top_vals - t_lse.unsqueeze(-1)
            else:
                assert teacher is not None
                with torch.no_grad():
                    t_logits = teacher(input_ids=ids, use_cache=False).logits[:, :-1].float()
                    t_lse = torch.logsumexp(t_logits, dim=-1)
                    top_vals, top_idx = torch.topk(t_logits, k=args.topk, dim=-1)
                    top_logp = top_vals - t_lse.unsqueeze(-1)

            qec.cur_tokens = ids
            ctx = model_run.no_sync() if (distributed and micro + 1 < grad_accum) else torch.enable_grad()
            with ctx:
                s_logits = qec.model(input_ids=ids, use_cache=False).logits[:, :-1]
                loss = (
                    topk_kl_loss(
                        s_logits,
                        top_logp,
                        top_idx,
                        teacher_logsumexp=t_lse,
                        teacher_top_logits=top_vals,
                    )
                    / grad_accum
                )
                loss.backward()
                acc_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        if is_long:
            long_tokens_seen += long_tokens_per_step
        else:
            short_tokens_seen += short_tokens_per_step

        if rank == 0 and ((step + 1) % 25 == 0 or step + 1 == total_steps):
            seen = short_tokens_seen + long_tokens_seen
            print(
                f"step={step+1}/{total_steps} phase={phase} tokens={seen/1e6:.2f}M "
                f"(short={short_tokens_seen/1e6:.2f}M long={long_tokens_seen/1e6:.2f}M) "
                f"loss={acc_loss:.4f} t={time.time()-t0:.1f}s",
                flush=True,
            )

    if rank == 0:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = qec.merge_lora_into_model()
        torch.save(qec.engrams.state_dict(), out_dir / "engram_qec_sidecar.pt")
        norm_state = {
            k: v.detach().cpu()
            for k, v in qec.model.state_dict().items()
            if (
                ("norm" in k.lower() and "indexer" not in k.lower() and "router" not in k.lower())
                or any(p in k for p in profile.attn_projections)
            )
        }
        torch.save(norm_state, out_dir / "merged_non_expert_qec.pt")
        total_seen = short_tokens_seen + long_tokens_seen
        (out_dir / "qec_manifest.json").write_text(
            json.dumps(
                {
                    "profile": profile.name,
                    "tokens": total_seen,
                    "short_tokens": short_tokens_seen,
                    "long_tokens": long_tokens_seen,
                    "short_seq_len": short_seq_len,
                    "long_seq_len": long_seq_len if has_long else 0,
                    "long_fraction_target": effective_long_fraction,
                    "curriculum_mode": args.curriculum_mode,
                    "merged_lora_projections": merged,
                    "engram_layers": list(engram_layers),
                    "unigram_only": args.unigram_only,
                    "buckets": args.buckets,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved QEC artifacts to {out_dir}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
