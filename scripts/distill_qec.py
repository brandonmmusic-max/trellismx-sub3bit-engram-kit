#!/usr/bin/env python3
"""Model-neutral DDP Full-Stack QEC distillation CLI with 80/20 context curriculum.

Freezes all routed MoE expert weights, routers, and sparse attention indexers of
a quantized Hugging Face checkpoint and distills mergeable non-expert LoRA
(attention, optional dense prefix MLP, optional shared experts), FP32-promoted
text-backbone RMSNorm scales, and Dual-Trellis Host-RAM Engram QEC tables
against either a live teacher or pre-dumped sparse top-K teacher shards.
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
                # Backward compatibility with raw top_logits + logsumexp shards
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
    parser.add_argument("--long-fraction", type=float, default=0.20, help="Fraction of distillation tokens on long-context sequences")
    parser.add_argument("--curriculum-mode", choices=("interleaved", "tail"), default="interleaved")
    parser.add_argument("--engram-layers", default="", help="Comma-separated layer indices for Engram QEC")
    parser.add_argument("--engram-table-device", choices=("auto", "cpu"), default="auto", help="Pin Engram tables in host RAM ('cpu') for large MoE models")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--train-dense-lora", action="store_true", help="Also attach mergeable LoRA to dense prefix MLP layers")
    parser.add_argument("--train-shared-lora", action="store_true", help="Also attach mergeable LoRA to BF16/FP8 shared experts")
    parser.add_argument("--buckets", type=int, default=262_139)
    parser.add_argument("--skip-heads", type=int, default=2)
    parser.add_argument("--unigram-only", action="store_true")
    parser.add_argument("--tokens-m", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--long-batch-size", type=int, default=1)
    parser.add_argument("--long-grad-accum", type=int, default=2)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--loss-chunk-tokens", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=0, help="Save resumable checkpoint every N optimizer steps (0 disables)")
    parser.add_argument("--resume", default="", help="Optional path to a saved checkpoint .pt to resume from")
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
            args.teacher_model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        ).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    if args.grad_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

    engram_layers = [int(x) for x in args.engram_layers.split(",") if x.strip()]
    table_dev = torch.device("cpu") if args.engram_table_device == "cpu" else None
    ek = (
        {"d_uni": 256, "n_bi_heads": 0, "n_tri_heads": 0, "table_device": table_dev}
        if args.unigram_only
        else {
            "d_uni": 128,
            "d_bi": 64,
            "n_bi_heads": 4,
            "bi_buckets": args.buckets,
            "d_tri": 64,
            "n_tri_heads": 2,
            "tri_buckets": args.buckets,
            "d_skip": 64 if args.skip_heads > 0 else 0,
            "n_skip_heads": args.skip_heads,
            "skip_buckets": args.buckets,
            "table_device": table_dev,
        }
    )
    qec = FullStackQEC(
        student,
        profile,
        engram_layers=engram_layers,
        lora_rank=args.lora_rank,
        engram_kwargs=ek,
        train_norms=True,
        train_dense_lora=args.train_dense_lora,
        train_shared_lora=args.train_shared_lora,
    ).to(device)
    qec.train()

    # Use DDP when all trainable parameters live on GPU; CPU host-RAM table mode uses manual sync or single-rank TP
    use_ddp = distributed and table_dev is None
    model_run: torch.nn.Module = DDP(qec, device_ids=[local_rank]) if use_ddp else qec
    param_groups = qec.parameter_groups()
    opt = torch.optim.AdamW(param_groups)

    short_chunks_per_step = world_size * args.batch_size * args.grad_accum
    long_chunks_per_step = world_size * args.long_batch_size * args.long_grad_accum
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
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        qec.engrams.load_state_dict(ckpt["engrams"])
        qec.loras.load_state_dict(ckpt["loras"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"])
        short_cursor = int(ckpt["short_cursor"])
        long_cursor = int(ckpt["long_cursor"])
        short_tokens_seen = int(ckpt["short_tokens_seen"])
        long_tokens_seen = int(ckpt["long_tokens_seen"])

    t0 = time.time()
    out_dir = Path(args.output)
    for step in range(start_step, total_steps):
        phase = schedule[step]
        is_long = phase == "long"
        micro_bs = args.long_batch_size if is_long else args.batch_size
        grad_accum = args.long_grad_accum if is_long else args.grad_accum
        chunks_this_step = world_size * micro_bs * grad_accum

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
            offset = (micro * world_size + rank) * micro_bs
            ids = step_ids[offset : offset + micro_bs].to(device=device, dtype=torch.long)
            if use_shards:
                assert step_idx is not None and step_logp is not None
                top_idx = step_idx[offset : offset + micro_bs].to(device=device, dtype=torch.long)
                top_logp = step_logp[offset : offset + micro_bs].to(device=device, dtype=torch.float32)
                p_tail = (
                    step_ptail[offset : offset + micro_bs].to(device=device, dtype=torch.float32)
                    if step_ptail is not None
                    else None
                )
            else:
                assert teacher is not None
                with torch.no_grad():
                    t_logits = teacher(input_ids=ids, use_cache=False).logits[:, :-1].float()
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
            long_tokens_seen += long_tokens_per_step
        else:
            short_tokens_seen += short_tokens_per_step

        if rank == 0 and ((step + 1) % 25 == 0 or step + 1 == total_steps):
            seen = short_tokens_seen + long_tokens_seen
            print(
                f"step={step+1}/{total_steps} phase={phase} positions={seen/1e6:.2f}M "
                f"(short={short_tokens_seen/1e6:.2f}M long={long_tokens_seen/1e6:.2f}M) "
                f"loss={acc_loss:.4f} t={time.time()-t0:.1f}s",
                flush=True,
            )

        if rank == 0 and args.save_every > 0 and (step + 1) % args.save_every == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step + 1,
                    "engrams": qec.engrams.state_dict(),
                    "loras": qec.loras.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "short_cursor": short_cursor,
                    "long_cursor": long_cursor,
                    "short_tokens_seen": short_tokens_seen,
                    "long_tokens_seen": long_tokens_seen,
                },
                out_dir / "checkpoint_latest.pt",
            )

    if distributed:
        dist.barrier()

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = qec.merge_lora_into_model()
        torch.save(qec.engrams.state_dict(), out_dir / "engram_qec_sidecar.pt")
        torch.save(qec.export_non_expert_state(), out_dir / "merged_non_expert_qec.pt")
        total_seen = short_tokens_seen + long_tokens_seen
        (out_dir / "qec_manifest.json").write_text(
            json.dumps(
                {
                    "profile": profile.name,
                    "supervised_positions": total_seen,
                    "short_positions": short_tokens_seen,
                    "long_positions": long_tokens_seen,
                    "short_seq_len": short_seq_len,
                    "long_seq_len": long_seq_len if has_long else 0,
                    "long_fraction_target": effective_long_fraction,
                    "curriculum_mode": args.curriculum_mode,
                    "merged_lora_projections": merged,
                    "engram_layers": list(engram_layers),
                    "unigram_only": args.unigram_only,
                    "buckets": args.buckets,
                    "skip_heads": args.skip_heads,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved QEC artifacts to {out_dir}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
