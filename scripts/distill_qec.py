#!/usr/bin/env python3
"""Model-neutral DDP Full-Stack QEC distillation CLI.

Freezes all routed MoE expert weights of a quantized Hugging Face checkpoint
and distills mergeable Attention LoRA, RMSNorm scales, and Host-RAM Engram QEC
tables against a teacher model (live or pre-cached top-K logprobs).

Usage (4-GPU DDP):
    torchrun --standalone --nproc_per_node=4 scripts/distill_qec.py \
        --profile configs/glm-5.3-flash.json \
        --teacher-model /path/to/teacher \
        --student-model /path/to/quantized-student \
        --corpus /path/to/tokens_2048.pt \
        --engram-layers 3,8,14,20,26,32,38,44 \
        --tokens-m 50 --output /path/to/qec-out
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

from trellismx_kit.engram_qec import FullStackQEC, topk_kl_loss
from trellismx_kit.profile import load_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill Full-Stack QEC on a frozen quantized MoE.")
    parser.add_argument("--profile", required=True, help="ModelProfile JSON path")
    parser.add_argument("--teacher-model", required=True, help="Teacher HF checkpoint path")
    parser.add_argument("--student-model", required=True, help="Quantized student HF checkpoint path")
    parser.add_argument("--corpus", required=True, help="Pre-tokenized int32 [N, seq_len] .pt file")
    parser.add_argument("--engram-layers", default="", help="Comma-separated layer indices for Engram QEC")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--buckets", type=int, default=262_139)
    parser.add_argument("--unigram-only", action="store_true")
    parser.add_argument("--tokens-m", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)

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

    corpus = torch.load(args.corpus, map_location="cpu", weights_only=True)
    seq_len = int(corpus.shape[1])

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device)

    engram_layers = [int(x) for x in args.engram_layers.split(",") if x.strip()]
    ek = (
        {"d_uni": 256, "n_bi_heads": 0, "n_tri_heads": 0}
        if args.unigram_only
        else {"d_uni": 128, "d_bi": 64, "n_bi_heads": 4, "bi_buckets": args.buckets, "d_tri": 64, "n_tri_heads": 2, "tri_buckets": args.buckets}
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

    chunks_per_step = world_size * args.batch_size * args.grad_accum
    tokens_per_step = chunks_per_step * seq_len
    total_steps = min(len(corpus) // chunks_per_step, max(1, int(round(args.tokens_m * 1e6)) // tokens_per_step))
    warmup = max(20, total_steps // 20)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    t0 = time.time()

    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        base = step * chunks_per_step
        acc_loss = 0.0
        for micro in range(args.grad_accum):
            start = base + (micro * world_size + rank) * args.batch_size
            ids = corpus[start : start + args.batch_size].to(device=device, dtype=torch.long)
            with torch.no_grad():
                t_logits = teacher(input_ids=ids, use_cache=False).logits[:, :-1].float()
                t_logp = F.log_softmax(t_logits, dim=-1)
                top_logp, top_idx = torch.topk(t_logp, k=args.topk, dim=-1)

            qec.cur_tokens = ids
            ctx = model_run.no_sync() if (distributed and micro + 1 < args.grad_accum) else torch.enable_grad()
            with ctx:
                s_logits = qec.model(input_ids=ids, use_cache=False).logits[:, :-1]
                loss = topk_kl_loss(s_logits, top_logp, top_idx) / args.grad_accum
                loss.backward()
                acc_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        if rank == 0 and ((step + 1) % 25 == 0 or step + 1 == total_steps):
            seen = (step + 1) * tokens_per_step
            print(f"step={step+1}/{total_steps} tokens={seen/1e6:.2f}M loss={acc_loss:.4f} t={time.time()-t0:.1f}s", flush=True)

    if rank == 0:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = qec.merge_lora_into_model()
        torch.save(qec.engrams.state_dict(), out_dir / "engram_qec_sidecar.pt")
        norm_state = {
            k: v.detach().cpu()
            for k, v in qec.model.state_dict().items()
            if "norm" in k.lower() or any(p in k for p in profile.attn_projections)
        }
        torch.save(norm_state, out_dir / "merged_non_expert_qec.pt")
        (out_dir / "qec_manifest.json").write_text(
            json.dumps(
                {
                    "profile": profile.name,
                    "tokens": total_steps * tokens_per_step,
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
