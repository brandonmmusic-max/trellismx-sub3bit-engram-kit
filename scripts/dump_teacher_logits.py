#!/usr/bin/env python3
"""Forward-only BF16 teacher logit dumper for dense evaluation and sparse QEC distillation.

Designed to run once on a high-VRAM node (for example, 8x H200 or 8x B200) and
emit two compact artifacts for downstream local quantization and QEC training:

1. Dense full-vocabulary evaluation logits (``eval_dense/<split>-window-XXXX.pt``)
   for exact FP64 KL divergence scoring across both 8K and 16K-64K untouched
   evaluation windows.
2. Sparse top-K distillation shards (``train_short/`` for 80% 2K-4K windows and
   ``train_long/`` for 20% 16K-32K+ windows), each storing ``token_ids``
   (int32), ``top_idx`` (int32), shifted ``top_logp`` (fp16, ~4000x more
   accurate near 0.0 than raw fp16 logits), exact ``p_tail`` (fp32), and
   ``logsumexp`` (fp32). Total size is 396 bytes/token at top-64 (~19.8 GB for
   50M tokens) or 780 bytes/token at top-128 (~39.0 GB for 50M tokens).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump dense eval and sparse top-K teacher logits.")
    parser.add_argument("--teacher-model", required=True, help="BF16 teacher HF checkpoint path or repo ID")
    parser.add_argument(
        "--eval-tokens",
        nargs="*",
        default=[],
        help="One or more int32 [N, seq_len] .pt files for dense eval logits (e.g. 8K and 32K/64K splits)",
    )
    parser.add_argument("--eval-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--short-tokens", default="", help="Optional int32 [N, seq_len] .pt file for 80% short distillation split")
    parser.add_argument("--long-tokens", default="", help="Optional int32 [N, seq_len] .pt file for 20% long distillation split")
    parser.add_argument("--topk", type=int, default=64, help="Top-K logits saved per position (64 -> 396 B/tok, 128 -> 780 B/tok)")
    parser.add_argument("--short-batch-size", type=int, default=8)
    parser.add_argument("--long-batch-size", type=int, default=1)
    parser.add_argument("--logit-chunk-tokens", type=int, default=2048, help="Sequence chunk size for FP32 logsumexp/topk to prevent 64K OOM")
    parser.add_argument("--rows-per-shard", type=int, default=1024)
    parser.add_argument("--long-rows-per-shard", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--resume", action="store_true", help="Skip already-written output shards")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight.device.type != "meta":
            return emb.weight.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def dump_dense_eval(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    out_dir: Path,
    split_name: str,
    dtype: torch.dtype,
    *,
    resume: bool = False,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _input_device(model)
    total_positions = 0
    for idx in range(int(tokens.shape[0])):
        out_path = out_dir / f"{split_name}-window-{idx:04d}.pt"
        pos = int(tokens.shape[1]) - 1
        if resume and out_path.exists():
            total_positions += pos
            continue
        ids = tokens[idx : idx + 1].to(device=dev, dtype=torch.long)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].to(dtype=dtype).cpu()
        torch.save(
            {
                "token_ids": tokens[idx].to(torch.int32).contiguous(),
                "logits": logits.contiguous(),
            },
            out_path,
        )
        total_positions += int(logits.shape[0])
    return {
        "split": split_name,
        "windows": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "positions": total_positions,
        "dtype": str(dtype),
    }


@torch.no_grad()
def dump_sparse_split(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    out_dir: Path,
    *,
    topk: int,
    batch_size: int,
    rows_per_shard: int,
    logit_chunk_tokens: int = 2048,
    resume: bool = False,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _input_device(model)
    n_rows = int(tokens.shape[0])
    seq_len = int(tokens.shape[1])
    pos_per_row = seq_len - 1
    shard_id = 0
    t0 = time.time()

    for shard_start in range(0, n_rows, rows_per_shard):
        shard_end = min(n_rows, shard_start + rows_per_shard)
        out_path = out_dir / f"shard-{shard_id:04d}.pt"
        if resume and out_path.exists():
            shard_id += 1
            continue

        ids_buf, idx_buf, logp_buf, ptail_buf, lse_buf = [], [], [], [], []
        for b0 in range(shard_start, shard_end, batch_size):
            b1 = min(shard_end, b0 + batch_size)
            ids = tokens[b0:b1].to(device=dev, dtype=torch.long)
            raw_logits = model(input_ids=ids, use_cache=False).logits[:, :-1]

            # Chunk FP32 logsumexp/topk along sequence length so 64K x 155K never OOMs
            c_idx, c_logp, c_ptail, c_lse = [], [], [], []
            for s0 in range(0, pos_per_row, logit_chunk_tokens):
                s1 = min(pos_per_row, s0 + logit_chunk_tokens)
                chunk_f = raw_logits[:, s0:s1].float()
                lse = torch.logsumexp(chunk_f, dim=-1)
                top_vals, top_idx = torch.topk(chunk_f, k=topk, dim=-1)
                top_logp = (top_vals - lse.unsqueeze(-1)).clamp_max(0.0)
                p_tail = (1.0 - top_logp.exp().sum(dim=-1)).clamp(0.0, 1.0)
                c_idx.append(top_idx.to(torch.int32).cpu())
                c_logp.append(top_logp.to(torch.float16).cpu())
                c_ptail.append(p_tail.to(torch.float32).cpu())
                c_lse.append(lse.to(torch.float32).cpu())

            ids_buf.append(tokens[b0:b1].to(torch.int32).cpu())
            idx_buf.append(torch.cat(c_idx, dim=1))
            logp_buf.append(torch.cat(c_logp, dim=1))
            ptail_buf.append(torch.cat(c_ptail, dim=1))
            lse_buf.append(torch.cat(c_lse, dim=1))

        torch.save(
            {
                "token_ids": torch.cat(ids_buf, dim=0).contiguous(),
                "top_idx": torch.cat(idx_buf, dim=0).contiguous(),
                "top_logp": torch.cat(logp_buf, dim=0).contiguous(),
                "p_tail": torch.cat(ptail_buf, dim=0).contiguous(),
                "logsumexp": torch.cat(lse_buf, dim=0).contiguous(),
            },
            out_path,
        )
        shard_id += 1
        done_positions = shard_end * pos_per_row
        rate = done_positions / max(1e-6, time.time() - t0)
        print(
            f"[{out_dir.name}] shard={shard_id} rows={shard_end}/{n_rows} "
            f"positions={done_positions/1e6:.2f}M rate={rate:.0f} pos/s",
            flush=True,
        )

    return {
        "shards": shard_id,
        "rows": n_rows,
        "seq_len": seq_len,
        "positions": n_rows * pos_per_row,
        "topk": topk,
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **load_kwargs).eval()

    manifest: dict[str, object] = {
        "teacher_model": args.teacher_model,
        "topk": args.topk,
    }
    if args.eval_tokens:
        eval_dtype = torch.bfloat16 if args.eval_dtype == "bfloat16" else torch.float32
        eval_summaries = []
        for path_str in args.eval_tokens:
            p = Path(path_str)
            eval_toks = torch.load(p, map_location="cpu", weights_only=True)
            eval_summaries.append(
                dump_dense_eval(
                    model,
                    eval_toks,
                    out_root / "eval_dense",
                    p.stem,
                    eval_dtype,
                    resume=args.resume,
                )
            )
        manifest["eval_dense"] = eval_summaries
    if args.short_tokens:
        short_toks = torch.load(args.short_tokens, map_location="cpu", weights_only=True)
        manifest["train_short"] = dump_sparse_split(
            model,
            short_toks,
            out_root / "train_short",
            topk=args.topk,
            batch_size=args.short_batch_size,
            rows_per_shard=args.rows_per_shard,
            logit_chunk_tokens=args.logit_chunk_tokens,
            resume=args.resume,
        )
    if args.long_tokens:
        long_toks = torch.load(args.long_tokens, map_location="cpu", weights_only=True)
        manifest["train_long"] = dump_sparse_split(
            model,
            long_toks,
            out_root / "train_long",
            topk=args.topk,
            batch_size=args.long_batch_size,
            rows_per_shard=args.long_rows_per_shard,
            logit_chunk_tokens=args.logit_chunk_tokens,
            resume=args.resume,
        )

    (out_root / "teacher_logits_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote teacher logits manifest to {out_root / 'teacher_logits_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
