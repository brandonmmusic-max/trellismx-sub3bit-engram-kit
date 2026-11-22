#!/usr/bin/env python3
"""Forward-only BF16 teacher logit dumper for dense evaluation and sparse QEC distillation.

Designed to run once on a high-VRAM node (for example, 8x H200 or 8x B200) and
emit two compact artifacts for downstream local quantization and QEC training:

1. Dense full-vocabulary evaluation logits (``eval_dense/window-XXXX.pt``) for
   exact FP64 KL divergence scoring across both 8K and 16K-64K untouched
   evaluation windows.
2. Sparse top-K distillation shards (``train_short/`` for 80% 2K-4K windows and
   ``train_long/`` for 20% 16K-32K+ windows), each storing ``token_ids``
   (int32), ``top_idx`` (int32), ``top_logits`` (fp16), and exact
   full-vocabulary ``logsumexp`` (fp32) so local student distillation includes
   exact residual tail probability mass.
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
    parser.add_argument("--eval-tokens", default="", help="Optional int32 [N, seq_len] .pt file for dense eval logits")
    parser.add_argument("--eval-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--short-tokens", default="", help="Optional int32 [N, seq_len] .pt file for 80% short distillation split")
    parser.add_argument("--long-tokens", default="", help="Optional int32 [N, seq_len] .pt file for 20% long distillation split")
    parser.add_argument("--topk", type=int, default=64, help="Top-K logits saved per position (64 -> 388 B/tok, 128 -> 772 B/tok)")
    parser.add_argument("--short-batch-size", type=int, default=8)
    parser.add_argument("--long-batch-size", type=int, default=1)
    parser.add_argument("--rows-per-shard", type=int, default=1024)
    parser.add_argument("--long-rows-per-shard", type=int, default=128)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _input_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def dump_dense_eval(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    out_dir: Path,
    dtype: torch.dtype,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _input_device(model)
    total_positions = 0
    for idx in range(int(tokens.shape[0])):
        ids = tokens[idx : idx + 1].to(device=dev, dtype=torch.long)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].to(dtype=dtype).cpu()
        torch.save(
            {
                "token_ids": tokens[idx].to(torch.int32).contiguous(),
                "logits": logits.contiguous(),
            },
            out_dir / f"window-{idx:04d}.pt",
        )
        total_positions += int(logits.shape[0])
    return {"windows": int(tokens.shape[0]), "positions": total_positions, "dtype": str(dtype)}


@torch.no_grad()
def dump_sparse_split(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    out_dir: Path,
    *,
    topk: int,
    batch_size: int,
    rows_per_shard: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _input_device(model)
    n_rows = int(tokens.shape[0])
    seq_len = int(tokens.shape[1])
    shard_id = 0
    t0 = time.time()

    for shard_start in range(0, n_rows, rows_per_shard):
        shard_end = min(n_rows, shard_start + rows_per_shard)
        ids_buf, idx_buf, val_buf, lse_buf = [], [], [], []
        for b0 in range(shard_start, shard_end, batch_size):
            b1 = min(shard_end, b0 + batch_size)
            ids = tokens[b0:b1].to(device=dev, dtype=torch.long)
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
            lse = torch.logsumexp(logits, dim=-1)
            top_vals, top_idx = torch.topk(logits, k=topk, dim=-1)
            ids_buf.append(tokens[b0:b1].to(torch.int32).cpu())
            idx_buf.append(top_idx.to(torch.int32).cpu())
            val_buf.append(top_vals.to(torch.float16).cpu())
            lse_buf.append(lse.to(torch.float32).cpu())

        torch.save(
            {
                "token_ids": torch.cat(ids_buf, dim=0).contiguous(),
                "top_idx": torch.cat(idx_buf, dim=0).contiguous(),
                "top_logits": torch.cat(val_buf, dim=0).contiguous(),
                "logsumexp": torch.cat(lse_buf, dim=0).contiguous(),
            },
            out_dir / f"shard-{shard_id:04d}.pt",
        )
        shard_id += 1
        done_tokens = shard_end * seq_len
        rate = done_tokens / max(1e-6, time.time() - t0)
        print(
            f"[{out_dir.name}] shard={shard_id} rows={shard_end}/{n_rows} "
            f"tokens={done_tokens/1e6:.2f}M rate={rate:.0f} tok/s",
            flush=True,
        )

    return {
        "shards": shard_id,
        "rows": n_rows,
        "seq_len": seq_len,
        "tokens": n_rows * seq_len,
        "topk": topk,
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    manifest: dict[str, object] = {
        "teacher_model": args.teacher_model,
        "topk": args.topk,
    }
    if args.eval_tokens:
        eval_toks = torch.load(args.eval_tokens, map_location="cpu", weights_only=True)
        eval_dtype = torch.bfloat16 if args.eval_dtype == "bfloat16" else torch.float32
        manifest["eval_dense"] = dump_dense_eval(model, eval_toks, out_root / "eval_dense", eval_dtype)
    if args.short_tokens:
        short_toks = torch.load(args.short_tokens, map_location="cpu", weights_only=True)
        manifest["train_short"] = dump_sparse_split(
            model,
            short_toks,
            out_root / "train_short",
            topk=args.topk,
            batch_size=args.short_batch_size,
            rows_per_shard=args.rows_per_shard,
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
        )

    (out_root / "teacher_logits_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote teacher logits manifest to {out_root / 'teacher_logits_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
