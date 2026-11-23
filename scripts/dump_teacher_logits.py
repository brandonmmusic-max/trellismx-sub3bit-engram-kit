#!/usr/bin/env python3
"""Forward-only BF16 teacher logit dumper for dense evaluation and sparse QEC distillation.

Supports both standard CausalLM checkpoints and multimodal conditional-generation
architectures (such as Glm5NextForConditionalGeneration) via AutoConfig detection,
and runs under either ``device_map='auto'`` or ``torchrun`` Expert/Tensor Parallel
(``--distributed-ep``) with optional ``--shard-rank/--shard-world`` window
partitioning across independent multi-GPU workers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from trellismx_kit.glm5_patch import patch_glm5_long_context

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump dense eval and sparse top-K teacher logits.")
    parser.add_argument("--teacher-model", required=True, help="BF16 teacher HF checkpoint path or repo ID")
    parser.add_argument(
        "--eval-tokens",
        nargs="*",
        default=[],
        help="One or more int32 [N, seq_len] .pt files for dense eval logits (e.g. 8K and 32K/64K splits)",
    )
    parser.add_argument("--eval-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--vocab-limit", type=int, default=0, help="Optional vocabulary slice limit (e.g. 154880)")
    parser.add_argument("--short-tokens", default="", help="Optional int32 [N, seq_len] .pt file for 80% short distillation split")
    parser.add_argument(
        "--long-tokens",
        nargs="*",
        default=[],
        help="One or more int32 [N, seq_len] .pt files for 20% long distillation splits (e.g. 16K, 32K, 64K)",
    )
    parser.add_argument("--topk", type=int, default=128, help="Top-K logits saved per position (64 -> 396 B/tok, 128 -> 780 B/tok)")
    parser.add_argument("--short-batch-size", type=int, default=8)
    parser.add_argument("--long-batch-size", type=int, default=1)
    parser.add_argument("--logit-chunk-tokens", type=int, default=2048, help="Sequence chunk size for FP32 logsumexp/topk")
    parser.add_argument("--rows-per-shard", type=int, default=512)
    parser.add_argument("--long-rows-per-shard", type=int, default=64)
    parser.add_argument("--distributed-ep", action="store_true", help="Load with Transformers DistributedConfig Expert Parallel")
    parser.add_argument(
        "--ep-implementation",
        choices=("all_to_all", "all_reduce"),
        default="all_to_all",
        help="EP token dispatch mode under --distributed-ep ('all_to_all' runs 8x DP on non-expert layers)",
    )
    parser.add_argument("--shard-rank", type=int, default=0, help="Independent worker shard index for partitioning shards across jobs")
    parser.add_argument("--shard-world", type=int, default=1, help="Total independent worker count for shard partitioning")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--experts-implementation", default="", help="Optional HF experts_implementation (e.g. grouped_mm)")
    parser.add_argument("--resume", action="store_true", help="Skip already-written output shards")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_hf_model(
    model_path: str,
    *,
    distributed_ep: bool,
    world_size: int,
    trust_remote_code: bool,
    attn_implementation: str = "",
    experts_implementation: str = "",
    ep_implementation: str = "all_to_all",
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
    if experts_implementation:
        kwargs["experts_implementation"] = experts_implementation

    if distributed_ep and world_size > 1:
        from transformers.distributed.configuration_utils import DistributedConfig

        # Keep pure EP on experts only; avoids upstream hybrid linear_attention o_proj TP mismatch
        # and avoids FSDP sharding overhead on replicated non-expert layers under all_to_all DP+EP.
        cfg.base_model_tp_plan = None
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg.text_config.base_model_tp_plan = None
        if ep_implementation == "all_to_all":
            cfg.base_model_fsdp_plan = None
            if hasattr(cfg, "text_config") and cfg.text_config is not None:
                cfg.text_config.base_model_fsdp_plan = None
        kwargs["config"] = cfg
        try:
            if ep_implementation == "all_to_all":
                kwargs["distributed_config"] = DistributedConfig(
                    ep_size=world_size,
                    tp_size=1,
                    fsdp_size=world_size,
                    fsdp_plan={},
                    pp_size=1,
                    ep_implementation="all_to_all",
                )
            else:
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
    else:
        kwargs["device_map"] = "auto"

    model = loader.from_pretrained(model_path, **kwargs).eval()
    patch_glm5_long_context(model)
    return model


def _input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight.device.type != "meta":
            w = emb.weight.to_local() if hasattr(emb.weight, "to_local") else emb.weight
            return w.device
    for param in model.parameters():
        if param.device.type != "meta":
            local = param.to_local() if hasattr(param, "to_local") else param
            return local.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def dump_dense_eval(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    out_dir: Path,
    split_name: str,
    dtype: torch.dtype,
    *,
    vocab_limit: int = 0,
    is_writer: bool = True,
    resume: bool = False,
) -> dict[str, object]:
    if is_writer:
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
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1]
        if vocab_limit > 0:
            logits = logits[:, :vocab_limit]
        if is_writer:
            tmp_path = out_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "token_ids": tokens[idx].to(torch.int32).contiguous(),
                    "logits": logits.cpu().to(dtype=dtype).contiguous(),
                },
                tmp_path,
            )
            os.replace(tmp_path, out_path)
        total_positions += pos
    return {
        "split": split_name,
        "windows": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "positions": total_positions,
        "dtype": str(dtype),
    }


@torch.no_grad()
def _extract_sparse_topk(
    raw_logits: torch.Tensor,
    *,
    topk: int,
    logit_chunk_tokens: int,
    to_cpu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_per_row = int(raw_logits.shape[1])
    c_idx, c_logp, c_ptail, c_lse = [], [], [], []
    for s0 in range(0, pos_per_row, logit_chunk_tokens):
        s1 = min(pos_per_row, s0 + logit_chunk_tokens)
        chunk_f = raw_logits[:, s0:s1].float()
        lse = torch.logsumexp(chunk_f, dim=-1)
        top_vals, top_idx = torch.topk(chunk_f, k=topk, dim=-1)
        top_logp = (top_vals - lse.unsqueeze(-1)).clamp_max(0.0)
        p_tail = (1.0 - top_logp.exp().sum(dim=-1)).clamp(0.0, 1.0)
        if to_cpu:
            c_idx.append(top_idx.to(torch.int32).cpu())
            c_logp.append(top_logp.to(torch.float16).cpu())
            c_ptail.append(p_tail.to(torch.float32).cpu())
            c_lse.append(lse.to(torch.float32).cpu())
        else:
            c_idx.append(top_idx.to(torch.int32))
            c_logp.append(top_logp.to(torch.float16))
            c_ptail.append(p_tail.to(torch.float32))
            c_lse.append(lse.to(torch.float32))
    return (
        torch.cat(c_idx, dim=1).contiguous(),
        torch.cat(c_logp, dim=1).contiguous(),
        torch.cat(c_ptail, dim=1).contiguous(),
        torch.cat(c_lse, dim=1).contiguous(),
    )


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
    vocab_limit: int = 0,
    is_writer: bool = True,
    shard_rank: int = 0,
    shard_world: int = 1,
    dp_rank: int = 0,
    dp_world: int = 1,
    resume: bool = False,
) -> dict[str, object]:
    if is_writer:
        out_dir.mkdir(parents=True, exist_ok=True)
    dev = _input_device(model)
    n_rows = int(tokens.shape[0])
    seq_len = int(tokens.shape[1])
    pos_per_row = seq_len - 1
    shard_id = 0
    new_positions = 0
    t0 = time.time()
    use_dp_gather = dp_world > 1 and dist.is_initialized()

    for shard_start in range(0, n_rows, rows_per_shard):
        shard_end = min(n_rows, shard_start + rows_per_shard)
        cur_shard = shard_id
        shard_id += 1
        if cur_shard % shard_world != shard_rank:
            continue
        out_path = out_dir / f"shard-{cur_shard:04d}.pt"
        if resume and out_path.exists():
            continue

        ids_buf, idx_buf, logp_buf, ptail_buf, lse_buf = [], [], [], [], []
        global_bs = batch_size * dp_world if use_dp_gather else batch_size
        for b0 in range(shard_start, shard_end, global_bs):
            b1 = min(shard_end, b0 + global_bs)
            n_real = b1 - b0
            if use_dp_gather:
                per_rank_bs = (n_real + dp_world - 1) // dp_world
                row_indices = [
                    min(b1 - 1, b0 + dp_rank * per_rank_bs + i)
                    for i in range(per_rank_bs)
                ]
                ids = tokens[row_indices].to(device=dev, dtype=torch.long)
            else:
                ids = tokens[b0:b1].to(device=dev, dtype=torch.long)

            raw_logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
            if vocab_limit > 0:
                raw_logits = raw_logits[:, :, :vocab_limit]

            if use_dp_gather:
                loc_idx, loc_logp, loc_ptail, loc_lse = _extract_sparse_topk(
                    raw_logits, topk=topk, logit_chunk_tokens=logit_chunk_tokens, to_cpu=False
                )
                del raw_logits
                g_idx = [torch.empty_like(loc_idx) for _ in range(dp_world)] if is_writer else None
                g_logp = [torch.empty_like(loc_logp) for _ in range(dp_world)] if is_writer else None
                g_ptail = [torch.empty_like(loc_ptail) for _ in range(dp_world)] if is_writer else None
                g_lse = [torch.empty_like(loc_lse) for _ in range(dp_world)] if is_writer else None
                dist.gather(loc_idx, g_idx, dst=0)
                dist.gather(loc_logp, g_logp, dst=0)
                dist.gather(loc_ptail, g_ptail, dst=0)
                dist.gather(loc_lse, g_lse, dst=0)
                del loc_idx, loc_logp, loc_ptail, loc_lse
                if is_writer:
                    assert g_idx is not None and g_logp is not None and g_ptail is not None and g_lse is not None
                    ids_buf.append(tokens[b0:b1].to(torch.int32).cpu())
                    idx_buf.append(torch.cat(g_idx, dim=0)[:n_real].cpu())
                    logp_buf.append(torch.cat(g_logp, dim=0)[:n_real].cpu())
                    ptail_buf.append(torch.cat(g_ptail, dim=0)[:n_real].cpu())
                    lse_buf.append(torch.cat(g_lse, dim=0)[:n_real].cpu())
                    del g_idx, g_logp, g_ptail, g_lse
            elif is_writer:
                loc_idx, loc_logp, loc_ptail, loc_lse = _extract_sparse_topk(
                    raw_logits, topk=topk, logit_chunk_tokens=logit_chunk_tokens, to_cpu=True
                )
                del raw_logits
                ids_buf.append(tokens[b0:b1].to(torch.int32).cpu())
                idx_buf.append(loc_idx)
                logp_buf.append(loc_logp)
                ptail_buf.append(loc_ptail)
                lse_buf.append(loc_lse)

        if is_writer:
            tmp_path = out_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "token_ids": torch.cat(ids_buf, dim=0).contiguous(),
                    "top_idx": torch.cat(idx_buf, dim=0).contiguous(),
                    "top_logp": torch.cat(logp_buf, dim=0).contiguous(),
                    "p_tail": torch.cat(ptail_buf, dim=0).contiguous(),
                    "logsumexp": torch.cat(lse_buf, dim=0).contiguous(),
                },
                tmp_path,
            )
            os.replace(tmp_path, out_path)
            new_positions += (shard_end - shard_start) * pos_per_row
            rate = new_positions / max(1e-6, time.time() - t0)
            print(
                f"[{out_dir.name}] shard={cur_shard}/{shard_id} worker={shard_rank}/{shard_world} "
                f"rows={shard_end}/{n_rows} rate={rate:.0f} pos/s",
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out_root = Path(args.output)

    distributed = args.distributed_ep and ("RANK" in os.environ)
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1

    is_writer = rank == 0
    if is_writer:
        out_root.mkdir(parents=True, exist_ok=True)

    model = load_hf_model(
        args.teacher_model,
        distributed_ep=args.distributed_ep,
        world_size=world_size,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        experts_implementation=args.experts_implementation,
        ep_implementation=args.ep_implementation,
    )
    dp_world = world_size if (distributed and args.ep_implementation == "all_to_all") else 1
    dp_rank = rank if dp_world > 1 else 0

    manifest: dict[str, object] = {
        "teacher_model": args.teacher_model,
        "topk": args.topk,
        "distributed_ep": bool(distributed),
        "ep_implementation": args.ep_implementation if distributed else "none",
        "world_size": world_size,
    }
    if args.eval_tokens and args.shard_rank == 0:
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
                    vocab_limit=args.vocab_limit,
                    is_writer=is_writer,
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
            vocab_limit=args.vocab_limit,
            is_writer=is_writer,
            shard_rank=args.shard_rank,
            shard_world=args.shard_world,
            dp_rank=dp_rank,
            dp_world=dp_world,
            resume=args.resume,
        )
    if args.long_tokens:
        long_paths = [Path(s) for s in args.long_tokens if s]
        long_summaries = {}
        for p in long_paths:
            long_toks = torch.load(p, map_location="cpu", weights_only=True)
            out_sub = out_root / ("train_long" if len(long_paths) == 1 else p.stem)
            long_summaries[p.stem] = dump_sparse_split(
                model,
                long_toks,
                out_sub,
                topk=args.topk,
                batch_size=args.long_batch_size,
                rows_per_shard=args.long_rows_per_shard,
                logit_chunk_tokens=args.logit_chunk_tokens,
                vocab_limit=args.vocab_limit,
                is_writer=is_writer,
                shard_rank=args.shard_rank,
                shard_world=args.shard_world,
                dp_rank=dp_rank,
                dp_world=dp_world,
                resume=args.resume,
            )
        manifest["train_long"] = long_summaries

    if distributed:
        dist.barrier()
    if is_writer and args.shard_rank == 0:
        (out_root / "teacher_logits_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote teacher logits manifest to {out_root / 'teacher_logits_manifest.json'}", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
