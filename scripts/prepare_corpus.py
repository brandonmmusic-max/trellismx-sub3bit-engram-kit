#!/usr/bin/env python3
"""Stream, tokenize, pack, and 13-gram screen a short/long QEC distillation corpus.

Produces int32 ``[N, seq_len]`` tensors for:
- ``eval_8k.pt`` (multi-domain held-out evaluation windows)
- ``eval_32k.pt`` (long-context held-out evaluation windows)
- ``train_short_<M>m_4k.pt`` (80% short-context distillation split)
- ``train_long_<M>m_16k.pt`` (20% long-context distillation split)

Every training window is screened against the union of 13-grams across all
evaluation windows (plus any optional external anchor token files) and rejected
if its 13-gram overlap exceeds ``--max-13gram-overlap`` (default 0.01 = 1%).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from transformers import AutoTokenizer


def _hash_13grams(tokens: np.ndarray, *, step: int = 1) -> set[bytes]:
    n = int(tokens.shape[0])
    if n < 13:
        return set()
    arr = np.ascontiguousarray(tokens, dtype=np.int32)
    out: set[bytes] = set()
    for i in range(0, n - 12, step):
        out.add(hashlib.blake2b(arr[i : i + 13].tobytes(), digest_size=8).digest())
    return out


def _overlap_fraction(tokens: np.ndarray, forbidden: set[bytes], *, step: int = 4) -> float:
    n = int(tokens.shape[0])
    if n < 13 or not forbidden:
        return 0.0
    arr = np.ascontiguousarray(tokens, dtype=np.int32)
    total = 0
    hits = 0
    for i in range(0, n - 12, step):
        total += 1
        if hashlib.blake2b(arr[i : i + 13].tobytes(), digest_size=8).digest() in forbidden:
            hits += 1
    return hits / max(1, total)


def _iter_hf_texts(dataset_specs: list[str]) -> Iterator[str]:
    from datasets import load_dataset

    streams = []
    for spec in dataset_specs:
        parts = spec.split(":")
        name = parts[0]
        config = parts[1] if len(parts) > 1 and parts[1] else None
        split = parts[2] if len(parts) > 2 and parts[2] else "train"
        field = parts[3] if len(parts) > 3 and parts[3] else "text"
        ds = load_dataset(name, config, split=split, streaming=True)
        streams.append((iter(ds), field))

    active = list(streams)
    while active:
        next_active = []
        for it, field in active:
            try:
                row = next(it)
            except StopIteration:
                continue
            txt = row.get(field)
            if isinstance(txt, str) and len(txt) >= 64:
                yield txt
            next_active.append((it, field))
        active = next_active


def _pack_windows(
    text_iter: Iterable[str],
    tokenizer,
    *,
    seq_len: int,
    target_windows: int,
    forbidden_13grams: set[bytes] | None = None,
    max_overlap: float = 0.01,
    min_doc_tokens: int = 0,
) -> tuple[torch.Tensor, int]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 0
    buf: list[int] = []
    accepted: list[np.ndarray] = []
    rejected = 0

    for text in text_iter:
        if len(accepted) >= target_windows:
            break
        ids = tokenizer.encode(text, add_special_tokens=False)
        if min_doc_tokens > 0 and len(ids) < min_doc_tokens:
            continue
        buf.extend(ids)
        buf.append(int(eos_id))
        while len(buf) >= seq_len and len(accepted) < target_windows:
            cand = np.asarray(buf[:seq_len], dtype=np.int32)
            del buf[:seq_len]
            if forbidden_13grams and _overlap_fraction(cand, forbidden_13grams) > max_overlap:
                rejected += 1
                continue
            accepted.append(cand)

    if len(accepted) < target_windows:
        raise RuntimeError(
            f"Stream exhausted after {len(accepted)}/{target_windows} windows of length {seq_len}"
        )
    return torch.from_numpy(np.stack(accepted, axis=0)), rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 13-gram screened short/long QEC corpus tensors.")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer directory or repo ID")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["HuggingFaceFW/fineweb-edu:sample-10BT:train:text"],
        help="Streaming dataset specs as 'repo[:config[:split[:text_field]]]'",
    )
    parser.add_argument("--external-eval-tokens", nargs="*", default=[], help="Optional extra .npy/.pt eval token files to include in 13-gram screen")
    parser.add_argument("--eval-8k-windows", type=int, default=21)
    parser.add_argument("--eval-32k-windows", type=int, default=4)
    parser.add_argument("--short-tokens-m", type=float, default=80.0)
    parser.add_argument("--short-seq-len", type=int, default=4096)
    parser.add_argument("--long-tokens-m", type=float, default=20.0)
    parser.add_argument("--long-seq-len", type=int, default=16384)
    parser.add_argument("--max-13gram-overlap", type=float, default=0.01)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)

    forbidden: set[bytes] = set()
    for p_str in args.external_eval_tokens:
        p = Path(p_str)
        arr = np.load(p) if p.suffix == ".npy" else torch.load(p, map_location="cpu", weights_only=True).numpy()
        if arr.ndim == 1:
            forbidden |= _hash_13grams(arr)
        else:
            for row in arr:
                forbidden |= _hash_13grams(row)

    text_stream = _iter_hf_texts(args.datasets)

    eval_8k, _ = _pack_windows(text_stream, tok, seq_len=8192, target_windows=args.eval_8k_windows)
    for row in eval_8k.numpy():
        forbidden |= _hash_13grams(row)
    torch.save(eval_8k, out_dir / "eval_8k.pt")
    print(f"Saved eval_8k.pt shape={tuple(eval_8k.shape)} forbidden_13grams={len(forbidden)}", flush=True)

    eval_32k, _ = _pack_windows(
        text_stream, tok, seq_len=32768, target_windows=args.eval_32k_windows, min_doc_tokens=4096
    )
    for row in eval_32k.numpy():
        forbidden |= _hash_13grams(row)
    torch.save(eval_32k, out_dir / "eval_32k.pt")
    print(f"Saved eval_32k.pt shape={tuple(eval_32k.shape)} forbidden_13grams={len(forbidden)}", flush=True)

    short_windows = max(1, int(round((args.short_tokens_m * 1e6) / (args.short_seq_len - 1))))
    train_short, rej_s = _pack_windows(
        text_stream,
        tok,
        seq_len=args.short_seq_len,
        target_windows=short_windows,
        forbidden_13grams=forbidden,
        max_overlap=args.max_13gram_overlap,
    )
    short_name = f"train_short_{int(round(args.short_tokens_m))}m_{args.short_seq_len // 1024}k.pt"
    torch.save(train_short, out_dir / short_name)
    print(f"Saved {short_name} shape={tuple(train_short.shape)} rejected={rej_s}", flush=True)

    long_windows = max(1, int(round((args.long_tokens_m * 1e6) / (args.long_seq_len - 1))))
    train_long, rej_l = _pack_windows(
        text_stream,
        tok,
        seq_len=args.long_seq_len,
        target_windows=long_windows,
        forbidden_13grams=forbidden,
        max_overlap=args.max_13gram_overlap,
        min_doc_tokens=2048,
    )
    long_name = f"train_long_{int(round(args.long_tokens_m))}m_{args.long_seq_len // 1024}k.pt"
    torch.save(train_long, out_dir / long_name)
    print(f"Saved {long_name} shape={tuple(train_long.shape)} rejected={rej_l}", flush=True)

    manifest = {
        "tokenizer": args.tokenizer,
        "datasets": args.datasets,
        "max_13gram_overlap": args.max_13gram_overlap,
        "forbidden_13grams": len(forbidden),
        "files": {
            "eval_8k": {"path": "eval_8k.pt", "shape": list(eval_8k.shape)},
            "eval_32k": {"path": "eval_32k.pt", "shape": list(eval_32k.shape)},
            "train_short": {"path": short_name, "shape": list(train_short.shape), "rejected": rej_s},
            "train_long": {"path": long_name, "shape": list(train_long.shape), "rejected": rej_l},
        },
    }
    (out_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
