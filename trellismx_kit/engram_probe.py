"""Training-free n-gram predictability diagnostic for routed MoE experts.

Given routed expert outputs y_t and corresponding token sequences, measures
held-out unigram and bigram explained variance (R^2) and normalized MSE (NMSE)
per expert. Used to quantify which layers behave as static local n-gram
lookups versus dynamic contextual transformations and to guide asymmetric
bit-rate plus Engram co-allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ExpertPredictability:
    unigram_coverage: float
    unigram_nmse: float
    unigram_r2: float
    bigram_coverage: float
    bigram_nmse: float
    bigram_r2: float


def score_expert_ngram_predictability(
    outputs: torch.Tensor,
    token_ids: np.ndarray,
    prev_token_ids: np.ndarray,
    is_build: np.ndarray,
    *,
    bigram_min_count: int = 2,
) -> ExpertPredictability:
    """Evaluate held-out unigram and bigram-backoff mean-table accuracy for one expert."""
    tok_all = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    prev_all = np.asarray(prev_token_ids, dtype=np.int64).reshape(-1)
    build_mask = np.asarray(is_build, dtype=bool).reshape(-1)
    n = int(outputs.shape[0])
    if tok_all.shape != (n,) or prev_all.shape != (n,) or build_mask.shape != (n,):
        raise ValueError("token_ids, prev_token_ids, and is_build must match outputs.shape[0]")

    device = outputs.device
    y_all = outputs.float()
    b_idx = torch.from_numpy(np.nonzero(build_mask)[0]).to(device)
    t_idx = torch.from_numpy(np.nonzero(~build_mask)[0]).to(device)
    if b_idx.numel() == 0 or t_idx.numel() == 0:
        raise ValueError("build and held-out splits must both be nonempty")

    y_b = y_all[b_idx]
    y_t = y_all[t_idx]
    d = y_all.shape[-1]
    mu0 = y_b.mean(dim=0)

    tok_b = tok_all[build_mask]
    tok_t = tok_all[~build_mask]
    vocab_stride = int(max(int(tok_all.max()), int(prev_all.max()), 1)) + 2
    bi_b = (prev_all[build_mask] + 1) * np.int64(vocab_stride) + tok_b
    bi_t = (prev_all[~build_mask] + 1) * np.int64(vocab_stride) + tok_t

    uniq_u, inv_u = np.unique(tok_b, return_inverse=True)
    inv_u_t = torch.from_numpy(inv_u).to(device)
    u_sum = torch.zeros((len(uniq_u), d), device=device, dtype=torch.float32)
    u_cnt = torch.zeros(len(uniq_u), device=device, dtype=torch.float32)
    u_sum.index_add_(0, inv_u_t, y_b)
    u_cnt.index_add_(0, inv_u_t, torch.ones_like(inv_u_t, dtype=torch.float32))
    u_mean = u_sum / u_cnt[:, None].clamp_min(1.0)

    idx_u = np.searchsorted(uniq_u, tok_t)
    hit_u = (idx_u < len(uniq_u)) & (uniq_u[np.minimum(idx_u, len(uniq_u) - 1)] == tok_t)
    pred_u = mu0.unsqueeze(0).expand_as(y_t).clone()
    if hit_u.any():
        pred_u[torch.from_numpy(np.nonzero(hit_u)[0]).to(device)] = u_mean[
            torch.from_numpy(idx_u[hit_u]).to(device)
        ]

    uniq_b, inv_b, cnt_b = np.unique(bi_b, return_inverse=True, return_counts=True)
    keep = cnt_b >= bigram_min_count
    pred_bi = pred_u.clone()
    bi_cov = 0.0
    if keep.any():
        kept_keys = uniq_b[keep]
        remap = np.full(len(uniq_b), -1, dtype=np.int64)
        remap[keep] = np.arange(keep.sum(), dtype=np.int64)
        mapped = remap[inv_b]
        has = mapped >= 0
        b_sum = torch.zeros((len(kept_keys), d), device=device, dtype=torch.float32)
        b_cnt = torch.zeros(len(kept_keys), device=device, dtype=torch.float32)
        h_pos = torch.from_numpy(np.nonzero(has)[0]).to(device)
        h_key = torch.from_numpy(mapped[has]).to(device)
        b_sum.index_add_(0, h_key, y_b[h_pos])
        b_cnt.index_add_(0, h_key, torch.ones_like(h_key, dtype=torch.float32))
        b_mean = b_sum / b_cnt[:, None].clamp_min(1.0)

        idx_b = np.searchsorted(kept_keys, bi_t)
        hit_b = (idx_b < len(kept_keys)) & (kept_keys[np.minimum(idx_b, len(kept_keys) - 1)] == bi_t)
        bi_cov = float(hit_b.mean())
        if hit_b.any():
            pred_bi[torch.from_numpy(np.nonzero(hit_b)[0]).to(device)] = b_mean[
                torch.from_numpy(idx_b[hit_b]).to(device)
            ]

    energy = max(float(y_t.square().sum().item()), 1e-12)
    var = max(float((y_t - mu0).square().sum().item()), 1e-12)
    u_err = float((y_t - pred_u).square().sum().item())
    b_err = float((y_t - pred_bi).square().sum().item())
    return ExpertPredictability(
        unigram_coverage=float(hit_u.mean()),
        unigram_nmse=u_err / energy,
        unigram_r2=1.0 - u_err / var,
        bigram_coverage=bi_cov,
        bigram_nmse=b_err / energy,
        bigram_r2=1.0 - b_err / var,
    )
