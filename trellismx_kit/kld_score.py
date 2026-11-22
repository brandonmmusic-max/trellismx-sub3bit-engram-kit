"""Exact FP64 full-vocabulary KL(teacher || student) scorer with BCa 95% CIs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np
import torch


ALIGNMENT_BAND = (0.20, 0.995)


@dataclass(frozen=True)
class KLDSummary:
    positions: int
    windows: int
    mean_kld: float
    ci95_low: float
    ci95_high: float
    median_kld: float
    p90_kld: float
    p99_kld: float
    top1_agreement_pct: float
    p99_9_kld: float | None = None
    max_kld: float = 0.0
    percentile_ci95_low: float = 0.0
    percentile_ci95_high: float = 0.0
    top1_wilson95_low_pct: float = 0.0
    top1_wilson95_high_pct: float = 0.0
    perplexity_ratio: float | None = None
    realized_alignment_rate: float | None = None


def _wilson_interval_pct(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    rad = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, (center - rad) * 100.0), min(100.0, (center + rad) * 100.0)


def _bca_interval(
    win_means: np.ndarray,
    win_weights: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Weighted window-block BCa and percentile 95% intervals."""
    n_win = len(win_means)
    observed = float((win_means * win_weights).sum() / win_weights.sum())
    if n_win < 2 or resamples < 1:
        return observed, observed, observed, observed
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_win, size=(resamples, n_win))
    boots = (win_means[idx] * win_weights[idx]).sum(axis=1) / win_weights[idx].sum(axis=1)
    p_lo, p_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    jack = np.empty(n_win, dtype=np.float64)
    total_num = float((win_means * win_weights).sum())
    total_den = float(win_weights.sum())
    for i in range(n_win):
        jack[i] = (total_num - win_means[i] * win_weights[i]) / (total_den - win_weights[i])
    diffs = jack.mean() - jack
    denom = float((diffs**2).sum())
    accel = float((diffs**3).sum() / (6.0 * denom**1.5)) if denom > 0.0 else 0.0

    prob_less = float((boots < observed).mean())
    normal = NormalDist()
    z0 = normal.inv_cdf(max(min(prob_less, 1.0 - 1e-9), 1e-9))
    adj: list[float] = []
    for q in (0.025, 0.975):
        zq = normal.inv_cdf(q)
        denom_a = 1.0 - accel * (z0 + zq)
        if abs(denom_a) < 1e-12:
            adj_q = q
        else:
            adj_q = min(max(normal.cdf(z0 + (z0 + zq) / denom_a), 0.0), 1.0)
        adj.append(float(np.quantile(boots, adj_q)))
    return adj[0], adj[1], p_lo, p_hi


def tokenwise_kld_fp64(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    chunk: int = 1024,
    vocab_limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact FP64 per-position KL(teacher || student) and uint8 top-1 match."""
    if teacher_logits.ndim != 2 or student_logits.ndim != 2:
        raise ValueError("teacher_logits and student_logits must be 2D [N, V] tensors")
    if vocab_limit is not None:
        teacher_logits = teacher_logits[:, :vocab_limit]
        student_logits = student_logits[:, :vocab_limit]
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher_logits and student_logits must have matching shapes")
    n = int(teacher_logits.shape[0])
    if n == 0:
        return torch.empty((0,), dtype=torch.float64), torch.empty((0,), dtype=torch.uint8)
    klds, agree = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        t_raw = teacher_logits[start:stop].double()
        s_raw = student_logits[start:stop].double()
        if (
            bool(torch.isnan(t_raw).any())
            or bool(torch.isposinf(t_raw).any())
            or bool(torch.isnan(s_raw).any())
            or bool(torch.isposinf(s_raw).any())
        ):
            raise ValueError("teacher and student logits must not contain NaN or +inf")
        t = t_raw.log_softmax(dim=-1)
        s = s_raw.log_softmax(dim=-1)
        t_p = t.exp()
        term = torch.where(t_p > 0.0, t_p * (t - s), torch.zeros_like(t))
        klds.append(term.sum(dim=-1).clamp_min(0.0).detach().cpu())
        agree.append((t.argmax(dim=-1) == s.argmax(dim=-1)).to(torch.uint8).detach().cpu())
    return torch.cat(klds), torch.cat(agree)


def tokenwise_kld_and_alignment_fp64(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    realized_tokens: torch.Tensor,
    *,
    chunk: int = 1024,
    vocab_limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (kld, top1_agree, teacher_logp_realized, student_logp_realized, teacher_hits_realized)."""
    if vocab_limit is not None:
        teacher_logits = teacher_logits[:, :vocab_limit]
        student_logits = student_logits[:, :vocab_limit]
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 2:
        raise ValueError("teacher_logits and student_logits must be matching 2D [N, V] tensors")
    n = int(teacher_logits.shape[0])
    realized = realized_tokens.detach().cpu().to(torch.long).reshape(-1)
    if realized.numel() != n:
        raise ValueError(f"realized_tokens length {realized.numel()} disagrees with {n} logit rows")

    klds, agree, t_rlp, s_rlp, t_hit = [], [], [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        t_raw = teacher_logits[start:stop].double()
        s_raw = student_logits[start:stop].double()
        if (
            bool(torch.isnan(t_raw).any())
            or bool(torch.isposinf(t_raw).any())
            or bool(torch.isnan(s_raw).any())
            or bool(torch.isposinf(s_raw).any())
        ):
            raise ValueError("teacher and student logits must not contain NaN or +inf")
        t = t_raw.log_softmax(dim=-1)
        s = s_raw.log_softmax(dim=-1)
        t_p = t.exp()
        term = torch.where(t_p > 0.0, t_p * (t - s), torch.zeros_like(t))
        klds.append(term.sum(dim=-1).clamp_min(0.0).detach().cpu())
        t_top1 = t.argmax(dim=-1)
        s_top1 = s.argmax(dim=-1)
        agree.append((t_top1 == s_top1).to(torch.uint8).detach().cpu())
        r_idx = realized[start:stop].to(t.device).unsqueeze(-1)
        t_rlp.append(t.gather(-1, r_idx).squeeze(-1).detach().cpu())
        s_rlp.append(s.gather(-1, r_idx).squeeze(-1).detach().cpu())
        t_hit.append((t_top1.detach().cpu() == realized[start:stop]).to(torch.uint8))
    return (
        torch.cat(klds),
        torch.cat(agree),
        torch.cat(t_rlp),
        torch.cat(s_rlp),
        torch.cat(t_hit),
    )


def summarize_windows(
    window_klds: Sequence[torch.Tensor],
    window_agree: Sequence[torch.Tensor],
    *,
    window_teacher_logp: Sequence[torch.Tensor] | None = None,
    window_student_logp: Sequence[torch.Tensor] | None = None,
    window_teacher_hits: Sequence[torch.Tensor] | None = None,
    resamples: int = 5000,
    seed: int = 20260829,
) -> KLDSummary:
    """Compute headline mean, window-block BCa 95% CI, tails, Wilson top-1 CI, and alignment rate."""
    if not window_klds or len(window_klds) != len(window_agree):
        raise ValueError("window_klds and window_agree must have equal nonempty length")
    kld_cpu = [w.detach().cpu().double().reshape(-1) for w in window_klds]
    agree_cpu = [a.detach().cpu().double().reshape(-1) for a in window_agree]
    for i, (wk, wa) in enumerate(zip(kld_cpu, agree_cpu, strict=True)):
        if wk.numel() == 0 or wk.numel() != wa.numel():
            raise ValueError(f"window {i} has empty or mismatched length ({wk.numel()} vs {wa.numel()})")

    all_kld = torch.cat(kld_cpu).numpy()
    all_agree = torch.cat(agree_cpu).numpy()
    win_means = np.array([float(w.mean().item()) for w in kld_cpu], dtype=np.float64)
    win_weights = np.array([int(w.numel()) for w in kld_cpu], dtype=np.float64)

    bca_lo, bca_hi, pct_lo, pct_hi = _bca_interval(
        win_means, win_weights, resamples=resamples, seed=seed
    )
    w_lo, w_hi = _wilson_interval_pct(int(all_agree.sum()), int(all_agree.size))
    p99_9 = float(np.quantile(all_kld, 0.999)) if all_kld.size * 0.001 >= 100 else None

    ppl_ratio: float | None = None
    if window_teacher_logp is not None and window_student_logp is not None:
        t_lp = torch.cat([x.detach().cpu().double().reshape(-1) for x in window_teacher_logp]).numpy()
        s_lp = torch.cat([x.detach().cpu().double().reshape(-1) for x in window_student_logp]).numpy()
        ppl_ratio = float(math.exp(float(t_lp.mean() - s_lp.mean())))

    align_rate: float | None = None
    if window_teacher_hits is not None:
        hits = torch.cat([x.detach().cpu().double().reshape(-1) for x in window_teacher_hits]).numpy()
        align_rate = float(hits.mean())

    return KLDSummary(
        positions=int(all_kld.size),
        windows=len(win_means),
        mean_kld=float(all_kld.mean()),
        ci95_low=float(bca_lo),
        ci95_high=float(bca_hi),
        median_kld=float(np.median(all_kld)),
        p90_kld=float(np.quantile(all_kld, 0.90)),
        p99_kld=float(np.quantile(all_kld, 0.99)),
        top1_agreement_pct=float(all_agree.mean() * 100.0),
        p99_9_kld=p99_9,
        max_kld=float(all_kld.max()),
        percentile_ci95_low=float(pct_lo),
        percentile_ci95_high=float(pct_hi),
        top1_wilson95_low_pct=float(w_lo),
        top1_wilson95_high_pct=float(w_hi),
        perplexity_ratio=ppl_ratio,
        realized_alignment_rate=align_rate,
    )
