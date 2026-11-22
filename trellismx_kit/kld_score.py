"""Exact FP64 full-vocabulary KL(teacher || student) scorer and bootstrap CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


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


def tokenwise_kld_fp64(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact FP64 per-position KL(teacher || student) and uint8 top-1 match."""
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 2:
        raise ValueError("teacher_logits and student_logits must be matching 2D [N, V] tensors")
    klds, agree = [], []
    n = teacher_logits.shape[0]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        t = teacher_logits[start:stop].double().log_softmax(dim=-1)
        s = student_logits[start:stop].double().log_softmax(dim=-1)
        klds.append((t.exp() * (t - s)).sum(dim=-1).cpu())
        agree.append((t.argmax(dim=-1) == s.argmax(dim=-1)).to(torch.uint8).cpu())
    return torch.cat(klds), torch.cat(agree)


def summarize_windows(
    window_klds: Sequence[torch.Tensor],
    window_agree: Sequence[torch.Tensor],
    *,
    resamples: int = 5000,
    seed: int = 20260829,
) -> KLDSummary:
    """Compute headline mean, window-block bootstrap 95% CI, tails, and top-1 agreement."""
    if not window_klds or len(window_klds) != len(window_agree):
        raise ValueError("window_klds and window_agree must have equal nonempty length")
    all_kld = torch.cat([w.double().reshape(-1) for w in window_klds]).numpy()
    all_agree = torch.cat([w.double().reshape(-1) for w in window_agree]).numpy()
    win_means = np.array([float(w.double().mean().item()) for w in window_klds], dtype=np.float64)
    win_weights = np.array([int(w.numel()) for w in window_klds], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_win = len(win_means)
    idx = rng.integers(0, n_win, size=(resamples, n_win))
    boot_num = (win_means[idx] * win_weights[idx]).sum(axis=1)
    boot_den = win_weights[idx].sum(axis=1)
    boot_means = boot_num / boot_den
    lo, hi = np.quantile(boot_means, [0.025, 0.975])

    return KLDSummary(
        positions=int(all_kld.size),
        windows=n_win,
        mean_kld=float(all_kld.mean()),
        ci95_low=float(lo),
        ci95_high=float(hi),
        median_kld=float(np.median(all_kld)),
        p90_kld=float(np.quantile(all_kld, 0.90)),
        p99_kld=float(np.quantile(all_kld, 0.99)),
        top1_agreement_pct=float(all_agree.mean() * 100.0),
    )
