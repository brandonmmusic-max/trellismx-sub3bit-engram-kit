"""Stratified calibration window selector enforcing rare-expert token floors.

Before encoding low-bit (K2/P25) experts, rare domain-specialized experts
(frequently triggered in code, agentic, and multilingual slices) can receive
poorly conditioned input Hessians under uniform sampling. This module greedily
selects a subset of candidate calibration windows to maximize minimum routed
token coverage across all experts while respecting per-domain quotas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WindowRecord:
    window_id: str
    domain: str
    expert_counts: np.ndarray  # [num_routed_layers, num_experts] int64


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: tuple[str, ...]
    domain_counts: dict[str, int]
    min_expert_tokens: int
    below_floor_experts: int
    total_expert_counts: np.ndarray


def select_stratified_windows(
    records: Sequence[WindowRecord],
    *,
    target_windows: int,
    token_floor: int = 2000,
    domain_caps: Mapping[str, int] | None = None,
) -> SelectionResult:
    """Greedily pick windows that raise the most under-covered experts toward token_floor."""
    if not records or target_windows <= 0:
        raise ValueError("records and target_windows must be nonempty/positive")
    if target_windows > len(records):
        raise ValueError(f"target_windows {target_windows} exceeds available records {len(records)}")

    shape = records[0].expert_counts.shape
    if any(r.expert_counts.shape != shape for r in records):
        raise ValueError("all WindowRecords must share one expert_counts shape")

    accumulated = np.zeros(shape, dtype=np.int64)
    chosen: list[str] = []
    used = np.zeros(len(records), dtype=bool)
    domain_counts: dict[str, int] = {}
    caps = dict(domain_caps or {})

    for _ in range(target_windows):
        deficit = np.maximum(0, token_floor - accumulated)
        best_idx = -1
        best_score = -1.0
        for idx, rec in enumerate(records):
            if used[idx]:
                continue
            cap = caps.get(rec.domain)
            if cap is not None and domain_counts.get(rec.domain, 0) >= cap:
                continue
            if deficit.any():
                # Gain is capped contribution toward closing remaining deficits,
                # weighted by inverse current count so zero-token experts dominate.
                gain = np.minimum(rec.expert_counts, deficit)
                weight = 1.0 / (accumulated + 1.0)
                score = float((gain * weight).sum())
            else:
                # Once every expert passes token_floor, balance least-seen experts
                score = float((rec.expert_counts / (accumulated + 1.0)).sum())
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            raise ValueError("domain_caps prevented reaching target_windows")
        used[best_idx] = True
        rec = records[best_idx]
        chosen.append(rec.window_id)
        accumulated += rec.expert_counts
        domain_counts[rec.domain] = domain_counts.get(rec.domain, 0) + 1

    return SelectionResult(
        selected_ids=tuple(chosen),
        domain_counts=domain_counts,
        min_expert_tokens=int(accumulated.min()),
        below_floor_experts=int((accumulated < token_floor).sum()),
        total_expert_counts=accumulated,
    )
