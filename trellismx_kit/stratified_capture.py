"""Stratified calibration window selector enforcing rare-expert token floors.

Before encoding low-bit (K2/P25) experts, rare domain-specialized experts
(frequently triggered in code, agentic, and multilingual slices) can receive
poorly conditioned input Hessians under uniform sampling. This module greedily
selects a subset of candidate calibration windows using exact submodular
log-coverage gain while respecting per-domain quotas and handling unreachable
experts cleanly.
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
    """Greedily pick windows that maximize submodular rare-expert coverage toward token_floor."""
    if not records or target_windows <= 0:
        raise ValueError("records and target_windows must be nonempty/positive")
    if target_windows > len(records):
        raise ValueError(f"target_windows {target_windows} exceeds available records {len(records)}")

    shape = records[0].expert_counts.shape
    stacked = np.empty((len(records), int(np.prod(shape))), dtype=np.int64)
    for i, rec in enumerate(records):
        if rec.expert_counts.shape != shape:
            raise ValueError("all WindowRecords must share one expert_counts shape")
        arr = np.asarray(rec.expert_counts, dtype=np.int64)
        if bool((arr < 0).any()):
            raise ValueError(f"window {rec.window_id!r} contains negative expert counts")
        stacked[i] = arr.reshape(-1)

    accumulated = np.zeros(stacked.shape[1], dtype=np.int64)
    chosen: list[str] = []
    used = np.zeros(len(records), dtype=bool)
    domain_counts: dict[str, int] = {}
    caps = dict(domain_caps or {})

    for _ in range(target_windows):
        eligible = [
            idx
            for idx, rec in enumerate(records)
            if not used[idx] and (caps.get(rec.domain) is None or domain_counts.get(rec.domain, 0) < caps[rec.domain])
        ]
        if not eligible:
            raise ValueError("domain_caps prevented reaching target_windows")

        cand_counts = stacked[eligible]  # [M, D]
        # Exact submodular marginal coverage toward token_floor:
        # sum(log1p(min(acc + c, floor)) - log1p(min(acc, floor)))
        cur_clamped = np.minimum(accumulated, token_floor)
        new_clamped = np.minimum(accumulated[None, :] + cand_counts, token_floor)
        floor_gains = (np.log1p(new_clamped) - np.log1p(cur_clamped[None, :])).sum(axis=1)

        if float(floor_gains.max()) > 0.0:
            best_pos = int(np.argmax(floor_gains))
        else:
            # All reachable experts have met token_floor (or remaining deficits are unreachable):
            # balance coverage across all experts via log1p marginal gain.
            bal_gains = (np.log1p(accumulated[None, :] + cand_counts) - np.log1p(accumulated[None, :])).sum(axis=1)
            best_pos = int(np.argmax(bal_gains))

        best_idx = eligible[best_pos]
        used[best_idx] = True
        rec = records[best_idx]
        chosen.append(rec.window_id)
        accumulated += stacked[best_idx]
        domain_counts[rec.domain] = domain_counts.get(rec.domain, 0) + 1

    final_counts = accumulated.reshape(shape)
    return SelectionResult(
        selected_ids=tuple(chosen),
        domain_counts=domain_counts,
        min_expert_tokens=int(final_counts.min()),
        below_floor_experts=int((final_counts < token_floor).sum()),
        total_expert_counts=final_counts,
    )
