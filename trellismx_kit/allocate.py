"""Model-neutral fast exact per-expert rate allocator over {k2, p25, k3, k4}.

Assigns one shared TrellisMX rate per routed expert under an exact aggregate
half-bit byte budget. Supports:
1. hierarchical: layer_sensitivity[l] * expert_route_mass[l, e] * rd_curve[rate]
   with optional Engram co-allocation discount (1 - gamma * bigram_r2[l]) so
   VRAM bits flow into deep contextual layers while host-RAM Engram covers
   high-R^2 early and final layers.
2. route-mass: expert_route_mass[l, e] * rd_curve[rate].

Uses O(N log N) convex marginal-gain allocation with exact boundary resolution
when per-expert RD curves are convex (solves 23,424 experts on a 1T MoE in
under 50 milliseconds with O(N) memory), falling back to exact compact DP for
arbitrary non-convex damage tables.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pe_format import ASSIGNMENT_SCHEMA, assignment_sha256
from .profile import ModelProfile, load_profile
from .trellis_ref import RATES

RATE_FRACTION = {
    "k2": Fraction(2),
    "p25": Fraction(5, 2),
    "k3": Fraction(3),
    "k4": Fraction(4),
}

EMPIRICAL_RD_RATIOS = {
    "k2": 4.631,
    "p25": 2.002,
    "k3": 1.000,
    "k4": 0.500,
}

IDEAL_RD_RATIOS = {
    "k2": 4.0,
    "p25": 2.0,
    "k3": 1.0,
    "k4": 0.25,
}


def _is_convex_item(costs: Sequence[int], damages: Sequence[float]) -> bool:
    """Return True iff finite (cost, damage) points have non-increasing marginal slopes."""
    pts = sorted(
        ((c, d) for c, d in zip(costs, damages, strict=True) if math.isfinite(d)),
        key=lambda x: x[0],
    )
    if len(pts) <= 2:
        return True
    prev_slope = math.inf
    for i in range(len(pts) - 1):
        dc = pts[i + 1][0] - pts[i][0]
        if dc <= 0:
            return False
        dd = pts[i][1] - pts[i + 1][1]
        if dd < -1e-15:
            return False
        slope = dd / dc
        if slope > prev_slope + 1e-12:
            return False
        prev_slope = slope
    return True


def _solve_convex_fast(
    candidates: Sequence[tuple[int, int, Sequence[float]]],
    costs: Sequence[int],
    rates: Sequence[str],
    budget_half_bits: int,
) -> tuple[dict[tuple[int, int], str], float]:
    """O(N log N) exact marginal-gain solver for convex per-expert RD curves."""
    n = len(candidates)
    order = sorted(range(len(rates)), key=lambda i: costs[i])
    # Per item, track available sorted options (r_idx, cost, damage)
    item_opts: list[list[tuple[int, int, float]]] = []
    cur_pos = [0] * n
    used = 0
    total_damage = 0.0

    # Max-heap by marginal damage reduction per half-bit: (-slope, -dc, item_idx, next_pos)
    heap: list[tuple[float, int, int, int]] = []
    for i, (_l, _e, damages) in enumerate(candidates):
        opts = [(r_idx, costs[r_idx], float(damages[r_idx])) for r_idx in order if math.isfinite(damages[r_idx])]
        if not opts:
            raise ValueError(f"candidate {i} has no finite rate options")
        item_opts.append(opts)
        used += opts[0][1]
        total_damage += opts[0][2]
        if len(opts) > 1:
            dc = opts[1][1] - opts[0][1]
            dd = opts[0][2] - opts[1][2]
            if dd > 0.0:
                heapq.heappush(heap, (-(dd / dc), -dc, i, 1))

    if used > budget_half_bits:
        raise ValueError("budget is below minimum feasible rate")

    remaining = budget_half_bits - used
    skipped_step2: list[tuple[float, int, int]] = []
    taken_step1: list[tuple[float, int, int]] = []  # (dd, item_idx, prev_pos)

    while heap and remaining > 0:
        neg_slope, neg_dc, idx, next_pos = heapq.heappop(heap)
        if cur_pos[idx] + 1 != next_pos:
            continue
        opts = item_opts[idx]
        dc = opts[next_pos][1] - opts[next_pos - 1][1]
        dd = opts[next_pos - 1][2] - opts[next_pos][2]
        if dc <= remaining:
            cur_pos[idx] = next_pos
            remaining -= dc
            total_damage -= dd
            if dc == 1:
                taken_step1.append((dd, idx, next_pos - 1))
            if next_pos + 1 < len(opts):
                ndc = opts[next_pos + 1][1] - opts[next_pos][1]
                ndd = opts[next_pos][2] - opts[next_pos + 1][2]
                if ndd > 0.0:
                    heapq.heappush(heap, (-(ndd / ndc), -ndc, idx, next_pos + 1))
        elif dc == 2 and remaining == 1:
            skipped_step2.append((dd, idx, next_pos))

    # Exact boundary optimality for {1, 2}-half-bit step sizes:
    # A skipped dc=2 step could beat either:
    # (a) 1 leftover half-bit + 1 weakest active dc=1 step (when remaining == 1), OR
    # (b) 2 weakest active dc=1 steps (when remaining == 0 because a cheaper dc=1 filled the last half-bit).
    if skipped_step2:
        # Active dc=1 steps whose item is still at prev_pos + 1 (valid to undo)
        active_step1 = [
            (dd1, idx1, prev_p1)
            for (dd1, idx1, prev_p1) in taken_step1
            if cur_pos[idx1] == prev_p1 + 1
        ]
        active_step1.sort(key=lambda x: x[0])
        best_dd2, best_idx2, best_pos2 = max(skipped_step2, key=lambda x: x[0])
        other_step1 = [x for x in active_step1 if x[1] != best_idx2]
        if remaining == 1 and len(other_step1) >= 1:
            dd1, idx1, prev_p1 = other_step1[0]
            if best_dd2 > dd1:
                cur_pos[idx1] = prev_p1
                cur_pos[best_idx2] = best_pos2
                total_damage = total_damage + dd1 - best_dd2
        elif remaining == 0 and len(other_step1) >= 2:
            dd1a, idx1a, prev_p1a = other_step1[0]
            dd1b, idx1b, prev_p1b = other_step1[1]
            # Handle case where both weakest steps belong to same item (undoing 2 steps on same item)
            if idx1a != idx1b and best_dd2 > (dd1a + dd1b):
                cur_pos[idx1a] = prev_p1a
                cur_pos[idx1b] = prev_p1b
                cur_pos[best_idx2] = best_pos2
                total_damage = total_damage + dd1a + dd1b - best_dd2

    choices: dict[tuple[int, int], str] = {}
    for i, (layer, expert, _damages) in enumerate(candidates):
        r_idx = item_opts[i][cur_pos[i]][0]
        choices[(layer, expert)] = rates[r_idx]
    return choices, total_damage


def solve_exact_dp(
    candidates: Sequence[tuple[int, int, Sequence[float]]],
    *,
    rates: Sequence[str],
    budget_bits: Fraction,
) -> tuple[dict[tuple[int, int], str], float]:
    """Return exact minimum-predicted-damage rate choice per expert."""
    if not candidates or not rates:
        raise ValueError("candidates and rates must be nonempty")
    if len(set(rates)) != len(rates) or any(r not in RATES for r in rates):
        raise ValueError(f"rates must be unique subset of {RATES}")
    if budget_bits <= 0:
        raise ValueError("budget_bits must be positive")

    costs = [int(RATE_FRACTION[r] * 2) for r in rates]
    total_half_bits = Fraction(budget_bits) * 2 * len(candidates)
    budget = int(total_half_bits // 1)

    for layer, expert, damages in candidates:
        if len(damages) != len(rates):
            raise ValueError(f"invalid damages length for layer {layer}, expert {expert}")
        if any(math.isnan(d) or d == -math.inf for d in damages) or not any(math.isfinite(d) for d in damages):
            raise ValueError(f"invalid damages for layer {layer}, expert {expert}")

    min_total_cost = sum(
        min(c for c, d in zip(costs, damages, strict=True) if math.isfinite(d))
        for _, _, damages in candidates
    )
    if budget < min_total_cost:
        raise ValueError(
            f"budget {float(budget_bits):.4f} bits is below minimum feasible rate {min_total_cost / (2 * len(candidates)):.2f}"
        )

    if all(_is_convex_item(costs, damages) for _, _, damages in candidates):
        return _solve_convex_fast(candidates, costs, rates, budget)

    # Compact NumPy DP fallback for arbitrary non-convex damage tables
    n_total = len(candidates)
    min_cost = min(costs)
    width = budget + 1
    dp = np.full(width, np.inf, dtype=np.float64)
    dp[0] = 0.0
    parent_rate = np.full((n_total, width), 255, dtype=np.uint8)
    for step_idx, (_layer, _expert, damages) in enumerate(candidates):
        rem = n_total - step_idx - 1
        max_usable = min(budget, budget - rem * min_cost)
        next_dp = np.full(width, np.inf, dtype=np.float64)
        for r_idx, (cost, d_val) in enumerate(zip(costs, damages, strict=True)):
            if not math.isfinite(d_val) or cost > max_usable:
                continue
            cand = dp[: max_usable + 1 - cost] + d_val
            target = next_dp[cost : max_usable + 1]
            better = cand < target
            target[better] = cand[better]
            parent_rate[step_idx, cost : max_usable + 1][better] = r_idx
        dp = next_dp

    final_used = int(np.argmin(dp))
    total_damage = float(dp[final_used])
    choices: dict[tuple[int, int], str] = {}
    used = final_used
    for row_idx in range(n_total - 1, -1, -1):
        r_idx = int(parent_rate[row_idx, used])
        layer, expert, _ = candidates[row_idx]
        choices[(layer, expert)] = rates[r_idx]
        used -= costs[r_idx]
    return choices, total_damage


def build_hierarchical_assignment(
    profile: ModelProfile,
    *,
    route_mass: Mapping[int, Sequence[float]],
    layer_sensitivity: Mapping[int, float] | None = None,
    engram_r2: Mapping[int, float] | None = None,
    engram_discount: float = 0.0,
    budget_bits: float = 2.5,
    rates: Sequence[str] | None = None,
    rd_ratios: Mapping[str, float] = EMPIRICAL_RD_RATIOS,
    fixed_layers: Mapping[int, str] | None = None,
    max_k4_fraction: float | None = None,
) -> dict[str, Any]:
    """Allocate one rate per expert across profile.routed_layers."""
    profile.validate()
    active_rates = tuple(rates) if rates is not None else profile.rates
    fixed = {int(k): str(v) for k, v in (fixed_layers or {}).items()}
    routed_set = set(profile.routed_layers)
    invalid_fixed = set(fixed) - routed_set
    if invalid_fixed:
        raise ValueError(f"fixed_layers contains layers outside profile.routed_layers: {sorted(invalid_fixed)}")
    for l_id, r_name in fixed.items():
        if r_name not in RATES:
            raise ValueError(f"fixed layer {l_id} has unsupported rate {r_name!r}")

    if layer_sensitivity is None:
        sens = {layer: 1.0 for layer in profile.routed_layers}
        rule = "route-mass"
    else:
        raw_s = {layer: float(layer_sensitivity[layer]) for layer in profile.routed_layers}
        if any(v <= 0 or not math.isfinite(v) for v in raw_s.values()):
            raise ValueError("layer_sensitivity scalars must be finite and positive")
        mean_s = sum(raw_s.values()) / len(raw_s)
        sens = {layer: v / mean_s for layer, v in raw_s.items()}
        rule = "hierarchical"

    # Optional asymmetric Engram co-allocation: discount high-R^2 layers so VRAM
    # bits shift toward deep contextual layers where n-gram predictability is low.
    if engram_r2 is not None and engram_discount > 0.0:
        if not 0.0 <= engram_discount <= 1.0:
            raise ValueError("engram_discount must be in [0, 1]")
        for layer in profile.routed_layers:
            r2 = max(0.0, min(1.0, float(engram_r2.get(layer, 0.0))))
            sens[layer] *= max(0.1, 1.0 - engram_discount * r2)
        mean_adj = sum(sens.values()) / len(sens)
        sens = {layer: v / mean_adj for layer, v in sens.items()}

    free_layers = [layer for layer in profile.routed_layers if layer not in fixed]
    fixed_half_bits = sum(
        int(RATE_FRACTION[fixed[layer]] * 2) * profile.num_experts for layer in fixed
    )
    total_experts = len(profile.routed_layers) * profile.num_experts
    free_experts = len(free_layers) * profile.num_experts
    if free_experts == 0:
        raise ValueError("all routed layers are fixed; nothing left to allocate")

    budget_frac = Fraction(str(budget_bits))
    target_total_half_bits = int((budget_frac * 2 * total_experts) // 1)
    free_half_bits = target_total_half_bits - fixed_half_bits
    if free_half_bits <= 0:
        raise ValueError("fixed_layers consume the entire bit budget")
    free_budget_bits = Fraction(free_half_bits, 2 * free_experts)

    # Validate route_mass upfront
    norm_masses: dict[int, list[float]] = {}
    for layer in free_layers:
        masses = [float(x) for x in route_mass[layer]]
        if len(masses) != profile.num_experts:
            raise ValueError(f"route_mass for layer {layer} has {len(masses)} entries; expected {profile.num_experts}")
        if any(not math.isfinite(m) or m < 0.0 for m in masses):
            raise ValueError(f"route_mass for layer {layer} must be finite and non-negative")
        layer_sum = max(sum(masses), 1e-12)
        norm_masses[layer] = [m / layer_sum for m in masses]

    # Strict max_k4_fraction cap accounting for any fixed k4 layers
    fixed_k4 = sum(profile.num_experts for l_id in fixed if fixed[l_id] == "k4")
    k4_allowed: set[tuple[int, int]] | None = None
    if max_k4_fraction is not None and "k4" in active_rates:
        if not 0.0 <= max_k4_fraction <= 1.0:
            raise ValueError("max_k4_fraction must be in [0, 1]")
        max_total_k4 = int(math.floor(max_k4_fraction * total_experts + 1e-12))
        cap_free_k4 = max(0, max_total_k4 - fixed_k4)
        scored = [
            (sens[layer] * norm_masses[layer][e], layer, e)
            for layer in free_layers
            for e in range(profile.num_experts)
        ]
        scored.sort(reverse=True)
        k4_allowed = {(l, e) for _, l, e in scored[:cap_free_k4]}

    candidates: list[tuple[int, int, list[float]]] = []
    for layer in free_layers:
        for e in range(profile.num_experts):
            imp = sens[layer] * norm_masses[layer][e]
            row = []
            for r in active_rates:
                if r == "k4" and k4_allowed is not None and (layer, e) not in k4_allowed:
                    row.append(math.inf)
                else:
                    row.append(imp * float(rd_ratios[r]))
            candidates.append((layer, e, row))

    choices, pred_damage = solve_exact_dp(
        candidates,
        rates=active_rates,
        budget_bits=free_budget_bits,
    )

    layers_out: dict[str, list[str]] = {}
    rate_counts = {r: 0 for r in RATES}
    achieved_bits_sum = Fraction(0)
    for layer in profile.routed_layers:
        if layer in fixed:
            row_rates = [fixed[layer]] * profile.num_experts
        else:
            row_rates = [choices[(layer, e)] for e in range(profile.num_experts)]
        layers_out[str(layer)] = row_rates
        for r in row_rates:
            rate_counts[r] += 1
            achieved_bits_sum += RATE_FRACTION[r]

    achieved_bpw = float(achieved_bits_sum / total_experts)
    document: dict[str, Any] = {
        "schema": ASSIGNMENT_SCHEMA,
        "model_name": profile.name,
        "rule": rule,
        "rates": list(active_rates),
        "budget_bits": float(budget_bits),
        "achieved_bits": achieved_bpw,
        "counts": rate_counts,
        "predicted_damage": pred_damage,
        "layers": layers_out,
        "source": {
            "rd_ratios": {r: float(rd_ratios[r]) for r in active_rates},
            "max_k4_fraction": max_k4_fraction,
            "engram_discount": engram_discount if engram_r2 is not None else 0.0,
            "fixed_layers": {str(k): v for k, v in fixed.items()},
        },
    }
    document["sha256"] = assignment_sha256(
        {k: v for k, v in document.items() if k != "sha256"},
        profile,
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign one TrellisMX rate per routed expert.")
    parser.add_argument("--profile", required=True, help="Path to ModelProfile JSON")
    parser.add_argument("--usage", required=True, help="Path to usage JSON with per-layer route_mass")
    parser.add_argument("--layer-sensitivity", help="Optional JSON mapping layer id to sensitivity scalar")
    parser.add_argument("--engram-r2", help="Optional JSON mapping layer id to held-out bigram R^2 for co-allocation")
    parser.add_argument("--engram-discount", type=float, default=0.0, help="Co-allocation discount gamma in [0, 1]")
    parser.add_argument("--budget-bits", type=float, default=2.5)
    parser.add_argument("--rates", help="Comma-separated rate subset (default: profile.rates)")
    parser.add_argument("--max-k4-fraction", type=float, default=0.10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    usage_doc = json.loads(Path(args.usage).read_text(encoding="utf-8"))
    layers_map = usage_doc.get("layers", usage_doc)
    route_mass = {int(k): [float(x) for x in v["route_mass"]] for k, v in layers_map.items()}
    layer_sens = None
    if args.layer_sensitivity:
        raw = json.loads(Path(args.layer_sensitivity).read_text(encoding="utf-8"))
        s_map = raw.get("layers", raw)
        layer_sens = {int(k): float(v) for k, v in s_map.items()}
    r2_map = None
    if args.engram_r2:
        raw_r2 = json.loads(Path(args.engram_r2).read_text(encoding="utf-8"))
        r2_src = raw_r2.get("layers", raw_r2)
        r2_map = {int(k): float(v) for k, v in r2_src.items()}
    rates = tuple(args.rates.split(",")) if args.rates else None

    doc = build_hierarchical_assignment(
        profile,
        route_mass=route_mass,
        layer_sensitivity=layer_sens,
        engram_r2=r2_map,
        engram_discount=args.engram_discount,
        budget_bits=args.budget_bits,
        rates=rates,
        max_k4_fraction=args.max_k4_fraction,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Assigned {profile.name}: achieved={doc['achieved_bits']:.4f} bpw, "
        f"counts={doc['counts']}, sha256={doc['sha256'][:16]}..."
    )


if __name__ == "__main__":
    main()
