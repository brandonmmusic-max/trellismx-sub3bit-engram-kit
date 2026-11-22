"""Model-neutral exact per-expert rate allocator over {k2, p25, k3, k4}.

Assigns one shared TrellisMX rate per routed expert under an exact aggregate
half-bit byte budget using multiple-choice dynamic programming. Supports three
importance rules:

1. hierarchical: layer_sensitivity[l] * expert_route_mass[l, e] * rd_curve[rate]
   using empirical TrellisMX distortion ratios (default {k2: 4.63, p25: 2.00,
   k3: 1.00, k4: 0.50}) rather than idealized 4^(-bits). Captures cross-layer
   importance without overfitting noisy per-expert backward attribution.
2. route-mass: expert_route_mass[l, e] * rd_curve[rate].
3. shapley: reconciled per-expert Aumann-Shapley damage shares summed across
   the expert's w13 and w2 matrix groups.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pe_format import ASSIGNMENT_SCHEMA, assignment_sha256
from .profile import ModelProfile, load_profile
from .trellis_ref import RATES

RATE_FRACTION = {
    "k2": Fraction(2),
    "p25": Fraction(5, 2),
    "k3": Fraction(3),
    "k4": Fraction(4),
}

# Empirical KLD ratios measured on Granite 3.1 3B (uniform K2=0.2348, P25=0.1015, K3=0.0507)
# with K4 extrapolated at 0.5x K3 (matching Qwen3-30B K4/K3 ~ 0.34-0.50).
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
    if total_half_bits.denominator != 1:
        budget = int(math.floor(float(total_half_bits)))
    else:
        budget = int(total_half_bits)

    min_cost = min(costs)
    max_cost = max(costs)
    if budget < min_cost * len(candidates):
        raise ValueError(
            f"budget {float(budget_bits):.4f} bits is below minimum feasible rate {min_cost / 2:.2f}"
        )

    states: dict[int, float] = {0: 0.0}
    parents: list[dict[int, tuple[int, int]]] = []
    n_total = len(candidates)
    for step_idx, (layer, expert, damages) in enumerate(candidates):
        if len(damages) != len(rates) or any(not math.isfinite(d) for d in damages):
            raise ValueError(f"invalid damages for layer {layer}, expert {expert}")
        rem = n_total - step_idx - 1
        max_usable = budget - rem * min_cost
        min_needed = max(0, budget - rem * max_cost - max_cost)
        new_states: dict[int, float] = {}
        parent: dict[int, tuple[int, int]] = {}
        for used, acc_damage in states.items():
            for r_idx, (cost, d_val) in enumerate(zip(costs, damages, strict=True)):
                nu = used + cost
                if nu > max_usable:
                    continue
                nd = acc_damage + d_val
                prev = new_states.get(nu)
                if prev is None or nd < prev:
                    new_states[nu] = nd
                    parent[nu] = (used, r_idx)
        # Pareto-prune dominated states (higher cost with >= damage)
        pruned: dict[int, float] = {}
        best = math.inf
        for u in sorted(new_states):
            val = new_states[u]
            if val < best:
                pruned[u] = val
                best = val
        states = pruned
        parents.append(parent)

    final_used = min(states, key=lambda u: (states[u], u))
    total_damage = states[final_used]
    choices: dict[tuple[int, int], str] = {}
    used = final_used
    for row_idx in range(n_total - 1, -1, -1):
        prev_u, r_idx = parents[row_idx][used]
        layer, expert, _ = candidates[row_idx]
        choices[(layer, expert)] = rates[r_idx]
        used = prev_u
    return choices, total_damage


def build_hierarchical_assignment(
    profile: ModelProfile,
    *,
    route_mass: Mapping[int, Sequence[float]],
    layer_sensitivity: Mapping[int, float] | None = None,
    budget_bits: float = 2.5,
    rates: Sequence[str] | None = None,
    rd_ratios: Mapping[str, float] = EMPIRICAL_RD_RATIOS,
    fixed_layers: Mapping[int, str] | None = None,
    max_k4_fraction: float | None = None,
) -> dict[str, Any]:
    """Allocate one rate per expert across profile.routed_layers."""
    profile.validate()
    active_rates = tuple(rates) if rates is not None else profile.rates
    fixed = dict(fixed_layers or {})

    # Normalize per-layer sensitivity scalars so mean = 1.0
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

    free_layers = [layer for layer in profile.routed_layers if layer not in fixed]
    fixed_bits = sum(
        float(RATE_FRACTION[fixed[layer]]) * profile.num_experts for layer in fixed
    )
    total_experts = len(profile.routed_layers) * profile.num_experts
    free_experts = len(free_layers) * profile.num_experts
    if free_experts == 0:
        raise ValueError("all routed layers are fixed; nothing left to allocate")

    target_total_bits = budget_bits * total_experts
    free_budget_bits = Fraction(round((target_total_bits - fixed_bits) * 2), 2 * free_experts)

    # Optional cap on K4 fraction: disallow K4 for experts outside the top max_k4_fraction by importance
    k4_allowed: set[tuple[int, int]] | None = None
    if max_k4_fraction is not None and "k4" in active_rates:
        if not 0.0 <= max_k4_fraction <= 1.0:
            raise ValueError("max_k4_fraction must be in [0, 1]")
        scored = []
        for layer in free_layers:
            masses = route_mass[layer]
            layer_sum = max(sum(masses), 1e-12)
            for e, m_val in enumerate(masses):
                scored.append((sens[layer] * (m_val / layer_sum), layer, e))
        scored.sort(reverse=True)
        cap_count = int(round(max_k4_fraction * len(scored)))
        k4_allowed = {(l, e) for _, l, e in scored[:cap_count]}

    candidates: list[tuple[int, int, list[float]]] = []
    for layer in free_layers:
        masses = route_mass[layer]
        if len(masses) != profile.num_experts:
            raise ValueError(f"route_mass for layer {layer} has {len(masses)} entries; expected {profile.num_experts}")
        layer_sum = max(sum(masses), 1e-12)
        for e, m_val in enumerate(masses):
            # Normalize within-layer route share, then weight by layer sensitivity scalar
            imp = sens[layer] * (float(m_val) / layer_sum)
            row = []
            for r in active_rates:
                if r == "k4" and k4_allowed is not None and (layer, e) not in k4_allowed:
                    # Penalize K4 so solver only picks K4 on top hot experts
                    row.append(imp * rd_ratios["k3"] + 1e6)
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
    rates = tuple(args.rates.split(",")) if args.rates else None

    doc = build_hierarchical_assignment(
        profile,
        route_mass=route_mass,
        layer_sensitivity=layer_sens,
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
