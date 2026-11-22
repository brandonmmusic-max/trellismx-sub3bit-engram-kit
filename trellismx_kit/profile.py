"""Model-neutral MoE architecture profile schema and validation.

A ModelProfile JSON describes every geometric parameter required by the
allocator, encoder wrapper, four-bank sidecar builder, QEC distillation loop,
and VRAM calculator so the toolkit applies to any routed MoE checkpoint without
hardcoded layer or expert counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .trellis_ref import RATES

PROFILE_SCHEMA = "trellismx.model-profile.v1"


@dataclass(frozen=True)
class ModelProfile:
    """Geometry and weight-path specification for one MoE model."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    routed_layers: tuple[int, ...]
    num_experts: int
    top_k: int
    vocab_size: int
    tp_size: int = 4
    rates: tuple[str, ...] = ("k2", "p25", "k3", "k4")
    mtp_layers: tuple[int, ...] = ()
    non_routed_params: int = 0
    layer_path_template: str = "model.layers.{layer}"
    moe_attr: str = "mlp"
    attn_attr: str = "self_attn"
    attn_projections: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    expert_layout: str = "per_expert_linear"  # "per_expert_linear" or "stacked_granite"
    logits_scaling: float = 1.0
    activation: str = "silu"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("profile name must be nonempty")
        if self.hidden_size <= 0 or self.hidden_size % 32 != 0:
            raise ValueError(f"hidden_size must be a positive multiple of 32, got {self.hidden_size}")
        if self.intermediate_size <= 0 or self.intermediate_size % 32 != 0:
            raise ValueError(
                f"intermediate_size must be a positive multiple of 32, got {self.intermediate_size}"
            )
        if self.tp_size <= 0 or self.intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"intermediate_size {self.intermediate_size} must divide evenly by tp_size {self.tp_size}"
            )
        if self.local_intermediate_size % 32 != 0:
            raise ValueError(
                f"local_intermediate_size {self.local_intermediate_size} must be a multiple of 32"
            )
        if self.num_layers <= 0 or not self.routed_layers:
            raise ValueError("profile must declare at least one routed layer")
        if any(layer < 0 or layer >= self.num_layers for layer in self.routed_layers):
            raise ValueError("routed_layers contains an index outside [0, num_layers)")
        if len(set(self.routed_layers)) != len(self.routed_layers):
            raise ValueError("routed_layers contains duplicates")
        if self.num_experts <= 0 or self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("num_experts and top_k must satisfy 0 < top_k <= num_experts")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not self.rates or any(rate not in RATES for rate in self.rates):
            raise ValueError(f"rates must be a nonempty subset of {RATES}")
        if len(set(self.rates)) != len(self.rates):
            raise ValueError("rates contains duplicates")
        if self.expert_layout not in ("per_expert_linear", "stacked_granite"):
            raise ValueError(f"unsupported expert_layout {self.expert_layout!r}")
        if self.logits_scaling <= 0:
            raise ValueError("logits_scaling must be positive")

    @property
    def local_intermediate_size(self) -> int:
        return self.intermediate_size // self.tp_size

    @property
    def weights_per_expert(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def local_weights_per_expert(self) -> int:
        return 3 * self.hidden_size * self.local_intermediate_size

    @property
    def total_routed_experts(self) -> int:
        return len(self.routed_layers) * self.num_experts

    @property
    def total_routed_weights(self) -> int:
        return self.total_routed_experts * self.weights_per_expert

    @property
    def hidden_tiles(self) -> int:
        return self.hidden_size // 16

    @property
    def local_intermediate_tiles(self) -> int:
        return self.local_intermediate_size // 16

    @property
    def global_intermediate_tiles(self) -> int:
        return self.intermediate_size // 16

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "name": self.name,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "routed_layers": list(self.routed_layers),
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "vocab_size": self.vocab_size,
            "tp_size": self.tp_size,
            "rates": list(self.rates),
            "mtp_layers": list(self.mtp_layers),
            "non_routed_params": self.non_routed_params,
            "layer_path_template": self.layer_path_template,
            "moe_attr": self.moe_attr,
            "attn_attr": self.attn_attr,
            "attn_projections": list(self.attn_projections),
            "expert_layout": self.expert_layout,
            "logits_scaling": self.logits_scaling,
            "activation": self.activation,
        }


def profile_from_mapping(document: Mapping[str, Any]) -> ModelProfile:
    schema = document.get("schema")
    if schema is not None and schema != PROFILE_SCHEMA:
        raise ValueError(f"unsupported model profile schema {schema!r}")
    profile = ModelProfile(
        name=str(document["name"]),
        hidden_size=int(document["hidden_size"]),
        intermediate_size=int(document["intermediate_size"]),
        num_layers=int(document["num_layers"]),
        routed_layers=tuple(int(x) for x in document["routed_layers"]),
        num_experts=int(document["num_experts"]),
        top_k=int(document["top_k"]),
        vocab_size=int(document["vocab_size"]),
        tp_size=int(document.get("tp_size", 4)),
        rates=tuple(str(x) for x in document.get("rates", ("k2", "p25", "k3", "k4"))),
        mtp_layers=tuple(int(x) for x in document.get("mtp_layers", ())),
        non_routed_params=int(document.get("non_routed_params", 0)),
        layer_path_template=str(document.get("layer_path_template", "model.layers.{layer}")),
        moe_attr=str(document.get("moe_attr", "mlp")),
        attn_attr=str(document.get("attn_attr", "self_attn")),
        attn_projections=tuple(
            str(x) for x in document.get("attn_projections", ("q_proj", "k_proj", "v_proj", "o_proj"))
        ),
        expert_layout=str(document.get("expert_layout", "per_expert_linear")),
        logits_scaling=float(document.get("logits_scaling", 1.0)),
        activation=str(document.get("activation", "silu")),
    )
    profile.validate()
    return profile


def load_profile(path: str | Path) -> ModelProfile:
    return profile_from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
