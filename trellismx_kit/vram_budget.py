"""Exact GPU VRAM and host-RAM sizing calculator for any MoE ModelProfile."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .profile import ModelProfile, load_profile

GIB = float(1 << 30)


@dataclass(frozen=True)
class MemoryEstimate:
    routed_params: int
    non_routed_params: int
    routed_payload_gib: float
    csf_scales_and_vectors_gib: float
    non_routed_vram_gib: float
    engram_gpu_proj_gib: float
    total_vram_weights_gib: float
    per_gpu_vram_weights_gib: float
    kv_headroom_per_gpu_gib: float
    engram_host_ram_gib: float


def estimate_memory(
    profile: ModelProfile,
    *,
    avg_expert_bits: float = 2.5,
    csf_scale_overhead_bits: float = 0.04,
    coupled_vector_overhead_bits: float = 0.02,
    non_routed_bits: float = 8.0,
    embed_in_host_ram: bool = True,
    gpu_count: int = 4,
    vram_per_gpu_gib: float = 95.59,  # 97,887 MiB (RTX PRO 6000 Blackwell)
    runtime_overhead_per_gpu_gib: float = 4.0,
    engram_layers: int = 8,
    engram_d_uni: int = 128,
    engram_bi_heads: int = 4,
    engram_tri_heads: int = 2,
    engram_d_head: int = 64,
    engram_buckets: int = 2_097_143,
    engram_table_bits: float = 8.0,
) -> MemoryEstimate:
    profile.validate()
    routed_w = profile.total_routed_weights
    non_routed_w = profile.non_routed_params
    embed_params = 2 * profile.vocab_size * profile.hidden_size if embed_in_host_ram else 0
    resident_non_routed = max(0, non_routed_w - embed_params)

    payload_bytes = routed_w * avg_expert_bits / 8.0
    overhead_bytes = routed_w * (csf_scale_overhead_bits + coupled_vector_overhead_bits) / 8.0
    non_routed_bytes = resident_non_routed * non_routed_bits / 8.0

    d_cat = engram_d_uni + (engram_bi_heads + engram_tri_heads) * engram_d_head
    engram_gpu_proj_bytes = engram_layers * (2 * d_cat * profile.hidden_size) * 2.0  # BF16 W_k, W_v
    engram_table_params = engram_layers * (
        profile.vocab_size * engram_d_uni
        + (engram_bi_heads + engram_tri_heads) * engram_buckets * engram_d_head
    )
    engram_host_bytes = engram_table_params * engram_table_bits / 8.0

    total_vram_gib = (payload_bytes + overhead_bytes + non_routed_bytes + engram_gpu_proj_bytes) / GIB
    per_gpu_gib = total_vram_gib / gpu_count
    kv_headroom = vram_per_gpu_gib - runtime_overhead_per_gpu_gib - per_gpu_gib

    return MemoryEstimate(
        routed_params=routed_w,
        non_routed_params=non_routed_w,
        routed_payload_gib=payload_bytes / GIB,
        csf_scales_and_vectors_gib=overhead_bytes / GIB,
        non_routed_vram_gib=non_routed_bytes / GIB,
        engram_gpu_proj_gib=engram_gpu_proj_bytes / GIB,
        total_vram_weights_gib=total_vram_gib,
        per_gpu_vram_weights_gib=per_gpu_gib,
        kv_headroom_per_gpu_gib=kv_headroom,
        engram_host_ram_gib=engram_host_bytes / GIB,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GPU VRAM and host-RAM budgets for an MoE profile.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--avg-expert-bits", type=float, default=2.5)
    parser.add_argument("--non-routed-bits", type=float, default=8.0)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--vram-per-gpu-gib", type=float, default=95.59)
    parser.add_argument("--engram-layers", type=int, default=8)
    parser.add_argument("--engram-buckets", type=int, default=2_097_143)
    parser.add_argument("--engram-table-bits", type=float, default=8.0)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    est = estimate_memory(
        profile,
        avg_expert_bits=args.avg_expert_bits,
        non_routed_bits=args.non_routed_bits,
        gpu_count=args.gpus,
        vram_per_gpu_gib=args.vram_per_gpu_gib,
        engram_layers=args.engram_layers,
        engram_buckets=args.engram_buckets,
        engram_table_bits=args.engram_table_bits,
    )
    print(json.dumps(est.__dict__, indent=2))


if __name__ == "__main__":
    main()
