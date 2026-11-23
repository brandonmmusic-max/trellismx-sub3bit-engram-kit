"""Exact GPU VRAM and host-RAM sizing calculator for any MoE ModelProfile.

Explicitly separates on-disk CSF scale compression (~0.06 bpw) from in-VRAM
UE8M0/K32 scale expansion (0.25 bpw in the b12x runtime unless in-VRAM
CSF decode is enabled), keeps lm_head resident in GPU VRAM, charges replicated
projections per GPU, and accounts for MTP routed layers when requested.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from .profile import ModelProfile, load_profile

GIB = float(1 << 30)


@dataclass(frozen=True)
class MemoryEstimate:
    routed_params: int
    non_routed_params: int
    routed_payload_gib: float
    in_vram_scales_and_vectors_gib: float
    on_disk_csf_scales_gib: float
    non_routed_vram_gib: float
    engram_gpu_proj_per_gpu_gib: float
    total_vram_weights_gib: float
    per_gpu_vram_weights_gib: float
    kv_headroom_per_gpu_gib: float
    engram_host_ram_gib: float
    offloaded_embed_host_ram_gib: float
    total_host_ram_gib: float


def estimate_memory(
    profile: ModelProfile,
    *,
    avg_expert_bits: float = 2.5,
    in_vram_csf: bool = False,
    csf_scale_bits: float = 0.0625,
    non_routed_bits: float = 8.0,
    embed_in_host_ram: bool = True,
    include_mtp: bool = True,
    gpu_count: int = 4,
    vram_per_gpu_gib: float = 95.59,  # 97,887 MiB (RTX PRO 6000 Blackwell)
    runtime_overhead_per_gpu_gib: float = 4.0,
    engram_layers: int = 8,
    engram_shared_banks: int = 3,
    engram_d_uni: int = 128,
    engram_bi_heads: int = 4,
    engram_tri_heads: int = 2,
    engram_skip_heads: int = 2,
    engram_d_head: int = 64,
    engram_buckets: int = 524_287,
    engram_table_bits: float = 16.0,
) -> MemoryEstimate:
    profile.validate()
    if gpu_count <= 0:
        raise ValueError(f"gpu_count must be positive; got {gpu_count}")
    if avg_expert_bits <= 0 or non_routed_bits <= 0:
        raise ValueError("bit widths must be positive")

    n_routed_layers = len(profile.routed_layers) + (len(profile.mtp_layers) if include_mtp else 0)
    routed_w = n_routed_layers * profile.num_experts * profile.weights_per_expert
    non_routed_w = profile.non_routed_params

    # Only embed_tokens (1 * V * H, when untied) can live in host RAM; lm_head stays in GPU VRAM.
    can_offload_embed = embed_in_host_ram and not profile.tie_word_embeddings
    offload_embed_params = profile.vocab_size * profile.hidden_size if can_offload_embed else 0
    resident_non_routed = max(0, non_routed_w - offload_embed_params)

    payload_bytes = routed_w * avg_expert_bits / 8.0
    # Plain UE8M0-K32 takes exactly 1 byte per 32 weights = 0.25 bpw when expanded in VRAM.
    in_vram_scale_bits = csf_scale_bits if in_vram_csf else 0.25
    in_vram_scale_bytes = routed_w * in_vram_scale_bits / 8.0
    on_disk_csf_bytes = routed_w * csf_scale_bits / 8.0

    # Exact coupled boundary vectors per layer:
    # - intermediate_scales_fp16: [E, 3 * I] fp16 (sharded across TP)
    # - gate_up_suh_fp16 + down_svh_fp16: 2 * H fp16 (replicated per GPU)
    sharded_coupled_bytes = n_routed_layers * (profile.num_experts * 3 * profile.intermediate_size * 2.0)
    replicated_coupled_per_gpu = n_routed_layers * (2 * profile.hidden_size * 2.0)

    non_routed_bytes = resident_non_routed * non_routed_bits / 8.0
    offload_embed_bytes = offload_embed_params * 2.0  # BF16 host lookup table

    # Engram GPU projections (W_k, W_v, conv, norms) are replicated per GPU unless TP-sharded
    n_hash_heads = engram_bi_heads + engram_tri_heads + engram_skip_heads
    d_cat = engram_d_uni + n_hash_heads * engram_d_head
    engram_gpu_proj_per_gpu = engram_layers * (2 * d_cat * profile.hidden_size + 4 * profile.hidden_size) * 2.0

    # Host-RAM tables use shared depth banks (or 1 bank per layer if engram_shared_banks <= 0)
    n_banks = engram_layers if engram_shared_banks <= 0 else min(engram_layers, engram_shared_banks)
    engram_table_params = n_banks * (
        profile.vocab_size * engram_d_uni
        + n_hash_heads * engram_buckets * engram_d_head
    )
    engram_host_bytes = engram_table_params * engram_table_bits / 8.0

    sharded_total_bytes = payload_bytes + in_vram_scale_bytes + sharded_coupled_bytes + non_routed_bytes
    per_gpu_bytes = (sharded_total_bytes / gpu_count) + replicated_coupled_per_gpu + engram_gpu_proj_per_gpu
    total_vram_bytes = per_gpu_bytes * gpu_count
    kv_headroom = vram_per_gpu_gib - runtime_overhead_per_gpu_gib - (per_gpu_bytes / GIB)

    return MemoryEstimate(
        routed_params=routed_w,
        non_routed_params=non_routed_w,
        routed_payload_gib=payload_bytes / GIB,
        in_vram_scales_and_vectors_gib=(in_vram_scale_bytes + sharded_coupled_bytes + replicated_coupled_per_gpu * gpu_count) / GIB,
        on_disk_csf_scales_gib=on_disk_csf_bytes / GIB,
        non_routed_vram_gib=non_routed_bytes / GIB,
        engram_gpu_proj_per_gpu_gib=engram_gpu_proj_per_gpu / GIB,
        total_vram_weights_gib=total_vram_bytes / GIB,
        per_gpu_vram_weights_gib=per_gpu_bytes / GIB,
        kv_headroom_per_gpu_gib=kv_headroom,
        engram_host_ram_gib=engram_host_bytes / GIB,
        offloaded_embed_host_ram_gib=offload_embed_bytes / GIB,
        total_host_ram_gib=(engram_host_bytes + offload_embed_bytes) / GIB,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GPU VRAM and host-RAM budgets for an MoE profile.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--avg-expert-bits", type=float, default=2.5)
    parser.add_argument("--non-routed-bits", type=float, default=8.0)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--vram-per-gpu-gib", type=float, default=95.59)
    parser.add_argument("--runtime-overhead-gib", type=float, default=4.0)
    parser.add_argument("--in-vram-csf", action="store_true", help="Assume CSF scales stay compressed in VRAM (~0.0625 bpw)")
    parser.add_argument("--no-embed-in-host-ram", action="store_true")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--engram-layers", type=int, default=12)
    parser.add_argument("--engram-shared-banks", type=int, default=3)
    parser.add_argument("--engram-buckets", type=int, default=524_287)
    parser.add_argument("--engram-table-bits", type=float, default=16.0)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    est = estimate_memory(
        profile,
        avg_expert_bits=args.avg_expert_bits,
        in_vram_csf=args.in_vram_csf,
        non_routed_bits=args.non_routed_bits,
        embed_in_host_ram=not args.no_embed_in_host_ram,
        include_mtp=not args.no_mtp,
        gpu_count=args.gpus,
        vram_per_gpu_gib=args.vram_per_gpu_gib,
        runtime_overhead_per_gpu_gib=args.runtime_overhead_gib,
        engram_layers=args.engram_layers,
        engram_shared_banks=args.engram_shared_banks,
        engram_buckets=args.engram_buckets,
        engram_table_bits=args.engram_table_bits,
    )
    print(json.dumps(est.__dict__, indent=2))


if __name__ == "__main__":
    main()
