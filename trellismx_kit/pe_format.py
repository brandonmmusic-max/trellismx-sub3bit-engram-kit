"""Model-neutral four-bank (K2, P25, K3, K4) TrellisMX rank sidecar format.

Supports any MoE geometry declared by ModelProfile, lossless CSF compression of
UE8M0 scale planes with globally optimal per-row [b, b+1] base selection and
bit-exact write/read verification, canonical assignment hashing, and flattened
device stream layout for per-expert P8 fused MoE kernels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .profile import ModelProfile
from .trellis_ref import (
    BITS_PER_WEIGHT,
    CODEC_LAW,
    COMPANDER_SCALE,
    PAIR_MIX,
    RATES,
    RATE_CODE,
    RATE_NAME,
    TILE_I16,
    TILE_U32,
    decode_tile_operands,
)

SIDECAR_SCHEMA = "trellismx-p8-coupled-per-expert-rank.v2"
ASSIGNMENT_SCHEMA = "trellismx.rate-assignment.v2"
CSF_ENCODING = "ue8m0-row-base-offset1-u32-exceptions.v1"
_ACCEPTED_CSF_ENCODINGS = frozenset({CSF_ENCODING, "ue8m0-row-base-offset1-u24-exceptions.v1"})
SCALE_STORAGE = ("plain", "csf")


@dataclass(frozen=True)
class CsfPlane:
    """Lossless row-base + 1-bit delta + uint32 exception table for a UE8M0 plane."""

    row_base: torch.Tensor      # uint8 [rows]
    bitplane: torch.Tensor      # uint8 [rows, ceil(cols / 8)]
    exceptions: torch.Tensor    # uint8 [num_exc]
    exc_index: torch.Tensor     # uint32 [num_exc] (packed row * cols + col)
    rows: int
    cols: int


@dataclass(frozen=True)
class Sidecar:
    path: Path
    layer: int
    rank: int
    metadata: dict[str, str]
    rates: torch.Tensor         # uint8 [E]
    slots: torch.Tensor         # uint16 [E]
    tensors: dict[str, torch.Tensor]

    @property
    def counts(self) -> dict[str, int]:
        return {name: int((self.rates == code).sum()) for name, code in RATE_CODE.items()}


@dataclass(frozen=True)
class DeviceLayout:
    expert_rate: torch.Tensor             # int32 [E]
    w13_stream: tuple[torch.Tensor, torch.Tensor]   # (gate_u32, up_u32)
    w2_stream: torch.Tensor               # down_u32
    w13_base_u32: tuple[torch.Tensor, torch.Tensor] # (gate_base[E], up_base[E])
    w2_base_u32: torch.Tensor             # down_base[E]


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _optimal_pair_base(flat_u8: torch.Tensor) -> torch.Tensor:
    """Choose b in [0, 254] per row maximizing count(b) + count(b + 1)."""
    rows = flat_u8.shape[0]
    hist = torch.zeros((rows, 256), dtype=torch.int32)
    hist.scatter_add_(1, flat_u8.to(torch.int64), torch.ones_like(flat_u8, dtype=torch.int32))
    pair_counts = hist[:, :-1] + hist[:, 1:]
    return pair_counts.argmax(dim=1).to(torch.uint8)


def encode_csf_plane(plane: torch.Tensor) -> CsfPlane:
    """Losslessly compress a 2D or 3D uint8 UE8M0 scale plane."""
    if plane.dtype != torch.uint8:
        raise TypeError("UE8M0 scale plane must be uint8")
    flat = plane.detach().cpu().contiguous().reshape(-1, plane.shape[-1])
    rows, cols = flat.shape
    base = _optimal_pair_base(flat)
    delta = flat.to(torch.int16) - base[:, None].to(torch.int16)
    in_range = (delta == 0) | (delta == 1)
    bits = (delta == 1).to(torch.uint8)
    pad = (8 - (cols % 8)) % 8
    if pad:
        bits = torch.cat((bits, torch.zeros((rows, pad), dtype=torch.uint8)), dim=1)
    bits_bytes = bits.reshape(rows, -1, 8)
    shifts = torch.arange(8, dtype=torch.uint8)
    bitplane = (bits_bytes << shifts).sum(dim=-1, dtype=torch.int16).to(torch.uint8).contiguous()

    exc_mask = ~in_range
    exc_flat_idx = torch.nonzero(exc_mask.reshape(-1), as_tuple=False).squeeze(-1).to(torch.int64)
    if exc_flat_idx.numel() and int(exc_flat_idx.max().item()) >= (1 << 32):
        raise ValueError("plane exceeds uint32 exception addressing limit")
    exceptions = flat.reshape(-1)[exc_flat_idx].to(torch.uint8).contiguous()
    exc_index = exc_flat_idx.to(torch.uint32).contiguous()
    encoded = CsfPlane(
        row_base=base.contiguous(),
        bitplane=bitplane,
        exceptions=exceptions,
        exc_index=exc_index,
        rows=rows,
        cols=cols,
    )
    decoded = decode_csf_plane(encoded, shape=plane.shape)
    if not torch.equal(decoded, plane.detach().cpu()):
        raise RuntimeError("CSF lossless round-trip verification failed")
    return encoded


def decode_csf_plane(csf: CsfPlane, shape: Sequence[int]) -> torch.Tensor:
    """Expand a CsfPlane back to the exact uint8 UE8M0 scale tensor."""
    rows, cols = csf.rows, csf.cols
    if math.prod(shape) != rows * cols:
        raise ValueError(f"target shape {tuple(shape)} disagrees with CSF geometry {(rows, cols)}")
    shifts = torch.arange(8, dtype=torch.uint8)
    unpacked = ((csf.bitplane[..., None] >> shifts) & 1).reshape(rows, -1)[:, :cols]
    out = (csf.row_base[:, None].to(torch.int16) + unpacked.to(torch.int16)).to(torch.uint8).contiguous()
    if csf.exceptions.numel():
        out_flat = out.reshape(-1)
        out_flat[csf.exc_index.to(torch.int64)] = csf.exceptions
    return out.reshape(tuple(shape)).contiguous()


def normalize_assignment_layers(
    layers: Mapping[int | str, Sequence[str]],
    profile: ModelProfile,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for layer_key, rates in layers.items():
        layer_id = int(layer_key)
        values = list(rates)
        if len(values) != profile.num_experts:
            raise ValueError(
                f"layer {layer_id} defines {len(values)} rates; profile expects {profile.num_experts}"
            )
        if any(rate not in profile.rates for rate in values):
            raise ValueError(f"layer {layer_id} contains a rate outside profile.rates {profile.rates}")
        result[str(layer_id)] = values
    if not result:
        raise ValueError("assignment must define at least one layer")
    return result


def assignment_sha256(document: Mapping[str, Any], profile: ModelProfile) -> str:
    body = dict(document)
    body["layers"] = normalize_assignment_layers(body["layers"], profile)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_rank_sidecar(
    path: str | Path,
    *,
    profile: ModelProfile,
    layer: int,
    rank: int,
    expert_rates: Sequence[str],
    w13_banks: Mapping[str, torch.Tensor],
    w2_banks: Mapping[str, torch.Tensor],
    w13_scale_ue8m0: torch.Tensor,
    w2_scale_ue8m0: torch.Tensor,
    coupled_vectors: Mapping[str, torch.Tensor],
    assignment_hash: str,
    scale_storage: str = "csf",
) -> Path:
    """Write and verify one four-bank per-expert rank sidecar file."""
    profile.validate()
    if scale_storage not in SCALE_STORAGE:
        raise ValueError(f"scale_storage must be one of {SCALE_STORAGE}")
    if len(expert_rates) != profile.num_experts:
        raise ValueError("expert_rates length must equal profile.num_experts")
    if any(r not in profile.rates for r in expert_rates):
        raise ValueError("expert_rates contains a rate outside profile.rates")

    rates_u8 = torch.tensor([RATE_CODE[r] for r in expert_rates], dtype=torch.uint8)
    slots_u16 = torch.empty(profile.num_experts, dtype=torch.uint16)
    counts = {r: 0 for r in RATES}
    for e, r in enumerate(expert_rates):
        slots_u16[e] = counts[r]
        counts[r] += 1

    h_tiles = profile.hidden_tiles
    i_tiles = profile.local_intermediate_tiles
    tensors: dict[str, torch.Tensor] = {
        "expert_rate_u8": rates_u8,
        "expert_slot_u16": slots_u16,
    }

    payload_bits = 0
    for r in RATES:
        c = counts[r]
        if c == 0:
            if r in w13_banks or r in w2_banks:
                raise ValueError(f"bank {r!r} must be omitted when expert count is zero")
            continue
        w13 = w13_banks[r].detach().cpu().contiguous()
        w2 = w2_banks[r].detach().cpu().contiguous()
        exp_w13 = (2, c, h_tiles, i_tiles, TILE_I16[r])
        exp_w2 = (c, i_tiles, h_tiles, TILE_I16[r])
        if w13.dtype != torch.int16 or tuple(w13.shape) != exp_w13:
            raise ValueError(f"w13_trellis_{r} expected int16 {exp_w13}, got {w13.dtype} {tuple(w13.shape)}")
        if w2.dtype != torch.int16 or tuple(w2.shape) != exp_w2:
            raise ValueError(f"w2_trellis_{r} expected int16 {exp_w2}, got {w2.dtype} {tuple(w2.shape)}")
        tensors[f"w13_trellis_{r}"] = w13
        tensors[f"w2_trellis_{r}"] = w2
        payload_bits += int(c * profile.local_weights_per_expert * BITS_PER_WEIGHT[r])

    exp_s13 = (profile.num_experts, 2 * profile.local_intermediate_size, profile.hidden_size // 32)
    exp_s2 = (profile.num_experts, profile.hidden_size, profile.local_intermediate_size // 32)
    s13 = w13_scale_ue8m0.detach().cpu().contiguous()
    s2 = w2_scale_ue8m0.detach().cpu().contiguous()
    if s13.dtype != torch.uint8 or tuple(s13.shape) != exp_s13:
        raise ValueError(f"w13_scale_ue8m0 expected uint8 {exp_s13}, got {s13.dtype} {tuple(s13.shape)}")
    if s2.dtype != torch.uint8 or tuple(s2.shape) != exp_s2:
        raise ValueError(f"w2_scale_ue8m0 expected uint8 {exp_s2}, got {s2.dtype} {tuple(s2.shape)}")

    metadata: dict[str, str] = {
        "schema": SIDECAR_SCHEMA,
        "model_name": profile.name,
        "layer": str(layer),
        "rank": str(rank),
        "tp_size": str(profile.tp_size),
        "num_experts": str(profile.num_experts),
        "hidden_size": str(profile.hidden_size),
        "local_intermediate_size": str(profile.local_intermediate_size),
        "rate_set": ",".join(profile.rates),
        "law": CODEC_LAW,
        "compander_scale": str(COMPANDER_SCALE),
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "pair_mix_multiplier": str(PAIR_MIX[0]),
        "pair_mix_increment": str(PAIR_MIX[1]),
        "rate_assignment_sha256": assignment_hash,
        "weight_payload_bits": str(payload_bits),
        "scale_storage": scale_storage,
        "sha256_w13_scale_ue8m0": tensor_sha256(s13),
        "sha256_w2_scale_ue8m0": tensor_sha256(s2),
    }
    for r in RATES:
        metadata[f"experts_{r}"] = str(counts[r])

    if scale_storage == "plain":
        tensors["w13_scale_ue8m0"] = s13
        tensors["w2_scale_ue8m0"] = s2
    else:
        metadata["csf_encoding"] = CSF_ENCODING
        for prefix, plane in (("w13", s13), ("w2", s2)):
            csf = encode_csf_plane(plane)
            tensors[f"{prefix}_scale_csf_base"] = csf.row_base
            tensors[f"{prefix}_scale_csf_bitplane"] = csf.bitplane
            tensors[f"{prefix}_scale_csf_exceptions"] = csf.exceptions
            tensors[f"{prefix}_scale_csf_exc_index"] = csf.exc_index

    for name, value in coupled_vectors.items():
        tensors[name] = value.detach().cpu().contiguous()

    for name, value in tensors.items():
        metadata[f"sha256_{name}"] = tensor_sha256(value)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)
    return out_path


def read_rank_sidecar(path: str | Path, profile: ModelProfile) -> Sidecar:
    """Read, validate hashes/geometry/dtypes, and expand CSF scales for one rank sidecar."""
    profile.validate()
    p = Path(path)
    with safe_open(str(p), framework="pt", device="cpu") as fh:
        metadata = dict(fh.metadata() or {})
        if metadata.get("schema") != SIDECAR_SCHEMA:
            raise ValueError(f"unexpected sidecar schema {metadata.get('schema')!r}")
        if metadata.get("law") != CODEC_LAW:
            raise ValueError(f"unexpected codec law {metadata.get('law')!r}; expected {CODEC_LAW!r}")
        if float(metadata.get("compander_scale", "-1")) != COMPANDER_SCALE:
            raise ValueError(
                f"unexpected compander_scale {metadata.get('compander_scale')!r}; expected {COMPANDER_SCALE}"
            )
        if (
            int(metadata.get("pair_mix_multiplier", -1)) != PAIR_MIX[0]
            or int(metadata.get("pair_mix_increment", -1)) != PAIR_MIX[1]
        ):
            raise ValueError("sidecar pair mix constants disagree with format specification")
        if (
            int(metadata["num_experts"]) != profile.num_experts
            or int(metadata["hidden_size"]) != profile.hidden_size
            or int(metadata["local_intermediate_size"]) != profile.local_intermediate_size
        ):
            raise ValueError("sidecar geometry disagrees with ModelProfile")
        tensors = {key: fh.get_tensor(key) for key in fh.keys()}

    for name, value in tensors.items():
        expected_hash = metadata.get(f"sha256_{name}")
        if expected_hash is None or tensor_sha256(value) != expected_hash:
            raise ValueError(f"SHA-256 verification failed for tensor {name!r}")

    rates_u8 = tensors["expert_rate_u8"]
    slots_u16 = tensors["expert_slot_u16"]
    if rates_u8.dtype != torch.uint8:
        raise ValueError(f"expert_rate_u8 must be uint8, got {rates_u8.dtype}")
    if slots_u16.dtype not in (torch.uint16, torch.int16):
        raise ValueError(f"expert_slot_u16 must be uint16/int16, got {slots_u16.dtype}")
    if tuple(rates_u8.shape) != (profile.num_experts,) or tuple(slots_u16.shape) != (profile.num_experts,):
        raise ValueError("expert rate/slot table shape disagrees with profile.num_experts")
    allowed_codes = {RATE_CODE[r] for r in profile.rates}
    if any(int(c) not in allowed_codes for c in rates_u8.tolist()):
        raise ValueError("expert_rate_u8 contains a rate code outside profile.rates")

    running = {code: 0 for code in RATE_CODE.values()}
    for e in range(profile.num_experts):
        code = int(rates_u8[e].item())
        if int(slots_u16[e].item()) != running[code]:
            raise ValueError(f"expert {e} slot index disagrees with ascending bank order")
        running[code] += 1

    h_tiles = profile.hidden_tiles
    i_tiles = profile.local_intermediate_tiles
    for r, code in RATE_CODE.items():
        cnt = running[code]
        if int(metadata[f"experts_{r}"]) != cnt:
            raise ValueError(f"metadata count for {r} disagrees with expert_rate_u8")
        present_13 = f"w13_trellis_{r}" in tensors
        present_2 = f"w2_trellis_{r}" in tensors
        if (cnt > 0) != present_13 or (cnt > 0) != present_2:
            raise ValueError(f"bank tensor presence for {r} disagrees with count {cnt}")
        if cnt > 0:
            w13 = tensors[f"w13_trellis_{r}"]
            w2 = tensors[f"w2_trellis_{r}"]
            exp_w13 = (2, cnt, h_tiles, i_tiles, TILE_I16[r])
            exp_w2 = (cnt, i_tiles, h_tiles, TILE_I16[r])
            if w13.dtype != torch.int16 or tuple(w13.shape) != exp_w13:
                raise ValueError(f"w13_trellis_{r} shape/dtype mismatch: {w13.dtype} {tuple(w13.shape)}")
            if w2.dtype != torch.int16 or tuple(w2.shape) != exp_w2:
                raise ValueError(f"w2_trellis_{r} shape/dtype mismatch: {w2.dtype} {tuple(w2.shape)}")

    scale_storage = metadata["scale_storage"]
    exp_s13 = (profile.num_experts, 2 * profile.local_intermediate_size, profile.hidden_size // 32)
    exp_s2 = (profile.num_experts, profile.hidden_size, profile.local_intermediate_size // 32)
    if scale_storage == "csf":
        enc = metadata.get("csf_encoding")
        if enc not in _ACCEPTED_CSF_ENCODINGS:
            raise ValueError(f"unsupported csf_encoding {enc!r}")
        for prefix, shape in (("w13", exp_s13), ("w2", exp_s2)):
            rows = shape[0] * shape[1]
            cols = shape[2]
            csf = CsfPlane(
                row_base=tensors.pop(f"{prefix}_scale_csf_base"),
                bitplane=tensors.pop(f"{prefix}_scale_csf_bitplane"),
                exceptions=tensors.pop(f"{prefix}_scale_csf_exceptions"),
                exc_index=tensors.pop(f"{prefix}_scale_csf_exc_index"),
                rows=rows,
                cols=cols,
            )
            expanded = decode_csf_plane(csf, shape=shape)
            if tensor_sha256(expanded) != metadata[f"sha256_{prefix}_scale_ue8m0"]:
                raise ValueError(f"expanded CSF hash mismatch for {prefix}_scale_ue8m0")
            tensors[f"{prefix}_scale_ue8m0"] = expanded
    elif scale_storage == "plain":
        for name, shape in (("w13_scale_ue8m0", exp_s13), ("w2_scale_ue8m0", exp_s2)):
            st = tensors[name]
            if st.dtype != torch.uint8 or tuple(st.shape) != shape:
                raise ValueError(f"{name} shape/dtype mismatch: {st.dtype} {tuple(st.shape)}")
    else:
        raise ValueError(f"unsupported scale_storage {scale_storage!r}")

    return Sidecar(
        path=p,
        layer=int(metadata["layer"]),
        rank=int(metadata["rank"]),
        metadata=metadata,
        rates=rates_u8,
        slots=slots_u16.to(torch.uint16),
        tensors=tensors,
    )


def device_layout(sidecar: Sidecar, profile: ModelProfile) -> DeviceLayout:
    """Build flat uint32 streams and base-offset tables for per-expert P8 kernels."""
    n_tiles_carrier = profile.hidden_tiles * profile.local_intermediate_tiles
    gate_chunks: list[torch.Tensor] = []
    up_chunks: list[torch.Tensor] = []
    down_chunks: list[torch.Tensor] = []
    gate_bank_offset: dict[int, int] = {}
    up_bank_offset: dict[int, int] = {}
    down_bank_offset: dict[int, int] = {}
    g_cursor = u_cursor = d_cursor = 0

    for r in RATES:
        code = RATE_CODE[r]
        key13 = f"w13_trellis_{r}"
        key2 = f"w2_trellis_{r}"
        if key13 not in sidecar.tensors:
            continue
        w13 = sidecar.tensors[key13]
        w2 = sidecar.tensors[key2]
        g_u32 = w13[0].contiguous().view(torch.int32).reshape(-1)
        u_u32 = w13[1].contiguous().view(torch.int32).reshape(-1)
        d_u32 = w2.contiguous().view(torch.int32).reshape(-1)
        gate_bank_offset[code] = g_cursor
        up_bank_offset[code] = u_cursor
        down_bank_offset[code] = d_cursor
        gate_chunks.append(g_u32)
        up_chunks.append(u_u32)
        down_chunks.append(d_u32)
        g_cursor += int(g_u32.numel())
        u_cursor += int(u_u32.numel())
        d_cursor += int(d_u32.numel())

    if max(g_cursor, u_cursor, d_cursor) >= (1 << 31):
        raise ValueError("flattened stream exceeds signed int32 word addressing limit")

    gate_base = torch.empty(profile.num_experts, dtype=torch.int32)
    up_base = torch.empty(profile.num_experts, dtype=torch.int32)
    down_base = torch.empty(profile.num_experts, dtype=torch.int32)
    for e in range(profile.num_experts):
        code = int(sidecar.rates[e].item())
        slot = int(sidecar.slots[e].item())
        stride = n_tiles_carrier * TILE_U32[RATE_NAME[code]]
        gate_base[e] = gate_bank_offset[code] + slot * stride
        up_base[e] = up_bank_offset[code] + slot * stride
        down_base[e] = down_bank_offset[code] + slot * stride

    return DeviceLayout(
        expert_rate=sidecar.rates.to(torch.int32),
        w13_stream=(torch.cat(gate_chunks), torch.cat(up_chunks)),
        w2_stream=torch.cat(down_chunks),
        w13_base_u32=(gate_base, up_base),
        w2_base_u32=down_base,
    )


def decode_expert_operands_from_layout(
    layout: DeviceLayout,
    profile: ModelProfile,
    expert: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one expert's (gate, up, down) E4M3 operand tiles from DeviceLayout."""
    rate = RATE_NAME[int(layout.expert_rate[expert].item())]
    u32_per_tile = TILE_U32[rate]
    n_tiles = profile.hidden_tiles * profile.local_intermediate_tiles
    words = n_tiles * u32_per_tile
    g_base = int(layout.w13_base_u32[0][expert].item())
    u_base = int(layout.w13_base_u32[1][expert].item())
    d_base = int(layout.w2_base_u32[expert].item())
    g_tiles = layout.w13_stream[0][g_base : g_base + words].view(torch.int16).reshape(n_tiles, TILE_I16[rate])
    u_tiles = layout.w13_stream[1][u_base : u_base + words].view(torch.int16).reshape(n_tiles, TILE_I16[rate])
    d_tiles = layout.w2_stream[d_base : d_base + words].view(torch.int16).reshape(n_tiles, TILE_I16[rate])
    return (
        decode_tile_operands(g_tiles, rate),
        decode_tile_operands(u_tiles, rate),
        decode_tile_operands(d_tiles, rate),
    )


def synthetic_rank_sidecar(
    path: str | Path,
    profile: ModelProfile,
    *,
    layer: int = 0,
    rank: int = 0,
    expert_rates: Sequence[str] | None = None,
    scale_storage: str = "csf",
    seed: int = 20261121,
) -> Path:
    """Generate a valid random four-bank rank sidecar for testing any ModelProfile."""
    gen = torch.Generator().manual_seed(seed)
    if expert_rates is None:
        rates_list = [profile.rates[i % len(profile.rates)] for i in range(profile.num_experts)]
    else:
        rates_list = list(expert_rates)
    counts = {r: sum(1 for x in rates_list if x == r) for r in RATES}
    w13_banks, w2_banks = {}, {}
    for r in RATES:
        if counts[r] == 0:
            continue
        w13_banks[r] = torch.randint(
            -32768, 32768, (2, counts[r], profile.hidden_tiles, profile.local_intermediate_tiles, TILE_I16[r]),
            generator=gen, dtype=torch.int16,
        )
        w2_banks[r] = torch.randint(
            -32768, 32768, (counts[r], profile.local_intermediate_tiles, profile.hidden_tiles, TILE_I16[r]),
            generator=gen, dtype=torch.int16,
        )
    s13 = torch.randint(120, 122, (profile.num_experts, 2 * profile.local_intermediate_size, profile.hidden_size // 32), generator=gen, dtype=torch.uint8)
    s2 = torch.randint(121, 123, (profile.num_experts, profile.hidden_size, profile.local_intermediate_size // 32), generator=gen, dtype=torch.uint8)
    s13.reshape(-1)[::97] = 135
    s2.reshape(-1)[::89] = 112
    coupled = {
        "gate_up_suh_fp16": (torch.randint(0, 2, (profile.hidden_size,), generator=gen) * 2 - 1).to(torch.float16),
        "down_svh_fp16": (torch.randint(0, 2, (profile.hidden_size,), generator=gen) * 2 - 1).to(torch.float16),
        "intermediate_scales_fp16": (torch.randint(0, 2, (profile.num_experts, 3 * profile.local_intermediate_size), generator=gen) * 2 - 1).to(torch.float16),
        "coupled_sign_draw_u8": torch.zeros(profile.num_experts, dtype=torch.uint8),
    }
    assign_doc = {
        "schema": ASSIGNMENT_SCHEMA,
        "layers": {str(layer): rates_list},
        "rule": "synthetic",
        "budget_bits": sum(BITS_PER_WEIGHT[r] for r in rates_list) / profile.num_experts,
        "source": {"seed": seed},
    }
    ahash = assignment_sha256(assign_doc, profile)
    return write_rank_sidecar(
        path,
        profile=profile,
        layer=layer,
        rank=rank,
        expert_rates=rates_list,
        w13_banks=w13_banks,
        w2_banks=w2_banks,
        w13_scale_ue8m0=s13,
        w2_scale_ue8m0=s2,
        coupled_vectors=coupled,
        assignment_hash=ahash,
        scale_storage=scale_storage,
    )
