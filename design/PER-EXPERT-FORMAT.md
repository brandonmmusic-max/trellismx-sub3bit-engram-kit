# Four-bank per-expert TrellisMX rank sidecar format

Status: implemented in `trellismx_kit/pe_format.py` and `trellismx_kit/trellis_ref.py`.

## Purpose

A uniform TrellisMX sidecar stores one bit rate for every expert of a layer.
This format stores one shared bit rate per routed expert chosen from four
rates: `k2` (2.0 bits), `p25` (2.5 bits, pair rate), `k3` (3.0 bits), and `k4`
(4.0 bits). Both matrix carriers of an expert (`w13`: gate and up projections;
`w2`: down projection) share that expert's assigned rate. All dimensions derive
from a `ModelProfile` JSON (`hidden_size`, `intermediate_size`, `tp_size`,
`num_experts`).

## Rates and procedural NPK state law

Every rate decodes 256-weight tiles directly to FP8 (E4M3) operands using the
zero-storage procedural NPK (Neural Processing of K) 16-bit state law with
`compander_scale = 2.0`:

| Rate | Code | Bits per weight | int16 words per tile | uint32 words per tile | Stream layout |
|---|---:|---:|---:|---:|---|
| `k2` | 0 | 2.0 | 32 | 16 | 256 symbols of 2 bits |
| `p25` | 1 | 2.5 | 40 | 20 | 128 symbols of 5 bits; two operands per 16-bit state |
| `k3` | 2 | 3.0 | 48 | 24 | 256 symbols of 3 bits |
| `k4` | 3 | 4.0 | 64 | 32 | 256 symbols of 4 bits |

For `p25`, symbol `m` occupies bits `[5m, 5m + 5)` of the tail-biting ring, its
16-bit state `s` yields operand `2m = NPK(s)` and operand
`2m + 1 = NPK((s * 1664525 + 1013904223) & 0xFFFF)`. Any sidecar recording
different pair constants or a different codec law string (`procedural-npk`) is
refused by the reader.

## Sidecar tensors and metadata

Schema string: `trellismx-p8-coupled-per-expert-rank.v2`.

Let `E = profile.num_experts`, `H = profile.hidden_size`,
`I_loc = profile.local_intermediate_size`, `HT = H / 16`, `IT = I_loc / 16`,
and `E_r` be the count of experts assigned to rate `r` in the layer.

| Tensor | Dtype and shape | Condition |
|---|---|---|
| `expert_rate_u8` | uint8 `[E]` | Always present; values in `{0, 1, 2, 3}` |
| `expert_slot_u16` | uint16 `[E]` | Always present; rank of expert within its rate bank |
| `w13_trellis_<r>` | int16 `[2, E_r, HT, IT, TILE_I16[r]]` | Present iff `E_r > 0` |
| `w2_trellis_<r>` | int16 `[E_r, IT, HT, TILE_I16[r]]` | Present iff `E_r > 0` |
| `w13_scale_ue8m0` | uint8 `[E, 2 * I_loc, H / 32]` | Plain storage (`scale_storage = plain`) |
| `w2_scale_ue8m0` | uint8 `[E, H, I_loc / 32]` | Plain storage (`scale_storage = plain`) |
| `<p>_scale_csf_*` | base, bitplane, exceptions, exc_index | Lossless CSF storage (`scale_storage = csf`) |
| Coupled vectors | `gate_up_suh_fp16`, `down_svh_fp16`, `intermediate_scales_fp16`, `coupled_sign_draw_u8` | Global expert order |

## Lossless CSF scale planes

When `scale_storage = csf`, the writer compresses each UE8M0 plane into a
per-row optimal adjacent-pair `[b, b + 1]` base (`_optimal_pair_base`), 1-bit
offset bitplane, and exact exception list, then expands it immediately and
verifies bitwise equality before writing. The metadata records `sha256_w13_scale_ue8m0` and `sha256_w2_scale_ue8m0` of the
plain planes. The reader verifies every stored tensor hash, expands CSF planes,
and verifies the expanded bytes against the plain-plane hashes.

## Flattened device layout

At load time, the runtime concatenates non-empty banks in `(k2, p25, k3, k4)`
order into flat int32 streams (`w13_stream` per half and `w2_stream`) plus two
tables indexed by global expert ID: `expert_rate[e]` and `expert_base_u32[e]`.
Tile `t` of expert `e` begins at `expert_base_u32[e] + t * TILE_U32[rate]`.
