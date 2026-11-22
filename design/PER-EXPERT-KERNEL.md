# Four-rate per-expert dispatch in P8 fused MoE kernels

Status: design specification for extending b12x P8 grouped and direct owners to
the four-rate set `("k2", "p25", "k3", "k4")`.

## Architecture

A P8 fused MoE layer executes work as tasks where every task owns one global
expert ID and one row tile. Because every expert carries a single shared rate
for both `w13` and `w2`, the decode rate becomes a task-level scalar uniform
across the CTA rather than a compile-time layer constant.

1. Program specialization: `P8ProgramSpec` replaces scalar `trellis_bits` with
   a sorted tuple `rate_set`, such as `("k2", "p25", "k3", "k4")`. Single-rate
   tuples fold away the runtime branch and emit the uniform kernel unchanged.
2. Device tables: each carrier uploads one concatenated int32 trellis stream
   plus `expert_rate[E]` and `expert_base_u32[E]` (interleaved `[gate, up]`
   pairs of length `2 * E` for `w13`). Scale planes and coupled vectors remain
   in global expert order.
3. Shared-memory staging: the B-operand staging buffer is sized for the largest
   `TILE_U32` in `rate_set` (`32` uint32 words per 16x16 tile when `k4` is
   present; `24` when `k3` is the maximum). All downstream shared-memory offsets
   derive from that ceiling so smaller rates read a prefix of the same staging
   buffer without changing layout.
4. Task dispatch: at the top of each task, the shell loads `rate` and both
   carrier base offsets once, pre-offsets each K64 payload and SFB slice base
   prior to invoking the staging helper (preventing loop-carried address
   aliasing on the second `w13` half), and branches once per task into the
   rate-specialized decode core (`k2`, `p25`, `k3`, or `k4`).

## Acceptance gates (single GPU)

Run on synthetic layers matching the target `ModelProfile` geometry with mixed
assignments covering all four rates plus corner cases (all-K2, all-P25, all-K3,
all-K4, and one empty bank):

1. Probe decode: bit-exact E4M3 operand equality against
   `trellismx_kit.pe_format.decode_expert_operands_from_layout` on every tile
   and wrap-around lane at all four rates.
2. Uniform equivalence: per-expert program on an all-K3 or all-K4 assignment
   matches the uniform P8 owner bit-for-bit.
3. Mixed-layer output parity: fused MoE output matches PyTorch serving-format
   reference at row counts 1, 8, 16, 64, and 512 within the uniform K3/K4
   tolerance.
4. Permutation invariance: permuting expert rate assignments without changing
   payloads leaves each individual routed expert's output unchanged.
5. Memory safety: `compute-sanitizer --tool memcheck` reports zero errors on
   decode and mixed-layer tests at row counts 1 and 64.
