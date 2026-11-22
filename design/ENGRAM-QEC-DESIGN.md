# Full-Stack Post-Quantization Error Correction (QEC)

Status: implemented in `trellismx_kit/engram_qec.py`; validated on Granite 3.1
3B-A800M-Instruct at 2.0, 2.25, and 2.50 stored bits per expert weight.

## Problem

Sub-3-bit trellis quantization compresses routed MoE experts to 2.0 to 2.5 bits
per weight so 1T+ models fit in four 96 GB GPUs without offloading expert
weights to host RAM. At 2.0 to 2.25 bits, weight matrices hit the
distortion-rate floor: static lexical idioms, named entities, syntax tokens,
and global attention calibration drift contribute measurable KL divergence
against the BF16 teacher. Standard Quantization-Aware Distillation (QAD) cannot
differentiate cleanly through discrete tail-biting Viterbi trellis search and
requires terabytes of FP32 master weights and optimizer states for 300B+ MoEs.

## Two-tier frozen-backbone recovery

Every quantized MoE expert weight and router remains permanently frozen in its
exact packed TrellisMX reconstruction. Two lightweight, orthogonal recovery
components train jointly against top-64 teacher KL divergence on a leak-free
text stream:

1. Mergeable non-expert QEC (zero extra VRAM, zero extra inference latency):
   rank-r LoRA adapters on BF16/FP8 attention projections (`q_proj`, `k_proj`,
   `v_proj`, `o_proj`), RMSNorm scale weights, and optional per-expert scalar
   gains. Once distillation finishes, attention LoRA deltas fold algebraically
   into the BF16/FP8 attention weights (`W <- W + (alpha/r) * B @ A`), leaving
   the serving graph and memory footprint unchanged.
2. Host-RAM Engram QEC sidecar (zero extra GPU expert VRAM, O(1) deterministic
   prefetch): context-gated n-gram lookup tables attached as additive residual
   branches at selected layers `l`:
   - Lookup vector `e_t` concatenates a unigram embedding `E_uni[t_i]` with
     optional prime-modulo hashed bigram and trigram embeddings keyed purely by
     `(t_{i-2}, t_{i-1}, t_i)`.
   - Two compact GPU-resident projections `W_k` and `W_v` map `e_t` to a
     context key `k_t` and residual update `v_t` (`W_v` initialized to exact
     zero so step 0 reproduces the quantized baseline identically).
   - Context gate `alpha_t = sigmoid(<RMSNorm(h_t), RMSNorm(k_t)> / sqrt(d) + b)`
     scales `v_t` so hash collisions or mismatched word senses are suppressed.
   - Because hash addresses depend exclusively on input token IDs, the serving
     runtime prefetches all required table rows from host DDR5 before Layer 0
     begins (a few kilobytes per token per layer, hidden behind GPU compute).

## Stability invariants

1. Do not apply RMSNorm directly to the small-std (`0.02`) concatenated lookup
   vector `e_t` before `W_v`; amplifying `e_t` by 50x causes immediate AdamW
   projection divergence. Keep `e_t` unnormalized or scale the projection
   learning rate proportionally.
2. Keep router weights frozen during QEC distillation unless using a very small
   learning rate (`<= 1e-5`) on large token counts; aggressive router updates
   degrade held-out KLD.
3. Screen the distillation stream with a 13-gram overlap check against every
   evaluation window so held-out KLD remains an untouched measurement.
