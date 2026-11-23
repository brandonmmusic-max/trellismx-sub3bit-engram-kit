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

## Two-tier frozen-backbone recovery (Dual-Trellis QEC v2)

Every quantized MoE trellis index stream, router, and sparse attention indexer
remains permanently frozen. Two orthogonal recovery tiers train jointly against
chunked, log-space exact tail-calibrated top-128 (or top-64) teacher KL
divergence on a leak-free 80/20 short-to-long text stream:

1. Zero-VRAM codebook recalibration and mergeable non-expert QEC:
   - Fine-tune the existing `intermediate_scales_fp16` tensor (`[E, 3, I_local]`)
     wrapping `gate_proj`, `up_proj`, and `down_proj` around SwiGLU in every
     routed layer (folds directly into existing sidecar FP16 tensors: 0 extra
     VRAM, 0 extra FLOPs, 0 kernel changes).
   - Add a compact per-expert output centroid bias `beta[l, e]` (`[L, E, H]`,
     0.0026 bpw VRAM on GLM-5.3 Flash) to cancel asymmetric SwiGLU Jensen drift.
   - Attach mergeable rank-32 LoRA adapters on attention projections (supporting
     hybrid KDA + MLA topologies via `ModelProfile.projections_for_layer`),
     BF16/FP8 dense prefix MLP layers, and shared experts, plus FP32-promoted
     text-backbone RMSNorm weights. Once distillation finishes, LoRA deltas fold
     algebraically into base weights (`W <- W + (alpha/r) * B @ A`) and LoRA
     hooks detach cleanly.
2. Dual-Trellis Host-RAM Engram QEC sidecar (zero extra GPU expert VRAM, O(1)
   deterministic prefetch):
   - Share 3 cross-layer host-RAM depth banks (early, mid, late) across attached
     layers with per-layer GPU projections, cutting host-RAM footprint 3x and
     DDR5/PCIe transfers 4x while multiplying bucket gradient hits 4x.
   - Combine raw unigram embeddings with zero-initialized residual prime-hashed
     bigram (half raw, half NFKC-canonicalized), trigram, and skip-gram
     `(t_{i-2}, t_i)` tables using distinct prime moduli per head and zero
     embedding weight decay so unseen held-out buckets emit exact zero instead
     of Gaussian noise.
   - Gate each n-gram order independently across `G = 8` channel groups with
     learnable RMSNorm scales and an optional router-aware gate scalar, then
     pass the gated residual stream through a zero-initialized width-4 causal
     depthwise Conv1D + SiLU residual (activation-side shift-register memory
     expanding 3-token context to 6+ tokens for 24 KB/layer decode state).
   - Because all hash addresses depend exclusively on input token IDs, the
     serving runtime prefetches all required table rows from host DDR5 before
     Layer 0 begins (~10 to 20 KB per token across all layers, hidden behind
     Layer 0 GPU compute).

Empirical ablation note: adding a dense rank-64 shared-basis Wiener matrix or
hidden-state MSE pre-alignment hurt held-out KLD on Granite 3.1 3B relative
to the leaner stack above and is excluded from the canonical recipe.

## Context-length curriculum (80/20 short-to-long split)

Host-RAM n-gram tables depend only on local `(t_{i-2}, t_{i-1}, t_i)` triples
and saturate fastest on high-throughput 2K to 4K sequences. Mergeable Attention
LoRA and Engram context gates, however, must also calibrate across long-context
attention distributions, especially on architectures with sparse indexed
attention (for example, `index_topk = 2048` on GLM-5.3 Flash) where behavior
transitions from dense attention to sparse top-K key retrieval past 2K tokens.

Recommended 35M to 50M token distillation split:

1. 80% short-context sequences (2K to 4K tokens): maximizes tokens/sec during
   backward and saturates prime-hashed bigram and trigram buckets.
2. 20% long-context sequences (16K to 32K tokens, plus an optional 1M to 2M
   token 64K tail when activation checkpointing fits comfortably in VRAM):
   calibrates Attention LoRA (`q`, `k`, `v`, `o`) and Engram context gates in
   the sparse long-context retrieval regime.

Because forward-only teacher prefill handles 16K to 64K sequences at full speed
and sparse top-K shards (`int32` token_ids + `int32` indices + shifted `fp16`
`top_logp` + `fp32` exact `p_tail` and `logsumexp`) cost 396 bytes/token
(top-64, ~19.8 GB per 50M tokens) or 780 bytes/token (top-128, ~39.0 GB per
50M tokens) at any context length, pre-dumping both short and long distillation
shards plus untouched 8K and 16K to 64K dense evaluation windows takes a single
forward-only pass.

## Stability invariants

1. Do not apply RMSNorm directly to the small-std (`0.02`) concatenated lookup
   vector `e_t` before `W_v`; amplifying `e_t` by 50x causes immediate AdamW
   projection divergence. Keep `e_t` unnormalized or scale the projection
   learning rate proportionally.
2. Promote trainable text-backbone RMSNorm weights to FP32 during distillation;
   BF16 ULP near 1.0 (`0.0078`) exceeds typical `3e-4` AdamW steps by more than
   10x and otherwise causes silent update stagnation.
3. Keep router weights and sparse attention indexer weights permanently frozen
   during QEC distillation so routing decisions and long-context key selection
   cannot destabilize.
4. Screen every short and long distillation sequence with a 13-gram overlap
   check against every evaluation window so held-out KLD remains an untouched
   measurement.
5. Store shifted `top_logp = top_logits - logsumexp` in FP16 plus exact FP32
   `p_tail` (never raw FP16 `top_logits` subtracted against FP32 `logsumexp`),
   and compute `log_q_tail` in chunked log-space via masked `logsumexp` so 32K
   to 64K sequences never OOM or suffer FP32 tail cancellation.
