# TrellisMX Sub-3-Bit Per-Expert Quantization and Host-RAM Engram QEC Kit

Model-neutral toolkit, format specifications, kernel patches, and empirical KLD
benchmarks for compressing large Mixture-of-Experts (MoE) language models below
3 bits per routed-expert weight without expert weight offloading to host RAM.
GLM-5.3-Flash targets are quantized directly from `zai-org/GLM-5.3-Flash-BF16`,
with non-routed modules kept in BF16 as stored.

Target deployment envelope: fitting 1T+ parameter MoE checkpoints onto four
96 GB GPUs (382 GiB VRAM) with 512 GB host DDR5 while preserving decode speed.

## Three-layer recipe

1. **Per-expert four-rate TrellisMX codec (`k2`, `p25`, `k3`, `k4`).**
   Every routed expert receives one shared rate across both `w13` (gate and up)
   and `w2` (down), selected under an exact byte budget by a two-level
   hierarchical allocator (`layer_sensitivity[l] * expert_route_mass[l, e] *
   rd_ratio[rate]`, with optional `k4` cap for top hot experts) or reconciled
   Aumann-Shapley shares. Encoding uses standard local GPTQ column-block error
   feedback inside the coupled Hadamard serving boundary (`procedural-npk`
   zero-storage E4M3 state law, `compander_scale = 2.0`, 5-pass continuous-to-
   UE8M0 scale refit, tail-biting Viterbi search, lossless CSF scale planes).
   YAQA-style two-sided Fisher rounding is excluded (measured 1% to 6% worse
   on Granite 3.1 3B).
2. **Mergeable non-expert QEC and zero-VRAM codebook recalibration.**
   With all quantized MoE trellis indices, routers, and sparse attention
   indexers permanently frozen, distillation trains:
   - Existing `intermediate_scales_fp16` (`[L, E, 3, I_local]`) continuous
     per-neuron codebook gains around SwiGLU (folds directly back into the
     sidecar tensors for zero extra VRAM, zero extra FLOPs, and zero kernel
     changes).
   - Rank-32 LoRA adapters on BF16/FP8 attention projections (supporting hybrid
     KDA/MLA topologies), optional dense prefix MLP layers, and shared experts,
     plus FP32-promoted text-backbone RMSNorm weights (all attention/dense/shared
     LoRA deltas merge algebraically back into base weights).
   - Compact per-expert output centroid biases (`[L, E, H]`, ~0.0026 bpw VRAM
     on GLM-5.3 Flash) correcting asymmetric SwiGLU Jensen drift.
3. **Dual-Trellis Host-RAM Engram QEC sidecar (zero GPU expert VRAM, O(1) deterministic prefetch).**
   Three shared cross-layer host-RAM depth banks (early, mid, late; 3x less
   host RAM and 4x fewer DDR5 lookups than per-layer tables) combine raw
   unigrams with zero-initialized residual prime-hashed bigram, trigram, and
   skip-gram tables over NFKC-canonicalized token IDs, per-order 8-group
   context gating with optional router-aware gating, and a width-4 causal
   depthwise Conv1D shift-register residual. Hash addresses depend exclusively
   on input token IDs `(t_{i-2}, t_{i-1}, t_i)`, allowing deterministic
   prefetch from host DDR5 before Layer 0 executes.

## Repository layout

| Path | Content | Status |
|---|---|---|
| `configs/*.json` | Model profiles (`granite-3.1-3b-a800m`, `glm-5.3-flash`, `qwen3-30b-a3b`, `template-1t-moe`) | implemented |
| `trellismx_kit/profile.py` | `ModelProfile` schema and validator for arbitrary MoE geometries | implemented |
| `trellismx_kit/trellis_ref.py` | Standalone `procedural-npk` packer and decoder for `k2`, `p25`, `k3`, `k4` | implemented |
| `trellismx_kit/pe_format.py` | Four-bank rank sidecar writer/reader, lossless CSF scales, device layout | implemented |
| `trellismx_kit/allocate.py` | Exact multiple-choice DP allocator (hierarchical, route-mass, K4 cap) | implemented |
| `trellismx_kit/stratified_capture.py` | Calibration window selector enforcing rare-expert routed-token floors | implemented |
| `trellismx_kit/layer_sensitivity.py` | Fast per-layer KLD perturbation sensitivity probe | implemented |
| `trellismx_kit/engram_qec.py` | Mergeable Attention LoRA + context-gated Host-RAM Engram QEC | implemented |
| `trellismx_kit/engram_probe.py` | Training-free n-gram predictability (`R^2`, NMSE) diagnostic | implemented |
| `trellismx_kit/kld_score.py` | Exact FP64 full-vocabulary KL scorer and window-block bootstrap CI | implemented |
| `trellismx_kit/vram_budget.py` | GPU VRAM and host-RAM sizing calculator | implemented |
| `scripts/distill_qec.py` | Model-neutral DDP CLI for Full-Stack QEC (80/20 short/long curriculum, live or sparse shards) | implemented |
| `scripts/dump_teacher_logits.py` | Forward-only dumper for dense eval logits and sparse top-K + logsumexp shards | implemented |
| `design/` | Format, four-rate P8 kernel, and Engram QEC specifications | implemented |
| `patches/` | `b12x-per-expert.patch` (qualified on SM120 synthetic layers for `{k2, p25, k3, k4}`), `kquant-pair.patch`, `flare-per-expert.patch` | SM120 kernel verified |
| `results/` | Machine-readable JSON summary of Granite 3.1 3B empirical sweeps | qualified on Granite 3B |
| `RESULTS.md` | Full Granite 3.1 3B empirical measurements (per-expert, zero-train LUT, 10M and 50M QEC) | qualified on Granite 3B |
| `AGENT_RUNBOOK.md` | Step-by-step workflow for quantizing an arbitrary MoE checkpoint | implemented |
| `LAMBDA_100M_PLAN.md` | Complete 100M-token GLM-5.3 Flash Lambda execution plan targeting DGX Spark 128 GB | implemented |
| `tests/test_kit.py` | CPU unit and invariant test suite (`pytest`) | implemented |

## Quick check

```bash
pytest -v tests/test_kit.py
python3 -m trellismx_kit.vram_budget --profile configs/template-1t-moe.json --avg-expert-bits 2.25 --gpus 4
```

## Credits and precedents

Built on prior open research and software: IBM Granite 3.1 3B-A800M-Instruct
(test model), NVIDIA Nemotron and Nebius SWE-agent datasets (calibration and
evaluation text), ExLlamaV3 / turboderp (EXL3 trellis lineage), QTIP
(`arXiv:2406.11235`), Q-Palette (`arXiv:2509.20214`), DeepSeek Engram
(`arXiv:2601.07372`), Mixture of Lookup Experts (`arXiv:2503.15798`), MC-MoE
(`arXiv:2410.06270`), GEMQ (`arXiv:2605.23078`), Q-Strata (`arXiv:2608.30564`),
Hill (`arXiv:2607.12266`), and the Local Inference Lab b12x kernel contributors.
