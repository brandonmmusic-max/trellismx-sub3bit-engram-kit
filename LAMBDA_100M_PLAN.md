# GLM-5.3 Flash 100M-Token Dual-Trellis QEC v2 Plan (8x H200 / 8x B200)

End-to-end execution plan for running a 100M-token Dual-Trellis QEC v2
distillation campaign on `zai-org/GLM-5.3-Flash-BF16` on a single 8x H200
(1.13 TiB VRAM) or 8x B200 (1.44 TiB VRAM) node, targeting near-4-bit KLD
fidelity at a footprint that fits on a single 128 GB DGX Spark (~119.2 GiB
unified memory) with zero expert offloading.

## 1. DGX Spark (128 GB) footprint sizing for GLM-5.3 Flash

GLM-5.3 Flash contains 42 main routed MoE layers (304.41B routed weights, 288
experts/layer, hidden 4096, intermediate 2048) plus 1 optional MTP routed layer
(layer 45, 7.25B weights) and 9.67B non-routed text/multimodal parameters
(18.03 GiB native carrier, or ~10.5 GiB when dense/attention/shared layers use
FP8 and vision/MTP are omitted for text-only serving).

A 3-bank 1,048,573-bucket Dual-Trellis Engram v2 sidecar (154,880 unigrams at
`d=128` plus 8 hash heads of 1,048,573 buckets at `d=64` across 3 shared depth
banks) totals 1.67B host-RAM parameters: **3.11 GiB in BF16** or **1.56 GiB in
FP8/INT8**.

Target total model + sidecar footprint on DGX Spark (119.2 GiB raw unified
memory): **100 to 108 GiB**, leaving **11 to 19 GiB** for OS, CUDA context, and
FP8 KV cache (42 main routed layers + 1.56 GiB FP8 Engram sidecar):

| Routed avg bpw | Routed payload | TrellisMX (expanded 0.25b scales, 18 GiB carrier) | TrellisMX (in-VRAM CSF, 10.5 GiB carrier) | EXL3/TR3 (18 GiB carrier) | EXL3/TR3 (10.5 GiB carrier) |
|---:|---:|---:|---:|---:|---:|
| 2.000 | 70.88 GiB | 99.46 GiB | 85.29 GiB | 90.88 GiB | 83.35 GiB |
| 2.250 | 79.73 GiB | 108.32 GiB | 94.15 GiB | 99.74 GiB | 92.21 GiB |
| 2.375 | 84.16 GiB | 112.75 GiB | 98.58 GiB | 104.17 GiB | 96.64 GiB |
| **2.500** | **88.59 GiB** | 117.18 GiB (tight) | **103.01 GiB** | **108.60 GiB** | **101.07 GiB** |
| 2.625 | 93.02 GiB | 121.61 GiB | 107.44 GiB | 113.03 GiB | 105.50 GiB |

Recommended quantization arm for the DGX Spark near-4-bit target:

1. **Primary arm: Per-Expert 2.50 bpw (`{k2, p25, k3, k4}`, 5% to 10% K4 cap)**
   with asymmetric Engram R-squared co-allocation (shifting K3/K4 bits to deep
   contextual layers 10 to 38 while Engram covers early layers 3 to 9 and late
   layers 39 to 44). Fits at **101.1 to 103.0 GiB** on DGX Spark with a slimmed
   10.5 GiB FP8/text-only carrier (16 to 18 GiB left for OS + KV cache), or
   **108.6 GiB** in EXL3/TR3 with the untouched 18.0 GiB multimodal carrier.
2. **Expanded-scale fallback arm: Per-Expert 2.25 to 2.375 bpw** if keeping
   both the untouched 18.0 GiB multimodal carrier and current expanded 0.25 bpw
   UE8M0 scales in VRAM (**108.3 GiB** at 2.25 bpw).

## 2. 100M-token 80/20 corpus and evaluation split

All training tokens must pass a strict 13-gram overlap screen (`<= 1%` overlap)
against every evaluation window before uploading to the node:

| Split | Sequence length | Count | Supervised positions | Storage format | Approx. size |
|---|---:|---:|---:|---|---:|
| `eval_2k` (anchor) | 2,048 | 25 | 51,175 | Dense FP32/BF16 (reusable from existing K4 panel) | 15.5–31.0 GiB |
| `eval_8k` (multi-domain) | 8,192 | 21 | 172,011 | Dense BF16 (`eval_dense/`) | 53.3 GiB |
| `eval_32k` (long-context) | 32,768 | 4 | 131,068 | Dense BF16 (`eval_dense/`) | 40.6 GiB |
| `train_short` (80%) | 4,096 | 19,536 | 80,000,000 | Sparse top-128 (`top_logp` fp16 + `p_tail`/`lse` fp32) | 62.4 GB |
| `train_long` (20%) | 16,384 | 1,221 | 20,000,000 | Sparse top-128 (`top_logp` fp16 + `p_tail`/`lse` fp32) | 15.6 GB |

Total reusable teacher-logit output: **~78 GB** (100M sparse top-128 training
shards) plus **~94 GB** (dense 8K + 32K BF16 eval windows). Using top-64
instead of top-128 halves the sparse training shards to **~39.6 GB** (396 B/tok).

## 3. Execution steps and wall-clock budget (8x H200 / 8x B200)

### Step 0: Pre-flight (~15 min)

1. Verify at least 1.5 TB free local NVMe, clone `trellismx-sub3bit-engram-kit`,
   and install dependencies (`pytest -q tests/test_kit.py` must pass 15/15).
2. Download `zai-org/GLM-5.3-Flash-BF16` (~643 GiB), the quantized per-expert
   student checkpoint (`2.50` or `2.375` bpw), and the pre-screened tokenized
   `.pt` files (`train_short_80m_4k.pt`, `train_long_20m_16k.pt`,
   `eval_8k.pt`, `eval_32k.pt`).
3. Run untouched baseline KLD on `eval_2k`, `eval_8k`, and `eval_32k` to anchor
   the 0.0M starting point before any parameter moves.

### Step 1: One-shot BF16 teacher logit dump (~45 to 60 min)

Run forward-only prefill across all 8 GPUs (TP8, or two independent TP4 workers
sharding the input files in half for ~2x higher throughput than sequential
`device_map="auto"`), saving `eval_dense/`, `train_short/`, and `train_long/`:

```bash
python3 scripts/dump_teacher_logits.py \
  --teacher-model /data/GLM-5.3-Flash-BF16 \
  --eval-tokens /data/eval_8k.pt /data/eval_32k.pt \
  --eval-dtype bfloat16 \
  --short-tokens /data/train_short_80m_4k.pt \
  --long-tokens /data/train_long_20m_16k.pt \
  --topk 128 --short-batch-size 8 --long-batch-size 1 \
  --logit-chunk-tokens 2048 --trust-remote-code --resume \
  --output /data/glm53-teacher-logits-100m
```

Verify shard counts, zero NaN/Inf values, and exact `eval_dense/` shapes, then
**unload the 643 GiB BF16 teacher completely from VRAM** before starting Step 2.

### Step 2: 100M-token Dual-Trellis QEC v2 distillation (~1.5 to 2.5 hr)

With the BF16 teacher unloaded, run TP4 student training (or 2x TP4 DDP replicas
across the 8 GPUs) from the local NVMe sparse top-128 shards using the winning
Dual-Trellis QEC v2 stack:

1. Permanently frozen: all quantized routed MoE trellis streams, all routers,
   all sparse attention indexers (`index_topk = 2048`), and any vision tower
   parameters.
2. Zero-VRAM codebook recalibration: existing `intermediate_scales_fp16`
   (`[42, 288, 3, 2048]` = 74.3M continuous per-neuron gains, `lr=6e-4, wd=0.0`).
3. Tiny VRAM centroid correction: per-expert output bias `[42, 288, 4096]`
   (94.5 MiB total, `lr=3e-4, wd=0.0`).
4. Mergeable non-expert LoRA (zero extra VRAM/latency after merge): rank-32
   LoRA (`lr=2.5e-4, wd=1e-2`) on all hybrid attention projections (34 KDA + 11
   MLA-NoPE via `configs/glm-5.3-flash.json`), dense prefix layers 0 to 2, all
   42 shared experts, and untied `lm_head`, plus FP32-promoted text-backbone
   RMSNorm weights (`lr=3e-4, wd=0.0`).
5. Dual-Trellis Host-RAM Engram v2 sidecar (3 shared depth banks: Early
   `{3, 5, 7, 9}`, Mid `{12, 16, 20, 24, 28, 32}`, Late `{36, 39, 42, 44}`;
   `1,048,573` distinct prime buckets per head; raw unigram + 4 bigram [2 raw,
   2 NFKC] + 2 trigram + 2 skip-gram; zero-init residual hash tables at
   `lr=1.5e-3, wd=0.0`; per-order 8-group context gating + router-aware gate
   scalar; width-4 causal DWConv1D; `lr_proj=4e-4, wd=1e-2`).
6. Loss and schedule: chunked log-space exact tail-calibrated `topk=128` KL
   loss (`chunk_tokens=512`), 80/20 interleaved 4K/16K curriculum, cosine LR
   decay with 5% warmup, gradient checkpointing with `enable_input_require_grads()`.
7. Evaluation and checkpointing milestones: evaluate untouched exact FP64 KLD
   on `eval_2k`, `eval_8k`, and `eval_32k` and save resumable checkpoints at
   **10M, 35M, 50M, 75M, and 100M tokens**.

### Step 3: Merge, post-merge verification, and Hugging Face upload (~20 min)

1. Algebraically merge all attention, dense, shared-expert, and `lm_head` LoRA
   deltas into the base weights and detach all LoRA hooks; fold trained
   `intermediate_scales_fp16` back into the sidecars; export the 3-bank Engram
   sidecar in both BF16 (`3.11 GiB`) and FP8 (`1.56 GiB`).
2. Re-score untouched KLD after merge and verify `|KLD_post_merge - KLD_pre_merge| < 1e-4`
   for both the BF16 and FP8 Engram sidecar exports.
3. Upload to Hugging Face:
   - The sealed quantized student + merged non-expert weights + Engram v2
     sidecar + milestone KLD JSON/markdown report.
   - The reusable 100M sparse top-128 teacher logit shards (`~78 GB`) and dense
     8K/32K evaluation windows (`~94 GB`) for all future local experiments.
