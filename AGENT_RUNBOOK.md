# Model-Neutral Quantization and QEC Agent Runbook

Step-by-step procedure for quantizing an arbitrary Hugging Face MoE checkpoint
with the per-expert TrellisMX codec and Full-Stack Post-Quantization Error
Correction (QEC).

## Step 1: Author a ModelProfile JSON

Copy the closest template under `configs/` and fill in the target model's exact
`config.json` parameters: `hidden_size`, `intermediate_size`, `num_layers`,
`routed_layers`, `num_experts`, `top_k`, `vocab_size`, `tp_size`,
`non_routed_params`, `layer_path_template`, `moe_attr`, `attn_attr`,
`attn_projections`, and `expert_layout`.

Validate the profile and verify VRAM headroom before allocating GPU hours:

```bash
python3 -m trellismx_kit.vram_budget \
  --profile configs/<model>.json \
  --avg-expert-bits 2.25 \
  --non-routed-bits 8.0 \
  --gpus 4
```

Target at least 10 GiB per GPU of remaining KV-cache headroom after weights and
runtime scratch (4 GiB/GPU). Embeddings should stay in host RAM when the
serving engine supports it.

## Step 2: Build leak-free calibration, selection, and evaluation splits

Prepare three strictly disjoint tokenized splits covering all target domains
(agentic, chat, code, legal, long context, math, multilingual, pretraining,
safety, science, software engineering):

1. Evaluation split: 20+ untouched windows of 8,192 tokens plus 4 to 8 long
   windows of 16,384 to 65,536 tokens. Capture and seal dense full-vocabulary
   BF16 teacher logits once. Never read evaluation tokens during capture,
   allocation, or QEC distillation.
2. Selection split: 8 to 16 windows used only for `layer_sensitivity.py` and
   hyperparameter checks.
3. Calibration and QEC distillation stream: prepare an 80/20 context-length
   split (80% at 2K to 4K tokens for fast n-gram bucket saturation, 20% at 16K
   to 32K tokens with an optional 1M to 2M token 64K tail for long-context
   Attention LoRA and context-gate calibration). Screen every document with a
   13-gram overlap check (`<= 1%` overlap) against all evaluation windows.

When the BF16 teacher exceeds local VRAM, run `scripts/dump_teacher_logits.py`
once on a high-VRAM forward-only node (such as 8x H200 or 8x B200) to emit:

1. Dense full-vocabulary BF16 evaluation windows (`eval_dense/`, e.g. 21
   windows at 8K tokens plus 4 to 8 windows at 32K to 64K tokens).
2. 50M sparse top-128 (or top-64) distillation positions split 80/20: 40M
   tokens at 4,096 sequence length (`train_short/`) and 10M tokens at 16,384 to
   32,768 sequence length (`train_long/`), storing shifted FP16 `top_logp`,
   exact FP32 `p_tail`, and `logsumexp` (396 B/tok at top-64 = ~19.8 GB per 50M
   tokens; 780 B/tok at top-128 = ~39.0 GB per 50M tokens).

Example one-shot teacher logit dump command:

```bash
python3 scripts/dump_teacher_logits.py \
  --teacher-model /path/to/bf16-teacher \
  --eval-tokens data/eval_8k.pt data/eval_32k.pt \
  --eval-dtype bfloat16 \
  --short-tokens data/train_short_40m_4k.pt \
  --long-tokens data/train_long_10m_16k.pt \
  --topk 128 --short-batch-size 8 --long-batch-size 1 \
  --logit-chunk-tokens 2048 --trust-remote-code --resume \
  --output work/teacher-logits
```

## Step 3: Stratified calibration capture and layer sensitivity

Run a forward pass over candidate calibration windows to record per-layer
routing counts and route mass, then invoke
`trellismx_kit.stratified_capture.select_stratified_windows` to select the
encoding capture windows while enforcing a minimum routed-token floor
(recommended: 2,000 tokens per expert) so cold domain-specialized experts get
well-conditioned input Hessians.

Next, run `trellismx_kit.layer_sensitivity.measure_layer_sensitivity` on the
selection split to produce one KLD sensitivity scalar per routed layer
(`layer-sensitivity.json`).

## Step 4: Assign one rate per expert

Run the exact multiple-choice DP allocator over `{k2, p25, k3, k4}`:

```bash
python3 -m trellismx_kit.allocate \
  --profile configs/<model>.json \
  --usage work/usage.json \
  --layer-sensitivity work/layer-sensitivity.json \
  --budget-bits 2.5 \
  --max-k4-fraction 0.10 \
  --output work/rate-assignment.json
```

Guidelines:

1. One shared rate per expert across `w13` and `w2` simplifies serving and
   reduces allocation variance.
2. On models with 128+ experts, cap `k4` at 5% to 10% (`--max-k4-fraction
   0.10`) so the hottest experts get 4-bit precision funded by cold `k2`
   experts.
3. Verify the output JSON's `achieved_bits` and SHA-256 hash before encoding.

## Step 5: Encode each expert once and pack four-bank CSF sidecars

Encode every routed expert once at its assigned rate using the TrellisMX
codec-v2 schedule (coupled Hadamard boundary, 50/50 routed-squared plus
all-token input Hessian, conditional post-SiLU down-projection Hessian, 2
continuous + 3 power-of-two UE8M0 scale refit passes, standard GPTQ column
block feedback, `procedural-npk` tail-biting Viterbi search, bitwise closure
check).

Pack each layer and TP rank into the four-bank sidecar schema via
`trellismx_kit.pe_format.write_rank_sidecar(..., scale_storage="csf")`, then
verify every written file with `read_rank_sidecar` (checks all tensor SHA-256
hashes, bank counts, slot ordering, and lossless CSF expansion).

## Step 6: Full-Stack QEC distillation (80/20 short-to-long curriculum)

Freeze all quantized MoE expert weights, routers, and sparse attention indexers
permanently, then run DDP Full-Stack QEC distillation against either a live
teacher or pre-dumped sparse top-64/top-128 teacher shards using the 80/20
short-to-long context curriculum:

```bash
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 scripts/distill_qec.py \
  --profile configs/<model>.json \
  --teacher-model /path/to/teacher \
  --student-model /path/to/quantized-student \
  --corpus /path/to/screened_short_2048_or_4096.pt \
  --long-corpus /path/to/screened_long_16384_or_32768.pt \
  --long-fraction 0.20 \
  --long-batch-size 1 \
  --grad-checkpointing \
  --engram-layers <6_to_16_spread_layer_ids> \
  --lora-rank 32 \
  --buckets 524287 \
  --tokens-m 50 \
  --output work/qec-artifacts
```

For large MoE models whose BF16 teacher exceeds local VRAM, omit
`--teacher-model` and pass `--teacher-shards /path/to/short_shards_dir` plus
`--long-teacher-shards /path/to/long_shards_dir` produced by
`scripts/dump_teacher_logits.py`.

Recommended token and bucket budgets (35M to 50M base for frontier MoEs):

1. Fast local sanity check / ablation: 10M tokens (already captures >85% of the
   gain on 40-expert models like Granite 3.1 3B, useful for validating loss
   curves before launching a full run).
2. Production 288+ expert MoE (GLM-5.3 Flash, 1T+ MoE): **35M to 50M tokens**
   (dump 50M once on the high-VRAM teacher node: 40M short at 4K + 10M long at
   16K to 32K). Higher expert sparsity (288 vs 40 experts) reduces per-expert
   token hits by 7.2x, so 35M to 50M tokens is required to converge both the
   74M+ `intermediate_scales_fp16` codebook gains and the 3 shared host-RAM
   depth banks (`524287` prime buckets per bigram/trigram/skip head).
3. Keep router weights and sparse attention indexers permanently frozen during
   QEC distillation, and keep text-backbone RMSNorm weights promoted to FP32.

The script automatically merges Attention LoRA deltas back into the attention
weights (`merged_non_expert_qec.pt`) and exports the Host-RAM Engram sidecar
(`engram_qec_sidecar.pt`).

## Step 7: Untouched KLD verification (8K and 16K to 64K) and kernel qualification

Score the assembled model (quantized experts + merged non-expert QEC + Host-RAM
Engram sidecar) on both the 8,192-token untouched evaluation windows and the
16,384 to 65,536-token long-context evaluation windows with
`trellismx_kit.kld_score.tokenwise_kld_fp64` and `summarize_windows` (single
window per forward pass, deterministic algorithms), plus needle-in-a-haystack
retrieval across the full supported context length. Before serving, apply
`patches/b12x-per-expert.patch` and `patches/flare-per-expert.patch` and pass
all five kernel acceptance gates in `design/PER-EXPERT-KERNEL.md`.
