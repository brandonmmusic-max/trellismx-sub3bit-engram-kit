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

1. Evaluation split: 20+ windows of 8,192 tokens. Capture and seal BF16 teacher
   logits once. Never read evaluation tokens during capture, allocation, or QEC
   distillation.
2. Selection split: 8 to 16 windows used only for `layer_sensitivity.py` and
   hyperparameter checks.
3. Calibration and QEC distillation stream: screen every document with a
   13-gram overlap check (`<= 1%` overlap) against all evaluation windows.

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

## Step 6: Full-Stack QEC distillation

Freeze all quantized MoE expert weights and routers permanently, then run DDP
Full-Stack QEC distillation against top-64 teacher logprobs (BF16 teacher or a
high-fidelity uniform K4 teacher when the BF16 teacher exceeds local VRAM):

```bash
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 scripts/distill_qec.py \
  --profile configs/<model>.json \
  --teacher-model /path/to/teacher \
  --student-model /path/to/quantized-student \
  --corpus /path/to/screened_tokens.pt \
  --engram-layers <6_to_12_spread_layer_ids> \
  --lora-rank 32 \
  --buckets 262139 \
  --tokens-m 50 \
  --output work/qec-artifacts
```

Recommended token budgets:

1. Unigram-only Engram + Attention LoRA (lowest host-RAM footprint): 5M to 15M
   tokens.
2. Full multi-head bigram/trigram Engram + Attention LoRA: 50M to 200M tokens
   (increase `--buckets` to `1048573` or `2097143` when training past 100M
   tokens).
3. Keep router weights frozen during QEC distillation.

The script automatically merges Attention LoRA deltas back into the attention
weights (`merged_non_expert_qec.pt`) and exports the Host-RAM Engram sidecar
(`engram_qec_sidecar.pt`).

## Step 7: Untouched KLD verification and kernel qualification

Score the final model (quantized experts + merged non-expert QEC + Host-RAM
Engram sidecar) on the untouched evaluation windows with
`trellismx_kit.kld_score.tokenwise_kld_fp64` and `summarize_windows` (single
window per forward pass, deterministic algorithms). Before serving, apply
`patches/b12x-per-expert.patch` and `patches/flare-per-expert.patch` and pass
all five kernel acceptance gates in `design/PER-EXPERT-KERNEL.md`.
