# Dual-Trellis Post-Quantization Error Correction: Sub-3-Bit Per-Expert TrellisMX Quantization and Host-RAM Engram Recovery from 3B Validation to a 304B-Expert GLM-5.3-Flash Pipeline

**Author:** Brandon M. Music  
**Target Venues:** arXiv (`cs.LG`, `cs.CL`), Hugging Face Papers, OpenReview, TechRxiv  
**Terminology Note:** Throughout this paper, **QEC** stands for **Post-Quantization Error Correction** (not quantum error correction). **Dual-Trellis** refers to the coupled two-pathway architecture pairing (1) GPU-resident trellis-coded quantized expert weights with (2) a parallel post-quantization error-correction pathway composed of mergeable non-expert adapters and host-RAM n-gram lookup tables.  
**Status Label Definitions:** `implemented` means code and artifacts exist and pass unit or integrity verification; `qualified` means end-to-end empirical accuracy or hardware correctness has been verified on held-out measurements; `research-only` means active experimental work or unshipped runtime integration not yet qualified end-to-end for production serving.

**Qualification Status Summary:**
- **Per-Expert TrellisMX Codec (`k2`, `p25`, `k3`, `k4`) and Dual-Trellis QEC Stack:** `qualified` on `ibm-granite/granite-3.1-3b-a800m-instruct` (32 routed MoE layers, 40 experts/layer, 3.02B expert weights) under exact IEEE-754 FP64 full-vocabulary KL divergence (`KL(teacher || student)`).
- **SM120 `b12x` Four-Rate Per-Expert Grouped MoE Kernel:** `qualified` on synthetic single-layer tensor-parallel (`TP4`) benchmarks matching GLM-5.3-Flash geometry (NVIDIA RTX PRO 6000 Blackwell, bit-exact decode parity, zero `compute-sanitizer` memory errors).
- **GLM-5.3-Flash 100M-Token QEC Distillation Pipeline (`2.375` and `2.500` bpw payload releases):** `implemented` / `research-only`. Calibration capture, dual exact rate allocation, union encoding, lossless Compressed Sparse Format (CSF) TP4 sidecar packaging (`85.651 GiB` and `90.080 GiB`), zero-overlap 100.1M-token corpus construction, the `646.975 GiB` BF16 teacher logit archive (public at `brandonmusic/GLM-5.3-Flash-BF16-Nemotron-100M-Teacher-Logits`), and all six pre-distillation (`0.0M`) exact FP64 KLD panels for `release_2375` (`2.375 bpw` payload / `2.417 bpw` total on-disk CSF sidecars) are complete: `ood_confirmation_2k = 0.05303` (`92.43%` Top-1), `ood_final_all25 = 0.06340` (`92.03%`), `ood_final_clean17 = 0.07473` (`90.54%`), `eval_8k = 0.09700` (`92.35%`), `eval_32k = 0.03910` (`93.77%`), and `eval_64k = 0.01080` (`75.36%`). Post-distillation 100M-token GLM-5.3-Flash KLD numbers from the active 8x NVIDIA B200 campaign are explicitly marked pending.
- **Provenance Statement:** All GLM-5.3-Flash TrellisMX checkpoints described in this paper are quantized directly from `zai-org/GLM-5.3-Flash-BF16`, with non-routed modules kept in BF16 as stored.

---

## Abstract

Running large Mixture-of-Experts (MoE) language models (300B to 1T+ total parameters) on local workstations or single unified-memory nodes faces a hard physical bottleneck. Although an MoE model activates only a small subset of its subnetworks (*experts*) on any single token, almost every expert is needed over the course of a conversation. Storing all experts in 16-bit precision exceeds available GPU video memory (VRAM), while offloading cold experts to slower host system RAM over PCIe collapses interactive token generation speed. Compressing every routed expert uniformly below 3 bits per weight (bpw) solves the VRAM capacity problem, but naive uniform 2-bit compression hits an exponential distortion cliff that severely degrades next-token probability distributions.

This paper specifies **Dual-Trellis Post-Quantization Error Correction (Dual-Trellis QEC)**, a three-layer compression and recovery system that keeps 100% of routed MoE experts permanently in GPU VRAM below 2.5 stored bits per weight while recovering high-precision output fidelity:

1. **Per-Expert Four-Rate TrellisMX Quantization.** Instead of forcing one uniform bit width across an entire model or layer, an exact integer half-bit budget allocator assigns each routed expert one shared bit rate from `{2.0, 2.5, 3.0, 4.0}` bpw based on layer sensitivity, squared routing-weight mass, and an asymmetric n-gram predictability discount. Expert matrices undergo coupled Hadamard orthogonal rotations, damped routed-plus-global Hessian conditioning with conditional downstream propagation, 5-pass GPTQ+Viterbi scale refitting onto a zero-storage 16-bit procedural FP8 state law (`procedural-npk`, Neural Processing of K), and lossless block-scale compression.
2. **Mergeable Non-Expert Error Correction.** With all quantized expert streams, routers, sparse attention indexers, and vision modules permanently frozen, post-quantization knowledge distillation optimizes low-rank adapters (LoRA) on attention projections, dense prefix layers, shared experts, and untied output heads alongside FP32-promoted normalization scales. After training, adapter deltas fold algebraically into the stored BF16 non-expert weights and norms return to BF16, adding zero serving-time VRAM or latency.
3. **Host-RAM Engram Error-Correction Sidecar.** Three cross-layer shared lookup banks residing in host DDR5 RAM map local token patterns (unigrams, bigrams, trigrams, and skip-grams) into context-gated, router-modulated, causally convolved residual corrections. Because every lookup address depends strictly on input token IDs, the serving runtime can prefetch all required table rows (`~3.75 KiB/token` in BF16 or `~1.9 KiB/token` in FP8) deterministically in $O(1)$ time before Layer 0 executes.

On `ibm-granite/granite-3.1-3b-a800m-instruct` (`172,011` untouched evaluation positions across 11 domains), per-expert rate allocation reduces training-free 2.50 bpw KL divergence (KLD) from `0.1015` (uniform) and `0.0975` (per-layer) to `0.0799` (`-21.3%` vs uniform). Adding Full-Stack QEC v1 (`1,283M` host-RAM parameters, `+6.8 bpw` host RAM) to strictly frozen 2.00 bpw uniform experts reduces KLD from `0.2348` to `0.1312` (`-44.1%`, `+4.65` percentage points Top-1 agreement, matching `2.347` unassisted uniform VRAM bits per weight). Upgrading to Dual-Trellis QEC v2 (`421M` host-RAM parameters, `+2.23 bpw` BF16 / `+1.12 bpw` FP8 host RAM) lowers 10M-token Uniform 2.00 bpw KLD to `0.1384` (with per-expert intermediate scale and bias recalibration) and `0.1436` (`-38.8%`) without those expert-domain terms (the closer architectural proxy for GLM-5.3-Flash). On Per-Expert 2.25 bpw, the 10M-token v2 stack lowers KLD from `0.1106` to `0.0891` (`-19.4%` vs `0.1106` baseline; `-19.3%` vs `0.1104` pre-train check; `89.06%` Top-1), beating unassisted Uniform 2.50 bpw (`0.1015`), Per-Layer 2.50 bpw (`0.0975`), and Per-Expert 2.375 bpw (`0.0934`).

Scaling the pipeline to `zai-org/GLM-5.3-Flash-BF16` (`304,405,807,104` routed parameters) yields verified lossless-CSF TP4 releases at `2.375 bpw` trellis payload (`2.417 bpw` / `85.651 GiB` total on-disk sidecars; `2.464` total effective bpw with the `1.651 GiB` FP8 Engram sidecar or `2.505` with the `3.111 GiB` BF16 sidecar) and `2.500 bpw` payload (`2.542 bpw` / `90.080 GiB` total on-disk sidecars). Prior to distillation (`0.0M` tokens), the training-free `2.375 bpw` GLM-5.3-Flash release already achieves exact FP64 mean KLD of **`0.05303`** (`92.43%` Top-1) on untouched `ood_confirmation_2k`, **`0.06340`** (`92.03%`) on `ood_final_all25`, **`0.07473`** (`90.54%`) on `ood_final_clean17`, **`0.09700`** (`92.35%`) on `eval_8k`, **`0.03910`** (`93.77%`) on `eval_32k`, and **`0.01080`** (`75.36%`) on `eval_64k`, paired with a public 100.1M-token zero-overlap 11-domain Nemotron teacher-logit archive undergoing an active 8x NVIDIA B200 distillation campaign.

---

## 1. Introduction and Systems Motivation

### 1.1 Why Frontier MoE Models Hit a Memory Wall

A modern transformer language model predicts the next token in a sequence by passing a vector representation (the *residual stream* or *hidden state*) through dozens of stacked neural layers. Each layer performs two main operations: an *attention* step that mixes information across earlier tokens in the prompt, and a *feed-forward network* (multi-layer perceptron, or MLP) that transforms each token independently via large matrix multiplications.

In a traditional *dense* transformer, every token multiplies against the exact same feed-forward weight matrices, so increasing model capacity increases both memory footprint and per-token arithmetic linearly. A **Mixture-of-Experts (MoE)** architecture decouples total model capacity from per-token compute. Instead of one huge feed-forward network per layer, each routed MoE layer contains dozens or hundreds of smaller feed-forward subnetworks called **experts** (for example, `288` experts per layer in `zai-org/GLM-5.3-Flash-BF16`). For each token, a lightweight linear classifier called a **router** scores all experts and activates only a tiny subset (typically `top_k = 8`). Only those 8 active experts execute matrix multiplications for that token.

This sparse design makes MoE models fast *when every weight already lives in high-bandwidth GPU memory (VRAM)*, but it creates a severe deployment wall:

1. **Capacity vs. Activity Mismatch.** Even though one token activates fewer than 3% of the experts in a layer, different tokens across a prompt or session trigger nearly all experts. Consequently, *every* expert weight must remain stored in memory. In `zai-org/GLM-5.3-Flash-BF16`, the 42 main routed MoE layers contain `304,405,807,104` parameters (`~567 GiB` in 16-bit Brain Floating Point, BF16), plus `9.67B` non-routed text and multimodal parameters (`18.03 GiB` in BF16), yielding `~585 GiB` of main weights (`~600` to `643 GiB` total on-disk checkpoint including the optional Multi-Token Prediction layer `45`, vision tower, and FP32 buffers). A 1T-parameter frontier MoE requires roughly `1.82 TiB` in BF16.
2. **Autoregressive Decode Is Memory-Bandwidth Bound.** During interactive generation (*decode*), the model produces one token per forward pass. Reading the active weights from memory dominates wall-clock latency far more than arithmetic. Modern GPUs provide multiple terabytes per second of internal VRAM bandwidth (HBM3e or GDDR7), whereas the PCIe bus connecting a GPU to host CPU RAM (DDR5) delivers roughly 32 to 64 GB/s per direction, two orders of magnitude slower.
3. **Why Expert Weight Offloading Ruins Latency.** Prior MoE serving work (Eliseev and Mazur 2023; MoE-Infinity; Pre-gated MoE; Fiddler) mitigates VRAM shortfalls by keeping hot experts on GPU and streaming cold experts from host DDR5 RAM over PCIe. Because the router at Layer $l$ depends on the exact hidden state exiting Layer $l-1$, predicting exact expert IDs multiple layers ahead is imperfect. Any cache miss stalls the GPU while tens of megabytes traverse PCIe, collapsing interactive generation throughput.

This work targets two concrete local deployment envelopes with **zero expert weight offloading**:
- **Single 128 GB unified-memory node** (e.g., NVIDIA DGX Spark, `119.2 GiB` total unified physical memory shared between GPU and host pools) running a ~314B-parameter MoE (`zai-org/GLM-5.3-Flash-BF16`) with headroom for the OS, CUDA context, and KV cache.
- **Four 96 GB GPUs** (`382 GiB` aggregate VRAM paired with `512 GB` host DDR5 RAM) running a 1T+ parameter MoE checkpoint at native VRAM decode speed.

Meeting either envelope requires compressing routed MoE expert weights below **2.5 bits per weight (bpw)** in VRAM.

### 1.2 Why Naive Sub-3-Bit Quantization Collapses Quality

**Quantization** shrinks a neural network by replacing each 16-bit weight with a compact code using far fewer bits (typically 2 to 4 bits). At 4 bits per weight, 16 discrete levels per value preserve most language-model behavior. Below 3 bits, naive scalar quantization fails sharply for three structural reasons:

- **Exponential Rate-Distortion Cliff.** Information theory (Shannon's Gaussian rate-distortion bound $D(R) = 2^{-2R}$) and empirical transformer measurements show that removing 1 bit roughly quadruples mean squared quantization error ($\approx 4\times$ per bit, or $\approx 2\times$ per half-bit). On `ibm-granite/granite-3.1-3b-a800m-instruct`, uniform 2.0-bit KL divergence (`0.2348`) is `4.63x` worse than uniform 3.0-bit (`0.0507`).
- **Outlier Amplification.** Neural weight matrices contain occasional large-magnitude entries (*outliers*). Stretching a coarse 2-bit grid to cover one outlier forces the remaining 99% of normal weights onto one or two discrete levels.
- **Extreme MoE Routing Skew.** MoE experts do not contribute equally to model outputs. A small fraction of frequently routed heavy-hitter experts carry most of the routing probability mass, while cold experts trigger rarely. Uniform 2.0-bit or 2.5-bit precision wastes bits on rarely triggered matrices while starving the heavy hitters that govern output quality.

### 1.3 Core Insight: Dual-Trellis QEC

Standard **Quantization-Aware Distillation (QAD)** attempts to recover low-bit accuracy by backpropagating gradients through quantized weights during fine-tuning. On a 300B to 1T+ MoE compressed with discrete trellis coding, standard QAD is impractical for two reasons: (1) discrete tail-biting Viterbi trellis search is non-differentiable, and (2) holding FP32 master weights plus AdamW optimizer states for 300B+ parameters requires multiple terabytes of VRAM.

**Dual-Trellis QEC** overcomes both the VRAM wall and the sub-3-bit quality cliff by coupling two orthogonal pathways across three engineering layers:

```
+------------------------------------------------------------------------------+
|                   DUAL-TRELLIS QEC SYSTEM ARCHITECTURE                       |
+------------------------------------------------------------------------------+
|                                                                              |
|  INPUT TOKENS (t_{i-2}, t_{i-1}, t_i)                                        |
|       |                                              |                       |
|       | (1) Primary Trellis Path (GPU VRAM)          | (2) O(1) Deterministic|
|       v                                              |     Prefetch BEFORE   |
|  +-------------------------------------------+       |     Layer 0 (Host RAM)|
|  | Non-Routed Modules (BF16, Zero Overhead)  |       v                       |
|  | - Attention (KDA/MLA) + Merged LoRA       |  +-------------------------+  |
|  | - Dense Prefix MLPs + Merged LoRA         |  | 3 Shared Depth Banks    |  |
|  | - Shared Experts + Merged LoRA            |  | (Early / Mid / Late)    |  |
|  | - Untied lm_head + Merged LoRA            |  | - Raw Unigram (d=128)   |  |
|  | - Calibrated RMSNorms (carrier BF16)      |  | - 4x Bigram (2 raw,     |  |
|  +-------------------------------------------+  |   2 NFKC) [zero-init]   |  |
|       |                                         | - 2x Trigram [NFKC]     |  |
|       v                                         | - 2x Skip-gram [NFKC]   |  |
|  +-------------------------------------------+  | 3.75 KiB/tok BF16 or    |  |
|  | Routed MoE Experts (ALL IN GPU VRAM)      |  | ~1.9 KiB/tok FP8+scales |  |
|  | - Permanently Frozen TrellisMX Streams    |  +-------------------------+  |
|  | - Per-Expert Rate in {k2, p25, k3, k4}    |               |               |
|  | - procedural-npk E4M3 Law + CSF Scales    |               | Async DMA     |
|  | - Zero Expert Offloading                  |               v               |
|  +-------------------------------------------+  +-------------------------+  |
|       |                                         | Per-Layer GPU Module    |  |
|       |                                         | - W_k / W_v Projections |  |
|       +---------------- (+) <-------------------| - G=8 Context Gating    |  |
|                          |                      | - Router-Aware Gate     |  |
|                          v                      | - Width-4 Causal DWConv |  |
|              Corrected Residual Stream          +-------------------------+  |
+------------------------------------------------------------------------------+
```

First, **every routed expert stays permanently in GPU VRAM**, compressed below 2.5 bpw via per-expert four-rate trellis coding (`k2`, `p25`, `k3`, `k4`) with lossless CSF scale planes. Second, **mergeable non-expert adapters** absorb systematic attention, normalization, dense-prefix, shared-expert, and output-head drift and merge back into the stored BF16 carrier weights with zero serving overhead. Third, a **Host-RAM Engram QEC sidecar** stores millions of sparse n-gram residual corrections in host DDR5 RAM (or unified memory on single-pool machines). Unlike expert offloading, Engram lookup addresses depend *solely* on input token IDs `(t_{i-2}, t_{i-1}, t_i)` known before Layer 0 begins. Because 3 shared depth banks serve all 14 attached layers, total host-to-GPU payload is only **`3.75 KiB/token`** in BF16 (or **`~1.9 KiB/token`** in FP8 plus per-row scales), which can be prefetched deterministically behind Layer 0 compute without stalling GPU execution.

---

## 2. Step-by-Step Per-Expert Four-Rate TrellisMX Quantization Pipeline

The quantization pipeline converts an unmodified BF16 MoE checkpoint (`zai-org/GLM-5.3-Flash-BF16`) into SHA-256-verified, CSF-compressed, four-bank tensor-parallel (`TP4`) rank sidecars. Throughout Sections 2 and 3, **toolkit library and CLI modules** shipped inside `trellismx-sub3bit-engram-kit` (`trellismx_kit/*.py`, `scripts/dump_teacher_logits.py`, `scripts/distill_qec.py`) are distinguished from **campaign orchestration scripts and kernels** executed on the 8x NVIDIA B200 build node (`capture_fit64.py`, `assign_dual.py`, `encode_union_worker.py`, `build_triple_releases.py`, `build_nemotron_100m.py`, `package_and_upload_2375.py`, and `sqg_quantize.cu`).

### 2.1 Step 1: Calibration Activation Capture (`capture_fit64.py`)

**Intuition: Why weight quantization requires real activation data.**  
A weight matrix $W$ affects model predictions strictly through matrix-vector products $y = W x$, where $x$ is the incoming activation vector. If coordinate $x_j$ is usually near zero on real text, rounding error in column $j$ of $W$ barely perturbs $y$. Conversely, if $x_j$ carries large variance or aligns with a frequent semantic feature, small errors in column $j$ cascade through downstream layers. Recording real activations entering every routed layer and expert allows the quantizer to protect high-impact directions while absorbing distortion along quiet directions.

**Executed Specification.**  
For `zai-org/GLM-5.3-Flash-BF16` (`45` main transformer layers: dense prefix layers `0..2`, `42` routed MoE layers `3..44`, plus optional Multi-Token Prediction layer `45`), the campaign orchestration script `capture_fit64.py` executes unmodified BF16 forward passes over 64 preregistered `fit` calibration windows drawn from the sealed `roles-v5` multi-domain panel (`64 x 2,048` tokens = `131,072` tokens). Across all 42 routed layers (`3..44`), the capture saves three binary streams per layer directly from BF16:
- `hidden.bf16.bin`: incoming hidden states `[131072, 4096]` entering each MoE block.
- `topk_ids.u16le.bin`: top-8 selected expert indices `[131072, 8]` per token (`0..287`).
- `topk_weights.f32le.bin`: normalized scalar routing weights `[131072, 8]` assigned by the router.

Zero weights are altered during capture, and every calibration window is disjoint from all held-out evaluation panels.

### 2.2 Step 2: Per-Expert Hierarchical Rate Allocation (`assign_dual.py`, `trellismx_kit/allocate.py`)

**Intuition: Why per-expert rate allocation beats uniform and per-layer schedules.**  
Under a fixed VRAM budget (for example, `2.375` trellis payload bits per routed weight), a uniform schedule forces every expert into the same bit width. A per-layer schedule can allocate more bits to Layer 15 than Layer 4, yet within Layer 15 the hottest expert can carry far more routing mass than the coldest expert. Assigning one discrete bit rate *individually to each routed expert* directs higher precision toward the heavy-hitter experts that dominate residual-stream variance.

Both matrix carriers of any expert (`w13` holding `gate_proj` and `up_proj`; `w2` holding `down_proj`) share **one assigned rate** chosen from the four-rate set:
$$\mathcal{R} = \{\texttt{k2}: 2.0,\; \texttt{p25}: 2.5,\; \texttt{k3}: 3.0,\; \texttt{k4}: 4.0\} \text{ bpw}.$$
Sharing one rate per expert simplifies the fused GPU kernel (each expert task decodes using a single task-level scalar uniform across its cooperative thread array), and controlled Granite 3.1 3B sweeps (`Section 4.1`) confirm that sharing one rate per expert matches or slightly beats splitting rates between `w13` and `w2` (`0.0799` vs `0.0810` KLD at `2.50` bpw).

**Mathematical Formulation and Exact Half-Bit Integer Budget Solver.**  
Let $L = 42$ routed layers (`3..44`) and $E = 288$ experts per layer (`12,096` routed experts total). From the 64-window calibration capture, upstream preprocessing in `assign_dual.py` computes each expert's **squared routing-weight mass**:
$$m_{l, e} = \sum_{t=1}^{T} \sum_{k=1}^{K} \mathbf{1}[\text{topk\_ids}_{l}(t, k) = e] \cdot \left(\text{topk\_weights}_{l}(t, k)\right)^2.$$
Squaring routing weights reflects error physics: an expert's output enters the residual stream multiplied by scalar routing weight $w$, so its quantization noise variance scales with $w^2$. To prevent rarely routed experts in the calibration sample from collapsing to zero importance, upstream preprocessing in `assign_dual.py` clamps $m_{l, e}$ from below at each layer's **5th-percentile floor** before `trellismx_kit/allocate.py` normalizes within each layer to $\tilde{m}_{l, e} = m_{l, e} / \sum_{e'} m_{l, e'}$.

Each layer also carries a positive layer sensitivity scalar $s_l$ adjusted by **asymmetric Engram co-allocation**:
$$s_l^{\text{eff}} \propto s_l \cdot \max\bigl(0.1,\; 1 - \gamma \cdot R^2_l\bigr),$$
with $\gamma = \texttt{engram\_discount} = 0.45$ and re-normalization to unit mean across all 42 routed layers. On `zai-org/GLM-5.3-Flash-BF16`, `assign_dual.py` supplies explicit structural priors informed by the Granite U-shaped predictability curve (`Section 4.6`):
- **Layer sensitivity prior $s_l$:** `1.15` for early transition layers `3..5`, `1.00` for layers `6..9`, `1.10` for deep contextual core layers `10..38`, and `1.05` for final output layers `39..44`.
- **Engram bigram predictability prior $R^2_l$:** linearly decreasing `0.55 - 0.04 * (l - 3)` across early layers `3..9`, flat `0.12` across deep contextual layers `10..38`, and linearly rising `0.35 + 0.04 * (l - 39)` across late layers `39..44`.

Discounting high-$R^2_l$ early and late layers shifts scarce `k3` and `k4` VRAM bits into deep contextual middle layers (`10..38`) where local n-gram tables have low predictive power.

The allocator models predicted distortion of assigning rate $r \in \mathcal{R}$ to expert $(l, e)$ as:
$$D_{l, e}(r) = s_l^{\text{eff}} \cdot \tilde{m}_{l, e} \cdot \rho(r),$$
using `EMPIRICAL_RD_RATIOS` in `trellismx_kit/allocate.py`:
$$\rho(\texttt{k2}) = 4.631, \quad \rho(\texttt{p25}) = 2.002, \quad \rho(\texttt{k3}) = 1.000, \quad \rho(\texttt{k4}) = 0.500.$$
Here $\rho(\texttt{k2})$, $\rho(\texttt{p25})$, and $\rho(\texttt{k3})$ are calibrated directly from measured Granite 3.1 3B uniform KLD ratios (`0.2348 / 0.0507 = 4.631`, `0.1015 / 0.0507 = 2.002`, `1.000`), whereas $\rho(\texttt{k4}) = 0.500$ is a conservative heuristic prior (ideal $4\times$-per-bit scaling would predict `0.25`, while a gentler `0.50` prior prevents the solver from starving sub-3-bit experts to over-allocate 4-bit experts).

Because every rate in $\mathcal{R}$ is an exact integer multiple of `0.5` bits (`costs = {k2: 4, p25: 5, k3: 6, k4: 8}` half-bits per weight), exact global allocation is a multiple-choice knapsack problem under integer budget $B_{\text{half}} = \lfloor 2 \cdot b_{\text{target}} \cdot L \cdot E \rfloor$. An optional `max_k4_fraction` guardrail restricts `k4` eligibility to the top $\lfloor \text{max\_k4\_fraction} \cdot L \cdot E \rfloor$ experts ranked by importance $s_l^{\text{eff}} \tilde{m}_{l, e}$. When every expert's rate-distortion curve is convex (strictly satisfied by `EMPIRICAL_RD_RATIOS`), `_solve_convex_fast` computes the exact global optimum in $O(N \log N)$ time using a max-heap of marginal damage reductions per half-bit plus exact boundary resolution for $\{1, 2\}$-half-bit swaps; for arbitrary non-convex damage tables, `solve_exact_dp` falls back to exact compact dynamic programming.

**Executed GLM-5.3-Flash Allocations and Union Deduplication (`assign_dual.py`).**  
Running `assign_dual.py` with `engram_discount = 0.45` produces two sealed exact-budget assignments across all `12,096` routed experts:

| Release Target | `max_k4_fraction` (Cap Experts) | `k2` (`2.0` bpw) | `p25` (`2.5` bpw) | `k3` (`3.0` bpw) | `k4` (`4.0` bpw) | Achieved Payload bpw | Canonical Assignment SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| `release_2375` | `0.06` (`725` cap; non-binding) | `4,839` | `5,624` | `1,542` | `91` | `2.3750` | `fe2ec2d094f3314a25d7b514a5eed633f299b70eeb93dfb09fa04d3281d9d931` |
| `release_2500` | `0.08` (`967` cap; non-binding) | `3,129` | `6,144` | `2,670` | `153` | `2.5000` | `5595524ab3568f534ae975b30a2c8079a00ba871b688fd789bbb5a970dad5ffc` |

At both `2.375` and `2.500` bpw, the `max_k4_fraction` guardrail is non-binding (`91 < 725` and `153 < 967`), serving primarily as a safety ceiling at higher average bit budgets. Encoding both releases independently would require $2 \times 12,096 = 24,192$ expert encodes; taking the set union of unique `(layer, expert, rate)` triples reduces the queue to `15,058` unique encodes, eliminating **`9,134 / 24,192 = 37.8%`** of encoding work.

**Why YAQA-Style Two-Sided Fisher Rounding Is Excluded.**  
YAQA (`arXiv:2505.22988`) replaces standard input-Hessian GPTQ rounding with two-sided Kronecker-factored Fisher rounding weighted by whole-model KL sensitivity. As documented in `Section 4.1`, an independent implementation of YAQA-style rounding on Granite 3.1 3B lowered KLD on 3 test windows when *only Layer 0* was quantized while all other 31 layers remained BF16, yet degraded every full-model configuration by `+1.7%` to `+5.9%` and hurt 25 of 32 single-layer swaps inside a quantized model while doubling encode time. Once every layer carries sub-3-bit error, unquantized Fisher factors no longer describe the perturbed loss landscape, so accepting higher raw weight error (`0.029` vs `0.023` NMSE on Layer 0) harms end-to-end KLD (while noting the published YAQA paper reported gains in its own setting). Standard local input-Hessian GPTQ rounding exhibits no such cross-layer fragility and is used exclusively.

### 2.3 Step 3: Coupled Hadamard Incoherence Processing

**Intuition: Why orthogonal rotations crush weight outliers.**  
Consider a vector of 128 numbers where one entry is `10.0` and the remaining 127 entries are `0.1`. Quantizing directly stretches the scale factor to cover `10.0`, crushing the 127 small entries onto nearly zero. Multiplying the vector by an orthogonal **Hadamard matrix** $H$ (structured $\pm 1/\sqrt{n}$ entries satisfying $H^\top H = I$, executable in $O(n \log n)$ additions/subtractions via the fast Walsh-Hadamard transform) spreads the single `10.0` spike evenly across all 128 coordinates as values near $\pm 10/\sqrt{128} \approx \pm 0.88$. The resulting Gaussian-like distribution compresses cleanly under trellis coding. Because $(W H^\top)(H x) = W x$, the rotation is algebraically lossless up to floating-point rounding.

**Coupled H512 / H128 Rotation Sandwich (`trellismx_kit/pe_format.py`, `encode_union_worker.py`).**  
TrellisMX wraps every expert's `(gate_proj, up_proj, down_proj)` triad inside a coupled two-tier block-Hadamard transform paired with deterministic Rademacher ($\pm 1$) sign vectors and diagonal scale vectors:

```
  Incoming Residual x [H=4096]
       |
       +--> Block Hadamard H512 (residual axis)
       +--> Multiply by gate_up_suh_fp16 [H=4096]
       +--> Block Hadamard H128 (input axis)
       v
  [Quantized TrellisMX slot0 (gate) & slot1 (up)]   <-- Encoded in rotated domain
       |
       +--> Atom-interleave rows (32-channel atoms)
       +--> Block Hadamard H128 (middle axis)
       +--> Multiply by pair_scales (gate_svh, up_svh) from intermediate_scales_fp16
       +--> Block Hadamard H128 (middle axis) * pre_signs (axis=1)
       +--> Split even/odd rows -> logical (gate, up) -> SwiGLU (silu-cap10)
       v
  [Middle Activation m = silu(clamp(gate, <=10)) * clamp(up, -10..10)]
       |
       +--> Multiply by post_signs (axis=2) -> Block Hadamard H128
       +--> Multiply by down_suh (from intermediate_scales_fp16) -> Block Hadamard H128
       v
  [Quantized TrellisMX down_core (down)]            <-- Encoded in rotated domain
       |
       +--> Block Hadamard H128 (output axis)
       +--> Multiply by down_svh_fp16 [H=4096]
       +--> Block Hadamard H512 (residual axis)
       v
  Expert Output y [H=4096]
```

Each layer sidecar stores four coupled metadata tensors in global expert order:
- `gate_up_suh_fp16` (`[H]`): shared per-layer input-side scale/sign vector after the `H512` residual rotation.
- `down_svh_fp16` (`[H]`): shared per-layer output-side scale/sign vector before the final `H512` residual rotation.
- `intermediate_scales_fp16` (`[E, 3 * I_loc]` per TP4 rank): per-expert diagonal scales partitioning into `gate_svh`, `up_svh`, and `down_suh` along the intermediate dimension.
- `coupled_sign_draw_u8` (`[E]`): deterministic integer draw seed selecting Rademacher sign vectors (`_rotation_signs` using 64-bit mixing constants `0x6A09E667F3BCC909` and `0xBB67AE8584CAA73B` for `axis=1` pre-SwiGLU and `axis=2` post-SwiGLU).

When `decode_coupled_scale_weights` inverts this sandwich for verification or BF16 installation (`scripts/distill_qec.py --per-expert-release`), it maps `(slot0, slot1, down_core)` back to physical logical `(gate, up, down)` weight matrices.

### 2.4 Step 4: Routed-Plus-Global Hessian Construction and Conditional Downstream Propagation

**Intuition: Solving universal MoE Hessian rank-deficiency and compounding SwiGLU error.**  
Minimizing squared output error $\|W x - \widehat{W} x\|_2^2$ over calibration tokens $x$ relies on the input second-moment matrix (the **Hessian** of the squared output loss with respect to $W$):
$$H_{\text{in}} = \frac{1}{N} \sum_{t=1}^{N} x_t x_t^\top.$$
Two severe failure modes occur in large MoE models if $H_{\text{in}}$ is built naively:
1. **Universal Rank Deficiency on Routed Experts.** In GLM-5.3-Flash's 64-window capture (`131,072` tokens, `top_k = 8`, `288` experts), the *average* expert receives only $131{,}072 \times 8 / 288 = 3{,}641$ routed tokens, which is strictly **smaller than the hidden dimension `4,096`**. Thus nearly *every* expert's purely routed Hessian is mathematically rank-deficient (zero eigenvalues along unsampled directions). Inverting a near-singular Hessian causes GPTQ error feedback to blow up on unseen prompts.
2. **Error Compounding Across SwiGLU.** Inside an expert, `down_proj` does not receive ideal BF16 intermediate activations; at inference time it ingests activations produced by the *already quantized* `gate_proj` and `up_proj` matrices passed through the nonlinear `SwiGLU` gate. Calibrating `down_proj` on clean BF16 middle activations optimizes against a mismatched input distribution.

**Executed Formulation (`encode_union_worker.py`).**  
TrellisMX resolves both failure modes explicitly:
- **Routed-Plus-Global Trace-Normalized Hessian Mix ($\beta = 0.5$).** Each expert combines its squared-routing-weight routed Hessian $H_{\text{routed}}^{(e)}$ with the full-rank global all-token Hessian $H_{\text{all}}$ across all `131,072` layer tokens, trace-normalized to matched scale:
  $$H_{\text{mix}}^{(e)} = 0.5 \cdot \frac{H_{\text{routed}}^{(e)}}{\text{tr}(H_{\text{routed}}^{(e)}) / d} + 0.5 \cdot \frac{H_{\text{all}}}{\text{tr}(H_{\text{all}}) / d}.$$
  Zero-route experts fall back directly to $H_{\text{all}}$.
- **Heavy Damping and Cholesky Inverse (`percdamp = 0.3`).** Prior codec sensitivity screening on GLM-5.3-Flash established that light 1% damping (`percdamp = 0.01`) on finite MoE routed token samples destabilizes GPTQ feedback. Adding diagonal damping `0.3` times the mean diagonal before computing the Cholesky inverse (`_prepare_full_inverse`) guarantees strict positive-definiteness and cuts expert-output damage sharply.
- **Conditional `down_proj` Hessian via Quantized Middle Carrier.** For every expert, the encoder first quantizes the stacked `[gate; up]` matrix (`gate_rec`, `up_rec`), propagates the calibration activations through the quantized `gate_rec` and `up_rec` weights plus SwiGLU (`coupled_middle_carrier(..., quantize=True)`), and builds the `down_proj` Hessian directly from those actual quantized middle activations.

### 2.5 Step 5: Procedural NPK Trellis Coding, 5-Pass Scale Refitting, and the Fused SMEM CUDA Kernel

**Intuition: Why trellis-coded vector quantization beats scalar rounding.**  
Consider compressing 256 numbers using 2 bits each (512 bits total). A scalar quantizer maps each number independently to one of only $2^2 = 4$ fixed levels. **Trellis-Coded Quantization (TCQ)** (Marcellin and Fischer 1990) borrows convolutional state machines and dynamic-programming decoding (Viterbi 1967) from digital communications. Rather than quantizing each weight independently, the decoder maintains a 16-bit sliding-window *state* $s_t$ encoding the most recent bit symbols. Advancing one step shifts $b$ new bits into the 16-bit state, and a deterministic law maps each state to an FP8 (`E4M3`) weight value. Although the underlying E4M3 format contains at most `254` finite signed values (`~161` distinct non-negative magnitudes after the procedural mapping), TCQ's large gain over scalar quantization arises from *sequential state-path optimization*: choosing a globally optimal multi-weight trajectory across overlapping 16-bit windows yields dense vector-quantization packing while transmitting only $2.0$ to $4.0$ bits per weight.

**The Zero-Storage `procedural-npk` State Law (`trellismx_kit/trellis_ref.py`).**  
TrellisMX uses Brandon M. Music's zero-storage **procedural NPK** (*Neural Processing of K*, where $K$ denotes the per-weight bit rate / trellis state index; machine label `procedural-npk`) 16-bit state law with `compander_scale = 2.0`. Its integer-to-FP16-pair bit-mixing lineage derives from QTIP's 3INST bit-masking (`0x8FFF8FFF`, `0x3B603B60`) and ExLlamaV3's MCG multiplier (`0xCBAC1FED`), extended with direct FP8 E4M3 register projection, `-0.0 -> +0.0` canonicalization, `compander_scale = 2.0` gain, linear congruential pair emission `(1664525, 1013904223)`, and a deterministic finite-output guarantee. For any 16-bit state $s \in [0, 65535]$:
1. Multiply modulo $2^{32}$: $p = (s \times \texttt{0xCBAC1FED}) \bmod 2^{32}$.
2. Mask and XOR into two IEEE-754 FP16 bit patterns: $w = (p \mathbin{\&} \texttt{0x8FFF8FFF}) \oplus \texttt{0x3B603B60}$.
3. Interpret the lower and upper 16 bits of $w$ as signed FP16 values $h_{\text{lo}}, h_{\text{hi}}$, sum them, and apply fixed gain `2.0`: $c = \text{fp32}(h_{\text{lo}} + h_{\text{hi}}) \times 2.0$.
4. Project $c$ to `float8_e4m3fn` and canonicalize `-0.0` to `+0.0`.

*Bounded Finite-Output Proof:* In step 2, masking each 16-bit lane with `0x8FFF` clears the top three exponent bits and XORing with `0x3B60` pins the 5-bit FP16 biased exponent strictly inside `{12, 13, 14, 15}` (actual exponents `2^-3` through `2^0`). Consequently, every half satisfies $|h_{\text{lo}}|, |h_{\text{hi}}| < 2.0$, their sum satisfies $|h_{\text{lo}} + h_{\text{hi}}| < 4.0$, and multiplying by `compander_scale = 2.0` yields $|c| < 8.0 \ll 448.0$ (the E4M3 finite ceiling). Thus `procedural-npk` is provably free of `NaN`, `Inf`, and E4M3 overflow across all `65,536` states while approximating a bell-shaped Gaussian sum distribution with zero stored codebook memory.

**Four-Rate Tile Geometry (`256` weights per `16 x 16` tile).**  
Every matrix is tiled into `16 x 16` blocks (256 weights) permuted into native GPU Tensor Core MMA B-fragment lane order (`tensor_core_permutation`):

| Rate | Code | Bits / Weight | Bits / Symbol ($b$) | Symbols / Tile | `int16` Words / Tile | `uint32` Words / Tile | State Emission Rule |
|---|---:|---:|---:|---:|---:|---:|---|
| `k2` | `0` | `2.0` | `2` | `256` | `32` | `16` | 1 operand per state: $\text{NPK}(s_m)$ |
| `p25` | `1` | `2.5` | `5` | `128` | `40` | `20` | 2 operands per state: $\text{NPK}(s_m)$ and $\text{NPK}((1664525 \cdot s_m + 1013904223) \mathbin{\&} \texttt{0xFFFF})$ |
| `k3` | `2` | `3.0` | `3` | `256` | `48` | `24` | 1 operand per state: $\text{NPK}(s_m)$ |
| `k4` | `3` | `4.0` | `4` | `256` | `64` | `32` | 1 operand per state: $\text{NPK}(s_m)$ |

In the `p25` pair rate, each 5-bit trellis step emits *two* E4M3 weights ($5 / 2 = 2.5$ bpw), using the linear congruential mixer `(1664525, 1013904223)` to decorrelate the second operand. Every tile is **tail-biting** (`tile_states`): the shift register wraps cyclically from the end of the tile back to index `0`, eliminating boundary distortion without wasting flush bits.

**5-Pass GPTQ + Viterbi Scale Refitting (`encode_union_worker.py`).**  
Matrices encode in **128-column slabs** (processing stacked `[gate; up]` together to share Hessian inversion). Inside each slab, **16-column blocks** apply GPTQ error feedback via triangular solve against the Cholesky factor of $H_{\text{inv}}$. Every contiguous group of `K = 32` weights shares one `UE8M0` power-of-two microscaling exponent ($2^{\text{code} - 127}$, following OCP MX v1.0). Scale optimization alternates with tail-biting Viterbi search (wrap context `128` scalar steps / `64` pair steps) across 5 passes:
- **Passes 1 and 2 (Continuous Scale Refit):** Run Viterbi search and solve each `K = 32` block scale to its exact continuous least-squares optimum.
- **Snap Transition:** Round each continuous scale to its nearest `UE8M0` power-of-two exponent (measured `-11%` expert-output damage vs scalar nearest-level scaling in prior codec screening).
- **Passes 3, 4, and 5 (Power-of-Two Refit):** Re-run GPTQ+Viterbi search with discrete `UE8M0` scales until convergence.
- **Bit-Exact Closure Check:** Immediately unpack the packed `int16` words and `uint8` `UE8M0` codes via the reference decoder and assert `torch.equal` bit-exact equality against the final in-memory reconstruction.

**Fused Shared-Memory CUDA Pair Kernel (`sqg_quantize.cu`).**  
In the 5-bit `p25` pair trellis, each transition maps an 11-bit base state ($2^{11} = 2{,}048$ states) plus a 5-bit incoming symbol ($32$ predecessors per state) into $2{,}048 \times 32 = 65{,}536$ full 16-bit states per step. Naive GPU Viterbi kernels spill intermediate path costs (`temp_costs`) to global HBM memory on every symbol step. Targeting the **NVIDIA B200 (`SM100`)** encoding GPUs with `__launch_bounds__(1024, 2)`, the fused shared-memory CUDA pair kernel holds the entire dynamic-programming working set on-chip in exactly **`83,200` bytes of Shared Memory (SMEM)**, eliminating global `temp_costs` traffic completely:
- `65,536` B: full resident `procedural-npk` E4M3 lookup table `lut_smem[65536]` (`uint8`).
- `16,384` B: double-buffered FP32 path-cost arrays `prev_smem_a[2048]` + `prev_smem_b[2048]` (`2 x 2,048 x 4` B).
- `1,024` B: FP32 tile input weights `input[256]` (`256 x 4` B).
- `128` B: warp-reduction minima `min_values[32]` (FP32) + `128` B: `min_indices[32]` (`int32`).
- Total: $65{,}536 + 16{,}384 + 1{,}024 + 128 + 128 = \mathbf{83{,}200}$ bytes exact.

### 2.6 Step 6: Four-Bank TP4 Rank Sidecar Packaging and Lossless CSF Scale Compression (`build_triple_releases.py`, `trellismx_kit/pe_format.py`)

**Intuition: Eliminating block-scale overhead losslessly.**  
Storing one uncompressed 8-bit `UE8M0` scale per 32 weights costs $8 / 32 = 0.25$ bpw (`9.51 GB = 8.86 GiB` across `304.41B` routed parameters). After coupled Hadamard rotation, however, contiguous 32-weight blocks along any output row share tightly clustered RMS norms, so nearly every `UE8M0` exponent in a row takes one of just two adjacent integer values, $b$ or $b + 1$.

**Lossless CSF Compression (`ue8m0-row-base-offset1-u32-exceptions.v1`).**  
For every row of a `UE8M0` scale plane, `encode_csf_plane` constructs an exact 256-bin histogram and picks the globally optimal adjacent base pair $[b, b + 1]$ maximizing $\text{count}(b) + \text{count}(b + 1)$ (`_optimal_pair_base`). Values in $\{b, b + 1\}$ pack into a 1-bit bitplane (`~0.031` bpw), while rare out-of-range exponents store verbatim in a sparse `(exceptions: uint8, exc_index: uint32)` table. This reduces total scale plus coupled-vector overhead from `0.25` bpw down to **`~0.042` bpw** with **zero numerical loss**: `encode_csf_plane` immediately calls `decode_csf_plane`, asserts bitwise `torch.equal`, and embeds SHA-256 digests of both the compressed tensors and the uncompressed plain `UE8M0` planes in safetensors metadata.

**Four-Bank TP4 Rank Sidecar Schema (`trellismx-p8-coupled-per-expert-rank.v2`).**  
Each routed layer shards across `tp_size = 4` tensor-parallel ranks (`local_intermediate_size = 512` per rank, matching `TP4` tensor-parallel expert serving). Each file (`p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors`) stores:
- Routing metadata: `expert_rate_u8` (`uint8 [288]`) and `expert_slot_u16` (`uint16 [288]`).
- Per-rate trellis banks: `w13_trellis_<r>` (`int16 [2, E_r, H/16, I_loc/16, TILE_I16[r]]`, dim-0 `[gate, up]`) and `w2_trellis_<r>` (`int16 [E_r, I_loc/16, H/16, TILE_I16[r]]`), omitted when $E_r = 0$.
- Lossless CSF scale planes for `w13` (stored in serving-kernel `up-gate` dim-0 order: `s13[:I_loc]` for `up`, `s13[I_loc:]` for `gate`) and `w2`.
- Coupled rotation vectors (`gate_up_suh_fp16`, `down_svh_fp16`, `intermediate_scales_fp16`, `coupled_sign_draw_u8`).
- Fail-closed metadata: schema string, `law = "procedural-npk"`, `compander_scale = 2.0`, pair-mix constants `(1664525, 1013904223)`, `rate_assignment_sha256`, and per-tensor SHA-256 hashes.

**Verified GLM-5.3-Flash Release Accounting (`build_triple_releases.py`).**  
Over GLM-5.3-Flash's exact `304,405,807,104` routed parameters, both 42-layer TP4 CSF releases (`168` sidecars each) verify bitwise and SHA-256 cleanly:

| Release | Sidecars | Trellis Payload (bpw / GiB) | Total On-Disk CSF Sidecar (Bytes / GiB / bpw) | Verified Install NMSE vs BF16 (`gate_up` / `down`) |
|---|---:|---|---:|---|
| `release_2375` | `168` | `2.375 bpw` / `84.164 GiB` | `91,966,908,114` B / `85.651 GiB` / **`2.417 bpw`** | L03: `0.0587` / `0.0557`; L24: `0.0572` / `0.0542`; L44: `0.0654` / `0.0623` |
| `release_2500` | `168` | `2.500 bpw` / `88.594 GiB` | `96,722,165,951` B / `90.080 GiB` / **`2.542 bpw`** | Verified via `read_rank_sidecar` SHA-256 + bitwise CSF round-trip |

During distillation startup (`_install_per_expert_release` in `scripts/distill_qec.py`), `release_2375` decodes in 16-expert chunks (holding FP32 decode peak to `~1.5 GiB` instead of `27 GiB`), inverts the coupled `H512`/`H128` scales into logical BF16 expert tensors, enforces `torch.equal` copy checks on every expert, and confirms empirical NMSE against unmodified `zai-org/GLM-5.3-Flash-BF16` (`0.0542` to `0.0654`, well below the `0.25` fail-closed sanity ceiling).


---

## 3. Step-by-Step Dual-Trellis QEC v2 Post-Quantization Distillation Pipeline

Even with per-expert four-rate TrellisMX allocation, compressing `304.41B` routed parameters below `2.5` bpw introduces systematic distortion: mild magnitude shrinkage in hidden streams, slight miscalibration across long-context sparse attention heads, and blurred recall of static lexical idioms, named entities, and syntactic patterns. **Knowledge distillation** (Hinton et al. 2015) repairs these systematic errors by training compact auxiliary modules so the quantized *student* model reproduces the next-token probability distribution of the unmodified BF16 *teacher* model, while keeping every quantized expert weight, router, sparse attention indexer, and vision parameter **permanently frozen**.

This section specifies the exact executed pipeline across campaign orchestration scripts (`build_nemotron_100m.py`, `package_and_upload_2375.py`) and toolkit library/CLI modules (`scripts/dump_teacher_logits.py`, `scripts/distill_qec.py`, `trellismx_kit/engram_qec.py`, `trellismx_kit/glm5_patch.py`, and `trellismx_kit/kld_score.py`).

### 3.1 Step 1: Zero-Overlap 11-Domain 100.1M-Token Nemotron Corpus Construction (`build_nemotron_100m.py`)

**Intuition: Why strict zero-overlap n-gram screening is mandatory for scientific validity.**  
In post-quantization distillation, especially when Host-RAM n-gram lookup tables memorize local lexical patterns, *data contamination* is a fatal experimental threat. If evaluation passages leak into the distillation corpus, lookup tables and adapters can memorize exact held-out spans and artificially depress evaluation KLD without improving generalization on unseen prompts. Screening every candidate training sequence against *every* evaluation and calibration sequence using exact stride-1 13-gram matching guarantees that held-out KLD reflects genuine out-of-distribution error correction.

**Executed Corpus Specification.**  
The corpus builder `build_nemotron_100m.py` streams 18 short-context sources and 9 long-context sources spanning 11 public NVIDIA Nemotron domains (agentic workflows, chat/instruction following, competitive programming/code, legal, long-context transduction, math, multilingual, pretraining text, safety, science, and software engineering). Documents undergo two hygiene gates before tokenization with the GLM-5.3-Flash tokenizer (`vocab_size = 154,880`):
1. **Document-Level Deduplication:** Exact BLAKE2b cryptographic hashing removes duplicate documents across and within streams.
2. **Zero-Tolerance Stride-1 13-Gram Exclusion (`max_13gram_overlap = 0.0`):** A forbidden hash set of **`1,895,227`** unique token 13-grams is compiled from all `665` windows in the sealed `roles-v5` registry plus the extended `eval_8k`, `eval_32k`, and `eval_64k` evaluation panels. Any candidate training sequence sharing even a *single* 13-token span with any forbidden window is rejected unconditionally.

The resulting zero-overlap training corpus partitions into an 80% short-context / 20% multi-length long-context curriculum totaling **`100,102,231`** supervised next-token positions:

| Training Split | Sequence Length | Windows | Supervised Positions (`seq_len - 1`) | Share of Corpus | Candidate Sequences Rejected (13-gram Screen) |
|---|---:|---:|---:|---:|---:|
| `train_short` | `4,096` | `19,536` | `79,999,920` (`80.00M`) | `79.92%` | `12,560` |
| `train_long_16k` | `16,384` | `611` | `10,010,013` (`10.01M`) | `10.00%` | `1,320` |
| `train_long_32k` | `32,768` | `184` | `6,029,128` (`6.03M`) | `6.02%` | `1,145` |
| `train_long_64k` | `65,536` | `62` | `4,063,170` (`4.06M`) | `4.06%` | `1,262` |
| **Total** | **Multi-length** | **`20,393`** | **`100,102,231`** | **`100.00%`** | **`16,287`** |

### 3.2 Step 2: One-Shot BF16 Teacher Logit Dumping and Tail-Lumped Top-K KL Lower Bound (`scripts/dump_teacher_logits.py`, `trellismx_kit/engram_qec.py`)

**Intuition: Why pre-dumping teacher logits saves VRAM and why naive top-K KL distorts tail entropy.**  
Holding a `~600` to `643 GiB` BF16 teacher alongside a quantized student during training wastes half of a cluster's VRAM and recomputes identical teacher outputs on every ablation. Because the teacher is static, one forward-only prefill pass archives its distributions to NVMe storage, allowing the teacher to be unloaded completely from VRAM before distillation starts.

Saving dense FP32 logits across `154,880` vocabulary entries for `100.1M` tokens would require `62 TB`. Instead, `scripts/dump_teacher_logits.py` exploits distribution sparsity by storing only the top $K = 128$ teacher tokens per position alongside the exact probability mass outside the top $K$.

A critical bias arises if one naively renormalizes the teacher's top-$K$ probabilities to sum to `1.0` and drops the remaining tail. Whenever the teacher places `15%` probability mass across the `154,752` tail tokens, renormalized top-$K$ KL pushes the student to place `100%` mass inside the top 128 tokens and `0%` in the tail, artificially over-sharpening student entropy.

**Sparse Exact-Tail Archive and Tail-Lumped Top-K KL Lower Bound.**  
For every training position $t$, `scripts/dump_teacher_logits.py` computes the FP32 log-partition function $\text{LSE}_t = \log \sum_{v=1}^{V} \exp(z_t[v])$ in chunks (`logit_chunk_tokens = 2048`), extracts `top_idx` (`int32`, $K = 128$), stores shifted log-probabilities $\log p_t[k] = z_t[\text{top\_idx}[k]] - \text{LSE}_t$ in `fp16` (`top_logp`, preventing catastrophic cancellation against FP32 `LSE`), and stores exact FP32 tail mass $p_{\text{tail}, t} = \text{clamp}(1 - \sum_{k=1}^{128} p_t[k], 0, 1)$ plus `logsumexp` (`fp32`). At `~780 bytes/token`, `100,102,231` supervised positions pack into `53` shards totaling **`72.719 GiB`**. In parallel, the unmodified BF16 teacher writes full-vocabulary FP32 dense logits for all 6 held-out evaluation panels (`eval_dense/`, `108` files totaling **`574.256 GiB`**). The complete reusable `646.975 GiB` teacher archive is published publicly at `brandonmusic/GLM-5.3-Flash-BF16-Nemotron-100M-Teacher-Logits`.

Before committing any shard under multi-rank DP+EP execution, `scripts/dump_teacher_logits.py` enforces an unconditional fail-closed **per-rank next-token alignment canary**: each rank's slice must achieve top-1 teacher agreement against `token_ids[:, 1:]` of at least `0.15`, catching any cross-rank gather or sequence-shift bug immediately.

During distillation, `topk_kl_loss` computes a **tail-lumped top-K KL lower bound** in `chunk_tokens = 512` gradient-checkpointed slices (preventing FP32 `[B, 65536, 154880]` logit allocations at 64K context). To guard against FP16 rounding where $\sum_{k \in \mathcal{S}_t} p_t[k] + p_{\text{tail}, t}$ slightly exceeds `1.0`, the implementation computes `scale = (t_top_p.sum(dim=-1) + p_tail).clamp_min(1.0)` and renormalizes `t_top_p`, `t_top_logp`, and `p_tail` consistently. Letting $\log q_t = \text{log\_softmax}(s_t)$ be the student's FP32 full-vocabulary log-probabilities and $\mathcal{S}_t$ be the teacher's top-128 indices, the student's lumped log tail probability is evaluated strictly in log-space by masking out $\mathcal{S}_t$ with $-\infty$:
$$\log q_{\text{tail}, t} = \operatorname{logsumexp}_{v \notin \mathcal{S}_t}\bigl(\log q_t[v]\bigr),$$
avoiding FP32 subtraction cancellation (`1 - sum(q_top)` collapsing to zero when $q_{\text{top}} \approx 1$). Guarding `has_tail = (p_tail > 1e-12) & torch.isfinite(log_q_tail)` prevents `0 * (-inf) = NaN` in backward passes, yielding:
$$\mathcal{L}_{\text{tail-lumped}}(t) = \underbrace{\sum_{k \in \mathcal{S}_t} p_t[k] \bigl(\log p_t[k] - \log q_t[k]\bigr)}_{\texttt{kl\_top}} + \underbrace{\mathbf{1}[\texttt{has\_tail}] \cdot p_{\text{tail}, t} \bigl(\log p_{\text{tail}, t} - \log q_{\text{tail}, t}\bigr)}_{\texttt{kl\_tail}} \le D_{\text{KL}}(p_t \,\|\, q_t).$$
By the data-processing inequality (lumping the `V - K` tail states into one super-state), $\mathcal{L}_{\text{tail-lumped}}$ is a tight mathematical lower bound on full-vocabulary KL divergence: it preserves exact total probability mass between the head and the tail while leaving the internal shape of the student's `V - K` tail unconstrained.

### 3.3 Step 3: Three-Component Dual-Trellis QEC v2 Architecture (`trellismx_kit/engram_qec.py`)

With all 42 layers of quantized routed experts, all routers, all sparse attention indexers (`index_topk = 2048`), and the vision tower **permanently frozen**, Dual-Trellis QEC v2 trains three compact components jointly:

#### Component A: Zero-Serving-Overhead Mergeable Non-Expert LoRA (`rank = 32`, `alpha = 64`)
A **Low-Rank Adapter (LoRA)** (Hu et al. 2022) adds a factored update $\Delta W = \frac{\alpha}{r} B A$ in parallel to an existing linear weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is Kaiming-initialized and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ starts at exact zero (guaranteeing an exact zero-perturbation step-0 starting point). After distillation, `merge_lora_into_model` folds $W \leftarrow \text{BF16}(\text{FP32}(W) + \frac{\alpha}{r} B A)$ directly into the stored BF16 carrier weights and detaches all LoRA hooks, incurring **zero serving VRAM and zero serving latency**.

On `zai-org/GLM-5.3-Flash-BF16` (`configs/glm-5.3-flash.json`), rank-32 LoRA (`lr = 2.5e-4`, `wd = 1e-2`) attaches to every non-routed linear projection across the text backbone:
- **34 Kimi Delta Attention (KDA) layers:** `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- **11 Multi-Head Latent Attention (MLA-NoPE) layers** (`3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43`): `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`.
- **3 dense prefix MLP layers** (`0, 1, 2`): `gate_proj`, `up_proj`, `down_proj`.
- **42 shared experts** (`3..44`): `gate_proj`, `up_proj`, `down_proj`.
- **Untied output head (`lm_head`):** Because `tie_word_embeddings = false`, adapting `lm_head` directly recalibrates output logit geometry. Crucially, `AttentionLoRA.forward` executes only the rank-32 bottleneck $x A^\top$ in FP32 and casts back to BF16 prior to multiplying by $B^\top$; materializing an FP32 `[1, 65536, 154880]` output at 64K context would otherwise consume **`37.81 GiB`** (`40.6 GB` decimal) of transient VRAM.

#### Component B: FP32-Promoted Text-Backbone RMSNorms
**Why BF16 RMSNorms silently stagnate during distillation.**  
Near `1.0`, BF16's 7-bit mantissa has a Unit in the Last Place (ULP) step size of $2^{-7} = 0.0078125$, which is more than `25x` larger than a typical `3e-4` AdamW step. Leaving RMSNorm scales in BF16 causes updates to round back to the original value, silently freezing the norms. `FullStackQEC._promote_norm_fp32` promotes all trainable text-backbone RMSNorm weights (strictly excluding routers, sparse indexers, and vision modules) to FP32 during training (`lr = 3e-4`, `wd = 0.0`). At export, `finalize_norms_to_carrier_dtype` casts the trained norms back to stock carrier BF16 and restores unmodified stock forward paths.

#### Component C: Dual-Trellis Host-RAM Engram v2 Sidecar (`SharedHostNGramBank` + `DualTrellisEngramLayer`)
Adapting DeepSeek's Engram conditional memory (`arXiv:2601.07372`) and classical hash embeddings (Svenstrup et al. 2017), the Host-RAM Engram v2 sidecar injects learned, context-gated lexical error corrections into the residual stream after selected MoE blocks:

1. **Three Shared Cross-Layer Depth Banks (`early`, `mid`, `late`).**  
   Fourteen attached routed layers share three host-RAM banks while keeping compact per-layer GPU projections (`0.138 GiB` total across all 14 layers in BF16):
   - `early`: layers `3, 5, 7, 9`
   - `mid`: layers `12, 16, 20, 24, 28, 32`
   - `late`: layers `36, 39, 42, 44`
   Sharing 3 banks across 14 layers cuts host-RAM footprint by **3x**, reduces host-RAM lookups by **~4x**, and **multiplies gradient updates per hash bucket by 4x**. Across all 3 banks, the exact table size is **`1,669,999,872` parameters** (`25,629,108` rows total): **`3.111 GiB`** in BF16 or **`1.651 GiB`** in FP8 (`1.555 GiB` E4M3 weights + `102,516,432` B = `0.0955 GiB` FP32 per-row scales).

2. **Multi-Order Distinct-Prime Hash Tables with NFKC Canonicalization and Zero-Init Residuals.**  
   Each shared bank contains:
   - **Raw Unigram Table:** exact vocabulary table `[154880, 128]` (`d_uni = 128`, initialized $\mathcal{N}(0, 0.02^2)$) preserving exact casing and leading whitespace.
   - **4 Bigram Heads** (`d_bi = 64` each): 2 heads hash raw pairs $(t_{i-1}, t_i)$ to preserve code/punctuation syntax, and 2 heads hash **NFKC-canonicalized** pairs (`build_nfkc_canon_map`: Unicode NFKC normalization, lowercasing, and whitespace stripping, skipping incomplete UTF-8 BPE fragments).
   - **2 NFKC Trigram Heads** (`d_tri = 64` each): hash canonicalized triples $(t_{i-2}, t_{i-1}, t_i)$.
   - **2 NFKC Skip-Gram Heads** (`d_skip = 64` each): hash non-adjacent canonicalized pairs $(t_{i-2}, t_i)$, capturing gapped syntactic templates even when the middle token changes.

   Each of the 8 hash heads per bank uses a distinct descending prime modulus from `1,048,573` (`[1048573, 1048571, 1048559, 1048549, 1048517, 1048507, 1048447, 1048433]`) with independent odd multiplicative mixers and salts (`(((a+1)*m1) ^ ((b+1)*m2) ^ salt).abs() % prime`) so collisions in one head do not co-occur in another. All bigram, trigram, and skip-gram tables initialize to **exact zero** (paired with orthogonal `gain = 0.05` on `w_v_bi/tri/skip` and zero init on `w_v_uni`) and train with `weight_decay = 0.0` (`lr_emb = 1.5e-3`). Zero-initialization ensures a clean zero-perturbation step-0 starting point and prevents random Gaussian noise from *untouched* buckets in small-token regimes (such as 5M to 10M Granite runs). At 100M tokens over `~1.05M` buckets per head, virtually every bucket receives multiple updates, so unseen held-out n-grams collide into trained buckets rather than zero rows; in that high-saturation regime, multi-head distinct-prime averaging combined with per-order context gating serves as the primary collision-mitigation mechanism (isolating zero-initialization alone at 100M tokens awaits a dedicated controlled ablation). Lookup vectors `e_t` are intentionally **not** passed through RMSNorm before `W_v`: normalizing small-std (`0.02`) lookup vectors amplifies them `50x` and causes immediate AdamW projection divergence.

3. **Per-Order Multi-Group Context Gating (`G = 8`) + Router-Aware Gate Modulation.**  
   Each attached `DualTrellisEngramLayer` projects each order $o \in \{\text{uni}, \text{bi}, \text{tri}, \text{skip}\}$ independently into key $k_o = W_{k,o} e_o$ and value $v_o = W_{v,o} e_o$, applies learnable RMSNorm to hidden state $h$ and keys $k_o$, splits the `4,096` channels into `G = 8` groups (`512` channels/group), and computes a per-group sigmoid gate:
   $$\alpha_{o, g} = \sigma\!\left(\frac{\langle \hat{h}_g,\; \hat{k}_{o, g} \rangle}{\sqrt{d_g}} + b_{o, g} + r_o(w_{\text{route}})\right),$$
   where $r_o(w_{\text{route}}) = \sum_{e=1}^{288} w_{\text{route}}[e] \cdot R_{o, e}$ is a zero-initialized **router-aware gate** (`route_gate_w` of shape `[4, 288]`). This lets each layer boost Engram error correction when a token routes to `2.0`-bit cold experts and attenuate it when `4.0`-bit experts already match BF16 closely.

4. **Zero-Initialized Width-4 Causal Depthwise Conv1D Shift Register.**  
   Summing the four gated order outputs $y = \sum_o \alpha_o \odot v_o$, `DualTrellisEngramLayer` applies `y <- y + silu(DWConv1d_k4(RMSNorm(y)))` with a zero-initialized width-4 causal kernel (`groups = 4096`). Convolving 3-token n-gram outputs across 4 causal steps (`t-3..t`) expands the effective lexical receptive field to 6+ tokens while adding only `3 x 4096 x 2 B = 24 KiB`/layer of decode shift-register state.

5. **Hook Placement on GLM-5.3-Flash Multi-Stream Hyper-Connections (`engram_hook_target = "mlp"`).**  
   GLM-5.3-Flash wraps each transformer sub-block inside a 4-stream **Manifold-Constrained Hyper-Connection (mHC)** bus (`block.ffn_hc`; Xie et al. 2026), whose full block output is 4-stream rather than 3D `[B, T, H]`. `FullStackQEC` automatically detects `hasattr(block, "ffn_hc")` and hooks Engram directly onto `block.mlp` (which always returns the collapsed 3D `[B, T, 4096]` tensor).

6. **$O(1)$ Deterministic Host-RAM Prefetch.**  
   Because every unigram, bigram, trigram, and skip-gram address in all 3 banks depends strictly on input token IDs `(t_{i-2}, t_{i-1}, t_i)`, a serving runtime can compute all addresses before Layer 0 starts and prefetch exactly `3 banks x (128 uni + 8 x 64 hash) = 1,920` elements/token: **`3.75 KiB/token`** in BF16 or **`~1.9 KiB/token`** in FP8 plus per-row scales (shared across all 14 attached layers), small enough to hide behind Layer 0 GPU compute.

### 3.4 Step 4: 80/20 Context Curriculum, Optimization, and 8x NVIDIA B200 Systems Engineering (`scripts/distill_qec.py`, `trellismx_kit/glm5_patch.py`)

Distilling a `314B`-parameter MoE up to `65,536`-token sequences on one `8x NVIDIA B200` node (`8 x 180 GB = 1.44 TB = 1.31 TiB` aggregate HBM3e) requires four targeted systems engineering solutions:

- **DP8 + EP8 `all_to_all` Without FSDP2 (`patch_ep_dp_all_to_all`).**  
  FSDP2 fails on GLM-5.3-Flash because BF16 linear weights and FP32 hyper-connection scalars / router biases coexist inside the same module tree. Instead, `scripts/distill_qec.py` loads the frozen backbone with pure Expert Parallelism (`ep_size = 8`, 36 local experts/GPU), overrides the class-level TP plan with `tp_plan = {"lm_head": "replicate"}` so every rank holds an independent full output head, and switches every `EpDispatchExpertsParallel` module to `AllToAllEP` with `tp_mesh.size() == 1`. All 8 GPUs process independent Data-Parallel (`DP8`) microbatches on non-expert layers while exchanging routed tokens over NVLink `all_to_all` inside MoE layers. Replicated QEC parameters and hash buffers broadcast from rank 0 at startup (`_sync_ep_qec_params`), and gradients synchronize via NVLink `dist.all_reduce(AVG)` before gradient clipping (`max_norm = 1.0`) and the AdamW step.
- **Chunked Local `grouped_mm` (`max_rows = 65,536`).**  
  At 32K and 64K sequence lengths, `all_to_all` dispatch can route over `65,536` token-expert rows to one GPU's local expert group, exceeding PyTorch `grouped_mm`'s internal row limit. `_make_chunked_local_experts` slices local expert batches above `65,536` rows and concatenates the outputs identically.
- **Triton `causal_conv1d` Acceleration (`patch_glm5_fla_causal_conv1d`).**  
  Replaces the slow pure-PyTorch `conv1d` fallback inside GLM-5.3-Flash's 34 KDA layers with Flash Linear Attention (`fla`; Yang and Zhang) Triton kernels.
- **Query-Chunked Long-Context DSA Indexer and Sparse Attention (`patch_glm5_long_context`).**  
  At `65,536` tokens with key-pool stride `index_kpool = 4` (`65,536 / 4 = 16,384` pooled keys), stock `Glm5NextTextIndexer.forward` materializes `[B, 65536, 32, 16384]` FP32 scores twice (**`256 GiB`** -> immediate OOM), and `Glm5NextTextAttention.build_attention_mask_from_topk` materializes `[1, 65536, 65536]` in both `int32` (`selected_counts`, `16 GiB`) and `bool` (`4 GiB`) = **`20 GiB`** combined. Chunking along the independent query dimension (`indexer_q_chunk = 2048`, `attn_q_chunk = 4096`) produces **algebraically identical** outputs (bit-identical within FP32 accumulation precision) while keeping peak VRAM **under `5 GiB`**.
- **80/20 Interleaved Multi-Length Curriculum and Schedule.**  
  `build_context_curriculum` interleaves short (`4K`) and long (`16K`/`32K`/`64K` via proportional round-robin `_MultiLengthShardStream`) steps uniformly across the 100M-token campaign:
  - Active 8x B200 launch configuration: `batch_size = 2`, `grad_accum = 1` on short steps (`16` sequences = `65,520` supervised positions/step across `DP8`) and `long_batch_size = 1`, `long_grad_accum = 1` on long steps (`8` sequences/step across `DP8`), contrasted with the `scripts/distill_qec.py` CLI defaults (`grad_accum = 8` / `long_grad_accum = 4` intended for smaller single-GPU setups).
  - Optimizer: AdamW (`lr_emb = 1.5e-3, wd = 0.0`; `lr_proj = 4e-4, wd = 1e-2`; `lr_lora = 2.5e-4, wd = 1e-2`; `lr_norm = 3e-4, wd = 0.0`), 5% linear warmup (`max(20, total_steps // 20)`), cosine decay to `0.02x` floor, gradient checkpointing (`use_reentrant = False`) with `enable_input_require_grads()`, `allow_tf32 = False`, and `highest` FP32 matmul precision.
  - Evaluation milestones: **`10M, 35M, 50M, 75M, and 100M`** tokens, scored via **GPU-chunked IEEE-754 FP64 KLD** (`compute_device = out_logits.device`, `chunk = 512` on GPU 0, with a `64`-thread CPU FP64 fallback `chunk = 1024`), verified bit-identical to 64-thread CPU FP64 on live `eval_32k` (`Section 6.2`).
- **3-Way Post-Merge PyTorch Artifact-Roundtrip Verification.**  
  Upon completing training, `scripts/distill_qec.py` merges all LoRA deltas into the BF16 carrier weights, restores FP32 norms back to carrier BF16, and scores three exact PyTorch artifact variants against `eval_dense/`: (1) **Zero-Sidecar Merged Carrier Only** (`engram_enabled = False`), (2) **Shipped BF16 Engram Artifact** (`3.111 GiB` BF16 tables + BF16 projections + stock BF16 carrier), and (3) **Shipped FP8 Engram Artifact** (`1.651 GiB` per-row-scaled FP8 E4M3 tables + BF16 projections + stock BF16 carrier). This establishes exact PyTorch artifact-roundtrip parity (`b12x`/Flare serving KLD remains to be measured on the finished checkpoint).

---

## 4. Prior Empirical Results on `ibm-granite/granite-3.1-3b-a800m-instruct` and Cross-Model Precedents

All Granite 3.1 3B measurements below (`status: qualified`) were executed under identical controlled conditions on `ibm-granite/granite-3.1-3b-a800m-instruct` (32 routed MoE layers, 40 experts/layer, `top_k = 8`, hidden size `1,536`, intermediate width `512`, vocabulary `49,155`, `3.02B` routed expert weights): **21 untouched evaluation windows** of `8,192` tokens across 11 domains (`172,011` supervised positions), exact IEEE-754 FP64 full-vocabulary `KL(BF16 teacher || student)`, single-window deterministic forward passes (`batch_size = 1`), and strict 13-gram screening (`<= 1%` overlap for QEC distillation corpora; `<= 3.6%` max across calibration/eval splits in the training-free study). All scorer self-checks (`teacher vs teacher`, `original weights through install path`, and independent second-scorer reproduction across 6 arms) verified exact `0.000000` error (`+0.0e+00` difference of means). Lower KLD and higher Top-1 agreement are better.

### 4.1 Training-Free Per-Expert vs. Per-Layer vs. Uniform TrellisMX Frontier (2.00 to 4.50 bpw)

Table 4.1 compares the training-free quantization frontier on the weights-only path. **Bit-accounting note:** TrellisMX rows list the trellis weight payload (`2.000` to `3.000` bpw; lossless CSF block scales add `~0.04` bpw on disk), whereas the two NVFP4 reference rows list total stored bits (`4.500` bpw = `4.0` payload + `0.5` UE4M3 `K = 16` block scales).

| Stored bpw | Configuration | Rounding | Mean KLD | BCa 95% CI | Median | p90 | p99 | Top-1 |
|---:|---|---|---:|---|---:|---:|---:|---:|
| `2.000` | Uniform K2 | Standard GPTQ | `0.2348` | `[0.2130, 0.2596]` | `0.0728` | `0.595` | `2.274` | `82.28%` |
| `2.125` | Per-expert `{k2, p25, k3}`, attribution | Standard GPTQ | `0.1359` | `[0.1248, 0.1523]` | `0.0400` | `0.332` | `1.393` | `86.56%` |
| `2.250` | Per-layer `{k2, p25, k3}`, attribution | Standard GPTQ | `0.1390` | `[0.1283, 0.1518]` | `0.0408` | `0.340` | `1.404` | `86.41%` |
| `2.250` | Per-expert `{k2, k3}`, attribution | Standard GPTQ | `0.1143` | `[0.1039, 0.1306]` | `0.0314` | `0.273` | `1.228` | `87.78%` |
| `2.250` | Per-expert `{k2, p25, k3}`, attribution | Standard GPTQ | `0.1106` | `[0.1007, 0.1257]` | `0.0313` | `0.267` | `1.195` | `87.83%` |
| `2.250` | Per-expert `{k2, p25, k3}`, attribution | YAQA-style (on K3) | `0.1125` | `[0.1025, 0.1281]` | `0.0316` | `0.272` | `1.208` | `87.70%` |
| `2.375` | Per-expert `{k2, p25, k3}`, attribution | Standard GPTQ | `0.0934` | `[0.0852, 0.1068]` | `0.0260` | `0.218` | `1.017` | `88.89%` |
| `2.500` | Uniform P25 (pair rate) | Standard GPTQ | `0.1015` | `[0.0938, 0.1106]` | `0.0278` | `0.239` | `1.068` | `88.32%` |
| `2.500` | Per-layer `{k2, k3}`, attribution | Standard GPTQ | `0.1097` | `[0.1012, 0.1200]` | `0.0317` | `0.265` | `1.116` | `87.89%` |
| `2.500` | Per-layer `{k2, p25, k3}`, attribution | Standard GPTQ | `0.0975` | `[0.0905, 0.1058]` | `0.0278` | `0.234` | `1.013` | `88.50%` |
| `2.500` | Per-expert `{k2, p25, k3}`, route mass, 1 rate/expert | Standard GPTQ | `0.0819` | `[0.0738, 0.0949]` | `0.0209` | `0.187` | `0.904` | `89.65%` |
| `2.500` | Per-expert `{k2, p25, k3}`, route mass (per group) | Standard GPTQ | `0.0815` | `[0.0737, 0.0934]` | `0.0214` | `0.188` | `0.894` | `89.64%` |
| `2.500` | Per-expert `{k2, p25, k3}`, attribution (per group) | Standard GPTQ | `0.0810` | `[0.0739, 0.0916]` | `0.0225` | `0.188` | `0.871` | `89.61%` |
| `2.500` | Per-expert `{k2, p25, k3}`, attribution, 1 rate/expert | Standard GPTQ | `0.0799` | `[0.0731, 0.0900]` | `0.0222` | `0.184` | `0.865` | `89.68%` |
| `2.500` | Per-expert `{k2, p25, k3}`, attribution | YAQA-style (on K3) | `0.0830` | `[0.0756, 0.0943]` | `0.0226` | `0.193` | `0.899` | `89.53%` |
| `2.625` | Per-expert `{k2, p25, k3}`, attribution | Standard GPTQ | `0.0715` | `[0.0655, 0.0801]` | `0.0195` | `0.163` | `0.771` | `90.35%` |
| `2.750` | Per-expert `{k2, p25, k3}`, attribution | Standard GPTQ | `0.0632` | `[0.0582, 0.0701]` | `0.0175` | `0.144` | `0.691` | `90.80%` |
| `2.750` | Per-layer `{k2, p25, k3}`, attribution | Standard GPTQ | `0.0721` | `[0.0667, 0.0784]` | `0.0192` | `0.166` | `0.748` | `90.19%` |
| `2.8508` | Per-expert `{k2, p25, k3}`, attribution | Standard GPTQ | `0.0585` | `[0.0542, 0.0642]` | `0.0159` | `0.134` | `0.630` | `91.08%` |
| `3.000` | Uniform K3 | Standard GPTQ | `0.0507` | `[0.0468, 0.0550]` | `0.0132` | `0.116` | `0.536` | `91.71%` |
| `3.000` | Uniform K3 | YAQA-style (all 32L) | `0.0536` | `[0.0494, 0.0590]` | `0.0140` | `0.127` | `0.557` | `91.56%` |
| `4.500` | NVFP4 on expert matrices only | N/A | `0.0370` | `[0.0343, 0.0398]` | `0.0099` | `0.082` | `0.369` | `92.79%` |
| `4.500` | NVFP4 on experts + attention projections | N/A | `0.1119` | `[0.1047, 0.1180]` | `0.0373` | `0.280` | `1.005` | `87.49%` |

**Key Findings from Table 4.1:**
1. **Per-expert allocation delivers an 18% to 21% KLD reduction across all 21/21 evaluation windows.** At `2.50 bpw`, one-rate-per-expert allocation reaches `0.0799` (attribution) and `0.0819` (route mass), beating Uniform P25 (`0.1015`, ratio `0.787`, 95% CI `[0.759, 0.817]`, `21/21` wins) and Per-Layer 2.50 (`0.0975`, ratio `0.819`, `21/21` wins). Per-expert at `2.375 bpw` (`0.0934`) already beats Uniform 2.50 bpw (`0.1015`, ratio `0.921`, `18/21` wins), and Per-expert at `2.125 bpw` (`0.1359`) beats Per-Layer 2.25 bpw (`0.1390`).
2. **Route mass matches expensive attribution within 0.6% to 2.5% when `{k2, p25, k3}` are all available.** At `2.50 bpw`, route mass (`0.0815` per-group / `0.0819` one-rate-per-expert) tracks full Aumann-Shapley attribution (`0.0810` / `0.0799`) well within the `0.0028` numerical tie floor (`Section 6.3`), validating calibration-only hierarchical allocation on 300B+ models.
3. **Keeping attention in BF16 is critical.** Extending 4.5-bit NVFP4 from experts (`0.0370`) to attention projections (`0.1119`) triples KLD (`+202%`), making `4.5`-bit experts+attention worse than `2.25`-bit per-expert TrellisMX with BF16 attention (`0.1106`).
4. **YAQA-style two-sided Fisher rounding degrades every full-model arm in this independent implementation.** At Uniform K3, YAQA-style rounding raises KLD by `+5.7%` weights-only (`0.0536` vs `0.0507`), `+5.9%` serving-format (`0.0567` vs `0.0535`), and `+5.3%` FP32 arithmetic (`0.0530` vs `0.0503`), and hurts single-layer swaps in `25/32` layers (`+0.004366` sum of single-layer deltas; `+0.002906` all-32 combined).

### 4.2 50M-Token Full-Stack QEC v1 Scaling Curve on Uniform K2 (2.00 Stored GPU bpw)

Freezing Uniform K2 experts strictly at `2.00` stored GPU bpw (zero expert offloading) and training Full-Stack QEC v1 (mergeable rank-32 Attention LoRA on `q/k/v/o`, RMSNorm scales, 1,280 per-expert scalar gains, plus a 12-layer Host-RAM Engram QEC sidecar on layers `0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 31` with `262,139` prime buckets per head = `1,283M` host-RAM parameters = **`+6.8 bpw` amortized host RAM in BF16** over Granite's `3.02B` expert weights) across 4x RTX PRO 6000 (`~35,000` tok/s, `24.2` min for `49.94M` 13-gram-screened tokens) produces Table 4.2:

| Distillation Tokens | Mean KLD | Delta vs K2 Baseline | Median | p90 | Top-1 Agreement |
|---:|---:|---:|---:|---:|---:|
| `0.0M` (Uniform K2 baseline) | `0.2348` | `0.0%` | `0.0728` | `0.5954` | `82.28%` |
| `1.05M` | `0.1843` | `-21.5%` | `0.0602` | `0.4581` | `84.41%` |
| `5.05M` | `0.1527` | `-34.9%` | `0.0483` | `0.3808` | `85.82%` |
| `10.03M` | `0.1454` | `-38.1%` | `0.0475` | `0.3617` | `86.04%` |
| `20.05M` | `0.1390` | `-40.8%` | `0.0438` | `0.3494` | `86.45%` |
| `35.06M` | `0.1330` | `-43.4%` | `0.0416` | `0.3341` | `86.77%` |
| `49.94M` | **`0.1312`** | **`-44.1%`** | **`0.0409`** | **`0.3309`** | **`86.93%`** |

At `49.94M` tokens, strictly 2.00-bit Uniform K2 + Full-Stack QEC v1 reaches `0.1312` KLD (`+4.65` percentage points Top-1), equivalent to `2.347` unassisted uniform GPU VRAM bits per weight (`+0.347` free VRAM bpw, trading off `+6.8` bpw of cheap host DDR5 RAM on this 3B test model).

### 4.3 Host-RAM Engram QEC Alone Across Bit Rates (10M Tokens)

Isolating an 8-layer Host-RAM Engram QEC sidecar (layers `0, 1, 2, 4, 8, 16, 24, 31`; `262,139` prime buckets; `868M` host-RAM params; experts, attention, and routers all permanently frozen) confirms strong monotonic recovery across every base bit rate:

| Base Model (Stored GPU bpw) | `0M` | `1M` | `3M` | `5M` | `10M` | Delta @ `10M` | Top-1 @ `10M` | Equivalent Unassisted VRAM Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Uniform K2 (`2.00`) | `0.2348` | `0.1753` | `0.1636` | `0.1603` | `0.1544` | `-34.2%` | `85.77%` | `+0.250` uniform bpw (`2.250` equiv.) |
| Per-Expert `2.25` | `0.1106` | `0.1019` | `0.0991` | `0.0978` | `0.0951` | `-14.0%` | `88.86%` | `+0.113` PE bpw (beats Uniform `2.50` at `0.1015`) |
| Per-Expert `2.50` | `0.0799` | `0.0757` | `0.0751` | `0.0743` | `0.0720` | `-9.9%` | `90.29%` | `+0.117` PE bpw (matches PE `2.625` at `0.0715`) |

### 4.4 Component Ablation on Per-Expert 2.25 bpw (5M Tokens)

Testing each recovery tier individually and jointly on Per-Expert 2.25 bpw over 5M tokens shows that Host-RAM Engram QEC and mergeable Attention LoRA are **complementary with substantial overlap** (`-11.5%` Engram alone, `-13.4%` LoRA alone, `-14.5%` combined):

| Configuration (`2.25` Stored GPU bpw) | Mean KLD | Delta vs Baseline | Median | p90 | Top-1 |
|---|---:|---:|---:|---:|---:|
| Baseline Per-Expert `2.25` bpw | `0.1106` | `0.0%` | `0.0313` | `0.2673` | `87.83%` |
| + 8-layer Unigram-only Engram (`107M` host-RAM params) | `0.0978` | `-11.6%` | `0.0287` | `0.2357` | `88.66%` |
| + 8-layer Full Engram uni+bi+tri (`868M` host-RAM params) | `0.0978` | `-11.5%` | `0.0290` | `0.2357` | `88.74%` |
| + Rank-16 Attention LoRA only (`5.24M` mergeable params) | `0.0958` | `-13.4%` | `0.0285` | `0.2343` | `88.76%` |
| **+ 8-layer Full Engram + Rank-16 Attention LoRA** | **`0.0945`** | **`-14.5%`** | **`0.0276`** | **`0.2329`** | **`88.90%`** |

Note why v1 Full Engram (`868M` params) tied Unigram-only (`107M` params) at 5M tokens: v1 used Gaussian-initialized bigram/trigram tables across 8 separate layer banks, so at 5M tokens untouched held-out buckets injected Gaussian noise that offset bigram/trigram gains. That finding directly motivated **Dual-Trellis QEC v2**.

### 4.5 Dual-Trellis QEC v2 Ablation Suite and Active Component Matrix (10M Fresh Tokens)

Table 4.5 presents the controlled 10M-token comparison of Dual-Trellis QEC v2 against the v1 reference on identical 21-window evaluation, showing the exact active modules in every arm and the amortized host-RAM footprint on Granite (`421M` v2 params = **`+2.23 bpw` BF16 / `+1.12 bpw` FP8** over `3.02B` expert weights, vs `1,283M` v1 params = **`+6.8 bpw` BF16**):

| Arm (`10M` Fresh Tokens) | Base (GPU bpw) | Active Modules (LoRA / Norms / Engram / Expert Terms) | Host-RAM Params (BF16 bpw) | KLD @ `1M` | KLD @ `5M` | KLD @ `10M` | Delta vs Baseline | Top-1 @ `10M` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v1 Full-Stack QEC reference | Uniform K2 (`2.00`) | Attn LoRA r32 / BF16 norms / 12L v1 Engram / scalar gains | `1,283M` (`+6.80`) | `0.1843` | `0.1527` | `0.1454` | `-38.1%` | `86.04%` |
| **Dual-Trellis Engram v2 + FP32 norms (GLM-5.3-Flash architectural proxy)** | Uniform K2 (`2.00`) | Attn LoRA r32 / FP32 norms / 3-bank v2 Engram / none | `421M` (`+2.23`) | `0.1629` | `0.1493` | **`0.1436`** | **`-38.8%`** | **`86.16%`** |
| Intermediate scales + bias + FP32 norms (v1 Engram) | Uniform K2 (`2.00`) | Attn LoRA r32 / FP32 norms / 12L v1 Engram / `inter_scales` + `expert_bias` | `1,283M` (`+6.80`) | `0.1642` | `0.1464` | `0.1411` | `-39.9%` | `86.29%` |
| **Combined Dual-Trellis QEC v2 stack (Uniform K2)** | Uniform K2 (`2.00`) | Attn LoRA r32 / FP32 norms / 3-bank v2 Engram / `inter_scales` + `expert_bias` | `421M` (`+2.23`) | `0.1613` | `0.1438` | **`0.1384`** | **`-41.1%`** | **`86.41%`** |
| **Combined Dual-Trellis QEC v2 stack (Per-Expert 2.25)** | Per-Expert `2.25` | Attn LoRA r32 / FP32 norms / 3-bank v2 Engram / `inter_scales` + `expert_bias` | `421M` (`+2.23`) | `0.0982` | `0.0937` | **`0.0891`** | **`-19.4%`** (vs `0.1106`; `-19.3%` vs `0.1104` pre-train) | **`89.06%`** |

Three findings govern the GLM-5.3-Flash design:
- **Architectural Proxy Honesty:** The strongest Uniform K2 arm (`0.1384`, `-41.1%`, median `0.0455`, p90 `0.3400`) includes logical-domain `intermediate_scales_fp16` and per-expert output centroid biases (`3.75 MiB`), which are omitted on GLM-5.3-Flash (`Section 5.2`). Without those expert-domain terms, the **Dual-Trellis Engram v2 + FP32 norms + Attention LoRA** arm (`0.1436`, `-38.8%`, `86.16%` Top-1) is the closer architectural proxy for GLM-5.3-Flash, still beating v1 (`0.1454`) with 3x less host RAM and 4x fewer host lookups.
- **Data Efficiency and Bit-Tier Breakthrough:** At `10M` tokens, the combined 2.00-bit stack (`0.1384`) matches v1's `20M`-token checkpoint (`0.1390`, a statistical tie within the `0.0028` noise floor) in half the tokens with 3x fewer host parameters. On Per-Expert `2.25` bpw, the combined 10M-token stack reaches **`0.0891`** (median `0.0272`, p90 `0.2140`, `89.06%` Top-1), beating unassisted Uniform `2.50` (`0.1015`), Per-Layer `2.50` (`0.0975`), and Per-Expert `2.375` (`0.0934`).
- **Excluded Negative Ablation (Shared-Basis Wiener Branch + Hidden-State MSE):** Adding a dense rank-64 shared-basis Wiener branch at `lr = 8e-4` overfit held-out evaluation, and even regularized (`lr = 1.5e-4, wd = 1e-2`) with hidden-state MSE alignment it trailed the clean combined stack at 5M tokens (`0.1453` vs `0.1438`). It is excluded from the canonical recipe.

### 4.6 Negative Controls (Expert Deletion vs. N-Gram LUT) and Theoretical Basis for Asymmetric Co-Allocation

Why keep cold experts in GPU VRAM at 2.0 bits rather than deleting the coldest 20% of experts and replacing them with host-RAM lookup tables? Table 4.6 resolves this directly on Granite 3.1 3B:

| Configuration | Effective GPU bpw | Mean KLD | Median | Top-1 |
|---|---:|---:|---:|---:|
| BF16 + Layer 0 deleted (zeroed, 3.1% experts) | `15.50` | `10.8053` | `9.7889` | `2.41%` |
| BF16 + Layer 0 zero-training bigram LUT | `15.50` | `0.7007` | `0.1372` | `74.68%` |
| BF16 + coldest 20% experts deleted (zeroed) | `12.80` | `0.6085` | `0.1432` | `74.88%` |
| BF16 + coldest 20% experts zero-training bigram LUT | `12.80` | `0.1302` | `0.0152` | `89.37%` |
| PE `2.50` + coldest 20% experts deleted (zeroed) | `2.06` | `0.7275` | `0.2105` | `71.53%` |
| PE `2.50` + coldest 20% experts zero-training bigram LUT | `2.06` | `0.1970` | `0.0406` | `85.44%` |
| K3 + coldest 20% experts deleted (zeroed) | `2.40` | `0.6633` | `0.1755` | `73.06%` |
| K3 + coldest 20% deleted + 6L distilled Engram (`450K` tok) | `2.40` | `0.2060` | `0.0496` | `84.97%` |
| K3 + Layers 0 to 5 deleted + 6L distilled Engram (`450K` tok) | `2.44` | `2.0389` | `1.2711` | `47.64%` |

While n-gram LUT replacement reduces expert-deletion damage `3.2x` to `4.7x` (`0.7275 -> 0.1970` at `2.06` bpw), keeping all experts alive in VRAM at `2.00` to `2.25` bpw plus residual Engram QEC (`0.0891` to `0.1312` KLD) dominates expert deletion at matched VRAM (`0.1970` to `0.2060`).

The training-free **n-gram predictability probe** (`trellismx_kit/engram_probe.py`, held-out across 9 of 44 fit windows on Granite 3.1 3B) explains *why* asymmetric co-allocation works:

| Layer Index | `L0` | `L1` | `L2` | `L4` | `L8` | `L16` | `L24` | `L31` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Held-Out BF16 Bigram $R^2$ | **`0.937`** | `0.687` | `0.536` | `0.330` | `0.178` | **`0.074`** | `0.162` | **`0.246`** |

At Layer 0, simple token bigrams explain **`93.7%`** of BF16 expert output variance (`NMSE = 0.0541`, best single expert $R^2 = 0.987$, `NMSE = 0.0093`). Local predictability drops steeply into the deep contextual core (`L16` = `7.4%`) before rebounding near the output boundary (`L31` = `24.6%`). This U-shaped profile justifies discounting early and final layers (`engram_discount = 0.45`) so scarce `k3`/`k4` VRAM bits flow into deep contextual layers.

### 4.7 SM120 `b12x` Four-Rate Per-Expert Grouped MoE Kernel Verification

Extending serving kernels to dispatch four per-expert rates inside one MoE layer (`patches/b12x-per-expert.patch`, `design/PER-EXPERT-KERNEL.md`) was verified on an NVIDIA RTX PRO 6000 Blackwell (`SM120`) using synthetic single-layer `TP4` MoE layers matching GLM-5.3-Flash geometry (`288` experts, `hidden_size = 4096`, `local_intermediate_size = 512`, `top_k = 8`):

| Verification Gate (`SM120` Single-Layer Synthetic `TP4`) | Scope | Result |
|---|---|---|
| Per-expert stream decode probe | All 288 experts across `{k2, p25, k3}` | Bit-exact E4M3 operand match vs `trellismx_kit/pe_format.py` |
| Uniform parity (`k2`, `k3`, `k4`) | Per-expert runtime vs uniform program | Bit-for-bit identical (`torch.equal`) |
| 3-rate mixed layer (`k2, p25, k3`) | Rows `1, 8, 16, 64, 512` across all 3 banks | Cosine `0.988` to `0.998` vs FP32 ref; bit-exact to uniform K2 |
| 4-rate mixed layer (`k2, p25, k3, k4`) | Rows `1, 16, 64` across all 4 banks | Cosine `0.988` to `0.995` vs FP32 ref |
| Bank permutation invariance | Roll-96 expert rate permutation | Bit-for-bit identical (`torch.equal`) |
| Small-M guard | `small_m_scheduler = True` on per-expert source | Fails closed as specified |
| `compute-sanitizer --tool memcheck` | 3-rate and 4-rate mixed layers across all banks | `0` errors |
| Single-layer timing: Mixed `{k2, p25, k3}` vs Uniform `k3` (5 replays) | Rows `1 / 8 / 16 / 64 / 512` | `2.15 / 1.94 / 1.94 / 1.93 / 2.90` ms vs `1.96 / 1.91 / 2.04 / 1.96 / 2.62` ms |

In these single-layer micro-benchmarks, mixed-rate `{k2, p25, k3}` latency tracks uniform `k3` within `~2%` to `11%` (slightly slower at `rows = 1` and `512` due to multi-bank dispatch overhead, slightly faster at `rows = 16` and `64`). Status: `qualified` on synthetic single-layer SM120 `TP4` benchmarks; full end-to-end token/s serving throughput and live serving-engine integration (`patches/flare-per-expert.patch`) remain `research-only` and deferred to deployment.

### 4.8 Cross-Model Precedents Supporting Codec and Allocation Primitives

Three independent empirical milestones anchor the underlying `procedural-npk` codec law, per-expert allocation theory, and GLM-5.3-Flash coupled P8 lineage:

| Precedent System / Model | Scope and Metric | Verified Empirical Result |
|---|---|---|
| **Gaussian Codebook Screening (`procedural-npk`, `compander_scale = 2.0`, E4M3)** | Held-out standard normal $\mathcal{N}(0, 1)$ MSE vs Shannon bound $D(R) = 2^{-2R}$ (`0.0625` at 2.0b, `0.03125` at 2.5b, `0.015625` at 3.0b, `0.003906` at 4.0b) | `k2`: **`0.065128`** (`1.042x` Shannon); `p25`: **`0.033772`** (`1.081x` Shannon); `k3`: **`0.016617`** (`1.063x` Shannon), confirming near-Shannon MSE halving per half-bit |
| **ShapleyMCG on `Qwen/Qwen3-30B-A3B`** (`128` experts/layer, `3.50` exact expert bpw) | `20,480`-pos WikiText KLD (`Base`: `K3/K4` mix vs uniform `K3 = 0.0994`, `K4 = 0.0340`; post-trained: vs carried-surplus on matched reconstructions) | `Qwen/Qwen3-30B-A3B-Base`: `0.045294` (`22%` below geometric mean `0.0581`, **`-7.73%`** vs routed-p2 `0.049089`, beats 5-seed random `0.0700`); Post-trained: **`-13.77%`** KLD reduction and `+0.60` pp Top-1 vs TurboDerp v0.0.1 carried-surplus (`10/10` rows) |
| **GLM-5.3-Flash Coupled P8 `K4/K5` Precedent** (also quantized directly from `zai-org/GLM-5.3-Flash-BF16`) | Sealed 32-window confirmation panel, true-decode KLD (published coupled P8 `K4/K5` reference vs uniform coupled P8 `K4`) | Published coupled P8 `K4/K5` reference: `0.03106` (FP8 KV) / `0.03501` (NVFP4 KV) (`0.03195` / `0.03546` in matched true-decode replay); uniform coupled `K4`: `0.03697`; 17-layer `K5` probe (`4.659` bpw): `0.03418` (`-7.54%`, `25/32` wins, proving rate monotonicity in served P8 kernels) |


---

## 5. Architectural and Recipe Changes Made for `zai-org/GLM-5.3-Flash-BF16`

Scaling Dual-Trellis QEC from the `3.02B`-expert `ibm-granite/granite-3.1-3b-a800m-instruct` validation benchmark up to the `304.41B`-expert `zai-org/GLM-5.3-Flash-BF16` target (`100x` more routed parameters, hybrid linear/sparse attention, 4-stream hyper-connections, and 64K long-context training) required deliberate architectural adaptations. Table 5.1 compares the two configurations directly.

### 5.1 Controlled Side-by-Side Comparison: Granite 3.1 3B v1 vs. GLM-5.3-Flash v2

| Dimension | Granite 3.1 3B QEC v1 (`qualified`) | GLM-5.3-Flash Dual-Trellis QEC v2 (`implemented` / active campaign) | Failure Mode Prevented or Gain Unlocked |
|---|---|---|---|
| **Base Checkpoint Provenance** | `ibm-granite/granite-3.1-3b-a800m-instruct` (`32` routed layers, `40` experts/layer, `3.02B` expert weights, vocab `49,155`) | Quantized directly from `zai-org/GLM-5.3-Flash-BF16` (`42` routed layers `3..44`, `288` experts/layer, `304,405,807,104` routed weights, vocab `154,880`; non-routed kept in BF16 as stored) | Clean BF16 provenance; zero double-quantization error |
| **Rate Set & Allocation** | `{k2, p25, k3}` (uniform or per-expert attribution / route mass) | `{k2, p25, k3, k4}` hierarchical exact half-bit knapsack + `engram_discount = 0.45` + non-binding `max_k4_fraction` guardrail (`0.06` / `0.08`) | Reserves `k4` for heavy hitters while shifting bits into deep low-$R^2$ layers |
| **Engram Bank Topology** | 12 independent per-layer banks (`1,283M` host params = `+6.80 bpw` BF16 on Granite) | 3 shared depth banks (`early: 3,5,7,9`; `mid: 12,16,20,24,28,32`; `late: 36,39,42,44`; `1,669,999,872` table params across `25,629,108` rows = `3.111 GiB` BF16 / `1.651 GiB` FP8+scales) | `3x` smaller host RAM, `~4x` fewer host lookups (`3.75 KiB/tok` BF16 / `~1.9 KiB/tok` FP8), `4x` gradient density per bucket |
| **N-Gram Features & Init** | Raw uni + raw bi + raw tri; Gaussian-init hash tables; scalar gate | Raw uni (`d=128`) + zero-init residual 4x bi (`2` raw, `2` NFKC) + 2x NFKC tri + 2x NFKC skip-gram (`d=64`); `G = 8` per-order gating; zero-init router gate `[4, 288]`; zero-init width-4 causal DWConv1D | Clean zero-residual step-0 start; improved casing/whitespace generalization; 6+ token local receptive field; routing-aware gating |
| **Engram Hook Placement** | Full transformer block output (`block`) | `block.mlp` (`engram_hook_target = "mlp"`, auto-detected via `hasattr(block, "ffn_hc")`) | GLM-5.3-Flash uses 4-stream mHC hyper-connections; `block.mlp` operates on the collapsed 3D `[B, T, 4096]` tensor |
| **Mergeable LoRA Scope** | Standard GQA `q/k/v/o` projections only | Hybrid 34 KDA (`q/k/v/o`) + 11 MLA-NoPE (`q_a, q_b, kv_a_with_mqa, kv_b, o`) + 3 dense prefix MLPs (`0..2`) + 42 shared experts + untied `lm_head` (FP32 bottleneck only) | Captures hybrid linear+sparse attention calibration, dense prefix drift, shared experts, and untied head geometry at zero serving overhead |
| **RMSNorm Precision** | Native BF16 norms in v1 | FP32-promoted text-backbone RMSNorms (routers/indexers/vision excluded), restored to carrier BF16 at merge | Eliminates BF16 ULP (`0.0078125`) gradient stagnation against `3e-4` AdamW updates |
| **Per-Expert `inter_scales` & `expert_bias`** | Tested in Granite v2 (`[L, E, 3, I]` scales + `[L, E, H]` bias) | **Intentionally omitted** on GLM-5.3-Flash (`--train-inter-scales` and `--train-expert-bias` disabled) | P8 `intermediate_scales_fp16` live in the coupled `H128`-rotated domain (not logical BF16), and DP8+EP8 replicated-QEC sync forbids EP-sharded trainable tensors |
| **Distillation Corpus & Curriculum** | `10M` to `50M` short sequences (`<= 1%` 13-gram screen) | `100,102,231` supervised positions, 11 Nemotron domains, **strict `0.0` stride-1 13-gram overlap**, 80/20 interleaved `4K` / `{16K, 32K, 64K}` curriculum | Calibrates Attention LoRA and Engram gates across the `index_topk = 2048` dense-to-sparse DSA regime up to 64K |
| **Long-Context & Distributed Engine** | 4x RTX PRO 6000 standard PyTorch DDP | 8x B200 (`1.31 TiB` HBM3e) DP8+EP8 `all_to_all` (`tp_plan={'lm_head':'replicate'}`), `65,536`-row chunked `grouped_mm`, `fla` Triton `causal_conv1d`, query-chunked DSA indexer (`2048`) & attention (`4096`) | Eliminates `256 + 20 GiB` 64K OOM (peak `< 5 GiB`), avoids FSDP2 mixed-dtype crash, maintains 8x DP throughput |
| **Distillation Objective** | Top-K KL (v1) -> tail-lumped top-K KL (v2) | Chunked (`512`-tok checkpointed) log-space tail-lumped top-128 KL lower bound from pre-dumped BF16 shards (`72.719 GiB`) | Exact head-vs-tail mass calibration without FP32 subtraction cancellation or 64K logit OOM |

### 5.2 Why Logical-Domain `inter_scales` and `expert_bias` Are Omitted on GLM-5.3-Flash

Although fine-tuning logical-domain intermediate neuron scales and per-expert output centroid biases reduced Uniform K2 KLD on Granite 3.1 3B (`0.1436 -> 0.1384`, `Section 4.5`), omitting both on GLM-5.3-Flash enforces two strict correctness invariants:

1. **Coupled-Domain Incompatibility for `intermediate_scales_fp16`.** Inside TrellisMX P8 sidecars (`Section 2.3`), `intermediate_scales_fp16` sits *between* two `H128` block-Hadamard transforms in the rotated serving domain. Because the distillation harness decodes quantized sidecars back to logical BF16 expert weights (`decode_coupled_scale_weights(..., invert_coupled_scales=True)`) to execute PyTorch `grouped_mm`, scaling logical post-SwiGLU activations cannot fold cleanly back into the rotated `intermediate_scales_fp16` sidecar tensor without re-encoding or adding extra serving ops. Keeping routed expert sidecars 100% immutable guarantees bit-exact serving layout preservation.
2. **DP8+EP8 Replicated-QEC Synchronization.** Under `--distributed-ep` (`ep_size = 8`), each GPU owns only `36` of the `288` experts per layer, whereas all QEC parameters (LoRAs, RMSNorms, Engram banks/projections) are replicated across all 8 GPUs and averaged via NVLink `all_reduce(AVG)`. Mixing EP-sharded trainable tensors into the replicated optimizer would corrupt cross-rank averaging; `scripts/distill_qec.py` enforces this fail-closed (`ValueError`).

### 5.3 Exact Three-Layer Bit-Width and Unified-Memory Footprint Accounting for GLM-5.3-Flash

Transparent footprint accounting over GLM-5.3-Flash's exact **`304,405,807,104`** routed parameters (`42 layers x 288 experts x 3 matrices x 4096 x 2048`) separates (1) trellis weight payload, (2) total on-disk GPU CSF sidecars (trellis + lossless CSF scales + coupled vectors), (3) Host-RAM Engram v2 tables (`1,669,999,872` parameters across `25,629,108` rows), and (4) 14 per-layer BF16 Engram GPU projections (`0.138 GiB` = `+0.0039 bpw`):

| Artifact Variant (`release_2375`) | Trellis Payload (GiB / bpw) | Total On-Disk GPU CSF Sidecar (GiB / bpw) | Host-RAM Engram Tables (GiB / bpw) | Total Effective bpw (over Payload / over Stored Sidecar) | Total Effective bpw (+ `0.138 GiB` GPU Engram Proj.) |
|---|---:|---:|---:|---:|---:|
| Zero-Sidecar Merged Carrier Only | `84.164` / `2.375` | `85.651` / `2.417` | `0.000` / `0.0000` | `2.375` / **`2.417`** | **`2.417`** |
| **Shipped FP8 Engram Sidecar** (`1.555 GiB` E4M3 + `0.0955 GiB` FP32 row scales) | `84.164` / `2.375` | `85.651` / `2.417` | `1.651` / `+0.0466` | `2.422` / **`2.464`** | **`2.468`** |
| **Shipped BF16 Engram Sidecar** | `84.164` / `2.375` | `85.651` / `2.417` | `3.111` / `+0.0878` | `2.463` / **`2.505`** | **`2.509`** |

**Single-Node 128 GB Unified-Memory Footprint (e.g., NVIDIA DGX Spark, `119.2 GiB` Total Unified Memory).**  
On a unified-memory architecture such as DGX Spark, GPU VRAM and host RAM share the same `119.2 GiB` physical memory pool, so the Engram sidecar counts directly against the same physical budget (while still avoiding random PCIe expert-offload stalls on discrete-GPU clusters). Table 5.3b reports primary totals using the **tested, unmodified `18.03 GiB` BF16 non-routed carrier** alongside an explicitly marked *unevaluated future optimization* using a hypothetical `~10.5 GiB` text-only/FP8 non-routed carrier:

| Deployment Configuration (with `1.651 GiB` FP8 Engram + `0.138 GiB` GPU Engram Proj.) | Routed GPU CSF Sidecars | Non-Routed Carrier | Engram Tables + GPU Proj. | **Total Model Footprint** | Remaining Pool Inside `119.2 GiB` | Status |
|---|---:|---:|---:|---:|---:|---|
| **`release_2375` + Tested BF16 Carrier** | `85.651 GiB` | `18.03 GiB` (BF16 as stored) | `1.789 GiB` | **`105.47 GiB`** | `13.73 GiB` | Primary tested carrier |
| **`release_2500` + Tested BF16 Carrier** | `90.080 GiB` | `18.03 GiB` (BF16 as stored) | `1.789 GiB` | **`109.90 GiB`** | `9.30 GiB` | Primary tested carrier |
| `release_2375` + Hypothetical Slimmed Carrier | `85.651 GiB` | `~10.5 GiB` (text-only/FP8) | `1.789 GiB` | `~97.94 GiB` | `~21.26 GiB` | Unevaluated future optimization |
| `release_2500` + Hypothetical Slimmed Carrier | `90.080 GiB` | `~10.5 GiB` (text-only/FP8) | `1.789 GiB` | `~102.37 GiB` | `~16.83 GiB` | Unevaluated future optimization |

Both `release_2375` (`105.47 GiB`) and `release_2500` (`109.90 GiB`) fit inside `119.2 GiB` unified memory *even while retaining the full unmodified `18.03 GiB` BF16 multimodal non-routed carrier*, with zero expert offloading.

---

## 6. Evaluation Protocol, Live Pre-QEC Baseline, Active Campaign Status, and Limitations

### 6.1 Local Inference Lab Exact FP64 Full-Vocabulary KLD Protocol (`trellismx_kit/kld_score.py`)

All qualification scores follow the **Local Inference Lab exact FP64 full-vocabulary KLD protocol**:

1. **Exact Metric Definition (Kullback and Leibler 1951):** At every supervised position $r \in \{0, \dots, T-2\}$ (row $r$ predicts realized token $r+1$) across the full vocabulary ($V = 154,880$ on GLM-5.3-Flash; $V = 49,155$ on Granite 3.1 3B), teacher and student logits cast to IEEE-754 double precision (`float64`), compute exact `log_softmax`, and evaluate:
   $$D_{\text{KL}}\bigl(p_{\text{teacher}} \,\|\, q_{\text{student}}\bigr) = \sum_{v=1}^{V} p_{\text{teacher}}(v) \Bigl(\log p_{\text{teacher}}(v) - \log q_{\text{student}}(v)\Bigr) \ge 0.$$
   Evaluation executes via **GPU-chunked IEEE-754 FP64** (`compute_device = out_logits.device`, `chunk = 512` on GPU 0, with a `64`-thread CPU FP64 fallback at `chunk = 1024`). Any `NaN` or `+Inf` logit raises a hard error immediately.
2. **Strict Single-Window Determinism (`batch_size = 1`).** Running evaluation window 0 batched alongside window 1 introduces a `0.002770` mean KLD drift on Granite 3.1 3B due to non-associative GEMM reduction order across batch rows. Every evaluation window therefore runs singly (`batch_size = 1`, `use_cache = False`) with `allow_tf32 = False` and `highest` FP32 matmul precision.
3. **Integrity Canaries and Fail-Closed Policy:**
   - **Always Fail-Closed:** Teacher self-KLD canary on window 0 of every split (`max(KL(t || t)) == 0.0` verbatim), `NaN`/`+Inf` rejection in `kld_score.py`, and `scripts/dump_teacher_logits.py`'s per-rank `>= 0.15` next-token alignment canary before shard commit.
   - **Realized-Token Alignment Band (`ALIGNMENT_BAND = [0.20, 0.995]`):** In `evaluate_dense_dir`, checking that teacher `argmax` matches realized next token `token_ids[1:]` inside `[0.20, 0.995]` flags anomalies via `alignment_ok` and console warnings by default, and fails closed with `RuntimeError` when `strict_alignment = True`.
4. **Uncertainty and Tail Statistics (`summarize_windows`):** Weighted window-block **BCa 95% confidence interval** (Efron 1987; `5,000` bootstrap resamples, `seed = 20260829`, jackknife acceleration), median / `p90` / `p99` / `p99.9` / `max`, **Top-1 agreement + 95% Wilson score interval** (Wilson 1927), and **perplexity ratio** $\exp(\overline{\log p_{\text{teacher}}(\text{realized})} - \overline{\log q_{\text{student}}(\text{realized})})$.

### 6.2 Multi-Panel Evaluation Matrix, Complete Measured Pre-QEC `release_2375` Baseline, and Active Campaign Status

The 100M-token GLM-5.3-Flash QEC distillation campaign evaluates across six zero-overlap dense FP32 teacher panels (`108` files = `574.256 GiB`) and reports all three post-merge PyTorch artifact variants (`zero-sidecar merged carrier`, `shipped BF16 Engram`, and `shipped FP8 Engram`) alongside both full (`ood_final_all25`) and clean-subset (`ood_final_clean17`) 2K panels. All six pre-distillation (`0.0M` token) panels for `release_2375` (`2.375 bpw` trellis payload / `2.417 bpw` total on-disk CSF sidecars) are measured under GPU-chunked IEEE-754 FP64 scoring (`eval_32k` cross-checked bit-identical against 64-thread CPU FP64):

| Evaluation Panel | Context Length | Windows / Positions | Pre-QEC `release_2375` Baseline (`0.0M` Tokens, Exact FP64) | Milestones (`10M / 35M / 50M / 75M / 100M`) | Post-Merge 3-Way Variants (`Carrier-Only` / `BF16 Engram` / `FP8 Engram`) |
|---|---:|---:|---|---|---|
| `ood_confirmation_2k` (Untouched OOD confirmation) | `2,048` | `28` / `57,316` | **Mean KLD = `0.05303`** (BCa `[0.04501, 0.06177]`, Med `0.00625`, p99 `0.64883`), **Top-1 = `92.43%`**, PPL ratio = `1.0144`, align = `0.7411` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` |
| `ood_final_all25` (`roles-v5` OOD anchor, all 25) | `2,048` | `25` / `51,175` | **Mean KLD = `0.06340`** (BCa `[0.04849, 0.09693]`, Med `0.00483`, p99 `0.91801`), **Top-1 = `92.03%`**, PPL ratio = `1.0253`, align = `0.7540` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` |
| `ood_final_clean17` (`roles-v5` OOD anchor, clean 17) | `2,048` | `17` / `34,799` | **Mean KLD = `0.07473`** (BCa `[0.05570, 0.12428]`, Med `0.01306`, p99 `1.02417`), **Top-1 = `90.54%`**, PPL ratio = `1.0350`, align = `0.7077` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` |
| `eval_8k` (Multi-domain 8K, past `index_topk = 2048`) | `8,192` | `24` / `196,584` | **Mean KLD = `0.09700`** (BCa `[0.06837, 0.16720]`, Med `0.00572`, p99 `1.69367`), **Top-1 = `92.35%`**, PPL ratio = `1.0176`, align = `0.7989` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` |
| `eval_32k` (Long-context sparse retrieval) | `32,768` | `8` / `262,136` | **Mean KLD = `0.03910`** (BCa `[0.02971, 0.05347]`, Med `0.00022`, p99 `0.38802`), **Top-1 = `93.77%`**, PPL ratio = `1.0037`, align = `0.8072` *(bit-identical CPU/GPU FP64)* | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` |
| `eval_64k` (Extreme 64K sparse retrieval) | `65,536` | `6` / `393,210` | **Mean KLD = `0.01080`** (BCa `[0.00698, 0.01581]`, Med `0.00195`, p99 `0.06068`), **Top-1 = `75.36%`**, PPL ratio = `1.0054`, align = `0.5502` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` | `[Pending completion of active 100M-token GLM-5.3-Flash QEC distillation campaign]` |

**Measured Pre-QEC Baseline Significance:** Even *before a single step of QEC distillation* (`0.0M` tokens), the four-rate per-expert `release_2375` checkpoint (`2.375 bpw` trellis payload / `2.417 bpw` on-disk CSF sidecars) achieves **`0.05303` to `0.07473` mean KLD (`90.54%` to `92.43%` Top-1)** across the three 2K OOD panels and **`0.03910` mean KLD (`93.77%` Top-1)** at 32K context, with every panel passing the `[0.20, 0.995]` realized-token alignment check and keeping perplexity ratio within `1.0037` to `1.0350`. On `eval_64k`, the low mean KLD (`0.01080`) combined with lower Top-1 agreement (`75.36%`) and lower teacher alignment (`0.5502`) reflects high-entropy synthetic long-context token streams where near-tied probabilities flip `argmax` despite tiny distributional divergence.

**Campaign Status (`research-only` / in-progress):** Completed artifacts include the 64-window calibration capture, dual exact rate assignments, `15,058`-job union encode, verified `168`-file TP4 CSF releases (`85.651 GiB` and `90.080 GiB`), `100,102,231`-position zero-overlap Nemotron corpus, public `646.975 GiB` teacher archive (`brandonmusic/GLM-5.3-Flash-BF16-Nemotron-100M-Teacher-Logits`), and all six pre-QEC baseline panels above. Active 100M-token distillation on 8x NVIDIA B200 targets sealed publication at `brandonmusic/GLM-5.3-Flash-TrellisMX-2.375bpw-QEC` upon completion and passing of the 3-way post-merge PyTorch artifact-roundtrip verification.

### 6.3 Numerical Noise Floor and Limitations

Rigorous interpretation of all results requires six explicit caveats:

1. **Numerical Noise Floor (`~0.0028` KLD).** Single-window vs. 2-window batched forward passes on Granite 3.1 3B differ by `0.002770` mean KLD strictly from floating-point GEMM reduction order. Differences smaller than `~0.0028` nats on Granite (such as `0.0799` vs `0.0810`, `0.0815` vs `0.0819`, and `0.1384` vs `0.1390`) should therefore be interpreted as statistical or numerical ties rather than decisive margins, whereas large structural gaps (`0.2348 -> 0.1312`, `0.1106 -> 0.0891`, `0.1015 -> 0.0799`) sit an order of magnitude above the noise floor.
2. **Repeated Evaluation Window Reuse on Granite 3.1 3B.** The Granite 3.1 3B v1 -> v2 design iteration evaluated roughly 20 training-free variants and 10 QEC arms on the same 21-window held-out set, introducing mild selection bias for 1% to 3% differences. By contrast, GLM-5.3-Flash preserves an untouched `ood_confirmation_2k` panel alongside sealed `roles-v5` panels.
3. **25x Sidecar-to-Expert Capacity Ratio Gap Between 3B and 304B.** On Granite 3.1 3B, the Dual-Trellis v2 host-RAM sidecar (`421M` params) equals `14%` of routed expert parameters (`3.02B`), adding `+2.23 bpw` (BF16) or `+1.12 bpw` (FP8) in host RAM. On GLM-5.3-Flash, the `1.67B`-param sidecar equals only **`0.55%`** of routed expert parameters (`304.41B`), adding just `+0.0466 bpw` (FP8) or `+0.0878 bpw` (BF16). Because the relative capacity ratio is `~25x` smaller at 304B scale, Granite's fractional KLD reductions (`-19%` to `-41%`) cannot be assumed to transfer linearly; measuring GLM-5.3-Flash directly across the 100M-token campaign is empirically mandatory.
4. **Proxy Assumptions in Asymmetric Co-Allocation.** `trellismx_kit/engram_probe.py` measures held-out bigram $R^2$ on unmodified BF16 expert outputs rather than directly on sub-3-bit quantization residuals, and `assign_dual.py` applies a smooth U-shaped heuristic $R^2_l$ prior across all 42 routed GLM-5.3-Flash layers (including layers without attached Engram hooks).
5. **In-Domain Held-Out vs. True Out-of-Distribution Panels.** Although every training sequence passes a strict `0.0` stride-1 13-gram screen against every evaluation window, `eval_8k`, `eval_32k`, and `eval_64k` draw from the same Nemotron domain families as the training corpus (in-domain held-out), whereas `ood_final_all25`, `ood_final_clean17`, and `ood_confirmation_2k` measure true out-of-distribution generalization. Both families must be inspected together when evaluating post-distillation results.
6. **Unmeasured End-to-End Serving KLD and Prefetch Overlap.** Post-merge verification establishes exact PyTorch artifact-roundtrip parity (with decoded logical BF16 experts). End-to-end serving KLD through fused `b12x`/Flare four-rate kernels, real-world interactive token/s throughput, and measured DDR5/PCIe overlap for `3.75 KiB/tok` (or `~1.9 KiB/tok`) Engram prefetch remain deferred to serving-engine integration and deployment benchmarking.

---

## 7. Acknowledgments, Lineage, Related Work, and Third-Party Credits

### 7.1 Compute Sponsorship and Acknowledgments

Sincere gratitude is extended to **Lambda Labs** for generously sponsoring and donating the 8x NVIDIA B200 GPU compute that powered this work. The GLM-5.3-Flash scale experiments, including calibration capture, four-rate per-expert quantization of `304.41B` routed parameters, generation and public archival of the `646.975 GiB` 100.1M-token BF16 teacher-logit dataset, and the 100M-token Dual-Trellis QEC distillation and exact FP64 KLD qualification campaign, would not have been possible without Lambda Labs' compute sponsorship.

### 7.2 Author Disclosure and Upstream Attribution

**Author Disclosure:** Brandon M. Music is the author of this paper, the founder and license author of Local Inference Lab, and the author of the `procedural-npk` codec law and `ShapleyMCG` allocation framework.

Table 7.1 provides specific attribution to every upstream research paper, software component, hardware partner, and dataset foundation underlying this work.

| Component / Lineage | Author(s) / Organization | Specific Technical Contribution and Role |
|---|---|---|
| **GLM-5.3-Flash Base Model** | Z.ai (`zai-org/GLM-5.3-Flash-BF16`) | Unmodified BF16 source checkpoint (`304.41B` routed + `9.67B` non-routed; hybrid 34 KDA / 11 MLA-NoPE + DSA + 4-stream mHC). All GLM-5.3-Flash TrellisMX artifacts (including the P8 K4/K5 precedent and the `2.375` / `2.500` bpw releases) are quantized directly from this BF16 checkpoint. |
| **Granite 3.1 3B Validation Model** | IBM (`ibm-granite/granite-3.1-3b-a800m-instruct`, Apache-2.0) | 32-layer, 40-expert MoE qualification model used for the training-free frontier, YAQA negative controls, 50M-token v1 scaling curve, and 10M-token Dual-Trellis QEC v2 ablation suite. |
| **Donated Compute Infrastructure** | **Lambda Labs** | Generously donated 8x NVIDIA B200 GPU compute powering GLM-5.3-Flash calibration, four-rate quantization, public 100.1M-token teacher-logit archiving, and the 100M-token Dual-Trellis QEC distillation campaign. |
| **vLLM Serving Image** | Local Inference Lab and Luke Alonso | Containerized vLLM serving image lineage (`ghcr.io/local-inference-lab/vllm`, Apache-2.0). |
| **CSF Scale Encoder/Decoder & FP64 KLD Protocol** | Local Inference Lab (CSF first added by Martin Vit, updated by Luke Alonso) | Lossless Compressed Sparse Format (CSF) UE8M0 scale codec design and the sealed exact IEEE-754 FP64 full-vocabulary KLD evaluation protocol. |
| **`b12x` Kernel Suite** | Luke Alonso (`github.com/lukealonso/b12x`), with Martin Vit and Local Inference Lab contributors (Apache-2.0) | SM120 Blackwell CUDA/CuTe P8 fused MoE and trellis kernels extended by `patches/b12x-per-expert.patch` for four-rate `{k2, p25, k3, k4}` per-expert dispatch. |
| **Codec Operand Adaptation (`KQuant` / `QSRT`)** | Luke Alonso | Inspiration for decoding trellis paths directly to native hardware FP8/FP4 operands (`W4A8` trellis path) and the tail-biting Viterbi tile-search CUDA lineage. |
| **QTIP** | Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa (NeurIPS 2024, `arXiv:2406.11235`) | Bitshift trellis-coded quantization with incoherence processing, multi-weight emission, and the 3INST bit-masking lineage (`0x8FFF8FFF`, `0x3B603B60`). |
| **ExLlamaV3 / EXL3** | `turboderp` and ExLlamaV3 contributors (MIT) | Practical LLM trellis quantization, packed tile layouts, and MCG multiplier lineage (`0xCBAC1FED`) from which TrellisMX descends. |
| **`procedural-npk` and `ShapleyMCG`** | Brandon M. Music | Zero-storage 16-bit procedural E4M3 state law (`procedural-npk`, Neural Processing of K, `compander_scale = 2.0`, LCG pair mixer, bounded finite-output guarantee) and exact-budget Aumann-Shapley MoE bit-allocation framework (`ShapleyMCG`). |
| **DeepSeek Engram & Hash Embeddings** | DeepSeek (`arXiv:2601.07372`); Svenstrup, Hansen, and Winther (NeurIPS 2017) | Deterministic $O(1)$ multi-head prime-hashed n-gram conditional memory with context gating (Engram) and classical hashed embedding tables, adapted in Dual-Trellis QEC v2 as a shared 3-bank post-quantization error-correction coprocessor. |
| **Core Model & Training Primitives** | Hu et al. (**LoRA**, ICLR 2022, `arXiv:2106.09685`); Hinton, Vinyals, and Dean (**Knowledge Distillation**, 2015); Yang and Zhang (**Flash Linear Attention / `fla`**); **Kimi Linear / KDA** (`arXiv:2510.26692`); **DeepSeek-V3.2 / DSA**; Xie et al. (**mHC**, 2026) | Mergeable low-rank adapters, teacher-student logit distillation, Triton `causal_conv1d` acceleration, hybrid KDA linear attention, DeepSeek Sparse Attention, and Manifold-Constrained Hyper-Connections. |
| **Quantization, Information Theory, & Statistics Foundations** | Frantar et al. (**GPTQ**); Chee et al. (**QuIP**); Tseng et al. (**QuIP#**); Ashkboos et al. (**QuaRot**); Liu et al. (**SpinQuant**); **OCP MX v1.0**; Marcellin and Fischer (**TCQ**, 1990); **Viterbi** (1967); Kullback and Leibler (**KL**, 1951); Efron (**BCa**, 1987); **Wilson** (1927) | Second-order Hessian column-block error feedback, Hadamard incoherence processing, `UE8M0` `K = 32` microscaling, trellis-coded quantization, Viterbi dynamic programming, KL divergence, and bootstrap/binomial confidence intervals. |
| **MoE Offloading, Fractional-Bit, & Attribution Literature** | Eliseev and Mazur (2023), **MoE-Infinity**, **Pre-gated MoE**, **Fiddler** (MoE offloading/prefetch); Lee and Song (**Q-Palette**, NeurIPS 2025, `arXiv:2509.20214`); Huang et al. (**MC-MoE**, ICLR 2025, `arXiv:2410.06270`); **MoPEQ** (`arXiv:2509.02512`); **GEMQ** (ICML 2026, `arXiv:2605.23078`); **Q-Strata** (`arXiv:2608.30564`); **IMPQ / CoopQ** (`arXiv:2509.15455`); **Mixture of Lookup Experts** (`arXiv:2503.15798`); Tseng et al. (**YAQA**, `arXiv:2505.22988`); Joshua Hill (*Saturation Makes Quantization Error Additive*, `arXiv:2607.12266`) | MoE offloading baselines motivating zero-offload sub-3-bit compression; fractional-bit trellis kernels, per-expert bit allocation, lookup-augmented MoE layers, Shapley/coverage attribution, and two-sided Fisher rounding (evaluated and excluded in `Section 4.1`). |
| **Calibration, Distillation, and Evaluation Datasets** | **NVIDIA Nemotron** (Science under CC-BY-SA-4.0; Long-Context Transduction under OpenMDW-1.1; Legal under CC-BY-4.0; other Nemotron SFT and Pretraining splits under the NVIDIA Open Data License Agreement) and **Nebius** (`nebius/SWE-agent-trajectories`) | Multi-domain calibration, evaluation, and 100.1M-token zero-overlap distillation text across 11 technical and natural-language domains. |


---

## 8. Comprehensive Plain-English Technical Glossary

This alphabetical glossary defines every machine-learning, quantization, statistical, and GPU-systems term referenced in this paper so a systems engineer, physicist, or software architect can read the entire specification without external references.

- **AdamW / Weight Decay / Cosine Schedule:** The standard adaptive gradient optimizer used in neural-network training. AdamW tracks running first and second moments of gradients per parameter, applies decoupled L2 shrinkage (*weight decay*) to weights (disabled on Engram hash tables and RMSNorms here so sparse buckets and normalization gains do not shrink artificially), and smoothly ramps the learning rate up linearly (*warmup*) before decaying along a half-cosine curve to a small floor (`0.02x`).
- **Attention (Self-Attention):** The core transformer mechanism that lets each token look back at earlier tokens in a sequence and blend their representations via learned query ($Q$), key ($K$), and value ($V$) projections. Attention mixes information *across sequence positions*, whereas feed-forward and MoE layers transform each token *independently*.
- **Aumann-Shapley Attribution:** A cooperative game-theory attribution rule integrated along a continuous path from unquantized to quantized weights that fairly divides total model quality loss (KL divergence) across individual layers or experts, accounting for nonlinear interactions among them.
- **BCa Bootstrap Confidence Interval (Bias-Corrected and Accelerated):** A non-parametric statistical interval (Efron 1987) computed by resampling entire evaluation windows with replacement (`5,000` resamples in this protocol) and adjusting for median bias and skewness via leave-one-out jackknife acceleration. Resampling whole windows respects token autocorrelation inside documents.
- **BF16 (Brain Floating Point 16-bit):** A 16-bit floating-point format with 1 sign bit, 8 exponent bits (matching 32-bit FP32 dynamic range), and 7 mantissa bits (step size `0.0078125` near `1.0`). Standard format for unmodified LLM reference weights.
- **BPE (Byte-Pair Encoding):** A subword tokenization algorithm that iteratively merges frequent character or byte pairs into vocabulary tokens (`154,880` tokens in GLM-5.3-Flash). Incomplete UTF-8 multi-byte BPE pieces must be skipped during NFKC canonicalization to avoid corrupting raw byte tokens.
- **Carrier and Sidecar:** In a TrellisMX deployment, the **carrier** is the standard Hugging Face model directory holding configuration, tokenizer, and unmodified BF16 non-routed tensors (embeddings, attention, norms, dense prefix, shared experts, `lm_head`), while **sidecars** are compact safetensors files storing the compressed per-expert trellis streams, lossless CSF scale planes, coupled rotation vectors, or Host-RAM Engram lookup tables.
- **Cholesky Decomposition:** Factoring a symmetric positive-definite matrix $H$ into $L L^\top$, where $L$ is lower-triangular. Used in GPTQ to invert damped input Hessians stably and solve triangular error-feedback systems.
- **Compander (`compander_scale`):** A fixed scalar gain (`2.0` in `procedural-npk`) applied to the summed FP16 halves before E4M3 projection so the procedural codebook amplitude matches the scale of Hadamard-rotated neural weights while staying strictly below the `448.0` E4M3 ceiling.
- **CSF Scale Compression (Compressed Sparse Format):** Lossless compression for `UE8M0` power-of-two block scale planes (`ue8m0-row-base-offset1-u32-exceptions.v1`). Each row stores an optimal adjacent exponent pair $[b, b+1]$ as a packed 1-bit bitplane plus a sparse `(value, index)` exception table, shrinking scale overhead from `0.25` bpw down to `~0.04` bpw with bit-exact round-trip recovery.
- **CTA (Cooperative Thread Array) and Tensor Core MMA:** In NVIDIA CUDA, a **CTA** (thread block) is a group of threads cooperating via fast on-chip shared memory to compute a tile of matrix output. **Tensor Core MMA** (Matrix Multiply-Accumulate) instructions execute hardware-accelerated dense tile multiplications (e.g., FP8 `E4M3` operands) requiring weights to be packed in a specific thread-lane permutation (`tensor_core_permutation`).
- **Data Parallelism (`DP8`):** Running 8 independent mini-batches concurrently across 8 GPUs, each holding a replica of the non-expert parameters, and averaging gradients across ranks (`dist.all_reduce(AVG)`) before each optimizer step.
- **DeepSeek Sparse Attention (DSA):** A long-context sparse attention mechanism (used across the 11 full-attention layers of GLM-5.3-Flash's 45-layer backbone) that runs a lightweight learned **indexer** (which scores all past pooled keys quadratically unless query-chunked) to select the top `2,048` most relevant past tokens for full attention, reducing attention compute at 32K to 64K context lengths.
- **Depthwise Causal Conv1D:** A 1-dimensional convolution filtering each hidden channel independently across sequence positions (`groups = hidden_size`) using only current and past tokens (never future tokens). In Engram v2, a zero-initialized width-4 causal convolution expands a 3-token n-gram lookup's receptive field to 6+ tokens for `24 KiB`/layer of decode shift-register state.
- **Distillation (Knowledge Distillation):** Training compact parameters in a compressed *student* model so its output next-token probability distribution matches that of an unmodified full-precision *teacher* model (Hinton et al. 2015).
- **DMA / NVLink / PCIe / HBM3e:** Hardware memory and interconnect layers on modern GPU nodes. **HBM3e** is ultra-high-bandwidth stacked GPU memory (`~8 TB/s` per B200); **NVLink** is a high-speed GPU-to-GPU fabric (`~1.8 TB/s` bidirectional on B200) used for `all_to_all` expert dispatch and gradient averaging; **PCIe** connects GPUs to host CPU DDR5 RAM (`~32` to `64 GB/s` per direction); **DMA** (Direct Memory Access) transfers small asynchronous payloads (like Engram's `1.9` to `3.75 KiB/token` rows) over PCIe concurrently with GPU compute.
- **Dual-Trellis and QEC (Post-Quantization Error Correction):** The three-layer compression and recovery architecture specified in this paper, pairing (1) GPU-resident per-expert four-rate trellis-coded quantized experts with (2) a parallel post-quantization error-correction pathway of mergeable non-expert adapters and deterministic Host-RAM Engram lookup tables. Distinct from quantum error correction.
- **E4M3 (`float8_e4m3fn`, FP8):** An 8-bit floating-point format with 1 sign bit, 4 exponent bits, and 3 mantissa bits (finite range up to `448.0`, no infinities, at most `254` distinct finite signed values). Native Tensor Core operand format on NVIDIA Hopper and Blackwell GPUs.
- **Effective Bits Per Weight (Effective bpw):** Total compressed storage across routed GPU payloads, block scales/coupled metadata, and auxiliary Host-RAM error-correction tables (and optionally GPU Engram projections) divided by total routed expert parameters (`304,405,807,104` on GLM-5.3-Flash). Reported alongside raw trellis payload bpw and total on-disk GPU CSF sidecar bpw for transparent footprint comparison.
- **Engram / Host-RAM Lookup Table:** A sparse conditional-memory coprocessor (DeepSeek `arXiv:2601.07372`) that hashes local token n-grams into multi-gigabyte embedding tables stored in host DDR5 RAM and injects context-gated residual corrections into selected transformer layers.
- **Expert Parallelism (`EP8`) and `All-to-All` Dispatch:** Sharding an MoE layer's experts across 8 GPUs (`288 / 8 = 36` experts per GPU). During forward and backward passes, an NVLink `all_to_all` collective dispatches each token's hidden state to the GPU owning its chosen experts and returns the weighted outputs.
- **Fisher Information and YAQA Rounding:** The Fisher information matrix measures how strongly weight perturbations shift whole-model output KL divergence. YAQA (`arXiv:2505.22988`) approximates the Fisher matrix with Kronecker input/output factors for two-sided rounding; in this independent TrellisMX implementation (`Section 4.1`), YAQA-style rounding degraded full-model KLD by `1.7%` to `5.9%` and is excluded.
- **FP16 / FP32 / FP64 / TF32 / ULP:** IEEE-754 floating-point precision formats: **FP16** (16-bit half precision, 10-bit mantissa), **FP32** (32-bit single precision, 23-bit mantissa), **FP64** (64-bit double precision, 52-bit mantissa, required for exact KLD scoring), and **TF32** (NVIDIA's 19-bit TensorFloat-32 matmul mode, explicitly disabled here via `allow_tf32 = False` for deterministic reproducibility). **ULP** (Unit in the Last Place) is the spacing between adjacent representable floats; near `1.0`, BF16's ULP is `0.0078125`, which stalls small AdamW norm updates unless norms are promoted to FP32.
- **FSDP2 (Fully Sharded Data Parallel v2):** PyTorch's parameter-sharding wrapper for distributed training. Incompatible with GLM-5.3-Flash's mixed BF16/FP32 module parameters, so this pipeline uses replicated non-expert weights with pure `DP8+EP8` `all_to_all` instead.
- **GPTQ (Generative Pre-Trained Transformer Quantization):** A second-order post-training quantization algorithm (Frantar et al.) that quantizes columns in blocks and propagates each block's rounding error onto remaining unquantized columns using the inverse input Hessian.
- **GQA (Grouped-Query Attention):** An attention variant (used in Granite 3.1 3B) where multiple query heads share fewer key/value heads to shrink KV-cache memory while preserving standard `q/k/v/o` projections.
- **Gradient Checkpointing:** Recomputing selected forward activations during backpropagation instead of retaining them in VRAM, trading modest compute for large memory savings on 16K to 64K sequences.
- **Grouped GEMM (`grouped_mm`):** A fused GPU kernel executing dozens of variable-sized matrix multiplications (one per routed expert) in a single launch. Chunked at `65,536` rows per call in `trellismx_kit/glm5_patch.py` to avoid kernel indexing limits at 32K to 64K sequence lengths.
- **Hadamard Transform / Incoherence Processing:** Multiplying weight matrices and activations by orthogonal Walsh-Hadamard matrices ($\pm 1/\sqrt{n}$ entries executed via fast $O(n \log n)$ butterfly sums) plus deterministic $\pm 1$ signs, spreading large outlier spikes uniformly across all channels before quantization (QuIP / QuIP# / QuaRot / SpinQuant).
- **Hessian Matrix (Input Second-Moment Matrix):** The $d \times d$ activation matrix $H_{\text{in}} = \frac{1}{N} \sum_t x_t x_t^\top$. Minimizing squared output error $\|W x - \widehat{W} x\|_2^2$ equals minimizing $\operatorname{tr}((W - \widehat{W}) H_{\text{in}} (W - \widehat{W})^\top)$.
- **Hyper-Connections / mHC (Manifold-Constrained Hyper-Connections):** An architectural generalization of the single transformer residual stream into multiple parallel streams (4 streams in GLM-5.3-Flash; Xie et al. 2026) mixed by constrained matrices. Requires attaching Engram hooks to the collapsed 3D `block.mlp` output rather than the 4-stream block output.
- **KDA (Kimi Delta Attention):** A gated linear-attention recurrence mechanism (`arXiv:2510.26692`) used in 34 of GLM-5.3-Flash's 45 layers (with short causal depthwise convolutions accelerated via `fla` Triton kernels), giving linear $O(T)$ sequence mixing between periodic sparse full-attention layers.
- **KL Divergence (KLD, Kullback-Leibler Divergence):** An information-theoretic measure (Kullback and Leibler 1951) of how much a student probability distribution $q$ diverges from a teacher distribution $p$: $D_{\text{KL}}(p \| q) = \sum_v p(v) \log(p(v)/q(v)) \ge 0$ (measured in nats). Zero means exact distributional identity; lower is strictly better.
- **KV Cache (Key-Value Cache):** Cached attention keys and values from prior prompt tokens so autoregressive decoding does not recompute the full prompt on every generated token.
- **Logits / Softmax / LogSumExp:** **Logits** are the unnormalized real-valued scores output by `lm_head` for every vocabulary token (`154,880` entries). **Softmax** converts logits into probabilities $p(v) = \exp(z_v) / \sum_{v'} \exp(z_{v'})$. **LogSumExp** ($\log \sum_{v'} \exp(z_{v'})$) computes the log normalizing constant stably without floating-point overflow.
- **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning (Hu et al. 2022) expressing a linear weight delta as $\Delta W = (\alpha / r) B A$ (rank `32`, $\alpha = 64$). Because matrix multiplication distributes over matrix addition, $(W + \Delta W) x = W x + \Delta W x$, so trained LoRA weights fold algebraically into base BF16 weights with zero serving overhead.
- **Mixture-of-Experts (MoE):** A sparse transformer architecture replacing dense feed-forward blocks with $E$ parallel subnetworks (*experts*), routing each token to only the top $K$ active experts via a learned linear router.
- **MLA (Multi-Head Latent Attention):** An attention design (used in 11 of GLM-5.3-Flash's 45 layers, without positional embeddings on the latent path, MLA-NoPE) that compresses keys and values into a low-rank latent vector (`kv_a_proj_with_mqa` -> `kv_b_proj`) to shrink KV-cache memory.
- **MTP (Multi-Token Prediction):** An auxiliary speculative-decoding head (`layer 45` in GLM-5.3-Flash, `7.25B` routed parameters) that predicts extra lookahead tokens beyond the primary next-token head.
- **Multiple-Choice Knapsack and Dynamic Programming:** A discrete combinatorial optimization problem where each item (routed expert) must choose exactly one option (bit rate in `{k2, p25, k3, k4}`) to minimize total distortion under an exact integer budget. Solved globally in $O(N \log N)$ via convex marginal-gain heaps (`_solve_convex_fast`) with an exact compact **dynamic programming** fallback (`solve_exact_dp`) for arbitrary non-convex curves (distinct from `DP8` Data Parallelism).
- **NFKC Canonicalization (Unicode Compatibility Composition):** A deterministic token-ID map (`build_nfkc_canon_map`) collapsing superficial casing, leading-space, and Unicode compatibility variants onto shared canonical IDs while leaving incomplete UTF-8 BPE byte fragments untouched.
- **N-Gram (Unigram / Bigram / Trigram / Skip-Gram):** Short local sequences of tokens: **unigram** is the current token $t_i$, **bigram** is the adjacent pair $(t_{i-1}, t_i)$, **trigram** is the 3-token span $(t_{i-2}, t_{i-1}, t_i)$, and **skip-gram** is the gapped pair $(t_{i-2}, t_i)$ skipping $t_{i-1}$.
- **NMSE (Normalized Mean Squared Error):** Relative tensor reconstruction error $\|W - \widehat{W}\|_F^2 / \|W\|_F^2$, used as a fast sanity check when installing decoded sidecars (`0.054` to `0.065` for `release_2375`).
- **NVFP4:** NVIDIA's 4-bit microscaled floating-point format (E2M1 weight payload at `4.0` bpw plus `UE4M3` block scales per `K = 16` weights at `0.5` bpw = `4.500` total stored bpw), used as a higher-bit reference baseline in Table 4.1.
- **OOD (Out-of-Distribution):** Evaluation panels (`ood_final_all25`, `ood_final_clean17`, `ood_confirmation_2k`) drawn from sources outside the distillation training streams to verify true generalization beyond in-domain held-out splits.
- **Perplexity Ratio:** $\exp(\text{mean}(\log p_{\text{teacher}} - \log q_{\text{student}}))$ on realized held-out next tokens; `1.000` equals zero cross-entropy loss relative to the BF16 teacher.
- **Prefill and Decode:** The two phases of LLM inference. **Prefill** processes the entire input prompt in parallel (compute-bound); **decode** generates output tokens autoregressively one step at a time (memory-bandwidth-bound).
- **Procedural NPK (`procedural-npk`, Neural Processing of K):** Brandon M. Music's zero-storage 16-bit deterministic state law (where $K$ denotes bit rate / trellis state index) mapping any 16-bit state $s$ to a provably finite FP8 E4M3 operand via `(s * 0xCBAC1FED) & 0xFFFFFFFF`, bit-masking/XORing with `0x8FFF8FFF` and `0x3B603B60`, summing the two FP16 halves, multiplying by `compander_scale = 2.0`, and projecting to E4M3 (`-0.0 -> +0.0`).
- **QAD (Quantization-Aware Distillation):** Fine-tuning quantized neural weights directly via straight-through gradient estimators against a teacher model; intractable on discrete tail-biting trellis codecs at 300B+ scale.
- **Quantization:** Compressing high-precision neural weights (e.g., 16-bit BF16) into compact discrete representations (2 to 4 bits per weight) to fit large models into GPU VRAM and accelerate memory-bandwidth-bound decoding.
- **Rademacher Sign Vector:** A deterministic vector of $+1$ and $-1$ entries multiplied elementwise into activations/weights before Hadamard rotations to break structured channel alignments.
- **Rate-Distortion Curve (`EMPIRICAL_RD_RATIOS`):** The trade-off between bit rate and distortion. In `trellismx_kit/allocate.py`, `EMPIRICAL_RD_RATIOS` sets `{k2: 4.631, p25: 2.002, k3: 1.000}` from measured Granite KLD ratios plus a conservative heuristic prior `k4: 0.500`.
- **Residual Stream:** The main hidden-state highway passing from Layer `0` through the final layer of a transformer, onto which every attention block, MoE block, and Engram QEC module adds incremental vector updates.
- **RMSNorm (Root Mean Square Layer Normalization):** A normalization layer dividing a hidden vector by its root-mean-square magnitude and multiplying by a learned per-channel scale vector. Promoted to FP32 during distillation to avoid BF16 ULP stagnation.
- **Routing Mass (`route_mass`):** The sum of squared router weights $\sum w^2$ routed to an expert across calibration tokens (clamped at a 5th-percentile floor per layer in upstream preprocessing), measuring an expert's residual-stream variance impact.
- **Safetensors and SHA-256:** **Safetensors** is a memory-mappable, zero-code-execution binary tensor container format storing tensors and string metadata; **SHA-256** is a 256-bit cryptographic digest used fail-closed across every TrellisMX sidecar tensor, expanded CSF scale plane, and canonical rate-assignment JSON.
- **ShapleyMCG:** Brandon M. Music's exact-budget MoE bit-allocation framework combining Aumann-Shapley attribution shares with exact integer knapsack optimization (`Section 4.8`).
- **Shared Memory (SMEM):** Ultra-fast on-chip SRAM local to each GPU streaming multiprocessor. The fused CUDA pair kernel (`sqg_quantize.cu`, targeting NVIDIA B200 `SM100`) packs the 64 KiB `procedural-npk` LUT plus double-buffered `2,048`-state FP32 Viterbi costs inside exactly `83,200` bytes of SMEM, eliminating global `temp_costs` spills.
- **SiLU (Sigmoid Linear Unit):** The smooth nonlinear activation $\text{SiLU}(x) = x \cdot \sigma(x)$, used both inside expert SwiGLU gates and after Engram v2's width-4 causal depthwise convolution.
- **SwiGLU (`silu-cap10`):** A gated feed-forward nonlinearity $\text{SiLU}(\text{gate}) \odot \text{up}$ used inside MoE experts (clamped to `gate <= 10` and `up in [-10, 10]` in GLM-5.3-Flash).
- **Tail-Biting Viterbi Search:** Dynamic-programming optimization (Viterbi 1967) over a cyclic ring-wrapped 16-bit trellis state machine (`128` scalar / `64` pair wrap context) finding the minimum-distortion symbol path for a 256-weight tile without boundary flush overhead.
- **Tail-Lumped Top-K KL Lower Bound (`topk_kl_loss`):** A chunked, gradient-checkpointed distillation loss combining exact KL over the teacher's top `128` tokens (`kl_top`) with a lumped tail KL term (`kl_tail`) evaluated in log-space via masked `logsumexp`. By the data-processing inequality, `kl_top + kl_tail <= KL_full`, preserving exact head-vs-tail mass calibration without 64K VRAM OOM or FP32 cancellation.
- **Tensor Parallelism (`TP4`):** Splitting each MoE expert's intermediate dimension across 4 GPUs (`2,048 -> 512` local width per rank) so all 4 GPUs execute every routed expert in parallel and sum partial outputs cleanly; matches the `TP4` four-bank rank sidecar layout.
- **Top-1 Agreement:** The percentage of evaluation positions where the student's highest-probability next token (`argmax`) matches the unmodified BF16 teacher's `argmax`.
- **Trellis-Coded Quantization (TCQ):** A vector-quantization scheme (Marcellin and Fischer 1990) in which consecutive weights traverse an overlapping finite state machine. Even though the E4M3 output alphabet has at most `254` signed finite values (`~161` distinct non-negative magnitudes after `procedural-npk`), optimizing sequential state-path constraints across overlapping 16-bit windows approaches Shannon's rate-distortion limit at `2.0` to `4.0` bpw.
- **TrellisMX:** Brandon M. Music's sub-4-bit coupled Hadamard + `procedural-npk` E4M3 trellis codec with OCP `UE8M0` `K = 32` microscaling and lossless CSF scale planes, descending from QTIP, EXL3, and `KQuant`/`QSRT`.
- **Triton Kernel:** A custom GPU kernel written in the Triton GPU programming language (notably Flash Linear Attention `fla`'s `causal_conv1d` kernel used here to accelerate GLM-5.3-Flash's 34 KDA linear-attention layers).
- **UE8M0 Microscaling:** An 8-bit unsigned pure-exponent block scale format under OCP MX v1.0 (0 sign bits, 8 exponent bits, 0 mantissa bits; codes `0..254` represent exact powers of two $2^{\text{code} - 127}$, while code `255` is reserved for `NaN`), shared across every `K = 32` contiguous weights.
- **Wilson Score Interval:** A binomial 95% confidence interval (Wilson 1927) used to bound Top-1 agreement percentages accurately even near `100%` where symmetric normal intervals break down.
