# Empirical Results: Sub-3-Bit Per-Expert TrellisMX and Full-Stack QEC

All measurements below were run on `ibm-granite/granite-3.1-3b-a800m-instruct`
(32 routed MoE layers, 40 experts per layer, top-8 routing, 3.02B expert
weights) using identical evaluation conditions: 21 untouched evaluation windows
of 8,192 tokens across 11 domains (172,011 positions, exact FP64
`KL(BF16 teacher || student)`, single-window deterministic forward passes,
zero evaluation leakage with strict 13-gram screening `<= 1%` overlap).

Lower KLD and higher Top-1 agreement are better.

## 1. Per-expert TrellisMX baseline frontier (no QEC)

Standard local GPTQ rounding, weights-only path (from
`trellismx-sub3bit-per-expert-results`):

| Stored bits | Configuration | Mean KLD | Median | p90 | Top-1 |
|---:|---|---:|---:|---:|---:|
| 2.000 | Uniform K2 | 0.2348 | 0.0728 | 0.5954 | 82.28% |
| 2.125 | Per-expert `{k2, p25, k3}`, attribution | 0.1359 | 0.0400 | 0.3320 | 86.56% |
| 2.250 | Per-expert `{k2, p25, k3}`, attribution | 0.1106 | 0.0313 | 0.2673 | 87.83% |
| 2.250 | Per-layer `{k2, p25, k3}`, attribution | 0.1390 | 0.0408 | 0.3400 | 86.41% |
| 2.375 | Per-expert `{k2, p25, k3}`, attribution | 0.0934 | 0.0260 | 0.2180 | 88.89% |
| 2.500 | Uniform P25 (pair rate) | 0.1015 | 0.0278 | 0.2390 | 88.32% |
| 2.500 | Per-layer `{k2, p25, k3}`, attribution | 0.0975 | 0.0278 | 0.2340 | 88.50% |
| 2.500 | Per-expert `{k2, p25, k3}`, one rate per expert, route mass | 0.0819 | 0.0209 | 0.1870 | 89.65% |
| 2.500 | Per-expert `{k2, p25, k3}`, one rate per expert, attribution | 0.0799 | 0.0222 | 0.1843 | 89.68% |
| 2.625 | Per-expert `{k2, p25, k3}`, attribution | 0.0715 | 0.0195 | 0.1630 | 90.35% |
| 2.750 | Per-expert `{k2, p25, k3}`, attribution | 0.0632 | 0.0175 | 0.1440 | 90.80% |
| 3.000 | Uniform K3 | 0.0507 | 0.0132 | 0.1163 | 91.71% |

## 2. Full-Stack QEC on Uniform 2.0-bit (50M-token scaling curve)

Frozen Uniform K2 experts (strictly 2.00 stored bits per expert weight in VRAM,
zero expert offloading) plus mergeable BF16 non-expert QEC (rank-32 Attention
LoRA on `q/k/v/o`, RMSNorm scales, 1,280 per-expert scalar gains) plus a
12-layer Host-RAM Engram QEC sidecar (layers 0, 1, 2, 4, 6, 8, 12, 16, 20, 24,
28, 31; 262,139 prime buckets per bigram/trigram head; context-aware gating).
Trained across 4x RTX PRO 6000 (DDP, ~35,000 tokens/s, 24.2 minutes total for
50M fresh 13-gram-screened tokens):

| Distillation tokens | Mean KLD | vs K2 baseline | Median | p90 | Top-1 |
|---:|---:|---:|---:|---:|---:|
| 0.0M (baseline) | 0.2348 | 0.0% | 0.0728 | 0.5954 | 82.28% |
| 1.05M | 0.1843 | -21.5% | 0.0602 | 0.4581 | 84.41% |
| 5.05M | 0.1527 | -34.9% | 0.0483 | 0.3808 | 85.82% |
| 10.03M | 0.1454 | -38.1% | 0.0475 | 0.3617 | 86.04% |
| 20.05M | 0.1390 | -40.8% | 0.0438 | 0.3494 | 86.45% |
| 35.06M | 0.1330 | -43.4% | 0.0416 | 0.3341 | 86.77% |
| 49.94M | 0.1312 | -44.1% | 0.0409 | 0.3309 | 86.93% |

At 50M tokens, Uniform 2.00-bit + Full-Stack QEC reaches 0.1312 KLD (+4.65
percentage points Top-1), equivalent to 2.347 uniform bits per weight (+0.347
free bits per weight in VRAM) and lower KLD than both Per-Layer 2.25-bit
(0.1390) and Per-Expert 2.125-bit (0.1359). The knee of the curve sits between
35M and 50M tokens (-1.4% incremental gain across the last 15M tokens).

## 3. Host-RAM Engram QEC alone (10M-token scaling across bit rates)

8-layer Engram QEC (layers 0, 1, 2, 4, 8, 16, 24, 31; 262,139 prime buckets;
frozen experts, frozen attention, frozen routers):

| Base model (stored bpw) | 0M | 1M | 3M | 5M | 10M | Delta @ 10M | Top-1 @ 10M |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform K2 (2.00 bpw) | 0.2348 | 0.1753 | 0.1636 | 0.1603 | 0.1544 | -34.2% | 85.77% |
| Per-Expert 2.25 bpw | 0.1106 | 0.1019 | 0.0991 | 0.0978 | 0.0951 | -14.0% | 88.86% |
| Per-Expert 2.50 bpw | 0.0799 | 0.0757 | 0.0751 | 0.0743 | 0.0720 | -9.9% | 90.29% |

Equivalent uniform bit gains from Engram QEC alone at 10M tokens: +0.250 bpw
on Uniform K2 (2.250 uniform-bit equivalent), +0.113 per-expert bpw on PE2.25
(beats Uniform 2.50-bit at 0.1015), and +0.117 per-expert bpw on PE2.50
(matches Per-Expert 2.625-bit at 0.0715).

## 4. Component ablation on Per-Expert 2.25-bit (5M tokens)

| Configuration (2.25 stored bpw) | Mean KLD | vs baseline | Median | p90 | Top-1 |
|---|---:|---:|---:|---:|---:|
| Baseline Per-Expert 2.25 bpw | 0.1106 | 0.0% | 0.0313 | 0.2673 | 87.83% |
| + 8-layer Unigram-only Engram (107M host-RAM params) | 0.0978 | -11.6% | 0.0287 | 0.2357 | 88.66% |
| + 8-layer Full Engram uni+bi+tri (868M host-RAM params) | 0.0978 | -11.5% | 0.0290 | 0.2357 | 88.74% |
| + Rank-16 Attention LoRA only (5.24M mergeable params) | 0.0958 | -13.4% | 0.0285 | 0.2343 | 88.76% |
| + 8-layer Full Engram + Rank-16 Attention LoRA | 0.0945 | -14.5% | 0.0276 | 0.2329 | 88.90% |

## 4b. Dual-Trellis QEC v2 ablation suite (10M fresh tokens, exact 21-window eval)

Controlled 10M-token comparison testing FP32-promoted RMSNorms, zero-VRAM
continuous codebook recalibration (`intermediate_scales_fp16` `[L, E, 3, I]`),
per-expert output centroid bias (`3.75 MiB`), and Dual-Trellis Engram v2
(zero-init residual bi/tri/skip tables, NFKC token canonicalization, per-order
8-group gating, width-4 causal depthwise Conv1D, 3 shared cross-layer depth
banks cutting host-RAM parameters 3x from 1.28B to 421M, zero embedding weight
decay, and log-space tail-calibrated top-128 KL):

| Arm (10M fresh tokens) | Base (stored bpw) | Host-RAM params | KLD @ 1M | KLD @ 5M | KLD @ 10M | vs baseline | Top-1 @ 10M |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 Full-Stack QEC (reference) | Uniform K2 (2.00) | 1,283M | 0.1843 | 0.1527 | 0.1454 | -38.1% | 86.04% |
| Dual-Trellis Engram v2 only + FP32 norms | Uniform K2 (2.00) | 421M | 0.1629 | 0.1493 | 0.1436 | -38.8% | 86.16% |
| `intermediate_scales_fp16` + bias + FP32 norms | Uniform K2 (2.00) | 1,283M | 0.1642 | 0.1464 | 0.1411 | -39.9% | 86.29% |
| **Combined Dual-Trellis QEC v2 stack** | **Uniform K2 (2.00)** | **421M** | **0.1613** | **0.1438** | **0.1384** | **-41.1%** | **86.41%** |
| **Combined Dual-Trellis QEC v2 stack** | **Per-Expert 2.25** | **421M** | **0.0982** | **0.0937** | **0.0891** | **-19.3%** | **89.06%** |

At 10M tokens, the combined Dual-Trellis QEC v2 stack on Uniform K2 (`0.1384`,
median `0.0455`, p90 `0.3400`) beats v1's 20M-token checkpoint (`0.1390`) in
half the tokens while using 3x less host RAM and 4x fewer host-RAM lookups. On
Per-Expert 2.25-bit, the exact same 10M-token stack drops KLD from `0.1104` to
`0.0891` (-19.3%, `89.06%` top-1), beating Uniform 2.50-bit (`0.1015`),
Per-Layer 2.50-bit (`0.0975`), and Per-Expert 2.375-bit (`0.0934`). Adding a
dense rank-64 shared-basis Wiener branch at `lr=8e-4` overfit held-out eval,
and even with `lr=1.5e-4, wd=1e-2` plus hidden-state MSE alignment trailed the
cleaner combined stack at 5M (`0.1453` vs `0.1438`), confirming the lean
`intermediate_scales_fp16` + per-expert bias + Dual-Trellis Engram v2 +
mergeable Attention LoRA + FP32 norms stack is the Pareto-optimal recipe.

## 5. Negative controls: expert deletion vs n-gram replacement

Replacing deleted experts with n-gram lookup tables outperforms pure deletion
by 3.2x to 4.7x, but keeping cold experts at 2 bits in VRAM still outperforms
deleting 20% of experts at matched total VRAM:

| Configuration | Effective bpw | Mean KLD | Median | Top-1 |
|---|---:|---:|---:|---:|
| BF16 + Layer 0 deleted (zeroed, 3.1% experts) | 15.50 | 10.8053 | 9.7889 | 2.41% |
| BF16 + Layer 0 zero-training bigram LUT | 15.50 | 0.7007 | 0.1372 | 74.68% |
| BF16 + coldest 20% experts deleted (zeroed) | 12.80 | 0.6085 | 0.1432 | 74.88% |
| BF16 + coldest 20% experts zero-training bigram LUT | 12.80 | 0.1302 | 0.0152 | 89.37% |
| PE2.50 + coldest 20% experts deleted (zeroed) | 2.06 | 0.7275 | 0.2105 | 71.53% |
| PE2.50 + coldest 20% experts zero-training bigram LUT | 2.06 | 0.1970 | 0.0406 | 85.44% |
| K3 + coldest 20% experts deleted (zeroed) | 2.40 | 0.6633 | 0.1755 | 73.06% |
| K3 + coldest 20% deleted + 6L distilled Engram (450K tok) | 2.40 | 0.2060 | 0.0496 | 84.97% |
| K3 + Layers 0-5 deleted + 6L distilled Engram (450K tok) | 2.44 | 2.0389 | 1.2711 | 47.64% |

Training-free n-gram predictability probe (`engram_probe.py`, held-out across
9 of 44 fit windows): Layer 0 mean bigram R-squared = 0.937 (NMSE 0.0541, best
expert 0.987 / NMSE 0.0093), dropping steeply to Layer 1 = 0.687, Layer 2 =
0.536, Layer 4 = 0.330, Layer 8 = 0.178, Layer 16 = 0.074, Layer 24 = 0.162,
and Layer 31 = 0.246. This explains why residual QEC (keeping cold experts at
2 bits and correcting systematic drift via host-RAM Engram + mergeable
Attention LoRA) dominates expert deletion at equal VRAM.

## 6. SM120 b12x per-expert grouped kernel acceptance (`patches/b12x-per-expert.patch`)

Verified on an RTX PRO 6000 Blackwell (SM120) against synthetic TP4 MoE layers
(288 experts, hidden 4096, rank-local intermediate 512, top-8 routing):

| Check | Scope | Result |
|---|---|---|
| Per-expert stream decode probe | All 288 experts across `{k2, p25, k3}` | Bit-exact E4M3 match |
| Uniform parity (K2, K3, K4) | Per-expert runtime vs uniform program | Bit-for-bit identical (`torch.equal`) |
| 3-rate mixed layer (`k2,p25,k3`) | Rows 1, 8, 16, 64, 512 across all 3 banks | Cosine 0.988 to 0.998 vs FP32 reference; bit-exact to uniform K2 |
| 4-rate mixed layer (`k2,p25,k3,k4`) | Rows 1, 16, 64 across all 4 banks | Cosine 0.988 to 0.995 vs FP32 reference |
| Bank permutation invariance | Roll-96 expert permutation | Bit-for-bit identical (`torch.equal`) |
| Small-M guard | `small_m_scheduler=True` on per-expert source | Fails closed as specified |
| `compute-sanitizer --tool memcheck` | 3-rate and 4-rate mixed layers across all banks | 0 errors |
| Mixed `{k2,p25,k3}` vs Uniform K3 timing (5 replays) | Rows 1 / 8 / 16 / 64 / 512 | 2.15 / 1.94 / 1.94 / 1.93 / 2.90 ms vs 1.96 / 1.91 / 2.04 / 1.96 / 2.62 ms |
