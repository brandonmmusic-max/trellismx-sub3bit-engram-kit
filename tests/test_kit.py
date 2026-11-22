"""CPU unit and invariant test suite for trellismx_kit."""

from __future__ import annotations

import itertools
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from trellismx_kit.allocate import (
    EMPIRICAL_RD_RATIOS,
    RATE_FRACTION,
    build_hierarchical_assignment,
    solve_exact_dp,
)
from trellismx_kit.engram_probe import score_expert_ngram_predictability
from trellismx_kit.engram_qec import (
    AttentionLoRA,
    DualTrellisEngramLayer,
    EngramQECLayer,
    FullStackQEC,
    SharedHostNGramBank,
    build_context_curriculum,
    build_nfkc_canon_map,
    topk_kl_loss,
)
from trellismx_kit.kld_score import summarize_windows, tokenwise_kld_fp64
from trellismx_kit.layer_sensitivity import measure_layer_sensitivity
from trellismx_kit.pe_format import (
    decode_csf_plane,
    decode_expert_operands_from_layout,
    device_layout,
    encode_csf_plane,
    read_rank_sidecar,
    synthetic_rank_sidecar,
)
from trellismx_kit.profile import ModelProfile, load_profile
from trellismx_kit.stratified_capture import WindowRecord, select_stratified_windows
from trellismx_kit.trellis_ref import (
    RATES,
    STREAM_BITS,
    SYMBOLS_PER_TILE,
    TILE_I16,
    decode_tile_operands,
    decode_tile_values,
    pack_symbols,
    state_lut,
    unpack_symbols,
)
from trellismx_kit.vram_budget import estimate_memory

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_profile() -> ModelProfile:
    return ModelProfile(
        name="synthetic-tiny-moe",
        hidden_size=64,
        intermediate_size=64,
        num_layers=4,
        routed_layers=(0, 1, 2, 3),
        num_experts=8,
        top_k=2,
        vocab_size=256,
        tp_size=2,
        rates=("k2", "p25", "k3", "k4"),
        non_routed_params=100_000,
    )


def test_shipped_profiles_validate() -> None:
    for cfg in sorted((REPO_ROOT / "configs").glob("*.json")):
        prof = load_profile(cfg)
        prof.validate()
        assert prof.total_routed_weights > 0
    glm = load_profile(REPO_ROOT / "configs" / "glm-5.3-flash.json")
    assert glm.layer_path_template == "model.language_model.layers.{layer}"
    assert "q_a_proj" in glm.projections_for_layer(3)
    assert "q_proj" in glm.projections_for_layer(4)


@pytest.mark.parametrize("rate", RATES)
def test_trellis_pack_unpack_and_decode_all_four_rates(rate: str) -> None:
    gen = torch.Generator().manual_seed(42)
    bits = STREAM_BITS[rate]
    n_sym = SYMBOLS_PER_TILE[rate]
    symbols = torch.randint(0, 1 << bits, (5, n_sym), generator=gen, dtype=torch.int16)
    packed = pack_symbols(symbols, rate)
    assert packed.dtype == torch.int16
    assert packed.shape == (5, TILE_I16[rate])
    unpacked = unpack_symbols(packed, rate)
    assert torch.equal(unpacked, symbols)

    operands = decode_tile_operands(packed, rate)
    values = decode_tile_values(packed, rate)
    assert operands.dtype == torch.uint8 and operands.shape == (5, 256)
    assert values.dtype == torch.float32 and values.shape == (5, 256)
    assert bool(torch.isfinite(values).all())
    assert state_lut().shape == (65536,)


def test_csf_lossless_round_trip_optimal_base_and_odd_cols() -> None:
    gen = torch.Generator().manual_seed(99)
    # Odd cols (13, not divisible by 8) exercises bitplane padding branch
    plane = torch.randint(120, 122, (8, 13), generator=gen, dtype=torch.uint8)
    plane[0, :5] = torch.tensor([0, 1, 127, 254, 255], dtype=torch.uint8)
    csf = encode_csf_plane(plane)
    recovered = decode_csf_plane(csf, shape=plane.shape)
    assert torch.equal(recovered, plane)

    # Bimodal 49%/51% smooth row must choose base=120 -> 0 exceptions!
    bimodal = torch.cat((torch.full((4, 49), 120, dtype=torch.uint8), torch.full((4, 51), 121, dtype=torch.uint8)), dim=1)
    csf_bi = encode_csf_plane(bimodal)
    assert csf_bi.exceptions.numel() == 0
    assert torch.equal(decode_csf_plane(csf_bi, shape=bimodal.shape), bimodal)


@pytest.mark.parametrize("scale_storage", ["plain", "csf"])
def test_four_bank_sidecar_round_trip_and_device_layout(tmp_path: Path, scale_storage: str) -> None:
    prof = _tiny_profile()
    path = synthetic_rank_sidecar(
        tmp_path / f"layer-000-{scale_storage}.safetensors",
        prof,
        layer=0,
        rank=0,
        scale_storage=scale_storage,
    )
    sc = read_rank_sidecar(path, prof)
    assert sc.counts == {"k2": 2, "p25": 2, "k3": 2, "k4": 2}
    layout = device_layout(sc, prof)
    assert layout.expert_rate.shape == (prof.num_experts,)
    for e in range(prof.num_experts):
        g, u, d = decode_expert_operands_from_layout(layout, prof, e)
        n_tiles = prof.hidden_tiles * prof.local_intermediate_tiles
        assert g.shape == (n_tiles, 256)
        assert u.shape == (n_tiles, 256)
        assert d.shape == (n_tiles, 256)


def test_sidecar_rejects_tampered_tensor(tmp_path: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    prof = _tiny_profile()
    path = synthetic_rank_sidecar(tmp_path / "good.safetensors", prof, scale_storage="csf")
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})
        tensors = {k: fh.get_tensor(k) for k in fh.keys()}
    tensors["w13_trellis_k4"][0, 0, 0, 0, 0] ^= 1
    bad_path = tmp_path / "tampered.safetensors"
    save_file(tensors, str(bad_path), metadata=meta)
    with pytest.raises(ValueError, match="SHA-256"):
        read_rank_sidecar(bad_path, prof)


def test_allocator_exact_optimality_fast_1t_scale_and_engram_coalloc() -> None:
    # 1) Brute-force exactness check on small 5-expert problem across all 4 rates
    rng = np.random.default_rng(11)
    cands = []
    for i in range(5):
        imp = float(rng.uniform(0.1, 3.0))
        cands.append((0, i, [imp * EMPIRICAL_RD_RATIOS[r] for r in RATES]))
    for budget_num in (22, 25, 27, 30, 33):
        budget_bits = Fraction(budget_num, 10)
        got_choices, got_dmg = solve_exact_dp(cands, rates=RATES, budget_bits=budget_bits)
        best_dmg = float("inf")
        max_hb = int((budget_bits * 2 * len(cands)) // 1)
        for combo in itertools.product(range(4), repeat=5):
            hb = sum(int(RATE_FRACTION[RATES[r]] * 2) for r in combo)
            if hb <= max_hb:
                dmg = sum(cands[i][2][combo[i]] for i in range(5))
                if dmg < best_dmg:
                    best_dmg = dmg
        assert abs(got_dmg - best_dmg) < 1e-9

    # 2) Fast O(N log N) scaling check on 1T MoE (23,424 experts) must finish well under 1s
    prof_1t = load_profile(REPO_ROOT / "configs" / "template-1t-moe.json")
    masses_1t = {l: rng.exponential(scale=1.0, size=prof_1t.num_experts).tolist() for l in prof_1t.routed_layers}
    sens_1t = {l: 1.0 + 0.01 * l for l in prof_1t.routed_layers}
    t0 = time.time()
    doc_1t = build_hierarchical_assignment(
        prof_1t,
        route_mass=masses_1t,
        layer_sensitivity=sens_1t,
        budget_bits=2.25,
        max_k4_fraction=0.05,
    )
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"1T MoE allocator took {elapsed:.2f}s"
    assert doc_1t["achieved_bits"] <= 2.25 + 1e-9
    assert doc_1t["counts"]["k4"] <= int(0.05 * prof_1t.total_routed_experts)

    # 3) Asymmetric Engram co-allocation shifts bits away from high-R^2 Layer 0 toward deep Layer 2
    prof = _tiny_profile()
    uniform_mass = {l: [1.0] * prof.num_experts for l in prof.routed_layers}
    equal_sens = {l: 1.0 for l in prof.routed_layers}
    r2_map = {0: 0.95, 1: 0.50, 2: 0.05, 3: 0.10}
    doc_co = build_hierarchical_assignment(
        prof,
        route_mass=uniform_mass,
        layer_sensitivity=equal_sens,
        engram_r2=r2_map,
        engram_discount=0.8,
        budget_bits=2.5,
        max_k4_fraction=0.25,
    )
    l0_bits = sum(float(RATE_FRACTION[r]) for r in doc_co["layers"]["0"])
    l2_bits = sum(float(RATE_FRACTION[r]) for r in doc_co["layers"]["2"])
    assert l2_bits > l0_bits


def test_stratified_capture_rare_floor_dead_expert_and_domain_caps() -> None:
    rng = np.random.default_rng(7)
    records = []
    for i in range(20):
        counts = rng.integers(100, 300, size=(2, 4), dtype=np.int64)
        counts[0, 0] = 0  # unreachable dead expert must not break fallback balancing
        if i in (3, 11, 17):
            counts[1, 3] = 900
        else:
            counts[1, 3] = 5
        records.append(WindowRecord(window_id=f"w{i}", domain="code" if i % 2 else "chat", expert_counts=counts))
    sel = select_stratified_windows(records, target_windows=5, token_floor=1500, domain_caps={"code": 3, "chat": 3})
    assert len(sel.selected_ids) == 5
    assert sel.domain_counts.get("code", 0) <= 3
    assert sum(w in sel.selected_ids for w in ("w3", "w11", "w17")) >= 2
    assert int(sel.total_expert_counts[1, 3]) >= 1500


class _TinyBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d, bias=False)
        self.self_attn.k_proj = nn.Linear(d, d, bias=False)
        self.self_attn.v_proj = nn.Linear(d, d, bias=False)
        self.self_attn.o_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.self_attn.o_proj(self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x))
        return self.norm(x + 0.1 * y + 0.1 * self.mlp(x))


class _TinyModel(nn.Module):
    def __init__(self, d: int, v: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(v, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyBlock(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, v, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        h = self.embed(input_ids)
        for blk in self.model.layers:
            h = blk(h)
        return self.head(h)


def test_fullstack_qec_bf16_norms_single_token_decode_host_tables_and_merge() -> None:
    torch.manual_seed(0)
    prof = _tiny_profile()
    model = _TinyModel(prof.hidden_size, prof.vocab_size, prof.num_layers).to(torch.bfloat16)
    ids = torch.randint(0, prof.vocab_size, (2, 16))
    with torch.no_grad():
        base_out = model(ids).clone()

    qec = FullStackQEC(
        model,
        prof,
        engram_layers=(0, 3),
        lora_rank=4,
        engram_kwargs={
            "d_uni": 16,
            "d_bi": 8,
            "n_bi_heads": 2,
            "bi_buckets": 257,
            "d_tri": 8,
            "n_tri_heads": 1,
            "tri_buckets": 257,
            "d_skip": 8,
            "n_skip_heads": 1,
            "skip_buckets": 257,
            "table_device": "cpu",
        },
    )
    # Promoted norms must be FP32 even though student backbone is BF16!
    assert model.model.layers[0].norm.weight.dtype == torch.float32
    assert model.model.layers[0].norm.weight.requires_grad

    # Single-token (T == 1) and T == 2 lookup must not crash and must respect prev_tokens
    layer0_eng = qec.engrams["0"]
    one_tok = layer0_eng.lookup(ids[:, :1])
    assert one_tok.shape == (2, 1, 16 + 16 + 8 + 8)
    cont = layer0_eng.lookup(ids[:, 2:3], prev_tokens=ids[:, 0:2])
    full = layer0_eng.lookup(ids[:, :3])
    assert torch.allclose(cont[:, 0], full[:, 2], atol=1e-6)

    # Zero-init matches baseline at step 0 via standard forward()
    qec.eval()
    with torch.no_grad():
        zero_out = qec(ids)
    assert torch.allclose(zero_out.float(), base_out.float(), atol=1e-2)

    # Two optimization steps verify gradients reach w_k, gate_bias, lora.a, uni_emb, AND FP32 norms
    qec.train()
    assert not qec.model.training  # frozen student stays in eval mode!
    opt = torch.optim.AdamW(qec.parameter_groups())
    norm_before = model.model.layers[0].norm.weight.detach().clone()
    target_logits = base_out.detach().float() + 0.5 * torch.randn_like(base_out.float())
    t_lse = torch.logsumexp(target_logits, dim=-1)
    top_vals, top_idx = torch.topk(target_logits, k=16, dim=-1)
    top_logp = (top_vals - t_lse.unsqueeze(-1)).clamp_max(0.0)
    p_tail = (1.0 - top_logp.exp().sum(dim=-1)).clamp(0.0, 1.0)
    for _ in range(2):
        opt.zero_grad()
        logits = qec(ids)
        loss = topk_kl_loss(logits, top_logp, top_idx, teacher_p_tail=p_tail, chunk_tokens=8)
        loss.backward()
        opt.step()
    assert not torch.equal(model.model.layers[0].norm.weight, norm_before)
    assert float(qec.loras["0_attn_q_proj"].a.weight.grad.abs().sum()) > 0.0
    assert float(qec.engrams["0"].w_k.weight.grad.abs().sum()) > 0.0

    # Algebraic LoRA merge preserves outputs and detaches all LoRA hooks
    qec.eval()
    with torch.no_grad():
        pre_merge = qec(ids).clone()
    merged = qec.merge_lora_into_model()
    assert merged == 4 * prof.num_layers
    assert len(qec._lora_handles) == 0
    with torch.no_grad():
        post_merge = qec(ids)
    assert torch.allclose(pre_merge.float(), post_merge.float(), atol=5e-2)
    qec.remove_hooks()


def test_layer_sensitivity_3d_and_2d_logits() -> None:
    torch.manual_seed(3)
    prof = _tiny_profile()
    model = _TinyModel(prof.hidden_size, prof.vocab_size, prof.num_layers)
    wins = [torch.randint(0, prof.vocab_size, (1, 16)) for _ in range(2)]
    sens_3d = measure_layer_sensitivity(model, prof, wins, lambda m, x: m(x), noise_rel_std=0.1)
    assert set(sens_3d.keys()) == set(prof.routed_layers)
    assert all(v > 1e-7 for v in sens_3d.values())
    sens_2d = measure_layer_sensitivity(model, prof, wins, lambda m, x: m(x)[0], noise_rel_std=0.1)
    for l in prof.routed_layers:
        assert abs(sens_3d[l] - sens_2d[l]) < 1e-6


def test_engram_probe_large_vocab_int32_kld_inf_mask_and_vram_budget() -> None:
    rng = np.random.default_rng(5)
    n = 1000
    # int32 IDs above 46,340 previously overflowed in NumPy 2.x!
    toks = (rng.integers(0, 20, size=n, dtype=np.int32) + np.int32(150_000))
    prev = np.concatenate([np.array([149_999], dtype=np.int32), toks[:-1]])
    table = torch.randn(20, 32)
    y = table[torch.from_numpy((toks - 150_000).astype(np.int64))] + 0.05 * torch.randn(n, 32)
    is_build_int = (np.arange(n) < 800).astype(np.int64)  # integer 0/1 mask
    res = score_expert_ngram_predictability(y, toks, prev, is_build_int)
    assert res.unigram_r2 > 0.95 and res.unigram_nmse < 0.05

    # Masked -inf vocab slots must not produce NaN in tokenwise_kld_fp64
    t_logits = torch.randn(32, 50)
    s_logits = t_logits + 0.1 * torch.randn(32, 50)
    t_logits[:, -2:] = float("-inf")
    s_logits[:, -2:] = float("-inf")
    kld_vals, agree_vals = tokenwise_kld_fp64(t_logits, s_logits)
    assert bool(torch.isfinite(kld_vals).all()) and float(kld_vals.mean()) > 0.0
    summ = summarize_windows([kld_vals[:10], kld_vals[10:]], [agree_vals[:10], agree_vals[10:]], resamples=200)
    assert summ.positions == 32 and summ.windows == 2

    # VRAM budget explicitly reflects expanded 0.25 bpw UE8M0 vs 0.0625 bpw in-VRAM CSF
    prof_1t = load_profile(REPO_ROOT / "configs" / "template-1t-moe.json")
    est_plain = estimate_memory(prof_1t, avg_expert_bits=2.25, in_vram_csf=False, gpu_count=4)
    est_csf = estimate_memory(prof_1t, avg_expert_bits=2.25, in_vram_csf=True, gpu_count=4)
    assert est_plain.in_vram_scales_and_vectors_gib > est_csf.in_vram_scales_and_vectors_gib + 15.0
    assert est_plain.offloaded_embed_host_ram_gib > 0.0


def test_80_20_curriculum_and_tail_calibrated_kl_and_frozen_indexer() -> None:
    sched = build_context_curriculum(
        total_tokens=50_000_000,
        short_tokens_per_step=65_536,
        long_tokens_per_step=131_072,
        long_fraction=0.20,
        mode="interleaved",
    )
    short_tok = sched.count("short") * 65_536
    long_tok = sched.count("long") * 131_072
    ratio = long_tok / (short_tok + long_tok)
    assert 0.195 <= ratio <= 0.205

    # Exact full-vocab equivalence when K == V AND finite backward gradients (no 0 * -inf NaN!)
    torch.manual_seed(7)
    t_logits = torch.randn(2, 8, 64)
    s_logits = (t_logits + 0.2 * torch.randn(2, 8, 64)).requires_grad_(True)
    t_lse = torch.logsumexp(t_logits, dim=-1)
    top_vals_all, top_idx_all = torch.topk(t_logits, k=64, dim=-1)
    kl_exact = torch.nn.functional.kl_div(
        torch.log_softmax(s_logits.detach(), dim=-1),
        torch.log_softmax(t_logits, dim=-1),
        log_target=True,
        reduction="none",
    ).sum(dim=-1).mean()
    kl_sparse_all = topk_kl_loss(
        s_logits,
        top_vals_all - t_lse.unsqueeze(-1),
        top_idx_all,
        teacher_logsumexp=t_lse,
        teacher_top_logits=top_vals_all,
        chunk_tokens=4,
    )
    assert torch.allclose(kl_exact, kl_sparse_all, atol=1e-5)
    kl_sparse_all.backward()
    assert bool(torch.isfinite(s_logits.grad).all())

    # Shifted top_logp + p_tail with teacher_logsumexp must NOT double-subtract!
    s2 = t_logits.clone().requires_grad_(True)
    top_vals_16, top_idx_16 = torch.topk(t_logits, k=16, dim=-1)
    top_logp_16 = (top_vals_16 - t_lse.unsqueeze(-1)).clamp_max(0.0)
    p_tail_16 = (1.0 - top_logp_16.exp().sum(dim=-1)).clamp(0.0, 1.0)
    kl_self = topk_kl_loss(s2, top_logp_16, top_idx_16, teacher_logsumexp=t_lse, teacher_p_tail=p_tail_16)
    assert float(kl_self.detach()) < 1e-6
    kl_self.backward()
    assert bool(torch.isfinite(s2.grad).all())

    # Verify sparse attention indexer norms remain frozen when train_norms=True
    prof = _tiny_profile()
    model = _TinyModel(prof.hidden_size, prof.vocab_size, prof.num_layers)
    model.model.layers[0].self_attn.indexer_norm = nn.LayerNorm(prof.hidden_size)
    qec = FullStackQEC(model, prof, engram_layers=(0,), lora_rank=4, train_norms=True)
    assert not model.model.layers[0].self_attn.indexer_norm.weight.requires_grad
    assert model.model.layers[0].norm.weight.requires_grad
    qec.remove_hooks()


class _Tiny3DExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int, inter: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.swiglu_limit = 10.0
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.05)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.05)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(hidden_states)
        for e in range(self.num_experts):
            pos, tok = torch.where(top_k_index.t() == e)
            if tok.numel() == 0:
                continue
            gu = torch.nn.functional.linear(hidden_states[tok], self.gate_up_proj[e])
            g, u = gu.chunk(2, dim=-1)
            act = torch.nn.functional.silu(g.clamp(max=self.swiglu_limit)) * u.clamp(-self.swiglu_limit, self.swiglu_limit)
            y = torch.nn.functional.linear(act, self.down_proj[e]) * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


class _TinyGlmStyleMoE(nn.Module):
    def __init__(self, hidden: int, inter: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.router_w = nn.Linear(hidden, num_experts, bias=False)
        self.experts = _Tiny3DExperts(num_experts, hidden, inter)
        self.shared_experts = nn.Module()
        self.shared_experts.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.shared_experts.up_proj = nn.Linear(hidden, inter, bias=False)
        self.shared_experts.down_proj = nn.Linear(inter, hidden, bias=False)
        self.shared_experts.forward = lambda x: self.shared_experts.down_proj(
            torch.nn.functional.silu(self.shared_experts.gate_proj(x)) * self.shared_experts.up_proj(x)
        )

    def gate(self, hidden_states: torch.Tensor):
        logits = self.router_w(hidden_states.reshape(-1, hidden_states.shape[-1]))
        probs = torch.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(probs, k=self.top_k, dim=-1)
        return logits, w.to(hidden_states.dtype), idx

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig = hidden_states.shape
        _, w, idx = self.gate(hidden_states)
        out = self.experts(hidden_states.reshape(-1, orig[-1]), idx, w).view(*orig)
        return out + self.shared_experts(hidden_states)


class _TinyMHCBlock(nn.Module):
    """Simulates a 4-stream hyper-connection decoder layer (like GLM-5.3 Flash)."""

    def __init__(self, d: int, inter: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.ffn_hc = nn.Identity()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d, bias=False)
        self.self_attn.k_proj = nn.Linear(d, d, bias=False)
        self.self_attn.v_proj = nn.Linear(d, d, bias=False)
        self.self_attn.o_proj = nn.Linear(d, d, bias=False)
        self.mlp = _TinyGlmStyleMoE(d, inter, num_experts, top_k)
        self.input_layernorm = nn.LayerNorm(d)

    def forward(self, h4: torch.Tensor) -> tuple[torch.Tensor, None]:
        # h4: [B, T, 4, D]
        x = self.input_layernorm(h4.mean(dim=-2))
        a = self.self_attn.o_proj(self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x))
        m = self.mlp(x + 0.1 * a)
        return h4 + 0.1 * m.unsqueeze(-2), None


class _TinyMHCModel(nn.Module):
    def __init__(self, d: int, inter: int, v: int, n_layers: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(v, d)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [_TinyMHCBlock(d, inter, num_experts, top_k) for _ in range(n_layers)]
        )
        self.model.language_model.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, v, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        h = self.embed(input_ids).unsqueeze(-2).expand(-1, -1, 4, -1).contiguous()
        for blk in self.model.language_model.layers:
            h, _ = blk(h)
        return self.lm_head(self.model.language_model.norm(h.mean(dim=-2)))


def test_dual_trellis_v2_shared_banks_mhc_inter_scales_bias_lm_head_and_fp8_export() -> None:
    torch.manual_seed(13)
    prof = ModelProfile(
        name="synthetic-mhc-moe",
        hidden_size=64,
        intermediate_size=64,
        num_layers=4,
        routed_layers=(0, 1, 2, 3),
        num_experts=4,
        top_k=2,
        vocab_size=128,
        tp_size=2,
        rates=("k2", "p25", "k3", "k4"),
        non_routed_params=50_000,
        layer_path_template="model.language_model.layers.{layer}",
        moe_attr="mlp",
        attn_attr="self_attn",
        shared_expert_attr="shared_experts",
        tie_word_embeddings=False,
    )
    model = _TinyMHCModel(64, 64, 128, 4, 4, 2).to(torch.bfloat16)
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        base_out = model(ids).clone()

    class _FakeTok:
        def convert_ids_to_tokens(self, tid: int) -> str:
            return f"\u0120Tok{tid % 32}" if tid > 1 else "<bos>"

    canon = build_nfkc_canon_map(_FakeTok(), 128)
    assert int(canon[0]) == 0 and int(canon[34]) == int(canon[2])

    qec = FullStackQEC(
        model,
        prof,
        engram_mode="v2_shared_banks",
        engram_bank_groups={"early": [0, 1], "late": [2, 3]},
        lora_rank=4,
        engram_kwargs={
            "d_uni": 16,
            "d_bi": 8,
            "n_bi_heads": 2,
            "d_tri": 8,
            "n_tri_heads": 1,
            "d_skip": 8,
            "n_skip_heads": 1,
            "buckets": 131,
            "canon_map": canon,
            "table_device": "cpu",
        },
        train_norms=True,
        train_shared_lora=True,
        train_lm_head_lora=True,
        train_inter_scales=True,
        train_expert_bias=True,
        use_router_gate=True,
    )
    assert "lm_head" in qec.loras
    assert len(qec.banks) == 2 and len(qec.engrams) == 4
    assert len(qec.inter_scales) == 4 and len(qec.expert_bias) == 4

    # SharedHostNGramBank T == 1 and prev_tokens continuity
    b0 = qec.banks["early"]
    u1, bi1, tri1, sk1 = b0.lookup(ids[:, 2:3], prev_tokens=ids[:, 0:2])
    uf, bif, trif, skf = b0.lookup(ids[:, :3])
    assert torch.allclose(u1[:, 0], uf[:, 2], atol=1e-6)
    assert bi1 is not None and bif is not None and torch.allclose(bi1[:, 0], bif[:, 2], atol=1e-6)
    assert tri1 is not None and trif is not None and torch.allclose(tri1[:, 0], trif[:, 2], atol=1e-6)
    assert sk1 is not None and skf is not None and torch.allclose(sk1[:, 0], skf[:, 2], atol=1e-6)

    # Zero-init matches baseline on 4-stream mHC model
    qec.eval()
    with torch.no_grad():
        zero_out = qec(ids)
    assert torch.allclose(zero_out.float(), base_out.float(), atol=2e-2)

    # Gradients flow into shared banks, per-layer projections, router gate, inter_scales, expert_bias, lm_head LoRA, norms
    qec.train()
    opt = torch.optim.AdamW(qec.parameter_groups())
    target_logits = base_out.detach().float() + 0.5 * torch.randn_like(base_out.float())
    t_lse = torch.logsumexp(target_logits, dim=-1)
    top_vals, top_idx = torch.topk(target_logits, k=16, dim=-1)
    top_logp = (top_vals - t_lse.unsqueeze(-1)).clamp_max(0.0)
    p_tail = (1.0 - top_logp.exp().sum(dim=-1)).clamp(0.0, 1.0)
    for _ in range(2):
        opt.zero_grad()
        loss = topk_kl_loss(qec(ids), top_logp, top_idx, teacher_p_tail=p_tail, chunk_tokens=6)
        loss.backward()
        opt.step()

    assert float(qec.inter_scales["0"].grad.abs().sum()) > 0.0
    assert float(qec.expert_bias["0"].grad.abs().sum()) > 0.0
    assert float(qec.loras["lm_head"].a.weight.grad.abs().sum()) > 0.0
    assert float(qec.banks["early"].bi_embs[0].weight.grad.abs().sum()) > 0.0
    assert isinstance(qec.engrams["0"], DualTrellisEngramLayer)
    assert qec.engrams["0"].route_gate_w is not None
    assert float(qec.engrams["0"].route_gate_w.grad.abs().sum()) > 0.0

    # Merge parity + BF16/FP8 sidecar export
    qec.eval()
    with torch.no_grad():
        pre_merge = qec(ids).clone()
    qec.merge_lora_into_model()
    with torch.no_grad():
        post_merge = qec(ids)
    assert torch.allclose(pre_merge.float(), post_merge.float(), atol=5e-2)

    bf16_sc = qec.export_engram_sidecar(table_dtype=torch.bfloat16)
    fp8_sc = qec.export_engram_sidecar(table_dtype=torch.float8_e4m3fn)
    assert bf16_sc["banks"]["early"]["uni_emb.weight"].dtype == torch.bfloat16
    assert fp8_sc["banks"]["early"]["uni_emb.weight"].dtype == torch.float8_e4m3fn
    assert "uni_emb.weight.row_scale" in fp8_sc["banks"]["early"]
    eqec = qec.export_expert_qec_state()
    assert eqec["layer_0.intermediate_scales_fp16"].dtype == torch.float16
    assert eqec["layer_0.expert_bias_bf16"].dtype == torch.bfloat16
    qec.remove_hooks()
