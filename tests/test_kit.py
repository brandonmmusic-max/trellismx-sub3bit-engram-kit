"""CPU unit and invariant test suite for trellismx_kit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from trellismx_kit.allocate import (
    EMPIRICAL_RD_RATIOS,
    build_hierarchical_assignment,
)
from trellismx_kit.engram_probe import score_expert_ngram_predictability
from trellismx_kit.engram_qec import (
    AttentionLoRA,
    EngramQECLayer,
    FullStackQEC,
    build_context_curriculum,
    topk_kl_loss,
)
from trellismx_kit.kld_score import summarize_windows, tokenwise_kld_fp64
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
    p = ModelProfile(
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
    p.validate()
    return p


def test_shipped_profiles_validate() -> None:
    configs = sorted((REPO_ROOT / "configs").glob("*.json"))
    assert len(configs) >= 4
    for cfg in configs:
        prof = load_profile(cfg)
        prof.validate()
        assert prof.local_intermediate_size % 32 == 0


@pytest.mark.parametrize("rate", RATES)
def test_trellis_pack_unpack_and_decode_all_four_rates(rate: str) -> None:
    gen = torch.Generator().manual_seed(101 + STREAM_BITS[rate])
    symbols = torch.randint(0, 1 << STREAM_BITS[rate], (16, SYMBOLS_PER_TILE[rate]), generator=gen, dtype=torch.int16)
    packed = pack_symbols(symbols, rate)
    assert packed.shape == (16, TILE_I16[rate])
    unpacked = unpack_symbols(packed, rate)
    assert torch.equal(unpacked, symbols)
    ops = decode_tile_operands(packed, rate)
    vals = decode_tile_values(packed, rate)
    assert ops.shape == (16, 256)
    assert ops.dtype == torch.uint8
    assert torch.isfinite(vals).all()
    assert state_lut().shape == (65536,)


def test_csf_lossless_round_trip_and_all_byte_values() -> None:
    gen = torch.Generator().manual_seed(7)
    smooth = torch.randint(120, 122, (8, 64, 16), generator=gen, dtype=torch.uint8)
    smooth.reshape(-1)[::37] = 135
    csf = encode_csf_plane(smooth)
    assert torch.equal(decode_csf_plane(csf, smooth.shape), smooth)
    # All 256 possible uint8 values
    every = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    assert torch.equal(decode_csf_plane(encode_csf_plane(every), every.shape), every)


@pytest.mark.parametrize("scale_storage", ("plain", "csf"))
def test_four_bank_sidecar_round_trip_and_device_layout(tmp_path: Path, scale_storage: str) -> None:
    prof = _tiny_profile()
    rates = ["k2", "p25", "k3", "k4", "k2", "k4", "p25", "k3"]
    sidecar_path = synthetic_rank_sidecar(
        tmp_path / f"rank-{scale_storage}.safetensors",
        prof,
        layer=2,
        rank=0,
        expert_rates=rates,
        scale_storage=scale_storage,
        seed=42,
    )
    loaded = read_rank_sidecar(sidecar_path, prof)
    assert loaded.counts == {"k2": 2, "p25": 2, "k3": 2, "k4": 2}
    layout = device_layout(loaded, prof)
    for e in range(prof.num_experts):
        g_ops, u_ops, d_ops = decode_expert_operands_from_layout(layout, prof, e)
        n_tiles = prof.hidden_tiles * prof.local_intermediate_tiles
        assert g_ops.shape == (n_tiles, 256)
        assert u_ops.shape == (n_tiles, 256)
        assert d_ops.shape == (n_tiles, 256)


def test_sidecar_rejects_tampered_tensor(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open

    prof = _tiny_profile()
    p = synthetic_rank_sidecar(tmp_path / "ok.safetensors", prof, scale_storage="csf")
    with safe_open(str(p), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})
    tensors = load_file(str(p))
    tensors["w13_scale_csf_base"][0] = (int(tensors["w13_scale_csf_base"][0]) + 1) % 256
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, str(bad), metadata=meta)
    with pytest.raises(ValueError, match="SHA-256"):
        read_rank_sidecar(bad, prof)


def test_hierarchical_allocator_exact_budget_and_k4_cap() -> None:
    prof = _tiny_profile()
    rng = np.random.default_rng(123)
    route_mass = {l: rng.exponential(scale=1.0, size=prof.num_experts).tolist() for l in prof.routed_layers}
    layer_sens = {0: 3.0, 1: 1.0, 2: 0.5, 3: 2.0}
    doc = build_hierarchical_assignment(
        prof,
        route_mass=route_mass,
        layer_sensitivity=layer_sens,
        budget_bits=2.5,
        max_k4_fraction=0.125,
    )
    assert abs(doc["achieved_bits"] - 2.5) <= 1e-6
    assert doc["counts"]["k4"] <= 4
    assert doc["rule"] == "hierarchical"
    assert len(doc["sha256"]) == 64


def test_stratified_capture_raises_rare_expert_floor() -> None:
    rng = np.random.default_rng(9)
    records = []
    for i in range(20):
        counts = rng.integers(100, 300, size=(2, 4), dtype=np.int64)
        if i in (3, 11, 17):
            counts[1, 3] = 900  # rare expert triggered only in these windows
        else:
            counts[1, 3] = 5
        records.append(WindowRecord(window_id=f"w{i}", domain="code" if i % 2 else "chat", expert_counts=counts))
    sel = select_stratified_windows(records, target_windows=5, token_floor=1500)
    assert len(sel.selected_ids) == 5
    assert sel.min_expert_tokens >= 900
    assert "w3" in sel.selected_ids and "w11" in sel.selected_ids


class _TinyBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d, bias=False)
        self.self_attn.k_proj = nn.Linear(d, d, bias=False)
        self.self_attn.v_proj = nn.Linear(d, d, bias=False)
        self.self_attn.o_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.self_attn.o_proj(self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x))
        return self.norm(x + 0.1 * y)


class _TinyModel(nn.Module):
    def __init__(self, d: int, v: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(v, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyBlock(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, v, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        for blk in self.model.layers:
            h = blk(h)
        return self.head(h)


def test_fullstack_qec_zero_init_and_exact_lora_merge() -> None:
    torch.manual_seed(0)
    prof = _tiny_profile()
    model = _TinyModel(prof.hidden_size, prof.vocab_size, prof.num_layers)
    ids = torch.randint(0, prof.vocab_size, (2, 16))
    with torch.no_grad():
        base_out = model(ids).clone()

    qec = FullStackQEC(
        model,
        prof,
        engram_layers=(0, 3),
        lora_rank=4,
        engram_kwargs={"d_uni": 16, "d_bi": 8, "n_bi_heads": 2, "bi_buckets": 257, "d_tri": 8, "n_tri_heads": 1, "tri_buckets": 257},
    )
    qec.eval()
    qec.cur_tokens = ids
    with torch.no_grad():
        zero_out = qec.model(ids)
    assert torch.allclose(zero_out, base_out, atol=1e-6)

    # One optimization step verifies gradients flow to both LoRA and Engram
    qec.train()
    opt = torch.optim.AdamW(qec.parameter_groups())
    logits = qec.model(ids)
    t_logp = torch.log_softmax(base_out.detach(), dim=-1)
    top_logp, top_idx = torch.topk(t_logp, k=8, dim=-1)
    loss = topk_kl_loss(logits, top_logp, top_idx)
    loss.backward()
    opt.step()

    # Verify algebraic LoRA merge preserves model outputs identically
    qec.eval()
    qec.cur_tokens = ids
    with torch.no_grad():
        pre_merge = qec.model(ids).clone()
    merged = qec.merge_lora_into_model()
    assert merged == 4 * prof.num_layers
    with torch.no_grad():
        post_merge = qec.model(ids)
    assert torch.allclose(pre_merge, post_merge, atol=1e-5)
    qec.remove_hooks()


def test_engram_probe_and_kld_and_vram_budget() -> None:
    rng = np.random.default_rng(5)
    n = 1000
    toks = rng.integers(0, 20, size=n, dtype=np.int64)
    prev = np.concatenate([np.array([-1]), toks[:-1]])
    table = torch.randn(20, 32)
    y = table[toks] + 0.05 * torch.randn(n, 32)
    is_build = np.arange(n) < 800
    res = score_expert_ngram_predictability(y, toks, prev, is_build)
    assert res.unigram_r2 > 0.95 and res.unigram_nmse < 0.05

    t_logits = torch.randn(64, 50)
    kld_zero, agree_zero = tokenwise_kld_fp64(t_logits, t_logits)
    assert float(kld_zero.abs().max()) < 1e-12
    assert int(agree_zero.sum()) == 64
    summ = summarize_windows([kld_zero, kld_zero], [agree_zero, agree_zero], resamples=200)
    assert summ.mean_kld < 1e-12 and summ.top1_agreement_pct == 100.0

    prof_1t = load_profile(REPO_ROOT / "configs" / "template-1t-moe.json")
    est = estimate_memory(prof_1t, avg_expert_bits=2.25, gpu_count=4)
    assert est.kv_headroom_per_gpu_gib > 10.0


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
    assert sched[0] == "short" and "long" in sched[:10]

    sched_tail = build_context_curriculum(
        total_tokens=1_000_000,
        short_tokens_per_step=10_000,
        long_tokens_per_step=20_000,
        long_fraction=0.20,
        mode="tail",
    )
    assert sched_tail == ["short"] * 80 + ["long"] * 10

    # Exact full-vocab equivalence when K == V and zero loss at teacher == student
    torch.manual_seed(7)
    t_logits = torch.randn(2, 8, 64)
    s_logits = t_logits + 0.2 * torch.randn(2, 8, 64)
    t_lse = torch.logsumexp(t_logits, dim=-1)
    top_vals_all, top_idx_all = torch.topk(t_logits, k=64, dim=-1)
    kl_exact = torch.nn.functional.kl_div(
        torch.log_softmax(s_logits, dim=-1),
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
    )
    assert torch.allclose(kl_exact, kl_sparse_all, atol=1e-5)

    top_vals_16, top_idx_16 = torch.topk(t_logits, k=16, dim=-1)
    kl_self = topk_kl_loss(
        t_logits,
        top_vals_16 - t_lse.unsqueeze(-1),
        top_idx_16,
        teacher_logsumexp=t_lse,
        teacher_top_logits=top_vals_16,
    )
    assert float(kl_self) < 1e-6

    # Verify sparse attention indexer norms remain frozen when train_norms=True
    prof = _tiny_profile()
    model = _TinyModel(prof.hidden_size, prof.vocab_size, prof.num_layers)
    model.model.layers[0].self_attn.indexer_norm = nn.LayerNorm(prof.hidden_size)
    qec = FullStackQEC(model, prof, engram_layers=(0,), lora_rank=4, train_norms=True)
    assert not model.model.layers[0].self_attn.indexer_norm.weight.requires_grad
    assert model.model.layers[0].norm.weight.requires_grad
    qec.remove_hooks()
