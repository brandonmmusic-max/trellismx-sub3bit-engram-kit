"""Memory-safe query-chunked wrappers for Glm5Next DSA indexer and sparse attention.

At 64K sequence length, stock ``Glm5NextTextIndexer.forward`` materializes
``[B, 65536, 32, 16384]`` FP32 scores twice (256 GiB) and
``Glm5NextTextAttention.build_attention_mask_from_topk`` materializes a 20 GiB
``[B, 65536, 65536]`` int32/bool mask. Chunking along the independent query
dimension produces bit-identical ``topk_indices`` and algebraically identical
differentiable attention outputs while keeping peak VRAM under 5 GiB.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_chunked_indexer_forward(indexer: nn.Module, q_chunk: int = 2048):
    orig_forward = indexer.forward

    @torch.no_grad()
    def chunked_forward(
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.BoolTensor,
        cache=None,
        **kwargs,
    ) -> torch.Tensor:
        if cache is None and "past_key_values" in kwargs:
            cache = kwargs.pop("past_key_values")
        batch_size, seq_len = hidden_states.shape[:2]
        if seq_len <= q_chunk:
            return orig_forward(
                hidden_states=hidden_states,
                q_resid=q_resid,
                attention_mask=attention_mask,
                cache=cache,
            )

        hidden_shape = (batch_size, seq_len, -1, indexer.head_dim)
        q = indexer.wq_b(q_resid).view(hidden_shape)
        k = indexer.k_norm(indexer.wk(hidden_states)).view(hidden_shape).squeeze(2)

        gate_scores = F.linear(hidden_states, indexer.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]
        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

        kv_len = seq_len
        current_length = seq_len
        if cache is not None:
            cache_layer = cache.layers[indexer.layer_idx]
            packed_states = cache.update_indexer(packed_states, indexer.layer_idx)
            kv_len = cache_layer.keys.shape[-2]
            current_length = cache_layer.get_seq_length()

        valid_keys = packed_states[..., -1].bool()
        pool_keys, pool_indices, pool_valid = indexer.get_pooled_states(packed_states=packed_states)
        pool_keys_t = pool_keys.transpose(-1, -2).float().unsqueeze(1)
        weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype)).float() * (
            indexer.n_heads**-0.5
        )
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        kv_positions = torch.arange(valid_keys.shape[-1], device=valid_keys.device)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]

        select_k = min(indexer.index_topk // indexer.index_kpool, pool_keys.shape[-2])
        output_width = indexer.index_topk + (indexer.index_kpool - 1 if indexer.index_kpool_always_select_tail else 0)
        min_f = torch.finfo(torch.float32).min

        chunks: list[torch.Tensor] = []
        for q0 in range(0, seq_len, q_chunk):
            q1 = min(seq_len, q0 + q_chunk)
            q_len_c = q1 - q0
            q_pos = (current_length - seq_len) + torch.arange(q0, q1, device=valid_keys.device)
            vis_c = (kv_positions[None, None, :] <= q_pos[None, :, None]) & valid_keys[:, None, :]

            scores_c = torch.matmul(q[:, q0:q1].float(), pool_keys_t)
            scores_c.mul_(indexer.softmax_scale).relu_()
            index_scores_c = torch.matmul(weights[:, q0:q1].unsqueeze(-2), scores_c).squeeze(-2)
            del scores_c

            pool_visible_c = vis_c.gather(
                dim=-1,
                index=pool_end[:, None, :].expand(batch_size, q_len_c, -1),
            )
            valid_candidates_c = pool_visible_c & pool_valid[:, None]
            index_scores_c = index_scores_c.masked_fill(~valid_candidates_c, min_f)

            selected = index_scores_c.topk(select_k, dim=-1).indices
            selected_valid = valid_candidates_c.gather(-1, selected)
            selected_indices = pool_indices[batch_idx, selected]

            topk_c = selected_indices.flatten(-2)
            topk_c = topk_c.masked_fill(
                ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
                -1,
            )
            if indexer.index_kpool_always_select_tail:
                topk_c = indexer.append_visible_tail(topk_c, vis_c, valid_keys)
            if topk_c.shape[-1] < output_width:
                topk_c = F.pad(topk_c, (0, output_width - topk_c.shape[-1]), value=-1)
            topk_c = topk_c[..., :output_width]
            topk_c = topk_c.masked_fill(~attention_mask[:, q0:q1, None], -1)
            chunks.append(topk_c.to(torch.int32))
            del vis_c, index_scores_c

        return torch.cat(chunks, dim=1)

    return chunked_forward


def _make_chunked_attention_forward(attn: nn.Module, q_chunk: int = 4096):
    orig_forward = attn.forward

    def chunked_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache=None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ):
        if cache is None and "past_key_values" in kwargs:
            cache = kwargs.pop("past_key_values")
        batch_size, seq_length = hidden_states.shape[:-1]
        if seq_length <= q_chunk:
            return orig_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache=cache,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.glm5_next.modeling_glm5_next import eager_attention_forward

        query_shape = (batch_size, seq_length, -1, attn.qk_head_dim)
        q_resid = attn.q_a_layernorm(attn.q_a_proj(hidden_states))
        query_states = attn.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

        compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
        k_pass = attn.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, attn.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, attn.qk_rope_head_dim)

        if cache is not None:
            k_pass, k_rot = cache.update(k_pass, k_rot, attn.layer_idx, kwargs=kwargs)

        key_states, value_states = attn.expand_kv(k_pass, k_rot)

        if attn.indexer is not None:
            topk_indices = attn.indexer(
                hidden_states=hidden_states,
                q_resid=q_resid,
                attention_mask=attention_mask,
                cache=cache,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, eager_attention_forward
        )
        kv_len = key_states.shape[2]
        drop_p = 0.0 if not attn.training else attn.attention_dropout

        use_ckpt = bool(
            torch.is_grad_enabled()
            and (query_states.requires_grad or key_states.requires_grad or value_states.requires_grad)
        )

        def _attn_chunk(
            q_c: torch.Tensor,
            k_s: torch.Tensor,
            v_s: torch.Tensor,
            topk_c: torch.Tensor,
        ) -> torch.Tensor:
            mask_c = attn.build_attention_mask_from_topk(
                topk_indices=topk_c,
                query_states=q_c,
                kv_length=kv_len,
            )
            out_c, _ = attention_interface(
                attn,
                q_c,
                k_s,
                v_s,
                mask_c,
                dropout=drop_p,
                scaling=attn.scaling,
                **kwargs,
            )
            return out_c

        out_chunks: list[torch.Tensor] = []
        for q0 in range(0, seq_length, q_chunk):
            q1 = min(seq_length, q0 + q_chunk)
            q_c = query_states[:, :, q0:q1, :]
            topk_c = topk_indices[:, q0:q1, :]
            if use_ckpt:
                out_c = torch.utils.checkpoint.checkpoint(
                    _attn_chunk,
                    q_c,
                    key_states,
                    value_states,
                    topk_c,
                    use_reentrant=False,
                )
            else:
                out_c = _attn_chunk(q_c, key_states, value_states, topk_c)
            out_chunks.append(out_c)

        attn_output = torch.cat(out_chunks, dim=1).reshape(batch_size, seq_length, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, None, topk_indices if attn.next_skip_topk else None

    return chunked_forward


def patch_glm5_fla_causal_conv1d() -> bool:
    """Replace pure-PyTorch Glm5Next causal_conv1d_fn fallback with fla Triton kernel when available."""
    try:
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
        import transformers.models.glm5_next.modeling_glm5_next as glm5_mod
    except Exception:
        return False

    if getattr(glm5_mod, "_trellismx_fla_conv1d_patched", False):
        return True

    def _fla_causal_conv1d_fn(
        hidden_states: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter | None = None,
        activation: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            _, hidden_size, seq_len = hidden_states.shape
            padding = weight.shape[-1] - 1
            out = F.conv1d(
                hidden_states.to(weight.dtype),
                weight=weight.unsqueeze(1),
                bias=bias,
                padding=padding,
                groups=hidden_size,
            )[:, :, :seq_len]
            if activation == "silu":
                out = F.silu(out)
            return out.to(hidden_states.dtype)
        # fla causal_conv1d expects [B, T, D]; Glm5Next passes [B, D, T].
        x_btd = hidden_states.transpose(1, 2).contiguous()
        out_btd, _ = fla_causal_conv1d(
            x_btd,
            weight=weight,
            bias=bias,
            activation=activation,
            backend="triton",
        )
        return out_btd.transpose(1, 2).to(hidden_states.dtype)

    glm5_mod.causal_conv1d_fn = _fla_causal_conv1d_fn  # type: ignore[attr-defined]
    glm5_mod._trellismx_fla_conv1d_patched = True  # type: ignore[attr-defined]
    return True


def patch_ep_dp_all_to_all(model: nn.Module) -> int:
    """Switch all_reduce EP experts to DP+EP all_to_all without wrapping the model in FSDP2.

    Loads keep pure replicated non-expert weights (supporting mixed BF16/FP32 parameters
    such as Glm5Next hyper-connections and router biases) while converting every
    ``EpDispatchExpertsParallel`` module to ``AllToAllEP`` with ``tp_mesh.size() == 1``
    so each rank processes an independent data-parallel batch in parallel.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers.distributed.expert_parallel import (
        AllToAllEP,
        EPImplementation,
        EpDispatchExpertsParallel,
    )

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return 0

    world_size = dist.get_world_size()
    tp1_mesh = init_device_mesh("cuda", (world_size, 1), mesh_dim_names=("dp", "tp"))["tp"]
    patched = 0
    for mod in model.modules():
        fwd = getattr(mod, "forward", None)
        closure = getattr(fwd, "__closure__", None)
        if not closure:
            continue
        orig_impl = None
        orig_parallel = None
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, EPImplementation):
                orig_impl = val
            elif isinstance(val, EpDispatchExpertsParallel):
                orig_parallel = val
        if orig_impl is None:
            continue
        if isinstance(orig_impl, AllToAllEP) and orig_impl.tp_mesh.size() == 1:
            patched += 1
            continue
        base_experts_fwd = orig_impl.experts_forward

        def _make_chunked_local_experts(fn, max_rows: int = 65536):
            def _chunked(hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
                n_rows = int(hidden_states.shape[0])
                if n_rows <= max_rows:
                    return fn(hidden_states, top_k_index, top_k_weights)
                parts = [
                    fn(hidden_states[i : i + max_rows], top_k_index[i : i + max_rows], top_k_weights[i : i + max_rows])
                    for i in range(0, n_rows, max_rows)
                ]
                return torch.cat(parts, dim=0)

            return _chunked

        mod.forward = _make_chunked_local_experts(base_experts_fwd)  # type: ignore[method-assign]
        parallel = orig_parallel if orig_parallel is not None else EpDispatchExpertsParallel()
        parallel.install_forward(
            mod,
            orig_impl.ep_mesh,
            tp_mesh=tp1_mesh,
            ep_implementation="all_to_all",
        )
        ep_fwd = mod.forward

        def _make_chunked_ep_dispatch(fn, max_tokens: int = 8192):
            def _chunked_ep(
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ):
                n_tok = int(hidden_states.shape[0])
                if n_tok <= max_tokens:
                    return fn(hidden_states, top_k_index, top_k_weights)
                use_ckpt = bool(torch.is_grad_enabled() and hidden_states.requires_grad)
                parts = []
                for i in range(0, n_tok, max_tokens):
                    h_c = hidden_states[i : i + max_tokens]
                    idx_c = top_k_index[i : i + max_tokens]
                    w_c = top_k_weights[i : i + max_tokens]
                    if use_ckpt:
                        parts.append(
                            torch.utils.checkpoint.checkpoint(
                                fn, h_c, idx_c, w_c, use_reentrant=False
                            )
                        )
                    else:
                        parts.append(fn(h_c, idx_c, w_c))
                return torch.cat(parts, dim=0)

            return _chunked_ep

        mod.forward = _make_chunked_ep_dispatch(ep_fwd, max_tokens=8192)  # type: ignore[method-assign]
        patched += 1
    # Verify no non-expert parameter (especially lm_head or embed_tokens) remained TP-sharded.
    for pname, param in model.named_parameters():
        if hasattr(param, "to_local") and ".experts." not in pname:
            raise RuntimeError(
                f"Non-expert parameter {pname!r} is still a sharded DTensor under DP+EP all_to_all; "
                "pass tp_plan={'lm_head': 'replicate'} in DistributedConfig."
            )
    return patched


def patch_glm5_long_context(
    model: nn.Module,
    *,
    indexer_q_chunk: int = 2048,
    attn_q_chunk: int = 4096,
) -> tuple[int, int]:
    """Install query-chunked forwards on Glm5Next DSA indexer and full-attention modules, plus fla conv1d."""
    patch_glm5_fla_causal_conv1d()
    n_idx = 0
    n_attn = 0
    for mod in model.modules():
        cls_name = mod.__class__.__name__
        if cls_name == "Glm5NextTextIndexer" and not getattr(mod, "_trellismx_chunked", False):
            mod.forward = _make_chunked_indexer_forward(mod, q_chunk=indexer_q_chunk)  # type: ignore[method-assign]
            mod._trellismx_chunked = True  # type: ignore[attr-defined]
            n_idx += 1
        elif cls_name == "Glm5NextTextAttention" and not getattr(mod, "_trellismx_chunked", False):
            mod.forward = _make_chunked_attention_forward(mod, q_chunk=attn_q_chunk)  # type: ignore[method-assign]
            mod._trellismx_chunked = True  # type: ignore[attr-defined]
            n_attn += 1
    return n_idx, n_attn
