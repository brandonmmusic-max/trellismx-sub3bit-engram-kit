"""Fast per-layer KLD sensitivity probe for hierarchical rate allocation.

Measures one positive sensitivity scalar per routed layer by injecting matched
per-token relative Gaussian noise (with antithetic +/- variance reduction) into
one routed layer's MoE output at a time on a small selection split and recording
exact chunked FP64 mean KL(teacher || perturbed). Supports both single-GPU and
multi-GPU (device_map='auto') models and both [B, T, V] and [T, V] logit shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .engram_qec import _resolve_attr
from .kld_score import tokenwise_kld_fp64
from .profile import ModelProfile


def _normalize_logits_2d(logits: torch.Tensor) -> torch.Tensor:
    """Slice autoregressive prefix positions and return 2D [N, V]."""
    if logits.ndim == 3:
        if logits.shape[1] < 2:
            raise ValueError(f"logit sequence length must be >= 2, got {tuple(logits.shape)}")
        return logits[:, :-1, :].reshape(-1, logits.shape[-1])
    if logits.ndim == 2:
        if logits.shape[0] < 2:
            raise ValueError(f"logit sequence length must be >= 2, got {tuple(logits.shape)}")
        return logits[:-1, :]
    raise ValueError(f"expected 2D [T, V] or 3D [B, T, V] logits, got shape {tuple(logits.shape)}")


def _input_device(model: nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight.device.type != "meta":
            return emb.weight.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


@torch.no_grad()
def measure_layer_sensitivity(
    model: nn.Module,
    profile: ModelProfile,
    selection_windows: Sequence[torch.Tensor],
    forward_logits_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    *,
    noise_rel_std: float = 0.05,
    antithetic: bool = True,
    seed: int = 20261121,
) -> dict[int, float]:
    """Return a positive KLD sensitivity scalar for every routed layer."""
    profile.validate()
    if not selection_windows:
        raise ValueError("selection_windows must be nonempty")
    if noise_rel_std <= 0:
        raise ValueError("noise_rel_std must be positive")

    model.eval()
    in_dev = _input_device(model)
    # Cache compact BF16 teacher logits on CPU (4x smaller than FP64 logprobs)
    teacher_logits_cpu: list[torch.Tensor] = []
    for ids in selection_windows:
        raw = forward_logits_fn(model, ids.to(in_dev))
        teacher_logits_cpu.append(_normalize_logits_2d(raw).to(dtype=torch.bfloat16, device="cpu"))

    signs = (1.0, -1.0) if antithetic else (1.0,)
    sensitivities: dict[int, float] = {}
    for layer in profile.routed_layers:
        block = _resolve_attr(model, profile.layer_path_template.format(layer=layer))
        moe = getattr(block, profile.moe_attr)

        kl_sum = 0.0
        n_pos = 0
        for sign in signs:
            # Lazily allocate generator on h.device so multi-GPU device_map='auto' never mismatches
            gen_holder: dict[torch.device, torch.Generator] = {}

            def make_hook(cur_sign: float):
                def hook(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    gen = gen_holder.get(h.device)
                    if gen is None:
                        gen = torch.Generator(device=h.device).manual_seed(seed + layer)
                        gen_holder[h.device] = gen
                    # Per-token RMS avoids distortion from BOS/attention sink tokens
                    rms = h.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
                    noise = torch.randn(h.shape, generator=gen, device=h.device, dtype=torch.float32) * (
                        cur_sign * noise_rel_std * rms
                    )
                    perturbed = h + noise.to(h.dtype)
                    return ((perturbed,) + out[1:]) if isinstance(out, tuple) else perturbed
                return hook

            handle = moe.register_forward_hook(make_hook(sign))
            try:
                for idx, ids in enumerate(selection_windows):
                    s_raw = forward_logits_fn(model, ids.to(in_dev))
                    s_2d = _normalize_logits_2d(s_raw)
                    t_2d = teacher_logits_cpu[idx].to(device=s_2d.device)
                    kld, _ = tokenwise_kld_fp64(t_2d, s_2d)
                    kl_sum += float(kld.sum().item())
                    n_pos += int(kld.numel())
            finally:
                handle.remove()

        if n_pos == 0:
            raise RuntimeError(f"layer {layer} sensitivity probe scored zero positions")
        sensitivities[layer] = max(kl_sum / n_pos, 1e-9)

    return sensitivities


def save_layer_sensitivity(path: str | Path, profile: ModelProfile, sensitivities: dict[int, float]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "trellismx.layer-sensitivity.v1",
        "model_name": profile.name,
        "layers": {str(k): float(v) for k, v in sorted(sensitivities.items())},
    }
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
