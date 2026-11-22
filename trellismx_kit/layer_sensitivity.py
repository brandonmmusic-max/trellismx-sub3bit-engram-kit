"""Fast per-layer KLD sensitivity probe for hierarchical rate allocation.

Measures one positive sensitivity scalar per routed layer by injecting matched
relative Gaussian noise into one routed layer's MoE output at a time on a small
selection split and recording mean KL(teacher || perturbed). Produces one scalar
per layer rather than thousands of noisy per-expert estimates, feeding directly
into trellismx_kit.allocate.build_hierarchical_assignment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .engram_qec import _resolve_attr
from .profile import ModelProfile


@torch.no_grad()
def measure_layer_sensitivity(
    model: nn.Module,
    profile: ModelProfile,
    selection_windows: Sequence[torch.Tensor],
    forward_logits_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    *,
    noise_rel_std: float = 0.05,
    seed: int = 20261121,
) -> dict[int, float]:
    """Return a positive KLD sensitivity scalar for every routed layer."""
    profile.validate()
    if not selection_windows:
        raise ValueError("selection_windows must be nonempty")
    if noise_rel_std <= 0:
        raise ValueError("noise_rel_std must be positive")

    model.eval()
    device = next(model.parameters()).device
    # Cache clean teacher log-probabilities per window
    teacher_logps: list[torch.Tensor] = []
    for ids in selection_windows:
        logits = forward_logits_fn(model, ids.to(device))[:-1]
        teacher_logps.append(logits.double().log_softmax(dim=-1).cpu())

    sensitivities: dict[int, float] = {}
    for layer in profile.routed_layers:
        block = _resolve_attr(model, profile.layer_path_template.format(layer=layer))
        moe = getattr(block, profile.moe_attr)
        gen = torch.Generator(device=device).manual_seed(seed + layer)

        def make_hook(g: torch.Generator):
            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    h = out[0]
                    rms = h.float().square().mean().sqrt().clamp_min(1e-6)
                    noise = torch.randn(h.shape, generator=g, device=h.device, dtype=torch.float32) * (noise_rel_std * rms)
                    return (h + noise.to(h.dtype),) + out[1:]
                rms = out.float().square().mean().sqrt().clamp_min(1e-6)
                noise = torch.randn(out.shape, generator=g, device=out.device, dtype=torch.float32) * (noise_rel_std * rms)
                return out + noise.to(out.dtype)
            return hook

        handle = moe.register_forward_hook(make_hook(gen))
        try:
            kl_sum = 0.0
            n_pos = 0
            for idx, ids in enumerate(selection_windows):
                t_logp = teacher_logps[idx].to(device)
                s_logp = forward_logits_fn(model, ids.to(device))[:-1].double().log_softmax(dim=-1)
                kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
                kl_sum += float(kl.sum().item())
                n_pos += int(kl.numel())
        finally:
            handle.remove()
        sensitivities[layer] = max(kl_sum / max(n_pos, 1), 1e-9)

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
