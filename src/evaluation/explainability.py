"""Explainability utilities (saliency + SHAP/LIME wrappers)."""
from __future__ import annotations

from typing import Dict, Optional

import torch


def saliency_map(model: torch.nn.Module, inputs: torch.Tensor, target_index: int = 0) -> Dict[str, torch.Tensor]:
    """Return absolute gradients w.r.t inputs as saliency."""

    model.eval()
    inputs = inputs.clone().detach().requires_grad_(True)
    output = model(inputs)
    if output.ndim > 1:
        scalar = output[..., target_index].sum()
    else:
        scalar = output.sum()
    scalar.backward()
    saliency = inputs.grad.abs()
    return {"saliency": saliency, "output": output.detach()}


def shap_values(model: torch.nn.Module, background: torch.Tensor, samples: torch.Tensor):
    """SHAP wrapper (requires shap dependency)."""

    import shap  # type: ignore

    model.eval()
    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(model, masker)
    return explainer(samples)


def lime_explanation(model: torch.nn.Module, sample: torch.Tensor, num_features: int = 5):
    """LIME wrapper (requires lime dependency)."""

    from lime import lime_tabular  # type: ignore
    import numpy as np

    model.eval()
    sample_np = sample.detach().cpu().numpy()
    explainer = lime_tabular.LimeTabularExplainer(sample_np, mode="regression")
    predict_fn = lambda x: model(torch.tensor(x, dtype=sample.dtype)).detach().cpu().numpy()
    exp = explainer.explain_instance(sample_np[0], predict_fn, num_features=num_features)
    return exp


__all__ = ["saliency_map"]
