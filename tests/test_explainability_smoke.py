import pytest
import torch

from evaluation.explainability import saliency_map


def test_saliency_map_runs():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)

        def forward(self, x):
            return self.linear(x)

    model = Tiny()
    x = torch.randn(1, 4, requires_grad=True)
    result = saliency_map(model, x)
    assert "saliency" in result
    assert result["saliency"].shape == x.shape


@pytest.mark.skip(reason="shap/lime heavy; run selectively if deps installed")
def test_shap_lime_optional():
    pytest.importorskip("shap")
    pytest.importorskip("lime")
