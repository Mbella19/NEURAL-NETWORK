"""Multi-task wrapper to add auxiliary heads to any backbone."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .base_model import BaseModel, ModelMetadata


class MultiTaskModel(BaseModel):
    def __init__(
        self,
        backbone: BaseModel,
        aux_heads: Dict[str, int],
        dropout: float = 0.2,
        policy_task_name: str = "PolicyExecutionTask",
    ) -> None:
        metadata = ModelMetadata(
            model_name=f"{backbone.metadata.model_name}-multitask",
            input_dim=backbone.metadata.input_dim,
            output_dim=backbone.metadata.output_dim,
        )
        super().__init__(metadata=metadata)
        self.backbone = backbone
        feat_dim = backbone.metadata.output_dim or 1
        self.policy_task_name = policy_task_name
        self.aux_heads = nn.ModuleDict()
        for name, dim in aux_heads.items():
            self.aux_heads[name] = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, feat_dim),  # Hidden layer
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, dim),       # Output layer
            )

    def forward(self, inputs: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        main_out = self.backbone(inputs, **kwargs)
        outputs = {"main": main_out}
        for name, head in self.aux_heads.items():
            head_in = main_out
            if name == self.policy_task_name and main_out.dim() > 2:
                # Pool over time for decision/execution head
                head_in = main_out.mean(dim=1)
            outputs[name] = head(head_in)
        return outputs


__all__ = ["MultiTaskModel"]
