"""LSTM module with attention support (Phase 4.2)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base_model import BaseModel, ModelMetadata


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attention_dim)
        self.context = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        energy = torch.tanh(self.proj(hidden_states))
        scores = self.context(energy)
        weights = torch.softmax(scores, dim=1)
        context = (weights * hidden_states).sum(dim=1)
        return context


class LSTMModule(BaseModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = False,
        attention_dim: Optional[int] = None,
        output_dim: int = 1,
    ) -> None:
        metadata = ModelMetadata(
            model_name="LSTMModule",
            input_dim=input_dim,
            output_dim=output_dim,
        )
        super().__init__(metadata=metadata)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.attention = AttentionLayer(hidden_dim * self.num_directions, attention_dim or hidden_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_directions, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        try:
            output, _ = self.lstm(inputs)
        except RuntimeError as exc:
            if "Placeholder storage has not been allocated on MPS device" in str(exc):
                cpu_inputs = inputs.to("cpu")
                cpu_output, _ = self.lstm.to("cpu")(cpu_inputs)
                output = cpu_output.to(inputs.device)
            else:
                raise
        context = self.attention(output)
        logits = self.fc(context)
        return logits


__all__ = ["LSTMModule"]
