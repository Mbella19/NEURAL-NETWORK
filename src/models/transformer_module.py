"""Transformer encoder module (Phase 4.3)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base_model import BaseModel, ModelMetadata
from .attention_fs import FSAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModule(BaseModel):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.5,
        output_dim: int = 1,
        use_fsatten: bool = False,
    ) -> None:
        metadata = ModelMetadata(
            model_name="TransformerModule",
            input_dim=input_dim,
            output_dim=output_dim,
        )
        super().__init__(metadata=metadata)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout=dropout)
        self.use_fsatten = use_fsatten
        self.fsatten = FSAttention(model_dim, dropout=dropout) if use_fsatten else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(inputs)
        x = self.positional_encoding(x)
        if self.fsatten is not None:
            x = x + self.fsatten(x)
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        pooled = encoded.mean(dim=1)
        logits = self.output_head(pooled)
        return logits


__all__ = ["TransformerModule"]
