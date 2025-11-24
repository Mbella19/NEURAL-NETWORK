"""Lightweight Temporal Fusion Transformer (TFT) variant for multi-horizon forecasting."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base_model import BaseModel, ModelMetadata
from .attention_fs import FSAttention


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        out = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out),
        )
        self.gate = nn.Sequential(
            nn.Linear(out, out),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        gate = self.gate(y)
        return self.norm(x[..., : y.shape[-1]] + gate * y)


class TemporalFusionTransformer(BaseModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 20,
        use_fsatten: bool = True,
        output_features: bool = False,  # NEW: Output features instead of predictions
    ) -> None:
        # Determine output_dim based on mode
        output_dim = d_model if output_features else horizon

        metadata = ModelMetadata(
            model_name="TemporalFusionTransformer",
            input_dim=input_dim,
            output_dim=output_dim,
        )
        super().__init__(metadata=metadata)

        # Store for later use (reset_head, etc.)
        self.d_model = d_model
        self.dropout_rate = dropout
        self.output_features = output_features

        self.encoder = nn.GRU(input_dim, d_model, batch_first=True, num_layers=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.context_gate = GatedResidualNetwork(d_model, d_model * 2, dropout=dropout)
        self.fsatten = FSAttention(d_model, dropout=dropout) if use_fsatten else None

        # Head is only used when output_features=False
        if not output_features:
            self.head = self._build_head(horizon)
        else:
            self.head = None  # No head when outputting features

    def _build_head(self, output_dim: int) -> nn.Module:
        """Build output head for prediction mode."""
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, output_dim),
        )

    def reset_head(self, output_dim: int) -> None:
        """Rebuild output head with new output dimension (only works when output_features=False)."""
        if self.output_features:
            raise ValueError("Cannot reset head when output_features=True - use feature mode with MultiTaskModel instead")
        device = next(self.parameters()).device
        self.head = self._build_head(output_dim).to(device)
        self.metadata.output_dim = output_dim

    def forward(self, inputs: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc_out, _ = self.encoder(inputs)
        if self.fsatten is not None:
            enc_out = enc_out + self.fsatten(enc_out)
        x = self.transformer(enc_out, src_key_padding_mask=src_key_padding_mask)
        fused = self.context_gate(x)
        pooled = fused.mean(dim=1)  # [batch, d_model]

        # Return features or predictions based on mode
        if self.output_features:
            return pooled  # Return features for MultiTaskModel
        return self.head(pooled)  # Return predictions for standalone use


__all__ = ["TemporalFusionTransformer"]
