"""Hybrid LSTM-Transformer model (Phase 4.4)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base_model import BaseModel, ModelMetadata
from .lstm_module import LSTMModule
from .transformer_module import TransformerModule


class HybridModel(BaseModel):
    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        transformer_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.5,
        output_dim: int = 1,
        use_fsatten: bool = False,
        use_lstm: bool = True,
        *,
        timeframe_slices: Optional[dict[str, list[int]]] = None,
        per_timeframe_dim: int = 64,
    ) -> None:
        metadata = ModelMetadata(
            model_name="HybridLSTMTransformer",
            input_dim=input_dim,
            output_dim=output_dim,
        )
        super().__init__(metadata=metadata)
        self.use_lstm = use_lstm
        self.timeframe_slices = {k: v for k, v in (timeframe_slices or {}).items() if v}
        self.per_timeframe_dim = per_timeframe_dim
        self.use_mtf = bool(self.timeframe_slices)

        self.mtf_encoders = nn.ModuleDict()
        if self.use_mtf:
            for tf_name, idx in self.timeframe_slices.items():
                self.mtf_encoders[tf_name] = nn.Sequential(
                    nn.LayerNorm(len(idx)),
                    nn.Linear(len(idx), per_timeframe_dim),
                    nn.SiLU(),
                )
            effective_input_dim = per_timeframe_dim * len(self.timeframe_slices)
        else:
            effective_input_dim = input_dim
        self.effective_input_dim = effective_input_dim
        self.metadata.input_dim = effective_input_dim

        self.fusion_dim = (lstm_hidden if use_lstm else 0) + transformer_dim
        self.lstm_branch = (
            LSTMModule(
                input_dim=effective_input_dim,
                hidden_dim=lstm_hidden,
                num_layers=lstm_layers,
                dropout=dropout,
                output_dim=lstm_hidden,
            )
            if use_lstm
            else None
        )
        self.transformer_branch = TransformerModule(
            input_dim=effective_input_dim,
            model_dim=transformer_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            output_dim=transformer_dim,
            use_fsatten=use_fsatten,
        )
        self.dropout = dropout
        self.head = self._build_head(output_dim)

    def forward(self, inputs: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_mtf:
            tf_feats = []
            for tf_name, idx in self.timeframe_slices.items():
                slice_x = inputs[..., idx]
                tf_feats.append(self.mtf_encoders[tf_name](slice_x))
            inputs = torch.cat(tf_feats, dim=-1)

        features = []
        if self.lstm_branch is not None:
            lstm_features = self.lstm_branch(inputs)
            features.append(lstm_features)
        transformer_features = self.transformer_branch(inputs, src_key_padding_mask=src_key_padding_mask)
        features.append(transformer_features)
        fused = torch.cat(features, dim=-1)
        return self.head(fused)

    def reset_head(self, output_dim: int) -> None:
        """Rebuild only the output head to support different phase targets."""
        if output_dim != self.metadata.output_dim:
            self.metadata.output_dim = output_dim
        device = next(self.parameters()).device
        self.head = self._build_head(output_dim).to(device)

    def _build_head(self, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_dim, output_dim),
        )


__all__ = ["HybridModel"]
