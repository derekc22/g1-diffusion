"""
Stage 1 Diffusion Model for HuggingFace Dataset: Object Motion → Hand Positions

Adapted from stage1_diffusion.py to work with compact object motion features
instead of BPS encoding. The object motion encoder is simpler since the input
is only 15D (pos + rot_6d + lin_vel + ang_vel) vs 3075D (BPS + centroid).

Architecture:
1. ObjectMotionEncoder (MLP): 15D → 256D object features
2. Stage1TransformerDenoiser: Same as original (hand_dim=6, cond_dim=256)

Both components trained jointly end-to-end with DDPM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Reuse the denoiser and helpers from the original stage 1
from models.stage1_diffusion import (
    Stage1TransformerDenoiser,
    timestep_embedding,
)


class ObjectMotionEncoder(nn.Module):
    """
    MLP encoder for object motion features.

    Encodes compact object motion representation into a feature vector.

    Input per frame:
        - object_features: (15,) = [pos(3), rot_6d(6), lin_vel(3), ang_vel(3)]

    Output per frame:
        - encoded_features: (out_dim,) - compact representation (default 256D)
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        layers = []
        dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            object_features: (B, T, input_dim) object motion features

        Returns:
            encoded: (B, T, out_dim) encoded features
        """
        return self.mlp(object_features)


class Stage1HFHandDiffusion(nn.Module):
    """
    Complete Stage 1 model for HuggingFace data:
    Object motion features → Hand positions.

    Combines:
    1. ObjectMotionEncoder: compact motion features → 256D
    2. Stage1TransformerDenoiser: diffusion denoiser for hand positions
    """

    def __init__(
        self,
        # Encoder params
        object_feature_input_dim: int = 15,
        encoder_hidden: int = 256,
        object_feature_dim: int = 256,
        encoder_layers: int = 3,
        # Denoiser params
        hand_dim: int = 6,
        d_model: int = 256,
        nhead: int = 4,
        num_transformer_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        self.hand_dim = hand_dim
        self.object_feature_dim = object_feature_dim

        # Object motion encoder (compact MLP)
        self.encoder = ObjectMotionEncoder(
            input_dim=object_feature_input_dim,
            hidden_dim=encoder_hidden,
            out_dim=object_feature_dim,
            num_layers=encoder_layers,
        )

        # Denoiser (same transformer as original stage 1)
        self.denoiser = Stage1TransformerDenoiser(
            hand_dim=hand_dim,
            cond_dim=object_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        object_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 6) noisy hand positions
            t: (B,) diffusion timesteps
            object_features: (B, T, 15) object motion features

        Returns:
            (B, T, 6) predicted clean hand positions
        """
        # Encode object features
        cond = self.encoder(object_features)  # (B, T, 256)

        # Denoise hand positions
        return self.denoiser(x, t, cond)  # (B, T, 6)


class Stage1HFHandDiffusionMLP(nn.Module):
    """
    MLP variant of Stage 1 for HuggingFace data.
    Uses MLP denoiser instead of transformer (for ablation).
    """

    def __init__(
        self,
        object_feature_input_dim: int = 15,
        encoder_hidden: int = 256,
        object_feature_dim: int = 256,
        encoder_layers: int = 3,
        hand_dim: int = 6,
        denoiser_hidden: int = 512,
        denoiser_layers: int = 4,
    ):
        super().__init__()

        self.hand_dim = hand_dim
        self.object_feature_dim = object_feature_dim

        self.encoder = ObjectMotionEncoder(
            input_dim=object_feature_input_dim,
            hidden_dim=encoder_hidden,
            out_dim=object_feature_dim,
            num_layers=encoder_layers,
        )

        # MLP denoiser
        input_dim = hand_dim + object_feature_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )

        layers = []
        dim = input_dim
        for _ in range(denoiser_layers):
            layers.append(nn.Linear(dim, denoiser_hidden))
            layers.append(nn.SiLU())
            dim = denoiser_hidden
        layers.append(nn.Linear(denoiser_hidden, hand_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        object_features: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        cond = self.encoder(object_features)
        h = torch.cat([x, cond], dim=-1)

        t_emb = timestep_embedding(t, h.shape[-1], dtype=x.dtype)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)
        h = h + t_emb

        return self.mlp(h)
