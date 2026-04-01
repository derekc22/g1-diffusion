"""
Stage 1 Flow Matching: Object Geometry → Hand Positions

Flow matching variant of Stage 1 that uses OT-CFM (Optimal Transport
Conditional Flow Matching) instead of DDPM for the generative process.

Architecture is identical to the DDPM version (same encoder, same transformer
denoiser backbone) — the only difference is in how the model is trained
and sampled:

DDPM:
    - Forward:   x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps
    - Model predicts: x_0 (clean data)
    - Loss: ||model(x_t, t) - x_0||^2
    - Sampling: 1000-step Markov chain

Flow Matching:
    - Forward:   x_t = (1-t)*x_1 + t*x_0, t ∈ [0,1]
    - Model predicts: velocity v = x_0 - x_1 (direction from data to noise)
    - Loss: ||model(x_t, t) - (x_0 - x_1)||^2
    - Sampling: ODE integration (Euler/midpoint/RK4), typically 20-100 steps

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn

# Reuse the object geometry encoder and positional encoding from the diffusion model
from models.stage1_diffusion import (
    ObjectGeometryEncoder,
    SinusoidalPositionalEncoding,
)
from utils.flow_matching import flow_matching_timestep_embedding


class Stage1FMTransformerDenoiser(nn.Module):
    """
    Transformer-based velocity predictor for flow matching.
    
    Same architecture as Stage1TransformerDenoiser but predicts velocity
    instead of clean data, and accepts continuous timesteps t ∈ [0, 1].
    
    Input:
        - x: (B, T, 6) noisy hand positions at time t
        - t: (B,) continuous timesteps in [0, 1]
        - cond: (B, T, cond_dim) object geometry features from encoder
    
    Output:
        - (B, T, 6) predicted velocity field v(x_t, t)
    """
    
    def __init__(
        self,
        hand_dim: int = 6,
        cond_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.hand_dim = hand_dim
        self.cond_dim = cond_dim
        self.d_model = d_model
        
        # Project hand positions + conditions to model dimension
        self.input_proj = nn.Linear(hand_dim + cond_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Timestep embedding MLP (continuous t ∈ [0,1])
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to velocity dimension (same as hand_dim)
        self.out_proj = nn.Linear(d_model, hand_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 6) noisy hand positions
            t: (B,) continuous timesteps in [0, 1]
            cond: (B, T, cond_dim) object geometry features
        
        Returns:
            (B, T, 6) predicted velocity field
        """
        B, T, D = x.shape
        
        # Concatenate hand positions with conditions
        h = torch.cat([x, cond], dim=-1)  # (B, T, 6 + cond_dim)
        
        # Project to model dimension
        h = self.input_proj(h)  # (B, T, d_model)
        
        # Add timestep embedding
        t_emb = flow_matching_timestep_embedding(t, self.d_model, dtype=x.dtype)
        t_emb = self.time_mlp(t_emb)  # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)
        h = h + t_emb
        
        # Add positional encoding
        h = self.pos_encoding(h)
        
        # Transformer encoder
        h = self.encoder(h)
        
        # Project to velocity output
        return self.out_proj(h)


class Stage1HandFlowMatching(nn.Module):
    """
    Complete Stage 1 Flow Matching model: Object geometry → Hand positions.
    
    Combines:
    1. ObjectGeometryEncoder: BPS + centroid → object features (256D)
       (Same encoder as DDPM version — shared architecture)
    2. Stage1FMTransformerDenoiser: Flow matching velocity predictor
    
    Both components are trained jointly end-to-end.
    """
    
    def __init__(
        self,
        # Encoder params
        bps_dim: int = 1024 * 3,
        centroid_dim: int = 3,
        encoder_hidden: int = 512,
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
        
        # Object geometry encoder (same MLP as DDPM version)
        self.encoder = ObjectGeometryEncoder(
            bps_dim=bps_dim,
            centroid_dim=centroid_dim,
            hidden_dim=encoder_hidden,
            out_dim=object_feature_dim,
            num_layers=encoder_layers,
        )
        
        # Flow matching velocity predictor (Transformer)
        self.denoiser = Stage1FMTransformerDenoiser(
            hand_dim=hand_dim,
            cond_dim=object_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
    
    def encode_object(
        self,
        bps_encoding: torch.Tensor,
        object_centroid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode object geometry into feature vectors."""
        return self.encoder(bps_encoding, object_centroid)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        bps_encoding: torch.Tensor,
        object_centroid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass: encode object + predict velocity.
        
        Args:
            x: (B, T, 6) noisy hand positions at time t
            t: (B,) continuous timesteps in [0, 1]
            bps_encoding: (B, T, 1024, 3) or (B, T, 3072)
            object_centroid: (B, T, 3)
        
        Returns:
            (B, T, 6) predicted velocity field v(x_t, t)
        """
        object_features = self.encode_object(bps_encoding, object_centroid)
        return self.denoiser(x, t, object_features)


class Stage1FMMLPDenoiser(nn.Module):
    """
    MLP-based velocity predictor (simpler alternative to transformer).
    """
    
    def __init__(
        self,
        hand_dim: int = 6,
        cond_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.hand_dim = hand_dim
        self.cond_dim = cond_dim
        
        input_dim = hand_dim + cond_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )
        
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, hand_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h = torch.cat([x, cond], dim=-1)
        
        input_dim = h.shape[-1]
        t_emb = flow_matching_timestep_embedding(t, input_dim, dtype=x.dtype)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)
        h = h + t_emb
        
        return self.mlp(h)


class Stage1HandFlowMatchingMLP(nn.Module):
    """
    Stage 1 Flow Matching with MLP velocity predictor.
    """
    
    def __init__(
        self,
        bps_dim: int = 1024 * 3,
        centroid_dim: int = 3,
        encoder_hidden: int = 512,
        object_feature_dim: int = 256,
        encoder_layers: int = 3,
        hand_dim: int = 6,
        denoiser_hidden: int = 512,
        denoiser_layers: int = 4,
    ):
        super().__init__()
        
        self.hand_dim = hand_dim
        self.object_feature_dim = object_feature_dim
        
        self.encoder = ObjectGeometryEncoder(
            bps_dim=bps_dim,
            centroid_dim=centroid_dim,
            hidden_dim=encoder_hidden,
            out_dim=object_feature_dim,
            num_layers=encoder_layers,
        )
        
        self.denoiser = Stage1FMMLPDenoiser(
            hand_dim=hand_dim,
            cond_dim=object_feature_dim,
            hidden_dim=denoiser_hidden,
            num_layers=denoiser_layers,
        )
    
    def encode_object(self, bps_encoding, object_centroid):
        return self.encoder(bps_encoding, object_centroid)
    
    def forward(self, x, t, bps_encoding, object_centroid):
        object_features = self.encode_object(bps_encoding, object_centroid)
        return self.denoiser(x, t, object_features)
