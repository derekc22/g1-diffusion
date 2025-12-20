"""
Stage 1: Hand Position Diffusion Model (OMOMO)

This module implements the first stage of the OMOMO pipeline:
Object geometry features → Hand joint positions

The MLP encoder and diffusion denoiser are trained jointly.

Architecture:
1. Object Geometry Encoder (MLP):
   - Input: [g_t, d(B_t, V_t)] where g_t is centroid (3D), d() is BPS encoding (1024x3)
   - Output: O_t (256D) per-frame object features

2. Hand Position Denoiser (Transformer):
   - Input: noisy hand positions H_t^n, object features O_t, diffusion timestep n
   - Output: denoised hand positions H_t (6D = left_xyz + right_xyz)

Reference: "Object Motion Guided Human Motion Synthesis" (Li et al., 2023)
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn


class ObjectGeometryEncoder(nn.Module):
    """
    MLP encoder for object geometry features.
    
    Encodes BPS representation + object centroid into a compact feature vector.
    The BPS representation captures local object shape relative to the centroid.
    
    Input per frame:
        - object_centroid: (3,) - g_t = mean of object vertices
        - bps_encoding: (1024, 3) - difference vectors from BPS basis to nearest vertices
    
    Output per frame:
        - object_features: (out_dim,) - compact geometry representation (default 256D)
    """
    
    def __init__(
        self,
        bps_dim: int = 1024 * 3,  # Flattened BPS encoding
        centroid_dim: int = 3,
        hidden_dim: int = 512,
        out_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.bps_dim = bps_dim
        self.centroid_dim = centroid_dim
        self.out_dim = out_dim
        
        input_dim = bps_dim + centroid_dim  # 3072 + 3 = 3075
        
        layers = []
        dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        bps_encoding: torch.Tensor,
        object_centroid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            bps_encoding: (B, T, 1024, 3) or (B, T, 3072) - BPS difference vectors
            object_centroid: (B, T, 3) - object centroid positions
        
        Returns:
            object_features: (B, T, out_dim) - encoded geometry features
        """
        B, T = object_centroid.shape[:2]
        
        # Flatten BPS if needed
        if bps_encoding.ndim == 4:
            bps_flat = bps_encoding.reshape(B, T, -1)  # (B, T, 3072)
        else:
            bps_flat = bps_encoding
        
        # Concatenate centroid and BPS
        x = torch.cat([object_centroid, bps_flat], dim=-1)  # (B, T, 3075)
        
        # Encode
        return self.mlp(x)  # (B, T, out_dim)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10000.0
) -> torch.Tensor:
    """
    Sinusoidal timestep embedding for diffusion models.
    
    Args:
        timesteps: (B,) integer timesteps
        dim: embedding dimension
        max_period: controls frequency range
    
    Returns:
        (B, dim) timestep embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SinusoidalPositionalEncoding(nn.Module):
    """Learnable-free sinusoidal positional encoding over time."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        _, T, _ = x.shape
        return x + self.pe[:, :T, :]


class Stage1TransformerDenoiser(nn.Module):
    """
    Transformer-based denoiser for hand position prediction.
    
    Follows OMOMO paper architecture:
    - 4 self-attention blocks
    - Each block: multi-head attention + position-wise feedforward
    - Noise level embedding added to input
    
    Input:
        - x: (B, T, 6) noisy hand positions [left_xyz, right_xyz]
        - t: (B,) diffusion timesteps
        - cond: (B, T, cond_dim) object geometry features from encoder
    
    Output:
        - (B, T, 6) predicted clean hand positions (x0 prediction)
    """
    
    def __init__(
        self,
        hand_dim: int = 6,  # 3D position for each hand
        cond_dim: int = 256,  # Object feature dimension from encoder
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
        
        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Transformer encoder (4 self-attention blocks)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection back to hand dimension
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
            t: (B,) diffusion timesteps
            cond: (B, T, cond_dim) object geometry features
        
        Returns:
            (B, T, 6) predicted clean hand positions
        """
        B, T, D = x.shape
        
        # Concatenate hand positions with conditions
        h = torch.cat([x, cond], dim=-1)  # (B, T, 6 + cond_dim)
        
        # Project to model dimension
        h = self.input_proj(h)  # (B, T, d_model)
        
        # Add timestep embedding
        t_emb = timestep_embedding(t, self.d_model)  # (B, d_model)
        t_emb = self.time_mlp(t_emb)  # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)
        h = h + t_emb
        
        # Add positional encoding
        h = self.pos_encoding(h)  # (B, T, d_model)
        
        # Transformer encoder
        h = self.encoder(h)  # (B, T, d_model)
        
        # Project to output
        out = self.out_proj(h)  # (B, T, 6)
        
        return out


class Stage1HandDiffusion(nn.Module):
    """
    Complete Stage 1 model: Object geometry → Hand positions.
    
    Combines:
    1. ObjectGeometryEncoder: BPS + centroid → object features (256D)
    2. Stage1TransformerDenoiser: Diffusion model for hand positions
    
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
        
        # Object geometry encoder (MLP)
        self.encoder = ObjectGeometryEncoder(
            bps_dim=bps_dim,
            centroid_dim=centroid_dim,
            hidden_dim=encoder_hidden,
            out_dim=object_feature_dim,
            num_layers=encoder_layers,
        )
        
        # Hand position denoiser (Transformer)
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
    
    def encode_object(
        self,
        bps_encoding: torch.Tensor,
        object_centroid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode object geometry into feature vectors.
        
        Args:
            bps_encoding: (B, T, 1024, 3) or (B, T, 3072)
            object_centroid: (B, T, 3)
        
        Returns:
            object_features: (B, T, object_feature_dim)
        """
        return self.encoder(bps_encoding, object_centroid)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        bps_encoding: torch.Tensor,
        object_centroid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass: encode object + denoise hands.
        
        Args:
            x: (B, T, 6) noisy hand positions
            t: (B,) diffusion timesteps
            bps_encoding: (B, T, 1024, 3) or (B, T, 3072) BPS encoding
            object_centroid: (B, T, 3) object centroid positions
        
        Returns:
            (B, T, 6) predicted clean hand positions
        """
        # Encode object geometry
        object_features = self.encode_object(bps_encoding, object_centroid)
        
        # Denoise hand positions
        return self.denoiser(x, t, object_features)


class Stage1MLPDenoiser(nn.Module):
    """
    MLP-based denoiser alternative (simpler than transformer).
    
    For ablation or faster training.
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
        """
        Args:
            x: (B, T, 6) noisy hand positions
            t: (B,) diffusion timesteps  
            cond: (B, T, cond_dim) object features
        
        Returns:
            (B, T, 6) predicted clean hand positions
        """
        B, T, _ = x.shape
        
        # Concatenate inputs
        h = torch.cat([x, cond], dim=-1)  # (B, T, hand_dim + cond_dim)
        
        # Timestep embedding
        input_dim = h.shape[-1]
        t_emb = timestep_embedding(t, input_dim)  # (B, input_dim)
        t_emb = self.time_mlp(t_emb)  # (B, input_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, input_dim)
        h = h + t_emb
        
        return self.mlp(h)


class Stage1HandDiffusionMLP(nn.Module):
    """
    Stage 1 with MLP denoiser (alternative to transformer).
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
        
        self.denoiser = Stage1MLPDenoiser(
            hand_dim=hand_dim,
            cond_dim=object_feature_dim,
            hidden_dim=denoiser_hidden,
            num_layers=denoiser_layers,
        )
    
    def encode_object(
        self,
        bps_encoding: torch.Tensor,
        object_centroid: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(bps_encoding, object_centroid)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        bps_encoding: torch.Tensor,
        object_centroid: torch.Tensor,
    ) -> torch.Tensor:
        object_features = self.encode_object(bps_encoding, object_centroid)
        return self.denoiser(x, t, object_features)
