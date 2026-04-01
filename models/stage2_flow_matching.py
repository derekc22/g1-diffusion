"""
Stage 2 Flow Matching: Hand Positions → Full-Body Motion

Flow matching variant of Stage 2 that generates full-body robot poses
conditioned on (rectified) hand positions, using OT-CFM instead of DDPM.

Architecture is identical to the DDPM version (same transformer backbone) —
only the training objective and sampling procedure differ.

DDPM Stage 2:
    - Forward:   x_t = sqrt(alpha_bar_t)*state + sqrt(1-alpha_bar_t)*eps
    - Model predicts: x_0 (clean state)
    - Loss: ||model(x_t, t, hands) - state||^2
    - Sampling: 1000-step reverse chain

Flow Matching Stage 2:
    - Forward:   x_t = (1-t)*x_1 + t*x_0, t ∈ [0,1]
    - Model predicts: velocity v = x_0 - x_1
    - Loss: ||model(x_t, t, hands) - (x_0 - x_1)||^2
    - Sampling: ODE integration, typically 20-100 steps
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn

# Reuse positional encoding from the diffusion model
from models.stage2_diffusion import SinusoidalPositionalEncoding
from utils.flow_matching import flow_matching_timestep_embedding


class Stage2FMTransformerModel(nn.Module):
    """
    Transformer-based velocity predictor for Stage 2 flow matching.
    
    Same architecture as Stage2TransformerModel but predicts velocity
    field and accepts continuous timesteps t ∈ [0, 1].
    
    Input:
        - x: (B, T, state_dim) noisy state at time t
        - t: (B,) continuous timesteps in [0, 1]
        - cond: (B, T, cond_dim) hand position conditioning
    
    Output:
        - (B, T, state_dim) predicted velocity field v(x_t, t)
    """
    
    def __init__(
        self,
        state_dim: int,
        cond_dim: int = 6,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.d_model = d_model
        
        input_dim = state_dim + cond_dim
        
        # Project per-frame state (+cond) to model dimension
        self.state_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding over time
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        
        # Timestep embedding (continuous t ∈ [0,1])
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
        
        # Project back to state dimension (velocity output)
        self.out_proj = nn.Linear(d_model, state_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, state_dim) noisy state
            t: (B,) continuous timesteps in [0, 1]
            cond: (B, T, cond_dim) hand position conditioning
            
        Returns:
            (B, T, state_dim) predicted velocity field
        """
        B, T, D = x.shape
        if D != self.state_dim:
            raise ValueError(f"Input state_dim {D} does not match model.state_dim {self.state_dim}")
        
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0")
            if cond.shape[0] != B or cond.shape[1] != T:
                raise ValueError(f"cond shape {cond.shape} incompatible with x shape {x.shape}")
            x_in = torch.cat([x, cond], dim=-1)
        else:
            x_in = x
        
        # Project to d_model
        h = self.state_proj(x_in)  # (B, T, d_model)
        
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


class Stage2FMMLPModel(nn.Module):
    """
    MLP-based velocity predictor for Stage 2 flow matching (ablation).
    """
    
    def __init__(
        self,
        state_dim: int,
        cond_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        input_dim = state_dim + cond_dim
        
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
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0")
            h = torch.cat([x, cond], dim=-1)
        else:
            h = x
        
        input_dim = h.shape[-1]
        t_emb = flow_matching_timestep_embedding(t, input_dim, dtype=h.dtype)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)
        h = h + t_emb
        
        return self.mlp(h)
