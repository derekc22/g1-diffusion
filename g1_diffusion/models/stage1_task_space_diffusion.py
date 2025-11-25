from __future__ import annotations
import torch
import torch.nn as nn
import math


class Stage1TaskSpaceModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        cond_dim: int = 0,
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
        """
        x:    (B, T, state_dim)
        t:    (B,)
        cond: (B, T, cond_dim) or None
        """
        B, T, _ = x.shape
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0")
            if cond.shape[0] != B or cond.shape[1] != T:
                raise ValueError(f"cond shape {cond.shape} incompatible with x shape {x.shape}")
            h = torch.cat([x, cond], dim=-1)
        else:
            h = x

        input_dim = h.shape[-1]
        t_emb = timestep_embedding(t, input_dim).to(h.device)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)
        h = h + t_emb

        out = self.mlp(h)
        return out




class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        _, T, _ = x.shape
        return x + self.pe[:, :T, :]


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Standard sinusoidal timestep embedding used in diffusion models.

    timesteps: (B,) integer timesteps
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        # pad one dimension if dim is odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Stage1TransformerModel(nn.Module):
    """
    Transformer-based denoiser for G1 diffusion.

    Input:  x: (B, T, D)  normalized state
            t: (B,)       integer diffusion timesteps
    Output: (B, T, D)     same shape as input, predicting x0 in normalized space
    """

    def __init__(
        self,
        state_dim: int,
        cond_dim: int = 0,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
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

        # Timestep embedding -> d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project back to state dimension
        self.out_proj = nn.Linear(d_model, state_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, state_dim)
        t: (B,)
        cond: (B, T, cond_dim) or None
        returns: (B, T, state_dim)
        """
        B, T, D = x.shape
        if D != self.state_dim:
            raise ValueError("Input state_dim does not match model.state_dim")

        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0")
            if cond.shape[0] != B or cond.shape[1] != T:
                raise ValueError(f"cond shape {cond.shape} incompatible with x shape {x.shape}")
            x_in = torch.cat([x, cond], dim=-1)
        else:
            x_in = x

        # Project state (+cond) to d_model
        h = self.state_proj(x_in)  # (B, T, d_model)

        # Add timestep embedding (same for all time steps in a sequence)
        t_emb = timestep_embedding(t, self.d_model)       # (B, d_model)
        t_emb = self.time_mlp(t_emb)                      # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)      # (B, T, d_model)
        h = h + t_emb

        # Add positional encoding over time
        h = self.pos_encoding(h)  # (B, T, d_model)

        # Transformer encoder over the time dimension
        h = self.encoder(h)       # (B, T, d_model)

        # Project back to state dimension
        out = self.out_proj(h)    # (B, T, D)
        return out
