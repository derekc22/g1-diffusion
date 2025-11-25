from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


class DiffusionSchedule:
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.timesteps = config.timesteps
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
