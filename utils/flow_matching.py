"""
Flow Matching Utilities for G1 Diffusion Models

Implements Optimal Transport Conditional Flow Matching (OT-CFM) as described in:
- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- "Improving and Generalizing Flow-Based Generative Models" (Liu et al., 2022)

Key differences from DDPM:
- Forward process: Linear interpolation x_t = (1-t)*x_0 + t*epsilon, t in [0,1]
- Model predicts velocity field v_theta(x_t, t) ≈ dx/dt = epsilon - x_0
- Loss: ||v_theta(x_t, t) - (epsilon - x_0)||^2
- Sampling: ODE integration from t=0 (noise) to t=1 (data) using Euler or midpoint
  NOTE: We use the convention t=0 is noise and t=1 is clean data for the ODE solver,
  while the interpolation uses x_t = (1-t)*x_1 + t*x_0 where x_1 is data and x_0 is noise.

Advantages over DDPM:
- Simpler training objective (no noise schedule tuning)
- Straighter ODE paths → fewer integration steps needed
- Deterministic sampling (ODE vs SDE)
- Naturally supports variable number of inference steps
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union

import math
import torch
import torch.nn as nn
import numpy as np


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training and inference."""
    # Training
    sigma_min: float = 1e-4  # Minimum noise level (avoids singularity at t=1)
    timestep_sampling: str = "uniform"  # "uniform", "logit_normal", "beta", "cosmap"
    logit_normal_mean: float = 0.0  # Mean of logit-normal (0.0 = symmetric, negative = bias toward t=0)
    logit_normal_std: float = 1.0   # Std of logit-normal (lower = more concentrated at mean)
    beta_alpha: float = 2.0  # Beta distribution alpha (>1 shifts mass from edges)
    beta_beta: float = 2.0   # Beta distribution beta
    
    # Inference
    num_inference_steps: int = 50  # ODE integration steps
    solver: str = "euler"  # "euler", "midpoint", "rk4"


class FlowMatchingSchedule:
    """
    Optimal Transport Conditional Flow Matching schedule.
    
    Defines the interpolation path and velocity field target for training,
    plus ODE solvers for inference.
    
    Convention:
        - t ∈ [0, 1] where t=0 is clean data (x_1) and t=1 is noise (x_0)
        - Forward: x_t = (1-t)*x_1 + t*x_0  (linear interpolation from data to noise)
        - Velocity target: u_t = x_0 - x_1  (direction from data to noise)
        - Model learns: v_theta(x_t, t) ≈ u_t
        - Sampling: integrate ODE dx/dt = v_theta(x_t, t) from t=1 (noise) to t=0 (data)
    
    NOTE: For the ODE solver, we reverse time: start at t_start=1 (noise) and
    integrate to t_end=0 (data), with negative dt.
    """
    
    def __init__(self, config: FlowMatchingConfig):
        self.config = config
        self.sigma_min = config.sigma_min
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample random timesteps for training using the configured strategy.
        
        Strategies:
          - "uniform":      t ~ U[0, 1)  (baseline, equal weight everywhere)
          - "logit_normal": t = sigmoid(N(mu, sigma))  (SD3/rectified flow style,
                            concentrates samples at mid-t, tunable toward edges)
          - "beta":         t ~ Beta(alpha, beta)  (flexible, e.g. Beta(0.5, 0.5)
                            = U-shaped = more weight at edges where model is weakest)
          - "cosmap":       t = 1 - cos(u * pi/2)^2 for u ~ U[0,1]
                            (cosine schedule mapping, more weight near t=0 and t=1)
        
        Args:
            batch_size: number of timesteps to sample
            device: target device
            
        Returns:
            t: (B,) timesteps in [0, 1)
        """
        strategy = self.config.timestep_sampling
        
        if strategy == "uniform":
            return torch.rand(batch_size, device=device)
        
        elif strategy == "logit_normal":
            # Logit-normal: sample from normal, then sigmoid to [0, 1]
            # Used in Stable Diffusion 3 and recent rectified flow papers
            mu = self.config.logit_normal_mean
            sigma = self.config.logit_normal_std
            normal_samples = torch.randn(batch_size, device=device) * sigma + mu
            t = torch.sigmoid(normal_samples)
            return t
        
        elif strategy == "beta":
            # Beta distribution — very flexible:
            #   alpha=beta=1.0  → uniform
            #   alpha=beta=0.5  → U-shaped (more weight at edges — good for FM!)
            #   alpha=beta=2.0  → bell-shaped (more weight at center)
            alpha = self.config.beta_alpha
            beta_param = self.config.beta_beta
            dist = torch.distributions.Beta(alpha, beta_param)
            t = dist.sample((batch_size,)).to(device)
            return t
        
        elif strategy == "cosmap":
            # CosMap: maps uniform samples through cosine to get more
            # weight near t=0 and t=1 (the edges where FM models struggle)
            u = torch.rand(batch_size, device=device)
            t = 1.0 - torch.cos(u * math.pi / 2).pow(2)
            return t
        
        else:
            raise ValueError(f"Unknown timestep sampling strategy: '{strategy}'. "
                             f"Available: uniform, logit_normal, beta, cosmap")
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute noisy sample x_t via OT interpolation.
        
        x_t = (1-t)*x_1 + t*x_0
        
        Args:
            x1: (B, T, D) clean data
            x0: (B, T, D) noise samples (from N(0,I))
            t: (B,) timesteps in [0, 1]
            
        Returns:
            x_t: (B, T, D) interpolated samples
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1)
        t_expand = t.view(-1, 1, 1)
        return (1.0 - t_expand) * x1 + t_expand * x0
    
    def compute_velocity_target(
        self,
        x1: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the target velocity field for OT-CFM.
        
        u_t = x_0 - x_1  (constant velocity along the OT path)
        
        Args:
            x1: (B, T, D) clean data
            x0: (B, T, D) noise samples
            
        Returns:
            velocity: (B, T, D) target velocity field
        """
        return x0 - x1
    
    def training_step(
        self,
        x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method for a single training step.
        
        Samples noise, timesteps, computes interpolation and velocity target.
        
        Args:
            x1: (B, T, D) clean data
            
        Returns:
            x_t: (B, T, D) noisy interpolated samples
            t: (B,) sampled timesteps
            velocity_target: (B, T, D) target velocity
            x0: (B, T, D) noise samples
        """
        B = x1.shape[0]
        device = x1.device
        
        # Sample noise
        x0 = torch.randn_like(x1)
        
        # Sample timesteps
        t = self.sample_timesteps(B, device)
        
        # Interpolate
        x_t = self.interpolate(x1, x0, t)
        
        # Velocity target
        velocity_target = self.compute_velocity_target(x1, x0)
        
        return x_t, t, velocity_target, x0


@torch.no_grad()
def euler_solve(
    model_fn: Callable,
    x_init: torch.Tensor,
    num_steps: int = 50,
) -> torch.Tensor:
    """
    Euler ODE solver for flow matching inference.
    
    Integrates from t=1 (noise) to t=0 (data):
        x_{t-dt} = x_t + dt * v_theta(x_t, t)
    where dt = -1/num_steps (negative because we go from t=1 to t=0).
    
    Args:
        model_fn: function(x_t, t) -> velocity prediction
            t is a (B,) tensor of scalar timesteps
        x_init: (B, T, D) initial noise at t=1
        num_steps: number of Euler steps
        
    Returns:
        x_final: (B, T, D) denoised sample at t=0
    """
    B = x_init.shape[0]
    device = x_init.device
    dt = 1.0 / num_steps
    
    x = x_init
    for i in range(num_steps):
        # Current time: goes from 1.0 to dt (approaching 0)
        t_val = 1.0 - i * dt
        t = torch.full((B,), t_val, device=device, dtype=x.dtype)
        
        # Predict velocity
        v = model_fn(x, t)
        
        # Euler step: x moves from noise toward data
        # dx/dt = v, and we step by -dt (going backward in t)
        x = x - dt * v
    
    return x


@torch.no_grad()
def midpoint_solve(
    model_fn: Callable,
    x_init: torch.Tensor,
    num_steps: int = 50,
) -> torch.Tensor:
    """
    Midpoint (RK2) ODE solver for flow matching inference.
    
    More accurate than Euler with same number of model evaluations (2x per step).
    
    Integrates from t=1 (noise) to t=0 (data):
        t_mid = t - dt/2
        v_mid = v_theta(x + dt/2 * v_theta(x, t), t_mid)
        x_{t-dt} = x_t - dt * v_mid
    
    Args:
        model_fn: function(x_t, t) -> velocity prediction
        x_init: (B, T, D) initial noise
        num_steps: number of midpoint steps
        
    Returns:
        x_final: (B, T, D) denoised sample
    """
    B = x_init.shape[0]
    device = x_init.device
    dt = 1.0 / num_steps
    
    x = x_init
    for i in range(num_steps):
        t_val = 1.0 - i * dt
        t = torch.full((B,), t_val, device=device, dtype=x.dtype)
        
        # First evaluation at current point
        v1 = model_fn(x, t)
        
        # Midpoint evaluation
        t_mid_val = t_val - dt / 2
        t_mid = torch.full((B,), t_mid_val, device=device, dtype=x.dtype)
        x_mid = x - (dt / 2) * v1
        v_mid = model_fn(x_mid, t_mid)
        
        # Full step using midpoint velocity
        x = x - dt * v_mid
    
    return x


@torch.no_grad()
def rk4_solve(
    model_fn: Callable,
    x_init: torch.Tensor,
    num_steps: int = 50,
) -> torch.Tensor:
    """
    Classic RK4 ODE solver for flow matching inference.
    
    Most accurate but uses 4 model evaluations per step.
    Good for small step counts where accuracy matters.
    
    Args:
        model_fn: function(x_t, t) -> velocity prediction
        x_init: (B, T, D) initial noise
        num_steps: number of RK4 steps
        
    Returns:
        x_final: (B, T, D) denoised sample
    """
    B = x_init.shape[0]
    device = x_init.device
    dt = 1.0 / num_steps
    
    x = x_init
    for i in range(num_steps):
        t_val = 1.0 - i * dt
        t = torch.full((B,), t_val, device=device, dtype=x.dtype)
        
        # k1
        k1 = model_fn(x, t)
        
        # k2
        t2 = torch.full((B,), t_val - dt / 2, device=device, dtype=x.dtype)
        k2 = model_fn(x - (dt / 2) * k1, t2)
        
        # k3
        k3 = model_fn(x - (dt / 2) * k2, t2)
        
        # k4
        t4 = torch.full((B,), t_val - dt, device=device, dtype=x.dtype)
        k4 = model_fn(x - dt * k3, t4)
        
        # Weighted step
        x = x - (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x


def get_ode_solver(solver_name: str):
    """
    Get ODE solver function by name.
    
    Args:
        solver_name: "euler", "midpoint", or "rk4"
        
    Returns:
        Solver function with signature (model_fn, x_init, num_steps) -> x_final
    """
    solvers = {
        "euler": euler_solve,
        "midpoint": midpoint_solve,
        "rk4": rk4_solve,
    }
    if solver_name not in solvers:
        raise ValueError(f"Unknown solver '{solver_name}'. Available: {list(solvers.keys())}")
    return solvers[solver_name]


def flow_matching_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Sinusoidal embedding for continuous timesteps t ∈ [0, 1].
    
    Same as diffusion timestep embedding but accepts continuous values.
    Scaled by max_period to spread across frequency bands.
    
    Args:
        t: (B,) continuous timesteps in [0, 1]
        dim: embedding dimension
        max_period: frequency scaling
        dtype: output dtype
        
    Returns:
        (B, dim) timestep embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    # Scale t to match diffusion embedding range  
    args = (t.float() * 1000.0).unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    
    if dtype is not None:
        emb = emb.to(dtype)
    
    return emb
