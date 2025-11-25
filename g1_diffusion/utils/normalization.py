import numpy as np
import torch
from typing import Tuple


def compute_mean_std(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    data: (N, D) or (N, T, D)
    returns mean, std over axes (N, T)
    """
    assert data.ndim in (2, 3)
    if data.ndim == 3:
        flat = data.reshape(-1, data.shape[-1])
    else:
        flat = data
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (data - mean) / std


def denormalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return data * std + mean
