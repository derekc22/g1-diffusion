from __future__ import annotations

from typing import Optional

import numpy as np
import torch


OBJECT_CONDITIONING_VARIANTS = ("variant0", "variant1", "variant2")

_VARIANT_ALIASES = {
    "0": "variant0",
    "baseline": "variant0",
    "exact": "variant0",
    "full": "variant0",
    "trajectory": "variant0",
    "variant0": "variant0",
    "1": "variant1",
    "endpoints": "variant1",
    "initial_final": "variant1",
    "start_end": "variant1",
    "variant1": "variant1",
    "2": "variant2",
    "linear": "variant2",
    "linear_interp": "variant2",
    "interpolated": "variant2",
    "variant2": "variant2",
}

VARIANT_DESCRIPTIONS = {
    "variant0": "exact object trajectory",
    "variant1": "initial frame for first half, final frame for second half",
    "variant2": "linear interpolation between initial/final object frames",
}


def normalize_object_conditioning_variant(variant: Optional[str]) -> str:
    """Return the canonical variant name."""
    if variant is None:
        return "variant0"

    key = str(variant).strip().lower().replace("-", "_")
    if key not in _VARIANT_ALIASES:
        valid = ", ".join(OBJECT_CONDITIONING_VARIANTS)
        aliases = ", ".join(sorted(_VARIANT_ALIASES))
        raise ValueError(
            f"Unknown object conditioning variant '{variant}'. "
            f"Use one of: {valid}. Accepted aliases: {aliases}."
        )
    return _VARIANT_ALIASES[key]


def describe_object_conditioning_variant(variant: Optional[str]) -> str:
    canonical = normalize_object_conditioning_variant(variant)
    return f"{canonical} ({VARIANT_DESCRIPTIONS[canonical]})"


def _torch_endpoint_hold(sequence: torch.Tensor, time_dim: int) -> torch.Tensor:
    T = sequence.shape[time_dim]
    out = torch.empty_like(sequence)
    midpoint = max(1, T // 2)

    first = sequence.narrow(time_dim, 0, 1)
    last = sequence.narrow(time_dim, T - 1, 1)

    first_slice = [slice(None)] * sequence.ndim
    first_slice[time_dim] = slice(0, midpoint)
    out[tuple(first_slice)] = first

    last_slice = [slice(None)] * sequence.ndim
    last_slice[time_dim] = slice(midpoint, T)
    out[tuple(last_slice)] = last
    return out


def _torch_linear_interpolate(sequence: torch.Tensor, time_dim: int) -> torch.Tensor:
    T = sequence.shape[time_dim]
    first = sequence.narrow(time_dim, 0, 1)
    last = sequence.narrow(time_dim, T - 1, 1)

    weight_shape = [1] * sequence.ndim
    weight_shape[time_dim] = T
    weights = torch.linspace(
        0.0,
        1.0,
        T,
        device=sequence.device,
        dtype=sequence.dtype,
    ).reshape(weight_shape)
    return first * (1.0 - weights) + last * weights


def _numpy_endpoint_hold(sequence: np.ndarray, time_dim: int) -> np.ndarray:
    T = sequence.shape[time_dim]
    out = np.empty_like(sequence)
    midpoint = max(1, T // 2)

    first = np.take(sequence, [0], axis=time_dim)
    last = np.take(sequence, [T - 1], axis=time_dim)

    first_slice = [slice(None)] * sequence.ndim
    first_slice[time_dim] = slice(0, midpoint)
    out[tuple(first_slice)] = first

    last_slice = [slice(None)] * sequence.ndim
    last_slice[time_dim] = slice(midpoint, T)
    out[tuple(last_slice)] = last
    return out


def _numpy_linear_interpolate(sequence: np.ndarray, time_dim: int) -> np.ndarray:
    T = sequence.shape[time_dim]
    first = np.take(sequence, [0], axis=time_dim)
    last = np.take(sequence, [T - 1], axis=time_dim)

    weight_shape = [1] * sequence.ndim
    weight_shape[time_dim] = T
    weight_dtype = sequence.dtype if np.issubdtype(sequence.dtype, np.floating) else np.float32
    weights = np.linspace(0.0, 1.0, T, dtype=weight_dtype).reshape(weight_shape)
    return first * (1.0 - weights) + last * weights


def apply_temporal_conditioning_variant(
    sequence: np.ndarray | torch.Tensor,
    variant: Optional[str],
    *,
    time_dim: int = 0,
) -> np.ndarray | torch.Tensor:
    """
    Replace a time-varying conditioning sequence with one of the experiment variants.

    Shapes are preserved for all variants:
    - variant0: exact sequence.
    - variant1: first-frame value for the first half of the window and last-frame
      value for the second half. This exposes only endpoint object frames.
    - variant2: elementwise linear interpolation between the first and last frames.
    """
    canonical = normalize_object_conditioning_variant(variant)
    if canonical == "variant0":
        return sequence

    if time_dim < 0:
        time_dim += sequence.ndim
    if time_dim < 0 or time_dim >= sequence.ndim:
        raise ValueError(f"time_dim={time_dim} is invalid for shape {tuple(sequence.shape)}")

    T = sequence.shape[time_dim]
    if T <= 1:
        if isinstance(sequence, torch.Tensor):
            return sequence.clone()
        return sequence.copy()

    if isinstance(sequence, torch.Tensor):
        if canonical == "variant1":
            return _torch_endpoint_hold(sequence, time_dim)
        return _torch_linear_interpolate(sequence, time_dim)

    if canonical == "variant1":
        return _numpy_endpoint_hold(sequence, time_dim)
    return _numpy_linear_interpolate(sequence, time_dim)


def apply_object_conditioning_variant(
    *,
    variant: Optional[str],
    time_dim: int = 0,
    bps_encoding: np.ndarray | torch.Tensor | None = None,
    object_centroid: np.ndarray | torch.Tensor | None = None,
    object_features: np.ndarray | torch.Tensor | None = None,
) -> dict[str, np.ndarray | torch.Tensor | None]:
    """Apply the selected variant to every object-conditioning tensor provided."""
    return {
        "bps_encoding": (
            None
            if bps_encoding is None
            else apply_temporal_conditioning_variant(bps_encoding, variant, time_dim=time_dim)
        ),
        "object_centroid": (
            None
            if object_centroid is None
            else apply_temporal_conditioning_variant(object_centroid, variant, time_dim=time_dim)
        ),
        "object_features": (
            None
            if object_features is None
            else apply_temporal_conditioning_variant(object_features, variant, time_dim=time_dim)
        ),
    }
