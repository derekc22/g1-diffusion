from __future__ import annotations

from typing import Optional

import numpy as np
import torch


OBJECT_CONDITIONING_VARIANTS = ("variant0",)

_VARIANT_ALIASES = {
    "0": "variant0",
    "baseline": "variant0",
    "exact": "variant0",
    "full": "variant0",
    "trajectory": "variant0",
    "variant0": "variant0",
}

VARIANT_DESCRIPTIONS = {
    "variant0": "exact object trajectory",
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
            f"Use one of: {valid}. Accepted aliases: {aliases}. "
            "The old variant1/variant2 ablations were archived."
        )
    return _VARIANT_ALIASES[key]


def describe_object_conditioning_variant(variant: Optional[str]) -> str:
    canonical = normalize_object_conditioning_variant(variant)
    return f"{canonical} ({VARIANT_DESCRIPTIONS[canonical]})"


def apply_temporal_conditioning_variant(
    sequence: np.ndarray | torch.Tensor,
    variant: Optional[str],
    *,
    time_dim: int = 0,
) -> np.ndarray | torch.Tensor:
    """
    Return the exact time-varying conditioning sequence.

    The previous variant1/variant2 ablations were archived; active training and
    sampling use variant0 only.
    """
    normalize_object_conditioning_variant(variant)
    _ = time_dim
    return sequence


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
