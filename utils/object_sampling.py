import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def infer_object_name(value: Any, fallback_path: Optional[str] = None) -> str:
    text = str(value or "")
    if fallback_path and not text:
        text = os.path.basename(fallback_path)
    stem = os.path.splitext(os.path.basename(text))[0].lower()

    parts = stem.split("_")
    sub_idx = next(
        (idx for idx, part in enumerate(parts) if re.fullmatch(r"sub\d+", part)),
        None,
    )
    if sub_idx is not None and sub_idx + 1 < len(parts):
        object_parts = []
        for part in parts[sub_idx + 1 :]:
            if part.isdigit() or re.fullmatch(r"sample\d+", part):
                break
            object_parts.append(part)
        if object_parts:
            return "_".join(object_parts)

    if len(parts) >= 2 and parts[0] == "omomo":
        return parts[1]
    return stem or "unknown"


def object_name_from_data(data: Dict[str, Any], path: str) -> str:
    if data.get("object_name"):
        return infer_object_name(data["object_name"])
    if data.get("mesh_file"):
        return infer_object_name(os.path.basename(str(data["mesh_file"])))
    return infer_object_name(data.get("seq_name"), fallback_path=path)


def build_balanced_sampler(
    labels: Sequence[str],
    power: float = 1.0,
    min_count: int = 1,
    seed: Optional[int] = None,
) -> WeightedRandomSampler:
    counts = Counter(labels)
    min_count = max(int(min_count), 1)
    weights = [
        float(max(counts[label], min_count)) ** (-float(power))
        for label in labels
    ]
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def format_label_counts(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    return ", ".join(f"{name}:{count}" for name, count in sorted(counts.items()))
