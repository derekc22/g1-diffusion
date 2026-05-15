from typing import Tuple

import numpy as np

from utils.contact_constraints import temporal_smooth_np


def smooth_body_motion_np(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    strength: float = 0.0,
    window: int = 5,
    iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strength <= 0.0:
        return root_pos, root_rot, dof_pos

    root_pos_s = temporal_smooth_np(
        root_pos,
        strength=strength,
        window=window,
        iterations=iterations,
    )
    root_rot_s = temporal_smooth_np(
        root_rot,
        strength=strength,
        window=window,
        iterations=iterations,
    )
    root_rot_s = root_rot_s / np.maximum(
        np.linalg.norm(root_rot_s, axis=-1, keepdims=True),
        1e-8,
    )
    dof_pos_s = temporal_smooth_np(
        dof_pos,
        strength=strength,
        window=window,
        iterations=iterations,
    )
    return root_pos_s, root_rot_s.astype(root_rot.dtype, copy=False), dof_pos_s
