from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from utils.rotation import mat_to_quat_xyzw, quat_to_rot6d_xyzw


ROBOT_STATE_DIM = 38
OBJECT_POSE_DIM = 9
ROBOT_OBJECT_STATE_DIM = ROBOT_STATE_DIM + OBJECT_POSE_DIM


def fix_length(arr: Any, length: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == length:
        return arr
    if arr.shape[0] > length:
        return arr[:length]
    if arr.shape[0] == 0:
        raise ValueError(f"{name} has zero frames")
    pad = np.repeat(arr[-1:], length - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def rotation_to_6d(rot: Any, length: int, name: str) -> np.ndarray:
    rot_arr = fix_length(np.asarray(rot, dtype=np.float32), length, name)
    if rot_arr.ndim == 2 and rot_arr.shape[1] == 4:
        return quat_to_rot6d_xyzw(torch.from_numpy(rot_arr).float()).numpy()
    if rot_arr.ndim == 2 and rot_arr.shape[1] == 6:
        return rot_arr
    if rot_arr.ndim == 2 and rot_arr.shape[1] == 9:
        rot_arr = rot_arr.reshape(length, 3, 3)
    if rot_arr.ndim == 3 and rot_arr.shape[1:] == (3, 3):
        quat = mat_to_quat_xyzw(torch.from_numpy(rot_arr).float())
        return quat_to_rot6d_xyzw(quat).numpy()
    raise ValueError(f"{name} has unsupported rotation shape {rot_arr.shape}")


def object_pose_from_data(data: Dict[str, Any], length: int | None = None) -> np.ndarray:
    if "object_pos" in data and data["object_pos"] is not None:
        object_pos_raw = data["object_pos"]
    elif "object_centroid" in data and data["object_centroid"] is not None:
        object_pos_raw = data["object_centroid"]
    else:
        raise KeyError("missing object_pos/object_centroid")

    if "object_rot" in data and data["object_rot"] is not None:
        object_rot_raw = data["object_rot"]
    elif "object_rotation" in data and data["object_rotation"] is not None:
        object_rot_raw = data["object_rotation"]
    else:
        raise KeyError("missing object_rot/object_rotation")

    pos_arr = np.asarray(object_pos_raw, dtype=np.float32)
    rot_arr = np.asarray(object_rot_raw, dtype=np.float32)
    T = int(length) if length is not None else min(pos_arr.shape[0], rot_arr.shape[0])
    object_pos = fix_length(pos_arr, T, "object_pos")
    object_rot_6d = rotation_to_6d(rot_arr, T, "object_rot")
    return np.concatenate([object_pos, object_rot_6d], axis=-1).astype(np.float32)


def robot_state_from_data(data: Dict[str, Any], length: int | None = None) -> np.ndarray:
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in data:
            raise KeyError(f"missing {key}")

    root_pos_raw = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_raw = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos_raw = np.asarray(data["dof_pos"], dtype=np.float32)
    T = int(length) if length is not None else min(
        root_pos_raw.shape[0],
        root_rot_raw.shape[0],
        dof_pos_raw.shape[0],
    )
    root_pos = fix_length(root_pos_raw, T, "root_pos")
    root_rot_6d = rotation_to_6d(root_rot_raw, T, "root_rot")
    dof_pos = fix_length(dof_pos_raw, T, "dof_pos")
    state = np.concatenate([root_pos, root_rot_6d, dof_pos], axis=-1).astype(np.float32)
    if state.shape[-1] != ROBOT_STATE_DIM:
        raise ValueError(f"expected robot state dim {ROBOT_STATE_DIM}, got {state.shape[-1]}")
    return state


def robot_object_motion_from_data(data: Dict[str, Any], length: int | None = None) -> np.ndarray:
    robot = robot_state_from_data(data, length)
    object_pose = object_pose_from_data(data, robot.shape[0])
    motion = np.concatenate([robot, object_pose], axis=-1).astype(np.float32)
    if motion.shape[-1] != ROBOT_OBJECT_STATE_DIM:
        raise ValueError(
            f"expected robot-object state dim {ROBOT_OBJECT_STATE_DIM}, got {motion.shape[-1]}"
        )
    return motion


def robot_object_layout() -> Dict[str, Any]:
    return {
        "motion_dim": ROBOT_OBJECT_STATE_DIM,
        "robot_state_dim": ROBOT_STATE_DIM,
        "object_pose_dim": OBJECT_POSE_DIM,
        "robot_state": [0, ROBOT_STATE_DIM],
        "object_pose": [ROBOT_STATE_DIM, ROBOT_OBJECT_STATE_DIM],
        "robot_state_fields": {
            "root_pos": [0, 3],
            "root_rot_6d": [3, 9],
            "dof_pos": [9, ROBOT_STATE_DIM],
        },
        "object_pose_fields": {
            "object_pos": [ROBOT_STATE_DIM, ROBOT_STATE_DIM + 3],
            "object_rot_6d": [ROBOT_STATE_DIM + 3, ROBOT_OBJECT_STATE_DIM],
        },
        "goal_dim": OBJECT_POSE_DIM,
    }
