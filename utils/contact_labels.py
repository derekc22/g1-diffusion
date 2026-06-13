from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is optional at import time
    cKDTree = None

from utils.robot_kinematics import robot_foot_proxy_positions


CONTACT_ORDER = ("LH_object", "RH_object", "LF_floor", "RF_floor")
CONTACT_MODE_NONE = 0
CONTACT_MODE_STICK = 1
CONTACT_MODE_SLIDE = 2


def _soft_from_distance(distance: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-(distance.astype(np.float32) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


def _nearest_vertices(points: np.ndarray, verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    verts = np.asarray(verts, dtype=np.float32)
    if verts.shape[0] == 0:
        return np.zeros_like(points), np.full(points.shape[0], np.inf, dtype=np.float32)
    if cKDTree is not None:
        dists, idx = cKDTree(verts).query(points)
        return verts[np.asarray(idx, dtype=np.int64)].astype(np.float32), np.asarray(dists, dtype=np.float32)
    diff = verts[None, :, :] - points[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    idx = np.argmin(dists, axis=1)
    return verts[idx].astype(np.float32), dists[np.arange(points.shape[0]), idx].astype(np.float32)


def _classify_modes(
    soft: np.ndarray,
    relative_speed: np.ndarray,
    contact_eps: float,
    stick_speed_threshold: float,
) -> np.ndarray:
    mode = np.zeros_like(soft, dtype=np.int64)
    active = soft >= float(contact_eps)
    mode[active & (relative_speed < float(stick_speed_threshold))] = CONTACT_MODE_STICK
    mode[active & (relative_speed >= float(stick_speed_threshold))] = CONTACT_MODE_SLIDE
    return mode


def compute_fixed_contact_labels(
    hand_positions: np.ndarray,
    object_verts: Optional[np.ndarray] = None,
    root_pos: Optional[np.ndarray] = None,
    root_rot_xyzw: Optional[np.ndarray] = None,
    dof_pos: Optional[np.ndarray] = None,
    floor_height: float = 0.0,
    object_contact_sigma: float = 0.05,
    floor_contact_sigma: float = 0.04,
    contact_eps: float = 0.2,
    stick_speed_threshold: float = 0.04,
) -> Dict[str, np.ndarray]:
    """Build fixed-size contact labels ordered as LH, RH, LF, RF.

    Hands use nearest moving object vertices. Feet use G1 sole proxy points and a
    static floor plane. Variable-size geometry never appears in the returned
    dict, so all arrays can be batched directly by a DataLoader.
    """
    hand_positions = np.asarray(hand_positions, dtype=np.float32)
    T = hand_positions.shape[0]

    contact_soft = np.zeros((T, 4), dtype=np.float32)
    contact_anchor_world = np.zeros((T, 4, 3), dtype=np.float32)
    contact_mode = np.zeros((T, 4), dtype=np.int64)
    contact_available = np.zeros((T, 4), dtype=np.float32)

    if object_verts is not None and T > 0:
        verts = np.asarray(object_verts, dtype=np.float32)
        if verts.ndim == 2:
            verts = np.repeat(verts[None], T, axis=0)
        frame_count = min(T, verts.shape[0])
        for t in range(frame_count):
            anchors, dists = _nearest_vertices(
                hand_positions[t].reshape(2, 3),
                verts[t],
            )
            contact_anchor_world[t, :2] = anchors
            contact_soft[t, :2] = _soft_from_distance(dists, object_contact_sigma)
            contact_available[t, :2] = np.isfinite(dists).astype(np.float32)

        hand_vel = np.zeros((T, 2, 3), dtype=np.float32)
        anchor_vel = np.zeros((T, 2, 3), dtype=np.float32)
        if T > 1:
            hand_vel[1:] = hand_positions[1:].reshape(-1, 2, 3) - hand_positions[:-1].reshape(-1, 2, 3)
            anchor_vel[1:] = contact_anchor_world[1:, :2] - contact_anchor_world[:-1, :2]
        rel_speed = np.linalg.norm(hand_vel - anchor_vel, axis=-1)
        contact_mode[:, :2] = _classify_modes(
            contact_soft[:, :2],
            rel_speed,
            contact_eps=contact_eps,
            stick_speed_threshold=stick_speed_threshold,
        )

    if root_pos is not None and root_rot_xyzw is not None and dof_pos is not None and T > 0:
        try:
            root = np.asarray(root_pos, dtype=np.float32)[:T]
            quat = np.asarray(root_rot_xyzw, dtype=np.float32)[:T]
            dof = np.asarray(dof_pos, dtype=np.float32)[:T]
            frame_count = min(T, root.shape[0], quat.shape[0], dof.shape[0])
            feet = robot_foot_proxy_positions(root[:frame_count], quat[:frame_count], dof[:frame_count])
            centers = feet.mean(axis=2)
            z_min = feet[..., 2].min(axis=2)
            floor_distance = np.abs(z_min - float(floor_height)).astype(np.float32)
            contact_soft[:frame_count, 2:4] = _soft_from_distance(
                floor_distance,
                floor_contact_sigma,
            )
            contact_available[:frame_count, 2:4] = 1.0
            contact_anchor_world[:frame_count, 2:4, :2] = centers[:, :, :2]
            contact_anchor_world[:frame_count, 2:4, 2] = float(floor_height)

            speed = np.zeros((frame_count, 2), dtype=np.float32)
            if frame_count > 1:
                speed[1:] = np.linalg.norm(
                    centers[1:, :, :2] - centers[:-1, :, :2],
                    axis=-1,
                )
            contact_mode[:frame_count, 2:4] = _classify_modes(
                contact_soft[:frame_count, 2:4],
                speed,
                contact_eps=contact_eps,
                stick_speed_threshold=stick_speed_threshold,
            )
        except Exception:
            # Foot labels are useful but should never make dataset loading fail.
            pass

    return {
        "contact_soft": contact_soft,
        "contact_anchor_world": contact_anchor_world,
        "contact_mode": contact_mode,
        "contact_available": contact_available,
    }
