from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from utils.contact_constraints import (
    compute_contact_metrics,
    compute_hand_jpe,
    detect_contact_frames,
    temporal_smooth_np,
)
from utils.rotation import quat_to_rot6d_xyzw, rot6d_to_mat, rot6d_to_quat_xyzw


DEFAULT_G1_XML = "/home/learning/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml"


def _parse_vec(text: Optional[str], default: Iterable[float]) -> np.ndarray:
    if text is None:
        return np.asarray(list(default), dtype=np.float64)
    return np.asarray([float(part) for part in text.split()], dtype=np.float64)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return _quat_wxyz_to_matrix(np.asarray([quat[3], quat[0], quat[1], quat[2]]))


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.asarray(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BodySpec:
    name: str
    parent: int
    pos: np.ndarray
    quat: np.ndarray
    joint_name: Optional[str]
    joint_axis: Optional[np.ndarray]


class G1Kinematics:
    """Minimal MJCF forward kinematics for the 29-DoF G1 model used here."""

    def __init__(self, xml_path: str = DEFAULT_G1_XML):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"G1 XML not found: {xml_path}")
        self.xml_path = xml_path
        self.bodies, self.joint_names = self._parse_mjcf(xml_path)
        self.body_name_to_id = {body.name: idx for idx, body in enumerate(self.bodies)}
        self.left_hand_id = self._find_body(
            ("left_rubber_hand", "left_wrist_yaw_link", "left_wrist_pitch_link")
        )
        self.right_hand_id = self._find_body(
            ("right_rubber_hand", "right_wrist_yaw_link", "right_wrist_pitch_link")
        )
        self.left_ankle_roll_id = self._find_body(("left_ankle_roll_link",))
        self.right_ankle_roll_id = self._find_body(("right_ankle_roll_link",))

    def _find_body(self, names: Tuple[str, ...]) -> int:
        for name in names:
            if name in self.body_name_to_id:
                return self.body_name_to_id[name]
        raise ValueError(f"Could not find any body from {names}")

    @staticmethod
    def _parse_mjcf(xml_path: str) -> Tuple[List[BodySpec], List[str]]:
        root = ET.parse(xml_path).getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"Missing <worldbody> in {xml_path}")

        bodies: List[BodySpec] = [
            BodySpec(
                name="world",
                parent=-1,
                pos=np.zeros(3, dtype=np.float64),
                quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                joint_name=None,
                joint_axis=None,
            )
        ]
        joint_names: List[str] = []

        def visit(elem: ET.Element, parent: int) -> None:
            if elem.tag != "body":
                return

            joint_name = None
            joint_axis = None
            for child in elem:
                if child.tag != "joint":
                    continue
                if child.get("type") == "free" or child.get("name") == "pelvis":
                    continue
                joint_name = child.get("name")
                joint_axis = _parse_vec(child.get("axis"), (0.0, 0.0, 1.0))
                joint_names.append(joint_name or f"joint_{len(joint_names)}")
                break

            idx = len(bodies)
            bodies.append(
                BodySpec(
                    name=elem.get("name") or f"body_{idx}",
                    parent=parent,
                    pos=_parse_vec(elem.get("pos"), (0.0, 0.0, 0.0)),
                    quat=_parse_vec(elem.get("quat"), (1.0, 0.0, 0.0, 0.0)),
                    joint_name=joint_name,
                    joint_axis=joint_axis,
                )
            )
            for child in elem:
                if child.tag == "body":
                    visit(child, idx)

        for child in worldbody:
            if child.tag == "body":
                visit(child, 0)

        return bodies, joint_names

    def forward_body_positions(
        self,
        root_pos: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        positions, _ = self.forward_body_transforms(root_pos, root_rot_xyzw, dof_pos)
        return positions

    def forward_body_transforms(
        self,
        root_pos: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float64)
        dof_pos = np.asarray(dof_pos, dtype=np.float64)

        if dof_pos.shape[-1] < len(self.joint_names):
            raise ValueError(
                f"Expected at least {len(self.joint_names)} DoF values, got {dof_pos.shape[-1]}"
            )

        positions = [np.zeros(3, dtype=np.float64)]
        rotations = [np.eye(3, dtype=np.float64)]
        root_rotation = quat_xyzw_to_matrix(root_rot_xyzw)
        dof_idx = 0

        for body_idx, body in enumerate(self.bodies[1:], start=1):
            parent_pos = positions[body.parent]
            parent_rot = rotations[body.parent]
            pos = parent_pos + parent_rot @ body.pos
            rot = parent_rot @ _quat_wxyz_to_matrix(body.quat)

            # MuJoCo freejoint qpos overrides the default pelvis body pose.
            if body_idx == 1:
                pos = root_pos
                rot = root_rotation

            if body.joint_axis is not None:
                rot = rot @ _axis_angle_to_matrix(body.joint_axis, float(dof_pos[dof_idx]))
                dof_idx += 1

            positions.append(pos)
            rotations.append(rot)

        return (
            np.asarray(positions, dtype=np.float32),
            np.asarray(rotations, dtype=np.float32),
        )

    def hand_positions(
        self,
        root_pos: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        root_pos = np.asarray(root_pos)
        root_rot_xyzw = np.asarray(root_rot_xyzw)
        dof_pos = np.asarray(dof_pos)

        if root_pos.ndim == 1:
            bodies = self.forward_body_positions(root_pos, root_rot_xyzw, dof_pos)
            return np.concatenate(
                [bodies[self.left_hand_id], bodies[self.right_hand_id]], axis=0
            ).astype(np.float32)

        hands = np.empty((root_pos.shape[0], 6), dtype=np.float32)
        for frame in range(root_pos.shape[0]):
            bodies = self.forward_body_positions(
                root_pos[frame],
                root_rot_xyzw[frame],
                dof_pos[frame],
            )
            hands[frame, :3] = bodies[self.left_hand_id]
            hands[frame, 3:] = bodies[self.right_hand_id]
        return hands


@lru_cache(maxsize=4)
def load_g1_kinematics(xml_path: str = DEFAULT_G1_XML) -> G1Kinematics:
    return G1Kinematics(xml_path)


def robot_hand_positions(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    xml_path: str = DEFAULT_G1_XML,
) -> np.ndarray:
    return load_g1_kinematics(xml_path).hand_positions(root_pos, root_rot_xyzw, dof_pos)


def _axis_angle_to_matrix_torch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis.to(device=angle.device, dtype=angle.dtype)
    axis = axis / axis.norm().clamp_min(1e-12)
    x, y, z = axis.unbind(dim=0)
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1.0 - c
    row0 = torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], dim=-1)
    row1 = torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], dim=-1)
    row2 = torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _quat_wxyz_to_matrix_torch(quat: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    quat_t = torch.as_tensor(quat, device=device, dtype=dtype)
    quat_t = quat_t / quat_t.norm().clamp_min(1e-12)
    w, x, y, z = quat_t.unbind(dim=0)
    return torch.stack(
        [
            torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)]),
            torch.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)]),
            torch.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]),
        ],
        dim=0,
    )


FOOT_PROXY_OFFSETS = (
    (0.08, 0.04, -0.035),
    (0.08, -0.04, -0.035),
    (-0.08, 0.04, -0.035),
    (-0.08, -0.04, -0.035),
)


def robot_body_transforms_from_state_torch(
    state: torch.Tensor,
    xml_path: str = DEFAULT_G1_XML,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Differentiable FK from Stage 2 state vectors to all body transforms.

    Args:
        state: (..., D) with [root_pos(3), root_rot_6d(6), dof_pos(...)]

    Returns:
        positions: (..., num_bodies, 3)
        rotations: (..., num_bodies, 3, 3)
        body_names: list aligned with the body dimension
    """
    if state.shape[-1] < 38:
        raise ValueError(f"Expected state dim at least 38, got {state.shape[-1]}")

    leading_shape = state.shape[:-1]
    flat = state.reshape(-1, state.shape[-1])
    root_pos = flat[:, :3]
    root_rot = rot6d_to_mat(flat[:, 3:9])
    dof_pos = flat[:, 9:]

    kin = load_g1_kinematics(xml_path)
    device = state.device
    dtype = state.dtype
    positions = [torch.zeros((flat.shape[0], 3), device=device, dtype=dtype)]
    rotations = [torch.eye(3, device=device, dtype=dtype).expand(flat.shape[0], 3, 3)]

    dof_idx = 0
    for body_idx, body in enumerate(kin.bodies[1:], start=1):
        parent_pos = positions[body.parent]
        parent_rot = rotations[body.parent]
        body_pos = torch.as_tensor(body.pos, device=device, dtype=dtype)
        body_rot = _quat_wxyz_to_matrix_torch(body.quat, device, dtype)

        pos = parent_pos + torch.matmul(parent_rot, body_pos.view(1, 3, 1)).squeeze(-1)
        rot = torch.matmul(parent_rot, body_rot.view(1, 3, 3))

        if body_idx == 1:
            pos = root_pos
            rot = root_rot

        if body.joint_axis is not None:
            joint_rot = _axis_angle_to_matrix_torch(
                torch.as_tensor(body.joint_axis, device=device, dtype=dtype),
                dof_pos[:, dof_idx],
            )
            rot = torch.matmul(rot, joint_rot)
            dof_idx += 1

        positions.append(pos)
        rotations.append(rot)

    body_pos = torch.stack(positions, dim=1).reshape(*leading_shape, len(positions), 3)
    body_rot = torch.stack(rotations, dim=1).reshape(*leading_shape, len(rotations), 3, 3)
    return body_pos, body_rot, [body.name for body in kin.bodies]


def robot_hand_positions_from_state_torch(
    state: torch.Tensor,
    xml_path: str = DEFAULT_G1_XML,
) -> torch.Tensor:
    """Differentiable FK from Stage 2 state vectors to hand link positions.

    Args:
        state: (..., D) with [root_pos(3), root_rot_6d(6), dof_pos(...)]

    Returns:
        (..., 6) [left_xyz, right_xyz]
    """
    kin = load_g1_kinematics(xml_path)
    body_pos, _, _ = robot_body_transforms_from_state_torch(state, xml_path=xml_path)
    left = body_pos[..., kin.left_hand_id, :]
    right = body_pos[..., kin.right_hand_id, :]
    return torch.cat([left, right], dim=-1)


def _foot_proxy_offsets_torch(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(FOOT_PROXY_OFFSETS, device=device, dtype=dtype)


def robot_foot_proxy_positions_from_state_torch(
    state: torch.Tensor,
    xml_path: str = DEFAULT_G1_XML,
) -> torch.Tensor:
    """Return differentiable sole proxy points as (..., 2, 4, 3)."""
    kin = load_g1_kinematics(xml_path)
    body_pos, body_rot, _ = robot_body_transforms_from_state_torch(state, xml_path=xml_path)
    offsets = _foot_proxy_offsets_torch(state.device, state.dtype)

    def transform(body_id: int) -> torch.Tensor:
        pos = body_pos[..., body_id, :]
        rot = body_rot[..., body_id, :, :]
        points = torch.matmul(rot, offsets.transpose(0, 1)).transpose(-1, -2)
        return points + pos.unsqueeze(-2)

    return torch.stack(
        [transform(kin.left_ankle_roll_id), transform(kin.right_ankle_roll_id)],
        dim=-3,
    )


def robot_foot_proxy_positions(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    xml_path: str = DEFAULT_G1_XML,
) -> np.ndarray:
    """Return sole proxy points as (T, 2, 4, 3) for left/right feet."""
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    single = root_pos.ndim == 1
    if single:
        root_pos = root_pos[None]
        root_rot_xyzw = root_rot_xyzw[None]
        dof_pos = dof_pos[None]

    kin = load_g1_kinematics(xml_path)
    offsets = np.asarray(FOOT_PROXY_OFFSETS, dtype=np.float32)
    points = np.empty((root_pos.shape[0], 2, offsets.shape[0], 3), dtype=np.float32)
    for frame in range(root_pos.shape[0]):
        body_pos, body_rot = kin.forward_body_transforms(
            root_pos[frame],
            root_rot_xyzw[frame],
            dof_pos[frame],
        )
        for foot_idx, body_id in enumerate((kin.left_ankle_roll_id, kin.right_ankle_roll_id)):
            points[frame, foot_idx] = body_pos[body_id] + offsets @ body_rot[body_id].T
    return points[0] if single else points


def nearest_surface_distances(hands: np.ndarray, object_verts: np.ndarray) -> np.ndarray:
    hands = np.asarray(hands, dtype=np.float32)
    object_verts = np.asarray(object_verts, dtype=np.float32)
    distances = []
    for frame in range(min(len(hands), len(object_verts))):
        distances.append(float(np.linalg.norm(object_verts[frame] - hands[frame, :3], axis=1).min()))
        distances.append(float(np.linalg.norm(object_verts[frame] - hands[frame, 3:], axis=1).min()))
    return np.asarray(distances, dtype=np.float64)


def compute_robot_contact_report(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    target_hands: Optional[np.ndarray] = None,
    gt_hands: Optional[np.ndarray] = None,
    object_verts: Optional[np.ndarray] = None,
    contact_threshold: float = 0.05,
    xml_path: str = DEFAULT_G1_XML,
) -> Dict[str, float]:
    robot_hands = robot_hand_positions(root_pos, root_rot_xyzw, dof_pos, xml_path=xml_path)
    T = robot_hands.shape[0]
    report: Dict[str, float] = {}

    if target_hands is not None:
        target = np.asarray(target_hands, dtype=np.float32)[:T]
        report["robot_to_target_hand_jpe_cm"] = compute_hand_jpe(robot_hands[: len(target)], target)
    if gt_hands is not None:
        gt = np.asarray(gt_hands, dtype=np.float32)[:T]
        report["robot_hand_jpe_cm"] = compute_hand_jpe(robot_hands[: len(gt)], gt)
    if object_verts is not None:
        verts = np.asarray(object_verts, dtype=np.float32)[:T]
        surface = nearest_surface_distances(robot_hands[: len(verts)], verts)
        report["robot_surface_mean_cm"] = float(surface.mean() * 100.0)
        report["robot_surface_p90_cm"] = float(np.percentile(surface, 90) * 100.0)
        if gt_hands is not None:
            gt = np.asarray(gt_hands, dtype=np.float32)[: len(verts)]
            contact = compute_contact_metrics(
                robot_hands[: len(verts)],
                gt,
                verts,
                contact_threshold=contact_threshold,
            )
            report["robot_contact_precision"] = float(contact["precision"])
            report["robot_contact_recall"] = float(contact["recall"])
            report["robot_contact_f1"] = float(contact["f1"])

    return report


def apply_robot_contact_root_correction(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    target_hands: np.ndarray,
    object_verts: Optional[np.ndarray] = None,
    contact_threshold: float = 0.06,
    activation_threshold: float = 0.12,
    max_translation: float = 0.08,
    smooth_strength: float = 0.55,
    smooth_window: int = 9,
    smooth_iterations: int = 2,
    xml_path: str = DEFAULT_G1_XML,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Translate the generated root so robot hand links match active contact anchors.

    The Stage 1 hand trajectory is the object-relative contact representation.
    Stage 2 can drift away from it, so this applies a bounded root-only
    correction. This preserves the generated joint motion while enforcing the
    shared robot/object contact anchor more directly.
    """
    root_pos = np.asarray(root_pos, dtype=np.float32)
    target_hands = np.asarray(target_hands, dtype=np.float32)
    T = min(root_pos.shape[0], target_hands.shape[0], dof_pos.shape[0], root_rot_xyzw.shape[0])
    if T == 0:
        return root_pos.copy(), {"applied": False, "reason": "empty sequence"}

    robot_hands = robot_hand_positions(
        root_pos[:T],
        root_rot_xyzw[:T],
        dof_pos[:T],
        xml_path=xml_path,
    )

    if object_verts is not None:
        left_contact, right_contact = detect_contact_frames(
            target_hands[:T],
            np.asarray(object_verts, dtype=np.float32)[:T],
            contact_threshold=activation_threshold,
        )
    else:
        left_contact = np.ones(T, dtype=bool)
        right_contact = np.ones(T, dtype=bool)

    raw_delta = np.zeros((T, 3), dtype=np.float32)
    active_counts = np.zeros(T, dtype=np.float32)
    for frame in range(T):
        deltas = []
        if left_contact[frame]:
            deltas.append(target_hands[frame, :3] - robot_hands[frame, :3])
        if right_contact[frame]:
            deltas.append(target_hands[frame, 3:] - robot_hands[frame, 3:])
        if deltas:
            raw_delta[frame] = np.mean(np.stack(deltas, axis=0), axis=0)
            active_counts[frame] = float(len(deltas))

    if max_translation > 0.0:
        norms = np.linalg.norm(raw_delta, axis=-1, keepdims=True)
        scale = np.minimum(1.0, max_translation / np.maximum(norms, 1e-8))
        raw_delta = raw_delta * scale

    delta = temporal_smooth_np(
        raw_delta,
        strength=smooth_strength,
        window=smooth_window,
        iterations=smooth_iterations,
        preserve_ends=False,
    )

    # Keep non-contact spans exactly at zero after smoothing leakage.
    inactive = active_counts <= 0.0
    delta[inactive] = 0.0

    corrected = root_pos.copy()
    corrected[:T] = corrected[:T] + delta

    corrected_hands = robot_hands + np.tile(delta, (1, 2))
    before_error = np.linalg.norm(
        robot_hands.reshape(T, 2, 3) - target_hands[:T].reshape(T, 2, 3),
        axis=-1,
    )
    after_error = np.linalg.norm(
        corrected_hands.reshape(T, 2, 3) - target_hands[:T].reshape(T, 2, 3),
        axis=-1,
    )
    active_mask = np.stack([left_contact, right_contact], axis=-1)
    if np.any(active_mask):
        before_active = before_error[active_mask]
        after_active = after_error[active_mask]
        before_mean = float(before_active.mean() * 100.0)
        after_mean = float(after_active.mean() * 100.0)
    else:
        before_mean = 0.0
        after_mean = 0.0

    metadata: Dict[str, object] = {
        "applied": True,
        "active_frames": int(np.sum(left_contact | right_contact)),
        "left_active_frames": int(np.sum(left_contact)),
        "right_active_frames": int(np.sum(right_contact)),
        "max_translation_cm": float(np.linalg.norm(delta, axis=-1).max() * 100.0),
        "mean_translation_cm": float(np.linalg.norm(delta, axis=-1).mean() * 100.0),
        "active_robot_to_target_before_cm": before_mean,
        "active_robot_to_target_after_cm": after_mean,
        "contact_threshold": float(contact_threshold),
        "activation_threshold": float(activation_threshold),
        "max_translation": float(max_translation),
    }
    return corrected, metadata


def _state_refinement_mask(
    state: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    mask = torch.zeros_like(state)
    mode = str(mode).lower()
    if mode == "root":
        mask[..., :3] = 1.0
    elif mode == "root_pose":
        mask[..., :9] = 1.0
    elif mode == "arms":
        mask[..., 9 + 15 : 9 + 29] = 1.0
    elif mode == "upper":
        mask[..., 3:9] = 1.0
        mask[..., 9 + 12 : 9 + 29] = 1.0
    elif mode in ("root_legs_upper", "stabilize_all", "root_all"):
        mask[...] = 1.0
    elif mode == "legs_upper":
        mask[..., 9 : 9 + 29] = 1.0
    elif mode == "root_legs":
        mask[..., :9] = 1.0
        mask[..., 9 : 9 + 12] = 1.0
    elif mode in ("root_upper", "all"):
        if mode == "all":
            mask[...] = 1.0
        else:
            mask[..., :9] = 1.0
            mask[..., 9 + 12 : 9 + 29] = 1.0
    else:
        mask[..., 9 + 15 : 9 + 29] = 1.0
    return mask


def _rms_np(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def _motion_state_np(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    rot6d = quat_to_rot6d_xyzw(torch.from_numpy(root_rot_xyzw.astype(np.float32))).numpy()
    return np.concatenate([root_pos, rot6d, dof_pos], axis=-1).astype(np.float32)


def _dilate_time_mask_np(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 0 or mask.shape[0] == 0 or radius <= 0:
        return mask
    dilated = mask.copy()
    for offset in range(1, radius + 1):
        dilated[offset:] |= mask[:-offset]
        dilated[:-offset] |= mask[offset:]
    return dilated


def _foot_slide_cm_np(
    feet: np.ndarray,
    floor_height: float,
    stance_height_threshold: float,
    stance_speed_threshold: float,
) -> float:
    if feet.shape[0] < 2:
        return 0.0
    centers = feet.mean(axis=2)
    z_min = feet[..., 2].min(axis=2)
    xy_delta = centers[1:, :, :2] - centers[:-1, :, :2]
    speed = np.linalg.norm(xy_delta, axis=-1)
    stance = (
        (z_min[:-1] < floor_height + stance_height_threshold)
        & (z_min[1:] < floor_height + stance_height_threshold)
        & (speed < stance_speed_threshold)
    )
    stance = _dilate_time_mask_np(stance, radius=1)
    if not np.any(stance):
        return 0.0
    return float(speed[stance].mean() * 100.0)


def compute_robot_motion_quality_report(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    floor_height: float = 0.0,
    stance_height_threshold: float = 0.06,
    stance_speed_threshold: float = 0.04,
    xml_path: str = DEFAULT_G1_XML,
) -> Dict[str, float]:
    """Compute root/body smoothness and foot-floor metrics from generated motion."""
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    T = min(root_pos.shape[0], root_rot_xyzw.shape[0], dof_pos.shape[0])
    if T == 0:
        return {
            "root_acc_rms_cm": 0.0,
            "root_jerk_rms_cm": 0.0,
            "state_acc_rms": 0.0,
            "state_jerk_rms": 0.0,
            "dof_acc_rms": 0.0,
            "dof_jerk_rms": 0.0,
            "foot_penetration_mean_cm": 0.0,
            "foot_penetration_max_cm": 0.0,
            "foot_below_floor_frac": 0.0,
            "foot_slide_cm": 0.0,
        }

    root_pos = root_pos[:T]
    root_rot_xyzw = root_rot_xyzw[:T]
    dof_pos = dof_pos[:T]
    state = _motion_state_np(root_pos, root_rot_xyzw, dof_pos)
    feet = robot_foot_proxy_positions(root_pos, root_rot_xyzw, dof_pos, xml_path=xml_path)
    foot_z = feet[..., 2]
    penetration = np.maximum(float(floor_height) - foot_z, 0.0)

    return {
        "root_acc_rms_cm": _rms_np(np.diff(root_pos, n=2, axis=0)) * 100.0 if T > 2 else 0.0,
        "root_jerk_rms_cm": _rms_np(np.diff(root_pos, n=3, axis=0)) * 100.0 if T > 3 else 0.0,
        "state_acc_rms": _rms_np(np.diff(state, n=2, axis=0)) if T > 2 else 0.0,
        "state_jerk_rms": _rms_np(np.diff(state, n=3, axis=0)) if T > 3 else 0.0,
        "dof_acc_rms": _rms_np(np.diff(dof_pos, n=2, axis=0)) if T > 2 else 0.0,
        "dof_jerk_rms": _rms_np(np.diff(dof_pos, n=3, axis=0)) if T > 3 else 0.0,
        "foot_penetration_mean_cm": float(penetration.mean() * 100.0),
        "foot_penetration_max_cm": float(penetration.max() * 100.0),
        "foot_below_floor_frac": float(np.mean(foot_z < float(floor_height))),
        "foot_slide_cm": _foot_slide_cm_np(
            feet,
            floor_height=float(floor_height),
            stance_height_threshold=float(stance_height_threshold),
            stance_speed_threshold=float(stance_speed_threshold),
        ),
    }


def _temporal_difference_torch(values: torch.Tensor, order: int) -> torch.Tensor:
    if values.shape[0] <= order:
        return values.new_zeros((0,) + values.shape[1:])
    if order == 1:
        return values[1:] - values[:-1]
    if order == 2:
        return values[2:] - 2.0 * values[1:-1] + values[:-2]
    if order == 3:
        return values[3:] - 3.0 * values[2:-1] + 3.0 * values[1:-2] - values[:-3]
    raise ValueError(f"Unsupported temporal difference order: {order}")


def _masked_temporal_loss(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    order: int,
) -> torch.Tensor:
    diff = _temporal_difference_torch(values, order)
    if diff.numel() == 0:
        return values.new_zeros(())
    if mask is None:
        return diff.pow(2).mean()
    mask_slice = mask[order:]
    return (diff.pow(2) * mask_slice).sum() / mask_slice.sum().clamp_min(1.0)


def apply_contact_preserving_motion_stabilization(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    target_hands: np.ndarray,
    object_verts: Optional[np.ndarray] = None,
    floor_height: float = 0.0,
    foot_clearance: float = 0.0,
    activation_threshold: float = 0.16,
    steps: int = 50,
    lr: float = 0.01,
    mode: str = "root_legs_upper",
    hand_weight: float = 80.0,
    floor_weight: float = 120.0,
    foot_slide_weight: float = 10.0,
    root_acc_weight: float = 2.0,
    root_jerk_weight: float = 8.0,
    state_acc_weight: float = 0.25,
    state_jerk_weight: float = 1.0,
    pose_reg_weight: float = 0.02,
    velocity_reg_weight: float = 0.05,
    max_root_delta: float = 0.08,
    max_joint_delta: float = 0.20,
    stance_height_threshold: float = 0.06,
    stance_speed_threshold: float = 0.04,
    device: str = "cuda",
    xml_path: str = DEFAULT_G1_XML,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Stabilize generated motion while preserving active FK hand contacts.

    This is an inference-time optimizer over the sampled sequence. It keeps hands
    close to the rectified Stage 1 anchors on active contact frames, adds foot
    floor constraints, and regularizes temporal acceleration/jerk against the
    generated input motion.
    """
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    target_hands = np.asarray(target_hands, dtype=np.float32)
    T = min(root_pos.shape[0], root_rot_xyzw.shape[0], dof_pos.shape[0], target_hands.shape[0])
    if T == 0 or steps <= 0 or lr <= 0.0:
        return root_pos.copy(), root_rot_xyzw.copy(), dof_pos.copy(), {
            "applied": False,
            "reason": "empty sequence or disabled",
        }

    if object_verts is not None:
        left_contact, right_contact = detect_contact_frames(
            target_hands[:T],
            np.asarray(object_verts, dtype=np.float32)[:T],
            contact_threshold=activation_threshold,
        )
    else:
        left_contact = np.ones(T, dtype=bool)
        right_contact = np.ones(T, dtype=bool)
    active = np.stack([left_contact, right_contact], axis=-1).astype(np.float32)

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    root_t = torch.from_numpy(root_pos[:T]).to(torch_device, dtype=torch.float32)
    quat_t = torch.from_numpy(root_rot_xyzw[:T]).to(torch_device, dtype=torch.float32)
    rot6d_t = quat_to_rot6d_xyzw(quat_t)
    dof_t = torch.from_numpy(dof_pos[:T]).to(torch_device, dtype=torch.float32)
    state0 = torch.cat([root_t, rot6d_t, dof_t], dim=-1)
    state = state0.detach().clone().requires_grad_(True)
    target_t = torch.from_numpy(target_hands[:T]).to(torch_device, dtype=torch.float32).view(T, 2, 3)
    active_t = torch.from_numpy(active).to(torch_device, dtype=torch.float32)
    state_mask = _state_refinement_mask(state0, mode)
    body_mask = state_mask.clone()
    body_mask[:, :3] = 0.0

    with torch.no_grad():
        feet0 = robot_foot_proxy_positions_from_state_torch(state0, xml_path=xml_path)
        z_min0 = feet0[..., 2].amin(dim=-1)
        center0 = feet0.mean(dim=-2)
        speed0 = torch.linalg.norm(center0[1:, :, :2] - center0[:-1, :, :2], dim=-1)
        stance0 = (
            (z_min0[:-1] < float(floor_height) + float(stance_height_threshold))
            & (z_min0[1:] < float(floor_height) + float(stance_height_threshold))
            & (speed0 < float(stance_speed_threshold))
        )
        stance_np = _dilate_time_mask_np(stance0.cpu().numpy(), radius=1).astype(np.float32)
        stance_t = torch.from_numpy(stance_np).to(torch_device, dtype=torch.float32)

        before_hands = robot_hand_positions_from_state_torch(state0, xml_path=xml_path).view(T, 2, 3)
        before_dist = torch.linalg.norm(before_hands - target_t, dim=-1)
        if active_t.sum() > 0:
            before_active = (before_dist * active_t).sum() / active_t.sum().clamp_min(1.0)
        else:
            before_active = before_dist.new_zeros(())

    optimizer = torch.optim.Adam([state], lr=lr)
    final_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)

        hands = robot_hand_positions_from_state_torch(state, xml_path=xml_path).view(T, 2, 3)
        if active_t.sum() > 0:
            hand_delta = hands - target_t
            hand_loss = (
                hand_delta.pow(2).sum(dim=-1) * active_t
            ).sum() / active_t.sum().clamp_min(1.0)
        else:
            hand_loss = state.new_zeros(())

        feet = robot_foot_proxy_positions_from_state_torch(state, xml_path=xml_path)
        foot_z = feet[..., 2]
        floor_loss = torch.relu(float(floor_height) + float(foot_clearance) - foot_z).pow(2).mean()
        if T > 1 and stance_t.sum() > 0:
            centers = feet.mean(dim=-2)
            slide_delta = centers[1:, :, :2] - centers[:-1, :, :2]
            foot_slide_loss = (
                slide_delta.pow(2).sum(dim=-1) * stance_t
            ).sum() / stance_t.sum().clamp_min(1.0)
        else:
            foot_slide_loss = state.new_zeros(())

        pose_reg = ((state - state0).pow(2) * state_mask).sum() / state_mask.sum().clamp_min(1.0)
        velocity_reg = _masked_temporal_loss(state - state0, state_mask, order=1)
        root_acc_loss = _masked_temporal_loss(state[:, :3], None, order=2)
        root_jerk_loss = _masked_temporal_loss(state[:, :3], None, order=3)
        state_acc_loss = _masked_temporal_loss(state, body_mask, order=2)
        state_jerk_loss = _masked_temporal_loss(state, body_mask, order=3)

        loss = (
            float(hand_weight) * hand_loss
            + float(floor_weight) * floor_loss
            + float(foot_slide_weight) * foot_slide_loss
            + float(root_acc_weight) * root_acc_loss
            + float(root_jerk_weight) * root_jerk_loss
            + float(state_acc_weight) * state_acc_loss
            + float(state_jerk_weight) * state_jerk_loss
            + float(pose_reg_weight) * pose_reg
            + float(velocity_reg_weight) * velocity_reg
        )
        final_loss = float(loss.detach().cpu())
        loss.backward()
        if state.grad is not None:
            state.grad.mul_(state_mask)
        optimizer.step()

        with torch.no_grad():
            if max_root_delta > 0.0:
                root_delta = state[:, :3] - state0[:, :3]
                norms = torch.linalg.norm(root_delta, dim=-1, keepdim=True)
                scale = torch.clamp(float(max_root_delta) / norms.clamp_min(1e-8), max=1.0)
                state[:, :3] = state0[:, :3] + root_delta * scale
            if max_joint_delta > 0.0:
                dof_delta = (state[:, 9:] - state0[:, 9:]).clamp(
                    -float(max_joint_delta),
                    float(max_joint_delta),
                )
                state[:, 9:] = state0[:, 9:] + dof_delta
            state[state_mask <= 0.0] = state0[state_mask <= 0.0]

    refined = state.detach()
    with torch.no_grad():
        after_hands = robot_hand_positions_from_state_torch(refined, xml_path=xml_path).view(T, 2, 3)
        after_dist = torch.linalg.norm(after_hands - target_t, dim=-1)
        if active_t.sum() > 0:
            after_active = (after_dist * active_t).sum() / active_t.sum().clamp_min(1.0)
        else:
            after_active = after_dist.new_zeros(())

    refined_np = refined.cpu().numpy().astype(np.float32)
    root_out = root_pos.copy()
    quat_out = root_rot_xyzw.copy()
    dof_out = dof_pos.copy()
    root_out[:T] = refined_np[:, :3]
    quat_out[:T] = rot6d_to_quat_xyzw(refined[:, 3:9]).detach().cpu().numpy().astype(np.float32)
    dof_out[:T] = refined_np[:, 9:]

    before_quality = compute_robot_motion_quality_report(
        root_pos[:T],
        root_rot_xyzw[:T],
        dof_pos[:T],
        floor_height=floor_height,
        stance_height_threshold=stance_height_threshold,
        stance_speed_threshold=stance_speed_threshold,
        xml_path=xml_path,
    )
    after_quality = compute_robot_motion_quality_report(
        root_out[:T],
        quat_out[:T],
        dof_out[:T],
        floor_height=floor_height,
        stance_height_threshold=stance_height_threshold,
        stance_speed_threshold=stance_speed_threshold,
        xml_path=xml_path,
    )

    metadata: Dict[str, object] = {
        "applied": True,
        "steps": int(steps),
        "lr": float(lr),
        "mode": str(mode),
        "active_frames": int(np.sum(left_contact | right_contact)),
        "left_active_frames": int(np.sum(left_contact)),
        "right_active_frames": int(np.sum(right_contact)),
        "active_robot_to_target_before_cm": float(before_active.detach().cpu() * 100.0),
        "active_robot_to_target_after_cm": float(after_active.detach().cpu() * 100.0),
        "final_loss": final_loss,
        "hand_weight": float(hand_weight),
        "floor_weight": float(floor_weight),
        "foot_slide_weight": float(foot_slide_weight),
        "root_acc_weight": float(root_acc_weight),
        "root_jerk_weight": float(root_jerk_weight),
        "state_acc_weight": float(state_acc_weight),
        "state_jerk_weight": float(state_jerk_weight),
        "pose_reg_weight": float(pose_reg_weight),
        "velocity_reg_weight": float(velocity_reg_weight),
        "max_root_delta": float(max_root_delta),
        "max_joint_delta": float(max_joint_delta),
        "floor_height": float(floor_height),
        "foot_clearance": float(foot_clearance),
        "stance_transition_frames": int(np.sum(stance_np > 0.0)),
        "activation_threshold": float(activation_threshold),
        "quality_before": before_quality,
        "quality_after": after_quality,
    }
    return root_out, quat_out, dof_out, metadata


def apply_robot_contact_state_refinement(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    target_hands: np.ndarray,
    object_verts: Optional[np.ndarray] = None,
    activation_threshold: float = 0.16,
    steps: int = 20,
    lr: float = 0.03,
    pose_reg_weight: float = 0.002,
    velocity_reg_weight: float = 0.02,
    acceleration_reg_weight: float = 0.0,
    max_joint_delta: float = 0.35,
    mode: str = "upper",
    device: str = "cuda",
    xml_path: str = DEFAULT_G1_XML,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Refine generated upper-body state so FK hand links meet contact anchors.

    This is a lightweight post-denoising contact refiner: it optimizes the
    generated sequence itself, not a controller, and keeps the solution close to
    the diffusion sample with pose/velocity regularization.
    """
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    target_hands = np.asarray(target_hands, dtype=np.float32)
    T = min(root_pos.shape[0], root_rot_xyzw.shape[0], dof_pos.shape[0], target_hands.shape[0])
    if T == 0 or steps <= 0 or lr <= 0.0:
        return root_pos.copy(), root_rot_xyzw.copy(), dof_pos.copy(), {
            "applied": False,
            "reason": "empty sequence or disabled",
        }

    if object_verts is not None:
        left_contact, right_contact = detect_contact_frames(
            target_hands[:T],
            np.asarray(object_verts, dtype=np.float32)[:T],
            contact_threshold=activation_threshold,
        )
    else:
        left_contact = np.ones(T, dtype=bool)
        right_contact = np.ones(T, dtype=bool)
    active = np.stack([left_contact, right_contact], axis=-1).astype(np.float32)
    if not np.any(active):
        return root_pos.copy(), root_rot_xyzw.copy(), dof_pos.copy(), {
            "applied": False,
            "reason": "no active contact anchors",
            "activation_threshold": float(activation_threshold),
        }

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    root_t = torch.from_numpy(root_pos[:T]).to(torch_device, dtype=torch.float32)
    quat_t = torch.from_numpy(root_rot_xyzw[:T]).to(torch_device, dtype=torch.float32)
    rot6d_t = quat_to_rot6d_xyzw(quat_t)
    dof_t = torch.from_numpy(dof_pos[:T]).to(torch_device, dtype=torch.float32)
    state0 = torch.cat([root_t, rot6d_t, dof_t], dim=-1)
    state = state0.detach().clone().requires_grad_(True)
    target_t = torch.from_numpy(target_hands[:T]).to(torch_device, dtype=torch.float32).view(T, 2, 3)
    active_t = torch.from_numpy(active).to(torch_device, dtype=torch.float32)
    state_mask = _state_refinement_mask(state0, mode)

    before_hands = robot_hand_positions_from_state_torch(state0.unsqueeze(0), xml_path=xml_path)[0]
    before_dist = torch.linalg.norm(before_hands.view(T, 2, 3) - target_t, dim=-1)
    before_active = (before_dist * active_t).sum() / active_t.sum().clamp_min(1.0)

    optimizer = torch.optim.Adam([state], lr=lr)
    final_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        hands = robot_hand_positions_from_state_torch(state.unsqueeze(0), xml_path=xml_path)[0].view(T, 2, 3)
        delta = hands - target_t
        contact_loss = (delta.pow(2).sum(dim=-1) * active_t).sum() / active_t.sum().clamp_min(1.0)
        pose_reg = ((state - state0).pow(2) * state_mask).sum() / state_mask.sum().clamp_min(1.0)
        if T > 1:
            vel = state[1:] - state[:-1]
            vel0 = state0[1:] - state0[:-1]
            vel_reg = ((vel - vel0).pow(2) * state_mask[1:]).sum() / state_mask[1:].sum().clamp_min(1.0)
        else:
            vel_reg = state.new_zeros(())
        if T > 2 and acceleration_reg_weight > 0.0:
            acc = state[2:] - 2.0 * state[1:-1] + state[:-2]
            acc0 = state0[2:] - 2.0 * state0[1:-1] + state0[:-2]
            acc_reg = (
                ((acc - acc0).pow(2) * state_mask[2:]).sum()
                / state_mask[2:].sum().clamp_min(1.0)
            )
        else:
            acc_reg = state.new_zeros(())
        loss = (
            contact_loss
            + pose_reg_weight * pose_reg
            + velocity_reg_weight * vel_reg
            + acceleration_reg_weight * acc_reg
        )
        final_loss = float(loss.detach().cpu())
        loss.backward()
        if state.grad is not None:
            state.grad.mul_(state_mask)
        optimizer.step()

        with torch.no_grad():
            if max_joint_delta > 0.0:
                dof_start = 9
                dof_delta = (state[:, dof_start:] - state0[:, dof_start:]).clamp(
                    -max_joint_delta,
                    max_joint_delta,
                )
                state[:, dof_start:] = state0[:, dof_start:] + dof_delta
            locked = state_mask <= 0.0
            state[locked] = state0[locked]

    refined = state.detach()
    after_hands = robot_hand_positions_from_state_torch(refined.unsqueeze(0), xml_path=xml_path)[0]
    after_dist = torch.linalg.norm(after_hands.view(T, 2, 3) - target_t, dim=-1)
    after_active = (after_dist * active_t).sum() / active_t.sum().clamp_min(1.0)

    refined_np = refined.cpu().numpy().astype(np.float32)
    root_out = root_pos.copy()
    quat_out = root_rot_xyzw.copy()
    dof_out = dof_pos.copy()
    root_out[:T] = refined_np[:, :3]
    quat_out[:T] = rot6d_to_quat_xyzw(refined[:, 3:9]).detach().cpu().numpy().astype(np.float32)
    dof_out[:T] = refined_np[:, 9:]

    metadata: Dict[str, object] = {
        "applied": True,
        "steps": int(steps),
        "lr": float(lr),
        "mode": str(mode),
        "active_frames": int(np.sum(left_contact | right_contact)),
        "left_active_frames": int(np.sum(left_contact)),
        "right_active_frames": int(np.sum(right_contact)),
        "active_robot_to_target_before_cm": float(before_active.detach().cpu() * 100.0),
        "active_robot_to_target_after_cm": float(after_active.detach().cpu() * 100.0),
        "final_loss": final_loss,
        "pose_reg_weight": float(pose_reg_weight),
        "velocity_reg_weight": float(velocity_reg_weight),
        "acceleration_reg_weight": float(acceleration_reg_weight),
        "max_joint_delta": float(max_joint_delta),
        "activation_threshold": float(activation_threshold),
    }
    return root_out, quat_out, dof_out, metadata
