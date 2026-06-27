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
)
from utils.rotation import quat_to_rot6d_xyzw, rot6d_to_mat


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
