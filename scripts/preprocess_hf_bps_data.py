"""
Preprocess HuggingFace retargeted motion data into the BPS format used by the
OMOMO-style Stage 1 model.

This is intentionally separate from preprocess_hf_data.py, which keeps the
existing compact 15D object-motion HF path unchanged.

Output PKLs are compatible with the existing BPS datasets/scripts:
  - scripts/train_stage1.py
  - scripts/sample_stage1.py
  - scripts/sample_stage2_optimized.py

Required object mesh source defaults to the local OMOMO release meshes:
  /home/learning/Documents/omomo_release/data/captured_objects
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from preprocess_hf_data import discover_motions, preprocess_motion  # noqa: E402
from utils.rotation import mat_to_quat_xyzw  # noqa: E402


DEFAULT_OBJECTS_DIR = "/home/learning/Documents/omomo_release/data/captured_objects"
DEFAULT_ROBOT_XML = "/home/learning/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml"
DEFAULT_REFERENCE_DIR = "/media/learning/DATA/export_smplx_retargeted"
ARTICULATED_OBJECTS = {"mop", "vacuum"}


class MujocoSequenceFK:
    """Compute hand positions and local body positions for HF robot motions."""

    def __init__(self, xml_path: str):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("mujoco is required unless --skip_fk is used") from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq

        self.body_names = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            self.body_names.append(name or f"body_{i}")

        self.left_hand_id = self._find_body(
            ["left_rubber_hand", "left_wrist_yaw_link", "left_wrist_pitch_link"]
        )
        self.right_hand_id = self._find_body(
            ["right_rubber_hand", "right_wrist_yaw_link", "right_wrist_pitch_link"]
        )
        print(
            f"  MuJoCo model: nq={self.nq}, nbody={self.model.nbody}, "
            f"left_hand={self.body_names[self.left_hand_id]}, "
            f"right_hand={self.body_names[self.right_hand_id]}"
        )

    def _find_body(self, candidates: List[str]) -> int:
        for name in candidates:
            body_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id >= 0:
                return body_id
        raise ValueError(f"Could not find any body from: {candidates}")

    def _set_qpos(
        self,
        root_pos: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof_pos: np.ndarray,
    ) -> None:
        qpos = np.zeros(self.nq, dtype=np.float64)
        qpos[0:3] = root_pos
        qpos[3:7] = np.array(
            [
                root_rot_xyzw[3],
                root_rot_xyzw[0],
                root_rot_xyzw[1],
                root_rot_xyzw[2],
            ],
            dtype=np.float64,
        )
        qpos[7 : 7 + dof_pos.shape[0]] = dof_pos
        self.data.qpos[:] = qpos
        self.mujoco.mj_forward(self.model, self.data)

    def compute_hand_positions(
        self,
        root_pos: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        T = root_pos.shape[0]
        hands = np.zeros((T, 6), dtype=np.float32)
        for t in range(T):
            self._set_qpos(root_pos[t], root_rot_xyzw[t], dof_pos[t])
            hands[t, 0:3] = self.data.xpos[self.left_hand_id]
            hands[t, 3:6] = self.data.xpos[self.right_hand_id]
        return hands

    def compute_local_body_positions(self, dof_pos: np.ndarray) -> np.ndarray:
        T = dof_pos.shape[0]
        local = np.zeros((T, self.model.nbody, 3), dtype=np.float32)
        root_pos = np.zeros(3, dtype=np.float32)
        root_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for t in range(T):
            self._set_qpos(root_pos, root_rot, dof_pos[t])
            local[t] = self.data.xpos
        return local


def load_obj_vertices(path: str) -> np.ndarray:
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices found in object mesh: {path}")
    return np.asarray(verts, dtype=np.float32)


def mesh_key_from_filename(filename: str) -> str:
    key = filename
    key = key.replace("_cleaned_simplified", "")
    key = key.replace("_cleaned", "")
    key = key.replace(".obj", "")
    return key


def build_mesh_map(objects_dir: str) -> Dict[str, Tuple[str, np.ndarray]]:
    objects_path = Path(objects_dir)
    if not objects_path.is_dir():
        raise FileNotFoundError(f"Object mesh directory not found: {objects_dir}")

    mesh_map = {}
    for path in sorted(objects_path.glob("*.obj")):
        key = mesh_key_from_filename(path.name)
        if key in mesh_map and "cleaned_simplified" not in path.name:
            continue
        mesh_map[key] = (str(path), load_obj_vertices(str(path)))
    return mesh_map


def object_name_candidates(metadata: Dict[str, Any], motion_name: str) -> List[str]:
    object_path = str(metadata.get("object_path", ""))
    candidates = []
    if object_path:
        path = Path(object_path)
        if path.stem:
            candidates.append(path.stem)
        if path.parent.name:
            candidates.append(path.parent.name)

    parts = motion_name.split("_")
    for part in parts:
        if not part.startswith("sub") and not part.isdigit():
            candidates.append(part)

    normalized = []
    for name in candidates:
        key = mesh_key_from_filename(name.lower())
        if key and key not in normalized:
            normalized.append(key)
    return normalized


def resolve_object_mesh(
    metadata: Dict[str, Any],
    motion_name: str,
    mesh_map: Dict[str, Tuple[str, np.ndarray]],
) -> Tuple[Optional[str], Optional[str], Optional[np.ndarray]]:
    for name in object_name_candidates(metadata, motion_name):
        if name in mesh_map:
            path, verts = mesh_map[name]
            return name, path, verts
    return None, None, None


def generate_bps_points(n: int = 1024, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < n:
        proposal = rng.uniform(-1.0, 1.0, (n * 2, 3))
        points.extend(proposal[np.linalg.norm(proposal, axis=1) <= 1.0].tolist())
    return np.asarray(points[:n], dtype=np.float32)


def load_or_create_bps_basis(
    output_dir: Path,
    bps_basis_path: Optional[str],
    n_points: int,
    seed: int,
) -> np.ndarray:
    if bps_basis_path:
        basis = np.load(bps_basis_path).astype(np.float32)
    else:
        basis = generate_bps_points(n_points, seed)

    out_path = output_dir / "bps_basis_points.npy"
    np.save(out_path, basis)
    print(f"  BPS basis: {basis.shape} saved to {out_path}")
    return basis


def normalize_quat_xyzw(quat: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape (T, 4), got {quat.shape}")
    if order == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return quat / norm


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(quat, "xyzw").astype(np.float64)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty((quat.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def transform_object_vertices(
    vertices: np.ndarray,
    object_pos: np.ndarray,
    object_rotation: np.ndarray,
    mesh_scale: float,
) -> np.ndarray:
    centered = (vertices - vertices.mean(axis=0, keepdims=True)) * mesh_scale
    rotated = np.einsum("tij,kj->tki", object_rotation, centered)
    return (rotated + object_pos[:, None, :]).astype(np.float32)


def infer_reference_mesh_scale(
    reference_dir: Optional[str],
    motion_name: str,
    vertices: np.ndarray,
) -> Optional[float]:
    if not reference_dir:
        return None

    reference_path = Path(reference_dir) / f"{motion_name}.pkl"
    if not reference_path.exists():
        return None

    with open(reference_path, "rb") as f:
        reference_data = pickle.load(f)

    if "object_mesh_scale" in reference_data:
        scale = float(reference_data["object_mesh_scale"])
        return scale if np.isfinite(scale) and scale > 0.0 else None

    object_verts = reference_data.get("object_verts")
    if object_verts is None:
        return None

    object_verts = np.asarray(object_verts, dtype=np.float64)
    if object_verts.ndim != 3 or object_verts.shape[1] != vertices.shape[0]:
        return None

    if "object_rotation" in reference_data:
        object_rotation = np.asarray(reference_data["object_rotation"], dtype=np.float64)[0]
    elif "object_rot" in reference_data:
        object_quat = normalize_quat_xyzw(reference_data["object_rot"], "xyzw")
        object_rotation = quat_xyzw_to_matrix(object_quat).astype(np.float64)[0]
    else:
        object_rotation = np.eye(3, dtype=np.float64)

    mesh_centered = vertices.astype(np.float64) - vertices.astype(np.float64).mean(
        axis=0,
        keepdims=True,
    )
    fitted = (object_rotation @ mesh_centered.T).T
    target = object_verts[0] - object_verts[0].mean(axis=0, keepdims=True)
    denom = float(np.sum(fitted * fitted))
    if denom <= 0.0:
        return None

    scale = float(np.sum(fitted * target) / denom)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def compute_bps(
    object_verts: np.ndarray,
    object_centroid: np.ndarray,
    basis: np.ndarray,
    radius: float,
) -> np.ndarray:
    T = object_verts.shape[0]
    bps = np.empty((T, basis.shape[0], 3), dtype=np.float32)
    for t in range(T):
        basis_world = basis * radius + object_centroid[t]
        tree = cKDTree(object_verts[t])
        _, idx = tree.query(basis_world)
        bps[t] = basis_world - object_verts[t, idx]
    return bps


def add_bps_geometry(
    motion_data: Dict[str, Any],
    motion_info: Dict[str, Any],
    mesh_map: Dict[str, Tuple[str, np.ndarray]],
    scale_cache: Dict[str, Optional[float]],
    bps_basis: np.ndarray,
    bps_radius: float,
    fallback_mesh_scale: float,
    reference_dir: Optional[str],
    object_quat_order: str,
    store_object_verts: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    object_name, mesh_file, vertices = resolve_object_mesh(
        motion_info.get("metadata", {}),
        motion_info["motion_name"],
        mesh_map,
    )
    if object_name is None or mesh_file is None or vertices is None:
        return None, "object mesh not found"

    if "object_pos" not in motion_data or "object_rot" not in motion_data:
        return None, "missing object_pos or object_rot"

    scale_key = f"{reference_dir or ''}:{motion_info['motion_name']}:{object_name}"
    if scale_key not in scale_cache:
        scale_cache[scale_key] = infer_reference_mesh_scale(
            reference_dir,
            motion_info["motion_name"],
            vertices,
        )
    mesh_scale = scale_cache[scale_key]
    if mesh_scale is None:
        mesh_scale = fallback_mesh_scale

    object_pos = np.asarray(motion_data["object_pos"], dtype=np.float32)
    object_quat = normalize_quat_xyzw(motion_data["object_rot"], object_quat_order)
    object_rotation = quat_xyzw_to_matrix(object_quat)
    object_verts = transform_object_vertices(
        vertices,
        object_pos,
        object_rotation,
        mesh_scale,
    )
    object_centroid = object_verts.mean(axis=1).astype(np.float32)
    bps_encoding = compute_bps(object_verts, object_centroid, bps_basis, bps_radius)

    motion_data["object_centroid"] = object_centroid
    motion_data["object_rotation"] = object_rotation.astype(np.float32)
    motion_data["bps_encoding"] = bps_encoding
    motion_data["object_name"] = object_name
    motion_data["mesh_file"] = mesh_file
    motion_data["num_verts"] = int(vertices.shape[0])
    motion_data["is_articulated"] = object_name in ARTICULATED_OBJECTS
    motion_data["object_mesh_scale"] = float(mesh_scale)
    motion_data["object_rot"] = mat_to_quat_xyzw(
        torch.from_numpy(object_rotation).float()
    ).numpy().astype(np.float32)
    motion_data["object_pos"] = object_centroid

    if store_object_verts:
        motion_data["object_verts"] = object_verts

    return motion_data, None


def process_one(
    motion_info: Dict[str, Any],
    fk_engine: Optional[MujocoSequenceFK],
    mesh_map: Dict[str, Tuple[str, np.ndarray]],
    scale_cache: Dict[str, Optional[float]],
    bps_basis: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    motion_data = preprocess_motion(motion_info, fk_engine, args.min_length)
    seq_name = f"{motion_info['dataset']}_{motion_info['motion_name']}_{motion_info['sample_name']}"
    if motion_data is None:
        return False, "base HF preprocessing failed"

    if fk_engine is not None:
        motion_data["local_body_pos"] = fk_engine.compute_local_body_positions(
            np.asarray(motion_data["dof_pos"], dtype=np.float32)
        )
        motion_data["link_body_list"] = fk_engine.body_names

    enriched, err = add_bps_geometry(
        motion_data,
        motion_info,
        mesh_map,
        scale_cache,
        bps_basis,
        args.bps_radius,
        args.mesh_scale,
        args.reference_dir,
        args.object_quat_order,
        store_object_verts=not args.no_object_verts,
    )
    if enriched is None:
        return False, err or "BPS geometry failed"

    out_path = Path(args.output_dir) / f"{seq_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(enriched, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True, str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess HF retargeted data into OMOMO-style BPS PKLs"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="./data/hf_bps_preprocessed")
    parser.add_argument("--objects_dir", default=DEFAULT_OBJECTS_DIR)
    parser.add_argument("--robot_xml", default=DEFAULT_ROBOT_XML)
    parser.add_argument("--min_length", type=int, default=30)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--num_motions", type=int, default=None)
    parser.add_argument("--skip_fk", action="store_true")
    parser.add_argument("--skip_missing_mesh", action="store_true")
    parser.add_argument("--no_object_verts", action="store_true")
    parser.add_argument("--bps_basis_path", default=None)
    parser.add_argument("--bps_points", type=int, default=1024)
    parser.add_argument("--bps_seed", type=int, default=42)
    parser.add_argument("--bps_radius", type=float, default=1.0)
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--reference_dir", default=DEFAULT_REFERENCE_DIR)
    parser.add_argument(
        "--object_quat_order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order stored in HF object_motion.npz after preprocessing",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HF BPS preprocessing")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {output_dir}")
    print(f"Objects: {args.objects_dir}")
    print(f"Reference: {args.reference_dir}")
    print(f"Store object_verts: {not args.no_object_verts}")
    print(f"Object quaternion order: {args.object_quat_order}")

    mesh_map = build_mesh_map(args.objects_dir)
    print(f"Loaded {len(mesh_map)} mesh entries")

    bps_basis = load_or_create_bps_basis(
        output_dir,
        args.bps_basis_path,
        args.bps_points,
        args.bps_seed,
    )

    fk_engine = None
    if not args.skip_fk:
        fk_engine = MujocoSequenceFK(args.robot_xml)

    motions = discover_motions(args.input_dir)
    if args.datasets:
        wanted = set(args.datasets)
        motions = [m for m in motions if m["dataset"] in wanted]
        print(f"Filtered to datasets {sorted(wanted)}: {len(motions)} motions")
    if args.num_motions is not None:
        motions = motions[: args.num_motions]
        print(f"Limited to first {len(motions)} motions")

    successful = 0
    failed = 0
    skipped = 0
    failures: Dict[str, int] = {}
    scale_cache: Dict[str, Optional[float]] = {}

    for i, motion_info in enumerate(motions):
        seq_name = (
            f"{motion_info['dataset']}_{motion_info['motion_name']}_"
            f"{motion_info['sample_name']}"
        )
        if i == 0 or (i + 1) % 10 == 0:
            print(f"\nProcessing [{i + 1}/{len(motions)}]: {seq_name}")

        ok, detail = process_one(
            motion_info,
            fk_engine,
            mesh_map,
            scale_cache,
            bps_basis,
            args,
        )
        if ok:
            successful += 1
            continue

        if args.skip_missing_mesh and detail == "object mesh not found":
            skipped += 1
        else:
            failed += 1
        failures[detail] = failures.get(detail, 0) + 1
        print(f"  Skipped/failed {seq_name}: {detail}")

    meta = {
        "num_sequences": successful,
        "failed": failed,
        "skipped": skipped,
        "input_dir": args.input_dir,
        "objects_dir": args.objects_dir,
        "reference_dir": args.reference_dir,
        "bps_points": int(bps_basis.shape[0]),
        "bps_radius": args.bps_radius,
        "object_quat_order": args.object_quat_order,
        "stored_object_verts": not args.no_object_verts,
        "inferred_reference_scales": {
            key: value for key, value in sorted(scale_cache.items()) if value is not None
        },
        "failures": failures,
    }
    with open(output_dir / "preprocessing_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("HF BPS preprocessing complete")
    print("=" * 60)
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir.resolve()}")
    if failures:
        print(f"  Failure summary: {failures}")


if __name__ == "__main__":
    main()
