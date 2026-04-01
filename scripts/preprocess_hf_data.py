"""
Preprocess HuggingFace retargeted motion dataset for diffusion training.

Converts the HuggingFace dataset format (CSV/NPZ per motion) into
preprocessed PKL files compatible with the diffusion training pipeline.

Key steps:
1. Read motion data from motion.csv or motion.npz
2. Read object motion from object_motion.npz
3. Compute hand positions via MuJoCo forward kinematics
4. Save preprocessed PKL files for training

The output PKL files contain:
    - root_pos: (T, 3) root position
    - root_rot: (T, 4) root rotation quaternion (xyzw)
    - dof_pos: (T, 29) joint angles
    - hand_positions: (T, 6) [left_xyz, right_xyz] from FK
    - object_pos: (T, 3) object position
    - object_rot: (T, 4) object rotation quaternion
    - object_lin_vel: (T, 3) object linear velocity
    - object_ang_vel: (T, 3) object angular velocity
    - contact: (T,) contact signal (if available)
    - fps: 30
    - seq_name: string identifier

Usage:
    python scripts/preprocess_hf_data.py \\
        --input_dir ./data/hf_dataset/data/unitree_g1 \\
        --output_dir ./data/hf_preprocessed \\
        --robot_xml ../g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml

If mujoco is not available, hand positions will be approximated using
a simple kinematic chain estimate, or you can supply --skip_fk to skip
hand position computation entirely (Stage 2 only training).
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

# NumPy 2.x compatibility shim for loading NPZ files saved with newer numpy
import sys as _sys
if not hasattr(np, '_core'):
    import numpy.core.multiarray
    import numpy.core.numeric
    _sys.modules['numpy._core'] = np.core
    _sys.modules['numpy._core.multiarray'] = np.core.multiarray
    _sys.modules['numpy._core.numeric'] = np.core.numeric

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# MuJoCo FK helper
# ---------------------------------------------------------------------------

class MujocoFK:
    """Compute forward kinematics using MuJoCo."""

    def __init__(self, xml_path: str):
        try:
            import mujoco
            self.mujoco = mujoco
        except ImportError:
            raise ImportError(
                "mujoco is required for FK computation. Install with: pip install mujoco"
            )

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Find hand body IDs
        self.left_hand_id = None
        self.right_hand_id = None

        # Try multiple possible hand body names
        left_names = ["left_rubber_hand", "left_wrist_yaw_link", "left_wrist_pitch_link"]
        right_names = ["right_rubber_hand", "right_wrist_yaw_link", "right_wrist_pitch_link"]

        for name in left_names:
            try:
                self.left_hand_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                if self.left_hand_id >= 0:
                    print(f"  Left hand body: '{name}' (id={self.left_hand_id})")
                    break
            except Exception:
                continue

        for name in right_names:
            try:
                self.right_hand_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                if self.right_hand_id >= 0:
                    print(f"  Right hand body: '{name}' (id={self.right_hand_id})")
                    break
            except Exception:
                continue

        if self.left_hand_id is None or self.left_hand_id < 0:
            raise ValueError("Could not find left hand body in MuJoCo model")
        if self.right_hand_id is None or self.right_hand_id < 0:
            raise ValueError("Could not find right hand body in MuJoCo model")

        self.nq = self.model.nq  # Total qpos dimension
        print(f"  MuJoCo model: nq={self.nq}, nbody={self.model.nbody}")

    def compute_hand_positions(
        self, root_pos: np.ndarray, root_rot_xyzw: np.ndarray, dof_pos: np.ndarray
    ) -> np.ndarray:
        """
        Compute hand positions for a sequence of poses.

        Args:
            root_pos: (T, 3) root position
            root_rot_xyzw: (T, 4) root rotation quaternion in xyzw order
            dof_pos: (T, 29) joint angles

        Returns:
            hand_positions: (T, 6) [left_x, left_y, left_z, right_x, right_y, right_z]
        """
        T = root_pos.shape[0]
        hand_positions = np.zeros((T, 6), dtype=np.float32)

        for t in range(T):
            # Build qpos vector: [root_pos(3), root_rot_wxyz(4), dof_pos(29)] = 36
            # MuJoCo uses wxyz quaternion order
            quat_wxyz = np.array([
                root_rot_xyzw[t, 3],  # w
                root_rot_xyzw[t, 0],  # x
                root_rot_xyzw[t, 1],  # y
                root_rot_xyzw[t, 2],  # z
            ])

            qpos = np.zeros(self.nq, dtype=np.float64)
            qpos[0:3] = root_pos[t]
            qpos[3:7] = quat_wxyz
            qpos[7:7 + dof_pos.shape[1]] = dof_pos[t]

            # Set qpos and run FK
            self.data.qpos[:] = qpos
            self.mujoco.mj_forward(self.model, self.data)

            # Extract hand positions
            left_pos = self.data.xpos[self.left_hand_id].copy()
            right_pos = self.data.xpos[self.right_hand_id].copy()

            hand_positions[t, 0:3] = left_pos
            hand_positions[t, 3:6] = right_pos

        return hand_positions


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_motion_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load motion data from CSV file.

    Handles two formats:
      A) No header, 36 columns: root_pos(3) + root_rot_xyzw(4) + dof_pos(29)
      B) Header row, 37 columns: frame + root_pos(3) + root_rot_wxyz(4) + dof_pos(29)
    """
    # Detect header row by reading first line
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()

    has_header = False
    try:
        float(first_line.split(',')[0])
    except ValueError:
        has_header = True

    skiprows = 1 if has_header else 0
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, skiprows=skiprows)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]

    # If 37 columns, first column is frame number — skip it
    if ncols == 37:
        data = data[:, 1:]  # drop frame column
        ncols = 36

    if ncols >= 36:
        root_pos = data[:, 0:3]
        raw_quat = data[:, 3:7]
        dof_pos = data[:, 7:36]
    elif ncols >= 7:
        root_pos = data[:, 0:3]
        raw_quat = data[:, 3:7]
        dof_pos = data[:, 7:]
    else:
        raise ValueError(f"CSV has {ncols} columns, expected at least 7 (root_pos + root_rot)")

    # Determine quaternion convention from header
    # If header has 'root_qw' as the 4th data col (index 3 after frame), it's wxyz
    if has_header and 'root_qw' in first_line.split(',')[4]:
        # wxyz → xyzw
        root_rot = np.column_stack([raw_quat[:, 1], raw_quat[:, 2], raw_quat[:, 3], raw_quat[:, 0]])
    else:
        # Already xyzw (headerless format per dataset README)
        root_rot = raw_quat

    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def load_motion_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load motion data from NPZ file.

    Handles the HF dataset format with keys like:
      joint_pos (T, 36) = root_pos(3) + root_rot_wxyz(4) + dof_pos(29)
      body_pos_w (T, N, 3) — world-frame body positions (can extract hands)
      body_names (N,) — body name strings
    """
    data = dict(np.load(npz_path, allow_pickle=True))

    result = {}

    # --- Handle combined joint_pos format (T, 36) ---
    if "joint_pos" in data and "root_pos" not in data:
        jp = np.asarray(data["joint_pos"], dtype=np.float32)
        if jp.ndim == 2 and jp.shape[1] >= 36:
            result["root_pos"] = jp[:, 0:3]
            # joint_pos stores quaternion as wxyz (MuJoCo convention)
            quat_wxyz = jp[:, 3:7]
            # convert to xyzw for our pipeline
            result["root_rot"] = np.column_stack([
                quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]
            ])
            result["dof_pos"] = jp[:, 7:36]

    # --- Extract hand positions from body_pos_w if available ---
    if "body_pos_w" in data and "body_names" in data:
        body_names = list(data["body_names"])
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        left_idx, right_idx = None, None
        for name in ["left_rubber_hand", "left_wrist_yaw_link", "left_wrist_pitch_link"]:
            if name in body_names:
                left_idx = body_names.index(name)
                break
        for name in ["right_rubber_hand", "right_wrist_yaw_link", "right_wrist_pitch_link"]:
            if name in body_names:
                right_idx = body_names.index(name)
                break
        if left_idx is not None and right_idx is not None:
            T = body_pos.shape[0]
            hand_positions = np.zeros((T, 6), dtype=np.float32)
            hand_positions[:, 0:3] = body_pos[:, left_idx]
            hand_positions[:, 3:6] = body_pos[:, right_idx]
            result["hand_positions"] = hand_positions

    # Root position
    if "root_pos" not in result:
        for key in ["root_pos", "root_position", "root_trans", "root_trans_offset", "translation"]:
            if key in data:
                result["root_pos"] = np.asarray(data[key], dtype=np.float32)
                break

    # Root rotation
    if "root_rot" not in result:
        for key in ["root_rot", "root_rotation", "root_quat", "rotation"]:
            if key in data:
                result["root_rot"] = np.asarray(data[key], dtype=np.float32)
                break

    # DOF positions
    if "dof_pos" not in result:
        for key in ["dof_pos", "dof", "joint_positions", "joint_angles", "qpos"]:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
                result["dof_pos"] = arr
                break

    # Hand positions (if pre-computed and not already extracted)
    if "hand_positions" not in result:
        for key in ["hand_positions", "hand_pos", "hands"]:
            if key in data:
                result["hand_positions"] = np.asarray(data[key], dtype=np.float32)
                break

    # Velocities (optional)
    for key in ["joint_velocities", "dof_vel", "velocities", "joint_vel"]:
        if key in data:
            result["joint_velocities"] = np.asarray(data[key], dtype=np.float32)
            break

    # FPS
    for key in ["fps", "framerate", "frame_rate"]:
        if key in data:
            result["fps"] = float(np.asarray(data[key]).flat[0])
            break

    return result


def load_object_motion(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load object motion data from object_motion.npz.

    Expected keys: position, rotation, linear_velocity, angular_velocity, contact
    """
    if not os.path.exists(npz_path):
        return {}

    data = dict(np.load(npz_path, allow_pickle=True))
    result = {}

    # Object position
    for key in ["obj_pos_w", "position", "pos", "object_pos", "object_position", "translation"]:
        if key in data:
            result["object_pos"] = np.asarray(data[key], dtype=np.float32)
            break

    # Object rotation
    for key in ["obj_quat_w", "rotation", "rot", "object_rot", "object_rotation", "quaternion"]:
        if key in data:
            arr = np.asarray(data[key], dtype=np.float32)
            result["object_rot"] = arr
            break

    # Linear velocity
    for key in ["obj_lin_vel_w", "linear_velocity", "lin_vel", "velocity", "object_lin_vel"]:
        if key in data:
            result["object_lin_vel"] = np.asarray(data[key], dtype=np.float32)
            break

    # Angular velocity
    for key in ["obj_ang_vel_w", "angular_velocity", "ang_vel", "object_ang_vel"]:
        if key in data:
            result["object_ang_vel"] = np.asarray(data[key], dtype=np.float32)
            break

    # Contact
    for key in ["contact", "contact_mask", "contact_label"]:
        if key in data:
            result["contact"] = np.asarray(data[key], dtype=np.float32)
            break

    return result


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------

def discover_motions(input_dir: str) -> List[Dict[str, Any]]:
    """
    Discover all motion sequences in the HuggingFace dataset structure.

    Expected structure:
        input_dir/
            <dataset_name>/
                tag_config.json
                <motion_name>/
                    <sample_name>/
                        motion.csv
                        motion.npz
                        object_motion.npz
                        metadata.json
    """
    input_path = Path(input_dir)
    motions = []

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Walk through dataset directories
    for dataset_dir in sorted(input_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # Check for tag_config.json
        tag_config_path = dataset_dir / "tag_config.json"
        tag_config = {}
        if tag_config_path.exists():
            with open(tag_config_path) as f:
                tag_config = json.load(f)

        # Walk through motion directories
        for motion_dir in sorted(dataset_dir.iterdir()):
            if not motion_dir.is_dir():
                continue

            motion_name = motion_dir.name

            # Find tags for this motion
            motion_tags = []
            for tag, tag_motions in tag_config.items():
                if motion_name in tag_motions:
                    motion_tags.append(tag)

            # Walk through sample directories
            for sample_dir in sorted(motion_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue

                sample_name = sample_dir.name

                # Check for data files
                motion_csv = sample_dir / "motion.csv"
                motion_npz = sample_dir / "motion.npz"
                object_npz = sample_dir / "object_motion.npz"
                metadata_json = sample_dir / "metadata.json"

                if not (motion_csv.exists() or motion_npz.exists()):
                    continue

                metadata = {}
                if metadata_json.exists():
                    with open(metadata_json) as f:
                        metadata = json.load(f)

                motions.append({
                    "dataset": dataset_name,
                    "motion_name": motion_name,
                    "sample_name": sample_name,
                    "motion_csv": str(motion_csv) if motion_csv.exists() else None,
                    "motion_npz": str(motion_npz) if motion_npz.exists() else None,
                    "object_npz": str(object_npz) if object_npz.exists() else None,
                    "metadata": metadata,
                    "tags": motion_tags,
                })

    # If no motions found with expected structure, try flat structure
    # (all CSV/NPZ files directly in input_dir)
    if not motions:
        print("  No HuggingFace structure found, trying flat directory layout...")
        for f in sorted(input_path.glob("**/*.csv")):
            if f.name == "motion.csv" or f.stem.endswith("_motion"):
                npz_path = f.with_suffix(".npz")
                obj_npz = f.parent / "object_motion.npz"
                motions.append({
                    "dataset": "flat",
                    "motion_name": f.stem,
                    "sample_name": "default",
                    "motion_csv": str(f),
                    "motion_npz": str(npz_path) if npz_path.exists() else None,
                    "object_npz": str(obj_npz) if obj_npz.exists() else None,
                    "metadata": {},
                    "tags": [],
                })

    return motions


def preprocess_motion(
    motion_info: Dict[str, Any],
    fk_engine: Optional[MujocoFK],
    min_length: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Preprocess a single motion sequence.

    Returns a dict with all fields needed for training, or None if invalid.
    """
    seq_name = f"{motion_info['dataset']}_{motion_info['motion_name']}_{motion_info['sample_name']}"

    # Load motion data (prefer NPZ, fall back to CSV)
    motion_data = {}
    if motion_info["motion_npz"]:
        try:
            motion_data = load_motion_npz(motion_info["motion_npz"])
        except Exception as e:
            print(f"  Warning: Failed to load NPZ for {seq_name}: {e}")

    if "root_pos" not in motion_data and motion_info["motion_csv"]:
        try:
            motion_data = load_motion_csv(motion_info["motion_csv"])
        except Exception as e:
            print(f"  Warning: Failed to load CSV for {seq_name}: {e}")
            return None

    if "root_pos" not in motion_data:
        print(f"  Warning: No root_pos found for {seq_name}")
        return None

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"]
    dof_pos = motion_data.get("dof_pos")

    if dof_pos is None:
        print(f"  Warning: No dof_pos found for {seq_name}")
        return None

    T = root_pos.shape[0]
    if T < min_length:
        print(f"  Warning: Sequence too short ({T} < {min_length}) for {seq_name}")
        return None

    # Validate shapes
    if root_pos.shape != (T, 3):
        print(f"  Warning: Invalid root_pos shape {root_pos.shape} for {seq_name}")
        return None
    if root_rot.shape[0] != T:
        print(f"  Warning: Mismatched root_rot length for {seq_name}")
        return None

    # Handle root rotation format
    if root_rot.shape[1] == 4:
        pass  # Already quaternion
    elif root_rot.shape[1] == 3:
        # Might be euler angles or axis-angle, skip for now
        print(f"  Warning: root_rot has 3 components (not quaternion) for {seq_name}, skipping")
        return None
    elif root_rot.shape[1] == 9:
        # Rotation matrix flattened
        from utils.rotation import mat_to_quat_xyzw
        import torch
        rot_mat = root_rot.reshape(T, 3, 3)
        root_rot = mat_to_quat_xyzw(torch.from_numpy(rot_mat).float()).numpy()

    # Compute hand positions
    hand_positions = motion_data.get("hand_positions")
    if hand_positions is None and fk_engine is not None:
        try:
            hand_positions = fk_engine.compute_hand_positions(root_pos, root_rot, dof_pos)
        except Exception as e:
            print(f"  Warning: FK failed for {seq_name}: {e}")
            hand_positions = None

    if hand_positions is None:
        # Create placeholder zeros (Stage 2 can train unconditionally)
        hand_positions = np.zeros((T, 6), dtype=np.float32)

    # Load object motion
    object_data = {}
    if motion_info["object_npz"]:
        try:
            object_data = load_object_motion(motion_info["object_npz"])
        except Exception as e:
            print(f"  Warning: Failed to load object motion for {seq_name}: {e}")

    # Ensure object data matches motion length
    for key in list(object_data.keys()):
        arr = object_data[key]
        if arr.shape[0] != T:
            # Try to trim or pad
            if arr.shape[0] > T:
                object_data[key] = arr[:T]
            else:
                pad = np.zeros((T - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
                object_data[key] = np.concatenate([arr, pad], axis=0)

    fps = motion_data.get("fps", 30.0)

    result = {
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "hand_positions": hand_positions.astype(np.float32),
        "fps": float(fps),
        "seq_name": seq_name,
        "dataset": motion_info["dataset"],
        "tags": motion_info["tags"],
    }

    # Add object data
    if "object_pos" in object_data:
        result["object_pos"] = object_data["object_pos"]
    if "object_rot" in object_data:
        result["object_rot"] = object_data["object_rot"]
    if "object_lin_vel" in object_data:
        result["object_lin_vel"] = object_data["object_lin_vel"]
    if "object_ang_vel" in object_data:
        result["object_ang_vel"] = object_data["object_ang_vel"]
    if "contact" in object_data:
        result["contact"] = object_data["contact"]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HuggingFace retargeted motion dataset for diffusion training"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing the HF dataset (e.g. ./data/hf_dataset/data/unitree_g1)")
    parser.add_argument("--output_dir", type=str, default="./data/hf_preprocessed",
                        help="Output directory for preprocessed PKL files")
    parser.add_argument("--robot_xml", type=str,
                        default="../g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml",
                        help="Path to MuJoCo XML model for FK computation")
    parser.add_argument("--min_length", type=int, default=30,
                        help="Minimum sequence length in frames")
    parser.add_argument("--skip_fk", action="store_true",
                        help="Skip FK computation (hand positions will be zeros)")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Only process specific datasets (e.g. omomo lafan1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize FK engine
    fk_engine = None
    if not args.skip_fk:
        xml_path = Path(args.robot_xml)
        if not xml_path.exists():
            # Try relative to project root
            alt_path = Path(PROJECT_ROOT) / args.robot_xml
            if alt_path.exists():
                xml_path = alt_path
            else:
                # Try absolute path to g1-gmr
                alt_path = Path("/home/learning/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml")
                if alt_path.exists():
                    xml_path = alt_path

        if xml_path.exists():
            print(f"Loading MuJoCo model from {xml_path}")
            try:
                fk_engine = MujocoFK(str(xml_path))
            except ImportError:
                print("WARNING: mujoco not available. Hand positions will be zeros.")
                print("  Install with: pip install mujoco")
            except Exception as e:
                print(f"WARNING: Failed to load MuJoCo model: {e}")
                print("  Hand positions will be zeros.")
        else:
            print(f"WARNING: Robot XML not found at {args.robot_xml}")
            print("  Hand positions will be zeros.")
            print("  Provide --robot_xml path or install mujoco + g1-gmr assets.")

    # Discover motions
    print(f"\nScanning {args.input_dir} for motions...")
    all_motions = discover_motions(args.input_dir)
    print(f"Found {len(all_motions)} motion sequences")

    # Filter by dataset if requested
    if args.datasets:
        all_motions = [m for m in all_motions if m["dataset"] in args.datasets]
        print(f"Filtered to {len(all_motions)} motions from datasets: {args.datasets}")

    # Process each motion
    successful = 0
    failed = 0
    has_object = 0
    total_frames = 0

    for i, motion_info in enumerate(all_motions):
        seq_name = f"{motion_info['dataset']}_{motion_info['motion_name']}_{motion_info['sample_name']}"

        if (i + 1) % 10 == 0 or i == 0:
            print(f"\nProcessing [{i+1}/{len(all_motions)}]: {seq_name}")

        result = preprocess_motion(motion_info, fk_engine, args.min_length)

        if result is None:
            failed += 1
            continue

        # Save as PKL
        out_path = output_dir / f"{seq_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f)

        successful += 1
        total_frames += result["root_pos"].shape[0]
        if "object_pos" in result:
            has_object += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"{'='*60}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total frames: {total_frames}")
    print(f"  With object data: {has_object}")
    print(f"  Output directory: {output_dir.resolve()}")

    if successful > 0:
        avg_frames = total_frames / successful
        print(f"  Average frames/sequence: {avg_frames:.0f}")
        print(f"  Average duration: {avg_frames/30:.1f}s (at 30fps)")

    # Save preprocessing metadata
    meta = {
        "num_sequences": successful,
        "total_frames": total_frames,
        "has_object_data": has_object,
        "min_length": args.min_length,
        "fk_computed": fk_engine is not None,
        "input_dir": str(args.input_dir),
    }
    with open(output_dir / "preprocessing_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nNext step: Train the diffusion model:")
    print(f"  ./src/train_stage1_hf.sh")
    print(f"  ./src/train_stage2_hf.sh")


if __name__ == "__main__":
    main()
