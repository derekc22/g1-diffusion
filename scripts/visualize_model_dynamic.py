"""
Visualize generated model samples with the correct object mesh per motion.

The existing GMR object viewer builds one MuJoCo model around one object mesh.
This wrapper keeps that behavior, but reinitializes the viewer whenever the next
sample needs a different object mesh. That makes mixed-object sample folders
safe to inspect without forcing one object for the whole folder.
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


DEFAULT_GMR_ROOT = "/home/learning/Documents/g1-gmr"
DEFAULT_OBJECTS_DIR = "/home/learning/Documents/omomo_release/data/captured_objects"

paused = False
motion_id = 0
motion_num = 0
terminate = False


def keyboard_callback(keycode: int) -> None:
    global paused, motion_id, motion_num, terminate

    try:
        key = chr(keycode)
    except ValueError:
        return

    if key == " ":
        paused = not paused
    elif key == "[" and motion_num > 0:
        motion_id = (motion_id - 1) % motion_num
    elif key == "]" and motion_num > 0:
        motion_id = (motion_id + 1) % motion_num
    elif key == ".":
        terminate = True


def add_gmr_to_path(gmr_root: str) -> None:
    root = os.path.abspath(gmr_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def mesh_key_from_filename(filename: str) -> str:
    key = os.path.basename(str(filename)).lower()
    key = os.path.splitext(key)[0]
    key = key.replace("_cleaned_simplified", "")
    key = key.replace("_cleaned", "")
    return key


def build_mesh_index(objects_dir: str) -> Dict[str, str]:
    objects_path = Path(objects_dir)
    if not objects_path.is_dir():
        return {}

    mesh_index: Dict[str, str] = {}
    for path in sorted(objects_path.glob("*.obj")):
        key = mesh_key_from_filename(path.name)
        if key not in mesh_index or "cleaned_simplified" in path.name:
            mesh_index[key] = str(path)
    return mesh_index


def normalize_object_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = str(value).strip()
    if not name:
        return None
    return mesh_key_from_filename(name)


def object_from_seq_name(seq_name: str, mesh_index: Dict[str, str]) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(str(seq_name)))[0].lower()

    for key in sorted(mesh_index, key=len, reverse=True):
        if re.search(rf"(^|_){re.escape(key)}($|_)", stem):
            return key

    parts = stem.split("_")
    sub_idx = None
    for idx, part in enumerate(parts):
        if re.fullmatch(r"sub\d+", part):
            sub_idx = idx
            break
    if sub_idx is None or sub_idx + 1 >= len(parts):
        return None

    object_parts: List[str] = []
    for part in parts[sub_idx + 1 :]:
        if part.isdigit() or re.fullmatch(r"sample\d+", part):
            break
        object_parts.append(part)
    if not object_parts:
        return None
    return "_".join(object_parts)


def infer_object_name(
    motion_data: Dict[str, Any],
    motion_file: str,
    mesh_index: Dict[str, str],
    reference_data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    for source in (motion_data, reference_data):
        if not source:
            continue
        object_name = normalize_object_name(source.get("object_name"))
        if object_name:
            return object_name
        mesh_file = source.get("mesh_file")
        if mesh_file:
            return mesh_key_from_filename(str(mesh_file))

    seq_name = motion_data.get("seq_name") or motion_file
    return object_from_seq_name(str(seq_name), mesh_index)


def find_object_mesh(
    object_name: Optional[str],
    mesh_index: Dict[str, str],
    motion_data: Dict[str, Any],
    reference_data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    for source in (motion_data, reference_data):
        if not source:
            continue
        mesh_file = source.get("mesh_file")
        if mesh_file and os.path.exists(str(mesh_file)):
            return str(mesh_file)

    if object_name:
        return mesh_index.get(object_name)
    return None


def load_reference_motion(
    reference_motion_folder: Optional[str],
    motion_data: Dict[str, Any],
    motion_file: str,
) -> Optional[Dict[str, Any]]:
    if not reference_motion_folder:
        return None

    seq_name = motion_data.get("seq_name") or motion_file
    seq_name = os.path.splitext(os.path.basename(str(seq_name)))[0]
    candidates = [seq_name]

    parts = seq_name.split("_")
    sub_idx = next(
        (idx for idx, part in enumerate(parts) if re.fullmatch(r"sub\d+", part)),
        None,
    )
    if sub_idx is not None:
        ref_parts = parts[sub_idx:]
        if ref_parts and re.fullmatch(r"sample\d+", ref_parts[-1]):
            ref_parts = ref_parts[:-1]
        if ref_parts:
            candidates.append("_".join(ref_parts))

    for candidate in candidates:
        path = os.path.join(reference_motion_folder, f"{candidate}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    return None


def load_obj_vertices(mesh_path: str) -> Optional[np.ndarray]:
    vertices = []
    with open(mesh_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        return None
    return np.asarray(vertices, dtype=np.float64)


def infer_object_mesh_scale(
    motion_data: Optional[Dict[str, Any]],
    mesh_path: Optional[str],
) -> Optional[float]:
    if motion_data is None or mesh_path is None:
        return None

    if "object_mesh_scale" in motion_data:
        return float(motion_data["object_mesh_scale"])

    if "object_verts" not in motion_data:
        return None

    mesh_vertices = load_obj_vertices(mesh_path)
    if mesh_vertices is None:
        return None

    object_verts = np.asarray(motion_data["object_verts"], dtype=np.float64)
    if object_verts.ndim != 3 or object_verts.shape[1] != mesh_vertices.shape[0]:
        return None

    if "object_rotation" in motion_data:
        object_rotation = np.asarray(motion_data["object_rotation"], dtype=np.float64)[0]
    elif "object_rot" in motion_data:
        object_rotation = R.from_quat(
            np.asarray(motion_data["object_rot"], dtype=np.float64)[0]
        ).as_matrix()
    else:
        object_rotation = np.eye(3)

    mesh_centered = mesh_vertices - mesh_vertices.mean(axis=0)
    fitted = (object_rotation @ mesh_centered.T).T
    target = object_verts[0] - object_verts[0].mean(axis=0)
    denom = float(np.sum(fitted * fitted))
    if denom <= 0.0:
        return None

    scale = float(np.sum(fitted * target) / denom)
    if scale <= 0.0:
        return None
    return scale


def build_video_path(robot_motion_folder: str, save_dir: str) -> str:
    pattern = r"logs/(.*?)/samples/(.*)"
    match = re.search(pattern, robot_motion_folder)
    if match:
        log_id = match.group(1)
        sample_id = match.group(2).rstrip("/")
        video_dir = os.path.join(save_dir, log_id, sample_id)
        return os.path.join(video_dir, "render_dynamic.mp4")

    folder_name = os.path.basename(robot_motion_folder.rstrip("/"))
    return os.path.join(save_dir, f"{folder_name}_dynamic.mp4")


def as_wxyz_quat(rot: Any, key: str) -> np.ndarray:
    rot_arr = np.asarray(rot)
    if rot_arr.ndim == 2 and rot_arr.shape[1] == 4:
        return rot_arr[:, [3, 0, 1, 2]]
    if rot_arr.ndim == 3 and rot_arr.shape[1:] == (3, 3):
        quat_xyzw = R.from_matrix(rot_arr).as_quat()
        return quat_xyzw[:, [3, 0, 1, 2]]
    raise ValueError(f"{key} must be quaternion (T, 4) or matrix (T, 3, 3), got {rot_arr.shape}")


def load_motion_file(motion_path: str) -> Tuple[
    Dict[str, Any],
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[List[str]],
    Optional[np.ndarray],
]:
    with open(motion_path, "rb") as f:
        motion_data = pickle.load(f)

    motion_fps = float(motion_data.get("fps", 30.0))
    motion_root_pos = np.asarray(motion_data["root_pos"])
    motion_root_rot = as_wxyz_quat(motion_data["root_rot"], "root_rot")
    motion_dof_pos = np.asarray(motion_data["dof_pos"])

    if "object_pos" in motion_data:
        motion_object_pos = np.asarray(motion_data["object_pos"])
    elif "object_centroid" in motion_data:
        motion_object_pos = np.asarray(motion_data["object_centroid"])
    else:
        raise KeyError(f"{motion_path} missing object_pos/object_centroid")

    if "object_rot" in motion_data:
        motion_object_rot = as_wxyz_quat(motion_data["object_rot"], "object_rot")
    elif "object_rotation" in motion_data:
        motion_object_rot = as_wxyz_quat(motion_data["object_rotation"], "object_rotation")
    else:
        raise KeyError(f"{motion_path} missing object_rot/object_rotation")

    return (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_object_pos,
        motion_object_rot,
        motion_data.get("local_body_pos"),
        motion_data.get("link_body_list"),
        motion_data.get("hand_positions"),
    )


def load_motion_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:

    motion_files = sorted(
        f for f in os.listdir(args.robot_motion_folder) if f.endswith(".pkl")
    )
    dataset = []
    mesh_index = build_mesh_index(args.objects_dir)

    print(f"Found {len(motion_files)} motion files in {args.robot_motion_folder}")
    for motion_file in tqdm(motion_files, desc="Loading motions"):
        motion_path = os.path.join(args.robot_motion_folder, motion_file)
        (
            motion_data,
            motion_fps,
            motion_root_pos,
            motion_root_rot,
            motion_dof_pos,
            motion_object_pos,
            motion_object_rot,
            motion_local_body_pos,
            motion_link_body_list,
            motion_hand_positions,
        ) = load_motion_file(motion_path)

        reference_data = load_reference_motion(
            args.reference_motion_folder,
            motion_data,
            motion_file,
        )
        object_name = infer_object_name(
            motion_data,
            motion_file,
            mesh_index,
            reference_data=reference_data,
        )
        object_mesh_path = find_object_mesh(
            object_name,
            mesh_index,
            motion_data,
            reference_data=reference_data,
        )
        object_mesh_scale = infer_object_mesh_scale(motion_data, object_mesh_path)
        if object_mesh_scale is None:
            object_mesh_scale = infer_object_mesh_scale(reference_data, object_mesh_path)
        if object_mesh_scale is None:
            object_mesh_scale = args.object_scale

        dataset.append(
            {
                "motion_file": motion_file,
                "motion_data": motion_data,
                "motion_fps": motion_fps,
                "motion_root_pos": motion_root_pos,
                "motion_root_rot": motion_root_rot,
                "motion_dof_pos": motion_dof_pos,
                "motion_object_pos": motion_object_pos,
                "motion_object_rot": motion_object_rot,
                "motion_local_body_pos": motion_local_body_pos,
                "motion_link_body_list": motion_link_body_list,
                "motion_hand_positions": motion_hand_positions,
                "object_name": object_name,
                "object_mesh_path": object_mesh_path,
                "object_mesh_scale": object_mesh_scale,
            }
        )

    return dataset


def object_signature(item: Dict[str, Any]) -> Tuple[str, float]:
    mesh_path = item["object_mesh_path"] or ""
    return mesh_path, float(item["object_mesh_scale"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1_with_object")
    parser.add_argument("--robot_motion_folder", type=str, required=True)
    parser.add_argument("--gmr_root", type=str, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--objects_dir", type=str, default=DEFAULT_OBJECTS_DIR)
    parser.add_argument("--reference_motion_folder", type=str, default=None)
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--save_dir", type=str, default="videos")
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--no_rate_limit", action="store_true")
    args = parser.parse_args()

    add_gmr_to_path(args.gmr_root)
    from general_motion_retargeting import RobotMotionViewerWithObject

    if not os.path.exists(args.robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {args.robot_motion_folder} does not exist.")

    dataset = load_motion_dataset(args)
    if not dataset:
        raise RuntimeError(f"No .pkl files found in {args.robot_motion_folder}")

    global motion_num, motion_id
    motion_num = len(dataset)
    motion_id = 0

    writer = None
    video_path = None
    if args.record_video:
        video_path = build_video_path(args.robot_motion_folder, args.save_dir)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=float(dataset[0]["motion_fps"]))
        print(f"Recording dynamic video to: {video_path}")

    env = None
    renderer = None
    current_signature = None
    current_motion_id = -1
    frame_idx = 0

    try:
        while not terminate:
            if motion_id < 0:
                motion_id = 0
            if motion_id >= motion_num:
                if args.auto:
                    break
                motion_id = motion_num - 1

            item = dataset[motion_id]
            signature = object_signature(item)
            if env is None or signature != current_signature:
                if renderer is not None:
                    renderer.close()
                    renderer = None
                if env is not None:
                    env.close()

                if item["object_mesh_path"] is None:
                    print(
                        f"No mesh for {item['motion_file']} "
                        f"(object={item['object_name']}); using placeholder box."
                    )
                else:
                    print(
                        f"Using object for {item['motion_file']}: "
                        f"{item['object_name']} -> {item['object_mesh_path']}"
                    )

                env = RobotMotionViewerWithObject(
                    robot_type=args.robot,
                    motion_fps=item["motion_fps"],
                    camera_follow=False,
                    record_video=False,
                    keyboard_callback=keyboard_callback,
                    object_mesh_path=item["object_mesh_path"],
                    object_mesh_scale=item["object_mesh_scale"],
                )
                if writer is not None:
                    renderer = mj.Renderer(
                        env.model,
                        height=args.video_height,
                        width=args.video_width,
                    )
                current_signature = signature

            if current_motion_id != motion_id:
                current_motion_id = motion_id
                frame_idx = 0
                print(
                    f"Switched to motion {motion_id}: {item['motion_file']}, "
                    f"object={item['object_name']}, fps={item['motion_fps']}, "
                    f"frames={len(item['motion_root_pos'])}"
                )

            min_len = min(len(item["motion_object_pos"]), len(item["motion_root_pos"]))
            if not paused:
                env.step(
                    item["motion_root_pos"][frame_idx],
                    item["motion_root_rot"][frame_idx],
                    item["motion_dof_pos"][frame_idx],
                    item["motion_object_pos"][frame_idx],
                    item["motion_object_rot"][frame_idx],
                    rate_limit=not args.no_rate_limit,
                )
                if renderer is not None and writer is not None:
                    renderer.update_scene(env.data, camera=env.viewer.cam)
                    writer.append_data(renderer.render())

                frame_idx += 1
                if frame_idx >= min_len:
                    if args.auto:
                        motion_id += 1
                    else:
                        frame_idx = 0
    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        if writer is not None:
            writer.close()
            print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
