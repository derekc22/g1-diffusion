"""
Plot comparison of model-generated vs ground truth motion — HuggingFace samples.

Each HF sample PKL already contains both generated and GT data, so no separate
retargeted folder is needed.

Formatting matches plot_robot_motion_compare_w_object.py exactly.

Usage:
    python scripts/plot_compare_hf.py \
        --sample_folder logs/stage2_hf_.../samples/ts1000_... \
        --save_dir figures/compare_hf
"""

import argparse
import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def quat_xyzw_to_euler(q):
    """Convert xyzw quaternion array (N,4) to Euler angles (N,3) [roll, pitch, yaw].
    Equivalent to quat_wxyz_to_euler but for xyzw convention."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1)


def compute_sse(model_data, retargeted_data, min_len):
    """Compute sum of squared errors between model and retargeted data."""
    return np.sum((model_data[:min_len] - retargeted_data[:min_len])**2)


def plot_root_obj_hand_pos_rot(data, save_dir):
    seq_name = data.get("seq_name", "unknown")

    # Model (generated)
    model_root_pos = data["root_pos"]
    model_root_rot = data["root_rot"]       # xyzw
    model_object_pos = data["object_pos"]
    model_object_rot = data["object_rot"]   # xyzw
    model_hand_positions = data.get("hand_positions", None)
    source_start = data.get("source_start", 0)

    # GT (retargeted)
    retargeted_root_pos = data["gt_root_pos"]
    retargeted_root_rot = data["gt_root_rot"]  # xyzw
    retargeted_hand_positions = data.get("gt_hands", None)

    # For object, GT object = same as model object (it's the conditioning input)
    retargeted_object_pos = model_object_pos
    retargeted_object_rot = model_object_rot

    fps = data.get("fps", 30.0)

    # Model time axis
    min_len_model = min([len(model_object_pos), len(model_root_pos)])
    if model_hand_positions is not None:
        min_len_model = min(min_len_model, len(model_hand_positions))

    dt_model = 1/fps
    start_t = source_start * dt_model
    tf_model = (min_len_model-1)*dt_model + start_t
    t_model = np.linspace(start_t, tf_model, min_len_model)

    # Retargeted time axis
    min_len_retargeted = min([len(retargeted_object_pos), len(retargeted_root_pos)])
    if retargeted_hand_positions is not None:
        min_len_retargeted = min(min_len_retargeted, len(retargeted_hand_positions))

    dt_retargeted = 1/fps
    tf_retargeted = (min_len_retargeted-1)*dt_retargeted
    t_retargeted = np.linspace(0, tf_retargeted, min_len_retargeted)

    # Compute minimum length for SSE comparison (use overlapping portion)
    min_len_compare = min(min_len_model, min_len_retargeted)

    # Determine figure size and layout based on whether hand data is available
    has_hand_data = model_hand_positions is not None and retargeted_hand_positions is not None
    if has_hand_data:
        plt.figure(figsize=(17, 15))  # Taller figure for 6 rows
        nrows = 6
    else:
        plt.figure(figsize=(17, 9))
        nrows = 4

    # Convert rotations to euler (xyzw convention)
    retargeted_root_rot_eul = quat_xyzw_to_euler(retargeted_root_rot)
    retargeted_object_rot_eul = quat_xyzw_to_euler(retargeted_object_rot)
    model_root_rot_eul = quat_xyzw_to_euler(model_root_rot)
    model_object_rot_eul = quat_xyzw_to_euler(model_object_rot)

    # RETARGETED 

    # Row 1: Root position
    sse = compute_sse(model_root_pos[:, 0], retargeted_root_pos[:, 0], min_len_compare)
    plt.subplot(nrows, 3, 1)
    plt.plot(t_retargeted, retargeted_root_pos[:min_len_retargeted, 0], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('x_root [m]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    sse = compute_sse(model_root_pos[:, 1], retargeted_root_pos[:, 1], min_len_compare)
    plt.subplot(nrows, 3, 2)
    plt.plot(t_retargeted, retargeted_root_pos[:min_len_retargeted, 1], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('y_root [m]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    sse = compute_sse(model_root_pos[:, 2], retargeted_root_pos[:, 2], min_len_compare)
    plt.subplot(nrows, 3, 3)
    plt.plot(t_retargeted, retargeted_root_pos[:min_len_retargeted, 2], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('z_root [m]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    # Row 2: Object position
    sse = compute_sse(model_object_pos[:, 0], retargeted_object_pos[:, 0], min_len_compare)
    plt.subplot(nrows, 3, 4)
    plt.plot(t_retargeted, retargeted_object_pos[:min_len_retargeted, 0], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('x_obj [m]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    sse = compute_sse(model_object_pos[:, 1], retargeted_object_pos[:, 1], min_len_compare)
    plt.subplot(nrows, 3, 5)
    plt.plot(t_retargeted, retargeted_object_pos[:min_len_retargeted, 1], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('y_obj [m]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    sse = compute_sse(model_object_pos[:, 2], retargeted_object_pos[:, 2], min_len_compare)
    plt.subplot(nrows, 3, 6)
    plt.plot(t_retargeted, retargeted_object_pos[:min_len_retargeted, 2], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('z_obj [m]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    # Row 3: Root rotation
    sse = compute_sse(model_root_rot_eul[:, 0], retargeted_root_rot_eul[:, 0], min_len_compare)
    plt.subplot(nrows, 3, 7)
    plt.plot(t_retargeted, retargeted_root_rot_eul[:min_len_retargeted, 0], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('roll_root [rad]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    sse = compute_sse(model_root_rot_eul[:, 1], retargeted_root_rot_eul[:, 1], min_len_compare)
    plt.subplot(nrows, 3, 8)
    plt.plot(t_retargeted, retargeted_root_rot_eul[:min_len_retargeted, 1], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('pitch_root [rad]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    sse = compute_sse(model_root_rot_eul[:, 2], retargeted_root_rot_eul[:, 2], min_len_compare)
    plt.subplot(nrows, 3, 9)
    plt.plot(t_retargeted, retargeted_root_rot_eul[:min_len_retargeted, 2], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('yaw_root [rad]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    # Row 4: Object rotation
    sse = compute_sse(model_object_rot_eul[:, 0], retargeted_object_rot_eul[:, 0], min_len_compare)
    plt.subplot(nrows, 3, 10)
    plt.plot(t_retargeted, retargeted_object_rot_eul[:min_len_retargeted, 0], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('roll_obj [rad]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    sse = compute_sse(model_object_rot_eul[:, 1], retargeted_object_rot_eul[:, 1], min_len_compare)
    plt.subplot(nrows, 3, 11)
    plt.plot(t_retargeted, retargeted_object_rot_eul[:min_len_retargeted, 1], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('pitch_obj [rad]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    sse = compute_sse(model_object_rot_eul[:, 2], retargeted_object_rot_eul[:, 2], min_len_compare)
    plt.subplot(nrows, 3, 12)
    plt.plot(t_retargeted, retargeted_object_rot_eul[:min_len_retargeted, 2], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('yaw_obj [rad]')
    plt.title(f'sse = {sse:.4f}')
    plt.grid()

    # Row 5 & 6: Hand positions (if available)
    if has_hand_data:
        # Left hand
        sse = compute_sse(model_hand_positions[:, 0], retargeted_hand_positions[:, 0], min_len_compare)
        plt.subplot(nrows, 3, 13)
        plt.plot(t_retargeted, retargeted_hand_positions[:min_len_retargeted, 0], linewidth=2, label="retargeted")
        plt.xlabel('t [s]')
        plt.ylabel('x_left_hand [m]')
        plt.title(f'sse = {sse:.4f}')
        plt.grid()

        sse = compute_sse(model_hand_positions[:, 1], retargeted_hand_positions[:, 1], min_len_compare)
        plt.subplot(nrows, 3, 14)
        plt.plot(t_retargeted, retargeted_hand_positions[:min_len_retargeted, 1], linewidth=2, label="retargeted")
        plt.xlabel('t [s]')
        plt.ylabel('y_left_hand [m]')
        plt.title(f'sse = {sse:.4f}')
        plt.grid()

        sse = compute_sse(model_hand_positions[:, 2], retargeted_hand_positions[:, 2], min_len_compare)
        plt.subplot(nrows, 3, 15)
        plt.plot(t_retargeted, retargeted_hand_positions[:min_len_retargeted, 2], linewidth=2, label="retargeted")
        plt.xlabel('t [s]')
        plt.ylabel('z_left_hand [m]')
        plt.title(f'sse = {sse:.4f}')
        plt.grid()

        # Right hand
        sse = compute_sse(model_hand_positions[:, 3], retargeted_hand_positions[:, 3], min_len_compare)
        plt.subplot(nrows, 3, 16)
        plt.plot(t_retargeted, retargeted_hand_positions[:min_len_retargeted, 3], linewidth=2, label="retargeted")
        plt.xlabel('t [s]')
        plt.ylabel('x_right_hand [m]')
        plt.title(f'sse = {sse:.4f}')
        plt.grid()

        sse = compute_sse(model_hand_positions[:, 4], retargeted_hand_positions[:, 4], min_len_compare)
        plt.subplot(nrows, 3, 17)
        plt.plot(t_retargeted, retargeted_hand_positions[:min_len_retargeted, 4], linewidth=2, label="retargeted")
        plt.xlabel('t [s]')
        plt.ylabel('y_right_hand [m]')
        plt.title(f'sse = {sse:.4f}')
        plt.grid()

        sse = compute_sse(model_hand_positions[:, 5], retargeted_hand_positions[:, 5], min_len_compare)
        plt.subplot(nrows, 3, 18)
        plt.plot(t_retargeted, retargeted_hand_positions[:min_len_retargeted, 5], linewidth=2, label="retargeted")
        plt.xlabel('t [s]')
        plt.ylabel('z_right_hand [m]')
        plt.title(f'sse = {sse:.4f}')
        plt.grid()

    # MODEL (overlayed on same subplots)

    # Row 1: Root position
    plt.subplot(nrows, 3, 1)
    plt.plot(t_model, model_root_pos[:min_len_model, 0], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('x_root [m]')
    plt.legend()

    plt.subplot(nrows, 3, 2)
    plt.plot(t_model, model_root_pos[:min_len_model, 1], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('y_root [m]')    

    plt.subplot(nrows, 3, 3)
    plt.plot(t_model, model_root_pos[:min_len_model, 2], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('z_root [m]')    

    # Row 2: Object position
    plt.subplot(nrows, 3, 4)
    plt.plot(t_model, model_object_pos[:min_len_model, 0], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('x_obj [m]')

    plt.subplot(nrows, 3, 5)
    plt.plot(t_model, model_object_pos[:min_len_model, 1], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('y_obj [m]')    

    plt.subplot(nrows, 3, 6)
    plt.plot(t_model, model_object_pos[:min_len_model, 2], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('z_obj [m]')    

    # Row 3: Root rotation
    plt.subplot(nrows, 3, 7)
    plt.plot(t_model, model_root_rot_eul[:min_len_model, 0], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('roll_root [rad]')

    plt.subplot(nrows, 3, 8)
    plt.plot(t_model, model_root_rot_eul[:min_len_model, 1], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('pitch_root [rad]')    

    plt.subplot(nrows, 3, 9)
    plt.plot(t_model, model_root_rot_eul[:min_len_model, 2], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('yaw_root [rad]')    

    # Row 4: Object rotation
    plt.subplot(nrows, 3, 10)
    plt.plot(t_model, model_object_rot_eul[:min_len_model, 0], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('roll_obj [rad]')

    plt.subplot(nrows, 3, 11)
    plt.plot(t_model, model_object_rot_eul[:min_len_model, 1], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('pitch_obj [rad]')    

    plt.subplot(nrows, 3, 12)
    plt.plot(t_model, model_object_rot_eul[:min_len_model, 2], linewidth=2, label="sampled")
    plt.xlabel('t [s]')
    plt.ylabel('yaw_obj [rad]')    

    # Row 5 & 6: Hand positions (if available)
    if has_hand_data:
        # Left hand
        plt.subplot(nrows, 3, 13)
        plt.plot(t_model, model_hand_positions[:min_len_model, 0], linewidth=2, label="sampled")

        plt.subplot(nrows, 3, 14)
        plt.plot(t_model, model_hand_positions[:min_len_model, 1], linewidth=2, label="sampled")

        plt.subplot(nrows, 3, 15)
        plt.plot(t_model, model_hand_positions[:min_len_model, 2], linewidth=2, label="sampled")

        # Right hand
        plt.subplot(nrows, 3, 16)
        plt.plot(t_model, model_hand_positions[:min_len_model, 3], linewidth=2, label="sampled")

        plt.subplot(nrows, 3, 17)
        plt.plot(t_model, model_hand_positions[:min_len_model, 4], linewidth=2, label="sampled")

        plt.subplot(nrows, 3, 18)
        plt.plot(t_model, model_hand_positions[:min_len_model, 5], linewidth=2, label="sampled")

    # TITLE
    
    plt.suptitle(seq_name)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{seq_name}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_folder", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="figures/compare_hf")
    parser.add_argument("--num_motions", type=int)
    args = parser.parse_args()

    sample_folder = args.sample_folder
    save_dir = args.save_dir

    pattern = r'logs/(.*?)/samples/(.*)'
    match = re.search(pattern, sample_folder)
    if match:
        log_id = match.group(1)
        sample_id = match.group(2).rstrip("/")
        save_dir = os.path.join(save_dir, log_id, sample_id)
    else:
        raise FileNotFoundError("IDs not found.")

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(sample_folder):
        raise FileNotFoundError(f"Sample dir {sample_folder} does not exist.")

    motion_files = sorted([f for f in os.listdir(sample_folder) if f.endswith('.pkl')])
    motion_num = len(motion_files)
    print(f"Found {motion_num} motion files in {sample_folder}, loading...")

    for motion_file in tqdm(motion_files[:args.num_motions]):
        fpath = os.path.join(sample_folder, motion_file)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        plot_root_obj_hand_pos_rot(data, save_dir)

    print("Loading done.")
