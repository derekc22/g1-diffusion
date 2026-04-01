"""
Stage 1 DDPM Sampling Script — HuggingFace Dataset
Object Motion Features → Hand Positions

Runs standard DDPM reverse diffusion (1000 steps) on HF-trained Stage 1 model.
The HF model uses compact 15D object motion features instead of 3075D BPS encoding.

Usage:
    python scripts/sample_stage1_hf.py
    python scripts/sample_stage1_hf.py --config_path experiments/stage1_hf/my_config.yaml

Configuration is done via YAML config files placed in experiments/stage1_hf/.
"""

import os
import sys
from typing import Optional, Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm
import types
import time

from models.stage1_hf_diffusion import Stage1HFHandDiffusion, Stage1HFHandDiffusionMLP
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.contact_constraints import (
    apply_contact_constraints,
    ContactConstraintProcessor,
    compute_contact_metrics,
    compute_hand_jpe,
)
from utils.general import load_config

# ---------------------------------------------------------------------------
# Compatibility shim for pickles created by NumPy >= 2.0
# ---------------------------------------------------------------------------
if "numpy._core" not in sys.modules:
    core_pkg = types.ModuleType("numpy._core")
    core_pkg.__path__ = []
    sys.modules["numpy._core"] = core_pkg

if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

if "numpy._core.numerictypes" not in sys.modules:
    sys.modules["numpy._core.numerictypes"] = np.core.numerictypes

if "numpy._core.umath" not in sys.modules:
    sys.modules["numpy._core.umath"] = np.core.umath
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load Stage 1 HF checkpoint and reconstruct model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt["config"]
    arch = config.get("train", {}).get("architecture", "transformer")
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    # HF-specific model params
    object_feature_input_dim = model_cfg.get("object_feature_input_dim", 15)
    encoder_hidden = model_cfg.get("encoder_hidden", 512)
    encoder_layers = model_cfg.get("encoder_layers", 3)
    object_feature_dim = model_cfg.get("object_feature_dim", 256)
    hand_dim = model_cfg.get("hand_dim", 6)

    # Get max_len from config
    window_size = dataset_cfg.get("window_size", 120)
    max_len = model_cfg.get("max_len", window_size + 100)

    if arch == "transformer":
        model = Stage1HFHandDiffusion(
            object_feature_input_dim=object_feature_input_dim,
            encoder_hidden=encoder_hidden,
            object_feature_dim=object_feature_dim,
            encoder_layers=encoder_layers,
            hand_dim=hand_dim,
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nhead", 4),
            num_transformer_layers=model_cfg.get("num_layers", 4),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.1),
            max_len=max_len,
        )
    else:
        model = Stage1HFHandDiffusionMLP(
            object_feature_input_dim=object_feature_input_dim,
            encoder_hidden=encoder_hidden,
            object_feature_dim=object_feature_dim,
            encoder_layers=encoder_layers,
            hand_dim=hand_dim,
        )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Load normalization stats
    norm_stats = ckpt.get("norm_stats", {})
    hand_mean = norm_stats.get("hand_mean")
    hand_std = norm_stats.get("hand_std")

    if hand_mean is not None:
        hand_mean = hand_mean.to(device)
        hand_std = hand_std.to(device)

    # Get diffusion schedule
    train_cfg = config.get("train", {})
    timesteps = train_cfg.get("timesteps", 1000)
    schedule = DiffusionSchedule(
        DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
    ).to(device)

    return model, schedule, hand_mean, hand_std, config


def build_object_features(data: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Build the 15D object feature vector from PKL data fields.

    Returns:
        (T, 15) array or None if no object data available.
    """
    if "object_features" in data:
        # Already pre-built (e.g. from HF dataset preprocessing)
        return np.asarray(data["object_features"], dtype=np.float32)

    obj_pos = data.get("object_pos")
    if obj_pos is None:
        return None

    obj_pos = np.asarray(obj_pos, dtype=np.float32)  # (T, 3)
    T = obj_pos.shape[0]

    # Rotation (6D)
    obj_rot = data.get("object_rot")
    if obj_rot is not None:
        obj_rot = np.asarray(obj_rot, dtype=np.float32)
        if obj_rot.shape[-1] == 4:
            # Quaternion → 6D (take first two columns of rotation matrix)
            from utils.rotation import quat_to_rot6d_xyzw
            obj_rot_6d = quat_to_rot6d_xyzw(torch.from_numpy(obj_rot)).numpy()
        elif obj_rot.shape[-1] == 6:
            obj_rot_6d = obj_rot
        elif obj_rot.shape[-2:] == (3, 3):
            obj_rot_6d = obj_rot.reshape(T, 9)[:, :6]
        else:
            obj_rot_6d = np.zeros((T, 6), dtype=np.float32)
    else:
        obj_rot_6d = np.zeros((T, 6), dtype=np.float32)

    # Velocities
    obj_lin_vel = data.get("object_lin_vel")
    if obj_lin_vel is not None:
        obj_lin_vel = np.asarray(obj_lin_vel, dtype=np.float32)
    else:
        obj_lin_vel = np.zeros((T, 3), dtype=np.float32)

    obj_ang_vel = data.get("object_ang_vel")
    if obj_ang_vel is not None:
        obj_ang_vel = np.asarray(obj_ang_vel, dtype=np.float32)
    else:
        obj_ang_vel = np.zeros((T, 3), dtype=np.float32)

    # Concatenate: [pos(3), rot_6d(6), lin_vel(3), ang_vel(3)] = 15D
    features = np.concatenate([obj_pos, obj_rot_6d, obj_lin_vel, obj_ang_vel], axis=-1)
    return features


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    object_features: torch.Tensor,
) -> torch.Tensor:
    """
    Run full DDPM reverse diffusion to generate hand positions.

    Args:
        model: Stage1HFHandDiffusion model
        schedule: Diffusion schedule
        object_features: (1, T, 15) object motion features

    Returns:
        (1, T, 6) predicted (normalized) hand positions
    """
    device = object_features.device
    B, T, _ = object_features.shape
    x = torch.randn(B, T, 6, device=device)

    for n in reversed(range(schedule.timesteps)):
        t = torch.full((B,), n, device=device, dtype=torch.long)

        # x0 prediction
        x0_pred = model(x, t, object_features)

        if n > 0:
            alpha_bar_t = schedule.alpha_bar[n]
            alpha_bar_t_prev = schedule.alpha_bar[n - 1]
            alpha_t = schedule.alpha[n]

            mean = (
                torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
            )

            sigma = torch.sqrt(schedule.beta[n])
            x = mean + sigma * torch.randn_like(x)
        else:
            x = x0_pred

    return x


def denormalize_hands(
    hands: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    """Denormalize hand positions."""
    if mean is None:
        return hands
    if hands.ndim == 3:
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    elif hands.ndim == 2:
        mean = mean.view(1, -1)
        std = std.view(1, -1)
    return hands * std + mean


def process_sequence(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    data: Dict[str, Any],
    hand_mean: Optional[torch.Tensor],
    hand_std: Optional[torch.Tensor],
    device: torch.device,
    apply_constraints: bool = True,
    contact_threshold: float = 0.03,
    partial_motion_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Process a single sequence through Stage 1."""
    # Build object features
    obj_feat_np = build_object_features(data)
    if obj_feat_np is None:
        T = data.get("hand_positions", data.get("dof_pos")).shape[0]
        obj_feat_np = np.zeros((T, 15), dtype=np.float32)

    T_original = obj_feat_np.shape[0]
    partial = False
    target_len = None

    # Handle partial motion generation
    obj_verts = data.get("object_verts")
    obj_rot = data.get("object_rotation")

    if partial_motion_length is not None and partial_motion_length > 0:
        if partial_motion_length < T_original:
            obj_feat_np = obj_feat_np[:partial_motion_length]
            if obj_verts is not None:
                obj_verts = obj_verts[:partial_motion_length]
            if obj_rot is not None:
                obj_rot = obj_rot[:partial_motion_length]
            partial = True
            target_len = partial_motion_length

    # Convert to tensor
    obj_feat = torch.from_numpy(obj_feat_np).float().unsqueeze(0).to(device)  # (1, T, 15)

    # Sample
    hands_norm = sample_ddpm(model, schedule, obj_feat)

    # Denormalize
    hands_raw = denormalize_hands(hands_norm, hand_mean, hand_std)
    hands_raw_np = hands_raw.squeeze(0).cpu().numpy()  # (T, 6)

    result = {
        "hands_raw": hands_raw_np,
        "partial": partial,
        "target_len": target_len,
        "original_len": T_original,
    }

    # Apply contact constraints
    if apply_constraints and obj_verts is not None and obj_rot is not None:
        processor = ContactConstraintProcessor(contact_threshold=contact_threshold)
        hands_rect, metadata = processor.process(hands_raw_np, obj_verts, obj_rot)
        result["hands_rectified"] = hands_rect
        result["contact_metadata"] = metadata
    else:
        result["hands_rectified"] = hands_raw_np
        result["contact_metadata"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 1 DDPM Sampling — HuggingFace Dataset")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./experiments/stage1_hf/sample_stage1_hf.yaml",
        help="Path to YAML experiment config file",
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    sample_yml = yml["sample"]
    dataset_yml = yml.get("dataset", {})

    root_dir = yml["root_dir"]
    ckpt_path = sample_yml["ckpt_path"]
    device_str = sample_yml.get("device", "cuda")
    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    num_samples = sample_yml.get("num_samples", None)
    seed = sample_yml.get("seed", 42)
    timesteps = sample_yml.get("timesteps", 1000)
    partial_motion_length = sample_yml.get("partial_motion_length", None)

    if not ckpt_path:
        raise ValueError("ckpt_path must be set in config")

    # Derive output directory from checkpoint path
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = yml.get("exp_name", "")
    suffix = f"_{exp_name}" if exp_name else ""

    ckpt_parts = ckpt_path.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        window_size = dataset_yml.get("window_size", 120)
        stride = dataset_yml.get("stride", 10)
        sample_folder = f"ts{timesteps}_w{window_size}_s{stride}_{timestamp}{suffix}"
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    else:
        output_dir = os.path.join("out", "stage1_hf", timestamp)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)
    torch.manual_seed(seed)

    # Save config to samples directory for reproducibility
    import yaml

    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(yml, f, default_flow_style=False)

    # Load model
    model, schedule, hand_mean, hand_std, config = load_checkpoint(ckpt_path, device)

    # Find input files
    files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if num_samples:
        files = files[:num_samples]

    print(f"Processing {len(files)} files")
    print(f"Output directory: {output_dir}")

    # Metrics accumulators
    all_hand_jpe = []
    all_metrics = []

    # Performance tracking
    total_time = 0.0
    sample_durations = []
    all_frame_counts = []

    for fpath in tqdm(files, desc="Processing"):
        fname = os.path.basename(fpath)

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        start_time = time.perf_counter()

        result = process_sequence(
            model,
            schedule,
            data,
            hand_mean,
            hand_std,
            device,
            apply_constraints=apply_constraints,
            contact_threshold=contact_threshold,
            partial_motion_length=partial_motion_length,
        )

        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        # Track timing
        fps = data.get("fps", 30.0)
        num_frames = result["hands_raw"].shape[0]
        all_frame_counts.append(num_frames)
        sample_durations.append(num_frames / fps)

        # Evaluate against GT if available
        if "hand_positions" in data:
            gt_hands = data["hand_positions"]
            pred_hands = result["hands_rectified"]
            # Truncate GT to match prediction length (partial motion)
            min_len = min(pred_hands.shape[0], gt_hands.shape[0])
            pred_hands = pred_hands[:min_len]
            gt_hands = gt_hands[:min_len]
            jpe = compute_hand_jpe(pred_hands, gt_hands)
            all_hand_jpe.append(jpe)

            if "object_verts" in data:
                obj_verts_eval = data["object_verts"][:min_len]
                metrics = compute_contact_metrics(
                    pred_hands,
                    gt_hands,
                    obj_verts_eval,
                    contact_threshold=0.05,
                )
                all_metrics.append(metrics)

        # Save result
        output_data = {
            "seq_name": data.get("seq_name", fname),
            "hands_raw": result["hands_raw"],
            "hands_rectified": result["hands_rectified"],
            "contact_metadata": result["contact_metadata"],
        }

        if "hand_positions" in data:
            output_data["hands_gt"] = data["hand_positions"]

        out_path = os.path.join(output_dir, fname)
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)

    # Print evaluation summary
    if all_hand_jpe:
        print(f"\n{'='*50}")
        print("Evaluation Results:")
        print(f"  Hand JPE (cm): {np.mean(all_hand_jpe):.2f} ± {np.std(all_hand_jpe):.2f}")
        if all_metrics:
            avg_prec = np.mean([m["precision"] for m in all_metrics])
            avg_rec = np.mean([m["recall"] for m in all_metrics])
            avg_f1 = np.mean([m["f1"] for m in all_metrics])
            print(f"  Contact Precision: {avg_prec:.3f}")
            print(f"  Contact Recall: {avg_rec:.3f}")
            print(f"  Contact F1: {avg_f1:.3f}")
        print("=" * 50)

    # Print performance summary
    print(f"\n{'='*50}")
    print("Performance Summary:")
    print(f"  Total files: {len(files)}")
    print(f"  Total time: {total_time:.2f}s")
    if files:
        print(f"  Average time: {total_time / len(files) * 1000:.2f}ms per sample")
        print(f"  Throughput: {len(files) / total_time:.2f} samples/sec")
    if sample_durations:
        print(f"  Average sample duration: {np.mean(sample_durations):.2f}s")
    if all_frame_counts:
        print(f"  Frames per motion: {int(np.mean(all_frame_counts))} avg, {min(all_frame_counts)} min, {max(all_frame_counts)} max")
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        print(f"  Generated fps: {gen_fps:.1f}")
    print("=" * 50)

    # Save summary
    summary_lines = [
        "Performance Summary",
        "=" * 50,
        f"Total files: {len(files)}",
        f"Total time: {total_time:.2f}s",
    ]
    if files:
        summary_lines.extend([
            f"Average time: {total_time / len(files) * 1000:.2f}ms per sample",
            f"Throughput: {len(files) / total_time:.2f} samples/sec",
        ])
    if sample_durations:
        summary_lines.append(f"Average sample duration: {np.mean(sample_durations):.2f}s")
    if all_frame_counts:
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        summary_lines.append(f"Frames per motion: {int(np.mean(all_frame_counts))} avg, {min(all_frame_counts)} min, {max(all_frame_counts)} max")
        summary_lines.append(f"Generated fps: {gen_fps:.1f}")
    if all_hand_jpe:
        summary_lines.extend([
            "",
            "Evaluation Results",
            f"Hand JPE (cm): {np.mean(all_hand_jpe):.2f} ± {np.std(all_hand_jpe):.2f}",
        ])
        if all_metrics:
            summary_lines.extend([
                f"Contact Precision: {np.mean([m['precision'] for m in all_metrics]):.3f}",
                f"Contact Recall: {np.mean([m['recall'] for m in all_metrics]):.3f}",
                f"Contact F1: {np.mean([m['f1'] for m in all_metrics]):.3f}",
            ])

    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
