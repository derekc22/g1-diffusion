"""
Stage 1 Flow Matching Sampling Script: Object Geometry → Hand Positions

Generates hand positions from object motion using the trained Stage 1
flow matching model, then applies contact constraints.

Uses ODE integration (Euler/midpoint/RK4) instead of DDPM reverse chain.

Usage:
    python sample_stage1_fm.py --config_path ./config/sample_stage1_fm.yaml
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import glob
import pickle
import time
import numpy as np
import torch
from tqdm import tqdm
import types

from models.stage1_flow_matching import Stage1HandFlowMatching, Stage1HandFlowMatchingMLP
from utils.flow_matching import FlowMatchingConfig, get_ode_solver
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
    """Load Stage 1 FM checkpoint and reconstruct model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    config = ckpt["config"]
    arch = config.get("train", {}).get("architecture", "transformer")
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    
    bps_dim = model_cfg.get("bps_dim", 3072)
    centroid_dim = model_cfg.get("centroid_dim", 3)
    object_feature_dim = model_cfg.get("object_feature_dim", 256)
    hand_dim = model_cfg.get("hand_dim", 6)
    
    window_size = dataset_cfg.get("window_size", 120)
    max_len = model_cfg.get("max_len", window_size + 100)
    
    if arch == "transformer":
        model = Stage1HandFlowMatching(
            bps_dim=bps_dim,
            centroid_dim=centroid_dim,
            object_feature_dim=object_feature_dim,
            hand_dim=hand_dim,
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nhead", 4),
            num_transformer_layers=model_cfg.get("num_layers", 4),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.1),
            max_len=max_len,
            encoder_hidden=model_cfg.get("encoder_hidden", 512),
            encoder_layers=model_cfg.get("encoder_layers", 3),
        )
    else:
        model = Stage1HandFlowMatchingMLP(
            bps_dim=bps_dim,
            centroid_dim=centroid_dim,
            object_feature_dim=object_feature_dim,
            hand_dim=hand_dim,
        )
    
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    # Normalization stats
    norm_stats = ckpt.get("norm_stats", {})
    hand_mean = norm_stats.get("hand_mean")
    hand_std = norm_stats.get("hand_std")
    
    if hand_mean is not None:
        hand_mean = hand_mean.to(device)
        hand_std = hand_std.to(device)
    
    return model, hand_mean, hand_std, config


@torch.no_grad()
def sample_flow_matching(
    model: torch.nn.Module,
    bps_encoding: torch.Tensor,
    object_centroid: torch.Tensor,
    num_steps: int = 50,
    solver: str = "euler",
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Sample hand positions using ODE integration.
    
    Args:
        model: Stage 1 FM model
        bps_encoding: (1, T, 3072)
        object_centroid: (1, T, 3)
        num_steps: number of ODE integration steps
        solver: "euler", "midpoint", or "rk4"
        num_samples: number of samples to generate
    
    Returns:
        (num_samples, T, 6) sampled hand positions (normalized)
    """
    device = next(model.parameters()).device
    T_seq = object_centroid.shape[1]
    
    # Repeat conditions for multiple samples
    if num_samples > 1:
        bps = bps_encoding.repeat(num_samples, 1, 1)
        centroid = object_centroid.repeat(num_samples, 1, 1)
    else:
        bps = bps_encoding
        centroid = object_centroid
    
    # Define model function for ODE solver
    def model_fn(x_t, t):
        return model(x_t, t, bps, centroid)
    
    # Start from noise (t=1)
    x_init = torch.randn(num_samples, T_seq, 6, device=device)
    
    # Solve ODE from t=1 (noise) to t=0 (data)
    ode_solver = get_ode_solver(solver)
    x_final = ode_solver(model_fn, x_init, num_steps=num_steps)
    
    return x_final


def denormalize_hands(hands: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
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
    data: Dict[str, Any],
    hand_mean: Optional[torch.Tensor],
    hand_std: Optional[torch.Tensor],
    device: torch.device,
    num_steps: int = 50,
    solver: str = "euler",
    apply_constraints: bool = True,
    contact_threshold: float = 0.03,
    partial_motion_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Process a single sequence: sample hands and apply contact constraints."""
    T_original = data["object_centroid"].shape[0]
    partial = False
    target_len = None
    
    bps_enc = data["bps_encoding"]
    centroid = data["object_centroid"]
    obj_verts = data.get("object_verts")
    obj_rot = data.get("object_rotation")
    
    if partial_motion_length is not None and partial_motion_length > 0:
        if partial_motion_length < T_original:
            bps_enc = bps_enc[:partial_motion_length]
            centroid = centroid[:partial_motion_length]
            if obj_verts is not None:
                obj_verts = obj_verts[:partial_motion_length]
            if obj_rot is not None:
                obj_rot = obj_rot[:partial_motion_length]
            partial = True
            target_len = partial_motion_length
    
    # Prepare inputs
    bps = torch.from_numpy(bps_enc).float().to(device)
    centroid_t = torch.from_numpy(centroid).float().to(device)
    
    if bps.ndim == 3 and bps.shape[-1] == 3:
        bps = bps.reshape(bps.shape[0], -1)
    
    bps = bps.unsqueeze(0)
    centroid_t = centroid_t.unsqueeze(0)
    
    # Sample hand positions via ODE
    hands_norm = sample_flow_matching(
        model, bps, centroid_t,
        num_steps=num_steps,
        solver=solver,
        num_samples=1,
    )
    
    # Denormalize
    hands_raw = denormalize_hands(hands_norm, hand_mean, hand_std)
    hands_raw = hands_raw.squeeze(0).cpu().numpy()
    
    result = {
        "hands_raw": hands_raw,
        "partial": partial,
        "target_len": target_len,
        "original_len": T_original,
    }
    
    # Apply contact constraints
    if apply_constraints and obj_verts is not None and obj_rot is not None:
        processor = ContactConstraintProcessor(contact_threshold=contact_threshold)
        hands_rect, metadata = processor.process(hands_raw, obj_verts, obj_rot)
        result["hands_rectified"] = hands_rect
        result["contact_metadata"] = metadata
    else:
        result["hands_rectified"] = hands_raw
        result["contact_metadata"] = None
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 1 FM Sampling")
    parser.add_argument("--config_path", type=str, default="./config/sample_stage1_fm.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    
    yml = load_config(args.config_path)
    sample_yml = yml["sample"]
    dataset_yml = yml["dataset"]

    root_dir = yml["root_dir"]
    ckpt_path = sample_yml["ckpt_path"]
    device = sample_yml["device"]
    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    num_samples = sample_yml.get("num_samples", None)
    seed = sample_yml.get("seed", 42)
    num_steps = sample_yml.get("num_inference_steps", 50)
    solver = sample_yml.get("ode_solver", "euler")
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
        sample_folder = f"fm_{solver}{num_steps}_w{window_size}_s{stride}_{timestamp}{suffix}"
        
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    else:
        output_dir = f"./out/stage1_fm_{timestamp}{suffix}"

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    torch.manual_seed(seed)

    # Save config
    import yaml
    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(yml, f, default_flow_style=False)

    # Load model
    model, hand_mean, hand_std, config = load_checkpoint(ckpt_path, device)

    # Find input files
    files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if num_samples:
        files = files[:num_samples]

    print(f"Processing {len(files)} files")
    print(f"ODE solver: {solver}, steps: {num_steps}")
    print(f"Output directory: {output_dir}")

    # Metrics accumulators
    all_metrics = []
    all_hand_jpe = []
    
    total_time = 0.0
    sample_durations = []
    all_frame_counts = []

    for fpath in tqdm(files, desc="Processing"):
        fname = os.path.basename(fpath)

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        if "bps_encoding" not in data or "object_centroid" not in data:
            print(f"  Skipping {fname}: missing object data")
            continue

        start_time = time.perf_counter()
        
        result = process_sequence(
            model, data,
            hand_mean, hand_std, device,
            num_steps=num_steps,
            solver=solver,
            apply_constraints=apply_constraints,
            contact_threshold=contact_threshold,
            partial_motion_length=partial_motion_length,
        )
        
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        
        fps = data.get("fps", 30.0)
        num_frames = result["hands_raw"].shape[0]
        all_frame_counts.append(num_frames)
        sample_durations.append(num_frames / fps)

        # Evaluate against GT
        if "hand_positions" in data:
            gt_hands = data["hand_positions"]
            pred_hands = result["hands_rectified"]

            jpe = compute_hand_jpe(pred_hands, gt_hands)
            all_hand_jpe.append(jpe)

            if "object_verts" in data:
                metrics = compute_contact_metrics(
                    pred_hands, gt_hands, data["object_verts"],
                    contact_threshold=0.05,
                )
                all_metrics.append(metrics)

        # Save result
        output_data = {
            "seq_name": data.get("seq_name", fname),
            "hands_raw": result["hands_raw"],
            "hands_rectified": result["hands_rectified"],
            "contact_metadata": result["contact_metadata"],
            "generative_model": "flow_matching",
            "ode_solver": solver,
            "num_inference_steps": num_steps,
        }

        if "hand_positions" in data:
            output_data["hands_gt"] = data["hand_positions"]

        out_path = os.path.join(output_dir, fname)
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)

    # Print evaluation summary
    if all_hand_jpe:
        print("\n" + "=" * 50)
        print("Evaluation Results (Flow Matching):")
        print(f"  Hand JPE (cm): {np.mean(all_hand_jpe):.2f} ± {np.std(all_hand_jpe):.2f}")

        if all_metrics:
            avg_prec = np.mean([m["precision"] for m in all_metrics])
            avg_rec = np.mean([m["recall"] for m in all_metrics])
            avg_f1 = np.mean([m["f1"] for m in all_metrics])
            print(f"  Contact Precision: {avg_prec:.3f}")
            print(f"  Contact Recall: {avg_rec:.3f}")
            print(f"  Contact F1: {avg_f1:.3f}")
        print("=" * 50)

    # Performance summary
    print("\n" + "=" * 50)
    print("Performance Summary:")
    print(f"  Total files: {len(files)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  ODE solver: {solver}, steps: {num_steps}")
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
        "Performance Summary (Flow Matching)",
        "=" * 50,
        f"Total files: {len(files)}",
        f"Total time: {total_time:.2f}s",
        f"ODE solver: {solver}, steps: {num_steps}",
    ]
    if files:
        summary_lines.extend([
            f"Average time: {total_time / len(files) * 1000:.2f}ms per sample",
            f"Throughput: {len(files) / total_time:.2f} samples/sec",
        ])
    if sample_durations:
        summary_lines.append(f"Average sample duration: {np.mean(sample_durations):.2f}s")
    if all_frame_counts:
        summary_lines.append(f"Frames per motion: {int(np.mean(all_frame_counts))} avg")
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        summary_lines.append(f"Generated fps: {gen_fps:.1f}")
    
    if all_hand_jpe:
        summary_lines.extend([
            "",
            "Evaluation Results",
            f"Hand JPE (cm): {np.mean(all_hand_jpe):.2f} ± {np.std(all_hand_jpe):.2f}",
        ])
        if all_metrics:
            avg_prec = np.mean([m["precision"] for m in all_metrics])
            avg_rec = np.mean([m["recall"] for m in all_metrics])
            avg_f1 = np.mean([m["f1"] for m in all_metrics])
            summary_lines.extend([
                f"Contact Precision: {avg_prec:.3f}",
                f"Contact Recall: {avg_rec:.3f}",
                f"Contact F1: {avg_f1:.3f}",
            ])
    
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
