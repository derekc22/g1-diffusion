"""
Stage 1 Sampling Script: Object Geometry → Hand Positions

Generates hand positions from object motion using the trained Stage 1 model,
then applies contact constraints for physically plausible results.

Usage:
    python sample_stage1.py
"""

import os
import sys
from typing import Optional, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm
import types

from models.stage1_diffusion import Stage1HandDiffusion, Stage1HandDiffusionMLP
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
    """Load Stage 1 checkpoint and reconstruct model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    config = ckpt["config"]
    arch = config.get("train", {}).get("architecture", "transformer")
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    
    # Get dimensions from config or use defaults
    bps_dim = model_cfg.get("bps_dim", 3072)
    centroid_dim = model_cfg.get("centroid_dim", 3)
    object_feature_dim = model_cfg.get("object_feature_dim", 256)
    hand_dim = model_cfg.get("hand_dim", 6)
    
    # Get max_len from config - this is critical for matching checkpoint
    window_size = dataset_cfg.get("window_size", 120)
    max_len = model_cfg.get("max_len", window_size + 100)
    
    if arch == "transformer":
        model = Stage1HandDiffusion(
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
        model = Stage1HandDiffusionMLP(
            bps_dim=bps_dim,
            centroid_dim=centroid_dim,
            object_feature_dim=object_feature_dim,
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
    
    # Create diffusion schedule
    train_cfg = config.get("train", {})
    timesteps = train_cfg.get("timesteps", 1000)
    schedule = DiffusionSchedule(
        DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
    ).to(device)
    
    return model, schedule, hand_mean, hand_std, config


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    bps_encoding: torch.Tensor,
    object_centroid: torch.Tensor,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Sample hand positions using DDPM reverse process.
    
    Args:
        model: Stage 1 model
        schedule: Diffusion schedule
        bps_encoding: (1, T, 3072) or (1, T, 1024, 3)
        object_centroid: (1, T, 3)
        num_samples: Number of samples to generate
    
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
    
    # Start from random noise
    x = torch.randn(num_samples, T_seq, 6, device=device)
    
    # Reverse diffusion
    for n in reversed(range(schedule.timesteps)):
        t = torch.full((num_samples,), n, device=device, dtype=torch.long)
        
        # Predict x0
        x0_pred = model(x, t, bps, centroid)
        
        if n > 0:
            # Compute mean for x_{n-1}
            alpha_bar_t = schedule.alpha_bar[n]
            alpha_bar_t_prev = schedule.alpha_bar[n - 1]
            alpha_t = schedule.alpha[n]
            
            # DDPM sampling
            mean = (
                torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred +
                torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
            )
            
            # Add noise
            beta_t = schedule.beta[n]
            sigma = torch.sqrt(beta_t)
            noise = torch.randn_like(x)
            x = mean + sigma * noise
        else:
            x = x0_pred
    
    return x


def denormalize_hands(hands: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize hand positions."""
    if mean is None:
        return hands
    if hands.ndim == 3:  # (B, T, 6)
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    elif hands.ndim == 2:  # (T, 6)
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
) -> Dict[str, np.ndarray]:
    """
    Process a single sequence: sample hands and apply contact constraints.
    
    Args:
        model: Stage 1 model
        schedule: Diffusion schedule  
        data: Dict with bps_encoding, object_centroid, object_verts, object_rotation
        hand_mean, hand_std: Normalization stats
        device: Torch device
        apply_constraints: Whether to apply contact constraints
        contact_threshold: Contact detection threshold
    
    Returns:
        Dict with raw_hands, rectified_hands, contact_metadata
    """
    # Prepare inputs
    bps = torch.from_numpy(data["bps_encoding"]).float().to(device)
    centroid = torch.from_numpy(data["object_centroid"]).float().to(device)
    
    # Handle BPS shape
    if bps.ndim == 3 and bps.shape[-1] == 3:  # (T, 1024, 3)
        bps = bps.reshape(bps.shape[0], -1)  # (T, 3072)
    
    # Add batch dimension
    bps = bps.unsqueeze(0)  # (1, T, 3072)
    centroid = centroid.unsqueeze(0)  # (1, T, 3)
    
    # Sample hand positions
    hands_norm = sample_ddpm(model, schedule, bps, centroid, num_samples=1)  # (1, T, 6)
    
    # Denormalize
    hands_raw = denormalize_hands(hands_norm, hand_mean, hand_std)
    hands_raw = hands_raw.squeeze(0).cpu().numpy()  # (T, 6)
    
    result = {
        "hands_raw": hands_raw,
    }
    
    # Apply contact constraints if object data available
    if apply_constraints and "object_verts" in data and "object_rotation" in data:
        obj_verts = data["object_verts"]
        obj_rot = data["object_rotation"]
        
        processor = ContactConstraintProcessor(contact_threshold=contact_threshold)
        hands_rect, metadata = processor.process(hands_raw, obj_verts, obj_rot)
        
        result["hands_rectified"] = hands_rect
        result["contact_metadata"] = metadata
    else:
        result["hands_rectified"] = hands_raw
        result["contact_metadata"] = None
    
    return result


def main():
    yml = load_config("./config/sample_stage1.yaml")
    sample_yml = yml["sample"]
    dataset_yml = yml["dataset"]

    root_dir = yml["root_dir"]
    ckpt_path = sample_yml["ckpt_path"]
    device = sample_yml["device"]
    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    num_samples = sample_yml.get("num_samples", None)
    evaluate = sample_yml.get("evaluate", False)
    seed = sample_yml.get("seed", 42)
    timesteps = sample_yml.get("timesteps", 1000)

    if not ckpt_path:
        raise ValueError("ckpt_path must be set in config/sample_stage1.yaml")

    # Derive output directory from checkpoint path
    # Expected format: logs/<log_id>/checkpoints/<ckpt_file>
    # Output to: logs/<log_id>/samples/<timestamp>/
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    # Extract log_id from checkpoint path
    ckpt_parts = ckpt_path.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        
        # Create sample folder name with config info
        window_size = dataset_yml.get("window_size", 120)
        stride = dataset_yml.get("stride", 10)
        sample_folder = f"ts{timesteps}_w{window_size}_s{stride}_{timestamp}"
        
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    # else:
    #     # Fallback to config output_dir if checkpoint path doesn't match expected pattern
    #     output_dir = sample_yml.get("output_dir", "./out/stage1")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
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
    all_metrics = []
    all_hand_jpe = []

    for fpath in tqdm(files, desc="Processing"):
        fname = os.path.basename(fpath)

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        # Check required keys
        if "bps_encoding" not in data or "object_centroid" not in data:
            print(f"  Skipping {fname}: missing object data")
            continue

        # Process
        result = process_sequence(
            model, schedule, data,
            hand_mean, hand_std, device,
            apply_constraints=apply_constraints,
            contact_threshold=contact_threshold,
        )

        # Evaluate against GT if available
        if evaluate and "hand_positions" in data:
            gt_hands = data["hand_positions"]
            pred_hands = result["hands_rectified"]

            # Hand JPE
            jpe = compute_hand_jpe(pred_hands, gt_hands)
            all_hand_jpe.append(jpe)

            # Contact metrics (if object verts available)
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
        }

        # Optionally include GT for comparison
        if "hand_positions" in data:
            output_data["hands_gt"] = data["hand_positions"]

        out_path = os.path.join(output_dir, fname)
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)

    # Print evaluation summary
    if evaluate and all_hand_jpe:
        print("\n" + "=" * 50)
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

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
