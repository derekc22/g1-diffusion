"""
End-to-End OMOMO Pipeline: Object Motion → Full-Body Motion

Complete pipeline that:
1. Stage 1: Generates hand positions from object geometry
2. Apply contact constraints to hand positions
3. Stage 2: Generates full-body motion from hand positions

Usage:
    python sample_stage2.py
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple

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
from models.stage2_diffusion import Stage2TransformerModel, Stage2MLPModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.contact_constraints import apply_contact_constraints, ContactConstraintProcessor
from utils.rotation import rot6d_to_quat_xyzw, mat_to_quat_xyzw
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


class OmomoPipeline:
    """
    Complete OMOMO-style pipeline for object-guided motion synthesis.
    
    Pipeline:
        Object geometry (BPS, centroid) 
            → Stage 1 (diffusion) 
            → Raw hand positions
            → Contact constraints
            → Rectified hand positions
            → Stage 2 (diffusion)
            → Full-body robot motion
    """
    
    def __init__(
        self,
        stage1_ckpt: str,
        stage2_ckpt: str,
        device: str = "cuda:0",
        contact_threshold: float = 0.03,
    ):
        self.device = torch.device(device)
        self.contact_threshold = contact_threshold
        
        # Load Stage 1
        self.stage1_model, self.stage1_schedule, self.stage1_norm, stage1_max_len = self._load_stage1(stage1_ckpt)
        
        # Load Stage 2  
        self.stage2_model, self.stage2_schedule, self.stage2_norm, stage2_max_len = self._load_stage2(stage2_ckpt)
        
        # Store max_len (use minimum of both stages for safety)
        self.max_len = min(stage1_max_len, stage2_max_len)
        
        # Contact processor
        self.contact_processor = ContactConstraintProcessor(contact_threshold=contact_threshold)
    
    def _load_stage1(self, ckpt_path: str) -> Tuple[torch.nn.Module, DiffusionSchedule, dict, int]:
        """Load Stage 1 model and schedule."""
        print(f"Loading Stage 1: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        
        # Get max_len from config - critical for matching checkpoint
        window_size = dataset_cfg.get("window_size", 120)
        max_len = model_cfg.get("max_len", window_size + 100)
        
        if arch == "transformer":
            model = Stage1HandDiffusion(
                bps_dim=model_cfg.get("bps_dim", 3072),
                centroid_dim=model_cfg.get("centroid_dim", 3),
                object_feature_dim=model_cfg.get("object_feature_dim", 256),
                hand_dim=model_cfg.get("hand_dim", 6),
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
                bps_dim=model_cfg.get("bps_dim", 3072),
                centroid_dim=model_cfg.get("centroid_dim", 3),
                object_feature_dim=model_cfg.get("object_feature_dim", 256),
                hand_dim=6,
            )
        
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        
        timesteps = config.get("train", {}).get("timesteps", 1000)
        schedule = DiffusionSchedule(
            DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
        ).to(self.device)
        
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "hand_mean": norm_stats.get("hand_mean"),
            "hand_std": norm_stats.get("hand_std"),
        }
        if norm["hand_mean"] is not None:
            norm["hand_mean"] = norm["hand_mean"].to(self.device)
            norm["hand_std"] = norm["hand_std"].to(self.device)
        
        return model, schedule, norm, max_len
    
    def _load_stage2(self, ckpt_path: str) -> Tuple[torch.nn.Module, DiffusionSchedule, dict, int]:
        """Load Stage 2 model and schedule."""
        print(f"Loading Stage 2: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        
        # Get max_len from config - critical for matching checkpoint
        window_size = dataset_cfg.get("window_size", 120)
        max_len = model_cfg.get("max_len", window_size + 100)
        
        # Infer state_dim from checkpoint
        state_dict = ckpt["model"]
        # Find state_dim from output projection layer
        if "out_proj.weight" in state_dict:
            state_dim = state_dict["out_proj.weight"].shape[0]
        else:
            # Fallback: try to get from config or use default
            state_dim = model_cfg.get("state_dim", 38)  # 3+6+29 for G1
        
        cond_dim = 6  # Hand positions
        
        if arch == "transformer":
            model = Stage2TransformerModel(
                state_dim=state_dim,
                cond_dim=cond_dim,
                d_model=model_cfg.get("d_model", 256),
                nhead=model_cfg.get("nhead", 4),
                num_layers=model_cfg.get("num_layers", 4),
                dim_feedforward=model_cfg.get("dim_feedforward", 512),
                dropout=model_cfg.get("dropout", 0.1),
                max_len=max_len,
            )
        else:
            model = Stage2MLPModel(
                state_dim=state_dim,
                cond_dim=cond_dim,
            )
        
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        
        timesteps = config.get("train", {}).get("timesteps", 1000)
        schedule = DiffusionSchedule(
            DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
        ).to(self.device)
        
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "state_mean": norm_stats.get("state_mean"),
            "state_std": norm_stats.get("state_std"),
        }
        if norm["state_mean"] is not None:
            norm["state_mean"] = norm["state_mean"].to(self.device)
            norm["state_std"] = norm["state_std"].to(self.device)
        
        return model, schedule, norm, max_len
    
    @torch.no_grad()
    def _sample_ddpm(
        self,
        model: torch.nn.Module,
        schedule: DiffusionSchedule,
        cond: torch.Tensor,
        output_dim: int,
        cond_fn: callable = None,
    ) -> torch.Tensor:
        """Generic DDPM sampling."""
        B, T, _ = cond.shape
        x = torch.randn(B, T, output_dim, device=self.device)
        
        for n in reversed(range(schedule.timesteps)):
            t = torch.full((B,), n, device=self.device, dtype=torch.long)
            
            # Predict x0
            if cond_fn is not None:
                x0_pred = cond_fn(model, x, t, cond)
            else:
                x0_pred = model(x, t, cond)
            
            if n > 0:
                alpha_bar_t = schedule.alpha_bar[n]
                alpha_bar_t_prev = schedule.alpha_bar[n - 1]
                alpha_t = schedule.alpha[n]
                
                mean = (
                    torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred +
                    torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
                )
                
                sigma = torch.sqrt(schedule.beta[n])
                x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        
        return x
    
    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor."""
        if mean is None:
            return x
        if x.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return x * std + mean
    
    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Normalize tensor."""
        if mean is None:
            return x
        if x.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return (x - mean) / std
    
    def generate(
        self,
        bps_encoding: np.ndarray,
        object_centroid: np.ndarray,
        object_verts: Optional[np.ndarray] = None,
        object_rotation: Optional[np.ndarray] = None,
        apply_constraints: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate full-body motion from object geometry.
        
        Args:
            bps_encoding: (T, 1024, 3) or (T, 3072) BPS encoding
            object_centroid: (T, 3) object centroid trajectory
            object_verts: (T, K, 3) object vertices (for contact constraints)
            object_rotation: (T, 3, 3) object rotations (for contact constraints)
            apply_constraints: Whether to apply contact constraints
        
        Returns:
            Dict with:
                - hands_raw: (T, 6) raw hand positions from Stage 1
                - hands_rectified: (T, 6) contact-constrained hands
                - state: (T, D) full-body state from Stage 2
                - root_pos: (T, 3)
                - root_rot: (T, 4) quaternion xyzw
                - dof_pos: (T, Dq)
                - truncated: bool, True if sequence was truncated
                - original_len: int, original sequence length before truncation
        """
        T_original = object_centroid.shape[0]
        truncated = False
        
        # Truncate if sequence exceeds max_len (positional encoding limit)
        if T_original > self.max_len:
            print(f"  Warning: Truncating sequence from {T_original} to {self.max_len} frames")
            bps_encoding = bps_encoding[:self.max_len]
            object_centroid = object_centroid[:self.max_len]
            if object_verts is not None:
                object_verts = object_verts[:self.max_len]
            if object_rotation is not None:
                object_rotation = object_rotation[:self.max_len]
            truncated = True
        
        T_seq = object_centroid.shape[0]
        
        # Prepare BPS
        bps = torch.from_numpy(bps_encoding).float().to(self.device)
        if bps.ndim == 3 and bps.shape[-1] == 3:
            bps = bps.reshape(T_seq, -1)
        bps = bps.unsqueeze(0)  # (1, T, 3072)
        
        centroid = torch.from_numpy(object_centroid).float().to(self.device).unsqueeze(0)  # (1, T, 3)
        
        # =====================================================================
        # Stage 1: Object → Hand positions
        # =====================================================================
        def stage1_forward(model, x, t, _):
            return model(x, t, bps, centroid)
        
        hands_norm = self._sample_ddpm(
            self.stage1_model,
            self.stage1_schedule,
            bps,  # Not used directly, passed via closure
            output_dim=6,
            cond_fn=stage1_forward,
        )
        
        hands_raw = self._denormalize(
            hands_norm,
            self.stage1_norm["hand_mean"],
            self.stage1_norm["hand_std"],
        )
        hands_raw_np = hands_raw.squeeze(0).cpu().numpy()  # (T, 6)
        
        # =====================================================================
        # Apply Contact Constraints
        # =====================================================================
        if apply_constraints and object_verts is not None and object_rotation is not None:
            hands_rect_np, contact_meta = self.contact_processor.process(
                hands_raw_np, object_verts, object_rotation
            )
        else:
            hands_rect_np = hands_raw_np
            contact_meta = None
        
        # =====================================================================
        # Stage 2: Hand positions → Full-body motion
        # =====================================================================
        hands_rect = torch.from_numpy(hands_rect_np).float().to(self.device).unsqueeze(0)  # (1, T, 6)
        
        # Get state dimension
        # Try to infer from model
        if hasattr(self.stage2_model, 'state_dim'):
            state_dim = self.stage2_model.state_dim
        else:
            # Infer from output projection
            state_dim = self.stage2_model.out_proj.out_features
        
        state_norm = self._sample_ddpm(
            self.stage2_model,
            self.stage2_schedule,
            hands_rect,
            output_dim=state_dim,
        )
        
        state = self._denormalize(
            state_norm,
            self.stage2_norm["state_mean"],
            self.stage2_norm["state_std"],
        )
        state_np = state.squeeze(0).cpu().numpy()  # (T, D)
        
        # Parse state into components
        root_pos = state_np[:, :3]
        root_rot_6d = state_np[:, 3:9]
        dof_pos = state_np[:, 9:]
        
        # Convert 6D rotation to quaternion
        root_rot_6d_t = torch.from_numpy(root_rot_6d).float()
        root_rot_quat = rot6d_to_quat_xyzw(root_rot_6d_t).numpy()
        
        return {
            "hands_raw": hands_raw_np,
            "hands_rectified": hands_rect_np,
            "contact_metadata": contact_meta,
            "state": state_np,
            "root_pos": root_pos,
            "root_rot": root_rot_quat,
            "dof_pos": dof_pos,
            "truncated": truncated,
            "original_len": T_original,
        }


def main():
    yml = load_config("./config/sample_stage2.yaml")
    sample_yml = yml["sample"]

    root_dir = yml["root_dir"]
    stage1_ckpt = sample_yml["stage1_ckpt"]
    stage2_ckpt = sample_yml["stage2_ckpt"]
    device = sample_yml["device"]
    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    num_samples = sample_yml.get("num_samples", None)
    seed = sample_yml.get("seed", 42)
    timesteps = sample_yml.get("timesteps", 1000)

    if not stage1_ckpt or not stage2_ckpt:
        raise ValueError("stage1_ckpt and stage2_ckpt must be set in config/sample_stage2.yaml")

    # Derive output directory from stage2 checkpoint path
    # Expected format: logs/<log_id>/checkpoints/<ckpt_file>
    # Output to: logs/<log_id>/samples/<timestamp>/
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    # Extract log_id from checkpoint path
    ckpt_parts = stage2_ckpt.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        
        # Create sample folder name with config info
        dataset_yml = yml.get("dataset", {})
        window_size = dataset_yml.get("window_size", 120)
        stride = dataset_yml.get("stride", 10)
        sample_folder = f"ts{timesteps}_w{window_size}_s{stride}_{timestamp}"
        
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    # else:
    #     # Fallback to config output_dir if checkpoint path doesn't match expected pattern
    #     output_dir = sample_yml.get("output_dir", "./out/stage2")

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    # Save config to samples directory for reproducibility
    # This makes it clear which stage1 and stage2 checkpoints were used
    import yaml
    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(yml, f, default_flow_style=False)

    # Create pipeline
    pipeline = OmomoPipeline(
        stage1_ckpt=stage1_ckpt,
        stage2_ckpt=stage2_ckpt,
        device=device,
        contact_threshold=contact_threshold,
    )

    # Find input files
    files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if num_samples:
        files = files[:num_samples]

    print(f"\nProcessing {len(files)} files")
    print(f"Output directory: {output_dir}")

    for fpath in tqdm(files, desc="Generating"):
        fname = os.path.basename(fpath)

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        # Check required keys
        if "bps_encoding" not in data or "object_centroid" not in data:
            print(f"  Skipping {fname}: missing object data")
            continue

        # Generate
        result = pipeline.generate(
            bps_encoding=data["bps_encoding"],
            object_centroid=data["object_centroid"],
            object_verts=data.get("object_verts"),
            object_rotation=data.get("object_rotation"),
            apply_constraints=apply_constraints,
        )

        # Save
        output_data = {
            "seq_name": data.get("seq_name", fname),
            "fps": data.get("fps", 30.0),
            **result,
        }

        # Add hand_positions for visualization compatibility
        # (plotting script expects 'hand_positions' key)
        if "hands_rectified" in result:
            output_data["hand_positions"] = result["hands_rectified"]

        # Add object data for visualization compatibility
        # object_pos: (T, 3) from object_centroid
        if "object_centroid" in data:
            output_data["object_pos"] = data["object_centroid"]
        
        # object_rot: (T, 4) quaternion in xyzw format from (T, 3, 3) rotation matrix
        if "object_rotation" in data:
            obj_rot_mat = data["object_rotation"]  # (T, 3, 3)
            obj_rot_mat_t = torch.from_numpy(obj_rot_mat).float()
            obj_rot_quat = mat_to_quat_xyzw(obj_rot_mat_t).numpy()  # (T, 4) xyzw
            output_data["object_rot"] = obj_rot_quat
        
        # Copy local_body_pos and link_body_list from source data
        # (Required for visualization but cannot be computed without FK)
        if "local_body_pos" in data:
            output_data["local_body_pos"] = data["local_body_pos"]
        if "link_body_list" in data:
            output_data["link_body_list"] = data["link_body_list"]
        
        # source_start: frame index in original source for alignment in comparison plots
        output_data["source_start"] = 0

        # Optionally include GT for comparison
        if "root_pos" in data:
            output_data["gt_root_pos"] = data["root_pos"]
            output_data["gt_root_rot"] = data["root_rot"]
            output_data["gt_dof_pos"] = data["dof_pos"]
        if "hand_positions" in data:
            output_data["gt_hands"] = data["hand_positions"]

        # Output filename: {source_name}_sample_000.pkl
        # This format is required for plot_robot_motion_compare_w_object.py
        # which uses regex to match model outputs to ground truth files
        fname_base = os.path.splitext(fname)[0]
        out_fname = f"{fname_base}.pkl"
        out_path = os.path.join(output_dir, out_fname)
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
