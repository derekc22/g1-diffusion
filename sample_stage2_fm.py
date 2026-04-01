"""
End-to-End Flow Matching Pipeline: Object Motion → Full-Body Motion

Complete pipeline using flow matching for both stages:
1. Stage 1 FM: Object geometry → Hand positions (ODE integration)
2. Apply contact constraints to hand positions
3. Stage 2 FM: Hand positions → Full-body motion (ODE integration)

Output format is identical to sample_stage2.py for direct metric comparison.

Usage:
    python sample_stage2_fm.py --config_path ./config/sample_stage2_fm.yaml
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm
import types

from models.stage1_flow_matching import Stage1HandFlowMatching, Stage1HandFlowMatchingMLP
from models.stage2_flow_matching import Stage2FMTransformerModel, Stage2FMMLPModel
from utils.flow_matching import FlowMatchingConfig, get_ode_solver
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


class FlowMatchingPipeline:
    """
    Complete OMOMO-style pipeline using flow matching for both stages.
    
    Pipeline:
        Object geometry (BPS, centroid) 
            → Stage 1 FM (ODE)
            → Raw hand positions
            → Contact constraints
            → Rectified hand positions
            → Stage 2 FM (ODE)
            → Full-body robot motion
    
    Output format matches OmomoPipeline for direct comparison.
    """
    
    def __init__(
        self,
        stage1_ckpt_path: str,
        stage2_ckpt_path: str,
        device: str = "cuda:0",
        contact_threshold: float = 0.03,
        num_inference_steps: int = 50,
        ode_solver: str = "euler",
    ):
        self.device = torch.device(device)
        self.contact_threshold = contact_threshold
        self.num_inference_steps = num_inference_steps
        self.ode_solver = ode_solver
        self.solver_fn = get_ode_solver(ode_solver)
        
        # Load Stage 1
        self.stage1_model, self.stage1_norm, stage1_max_len = self._load_stage1(stage1_ckpt_path)
        
        # Load Stage 2
        self.stage2_model, self.stage2_norm, stage2_max_len = self._load_stage2(stage2_ckpt_path)
        
        self.max_len = min(stage1_max_len, stage2_max_len)
        
        # Contact processor
        self.contact_processor = ContactConstraintProcessor(contact_threshold=contact_threshold)
    
    def _load_stage1(self, ckpt_path: str) -> Tuple[torch.nn.Module, dict, int]:
        """Load Stage 1 FM model."""
        print(f"Loading Stage 1 FM: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        
        window_size = dataset_cfg.get("window_size", 120)
        max_len = model_cfg.get("max_len", window_size + 100)
        
        if arch == "transformer":
            model = Stage1HandFlowMatching(
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
            model = Stage1HandFlowMatchingMLP(
                bps_dim=model_cfg.get("bps_dim", 3072),
                centroid_dim=model_cfg.get("centroid_dim", 3),
                object_feature_dim=model_cfg.get("object_feature_dim", 256),
                hand_dim=6,
            )
        
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "hand_mean": norm_stats.get("hand_mean"),
            "hand_std": norm_stats.get("hand_std"),
        }
        if norm["hand_mean"] is not None:
            norm["hand_mean"] = norm["hand_mean"].to(self.device)
            norm["hand_std"] = norm["hand_std"].to(self.device)
        
        return model, norm, max_len
    
    def _load_stage2(self, ckpt_path: str) -> Tuple[torch.nn.Module, dict, int]:
        """Load Stage 2 FM model."""
        print(f"Loading Stage 2 FM: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        
        window_size = dataset_cfg.get("window_size", 120)
        max_len = model_cfg.get("max_len", window_size + 100)
        
        # Infer state_dim from checkpoint
        state_dict = ckpt["model"]
        if "out_proj.weight" in state_dict:
            state_dim = state_dict["out_proj.weight"].shape[0]
        else:
            state_dim = model_cfg.get("state_dim", 38)
        
        cond_dim = 6
        
        if arch == "transformer":
            model = Stage2FMTransformerModel(
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
            model = Stage2FMMLPModel(
                state_dim=state_dim,
                cond_dim=cond_dim,
            )
        
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "state_mean": norm_stats.get("state_mean"),
            "state_std": norm_stats.get("state_std"),
        }
        if norm["state_mean"] is not None:
            norm["state_mean"] = norm["state_mean"].to(self.device)
            norm["state_std"] = norm["state_std"].to(self.device)
        
        self.state_dim = state_dim
        return model, norm, max_len
    
    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if mean is None:
            return x
        if x.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return x * std + mean
    
    @torch.no_grad()
    def generate(
        self,
        bps_encoding: np.ndarray,
        object_centroid: np.ndarray,
        object_verts: Optional[np.ndarray] = None,
        object_rotation: Optional[np.ndarray] = None,
        apply_constraints: bool = True,
        partial_motion_length: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate full-body motion from object geometry using flow matching.
        
        Output format is identical to OmomoPipeline.generate() for direct comparison.
        """
        T_original = object_centroid.shape[0]
        truncated = False
        partial = False
        target_len = None
        
        # Handle partial motion
        if partial_motion_length is not None and partial_motion_length > 0:
            if partial_motion_length < T_original:
                bps_encoding = bps_encoding[:partial_motion_length]
                object_centroid = object_centroid[:partial_motion_length]
                if object_verts is not None:
                    object_verts = object_verts[:partial_motion_length]
                if object_rotation is not None:
                    object_rotation = object_rotation[:partial_motion_length]
                partial = True
                target_len = partial_motion_length
        
        T_working = object_centroid.shape[0]
        
        # Truncate if exceeds max_len
        if T_working > self.max_len:
            print(f"  Warning: Truncating sequence from {T_working} to {self.max_len} frames")
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
        bps = bps.unsqueeze(0)
        
        centroid = torch.from_numpy(object_centroid).float().to(self.device).unsqueeze(0)
        
        # =====================================================================
        # Stage 1: Object → Hand positions (ODE integration)
        # =====================================================================
        def stage1_fn(x_t, t):
            return self.stage1_model(x_t, t, bps, centroid)
        
        x_init_s1 = torch.randn(1, T_seq, 6, device=self.device)
        hands_norm = self.solver_fn(stage1_fn, x_init_s1, num_steps=self.num_inference_steps)
        
        hands_raw = self._denormalize(
            hands_norm,
            self.stage1_norm["hand_mean"],
            self.stage1_norm["hand_std"],
        )
        hands_raw_np = hands_raw.squeeze(0).cpu().numpy()
        
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
        # Stage 2: Hand positions → Full-body motion (ODE integration)
        # =====================================================================
        hands_rect = torch.from_numpy(hands_rect_np).float().to(self.device).unsqueeze(0)
        
        if hasattr(self.stage2_model, 'state_dim'):
            state_dim = self.stage2_model.state_dim
        else:
            state_dim = self.stage2_model.out_proj.out_features
        
        def stage2_fn(x_t, t):
            return self.stage2_model(x_t, t, hands_rect)
        
        x_init_s2 = torch.randn(1, T_seq, state_dim, device=self.device)
        state_norm = self.solver_fn(stage2_fn, x_init_s2, num_steps=self.num_inference_steps)
        
        state = self._denormalize(
            state_norm,
            self.stage2_norm["state_mean"],
            self.stage2_norm["state_std"],
        )
        state_np = state.squeeze(0).cpu().numpy()
        
        # Parse state into components
        root_pos = state_np[:, :3]
        root_rot_6d = state_np[:, 3:9]
        dof_pos = state_np[:, 9:]
        
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
            "partial": partial,
            "target_len": target_len,
        }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 FM End-to-End Sampling")
    parser.add_argument("--config_path", type=str, default="./config/sample_stage2_fm.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    
    yml = load_config(args.config_path)
    sample_yml = yml["sample"]

    root_dir = yml["root_dir"]
    stage1_ckpt_path = sample_yml["stage1_ckpt_path"]
    stage2_ckpt_path = sample_yml["stage2_ckpt_path"]
    device = sample_yml["device"]
    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    num_samples = sample_yml.get("num_samples", None)
    seed = sample_yml.get("seed", 42)
    num_inference_steps = sample_yml.get("num_inference_steps", 50)
    ode_solver = sample_yml.get("ode_solver", "euler")
    partial_motion_length = sample_yml.get("partial_motion_length", None)

    if not stage1_ckpt_path or not stage2_ckpt_path:
        raise ValueError("stage1_ckpt_path and stage2_ckpt_path must be set")

    # Derive output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    exp_name = yml.get("exp_name", "")
    suffix = f"_{exp_name}" if exp_name else ""
    
    ckpt_parts = stage2_ckpt_path.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        
        dataset_yml = yml.get("dataset", {})
        window_size = dataset_yml.get("window_size", 120)
        stride = dataset_yml.get("stride", 10)
        sample_folder = f"fm_{ode_solver}{num_inference_steps}_w{window_size}_s{stride}_{timestamp}{suffix}"
        
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    else:
        output_dir = f"./out/e2e_fm_{timestamp}{suffix}"

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    # Save config
    import yaml
    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(yml, f, default_flow_style=False)

    # Create pipeline
    pipeline = FlowMatchingPipeline(
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        device=device,
        contact_threshold=contact_threshold,
        num_inference_steps=num_inference_steps,
        ode_solver=ode_solver,
    )

    # Find input files
    files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if num_samples:
        files = files[:num_samples]

    print(f"\nProcessing {len(files)} files")
    print(f"ODE solver: {ode_solver}, steps: {num_inference_steps}")
    print(f"Output directory: {output_dir}")
    
    total_time = 0.0
    sample_durations = []
    all_frame_counts = []

    for fpath in tqdm(files, desc="Generating"):
        fname = os.path.basename(fpath)

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        if "bps_encoding" not in data or "object_centroid" not in data:
            print(f"  Skipping {fname}: missing object data")
            continue

        start_time = time.perf_counter()
        result = pipeline.generate(
            bps_encoding=data["bps_encoding"],
            object_centroid=data["object_centroid"],
            object_verts=data.get("object_verts"),
            object_rotation=data.get("object_rotation"),
            apply_constraints=apply_constraints,
            partial_motion_length=partial_motion_length,
        )
        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        # Save — same output format as DDPM pipeline for comparison
        output_data = {
            "seq_name": data.get("seq_name", fname),
            "fps": data.get("fps", 30.0),
            "generative_model": "flow_matching",
            "ode_solver": ode_solver,
            "num_inference_steps": num_inference_steps,
            **result,
        }
        
        fps = output_data["fps"]
        num_frames = result.get("root_pos", result.get("dof_pos")).shape[0] if "root_pos" in result or "dof_pos" in result else 0
        if num_frames > 0:
            all_frame_counts.append(num_frames)
            sample_durations.append(num_frames / fps)

        # Visualization compatibility keys (same as DDPM sample_stage2.py)
        if "hands_rectified" in result:
            output_data["hand_positions"] = result["hands_rectified"]
        if "object_centroid" in data:
            output_data["object_pos"] = data["object_centroid"]
        if "object_rotation" in data:
            obj_rot_mat = data["object_rotation"]
            obj_rot_mat_t = torch.from_numpy(obj_rot_mat).float()
            obj_rot_quat = mat_to_quat_xyzw(obj_rot_mat_t).numpy()
            output_data["object_rot"] = obj_rot_quat
        if "local_body_pos" in data:
            output_data["local_body_pos"] = data["local_body_pos"]
        if "link_body_list" in data:
            output_data["link_body_list"] = data["link_body_list"]
        output_data["source_start"] = 0

        # GT for comparison
        if "root_pos" in data:
            output_data["gt_root_pos"] = data["root_pos"]
            output_data["gt_root_rot"] = data["root_rot"]
            output_data["gt_dof_pos"] = data["dof_pos"]
        if "hand_positions" in data:
            output_data["gt_hands"] = data["hand_positions"]

        fname_base = os.path.splitext(fname)[0]
        out_fname = f"{fname_base}.pkl"
        out_path = os.path.join(output_dir, out_fname)
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)

    # Performance summary
    num_files = len(files)
    avg_time = (total_time / num_files * 1000) if num_files > 0 else 0
    throughput = num_files / total_time if total_time > 0 else 0
    avg_duration = sum(sample_durations) / len(sample_durations) if sample_durations else 0
    
    print(f"\n{'='*50}")
    print("Performance Summary (Flow Matching)")
    print(f"{'='*50}")
    print(f"Total files: {num_files}")
    print(f"Total time: {total_time:.2f}s")
    print(f"ODE solver: {ode_solver}, steps: {num_inference_steps}")
    print(f"Average time: {avg_time:.1f}ms per sample")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Average sample duration: {avg_duration:.2f}s")
    if all_frame_counts:
        print(f"Frames per motion: {int(np.mean(all_frame_counts))} avg, {min(all_frame_counts)} min, {max(all_frame_counts)} max")
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        print(f"Generated fps: {gen_fps:.1f}")
    print(f"{'='*50}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Performance Summary (Flow Matching)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total files: {num_files}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"ODE solver: {ode_solver}, steps: {num_inference_steps}\n")
        f.write(f"Average time: {avg_time:.1f}ms per sample\n")
        f.write(f"Throughput: {throughput:.2f} samples/sec\n")
        f.write(f"Average sample duration: {avg_duration:.2f}s\n")
        if all_frame_counts:
            f.write(f"Frames per motion: {int(np.mean(all_frame_counts))} avg\n")
            total_frames = sum(all_frame_counts)
            gen_fps = total_frames / total_time if total_time > 0 else 0
            f.write(f"Generated fps: {gen_fps:.1f}\n")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
