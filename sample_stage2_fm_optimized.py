"""
Optimized Flow Matching Pipeline with Inference Optimizations

Complete pipeline with:
1. Stage 1 FM: Object geometry → Hand positions (optimized ODE)
2. Contact constraints
3. Stage 2 FM: Hand positions → Full-body motion (optimized ODE)

Supports:
- Mixed precision (FP16/BF16) for ~2x speedup
- torch.compile for graph optimization
- Configurable ODE solver (Euler/midpoint/RK4) and step count
- Warmup for consistent performance

Output format matches both DDPM and FM baselines for comparison.

Usage:
    python sample_stage2_fm_optimized.py --config_path ./config/sample_stage2_fm_optimized.yaml
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import argparse
import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm
import types
import time

from models.stage1_flow_matching import Stage1HandFlowMatching, Stage1HandFlowMatchingMLP
from models.stage2_flow_matching import Stage2FMTransformerModel, Stage2FMMLPModel
from utils.flow_matching import FlowMatchingConfig, get_ode_solver
from utils.inference_optimization import PrecisionMode
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


class OptimizedFlowMatchingPipeline:
    """
    Optimized flow matching pipeline with mixed precision and torch.compile.
    """
    
    def __init__(
        self,
        stage1_ckpt_path: str,
        stage2_ckpt_path: str,
        device: str = "cuda:0",
        contact_threshold: float = 0.03,
        num_inference_steps: int = 50,
        ode_solver: str = "euler",
        precision: str = "fp16",
        use_torch_compile: bool = True,
        compile_mode: str = "reduce-overhead",
        warmup_iterations: int = 3,
    ):
        self.device = torch.device(device)
        self.contact_threshold = contact_threshold
        self.num_inference_steps = num_inference_steps
        self.ode_solver = ode_solver
        self.solver_fn = get_ode_solver(ode_solver)
        self.warmup_iterations = warmup_iterations
        
        # Precision
        precision_mode = PrecisionMode(precision)
        if precision_mode == PrecisionMode.FP16:
            self.dtype = torch.float16
        elif precision_mode == PrecisionMode.BF16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        # Load Stage 1
        print("Loading Stage 1 FM...")
        self.stage1_model, self.stage1_norm, self.stage1_max_len = self._load_stage1(stage1_ckpt_path)
        
        # Load Stage 2
        print("Loading Stage 2 FM...")
        self.stage2_model, self.stage2_norm, self.stage2_max_len = self._load_stage2(stage2_ckpt_path)
        
        # Apply precision
        if self.dtype != torch.float32:
            self.stage1_model = self.stage1_model.to(self.dtype)
            self.stage2_model = self.stage2_model.to(self.dtype)
        
        # Apply torch.compile
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                self.stage1_model = torch.compile(
                    self.stage1_model, mode=compile_mode, fullgraph=False
                )
                self.stage2_model = torch.compile(
                    self.stage2_model, mode=compile_mode, fullgraph=False
                )
                print(f"Models compiled with mode='{compile_mode}'")
            except Exception as e:
                print(f"torch.compile failed: {e}")
        
        self.contact_processor = ContactConstraintProcessor(contact_threshold=contact_threshold)
        self._is_warmed_up = False
    
    def _load_stage1(self, ckpt_path: str):
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
    
    def _load_stage2(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        
        window_size = dataset_cfg.get("window_size", 120)
        max_len = model_cfg.get("max_len", window_size + 100)
        
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
    
    def warmup(self, seq_len: int = 120):
        """Warmup both stages for consistent performance."""
        print("Warming up models...")
        
        dummy_bps = torch.randn(1, seq_len, 3072, device=self.device, dtype=self.dtype)
        dummy_centroid = torch.randn(1, seq_len, 3, device=self.device, dtype=self.dtype)
        dummy_x1 = torch.randn(1, seq_len, 6, device=self.device, dtype=self.dtype)
        dummy_t = torch.rand(1, device=self.device, dtype=self.dtype)
        
        for _ in range(self.warmup_iterations):
            with torch.inference_mode():
                _ = self.stage1_model(dummy_x1, dummy_t, dummy_bps, dummy_centroid)
        
        dummy_cond = torch.randn(1, seq_len, 6, device=self.device, dtype=self.dtype)
        dummy_x2 = torch.randn(1, seq_len, self.state_dim, device=self.device, dtype=self.dtype)
        
        for _ in range(self.warmup_iterations):
            with torch.inference_mode():
                _ = self.stage2_model(dummy_x2, dummy_t, dummy_cond)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self._is_warmed_up = True
        print("Warmup complete")
    
    def _denormalize(self, x, mean, std):
        if mean is None:
            return x
        if x.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return x * std.to(x.dtype) + mean.to(x.dtype)
    
    @torch.inference_mode()
    def generate(
        self,
        bps_encoding: np.ndarray,
        object_centroid: np.ndarray,
        object_verts: Optional[np.ndarray] = None,
        object_rotation: Optional[np.ndarray] = None,
        apply_constraints: bool = True,
        partial_motion_length: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate full-body motion using optimized flow matching."""
        T_original = object_centroid.shape[0]
        truncated = False
        partial = False
        target_len = None
        
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
        max_len = min(self.stage1_max_len, self.stage2_max_len)
        
        if T_working > max_len:
            print(f"  Warning: Truncating sequence from {T_working} to {max_len} frames")
            bps_encoding = bps_encoding[:max_len]
            object_centroid = object_centroid[:max_len]
            if object_verts is not None:
                object_verts = object_verts[:max_len]
            if object_rotation is not None:
                object_rotation = object_rotation[:max_len]
            truncated = True
        
        T_seq = object_centroid.shape[0]
        
        # Prepare inputs
        bps = torch.from_numpy(bps_encoding).to(self.device, dtype=self.dtype)
        centroid = torch.from_numpy(object_centroid).to(self.device, dtype=self.dtype)
        
        if bps.ndim == 3 and bps.shape[-1] == 3:
            bps = bps.reshape(bps.shape[0], -1)
        bps = bps.unsqueeze(0)
        centroid = centroid.unsqueeze(0)
        
        # Stage 1: ODE integration
        def stage1_fn(x_t, t):
            return self.stage1_model(x_t, t, bps, centroid)
        
        x_init_s1 = torch.randn(1, T_seq, 6, device=self.device, dtype=self.dtype)
        hands_norm = self.solver_fn(stage1_fn, x_init_s1, num_steps=self.num_inference_steps)
        
        hands_raw = self._denormalize(
            hands_norm,
            self.stage1_norm["hand_mean"],
            self.stage1_norm["hand_std"],
        )
        hands_raw_np = hands_raw.squeeze(0).float().cpu().numpy()
        
        # Contact constraints
        if apply_constraints and object_verts is not None and object_rotation is not None:
            hands_rect_np, contact_metadata = self.contact_processor.process(
                hands_raw_np, object_verts, object_rotation
            )
            hands_rect = torch.from_numpy(hands_rect_np).to(self.device, dtype=self.dtype).unsqueeze(0)
        else:
            hands_rect = hands_raw
            hands_rect_np = hands_raw_np
            contact_metadata = None
        
        # Stage 2: ODE integration
        def stage2_fn(x_t, t):
            return self.stage2_model(x_t, t, hands_rect)
        
        x_init_s2 = torch.randn(1, T_seq, self.state_dim, device=self.device, dtype=self.dtype)
        state_norm = self.solver_fn(stage2_fn, x_init_s2, num_steps=self.num_inference_steps)
        
        state = self._denormalize(
            state_norm,
            self.stage2_norm["state_mean"],
            self.stage2_norm["state_std"],
        )
        state_np = state.squeeze(0).float().cpu().numpy()
        
        root_pos = state_np[:, :3]
        root_rot_6d = state_np[:, 3:9]
        dof_pos = state_np[:, 9:]
        
        root_rot_6d_t = torch.from_numpy(root_rot_6d).float()
        root_rot_quat = rot6d_to_quat_xyzw(root_rot_6d_t).numpy()
        
        return {
            "hands_raw": hands_raw_np,
            "hands_rectified": hands_rect_np,
            "contact_metadata": contact_metadata,
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
    parser = argparse.ArgumentParser(description="Optimized FM End-to-End Sampling")
    parser.add_argument("--config_path", type=str, default="./config/sample_stage2_fm_optimized.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    
    yml = load_config(args.config_path)
    
    sample_yml = yml["sample"]
    dataset_yml = yml.get("dataset", {})
    opt_yml = yml.get("optimization", {})
    
    root_dir = yml["root_dir"]
    stage1_ckpt_path = sample_yml["stage1_ckpt_path"]
    stage2_ckpt_path = sample_yml["stage2_ckpt_path"]
    device_str = sample_yml.get("device", "cuda")
    num_samples = sample_yml.get("num_samples", None)
    seed = sample_yml.get("seed", 42)
    partial_motion_length = sample_yml.get("partial_motion_length", None)
    
    # FM-specific settings
    num_inference_steps = sample_yml.get("num_inference_steps", 50)
    ode_solver = sample_yml.get("ode_solver", "euler")
    
    # Optimization settings
    precision = opt_yml.get("precision", "fp16")
    use_torch_compile = opt_yml.get("use_torch_compile", True)
    compile_mode = opt_yml.get("compile_mode", "reduce-overhead")
    warmup_iterations = opt_yml.get("warmup_iterations", 3)
    
    if not stage1_ckpt_path or not stage2_ckpt_path:
        raise ValueError("Both stage1_ckpt_path and stage2_ckpt_path must be set")
    
    device = torch.device(device_str if torch.cuda.is_available() or device_str != "cuda" else "cpu")
    torch.manual_seed(seed)
    
    print(f"\nInference Configuration:")
    print(f"  Precision: {precision}")
    print(f"  ODE solver: {ode_solver}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  torch.compile: {use_torch_compile}")
    
    # Create pipeline
    pipeline = OptimizedFlowMatchingPipeline(
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        device=str(device),
        contact_threshold=sample_yml.get("contact_threshold", 0.03),
        num_inference_steps=num_inference_steps,
        ode_solver=ode_solver,
        precision=precision,
        use_torch_compile=use_torch_compile,
        compile_mode=compile_mode,
        warmup_iterations=warmup_iterations,
    )
    
    # Output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    exp_name = yml.get("exp_name", "")
    suffix = f"_{exp_name}" if exp_name else ""
    
    window_size = dataset_yml.get("window_size", 120)
    stride = dataset_yml.get("stride", 10)
    
    ckpt_parts = stage2_ckpt_path.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        sample_folder = f"fm_opt_{ode_solver}{num_inference_steps}_w{window_size}_s{stride}_{timestamp}{suffix}"
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    else:
        output_dir = f"./out/e2e_fm_opt_{timestamp}{suffix}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    import yaml
    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump({
            "inference": {
                "precision": precision,
                "ode_solver": ode_solver,
                "steps": num_inference_steps,
                "torch_compile": use_torch_compile,
            },
            **yml,
        }, f, default_flow_style=False)
    
    # Find input files
    files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if num_samples:
        files = files[:num_samples]
    
    print(f"\nProcessing {len(files)} files")
    print(f"Output directory: {output_dir}")
    
    # Warmup
    if files:
        with open(files[0], "rb") as f:
            warmup_data = pickle.load(f)
        if "bps_encoding" in warmup_data:
            T = warmup_data["object_centroid"].shape[0]
            pipeline.warmup(seq_len=T)
    
    # Reset seed after warmup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Process files
    total_time = 0
    sample_durations = []
    all_frame_counts = []
    apply_constraints = sample_yml.get("apply_constraints", True)
    
    for fpath in tqdm(files, desc="Processing"):
        fname = os.path.basename(fpath)
        
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        
        if "bps_encoding" not in data or "object_centroid" not in data:
            print(f"  Skipping {fname}: missing data")
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
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        
        fps = data.get("fps", 30.0)
        num_frames = result.get("root_pos", result.get("dof_pos")).shape[0] if "root_pos" in result or "dof_pos" in result else 0
        if num_frames > 0:
            all_frame_counts.append(num_frames)
            sample_durations.append(num_frames / fps)
        
        # Save result — same format as all other pipelines
        output_data = {
            "seq_name": data.get("seq_name", fname),
            "fps": data.get("fps", 30.0),
            "generative_model": "flow_matching",
            "ode_solver": ode_solver,
            "num_inference_steps": num_inference_steps,
            **result,
            "inference_time_ms": elapsed * 1000,
        }
        
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
        
        if "root_pos" in data:
            output_data["gt_root_pos"] = data["root_pos"]
            output_data["gt_root_rot"] = data["root_rot"]
            output_data["gt_dof_pos"] = data["dof_pos"]
        if "hand_positions" in data:
            output_data["gt_hands"] = data["hand_positions"]
        
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)
    
    # Summary
    avg_duration = sum(sample_durations) / len(sample_durations) if sample_durations else 0
    
    print("\n" + "=" * 50)
    print("Performance Summary (Optimized Flow Matching):")
    print(f"  Total files: {len(files)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  ODE solver: {ode_solver}, steps: {num_inference_steps}")
    print(f"  Precision: {precision}")
    if files:
        print(f"  Average time: {total_time / len(files) * 1000:.2f}ms per sample")
        print(f"  Throughput: {len(files) / total_time:.2f} samples/sec")
    print(f"  Average sample duration: {avg_duration:.2f}s")
    if all_frame_counts:
        print(f"  Frames per motion: {int(np.mean(all_frame_counts))} avg")
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        print(f"  Generated fps: {gen_fps:.1f}")
    
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Performance Summary (Optimized Flow Matching)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total files: {len(files)}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"ODE solver: {ode_solver}, steps: {num_inference_steps}\n")
        f.write(f"Precision: {precision}\n")
        if files:
            f.write(f"Average time: {total_time / len(files) * 1000:.2f}ms per sample\n")
            f.write(f"Throughput: {len(files) / total_time:.2f} samples/sec\n")
        f.write(f"Average sample duration: {avg_duration:.2f}s\n")
        if all_frame_counts:
            f.write(f"Frames per motion: {int(np.mean(all_frame_counts))} avg\n")
            total_frames = sum(all_frame_counts)
            gen_fps = total_frames / total_time if total_time > 0 else 0
            f.write(f"Generated fps: {gen_fps:.1f}\n")
    
    print("=" * 50)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
