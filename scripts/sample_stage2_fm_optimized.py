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
    python sample_stage2_fm_optimized.py --config_path ./experiments/sample_stage2_fm_optimized_balanced.yaml
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple

# Ensure project root is on sys.path
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


# Import the optimized pipeline class from the root-level script
sys.path.insert(0, PROJECT_ROOT)
from sample_stage2_fm_optimized import OptimizedFlowMatchingPipeline


def main():
    parser = argparse.ArgumentParser(description="Optimized FM End-to-End Sampling")
    parser.add_argument("--config_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments", "sample_stage2_fm_optimized_balanced.yaml"),
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
