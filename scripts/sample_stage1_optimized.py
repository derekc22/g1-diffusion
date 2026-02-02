"""
Optimized Stage 1 Sampling Script with Inference Time Optimizations

This script provides fast inference for Stage 1 diffusion model using:
1. Mixed precision (FP16/BF16) for ~2x speedup
2. torch.compile for graph optimization
3. DDIM/DPM-Solver for 10-20x faster sampling (50 steps vs 1000)
4. Model warmup for consistent performance

Typical speedup: 3-5 seconds → 0.2-0.5 seconds per sample

Usage:
    python sample_stage1_optimized.py

Configuration is done via ./config/sample_stage1_optimized.yaml
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

from models.stage1_diffusion import Stage1HandDiffusion, Stage1HandDiffusionMLP
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.inference_optimization import (
    InferenceConfig,
    PrecisionMode,
    SamplerType,
    OptimizedInferenceWrapper,
    DDIMSampler,
    DPMSolverSampler,
    create_sampler,
    benchmark_inference,
    get_optimal_config,
    ONNXExporter,
)
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
    
    # Get max_len from config
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
    
    # Get timesteps from config
    train_cfg = config.get("train", {})
    timesteps = train_cfg.get("timesteps", 1000)
    
    return model, timesteps, hand_mean, hand_std, config


@torch.inference_mode()
def sample_optimized(
    model: torch.nn.Module,
    sampler: DDIMSampler | DPMSolverSampler,
    bps_encoding: torch.Tensor,
    object_centroid: torch.Tensor,
    num_samples: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample hand positions using fast sampler (DDIM or DPM-Solver).
    
    Args:
        model: Stage 1 model (optimized wrapper or raw)
        sampler: DDIM or DPM-Solver sampler
        bps_encoding: (1, T, 3072) or (1, T, 1024, 3)
        object_centroid: (1, T, 3)
        num_samples: Number of samples to generate
        dtype: Data type for computation
    
    Returns:
        (num_samples, T, 6) sampled hand positions (normalized)
    """
    device = next(model.parameters()).device
    T_seq = object_centroid.shape[1]
    
    # Prepare conditions
    if num_samples > 1:
        bps = bps_encoding.repeat(num_samples, 1, 1)
        centroid = object_centroid.repeat(num_samples, 1, 1)
    else:
        bps = bps_encoding
        centroid = object_centroid
    
    # Convert to target dtype
    bps = bps.to(device=device, dtype=dtype)
    centroid = centroid.to(device=device, dtype=dtype)
    
    # Create condition function for sampler
    def condition_fn():
        return (bps, centroid)
    
    # Run sampling
    shape = (num_samples, T_seq, 6)
    x = sampler.sample(
        model=model,
        shape=shape,
        condition_fn=condition_fn,
        device=device,
        dtype=dtype,
    )
    
    return x


@torch.inference_mode()
def sample_ddpm_optimized(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    bps_encoding: torch.Tensor,
    object_centroid: torch.Tensor,
    num_samples: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample using original DDPM but with precision optimization.
    
    For cases where exact DDPM behavior is needed.
    """
    device = next(model.parameters()).device
    T_seq = object_centroid.shape[1]
    
    if num_samples > 1:
        bps = bps_encoding.repeat(num_samples, 1, 1)
        centroid = object_centroid.repeat(num_samples, 1, 1)
    else:
        bps = bps_encoding
        centroid = object_centroid
    
    bps = bps.to(device=device, dtype=dtype)
    centroid = centroid.to(device=device, dtype=dtype)
    
    x = torch.randn(num_samples, T_seq, 6, device=device, dtype=dtype)
    
    for n in reversed(range(schedule.timesteps)):
        t = torch.full((num_samples,), n, device=device, dtype=torch.long)
        
        x0_pred = model(x, t, bps, centroid)
        
        if n > 0:
            alpha_bar_t = schedule.alpha_bar[n]
            alpha_bar_t_prev = schedule.alpha_bar[n - 1]
            alpha_t = schedule.alpha[n]
            
            mean = (
                torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred +
                torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
            )
            
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
    if hands.ndim == 3:
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    elif hands.ndim == 2:
        mean = mean.view(1, -1)
        std = std.view(1, -1)
    return hands * std + mean


def process_sequence_optimized(
    model: torch.nn.Module,
    sampler: DDIMSampler | DPMSolverSampler | None,
    schedule: DiffusionSchedule | None,
    data: Dict[str, Any],
    hand_mean: Optional[torch.Tensor],
    hand_std: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    apply_constraints: bool = True,
    contact_threshold: float = 0.03,
    use_fast_sampler: bool = True,
    partial_motion_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Process a single sequence with optimized sampling.
    
    Args:
        partial_motion_length: If set, generate motion of this length instead of full object motion length
    """
    # Get original length
    T_original = data["object_centroid"].shape[0]
    partial = False
    target_len = None
    
    # Handle partial motion generation
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
    bps = torch.from_numpy(bps_enc).float()
    centroid_t = torch.from_numpy(centroid).float()
    
    # Handle BPS shape
    if bps.ndim == 3 and bps.shape[-1] == 3:
        bps = bps.reshape(bps.shape[0], -1)
    
    bps = bps.unsqueeze(0).to(device)
    centroid_t = centroid_t.unsqueeze(0).to(device)
    
    # Sample
    if use_fast_sampler and sampler is not None:
        hands_norm = sample_optimized(model, sampler, bps, centroid_t, num_samples=1, dtype=dtype)
    else:
        hands_norm = sample_ddpm_optimized(model, schedule, bps, centroid_t, num_samples=1, dtype=dtype)
    
    # Denormalize
    hands_raw = denormalize_hands(hands_norm, hand_mean, hand_std)
    hands_raw = hands_raw.squeeze(0).float().cpu().numpy()
    
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


def run_benchmark_fn(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int = 120,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different inference configurations.
    """
    print("\n" + "=" * 60)
    print("Running inference benchmarks...")
    print("=" * 60)
    
    results = {}
    
    # Create sample inputs
    batch_size = 1
    sample_x = torch.randn(batch_size, seq_len, 6, device=device)
    sample_t = torch.randint(0, 1000, (batch_size,), device=device)
    sample_bps = torch.randn(batch_size, seq_len, 3072, device=device)
    sample_centroid = torch.randn(batch_size, seq_len, 3, device=device)
    
    inputs = (sample_x, sample_t, sample_bps, sample_centroid)
    
    configs_to_test = [
        ("FP32 baseline", PrecisionMode.FP32, False),
        ("FP16", PrecisionMode.FP16, False),
        ("FP16 + compile", PrecisionMode.FP16, True),
    ]
    
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        configs_to_test.append(("BF16 + compile", PrecisionMode.BF16, True))
    
    for name, precision, use_compile in configs_to_test:
        print(f"\nBenchmarking: {name}")
        
        # Create optimized wrapper
        config = InferenceConfig(
            precision=precision,
            use_torch_compile=use_compile,
            compile_mode="reduce-overhead",
        )
        
        try:
            # Clone model for each config
            import copy
            test_model = copy.deepcopy(model)
            wrapper = OptimizedInferenceWrapper(test_model, config, device)
            
            # Warmup
            wrapper.warmup(inputs)
            
            # Benchmark
            timing = benchmark_inference(
                wrapper.model,
                inputs,
                num_iterations=50,
                warmup_iterations=5,
            )
            
            results[name] = timing
            print(f"  Mean: {timing['mean_ms']:.2f} ms")
            print(f"  Throughput: {timing['throughput_hz']:.1f} Hz")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {"error": str(e)}
    
    # Test different samplers
    print("\n" + "-" * 40)
    print("Sampler benchmarks (full sampling loop):")
    print("-" * 40)
    
    sampler_configs = [
        ("DDPM (1000 steps)", SamplerType.DDPM, 1000),
        ("DDIM (100 steps)", SamplerType.DDIM, 100),
        ("DDIM (50 steps)", SamplerType.DDIM, 50),
        ("DPM-Solver (20 steps)", SamplerType.DPM_SOLVER, 20),
    ]
    
    # Use FP16 for sampler benchmarks
    config = InferenceConfig(
        precision=PrecisionMode.FP16 if device.type == "cuda" else PrecisionMode.FP32,
        use_torch_compile=True,
    )
    
    import copy
    test_model = copy.deepcopy(model)
    wrapper = OptimizedInferenceWrapper(test_model, config, device)
    wrapper.warmup(inputs)
    
    for name, sampler_type, steps in sampler_configs:
        print(f"\nBenchmarking: {name}")
        
        try:
            if sampler_type == SamplerType.DDPM:
                schedule = DiffusionSchedule(DiffusionConfig(timesteps=steps)).to(device)
                
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    _ = sample_ddpm_optimized(
                        wrapper.model, schedule,
                        sample_bps[:1], sample_centroid[:1],
                        dtype=wrapper.dtype,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
            else:
                sampler = create_sampler(sampler_type, num_inference_steps=steps).to(device)
                
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    _ = sample_optimized(
                        wrapper.model, sampler,
                        sample_bps[:1], sample_centroid[:1],
                        dtype=wrapper.dtype,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
            
            mean_time = np.mean(times)
            results[name] = {"mean_ms": mean_time}
            print(f"  Mean: {mean_time:.2f} ms")
            print(f"  Speedup vs DDPM: {results.get('DDPM (1000 steps)', {}).get('mean_ms', mean_time) / mean_time:.1f}x")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    return results


def export_model(
    model: torch.nn.Module,
    output_dir: str,
    device: torch.device,
    seq_len: int = 120,
):
    """Export model to ONNX format."""
    print(f"\nExporting model to {output_dir}")
    
    sample_inputs = {
        "x": torch.randn(1, seq_len, 6, device=device),
        "t": torch.randint(0, 1000, (1,), device=device),
        "bps": torch.randn(1, seq_len, 3072, device=device),
        "centroid": torch.randn(1, seq_len, 3, device=device),
    }
    
    try:
        paths = ONNXExporter.export_stage1_model(
            model,
            output_dir,
            sample_inputs,
        )
        print(f"Exported to: {paths}")
    except Exception as e:
        print(f"Export failed: {e}")


def build_inference_config_from_yaml(opt_yml: dict) -> InferenceConfig:
    """Build InferenceConfig from YAML optimization section."""
    config = InferenceConfig()
    
    # Load settings from YAML
    if "precision" in opt_yml:
        config.precision = PrecisionMode(opt_yml["precision"])
    if "sampler" in opt_yml:
        config.sampler = SamplerType(opt_yml["sampler"])
    if "num_inference_steps" in opt_yml:
        config.num_inference_steps = opt_yml["num_inference_steps"]
    if "ddim_eta" in opt_yml:
        config.ddim_eta = opt_yml["ddim_eta"]
    
    # torch.compile settings
    if "use_torch_compile" in opt_yml:
        config.use_torch_compile = opt_yml["use_torch_compile"]
    if "compile_mode" in opt_yml:
        config.compile_mode = opt_yml["compile_mode"]
    if "compile_fullgraph" in opt_yml:
        config.compile_fullgraph = opt_yml["compile_fullgraph"]
    if "warmup_iterations" in opt_yml:
        config.warmup_iterations = opt_yml["warmup_iterations"]
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Optimized Sampling")
    parser.add_argument("--config_path", type=str, default="./config/sample_stage1_optimized.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load all settings from YAML config
    yml = load_config(args.config_path)
    
    sample_yml = yml["sample"]
    dataset_yml = yml["dataset"]
    opt_yml = yml.get("optimization", {})

    root_dir = yml["root_dir"]
    ckpt_path = sample_yml["ckpt_path"]
    device_str = sample_yml.get("device", "cuda")
    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    num_samples = sample_yml.get("num_samples", None)
    evaluate = sample_yml.get("evaluate", False)
    seed = sample_yml.get("seed", 42)
    run_benchmark = sample_yml.get("benchmark", False)
    export_onnx_path = sample_yml.get("export_onnx", None)
    partial_motion_length = sample_yml.get("partial_motion_length", None)

    if not ckpt_path:
        raise ValueError("ckpt_path must be set in config")

    device = torch.device(device_str if torch.cuda.is_available() or device_str != "cuda" else "cpu")
    torch.manual_seed(seed)

    # Load model
    model, num_timesteps, hand_mean, hand_std, config = load_checkpoint(ckpt_path, device)

    # Build inference config from YAML
    inf_config = build_inference_config_from_yaml(opt_yml)

    print(f"\nInference Configuration:")
    print(f"  Precision: {inf_config.precision.value}")
    print(f"  Sampler: {inf_config.sampler.value}")
    print(f"  Steps: {inf_config.num_inference_steps if inf_config.sampler != SamplerType.DDPM else num_timesteps}")
    print(f"  torch.compile: {inf_config.use_torch_compile}")

    # Run benchmark if requested
    if run_benchmark:
        run_benchmark_fn(model, device)
        return

    # Export if requested
    if export_onnx_path:
        export_model(model, export_onnx_path, device)
        return

    # Create optimized wrapper
    wrapper = OptimizedInferenceWrapper(model, inf_config, device)

    # Create sampler
    if inf_config.sampler == SamplerType.DDPM:
        sampler = None
        schedule = DiffusionSchedule(DiffusionConfig(timesteps=num_timesteps)).to(device)
        use_fast = False
    else:
        sampler = create_sampler(
            inf_config.sampler,
            num_train_timesteps=num_timesteps,
            num_inference_steps=inf_config.num_inference_steps,
            ddim_eta=inf_config.ddim_eta,
        ).to(device)
        schedule = None
        use_fast = True

    # Output directory - match original naming pattern + optional exp_name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    # Get optional experiment name for folder suffix
    exp_name = yml.get("exp_name", "")
    suffix = f"_{exp_name}" if exp_name else ""
    
    # Get dataset config for folder naming
    window_size = dataset_yml.get("window_size", 120)
    stride = dataset_yml.get("stride", 10)

    ckpt_parts = ckpt_path.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        sample_folder = f"ts{num_timesteps}_w{window_size}_s{stride}_{timestamp}{suffix}"
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    else:
        output_dir = f"./out/stage1_{timestamp}{suffix}"

    os.makedirs(output_dir, exist_ok=True)

    # Save config
    import yaml
    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump({
            "inference": {
                "precision": inf_config.precision.value,
                "sampler": inf_config.sampler.value,
                "steps": inf_config.num_inference_steps,
                "torch_compile": inf_config.use_torch_compile,
            },
            **yml,
        }, f, default_flow_style=False)

    # Find input files
    files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if num_samples:
        files = files[:num_samples]

    print(f"\nProcessing {len(files)} files")
    print(f"Output directory: {output_dir}")
    
    # Warmup with first file
    if files:
        print("\nWarming up model...")
        with open(files[0], "rb") as f:
            warmup_data = pickle.load(f)
        
        if "bps_encoding" in warmup_data and "object_centroid" in warmup_data:
            warmup_bps = torch.from_numpy(warmup_data["bps_encoding"]).float()
            warmup_centroid = torch.from_numpy(warmup_data["object_centroid"]).float()
            
            if warmup_bps.ndim == 3 and warmup_bps.shape[-1] == 3:
                warmup_bps = warmup_bps.reshape(warmup_bps.shape[0], -1)
            
            warmup_bps = warmup_bps.unsqueeze(0).to(device)
            warmup_centroid = warmup_centroid.unsqueeze(0).to(device)
            
            T = warmup_centroid.shape[1]
            sample_x = torch.randn(1, T, 6, device=device)
            sample_t = torch.randint(0, num_timesteps, (1,), device=device)
            
            wrapper.warmup((sample_x, sample_t, warmup_bps, warmup_centroid))
    
    # Reset seed AFTER warmup to ensure reproducibility
    # Warmup consumes RNG state, so we need to reset it
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Process files
    total_time = 0
    sample_durations = []

    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    evaluate = sample_yml.get("evaluate", False)
    
    all_metrics = []
    all_hand_jpe = []
    
    for fpath in tqdm(files, desc="Processing"):
        fname = os.path.basename(fpath)
        
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        
        if "bps_encoding" not in data or "object_centroid" not in data:
            print(f"  Skipping {fname}: missing object data")
            continue
        
        start_time = time.perf_counter()
        
        result = process_sequence_optimized(
            wrapper.model, sampler, schedule, data,
            hand_mean, hand_std, device, wrapper.dtype,
            apply_constraints=apply_constraints,
            contact_threshold=contact_threshold,
            use_fast_sampler=use_fast,
            partial_motion_length=partial_motion_length,
        )
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        
        # Track sample duration
        fps = data.get("fps", 30.0)
        num_frames = result["hands_rectified"].shape[0]
        sample_durations.append(num_frames / fps)
        
        # Evaluate
        if evaluate and "hand_positions" in data:
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
            "inference_time_ms": elapsed * 1000,
        }
        
        if "hand_positions" in data:
            output_data["hands_gt"] = data["hand_positions"]
        
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)
    
    # Print summary
    # Calculate average sample duration
    avg_duration = sum(sample_durations) / len(sample_durations) if sample_durations else 0
    
    print("\n" + "=" * 50)
    print("Performance Summary:")
    print(f"  Total files: {len(files)}")
    print(f"  Total time: {total_time:.2f}s")
    if files:
        print(f"  Average time: {total_time / len(files) * 1000:.2f}ms per sample")
        print(f"  Throughput: {len(files) / total_time:.2f} samples/sec")
    print(f"  Average sample duration: {avg_duration:.2f}s")
    
    # Build summary text
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
    summary_lines.append(f"Average sample duration: {avg_duration:.2f}s")
    
    if evaluate and all_hand_jpe:
        print("\nEvaluation Results:")
        print(f"  Hand JPE (cm): {np.mean(all_hand_jpe):.2f} ± {np.std(all_hand_jpe):.2f}")
        summary_lines.extend([
            "",
            "Evaluation Results",
            f"Hand JPE (cm): {np.mean(all_hand_jpe):.2f} ± {np.std(all_hand_jpe):.2f}",
        ])
        
        if all_metrics:
            avg_prec = np.mean([m["precision"] for m in all_metrics])
            avg_rec = np.mean([m["recall"] for m in all_metrics])
            avg_f1 = np.mean([m["f1"] for m in all_metrics])
            print(f"  Contact Precision: {avg_prec:.3f}")
            print(f"  Contact Recall: {avg_rec:.3f}")
            print(f"  Contact F1: {avg_f1:.3f}")
            summary_lines.extend([
                f"Contact Precision: {avg_prec:.3f}",
                f"Contact Recall: {avg_rec:.3f}",
                f"Contact F1: {avg_f1:.3f}",
            ])
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    
    print("=" * 50)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
