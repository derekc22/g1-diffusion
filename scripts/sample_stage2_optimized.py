"""
Optimized End-to-End OMOMO Pipeline with Inference Time Optimizations

Complete pipeline with:
1. Stage 1: Object geometry → Hand positions (optimized)
2. Contact constraints
3. Stage 2: Hand positions → Full-body motion (optimized)

Supports:
- Mixed precision (FP16/BF16) for ~2x speedup
- torch.compile for graph optimization
- DDIM/DPM-Solver for 10-50x faster sampling
- Batch processing for throughput

Usage:
    python sample_stage2_optimized.py

Configuration is done via ./config/sample_stage2_optimized.yaml
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple

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
from models.stage2_diffusion import Stage2TransformerModel, Stage2MLPModel
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
)
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


class OptimizedOmomoPipeline:
    """
    Optimized OMOMO pipeline with inference time optimizations.
    
    Applies precision optimization and fast sampling to both stages.
    """
    
    def __init__(
        self,
        stage1_ckpt_path: str,
        stage2_ckpt_path: str,
        config: InferenceConfig,
        device: str = "cuda:0",
        contact_threshold: float = 0.03,
    ):
        self.device = torch.device(device)
        self.config = config
        self.contact_threshold = contact_threshold
        
        # Determine dtype from config
        if config.precision == PrecisionMode.FP16:
            self.dtype = torch.float16
        elif config.precision == PrecisionMode.BF16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        # Load Stage 1
        print("Loading Stage 1...")
        (
            self.stage1_model,
            self.stage1_timesteps,
            self.stage1_norm,
            self.stage1_max_len,
        ) = self._load_stage1(stage1_ckpt_path)
        
        # Load Stage 2
        print("Loading Stage 2...")
        (
            self.stage2_model,
            self.stage2_timesteps,
            self.stage2_norm,
            self.stage2_max_len,
            self.state_dim,
        ) = self._load_stage2(stage2_ckpt_path)
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Create samplers
        self._create_samplers()
        
        # Contact processor
        self.contact_processor = ContactConstraintProcessor(contact_threshold=contact_threshold)
        
        self._is_warmed_up = False
    
    def _load_stage1(self, ckpt_path: str):
        """Load Stage 1 model."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        
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
        
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "hand_mean": norm_stats.get("hand_mean"),
            "hand_std": norm_stats.get("hand_std"),
        }
        if norm["hand_mean"] is not None:
            norm["hand_mean"] = norm["hand_mean"].to(self.device)
            norm["hand_std"] = norm["hand_std"].to(self.device)
        
        return model, timesteps, norm, max_len
    
    def _load_stage2(self, ckpt_path: str):
        """Load Stage 2 model."""
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
        
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "state_mean": norm_stats.get("state_mean"),
            "state_std": norm_stats.get("state_std"),
        }
        if norm["state_mean"] is not None:
            norm["state_mean"] = norm["state_mean"].to(self.device)
            norm["state_std"] = norm["state_std"].to(self.device)
        
        return model, timesteps, norm, max_len, state_dim
    
    def _apply_optimizations(self):
        """Apply precision and compilation optimizations."""
        # Apply precision
        if self.config.precision == PrecisionMode.FP16:
            self.stage1_model = self.stage1_model.half()
            self.stage2_model = self.stage2_model.half()
        elif self.config.precision == PrecisionMode.BF16:
            self.stage1_model = self.stage1_model.to(torch.bfloat16)
            self.stage2_model = self.stage2_model.to(torch.bfloat16)
        elif self.config.precision == PrecisionMode.INT8:
            import warnings
            warnings.warn("INT8 quantization not recommended for diffusion models")
        
        # Apply torch.compile
        if self.config.use_torch_compile and hasattr(torch, 'compile'):
            try:
                self.stage1_model = torch.compile(
                    self.stage1_model,
                    mode=self.config.compile_mode,
                    fullgraph=self.config.compile_fullgraph,
                )
                self.stage2_model = torch.compile(
                    self.stage2_model,
                    mode=self.config.compile_mode,
                    fullgraph=self.config.compile_fullgraph,
                )
                print(f"Models compiled with mode='{self.config.compile_mode}'")
            except Exception as e:
                print(f"torch.compile failed: {e}")
    
    def _create_samplers(self):
        """Create fast samplers for both stages."""
        if self.config.sampler == SamplerType.DDPM:
            self.stage1_sampler = None
            self.stage2_sampler = None
            self.stage1_schedule = DiffusionSchedule(
                DiffusionConfig(timesteps=self.stage1_timesteps)
            ).to(self.device)
            self.stage2_schedule = DiffusionSchedule(
                DiffusionConfig(timesteps=self.stage2_timesteps)
            ).to(self.device)
        else:
            self.stage1_sampler = create_sampler(
                self.config.sampler,
                num_train_timesteps=self.stage1_timesteps,
                num_inference_steps=self.config.num_inference_steps,
                ddim_eta=self.config.ddim_eta,
            ).to(self.device)
            self.stage2_sampler = create_sampler(
                self.config.sampler,
                num_train_timesteps=self.stage2_timesteps,
                num_inference_steps=self.config.num_inference_steps,
                ddim_eta=self.config.ddim_eta,
            ).to(self.device)
            self.stage1_schedule = None
            self.stage2_schedule = None
    
    def warmup(self, seq_len: int = 120):
        """Warmup both stages for consistent performance."""
        print("Warming up models...")
        
        # Warmup Stage 1
        dummy_bps = torch.randn(1, seq_len, 3072, device=self.device, dtype=self.dtype)
        dummy_centroid = torch.randn(1, seq_len, 3, device=self.device, dtype=self.dtype)
        dummy_x1 = torch.randn(1, seq_len, 6, device=self.device, dtype=self.dtype)
        dummy_t = torch.randint(0, self.stage1_timesteps, (1,), device=self.device)
        
        for _ in range(self.config.warmup_iterations):
            with torch.inference_mode():
                _ = self.stage1_model(dummy_x1, dummy_t, dummy_bps, dummy_centroid)
        
        # Warmup Stage 2
        dummy_cond = torch.randn(1, seq_len, 6, device=self.device, dtype=self.dtype)
        dummy_x2 = torch.randn(1, seq_len, self.state_dim, device=self.device, dtype=self.dtype)
        
        for _ in range(self.config.warmup_iterations):
            with torch.inference_mode():
                _ = self.stage2_model(dummy_x2, dummy_t, dummy_cond)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self._is_warmed_up = True
        print("Warmup complete")
    
    @torch.inference_mode()
    def _sample_stage1_fast(
        self,
        bps: torch.Tensor,
        centroid: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Stage 1 with fast sampler."""
        B, T, _ = centroid.shape
        
        def condition_fn():
            return (bps, centroid)
        
        return self.stage1_sampler.sample(
            model=self.stage1_model,
            shape=(B, T, 6),
            condition_fn=condition_fn,
            device=self.device,
            dtype=self.dtype,
        )
    
    @torch.inference_mode()
    def _sample_stage1_ddpm(
        self,
        bps: torch.Tensor,
        centroid: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Stage 1 with DDPM."""
        B, T, _ = centroid.shape
        x = torch.randn(B, T, 6, device=self.device, dtype=self.dtype)
        
        for n in reversed(range(self.stage1_schedule.timesteps)):
            t = torch.full((B,), n, device=self.device, dtype=torch.long)
            x0_pred = self.stage1_model(x, t, bps, centroid)
            
            if n > 0:
                alpha_bar_t = self.stage1_schedule.alpha_bar[n]
                alpha_bar_t_prev = self.stage1_schedule.alpha_bar[n - 1]
                alpha_t = self.stage1_schedule.alpha[n]
                
                mean = (
                    torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred +
                    torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
                )
                sigma = torch.sqrt(self.stage1_schedule.beta[n])
                x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        
        return x
    
    @torch.inference_mode()
    def _sample_stage2_fast(
        self,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Stage 2 with fast sampler."""
        B, T, _ = cond.shape
        
        def condition_fn():
            return cond
        
        return self.stage2_sampler.sample(
            model=self.stage2_model,
            shape=(B, T, self.state_dim),
            condition_fn=condition_fn,
            device=self.device,
            dtype=self.dtype,
        )
    
    @torch.inference_mode()
    def _sample_stage2_ddpm(
        self,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Stage 2 with DDPM."""
        B, T, _ = cond.shape
        x = torch.randn(B, T, self.state_dim, device=self.device, dtype=self.dtype)
        
        for n in reversed(range(self.stage2_schedule.timesteps)):
            t = torch.full((B,), n, device=self.device, dtype=torch.long)
            x0_pred = self.stage2_model(x, t, cond)
            
            if n > 0:
                alpha_bar_t = self.stage2_schedule.alpha_bar[n]
                alpha_bar_t_prev = self.stage2_schedule.alpha_bar[n - 1]
                alpha_t = self.stage2_schedule.alpha[n]
                
                mean = (
                    torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred +
                    torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
                )
                sigma = torch.sqrt(self.stage2_schedule.beta[n])
                x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        
        return x
    
    def _denormalize(
        self,
        x: torch.Tensor,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Denormalize tensor."""
        if mean is None:
            return x
        if x.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return x * std.to(x.dtype) + mean.to(x.dtype)
    
    def _normalize(
        self,
        x: torch.Tensor,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize tensor."""
        if mean is None:
            return x
        if x.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return (x - mean.to(x.dtype)) / std.to(x.dtype)
    
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
        """
        Generate full-body motion from object geometry.
        
        Args:
            bps_encoding: (T, 1024, 3) or (T, 3072) BPS encoding
            object_centroid: (T, 3) object centroid positions
            object_verts: (T, V, 3) object vertices (optional, for constraints)
            object_rotation: (T, 3, 3) object rotation (optional, for constraints)
            apply_constraints: Whether to apply contact constraints
            partial_motion_length: If set, generate motion of this length instead of full object motion length
            
        Returns:
            Dict with hands_raw, hands_rectified, state, root_pos, root_rot, dof_pos, truncated, original_len, partial, target_len
        """
        T_original = object_centroid.shape[0]
        truncated = False
        partial = False
        target_len = None
        
        # Handle partial motion generation
        if partial_motion_length is not None and partial_motion_length > 0:
            if partial_motion_length < T_original:
                # Use only the first partial_motion_length frames of object motion
                bps_encoding = bps_encoding[:partial_motion_length]
                object_centroid = object_centroid[:partial_motion_length]
                if object_verts is not None:
                    object_verts = object_verts[:partial_motion_length]
                if object_rotation is not None:
                    object_rotation = object_rotation[:partial_motion_length]
                partial = True
                target_len = partial_motion_length
            else:
                print(f"  Warning: partial_motion_length ({partial_motion_length}) >= object motion length ({T_original}), using full length")
        
        # Update working length to reflect potential partial motion
        T_working = object_centroid.shape[0]
        
        # Truncate if sequence exceeds max_len (positional encoding limit)
        max_len = self.stage1_max_len  # Use stage1 max_len as reference
        if T_working > max_len:
            print(f"  Warning: Truncating sequence from {T_working} to {max_len} frames")
            bps_encoding = bps_encoding[:max_len]
            object_centroid = object_centroid[:max_len]
            if object_verts is not None:
                object_verts = object_verts[:max_len]
            if object_rotation is not None:
                object_rotation = object_rotation[:max_len]
            truncated = True
        
        # Prepare inputs
        bps = torch.from_numpy(bps_encoding).to(self.device, dtype=self.dtype)
        centroid = torch.from_numpy(object_centroid).to(self.device, dtype=self.dtype)
        
        # Handle BPS shape
        if bps.ndim == 3 and bps.shape[-1] == 3:
            bps = bps.reshape(bps.shape[0], -1)
        
        # Add batch dimension
        bps = bps.unsqueeze(0)
        centroid = centroid.unsqueeze(0)
        
        # Stage 1: Generate hands
        if self.stage1_sampler is not None:
            hands_norm = self._sample_stage1_fast(bps, centroid)
        else:
            hands_norm = self._sample_stage1_ddpm(bps, centroid)
        
        # Denormalize hands
        hands_raw = self._denormalize(
            hands_norm,
            self.stage1_norm["hand_mean"],
            self.stage1_norm["hand_std"],
        )
        hands_raw_np = hands_raw.squeeze(0).float().cpu().numpy()
        
        # Apply contact constraints
        if apply_constraints and object_verts is not None and object_rotation is not None:
            hands_rect_np, contact_metadata = self.contact_processor.process(
                hands_raw_np, object_verts, object_rotation
            )
            hands_rect = torch.from_numpy(hands_rect_np).to(self.device, dtype=self.dtype)
            hands_rect = hands_rect.unsqueeze(0)
        else:
            hands_rect = hands_raw
            hands_rect_np = hands_raw_np
            contact_metadata = None
        
        # Stage 2: Generate full body
        # NOTE: Stage 2 takes DENORMALIZED hand positions as conditioning
        # (This matches baseline - do NOT normalize hands_rect here)
        if self.stage2_sampler is not None:
            state_norm = self._sample_stage2_fast(hands_rect)
        else:
            state_norm = self._sample_stage2_ddpm(hands_rect)
        
        # Denormalize state
        state = self._denormalize(
            state_norm,
            self.stage2_norm["state_mean"],
            self.stage2_norm["state_std"],
        )
        state_np = state.squeeze(0).float().cpu().numpy()
        
        # Parse state into components (matches original sample_stage2.py)
        root_pos = state_np[:, :3]
        root_rot_6d = state_np[:, 3:9]
        dof_pos = state_np[:, 9:]
        
        # Convert 6D rotation to quaternion
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
    parser = argparse.ArgumentParser(description="Stage 2 Optimized End-to-End Sampling")
    parser.add_argument("--config_path", type=str, default="./config/sample_stage2_optimized.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load all settings from YAML config
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
    
    if not stage1_ckpt_path or not stage2_ckpt_path:
        raise ValueError("Both stage1_ckpt_path and stage2_ckpt_path must be set")
    
    device = torch.device(device_str if torch.cuda.is_available() or device_str != "cuda" else "cpu")
    torch.manual_seed(seed)
    
    # Build inference config from YAML
    inf_config = build_inference_config_from_yaml(opt_yml)
    
    print(f"\nInference Configuration:")
    print(f"  Precision: {inf_config.precision.value}")
    print(f"  Sampler: {inf_config.sampler.value}")
    print(f"  Steps: {inf_config.num_inference_steps if inf_config.sampler != SamplerType.DDPM else 'N/A (DDPM)'}")
    print(f"  torch.compile: {inf_config.use_torch_compile}")
    
    # Create pipeline
    pipeline = OptimizedOmomoPipeline(
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        config=inf_config,
        device=str(device),
        contact_threshold=sample_yml.get("contact_threshold", 0.03),
    )
    
    # Output directory - match original naming pattern + optional exp_name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    # Get optional experiment name for folder suffix
    exp_name = yml.get("exp_name", "")
    suffix = f"_{exp_name}" if exp_name else ""
    
    # Get dataset config for folder naming
    window_size = dataset_yml.get("window_size", 120)
    stride = dataset_yml.get("stride", 10)
    timesteps = 1000  # Default timesteps
    
    ckpt_parts = stage2_ckpt_path.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        sample_folder = f"ts{timesteps}_w{window_size}_s{stride}_{timestamp}{suffix}"
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    else:
        output_dir = f"./out/e2e_{timestamp}{suffix}"
    
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
    
    # Warmup
    if files:
        with open(files[0], "rb") as f:
            warmup_data = pickle.load(f)
        if "bps_encoding" in warmup_data:
            T = warmup_data["object_centroid"].shape[0]
            pipeline.warmup(seq_len=T)
    
    # Reset seed AFTER warmup to ensure reproducibility
    # Warmup consumes RNG state, so we need to reset it
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
        
        # Track sample duration
        fps = data.get("fps", 30.0)
        num_frames = result.get("root_pos", result.get("dof_pos")).shape[0] if "root_pos" in result or "dof_pos" in result else 0
        if num_frames > 0:
            all_frame_counts.append(num_frames)
            sample_durations.append(num_frames / fps)
        
        # Save result
        output_data = {
            "seq_name": data.get("seq_name", fname),
            "fps": data.get("fps", 30.0),
            **result,
            "inference_time_ms": elapsed * 1000,
        }
        
        # Add hand_positions for visualization compatibility
        if "hands_rectified" in result:
            output_data["hand_positions"] = result["hands_rectified"]
        
        # Add object data for visualization compatibility
        if "object_centroid" in data:
            output_data["object_pos"] = data["object_centroid"]
        
        # object_rot: (T, 4) quaternion in xyzw format from (T, 3, 3) rotation matrix
        if "object_rotation" in data:
            obj_rot_mat = data["object_rotation"]  # (T, 3, 3)
            obj_rot_mat_t = torch.from_numpy(obj_rot_mat).float()
            obj_rot_quat = mat_to_quat_xyzw(obj_rot_mat_t).numpy()  # (T, 4) xyzw
            output_data["object_rot"] = obj_rot_quat
        
        # Copy local_body_pos and link_body_list from source data
        if "local_body_pos" in data:
            output_data["local_body_pos"] = data["local_body_pos"]
        if "link_body_list" in data:
            output_data["link_body_list"] = data["link_body_list"]
        
        # source_start for comparison plots
        output_data["source_start"] = 0
        
        # Optionally include GT for comparison
        if "root_pos" in data:
            output_data["gt_root_pos"] = data["root_pos"]
            output_data["gt_root_rot"] = data["root_rot"]
            output_data["gt_dof_pos"] = data["dof_pos"]
        if "hand_positions" in data:
            output_data["gt_hands"] = data["hand_positions"]
        
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)
    
    # Calculate average sample duration
    avg_duration = sum(sample_durations) / len(sample_durations) if sample_durations else 0
    
    # Print summary
    print("\n" + "=" * 50)
    print("Performance Summary:")
    print(f"  Total files: {len(files)}")
    print(f"  Total time: {total_time:.2f}s")
    if files:
        print(f"  Average time: {total_time / len(files) * 1000:.2f}ms per sample")
        print(f"  Throughput: {len(files) / total_time:.2f} samples/sec")
    print(f"  Average sample duration: {avg_duration:.2f}s")
    if all_frame_counts:
        print(f"  Frames per motion: {int(np.mean(all_frame_counts))} avg, {min(all_frame_counts)} min, {max(all_frame_counts)} max")
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        print(f"  Generated fps: {gen_fps:.1f}")
    
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
    if all_frame_counts:
        summary_lines.append(f"Frames per motion: {int(np.mean(all_frame_counts))} avg, {min(all_frame_counts)} min, {max(all_frame_counts)} max")
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        summary_lines.append(f"Generated fps: {gen_fps:.1f}")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    
    print("=" * 50)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
