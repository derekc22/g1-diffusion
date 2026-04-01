"""
Optimized End-to-End HF Pipeline with Inference Time Optimizations

Complete pipeline with:
1. Stage 1: Object motion features → Hand positions (optimized)
2. Contact constraints
3. Stage 2: Hand positions → Full-body motion (optimized)

Supports:
- Mixed precision (FP16/BF16) for ~2x speedup
- torch.compile for graph optimization
- DDIM/DPM-Solver for 10-50x faster sampling (50 steps vs 1000)
- Model warmup for consistent performance

Usage:
    python scripts/sample_stage2_hf_optimized.py
    python scripts/sample_stage2_hf_optimized.py --config_path experiments/stage2_hf_optimized/my_config.yaml
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

from models.stage1_hf_diffusion import Stage1HFHandDiffusion, Stage1HFHandDiffusionMLP
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


def _build_object_features(data: Dict[str, Any]) -> Optional[np.ndarray]:
    """Build 15D object feature vector from PKL fields."""
    if "object_features" in data:
        return np.asarray(data["object_features"], dtype=np.float32)

    obj_pos = data.get("object_pos")
    if obj_pos is None:
        return None

    obj_pos = np.asarray(obj_pos, dtype=np.float32)
    T = obj_pos.shape[0]

    obj_rot = data.get("object_rot")
    if obj_rot is not None:
        obj_rot = np.asarray(obj_rot, dtype=np.float32)
        if obj_rot.shape[-1] == 4:
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

    obj_lin_vel = data.get("object_lin_vel")
    obj_lin_vel = np.asarray(obj_lin_vel, dtype=np.float32) if obj_lin_vel is not None else np.zeros((T, 3), dtype=np.float32)

    obj_ang_vel = data.get("object_ang_vel")
    obj_ang_vel = np.asarray(obj_ang_vel, dtype=np.float32) if obj_ang_vel is not None else np.zeros((T, 3), dtype=np.float32)

    return np.concatenate([obj_pos, obj_rot_6d, obj_lin_vel, obj_ang_vel], axis=-1)


class OptimizedHFPipeline:
    """
    Optimized HF pipeline with inference time optimizations.

    Applies precision optimization and fast sampling to both stages.
    The HF Stage 1 uses 15D object motion features instead of 3075D BPS.
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

        # Determine dtype
        if config.precision == PrecisionMode.FP16:
            self.dtype = torch.float16
        elif config.precision == PrecisionMode.BF16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Load Stage 1 (HF)
        print("Loading Stage 1 (HF)...")
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
        """Load Stage 1 HF model."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})

        window_size = dataset_cfg.get("window_size", 120)
        max_len = model_cfg.get("max_len", window_size + 100)

        object_feature_input_dim = model_cfg.get("object_feature_input_dim", 15)
        encoder_hidden = model_cfg.get("encoder_hidden", 512)
        encoder_layers = model_cfg.get("encoder_layers", 3)
        object_feature_dim = model_cfg.get("object_feature_dim", 256)
        hand_dim = model_cfg.get("hand_dim", 6)

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
        if self.config.precision == PrecisionMode.FP16:
            self.stage1_model = self.stage1_model.half()
            self.stage2_model = self.stage2_model.half()
        elif self.config.precision == PrecisionMode.BF16:
            self.stage1_model = self.stage1_model.to(torch.bfloat16)
            self.stage2_model = self.stage2_model.to(torch.bfloat16)
        elif self.config.precision == PrecisionMode.INT8:
            import warnings
            warnings.warn("INT8 quantization not recommended for diffusion models")

        if self.config.use_torch_compile and hasattr(torch, "compile"):
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

        # Warmup Stage 1 (HF: 15D object features instead of BPS)
        dummy_obj_feat = torch.randn(1, seq_len, 15, device=self.device, dtype=self.dtype)
        dummy_x1 = torch.randn(1, seq_len, 6, device=self.device, dtype=self.dtype)
        dummy_t = torch.randint(0, self.stage1_timesteps, (1,), device=self.device)

        for _ in range(self.config.warmup_iterations):
            with torch.inference_mode():
                _ = self.stage1_model(dummy_x1, dummy_t, dummy_obj_feat)

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
        object_features: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Stage 1 with fast sampler (DDIM/DPM-Solver)."""
        B, T, _ = object_features.shape

        def condition_fn():
            return (object_features,)

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
        object_features: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Stage 1 with full DDPM."""
        B, T, _ = object_features.shape
        x = torch.randn(B, T, 6, device=self.device, dtype=self.dtype)

        for n in reversed(range(self.stage1_schedule.timesteps)):
            t = torch.full((B,), n, device=self.device, dtype=torch.long)
            x0_pred = self.stage1_model(x, t, object_features)

            if n > 0:
                alpha_bar_t = self.stage1_schedule.alpha_bar[n]
                alpha_bar_t_prev = self.stage1_schedule.alpha_bar[n - 1]
                alpha_t = self.stage1_schedule.alpha[n]

                mean = (
                    torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                    + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
                )
                sigma = torch.sqrt(self.stage1_schedule.beta[n])
                x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x

    @torch.inference_mode()
    def _sample_stage2_fast(
        self,
        hands: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Stage 2 with fast sampler."""
        B, T, _ = hands.shape

        def condition_fn():
            return (hands,)

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
        hands: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Stage 2 with full DDPM."""
        B, T, _ = hands.shape
        x = torch.randn(B, T, self.state_dim, device=self.device, dtype=self.dtype)

        for n in reversed(range(self.stage2_schedule.timesteps)):
            t = torch.full((B,), n, device=self.device, dtype=torch.long)
            x0_pred = self.stage2_model(x, t, hands)

            if n > 0:
                alpha_bar_t = self.stage2_schedule.alpha_bar[n]
                alpha_bar_t_prev = self.stage2_schedule.alpha_bar[n - 1]
                alpha_t = self.stage2_schedule.alpha[n]

                mean = (
                    torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                    + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
                )
                sigma = torch.sqrt(self.stage2_schedule.beta[n])
                x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x

    def _denormalize(self, x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
        if mean is None:
            return x
        if x.ndim == 3:
            mean = mean.view(1, 1, -1).to(x.dtype)
            std = std.view(1, 1, -1).to(x.dtype)
        else:
            mean = mean.view(1, -1).to(x.dtype)
            std = std.view(1, -1).to(x.dtype)
        return x * std + mean

    def generate(
        self,
        object_features: np.ndarray,
        object_verts: Optional[np.ndarray] = None,
        object_rotation: Optional[np.ndarray] = None,
        apply_constraints: bool = True,
        partial_motion_length: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate full-body motion from object motion features (optimized).

        Args:
            object_features: (T, 15) object motion features
            object_verts: (T, K, 3) object vertices (for contact constraints)
            object_rotation: (T, 3, 3) object rotations (for contact constraints)
            apply_constraints: Whether to apply contact constraints
            partial_motion_length: If set, generate this many frames

        Returns:
            Dict with hands_raw, hands_rectified, state, root_pos, root_rot, dof_pos, etc.
        """
        T_original = object_features.shape[0]
        truncated = False
        partial = False
        target_len = None

        max_len = min(self.stage1_max_len, self.stage2_max_len)

        # Handle partial motion
        if partial_motion_length is not None and partial_motion_length > 0:
            if partial_motion_length < T_original:
                object_features = object_features[:partial_motion_length]
                if object_verts is not None:
                    object_verts = object_verts[:partial_motion_length]
                if object_rotation is not None:
                    object_rotation = object_rotation[:partial_motion_length]
                partial = True
                target_len = partial_motion_length

        T_working = object_features.shape[0]
        if T_working > max_len:
            print(f"  Warning: Truncating from {T_working} to {max_len} frames")
            object_features = object_features[:max_len]
            if object_verts is not None:
                object_verts = object_verts[:max_len]
            if object_rotation is not None:
                object_rotation = object_rotation[:max_len]
            truncated = True

        # Prepare tensors
        obj_feat = torch.from_numpy(object_features).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        # =====================================================================
        # Stage 1: Object features → Hand positions
        # =====================================================================
        use_fast = self.config.sampler != SamplerType.DDPM and self.stage1_sampler is not None

        if use_fast:
            hands_norm = self._sample_stage1_fast(obj_feat)
        else:
            hands_norm = self._sample_stage1_ddpm(obj_feat)

        hands_raw = self._denormalize(
            hands_norm,
            self.stage1_norm["hand_mean"],
            self.stage1_norm["hand_std"],
        )
        hands_raw_np = hands_raw.squeeze(0).float().cpu().numpy()

        # =====================================================================
        # Contact Constraints
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
        hands_rect = torch.from_numpy(hands_rect_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        if use_fast:
            state_norm = self._sample_stage2_fast(hands_rect)
        else:
            state_norm = self._sample_stage2_ddpm(hands_rect)

        state = self._denormalize(
            state_norm,
            self.stage2_norm["state_mean"],
            self.stage2_norm["state_std"],
        )
        state_np = state.squeeze(0).float().cpu().numpy()

        # Parse state
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


def parse_optimization_config(yml: Dict[str, Any]) -> InferenceConfig:
    """Parse optimization settings from YAML config."""
    opt = yml.get("optimization", {})

    # Parse precision
    precision_str = opt.get("precision", "fp16").lower()
    precision_map = {
        "fp32": PrecisionMode.FP32,
        "fp16": PrecisionMode.FP16,
        "bf16": PrecisionMode.BF16,
        "int8": PrecisionMode.INT8,
    }
    precision = precision_map.get(precision_str, PrecisionMode.FP16)

    # Parse sampler
    sampler_str = opt.get("sampler", "ddim").lower()
    sampler_map = {
        "ddpm": SamplerType.DDPM,
        "ddim": SamplerType.DDIM,
        "dpm_solver": SamplerType.DPM_SOLVER,
    }
    sampler = sampler_map.get(sampler_str, SamplerType.DDIM)

    return InferenceConfig(
        precision=precision,
        sampler=sampler,
        num_inference_steps=opt.get("num_inference_steps", 50),
        ddim_eta=opt.get("ddim_eta", 0.0),
        use_torch_compile=opt.get("use_torch_compile", True),
        compile_mode=opt.get("compile_mode", "reduce-overhead"),
        compile_fullgraph=opt.get("compile_fullgraph", False),
        warmup_iterations=opt.get("warmup_iterations", 3),
    )


def main():
    parser = argparse.ArgumentParser(description="Optimized Stage 2 End-to-End HF Pipeline")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./experiments/stage2_hf_optimized/sample_stage2_hf_optimized.yaml",
        help="Path to YAML experiment config file",
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    sample_yml = yml["sample"]

    root_dir = yml["root_dir"]
    stage1_ckpt_path = sample_yml["stage1_ckpt_path"]
    stage2_ckpt_path = sample_yml["stage2_ckpt_path"]
    device_str = sample_yml.get("device", "cuda")
    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    num_samples = sample_yml.get("num_samples", None)
    seed = sample_yml.get("seed", 42)
    timesteps = sample_yml.get("timesteps", 1000)
    partial_motion_length = sample_yml.get("partial_motion_length", None)

    if not stage1_ckpt_path or not stage2_ckpt_path:
        raise ValueError("stage1_ckpt_path and stage2_ckpt_path must be set in config")

    # Parse optimization config
    opt_config = parse_optimization_config(yml)
    print(f"\nOptimization config:")
    print(f"  Precision: {opt_config.precision.value}")
    print(f"  Sampler: {opt_config.sampler.value}")
    print(f"  Inference steps: {opt_config.num_inference_steps}")
    print(f"  torch.compile: {opt_config.use_torch_compile}")
    print(f"  Warmup: {opt_config.warmup_iterations} iterations")

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
        sample_folder = f"opt_{opt_config.sampler.value}{opt_config.num_inference_steps}_{opt_config.precision.value}_w{window_size}_s{stride}_{timestamp}{suffix}"
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    else:
        output_dir = os.path.join("out", "stage2_hf_optimized", timestamp)

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    # Save config for reproducibility
    import yaml

    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(yml, f, default_flow_style=False)

    # Create pipeline
    pipeline = OptimizedHFPipeline(
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        config=opt_config,
        device=device_str,
        contact_threshold=contact_threshold,
    )

    # Warmup
    dataset_yml = yml.get("dataset", {})
    seq_len = dataset_yml.get("window_size", 120)
    pipeline.warmup(seq_len=seq_len)

    # Find input files
    files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if num_samples:
        files = files[:num_samples]

    print(f"\nProcessing {len(files)} files")
    print(f"Output directory: {output_dir}")

    # Performance tracking
    total_time = 0.0
    sample_durations = []
    all_frame_counts = []

    for fpath in tqdm(files, desc="Generating (optimized)"):
        fname = os.path.basename(fpath)

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        # Build object features
        obj_feat = _build_object_features(data)
        if obj_feat is None:
            T = data.get("hand_positions", data.get("dof_pos")).shape[0]
            obj_feat = np.zeros((T, 15), dtype=np.float32)

        # Generate with timing
        start_time = time.perf_counter()
        result = pipeline.generate(
            object_features=obj_feat,
            object_verts=data.get("object_verts"),
            object_rotation=data.get("object_rotation"),
            apply_constraints=apply_constraints,
            partial_motion_length=partial_motion_length,
        )
        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        # Save
        output_data = {
            "seq_name": data.get("seq_name", fname),
            "fps": data.get("fps", 30.0),
            **result,
        }

        # Visualization compatibility
        if "hands_rectified" in result:
            output_data["hand_positions"] = result["hands_rectified"]
        if "object_pos" in data:
            output_data["object_pos"] = data["object_pos"]
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

        # Track timing
        fps = output_data["fps"]
        num_frames = result.get("root_pos", result.get("dof_pos", np.zeros((1,)))).shape[0]
        if num_frames > 0:
            all_frame_counts.append(num_frames)
            sample_durations.append(num_frames / fps)

        fname_base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{fname_base}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)

    # Performance summary
    num_files = len(files)
    avg_time = (total_time / num_files * 1000) if num_files > 0 else 0
    throughput = num_files / total_time if total_time > 0 else 0
    avg_duration = sum(sample_durations) / len(sample_durations) if sample_durations else 0

    print(f"\n{'='*60}")
    print("Optimized Performance Summary")
    print(f"{'='*60}")
    print(f"Sampler: {opt_config.sampler.value} ({opt_config.num_inference_steps} steps)")
    print(f"Precision: {opt_config.precision.value}")
    print(f"torch.compile: {opt_config.use_torch_compile}")
    print(f"Total files: {num_files}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {avg_time:.1f}ms per sample")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Average sample duration: {avg_duration:.2f}s")
    if all_frame_counts:
        print(f"Frames per motion: {int(np.mean(all_frame_counts))} avg, {min(all_frame_counts)} min, {max(all_frame_counts)} max")
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        print(f"Generated fps: {gen_fps:.1f}")
    print(f"{'='*60}")

    # Save summary
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Optimized Performance Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Sampler: {opt_config.sampler.value} ({opt_config.num_inference_steps} steps)\n")
        f.write(f"Precision: {opt_config.precision.value}\n")
        f.write(f"torch.compile: {opt_config.use_torch_compile}\n")
        f.write(f"Total files: {num_files}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Average time: {avg_time:.1f}ms per sample\n")
        f.write(f"Throughput: {throughput:.2f} samples/sec\n")
        if all_frame_counts:
            f.write(f"Frames per motion: {int(np.mean(all_frame_counts))} avg\n")
            total_frames = sum(all_frame_counts)
            gen_fps = total_frames / total_time if total_time > 0 else 0
            f.write(f"Generated fps: {gen_fps:.1f}\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
