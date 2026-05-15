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
from utils.rotation import quat_to_rot6d_xyzw, rot6d_to_quat_xyzw, mat_to_quat_xyzw
from utils.general import load_config
from utils.motion_postprocess import smooth_body_motion_np
from utils.robot_kinematics import (
    apply_robot_contact_root_correction,
    apply_robot_contact_state_refinement,
    robot_hand_positions,
    robot_hand_positions_from_state_torch,
)
from utils.object_conditioning import (
    apply_object_conditioning_variant,
    describe_object_conditioning_variant,
    normalize_object_conditioning_variant,
)

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
        body_smooth_strength: float = 0.0,
        body_smooth_window: int = 5,
        body_smooth_iterations: int = 1,
        stage1_contact_search_threshold: Optional[float] = None,
        stage1_max_contact_offset: Optional[float] = 0.02,
        stage1_max_contact_correction: Optional[float] = 0.06,
        stage1_fallback_contact_search_threshold: Optional[float] = None,
        stage1_fallback_max_contact_correction: Optional[float] = None,
        robot_contact_correction: bool = False,
        robot_contact_activation_threshold: float = 0.12,
        robot_contact_max_translation: float = 0.08,
        robot_contact_smooth_strength: float = 0.55,
        robot_contact_smooth_window: int = 9,
        robot_contact_smooth_iterations: int = 2,
        robot_contact_guidance: bool = False,
        robot_contact_guidance_steps: int = 3,
        robot_contact_guidance_lr: float = 0.08,
        robot_contact_guidance_activation_threshold: float = 0.12,
        robot_contact_guidance_surface_weight: float = 0.7,
        robot_contact_guidance_margin: float = 0.0,
        robot_contact_guidance_max_delta: float = 0.35,
        robot_contact_guidance_mode: str = "upper",
        robot_contact_refinement: bool = False,
        robot_contact_refinement_steps: int = 20,
        robot_contact_refinement_lr: float = 0.03,
        robot_contact_refinement_activation_threshold: float = 0.16,
        robot_contact_refinement_pose_reg_weight: float = 0.002,
        robot_contact_refinement_velocity_reg_weight: float = 0.02,
        robot_contact_refinement_acceleration_reg_weight: float = 0.0,
        robot_contact_refinement_max_joint_delta: float = 0.35,
        robot_contact_refinement_mode: str = "upper",
        allow_full_length: bool = False,
    ):
        self.device = torch.device(device)
        self.config = config
        self.contact_threshold = contact_threshold
        self.body_smooth_strength = float(body_smooth_strength)
        self.body_smooth_window = int(body_smooth_window)
        self.body_smooth_iterations = int(body_smooth_iterations)
        self.stage1_contact_search_threshold = stage1_contact_search_threshold
        self.stage1_max_contact_offset = stage1_max_contact_offset
        self.stage1_max_contact_correction = stage1_max_contact_correction
        self.stage1_fallback_contact_search_threshold = stage1_fallback_contact_search_threshold
        self.stage1_fallback_max_contact_correction = stage1_fallback_max_contact_correction
        self.robot_contact_correction = bool(robot_contact_correction)
        self.robot_contact_activation_threshold = float(robot_contact_activation_threshold)
        self.robot_contact_max_translation = float(robot_contact_max_translation)
        self.robot_contact_smooth_strength = float(robot_contact_smooth_strength)
        self.robot_contact_smooth_window = int(robot_contact_smooth_window)
        self.robot_contact_smooth_iterations = int(robot_contact_smooth_iterations)
        self.robot_contact_guidance = bool(robot_contact_guidance)
        self.robot_contact_guidance_steps = int(robot_contact_guidance_steps)
        self.robot_contact_guidance_lr = float(robot_contact_guidance_lr)
        self.robot_contact_guidance_activation_threshold = float(
            robot_contact_guidance_activation_threshold
        )
        self.robot_contact_guidance_surface_weight = float(robot_contact_guidance_surface_weight)
        self.robot_contact_guidance_margin = float(robot_contact_guidance_margin)
        self.robot_contact_guidance_max_delta = float(robot_contact_guidance_max_delta)
        self.robot_contact_guidance_mode = str(robot_contact_guidance_mode).lower()
        self.robot_contact_refinement = bool(robot_contact_refinement)
        self.robot_contact_refinement_steps = int(robot_contact_refinement_steps)
        self.robot_contact_refinement_lr = float(robot_contact_refinement_lr)
        self.robot_contact_refinement_activation_threshold = float(
            robot_contact_refinement_activation_threshold
        )
        self.robot_contact_refinement_pose_reg_weight = float(
            robot_contact_refinement_pose_reg_weight
        )
        self.robot_contact_refinement_velocity_reg_weight = float(
            robot_contact_refinement_velocity_reg_weight
        )
        self.robot_contact_refinement_acceleration_reg_weight = float(
            robot_contact_refinement_acceleration_reg_weight
        )
        self.robot_contact_refinement_max_joint_delta = float(
            robot_contact_refinement_max_joint_delta
        )
        self.robot_contact_refinement_mode = str(robot_contact_refinement_mode).lower()
        self.allow_full_length = bool(allow_full_length)
        
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
        self.object_conditioning_variant = normalize_object_conditioning_variant(
            self.stage1_checkpoint_variant
        )
        print(f"Object conditioning: {describe_object_conditioning_variant(self.object_conditioning_variant)}")
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Create samplers
        self._create_samplers()
        
        # Contact processor
        self.contact_processor = ContactConstraintProcessor(
            contact_threshold=contact_threshold,
            contact_search_threshold=stage1_contact_search_threshold,
            max_contact_offset=stage1_max_contact_offset,
            max_contact_correction=stage1_max_contact_correction,
            fallback_contact_search_threshold=stage1_fallback_contact_search_threshold,
            fallback_max_contact_correction=stage1_fallback_max_contact_correction,
        )
        
        self._is_warmed_up = False
    
    def _load_stage1(self, ckpt_path: str):
        """Load Stage 1 model."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        self.stage1_checkpoint_variant = normalize_object_conditioning_variant(
            dataset_cfg.get("object_conditioning_variant", "variant0")
        )
        
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
                encoder_hidden=model_cfg.get("encoder_hidden", 512),
                object_feature_dim=model_cfg.get("object_feature_dim", 256),
                encoder_layers=model_cfg.get("encoder_layers", 3),
                hand_dim=model_cfg.get("hand_dim", 6),
                denoiser_hidden=model_cfg.get("denoiser_hidden", 512),
                denoiser_layers=model_cfg.get("denoiser_layers", 4),
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
                hidden_dim=model_cfg.get("mlp_hidden", 512),
                num_layers=model_cfg.get("mlp_layers", 4),
            )
        
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        
        timesteps = config.get("train", {}).get("timesteps", 1000)
        
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "state_mean": norm_stats.get("state_mean"),
            "state_std": norm_stats.get("state_std"),
            "hand_mean": norm_stats.get("hand_mean"),
            "hand_std": norm_stats.get("hand_std"),
            "normalize_hands": dataset_cfg.get("normalize_hands", False),
        }
        if norm["state_mean"] is not None:
            norm["state_mean"] = norm["state_mean"].to(self.device)
            norm["state_std"] = norm["state_std"].to(self.device)
        if norm["hand_mean"] is not None:
            norm["hand_mean"] = norm["hand_mean"].to(self.device)
            norm["hand_std"] = norm["hand_std"].to(self.device)
        
        return model, timesteps, norm, max_len, state_dim

    def _prepare_stage2_condition(self, hands: torch.Tensor) -> torch.Tensor:
        if not self.stage2_norm.get("normalize_hands", False):
            return hands
        hand_mean = self.stage2_norm.get("hand_mean")
        hand_std = self.stage2_norm.get("hand_std")
        if hand_mean is None or hand_std is None:
            return hands
        mean = hand_mean.to(device=hands.device, dtype=hands.dtype).view(1, 1, -1)
        std = hand_std.to(device=hands.device, dtype=hands.dtype).view(1, 1, -1)
        return (hands - mean) / std
    
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

    def _stage2_guidance_state_mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(x)
        mode = self.robot_contact_guidance_mode

        if mode == "root":
            mask[..., :3] = 1.0
        elif mode == "root_pose":
            mask[..., :9] = 1.0
        elif mode in ("upper", "arms", "all"):
            if mode == "all":
                mask[...] = 1.0
            else:
                mask[..., :9] = 1.0
                # State layout is [root_pos(3), root_rot_6d(6), dof_pos(29)].
                # DoF 12:28 are waist and arm joints for the G1 model.
                mask[..., 9 + 12 : 9 + 29] = 1.0
        else:
            mask[..., :3] = 1.0

        return mask

    def _prepare_robot_contact_guidance_targets(
        self,
        hands_rectified: np.ndarray,
        object_verts: Optional[np.ndarray],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, object]]:
        if object_verts is None:
            target = torch.from_numpy(hands_rectified).to(self.device, dtype=torch.float32)
            mask = torch.ones((target.shape[0], 2), device=self.device, dtype=torch.float32)
            return target.unsqueeze(0), mask.unsqueeze(0), {
                "available": True,
                "source": "hands_rectified",
                "active_frames": int(target.shape[0]),
                "left_active_frames": int(target.shape[0]),
                "right_active_frames": int(target.shape[0]),
            }

        hands = np.asarray(hands_rectified, dtype=np.float32)
        verts = np.asarray(object_verts, dtype=np.float32)
        T = min(hands.shape[0], verts.shape[0])
        if T == 0:
            return None, None, {"available": False, "reason": "empty sequence"}

        target = hands[:T].reshape(T, 2, 3).copy()
        active = np.zeros((T, 2), dtype=np.float32)
        surface_weight = float(np.clip(self.robot_contact_guidance_surface_weight, 0.0, 1.0))

        for frame in range(T):
            verts_t = verts[frame]
            for hand_idx in range(2):
                hand = hands[frame, hand_idx * 3 : hand_idx * 3 + 3]
                dists = np.linalg.norm(verts_t - hand[None, :], axis=1)
                nearest_idx = int(np.argmin(dists))
                nearest = verts_t[nearest_idx]
                if float(dists[nearest_idx]) <= self.robot_contact_guidance_activation_threshold:
                    active[frame, hand_idx] = 1.0
                    target[frame, hand_idx] = (
                        (1.0 - surface_weight) * hand + surface_weight * nearest
                    )

        if not np.any(active):
            return None, None, {
                "available": False,
                "reason": "no active Stage 1/object contact frames",
                "activation_threshold": self.robot_contact_guidance_activation_threshold,
            }

        target_t = torch.from_numpy(target.reshape(T, 6)).to(self.device, dtype=torch.float32)
        active_t = torch.from_numpy(active).to(self.device, dtype=torch.float32)
        return target_t.unsqueeze(0), active_t.unsqueeze(0), {
            "available": True,
            "source": "object_surface_anchor_blend",
            "active_frames": int(np.sum(np.any(active > 0.0, axis=1))),
            "left_active_frames": int(np.sum(active[:, 0] > 0.0)),
            "right_active_frames": int(np.sum(active[:, 1] > 0.0)),
            "activation_threshold": self.robot_contact_guidance_activation_threshold,
            "surface_weight": surface_weight,
        }

    def _guide_stage2_x0(
        self,
        x0_pred: torch.Tensor,
        target_hands: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        if active_mask.sum() <= 0:
            return x0_pred, 0.0

        x0 = x0_pred.detach().float().clone().requires_grad_(True)
        state = self._denormalize(
            x0,
            self.stage2_norm["state_mean"],
            self.stage2_norm["state_std"],
        )
        robot_hands = robot_hand_positions_from_state_torch(state).view(
            state.shape[0], state.shape[1], 2, 3
        )
        target = target_hands[:, : state.shape[1]].view(state.shape[0], state.shape[1], 2, 3)
        mask = active_mask[:, : state.shape[1]].unsqueeze(-1)

        delta = robot_hands - target
        distance = torch.linalg.norm(delta, dim=-1)
        if self.robot_contact_guidance_margin > 0.0:
            contact_error = torch.relu(distance - self.robot_contact_guidance_margin).pow(2)
            energy = (contact_error * active_mask[:, : state.shape[1]]).sum()
        else:
            energy = (delta.pow(2) * mask).sum()
        energy = energy / active_mask[:, : state.shape[1]].sum().clamp_min(1.0)

        grad = torch.autograd.grad(energy, x0, retain_graph=False, create_graph=False)[0]
        grad = grad * self._stage2_guidance_state_mask(grad)
        guided = x0 - self.robot_contact_guidance_lr * grad

        if self.robot_contact_guidance_max_delta > 0.0:
            step_delta = (guided - x0).clamp(
                -self.robot_contact_guidance_max_delta,
                self.robot_contact_guidance_max_delta,
            )
            guided = x0 + step_delta

        return guided.detach().to(dtype=x0_pred.dtype), float(energy.detach().cpu())

    def _sample_stage2_fast_guided(
        self,
        cond: torch.Tensor,
        target_hands: Optional[torch.Tensor],
        active_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if (
            not self.robot_contact_guidance
            or target_hands is None
            or active_mask is None
            or not isinstance(self.stage2_sampler, DDIMSampler)
        ):
            return self._sample_stage2_fast(cond), {"applied": False}

        B, T, _ = cond.shape
        x = torch.randn((B, T, self.state_dim), device=self.device, dtype=self.dtype)
        guide_steps = max(0, min(self.robot_contact_guidance_steps, len(self.stage2_sampler.timesteps)))
        guide_start = len(self.stage2_sampler.timesteps) - guide_steps
        energies = []

        for step_idx, t in enumerate(self.stage2_sampler.timesteps):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                x0_pred = self.stage2_model(x, t_batch, cond)

            if step_idx >= guide_start:
                x0_pred, energy = self._guide_stage2_x0(x0_pred, target_hands, active_mask)
                energies.append(energy)

            if step_idx < len(self.stage2_sampler.timesteps) - 1:
                alpha_t = self.stage2_sampler.alphas_cumprod[t]
                alpha_prev = self.stage2_sampler.alphas_cumprod[
                    self.stage2_sampler.timesteps[step_idx + 1]
                ]
                pred_noise = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t)
                sigma = self.stage2_sampler.eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                )
                dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise
                x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
                if self.stage2_sampler.eta > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        metadata: Dict[str, object] = {
            "applied": True,
            "steps": int(guide_steps),
            "lr": self.robot_contact_guidance_lr,
            "mode": self.robot_contact_guidance_mode,
            "max_delta": self.robot_contact_guidance_max_delta,
        }
        if energies:
            metadata["energy_first"] = float(energies[0])
            metadata["energy_last"] = float(energies[-1])
        return x, metadata
    
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
    
    def generate(
        self,
        bps_encoding: np.ndarray,
        object_centroid: np.ndarray,
        object_verts: Optional[np.ndarray] = None,
        object_rotation: Optional[np.ndarray] = None,
        contact_labels: Optional[np.ndarray] = None,
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
                if contact_labels is not None:
                    contact_labels = contact_labels[:partial_motion_length]
                partial = True
                target_len = partial_motion_length
            else:
                print(f"  Warning: partial_motion_length ({partial_motion_length}) >= object motion length ({T_original}), using full length")
        
        # Update working length to reflect potential partial motion
        T_working = object_centroid.shape[0]
        
        # Truncate if sequence exceeds the configured checkpoint max_len unless
        # explicitly running a full-length extrapolation pass.
        max_len = self.stage1_max_len  # Use stage1 max_len as reference
        if T_working > max_len and not self.allow_full_length:
            print(f"  Warning: Truncating sequence from {T_working} to {max_len} frames")
            bps_encoding = bps_encoding[:max_len]
            object_centroid = object_centroid[:max_len]
            if object_verts is not None:
                object_verts = object_verts[:max_len]
            if object_rotation is not None:
                object_rotation = object_rotation[:max_len]
            if contact_labels is not None:
                contact_labels = contact_labels[:max_len]
            truncated = True
        elif T_working > max_len:
            print(
                f"  Warning: Running full length {T_working} frames "
                f"past checkpoint max_len={max_len}"
            )
        
        object_conditioning = apply_object_conditioning_variant(
            variant=self.object_conditioning_variant,
            bps_encoding=bps_encoding,
            object_centroid=object_centroid,
        )
        bps_encoding = object_conditioning["bps_encoding"]
        object_centroid = object_conditioning["object_centroid"]

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
                hands_raw_np, object_verts, object_rotation, contact_labels=contact_labels
            )
            hands_rect = torch.from_numpy(hands_rect_np).to(self.device, dtype=self.dtype)
            hands_rect = hands_rect.unsqueeze(0)
        else:
            hands_rect = hands_raw
            hands_rect_np = hands_raw_np
            contact_metadata = None
        
        # Stage 2: Generate full body
        stage2_cond = self._prepare_stage2_condition(hands_rect)
        if self.stage2_sampler is not None:
            guidance_target, guidance_mask, guidance_metadata = (
                self._prepare_robot_contact_guidance_targets(hands_rect_np, object_verts)
                if self.robot_contact_guidance
                else (None, None, {"available": False})
            )
            state_norm, robot_contact_guidance_metadata = self._sample_stage2_fast_guided(
                stage2_cond,
                guidance_target,
                guidance_mask,
            )
            robot_contact_guidance_metadata.update(guidance_metadata)
        else:
            state_norm = self._sample_stage2_ddpm(stage2_cond)
            robot_contact_guidance_metadata = {"applied": False, "reason": "ddpm sampler"}
        
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
        root_pos, root_rot_quat, dof_pos = smooth_body_motion_np(
            root_pos,
            root_rot_quat,
            dof_pos,
            strength=self.body_smooth_strength,
            window=self.body_smooth_window,
            iterations=self.body_smooth_iterations,
        )

        robot_contact_metadata = None
        if self.robot_contact_correction:
            root_pos_corrected, robot_contact_metadata = apply_robot_contact_root_correction(
                root_pos=root_pos,
                root_rot_xyzw=root_rot_quat,
                dof_pos=dof_pos,
                target_hands=hands_rect_np,
                object_verts=object_verts,
                contact_threshold=self.contact_threshold,
                activation_threshold=self.robot_contact_activation_threshold,
                max_translation=self.robot_contact_max_translation,
                smooth_strength=self.robot_contact_smooth_strength,
                smooth_window=self.robot_contact_smooth_window,
                smooth_iterations=self.robot_contact_smooth_iterations,
            )
            root_pos = root_pos_corrected
            state_np[:, :3] = root_pos

        robot_contact_refinement_metadata = None
        if self.robot_contact_refinement:
            root_pos, root_rot_quat, dof_pos, robot_contact_refinement_metadata = (
                apply_robot_contact_state_refinement(
                    root_pos=root_pos,
                    root_rot_xyzw=root_rot_quat,
                    dof_pos=dof_pos,
                    target_hands=hands_rect_np,
                    object_verts=object_verts,
                    activation_threshold=self.robot_contact_refinement_activation_threshold,
                    steps=self.robot_contact_refinement_steps,
                    lr=self.robot_contact_refinement_lr,
                    pose_reg_weight=self.robot_contact_refinement_pose_reg_weight,
                    velocity_reg_weight=self.robot_contact_refinement_velocity_reg_weight,
                    acceleration_reg_weight=self.robot_contact_refinement_acceleration_reg_weight,
                    max_joint_delta=self.robot_contact_refinement_max_joint_delta,
                    mode=self.robot_contact_refinement_mode,
                    device=str(self.device),
                )
            )
            state_np[:, :3] = root_pos
            state_np[:, 3:9] = quat_to_rot6d_xyzw(torch.from_numpy(root_rot_quat).float()).numpy()
            state_np[:, 9:] = dof_pos

        robot_hands = robot_hand_positions(root_pos, root_rot_quat, dof_pos)
        
        return {
            "hands_raw": hands_raw_np,
            "hands_rectified": hands_rect_np,
            "contact_metadata": contact_metadata,
            "robot_hands": robot_hands,
            "robot_contact_metadata": robot_contact_metadata,
            "robot_contact_guidance_metadata": robot_contact_guidance_metadata,
            "robot_contact_refinement_metadata": robot_contact_refinement_metadata,
            "state": state_np,
            "root_pos": root_pos,
            "root_rot": root_rot_quat,
            "dof_pos": dof_pos,
            "truncated": truncated,
            "original_len": T_original,
            "partial": partial,
            "target_len": target_len,
            "object_conditioning_variant": self.object_conditioning_variant,
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
        body_smooth_strength=sample_yml.get("body_smooth_strength", 0.0),
        body_smooth_window=sample_yml.get("body_smooth_window", 5),
        body_smooth_iterations=sample_yml.get("body_smooth_iterations", 1),
        stage1_contact_search_threshold=sample_yml.get("stage1_contact_search_threshold", None),
        stage1_max_contact_offset=sample_yml.get("stage1_max_contact_offset", 0.02),
        stage1_max_contact_correction=sample_yml.get("stage1_max_contact_correction", 0.06),
        stage1_fallback_contact_search_threshold=sample_yml.get(
            "stage1_fallback_contact_search_threshold", None
        ),
        stage1_fallback_max_contact_correction=sample_yml.get(
            "stage1_fallback_max_contact_correction", None
        ),
        robot_contact_correction=sample_yml.get("robot_contact_correction", False),
        robot_contact_activation_threshold=sample_yml.get("robot_contact_activation_threshold", 0.12),
        robot_contact_max_translation=sample_yml.get("robot_contact_max_translation", 0.08),
        robot_contact_smooth_strength=sample_yml.get("robot_contact_smooth_strength", 0.55),
        robot_contact_smooth_window=sample_yml.get("robot_contact_smooth_window", 9),
        robot_contact_smooth_iterations=sample_yml.get("robot_contact_smooth_iterations", 2),
        robot_contact_guidance=sample_yml.get("robot_contact_guidance", False),
        robot_contact_guidance_steps=sample_yml.get("robot_contact_guidance_steps", 3),
        robot_contact_guidance_lr=sample_yml.get("robot_contact_guidance_lr", 0.08),
        robot_contact_guidance_activation_threshold=sample_yml.get(
            "robot_contact_guidance_activation_threshold", 0.12
        ),
        robot_contact_guidance_surface_weight=sample_yml.get(
            "robot_contact_guidance_surface_weight", 0.7
        ),
        robot_contact_guidance_margin=sample_yml.get("robot_contact_guidance_margin", 0.0),
        robot_contact_guidance_max_delta=sample_yml.get("robot_contact_guidance_max_delta", 0.35),
        robot_contact_guidance_mode=sample_yml.get("robot_contact_guidance_mode", "upper"),
        robot_contact_refinement=sample_yml.get("robot_contact_refinement", False),
        robot_contact_refinement_steps=sample_yml.get("robot_contact_refinement_steps", 20),
        robot_contact_refinement_lr=sample_yml.get("robot_contact_refinement_lr", 0.03),
        robot_contact_refinement_activation_threshold=sample_yml.get(
            "robot_contact_refinement_activation_threshold", 0.16
        ),
        robot_contact_refinement_pose_reg_weight=sample_yml.get(
            "robot_contact_refinement_pose_reg_weight", 0.002
        ),
        robot_contact_refinement_velocity_reg_weight=sample_yml.get(
            "robot_contact_refinement_velocity_reg_weight", 0.02
        ),
        robot_contact_refinement_acceleration_reg_weight=sample_yml.get(
            "robot_contact_refinement_acceleration_reg_weight", 0.0
        ),
        robot_contact_refinement_max_joint_delta=sample_yml.get(
            "robot_contact_refinement_max_joint_delta", 0.35
        ),
        robot_contact_refinement_mode=sample_yml.get("robot_contact_refinement_mode", "upper"),
        allow_full_length=sample_yml.get("allow_full_length", False),
    )
    
    # Output directory - match original naming pattern + optional exp_name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    # Get optional experiment name for folder suffix
    exp_name = yml.get("exp_name", "")
    suffix = f"_{exp_name}" if exp_name else ""
    
    timesteps = 1000  # Default timesteps
    
    ckpt_parts = stage2_ckpt_path.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        sample_folder = f"ts{timesteps}_{timestamp}{suffix}"
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
    print(f"Object conditioning: {describe_object_conditioning_variant(pipeline.object_conditioning_variant)}")
    
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
            contact_labels=data.get("contact"),
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
        for key in ("object_name", "mesh_file", "num_verts", "is_articulated", "object_mesh_scale"):
            if key in data:
                output_data[key] = data[key]
        
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
