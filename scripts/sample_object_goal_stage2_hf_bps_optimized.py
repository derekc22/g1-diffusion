"""
Optimized object-goal Stage 2/end-to-end sampler.

This is a copied-and-adapted optimized sampler path for the corrected
object-goal two-stage model. It intentionally leaves the legacy optimized
sampler untouched.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
import types
from datetime import datetime
from typing import Any, Dict, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from tqdm import tqdm

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

from models.stage1_diffusion import Stage1HandDiffusion, Stage1HandDiffusionMLP
from models.stage2_diffusion import Stage2MLPModel, Stage2TransformerModel
from utils.contact_constraints import ContactConstraintProcessor
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.general import load_config, load_torch_checkpoint
from utils.inference_optimization import (
    InferenceConfig,
    PrecisionMode,
    SamplerType,
    create_sampler,
)
from utils.object_conditioning import apply_object_conditioning_variant, normalize_object_conditioning_variant
from utils.object_goal_features import OBJECT_POSE_DIM, ROBOT_STATE_DIM, object_pose_from_data, robot_object_layout
from utils.rotation import rot6d_to_quat_xyzw


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _require_checkpoint_path(sample_cfg: dict[str, Any], key: str, config_path: str) -> str:
    path = str(sample_cfg.get(key) or "").strip()
    if not path:
        raise ValueError(
            f"Missing {key}. Edit {config_path} and set:\n"
            "  sample.stage1_ckpt_path\n"
            "  sample.stage2_ckpt_path"
        )
    return _resolve(path)


def _stat_to_device(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _normalize(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mean is None or std is None:
        return x
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    if x.ndim == 3:
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    else:
        mean = mean.view(1, -1)
        std = std.view(1, -1)
    return (x - mean) / std.clamp_min(1e-8)


def _denormalize(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mean is None or std is None:
        return x
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    if x.ndim == 3:
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    else:
        mean = mean.view(1, -1)
        std = std.view(1, -1)
    return x * std + mean


def _prepare_object_inputs(data: dict[str, Any], max_len: int, variant: str):
    bps = np.asarray(data["bps_encoding"], dtype=np.float32)
    centroid = np.asarray(data["object_centroid"], dtype=np.float32)
    object_pose = object_pose_from_data(data)
    T = min(bps.shape[0], centroid.shape[0], object_pose.shape[0], max_len)
    bps = bps[:T]
    centroid = centroid[:T]
    object_pose = object_pose[:T]
    conditioned = apply_object_conditioning_variant(
        variant=variant,
        bps_encoding=bps,
        object_centroid=centroid,
    )
    bps = conditioned["bps_encoding"]
    centroid = conditioned["object_centroid"]
    if bps.ndim == 3:
        bps = bps.reshape(T, -1)
    static_bps = np.repeat(bps[:1], T, axis=0)
    return bps, centroid, object_pose, static_bps


def _output_dir_from_stage2_checkpoint(stage2_ckpt_path: str, yml: dict[str, Any], inf_config: InferenceConfig) -> str:
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = yml.get("exp_name", "")
    suffix = f"_{exp_name}" if exp_name else ""
    step_label = (
        str(inf_config.num_inference_steps)
        if inf_config.sampler != SamplerType.DDPM
        else str(inf_config.sampler.value)
    )
    sample_folder = f"ts{step_label}_{timestamp}{suffix}"

    ckpt_parts = stage2_ckpt_path.split(os.sep)
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        return os.path.join(PROJECT_ROOT, "logs", log_id, "samples", sample_folder)

    return os.path.join(PROJECT_ROOT, "out", f"object_goal_stage2_hf_bps_optimized_{timestamp}{suffix}")


class _Stage1GlobalCondAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, t, bps, centroid, object_pose, goal):
        return self.model(x, t, bps, centroid, object_pose=object_pose, global_cond=goal)


class _Stage2GlobalCondAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, t, cond, goal):
        return self.model(x, t, cond, global_cond=goal)


def build_inference_config_from_yaml(opt_yml: dict) -> InferenceConfig:
    config = InferenceConfig()
    if "precision" in opt_yml:
        config.precision = PrecisionMode(opt_yml["precision"])
    if "sampler" in opt_yml:
        config.sampler = SamplerType(opt_yml["sampler"])
    if "num_inference_steps" in opt_yml:
        config.num_inference_steps = int(opt_yml["num_inference_steps"])
    if "ddim_eta" in opt_yml:
        config.ddim_eta = float(opt_yml["ddim_eta"])
    if "use_torch_compile" in opt_yml:
        config.use_torch_compile = bool(opt_yml["use_torch_compile"])
    if "compile_mode" in opt_yml:
        config.compile_mode = opt_yml["compile_mode"]
    if "compile_fullgraph" in opt_yml:
        config.compile_fullgraph = bool(opt_yml["compile_fullgraph"])
    if "warmup_iterations" in opt_yml:
        config.warmup_iterations = int(opt_yml["warmup_iterations"])
    return config


class OptimizedObjectGoalStage2Pipeline:
    def __init__(
        self,
        stage1_ckpt_path: str,
        stage2_ckpt_path: str,
        config: InferenceConfig,
        device: str = "cuda:0",
        contact_threshold: float = 0.03,
        stage1_contact_search_threshold: Optional[float] = None,
        stage1_max_contact_offset: Optional[float] = 0.02,
        stage1_max_contact_correction: Optional[float] = 0.06,
        stage1_fallback_contact_search_threshold: Optional[float] = None,
        stage1_fallback_max_contact_correction: Optional[float] = None,
        require_contact_geometry: bool = True,
    ):
        self.device = torch.device(device)
        self.config = config
        self.require_contact_geometry = bool(require_contact_geometry)

        if config.precision == PrecisionMode.FP16:
            self.dtype = torch.float16
        elif config.precision == PrecisionMode.BF16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        print("Loading object-goal Stage 1...")
        self.stage1_model, self.stage1_timesteps, self.stage1_norm, self.stage1_max_len, self.stage1_schedule_cfg, self.object_conditioning_variant = self._load_stage1(stage1_ckpt_path)

        print("Loading object-goal Stage 2...")
        self.stage2_model, self.stage2_timesteps, self.stage2_norm, self.stage2_max_len, self.stage2_schedule_cfg, self.state_dim, self.cond_dim = self._load_stage2(stage2_ckpt_path)

        self._apply_optimizations()
        self._create_samplers()
        self.stage1_fast_model = _Stage1GlobalCondAdapter(self.stage1_model)
        self.stage2_fast_model = _Stage2GlobalCondAdapter(self.stage2_model)

        self.contact_processor = ContactConstraintProcessor(
            contact_threshold=contact_threshold,
            contact_search_threshold=stage1_contact_search_threshold,
            max_contact_offset=stage1_max_contact_offset,
            max_contact_correction=stage1_max_contact_correction,
            fallback_contact_search_threshold=stage1_fallback_contact_search_threshold,
            fallback_max_contact_correction=stage1_fallback_max_contact_correction,
        )

    def _load_stage1(self, ckpt_path: str):
        ckpt = load_torch_checkpoint(ckpt_path, map_location=self.device)
        if ckpt.get("pipeline_type") != "object_goal_two_stage" or ckpt.get("stage") != 1:
            raise ValueError("stage1_ckpt_path must point to an object-goal two-stage Stage 1 checkpoint")

        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        window_size = int(dataset_cfg.get("window_size", 300))
        common = {
            "bps_dim": int(model_cfg.get("bps_dim", 3072)),
            "centroid_dim": int(model_cfg.get("centroid_dim", 3)),
            "encoder_hidden": int(model_cfg.get("encoder_hidden", 512)),
            "object_feature_dim": int(model_cfg.get("object_feature_dim", 256)),
            "encoder_layers": int(model_cfg.get("encoder_layers", 3)),
            "hand_dim": int(model_cfg.get("hand_dim", 6)),
            "object_pose_dim": OBJECT_POSE_DIM,
            "global_cond_dim": OBJECT_POSE_DIM,
            "global_cond_hidden": model_cfg.get("global_cond_hidden"),
        }
        if arch == "transformer":
            model = Stage1HandDiffusion(
                **common,
                d_model=int(model_cfg.get("d_model", 256)),
                nhead=int(model_cfg.get("nhead", 4)),
                num_transformer_layers=int(model_cfg.get("num_layers", 4)),
                dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                max_len=int(model_cfg.get("max_len", window_size)),
            )
        else:
            model = Stage1HandDiffusionMLP(
                **common,
                denoiser_hidden=int(model_cfg.get("denoiser_hidden", 512)),
                denoiser_layers=int(model_cfg.get("denoiser_layers", 4)),
            )
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()

        train_cfg = config.get("train", {})
        schedule = ckpt.get("schedule", {})
        schedule_cfg = {
            "timesteps": int(schedule.get("timesteps", train_cfg.get("timesteps", 1000))),
            "beta_start": float(schedule.get("beta_start", train_cfg.get("beta_start", 1e-4))),
            "beta_end": float(schedule.get("beta_end", train_cfg.get("beta_end", 0.02))),
        }
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "hand_mean": _stat_to_device(norm_stats.get("hand_mean"), self.device),
            "hand_std": _stat_to_device(norm_stats.get("hand_std"), self.device),
            "goal_mean": _stat_to_device(norm_stats.get("goal_mean"), self.device),
            "goal_std": _stat_to_device(norm_stats.get("goal_std"), self.device),
        }
        variant = normalize_object_conditioning_variant(dataset_cfg.get("object_conditioning_variant", "variant0"))
        return model, schedule_cfg["timesteps"], norm, int(model_cfg.get("max_len", window_size)), schedule_cfg, variant

    def _load_stage2(self, ckpt_path: str):
        ckpt = load_torch_checkpoint(ckpt_path, map_location=self.device)
        if ckpt.get("pipeline_type") != "object_goal_two_stage" or ckpt.get("stage") != 2:
            raise ValueError("stage2_ckpt_path must point to an object-goal two-stage Stage 2 checkpoint")

        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        window_size = int(dataset_cfg.get("window_size", 300))
        state = ckpt["model"]
        if "out_proj.weight" in state:
            state_dim = int(state["out_proj.weight"].shape[0])
            input_dim = int(state["state_proj.weight"].shape[1])
            cond_dim = input_dim - state_dim
        else:
            state_dim = int(ckpt.get("state_dim", model_cfg.get("state_dim", 47)))
            cond_dim = int(ckpt.get("cond_dim", model_cfg.get("cond_dim", 3078)))
        if int(ckpt.get("state_dim", state_dim)) != 47 or int(ckpt.get("cond_dim", cond_dim)) != 3078:
            raise ValueError("Stage 2 checkpoint must be object-goal 47D target with 3078D condition")

        common = {
            "state_dim": state_dim,
            "cond_dim": cond_dim,
            "global_cond_dim": OBJECT_POSE_DIM,
            "global_cond_hidden": model_cfg.get("global_cond_hidden"),
            "contact_dim": int(model_cfg.get("contact_dim", 0)),
        }
        if arch == "transformer":
            model = Stage2TransformerModel(
                **common,
                d_model=int(model_cfg.get("d_model", 512)),
                nhead=int(model_cfg.get("nhead", 8)),
                num_layers=int(model_cfg.get("num_layers", 8)),
                dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                max_len=int(model_cfg.get("max_len", window_size)),
            )
        else:
            model = Stage2MLPModel(
                state_dim=common["state_dim"],
                cond_dim=common["cond_dim"],
                global_cond_dim=common["global_cond_dim"],
                contact_dim=common["contact_dim"],
                hidden_dim=int(model_cfg.get("mlp_hidden", 512)),
                num_layers=int(model_cfg.get("mlp_layers", 4)),
            )
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()

        train_cfg = config.get("train", {})
        schedule = ckpt.get("schedule", {})
        schedule_cfg = {
            "timesteps": int(schedule.get("timesteps", train_cfg.get("timesteps", 1000))),
            "beta_start": float(schedule.get("beta_start", train_cfg.get("beta_start", 1e-4))),
            "beta_end": float(schedule.get("beta_end", train_cfg.get("beta_end", 0.02))),
        }
        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "state_mean": _stat_to_device(norm_stats.get("state_mean"), self.device),
            "state_std": _stat_to_device(norm_stats.get("state_std"), self.device),
            "hand_mean": _stat_to_device(norm_stats.get("hand_mean"), self.device),
            "hand_std": _stat_to_device(norm_stats.get("hand_std"), self.device),
            "goal_mean": _stat_to_device(norm_stats.get("goal_mean"), self.device),
            "goal_std": _stat_to_device(norm_stats.get("goal_std"), self.device),
            "normalize_hands": bool(dataset_cfg.get("normalize_hands", False)),
        }
        return model, schedule_cfg["timesteps"], norm, int(model_cfg.get("max_len", window_size)), schedule_cfg, state_dim, cond_dim

    def _apply_optimizations(self) -> None:
        if self.config.precision == PrecisionMode.FP16:
            self.stage1_model = self.stage1_model.half()
            self.stage2_model = self.stage2_model.half()
        elif self.config.precision == PrecisionMode.BF16:
            self.stage1_model = self.stage1_model.to(torch.bfloat16)
            self.stage2_model = self.stage2_model.to(torch.bfloat16)

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
            except Exception as exc:
                print(f"torch.compile failed: {exc}")

    def _create_samplers(self) -> None:
        if self.config.sampler == SamplerType.DDPM:
            self.stage1_sampler = None
            self.stage2_sampler = None
            self.stage1_schedule = DiffusionSchedule(DiffusionConfig(**self.stage1_schedule_cfg)).to(self.device)
            self.stage2_schedule = DiffusionSchedule(DiffusionConfig(**self.stage2_schedule_cfg)).to(self.device)
        else:
            self.stage1_sampler = create_sampler(
                self.config.sampler,
                num_train_timesteps=self.stage1_schedule_cfg["timesteps"],
                num_inference_steps=self.config.num_inference_steps,
                beta_start=self.stage1_schedule_cfg["beta_start"],
                beta_end=self.stage1_schedule_cfg["beta_end"],
                ddim_eta=self.config.ddim_eta,
            ).to(self.device)
            self.stage2_sampler = create_sampler(
                self.config.sampler,
                num_train_timesteps=self.stage2_schedule_cfg["timesteps"],
                num_inference_steps=self.config.num_inference_steps,
                beta_start=self.stage2_schedule_cfg["beta_start"],
                beta_end=self.stage2_schedule_cfg["beta_end"],
                ddim_eta=self.config.ddim_eta,
            ).to(self.device)
            self.stage1_schedule = None
            self.stage2_schedule = None

    def warmup(self, seq_len: int = 300) -> None:
        print("Warming up object-goal optimized sampler...")
        dummy_t = torch.randint(0, self.stage1_timesteps, (1,), device=self.device)
        dummy_bps = torch.randn(1, seq_len, 3072, device=self.device, dtype=self.dtype)
        dummy_centroid = torch.randn(1, seq_len, 3, device=self.device, dtype=self.dtype)
        dummy_object_pose = torch.randn(1, seq_len, OBJECT_POSE_DIM, device=self.device, dtype=self.dtype)
        dummy_goal = torch.randn(1, OBJECT_POSE_DIM, device=self.device, dtype=self.dtype)
        dummy_hands = torch.randn(1, seq_len, 6, device=self.device, dtype=self.dtype)
        dummy_cond = torch.randn(1, seq_len, self.cond_dim, device=self.device, dtype=self.dtype)
        dummy_state = torch.randn(1, seq_len, self.state_dim, device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            for _ in range(self.config.warmup_iterations):
                _ = self.stage1_model(
                    dummy_hands,
                    dummy_t,
                    dummy_bps,
                    dummy_centroid,
                    object_pose=dummy_object_pose,
                    global_cond=dummy_goal,
                )
                _ = self.stage2_model(dummy_state, dummy_t, dummy_cond, global_cond=dummy_goal)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete")

    @torch.inference_mode()
    def _sample_stage1_fast(self, bps, centroid, object_pose, goal) -> torch.Tensor:
        B, T, _ = centroid.shape

        def condition_fn():
            return (bps, centroid, object_pose, goal)

        return self.stage1_sampler.sample(
            model=self.stage1_fast_model,
            shape=(B, T, 6),
            condition_fn=condition_fn,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def _sample_stage2_fast(self, cond, goal) -> torch.Tensor:
        B, T, _ = cond.shape

        def condition_fn():
            return (cond, goal)

        return self.stage2_sampler.sample(
            model=self.stage2_fast_model,
            shape=(B, T, self.state_dim),
            condition_fn=condition_fn,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def _sample_stage1_ddpm(self, bps, centroid, object_pose, goal) -> torch.Tensor:
        B, T, _ = centroid.shape
        x = torch.randn(B, T, 6, device=self.device, dtype=self.dtype)
        for n in reversed(range(self.stage1_schedule.timesteps)):
            t = torch.full((B,), n, device=self.device, dtype=torch.long)
            x0_pred = self.stage1_model(x, t, bps, centroid, object_pose=object_pose, global_cond=goal)
            if n > 0:
                alpha_bar_t = self.stage1_schedule.alpha_bar[n]
                alpha_bar_t_prev = self.stage1_schedule.alpha_bar[n - 1]
                alpha_t = self.stage1_schedule.alpha[n]
                mean = (
                    torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                    + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
                )
                x = mean + torch.sqrt(self.stage1_schedule.beta[n]) * torch.randn_like(x)
            else:
                x = x0_pred
        return x

    @torch.inference_mode()
    def _sample_stage2_ddpm(self, cond, goal) -> torch.Tensor:
        B, T, _ = cond.shape
        x = torch.randn(B, T, self.state_dim, device=self.device, dtype=self.dtype)
        for n in reversed(range(self.stage2_schedule.timesteps)):
            t = torch.full((B,), n, device=self.device, dtype=torch.long)
            x0_pred = self.stage2_model(x, t, cond, global_cond=goal)
            if n > 0:
                alpha_bar_t = self.stage2_schedule.alpha_bar[n]
                alpha_bar_t_prev = self.stage2_schedule.alpha_bar[n - 1]
                alpha_t = self.stage2_schedule.alpha[n]
                mean = (
                    torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                    + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
                )
                x = mean + torch.sqrt(self.stage2_schedule.beta[n]) * torch.randn_like(x)
            else:
                x = x0_pred
        return x

    def generate(self, data: dict[str, Any], max_len: int) -> dict[str, Any]:
        bps_np, centroid_np, object_pose_np, static_bps_np = _prepare_object_inputs(
            data,
            max_len=max_len,
            variant=self.object_conditioning_variant,
        )
        goal_np = object_pose_np[-1]

        bps = torch.from_numpy(bps_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        centroid = torch.from_numpy(centroid_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        object_pose = torch.from_numpy(object_pose_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        goal_raw = torch.from_numpy(goal_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        goal_stage1 = _normalize(goal_raw, self.stage1_norm["goal_mean"], self.stage1_norm["goal_std"])

        if self.stage1_sampler is not None:
            hands_norm = self._sample_stage1_fast(bps, centroid, object_pose, goal_stage1)
        else:
            hands_norm = self._sample_stage1_ddpm(bps, centroid, object_pose, goal_stage1)
        hands_raw = _denormalize(hands_norm, self.stage1_norm["hand_mean"], self.stage1_norm["hand_std"])
        hands_np = hands_raw.squeeze(0).float().cpu().numpy()

        object_verts = data.get("object_verts")
        object_rotation = data.get("object_rotation")
        if object_verts is not None and object_rotation is not None:
            hands_rect_np, contact_meta = self.contact_processor.process(
                hands_np,
                np.asarray(object_verts, dtype=np.float32)[: hands_np.shape[0]],
                np.asarray(object_rotation, dtype=np.float32)[: hands_np.shape[0]],
                contact_labels=data.get("contact"),
            )
        elif self.require_contact_geometry:
            raise RuntimeError("Missing object_verts/object_rotation; contact rectification is required")
        else:
            hands_rect_np = hands_np
            contact_meta = {"rectification": "skipped_missing_object_geometry"}

        hands_rect = torch.from_numpy(hands_rect_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        if self.stage2_norm.get("normalize_hands", False):
            hands_cond = _normalize(hands_rect, self.stage2_norm["hand_mean"], self.stage2_norm["hand_std"])
        else:
            hands_cond = hands_rect
        stage2_goal = _normalize(goal_raw, self.stage2_norm["goal_mean"], self.stage2_norm["goal_std"])
        static_bps = torch.from_numpy(static_bps_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        cond = torch.cat([hands_cond, static_bps], dim=-1)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Stage 2 condition dim mismatch: built {cond.shape[-1]}, checkpoint expects {self.cond_dim}")

        if self.stage2_sampler is not None:
            state_norm = self._sample_stage2_fast(cond, stage2_goal)
        else:
            state_norm = self._sample_stage2_ddpm(cond, stage2_goal)

        state = _denormalize(state_norm, self.stage2_norm["state_mean"], self.stage2_norm["state_std"])
        state_np = state.squeeze(0).float().cpu().numpy()
        robot_state = state_np[:, :ROBOT_STATE_DIM]
        object_pose_sample = state_np[:, ROBOT_STATE_DIM:]
        root_pos = robot_state[:, :3]
        root_rot = rot6d_to_quat_xyzw(torch.from_numpy(robot_state[:, 3:9]).float()).numpy()
        dof_pos = robot_state[:, 9:]

        return {
            "pipeline_type": "object_goal_two_stage",
            "layout": robot_object_layout(),
            "hands_raw": hands_np,
            "hands_rectified": hands_rect_np,
            "contact_metadata": contact_meta,
            "state": state_np,
            "robot_state": robot_state,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "object_pose": object_pose_sample,
            "object_pose_context": object_pose_np,
            "goal": goal_np,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized object-goal Stage 2/end-to-end sampling")
    parser.add_argument(
        "--config_path",
        default=os.path.join(PROJECT_ROOT, "experiments", "object_goal", "sample_object_goal_stage2_hf_bps_optimized.yaml"),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    sample_cfg = yml["sample"]
    opt_cfg = yml.get("optimization", {})
    root_dir = _resolve(sample_cfg.get("root_dir", yml.get("root_dir", "./data/hf_bps_preprocessed")))
    stage1_ckpt_path = _require_checkpoint_path(sample_cfg, "stage1_ckpt_path", args.config_path)
    stage2_ckpt_path = _require_checkpoint_path(sample_cfg, "stage2_ckpt_path", args.config_path)
    device_str = sample_cfg.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() or not str(device_str).startswith("cuda") else "cpu")
    seed = int(sample_cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    inf_config = build_inference_config_from_yaml(opt_cfg)
    print("\nObject-goal optimized inference:")
    print(f"  Precision: {inf_config.precision.value}")
    print(f"  Sampler: {inf_config.sampler.value}")
    print(f"  Steps: {inf_config.num_inference_steps if inf_config.sampler != SamplerType.DDPM else 'N/A (DDPM)'}")
    print(f"  torch.compile: {inf_config.use_torch_compile}")

    pipeline = OptimizedObjectGoalStage2Pipeline(
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        config=inf_config,
        device=str(device),
        contact_threshold=float(sample_cfg.get("contact_threshold", 0.03)),
        stage1_contact_search_threshold=sample_cfg.get("stage1_contact_search_threshold"),
        stage1_max_contact_offset=sample_cfg.get("stage1_max_contact_offset", 0.02),
        stage1_max_contact_correction=sample_cfg.get("stage1_max_contact_correction", 0.06),
        stage1_fallback_contact_search_threshold=sample_cfg.get("stage1_fallback_contact_search_threshold"),
        stage1_fallback_max_contact_correction=sample_cfg.get("stage1_fallback_max_contact_correction"),
        require_contact_geometry=bool(sample_cfg.get("require_contact_geometry", True)),
    )

    input_path = sample_cfg.get("input_path")
    if input_path:
        input_paths = [_resolve(input_path)]
    else:
        input_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        num_samples = sample_cfg.get("num_samples")
        if num_samples:
            input_paths = input_paths[: int(num_samples)]
    if not input_paths:
        raise RuntimeError(f"No input PKLs found in {root_dir}")

    output_dir = _output_dir_from_stage2_checkpoint(stage2_ckpt_path, yml, inf_config)
    os.makedirs(output_dir, exist_ok=True)

    if bool(opt_cfg.get("warmup", True)):
        with open(input_paths[0], "rb") as f:
            warmup_data = pickle.load(f)
        warmup_len = min(
            int(sample_cfg.get("max_len", 300)),
            int(np.asarray(warmup_data["object_centroid"]).shape[0]),
        )
        pipeline.warmup(seq_len=warmup_len)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    try:
        import yaml

        with open(os.path.join(output_dir, "config.yml"), "w") as f:
            yaml.dump({"inference": {"precision": inf_config.precision.value, "sampler": inf_config.sampler.value, "steps": inf_config.num_inference_steps}, **yml}, f, default_flow_style=False)
    except Exception as exc:
        print(f"Warning: could not save config.yml: {exc}")

    total_time = 0.0
    frame_counts = []
    max_len = int(sample_cfg.get("max_len", 300))
    print(f"\nProcessing {len(input_paths)} file(s)")
    print(f"Output directory: {output_dir}")

    for path in tqdm(input_paths, desc="Processing"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        start = time.perf_counter()
        result = pipeline.generate(data, max_len=max_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_time += elapsed

        gen_T = result["state"].shape[0]
        frame_counts.append(gen_T)
        out = {
            **result,
            "source_path": path,
            "seq_name": data.get("seq_name", os.path.basename(path)),
            "fps": float(data.get("fps", 30.0)),
            "inference_time_ms": elapsed * 1000,
            "optimized_sampler": inf_config.sampler.value,
            "num_inference_steps": inf_config.num_inference_steps,
        }
        for key in ("object_name", "mesh_file", "num_verts", "is_articulated", "object_mesh_scale"):
            if key in data:
                out[key] = data[key]
        out_path = os.path.join(output_dir, os.path.basename(path))
        with open(out_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n" + "=" * 50)
    print("Object-goal optimized sampling summary")
    print(f"  Total files: {len(input_paths)}")
    print(f"  Total time: {total_time:.2f}s")
    if input_paths:
        print(f"  Average time: {total_time / len(input_paths) * 1000:.2f}ms per sample")
    if frame_counts:
        print(f"  Frames per motion: {int(np.mean(frame_counts))} avg, {min(frame_counts)} min, {max(frame_counts)} max")
    print(f"  Results saved to {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
