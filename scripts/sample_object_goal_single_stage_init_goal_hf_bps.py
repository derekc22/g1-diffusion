"""Sample [robot_state, object_pose] from initial robot/object frames and an object goal."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from models.stage2_diffusion import Stage2MLPModel, Stage2TransformerModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.general import load_config, load_torch_checkpoint
from utils.inference_optimization import InferenceConfig, PrecisionMode, SamplerType, create_sampler
from utils.object_goal_features import (
    OBJECT_POSE_DIM,
    ROBOT_OBJECT_STATE_DIM,
    ROBOT_STATE_DIM,
    robot_object_layout,
    robot_object_motion_from_data,
)
from utils.rotation import rot6d_to_quat_xyzw


MODEL_TYPE = "object_goal_single_stage_init_goal"
GLOBAL_COND_DIM = 56
GLOBAL_COND_LAYOUT = {
    "final_object_frame": [0, 9],
    "initial_robot_frame": [9, 47],
    "initial_object_frame": [47, 56],
}


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def source_motion(data: dict[str, Any], max_len: int) -> np.ndarray:
    motion = robot_object_motion_from_data(data)
    seq_len = min(int(max_len), motion.shape[0])
    if seq_len <= 0:
        raise ValueError("Source item has no usable frames")
    return motion[:seq_len].astype(np.float32)


def condition_from_source(
    data: dict[str, Any], max_len: int, override: Optional[dict[str, Any]] = None
) -> tuple[np.ndarray, np.ndarray]:
    motion = source_motion(data, max_len)
    values = {
        "final_object_frame": motion[-1, ROBOT_STATE_DIM:],
        "initial_robot_frame": motion[0, :ROBOT_STATE_DIM],
        "initial_object_frame": motion[0, ROBOT_STATE_DIM:],
    }
    for key, dim in (("final_object_frame", 9), ("initial_robot_frame", 38), ("initial_object_frame", 9)):
        candidate = (override or {}).get(key)
        if candidate is not None:
            candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
            if candidate.shape != (dim,):
                raise ValueError(f"sample.global_cond_override.{key} must have {dim} values")
            values[key] = candidate
    g56 = np.concatenate(
        [values["final_object_frame"], values["initial_robot_frame"], values["initial_object_frame"]]
    ).astype(np.float32)
    return g56, motion


def inference_config(yml: dict[str, Any]) -> InferenceConfig:
    cfg = InferenceConfig()
    cfg.use_torch_compile = False
    if "precision" in yml:
        cfg.precision = PrecisionMode(str(yml["precision"]).lower())
    if "sampler" in yml:
        cfg.sampler = SamplerType(str(yml["sampler"]).lower())
    if "num_inference_steps" in yml:
        cfg.num_inference_steps = int(yml["num_inference_steps"])
    if "ddim_eta" in yml:
        cfg.ddim_eta = float(yml["ddim_eta"])
    return cfg


def _stat(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    return None if value is None else torch.as_tensor(value, device=device, dtype=torch.float32)


class _InitGoalAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, g56: torch.Tensor) -> torch.Tensor:
        return self.model(x, t, cond=None, global_cond=g56)


class SingleStageInitGoalPipeline:
    """One-checkpoint x0 sampling pipeline shared by sampling and evaluation."""

    def __init__(self, ckpt_path: str, inference: InferenceConfig, device: torch.device):
        self.device = device
        self.inference = inference
        self.dtype = (
            torch.float16
            if inference.precision == PrecisionMode.FP16 and device.type == "cuda"
            else torch.bfloat16
            if inference.precision == PrecisionMode.BF16
            else torch.float32
        )
        ckpt = load_torch_checkpoint(ckpt_path, map_location=device)
        expected = {
            "model_type": MODEL_TYPE,
            "prediction_type": "x0",
            "state_dim": 47,
            "cond_dim": 0,
            "global_cond_dim": 56,
        }
        for key, value in expected.items():
            if ckpt.get(key) != value:
                raise ValueError(f"Checkpoint {key} must be {value!r}, got {ckpt.get(key)!r}")
        if ckpt.get("global_cond_layout") != GLOBAL_COND_LAYOUT:
            raise ValueError(f"Checkpoint global_cond_layout must be {GLOBAL_COND_LAYOUT}")

        config = ckpt["config"]
        train_cfg, model_cfg = config.get("train", {}), config.get("model", {})
        window_size = int(config.get("dataset", {}).get("window_size", 300))
        common = dict(state_dim=47, cond_dim=0, global_cond_dim=56, contact_dim=0)
        architecture = str(train_cfg.get("architecture", "transformer"))
        if architecture == "transformer":
            model = Stage2TransformerModel(
                **common,
                global_cond_hidden=model_cfg.get("global_cond_hidden"),
                d_model=int(model_cfg.get("d_model", 512)),
                nhead=int(model_cfg.get("nhead", 8)),
                num_layers=int(model_cfg.get("num_layers", 8)),
                dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                max_len=int(model_cfg.get("max_len", window_size)),
            )
        elif architecture == "mlp":
            model = Stage2MLPModel(
                **common,
                hidden_dim=int(model_cfg.get("mlp_hidden", 512)),
                num_layers=int(model_cfg.get("mlp_layers", 4)),
            )
        else:
            raise ValueError(f"Unknown checkpoint architecture {architecture!r}")
        model.load_state_dict(ckpt["model"])
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.adapter = _InitGoalAdapter(self.model)

        schedule = ckpt.get("schedule", {})
        self.schedule_cfg = {
            "timesteps": int(schedule.get("timesteps", train_cfg.get("timesteps", 1000))),
            "beta_start": float(schedule.get("beta_start", train_cfg.get("beta_start", 1e-4))),
            "beta_end": float(schedule.get("beta_end", train_cfg.get("beta_end", 0.02))),
        }
        stats = ckpt.get("norm_stats", {})
        self.state_mean = _stat(stats.get("state_mean"), device)
        self.state_std = _stat(stats.get("state_std"), device)
        self.goal_mean = _stat(stats.get("goal_mean"), device)
        self.goal_std = _stat(stats.get("goal_std"), device)
        if any(x is None for x in (self.state_mean, self.state_std, self.goal_mean, self.goal_std)):
            raise ValueError("Checkpoint is missing state/goal normalization statistics")
        if inference.sampler == SamplerType.DDPM:
            self.fast_sampler = None
            self.schedule = DiffusionSchedule(DiffusionConfig(**self.schedule_cfg)).to(device)
        elif inference.sampler == SamplerType.DDIM:
            self.fast_sampler = create_sampler(
                inference.sampler,
                num_train_timesteps=self.schedule_cfg["timesteps"],
                num_inference_steps=inference.num_inference_steps,
                beta_start=self.schedule_cfg["beta_start"],
                beta_end=self.schedule_cfg["beta_end"],
                ddim_eta=inference.ddim_eta,
            ).to(device)
            self.schedule = None
        else:
            raise ValueError("This x0 pipeline currently supports sampler 'ddim' or 'ddpm'")

    def normalize_global_cond(self, raw: np.ndarray) -> torch.Tensor:
        g = torch.as_tensor(raw, device=self.device, dtype=self.dtype).reshape(1, GLOBAL_COND_DIM)
        state_mean = self.state_mean.to(dtype=self.dtype)
        state_std = self.state_std.to(dtype=self.dtype).clamp_min(1e-8)
        goal_mean = self.goal_mean.to(dtype=self.dtype)
        goal_std = self.goal_std.to(dtype=self.dtype).clamp_min(1e-8)
        parts = [
            (g[:, :9] - goal_mean) / goal_std,
            (g[:, 9:47] - state_mean[:38]) / state_std[:38],
            (g[:, 47:56] - state_mean[38:47]) / state_std[38:47],
        ]
        return torch.cat(parts, dim=-1)

    @torch.inference_mode()
    def _sample_ddpm(self, g56: torch.Tensor, seq_len: int) -> torch.Tensor:
        x = torch.randn(1, seq_len, 47, device=self.device, dtype=self.dtype)
        assert self.schedule is not None
        for n in reversed(range(self.schedule.timesteps)):
            t = torch.full((1,), n, device=self.device, dtype=torch.long)
            x0 = self.model(x, t, cond=None, global_cond=g56)
            if n == 0:
                x = x0
                continue
            abar = self.schedule.alpha_bar[n]
            abar_prev = self.schedule.alpha_bar[n - 1]
            alpha = self.schedule.alpha[n]
            mean = (
                torch.sqrt(abar_prev) * (1 - alpha) / (1 - abar) * x0
                + torch.sqrt(alpha) * (1 - abar_prev) / (1 - abar) * x
            )
            posterior_var = self.schedule.beta[n] * (1 - abar_prev) / (1 - abar)
            x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
        return x

    @torch.inference_mode()
    def generate(self, global_cond_raw: np.ndarray, seq_len: int) -> dict[str, Any]:
        raw = np.asarray(global_cond_raw, dtype=np.float32).reshape(GLOBAL_COND_DIM)
        g56 = self.normalize_global_cond(raw)
        if self.fast_sampler is None:
            state_norm = self._sample_ddpm(g56, int(seq_len))
        else:
            state_norm = self.fast_sampler.sample(
                model=self.adapter,
                shape=(1, int(seq_len), 47),
                condition_fn=lambda: (g56,),
                device=self.device,
                dtype=self.dtype,
            )
        mean = self.state_mean.to(dtype=self.dtype).view(1, 1, -1)
        std = self.state_std.to(dtype=self.dtype).view(1, 1, -1)
        state = (state_norm * std + mean).squeeze(0).float().cpu().numpy()
        robot_state, object_pose = state[:, :38], state[:, 38:47]
        return {
            "model_type": MODEL_TYPE,
            "prediction_type": "x0",
            "conditioning": "init_goal",
            "global_cond_layout": GLOBAL_COND_LAYOUT,
            "layout": robot_object_layout(),
            "state": state,
            "robot_state": robot_state,
            "object_pose": object_pose,
            "root_pos": robot_state[:, :3],
            "root_rot": rot6d_to_quat_xyzw(torch.from_numpy(robot_state[:, 3:9])).numpy(),
            "dof_pos": robot_state[:, 9:],
            "goal": raw[:9].copy(),
            "global_cond": raw.copy(),
            "final_object_frame": raw[:9].copy(),
            "initial_robot_frame": raw[9:47].copy(),
            "initial_object_frame": raw[47:56].copy(),
        }


def _output_dir(ckpt_path: str, yml: dict[str, Any], sampler: str) -> str:
    configured = yml.get("sample", {}).get("output_dir")
    if configured:
        return resolve_path(str(configured))
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    name = str(yml.get("exp_name", "object_goal_single_stage_init_goal"))
    return os.path.join(PROJECT_ROOT, "out", f"{name}_{sampler}_{timestamp}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample single-stage init+goal robot-object diffusion")
    parser.add_argument(
        "--config_path",
        default=os.path.join(PROJECT_ROOT, "experiments", "object_goal", "sample_object_goal_single_stage_init_goal_hf_bps.yaml"),
    )
    args = parser.parse_args()
    yml = load_config(args.config_path)
    cfg = yml["sample"]
    ckpt_value = str(cfg.get("ckpt_path") or "").strip()
    if not ckpt_value:
        raise ValueError(f"Missing sample.ckpt_path in {args.config_path}")
    ckpt_path = resolve_path(ckpt_value)
    root_dir = resolve_path(str(cfg.get("root_dir", yml.get("root_dir", "./data/hf_bps_preprocessed"))))
    requested_device = str(cfg.get("device", "cuda:0"))
    device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mode = str(cfg.get("mode", "A")).upper()
    if mode != "A":
        raise ValueError("Standalone sampling supports paired/A-like sources; use the eval script for B/C")
    input_path = cfg.get("input_path")
    paths = [resolve_path(str(input_path))] if input_path else sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if cfg.get("num_samples") is not None:
        paths = paths[: int(cfg["num_samples"])]
    if not paths:
        raise RuntimeError(f"No source PKLs found in {root_dir}")

    inference = inference_config(yml.get("optimization", {}))
    pipeline = SingleStageInitGoalPipeline(ckpt_path, inference, device)
    output_dir = _output_dir(ckpt_path, yml, inference.sampler.value)
    os.makedirs(output_dir, exist_ok=True)
    for index, path in enumerate(tqdm(paths, desc="Sampling")):
        with open(path, "rb") as f:
            data = pickle.load(f)
        g56, motion = condition_from_source(data, int(cfg.get("max_len", 300)), cfg.get("global_cond_override"))
        started = time.perf_counter()
        result = pipeline.generate(g56, motion.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize()
        result.update(
            source_path=path,
            eval_mode=cfg.get("eval_mode") or "A-like-paired",
            seq_name=data.get("seq_name", os.path.splitext(os.path.basename(path))[0]),
            fps=float(data.get("fps", 30.0)),
            sampler=inference.sampler.value,
            num_inference_steps=inference.num_inference_steps,
            inference_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        for key in ("object_name", "mesh_file", "num_verts", "is_articulated", "object_mesh_scale"):
            if key in data:
                result[key] = data[key]
        name = os.path.basename(path) if len(paths) == 1 else f"{index:05d}_{os.path.basename(path)}"
        with open(os.path.join(output_dir, name), "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(paths)} init+goal sample(s) to {output_dir}")


if __name__ == "__main__":
    main()
