"""Sample [robot_state, object_pose] directly from a final object goal.

This is an isolated single-stage path. It loads one checkpoint and one or more
HF-BPS source items, extracts only the final 9D object goal and sequence length,
then samples a 47D trajectory without Stage 1, hands, BPS, geometry, contact
rectification, or a clean per-frame object trajectory condition.
"""

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
    object_pose_from_data,
    robot_object_layout,
)
from utils.rotation import rot6d_to_quat_xyzw


MODEL_TYPE = "object_goal_single_stage_goal_only"


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _required_checkpoint(sample_cfg: dict[str, Any], config_path: str) -> str:
    path = str(sample_cfg.get("ckpt_path") or "").strip()
    if not path:
        raise ValueError(f"Missing sample.ckpt_path in {config_path}")
    return _resolve(path)


def _stat(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _normalize(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mean is None or std is None:
        return x
    return (x - mean.to(x)) / std.to(x).clamp_min(1e-8)


def _denormalize(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mean is None or std is None:
        return x
    return x * std.to(x).view(1, 1, -1) + mean.to(x).view(1, 1, -1)


def _source_goal_and_length(data: dict[str, Any], max_len: int) -> tuple[np.ndarray, int]:
    object_pose = object_pose_from_data(data)
    lengths = [object_pose.shape[0]]
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key in data and data[key] is not None:
            lengths.append(int(np.asarray(data[key]).shape[0]))
    seq_len = min(max_len, *lengths)
    if seq_len <= 0:
        raise ValueError("Source item has no usable frames")
    return object_pose[seq_len - 1].astype(np.float32), seq_len


def _inference_config(yml: dict[str, Any]) -> InferenceConfig:
    cfg = InferenceConfig()
    if "precision" in yml:
        cfg.precision = PrecisionMode(yml["precision"])
    if "sampler" in yml:
        cfg.sampler = SamplerType(yml["sampler"])
    if "num_inference_steps" in yml:
        cfg.num_inference_steps = int(yml["num_inference_steps"])
    if "ddim_eta" in yml:
        cfg.ddim_eta = float(yml["ddim_eta"])
    return cfg


class _GoalOnlyAdapter(torch.nn.Module):
    """Adapts optimized samplers to model(x_t, t, cond=None, global_cond=g)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.model(x, t, cond=None, global_cond=goal)


class SingleStageGoalOnlyPipeline:
    def __init__(self, ckpt_path: str, inference: InferenceConfig, device: torch.device):
        self.device = device
        self.inference = inference
        if inference.precision == PrecisionMode.FP16 and device.type == "cuda":
            self.dtype = torch.float16
        elif inference.precision == PrecisionMode.BF16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        ckpt = load_torch_checkpoint(ckpt_path, map_location=device)
        if ckpt.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Checkpoint model_type must be {MODEL_TYPE!r}")
        if ckpt.get("prediction_type") != "x0":
            raise ValueError("Checkpoint must predict x0")
        if int(ckpt.get("state_dim", -1)) != ROBOT_OBJECT_STATE_DIM:
            raise ValueError(f"Checkpoint state_dim must be {ROBOT_OBJECT_STATE_DIM}")
        if int(ckpt.get("cond_dim", -1)) != 0:
            raise ValueError("Checkpoint cond_dim must be 0")
        if int(ckpt.get("global_cond_dim", -1)) != OBJECT_POSE_DIM:
            raise ValueError(f"Checkpoint global_cond_dim must be {OBJECT_POSE_DIM}")

        config = ckpt["config"]
        train_cfg = config.get("train", {})
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        architecture = train_cfg.get("architecture", "transformer")
        window_size = int(dataset_cfg.get("window_size", 300))
        common = {
            "state_dim": ROBOT_OBJECT_STATE_DIM,
            "cond_dim": 0,
            "global_cond_dim": OBJECT_POSE_DIM,
            "contact_dim": 0,
        }
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
        self.adapter = _GoalOnlyAdapter(self.model)

        schedule = ckpt.get("schedule", {})
        self.schedule_cfg = {
            "timesteps": int(schedule.get("timesteps", train_cfg.get("timesteps", 1000))),
            "beta_start": float(schedule.get("beta_start", train_cfg.get("beta_start", 1e-4))),
            "beta_end": float(schedule.get("beta_end", train_cfg.get("beta_end", 0.02))),
        }
        norm_stats = ckpt.get("norm_stats", {})
        self.state_mean = _stat(norm_stats.get("state_mean"), device)
        self.state_std = _stat(norm_stats.get("state_std"), device)
        self.goal_mean = _stat(norm_stats.get("goal_mean"), device)
        self.goal_std = _stat(norm_stats.get("goal_std"), device)
        if any(v is None for v in (self.state_mean, self.state_std, self.goal_mean, self.goal_std)):
            raise ValueError("Checkpoint is missing state/goal normalization statistics")

        if inference.sampler == SamplerType.DDPM:
            self.fast_sampler = None
            self.schedule = DiffusionSchedule(DiffusionConfig(**self.schedule_cfg)).to(device)
        else:
            self.fast_sampler = create_sampler(
                inference.sampler,
                num_train_timesteps=self.schedule_cfg["timesteps"],
                num_inference_steps=inference.num_inference_steps,
                beta_start=self.schedule_cfg["beta_start"],
                beta_end=self.schedule_cfg["beta_end"],
                ddim_eta=inference.ddim_eta,
            ).to(device)
            self.schedule = None

    @torch.inference_mode()
    def _sample_ddpm(self, goal: torch.Tensor, seq_len: int) -> torch.Tensor:
        x = torch.randn(1, seq_len, ROBOT_OBJECT_STATE_DIM, device=self.device, dtype=self.dtype)
        assert self.schedule is not None
        for n in reversed(range(self.schedule.timesteps)):
            t = torch.full((1,), n, device=self.device, dtype=torch.long)
            x0_pred = self.model(x, t, cond=None, global_cond=goal)
            if n == 0:
                x = x0_pred
                continue
            alpha_bar_t = self.schedule.alpha_bar[n]
            alpha_bar_prev = self.schedule.alpha_bar[n - 1]
            alpha_t = self.schedule.alpha[n]
            mean = (
                torch.sqrt(alpha_bar_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                + torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
            )
            x = mean + torch.sqrt(self.schedule.beta[n]) * torch.randn_like(x)
        return x

    @torch.inference_mode()
    def generate(self, data: dict[str, Any], max_len: int) -> dict[str, Any]:
        goal_np, seq_len = _source_goal_and_length(data, max_len)
        goal_raw = torch.from_numpy(goal_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        goal = _normalize(goal_raw, self.goal_mean, self.goal_std)

        if self.fast_sampler is None:
            state_norm = self._sample_ddpm(goal, seq_len)
        else:
            state_norm = self.fast_sampler.sample(
                model=self.adapter,
                shape=(1, seq_len, ROBOT_OBJECT_STATE_DIM),
                condition_fn=lambda: (goal,),
                device=self.device,
                dtype=self.dtype,
            )

        state = _denormalize(state_norm, self.state_mean, self.state_std)
        state_np = state.squeeze(0).float().cpu().numpy()
        robot_state = state_np[:, :ROBOT_STATE_DIM]
        object_pose = state_np[:, ROBOT_STATE_DIM:]
        return {
            "pipeline_type": MODEL_TYPE,
            "prediction_type": "x0",
            "layout": robot_object_layout(),
            "state": state_np,
            "robot_state": robot_state,
            "object_pose": object_pose,
            "root_pos": robot_state[:, :3],
            "root_rot": rot6d_to_quat_xyzw(torch.from_numpy(robot_state[:, 3:9]).float()).numpy(),
            "dof_pos": robot_state[:, 9:],
            "goal": goal_np,
        }


def _output_dir(ckpt_path: str, yml: dict[str, Any], inference: InferenceConfig) -> str:
    configured = yml.get("sample", {}).get("output_dir")
    if configured:
        return _resolve(configured)
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = str(yml.get("exp_name", "goal_only"))
    parts = ckpt_path.split(os.sep)
    if "logs" in parts and "checkpoints" in parts:
        logs_idx = parts.index("logs")
        log_id = parts[logs_idx + 1]
        return os.path.join(
            PROJECT_ROOT,
            "logs",
            log_id,
            "samples",
            f"{inference.sampler.value}_{timestamp}_{exp_name}",
        )
    return os.path.join(PROJECT_ROOT, "out", f"object_goal_single_stage_{timestamp}_{exp_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample single-stage goal-only robot-object diffusion")
    parser.add_argument(
        "--config_path",
        default=os.path.join(
            PROJECT_ROOT,
            "experiments",
            "object_goal",
            "sample_object_goal_single_stage_hf_bps.yaml",
        ),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    sample_cfg = yml["sample"]
    inference = _inference_config(yml.get("optimization", {}))
    ckpt_path = _required_checkpoint(sample_cfg, args.config_path)
    root_dir = _resolve(sample_cfg.get("root_dir", yml.get("root_dir", "./data/hf_bps_preprocessed")))
    requested_device = str(sample_cfg.get("device", "cuda:0"))
    device = torch.device(
        requested_device
        if torch.cuda.is_available() or not requested_device.startswith("cuda")
        else "cpu"
    )
    seed = int(sample_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_path = sample_cfg.get("input_path")
    if input_path:
        input_paths = [_resolve(input_path)]
    else:
        input_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        num_samples = sample_cfg.get("num_samples")
        if num_samples is not None:
            input_paths = input_paths[: int(num_samples)]
    if not input_paths:
        raise RuntimeError(f"No source PKLs found in {root_dir}")

    pipeline = SingleStageGoalOnlyPipeline(ckpt_path, inference, device)
    output_dir = _output_dir(ckpt_path, yml, inference)
    os.makedirs(output_dir, exist_ok=True)
    max_len = int(sample_cfg.get("max_len", 300))

    print("Single-stage goal-only robot-object sampling")
    print(f"Checkpoint: {ckpt_path}")
    print("Condition: final 9D object goal only")
    print(f"Processing {len(input_paths)} source item(s)")
    print(f"Output: {output_dir}")

    for path in tqdm(input_paths, desc="Sampling"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        start = time.perf_counter()
        result = pipeline.generate(data, max_len=max_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        out = {
            **result,
            "source_path": path,
            "seq_name": data.get("seq_name", os.path.basename(path)),
            "fps": float(data.get("fps", 30.0)),
            "inference_time_ms": elapsed * 1000.0,
            "sampler": inference.sampler.value,
            "num_inference_steps": inference.num_inference_steps,
        }
        for key in ("object_name", "mesh_file", "num_verts", "is_articulated", "object_mesh_scale"):
            if key in data:
                out[key] = data[key]
        out_path = os.path.join(output_dir, os.path.basename(path))
        with open(out_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(input_paths)} sample(s) to {output_dir}")


if __name__ == "__main__":
    main()
