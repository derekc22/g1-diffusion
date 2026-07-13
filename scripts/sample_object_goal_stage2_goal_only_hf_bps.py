"""Sample the goal-only two-stage object-goal diffusion baseline.

The source PKL supplies only a sequence length, its final 9D object goal, and
optional output metadata. Stage 1 generates hands from the goal; Stage 2
generates [robot_state, object_pose] from those hands and the same goal.
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


STAGE1_MODEL_TYPE = "object_goal_stage1_goal_only"
STAGE2_MODEL_TYPE = "object_goal_stage2_goal_only"
PIPELINE_TYPE = "object_goal_two_stage_goal_only"
PREDICTION_TYPE = "x0"
HAND_DIM = 6


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _required_checkpoint(sample_cfg: dict[str, Any], key: str, config_path: str) -> str:
    path = str(sample_cfg.get(key) or "").strip()
    if not path:
        raise ValueError(f"Missing sample.{key} in {config_path}")
    return _resolve(path)


def _stat(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _required_stat(
    norm_stats: dict[str, Any], key: str, dim: int, device: torch.device, checkpoint: str
) -> torch.Tensor:
    value = _stat(norm_stats.get(key), device)
    if value is None:
        raise ValueError(f"{checkpoint} checkpoint is missing norm_stats.{key}")
    if value.numel() != dim:
        raise ValueError(
            f"{checkpoint} checkpoint norm_stats.{key} must have {dim} values, "
            f"got {value.numel()}"
        )
    return value.reshape(dim)


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.to(x).view(1, -1)) / std.to(x).view(1, -1).clamp_min(1e-8)


def _denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.to(x).view(1, 1, -1) + mean.to(x).view(1, 1, -1)


def _source_goal_and_length(data: dict[str, Any], max_len: int) -> tuple[np.ndarray, int]:
    object_pose = object_pose_from_data(data)
    lengths = [int(object_pose.shape[0])]
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key in data and data[key] is not None:
            lengths.append(int(np.asarray(data[key]).shape[0]))
    source_len = min(lengths)
    seq_len = min(source_len, max_len) if max_len > 0 else source_len
    if seq_len <= 0:
        raise ValueError("Source item has no usable frames")
    # The goal is the source motion's final pose, not a clean trajectory prefix.
    return object_pose[-1].astype(np.float32), seq_len


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


def _model_from_checkpoint(
    ckpt: dict[str, Any], state_dim: int, cond_dim: int, window_size: int
) -> torch.nn.Module:
    config = ckpt["config"]
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    architecture = str(train_cfg.get("architecture", "transformer"))
    common = {
        "state_dim": state_dim,
        "cond_dim": cond_dim,
        "global_cond_dim": OBJECT_POSE_DIM,
        "contact_dim": 0,
    }
    if architecture == "transformer":
        return Stage2TransformerModel(
            **common,
            global_cond_hidden=model_cfg.get("global_cond_hidden"),
            d_model=int(model_cfg.get("d_model", 512)),
            nhead=int(model_cfg.get("nhead", 8)),
            num_layers=int(model_cfg.get("num_layers", 8)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_len=int(model_cfg.get("max_len", window_size)),
        )
    if architecture == "mlp":
        return Stage2MLPModel(
            **common,
            hidden_dim=int(model_cfg.get("mlp_hidden", 512)),
            num_layers=int(model_cfg.get("mlp_layers", 4)),
        )
    raise ValueError(f"Unknown checkpoint architecture {architecture!r}")


def _schedule_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    train_cfg = ckpt["config"].get("train", {})
    schedule = ckpt.get("schedule", {})
    return {
        "timesteps": int(schedule.get("timesteps", train_cfg.get("timesteps", 1000))),
        "beta_start": float(schedule.get("beta_start", train_cfg.get("beta_start", 1e-4))),
        "beta_end": float(schedule.get("beta_end", train_cfg.get("beta_end", 0.02))),
    }


def _validate_common(
    ckpt: dict[str, Any], checkpoint: str, model_type: str, cond_dim: int
) -> None:
    if ckpt.get("model_type") != model_type:
        raise ValueError(f"{checkpoint} checkpoint model_type must be {model_type!r}")
    if ckpt.get("prediction_type") != PREDICTION_TYPE:
        raise ValueError(f"{checkpoint} checkpoint prediction_type must be 'x0'")
    if int(ckpt.get("cond_dim", -1)) != cond_dim:
        raise ValueError(f"{checkpoint} checkpoint cond_dim must be {cond_dim}")
    if int(ckpt.get("global_cond_dim", -1)) != OBJECT_POSE_DIM:
        raise ValueError(
            f"{checkpoint} checkpoint global_cond_dim must be {OBJECT_POSE_DIM}"
        )
    if "config" not in ckpt or "model" not in ckpt:
        raise ValueError(f"{checkpoint} checkpoint is missing config or model weights")


class _Stage1Adapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.model(x, t, cond=None, global_cond=goal)


class _Stage2Adapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, hands: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        return self.model(x, t, cond=hands, global_cond=goal)


class GoalOnlyTwoStagePipeline:
    def __init__(
        self,
        stage1_ckpt_path: str,
        stage2_ckpt_path: str,
        inference: InferenceConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.inference = inference
        if inference.precision == PrecisionMode.FP16 and device.type == "cuda":
            self.dtype = torch.float16
        elif inference.precision == PrecisionMode.BF16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        stage1_ckpt = load_torch_checkpoint(stage1_ckpt_path, map_location=device)
        _validate_common(stage1_ckpt, "Stage 1", STAGE1_MODEL_TYPE, 0)
        stage1_state_dim = stage1_ckpt.get("state_dim", stage1_ckpt.get("hand_dim", -1))
        if int(stage1_state_dim) != HAND_DIM:
            raise ValueError(f"Stage 1 checkpoint state_dim or hand_dim must be {HAND_DIM}")
        if "hand_dim" in stage1_ckpt and int(stage1_ckpt["hand_dim"]) != HAND_DIM:
            raise ValueError(f"Stage 1 checkpoint hand_dim must be {HAND_DIM}")

        stage2_ckpt = load_torch_checkpoint(stage2_ckpt_path, map_location=device)
        _validate_common(stage2_ckpt, "Stage 2", STAGE2_MODEL_TYPE, HAND_DIM)
        if int(stage2_ckpt.get("state_dim", -1)) != ROBOT_OBJECT_STATE_DIM:
            raise ValueError(
                f"Stage 2 checkpoint state_dim must be {ROBOT_OBJECT_STATE_DIM}"
            )

        stage1_window = int(stage1_ckpt["config"].get("dataset", {}).get("window_size", 300))
        stage2_window = int(stage2_ckpt["config"].get("dataset", {}).get("window_size", 300))
        stage1_model = _model_from_checkpoint(stage1_ckpt, HAND_DIM, 0, stage1_window)
        stage2_model = _model_from_checkpoint(
            stage2_ckpt, ROBOT_OBJECT_STATE_DIM, HAND_DIM, stage2_window
        )
        stage1_model.load_state_dict(stage1_ckpt["model"], strict=True)
        stage2_model.load_state_dict(stage2_ckpt["model"], strict=True)
        self.stage1_model = stage1_model.to(device=device, dtype=self.dtype).eval()
        self.stage2_model = stage2_model.to(device=device, dtype=self.dtype).eval()
        self.stage1_adapter = _Stage1Adapter(self.stage1_model)
        self.stage2_adapter = _Stage2Adapter(self.stage2_model)

        stage1_norm = stage1_ckpt.get("norm_stats", {})
        self.hand_mean = _required_stat(stage1_norm, "hand_mean", HAND_DIM, device, "Stage 1")
        self.hand_std = _required_stat(stage1_norm, "hand_std", HAND_DIM, device, "Stage 1")
        self.stage1_goal_mean = _required_stat(
            stage1_norm, "goal_mean", OBJECT_POSE_DIM, device, "Stage 1"
        )
        self.stage1_goal_std = _required_stat(
            stage1_norm, "goal_std", OBJECT_POSE_DIM, device, "Stage 1"
        )

        stage2_norm = stage2_ckpt.get("norm_stats", {})
        if bool(stage2_norm.get("normalize_hands", False)):
            raise ValueError("Stage 2 checkpoint must use raw, unnormalized hand conditioning")
        self.state_mean = _required_stat(
            stage2_norm, "state_mean", ROBOT_OBJECT_STATE_DIM, device, "Stage 2"
        )
        self.state_std = _required_stat(
            stage2_norm, "state_std", ROBOT_OBJECT_STATE_DIM, device, "Stage 2"
        )
        self.stage2_goal_mean = _required_stat(
            stage2_norm, "goal_mean", OBJECT_POSE_DIM, device, "Stage 2"
        )
        self.stage2_goal_std = _required_stat(
            stage2_norm, "goal_std", OBJECT_POSE_DIM, device, "Stage 2"
        )

        self.stage1_schedule_cfg = _schedule_config(stage1_ckpt)
        self.stage2_schedule_cfg = _schedule_config(stage2_ckpt)
        if inference.sampler == SamplerType.DDPM:
            self.stage1_sampler = None
            self.stage2_sampler = None
            self.stage1_schedule = DiffusionSchedule(
                DiffusionConfig(**self.stage1_schedule_cfg)
            ).to(device)
            self.stage2_schedule = DiffusionSchedule(
                DiffusionConfig(**self.stage2_schedule_cfg)
            ).to(device)
        else:
            self.stage1_sampler = create_sampler(
                inference.sampler,
                num_train_timesteps=self.stage1_schedule_cfg["timesteps"],
                num_inference_steps=inference.num_inference_steps,
                beta_start=self.stage1_schedule_cfg["beta_start"],
                beta_end=self.stage1_schedule_cfg["beta_end"],
                ddim_eta=inference.ddim_eta,
            ).to(device)
            self.stage2_sampler = create_sampler(
                inference.sampler,
                num_train_timesteps=self.stage2_schedule_cfg["timesteps"],
                num_inference_steps=inference.num_inference_steps,
                beta_start=self.stage2_schedule_cfg["beta_start"],
                beta_end=self.stage2_schedule_cfg["beta_end"],
                ddim_eta=inference.ddim_eta,
            ).to(device)
            self.stage1_schedule = None
            self.stage2_schedule = None

    @torch.inference_mode()
    def _sample_ddpm(
        self,
        model: torch.nn.Module,
        schedule: DiffusionSchedule,
        shape: tuple[int, int, int],
        cond: Optional[torch.Tensor],
        goal: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.randn(*shape, device=self.device, dtype=self.dtype)
        for n in reversed(range(schedule.timesteps)):
            t = torch.full((shape[0],), n, device=self.device, dtype=torch.long)
            x0_pred = model(x, t, cond=cond, global_cond=goal)
            if n == 0:
                x = x0_pred
                continue
            alpha_bar_t = schedule.alpha_bar[n]
            alpha_bar_prev = schedule.alpha_bar[n - 1]
            alpha_t = schedule.alpha[n]
            mean = (
                torch.sqrt(alpha_bar_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                + torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
            )
            x = mean + torch.sqrt(schedule.beta[n]) * torch.randn_like(x)
        return x

    @torch.inference_mode()
    def generate(self, data: dict[str, Any], max_len: int) -> dict[str, Any]:
        goal_np, seq_len = _source_goal_and_length(data, max_len)
        goal_raw = torch.from_numpy(goal_np).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        stage1_goal = _normalize(goal_raw, self.stage1_goal_mean, self.stage1_goal_std)

        if self.stage1_sampler is None:
            assert self.stage1_schedule is not None
            hands_norm = self._sample_ddpm(
                self.stage1_model,
                self.stage1_schedule,
                (1, seq_len, HAND_DIM),
                cond=None,
                goal=stage1_goal,
            )
        else:
            hands_norm = self.stage1_sampler.sample(
                model=self.stage1_adapter,
                shape=(1, seq_len, HAND_DIM),
                condition_fn=lambda: (stage1_goal,),
                device=self.device,
                dtype=self.dtype,
            )
        hands = _denormalize(hands_norm, self.hand_mean, self.hand_std)

        stage2_goal = _normalize(goal_raw, self.stage2_goal_mean, self.stage2_goal_std)
        if self.stage2_sampler is None:
            assert self.stage2_schedule is not None
            state_norm = self._sample_ddpm(
                self.stage2_model,
                self.stage2_schedule,
                (1, seq_len, ROBOT_OBJECT_STATE_DIM),
                cond=hands,
                goal=stage2_goal,
            )
        else:
            state_norm = self.stage2_sampler.sample(
                model=self.stage2_adapter,
                shape=(1, seq_len, ROBOT_OBJECT_STATE_DIM),
                condition_fn=lambda: (hands, stage2_goal),
                device=self.device,
                dtype=self.dtype,
            )

        state = _denormalize(state_norm, self.state_mean, self.state_std)
        state_np = state.squeeze(0).float().cpu().numpy()
        hands_np = hands.squeeze(0).float().cpu().numpy()
        robot_state = state_np[:, :ROBOT_STATE_DIM]
        object_pose = state_np[:, ROBOT_STATE_DIM:]
        return {
            "pipeline_type": PIPELINE_TYPE,
            "prediction_type": PREDICTION_TYPE,
            "layout": robot_object_layout(),
            "state": state_np,
            "robot_state": robot_state,
            "object_pose": object_pose,
            "root_pos": robot_state[:, :3],
            "root_rot": rot6d_to_quat_xyzw(
                torch.from_numpy(robot_state[:, 3:9]).float()
            ).numpy(),
            "dof_pos": robot_state[:, 9:],
            "goal": goal_np,
            "hands": hands_np,
            "hands_pred": hands_np,
        }


def _output_dir(stage2_ckpt_path: str, yml: dict[str, Any], inference: InferenceConfig) -> str:
    configured = yml.get("sample", {}).get("output_dir")
    if configured:
        return _resolve(configured)
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = str(yml.get("exp_name", "goal_only"))
    parts = stage2_ckpt_path.split(os.sep)
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
    return os.path.join(PROJECT_ROOT, "out", f"{PIPELINE_TYPE}_{timestamp}_{exp_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample goal-only object-goal two-stage diffusion")
    parser.add_argument(
        "--config_path",
        default=os.path.join(
            PROJECT_ROOT,
            "experiments",
            "object_goal",
            "sample_object_goal_stage2_goal_only_hf_bps.yaml",
        ),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    sample_cfg = yml["sample"]
    inference = _inference_config(yml.get("optimization", {}))
    stage1_ckpt_path = _required_checkpoint(
        sample_cfg, "stage1_ckpt_path", args.config_path
    )
    stage2_ckpt_path = _required_checkpoint(
        sample_cfg, "stage2_ckpt_path", args.config_path
    )
    root_dir = _resolve(
        sample_cfg.get("root_dir", yml.get("root_dir", "./data/hf_bps_preprocessed"))
    )
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

    pipeline = GoalOnlyTwoStagePipeline(
        stage1_ckpt_path, stage2_ckpt_path, inference, device
    )
    output_dir = _output_dir(stage2_ckpt_path, yml, inference)
    os.makedirs(output_dir, exist_ok=True)
    max_len = int(sample_cfg.get("max_len", 300))

    print("Goal-only object-goal two-stage sampling")
    print(f"Stage 1 checkpoint: {stage1_ckpt_path}")
    print(f"Stage 2 checkpoint: {stage2_ckpt_path}")
    print("Stage 1 condition: final 9D object goal only")
    print("Stage 2 condition: generated 6D hands + final 9D object goal")
    print("Contact rectification: disabled")
    print(f"Processing {len(input_paths)} source item(s)")
    print(f"Output: {output_dir}")

    for path in tqdm(input_paths, desc="Sampling"):
        with open(path, "rb") as file:
            data = pickle.load(file)
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
            "contact_rectification": False,
        }
        for key in ("object_name", "mesh_file", "num_verts", "is_articulated", "object_mesh_scale"):
            if key in data:
                out[key] = data[key]
        out_path = os.path.join(output_dir, os.path.basename(path))
        with open(out_path, "wb") as file:
            pickle.dump(out, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(input_paths)} sample(s) to {output_dir}")


if __name__ == "__main__":
    main()
