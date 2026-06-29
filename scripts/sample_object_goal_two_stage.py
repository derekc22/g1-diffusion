"""
Sample the corrected two-stage object-goal diffusion pipeline.

HF-BPS object geometry/object pose/final pose
  -> Stage 1 hand diffusion
  -> contact rectification
  -> Stage 2 robot-plus-object diffusion
  -> decoded sample pickle
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import types
from datetime import datetime
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

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
from utils.object_conditioning import apply_object_conditioning_variant, normalize_object_conditioning_variant
from utils.object_goal_features import OBJECT_POSE_DIM, ROBOT_STATE_DIM, object_pose_from_data, robot_object_layout
from utils.rotation import rot6d_to_quat_xyzw


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _normalize(x: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
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


def _denormalize(x: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
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


def _stage1_model_from_ckpt(ckpt: dict[str, Any], device: torch.device):
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
    model.to(device)
    model.eval()
    return model


def _stage2_model_from_ckpt(ckpt: dict[str, Any], device: torch.device):
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
        cond_dim_value = ckpt.get("cond_dim", model_cfg.get("cond_dim"))
        if cond_dim_value is None:
            raise ValueError("MLP Stage 2 checkpoint must store cond_dim")
        cond_dim = int(cond_dim_value)
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
            **common,
            hidden_dim=int(model_cfg.get("mlp_hidden", 512)),
            num_layers=int(model_cfg.get("mlp_layers", 4)),
        )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, state_dim, cond_dim


@torch.inference_mode()
def _sample_stage1(model, schedule: DiffusionSchedule, bps, centroid, object_pose, goal) -> torch.Tensor:
    B, T, _ = centroid.shape
    x = torch.randn(B, T, 6, device=centroid.device, dtype=centroid.dtype)
    for n in reversed(range(schedule.timesteps)):
        t = torch.full((B,), n, device=centroid.device, dtype=torch.long)
        x0_pred = model(x, t, bps, centroid, object_pose=object_pose, global_cond=goal)
        if n > 0:
            alpha_bar_t = schedule.alpha_bar[n]
            alpha_bar_t_prev = schedule.alpha_bar[n - 1]
            alpha_t = schedule.alpha[n]
            mean = (
                torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
            )
            x = mean + torch.sqrt(schedule.beta[n]) * torch.randn_like(x)
        else:
            x = x0_pred
    return x


@torch.inference_mode()
def _sample_stage2(model, schedule: DiffusionSchedule, cond, goal, state_dim: int) -> torch.Tensor:
    B, T, _ = cond.shape
    x = torch.randn(B, T, state_dim, device=cond.device, dtype=cond.dtype)
    for n in reversed(range(schedule.timesteps)):
        t = torch.full((B,), n, device=cond.device, dtype=torch.long)
        x0_pred = model(x, t, cond, global_cond=goal)
        if n > 0:
            alpha_bar_t = schedule.alpha_bar[n]
            alpha_bar_t_prev = schedule.alpha_bar[n - 1]
            alpha_t = schedule.alpha[n]
            mean = (
                torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
            )
            x = mean + torch.sqrt(schedule.beta[n]) * torch.randn_like(x)
        else:
            x = x0_pred
    return x


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample two-stage object-goal diffusion")
    parser.add_argument(
        "--config_path",
        default=os.path.join(PROJECT_ROOT, "experiments", "object_goal", "sample_object_goal_two_stage.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config_path)
    sample_cfg = cfg["sample"]
    device = torch.device(sample_cfg.get("device", "cuda:0"))
    dtype = torch.float32

    stage1_ckpt = load_torch_checkpoint(_resolve(sample_cfg["stage1_ckpt_path"]), map_location=device)
    stage2_ckpt = load_torch_checkpoint(_resolve(sample_cfg["stage2_ckpt_path"]), map_location=device)
    if stage1_ckpt.get("pipeline_type") != "object_goal_two_stage" or stage1_ckpt.get("stage") != 1:
        raise ValueError("stage1_ckpt_path must point to an object-goal two-stage Stage 1 checkpoint")
    if stage2_ckpt.get("pipeline_type") != "object_goal_two_stage" or stage2_ckpt.get("stage") != 2:
        raise ValueError("stage2_ckpt_path must point to an object-goal two-stage Stage 2 checkpoint")

    stage1_model = _stage1_model_from_ckpt(stage1_ckpt, device)
    stage2_model, state_dim, cond_dim = _stage2_model_from_ckpt(stage2_ckpt, device)

    s1_train = stage1_ckpt["config"].get("train", {})
    s2_train = stage2_ckpt["config"].get("train", {})
    s1_schedule = DiffusionSchedule(
        DiffusionConfig(
            timesteps=int(s1_train.get("timesteps", 1000)),
            beta_start=float(s1_train.get("beta_start", 1e-4)),
            beta_end=float(s1_train.get("beta_end", 0.02)),
        )
    ).to(device)
    s2_schedule = DiffusionSchedule(
        DiffusionConfig(
            timesteps=int(s2_train.get("timesteps", 1000)),
            beta_start=float(s2_train.get("beta_start", 1e-4)),
            beta_end=float(s2_train.get("beta_end", 0.02)),
        )
    ).to(device)

    root_dir = _resolve(sample_cfg.get("root_dir", "./data/hf_bps_preprocessed"))
    input_path = sample_cfg.get("input_path")
    if input_path:
        input_paths = [_resolve(input_path)]
    else:
        input_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))[: int(sample_cfg.get("num_samples", 1))]
    if not input_paths:
        raise RuntimeError(f"No input PKLs found in {root_dir}")

    output_dir = _resolve(sample_cfg.get("output_dir", f"out/object_goal_two_stage/{datetime.now():%Y%m%d_%H%M%S}"))
    os.makedirs(output_dir, exist_ok=True)
    variant = normalize_object_conditioning_variant(
        stage1_ckpt["config"].get("dataset", {}).get("object_conditioning_variant", "variant0")
    )
    contact_processor = ContactConstraintProcessor(
        contact_threshold=float(sample_cfg.get("contact_threshold", 0.03)),
        contact_search_threshold=sample_cfg.get("stage1_contact_search_threshold"),
        max_contact_offset=sample_cfg.get("stage1_max_contact_offset", 0.02),
        max_contact_correction=sample_cfg.get("stage1_max_contact_correction", 0.06),
        fallback_contact_search_threshold=sample_cfg.get("stage1_fallback_contact_search_threshold"),
        fallback_max_contact_correction=sample_cfg.get("stage1_fallback_max_contact_correction"),
    )

    s1_norm = stage1_ckpt["norm_stats"]
    s2_norm = stage2_ckpt["norm_stats"]
    max_len = int(sample_cfg.get("max_len", stage1_ckpt["config"].get("dataset", {}).get("window_size", 300)))
    require_contact_geometry = bool(sample_cfg.get("require_contact_geometry", True))

    for path in input_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        bps_np, centroid_np, object_pose_np, static_bps_np = _prepare_object_inputs(data, max_len, variant)
        goal_np = object_pose_np[-1]

        bps = torch.from_numpy(bps_np).to(device=device, dtype=dtype).unsqueeze(0)
        centroid = torch.from_numpy(centroid_np).to(device=device, dtype=dtype).unsqueeze(0)
        object_pose = torch.from_numpy(object_pose_np).to(device=device, dtype=dtype).unsqueeze(0)
        goal_raw = torch.from_numpy(goal_np).to(device=device, dtype=dtype).unsqueeze(0)
        goal_stage1 = _normalize(goal_raw, s1_norm.get("goal_mean"), s1_norm.get("goal_std"))

        hands_norm = _sample_stage1(stage1_model, s1_schedule, bps, centroid, object_pose, goal_stage1)
        hands_raw = _denormalize(hands_norm, s1_norm.get("hand_mean"), s1_norm.get("hand_std"))
        hands_np = hands_raw.squeeze(0).float().cpu().numpy()

        object_verts = data.get("object_verts")
        object_rotation = data.get("object_rotation")
        if object_verts is not None and object_rotation is not None:
            hands_rect_np, contact_meta = contact_processor.process(
                hands_np,
                np.asarray(object_verts, dtype=np.float32)[: hands_np.shape[0]],
                np.asarray(object_rotation, dtype=np.float32)[: hands_np.shape[0]],
                contact_labels=data.get("contact"),
            )
        elif require_contact_geometry:
            raise RuntimeError(
                f"Missing object_verts/object_rotation in {path}; contact rectification "
                "is required by default for object-goal sampling."
            )
        else:
            hands_rect_np = hands_np
            contact_meta = {"rectification": "skipped_missing_object_geometry"}

        hands_rect = torch.from_numpy(hands_rect_np).to(device=device, dtype=dtype).unsqueeze(0)
        if stage2_ckpt["config"].get("dataset", {}).get("normalize_hands", False):
            hands_cond = _normalize(hands_rect, s2_norm.get("hand_mean"), s2_norm.get("hand_std"))
        else:
            hands_cond = hands_rect
        stage2_goal = _normalize(goal_raw, s2_norm.get("goal_mean"), s2_norm.get("goal_std"))
        static_bps = torch.from_numpy(static_bps_np).to(device=device, dtype=dtype).unsqueeze(0)
        cond = torch.cat([hands_cond, static_bps], dim=-1)
        if cond.shape[-1] != cond_dim:
            raise ValueError(f"Stage 2 condition dim mismatch: built {cond.shape[-1]}, checkpoint expects {cond_dim}")

        state_norm = _sample_stage2(stage2_model, s2_schedule, cond, stage2_goal, state_dim)
        state = _denormalize(state_norm, s2_norm.get("state_mean"), s2_norm.get("state_std"))
        state_np = state.squeeze(0).float().cpu().numpy()
        robot_state = state_np[:, :ROBOT_STATE_DIM]
        object_pose_sample = state_np[:, ROBOT_STATE_DIM:]
        root_pos = robot_state[:, :3]
        root_rot = rot6d_to_quat_xyzw(torch.from_numpy(robot_state[:, 3:9]).float()).numpy()
        dof_pos = robot_state[:, 9:]

        out = {
            "pipeline_type": "object_goal_two_stage",
            "layout": robot_object_layout(),
            "source_path": path,
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
        out_path = os.path.join(output_dir, os.path.basename(path))
        with open(out_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
