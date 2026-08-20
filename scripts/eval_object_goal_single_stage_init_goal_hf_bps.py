"""Evaluate the single-stage init+goal model in paired, recombined, and arbitrary modes.

Parent distance is trajectory RMSE after linear time resampling and checkpoint
state normalization. This makes heterogeneous state dimensions comparable but
is intentionally a simple state-space similarity score, not a perceptual metric.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import shlex
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from scripts.sample_object_goal_single_stage_init_goal_hf_bps import (
    GLOBAL_COND_LAYOUT,
    MODEL_TYPE,
    SingleStageInitGoalPipeline,
    inference_config,
    resolve_path,
    source_motion,
)
from utils.general import load_config
from utils.object_sampling import object_name_from_data
from utils.rotation import rot6d_to_mat


ROBOT_DIM = 38


def _load_pickle(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _record(path: str, max_len: int) -> dict[str, Any] | None:
    try:
        data = _load_pickle(path)
        motion = source_motion(data, max_len)
    except Exception as exc:
        print(f"Skipping {path}: {exc}")
        return None
    return {
        "path": path,
        "seq_name": str(data.get("seq_name", os.path.splitext(os.path.basename(path))[0])),
        "object_name": str(object_name_from_data(data, path)),
        "fps": float(data.get("fps", 30.0)),
        "length": int(motion.shape[0]),
        "initial_robot": motion[0, :ROBOT_DIM].copy(),
        "initial_object": motion[0, ROBOT_DIM:].copy(),
        "final_object": motion[-1, ROBOT_DIM:].copy(),
    }


def _motion(record: dict[str, Any], max_len: int) -> tuple[np.ndarray, dict[str, Any]]:
    data = _load_pickle(record["path"])
    return source_motion(data, max_len), data


def _parent_meta(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("path", "seq_name", "object_name", "length")}


def _resample(motion: np.ndarray, target_len: int) -> np.ndarray:
    if motion.shape[0] == target_len:
        return motion
    old_t = np.linspace(0.0, 1.0, motion.shape[0])
    new_t = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(new_t, old_t, motion[:, d]) for d in range(motion.shape[1])], axis=-1)


def _rotation_error_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> float:
    a = rot6d_to_mat(torch.as_tensor(rot6d_a, dtype=torch.float32).reshape(1, 6))[0]
    b = rot6d_to_mat(torch.as_tensor(rot6d_b, dtype=torch.float32).reshape(1, 6))[0]
    relative = a.transpose(0, 1) @ b
    cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).item())


def _path_length(xyz: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum()) if len(xyz) > 1 else 0.0


def _base_metrics(state: np.ndarray, g56: np.ndarray, success: dict[str, Any]) -> dict[str, Any]:
    robot, obj = state[:, :ROBOT_DIM], state[:, ROBOT_DIM:]
    final_pos_error = float(np.linalg.norm(obj[-1, :3] - g56[:3]))
    final_rot_error = _rotation_error_deg(obj[-1, 3:9], g56[3:9])
    pos_threshold = float(success.get("final_object_position_threshold", 0.10))
    rot_threshold = float(success.get("final_object_rotation_threshold_deg", 20.0))
    return {
        "final_object_position_error": final_pos_error,
        "final_object_rotation_error_deg": final_rot_error,
        "initial_robot_position_error": float(np.linalg.norm(robot[0, :3] - g56[9:12])),
        "initial_object_position_error": float(np.linalg.norm(obj[0, :3] - g56[47:50])),
        "root_displacement": float(np.linalg.norm(robot[-1, :3] - robot[0, :3])),
        "object_displacement": float(np.linalg.norm(obj[-1, :3] - obj[0, :3])),
        "root_path_length": _path_length(robot[:, :3]),
        "object_path_length": _path_length(obj[:, :3]),
        "success": bool(final_pos_error <= pos_threshold and final_rot_error <= rot_threshold),
    }


def _parent_distance(
    generated: np.ndarray,
    parent: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> float:
    parent_aligned = _resample(parent, generated.shape[0])
    scale = np.maximum(state_std.reshape(1, -1), 1e-8)
    generated_norm = (generated - state_mean.reshape(1, -1)) / scale
    parent_norm = (parent_aligned - state_mean.reshape(1, -1)) / scale
    return float(np.sqrt(np.mean(np.square(generated_norm - parent_norm))))


def _goal_values(cfg: dict[str, Any]) -> list[np.ndarray]:
    result = []
    for index, item in enumerate(cfg.get("arbitrary_goals") or []):
        value = item.get("final_object_frame") if isinstance(item, dict) else item
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape != (9,):
            raise ValueError(f"eval.arbitrary_goals[{index}] must contain exactly 9 values")
        result.append(arr)
    return result


def _choose_b_pair(
    records: list[dict[str, Any]],
    same_object_only: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eligible = []
    for init in records:
        candidates = [
            goal
            for goal in records
            if goal["path"] != init["path"]
            and (not same_object_only or goal["object_name"] == init["object_name"])
        ]
        if candidates:
            eligible.append((init, candidates))
    if not eligible:
        qualifier = "same-object " if same_object_only else ""
        raise RuntimeError(f"Mode B has no {qualifier}pairs with distinct parent motions")
    init, candidates = eligible[int(rng.integers(len(eligible)))]
    return init, candidates[int(rng.integers(len(candidates)))]


def _copy_source_metadata(output: dict[str, Any], data: dict[str, Any]) -> None:
    for key in ("object_name", "mesh_file", "num_verts", "is_articulated", "object_mesh_scale"):
        if key in data:
            output[key] = data[key]


def _save_qualitative(
    output_root: str,
    mode: str,
    index: int,
    sample: dict[str, Any],
) -> str:
    sample_dir = os.path.join(output_root, "qualitative", mode, f"sample_{index:05d}")
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(sample_dir, "sample.pkl")
    with open(path, "wb") as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
    return sample_dir


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    result: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "conditioning": "init_goal",
        "global_cond_layout": GLOBAL_COND_LAYOUT,
        "parent_distance_metric": "RMSE over linearly time-resampled trajectories in checkpoint-normalized 47D state space",
        "success_thresholds": cfg.get("success", {}),
        "modes": {},
    }
    for mode, mode_rows in grouped.items():
        numeric_keys = [
            key
            for key, value in mode_rows[0].items()
            if isinstance(value, (int, float, np.integer, np.floating)) and key not in ("sample_index",)
        ]
        averages = {
            key: float(np.mean([float(row[key]) for row in mode_rows if row.get(key) not in (None, "")]))
            for key in numeric_keys
            if any(row.get(key) not in (None, "") for row in mode_rows)
        }
        result["modes"][mode] = {
            "num_samples": len(mode_rows),
            "success_rate": float(np.mean([bool(row["success"]) for row in mode_rows])),
            "mean_metrics": averages,
        }
    return result


def _visualization_command(sample_dir: str, video_dir: str, vis_cfg: dict[str, Any]) -> str:
    args = [
        "python",
        resolve_path(str(vis_cfg.get("script", "./scripts/visualize_model_dynamic.py"))),
        "--robot",
        "unitree_g1_with_object",
        "--robot_motion_folder",
        sample_dir,
        "--gmr_root",
        str(vis_cfg.get("gmr_root", "/home/learning/Documents/g1-gmr")),
        "--objects_dir",
        str(vis_cfg.get("objects_dir", "/home/learning/Documents/omomo_release/data/captured_objects")),
        "--record_video",
        "--save_dir",
        video_dir,
        "--no_rate_limit",
        "--auto",
    ]
    reference = vis_cfg.get("reference_motion_folder")
    if reference:
        args.extend(["--reference_motion_folder", str(reference)])
    return " ".join(shlex.quote(str(x)) for x in args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-stage init+goal robot-object diffusion")
    parser.add_argument(
        "--config_path",
        default=os.path.join(PROJECT_ROOT, "experiments", "object_goal", "eval_object_goal_single_stage_init_goal_hf_bps.yaml"),
    )
    args = parser.parse_args()
    yml = load_config(args.config_path)
    cfg = yml["eval"]
    modes = [str(mode).upper() for mode in cfg.get("modes", ["A", "B", "C"])]
    unknown = set(modes) - {"A", "B", "C"}
    if unknown:
        raise ValueError(f"Unknown eval modes: {sorted(unknown)}")
    goals = _goal_values(cfg)
    perturb = cfg.get("perturbation") or {}
    if "C" in modes and not goals and not bool(perturb.get("enabled", False)):
        raise ValueError("Mode C requires eval.arbitrary_goals or explicit perturbation.enabled=true")

    input_dir = resolve_path(str(cfg.get("input_source_dir", yml.get("root_dir", "./data/hf_bps_preprocessed"))))
    max_len, min_len = int(cfg.get("max_len", 300)), int(cfg.get("min_len", 30))
    records = []
    for path in tqdm(sorted(glob.glob(os.path.join(input_dir, "*.pkl"))), desc="Indexing eval sources"):
        record = _record(path, max_len)
        if record is not None and record["length"] >= min_len:
            records.append(record)
    if not records:
        raise RuntimeError(f"No valid evaluation PKLs found in {input_dir}")

    ckpt_value = str(cfg.get("ckpt_path") or "").strip()
    if not ckpt_value:
        raise ValueError(f"Missing eval.ckpt_path in {args.config_path}")
    requested_device = str(cfg.get("device", "cuda:0"))
    device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inference = inference_config(yml.get("optimization", {}))
    pipeline = SingleStageInitGoalPipeline(resolve_path(ckpt_value), inference, device)
    state_mean = pipeline.state_mean.float().cpu().numpy()
    state_std = pipeline.state_std.float().cpu().numpy()

    configured_output = cfg.get("output_dir")
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    output_root = resolve_path(str(configured_output)) if configured_output else os.path.join(
        PROJECT_ROOT, "evals", "object_goal_single_stage_init_goal", timestamp
    )
    os.makedirs(output_root, exist_ok=True)
    num_samples = int(cfg.get("num_samples", 1000))
    qualitative_count = min(int(cfg.get("qualitative_num_samples", 10)), num_samples)
    same_object_only = bool(cfg.get("same_object_only", True))
    unclear_margin = float(cfg.get("parent_unclear_margin", 0.05))
    success_cfg = cfg.get("success") or {}
    rows: list[dict[str, Any]] = []
    visualization_commands: list[str] = []

    for mode in modes:
        mode_rows = []
        for sample_index in tqdm(range(num_samples), desc=f"Eval {mode}"):
            if mode == "B":
                init_record, goal_record = _choose_b_pair(records, same_object_only, rng)
            else:
                init_record = records[int(rng.integers(len(records)))]
                goal_record = init_record
            init_motion, init_data = _motion(init_record, max_len)
            goal_motion = init_motion
            if mode == "B":
                goal_motion, _ = _motion(goal_record, max_len)

            final_goal = goal_record["final_object"].copy()
            arbitrary_goal_id: int | str = ""
            if mode == "C":
                if goals:
                    arbitrary_goal_id = int(sample_index % len(goals))
                    final_goal = goals[int(arbitrary_goal_id)].copy()
                else:
                    pos_delta = np.asarray(perturb.get("position_delta", [0, 0, 0]), dtype=np.float32).reshape(-1)
                    rot_delta = np.asarray(perturb.get("rotation_6d_delta", [0] * 6), dtype=np.float32).reshape(-1)
                    if pos_delta.shape != (3,) or rot_delta.shape != (6,):
                        raise ValueError("Mode C perturbation deltas must have 3 and 6 values")
                    final_goal[:3] += pos_delta
                    final_goal[3:9] += rot_delta
                    arbitrary_goal_id = "perturbation"
            g56 = np.concatenate([final_goal, init_record["initial_robot"], init_record["initial_object"]]).astype(np.float32)
            output = pipeline.generate(g56, init_motion.shape[0])
            output.update(
                eval_mode=mode,
                source_path=init_record["path"],
                seq_name=f"{init_record['seq_name']}_eval{mode}_{sample_index:05d}",
                fps=init_record["fps"],
                sampler=inference.sampler.value,
                num_inference_steps=inference.num_inference_steps,
                init_parent=_parent_meta(init_record),
                goal_parent=_parent_meta(goal_record) if mode in ("A", "B") else None,
                arbitrary_goal_id=arbitrary_goal_id,
            )
            _copy_source_metadata(output, init_data)
            metrics = _base_metrics(output["state"], g56, success_cfg)
            row: dict[str, Any] = {
                "mode": mode,
                "sample_index": sample_index,
                "init_parent": init_record["seq_name"],
                "init_parent_path": init_record["path"],
                "goal_parent": goal_record["seq_name"] if mode in ("A", "B") else "arbitrary",
                "goal_parent_path": goal_record["path"] if mode in ("A", "B") else "",
                "object_name": init_record["object_name"],
                "arbitrary_goal_id": arbitrary_goal_id,
                **metrics,
            }
            if mode == "B":
                init_distance = _parent_distance(output["state"], init_motion, state_mean, state_std)
                goal_distance = _parent_distance(output["state"], goal_motion, state_mean, state_std)
                if abs(init_distance - goal_distance) <= unclear_margin:
                    closer = "unclear"
                else:
                    closer = "init_parent" if init_distance < goal_distance else "goal_parent"
                row.update(
                    distance_to_init_parent_motion=init_distance,
                    distance_to_goal_parent_motion=goal_distance,
                    closer_parent=closer,
                )
                output["parent_comparison"] = {
                    "metric": "checkpoint-normalized 47D trajectory RMSE after linear time resampling",
                    "distance_to_init_parent_motion": init_distance,
                    "distance_to_goal_parent_motion": goal_distance,
                    "closer_parent": closer,
                    "unclear_margin": unclear_margin,
                }
            rows.append(row)
            mode_rows.append(row)
            if sample_index < qualitative_count:
                sample_dir = _save_qualitative(output_root, mode, sample_index, output)
                video_dir = os.path.join(output_root, "qualitative", mode, "videos")
                visualization_commands.append(_visualization_command(sample_dir, video_dir, yml.get("visualization", {})))
        _write_csv(os.path.join(output_root, "quantitative", mode, "metrics.csv"), mode_rows)

    _write_csv(os.path.join(output_root, "metrics.csv"), rows)
    summary = _summary(rows, cfg)
    summary.update(
        checkpoint=resolve_path(ckpt_value),
        input_source_dir=input_dir,
        output_dir=output_root,
        sampler=inference.sampler.value,
        num_inference_steps=inference.num_inference_steps,
        same_object_only=same_object_only,
    )
    with open(os.path.join(output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    commands_path = os.path.join(output_root, "qualitative", "visualize_commands.sh")
    os.makedirs(os.path.dirname(commands_path), exist_ok=True)
    with open(commands_path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -e\n\n")
        f.write("source /home/learning/miniconda3/etc/profile.d/conda.sh\n")
        f.write("conda activate g1-gmr\n\n")
        f.write("\n".join(visualization_commands) + "\n")
    print(f"Evaluation complete: {output_root}")
    print(f"Metrics: {os.path.join(output_root, 'metrics.csv')}")
    print(f"Summary: {os.path.join(output_root, 'summary.json')}")
    print(f"Visualization commands: {commands_path}")


if __name__ == "__main__":
    main()
