"""Evaluate the single-stage init+goal model and its conditioning robustness."""

from __future__ import annotations

import argparse
import copy
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
from utils.eval_plotting import save_plot
from utils.general import load_config
from utils.object_sampling import object_name_from_data
from utils.rotation import rot6d_to_mat


ROBOT_DIM = 38
VALID_DIRECTIONS = {"+x", "-x", "+y", "-y", "random_xy"}


def _load_pickle(path: str) -> dict[str, Any]:
    with open(path, "rb") as file:
        return pickle.load(file)


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
    return np.stack([np.interp(new_t, old_t, motion[:, dim]) for dim in range(motion.shape[1])], axis=-1)


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
    object_goal_delta = g56[:3] - g56[47:50]
    return {
        "final_object_position_error": final_pos_error,
        "final_object_rotation_error_deg": final_rot_error,
        "initial_robot_position_error": float(np.linalg.norm(robot[0, :3] - g56[9:12])),
        "initial_object_position_error": float(np.linalg.norm(obj[0, :3] - g56[47:50])),
        "root_displacement": float(np.linalg.norm(robot[-1, :3] - robot[0, :3])),
        "object_displacement": float(np.linalg.norm(obj[-1, :3] - obj[0, :3])),
        "root_path_length": _path_length(robot[:, :3]),
        "object_path_length": _path_length(obj[:, :3]),
        "initial_to_goal_object_distance": float(np.linalg.norm(object_goal_delta)),
        "object_goal_delta_x": float(object_goal_delta[0]),
        "object_goal_delta_y": float(object_goal_delta[1]),
        "object_goal_delta_z": float(object_goal_delta[2]),
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
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (9,):
            raise ValueError(f"eval.arbitrary_goals[{index}] must contain exactly 9 values")
        result.append(array)
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


def _save_pickle(path: str, sample: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(sample, file, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _save_qualitative(output_root: str, mode: str, index: int, sample: dict[str, Any]) -> str:
    return _save_pickle(
        os.path.join(output_root, "qualitative", mode, f"sample_{index:05d}", "sample.pkl"),
        sample,
    )


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(value, file, indent=2)


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = {
        "sample_index", "goal_index", "initial_condition_index", "source_index",
        "radius_m", "actual_radius_m", "success",
    }
    numeric_keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in excluded and key not in numeric_keys and isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
                numeric_keys.append(key)
    metrics: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        if values:
            metrics[key] = {
                "mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values)),
            }
    return {
        "num_samples": len(rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])) if rows else None,
        "metrics": metrics,
    }


def _legacy_summary(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
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
        detailed = _metric_summary(mode_rows)
        result["modes"][mode] = {
            "num_samples": detailed["num_samples"],
            "success_rate": detailed["success_rate"],
            "mean_metrics": {key: values["mean"] for key, values in detailed["metrics"].items()},
        }
    return result


def _visualization_command(sample_path: str, video_dir: str, vis_cfg: dict[str, Any]) -> str:
    args = [
        "python", resolve_path(str(vis_cfg.get("script", "./scripts/visualize_model_dynamic.py"))),
        "--robot", "unitree_g1_with_object", "--robot_motion_folder", os.path.dirname(sample_path),
        "--gmr_root", str(vis_cfg.get("gmr_root", "/home/learning/Documents/g1-gmr")),
        "--objects_dir", str(vis_cfg.get("objects_dir", "/home/learning/Documents/omomo_release/data/captured_objects")),
        "--record_video", "--save_dir", video_dir, "--no_rate_limit", "--auto",
    ]
    reference = vis_cfg.get("reference_motion_folder")
    if reference:
        args.extend(["--reference_motion_folder", str(reference)])
    return " ".join(shlex.quote(str(item)) for item in args)


def _ghost_command(sample_path: str, video_path: str, vis_cfg: dict[str, Any]) -> str:
    ghost_cfg = vis_cfg.get("ghost") or {}
    args = [
        "python", resolve_path(str(ghost_cfg.get("script", "./scripts/visualize_object_goal_ghost_trajectories.py"))),
        "--sample_path", sample_path, "--output_path", video_path,
        "--gmr_root", str(vis_cfg.get("gmr_root", "/home/learning/Documents/g1-gmr")),
        "--objects_dir", str(vis_cfg.get("objects_dir", "/home/learning/Documents/omomo_release/data/captured_objects")),
        "--ghost_alpha", str(float(ghost_cfg.get("alpha", 0.25))),
    ]
    return " ".join(shlex.quote(str(item)) for item in args)


def _write_command_script(path: str, commands: list[str], environment: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write("#!/usr/bin/env bash\nset -e\n\n")
        file.write("source /home/learning/miniconda3/etc/profile.d/conda.sh\n")
        file.write(f"conda activate {environment}\n\n")
        if commands:
            file.write("\n".join(commands) + "\n")


def _standard_modes(
    *, cfg: dict[str, Any], yml: dict[str, Any], modes: list[str], goals: list[np.ndarray],
    records: list[dict[str, Any]], pipeline: SingleStageInitGoalPipeline, inference: Any,
    output_root: str, rng: np.random.Generator, max_len: int,
    state_mean: np.ndarray, state_std: np.ndarray,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    num_samples = int(cfg.get("num_samples", 1000))
    qualitative_count = min(int(cfg.get("qualitative_num_samples", 10)), num_samples)
    same_object_only = bool(cfg.get("same_object_only", True))
    unclear_margin = float(cfg.get("parent_unclear_margin", 0.05))
    success_cfg = cfg.get("success") or {}
    perturb = cfg.get("perturbation") or {}
    vis_cfg = yml.get("visualization") or {}
    ghost_enabled = bool((vis_cfg.get("ghost") or {}).get("enabled", False))
    rows: list[dict[str, Any]] = []
    regular_commands: list[str] = []
    ghost_commands: list[str] = []
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
                eval_mode=mode, source_path=init_record["path"],
                seq_name=f"{init_record['seq_name']}_eval{mode}_{sample_index:05d}", fps=init_record["fps"],
                sampler=inference.sampler.value, num_inference_steps=inference.num_inference_steps,
                init_parent=_parent_meta(init_record),
                goal_parent=_parent_meta(goal_record) if mode in ("A", "B") else None,
                arbitrary_goal_id=arbitrary_goal_id,
            )
            _copy_source_metadata(output, init_data)
            metrics = _base_metrics(output["state"], g56, success_cfg)
            row: dict[str, Any] = {
                "mode": mode, "sample_index": sample_index,
                "init_parent": init_record["seq_name"], "init_parent_path": init_record["path"],
                "goal_parent": goal_record["seq_name"] if mode in ("A", "B") else "arbitrary",
                "goal_parent_path": goal_record["path"] if mode in ("A", "B") else "",
                "object_name": init_record["object_name"], "arbitrary_goal_id": arbitrary_goal_id, **metrics,
            }
            if mode == "C" and not goals:
                row["object_goal_radius_m"] = float(np.linalg.norm(final_goal[:3] - goal_record["final_object"][:3]))
            if mode == "B":
                init_distance = _parent_distance(output["state"], init_motion, state_mean, state_std)
                goal_distance = _parent_distance(output["state"], goal_motion, state_mean, state_std)
                closer = "unclear" if abs(init_distance - goal_distance) <= unclear_margin else (
                    "init_parent" if init_distance < goal_distance else "goal_parent"
                )
                row.update(distance_to_init_parent_motion=init_distance, distance_to_goal_parent_motion=goal_distance, closer_parent=closer)
                output["parent_comparison"] = {
                    "metric": "checkpoint-normalized 47D trajectory RMSE after linear time resampling",
                    "distance_to_init_parent_motion": init_distance,
                    "distance_to_goal_parent_motion": goal_distance,
                    "closer_parent": closer, "unclear_margin": unclear_margin,
                }
            rows.append(row)
            mode_rows.append(row)
            if sample_index < qualitative_count:
                sample_path = _save_qualitative(output_root, mode, sample_index, output)
                regular_commands.append(_visualization_command(sample_path, os.path.join(output_root, "qualitative", mode, "videos"), vis_cfg))
                if ghost_enabled:
                    ghost_commands.append(_ghost_command(
                        sample_path,
                        os.path.join(output_root, "qualitative", mode, "videos_ghost", f"sample_{sample_index:05d}.mp4"),
                        vis_cfg,
                    ))
        _write_csv(os.path.join(output_root, "quantitative", mode, "metrics.csv"), mode_rows)
    return rows, regular_commands, ghost_commands


def _group_summaries(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _metric_summary(group) for name, group in sorted(grouped.items())}


def _init_sweep(
    *, section: dict[str, Any], records: list[dict[str, Any]], pipeline: SingleStageInitGoalPipeline,
    inference: Any, output_root: str, success_cfg: dict[str, Any], vis_cfg: dict[str, Any], max_len: int,
) -> tuple[dict[str, Any], list[str]]:
    sweep_root = os.path.join(output_root, "init_sweep")
    rng = np.random.default_rng(int(section.get("seed", 42)))
    num_goals = int(section.get("num_goals", 5))
    num_initials = int(section.get("num_initial_conditions_per_goal", 20))
    same_object_only = bool(section.get("same_object_only", True))
    include_baseline = bool(section.get("include_goal_parent_initial", True))
    qualitative_per_goal = int(section.get("qualitative_num_samples_per_goal", 1))
    if num_goals <= 0 or num_initials <= 0:
        raise ValueError("init_sweep counts must be positive")
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_object[record["object_name"]].append(record)
    if same_object_only:
        eligible_objects = {name: group for name, group in by_object.items() if len(group) >= num_initials}
        eligible_goals = [record for group in eligible_objects.values() for record in group]
    else:
        eligible_objects = dict(by_object)
        eligible_goals = records if len(records) >= num_initials else []
    if len(eligible_goals) < num_goals:
        counts = ", ".join(f"{name}:{len(group)}" for name, group in sorted(by_object.items()))
        raise RuntimeError(f"init_sweep needs {num_goals} eligible goals with {num_initials} unique initials; object counts are {counts}")
    goal_indices = rng.choice(len(eligible_goals), size=num_goals, replace=False)
    goal_records = [eligible_goals[int(index)] for index in goal_indices]
    rows: list[dict[str, Any]] = []
    ghost_commands: list[str] = []
    ghost_enabled = bool((vis_cfg.get("ghost") or {}).get("enabled", False))
    for goal_index, goal_record in enumerate(tqdm(goal_records, desc="Init-sweep goals")):
        candidates = by_object[goal_record["object_name"]] if same_object_only else records
        remaining = [record for record in candidates if record["path"] != goal_record["path"]]
        selected: list[dict[str, Any]] = [goal_record] if include_baseline else []
        needed = num_initials - len(selected)
        if len(remaining) < needed:
            raise RuntimeError(f"Not enough distinct initial conditions for goal {goal_record['seq_name']}")
        chosen = rng.choice(len(remaining), size=needed, replace=False)
        selected.extend(remaining[int(index)] for index in chosen)
        target_goal_id = f"goal_{goal_index:03d}"
        for init_index, init_record in enumerate(selected):
            init_motion, init_data = _motion(init_record, max_len)
            g56 = np.concatenate([goal_record["final_object"], init_record["initial_robot"], init_record["initial_object"]]).astype(np.float32)
            output = pipeline.generate(g56, init_motion.shape[0])
            output.update(
                eval_mode="init_sweep", seq_name=f"{target_goal_id}_init_{init_index:03d}", fps=init_record["fps"],
                sampler=inference.sampler.value, num_inference_steps=inference.num_inference_steps,
                source_path=init_record["path"], init_parent=_parent_meta(init_record), goal_parent=_parent_meta(goal_record),
                target_goal_id=target_goal_id,
            )
            _copy_source_metadata(output, init_data)
            sample_path = _save_pickle(os.path.join(sweep_root, "samples", target_goal_id, f"sample_{init_index:03d}.pkl"), output)
            row = {
                "mode": "init_sweep", "sample_index": len(rows), "goal_index": goal_index,
                "target_goal_id": target_goal_id, "target_goal_parent": goal_record["seq_name"],
                "target_goal_parent_path": goal_record["path"], "initial_condition_index": init_index,
                "init_parent": init_record["seq_name"], "init_parent_path": init_record["path"],
                "object_name": goal_record["object_name"],
                "is_paired_baseline": bool(init_record["path"] == goal_record["path"]),
                **_base_metrics(output["state"], g56, success_cfg),
            }
            rows.append(row)
            if ghost_enabled and init_index < qualitative_per_goal:
                ghost_commands.append(_ghost_command(
                    sample_path,
                    os.path.join(sweep_root, "qualitative", "videos_ghost", f"{target_goal_id}_init_{init_index:03d}.mp4"),
                    vis_cfg,
                ))
    _write_csv(os.path.join(sweep_root, "metrics.csv"), rows)
    summary = {
        "definition": "Fixed final object goal with varied initial robot/object frames",
        "same_object_only": same_object_only, "include_goal_parent_initial": include_baseline,
        "eligible_object_counts": {name: len(group) for name, group in sorted(eligible_objects.items())},
        "all_samples": _metric_summary(rows), "per_target_goal": _group_summaries(rows, "target_goal_id"),
        "per_object_identity": _group_summaries(rows, "object_name"),
    }
    _write_json(os.path.join(sweep_root, "summary.json"), summary)
    goal_names = sorted(summary["per_target_goal"])
    positions = list(range(len(goal_names)))
    plot_root = os.path.join(sweep_root, "plots")
    save_plot(
        os.path.join(plot_root, "final_position_error_by_goal.png"), positions,
        {"mean": [summary["per_target_goal"][name]["metrics"]["final_object_position_error"]["mean"] for name in goal_names]},
        title="Final object position error by target goal", xlabel="Target goal", ylabel="Position error (m)",
        kind="bar", x_tick_labels=goal_names,
    )
    save_plot(
        os.path.join(plot_root, "success_rate_by_goal.png"), positions,
        {"success_rate": [summary["per_target_goal"][name]["success_rate"] for name in goal_names]},
        title="Success rate by target goal", xlabel="Target goal", ylabel="Success rate", kind="bar", x_tick_labels=goal_names,
    )
    distances = [float(row["initial_to_goal_object_distance"]) for row in rows]
    save_plot(
        os.path.join(plot_root, "final_position_error_vs_initial_to_goal_distance.png"), distances,
        {"final_position_error": [float(row["final_object_position_error"]) for row in rows]},
        title="Final position error vs initial-to-goal distance", xlabel="Initial-to-goal object distance (m)",
        ylabel="Final position error (m)", kind="scatter",
    )
    save_plot(
        os.path.join(plot_root, "path_length_vs_initial_to_goal_distance.png"), distances,
        {"root_path_length": [float(row["root_path_length"]) for row in rows], "object_path_length": [float(row["object_path_length"]) for row in rows]},
        title="Generated path length vs initial-to-goal distance", xlabel="Initial-to-goal object distance (m)",
        ylabel="Path length (m)", kind="scatter",
    )
    return summary, ghost_commands


def _direction_vectors(count: int, directions: list[str], rng: np.random.Generator) -> tuple[list[str], list[np.ndarray]]:
    labels = [directions[index % len(directions)] for index in range(count)]
    rng.shuffle(labels)
    vectors = []
    for label in labels:
        if label == "+x":
            vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif label == "-x":
            vector = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        elif label == "+y":
            vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif label == "-y":
            vector = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            vector = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        vectors.append(vector)
    return labels, vectors


def _radius_flat_summary(radius: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    detailed = _metric_summary(rows)
    flat: dict[str, Any] = {
        "radius_m": radius, "num_samples": detailed["num_samples"], "success_rate": detailed["success_rate"],
        "failure_rate": 1.0 - float(detailed["success_rate"]),
    }
    for key, values in detailed["metrics"].items():
        for statistic, value in values.items():
            flat[f"{statistic}_{key}"] = value
    return flat


def _ood_plots(sweep_root: str, summaries: list[dict[str, Any]], elbow: dict[str, Any]) -> None:
    radii = [float(row["radius_m"]) for row in summaries]
    plot_root = os.path.join(sweep_root, "plots")
    save_plot(os.path.join(plot_root, "final_position_error_vs_radius.png"), radii,
              {"mean": [row["mean_final_object_position_error"] for row in summaries], "median": [row["median_final_object_position_error"] for row in summaries]},
              title="Final object position error vs OOD goal radius", xlabel="Object-goal radius (m)", ylabel="Position error (m)")
    save_plot(os.path.join(plot_root, "final_rotation_error_vs_radius.png"), radii,
              {"mean": [row["mean_final_object_rotation_error_deg"] for row in summaries], "median": [row["median_final_object_rotation_error_deg"] for row in summaries]},
              title="Final object rotation error vs OOD goal radius", xlabel="Object-goal radius (m)", ylabel="Rotation error (deg)")
    save_plot(os.path.join(plot_root, "success_rate_vs_radius.png"), radii,
              {"success_rate": [row["success_rate"] for row in summaries]}, title="Success rate vs OOD goal radius",
              xlabel="Object-goal radius (m)", ylabel="Success rate", horizontal_lines={"80% threshold": float(elbow["success_rate_threshold"])})
    save_plot(os.path.join(plot_root, "initial_error_vs_radius.png"), radii,
              {"robot": [row["mean_initial_robot_position_error"] for row in summaries], "object": [row["mean_initial_object_position_error"] for row in summaries]},
              title="Initial conditioning error vs OOD goal radius", xlabel="Object-goal radius (m)", ylabel="Initial position error (m)")
    save_plot(os.path.join(plot_root, "path_length_vs_radius.png"), radii,
              {"root": [row["mean_root_path_length"] for row in summaries], "object": [row["mean_object_path_length"] for row in summaries]},
              title="Generated path length vs OOD goal radius", xlabel="Object-goal radius (m)", ylabel="Path length (m)")
    position_elbow = elbow["first_radius_mean_position_error_above_threshold_m"]
    success_elbow = elbow["first_radius_success_rate_below_threshold_m"]
    save_plot(os.path.join(plot_root, "elbow_final_position_error.png"), radii,
              {"mean_position_error": [row["mean_final_object_position_error"] for row in summaries]},
              title="Position-error elbow", xlabel="Object-goal radius (m)", ylabel="Mean position error (m)",
              vertical_lines={} if position_elbow is None else {"estimated elbow": position_elbow},
              horizontal_lines={"error threshold": float(elbow["position_error_threshold_m"])})
    save_plot(os.path.join(plot_root, "elbow_failure_rate.png"), radii,
              {"failure_rate": [row["failure_rate"] for row in summaries]}, title="Failure-rate elbow",
              xlabel="Object-goal radius (m)", ylabel="Failure rate",
              vertical_lines={} if success_elbow is None else {"estimated elbow": success_elbow},
              horizontal_lines={"20% failure": 1.0 - float(elbow["success_rate_threshold"])})


def _ood_radius_sweep(
    *, section: dict[str, Any], records: list[dict[str, Any]], pipeline: SingleStageInitGoalPipeline,
    inference: Any, output_root: str, success_cfg: dict[str, Any], vis_cfg: dict[str, Any], max_len: int,
) -> tuple[dict[str, Any], list[str]]:
    sweep_root = os.path.join(output_root, "ood_radius_sweep")
    rng = np.random.default_rng(int(section.get("seed", 42)))
    radii = [float(value) for value in section.get("radii_m", [])]
    if not radii or any(radius < 0.0 for radius in radii):
        raise ValueError("ood_radius_sweep.radii_m must contain non-negative radii")
    radii = sorted(set(radii))
    sample_count = int(section.get("samples_per_radius", 100))
    if sample_count <= 0:
        raise ValueError("ood_radius_sweep.samples_per_radius must be positive")
    directions = [str(value).lower() for value in section.get("directions", ["+x"])]
    unknown = set(directions) - VALID_DIRECTIONS
    if not directions or unknown:
        raise ValueError(f"Unknown OOD directions: {sorted(unknown)}; valid values are {sorted(VALID_DIRECTIONS)}")
    source_indices = rng.choice(len(records), size=sample_count, replace=sample_count > len(records))
    sources = [records[int(index)] for index in source_indices]
    direction_labels, direction_vectors = _direction_vectors(sample_count, directions, rng)
    save_samples = bool(section.get("save_samples", False))
    qualitative_per_radius = int(section.get("qualitative_num_samples_per_radius", 0))
    ghost_enabled = bool((vis_cfg.get("ghost") or {}).get("enabled", False))
    rows: list[dict[str, Any]] = []
    ghost_commands: list[str] = []
    for radius in tqdm(radii, desc="OOD radii"):
        for source_index, (record, direction, unit_vector) in enumerate(zip(sources, direction_labels, direction_vectors)):
            motion, data = _motion(record, max_len)
            delta = unit_vector * float(radius)
            final_goal = record["final_object"].copy()
            source_rotation = final_goal[3:9].copy()
            final_goal[:3] += delta
            if not np.array_equal(final_goal[3:9], source_rotation):
                raise AssertionError("OOD radius sweep must not modify object rotation")
            actual_radius = float(np.linalg.norm(final_goal[:3] - record["final_object"][:3]))
            g56 = np.concatenate([final_goal, record["initial_robot"], record["initial_object"]]).astype(np.float32)
            output = pipeline.generate(g56, motion.shape[0])
            output.update(
                eval_mode="ood_radius_sweep", seq_name=f"radius_{radius:.3f}_source_{source_index:04d}", fps=record["fps"],
                sampler=inference.sampler.value, num_inference_steps=inference.num_inference_steps, source_path=record["path"],
                init_parent=_parent_meta(record), goal_parent=_parent_meta(record), radius_m=float(radius), actual_radius_m=actual_radius,
                perturbation_direction=direction, perturbation_delta_xyz=delta.copy(),
            )
            _copy_source_metadata(output, data)
            sample_path = os.path.join(sweep_root, "samples", f"radius_{radius:.3f}", f"sample_{source_index:04d}.pkl")
            if save_samples or source_index < qualitative_per_radius:
                _save_pickle(sample_path, output)
            row = {
                "mode": "ood_radius_sweep", "sample_index": len(rows), "source_index": source_index,
                "source_parent": record["seq_name"], "source_parent_path": record["path"], "object_name": record["object_name"],
                "radius_m": float(radius), "actual_radius_m": actual_radius, "direction": direction,
                "goal_delta_x": float(delta[0]), "goal_delta_y": float(delta[1]), "goal_delta_z": float(delta[2]),
                **_base_metrics(output["state"], g56, success_cfg),
            }
            rows.append(row)
            if ghost_enabled and source_index < qualitative_per_radius:
                ghost_commands.append(_ghost_command(
                    sample_path,
                    os.path.join(sweep_root, "qualitative", "videos_ghost", f"radius_{radius:.3f}_sample_{source_index:04d}.mp4"),
                    vis_cfg,
                ))
    _write_csv(os.path.join(sweep_root, "metrics.csv"), rows)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["radius_m"])].append(row)
    summaries = [_radius_flat_summary(radius, grouped[radius]) for radius in sorted(grouped)]
    _write_csv(os.path.join(sweep_root, "summary_by_radius.csv"), summaries)
    _write_json(os.path.join(sweep_root, "summary_by_radius.json"), summaries)
    elbow_cfg = section.get("elbow") or {}
    success_threshold = float(elbow_cfg.get("success_rate_below", 0.80))
    position_threshold = float(elbow_cfg.get("mean_position_error_above_m", 0.10))
    success_elbow = next((row["radius_m"] for row in summaries if row["success_rate"] < success_threshold), None)
    position_elbow = next((row["radius_m"] for row in summaries if row["mean_final_object_position_error"] > position_threshold), None)
    elbow = {
        "method": "first configured radius crossing each threshold", "success_rate_threshold": success_threshold,
        "position_error_threshold_m": position_threshold, "first_radius_success_rate_below_threshold_m": success_elbow,
        "first_radius_mean_position_error_above_threshold_m": position_elbow,
    }
    _write_json(os.path.join(sweep_root, "elbow_summary.json"), elbow)
    summary = {
        "object_goal_radius_definition": "L2 norm of perturbed final object XYZ minus source final object XYZ, in meters",
        "rotation_policy": "source final object rotation is unchanged",
        "samples_per_radius_semantics": "total per radius, distributed approximately evenly across directions",
        "same_source_set_across_radii": True, "same_object_only": bool(section.get("same_object_only", True)),
        "directions": directions, "all_samples": _metric_summary(rows), "by_radius": summaries, "elbow": elbow,
    }
    _write_json(os.path.join(sweep_root, "summary.json"), summary)
    _ood_plots(sweep_root, summaries, elbow)
    return summary, ghost_commands


def _apply_smoke_overrides(yml: dict[str, Any]) -> None:
    cfg = yml["eval"]
    cfg["num_samples"] = 1
    cfg["qualitative_num_samples"] = 1
    cfg["max_len"] = min(int(cfg.get("max_len", 300)), 30)
    cfg["min_len"] = min(int(cfg.get("min_len", 30)), 30)
    optimization = yml.setdefault("optimization", {})
    optimization["sampler"] = "ddim"
    optimization["num_inference_steps"] = 2
    if (cfg.get("init_sweep") or {}).get("enabled", False):
        cfg["init_sweep"].update(num_goals=1, num_initial_conditions_per_goal=2, qualitative_num_samples_per_goal=1)
    if (cfg.get("ood_radius_sweep") or {}).get("enabled", False):
        radii = list(cfg["ood_radius_sweep"].get("radii_m", [0.0, 0.05]))
        cfg["ood_radius_sweep"].update(radii_m=radii[:2], samples_per_radius=2, save_samples=True, qualitative_num_samples_per_radius=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-stage init+goal robot-object diffusion")
    parser.add_argument("--config_path", default=os.path.join(PROJECT_ROOT, "experiments", "object_goal", "eval_object_goal_single_stage_init_goal_hf_bps.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Run a bounded infrastructure smoke evaluation")
    args = parser.parse_args()
    yml = copy.deepcopy(load_config(args.config_path))
    if args.smoke:
        _apply_smoke_overrides(yml)
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
    output_root = resolve_path(str(configured_output)) if configured_output else os.path.join(PROJECT_ROOT, "evals", "object_goal_single_stage_init_goal", timestamp)
    os.makedirs(output_root, exist_ok=True)
    rows, regular_commands, ghost_commands = _standard_modes(
        cfg=cfg, yml=yml, modes=modes, goals=goals, records=records, pipeline=pipeline, inference=inference,
        output_root=output_root, rng=rng, max_len=max_len, state_mean=state_mean, state_std=state_std,
    )
    _write_csv(os.path.join(output_root, "metrics.csv"), rows)
    summary = _legacy_summary(rows, cfg)
    vis_cfg = yml.get("visualization") or {}
    init_section = cfg.get("init_sweep") or {}
    if bool(init_section.get("enabled", False)):
        init_summary, commands = _init_sweep(
            section=init_section, records=records, pipeline=pipeline, inference=inference, output_root=output_root,
            success_cfg=cfg.get("success") or {}, vis_cfg=vis_cfg, max_len=max_len,
        )
        summary["init_sweep"] = init_summary
        ghost_commands.extend(commands)
    radius_section = cfg.get("ood_radius_sweep") or {}
    if bool(radius_section.get("enabled", False)):
        radius_summary, commands = _ood_radius_sweep(
            section=radius_section, records=records, pipeline=pipeline, inference=inference, output_root=output_root,
            success_cfg=cfg.get("success") or {}, vis_cfg=vis_cfg, max_len=max_len,
        )
        summary["ood_radius_sweep"] = radius_summary
        ghost_commands.extend(commands)
    summary.update(
        checkpoint=resolve_path(ckpt_value), input_source_dir=input_dir, output_dir=output_root,
        sampler=inference.sampler.value, num_inference_steps=inference.num_inference_steps,
        same_object_only=bool(cfg.get("same_object_only", True)), smoke=bool(args.smoke),
    )
    _write_json(os.path.join(output_root, "summary.json"), summary)
    _write_command_script(os.path.join(output_root, "qualitative", "visualize_commands.sh"), regular_commands, "g1-gmr")
    _write_command_script(os.path.join(output_root, "qualitative", "visualize_ghost_commands.sh"), ghost_commands, "g1-gmr")
    print(f"Evaluation complete: {output_root}")
    print(f"Metrics: {os.path.join(output_root, 'metrics.csv')}")
    print(f"Summary: {os.path.join(output_root, 'summary.json')}")
    print(f"Ghost visualization commands: {os.path.join(output_root, 'qualitative', 'visualize_ghost_commands.sh')}")


if __name__ == "__main__":
    main()
