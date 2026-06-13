import argparse
import glob
import os
import pickle
import re
import sys
import types

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

from utils.contact_constraints import compute_contact_metrics, compute_hand_jpe
from utils.robot_kinematics import (
    compute_robot_contact_report,
    compute_robot_motion_quality_report,
    robot_hand_positions,
    robot_foot_proxy_positions,
)


def nearest_surface_distances(hands, object_verts):
    distances = []
    for t in range(len(hands)):
        distances.append(np.linalg.norm(object_verts[t] - hands[t, :3], axis=1).min())
        distances.append(np.linalg.norm(object_verts[t] - hands[t, 3:], axis=1).min())
    return np.asarray(distances, dtype=np.float64)


def parse_generated_fps(sample_dir):
    summary_path = os.path.join(sample_dir, "performance_summary.txt")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        text = f.read()
    match = re.search(r"Generated fps:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def compute_foot_contact_report(
    pred_root_pos,
    pred_root_rot,
    pred_dof_pos,
    gt_root_pos,
    gt_root_rot,
    gt_dof_pos,
    floor_height,
    contact_height_threshold,
):
    T = min(
        len(pred_root_pos),
        len(pred_root_rot),
        len(pred_dof_pos),
        len(gt_root_pos),
        len(gt_root_rot),
        len(gt_dof_pos),
    )
    if T == 0:
        return {
            "foot_contact_precision": 0.0,
            "foot_contact_recall": 0.0,
            "foot_contact_f1": 0.0,
            "foot_floating_frac": 0.0,
        }

    pred_feet = robot_foot_proxy_positions(
        np.asarray(pred_root_pos[:T], dtype=np.float32),
        np.asarray(pred_root_rot[:T], dtype=np.float32),
        np.asarray(pred_dof_pos[:T], dtype=np.float32),
    )
    gt_feet = robot_foot_proxy_positions(
        np.asarray(gt_root_pos[:T], dtype=np.float32),
        np.asarray(gt_root_rot[:T], dtype=np.float32),
        np.asarray(gt_dof_pos[:T], dtype=np.float32),
    )
    pred_contact = pred_feet[..., 2].min(axis=2) <= float(floor_height) + float(contact_height_threshold)
    gt_contact = gt_feet[..., 2].min(axis=2) <= float(floor_height) + float(contact_height_threshold)
    tp = np.sum(pred_contact & gt_contact)
    fp = np.sum(pred_contact & ~gt_contact)
    fn = np.sum(~pred_contact & gt_contact)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "foot_contact_precision": float(precision),
        "foot_contact_recall": float(recall),
        "foot_contact_f1": float(f1),
        "foot_floating_frac": float(np.mean((~pred_contact) & gt_contact)),
    }


def evaluate(
    sample_dir,
    source_dir,
    contact_threshold,
    floor_height,
    stance_height_threshold,
    stance_speed_threshold,
):
    rows = []
    for sample_path in sorted(glob.glob(os.path.join(sample_dir, "*.pkl"))):
        name = os.path.basename(sample_path)
        source_path = os.path.join(source_dir, name)
        if not os.path.exists(source_path):
            continue

        with open(sample_path, "rb") as f:
            sample = pickle.load(f)
        with open(source_path, "rb") as f:
            source = pickle.load(f)

        if "object_verts" not in source:
            continue

        target_hands = np.asarray(
            sample.get("hands_rectified", sample.get("hand_positions")),
            dtype=np.float32,
        )
        raw_hands = sample.get("hands_raw")
        raw_hands = np.asarray(raw_hands, dtype=np.float32) if raw_hands is not None else target_hands
        T = min(len(target_hands), len(source["hand_positions"]), len(source["object_verts"]))
        target_hands = target_hands[:T]
        raw_hands = raw_hands[:T]
        gt_hands = np.asarray(source["hand_positions"][:T], dtype=np.float32)
        object_verts = np.asarray(source["object_verts"][:T], dtype=np.float32)

        contact = compute_contact_metrics(target_hands, gt_hands, object_verts, contact_threshold)
        surface_dist = nearest_surface_distances(target_hands, object_verts)
        acc = np.diff(target_hands, n=2, axis=0)
        row = {
            "name": name,
            "frames": T,
            "target_hand_jpe_cm": compute_hand_jpe(target_hands, gt_hands),
            "target_contact_f1": contact["f1"],
            "target_contact_precision": contact["precision"],
            "target_contact_recall": contact["recall"],
            "target_surface_mean_cm": float(surface_dist.mean() * 100.0),
            "target_surface_p90_cm": float(np.percentile(surface_dist, 90) * 100.0),
            "constraint_delta_cm": float(np.linalg.norm(target_hands - raw_hands, axis=-1).mean() * 100.0),
            "target_acc_rms_cm": float(np.sqrt((acc ** 2).mean()) * 100.0) if len(target_hands) > 2 else 0.0,
        }

        robot_hands = sample.get("robot_hands")
        if robot_hands is None and all(key in sample for key in ("root_pos", "root_rot", "dof_pos")):
            robot_hands = robot_hand_positions(
                np.asarray(sample["root_pos"][:T], dtype=np.float32),
                np.asarray(sample["root_rot"][:T], dtype=np.float32),
                np.asarray(sample["dof_pos"][:T], dtype=np.float32),
            )
        if robot_hands is not None:
            robot_hands = np.asarray(robot_hands, dtype=np.float32)[:T]
            root_pos = np.asarray(sample["root_pos"][:T], dtype=np.float32)
            root_rot = np.asarray(sample["root_rot"][:T], dtype=np.float32)
            dof_pos = np.asarray(sample["dof_pos"][:T], dtype=np.float32)
            robot_report = compute_robot_contact_report(
                root_pos,
                root_rot,
                dof_pos,
                target_hands=target_hands,
                gt_hands=gt_hands,
                object_verts=object_verts,
                contact_threshold=contact_threshold,
            )
            row.update(robot_report)
            row.update(
                compute_robot_motion_quality_report(
                    root_pos,
                    root_rot,
                    dof_pos,
                    floor_height=floor_height,
                    stance_height_threshold=stance_height_threshold,
                    stance_speed_threshold=stance_speed_threshold,
                )
            )
            if all(key in source for key in ("root_pos", "root_rot", "dof_pos")):
                row.update(
                    compute_foot_contact_report(
                        root_pos,
                        root_rot,
                        dof_pos,
                        np.asarray(source["root_pos"][:T], dtype=np.float32),
                        np.asarray(source["root_rot"][:T], dtype=np.float32),
                        np.asarray(source["dof_pos"][:T], dtype=np.float32),
                        floor_height=floor_height,
                        contact_height_threshold=stance_height_threshold,
                    )
                )
            robot_acc = np.diff(robot_hands, n=2, axis=0)
            row["robot_acc_rms_cm"] = (
                float(np.sqrt((robot_acc ** 2).mean()) * 100.0) if len(robot_hands) > 2 else 0.0
            )

        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate HF-BPS generated samples.")
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--source_dir", default="./data/hf_bps_preprocessed")
    parser.add_argument("--contact_threshold", type=float, default=0.05)
    parser.add_argument("--floor_height", type=float, default=0.0)
    parser.add_argument("--stance_height_threshold", type=float, default=0.06)
    parser.add_argument("--stance_speed_threshold", type=float, default=0.04)
    args = parser.parse_args()

    rows = evaluate(
        args.sample_dir,
        args.source_dir,
        args.contact_threshold,
        args.floor_height,
        args.stance_height_threshold,
        args.stance_speed_threshold,
    )
    if not rows:
        raise RuntimeError(f"No comparable HF-BPS samples found in {args.sample_dir}")

    candidate_keys = [
        "target_hand_jpe_cm",
        "target_contact_f1",
        "target_contact_precision",
        "target_contact_recall",
        "target_surface_mean_cm",
        "target_surface_p90_cm",
        "constraint_delta_cm",
        "target_acc_rms_cm",
        "robot_hand_jpe_cm",
        "robot_to_target_hand_jpe_cm",
        "robot_contact_f1",
        "robot_contact_precision",
        "robot_contact_recall",
        "robot_surface_mean_cm",
        "robot_surface_p90_cm",
        "robot_acc_rms_cm",
        "root_acc_rms_cm",
        "root_jerk_rms_cm",
        "state_acc_rms",
        "state_jerk_rms",
        "dof_acc_rms",
        "dof_jerk_rms",
        "foot_penetration_mean_cm",
        "foot_penetration_max_cm",
        "foot_below_floor_frac",
        "foot_slide_cm",
        "foot_contact_f1",
        "foot_contact_precision",
        "foot_contact_recall",
        "foot_floating_frac",
    ]
    keys = [key for key in candidate_keys if key in rows[0]]
    avg = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    generated_fps = parse_generated_fps(args.sample_dir)

    lines = ["HF-BPS Sample Evaluation", "=" * 50, f"sample_dir: {args.sample_dir}"]
    if generated_fps is not None:
        lines.append(f"generated_fps: {generated_fps:.1f}")
    lines.append(f"num_samples: {len(rows)}")
    for key in keys:
        lines.append(f"{key}: {avg[key]:.6f}")

    lines.append("")
    lines.append("Per-sample:")
    for row in rows:
        parts = [
            f"{row['name']} frames={row['frames']}",
            f"target_jpe_cm={row['target_hand_jpe_cm']:.3f}",
            f"target_f1={row['target_contact_f1']:.3f}",
            f"target_p90_cm={row['target_surface_p90_cm']:.3f}",
        ]
        if "robot_hand_jpe_cm" in row:
            parts.extend(
                [
                    f"robot_jpe_cm={row['robot_hand_jpe_cm']:.3f}",
                    f"robot_f1={row['robot_contact_f1']:.3f}",
                    f"robot_p90_cm={row['robot_surface_p90_cm']:.3f}",
                    f"robot_to_target_cm={row['robot_to_target_hand_jpe_cm']:.3f}",
                ]
            )
        if "root_jerk_rms_cm" in row:
            parts.extend(
                [
                    f"root_jerk_cm={row['root_jerk_rms_cm']:.3f}",
                    f"foot_pen_max_cm={row['foot_penetration_max_cm']:.3f}",
                    f"foot_slide_cm={row['foot_slide_cm']:.3f}",
                ]
            )
        if "foot_contact_f1" in row:
            parts.extend(
                [
                    f"foot_f1={row['foot_contact_f1']:.3f}",
                    f"foot_float={row['foot_floating_frac']:.3f}",
                ]
            )
        lines.append(" ".join(parts))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.sample_dir, "hf_bps_metrics.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
