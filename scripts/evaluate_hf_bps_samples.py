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
from utils.robot_kinematics import compute_robot_contact_report, robot_hand_positions


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


def evaluate(sample_dir, source_dir, contact_threshold):
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
            robot_report = compute_robot_contact_report(
                np.asarray(sample["root_pos"][:T], dtype=np.float32),
                np.asarray(sample["root_rot"][:T], dtype=np.float32),
                np.asarray(sample["dof_pos"][:T], dtype=np.float32),
                target_hands=target_hands,
                gt_hands=gt_hands,
                object_verts=object_verts,
                contact_threshold=contact_threshold,
            )
            row.update(robot_report)
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
    args = parser.parse_args()

    rows = evaluate(args.sample_dir, args.source_dir, args.contact_threshold)
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
        lines.append(" ".join(parts))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.sample_dir, "hf_bps_metrics.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
