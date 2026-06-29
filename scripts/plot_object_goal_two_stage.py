"""Quicklook plots for object-goal two-stage sample PKLs."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _as_array(data: dict[str, Any], key: str) -> np.ndarray | None:
    value = data.get(key)
    if value is None:
        return None
    return np.asarray(value)


def _sample_files(sample_folder: str) -> list[Path]:
    root = Path(sample_folder)
    files = sorted(root.glob("*.pkl"))
    if files:
        return files
    return sorted(root.glob("*/*.pkl"))


def _object_position_and_rot6d(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    object_pose = _as_array(data, "object_pose")
    if object_pose is not None:
        return object_pose[:, :3], object_pose[:, 3:9]
    object_pos = _as_array(data, "object_pos")
    if object_pos is not None:
        return object_pos, None
    object_centroid = _as_array(data, "object_centroid")
    if object_centroid is not None:
        return object_centroid, None
    raise KeyError("sample missing object_pose/object_pos/object_centroid")


def _plot_sample(path: Path, save_dir: Path) -> str:
    with path.open("rb") as f:
        data = pickle.load(f)

    object_pos, object_rot6d = _object_position_and_rot6d(data)
    root_pos = _as_array(data, "root_pos")
    hands_raw = _as_array(data, "hands_raw")
    hands_rect = _as_array(data, "hands_rectified")
    goal = _as_array(data, "goal")
    goal_pos = goal[:3] if goal is not None and goal.shape[0] >= 3 else object_pos[-1]
    fps = float(data.get("fps", 30.0))
    t = np.arange(object_pos.shape[0]) / fps

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.reshape(-1)

    axes[0].plot(object_pos[:, 0], object_pos[:, 1], label="object")
    axes[0].scatter([object_pos[0, 0]], [object_pos[0, 1]], s=25, label="start")
    axes[0].scatter([goal_pos[0]], [goal_pos[1]], s=35, label="goal")
    axes[0].set_title("Object XY")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, object_pos[:, 2], label="object z")
    axes[1].axhline(goal_pos[2], color="tab:red", linestyle="--", label="goal z")
    axes[1].set_title("Object Z")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if root_pos is not None:
        axes[2].plot(root_pos[:, 0], root_pos[:, 1], label="root")
        axes[2].set_title("Root XY")
        axes[2].set_xlabel("x [m]")
        axes[2].set_ylabel("y [m]")
        axes[2].axis("equal")
        axes[2].grid(alpha=0.3)
    else:
        axes[2].set_title("Root XY unavailable")

    dist_to_goal = np.linalg.norm(object_pos[:, :3] - goal_pos.reshape(1, 3), axis=-1)
    axes[3].plot(t, dist_to_goal)
    axes[3].set_title("Object Distance To Goal")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("m")
    axes[3].grid(alpha=0.3)

    if hands_raw is not None and hands_rect is not None:
        min_len = min(hands_raw.shape[0], hands_rect.shape[0])
        corr_l = np.linalg.norm(hands_rect[:min_len, :3] - hands_raw[:min_len, :3], axis=-1)
        corr_r = np.linalg.norm(hands_rect[:min_len, 3:6] - hands_raw[:min_len, 3:6], axis=-1)
        axes[4].plot(t[:min_len], corr_l, label="left")
        axes[4].plot(t[:min_len], corr_r, label="right")
        axes[4].set_title("Hand Rectification Magnitude")
        axes[4].set_xlabel("time [s]")
        axes[4].set_ylabel("m")
        axes[4].grid(alpha=0.3)
        axes[4].legend()
    else:
        axes[4].set_title("Hand rectification unavailable")

    if object_rot6d is not None:
        for dim in range(object_rot6d.shape[1]):
            axes[5].plot(t, object_rot6d[:, dim], linewidth=1, label=f"r{dim}")
        axes[5].set_title("Object Rot6D")
        axes[5].set_xlabel("time [s]")
        axes[5].grid(alpha=0.3)
    else:
        axes[5].set_title("Object Rot6D unavailable")

    seq_name = data.get("seq_name") or path.stem
    fig.suptitle(str(seq_name))
    fig.tight_layout()
    out_path = save_dir / f"{path.stem}_quicklook.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot object-goal two-stage quicklooks")
    parser.add_argument("--sample_folder", default="./out/object_goal_two_stage")
    parser.add_argument("--save_dir", default="./figures/object_goal_two_stage")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    files = _sample_files(args.sample_folder)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No sample PKLs found in {args.sample_folder}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs = [_plot_sample(path, save_dir) for path in files]

    summary_path = save_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("Object-goal two-stage quicklook plots\n")
        f.write(f"sample_folder: {args.sample_folder}\n")
        f.write(f"num_files: {len(files)}\n")
        for output in outputs:
            f.write(f"{output}\n")

    print(f"Wrote {len(outputs)} plot(s) to {save_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
