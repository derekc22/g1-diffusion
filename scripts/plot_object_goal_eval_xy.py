"""Create XY goal/endpoint diagnostics from object-goal evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Any, Iterable

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from matplotlib.colors import Normalize

from utils import eval_plotting

try:
    from PIL import ImageFont
except ImportError:
    ImageFont = None


XY_COLUMNS = (
    "query_goal_x", "query_goal_y",
    "generated_endpoint_x", "generated_endpoint_y",
    "source_final_x", "source_final_y",
    "initial_object_x", "initial_object_y",
    "endpoint_error_x", "endpoint_error_y",
)
CORE_PLOT_COLUMNS = (
    "query_goal_x", "query_goal_y",
    "generated_endpoint_x", "generated_endpoint_y",
    "endpoint_error_x", "endpoint_error_y",
    "final_object_position_error",
    "final_object_rotation_error_deg",
)
MODE_DISPLAY_NAMES = {
    "A": "A: Single motion",
    "B": "B: Interpolated motion",
    "C": "C: OOD goal-radius sweep",
    "D": "D: Initial-condition sweep",
}
LEGACY_MODE_LABELS = {
    "ood_radius_sweep": "C",
    "init_sweep": "D",
}
FIGURE_SIZE = (8.5, 8.5)
ERROR_CMAP = "plasma"
TARGET_LABEL = "Target goal"
ENDPOINT_LABEL = "Generated endpoint"
INITIAL_LABEL = "Initial object position"
ERROR_VECTOR_LABEL = "Error vector"


def _read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_rows(rows: Iterable[dict[str, str]], columns: Iterable[str]) -> list[dict[str, str]]:
    required = tuple(columns)
    return [row for row in rows if all(np.isfinite(_number(row, key)) for key in required)]


def _arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    result = {
        key: np.asarray([_number(row, key) for row in rows], dtype=np.float64)
        for key in XY_COLUMNS
    }
    result["error"] = np.asarray(
        [_number(row, "final_object_position_error") for row in rows], dtype=np.float64
    )
    result["rotation_error"] = np.asarray(
        [_number(row, "final_object_rotation_error_deg") for row in rows], dtype=np.float64
    )
    result["success"] = np.asarray([_truth(row.get("success")) for row in rows], dtype=bool)
    result["radius"] = np.asarray([_number(row, "radius_m") for row in rows], dtype=np.float64)
    return result


def _mode_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unknown"


def _canonical_mode(value: str) -> str:
    return LEGACY_MODE_LABELS.get(value, value)


def _display_mode(value: str) -> str:
    return MODE_DISPLAY_NAMES.get(value, value)


def _require_matplotlib() -> Any:
    if eval_plotting.plt is None:
        raise RuntimeError("Matplotlib is required for presentation-quality XY evaluation plots")
    return eval_plotting.plt


def _save_figure(
    figure: Any,
    axis: Any,
    path: str,
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
    subtitle: str | None = None,
) -> None:
    axis.set_title(title if subtitle is None else f"{title}\n{subtitle}", fontsize=14, pad=12)
    axis.set_xlabel("Object X position (m)", fontsize=12)
    axis.set_ylabel("Object Y position (m)", fontsize=12)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="best", frameon=True)
    figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    eval_plotting.plt.close(figure)


def _floor_map_limits(
    values: dict[str, np.ndarray],
    *,
    include_initial: bool = False,
    include_source: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_parts = [values["query_goal_x"], values["generated_endpoint_x"]]
    y_parts = [values["query_goal_y"], values["generated_endpoint_y"]]
    if include_initial:
        x_parts.append(values["initial_object_x"])
        y_parts.append(values["initial_object_y"])
    if include_source:
        x_parts.append(values["source_final_x"])
        y_parts.append(values["source_final_y"])
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    x_low, x_high = float(x.min()), float(x.max())
    y_low, y_high = float(y.min()), float(y.max())
    span = max(x_high - x_low, y_high - y_low, 0.10) * 1.12
    x_mid = (x_low + x_high) / 2.0
    y_mid = (y_low + y_high) / 2.0
    half = span / 2.0
    return (x_mid - half, x_mid + half), (y_mid - half, y_mid + half)


def _error_norm(values: dict[str, np.ndarray], key: str = "error") -> Normalize:
    low, high = _error_limits(values[key])
    return Normalize(vmin=low, vmax=high)


def _segments(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nan = np.full_like(x0, np.nan)
    return np.column_stack((x0, x1, nan)).reshape(-1), np.column_stack((y0, y1, nan)).reshape(-1)


def _error_limits(errors: np.ndarray) -> tuple[float, float]:
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    return (low, high) if high > low else (low, low + max(abs(low) * 0.01, 1e-6))


def _heat_color(value: float, low: float, high: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return 130, 130, 130
    ratio = min(max((value - low) / max(high - low, 1e-12), 0.0), 1.0)
    if ratio <= 0.5:
        blend = ratio * 2.0
        return int(40 + 215 * blend), int(90 + 145 * blend), int(210 - 170 * blend)
    blend = (ratio - 0.5) * 2.0
    return 255, int(235 - 185 * blend), int(40 - 20 * blend)


def _font(size: int) -> Any:
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


class _FallbackCanvas:
    def __init__(self, x_values: Iterable[float], y_values: Iterable[float]):
        self.width, self.height = 1400, 1000
        self.left, self.right, self.top, self.bottom = 130, 360, 105, 125
        self.image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        self.x_min, self.x_max = eval_plotting._bounds(list(x_values))
        self.y_min, self.y_max = eval_plotting._bounds(list(y_values))
        plot_width = self.width - self.left - self.right
        plot_height = self.height - self.top - self.bottom
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        units_per_pixel = max(x_range / plot_width, y_range / plot_height)
        x_mid = (self.x_min + self.x_max) / 2.0
        y_mid = (self.y_min + self.y_max) / 2.0
        self.x_min, self.x_max = x_mid - units_per_pixel * plot_width / 2.0, x_mid + units_per_pixel * plot_width / 2.0
        self.y_min, self.y_max = y_mid - units_per_pixel * plot_height / 2.0, y_mid + units_per_pixel * plot_height / 2.0
        for index in range(6):
            px = self.left + round((self.width - self.left - self.right) * index / 5)
            py = self.top + round((self.height - self.top - self.bottom) * index / 5)
            eval_plotting._draw_line(self.image, (px, self.top), (px, self.height - self.bottom), (230, 230, 230))
            eval_plotting._draw_line(self.image, (self.left, py), (self.width - self.right, py), (230, 230, 230))

    def pixel(self, x_value: float, y_value: float) -> tuple[int, int]:
        plot_width = self.width - self.left - self.right
        plot_height = self.height - self.top - self.bottom
        px = self.left + round(plot_width * (x_value - self.x_min) / max(self.x_max - self.x_min, 1e-12))
        py = self.top + round(plot_height * (self.y_max - y_value) / max(self.y_max - self.y_min, 1e-12))
        return px, py

    def line(self, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int], width: int = 1) -> None:
        eval_plotting._draw_line(self.image, self.pixel(x0, y0), self.pixel(x1, y1), color, width)

    def marker(self, x: float, y: float, color: tuple[int, int, int], kind: str = "square", size: int = 4) -> None:
        px, py = self.pixel(x, y)
        if kind == "x":
            eval_plotting._draw_line(self.image, (px - size, py - size), (px + size, py + size), color, 2)
            eval_plotting._draw_line(self.image, (px - size, py + size), (px + size, py - size), color, 2)
        elif kind == "dot":
            self.image[max(0, py - size):min(self.height, py + size + 1), max(0, px - size):min(self.width, px + size + 1)] = color
        else:
            eval_plotting._draw_line(self.image, (px - size, py - size), (px + size, py - size), color, 2)
            eval_plotting._draw_line(self.image, (px + size, py - size), (px + size, py + size), color, 2)
            eval_plotting._draw_line(self.image, (px + size, py + size), (px - size, py + size), color, 2)
            eval_plotting._draw_line(self.image, (px - size, py + size), (px - size, py - size), color, 2)

    def cell(self, x0: float, x1: float, y0: float, y1: float, color: tuple[int, int, int]) -> None:
        px0, py1 = self.pixel(x0, y0)
        px1, py0 = self.pixel(x1, y1)
        self.image[max(self.top, py0):min(self.height - self.bottom, py1 + 1), max(self.left, px0):min(self.width - self.right, px1 + 1)] = color

    def save(
        self,
        path: str,
        title: str,
        note: str = "",
        legend: str = "error",
        colorbar: tuple[float, float, str] | None = None,
    ) -> None:
        eval_plotting._draw_line(self.image, (self.left, self.top), (self.left, self.height - self.bottom), (0, 0, 0), 2)
        eval_plotting._draw_line(self.image, (self.left, self.height - self.bottom), (self.width - self.right, self.height - self.bottom), (0, 0, 0), 2)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if eval_plotting.Image is not None and eval_plotting.ImageDraw is not None:
            image = eval_plotting.Image.fromarray(self.image)
            draw = eval_plotting.ImageDraw.Draw(image)
            title_font = _font(28)
            label_font = _font(20)
            text_font = _font(17)
            small_font = _font(15)
            draw.text((self.left, 24), title, fill=(0, 0, 0), font=title_font)
            draw.text((self.left + 320, self.height - 55), "Object X position (m)", fill=(0, 0, 0), font=label_font)
            draw.text((12, self.top + 330), "Object Y position (m)", fill=(0, 0, 0), font=label_font)
            if note:
                draw.text((self.left, 63), note, fill=(75, 75, 75), font=small_font)
            for index in range(6):
                x_value = self.x_min + (self.x_max - self.x_min) * index / 5.0
                y_value = self.y_max - (self.y_max - self.y_min) * index / 5.0
                px = self.left + round((self.width - self.left - self.right) * index / 5)
                py = self.top + round((self.height - self.top - self.bottom) * index / 5)
                draw.text((px - 28, self.height - self.bottom + 12), f"{x_value:.2f}", fill=(45, 45, 45), font=small_font)
                draw.text((self.left - 67, py - 10), f"{y_value:.2f}", fill=(45, 45, 45), font=small_font)

            legend_x, legend_y = self.width - self.right + 45, self.top + 10
            if legend in {"error", "init", "ood"}:
                draw.rectangle((legend_x, legend_y, legend_x + 14, legend_y + 14), outline=(35, 35, 35), width=2)
                draw.text((legend_x + 25, legend_y - 4), "Query/target goal", fill=(0, 0, 0), font=text_font)
                draw.line((legend_x, legend_y + 42, legend_x + 14, legend_y + 56), fill=(160, 80, 160), width=2)
                draw.line((legend_x, legend_y + 56, legend_x + 14, legend_y + 42), fill=(160, 80, 160), width=2)
                draw.text((legend_x + 25, legend_y + 37), "Generated endpoint", fill=(0, 0, 0), font=text_font)
                draw.line((legend_x, legend_y + 84, legend_x + 18, legend_y + 84), fill=(185, 185, 185), width=2)
                draw.text((legend_x + 25, legend_y + 73), "Endpoint error vector", fill=(0, 0, 0), font=text_font)
            if legend == "success":
                draw.rectangle((legend_x, legend_y, legend_x + 14, legend_y + 14), outline=(40, 40, 40), width=2)
                draw.text((legend_x + 25, legend_y - 4), "Query/target goal", fill=(0, 0, 0), font=text_font)
                draw.line((legend_x, legend_y + 40, legend_x + 14, legend_y + 54), fill=(60, 60, 60), width=2)
                draw.line((legend_x, legend_y + 54, legend_x + 14, legend_y + 40), fill=(60, 60, 60), width=2)
                draw.text((legend_x + 25, legend_y + 35), "Generated endpoint", fill=(0, 0, 0), font=text_font)
                draw.line((legend_x, legend_y + 82, legend_x + 18, legend_y + 82), fill=(185, 185, 185), width=2)
                draw.text((legend_x + 25, legend_y + 71), "Endpoint error vector", fill=(0, 0, 0), font=text_font)
                draw.rectangle((legend_x, legend_y + 112, legend_x + 14, legend_y + 126), fill=(35, 165, 75))
                draw.text((legend_x + 25, legend_y + 107), "Success", fill=(0, 0, 0), font=text_font)
                draw.rectangle((legend_x, legend_y + 146, legend_x + 14, legend_y + 160), fill=(215, 50, 50))
                draw.text((legend_x + 25, legend_y + 141), "Failure", fill=(0, 0, 0), font=text_font)
            if legend == "init":
                draw.ellipse((legend_x, legend_y + 112, legend_x + 14, legend_y + 126), fill=(50, 120, 210))
                draw.text((legend_x + 25, legend_y + 107), "Initial object position", fill=(0, 0, 0), font=text_font)
            if legend == "ood":
                draw.ellipse((legend_x, legend_y + 112, legend_x + 14, legend_y + 126), fill=(60, 60, 60))
                draw.text((legend_x + 25, legend_y + 107), "Source final position", fill=(0, 0, 0), font=text_font)

            if colorbar is not None:
                low, high, label = colorbar
                bar_x, bar_y, bar_w, bar_h = legend_x, self.top + 310, 30, 330
                for offset in range(bar_h):
                    value = high - (high - low) * offset / max(bar_h - 1, 1)
                    draw.line((bar_x, bar_y + offset, bar_x + bar_w, bar_y + offset), fill=_heat_color(value, low, high))
                draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline=(40, 40, 40), width=1)
                draw.text((bar_x + 44, bar_y - 8), f"{high:.3f}", fill=(0, 0, 0), font=small_font)
                draw.text((bar_x + 44, bar_y + bar_h // 2 - 8), f"{(low + high) / 2.0:.3f}", fill=(0, 0, 0), font=small_font)
                draw.text((bar_x + 44, bar_y + bar_h - 16), f"{low:.3f}", fill=(0, 0, 0), font=small_font)
                draw.text((bar_x, bar_y + bar_h + 16), label, fill=(0, 0, 0), font=text_font)
            image.save(path, format="PDF", resolution=150.0)
        else:
            raise RuntimeError("PDF XY plotting requires Pillow when Matplotlib is unavailable")


def _fallback_goal_endpoint(path: str, values: dict[str, np.ndarray], title: str) -> None:
    x = np.concatenate((values["query_goal_x"], values["generated_endpoint_x"]))
    y = np.concatenate((values["query_goal_y"], values["generated_endpoint_y"]))
    canvas = _FallbackCanvas(x, y)
    low, high = _error_limits(values["error"])
    for gx, gy, ex, ey, error in zip(values["query_goal_x"], values["query_goal_y"], values["generated_endpoint_x"], values["generated_endpoint_y"], values["error"]):
        canvas.line(gx, gy, ex, ey, (205, 205, 205))
        canvas.marker(gx, gy, (35, 35, 35), "square")
        canvas.marker(ex, ey, _heat_color(error, low, high), "x", 6)
    canvas.save(
        path,
        title,
        "Square = query/target goal; X = generated endpoint; line = endpoint error vector",
        colorbar=(low, high, "Final position error (m)"),
    )


def _fallback_success(path: str, values: dict[str, np.ndarray], title: str) -> None:
    x = np.concatenate((values["query_goal_x"], values["generated_endpoint_x"]))
    y = np.concatenate((values["query_goal_y"], values["generated_endpoint_y"]))
    canvas = _FallbackCanvas(x, y)
    for gx, gy, ex, ey, success in zip(values["query_goal_x"], values["query_goal_y"], values["generated_endpoint_x"], values["generated_endpoint_y"], values["success"]):
        color = (35, 165, 75) if success else (215, 50, 50)
        canvas.line(gx, gy, ex, ey, (205, 205, 205))
        canvas.marker(gx, gy, (45, 45, 45), "square", 5)
        canvas.marker(ex, ey, color, "x", 6)
    canvas.save(path, title, "Generated endpoints: green = success; red = failure", legend="success")


def _plot_mode(mode: str, rows: list[dict[str, str]], plot_root: str) -> list[str]:
    rows = _finite_rows(rows, CORE_PLOT_COLUMNS)
    if not rows:
        print(f"Skipping XY plots for {mode}: required coordinate columns are absent or non-finite")
        return []
    values = _arrays(rows)
    safe_mode = _mode_name(mode)
    display_mode = _display_mode(mode)
    paths = {
        "goal_position": os.path.join(plot_root, f"{safe_mode}_goal_vs_endpoint_position_error.pdf"),
        "goal_rotation": os.path.join(plot_root, f"{safe_mode}_goal_vs_endpoint_rotation_error.pdf"),
        "success": os.path.join(plot_root, f"{safe_mode}_success_scatter.pdf"),
    }

    plt = _require_matplotlib()
    limits = _floor_map_limits(
        values,
        include_initial=mode == "D",
        include_source=mode == "C",
    )
    line_x, line_y = _segments(values["query_goal_x"], values["query_goal_y"], values["generated_endpoint_x"], values["generated_endpoint_y"])
    for metric_key, path_key, title_suffix, colorbar_label in (
        ("error", "goal_position", "Position error", "Final object position error (m)"),
        ("rotation_error", "goal_rotation", "Rotation error", "Final object rotation error (deg)"),
    ):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.plot(line_x, line_y, color="0.72", alpha=0.32, linewidth=0.65, label=ERROR_VECTOR_LABEL, zorder=1)
        axis.scatter(values["query_goal_x"], values["query_goal_y"], marker="s", s=42, facecolors="none", edgecolors="black", linewidths=1.2, label=TARGET_LABEL, zorder=3)
        endpoints = axis.scatter(
            values["generated_endpoint_x"], values["generated_endpoint_y"],
            marker="x", s=46, linewidths=1.4, c=values[metric_key],
            cmap=ERROR_CMAP, norm=_error_norm(values, metric_key), label=ENDPOINT_LABEL, zorder=4,
        )
        figure.colorbar(endpoints, ax=axis, label=colorbar_label)
        _save_figure(figure, axis, paths[path_key], f"{display_mode}: {title_suffix}", limits)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(line_x, line_y, color="0.72", alpha=0.32, linewidth=0.65, label=ERROR_VECTOR_LABEL, zorder=1)
    axis.scatter(values["query_goal_x"], values["query_goal_y"], marker="s", s=42, facecolors="none", edgecolors="black", linewidths=1.2, label=TARGET_LABEL, zorder=3)
    success = values["success"]
    if np.any(success):
        axis.scatter(values["generated_endpoint_x"][success], values["generated_endpoint_y"][success], color="tab:green", marker="x", s=46, linewidths=1.4, label="Success", zorder=4)
    if np.any(~success):
        axis.scatter(values["generated_endpoint_x"][~success], values["generated_endpoint_y"][~success], color="tab:red", marker="x", s=46, linewidths=1.4, label="Failure", zorder=4)
    _save_figure(
        figure,
        axis,
        paths["success"],
        f"{display_mode} — Success and failure",
        limits,
        subtitle="Success: object ≤10 cm, ≤20°",
    )

    return list(paths.values())


def _plot_init_sweep(rows: list[dict[str, str]], eval_dir: str) -> list[str]:
    rows = _finite_rows(rows, CORE_PLOT_COLUMNS + ("initial_object_x", "initial_object_y"))
    if not rows:
        return []
    values = _arrays(rows)
    paths = {
        "position": os.path.join(eval_dir, "init_sweep", "plots", "initial_states_and_endpoints_xy_position_error.pdf"),
        "rotation": os.path.join(eval_dir, "init_sweep", "plots", "initial_states_and_endpoints_xy_rotation_error.pdf"),
    }
    plt = _require_matplotlib()
    limits = _floor_map_limits(values, include_initial=True)
    error_x, error_y = _segments(values["query_goal_x"], values["query_goal_y"], values["generated_endpoint_x"], values["generated_endpoint_y"])
    for metric_key, path_key, title_suffix, colorbar_label in (
        ("error", "position", "Position error", "Final object position error (m)"),
        ("rotation_error", "rotation", "Rotation error", "Final object rotation error (deg)"),
    ):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.plot(error_x, error_y, color="0.72", alpha=0.32, linewidth=0.65, label=ERROR_VECTOR_LABEL, zorder=1)
        axis.scatter(values["initial_object_x"], values["initial_object_y"], marker="o", s=18, color="tab:blue", label=INITIAL_LABEL, zorder=2)
        axis.scatter(values["query_goal_x"], values["query_goal_y"], marker="s", s=42, facecolors="none", edgecolors="black", linewidths=1.2, label=TARGET_LABEL, zorder=3)
        endpoints = axis.scatter(
            values["generated_endpoint_x"], values["generated_endpoint_y"],
            marker="x", s=46, linewidths=1.4, c=values[metric_key],
            cmap=ERROR_CMAP, norm=_error_norm(values, metric_key), label=ENDPOINT_LABEL, zorder=4,
        )
        figure.colorbar(endpoints, ax=axis, label=colorbar_label)
        _save_figure(
            figure, axis, paths[path_key],
            f"D: Initial-condition sweep — Initial positions, goals, and endpoints: {title_suffix}", limits,
        )
    return list(paths.values())


def _plot_ood_sweep(rows: list[dict[str, str]], eval_dir: str) -> list[str]:
    rows = _finite_rows(
        rows,
        CORE_PLOT_COLUMNS + ("source_final_x", "source_final_y", "radius_m"),
    )
    if not rows:
        return []
    values = _arrays(rows)
    paths = {
        "position": os.path.join(eval_dir, "ood_radius_sweep", "plots", "xy_radius_goal_endpoint_position_error.pdf"),
        "rotation": os.path.join(eval_dir, "ood_radius_sweep", "plots", "xy_radius_goal_endpoint_rotation_error.pdf"),
    }
    plt = _require_matplotlib()
    limits = _floor_map_limits(values, include_source=True)
    error_x, error_y = _segments(values["query_goal_x"], values["query_goal_y"], values["generated_endpoint_x"], values["generated_endpoint_y"])
    for metric_key, path_key, title_suffix, colorbar_label in (
        ("error", "position", "Position error", "Final object position error (m)"),
        ("rotation_error", "rotation", "Rotation error", "Final object rotation error (deg)"),
    ):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        source_x, source_y = _segments(
            values["source_final_x"], values["source_final_y"],
            values["query_goal_x"], values["query_goal_y"],
        )
        axis.plot(
            source_x, source_y, color="0.5", alpha=0.5, linewidth=0.9,
            label="Random XY goal perturbation",
        )
        axis.plot(error_x, error_y, color="0.72", alpha=0.32, linewidth=0.65, label=ERROR_VECTOR_LABEL, zorder=1)
        axis.scatter(values["source_final_x"], values["source_final_y"], marker="o", s=16, color="0.25", alpha=0.65, label="Source final position", zorder=2)
        axis.scatter(values["query_goal_x"], values["query_goal_y"], marker="s", s=42, facecolors="none", edgecolors="black", linewidths=1.2, label=TARGET_LABEL, zorder=3)
        endpoints = axis.scatter(
            values["generated_endpoint_x"], values["generated_endpoint_y"],
            marker="x", s=46, linewidths=1.4, c=values[metric_key],
            cmap=ERROR_CMAP, norm=_error_norm(values, metric_key), label=ENDPOINT_LABEL, zorder=4,
        )
        figure.colorbar(endpoints, ax=axis, label=colorbar_label)
        _save_figure(
            figure, axis, paths[path_key],
            f"C: OOD goal-radius sweep — Source, goals, and endpoints: {title_suffix}", limits,
        )
    return list(paths.values())


def generate_xy_plots(eval_dir: str) -> list[str]:
    """Generate all available XY diagnostic plots and return their paths."""
    eval_dir = os.path.abspath(eval_dir)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    root_rows = _read_csv(os.path.join(eval_dir, "metrics.csv"))
    if not root_rows:
        for path in sorted(glob.glob(os.path.join(eval_dir, "quantitative", "*", "metrics.csv"))):
            root_rows.extend(_read_csv(path))
    for row in root_rows:
        mode = _canonical_mode(str(row.get("mode") or "standard"))
        grouped[mode].append(row)

    init_rows = _read_csv(os.path.join(eval_dir, "init_sweep", "metrics.csv"))
    ood_rows = _read_csv(os.path.join(eval_dir, "ood_radius_sweep", "metrics.csv"))
    for row in init_rows:
        mode = _canonical_mode(str(row.get("mode") or "D"))
        grouped[mode].append(row)
    for row in ood_rows:
        mode = _canonical_mode(str(row.get("mode") or "C"))
        grouped[mode].append(row)

    generated: list[str] = []
    plot_root = os.path.join(eval_dir, "plots", "xy")
    for mode, rows in sorted(grouped.items()):
        generated.extend(_plot_mode(mode, rows, plot_root))

    generated.extend(_plot_init_sweep(init_rows, eval_dir))
    generated.extend(_plot_ood_sweep(ood_rows, eval_dir))

    if not generated:
        print(f"No XY plots generated from {eval_dir}: no CSV contained all required XY columns")
    else:
        print(f"Generated {len(generated)} XY diagnostic plots under {eval_dir}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot object-goal query and generated endpoint diagnostics in XY")
    parser.add_argument("eval_dir", nargs="?", help="Evaluation output directory")
    parser.add_argument("--eval_dir", dest="eval_dir_option", help="Evaluation output directory")
    args = parser.parse_args()
    eval_dir = args.eval_dir_option or args.eval_dir
    if not eval_dir:
        parser.error("an evaluation output directory is required")
    for path in generate_xy_plots(eval_dir):
        print(path)


if __name__ == "__main__":
    main()
