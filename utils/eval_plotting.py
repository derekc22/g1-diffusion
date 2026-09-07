"""Small evaluation plotting helpers with no mandatory plotting dependency."""

from __future__ import annotations

import os
import struct
import zlib
from typing import Mapping, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


COLORS = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (148, 103, 189),
    (255, 127, 14),
]


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 1.0
    low, high = float(array.min()), float(array.max())
    if high <= low:
        pad = max(abs(low) * 0.05, 1e-6)
    else:
        pad = (high - low) * 0.08
    return low - pad, high + pad


def _draw_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        radius = max(width // 2, 0)
        image[
            max(0, y0 - radius) : min(image.shape[0], y0 + radius + 1),
            max(0, x0 - radius) : min(image.shape[1], x0 + radius + 1),
        ] = color
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _write_raw_png(image: np.ndarray, path: str) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + row.tobytes() for row in image)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as file:
        file.write(png)


def _fallback_plot(
    path: str,
    x_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    kind: str,
    x_tick_labels: Sequence[str] | None,
    vertical_lines: Mapping[str, float] | None,
    horizontal_lines: Mapping[str, float] | None,
) -> None:
    width, height = 1000, 600
    left, right, top, bottom = 90, 30, 55, 75
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = np.asarray(x_values, dtype=np.float64)
    y_all = [float(value) for values in series.values() for value in values]
    if vertical_lines:
        x = np.append(x, list(vertical_lines.values()))
    if horizontal_lines:
        y_all.extend(horizontal_lines.values())
    x_min, x_max = _bounds(x)
    y_min, y_max = _bounds(y_all)
    plot_width, plot_height = width - left - right, height - top - bottom

    def point(x_value: float, y_value: float) -> tuple[int, int]:
        px = left + round(plot_width * (x_value - x_min) / max(x_max - x_min, 1e-12))
        py = top + round(plot_height * (y_max - y_value) / max(y_max - y_min, 1e-12))
        return px, py

    for grid_index in range(6):
        y_pixel = top + round(plot_height * grid_index / 5)
        _draw_line(canvas, (left, y_pixel), (width - right, y_pixel), (225, 225, 225))
    _draw_line(canvas, (left, top), (left, height - bottom), (0, 0, 0), 2)
    _draw_line(canvas, (left, height - bottom), (width - right, height - bottom), (0, 0, 0), 2)

    if vertical_lines:
        for value in vertical_lines.values():
            px, _ = point(float(value), y_min)
            _draw_line(canvas, (px, top), (px, height - bottom), (120, 120, 120), 2)
    if horizontal_lines:
        for value in horizontal_lines.values():
            _, py = point(x_min, float(value))
            _draw_line(canvas, (left, py), (width - right, py), (120, 120, 120), 2)

    for series_index, values in enumerate(series.values()):
        color = COLORS[series_index % len(COLORS)]
        points = [point(float(xv), float(yv)) for xv, yv in zip(x_values, values)]
        if kind == "bar":
            baseline = point(x_min, max(0.0, y_min))[1]
            half_width = max(2, plot_width // max(3 * len(points), 1))
            for px, py in points:
                canvas[min(py, baseline) : max(py, baseline) + 1, max(left, px - half_width) : min(width - right, px + half_width + 1)] = color
        else:
            if kind == "line":
                for previous, current in zip(points, points[1:]):
                    _draw_line(canvas, previous, current, color, 3)
            for px, py in points:
                canvas[max(0, py - 3) : py + 4, max(0, px - 3) : px + 4] = color

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if Image is not None and ImageDraw is not None:
        pil_image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_image)
        draw.text((left, 15), title, fill=(0, 0, 0))
        draw.text((left, height - 25), xlabel, fill=(0, 0, 0))
        draw.text((8, top), ylabel, fill=(0, 0, 0))
        draw.text((left, height - bottom + 8), f"{x_min:.3g}", fill=(0, 0, 0))
        draw.text((width - right - 50, height - bottom + 8), f"{x_max:.3g}", fill=(0, 0, 0))
        draw.text((15, top), f"{y_max:.3g}", fill=(0, 0, 0))
        draw.text((15, height - bottom - 10), f"{y_min:.3g}", fill=(0, 0, 0))
        if x_tick_labels is not None:
            for x_value, label in zip(x_values, x_tick_labels):
                px, _ = point(float(x_value), y_min)
                draw.text((px - 20, height - bottom + 24), str(label), fill=(0, 0, 0))
        legend_x = max(left, width - right - 220)
        for index, name in enumerate(series):
            color = COLORS[index % len(COLORS)]
            y_pos = top + index * 17
            draw.rectangle((legend_x, y_pos, legend_x + 12, y_pos + 8), fill=color)
            draw.text((legend_x + 18, y_pos - 2), name, fill=(0, 0, 0))
        pil_image.save(path)
    else:
        _write_raw_png(canvas, path)


def save_plot(
    path: str,
    x_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    kind: str = "line",
    x_tick_labels: Sequence[str] | None = None,
    vertical_lines: Mapping[str, float] | None = None,
    horizontal_lines: Mapping[str, float] | None = None,
) -> None:
    """Save a line, scatter, or bar chart with optional threshold markers."""
    if kind not in {"line", "scatter", "bar"}:
        raise ValueError(f"Unknown plot kind {kind!r}")
    if not series:
        raise ValueError("At least one plot series is required")
    count = len(x_values)
    if any(len(values) != count for values in series.values()):
        raise ValueError("All plot series must match x_values")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if plt is None:
        _fallback_plot(
            path, x_values, series, title, xlabel, ylabel, kind,
            x_tick_labels, vertical_lines, horizontal_lines,
        )
        return

    figure, axis = plt.subplots(figsize=(10, 6))
    x = np.asarray(x_values, dtype=np.float64)
    for index, (name, values) in enumerate(series.items()):
        color = np.asarray(COLORS[index % len(COLORS)]) / 255.0
        if kind == "bar":
            width = 0.8 / max(len(series), 1)
            offset = (index - (len(series) - 1) / 2.0) * width
            axis.bar(x + offset, values, width=width, label=name, color=color)
        elif kind == "scatter":
            axis.scatter(x, values, label=name, color=color, alpha=0.75)
        else:
            axis.plot(x, values, marker="o", label=name, color=color)
    for name, value in (vertical_lines or {}).items():
        axis.axvline(value, linestyle="--", alpha=0.7, label=name)
    for name, value in (horizontal_lines or {}).items():
        axis.axhline(value, linestyle="--", alpha=0.7, label=name)
    if x_tick_labels is not None:
        axis.set_xticks(x, x_tick_labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
