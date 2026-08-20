"""Train direct [robot, object] diffusion conditioned on initial state and object goal.

The only denoiser inputs are the noised 47D state, diffusion timestep, and a
56D global condition: [final_object(9), initial_robot(38), initial_object(9)].
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import zlib
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from datasets.hf_motion_dataset import HFFullBodyDataset
from models.stage2_diffusion import Stage2MLPModel, Stage2TransformerModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.general import dump_config, load_config
from utils.motion_losses import denormalize, format_metrics, loss_config, temporal_reconstruction_loss
from utils.object_goal_features import OBJECT_POSE_DIM, ROBOT_OBJECT_STATE_DIM, ROBOT_STATE_DIM, robot_object_layout
from utils.object_sampling import build_balanced_sampler, format_label_counts


MODEL_TYPE = "object_goal_single_stage_init_goal"
PREDICTION_TYPE = "x0"
GLOBAL_COND_DIM = 56
GLOBAL_COND_LAYOUT = {
    "final_object_frame": [0, 9],
    "initial_robot_frame": [9, 47],
    "initial_object_frame": [47, 56],
}


def _draw_line(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    """Draw a small Bresenham line for the no-dependency PNG fallback."""
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


def _save_simple_loss_png(losses: list[float], path: str) -> None:
    """Write a valid loss-curve PNG when Matplotlib is not installed."""
    width, height = 1000, 600
    left, right, top, bottom = 80, 30, 40, 70
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = np.asarray(losses, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        values = np.where(finite, values, np.nan)
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    if y_max <= y_min:
        padding = max(abs(y_min) * 0.05, 1e-6)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.05
        y_min -= padding
        y_max += padding

    for grid_index in range(6):
        y = top + round(plot_height * grid_index / 5)
        _draw_line(image, left, y, width - right, y, (220, 220, 220))
    _draw_line(image, left, top, left, height - bottom, (0, 0, 0), width=2)
    _draw_line(image, left, height - bottom, width - right, height - bottom, (0, 0, 0), width=2)

    points = []
    denominator = max(len(values) - 1, 1)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            points.append(None)
            continue
        x = left + round(plot_width * index / denominator)
        y = top + round(plot_height * (y_max - float(value)) / (y_max - y_min))
        points.append((x, y))
    for previous, current in zip(points, points[1:]):
        if previous is not None and current is not None:
            _draw_line(image, *previous, *current, (31, 119, 180), width=3)
    if len(points) == 1 and points[0] is not None:
        x, y = points[0]
        image[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] = (31, 119, 180)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + row.tobytes() for row in image)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as file:
        file.write(png)


def _save_loss_plot(losses: list[float], path: str) -> None:
    if plt is None:
        _save_simple_loss_png(losses, path)
        return
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Single-stage init+goal robot-object diffusion loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=100)
    plt.close()


def _make_model(architecture: str, cfg: dict, window_size: int) -> torch.nn.Module:
    state_dim = int(cfg.get("state_dim", ROBOT_OBJECT_STATE_DIM))
    cond_dim = int(cfg.get("cond_dim", 0))
    global_cond_dim = int(cfg.get("global_cond_dim", GLOBAL_COND_DIM))
    contact_dim = int(cfg.get("contact_dim", 0))
    if (state_dim, cond_dim, global_cond_dim, contact_dim) != (47, 0, 56, 0):
        raise ValueError("Required model dimensions are state=47, cond=0, global_cond=56, contact=0")
    common = dict(state_dim=47, cond_dim=0, global_cond_dim=56, contact_dim=0)
    if architecture == "transformer":
        return Stage2TransformerModel(
            **common,
            global_cond_hidden=cfg.get("global_cond_hidden"),
            d_model=int(cfg.get("d_model", 512)),
            nhead=int(cfg.get("nhead", 8)),
            num_layers=int(cfg.get("num_layers", 8)),
            dim_feedforward=int(cfg.get("dim_feedforward", 512)),
            dropout=float(cfg.get("dropout", 0.1)),
            max_len=int(cfg.get("max_len", window_size)),
        )
    if architecture == "mlp":
        return Stage2MLPModel(
            **common,
            hidden_dim=int(cfg.get("mlp_hidden", 512)),
            num_layers=int(cfg.get("mlp_layers", 4)),
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


def _global_cond_from_batch(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Build normalized g56 from the normalized state and normalized final goal."""
    initial = state[:, 0]
    g56 = torch.cat(
        [goal, initial[:, :ROBOT_STATE_DIM], initial[:, ROBOT_STATE_DIM:]], dim=-1
    )
    if g56.shape[-1] != GLOBAL_COND_DIM:
        raise ValueError(f"Expected 56D global condition, got {g56.shape}")
    return g56


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-stage init+goal robot-object diffusion")
    parser.add_argument(
        "--config_path",
        default=os.path.join(PROJECT_ROOT, "config", "train_object_goal_single_stage_init_goal_hf_bps.yaml"),
    )
    args = parser.parse_args()
    yml = load_config(args.config_path)
    train_cfg, dataset_cfg, model_cfg = yml["train"], yml["dataset"], yml["model"]
    temporal_cfg = loss_config(yml, "single_stage")
    if str(train_cfg.get("prediction_type", "x0")) != "x0":
        raise ValueError("This model supports prediction_type='x0' only")

    root_dir = yml["root_dir"]
    root_dir = root_dir if os.path.isabs(root_dir) else os.path.join(PROJECT_ROOT, root_dir)
    save_dir = train_cfg["save_dir"]
    save_dir = save_dir if os.path.isabs(save_dir) else os.path.join(PROJECT_ROOT, save_dir)
    device = torch.device(train_cfg["device"])
    batch_size = int(train_cfg["batch_size"])
    num_epochs = int(train_cfg["num_epochs"])
    timesteps = int(train_cfg["timesteps"])
    lr = float(train_cfg["lr"])
    architecture = str(train_cfg.get("architecture", "transformer"))
    window_size = int(dataset_cfg["window_size"])
    stride = int(dataset_cfg["stride"])
    save_every = int(train_cfg.get("save_every", 100))
    max_steps = train_cfg.get("max_train_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    max_seconds = train_cfg.get("max_train_seconds")
    max_seconds = float(max_seconds) if max_seconds is not None else None
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    prefix = train_cfg.get("exp_prefix", MODEL_TYPE)
    exp_name = f"{prefix}_e{num_epochs}_b{batch_size}_lr{lr}_ts{timesteps}_w{window_size}_s{stride}_{architecture}_{timestamp}"
    log_path = os.path.join(save_dir, exp_name)
    figure_dir = os.path.join(log_path, "figures")
    ckpt_dir = os.path.join(log_path, "checkpoints")
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    dump_config(os.path.join(log_path, "config.yml"), yml)

    # The legacy loader may read hand fields while validating source PKLs. The
    # loop below consumes only state and goal; all optional conditioning paths
    # (hands, BPS, geometry, contacts) are disabled here.
    dataset = HFFullBodyDataset(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=int(dataset_cfg.get("min_seq_len", 30)),
        train=True,
        train_split=float(dataset_cfg.get("train_split", 0.99)),
        preload=bool(dataset_cfg.get("preload", True)),
        normalize_hands=False,
        hand_condition_dir=None,
        require_hand_condition=False,
        include_contact_data=False,
        target_includes_object_pose=True,
        include_object_context=False,
        include_goal=True,
    )
    sampler = None
    if bool(dataset_cfg.get("balance_by_object", False)):
        labels = dataset.get_window_object_names()
        sampler = build_balanced_sampler(
            labels,
            power=float(dataset_cfg.get("object_balance_power", 1.0)),
            min_count=int(dataset_cfg.get("object_balance_min_count", 1)),
            seed=seed,
        )
        print(f"Object-balanced sampler: {format_label_counts(labels)}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    sample = dataset[0]
    if tuple(sample["state"].shape) != (window_size, ROBOT_OBJECT_STATE_DIM):
        raise ValueError(f"Expected state ({window_size}, 47), got {sample['state'].shape}")
    if sample["goal"].shape[-1] != OBJECT_POSE_DIM:
        raise ValueError(f"Expected 9D goal, got {sample['goal'].shape}")

    model = _make_model(architecture, model_cfg, window_size).to(device)
    schedule_cfg = DiffusionConfig(
        timesteps=timesteps,
        beta_start=float(train_cfg.get("beta_start", 1e-4)),
        beta_end=float(train_cfg.get("beta_end", 0.02)),
    )
    schedule = DiffusionSchedule(schedule_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Single-stage init+goal robot-object diffusion training")
    print("x0/model output: (B, T, 47); cond=None; global_cond: (B, 56)")
    print(f"global_cond layout: {GLOBAL_COND_LAYOUT}")
    if plt is None:
        print("Matplotlib unavailable; using built-in loss PNG renderer")

    losses = []
    best_loss = float("inf")
    global_step = 0
    started = time.monotonic()
    stop = False
    final_ckpt = None
    for epoch in range(num_epochs):
        model.train()
        total, batches = 0.0, 0
        for step, batch in enumerate(loader):
            # No batch cond, hands, BPS, object trajectory condition, or contacts.
            state = batch["state"].to(device)
            goal = batch["goal"].to(device)
            global_cond = _global_cond_from_batch(state, goal)
            t = torch.randint(0, timesteps, (state.shape[0],), device=device)
            x_t = schedule.q_sample(state, t, torch.randn_like(state))
            x0_hat = model(x_t, t, cond=None, global_cond=global_cond)
            base_loss = F.mse_loss(x0_hat, state)
            pred_phys = denormalize(x0_hat, dataset.state_mean, dataset.state_std)
            state_phys = denormalize(state, dataset.state_mean, dataset.state_std)
            temporal_loss, temporal_metrics = temporal_reconstruction_loss(pred_phys, state_phys, temporal_cfg)
            loss = temporal_cfg["base_weight"] * base_loss + temporal_loss
            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
            if global_step % int(train_cfg.get("log_every", 50)) == 0:
                metrics = {"base": float(base_loss.detach().cpu()), **temporal_metrics}
                print(f"Epoch {epoch} Step {step} (global {global_step}): loss={loss.item():.6f} {format_metrics(metrics)}")
            global_step += 1
            stop = (max_steps is not None and global_step >= max_steps) or (
                max_seconds is not None and time.monotonic() - started >= max_seconds
            )
            if stop:
                break

        avg_loss = total / max(batches, 1)
        losses.append(avg_loss)
        best_loss = min(best_loss, avg_loss)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f} (best={best_loss:.6f})")
        np.savetxt(
            os.path.join(log_path, "losses.csv"),
            np.column_stack((np.arange(len(losses)), np.asarray(losses))),
            delimiter=",",
            header="epoch,loss",
            comments="",
        )
        should_save = (epoch + 1) % save_every == 0 or epoch == num_epochs - 1 or stop
        if should_save:
            final_ckpt = os.path.join(ckpt_dir, f"object_goal_single_stage_init_goal_epoch_{epoch:06d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss_history": losses,
                    "config": yml,
                    "generative_model": "ddpm",
                    "model_type": MODEL_TYPE,
                    "prediction_type": PREDICTION_TYPE,
                    "state_dim": 47,
                    "cond_dim": 0,
                    "global_cond_dim": 56,
                    "global_cond_layout": GLOBAL_COND_LAYOUT,
                    "stage_target": "robot_state_plus_object_pose",
                    "layout": robot_object_layout(),
                    "condition": {
                        "per_frame": "none",
                        "global": "final_object_frame+initial_robot_frame+initial_object_frame",
                        "uses_hands": False,
                        "uses_bps": False,
                        "uses_object_trajectory": False,
                        "uses_geometry": False,
                        "uses_contacts": False,
                    },
                    "norm_stats": {
                        "state_mean": dataset.state_mean,
                        "state_std": dataset.state_std,
                        "goal_mean": dataset.goal_mean,
                        "goal_std": dataset.goal_std,
                    },
                    "schedule": {
                        "timesteps": schedule_cfg.timesteps,
                        "beta_start": schedule_cfg.beta_start,
                        "beta_end": schedule_cfg.beta_end,
                    },
                },
                final_ckpt,
            )
            print(f"Saved checkpoint: {final_ckpt}")
            plot_path = os.path.join(figure_dir, f"loss_epoch_{epoch}.png")
            _save_loss_plot(losses, plot_path)
            print(f"Saved loss plot: {plot_path}")
        if stop:
            print("Stopping due to configured smoke-run limit")
            break
    print(f"Training complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
