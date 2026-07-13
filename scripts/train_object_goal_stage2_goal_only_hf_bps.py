"""Train goal-only Stage 2 robot-object diffusion with hand conditioning.

The target is [robot_state(38), object_pose(9)]. Ground-truth hands are the
only per-frame condition; the final 9D object goal is the global condition.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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
from utils.object_goal_features import (
    OBJECT_POSE_DIM,
    ROBOT_OBJECT_STATE_DIM,
    ROBOT_STATE_DIM,
    robot_object_layout,
)
from utils.object_sampling import build_balanced_sampler, format_label_counts


MODEL_TYPE = "object_goal_stage2_goal_only"
PREDICTION_TYPE = "x0"
HAND_DIM = 6


def _make_model(architecture: str, model_cfg: dict, window_size: int) -> torch.nn.Module:
    state_dim = int(model_cfg.get("state_dim", ROBOT_OBJECT_STATE_DIM))
    cond_dim = int(model_cfg.get("cond_dim", HAND_DIM))
    global_cond_dim = int(model_cfg.get("global_cond_dim", OBJECT_POSE_DIM))
    contact_dim = int(model_cfg.get("contact_dim", 0))
    if state_dim != ROBOT_OBJECT_STATE_DIM:
        raise ValueError(f"Goal-only Stage 2 requires state_dim={ROBOT_OBJECT_STATE_DIM}")
    if cond_dim != HAND_DIM:
        raise ValueError(f"Goal-only Stage 2 requires cond_dim={HAND_DIM}")
    if global_cond_dim != OBJECT_POSE_DIM:
        raise ValueError(f"Goal-only Stage 2 requires global_cond_dim={OBJECT_POSE_DIM}")
    if contact_dim != 0:
        raise ValueError("Goal-only Stage 2 does not use a contact head")

    common = {
        "state_dim": ROBOT_OBJECT_STATE_DIM,
        "cond_dim": HAND_DIM,
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
    raise ValueError(f"Unknown architecture {architecture!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train goal-only object-goal Stage 2")
    parser.add_argument(
        "--config_path",
        default=os.path.join(
            PROJECT_ROOT, "config", "train_object_goal_stage2_goal_only_hf_bps.yaml"
        ),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    train_cfg = yml["train"]
    dataset_cfg = yml["dataset"]
    model_cfg = yml["model"]
    if str(train_cfg.get("prediction_type", PREDICTION_TYPE)) != PREDICTION_TYPE:
        raise ValueError("Goal-only Stage 2 supports prediction_type='x0' only")

    root_dir = yml["root_dir"]
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(PROJECT_ROOT, root_dir)
    save_dir = train_cfg["save_dir"]
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(PROJECT_ROOT, save_dir)

    device = torch.device(train_cfg["device"])
    batch_size = int(train_cfg["batch_size"])
    num_epochs = int(train_cfg["num_epochs"])
    timesteps = int(train_cfg["timesteps"])
    lr = float(train_cfg["lr"])
    architecture = str(train_cfg.get("architecture", "transformer"))
    window_size = int(dataset_cfg["window_size"])
    stride = int(dataset_cfg["stride"])
    save_every = int(train_cfg.get("save_every", 100))
    max_train_steps = train_cfg.get("max_train_steps")
    max_train_steps = int(max_train_steps) if max_train_steps is not None else None
    max_train_seconds = train_cfg.get("max_train_seconds")
    max_train_seconds = float(max_train_seconds) if max_train_seconds is not None else None
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_prefix = train_cfg.get("exp_prefix", MODEL_TYPE)
    exp_name = (
        f"{exp_prefix}_e{num_epochs}_b{batch_size}_lr{lr}_ts{timesteps}_"
        f"w{window_size}_s{stride}_{architecture}_{timestamp}"
    )
    log_path = os.path.join(save_dir, exp_name)
    figure_path = os.path.join(log_path, "figures")
    ckpt_path = os.path.join(log_path, "checkpoints")
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    dump_config(os.path.join(log_path, "config.yml"), yml)

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
        object_labels = dataset.get_window_object_names()
        sampler = build_balanced_sampler(
            object_labels,
            power=float(dataset_cfg.get("object_balance_power", 1.0)),
            min_count=int(dataset_cfg.get("object_balance_min_count", 1)),
            seed=seed,
        )
        print(f"Object-balanced sampler: {format_label_counts(object_labels)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    sample = dataset[0]
    if int(sample["state"].shape[-1]) != ROBOT_OBJECT_STATE_DIM:
        raise ValueError(f"Expected {ROBOT_OBJECT_STATE_DIM}D robot-object target")
    if int(sample["cond_hands"].shape[-1]) != HAND_DIM:
        raise ValueError(f"Expected {HAND_DIM}D hand condition")
    if int(sample["goal"].shape[-1]) != OBJECT_POSE_DIM:
        raise ValueError(f"Expected {OBJECT_POSE_DIM}D final goal")
    model = _make_model(architecture, model_cfg, window_size).to(device)

    print("=" * 64)
    print("Goal-only object-goal Stage 2 training")
    print(f"Logs: {log_path}")
    print("Target x0: [robot_state(38), object_pose(9)] = 47D")
    print("Per-frame condition: ground-truth hands (6D)")
    print("Global condition: final object goal (9D)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 64)

    schedule_cfg = DiffusionConfig(
        timesteps=timesteps,
        beta_start=float(train_cfg.get("beta_start", 1e-4)),
        beta_end=float(train_cfg.get("beta_end", 0.02)),
    )
    schedule = DiffusionSchedule(schedule_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0
    best_loss = float("inf")
    losses = []
    robot_losses = []
    object_losses = []
    train_start = time.monotonic()
    ckpt_file = None
    stop_requested = False

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_robot_loss = 0.0
        epoch_object_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(dataloader):
            # Only the 47D target, hands, and final goal enter this loop.
            state = batch["state"].to(device)
            hands = batch["cond_hands"].to(device)
            goal = batch["goal"].to(device)
            t = torch.randint(0, timesteps, (state.shape[0],), device=device)
            state_noisy = schedule.q_sample(state, t, torch.randn_like(state))
            state_pred = model(state_noisy, t, cond=hands, global_cond=goal)

            loss = F.mse_loss(state_pred, state)
            robot_mse = F.mse_loss(
                state_pred[..., :ROBOT_STATE_DIM], state[..., :ROBOT_STATE_DIM]
            )
            object_mse = F.mse_loss(
                state_pred[..., ROBOT_STATE_DIM:], state[..., ROBOT_STATE_DIM:]
            )

            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_robot_loss += float(robot_mse.detach().cpu())
            epoch_object_loss += float(object_mse.detach().cpu())
            num_batches += 1
            if global_step % int(train_cfg.get("log_every", 50)) == 0:
                print(
                    f"Epoch {epoch} Step {step} (global {global_step}): "
                    f"state_mse={loss.item():.6f} robot_mse={robot_mse.item():.6f} "
                    f"object_mse={object_mse.item():.6f}"
                )
            global_step += 1
            if max_train_steps is not None and global_step >= max_train_steps:
                stop_requested = True
            if max_train_seconds is not None and time.monotonic() - train_start >= max_train_seconds:
                stop_requested = True
            if stop_requested:
                break

        if num_batches == 0:
            raise RuntimeError("Training dataloader produced no batches")
        avg_loss = epoch_loss / num_batches
        avg_robot_loss = epoch_robot_loss / num_batches
        avg_object_loss = epoch_object_loss / num_batches
        losses.append(avg_loss)
        robot_losses.append(avg_robot_loss)
        object_losses.append(avg_object_loss)
        best_loss = min(best_loss, avg_loss)
        print(f"Epoch {epoch}: avg_state_mse={avg_loss:.6f} (best={best_loss:.6f})")

        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1 or stop_requested:
            ckpt_file = os.path.join(
                ckpt_path, f"object_goal_stage2_goal_only_epoch_{epoch:06d}.pt"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": yml,
                    "generative_model": "ddpm",
                    "model_type": MODEL_TYPE,
                    "prediction_type": PREDICTION_TYPE,
                    "stage": 2,
                    "stage_target": "robot_state_plus_object_pose",
                    "state_dim": ROBOT_OBJECT_STATE_DIM,
                    "cond_dim": HAND_DIM,
                    "global_cond_dim": OBJECT_POSE_DIM,
                    "requires_stage1_hands": True,
                    "hand_contact_rectification_required": False,
                    "layout": robot_object_layout(),
                    "condition": {
                        "per_frame": "hands",
                        "global": "final_object_goal",
                        "uses_bps": False,
                        "uses_object_trajectory": False,
                        "uses_geometry": False,
                        "uses_contact": False,
                    },
                    "norm_stats": {
                        "state_mean": dataset.state_mean,
                        "state_std": dataset.state_std,
                        "hand_mean": dataset.hand_mean,
                        "hand_std": dataset.hand_std,
                        "goal_mean": dataset.goal_mean,
                        "goal_std": dataset.goal_std,
                        "normalize_hands": False,
                    },
                    "schedule": {
                        "timesteps": schedule_cfg.timesteps,
                        "beta_start": schedule_cfg.beta_start,
                        "beta_end": schedule_cfg.beta_end,
                    },
                },
                ckpt_file,
            )
            print(f"Saved checkpoint: {ckpt_file}")

        plot_every = int(train_cfg.get("plot_every", save_every))
        if plt is not None and (epoch % plot_every == 0 or epoch == num_epochs - 1 or stop_requested):
            epochs = np.arange(len(losses))
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, label="state MSE")
            plt.plot(epochs, robot_losses, label="robot MSE")
            plt.plot(epochs, object_losses, label="object MSE")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Goal-only object-goal Stage 2 loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            figure_file = os.path.join(figure_path, f"loss_epoch_{epoch:06d}.png")
            plt.savefig(figure_file, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"Saved loss plot: {figure_file}")
        elif plt is None and epoch == 0:
            print("Warning: matplotlib is unavailable; loss plots will not be generated")

        if stop_requested:
            print("Stopping due to configured smoke-run limit")
            break

    print("=" * 64)
    print("Goal-only object-goal Stage 2 training complete")
    print(f"Final checkpoint: {ckpt_file}")
    print("=" * 64)


if __name__ == "__main__":
    main()
