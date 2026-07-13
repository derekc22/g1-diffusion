"""Train a single-stage goal-only robot-object diffusion baseline.

The denoiser predicts x0 = [robot_state(38), object_pose(9)] from only the
noised 47D trajectory, the diffusion timestep, and the final 9D object goal.
HF-BPS PKLs are used as the data source, but hands, BPS, geometry, contacts,
and per-frame object trajectories are never passed as conditioning inputs.
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
from utils.motion_losses import denormalize, format_metrics, loss_config, temporal_reconstruction_loss
from utils.object_goal_features import (
    OBJECT_POSE_DIM,
    ROBOT_OBJECT_STATE_DIM,
    ROBOT_STATE_DIM,
    robot_object_layout,
)
from utils.object_sampling import build_balanced_sampler, format_label_counts


MODEL_TYPE = "object_goal_single_stage_goal_only"
PREDICTION_TYPE = "x0"


def _make_model(
    architecture: str,
    model_cfg: dict,
    window_size: int,
) -> torch.nn.Module:
    state_dim = int(model_cfg.get("state_dim", ROBOT_OBJECT_STATE_DIM))
    cond_dim = int(model_cfg.get("cond_dim", 0))
    global_cond_dim = int(model_cfg.get("global_cond_dim", OBJECT_POSE_DIM))
    contact_dim = int(model_cfg.get("contact_dim", 0))
    if state_dim != ROBOT_OBJECT_STATE_DIM:
        raise ValueError(f"model.state_dim must be {ROBOT_OBJECT_STATE_DIM}, got {state_dim}")
    if cond_dim != 0:
        raise ValueError("Goal-only baseline requires model.cond_dim=0")
    if global_cond_dim != OBJECT_POSE_DIM:
        raise ValueError(f"model.global_cond_dim must be {OBJECT_POSE_DIM}")
    if contact_dim != 0:
        raise ValueError("Goal-only baseline does not use a contact head; set model.contact_dim=0")

    common = {
        "state_dim": state_dim,
        "cond_dim": 0,
        "global_cond_dim": global_cond_dim,
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
    parser = argparse.ArgumentParser(description="Train single-stage goal-only robot-object diffusion")
    parser.add_argument(
        "--config_path",
        default=os.path.join(PROJECT_ROOT, "config", "train_object_goal_single_stage_hf_bps.yaml"),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    train_cfg = yml["train"]
    dataset_cfg = yml["dataset"]
    model_cfg = yml["model"]
    loss_cfg = loss_config(yml, "single_stage")

    if str(train_cfg.get("prediction_type", PREDICTION_TYPE)) != PREDICTION_TYPE:
        raise ValueError("Goal-only baseline supports prediction_type='x0' only")

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
    max_train_seconds = train_cfg.get("max_train_seconds")
    max_train_seconds = float(max_train_seconds) if max_train_seconds is not None else None
    max_train_steps = train_cfg.get("max_train_steps")
    max_train_steps = int(max_train_steps) if max_train_steps is not None else None
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    exp_prefix = train_cfg.get("exp_prefix", MODEL_TYPE)
    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
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

    # This existing dataset constructs a legacy hand field internally, but this
    # code path consumes only `state` and `goal` from each batch. BPS/object
    # context and contact label construction are explicitly disabled.
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
    state_dim = int(sample["state"].shape[-1])
    goal_dim = int(sample["goal"].shape[-1])
    if state_dim != ROBOT_OBJECT_STATE_DIM:
        raise ValueError(f"Expected {ROBOT_OBJECT_STATE_DIM}D target, got {state_dim}D")
    if goal_dim != OBJECT_POSE_DIM:
        raise ValueError(f"Expected {OBJECT_POSE_DIM}D final goal, got {goal_dim}D")

    model = _make_model(architecture, model_cfg, window_size).to(device)
    print("=" * 64)
    print("Single-stage goal-only robot-object diffusion training")
    print(f"Logs: {log_path}")
    print(f"Target x0: [robot(38), object_pose(9)] = {state_dim}D")
    print("Per-frame semantic condition: none (cond_dim=0, cond=None)")
    print(f"Global condition: final object goal ({goal_dim}D)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 64)

    schedule_cfg = DiffusionConfig(
        timesteps=timesteps,
        beta_start=float(train_cfg.get("beta_start", 1e-4)),
        beta_end=float(train_cfg.get("beta_end", 0.02)),
    )
    schedule = DiffusionSchedule(schedule_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    best_loss = float("inf")
    global_step = 0
    train_start = time.monotonic()
    ckpt_file = None
    stop_requested = False

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(dataloader):
            # Deliberately do not read batch["cond"], batch["cond_hands"], BPS,
            # contacts, geometry, or any clean per-frame object condition.
            state = batch["state"].to(device)
            goal = batch["goal"].to(device)
            batch_size_actual = state.shape[0]

            t = torch.randint(0, timesteps, (batch_size_actual,), device=device)
            state_noisy = schedule.q_sample(state, t, torch.randn_like(state))
            state_pred = model(state_noisy, t, cond=None, global_cond=goal)

            base_loss = F.mse_loss(state_pred, state)
            robot_base_loss = F.mse_loss(
                state_pred[..., :ROBOT_STATE_DIM], state[..., :ROBOT_STATE_DIM]
            )
            object_base_loss = F.mse_loss(
                state_pred[..., ROBOT_STATE_DIM:], state[..., ROBOT_STATE_DIM:]
            )
            state_pred_phys = denormalize(state_pred, dataset.state_mean, dataset.state_std)
            state_phys = denormalize(state, dataset.state_mean, dataset.state_std)
            temporal_loss, temporal_metrics = temporal_reconstruction_loss(
                state_pred_phys, state_phys, loss_cfg
            )
            loss = loss_cfg["base_weight"] * base_loss + temporal_loss

            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            num_batches += 1
            if global_step % int(train_cfg.get("log_every", 50)) == 0:
                metrics = {
                    "base": float(base_loss.detach().cpu()),
                    "robot_base": float(robot_base_loss.detach().cpu()),
                    "object_base": float(object_base_loss.detach().cpu()),
                    **temporal_metrics,
                }
                print(
                    f"Epoch {epoch} Step {step} (global {global_step}): "
                    f"loss={loss.item():.6f} {format_metrics(metrics)}"
                )
            global_step += 1

            if max_train_steps is not None and global_step >= max_train_steps:
                stop_requested = True
            if max_train_seconds is not None and time.monotonic() - train_start >= max_train_seconds:
                stop_requested = True
            if stop_requested:
                break

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        best_loss = min(best_loss, avg_loss)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f} (best={best_loss:.6f})")

        should_save = (epoch + 1) % save_every == 0 or epoch == num_epochs - 1 or stop_requested
        if should_save:
            ckpt_file = os.path.join(ckpt_path, f"object_goal_single_stage_epoch_{epoch:06d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": yml,
                    "generative_model": "ddpm",
                    "model_type": MODEL_TYPE,
                    "prediction_type": PREDICTION_TYPE,
                    "state_dim": ROBOT_OBJECT_STATE_DIM,
                    "cond_dim": 0,
                    "global_cond_dim": OBJECT_POSE_DIM,
                    "stage_target": "robot_state_plus_object_pose",
                    "layout": robot_object_layout(),
                    "condition": {
                        "per_frame": "none",
                        "global": "final_object_goal",
                        "global_goal_dim": OBJECT_POSE_DIM,
                        "uses_hands": False,
                        "uses_bps": False,
                        "uses_object_trajectory": False,
                        "uses_geometry": False,
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
                ckpt_file,
            )
            print(f"Saved checkpoint: {ckpt_file}")


        if plt is not None and (epoch % save_every == 0 or epoch == num_epochs - 1):
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(losses)), losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Single-stage goal-only robot-object diffusion loss")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figure_path, f"loss_epoch_{epoch}.png"), dpi=100)
            plt.close()

        if stop_requested:
            print("Stopping due to configured smoke-run limit")
            break

    print("=" * 64)
    print("Single-stage goal-only training complete")
    print(f"Final checkpoint: {ckpt_file}")
    print("=" * 64)


if __name__ == "__main__":
    main()
