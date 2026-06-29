"""
Train Stage 2 of the two-stage object-goal pipeline.

Target:
    robot state trajectory R_{0:T} plus object pose trajectory O_{0:T}

Condition:
    hand trajectory H_{0:T}, HF-BPS object geometry/context, and final full
    object pose g. During sampling, H is produced by Stage 1 and contact
    rectified before this Stage 2 model is called.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
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

from datasets.hf_motion_dataset import HFFullBodyDataset
from models.stage2_diffusion import Stage2MLPModel, Stage2TransformerModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.general import dump_config, load_config, load_torch_checkpoint
from utils.motion_losses import (
    denormalize,
    format_metrics,
    full_body_contact_loss,
    loss_config,
    robot_fk_hand_loss,
    temporal_reconstruction_loss,
)
from utils.object_goal_features import OBJECT_POSE_DIM, ROBOT_OBJECT_STATE_DIM, ROBOT_STATE_DIM, robot_object_layout
from utils.object_sampling import build_balanced_sampler, format_label_counts


def _make_model(architecture: str, model_cfg: dict, state_dim: int, cond_dim: int, window_size: int):
    contact_dim = int(model_cfg.get("contact_dim", 0))
    common = {
        "state_dim": state_dim,
        "cond_dim": cond_dim,
        "global_cond_dim": OBJECT_POSE_DIM,
        "global_cond_hidden": model_cfg.get("global_cond_hidden"),
        "contact_dim": contact_dim,
    }
    if architecture == "transformer":
        return Stage2TransformerModel(
            **common,
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            num_layers=int(model_cfg["num_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            max_len=int(model_cfg.get("max_len", window_size)),
        )
    if architecture == "mlp":
        return Stage2MLPModel(
            **common,
            hidden_dim=int(model_cfg.get("mlp_hidden", 512)),
            num_layers=int(model_cfg.get("mlp_layers", 4)),
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


def _load_partial(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device)
    src = ckpt["model"]
    dst = model.state_dict()
    filtered = {}
    skipped = {}
    for key, value in src.items():
        if key in dst and tuple(dst[key].shape) == tuple(value.shape):
            filtered[key] = value
        elif key in dst:
            skipped[key] = (tuple(value.shape), tuple(dst[key].shape))
    result = model.load_state_dict(filtered, strict=False)
    print(f"  Partial checkpoint load: {result}")
    if skipped:
        print("  Skipped shape-mismatched tensors:")
        for key, (src_shape, dst_shape) in skipped.items():
            print(f"    {key}: {src_shape} -> {dst_shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train object-goal Stage 2 HF-BPS prior")
    parser.add_argument(
        "--config_path",
        default=os.path.join(PROJECT_ROOT, "config", "train_object_goal_stage2_hf_bps.yaml"),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    train_cfg = yml["train"]
    dataset_cfg = yml["dataset"]
    model_cfg = yml["model"]
    loss_cfg = loss_config(yml, "stage2")

    root_dir = yml["root_dir"]
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(PROJECT_ROOT, root_dir)
    save_dir = train_cfg["save_dir"]
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(PROJECT_ROOT, save_dir)

    device = torch.device(train_cfg["device"])
    batch_size = int(train_cfg["batch_size"])
    timesteps = int(train_cfg["timesteps"])
    num_epochs = int(train_cfg["num_epochs"])
    lr = float(train_cfg["lr"])
    architecture = train_cfg.get("architecture", "transformer")
    save_every = int(train_cfg.get("save_every", 100))
    max_train_seconds = train_cfg.get("max_train_seconds")
    max_train_seconds = float(max_train_seconds) if max_train_seconds is not None else None
    seed = train_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    window_size = int(dataset_cfg["window_size"])
    stride = int(dataset_cfg["stride"])
    contact_dim = int(model_cfg.get("contact_dim", 0))
    include_contact_data = contact_dim > 0 or any(
        loss_cfg[name] > 0.0
        for name in (
            "contact_state_weight",
            "object_contact_dist_weight",
            "floor_contact_dist_weight",
            "contact_velocity_weight",
            "foot_slide_weight",
            "floor_penetration_weight",
            "support_weight",
        )
    )

    hand_condition_dir = dataset_cfg.get("hand_condition_dir")
    if hand_condition_dir and not os.path.isabs(hand_condition_dir):
        hand_condition_dir = os.path.join(PROJECT_ROOT, hand_condition_dir)

    exp_prefix = train_cfg.get("exp_prefix", "object_goal_stage2_hf_bps")
    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = f"{exp_prefix}_e{num_epochs}_b{batch_size}_lr{lr}_ts{timesteps}_w{window_size}_s{stride}_{architecture}_"
    log_path = os.path.join(save_dir, exp_name + dtn)
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
        normalize_hands=bool(dataset_cfg.get("normalize_hands", False)),
        hand_condition_dir=hand_condition_dir,
        hand_condition_key=dataset_cfg.get("hand_condition_key", "hand_positions"),
        require_hand_condition=bool(dataset_cfg.get("require_hand_condition", False)),
        include_contact_data=include_contact_data,
        floor_height=float(loss_cfg["floor_height"]),
        object_contact_sigma=float(dataset_cfg.get("object_contact_sigma", 0.05)),
        floor_contact_sigma=float(dataset_cfg.get("floor_contact_sigma", 0.04)),
        contact_eps=float(dataset_cfg.get("contact_eps", 0.2)),
        stick_speed_threshold=float(dataset_cfg.get("stick_speed_threshold", 0.04)),
        target_includes_object_pose=True,
        include_object_context=True,
        include_goal=True,
        flatten_bps=True,
        object_conditioning_variant=dataset_cfg.get("object_conditioning_variant", "variant0"),
        object_context_mode=dataset_cfg.get("object_context_mode", "static_bps"),
    )

    sampler = None
    if dataset_cfg.get("balance_by_object", False):
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
    state_dim = sample["state"].shape[-1]
    cond_dim = sample["cond"].shape[-1]
    if state_dim != ROBOT_OBJECT_STATE_DIM:
        raise ValueError(f"Expected Stage 2 target dim {ROBOT_OBJECT_STATE_DIM}, got {state_dim}")

    model = _make_model(architecture, model_cfg, state_dim, cond_dim, window_size).to(device)
    print("=" * 60)
    print("Object-goal Stage 2 training")
    print("=" * 60)
    print(f"Logs: {log_path}")
    print(f"Windows: {len(dataset)} from {dataset.num_files} files")
    print(f"Target: robot state + object pose ({state_dim}D)")
    print(f"Condition dim: {cond_dim} (hands + static BPS geometry context)")
    print("Global condition: final full object pose (9D)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    init_ckpt_path = train_cfg.get("init_ckpt_path")
    if init_ckpt_path:
        if not os.path.isabs(init_ckpt_path):
            init_ckpt_path = os.path.join(PROJECT_ROOT, init_ckpt_path)
        print(f"Initializing compatible tensors from {init_ckpt_path}")
        _load_partial(model, init_ckpt_path, device)

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

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(dataloader):
            state = batch["state"].to(device)
            cond = batch["cond"].to(device)
            goal = batch["goal"].to(device)
            cond_hands = batch["cond_hands"].to(device)
            B = state.shape[0]

            t = torch.randint(0, timesteps, (B,), device=device)
            noise = torch.randn_like(state)
            state_noisy = schedule.q_sample(state, t, noise)
            if contact_dim > 0 and hasattr(model, "forward_with_contact"):
                state_pred, contact_logits = model.forward_with_contact(
                    state_noisy,
                    t,
                    cond,
                    global_cond=goal,
                )
            else:
                state_pred = model(state_noisy, t, cond, global_cond=goal)
                contact_logits = None

            base_loss = F.mse_loss(state_pred, state)
            robot_base_loss = F.mse_loss(
                state_pred[..., :ROBOT_STATE_DIM],
                state[..., :ROBOT_STATE_DIM],
            )
            object_base_loss = F.mse_loss(
                state_pred[..., ROBOT_STATE_DIM:],
                state[..., ROBOT_STATE_DIM:],
            )
            state_pred_phys = denormalize(state_pred, dataset.state_mean, dataset.state_std)
            state_phys = denormalize(state, dataset.state_mean, dataset.state_std)
            temporal_loss, temporal_metrics = temporal_reconstruction_loss(
                state_pred_phys,
                state_phys,
                loss_cfg,
            )
            robot_pred_phys = state_pred_phys[..., :ROBOT_STATE_DIM]
            robot_loss, robot_metrics = robot_fk_hand_loss(
                robot_pred_phys,
                cond_hands,
                loss_cfg,
            )
            contact_loss, contact_metrics = full_body_contact_loss(
                robot_pred_phys,
                contact_logits,
                batch,
                loss_cfg,
                global_step=global_step,
            )
            loss = (
                loss_cfg["base_weight"] * base_loss
                + temporal_loss
                + robot_loss
                + contact_loss
            )

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
                    **robot_metrics,
                    **contact_metrics,
                }
                print(
                    f"Epoch {epoch} Step {step} (global {global_step}): "
                    f"loss={loss.item():.6f} {format_metrics(metrics)}"
                )
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        best_loss = min(best_loss, avg_loss)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f} (best={best_loss:.6f})")

        should_stop = (
            max_train_seconds is not None
            and time.monotonic() - train_start >= max_train_seconds
        )
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1 or should_stop:
            ckpt_file = os.path.join(ckpt_path, f"object_goal_stage2_epoch_{epoch:06d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": yml,
                    "generative_model": "ddpm",
                    "pipeline_type": "object_goal_two_stage",
                    "stage": 2,
                    "stage_target": "robot_state_plus_object_pose",
                    "prediction_type": "x0",
                    "state_dim": state_dim,
                    "cond_dim": cond_dim,
                    "requires_stage1_hands": True,
                    "hand_contact_rectification_required": True,
                    "layout": robot_object_layout(),
                    "condition": {
                        "per_frame": "hands + static_bps_context",
                        "global_goal_dim": OBJECT_POSE_DIM,
                        "object_context_mode": dataset_cfg.get("object_context_mode", "static_bps"),
                    },
                    "norm_stats": {
                        "state_mean": dataset.state_mean,
                        "state_std": dataset.state_std,
                        "hand_mean": dataset.hand_mean,
                        "hand_std": dataset.hand_std,
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
            print(f"  Saved checkpoint: {ckpt_file}")

        if plt is not None and (epoch % save_every == 0 or epoch == num_epochs - 1):
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(losses)), losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Object-goal Stage 2 loss")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figure_path, f"loss_epoch_{epoch}.png"), dpi=100)
            plt.close()

        if should_stop:
            elapsed = time.monotonic() - train_start
            print(f"Stopping after {elapsed:.1f}s due to train.max_train_seconds")
            break

    print("=" * 60)
    print("Object-goal Stage 2 training complete")
    print(f"Final checkpoint: {ckpt_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
