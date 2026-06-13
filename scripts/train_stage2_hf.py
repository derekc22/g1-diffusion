"""
Stage 2 DDPM Training Script — HuggingFace Dataset
Hand Positions → Full-Body Motion

Trains the Stage 2 diffusion model which generates full-body robot poses
conditioned on hand positions.

Uses standard DDPM with x0-prediction:
    q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Loss = ||f_theta(x_t, t, cond) - x_0||^2

State vector: [root_pos(3), root_rot_6d(6), dof_pos(29)] = 38D
Conditioning: hand_positions(6D)

Usage:
    python scripts/train_stage2_hf.py
"""

import os
import sys
import types
import time
import argparse
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Compatibility shim for pickles created by NumPy >= 2.0
# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------

from datasets.hf_motion_dataset import HFFullBodyDataset
from models.stage2_diffusion import Stage2TransformerModel, Stage2MLPModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.general import load_config, dump_config
from utils.object_sampling import build_balanced_sampler, format_label_counts
from utils.motion_losses import (
    denormalize,
    format_metrics,
    full_body_contact_loss,
    loss_config,
    robot_fk_hand_loss,
    temporal_reconstruction_loss,
)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 DDPM Training - HuggingFace Dataset")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "train_stage2_hf.yaml"),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)

    train_yml = yml["train"]
    dataset_yml = yml["dataset"]
    model_yml = yml["model"]
    loss_cfg = loss_config(yml, "stage2")

    root_dir = yml["root_dir"]
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(PROJECT_ROOT, root_dir)

    save_dir = train_yml["save_dir"]
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(PROJECT_ROOT, save_dir)

    batch_size = train_yml["batch_size"]
    timesteps = train_yml["timesteps"]
    num_epochs = train_yml["num_epochs"]
    lr = float(train_yml["lr"])
    device_str = train_yml["device"]
    architecture = train_yml["architecture"]
    save_every = train_yml.get("save_every", 200)
    init_ckpt_path = train_yml.get("init_ckpt_path")
    resume_optimizer = train_yml.get("resume_optimizer", False)
    max_train_seconds = train_yml.get("max_train_seconds")
    max_train_seconds = float(max_train_seconds) if max_train_seconds is not None else None

    window_size = dataset_yml["window_size"]
    stride = dataset_yml["stride"]
    min_seq_len = dataset_yml["min_seq_len"]
    train_split = dataset_yml["train_split"]
    preload = dataset_yml.get("preload", True)
    normalize_hands = dataset_yml.get("normalize_hands", False)
    hand_condition_dir = dataset_yml.get("hand_condition_dir")
    if hand_condition_dir and not os.path.isabs(hand_condition_dir):
        hand_condition_dir = os.path.join(PROJECT_ROOT, hand_condition_dir)
    hand_condition_key = dataset_yml.get("hand_condition_key", "hand_positions")
    require_hand_condition = dataset_yml.get("require_hand_condition", False)
    contact_dim = int(model_yml.get("contact_dim", 0))
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

    # Create experiment directory
    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = f"stage2_hf_e{num_epochs}_b{batch_size}_lr{lr}_ts{timesteps}_w{window_size}_s{stride}_{architecture}_"
    log_path = os.path.join(save_dir, exp_name + dtn)
    figure_path = os.path.join(log_path, "figures")
    ckpt_path = os.path.join(log_path, "checkpoints")

    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    dump_config(os.path.join(log_path, "config.yml"), yml)

    print(f"{'='*60}")
    print(f"Stage 2 DDPM Training — HuggingFace Dataset")
    print(f"{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Logs: {log_path}")
    print(f"Generative model: DDPM (x0-prediction)")

    device = torch.device(device_str)

    # Create dataset
    print(f"\nLoading dataset from {root_dir}")
    dataset = HFFullBodyDataset(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=min_seq_len,
        train=True,
        train_split=train_split,
        preload=preload,
        normalize_hands=normalize_hands,
        hand_condition_dir=hand_condition_dir,
        hand_condition_key=hand_condition_key,
        require_hand_condition=require_hand_condition,
        include_contact_data=include_contact_data,
        floor_height=loss_cfg["floor_height"],
        object_contact_sigma=dataset_yml.get("object_contact_sigma", 0.05),
        floor_contact_sigma=dataset_yml.get("floor_contact_sigma", 0.04),
        contact_eps=dataset_yml.get("contact_eps", 0.2),
        stick_speed_threshold=dataset_yml.get("stick_speed_threshold", 0.04),
    )

    print(f"  Windows: {len(dataset)}")
    print(f"  Files: {dataset.num_files}")
    print(f"  Normalize hand conditioning: {normalize_hands}")
    print(f"  Contact supervision: {include_contact_data}")
    if hand_condition_dir:
        print(f"  Hand conditioning source: {hand_condition_dir} [{hand_condition_key}]")

    sampler = None
    if dataset_yml.get("balance_by_object", False):
        object_labels = dataset.get_window_object_names()
        sampler = build_balanced_sampler(
            object_labels,
            power=float(dataset_yml.get("object_balance_power", 1.0)),
            min_count=int(dataset_yml.get("object_balance_min_count", 1)),
            seed=train_yml.get("seed"),
        )
        print(f"  Object-balanced sampler: {format_label_counts(object_labels)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # Get dimensions
    sample = dataset[0]
    state_dim = sample["state"].shape[-1]
    cond_dim = sample["cond"].shape[-1]

    print(f"\n  State dim: {state_dim}")
    print(f"  Cond dim (hands): {cond_dim}")

    # Create model (reuses the EXACT same Stage2TransformerModel)
    print(f"\nCreating Stage 2 {architecture} model")

    if architecture == "transformer":
        model = Stage2TransformerModel(
            state_dim=state_dim,
            cond_dim=cond_dim,
            d_model=model_yml["d_model"],
            nhead=model_yml["nhead"],
            num_layers=model_yml["num_layers"],
            dim_feedforward=model_yml["dim_feedforward"],
            dropout=model_yml["dropout"],
            max_len=model_yml.get("max_len", window_size + 100),
            contact_dim=contact_dim,
        ).to(device)
    else:
        model = Stage2MLPModel(
            state_dim=state_dim,
            cond_dim=cond_dim,
            hidden_dim=model_yml.get("mlp_hidden", 512),
            num_layers=model_yml.get("mlp_layers", 4),
            contact_dim=contact_dim,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Diffusion schedule
    schedule = DiffusionSchedule(
        DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if init_ckpt_path:
        print(f"Initializing from checkpoint: {init_ckpt_path}")
        init_ckpt = torch.load(init_ckpt_path, map_location=device)
        init_state = init_ckpt["model"]
        model_state = model.state_dict()
        filtered_state = {}
        skipped_shape_keys = {}
        for key, value in init_state.items():
            if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
                skipped_shape_keys[key] = (tuple(value.shape), tuple(model_state[key].shape))
                continue
            filtered_state[key] = value
        load_result = model.load_state_dict(
            filtered_state,
            strict=(contact_dim <= 0 and not skipped_shape_keys),
        )
        if skipped_shape_keys:
            print("  Skipped shape-mismatched checkpoint tensors:")
            for key, (src_shape, dst_shape) in skipped_shape_keys.items():
                print(f"    {key}: checkpoint {src_shape} -> model {dst_shape}")
        if contact_dim > 0:
            print(f"  Non-strict load for contact head: {load_result}")
        if resume_optimizer and "optimizer" in init_ckpt:
            optimizer.load_state_dict(init_ckpt["optimizer"])

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs")
    print(f"  Batch size: {batch_size}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Learning rate: {lr}")
    if max_train_seconds is not None:
        print(f"  Max train time: {max_train_seconds:.0f}s")
    print()

    global_step = 0
    losses = []
    best_loss = float("inf")
    train_start_time = time.monotonic()
    ckpt_file = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            state = batch["state"].to(device)    # (B, T, 38) normalized
            cond = batch["cond"].to(device)      # (B, T, 6) hand positions

            B = state.shape[0]

            # Sample timesteps
            t = torch.randint(0, timesteps, (B,), device=device)

            # Add noise to state
            noise = torch.randn_like(state)
            state_noisy = schedule.q_sample(state, t, noise)

            # Predict clean state (x0 prediction)
            if contact_dim > 0 and hasattr(model, "forward_with_contact"):
                state_pred, contact_logits = model.forward_with_contact(state_noisy, t, cond)
            else:
                state_pred = model(state_noisy, t, cond)
                contact_logits = None

            # Loss
            base_loss = F.mse_loss(state_pred, state)
            state_pred_phys = denormalize(state_pred, dataset.state_mean, dataset.state_std)
            state_phys = denormalize(state, dataset.state_mean, dataset.state_std)
            temporal_loss, temporal_metrics = temporal_reconstruction_loss(
                state_pred_phys,
                state_phys,
                loss_cfg,
            )
            cond_phys = dataset.denormalize_hands(cond)
            robot_loss, robot_metrics = robot_fk_hand_loss(
                state_pred_phys,
                cond_phys,
                loss_cfg,
            )
            contact_loss, contact_metrics = full_body_contact_loss(
                state_pred_phys,
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

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if global_step % 50 == 0:
                metrics = {
                    "base": float(base_loss.detach().cpu()),
                    **temporal_metrics,
                    **robot_metrics,
                    **contact_metrics,
                }
                print(
                    f"Epoch {epoch} Step {step} (global {global_step}): "
                    f"loss={loss.item():.6f} {format_metrics(metrics)}"
                )

            global_step += 1

        # Epoch stats
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f} (best={best_loss:.6f})")

        # Save checkpoint
        should_stop = (
            max_train_seconds is not None
            and time.monotonic() - train_start_time >= max_train_seconds
        )
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1 or should_stop:
            ckpt_file = os.path.join(ckpt_path, f"stage2_hf_epoch_{str(epoch).zfill(6)}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": yml,
                "generative_model": "ddpm",
                "norm_stats": {
                    "state_mean": dataset.state_mean,
                    "state_std": dataset.state_std,
                    "hand_mean": dataset.hand_mean,
                    "hand_std": dataset.hand_std,
                },
            }, ckpt_file)
            print(f"  Saved checkpoint: {ckpt_file}")

        # Plot loss
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(losses)), losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.title("Stage 2 DDPM Loss — HuggingFace Dataset")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figure_path, f"loss_epoch_{epoch}.png"), dpi=100)
            plt.close()

        if should_stop:
            elapsed = time.monotonic() - train_start_time
            print(f"Stopping after {elapsed:.1f}s due to train.max_train_seconds")
            break

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Final checkpoint: {ckpt_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
