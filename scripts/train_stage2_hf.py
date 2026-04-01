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


def main():
    yml = load_config(os.path.join(PROJECT_ROOT, "config", "train_stage2_hf.yaml"))

    train_yml = yml["train"]
    dataset_yml = yml["dataset"]
    model_yml = yml["model"]

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

    window_size = dataset_yml["window_size"]
    stride = dataset_yml["stride"]
    min_seq_len = dataset_yml["min_seq_len"]
    train_split = dataset_yml["train_split"]
    preload = dataset_yml.get("preload", True)

    # Model config
    d_model = model_yml["d_model"]
    nhead = model_yml["nhead"]
    num_layers = model_yml["num_layers"]
    dim_feedforward = model_yml["dim_feedforward"]
    dropout = model_yml["dropout"]

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
        normalize_hands=False,  # Same as original — hands not normalized
    )

    print(f"  Windows: {len(dataset)}")
    print(f"  Files: {dataset.num_files}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=window_size + 100,
        ).to(device)
    else:
        model = Stage2MLPModel(
            state_dim=state_dim,
            cond_dim=cond_dim,
            hidden_dim=512,
            num_layers=4,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Diffusion schedule
    schedule = DiffusionSchedule(
        DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs")
    print(f"  Batch size: {batch_size}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Learning rate: {lr}")
    print()

    global_step = 0
    losses = []
    best_loss = float("inf")

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
            state_pred = model(state_noisy, t, cond)

            # Loss
            loss = F.mse_loss(state_pred, state)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if global_step % 50 == 0:
                print(f"Epoch {epoch} Step {step} (global {global_step}): loss={loss.item():.6f}")

            global_step += 1

        # Epoch stats
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f} (best={best_loss:.6f})")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
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

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Final checkpoint: {ckpt_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
