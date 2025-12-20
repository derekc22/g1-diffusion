"""
Stage 2 Training Script: Hand Positions → Full-Body Motion (OMOMO)

Trains the Stage 2 diffusion model which generates full-body robot poses
conditioned on (rectified) hand positions from Stage 1.

Following the OMOMO paper:
- Stage 2 is trained using "human motion data only"
- It learns to generate plausible full-body poses that reach given hand positions
- At inference, hand positions come from Stage 1 with contact constraints applied

Usage:
    python train_stage2.py
"""

import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from datasets.g1_motion_dataset import G1MotionDatasetHandCond
from models.stage2_diffusion import Stage2TransformerModel, Stage2MLPModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.general import load_config, dump_config


def main():
    yml = load_config("./config/train_stage2.yaml")
    train_yml = yml["train"]
    dataset_yml = yml["dataset"]
    model_yml = yml["model"]

    root_dir = yml["root_dir"]

    save_dir = train_yml["save_dir"]
    batch_size = train_yml["batch_size"]
    timesteps = train_yml["timesteps"]
    num_epochs = train_yml["num_epochs"]
    lr = float(train_yml["lr"])
    device = train_yml["device"]
    architecture = train_yml["architecture"]
    save_every = train_yml.get("save_every", 100)

    window_size = dataset_yml["window_size"]
    stride = dataset_yml["stride"]
    min_seq_len = dataset_yml["min_seq_len"]
    train_split = dataset_yml["train_split"]
    preload = dataset_yml.get("preload", False)  # Default to False for large datasets

    # Model config
    d_model = model_yml["d_model"]
    nhead = model_yml["nhead"]
    num_layers = model_yml["num_layers"]
    dim_feedforward = model_yml["dim_feedforward"]
    dropout = model_yml["dropout"]

    # Create experiment directory
    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = f"stage2_e{num_epochs}_b{batch_size}_lr{lr}_ts{timesteps}_w{window_size}_s{stride}_{architecture}_"
    log_path = os.path.join(save_dir, exp_name + dtn)
    figure_path = os.path.join(log_path, "figures")
    ckpt_path = os.path.join(log_path, "checkpoints")

    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    dump_config(os.path.join(log_path, "config.yml"), yml)

    print(f"Experiment: {exp_name}")
    print(f"Logs: {log_path}")

    device = torch.device(device)

    # Create dataset
    print(f"\nLoading dataset from {root_dir}")
    dataset = G1MotionDatasetHandCond(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=min_seq_len,
        train=True,
        train_split=train_split,
        preload=preload,
        normalize_hands=False,
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

    # Create model
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
    global_step = 0
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            # Get data
            state = batch["state"].to(device)
            cond = batch["cond"].to(device)

            B = state.shape[0]

            # Sample timesteps
            t = torch.randint(0, timesteps, (B,), device=device)

            # Add noise to state
            noise = torch.randn_like(state)
            state_noisy = schedule.q_sample(state, t, noise)

            # Predict clean state
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
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            ckpt_file = os.path.join(ckpt_path, f"stage2_epoch_{str(epoch).zfill(6)}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": yml,
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
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("stage 2 loss")
            plt.grid(True)
            plt.savefig(os.path.join(figure_path, f"loss_epoch_{epoch}.png"))
            plt.close()

    print(f"\nTraining complete!")
    print(f"Final checkpoint: {ckpt_file}")


if __name__ == "__main__":
    main()
