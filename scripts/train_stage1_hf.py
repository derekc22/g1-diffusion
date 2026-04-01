"""
Stage 1 DDPM Training Script — HuggingFace Dataset
Object Motion Features → Hand Positions

Trains the Stage 1 diffusion model which:
1. Encodes object motion features (15D) via MLP → object features (256D)
2. Generates hand positions (6D) via transformer denoiser

Uses standard DDPM with x0-prediction:
    q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Loss = ||f_theta(x_t, t, cond) - x_0||^2

Usage:
    python scripts/train_stage1_hf.py
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

from datasets.hf_motion_dataset import HFHandMotionDataset
from models.stage1_hf_diffusion import Stage1HFHandDiffusion, Stage1HFHandDiffusionMLP
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.general import load_config, dump_config


def main():
    yml = load_config(os.path.join(PROJECT_ROOT, "config", "train_stage1_hf.yaml"))

    train_yml = yml["train"]
    dataset_yml = yml["dataset"]
    model_yml = yml["model"]

    root_dir = yml["root_dir"]
    # Resolve relative paths against project root
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
    save_every = train_yml.get("save_every", 100)

    # Dataset config
    window_size = dataset_yml["window_size"]
    stride = dataset_yml["stride"]
    min_seq_len = dataset_yml.get("min_seq_len", 30)
    train_split = dataset_yml.get("train_split", 0.9)
    preload = dataset_yml.get("preload", True)
    require_object = dataset_yml.get("require_object", False)

    # Model config
    object_feature_input_dim = model_yml.get("object_feature_input_dim", 15)
    encoder_hidden = model_yml["encoder_hidden"]
    encoder_layers = model_yml["encoder_layers"]
    object_feature_dim = model_yml["object_feature_dim"]
    d_model = model_yml["d_model"]
    nhead = model_yml["nhead"]
    num_layers = model_yml["num_layers"]
    dim_feedforward = model_yml["dim_feedforward"]
    dropout = model_yml["dropout"]

    # Create experiment directory
    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = f"stage1_hf_e{num_epochs}_b{batch_size}_lr{lr}_ts{timesteps}_w{window_size}_s{stride}_{architecture}_"
    log_path = os.path.join(save_dir, exp_name + dtn)
    figure_path = os.path.join(log_path, "figures")
    ckpt_path = os.path.join(log_path, "checkpoints")

    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    dump_config(os.path.join(log_path, "config.yml"), yml)

    print(f"{'='*60}")
    print(f"Stage 1 DDPM Training — HuggingFace Dataset")
    print(f"{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Logs: {log_path}")
    print(f"Generative model: DDPM (x0-prediction)")

    device = torch.device(device_str)

    # Create dataset
    print(f"\nLoading dataset from {root_dir}")
    dataset = HFHandMotionDataset(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=min_seq_len,
        train=True,
        train_split=train_split,
        preload=preload,
        require_object=require_object,
    )

    print(f"  Windows: {len(dataset)}")
    print(f"  Files: {dataset.num_files}")
    print(f"  Hand mean shape: {dataset.hand_mean.shape}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # Get dimensions from sample
    sample = dataset[0]
    hand_dim = sample["hand_positions"].shape[-1]   # 6
    obj_feat_dim = sample["object_features"].shape[-1]  # 15

    print(f"\n  Hand dim: {hand_dim}")
    print(f"  Object feature dim: {obj_feat_dim}")

    # Create model
    print(f"\nCreating Stage 1 HF {architecture} model")

    if architecture == "transformer":
        model = Stage1HFHandDiffusion(
            object_feature_input_dim=obj_feat_dim,
            encoder_hidden=encoder_hidden,
            object_feature_dim=object_feature_dim,
            encoder_layers=encoder_layers,
            hand_dim=hand_dim,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=window_size + 100,
        ).to(device)
    else:
        model = Stage1HFHandDiffusionMLP(
            object_feature_input_dim=obj_feat_dim,
            encoder_hidden=encoder_hidden,
            object_feature_dim=object_feature_dim,
            encoder_layers=encoder_layers,
            hand_dim=hand_dim,
            denoiser_hidden=512,
            denoiser_layers=4,
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
            hand_pos = batch["hand_positions"].to(device)    # (B, T, 6) normalized
            obj_feat = batch["object_features"].to(device)   # (B, T, 15)

            B = hand_pos.shape[0]

            # Sample random diffusion timesteps
            t = torch.randint(0, timesteps, (B,), device=device)

            # Add noise to hand positions
            noise = torch.randn_like(hand_pos)
            hand_noisy = schedule.q_sample(hand_pos, t, noise)

            # Predict clean hand positions (x0 prediction)
            hand_pred = model(hand_noisy, t, obj_feat)

            # Loss
            loss = F.mse_loss(hand_pred, hand_pos)

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
            ckpt_file = os.path.join(ckpt_path, f"stage1_hf_epoch_{str(epoch).zfill(6)}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": yml,
                "generative_model": "ddpm",
                "norm_stats": {
                    "hand_mean": dataset.hand_mean,
                    "hand_std": dataset.hand_std,
                },
            }, ckpt_file)
            print(f"  Saved checkpoint: {ckpt_file}")

        # Plot loss curve
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(losses)), losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.title("Stage 1 DDPM Loss — HuggingFace Dataset")
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
