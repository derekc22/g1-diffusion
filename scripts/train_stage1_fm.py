"""
Stage 1 Flow Matching Training Script: Object Geometry → Hand Positions

Trains the Stage 1 flow matching model which:
1. Encodes object geometry (BPS + centroid) via MLP → object features (256D)
2. Generates hand positions (6D) via transformer velocity predictor

Uses OT-CFM (Optimal Transport Conditional Flow Matching):
- Linear interpolation: x_t = (1-t)*x_1 + t*x_0 where x_1 is data, x_0 is noise
- Velocity target: v = x_0 - x_1
- Loss: ||v_theta(x_t, t) - v||^2

Usage:
    python train_stage1_fm.py
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure project root is on sys.path
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

from datasets.hand_motion_dataset import HandMotionDataset
from models.stage1_flow_matching import Stage1HandFlowMatching, Stage1HandFlowMatchingMLP
from utils.flow_matching import FlowMatchingConfig, FlowMatchingSchedule
from utils.general import load_config, dump_config
from utils.object_conditioning import (
    describe_object_conditioning_variant,
    normalize_object_conditioning_variant,
)
from utils.motion_losses import (
    contact_anchor_loss,
    denormalize,
    format_metrics,
    loss_config,
    temporal_reconstruction_loss,
)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Flow Matching Training")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "train_stage1_fm.yaml"),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)

    train_yml = yml["train"]
    dataset_yml = yml["dataset"]
    model_yml = yml["model"]
    loss_cfg = loss_config(yml, "stage1")

    root_dir = yml["root_dir"]

    save_dir = train_yml["save_dir"]
    batch_size = train_yml["batch_size"]
    num_epochs = train_yml["num_epochs"]
    lr = float(train_yml["lr"])
    device = train_yml["device"]
    architecture = train_yml["architecture"]
    save_every = train_yml.get("save_every", 100)

    # Flow matching config
    sigma_min = train_yml.get("sigma_min", 1e-4)
    timestep_sampling = train_yml.get("timestep_sampling", "uniform")
    logit_normal_mean = train_yml.get("logit_normal_mean", 0.0)
    logit_normal_std = train_yml.get("logit_normal_std", 1.0)
    beta_alpha = train_yml.get("beta_alpha", 2.0)
    beta_beta = train_yml.get("beta_beta", 2.0)

    # Dataset config
    window_size = dataset_yml["window_size"]
    stride = dataset_yml["stride"]
    min_seq_len = dataset_yml.get("min_seq_len", 30)
    train_split = dataset_yml.get("train_split", 0.9)
    preload = dataset_yml.get("preload", False)
    object_conditioning_variant = normalize_object_conditioning_variant(
        dataset_yml.get("object_conditioning_variant", "variant0")
    )
    dataset_yml["object_conditioning_variant"] = object_conditioning_variant

    # Model config
    encoder_hidden = model_yml["encoder_hidden"]
    encoder_layers = model_yml["encoder_layers"]
    object_feature_dim = model_yml["object_feature_dim"]

    # Create experiment directory
    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = (
        f"stage1_fm_{object_conditioning_variant}_e{num_epochs}_b{batch_size}_lr{lr}_"
        f"w{window_size}_s{stride}_{architecture}_"
    )
    log_path = os.path.join(save_dir, exp_name + dtn)
    figure_path = os.path.join(log_path, "figures")
    ckpt_path = os.path.join(log_path, "checkpoints")

    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    dump_config(os.path.join(log_path, "config.yml"), yml)

    print(f"Experiment: {exp_name}")
    print(f"Logs: {log_path}")
    print(f"Generative model: Flow Matching (OT-CFM)")
    print(f"Object conditioning: {describe_object_conditioning_variant(object_conditioning_variant)}")

    device = torch.device(device)

    # Create dataset (same as DDPM version)
    print(f"\nLoading dataset from {root_dir}")
    dataset = HandMotionDataset(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=min_seq_len,
        train=True,
        train_split=train_split,
        preload=preload,
        flatten_bps=True,
        include_contact_data=(
            loss_cfg["contact_weight"] > 0.0
            or loss_cfg["contact_offset_weight"] > 0.0
            or loss_cfg["contact_distance_weight"] > 0.0
        ),
        contact_threshold=loss_cfg["contact_margin"],
        object_conditioning_variant=object_conditioning_variant,
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
    hand_dim = sample["hand_positions"].shape[-1]  # 6
    bps_dim = sample["bps_encoding"].shape[-1]     # 3072
    centroid_dim = sample["object_centroid"].shape[-1]  # 3

    print(f"\n  Hand dim: {hand_dim}")
    print(f"  BPS dim: {bps_dim}")
    print(f"  Centroid dim: {centroid_dim}")

    # Create model
    print(f"\nCreating Stage 1 FM {architecture} model")

    if architecture == "transformer":
        model = Stage1HandFlowMatching(
            bps_dim=bps_dim,
            centroid_dim=centroid_dim,
            encoder_hidden=encoder_hidden,
            object_feature_dim=object_feature_dim,
            encoder_layers=encoder_layers,
            hand_dim=hand_dim,
            d_model=model_yml["d_model"],
            nhead=model_yml["nhead"],
            num_transformer_layers=model_yml["num_layers"],
            dim_feedforward=model_yml["dim_feedforward"],
            dropout=model_yml["dropout"],
            max_len=window_size + 100,
        ).to(device)
    else:
        model = Stage1HandFlowMatchingMLP(
            bps_dim=bps_dim,
            centroid_dim=centroid_dim,
            encoder_hidden=encoder_hidden,
            object_feature_dim=object_feature_dim,
            encoder_layers=encoder_layers,
            hand_dim=hand_dim,
            denoiser_hidden=model_yml.get("denoiser_hidden", 512),
            denoiser_layers=model_yml.get("denoiser_layers", 4),
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Flow matching schedule
    fm_config = FlowMatchingConfig(
        sigma_min=sigma_min,
        timestep_sampling=timestep_sampling,
        logit_normal_mean=logit_normal_mean,
        logit_normal_std=logit_normal_std,
        beta_alpha=beta_alpha,
        beta_beta=beta_beta,
    )
    schedule = FlowMatchingSchedule(fm_config)
    print(f"  Timestep sampling: {timestep_sampling}")

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
            hand_pos = batch["hand_positions"].to(device)   # (B, T, 6) - normalized
            bps = batch["bps_encoding"].to(device)          # (B, T, 3072)
            centroid = batch["object_centroid"].to(device)   # (B, T, 3)

            B = hand_pos.shape[0]

            # === Flow Matching Training Step ===
            # 1. Sample noise x_0 ~ N(0, I)
            x0 = torch.randn_like(hand_pos)

            # 2. Sample timestep t ~ U[0, 1]
            t = schedule.sample_timesteps(B, device)

            # 3. Compute interpolated sample: x_t = (1-t)*x_1 + t*x_0
            x_t = schedule.interpolate(hand_pos, x0, t)

            # 4. Compute velocity target: v = x_0 - x_1
            velocity_target = schedule.compute_velocity_target(hand_pos, x0)

            # 5. Predict velocity
            velocity_pred = model(x_t, t, bps, centroid)

            # 6. Loss: MSE between predicted and target velocity.
            base_loss = F.mse_loss(velocity_pred, velocity_target)
            hand_pred = x0 - velocity_pred
            hand_pred_phys = denormalize(hand_pred, dataset.hand_mean, dataset.hand_std)
            hand_pos_phys = denormalize(hand_pos, dataset.hand_mean, dataset.hand_std)
            temporal_loss, temporal_metrics = temporal_reconstruction_loss(
                hand_pred_phys,
                hand_pos_phys,
                loss_cfg,
            )
            contact_loss, contact_metrics = contact_anchor_loss(
                hand_pred_phys,
                batch,
                loss_cfg,
            )
            loss = loss_cfg["base_weight"] * base_loss + temporal_loss + contact_loss

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
                    **contact_metrics,
                }
                print(
                    f"Epoch {epoch} Step {step} (global {global_step}): "
                    f"loss={loss.item():.6f} {format_metrics(metrics)}"
                )

            global_step += 1

        # Epoch stats
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            ckpt_file = os.path.join(ckpt_path, f"stage1_fm_epoch_{str(epoch).zfill(6)}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": yml,
                "generative_model": "flow_matching",
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
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Stage 1 Flow Matching Loss")
            plt.grid(True)
            plt.savefig(os.path.join(figure_path, f"loss_epoch_{epoch}.png"))
            plt.close()

    print(f"\nTraining complete!")
    print(f"Final checkpoint: {ckpt_file}")


if __name__ == "__main__":
    main()
