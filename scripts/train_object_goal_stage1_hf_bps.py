"""
Train Stage 1 of the two-stage object-goal pipeline.

Target:
    hand trajectory H_{0:T}

Condition:
    HF-BPS object geometry/centroid, object pose trajectory O_{0:T}, and final
    full object pose g.
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

from datasets.hand_motion_dataset import HandMotionDataset
from models.stage1_diffusion import Stage1HandDiffusion, Stage1HandDiffusionMLP
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.general import dump_config, load_config
from utils.motion_losses import (
    contact_anchor_loss,
    denormalize,
    format_metrics,
    loss_config,
    temporal_reconstruction_loss,
)
from utils.object_goal_features import OBJECT_POSE_DIM
from utils.object_sampling import build_balanced_sampler, format_label_counts


def _make_model(architecture: str, model_cfg: dict, hand_dim: int, bps_dim: int, centroid_dim: int):
    common = {
        "bps_dim": bps_dim,
        "centroid_dim": centroid_dim,
        "encoder_hidden": int(model_cfg["encoder_hidden"]),
        "object_feature_dim": int(model_cfg["object_feature_dim"]),
        "encoder_layers": int(model_cfg["encoder_layers"]),
        "hand_dim": hand_dim,
        "object_pose_dim": OBJECT_POSE_DIM,
        "global_cond_dim": OBJECT_POSE_DIM,
        "global_cond_hidden": model_cfg.get("global_cond_hidden"),
    }
    if architecture == "transformer":
        return Stage1HandDiffusion(
            **common,
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            num_transformer_layers=int(model_cfg["num_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            max_len=int(model_cfg.get("max_len", 512)),
        )
    if architecture == "mlp":
        return Stage1HandDiffusionMLP(
            **common,
            denoiser_hidden=int(model_cfg.get("denoiser_hidden", 512)),
            denoiser_layers=int(model_cfg.get("denoiser_layers", 4)),
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


def _load_partial(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
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
    parser = argparse.ArgumentParser(description="Train object-goal Stage 1 HF-BPS prior")
    parser.add_argument(
        "--config_path",
        default=os.path.join(PROJECT_ROOT, "config", "train_object_goal_stage1_hf_bps.yaml"),
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    train_cfg = yml["train"]
    dataset_cfg = yml["dataset"]
    model_cfg = yml["model"]
    loss_cfg = loss_config(yml, "stage1")

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
    exp_prefix = train_cfg.get("exp_prefix", "object_goal_stage1_hf_bps")
    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = f"{exp_prefix}_e{num_epochs}_b{batch_size}_lr{lr}_ts{timesteps}_w{window_size}_s{stride}_{architecture}_"
    log_path = os.path.join(save_dir, exp_name + dtn)
    figure_path = os.path.join(log_path, "figures")
    ckpt_path = os.path.join(log_path, "checkpoints")
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    dump_config(os.path.join(log_path, "config.yml"), yml)

    include_contact_data = (
        loss_cfg["contact_weight"] > 0.0
        or loss_cfg["contact_offset_weight"] > 0.0
        or loss_cfg["contact_distance_weight"] > 0.0
    )
    dataset = HandMotionDataset(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=int(dataset_cfg.get("min_seq_len", 30)),
        train=True,
        train_split=float(dataset_cfg.get("train_split", 0.99)),
        preload=bool(dataset_cfg.get("preload", True)),
        flatten_bps=True,
        include_contact_data=include_contact_data,
        contact_threshold=float(loss_cfg["contact_margin"]),
        object_conditioning_variant=dataset_cfg.get("object_conditioning_variant", "variant0"),
        include_object_pose_goal=True,
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
    hand_dim = sample["hand_positions"].shape[-1]
    bps_dim = sample["bps_encoding"].shape[-1]
    centroid_dim = sample["object_centroid"].shape[-1]
    model = _make_model(architecture, model_cfg, hand_dim, bps_dim, centroid_dim).to(device)
    print("=" * 60)
    print("Object-goal Stage 1 training")
    print("=" * 60)
    print(f"Logs: {log_path}")
    print(f"Windows: {len(dataset)} from {dataset.num_files} files")
    print("Target: hands (6D)")
    print("Condition: BPS/centroid + object pose trajectory + final object pose")
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
            hand_pos = batch["hand_positions"].to(device)
            bps = batch["bps_encoding"].to(device)
            centroid = batch["object_centroid"].to(device)
            object_pose = batch["object_pose"].to(device)
            goal = batch["goal"].to(device)
            B = hand_pos.shape[0]

            t = torch.randint(0, timesteps, (B,), device=device)
            noise = torch.randn_like(hand_pos)
            hand_noisy = schedule.q_sample(hand_pos, t, noise)
            hand_pred = model(
                hand_noisy,
                t,
                bps,
                centroid,
                object_pose=object_pose,
                global_cond=goal,
            )

            base_loss = F.mse_loss(hand_pred, hand_pos)
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
                    **temporal_metrics,
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
            ckpt_file = os.path.join(ckpt_path, f"object_goal_stage1_epoch_{epoch:06d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": yml,
                    "generative_model": "ddpm",
                    "pipeline_type": "object_goal_two_stage",
                    "stage": 1,
                    "stage_target": "hands",
                    "prediction_type": "x0",
                    "condition": {
                        "object_geometry": "bps_encoding + object_centroid",
                        "object_pose_trajectory_dim": OBJECT_POSE_DIM,
                        "goal_dim": OBJECT_POSE_DIM,
                    },
                    "norm_stats": {
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
            plt.title("Object-goal Stage 1 loss")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figure_path, f"loss_epoch_{epoch}.png"), dpi=100)
            plt.close()

        if should_stop:
            elapsed = time.monotonic() - train_start
            print(f"Stopping after {elapsed:.1f}s due to train.max_train_seconds")
            break

    print("=" * 60)
    print("Object-goal Stage 1 training complete")
    print(f"Final checkpoint: {ckpt_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
