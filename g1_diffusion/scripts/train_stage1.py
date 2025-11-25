import argparse
import os
import sys

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.g1_motion_dataset import G1MotionDataset
from models.stage1_task_space_diffusion import (
    Stage1TaskSpaceModel,
    Stage1TransformerModel,
)
from utils.diffusion import DiffusionConfig, DiffusionSchedule
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 DDPM on G1 states")
    parser.add_argument("--root_dir", type=str, default="../../GMR-master/export_smplx_retargeted")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="runs/stage1")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mlp", "transformer"],
        default="mlp",
        help="Backbone type for Stage-1 model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("./plots", exist_ok=True)

    device = torch.device(args.device)

    dataset = G1MotionDataset(
        root_dir=args.root_dir,
        window_size=args.window_size,
        stride=args.stride,
        normalize=True,
        train=True,
        train_split=0.9,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    sample = dataset[0]
    _, state_dim = sample["state"].shape
    cond_dim = sample["cond"].shape[1]

    if args.backbone == "mlp":
        model = Stage1TaskSpaceModel(state_dim=state_dim, cond_dim=cond_dim).to(device)
    else:
        model = Stage1TransformerModel(state_dim=state_dim, cond_dim=cond_dim).to(device)

    schedule = DiffusionSchedule(
        DiffusionConfig(
            timesteps=args.timesteps,
            beta_start=1e-4,
            beta_end=0.02,
        )
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    losses = []
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            x0 = batch["state"].to(device)
            cond = batch["cond"].to(device)
            B = x0.shape[0]

            t = torch.randint(0, args.timesteps, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)

            x0_pred = model(x_t, t, cond)
            loss = F.mse_loss(x0_pred, x0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                print(
                    f"Epoch {epoch} Step {step} (global {global_step}): loss={loss.item():.6f}"
                )
            global_step += 1

        ckpt_path = os.path.join(args.save_dir, f"stage1_epoch_{epoch}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": vars(args),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            losses.append(loss.item())
            plt.plot(np.arange(0, epoch+1), losses)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.grid(True)
            plt.savefig(f"./plots/loss_{epoch}")
            plt.close()



if __name__ == "__main__":
    main()
