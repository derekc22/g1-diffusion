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
from models.stage2_diffusion import Stage1TaskSpaceModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 DDPM on G1 states")
    parser.add_argument("--root_dir", type=str, default="../../g1-gmr/export_smplx_retargeted")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="runs/stage1")
    return parser.parse_args()


def main():
    args = parse_args()

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

    model = Stage1TaskSpaceModel(state_dim=state_dim).to(device)
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
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            x0 = batch["state"].to(device)
            B = x0.shape[0]

            t = torch.randint(0, args.timesteps, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)

            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                print(f"Epoch {epoch} Step {step} (global {global_step}): loss={loss.item():.6f}")
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


if __name__ == "__main__":
    main()
