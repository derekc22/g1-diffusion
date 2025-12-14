import os
import sys
from datetime import datetime

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.g1_motion_dataset import G1MotionDataset
from models.stage2_task_space_diffusion import (
    Stage2MLPModel,
    Stage2TransformerModel,
)
from utils.diffusion import DiffusionConfig, DiffusionSchedule
import matplotlib.pyplot as plt
import numpy as np
from utils.general import load_config, dump_config


def main():
    yml = load_config("./config/train.yaml")
    train_yml = yml["train"]
    dataset_yml = yml["dataset"]

    root_dir      = yml["root_dir"]

    save_dir      = train_yml["save_dir"]
    batch_size    = train_yml["batch_size"]
    timesteps     = train_yml["timesteps"]
    num_epochs    = train_yml["num_epochs"]
    lr            = float(train_yml["lr"])
    device        = train_yml["device"]
    architecture = train_yml["architecture"]  # backbone architecture; choose between ["mlp", "transformer"]

    window_size     = dataset_yml["window_size"]
    stride          = dataset_yml["stride"]
    min_seq_len     = dataset_yml["min_seq_len"]
    # normalize       = dataset_yml["normalize"]
    train           = dataset_yml["train"]
    train_split     = dataset_yml["train_split"]
    preload         = dataset_yml["preload"]    

    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    EXP_NAME = f"e{num_epochs}_b{batch_size}_lr{lr}_ts{timesteps}_w{window_size}_s{stride}_{architecture}_"  

    log_path = os.path.join(save_dir, EXP_NAME + dtn)
    figure_path = os.path.join(log_path, "figures")
    ckpt_path = os.path.join(log_path, "checkpoints")
    dump_path = os.path.join(log_path, "config.yml")

    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    dump_config(dump_path, yml)

    device = torch.device(device)

    dataset = G1MotionDataset(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=min_seq_len,
        # normalize=normalize,
        train=train,
        train_split=train_split,
        preload=preload
    )
    print(dataset.mean.device)
    print(dataset.std.device)
    # exit()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    assert (batch_size <= dataset.num_files), "Requested batch size exceeds the number of files in dataset"

    sample = dataset[0]
    _, state_dim = sample["state"].shape
    cond_dim = sample["cond"].shape[1]

    if architecture == "mlp":
        model = Stage2MLPModel(state_dim=state_dim, cond_dim=cond_dim).to(device)
    elif architecture == "transformer":
        model = Stage2TransformerModel(state_dim=state_dim, cond_dim=cond_dim).to(device)

    schedule = DiffusionSchedule(
        DiffusionConfig(
            timesteps=timesteps,
            beta_start=1e-4,
            beta_end=0.02,
        )
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    losses = []
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            x0 = batch["state"].to(device)
            cond = batch["cond"].to(device)
            B = x0.shape[0]

            t = torch.randint(0, timesteps, (B,), device=device)
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

        ckpt_fpath = os.path.join(ckpt_path, f"stage2_epoch_{str(epoch).zfill(7)}.pt")
        # print(dataset.mean.device)
        # print(dataset.std.device)
        # exit()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": yml,
                "norm_stats": {"mean": dataset.mean, "std": dataset.std},
            },
            ckpt_fpath,
        )
        print(f"Saved checkpoint to {ckpt_fpath}")
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            plt.plot(np.arange(0, epoch+1), losses)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.grid(True)
            plt.savefig(f"{figure_path}/loss_{epoch}")
            plt.close()



if __name__ == "__main__":
    main()
