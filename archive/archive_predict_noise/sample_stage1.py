import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.g1_motion_dataset import G1MotionDataset
from models.stage1_task_space_diffusion import Stage1TaskSpaceModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.rotation import rot6d_to_quat_wxyz


def load_latest_checkpoint(save_dir: str) -> str:
    ckpts = sorted(glob.glob(os.path.join(save_dir, "stage1_epoch_*.pt")))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {save_dir}")
    return ckpts[-1]


def main():
    parser = argparse.ArgumentParser(description="Sample Stage 1 DDPM and export robot motions")
    parser.add_argument("--root_dir", type=str, default="../../g1-gmr/export_smplx_retargeted")
    parser.add_argument("--save_dir", type=str, default="runs/stage1")
    parser.add_argument("--out_dir", type=str, default="samples/stage1_robot")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = G1MotionDataset(
        root_dir=args.root_dir,
        window_size=120,
        stride=10,
        normalize=True,
        train=True,
        train_split=0.9,
    )
    sample0 = dataset[0]
    T, D = sample0["state"].shape
    state_dim = D
    data_mean = dataset.mean.to(device)  # (D,)
    data_std = dataset.std.to(device)    # (D,)

    # load a reference robot motion file to get local_body_pos and link_body_list
    ref_files = sorted(glob.glob(os.path.join(args.root_dir, "*.pkl")))
    if not ref_files:
        raise RuntimeError(f"No reference motion files found in {args.root_dir}")
    with open(ref_files[0], "rb") as f:
        ref_motion = pickle.load(f)

    ref_local_body_pos = ref_motion.get("local_body_pos", None)
    ref_link_body_list = ref_motion.get("link_body_list", None)

    diff_config = DiffusionConfig(timesteps=args.timesteps, beta_start=1e-4, beta_end=0.02)
    schedule = DiffusionSchedule(diff_config).to(device)

    model = Stage1TaskSpaceModel(state_dim=state_dim).to(device)
    ckpt_path = load_latest_checkpoint(args.save_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    B = args.num_samples
    x_t = torch.randn(B, T, D, device=device)

    for t_idx in reversed(range(args.timesteps)):
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_pred = model(x_t, t)

        beta_t = schedule.beta[t_idx]
        alpha_t = schedule.alpha[t_idx]
        alpha_bar_t = schedule.alpha_bar[t_idx]
        alpha_bar_prev = schedule.alpha_bar[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=device)

        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        if t_idx > 0:
            coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            posterior_mean = coef1.unsqueeze(0).unsqueeze(0) * x0_pred + coef2.unsqueeze(0).unsqueeze(0) * x_t
            var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x_t)
            x_t = posterior_mean + torch.sqrt(var).unsqueeze(0).unsqueeze(0) * noise
        else:
            x_t = x0_pred

    x0_norm = x_t  # (B, T, D)

    mean_batched = data_mean.view(1, 1, -1)
    std_batched = data_std.view(1, 1, -1)
    x0 = x0_norm * std_batched + mean_batched

    x0_np = x0.detach().cpu().numpy()

    for i in range(B):
        state_i = x0_np[i]
        root_pos = state_i[:, 0:3]
        root_rot6d = state_i[:, 3:9]
        dof_pos = state_i[:, 9:]

        root_rot6d_t = torch.from_numpy(root_rot6d).float().to(device)
        quat_wxyz = rot6d_to_quat_wxyz(root_rot6d_t).cpu().numpy()

        motion_data = {
            "fps": float(sample0["fps"]),
            "root_pos": root_pos.astype(np.float32),
            "root_rot": quat_wxyz.astype(np.float32),
            "dof_pos": dof_pos.astype(np.float32),
        }

        if ref_local_body_pos is not None:
            ref_lbp = np.asarray(ref_local_body_pos, dtype=np.float32)
            T_ref = ref_lbp.shape[0]
            if T_ref >= T:
                local_body_pos = ref_lbp[:T]
            else:
                pad_len = T - T_ref
                last = ref_lbp[-1:, :, :]
                pad = np.repeat(last, pad_len, axis=0)
                local_body_pos = np.concatenate([ref_lbp, pad], axis=0)
            motion_data["local_body_pos"] = local_body_pos.astype(np.float32)
        else:
            motion_data["local_body_pos"] = np.zeros((T, 0, 3), dtype=np.float32)

        if ref_link_body_list is not None:
            motion_data["link_body_list"] = ref_link_body_list
        else:
            motion_data["link_body_list"] = []

        out_path = os.path.join(args.out_dir, f"stage1_sample_{i:03d}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved sample to {out_path}")


if __name__ == "__main__":
    main()
