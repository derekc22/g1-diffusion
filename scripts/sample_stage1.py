import glob
import os
import pickle
import sys
from datetime import datetime
import re

import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.g1_motion_dataset import G1MotionDataset
from models.stage1_task_space_diffusion import Stage1MLPModel, Stage1TransformerModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.rotation import rot6d_to_quat_xyzw
from utils.general import load_config, dump_config


def load_latest_checkpoint(model_dir: str) -> str:
    ckpts = sorted(glob.glob(os.path.join(model_dir, "checkpoints", "stage1_epoch_*.pt")))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {model_dir}")
    return ckpts[-1]

def main():

    yml = load_config("./config/sample.yaml")
    sample_yml = yml["sample"]
    dataset_yml = yml["dataset"]

    root_dir      = yml["root_dir"]

    model_dir      = sample_yml["model_dir"]
    # save_dir       = sample_yml["save_dir"]
    num_samples   = sample_yml["num_samples"]
    timesteps     = sample_yml["timesteps"]
    device        = sample_yml["device"]
    ckpt_path     = sample_yml["ckpt_path"]   # optional explicit checkpoint path; defaults to latest in model_dir
    # architecture = sample_yml["architecture"]  # backbone architecture; choose between ["mlp", "transformer"]
    random_samples         = sample_yml["random_samples"]
    seed =  sample_yml["seed"]

    window_size     = dataset_yml["window_size"]
    stride          = dataset_yml["stride"]
    min_seq_len     = dataset_yml["min_seq_len"]
    # normalize       = dataset_yml["normalize"]
    train           = dataset_yml["train"]
    train_split     = dataset_yml["train_split"]
    preload         = dataset_yml["preload"]    

 

    ckpt_path = ckpt_path if ckpt_path else load_latest_checkpoint(model_dir)
    pattern = r"_epoch_(\d+)"
    match = re.search(pattern, ckpt_path)
    epoch_num = int(match.group(1))

    save_dir = os.path.join(model_dir, "samples")
    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    EXP_NAME = f"ts{timesteps}_w{window_size}_s{stride}_ckpt{epoch_num}_"

    log_path = os.path.join(save_dir, EXP_NAME + dtn)
    dump_path = os.path.join(log_path, "config.yml")

    os.makedirs(log_path, exist_ok=True)
    os.makedirs("./videos", exist_ok=True)
    dump_config(dump_path, yml)

    device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_config = ckpt.get("config", {}) or {}
    architecture = ckpt_config.get("train").get("architecture")

    # ckpt_architecture = ckpt_config.get("train").get("architecture")
    # if ckpt_architecture != architecture:
    #     print(
    #         f"Warning: checkpoint architecture '{ckpt_architecture}' differs from requested '{architecture}'. Using checkpoint architecture."
    #     )
    # architecture = ckpt_architecture

    norm_stats = ckpt.get("norm_stats")
    dataset_mean = norm_stats.get("mean")
    dataset_std = norm_stats.get("std")
    # print(dataset_mean.device)
    # print(dataset_std.device)
    # exit()
    
    dataset = G1MotionDataset(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=min_seq_len,
        # normalize=normalize,
        train=train,
        train_split=train_split,
        preload=preload,
        mean=dataset_mean.cpu(),
        std=dataset_std.cpu(),
    )
    sample0 = dataset[0]
    T, state_dim = sample0["state"].shape
    dataset_cond_dim = sample0["cond"].shape[-1]
    # dataset_mean = dataset.mean.to(device)  # (D,)
    # dataset_std = dataset.std.to(device)    # (D,)

    # # load a reference robot motion file to get local_body_pos and link_body_list
    # ref_files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    # if not ref_files:
    #     raise RuntimeError(f"No reference motion files found in {root_dir}")
    # with open(ref_files[0], "rb") as f:
    #     ref_motion = pickle.load(f)

    # ref_local_body_pos = ref_motion.get("local_body_pos", None)
    # ref_link_body_list = ref_motion.get("link_body_list", None)

    diff_config = DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
    schedule = DiffusionSchedule(diff_config).to(device)

    def infer_cond_dim(model_state: dict, state_dim_val: int) -> int:
        if "state_proj.weight" in model_state:
            weight = model_state["state_proj.weight"]
        elif "mlp.0.weight" in model_state:
            weight = model_state["mlp.0.weight"]
        else:
            return 0
        input_dim = weight.shape[1]
        cond_dim_est = int(input_dim - state_dim_val)
        # if cond_dim_est < 0:
        #     raise RuntimeError(
        #         f"Checkpoint input dim ({input_dim}) smaller than state_dim ({state_dim_val})."
        #     )
        return cond_dim_est

    model_cond_dim = infer_cond_dim(ckpt["model"], state_dim)
    # if model_cond_dim > dataset_cond_dim:
    #     raise RuntimeError(
    #         f"Checkpoint expects cond_dim={model_cond_dim} but dataset only provides {dataset_cond_dim}."
    #     )
    # if 0 < model_cond_dim < dataset_cond_dim:
    #     print(
    #         f"Info: checkpoint cond_dim={model_cond_dim}, dataset cond_dim={dataset_cond_dim}. Using the first {model_cond_dim} conditioning features."
    #     )

    if architecture == "mlp":
        model = Stage1MLPModel(state_dim=state_dim, cond_dim=model_cond_dim).to(device)
    else:
        model = Stage1TransformerModel(state_dim=state_dim, cond_dim=model_cond_dim).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    B = num_samples

    rng = np.random.default_rng() if random_samples else np.random.default_rng(seed=seed)
    indices = rng.integers(0, len(dataset), size=B)
    # indices = np.random.randint(0, len(dataset), size=B)
    cond_tensors = []
    window_meta = []

    for idx in indices:
        sample_i = dataset[int(idx)]
        cond_tensor_full = sample_i["cond"]  # (T, dataset_cond_dim)
        # if cond_tensor_full.shape[0] != T:
        #     raise ValueError(
        #         f"Conditioning window length {cond_tensor_full.shape[0]} does not match state length {T}"
        #     )

        if model_cond_dim > 0:
            # if cond_tensor_full.shape[1] < model_cond_dim:
            #     raise ValueError(
            #         f"Conditioning dim {cond_tensor_full.shape[1]} smaller than model requirement {model_cond_dim}"
            #     )
            cond_tensors.append(cond_tensor_full[:, :model_cond_dim])

        cond_np = cond_tensor_full.cpu().numpy().astype(np.float32)
        window_meta.append(
            {
                "fps": float(sample_i["fps"]),
                "seq_name": sample_i.get("seq_name", f"window_{idx}"),
                "file_idx": int(sample_i["file_idx"]),
                "start": int(sample_i["start"]),
                "object_pos": cond_np[:, :3],
                "object_rot_6d": cond_np[:, 3:9],
            }
        )

    cond_batch = None
    if model_cond_dim > 0:
        cond_batch = torch.stack(cond_tensors, dim=0).to(device)

    x_t = torch.randn(B, T, state_dim, device=device)
    cond_model_input = cond_batch if model_cond_dim > 0 else None

    for t_idx in reversed(range(timesteps)):
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        with torch.no_grad():
            # Model now directly predicts x0 in normalized space
            x0_pred = model(x_t, t, cond=cond_model_input)

        beta_t = schedule.beta[t_idx]
        alpha_t = schedule.alpha[t_idx]
        alpha_bar_t = schedule.alpha_bar[t_idx]
        alpha_bar_prev = schedule.alpha_bar[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=device)

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

    # print(dataset_mean.shape)
    # print(x0_norm.shape)
    mean_batched = dataset_mean.view(1, 1, -1)
    # print(mean_batched.shape)
    std_batched = dataset_std.view(1, 1, -1)
    # exit()
    x0 = x0_norm * std_batched + mean_batched

    x0_np = x0.detach().cpu().numpy()

    for i in range(B):
        state_i = x0_np[i]
        meta_i = window_meta[i]

        root_pos = state_i[:, 0:3]
        root_rot6d = state_i[:, 3:9]
        dof_pos = state_i[:, 9:]
        root_rot6d_t = torch.from_numpy(root_rot6d).float().to(device)
        root_quat_xyzw = rot6d_to_quat_xyzw(root_rot6d_t).cpu().numpy()

        obj_pos = meta_i["object_pos"]
        obj_rot6d = meta_i["object_rot_6d"]
        obj_rot6d_t = torch.from_numpy(obj_rot6d).float().to(device)
        obj_quat_xyzw = rot6d_to_quat_xyzw(obj_rot6d_t).cpu().numpy()

        fps_i = float(meta_i["fps"])

        motion_data = {
            "fps": fps_i,
            "root_pos": root_pos.astype(np.float32),
            "root_rot": root_quat_xyzw.astype(np.float32),
            "dof_pos": dof_pos.astype(np.float32),
            "object_pos": obj_pos.astype(np.float32),
            "object_rot": obj_quat_xyzw.astype(np.float32),
            "source_seq_name": meta_i["seq_name"],
            "source_file_idx": meta_i["file_idx"],
            "source_start": meta_i["start"],
            "local_body_pos": np.zeros((T, 0, 3), dtype=np.float32),
            "link_body_list": []
        }

        # if ref_local_body_pos is not None:
        #     ref_lbp = np.asarray(ref_local_body_pos, dtype=np.float32)
        #     T_ref = ref_lbp.shape[0]
        #     if T_ref >= T:
        #         local_body_pos = ref_lbp[:T]
        #     else:
        #         pad_len = T - T_ref
        #         last = ref_lbp[-1:, :, :]
        #         pad = np.repeat(last, pad_len, axis=0)
        #         local_body_pos = np.concatenate([ref_lbp, pad], axis=0)
        #     motion_data["local_body_pos"] = local_body_pos.astype(np.float32)
        # else:
            # motion_data["local_body_pos"] = np.zeros((T, 0, 3), dtype=np.float32)

        # if ref_link_body_list is not None:
        #     motion_data["link_body_list"] = ref_link_body_list
        # else:
        #     motion_data["link_body_list"] = []

        seq_for_name = meta_i.get("seq_name") or f"stage1_seq_{i:03d}"
        safe_base = seq_for_name.replace("/", "_")
        out_name = f"{safe_base}_sample_{i:03d}.pkl"
        out_path = os.path.join(log_path, out_name)
        with open(out_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved sample to {out_path}")


if __name__ == "__main__":
    main()
