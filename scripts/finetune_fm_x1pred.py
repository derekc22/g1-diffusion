"""
Quick fine-tune test: x1-prediction (data prediction) vs velocity prediction.

The hypothesis: velocity prediction at t≈0 is ill-conditioned because predicting
noise from clean data is impossible. Data prediction at t≈0 is trivial (identity).

This test fine-tunes the existing FM checkpoint with x1-prediction loss for 50 epochs
and compares against the velocity-prediction baseline.

If this shows clear improvement → full retrain with x1-prediction is justified.
If not → FM fundamentally can't compete with DDIM for this task.

Usage:
    python scripts/finetune_fm_x1pred.py
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time

from datasets.g1_motion_dataset import G1MotionDatasetHandCond
from models.stage2_flow_matching import Stage2FMTransformerModel
from models.stage2_diffusion import Stage2TransformerModel
from utils.flow_matching import FlowMatchingSchedule, FlowMatchingConfig, euler_solve
from utils.inference_optimization import DDIMSampler
from utils.diffusion import DiffusionSchedule, DiffusionConfig
from utils.general import load_config


@torch.no_grad()
def evaluate_all(model, dataset, device, state_dim, label, prediction_type="velocity",
                 ddpm_model=None, ddpm_timesteps=None, num_eval=10):
    """
    Comprehensive evaluation:
    1. Velocity MSE at key timesteps
    2. Implied x1-prediction MSE at key timesteps
    3. Generation MSE at various step counts
    4. Head-to-head against DDIM-2
    """
    model.eval()
    t_values = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]

    print(f"\n--- {label} ---")

    # 1. Velocity + data prediction accuracy at each timestep
    print(f"  {'t':>6s}  {'vel_MSE':>10s}  {'x1_MSE':>10s}")
    for t_val in t_values:
        vel_errors = []
        x1_errors = []
        for i in range(min(num_eval, len(dataset))):
            sample = dataset[i]
            state = sample["state"].unsqueeze(0).to(device)  # x_1 (data)
            cond = sample["cond"].unsqueeze(0).to(device)

            torch.manual_seed(42 + i)
            noise = torch.randn_like(state)  # x_0

            t = torch.full((1,), t_val, device=device)
            x_t = (1.0 - t_val) * state + t_val * noise

            velocity_target = noise - state

            if prediction_type == "velocity":
                v_pred = model(x_t, t, cond)
                # Implied data prediction
                x1_pred = x_t - t_val * v_pred
            else:  # x1-prediction
                x1_pred = model(x_t, t, cond)
                # Implied velocity
                v_pred = (x_t - x1_pred) / max(t_val, 1e-6)

            vel_errors.append(F.mse_loss(v_pred, velocity_target).item())
            x1_errors.append(F.mse_loss(x1_pred, state).item())

        print(f"  {t_val:>6.2f}  {np.mean(vel_errors):>10.4f}  {np.mean(x1_errors):>10.4f}")

    # 2. Generation quality at various step counts
    print(f"\n  Generation MSE (Euler):")
    for n_steps in [2, 10, 20, 50]:
        mses = []
        for i in range(min(num_eval, len(dataset))):
            sample = dataset[i]
            state_gt = sample["state"].unsqueeze(0).to(device)
            cond = sample["cond"].unsqueeze(0).to(device)
            T = state_gt.shape[1]

            torch.manual_seed(42 + i)
            x_init = torch.randn(1, T, state_dim, device=device)

            if prediction_type == "velocity":
                def model_fn(x_t, t_tensor, c=cond):
                    return model(x_t, t_tensor, c)
            else:
                # x1-prediction: convert to velocity for ODE solver
                def model_fn(x_t, t_tensor, c=cond):
                    x1_hat = model(x_t, t_tensor, c)
                    # v = (x_t - x1_hat) / t, but need to handle t→0
                    t_expand = t_tensor.view(-1, 1, 1).clamp(min=1e-6)
                    return (x_t - x1_hat) / t_expand

            x_out = euler_solve(model_fn, x_init, num_steps=n_steps)
            mse = F.mse_loss(x_out, state_gt).item()
            mses.append(mse)

        print(f"    {n_steps:>3d} steps: MSE = {np.mean(mses):.4f}")

    # 3. Compare against DDIM-2 if available
    if ddpm_model is not None and ddpm_timesteps is not None:
        ddim = DDIMSampler(
            num_train_timesteps=ddpm_timesteps,
            num_inference_steps=2,
            eta=0.0,
        ).to(device)

        ddim_mses = []
        for i in range(min(num_eval, len(dataset))):
            sample = dataset[i]
            state_gt = sample["state"].unsqueeze(0).to(device)
            cond_i = sample["cond"].unsqueeze(0).to(device)
            T = state_gt.shape[1]

            torch.manual_seed(42 + i)
            x_out = ddim.sample(
                model=ddpm_model,
                shape=(1, T, state_dim),
                condition_fn=lambda c=cond_i: c,
                device=device,
            )
            ddim_mses.append(F.mse_loss(x_out, state_gt).item())

        print(f"\n  DDIM-2 reference: MSE = {np.mean(ddim_mses):.4f}")


def x1_prediction_loss(model, x_t, t, cond, x1_target):
    """
    x1-prediction loss: model predicts clean data directly.
    
    Loss = ||model(x_t, t, cond) - x_1||^2
    
    Note: We use the SAME model architecture (it predicts a (B,T,D) output).
    The only change is what we train it to predict.
    """
    x1_pred = model(x_t, t, cond)
    return F.mse_loss(x1_pred, x1_target)


def velocity_prediction_loss(model, x_t, t, cond, velocity_target):
    """Standard FM velocity prediction loss for comparison."""
    v_pred = model(x_t, t, cond)
    return F.mse_loss(v_pred, velocity_target)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    fm_ckpt = "logs/stage2_fm_e10000_b128_lr0.0001_w120_s10_transformer_2026Mar29_20-52-56/checkpoints/stage2_fm_epoch_000999.pt"
    ddpm_ckpt = "logs/stage2_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026Feb15_21-41-59/checkpoints/stage2_epoch_001299.pt"
    root_dir = "/media/learning/DATA/export_smplx_retargeted_sub1_clothesstand"

    finetune_epochs = 50
    lr = 1e-4  # Same as original training
    batch_size = 64  # Reduced from 128 to avoid OOM

    # Load checkpoint
    print("Loading FM checkpoint...")
    ckpt = torch.load(fm_ckpt, map_location=device)
    config = ckpt["config"]
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    window_size = dataset_cfg.get("window_size", 120)
    max_len = model_cfg.get("max_len", window_size + 100)
    state_dict_orig = ckpt["model"]
    state_dim = state_dict_orig["out_proj.weight"].shape[0]

    # Load DDPM model for comparison
    print("Loading DDPM model...")
    ddpm_ckpt_data = torch.load(ddpm_ckpt, map_location=device)
    ddpm_config = ddpm_ckpt_data["config"]
    ddpm_model_cfg = ddpm_config.get("model", {})
    ddpm_dataset_cfg = ddpm_config.get("dataset", {})
    ddpm_window_size = ddpm_dataset_cfg.get("window_size", 120)
    ddpm_max_len = ddpm_model_cfg.get("max_len", ddpm_window_size + 100)
    ddpm_timesteps = ddpm_config.get("train", {}).get("timesteps", 1000)

    ddpm_model = Stage2TransformerModel(
        state_dim=state_dim, cond_dim=6,
        d_model=ddpm_model_cfg.get("d_model", 256),
        nhead=ddpm_model_cfg.get("nhead", 4),
        num_layers=ddpm_model_cfg.get("num_layers", 4),
        dim_feedforward=ddpm_model_cfg.get("dim_feedforward", 512),
        dropout=0.0,
        max_len=ddpm_max_len,
    ).to(device)
    ddpm_model.load_state_dict(ddpm_ckpt_data["model"])
    ddpm_model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = G1MotionDatasetHandCond(
        root_dir=root_dir,
        window_size=window_size,
        stride=10,
        min_seq_len=30,
        train=True,
        train_split=0.99,
        preload=True,
        normalize_hands=False,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=4, pin_memory=True)
    print(f"  {len(dataset)} windows, {len(dataloader)} batches/epoch")

    schedule = FlowMatchingSchedule(FlowMatchingConfig())

    # =========================================================
    # BASELINE: We already have these numbers from diagnose_fm.py
    # Velocity prediction baseline (from previous run):
    #   vel_MSE: t=0.01:0.94  t=0.50:0.09  t=0.99:0.17
    #   x1_MSE:  t=0.01:0.0001  t=0.50:0.023  t=0.99:0.167
    #   Gen MSE: 2-step=0.184, 50-step=0.259
    #   DDIM-2 reference: 0.034
    # =========================================================
    print("\nBaseline numbers (from previous diagnostic):")
    print("  FM velocity: Gen MSE 2-step=0.184, 50-step=0.259")
    print("  DDIM-2 reference: MSE=0.034")
    
    # Free DDPM model during training — reload for eval only
    del ddpm_model
    torch.cuda.empty_cache()

    # =========================================================
    # EXPERIMENT: Fine-tune with x1-prediction (data prediction)
    # =========================================================
    print("\n" + "=" * 60)
    print(f"FINE-TUNING: x1-prediction for {finetune_epochs} epochs")
    print("=" * 60)

    # Start from RANDOM INIT (not velocity-trained weights)
    # because the output head was trained to predict velocity, not data.
    # Fine-tuning velocity weights toward data prediction would fight the prior.
    #
    # Actually, let's test BOTH:
    #   A) Fine-tune from velocity-trained checkpoint
    #   B) Train from scratch

    for variant_name, use_pretrained in [("B: fresh init (x1-pred)", False)]:
        print(f"\n--- Variant {variant_name} ---")

        model = Stage2FMTransformerModel(
            state_dim=state_dim, cond_dim=6,
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 4),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.1),
            max_len=max_len,
        ).to(device)

        if use_pretrained:
            model.load_state_dict(state_dict_orig)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        t0 = time.time()
        for epoch in range(finetune_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                state = batch["state"].to(device)    # x_1 (data)
                cond = batch["cond"].to(device)
                B = state.shape[0]

                noise = torch.randn_like(state)       # x_0
                t = schedule.sample_timesteps(B, device)
                t_expand = t.view(-1, 1, 1)
                x_t = (1.0 - t_expand) * state + t_expand * noise

                # x1-prediction loss: predict clean data directly
                loss = x1_prediction_loss(model, x_t, t, cond, state)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.6f}  ({elapsed:.1f}s)")

        total_time = time.time() - t0
        print(f"  Total fine-tune time: {total_time:.1f}s")

        # Evaluate — reload DDPM model for comparison
        torch.cuda.empty_cache()
        ddpm_model_eval = Stage2TransformerModel(
            state_dim=state_dim, cond_dim=6,
            d_model=ddpm_model_cfg.get("d_model", 256),
            nhead=ddpm_model_cfg.get("nhead", 4),
            num_layers=ddpm_model_cfg.get("num_layers", 4),
            dim_feedforward=ddpm_model_cfg.get("dim_feedforward", 512),
            dropout=0.0, max_len=ddpm_max_len,
        ).to(device)
        ddpm_model_eval.load_state_dict(ddpm_ckpt_data["model"])
        ddpm_model_eval.eval()

        evaluate_all(model, dataset, device, state_dim,
                     f"x1-prediction ({variant_name})",
                     prediction_type="x1",
                     ddpm_model=ddpm_model_eval, ddpm_timesteps=ddpm_timesteps)
        del model, optimizer, ddpm_model_eval
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("DECISION GUIDE:")
    print("=" * 60)
    print("""
Compare x1-prediction FM vs velocity-prediction FM vs DDIM-2:

If x1-prediction at 2-10 steps approaches DDIM-2 quality:
  → Full retrain with x1-prediction is the answer.
  → The velocity parameterization was the bottleneck, not FM itself.

If x1-prediction is better than velocity but still far from DDIM-2:
  → FM with linear interpolation can't match DDPM + DDIM.
  → The nonlinear beta schedule gives DDIM a structural advantage.

If x1-prediction shows no improvement:
  → Something else is wrong, or 50 epochs wasn't enough.
""")


if __name__ == "__main__":
    main()
