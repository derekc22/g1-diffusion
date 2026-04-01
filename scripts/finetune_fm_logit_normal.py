"""
Quick fine-tune of existing FM checkpoint with logit-normal timestep sampling.

Purpose: Test whether logit-normal sampling improves edge-timestep velocity 
prediction WITHOUT committing to a full retrain. Runs ~50 epochs, should 
take ~5 minutes.

If velocity MSE at t≈0.01 drops significantly → full retrain is worth it.
If it flatlines → FM model has a deeper issue and retraining won't help.

Usage:
    python scripts/finetune_fm_logit_normal.py
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

from datasets.g1_motion_dataset import G1MotionDatasetHandCond
from models.stage2_flow_matching import Stage2FMTransformerModel
from utils.flow_matching import FlowMatchingConfig, FlowMatchingSchedule, euler_solve
from utils.general import load_config


def evaluate_velocity_at_edges(model, dataset, device, schedule_eval, num_samples=20):
    """Check velocity prediction quality at key timesteps."""
    model.eval()
    t_values = [0.01, 0.05, 0.10, 0.50, 0.90, 0.99]
    results = {}

    with torch.no_grad():
        for t_val in t_values:
            errors = []
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                state = sample["state"].unsqueeze(0).to(device)
                cond = sample["cond"].unsqueeze(0).to(device)

                noise = torch.randn_like(state)
                t = torch.full((1,), t_val, device=device)
                x_t = (1.0 - t_val) * state + t_val * noise
                velocity_target = noise - state
                velocity_pred = model(x_t, t, cond)
                mse = F.mse_loss(velocity_pred, velocity_target).item()
                errors.append(mse)

            results[t_val] = np.mean(errors)

    return results


def evaluate_generation_mse(model, dataset, device, state_dim, num_steps=50, num_samples=10):
    """Check generation quality (MSE vs GT)."""
    model.eval()
    mses = []
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            state_gt = sample["state"].unsqueeze(0).to(device)
            cond = sample["cond"].unsqueeze(0).to(device)
            T = state_gt.shape[1]

            torch.manual_seed(42 + i)
            x_init = torch.randn(1, T, state_dim, device=device)

            def model_fn(x_t, t, c=cond):
                return model(x_t, t, c)

            x_out = euler_solve(model_fn, x_init, num_steps=num_steps)
            mse = F.mse_loss(x_out, state_gt).item()
            mses.append(mse)

    return np.mean(mses)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Config ----
    fm_ckpt = "logs/stage2_fm_e10000_b128_lr0.0001_w120_s10_transformer_2026Mar29_20-52-56/checkpoints/stage2_fm_epoch_000999.pt"
    root_dir = "/media/learning/DATA/export_smplx_retargeted_sub1_clothesstand"
    finetune_epochs = 50
    lr = 1e-5  # Lower LR for fine-tuning (1/10th of original)
    batch_size = 128

    # Strategies to test
    strategies = {
        "logit_normal": FlowMatchingConfig(timestep_sampling="logit_normal", logit_normal_mean=0.0, logit_normal_std=1.0),
        "beta_u_shaped": FlowMatchingConfig(timestep_sampling="beta", beta_alpha=0.5, beta_beta=0.5),
    }

    # ---- Load model ----
    print("Loading FM checkpoint...")
    ckpt = torch.load(fm_ckpt, map_location=device)
    config = ckpt["config"]
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    window_size = dataset_cfg.get("window_size", 120)
    max_len = model_cfg.get("max_len", window_size + 100)
    state_dict = ckpt["model"]
    state_dim = state_dict["out_proj.weight"].shape[0]

    # ---- Load dataset ----
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    print(f"  {len(dataset)} windows, {len(dataloader)} batches")

    # ---- Baseline evaluation ----
    print("\n=== BASELINE (uniform-trained) ===")
    base_model = Stage2FMTransformerModel(
        state_dim=state_dim, cond_dim=6,
        d_model=model_cfg.get("d_model", 256),
        nhead=model_cfg.get("nhead", 4),
        num_layers=model_cfg.get("num_layers", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        dropout=0.0,  # No dropout for eval
        max_len=max_len,
    ).to(device)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    dummy_schedule = FlowMatchingSchedule(FlowMatchingConfig())
    baseline_vel = evaluate_velocity_at_edges(base_model, dataset, device, dummy_schedule)
    baseline_gen = evaluate_generation_mse(base_model, dataset, device, state_dim, num_steps=50)
    print(f"  Velocity MSE: " + "  ".join(f"t={t:.2f}:{v:.4f}" for t, v in baseline_vel.items()))
    print(f"  Generation MSE (50 steps): {baseline_gen:.4f}")
    del base_model

    # ---- Fine-tune with each strategy ----
    for strat_name, fm_config in strategies.items():
        print(f"\n{'='*60}")
        print(f"Fine-tuning with: {strat_name}")
        print(f"{'='*60}")

        schedule = FlowMatchingSchedule(fm_config)

        # Fresh copy from checkpoint
        model = Stage2FMTransformerModel(
            state_dim=state_dim, cond_dim=6,
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 4),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.1),
            max_len=max_len,
        ).to(device)
        model.load_state_dict(state_dict)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(finetune_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                state = batch["state"].to(device)
                cond = batch["cond"].to(device)
                B = state.shape[0]

                x0 = torch.randn_like(state)
                t = schedule.sample_timesteps(B, device)
                x_t = schedule.interpolate(state, x0, t)
                velocity_target = schedule.compute_velocity_target(state, x0)
                velocity_pred = model(x_t, t, cond)
                loss = F.mse_loss(velocity_pred, velocity_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                vel_results = evaluate_velocity_at_edges(model, dataset, device, schedule)
                gen_mse = evaluate_generation_mse(model, dataset, device, state_dim, num_steps=50)
                print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.6f}  gen_MSE={gen_mse:.4f}  "
                      f"vel@0.01={vel_results[0.01]:.4f}  vel@0.10={vel_results[0.10]:.4f}  "
                      f"vel@0.50={vel_results[0.50]:.4f}  vel@0.99={vel_results[0.99]:.4f}")
                model.train()

        # Final evaluation
        model.eval()
        final_vel = evaluate_velocity_at_edges(model, dataset, device, schedule)
        final_gen = evaluate_generation_mse(model, dataset, device, state_dim, num_steps=50)

        print(f"\n  --- {strat_name} SUMMARY ---")
        print(f"  {'Timestep':<10} {'Baseline':<12} {'After FT':<12} {'Change':<12}")
        for t_val in baseline_vel:
            b = baseline_vel[t_val]
            a = final_vel[t_val]
            pct = (a - b) / b * 100
            print(f"  t={t_val:<7.2f} {b:<12.4f} {a:<12.4f} {pct:+.1f}%")
        print(f"  Gen MSE:   {baseline_gen:<12.4f} {final_gen:<12.4f} {(final_gen-baseline_gen)/baseline_gen*100:+.1f}%")

        del model, optimizer

    print("\n" + "="*60)
    print("DECISION GUIDE:")
    print("="*60)
    print("""
If vel@0.01 dropped significantly (>30%) and gen_MSE dropped:
  → Full retrain with this strategy will likely help. Go for it.

If vel@0.01 barely moved or gen_MSE got worse:
  → Timestep sampling alone won't fix this. Don't waste the hour.
  → Consider: higher capacity model, different loss weighting, or
    accept that DDIM-2 is simply better for this task at low NFE.
""")


if __name__ == "__main__":
    main()
