"""
FM vs DDPM diagnostic — tests model quality and ODE solver correctness.

Tests:
1. Velocity prediction accuracy at different t values (is the FM model actually good?)
2. ODE reconstruction quality at different step counts (does more steps = better?)
3. Direct comparison: FM-N vs DDIM-2 MSE against ground truth

Usage:
    python scripts/diagnose_fm.py
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from datasets.g1_motion_dataset import G1MotionDatasetHandCond
from models.stage2_flow_matching import Stage2FMTransformerModel
from models.stage2_diffusion import Stage2TransformerModel
from utils.flow_matching import FlowMatchingSchedule, FlowMatchingConfig, euler_solve, midpoint_solve
from utils.inference_optimization import DDIMSampler
from utils.diffusion import DiffusionSchedule, DiffusionConfig
from utils.general import load_config


def load_fm_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    window_size = dataset_cfg.get("window_size", 120)
    max_len = model_cfg.get("max_len", window_size + 100)

    state_dict = ckpt["model"]
    state_dim = state_dict["out_proj.weight"].shape[0]

    model = Stage2FMTransformerModel(
        state_dim=state_dim,
        cond_dim=6,
        d_model=model_cfg.get("d_model", 256),
        nhead=model_cfg.get("nhead", 4),
        num_layers=model_cfg.get("num_layers", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        dropout=model_cfg.get("dropout", 0.1),
        max_len=max_len,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    norm = ckpt.get("norm_stats", {})
    return model, norm, state_dim


def load_ddpm_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    window_size = dataset_cfg.get("window_size", 120)
    max_len = model_cfg.get("max_len", window_size + 100)

    state_dict = ckpt["model"]
    state_dim = state_dict["out_proj.weight"].shape[0]

    model = Stage2TransformerModel(
        state_dim=state_dim,
        cond_dim=6,
        d_model=model_cfg.get("d_model", 256),
        nhead=model_cfg.get("nhead", 4),
        num_layers=model_cfg.get("num_layers", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        dropout=model_cfg.get("dropout", 0.1),
        max_len=max_len,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    norm = ckpt.get("norm_stats", {})
    timesteps = config.get("train", {}).get("timesteps", 1000)
    return model, norm, state_dim, timesteps


@torch.no_grad()
def test_velocity_accuracy(fm_model, dataset, device, num_samples=20):
    """
    Test 1: Does the FM model accurately predict velocity?
    
    For each sample, create (x_t, t) pairs and check if model(x_t, t, cond) ≈ noise - data.
    """
    print("\n" + "=" * 60)
    print("TEST 1: FM Velocity Prediction Accuracy")
    print("=" * 60)

    schedule = FlowMatchingSchedule(FlowMatchingConfig())
    t_values = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    for t_val in t_values:
        errors = []
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            state = sample["state"].unsqueeze(0).to(device)  # (1, T, D) - clean data
            cond = sample["cond"].unsqueeze(0).to(device)    # (1, T, 6) - hand positions

            noise = torch.randn_like(state)
            t = torch.full((1,), t_val, device=device)

            # Interpolate
            x_t = (1.0 - t_val) * state + t_val * noise

            # Target velocity
            velocity_target = noise - state

            # Model prediction
            velocity_pred = fm_model(x_t, t, cond)

            # Per-sample MSE
            mse = F.mse_loss(velocity_pred, velocity_target).item()
            errors.append(mse)

        mean_err = np.mean(errors)
        std_err = np.std(errors)
        print(f"  t={t_val:.2f}: velocity MSE = {mean_err:.4f} ± {std_err:.4f}")

    # Also check: what's the magnitude of the velocity target?
    sample = dataset[0]
    state = sample["state"].unsqueeze(0).to(device)
    noise = torch.randn_like(state)
    v_target = noise - state
    print(f"\n  Reference: ||velocity_target||² = {v_target.pow(2).mean().item():.4f}")
    print(f"  (This is the 'trivial' MSE if model predicted zeros)")


@torch.no_grad()
def test_ode_reconstruction(fm_model, dataset, device, state_dim, num_samples=10):
    """
    Test 2: Does ODE quality improve with more steps?
    
    Run FM sampling from the SAME noise seed at different step counts.
    Measure MSE vs ground truth (though ground truth isn't the target of generation,
    we expect the distribution statistics to be similar).
    """
    print("\n" + "=" * 60)
    print("TEST 2: ODE Reconstruction Quality vs Step Count")
    print("=" * 60)

    step_counts = [2, 5, 10, 20, 50, 100, 200]

    for n_steps in step_counts:
        mses = []
        output_stds = []
        output_means = []

        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            state_gt = sample["state"].unsqueeze(0).to(device)
            cond = sample["cond"].unsqueeze(0).to(device)
            T = state_gt.shape[1]

            # Fixed seed noise for fair comparison across step counts
            torch.manual_seed(42 + i)
            x_init = torch.randn(1, T, state_dim, device=device)

            def model_fn(x_t, t):
                return fm_model(x_t, t, cond)

            x_out = euler_solve(model_fn, x_init, num_steps=n_steps)

            mse = F.mse_loss(x_out, state_gt).item()
            mses.append(mse)
            output_stds.append(x_out.std().item())
            output_means.append(x_out.mean().item())

        print(f"  {n_steps:>4} steps: MSE={np.mean(mses):.4f} ± {np.std(mses):.4f}  "
              f"out_mean={np.mean(output_means):.3f}  out_std={np.mean(output_stds):.3f}")

    # Show ground truth stats for comparison
    gt_means, gt_stds = [], []
    for i in range(min(num_samples, len(dataset))):
        s = dataset[i]["state"]
        gt_means.append(s.mean().item())
        gt_stds.append(s.std().item())
    print(f"  Ground truth stats: mean={np.mean(gt_means):.3f}  std={np.mean(gt_stds):.3f}")


@torch.no_grad()
def test_ddpm_comparison(ddpm_model, fm_model, dataset, device, state_dim, timesteps, num_samples=10):
    """
    Test 3: Direct MSE comparison — FM-N vs DDIM-2.
    """
    print("\n" + "=" * 60)
    print("TEST 3: FM vs DDIM — MSE Against Ground Truth")
    print("=" * 60)

    # DDIM-2
    ddim = DDIMSampler(
        num_train_timesteps=timesteps,
        num_inference_steps=2,
        eta=0.0,
    ).to(device)

    ddim_mses = []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        state_gt = sample["state"].unsqueeze(0).to(device)
        cond = sample["cond"].unsqueeze(0).to(device)
        T = state_gt.shape[1]

        torch.manual_seed(42 + i)
        x_out = ddim.sample(
            model=ddpm_model,
            shape=(1, T, state_dim),
            condition_fn=lambda c=cond: c,
            device=device,
        )
        mse = F.mse_loss(x_out, state_gt).item()
        ddim_mses.append(mse)

    print(f"  DDIM-2:    MSE = {np.mean(ddim_mses):.4f} ± {np.std(ddim_mses):.4f}")

    # FM at various step counts
    for n_steps in [2, 10, 20, 50, 100]:
        fm_mses = []
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            state_gt = sample["state"].unsqueeze(0).to(device)
            cond = sample["cond"].unsqueeze(0).to(device)
            T = state_gt.shape[1]

            torch.manual_seed(42 + i)
            x_init = torch.randn(1, T, state_dim, device=device)

            def model_fn(x_t, t, c=cond):
                return fm_model(x_t, t, c)

            x_out = euler_solve(model_fn, x_init, num_steps=n_steps)
            mse = F.mse_loss(x_out, state_gt).item()
            fm_mses.append(mse)

        print(f"  FM-Euler-{n_steps:<3}: MSE = {np.mean(fm_mses):.4f} ± {np.std(fm_mses):.4f}")


@torch.no_grad()
def test_roundtrip(fm_model, dataset, device):
    """
    Test 4: Roundtrip sanity — noise a clean sample to t=0.99, then denoise with ODE.
    
    If the model is correct, many-step ODE from x_{0.99} should recover ~x_0 (data).
    This tests the model WITHOUT full noise (avoids the hard t=1 case).
    """
    print("\n" + "=" * 60)
    print("TEST 4: Partial Roundtrip (noise to t=0.99 then denoise)")
    print("=" * 60)

    for t_start in [0.5, 0.8, 0.99]:
        mses = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            state = sample["state"].unsqueeze(0).to(device)
            cond = sample["cond"].unsqueeze(0).to(device)

            torch.manual_seed(42 + i)
            noise = torch.randn_like(state)

            # Noise the clean data to t_start
            x_t = (1.0 - t_start) * state + t_start * noise

            # Now denoise from t_start to t=0 using 100 steps over [t_start, 0]
            num_steps = 100
            dt = t_start / num_steps
            x = x_t
            for step in range(num_steps):
                t_val = t_start - step * dt
                t_tensor = torch.full((1,), t_val, device=device)
                v = fm_model(x, t_tensor, cond)
                x = x - dt * v

            mse = F.mse_loss(x, state).item()
            mses.append(mse)

        print(f"  Start t={t_start}: roundtrip MSE = {np.mean(mses):.6f} ± {np.std(mses):.6f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths — update these if needed
    fm_ckpt = "logs/stage2_fm_e10000_b128_lr0.0001_w120_s10_transformer_2026Mar29_20-52-56/checkpoints/stage2_fm_epoch_000999.pt"
    ddpm_ckpt = "logs/stage2_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026Feb15_21-41-59/checkpoints/stage2_epoch_001299.pt"
    root_dir = "/media/learning/DATA/export_smplx_retargeted_sub1_clothesstand"

    print("Loading FM model...")
    fm_model, fm_norm, state_dim = load_fm_model(fm_ckpt, device)
    print(f"  state_dim = {state_dim}")

    print("Loading DDPM model...")
    ddpm_model, ddpm_norm, ddpm_state_dim, ddpm_timesteps = load_ddpm_model(ddpm_ckpt, device)
    print(f"  state_dim = {ddpm_state_dim}, timesteps = {ddpm_timesteps}")

    print("Loading dataset...")
    dataset = G1MotionDatasetHandCond(
        root_dir=root_dir,
        window_size=120,
        stride=10,
        min_seq_len=30,
        train=True,
        train_split=0.99,
        preload=False,
        normalize_hands=False,
    )
    print(f"  {len(dataset)} windows")

    # Run all tests
    test_velocity_accuracy(fm_model, dataset, device)
    test_ode_reconstruction(fm_model, dataset, device, state_dim)
    test_ddpm_comparison(ddpm_model, fm_model, dataset, device, state_dim, ddpm_timesteps)
    test_roundtrip(fm_model, dataset, device)

    print("\n" + "=" * 60)
    print("DIAGNOSIS GUIDE:")
    print("=" * 60)
    print("""
Test 1 (Velocity Accuracy):
  - If MSE is close to the 'trivial' reference → model learned nothing
  - If MSE << reference at all t values → model is good
  - If MSE is good at mid-t but bad at t≈0 or t≈1 → edge case issue

Test 2 (ODE vs Steps):
  - If MSE decreases monotonically with more steps → solver is correct
  - If MSE plateaus or increases → possible sign/convention bug
  - If out_std is very different from gt_std → model outputs wrong scale

Test 3 (FM vs DDIM):
  - Compare absolute MSE values
  - If FM-100 still worse than DDIM-2 → FM model quality is the bottleneck
  - If FM-100 is close to DDIM-2 → FM just needs more steps for same quality

Test 4 (Roundtrip):
  - If roundtrip from t=0.5 has low MSE → model velocity is accurate
  - If roundtrip from t=0.99 has high MSE → model struggles at high noise
  - If ALL roundtrips fail → fundamental model/convention issue
""")


if __name__ == "__main__":
    main()
