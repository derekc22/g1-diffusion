"""
Deep diagnostics for Stage 2 flow matching ideas.

This script goes beyond the existing FM diagnostics and tests:
1. Baseline FM solver behavior across NFE and solver families.
2. Non-uniform two-step schedules (directly tests whether t=0.5 is a bad split).
3. Time-warped Euler grids (more or less resolution near the endpoint t≈0).
4. Terminal projection samplers that stop at t=τ and project once to data.
5. Partial roundtrip tests to isolate endpoint failures from full-noise failures.
6. DDIM references at several NFE values.
7. Optional checkpoint progression checks.

Outputs a JSON report under logs/diagnostics/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from datasets.g1_motion_dataset import G1MotionDatasetHandCond
from models.stage2_diffusion import Stage2TransformerModel
from models.stage2_flow_matching import Stage2FMTransformerModel
from utils.inference_optimization import DDIMSampler


DEFAULT_FM_CKPT = (
    "logs/stage2_fm_e10000_b128_lr0.0001_w120_s10_transformer_2026Mar29_20-52-56/"
    "checkpoints/stage2_fm_epoch_000999.pt"
)
DEFAULT_DDPM_CKPT = (
    "logs/stage2_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026Feb15_21-41-59/"
    "checkpoints/stage2_epoch_001299.pt"
)
DEFAULT_ROOT_DIR = "/media/learning/DATA/export_smplx_retargeted_sub1_clothesstand"


@dataclass
class Batch:
    state: torch.Tensor
    cond: torch.Tensor
    meta: List[Dict[str, object]]
    seeds: List[int]


class BatchSubset:
    def __init__(self, dataset, indices: Sequence[int], device: torch.device, base_seed: int = 12345):
        self.dataset = dataset
        self.indices = list(indices)
        self.device = device
        self.base_seed = base_seed

    def iter_batches(self, batch_size: int) -> Iterable[Batch]:
        for start in range(0, len(self.indices), batch_size):
            batch_indices = self.indices[start:start + batch_size]
            states = []
            conds = []
            metas = []
            seeds = []
            for k, idx in enumerate(batch_indices):
                sample = self.dataset[idx]
                states.append(sample["state"])
                conds.append(sample["cond"])
                metas.append({
                    "dataset_idx": int(idx),
                    "seq_name": str(sample["seq_name"]),
                    "start": int(sample["start"]),
                })
                seeds.append(self.base_seed + start + k)
            yield Batch(
                state=torch.stack(states, dim=0).to(self.device),
                cond=torch.stack(conds, dim=0).to(self.device),
                meta=metas,
                seeds=seeds,
            )


def load_fm_model(ckpt_path: str, device: torch.device):
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
    return model, config, state_dim, ckpt.get("norm_stats", {})


def load_ddpm_model(ckpt_path: str, device: torch.device):
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
    timesteps = config.get("train", {}).get("timesteps", 1000)
    return model, config, state_dim, timesteps, ckpt.get("norm_stats", {})


@torch.no_grad()
def make_noise_like(x: torch.Tensor, seeds: Sequence[int]) -> torch.Tensor:
    pieces = []
    for seed in seeds:
        g = torch.Generator(device=x.device)
        g.manual_seed(seed)
        pieces.append(torch.randn((1,) + tuple(x.shape[1:]), generator=g, device=x.device, dtype=x.dtype))
    return torch.cat(pieces, dim=0)


@torch.no_grad()
def euler_on_grid(model_fn: Callable, x_init: torch.Tensor, t_grid: Sequence[float]) -> torch.Tensor:
    x = x_init
    B = x.shape[0]
    device = x.device
    dtype = x.dtype
    for i in range(len(t_grid) - 1):
        t_cur = float(t_grid[i])
        t_next = float(t_grid[i + 1])
        dt = t_cur - t_next
        t = torch.full((B,), t_cur, device=device, dtype=dtype)
        v = model_fn(x, t)
        x = x - dt * v
    return x


@torch.no_grad()
def midpoint_on_grid(model_fn: Callable, x_init: torch.Tensor, t_grid: Sequence[float]) -> torch.Tensor:
    x = x_init
    B = x.shape[0]
    device = x.device
    dtype = x.dtype
    for i in range(len(t_grid) - 1):
        t_cur = float(t_grid[i])
        t_next = float(t_grid[i + 1])
        dt = t_cur - t_next
        t = torch.full((B,), t_cur, device=device, dtype=dtype)
        v1 = model_fn(x, t)
        t_mid_val = t_cur - 0.5 * dt
        t_mid = torch.full((B,), t_mid_val, device=device, dtype=dtype)
        x_mid = x - 0.5 * dt * v1
        v_mid = model_fn(x_mid, t_mid)
        x = x - dt * v_mid
    return x


@torch.no_grad()
def rk4_on_grid(model_fn: Callable, x_init: torch.Tensor, t_grid: Sequence[float]) -> torch.Tensor:
    x = x_init
    B = x.shape[0]
    device = x.device
    dtype = x.dtype
    for i in range(len(t_grid) - 1):
        t_cur = float(t_grid[i])
        t_next = float(t_grid[i + 1])
        dt = t_cur - t_next
        t = torch.full((B,), t_cur, device=device, dtype=dtype)
        k1 = model_fn(x, t)
        t_mid = torch.full((B,), t_cur - 0.5 * dt, device=device, dtype=dtype)
        k2 = model_fn(x - 0.5 * dt * k1, t_mid)
        k3 = model_fn(x - 0.5 * dt * k2, t_mid)
        t_end = torch.full((B,), t_next, device=device, dtype=dtype)
        k4 = model_fn(x - dt * k3, t_end)
        x = x - (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@torch.no_grad()
def terminal_project(model_fn: Callable, x_tau: torch.Tensor, tau: float) -> torch.Tensor:
    B = x_tau.shape[0]
    t = torch.full((B,), float(tau), device=x_tau.device, dtype=x_tau.dtype)
    v = model_fn(x_tau, t)
    return x_tau - float(tau) * v


@torch.no_grad()
def two_eval_split_sampler(model_fn: Callable, x_init: torch.Tensor, tau: float) -> torch.Tensor:
    x_tau = euler_on_grid(model_fn, x_init, [1.0, float(tau)])
    return terminal_project(model_fn, x_tau, float(tau))


@torch.no_grad()
def direct_project_from_noise(model_fn: Callable, x_init: torch.Tensor) -> torch.Tensor:
    return terminal_project(model_fn, x_init, 1.0)


def uniform_grid(n_steps: int) -> List[float]:
    return np.linspace(1.0, 0.0, n_steps + 1).tolist()


def powered_grid(n_steps: int, power: float) -> List[float]:
    s = np.linspace(1.0, 0.0, n_steps + 1)
    t = np.power(s, power)
    t[0] = 1.0
    t[-1] = 0.0
    return t.tolist()


def cosine_grid(n_steps: int) -> List[float]:
    s = np.linspace(1.0, 0.0, n_steps + 1)
    t = np.sin(0.5 * math.pi * s) ** 2
    t[0] = 1.0
    t[-1] = 0.0
    return t.tolist()


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


@torch.no_grad()
def evaluate_generation_method(
    name: str,
    subset: BatchSubset,
    batch_size: int,
    sampler_fn: Callable[[torch.Tensor, torch.Tensor, Sequence[int]], torch.Tensor],
) -> Dict[str, object]:
    t0 = time.time()
    mses = []
    out_means = []
    out_stds = []
    for batch in subset.iter_batches(batch_size):
        x_out = sampler_fn(batch.state, batch.cond, batch.seeds)
        mse_per = ((x_out - batch.state) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        mses.extend(float(v) for v in mse_per)
        out_means.extend(float(v) for v in x_out.mean(dim=(1, 2)).detach().cpu().numpy())
        out_stds.extend(float(v) for v in x_out.std(dim=(1, 2)).detach().cpu().numpy())
    return {
        "name": name,
        "mse": summarize(mses),
        "out_mean": summarize(out_means),
        "out_std": summarize(out_stds),
        "elapsed_sec": float(time.time() - t0),
        "num_samples": len(mses),
    }


@torch.no_grad()
def evaluate_velocity_profiles(fm_model, subset: BatchSubset, batch_size: int, t_values: Sequence[float]) -> Dict[str, object]:
    out = {}
    for t_val in t_values:
        vel_errs = []
        x1_errs = []
        trivial_errs = []
        for batch in subset.iter_batches(batch_size):
            noise = make_noise_like(batch.state, batch.seeds)
            t = torch.full((batch.state.shape[0],), float(t_val), device=batch.state.device, dtype=batch.state.dtype)
            x_t = (1.0 - t_val) * batch.state + t_val * noise
            v_target = noise - batch.state
            v_pred = fm_model(x_t, t, batch.cond)
            x1_pred = x_t - t.view(-1, 1, 1) * v_pred
            vel_errs.extend(float(v) for v in ((v_pred - v_target) ** 2).mean(dim=(1, 2)).detach().cpu().numpy())
            x1_errs.extend(float(v) for v in ((x1_pred - batch.state) ** 2).mean(dim=(1, 2)).detach().cpu().numpy())
            trivial_errs.extend(float(v) for v in (v_target ** 2).mean(dim=(1, 2)).detach().cpu().numpy())
        out[f"t={t_val:.2f}"] = {
            "velocity_mse": summarize(vel_errs),
            "x1_mse": summarize(x1_errs),
            "trivial_zero_velocity_mse": summarize(trivial_errs),
        }
    return out


@torch.no_grad()
def evaluate_roundtrip_methods(
    subset: BatchSubset,
    batch_size: int,
    t_starts: Sequence[float],
    method_factories: Dict[str, Callable[[float], Callable[[torch.Tensor, torch.Tensor, Sequence[int]], torch.Tensor]]],
) -> Dict[str, object]:
    results = {}
    for method_name, factory in method_factories.items():
        per_t = {}
        for t_start in t_starts:
            mses = []
            sampler_fn = factory(float(t_start))
            for batch in subset.iter_batches(batch_size):
                noise = make_noise_like(batch.state, batch.seeds)
                x_t = (1.0 - t_start) * batch.state + t_start * noise
                x_out = sampler_fn(x_t, batch.cond, batch.seeds)
                mses.extend(float(v) for v in ((x_out - batch.state) ** 2).mean(dim=(1, 2)).detach().cpu().numpy())
            per_t[f"t_start={t_start:.2f}"] = summarize(mses)
        results[method_name] = per_t
    return results


def main():
    parser = argparse.ArgumentParser(description="Deep diagnostics for Stage 2 FM ideas")
    parser.add_argument("--fm_ckpt", type=str, default=DEFAULT_FM_CKPT)
    parser.add_argument("--ddpm_ckpt", type=str, default=DEFAULT_DDPM_CKPT)
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--use_train_split", action="store_true")
    parser.add_argument("--output_dir", type=str, default="logs/diagnostics")
    parser.add_argument("--progression_ckpts", nargs="*", default=[])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fm_model, fm_config, state_dim, fm_norm = load_fm_model(args.fm_ckpt, device)
    ddpm_model, _, ddpm_state_dim, ddpm_timesteps, _ = load_ddpm_model(args.ddpm_ckpt, device)
    if state_dim != ddpm_state_dim:
        raise ValueError(f"State dim mismatch: FM={state_dim}, DDPM={ddpm_state_dim}")

    dataset_cfg = fm_config.get("dataset", {})
    dataset = G1MotionDatasetHandCond(
        root_dir=args.root_dir,
        window_size=dataset_cfg.get("window_size", 120),
        stride=dataset_cfg.get("stride", 10),
        min_seq_len=dataset_cfg.get("min_seq_len", 30),
        train=args.use_train_split,
        train_split=args.train_split,
        preload=False,
        state_mean=fm_norm.get("state_mean").cpu() if fm_norm.get("state_mean") is not None else None,
        state_std=fm_norm.get("state_std").cpu() if fm_norm.get("state_std") is not None else None,
        hand_mean=fm_norm.get("hand_mean").cpu() if fm_norm.get("hand_mean") is not None else None,
        hand_std=fm_norm.get("hand_std").cpu() if fm_norm.get("hand_std") is not None else None,
        normalize_hands=False,
    )

    rng = np.random.default_rng(0)
    all_indices = np.arange(len(dataset))
    if args.num_samples < len(dataset):
        indices = rng.choice(all_indices, size=args.num_samples, replace=False)
        indices = np.sort(indices)
    else:
        indices = all_indices
    subset = BatchSubset(dataset, indices.tolist(), device=device)

    report: Dict[str, object] = {
        "fm_ckpt": args.fm_ckpt,
        "ddpm_ckpt": args.ddpm_ckpt,
        "root_dir": args.root_dir,
        "device": str(device),
        "num_samples": int(len(indices)),
        "dataset_len": int(len(dataset)),
        "dataset_mode": "train" if args.use_train_split else "val",
    }

    print("\n[1/7] Velocity / implied x1 diagnostics...")
    t_values = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    report["velocity_profiles"] = evaluate_velocity_profiles(fm_model, subset, args.batch_size, t_values)

    print("\n[2/7] DDIM references...")
    ddim_results = {}
    for nfe in [1, 2, 5, 10, 20]:
        sampler = DDIMSampler(
            num_train_timesteps=ddpm_timesteps,
            num_inference_steps=nfe,
            eta=0.0,
        ).to(device)

        def ddim_sampler_fn(state, cond, seeds, sampler=sampler):
            torch.manual_seed(seeds[0])
            return sampler.sample(
                model=ddpm_model,
                shape=tuple(state.shape),
                condition_fn=lambda c=cond: c,
                device=device,
                dtype=state.dtype,
            )

        result = evaluate_generation_method(f"ddim_nfe_{nfe}", subset, args.batch_size, ddim_sampler_fn)
        ddim_results[f"nfe_{nfe}"] = result
        print(f"  DDIM NFE={nfe}: MSE={result['mse']['mean']:.4f} ± {result['mse']['std']:.4f}")
    report["ddim"] = ddim_results

    print("\n[3/7] Baseline FM solvers on uniform grids...")
    fm_baselines = {}
    for nfe in [1, 2, 5, 10, 20, 50, 100, 200]:
        grid = uniform_grid(nfe)

        grid_tuple = tuple(grid)

        def fm_euler_sampler(state, cond, seeds, grid_tuple=grid_tuple):
            x_init = make_noise_like(state, seeds)
            return euler_on_grid(lambda x, t: fm_model(x, t, cond), x_init, grid_tuple)

        result = evaluate_generation_method(f"fm_euler_nfe_{nfe}", subset, args.batch_size, fm_euler_sampler)
        fm_baselines[f"euler_nfe_{nfe}"] = result
        print(f"  FM Euler NFE={nfe}: MSE={result['mse']['mean']:.4f} ± {result['mse']['std']:.4f}")

    for steps in [1, 2, 5, 10, 25]:
        grid = uniform_grid(steps)
        nfe = 2 * steps

        grid_tuple = tuple(grid)

        def fm_midpoint_sampler(state, cond, seeds, grid_tuple=grid_tuple):
            x_init = make_noise_like(state, seeds)
            return midpoint_on_grid(lambda x, t: fm_model(x, t, cond), x_init, grid_tuple)

        result = evaluate_generation_method(f"fm_midpoint_nfe_{nfe}", subset, args.batch_size, fm_midpoint_sampler)
        fm_baselines[f"midpoint_nfe_{nfe}"] = result
        print(f"  FM Midpoint NFE={nfe}: MSE={result['mse']['mean']:.4f} ± {result['mse']['std']:.4f}")

    for steps in [1, 2, 5, 10]:
        grid = uniform_grid(steps)
        nfe = 4 * steps

        grid_tuple = tuple(grid)

        def fm_rk4_sampler(state, cond, seeds, grid_tuple=grid_tuple):
            x_init = make_noise_like(state, seeds)
            return rk4_on_grid(lambda x, t: fm_model(x, t, cond), x_init, grid_tuple)

        result = evaluate_generation_method(f"fm_rk4_nfe_{nfe}", subset, args.batch_size, fm_rk4_sampler)
        fm_baselines[f"rk4_nfe_{nfe}"] = result
        print(f"  FM RK4 NFE={nfe}: MSE={result['mse']['mean']:.4f} ± {result['mse']['std']:.4f}")
    report["fm_uniform"] = fm_baselines

    print("\n[4/7] Two-eval split search (tests whether t=0.5 is the wrong split)...")
    split_results = {}
    for tau in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
        def split_sampler(state, cond, seeds, tau=tau):
            x_init = make_noise_like(state, seeds)
            return two_eval_split_sampler(lambda x, t: fm_model(x, t, cond), x_init, tau)

        result = evaluate_generation_method(f"fm_split_tau_{tau:.2f}", subset, args.batch_size, split_sampler)
        split_results[f"tau_{tau:.2f}"] = result
        print(f"  FM 2-eval split tau={tau:.2f}: MSE={result['mse']['mean']:.4f} ± {result['mse']['std']:.4f}")
    report["two_eval_split_search"] = split_results

    print("\n[5/7] Time-warped Euler and terminal projection families...")
    warp_results = {}
    warp_specs = {
        "power_0.5": lambda n: powered_grid(n, 0.5),
        "power_0.75": lambda n: powered_grid(n, 0.75),
        "power_1.0": lambda n: powered_grid(n, 1.0),
        "power_1.5": lambda n: powered_grid(n, 1.5),
        "power_2.0": lambda n: powered_grid(n, 2.0),
        "power_3.0": lambda n: powered_grid(n, 3.0),
        "cosine": cosine_grid,
    }
    for grid_name, grid_fn in warp_specs.items():
        for nfe in [2, 5, 10, 20, 50]:
            grid = grid_fn(nfe)

            grid_tuple = tuple(grid)

            def warp_sampler(state, cond, seeds, grid_tuple=grid_tuple):
                x_init = make_noise_like(state, seeds)
                return euler_on_grid(lambda x, t: fm_model(x, t, cond), x_init, grid_tuple)

            result = evaluate_generation_method(f"fm_{grid_name}_nfe_{nfe}", subset, args.batch_size, warp_sampler)
            warp_results[f"{grid_name}_nfe_{nfe}"] = result
            print(f"  FM warp={grid_name:>9s} NFE={nfe:>2d}: MSE={result['mse']['mean']:.4f} ± {result['mse']['std']:.4f}")

    projector_results = {}
    for tau in [0.50, 0.30, 0.20, 0.10, 0.05, 0.01]:
        for nfe in [2, 5, 10, 20, 50]:
            if nfe == 1:
                continue
            steps_before_project = max(nfe - 1, 1)
            grid = np.linspace(1.0, tau, steps_before_project + 1).tolist()

            grid_tuple = tuple(grid)

            def projector_sampler(state, cond, seeds, tau=tau, grid_tuple=grid_tuple):
                x_init = make_noise_like(state, seeds)
                x_tau = euler_on_grid(lambda x, t: fm_model(x, t, cond), x_init, grid_tuple)
                return terminal_project(lambda x, t: fm_model(x, t, cond), x_tau, tau)

            result = evaluate_generation_method(f"fm_project_tau_{tau:.2f}_nfe_{nfe}", subset, args.batch_size, projector_sampler)
            projector_results[f"tau_{tau:.2f}_nfe_{nfe}"] = result
            print(f"  FM project tau={tau:.2f} NFE={nfe:>2d}: MSE={result['mse']['mean']:.4f} ± {result['mse']['std']:.4f}")

    def direct_project_sampler(state, cond, seeds):
        x_init = make_noise_like(state, seeds)
        return direct_project_from_noise(lambda x, t: fm_model(x, t, cond), x_init)

    projector_results["direct_project_nfe_1"] = evaluate_generation_method(
        "fm_direct_project_nfe_1", subset, args.batch_size, direct_project_sampler
    )
    print(
        "  FM direct project NFE=1: "
        f"MSE={projector_results['direct_project_nfe_1']['mse']['mean']:.4f} ± "
        f"{projector_results['direct_project_nfe_1']['mse']['std']:.4f}"
    )
    report["warped_euler"] = warp_results
    report["terminal_projection"] = projector_results

    print("\n[6/7] Partial roundtrip diagnostics...")
    roundtrip_factories = {
        "uniform_euler_100": lambda t_start: (
            lambda x_start, cond, seeds, t_start=t_start: euler_on_grid(
                lambda x, t: fm_model(x, t, cond),
                x_start,
                np.linspace(t_start, 0.0, 101).tolist(),
            )
        ),
        "power2_euler_100": lambda t_start: (
            lambda x_start, cond, seeds, t_start=t_start: euler_on_grid(
                lambda x, t: fm_model(x, t, cond),
                x_start,
                (np.power(np.linspace(1.0, 0.0, 101), 2.0) * t_start).tolist(),
            )
        ),
        "project_tau_0.10": lambda t_start: (
            lambda x_start, cond, seeds, t_start=t_start: terminal_project(
                lambda x, t: fm_model(x, t, cond),
                euler_on_grid(
                    lambda x, t: fm_model(x, t, cond),
                    x_start,
                    np.linspace(t_start, 0.10, 100).tolist() if t_start > 0.10 else [t_start],
                ),
                min(t_start, 0.10),
            )
        ),
    }
    report["roundtrip"] = evaluate_roundtrip_methods(
        subset,
        args.batch_size,
        [0.25, 0.50, 0.75, 0.90, 0.99],
        roundtrip_factories,
    )

    print("\n[7/7] Optional checkpoint progression spot-check...")
    progression = {}
    for ckpt_path in args.progression_ckpts:
        model, _, _, _ = load_fm_model(ckpt_path, device)

        def prog_sampler(state, cond, seeds, model=model):
            x_init = make_noise_like(state, seeds)
            return euler_on_grid(lambda x, t: model(x, t, cond), x_init, uniform_grid(20))

        result = evaluate_generation_method(os.path.basename(ckpt_path), subset, args.batch_size, prog_sampler)
        progression[ckpt_path] = result
        print(f"  {os.path.basename(ckpt_path)} @ Euler-20: MSE={result['mse']['mean']:.4f} ± {result['mse']['std']:.4f}")
    report["checkpoint_progression"] = progression

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"fm_idea_diagnostics_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
