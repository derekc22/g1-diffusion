"""Smoke-test the corrected two-stage object-goal diffusion path."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from datasets.hf_motion_dataset import HFFullBodyDataset
from models.stage1_diffusion import Stage1HandDiffusion
from models.stage2_diffusion import Stage2TransformerModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.object_goal_features import OBJECT_POSE_DIM, ROBOT_OBJECT_STATE_DIM, robot_object_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test object-goal two-stage diffusion")
    parser.add_argument("--root_dir", default="./data/hf_bps_preprocessed")
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root_dir = args.root_dir
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(PROJECT_ROOT, root_dir)
    device = torch.device(args.device)

    stage1_ds = HandMotionDataset(
        root_dir=root_dir,
        window_size=args.window_size,
        stride=args.window_size,
        min_seq_len=30,
        train=True,
        train_split=0.99,
        preload=True,
        flatten_bps=True,
        include_object_pose_goal=True,
    )
    stage2_ds = HFFullBodyDataset(
        root_dir=root_dir,
        window_size=args.window_size,
        stride=args.window_size,
        min_seq_len=30,
        train=True,
        train_split=0.99,
        preload=True,
        target_includes_object_pose=True,
        include_object_context=True,
        include_goal=True,
        flatten_bps=True,
    )
    b1 = next(iter(DataLoader(stage1_ds, batch_size=args.batch_size, shuffle=False)))
    b2 = next(iter(DataLoader(stage2_ds, batch_size=args.batch_size, shuffle=False)))

    print(f"stage1.hand_positions: {tuple(b1['hand_positions'].shape)}")
    print(f"stage1.object_pose:    {tuple(b1['object_pose'].shape)}")
    print(f"stage1.goal:           {tuple(b1['goal'].shape)}")
    print(f"stage2.state:          {tuple(b2['state'].shape)}")
    print(f"stage2.cond:           {tuple(b2['cond'].shape)}")
    print(f"stage2.goal:           {tuple(b2['goal'].shape)}")
    assert b1["hand_positions"].shape[1:] == (args.window_size, 6)
    assert b1["object_pose"].shape[1:] == (args.window_size, OBJECT_POSE_DIM)
    assert b1["goal"].shape[1:] == (OBJECT_POSE_DIM,)
    assert b2["state"].shape[1:] == (args.window_size, ROBOT_OBJECT_STATE_DIM)
    assert b2["goal"].shape[1:] == (OBJECT_POSE_DIM,)

    stage1 = Stage1HandDiffusion(
        bps_dim=b1["bps_encoding"].shape[-1],
        centroid_dim=3,
        encoder_hidden=64,
        object_feature_dim=64,
        encoder_layers=2,
        hand_dim=6,
        d_model=64,
        nhead=4,
        num_transformer_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        max_len=args.window_size,
        object_pose_dim=OBJECT_POSE_DIM,
        global_cond_dim=OBJECT_POSE_DIM,
        global_cond_hidden=64,
    ).to(device)
    stage2 = Stage2TransformerModel(
        state_dim=ROBOT_OBJECT_STATE_DIM,
        cond_dim=b2["cond"].shape[-1],
        global_cond_dim=OBJECT_POSE_DIM,
        global_cond_hidden=64,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        max_len=args.window_size,
    ).to(device)
    schedule = DiffusionSchedule(DiffusionConfig(timesteps=8)).to(device)

    hands = b1["hand_positions"].to(device)
    bps = b1["bps_encoding"].to(device)
    centroid = b1["object_centroid"].to(device)
    object_pose = b1["object_pose"].to(device)
    goal1 = b1["goal"].to(device)
    t1 = torch.randint(0, schedule.timesteps, (hands.shape[0],), device=device)
    hand_noisy = schedule.q_sample(hands, t1, torch.randn_like(hands))
    hand_pred = stage1(hand_noisy, t1, bps, centroid, object_pose=object_pose, global_cond=goal1)
    loss1 = F.mse_loss(hand_pred, hands)
    print(f"stage1.pred:           {tuple(hand_pred.shape)} loss={float(loss1.detach()):.6f}")

    state = b2["state"].to(device)
    cond = b2["cond"].to(device)
    goal2 = b2["goal"].to(device)
    t2 = torch.randint(0, schedule.timesteps, (state.shape[0],), device=device)
    state_noisy = schedule.q_sample(state, t2, torch.randn_like(state))
    state_pred = stage2(state_noisy, t2, cond, global_cond=goal2)
    loss2 = F.mse_loss(state_pred, state)
    print(f"stage2.pred:           {tuple(state_pred.shape)} loss={float(loss2.detach()):.6f}")
    assert state_pred.shape[-1] == ROBOT_OBJECT_STATE_DIM

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "stage2_object_goal_smoke.pt")
        torch.save(
            {
                "model": stage2.state_dict(),
                "pipeline_type": "object_goal_two_stage",
                "stage": 2,
                "stage_target": "robot_state_plus_object_pose",
                "requires_stage1_hands": True,
                "hand_contact_rectification_required": True,
                "layout": robot_object_layout(),
                "state_dim": ROBOT_OBJECT_STATE_DIM,
                "cond_dim": b2["cond"].shape[-1],
                "norm_stats": {
                    "state_mean": stage2_ds.state_mean,
                    "state_std": stage2_ds.state_std,
                    "goal_mean": stage2_ds.goal_mean,
                    "goal_std": stage2_ds.goal_std,
                },
            },
            ckpt_path,
        )
        round_trip = torch.load(ckpt_path, map_location=device, weights_only=False)
        assert round_trip["pipeline_type"] == "object_goal_two_stage"
        assert round_trip["stage_target"] == "robot_state_plus_object_pose"
        assert round_trip["layout"]["motion_dim"] == ROBOT_OBJECT_STATE_DIM

    print("object-goal two-stage smoke test passed")


if __name__ == "__main__":
    main()
