from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.robot_kinematics import robot_hand_positions_from_state_torch


def loss_config(yml: Dict[str, Any], stage: str) -> Dict[str, float]:
    cfg = yml.get("loss", {})
    stage_cfg = cfg.get(stage, {})

    def value(name: str, default: float) -> float:
        if name in stage_cfg:
            return float(stage_cfg[name])
        return float(cfg.get(name, default))

    return {
        "base_weight": value("base_weight", 1.0),
        "velocity_weight": value("velocity_weight", 0.0),
        "acceleration_weight": value("acceleration_weight", 0.0),
        "jerk_weight": value("jerk_weight", 0.0),
        "smooth_acceleration_weight": value("smooth_acceleration_weight", 0.0),
        "smooth_jerk_weight": value("smooth_jerk_weight", 0.0),
        "contact_weight": value("contact_weight", 0.0),
        "contact_offset_weight": value("contact_offset_weight", 0.0),
        "contact_distance_weight": value("contact_distance_weight", 0.0),
        "contact_margin": value("contact_margin", 0.03),
        "robot_hand_weight": value("robot_hand_weight", 0.0),
        "robot_hand_contact_weight": value("robot_hand_contact_weight", 0.0),
        "robot_hand_contact_margin": value("robot_hand_contact_margin", 0.08),
    }


def denormalize(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mean is None or std is None:
        return x
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    if x.ndim == 3:
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    else:
        mean = mean.view(1, -1)
        std = std.view(1, -1)
    return x * std + mean


def _diff(x: torch.Tensor, order: int) -> torch.Tensor:
    out = x
    for _ in range(order):
        out = out[:, 1:] - out[:, :-1]
    return out


def temporal_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = pred.new_zeros(())
    metrics: Dict[str, float] = {}

    if cfg["velocity_weight"] > 0.0 and pred.shape[1] > 1:
        vel = F.mse_loss(_diff(pred, 1), _diff(target, 1))
        loss = loss + cfg["velocity_weight"] * vel
        metrics["vel_mse"] = float(vel.detach().cpu())

    if cfg["acceleration_weight"] > 0.0 and pred.shape[1] > 2:
        acc = F.mse_loss(_diff(pred, 2), _diff(target, 2))
        loss = loss + cfg["acceleration_weight"] * acc
        metrics["acc_mse"] = float(acc.detach().cpu())

    if cfg["jerk_weight"] > 0.0 and pred.shape[1] > 3:
        jerk = F.mse_loss(_diff(pred, 3), _diff(target, 3))
        loss = loss + cfg["jerk_weight"] * jerk
        metrics["jerk_mse"] = float(jerk.detach().cpu())

    if cfg["smooth_acceleration_weight"] > 0.0 and pred.shape[1] > 2:
        smooth_acc = _diff(pred, 2).pow(2).mean()
        loss = loss + cfg["smooth_acceleration_weight"] * smooth_acc
        metrics["smooth_acc"] = float(smooth_acc.detach().cpu())

    if cfg["smooth_jerk_weight"] > 0.0 and pred.shape[1] > 3:
        smooth_jerk = _diff(pred, 3).pow(2).mean()
        loss = loss + cfg["smooth_jerk_weight"] * smooth_jerk
        metrics["smooth_jerk"] = float(smooth_jerk.detach().cpu())

    return loss, metrics


def contact_anchor_loss(
    pred_hands: torch.Tensor,
    batch: Dict[str, Any],
    cfg: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if (
        cfg["contact_weight"] <= 0.0
        and cfg["contact_offset_weight"] <= 0.0
        and cfg["contact_distance_weight"] <= 0.0
    ):
        return pred_hands.new_zeros(()), {}
    if "contact_points" not in batch or "contact_mask" not in batch:
        return pred_hands.new_zeros(()), {"contact_available": 0.0}

    contact_points = batch["contact_points"].to(device=pred_hands.device, dtype=pred_hands.dtype)
    contact_offsets = batch.get("contact_offsets")
    if contact_offsets is None:
        contact_offsets = torch.zeros_like(contact_points)
    else:
        contact_offsets = contact_offsets.to(device=pred_hands.device, dtype=pred_hands.dtype)
    contact_mask = batch["contact_mask"].to(device=pred_hands.device, dtype=pred_hands.dtype)

    target = contact_points + contact_offsets
    pred = pred_hands.view(pred_hands.shape[0], pred_hands.shape[1], 2, 3)
    target = target.view(target.shape[0], target.shape[1], 2, 3)
    points = contact_points.view(contact_points.shape[0], contact_points.shape[1], 2, 3)
    mask = contact_mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    distance_mask = batch.get("contact_distance_mask")
    if distance_mask is None:
        distance_mask = contact_mask
    else:
        distance_mask = distance_mask.to(device=pred_hands.device, dtype=pred_hands.dtype)
    surface_mask = distance_mask.unsqueeze(-1)
    surface_denom = surface_mask.sum().clamp_min(1.0)

    loss = pred_hands.new_zeros(())
    metrics: Dict[str, float] = {
        "contact_frames": float(contact_mask.sum().detach().cpu()),
        "surface_contact_frames": float(distance_mask.sum().detach().cpu()),
    }

    if cfg["contact_weight"] > 0.0:
        surface = ((pred - points).pow(2) * surface_mask).sum() / surface_denom
        loss = loss + cfg["contact_weight"] * surface
        metrics["contact_surface_mse"] = float(surface.detach().cpu())

    if cfg["contact_offset_weight"] > 0.0:
        anchor = ((pred - target).pow(2) * mask).sum() / denom
        loss = loss + cfg["contact_offset_weight"] * anchor
        metrics["contact_offset_mse"] = float(anchor.detach().cpu())

    if cfg["contact_distance_weight"] > 0.0:
        dist = torch.linalg.norm(pred - points, dim=-1)
        violation = torch.relu(dist - cfg["contact_margin"]).pow(2)
        dist_loss = (violation * distance_mask).sum() / distance_mask.sum().clamp_min(1.0)
        loss = loss + cfg["contact_distance_weight"] * dist_loss
        metrics["contact_distance_violation"] = float(dist_loss.detach().cpu())

    return loss, metrics


def robot_fk_hand_loss(
    pred_state: torch.Tensor,
    target_hands: torch.Tensor,
    cfg: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if cfg["robot_hand_weight"] <= 0.0 and cfg["robot_hand_contact_weight"] <= 0.0:
        return pred_state.new_zeros(()), {}

    pred_hands = robot_hand_positions_from_state_torch(pred_state)
    target_hands = target_hands.to(device=pred_hands.device, dtype=pred_hands.dtype)
    pred_view = pred_hands.view(*pred_hands.shape[:-1], 2, 3)
    target_view = target_hands.view(*target_hands.shape[:-1], 2, 3)
    hand_dist = torch.linalg.norm(pred_view - target_view, dim=-1)

    loss = pred_state.new_zeros(())
    metrics: Dict[str, float] = {
        "robot_hand_jpe_cm": float(hand_dist.mean().detach().cpu() * 100.0),
    }

    if cfg["robot_hand_weight"] > 0.0:
        hand_mse = F.mse_loss(pred_hands, target_hands)
        loss = loss + cfg["robot_hand_weight"] * hand_mse
        metrics["robot_hand_mse"] = float(hand_mse.detach().cpu())

    if cfg["robot_hand_contact_weight"] > 0.0:
        margin = cfg["robot_hand_contact_margin"]
        contact_mask = (hand_dist.detach() <= margin).to(dtype=pred_state.dtype)
        if contact_mask.sum() > 0:
            contact_mse = (
                ((pred_view - target_view).pow(2).sum(dim=-1) * contact_mask).sum()
                / contact_mask.sum().clamp_min(1.0)
            )
            loss = loss + cfg["robot_hand_contact_weight"] * contact_mse
            metrics["robot_contact_hand_mse"] = float(contact_mse.detach().cpu())
            metrics["robot_contact_hand_frames"] = float(contact_mask.sum().detach().cpu())
        else:
            metrics["robot_contact_hand_frames"] = 0.0

    return loss, metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    if not metrics:
        return ""
    parts = []
    for key in sorted(metrics):
        value = metrics[key]
        if key.endswith("_frames") or key == "contact_available":
            parts.append(f"{key}={value:.0f}")
        else:
            parts.append(f"{key}={value:.6f}")
    return " ".join(parts)
