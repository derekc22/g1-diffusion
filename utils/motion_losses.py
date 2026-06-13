from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.robot_kinematics import (
    robot_foot_proxy_positions_from_state_torch,
    robot_hand_positions_from_state_torch,
)


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
        "contact_state_weight": value("contact_state_weight", 0.0),
        "object_contact_dist_weight": value("object_contact_dist_weight", 0.0),
        "floor_contact_dist_weight": value("floor_contact_dist_weight", 0.0),
        "contact_velocity_weight": value("contact_velocity_weight", 0.0),
        "foot_slide_weight": value("foot_slide_weight", 0.0),
        "floor_penetration_weight": value("floor_penetration_weight", 0.0),
        "support_weight": value("support_weight", 0.0),
        "geometric_warmup_steps": value("geometric_warmup_steps", 0.0),
        "floor_height": value("floor_height", 0.0),
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

    if "contact_soft" in batch and "contact_anchor_world" in batch:
        contact_soft = batch["contact_soft"].to(device=pred_hands.device, dtype=pred_hands.dtype)
        contact_anchor = batch["contact_anchor_world"].to(
            device=pred_hands.device,
            dtype=pred_hands.dtype,
        )
        contact_available = batch.get("contact_available")
        if contact_available is None:
            contact_available = torch.ones_like(contact_soft)
        else:
            contact_available = contact_available.to(device=pred_hands.device, dtype=pred_hands.dtype)

        pred = pred_hands.view(pred_hands.shape[0], pred_hands.shape[1], 2, 3)
        target = contact_anchor[:, :, :2]
        soft = contact_soft[:, :, :2]
        available = contact_available[:, :, :2]
        weight = soft * available
        denom = weight.sum().clamp_min(1.0)

        loss = pred_hands.new_zeros(())
        metrics: Dict[str, float] = {
            "contact_soft_frames": float(weight.sum().detach().cpu()),
        }

        anchor_weight = cfg["contact_weight"] + cfg["contact_offset_weight"]
        if anchor_weight > 0.0:
            anchor = F.smooth_l1_loss(pred, target, reduction="none").sum(dim=-1)
            anchor = (anchor * weight).sum() / denom
            loss = loss + anchor_weight * anchor
            metrics["contact_soft_anchor"] = float(anchor.detach().cpu())

        if cfg["contact_distance_weight"] > 0.0:
            dist = torch.linalg.norm(pred - target, dim=-1)
            violation = torch.relu(dist - cfg["contact_margin"]).pow(2)
            dist_loss = (violation * weight).sum() / denom
            loss = loss + cfg["contact_distance_weight"] * dist_loss
            metrics["contact_soft_distance_violation"] = float(dist_loss.detach().cpu())

        return loss, metrics

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


def _warmup_scale(cfg: Dict[str, float], global_step: int) -> float:
    warmup = int(cfg.get("geometric_warmup_steps", 0.0))
    if warmup <= 0:
        return 1.0
    return min(1.0, max(0.0, float(global_step) / float(warmup)))


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def full_body_contact_loss(
    pred_state: torch.Tensor,
    contact_logits: Optional[torch.Tensor],
    batch: Dict[str, Any],
    cfg: Dict[str, float],
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = (
        cfg["contact_state_weight"] > 0.0
        or cfg["object_contact_dist_weight"] > 0.0
        or cfg["floor_contact_dist_weight"] > 0.0
        or cfg["contact_velocity_weight"] > 0.0
        or cfg["foot_slide_weight"] > 0.0
        or cfg["floor_penetration_weight"] > 0.0
        or cfg["support_weight"] > 0.0
    )
    if not active:
        return pred_state.new_zeros(()), {}
    if "contact_soft" not in batch or "contact_anchor_world" not in batch:
        return pred_state.new_zeros(()), {"full_contact_available": 0.0}

    contact_soft = batch["contact_soft"].to(device=pred_state.device, dtype=pred_state.dtype)
    contact_anchor = batch["contact_anchor_world"].to(device=pred_state.device, dtype=pred_state.dtype)
    contact_available = batch.get("contact_available")
    if contact_available is None:
        contact_available = torch.ones_like(contact_soft)
    else:
        contact_available = contact_available.to(device=pred_state.device, dtype=pred_state.dtype)
    contact_mode = batch.get("contact_mode")
    if contact_mode is None:
        contact_mode = torch.zeros_like(contact_soft, dtype=torch.long, device=pred_state.device)
    else:
        contact_mode = contact_mode.to(device=pred_state.device)

    scale = _warmup_scale(cfg, global_step)
    loss = pred_state.new_zeros(())
    metrics: Dict[str, float] = {
        "full_contact_available": float(contact_available.sum().detach().cpu()),
        "geometric_warmup": float(scale),
    }

    if cfg["contact_state_weight"] > 0.0:
        if contact_logits is None:
            metrics["contact_state_head"] = 0.0
        else:
            bce = F.binary_cross_entropy_with_logits(
                contact_logits,
                contact_soft,
                reduction="none",
            )
            state_loss = _weighted_mean(bce, contact_available)
            loss = loss + scale * cfg["contact_state_weight"] * state_loss
            metrics["contact_state_bce"] = float(state_loss.detach().cpu())

    needs_hands = (
        cfg["object_contact_dist_weight"] > 0.0
        or cfg["contact_velocity_weight"] > 0.0
    )
    pred_hands = None
    if needs_hands:
        pred_hands = robot_hand_positions_from_state_torch(pred_state).view(
            pred_state.shape[0],
            pred_state.shape[1],
            2,
            3,
        )

    if cfg["object_contact_dist_weight"] > 0.0 and pred_hands is not None:
        weight = contact_soft[:, :, :2] * contact_available[:, :, :2]
        dist_loss = F.smooth_l1_loss(pred_hands, contact_anchor[:, :, :2], reduction="none").sum(dim=-1)
        dist_loss = _weighted_mean(dist_loss, weight)
        loss = loss + scale * cfg["object_contact_dist_weight"] * dist_loss
        metrics["object_contact_dist"] = float(dist_loss.detach().cpu())

    if cfg["contact_velocity_weight"] > 0.0 and pred_hands is not None and pred_state.shape[1] > 1:
        pred_vel = pred_hands[:, 1:] - pred_hands[:, :-1]
        anchor_vel = contact_anchor[:, 1:, :2] - contact_anchor[:, :-1, :2]
        stick = (contact_mode[:, 1:, :2] == 1).to(dtype=pred_state.dtype)
        weight = contact_soft[:, 1:, :2] * contact_available[:, 1:, :2] * stick
        vel_loss = F.smooth_l1_loss(pred_vel, anchor_vel, reduction="none").sum(dim=-1)
        vel_loss = _weighted_mean(vel_loss, weight)
        loss = loss + scale * cfg["contact_velocity_weight"] * vel_loss
        metrics["contact_velocity"] = float(vel_loss.detach().cpu())

    needs_feet = (
        cfg["floor_contact_dist_weight"] > 0.0
        or cfg["foot_slide_weight"] > 0.0
        or cfg["floor_penetration_weight"] > 0.0
    )
    pred_feet = None
    if needs_feet:
        pred_feet = robot_foot_proxy_positions_from_state_torch(pred_state)

    if cfg["floor_contact_dist_weight"] > 0.0 and pred_feet is not None:
        z_min = pred_feet[..., 2].amin(dim=-1)
        floor_z = contact_anchor[:, :, 2:4, 2]
        weight = contact_soft[:, :, 2:4] * contact_available[:, :, 2:4]
        floor_dist = F.smooth_l1_loss(z_min, floor_z, reduction="none")
        floor_dist = _weighted_mean(floor_dist, weight)
        loss = loss + scale * cfg["floor_contact_dist_weight"] * floor_dist
        metrics["floor_contact_dist"] = float(floor_dist.detach().cpu())

    if cfg["foot_slide_weight"] > 0.0 and pred_feet is not None and pred_state.shape[1] > 1:
        centers = pred_feet.mean(dim=-2)
        slide = (centers[:, 1:, :, :2] - centers[:, :-1, :, :2]).pow(2).sum(dim=-1)
        stick = (contact_mode[:, 1:, 2:4] == 1).to(dtype=pred_state.dtype)
        weight = contact_soft[:, 1:, 2:4] * contact_available[:, 1:, 2:4] * stick
        slide_loss = _weighted_mean(slide, weight)
        loss = loss + scale * cfg["foot_slide_weight"] * slide_loss
        metrics["foot_slide"] = float(slide_loss.detach().cpu())

    if cfg["floor_penetration_weight"] > 0.0 and pred_feet is not None:
        floor_height = float(cfg.get("floor_height", 0.0))
        pen = torch.relu(pred_feet.new_tensor(floor_height) - pred_feet[..., 2]).pow(2).mean()
        loss = loss + scale * cfg["floor_penetration_weight"] * pen
        metrics["floor_penetration"] = float(pen.detach().cpu())

    if cfg["support_weight"] > 0.0:
        foot_weight = contact_soft[:, :, 2:4] * contact_available[:, :, 2:4]
        support_sum = foot_weight.sum(dim=-1)
        support_xy = (
            contact_anchor[:, :, 2:4, :2] * foot_weight.unsqueeze(-1)
        ).sum(dim=-2) / support_sum.unsqueeze(-1).clamp_min(1e-6)
        root_xy = pred_state[:, :, :2]
        support_error = F.smooth_l1_loss(root_xy, support_xy, reduction="none").sum(dim=-1)
        support_loss = _weighted_mean(support_error, support_sum)
        loss = loss + scale * cfg["support_weight"] * support_loss
        metrics["support_proxy"] = float(support_loss.detach().cpu())

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
