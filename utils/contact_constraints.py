"""
Contact Constraints for OMOMO Stage 1

Implements the hand contact constraint post-processing described in the OMOMO paper
(Section 3.3 "Apply Hand Contact Constraints").

Algorithm:
1. For each predicted hand position sequence H_1, ..., H_T:
   a. Compute distance d_t from hand to nearest vertex on object mesh V_t
   b. Find first timestep k where d_k < threshold (contact detected)
   c. Compute offset vector p = H_k - V^i_k (hand position relative to nearest vertex)
   d. For all t > k, update hand position: H_hat_t = V^i_t + R_t @ R_k^{-1} @ p
      where R_t is object rotation at timestep t

This ensures hands maintain consistent relative position to object after contact.
"""

from typing import Tuple, Optional
import numpy as np
import torch


def find_nearest_vertex(
    hand_pos: np.ndarray,
    object_verts: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    """
    Find nearest vertex on object mesh to hand position.
    
    Args:
        hand_pos: (3,) hand position
        object_verts: (K, 3) object mesh vertices
    
    Returns:
        (vertex_idx, distance, vertex_position)
    """
    dists = np.linalg.norm(object_verts - hand_pos, axis=1)  # (K,)
    min_idx = np.argmin(dists)
    return min_idx, dists[min_idx], object_verts[min_idx]


def apply_contact_constraints_single_hand(
    hand_positions: np.ndarray,
    object_verts: np.ndarray,
    object_rotations: np.ndarray,
    contact_threshold: float = 0.03,
) -> np.ndarray:
    """
    Apply contact constraints to a single hand's trajectory.
    
    Args:
        hand_positions: (T, 3) hand position trajectory
        object_verts: (T, K, 3) object mesh vertices over time
        object_rotations: (T, 3, 3) object rotation matrices over time
        contact_threshold: threshold for contact detection (meters), paper uses 0.03
    
    Returns:
        (T, 3) rectified hand positions
    """
    T = hand_positions.shape[0]
    rectified = hand_positions.copy()
    
    # Find first contact frame
    contact_frame = None
    nearest_vertex_idx = None
    
    for t in range(T):
        v_idx, dist, _ = find_nearest_vertex(hand_positions[t], object_verts[t])
        if dist < contact_threshold:
            contact_frame = t
            nearest_vertex_idx = v_idx
            break
    
    # If no contact found, return original
    if contact_frame is None:
        return rectified
    
    # Compute offset vector p at contact frame
    k = contact_frame
    V_i_k = object_verts[k, nearest_vertex_idx]  # (3,)
    H_k = hand_positions[k]  # (3,)
    p = H_k - V_i_k  # offset vector (3,)
    
    R_k = object_rotations[k]  # (3, 3)
    R_k_inv = np.linalg.inv(R_k)
    
    # Update hand positions for all frames after contact
    for t in range(k + 1, T):
        R_t = object_rotations[t]
        V_i_t = object_verts[t, nearest_vertex_idx]
        
        # H_hat_t = V^i_t + R_t @ R_k^{-1} @ p
        rotated_offset = R_t @ R_k_inv @ p
        rectified[t] = V_i_t + rotated_offset
    
    return rectified


def apply_contact_constraints(
    hand_positions: np.ndarray,
    object_verts: np.ndarray,
    object_rotations: np.ndarray,
    contact_threshold: float = 0.03,
) -> np.ndarray:
    """
    Apply contact constraints to both hands.
    
    Args:
        hand_positions: (T, 6) = [left_x, left_y, left_z, right_x, right_y, right_z]
        object_verts: (T, K, 3) object mesh vertices
        object_rotations: (T, 3, 3) object rotation matrices
        contact_threshold: threshold for contact detection (meters)
    
    Returns:
        (T, 6) rectified hand positions
    """
    # Split left and right hands
    left_hand = hand_positions[:, :3]   # (T, 3)
    right_hand = hand_positions[:, 3:]  # (T, 3)
    
    # Apply constraints separately
    left_rectified = apply_contact_constraints_single_hand(
        left_hand, object_verts, object_rotations, contact_threshold
    )
    right_rectified = apply_contact_constraints_single_hand(
        right_hand, object_verts, object_rotations, contact_threshold
    )
    
    # Combine
    return np.concatenate([left_rectified, right_rectified], axis=-1)


def apply_contact_constraints_batch(
    hand_positions: torch.Tensor,
    object_verts: torch.Tensor,
    object_rotations: torch.Tensor,
    contact_threshold: float = 0.03,
) -> torch.Tensor:
    """
    Batch version of contact constraints for inference.
    
    Args:
        hand_positions: (B, T, 6) hand positions
        object_verts: (B, T, K, 3) object vertices
        object_rotations: (B, T, 3, 3) object rotations
        contact_threshold: contact detection threshold
    
    Returns:
        (B, T, 6) rectified hand positions
    """
    B = hand_positions.shape[0]
    device = hand_positions.device
    
    # Convert to numpy for processing
    hands_np = hand_positions.detach().cpu().numpy()
    verts_np = object_verts.detach().cpu().numpy()
    rots_np = object_rotations.detach().cpu().numpy()
    
    results = []
    for b in range(B):
        rectified = apply_contact_constraints(
            hands_np[b], verts_np[b], rots_np[b], contact_threshold
        )
        results.append(rectified)
    
    results_np = np.stack(results, axis=0)
    return torch.from_numpy(results_np).to(device=device, dtype=hand_positions.dtype)


def detect_contact_frames(
    hand_positions: np.ndarray,
    object_verts: np.ndarray,
    contact_threshold: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect which frames have contact for each hand.
    
    Used for computing contact metrics (precision, recall, F1).
    
    Args:
        hand_positions: (T, 6) hand positions
        object_verts: (T, K, 3) object vertices
        contact_threshold: threshold (paper uses 0.05 for metrics)
    
    Returns:
        (left_contact, right_contact): (T,) boolean arrays
    """
    T = hand_positions.shape[0]
    left_contact = np.zeros(T, dtype=bool)
    right_contact = np.zeros(T, dtype=bool)
    
    left_hand = hand_positions[:, :3]
    right_hand = hand_positions[:, 3:]
    
    for t in range(T):
        _, left_dist, _ = find_nearest_vertex(left_hand[t], object_verts[t])
        _, right_dist, _ = find_nearest_vertex(right_hand[t], object_verts[t])
        
        left_contact[t] = left_dist < contact_threshold
        right_contact[t] = right_dist < contact_threshold
    
    return left_contact, right_contact


def compute_contact_metrics(
    pred_hands: np.ndarray,
    gt_hands: np.ndarray,
    object_verts: np.ndarray,
    contact_threshold: float = 0.05,  # Paper uses 5cm for metrics
) -> dict:
    """
    Compute contact metrics: precision, recall, F1 score.
    
    As described in OMOMO paper evaluation (Section 5.1).
    
    Args:
        pred_hands: (T, 6) predicted hand positions
        gt_hands: (T, 6) ground truth hand positions
        object_verts: (T, K, 3) object vertices
        contact_threshold: threshold for contact (paper: 5cm)
    
    Returns:
        dict with precision, recall, F1 for each hand and combined
    """
    pred_left_contact, pred_right_contact = detect_contact_frames(
        pred_hands, object_verts, contact_threshold
    )
    gt_left_contact, gt_right_contact = detect_contact_frames(
        gt_hands, object_verts, contact_threshold
    )
    
    def compute_prf(pred, gt):
        """Compute precision, recall, F1 from boolean arrays."""
        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        fn = np.sum(~pred & gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    left_p, left_r, left_f1 = compute_prf(pred_left_contact, gt_left_contact)
    right_p, right_r, right_f1 = compute_prf(pred_right_contact, gt_right_contact)
    
    # Combined (any hand in contact)
    pred_any = pred_left_contact | pred_right_contact
    gt_any = gt_left_contact | gt_right_contact
    combined_p, combined_r, combined_f1 = compute_prf(pred_any, gt_any)
    
    return {
        "left_precision": left_p,
        "left_recall": left_r,
        "left_f1": left_f1,
        "right_precision": right_p,
        "right_recall": right_r,
        "right_f1": right_f1,
        "precision": combined_p,  # C_prec in paper
        "recall": combined_r,      # C_rec in paper
        "f1": combined_f1,         # F1 Score in paper
    }


def compute_hand_jpe(
    pred_hands: np.ndarray,
    gt_hands: np.ndarray,
) -> float:
    """
    Compute Mean Hand Joint Position Error (HandJPE).
    
    Args:
        pred_hands: (T, 6) predicted hand positions
        gt_hands: (T, 6) ground truth hand positions
    
    Returns:
        Mean error in centimeters
    """
    # Split into left/right
    pred_left = pred_hands[:, :3]
    pred_right = pred_hands[:, 3:]
    gt_left = gt_hands[:, :3]
    gt_right = gt_hands[:, 3:]
    
    # Compute per-hand errors
    left_error = np.linalg.norm(pred_left - gt_left, axis=1)  # (T,)
    right_error = np.linalg.norm(pred_right - gt_right, axis=1)  # (T,)
    
    # Average over time and hands, convert to cm
    mean_error = (left_error.mean() + right_error.mean()) / 2
    return mean_error * 100  # meters to centimeters


class ContactConstraintProcessor:
    """
    Processor class for applying contact constraints during inference.
    
    Maintains state for batch processing and provides utilities for
    determining contact mode (single-hand vs two-hand manipulation).
    """
    
    def __init__(
        self,
        contact_threshold: float = 0.03,
        two_hand_threshold: float = 0.03,  # Both hands within this distance = two-handed
    ):
        self.contact_threshold = contact_threshold
        self.two_hand_threshold = two_hand_threshold
    
    def process(
        self,
        hand_positions: np.ndarray,
        object_verts: np.ndarray,
        object_rotations: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply contact constraints and return metadata.
        
        Args:
            hand_positions: (T, 6) predicted hand positions
            object_verts: (T, K, 3) object vertices
            object_rotations: (T, 3, 3) object rotations
        
        Returns:
            rectified_hands: (T, 6) contact-constrained positions
            metadata: dict with contact info
        """
        rectified = apply_contact_constraints(
            hand_positions, object_verts, object_rotations, self.contact_threshold
        )
        
        # Determine manipulation mode
        left_contact, right_contact = detect_contact_frames(
            rectified, object_verts, self.two_hand_threshold
        )
        
        is_two_handed = np.any(left_contact) and np.any(right_contact)
        
        metadata = {
            "left_contact_frames": left_contact,
            "right_contact_frames": right_contact,
            "is_two_handed": is_two_handed,
            "left_first_contact": np.argmax(left_contact) if np.any(left_contact) else -1,
            "right_first_contact": np.argmax(right_contact) if np.any(right_contact) else -1,
        }
        
        return rectified, metadata
