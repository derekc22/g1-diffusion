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


def temporal_smooth_np(
    sequence: np.ndarray,
    strength: float = 0.25,
    window: int = 5,
    iterations: int = 1,
    preserve_ends: bool = True,
) -> np.ndarray:
    if strength <= 0.0 or iterations <= 0 or window <= 1 or sequence.shape[0] < 3:
        return sequence
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out = sequence.astype(np.float32, copy=True)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    original_first = out[0].copy()
    original_last = out[-1].copy()

    for _ in range(iterations):
        padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
        smoothed = np.empty_like(out)
        for dim in range(out.shape[1]):
            smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
        out = (1.0 - strength) * out + strength * smoothed
        if preserve_ends:
            out[0] = original_first
            out[-1] = original_last
    return out.astype(sequence.dtype, copy=False)


def limit_contact_correction_np(
    original: np.ndarray,
    corrected: np.ndarray,
    max_correction: Optional[float],
) -> np.ndarray:
    if max_correction is None or max_correction <= 0.0:
        return corrected

    out = original.astype(np.float32, copy=True)
    corrected = corrected.astype(np.float32, copy=False)
    for hand_idx in range(2):
        start = hand_idx * 3
        end = start + 3
        delta = corrected[:, start:end] - original[:, start:end]
        norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        scale = np.minimum(1.0, max_correction / np.maximum(norm, 1e-8))
        out[:, start:end] = original[:, start:end] + delta * scale
    return out.astype(corrected.dtype, copy=False)


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
    max_contact_offset: Optional[float] = None,
) -> np.ndarray:
    """
    Apply contact constraints to a single hand's trajectory.
    
    Args:
        hand_positions: (T, 3) hand position trajectory
        object_verts: (T, K, 3) object mesh vertices over time
        object_rotations: (T, 3, 3) object rotation matrices over time
        contact_threshold: threshold for contact detection (meters), paper uses 0.03
        max_contact_offset: clamp retained hand-to-surface offset after contact
    
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
    if max_contact_offset is not None and max_contact_offset >= 0.0:
        p_norm = float(np.linalg.norm(p))
        if p_norm > max_contact_offset and p_norm > 1e-8:
            p = p / p_norm * max_contact_offset
    
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
    max_contact_offset: Optional[float] = None,
) -> np.ndarray:
    """
    Apply contact constraints to both hands.
    
    Args:
        hand_positions: (T, 6) = [left_x, left_y, left_z, right_x, right_y, right_z]
        object_verts: (T, K, 3) object mesh vertices
        object_rotations: (T, 3, 3) object rotation matrices
        contact_threshold: threshold for contact detection (meters)
        max_contact_offset: clamp retained hand-to-surface offset after contact
    
    Returns:
        (T, 6) rectified hand positions
    """
    # Split left and right hands
    left_hand = hand_positions[:, :3]   # (T, 3)
    right_hand = hand_positions[:, 3:]  # (T, 3)
    
    # Apply constraints separately
    left_rectified = apply_contact_constraints_single_hand(
        left_hand, object_verts, object_rotations, contact_threshold, max_contact_offset
    )
    right_rectified = apply_contact_constraints_single_hand(
        right_hand, object_verts, object_rotations, contact_threshold, max_contact_offset
    )
    
    # Combine
    return np.concatenate([left_rectified, right_rectified], axis=-1)


def apply_labeled_contact_constraints(
    hand_positions: np.ndarray,
    object_verts: np.ndarray,
    object_rotations: np.ndarray,
    contact_labels: np.ndarray,
    max_contact_offset: Optional[float] = 0.02,
) -> np.ndarray:
    labels = np.asarray(contact_labels).reshape(-1)[: hand_positions.shape[0]] > 0
    if not np.any(labels):
        return hand_positions.copy()

    T = hand_positions.shape[0]
    rectified = hand_positions.copy()
    t = 0

    while t < T:
        if not labels[t]:
            t += 1
            continue

        start = t
        while t < T and labels[t]:
            t += 1
        end = t

        left_idx, left_dist, left_vertex = find_nearest_vertex(
            rectified[start, :3], object_verts[start]
        )
        right_idx, right_dist, right_vertex = find_nearest_vertex(
            rectified[start, 3:], object_verts[start]
        )
        if right_dist < left_dist:
            hand_slice = slice(3, 6)
            nearest_idx = right_idx
            nearest = right_vertex
        else:
            hand_slice = slice(0, 3)
            nearest_idx = left_idx
            nearest = left_vertex

        p = rectified[start, hand_slice] - nearest
        if max_contact_offset is not None and max_contact_offset >= 0.0:
            p_norm = float(np.linalg.norm(p))
            if p_norm > max_contact_offset and p_norm > 1e-8:
                p = p / p_norm * max_contact_offset

        R_start_inv = np.linalg.inv(object_rotations[start])
        for frame in range(start, end):
            rotated_offset = object_rotations[frame] @ R_start_inv @ p
            rectified[frame, hand_slice] = object_verts[frame, nearest_idx] + rotated_offset

    return rectified


def apply_contact_constraints_batch(
    hand_positions: torch.Tensor,
    object_verts: torch.Tensor,
    object_rotations: torch.Tensor,
    contact_threshold: float = 0.03,
    smooth_strength: float = 0.25,
    smooth_window: int = 5,
    smooth_iterations: int = 1,
    max_contact_offset: Optional[float] = 0.02,
    max_contact_correction: Optional[float] = 0.06,
) -> torch.Tensor:
    """
    Batch version of contact constraints for inference.
    
    Args:
        hand_positions: (B, T, 6) hand positions
        object_verts: (B, T, K, 3) object vertices
        object_rotations: (B, T, 3, 3) object rotations
        contact_threshold: contact detection threshold
        smooth_strength: temporal smoothing blend before contact constraints
        smooth_window: odd moving-average window for smoothing
        smooth_iterations: smoothing iterations
        max_contact_offset: clamp retained hand-to-surface offset after contact
        max_contact_correction: maximum per-frame correction applied to each hand
    
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
        hands_b = temporal_smooth_np(
            hands_np[b],
            strength=smooth_strength,
            window=smooth_window,
            iterations=smooth_iterations,
        )
        rectified = apply_contact_constraints(
            hands_b, verts_np[b], rots_np[b], contact_threshold, max_contact_offset
        )
        rectified = limit_contact_correction_np(hands_b, rectified, max_contact_correction)
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
        contact_search_threshold: Optional[float] = None,
        max_contact_offset: Optional[float] = 0.02,
        max_contact_correction: Optional[float] = 0.06,
        smooth_strength: float = 0.25,
        smooth_window: int = 5,
        smooth_iterations: int = 1,
        fallback_contact_search_threshold: Optional[float] = None,
        fallback_max_contact_correction: Optional[float] = None,
    ):
        self.contact_threshold = contact_threshold
        self.two_hand_threshold = two_hand_threshold
        self.contact_search_threshold = (
            max(contact_threshold, 0.08)
            if contact_search_threshold is None
            else contact_search_threshold
        )
        self.max_contact_offset = max_contact_offset
        self.max_contact_correction = max_contact_correction
        self.smooth_strength = smooth_strength
        self.smooth_window = smooth_window
        self.smooth_iterations = smooth_iterations
        self.fallback_contact_search_threshold = fallback_contact_search_threshold
        self.fallback_max_contact_correction = fallback_max_contact_correction
    
    def process(
        self,
        hand_positions: np.ndarray,
        object_verts: np.ndarray,
        object_rotations: np.ndarray,
        contact_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply contact constraints and return metadata.
        
        Args:
            hand_positions: (T, 6) predicted hand positions
            object_verts: (T, K, 3) object vertices
            object_rotations: (T, 3, 3) object rotations
            contact_labels: optional (T,) contact intervals from data
        
        Returns:
            rectified_hands: (T, 6) contact-constrained positions
            metadata: dict with contact info
        """
        smoothed_hands = temporal_smooth_np(
            hand_positions,
            strength=self.smooth_strength,
            window=self.smooth_window,
            iterations=self.smooth_iterations,
        )
        if contact_labels is not None:
            rectified = apply_labeled_contact_constraints(
                smoothed_hands,
                object_verts,
                object_rotations,
                contact_labels,
                self.max_contact_offset,
            )
        else:
            rectified = apply_contact_constraints(
                smoothed_hands,
                object_verts,
                object_rotations,
                self.contact_search_threshold,
                self.max_contact_offset,
            )
        rectified = limit_contact_correction_np(
            smoothed_hands,
            rectified,
            self.max_contact_correction,
        )
        
        # Determine manipulation mode
        left_contact, right_contact = detect_contact_frames(
            rectified, object_verts, self.two_hand_threshold
        )
        used_fallback = False

        if (
            self.fallback_contact_search_threshold is not None
            and self.fallback_contact_search_threshold > self.contact_search_threshold
            and not np.any(left_contact | right_contact)
        ):
            use_labeled_fallback = False
            if contact_labels is not None:
                labels = np.asarray(contact_labels).reshape(-1)[: smoothed_hands.shape[0]] > 0
                if np.any(labels):
                    min_labeled_dist = np.inf
                    for frame in np.where(labels)[0]:
                        left_dist = np.linalg.norm(
                            object_verts[frame] - smoothed_hands[frame, :3], axis=1
                        ).min()
                        right_dist = np.linalg.norm(
                            object_verts[frame] - smoothed_hands[frame, 3:], axis=1
                        ).min()
                        min_labeled_dist = min(min_labeled_dist, float(left_dist), float(right_dist))
                    use_labeled_fallback = min_labeled_dist <= self.fallback_contact_search_threshold

            if use_labeled_fallback:
                fallback = apply_labeled_contact_constraints(
                    smoothed_hands,
                    object_verts,
                    object_rotations,
                    contact_labels,
                    self.max_contact_offset,
                )
            else:
                fallback = apply_contact_constraints(
                    smoothed_hands,
                    object_verts,
                    object_rotations,
                    self.fallback_contact_search_threshold,
                    self.max_contact_offset,
                )
            fallback = limit_contact_correction_np(
                smoothed_hands,
                fallback,
                (
                    self.fallback_max_contact_correction
                    if self.fallback_max_contact_correction is not None
                    else self.max_contact_correction
                ),
            )
            fallback_left, fallback_right = detect_contact_frames(
                fallback, object_verts, self.two_hand_threshold
            )
            if np.any(fallback_left | fallback_right):
                rectified = fallback
                left_contact = fallback_left
                right_contact = fallback_right
                used_fallback = True
                used_labeled_fallback = use_labeled_fallback
            else:
                used_labeled_fallback = False
        else:
            used_labeled_fallback = False
        
        is_two_handed = np.any(left_contact) and np.any(right_contact)
        
        metadata = {
            "left_contact_frames": left_contact,
            "right_contact_frames": right_contact,
            "is_two_handed": is_two_handed,
            "left_first_contact": np.argmax(left_contact) if np.any(left_contact) else -1,
            "right_first_contact": np.argmax(right_contact) if np.any(right_contact) else -1,
            "used_contact_labels": contact_labels is not None,
            "contact_search_threshold": self.contact_search_threshold,
            "max_contact_offset": self.max_contact_offset,
            "max_contact_correction": self.max_contact_correction,
            "used_fallback": used_fallback,
            "used_labeled_fallback": used_labeled_fallback,
            "fallback_contact_search_threshold": self.fallback_contact_search_threshold,
            "fallback_max_contact_correction": self.fallback_max_contact_correction,
            "smoothing_strength": self.smooth_strength,
            "smoothing_window": self.smooth_window,
        }
        
        return rectified, metadata
