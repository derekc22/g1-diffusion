"""
Stage 1 Dataset: Object Geometry → Hand Positions

This dataset provides windowed sequences of:
- Input: BPS encoding + object centroid (object geometry features)
- Target: Hand positions (left_xyz, right_xyz)

Used for training the Stage 1 diffusion model that generates hand positions
from object motion.
"""

import os
import glob
import pickle
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import sys
import types
import torch
from torch.utils.data import Dataset

from utils.normalization import compute_mean_std

# ---------------------------------------------------------------------------
# Compatibility shim for pickles created by NumPy >= 2.0 in envs with older NumPy
# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------


class HandMotionDataset(Dataset):
    """
    Dataset for Stage 1: Object geometry → Hand positions.
    
    Each underlying file contains:
      - bps_encoding: (T, 1024, 3) - BPS representation of object
      - object_centroid: (T, 3) - object centroid trajectory
      - hand_positions: (T, 6) - [left_x, left_y, left_z, right_x, right_y, right_z]
      - object_verts: (T, K, 3) - object mesh vertices (for contact constraints)
      - object_rotation: (T, 3, 3) - object rotation matrices (for contact constraints)
    
    Returns per sample:
      - hand_positions: (window_size, 6) - target hand positions
      - bps_encoding: (window_size, 1024, 3) or (window_size, 3072) - object BPS
      - object_centroid: (window_size, 3) - object centroid
      - object_verts: (window_size, K, 3) - for contact constraint computation
      - object_rotation: (window_size, 3, 3) - for contact constraint computation
    """
    
    def __init__(
        self,
        root_dir: str,
        window_size: int,
        stride: int,
        min_seq_len: int,
        train: bool,
        train_split: float = 0.9,
        preload: bool = True,
        hand_mean: Optional[np.ndarray] = None,
        hand_std: Optional[np.ndarray] = None,
        flatten_bps: bool = True,  # Whether to flatten BPS to (T, 3072)
        include_object_geometry: bool = False,  # Include object_verts/rotation (variable size, can't batch)
    ):
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.min_seq_len = min_seq_len
        self.preload = preload
        self.flatten_bps = flatten_bps
        self.include_object_geometry = include_object_geometry
        
        # Normalization stats (only for hand positions)
        self.hand_mean = hand_mean
        self.hand_std = hand_std
        
        # Find all pkl files
        file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if not file_paths:
            raise RuntimeError(f"No motion .pkl files found in {root_dir}")
        
        # Train/val split at file level
        num_files = len(file_paths)
        self.num_files = num_files
        split_idx = int(num_files * train_split)
        
        if train:
            self.file_paths = file_paths[:split_idx]
        else:
            self.file_paths = file_paths[split_idx:]
        
        # Cache and windows
        self.windows: List[Tuple[int, int]] = []
        self._file_cache: Dict[int, Dict[str, Any]] = {}
        
        for fi, path in enumerate(self.file_paths):
            data = self._load_file(fi, path)
            if data is None:
                continue
            T = data["hand_positions"].shape[0]
            if T < self.min_seq_len or T < self.window_size:
                continue
            for start in range(0, T - self.window_size + 1, self.stride):
                self.windows.append((fi, start))
        
        if not self.windows:
            raise RuntimeError("No windows constructed. Check min_seq_len and window_size.")
        
        # Compute normalization stats for hand positions during training
        if train and self.hand_mean is None:
            all_hands = []
            for idx in range(len(self.windows)):
                sample = self._get_window(idx, normalized=False)
                all_hands.append(sample["hand_positions"].numpy())
            all_hands_arr = np.concatenate(all_hands, axis=0)  # (N, 6)
            mean, std = compute_mean_std(all_hands_arr)
            self.hand_mean = torch.from_numpy(mean).float()
            self.hand_std = torch.from_numpy(std).float()
    
    def _load_file(self, file_idx: int, path: str) -> Optional[Dict[str, Any]]:
        """Load and validate a single pkl file."""
        if self.preload and file_idx in self._file_cache:
            return self._file_cache[file_idx]
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None
        
        # Check required keys
        required_keys = ["hand_positions", "bps_encoding", "object_centroid"]
        for key in required_keys:
            if key not in data:
                print(f"Warning: Missing key '{key}' in {path}")
                return None
        
        # Extract and validate data
        hand_positions = np.asarray(data["hand_positions"], dtype=np.float32)
        bps_encoding = np.asarray(data["bps_encoding"], dtype=np.float32)
        object_centroid = np.asarray(data["object_centroid"], dtype=np.float32)
        
        if hand_positions.shape[1] != 6:
            print(f"Warning: Invalid hand_positions shape {hand_positions.shape} in {path}")
            return None
        
        data_proc = {
            "hand_positions": hand_positions,
            "bps_encoding": bps_encoding,
            "object_centroid": object_centroid,
            "seq_name": data.get("seq_name", os.path.splitext(os.path.basename(path))[0]),
            "fps": float(data.get("fps", 30.0)),
        }
        
        # Only load object geometry if explicitly requested (for contact constraints)
        # These are large and have variable sizes, so skip during training
        if self.include_object_geometry:
            object_verts = data.get("object_verts")
            object_rotation = data.get("object_rotation")
            data_proc["object_verts"] = np.asarray(object_verts, dtype=np.float32) if object_verts is not None else None
            data_proc["object_rotation"] = np.asarray(object_rotation, dtype=np.float32) if object_rotation is not None else None
        else:
            data_proc["object_verts"] = None
            data_proc["object_rotation"] = None
        
        if self.preload:
            self._file_cache[file_idx] = data_proc
        
        return data_proc
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def _get_window(self, idx: int, normalized: bool) -> Dict[str, torch.Tensor]:
        """Get a single windowed sample."""
        file_idx, start = self.windows[idx]
        path = self.file_paths[file_idx]
        data = self._load_file(file_idx, path)
        
        T = self.window_size
        end = start + T
        
        # Extract window
        hand_positions = data["hand_positions"][start:end]  # (T, 6)
        bps_encoding = data["bps_encoding"][start:end]      # (T, 1024, 3)
        object_centroid = data["object_centroid"][start:end]  # (T, 3)
        
        # Convert to tensors
        hand_t = torch.from_numpy(hand_positions).float()
        centroid_t = torch.from_numpy(object_centroid).float()
        
        # Handle BPS
        if self.flatten_bps:
            bps_flat = bps_encoding.reshape(T, -1)  # (T, 3072)
            bps_t = torch.from_numpy(bps_flat).float()
        else:
            bps_t = torch.from_numpy(bps_encoding).float()
        
        # Normalize hand positions if requested
        if normalized and self.hand_mean is not None:
            mean = self.hand_mean.view(1, -1)
            std = self.hand_std.view(1, -1)
            hand_t = (hand_t - mean) / std
        
        result = {
            "hand_positions": hand_t,
            "bps_encoding": bps_t,
            "object_centroid": centroid_t,
            "seq_name": data["seq_name"],
            "fps": data["fps"],
            "file_idx": file_idx,
            "start": start,
        }
        
        # Add optional data for contact constraints (only when explicitly requested)
        # These have variable sizes across samples and can't be batched
        if self.include_object_geometry:
            if data["object_verts"] is not None:
                result["object_verts"] = torch.from_numpy(data["object_verts"][start:end]).float()
            if data["object_rotation"] is not None:
                result["object_rotation"] = torch.from_numpy(data["object_rotation"][start:end]).float()
        
        return result
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get normalized sample."""
        return self._get_window(idx, normalized=True)
    
    def get_unnormalized(self, idx: int) -> Dict[str, Any]:
        """Get unnormalized sample (for visualization/evaluation)."""
        return self._get_window(idx, normalized=False)
    
    def denormalize_hands(self, hand_positions: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized hand positions back to original scale.
        
        Args:
            hand_positions: (..., 6) normalized positions
        
        Returns:
            (..., 6) denormalized positions
        """
        if self.hand_mean is None:
            return hand_positions
        
        mean = self.hand_mean.to(hand_positions.device)
        std = self.hand_std.to(hand_positions.device)
        
        # Handle various shapes
        orig_shape = hand_positions.shape
        if len(orig_shape) == 3:  # (B, T, 6)
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        elif len(orig_shape) == 2:  # (T, 6) or (B, 6)
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        
        return hand_positions * std + mean


class ObjectMotionDatasetWithHands(Dataset):
    """
    Extended dataset that provides both object geometry and hand positions.
    
    Used for:
    1. Stage 1 training: object → hands
    2. Stage 2 training: hands → full body (with object context)
    3. End-to-end evaluation
    """
    
    def __init__(
        self,
        root_dir: str,
        window_size: int,
        stride: int,
        min_seq_len: int,
        train: bool,
        train_split: float = 0.9,
        preload: bool = True,
        # Separate normalization for hands and body
        hand_mean: Optional[np.ndarray] = None,
        hand_std: Optional[np.ndarray] = None,
    ):
        """Initialize dataset with same parameters as HandMotionDataset."""
        # Delegate to HandMotionDataset for most functionality
        self._base = HandMotionDataset(
            root_dir=root_dir,
            window_size=window_size,
            stride=stride,
            min_seq_len=min_seq_len,
            train=train,
            train_split=train_split,
            preload=preload,
            hand_mean=hand_mean,
            hand_std=hand_std,
            flatten_bps=True,
        )
        
        # Expose normalization stats
        self.hand_mean = self._base.hand_mean
        self.hand_std = self._base.hand_std
        self.windows = self._base.windows
        self.file_paths = self._base.file_paths
        self.num_files = self._base.num_files
    
    def __len__(self) -> int:
        return len(self._base)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._base[idx]
    
    def denormalize_hands(self, hand_positions: torch.Tensor) -> torch.Tensor:
        return self._base.denormalize_hands(hand_positions)
