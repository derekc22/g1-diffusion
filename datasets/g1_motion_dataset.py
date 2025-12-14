"""
Stage 2 Dataset: Hand Positions → Full-Body Motion (OMOMO)

This dataset provides windowed sequences for Stage 2 training:
- Input/Target: Full-body robot motion (root_pos, root_rot, dof_pos)
- Conditioning: Hand positions (6D from Stage 1 output)

In the OMOMO paper, Stage 2 is trained using "human motion data only" -
meaning it learns to generate plausible full-body poses that reach the
given hand positions, without explicit object conditioning.
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

from utils.rotation import quat_to_rot6d_xyzw
from utils.normalization import compute_mean_std

# ---------------------------------------------------------------------------
# Compatibility shim for pickles created by NumPy >= 2.0
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


class G1MotionDatasetHandCond(Dataset):
    """
    Dataset for Stage 2: Hand positions → Full-body motion.
    
    Each underlying file contains:
      - fps: scalar
      - root_pos: (T, 3) root position
      - root_rot: (T, 4) root rotation quaternion (xyzw)
      - dof_pos: (T, Dq) joint positions
      - hand_positions: (T, 6) hand positions [left_xyz, right_xyz]
    
    State vector s_t = [root_pos, root_rot_6d, dof_pos]
    Conditioning c_t = [hand_positions]
    
    Following OMOMO: Stage 2 generates full-body poses conditioned on
    rectified hand positions (after contact constraints from Stage 1).
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
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        hand_mean: Optional[np.ndarray] = None,
        hand_std: Optional[np.ndarray] = None,
        normalize_hands: bool = False,  # Whether to normalize hand conditioning
    ):
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.min_seq_len = min_seq_len
        self.preload = preload
        self.normalize_hands = normalize_hands
        
        # Normalization stats
        self.state_mean = state_mean
        self.state_std = state_std
        self.hand_mean = hand_mean
        self.hand_std = hand_std
        
        # Find files
        file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if not file_paths:
            raise RuntimeError(f"No motion .pkl files found in {root_dir}")
        
        # Train/val split
        num_files = len(file_paths)
        self.num_files = num_files
        split_idx = int(num_files * train_split)
        
        if train:
            self.file_paths = file_paths[:split_idx]
        else:
            self.file_paths = file_paths[split_idx:]
        
        # Build windows
        self.windows: List[Tuple[int, int]] = []
        self._file_cache: Dict[int, Dict[str, Any]] = {}
        
        for fi, path in enumerate(self.file_paths):
            data = self._load_file(fi, path)
            if data is None:
                continue
            T = data["root_pos"].shape[0]
            if T < self.min_seq_len or T < self.window_size:
                continue
            for start in range(0, T - self.window_size + 1, self.stride):
                self.windows.append((fi, start))
        
        if not self.windows:
            raise RuntimeError("No windows constructed.")
        
        # Compute normalization stats during training
        if train and self.state_mean is None:
            all_states = []
            all_hands = []
            for idx in range(len(self.windows)):
                state, hand_cond, _ = self._get_window(idx, normalized=False)
                all_states.append(state.numpy())
                all_hands.append(hand_cond.numpy())
            
            states_arr = np.concatenate(all_states, axis=0)
            state_mean, state_std = compute_mean_std(states_arr)
            self.state_mean = torch.from_numpy(state_mean).float()
            self.state_std = torch.from_numpy(state_std).float()
            
            if self.normalize_hands:
                hands_arr = np.concatenate(all_hands, axis=0)
                hand_mean, hand_std = compute_mean_std(hands_arr)
                self.hand_mean = torch.from_numpy(hand_mean).float()
                self.hand_std = torch.from_numpy(hand_std).float()
    
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
        required = ["root_pos", "root_rot", "dof_pos", "hand_positions"]
        for key in required:
            if key not in data:
                print(f"Warning: Missing '{key}' in {path}")
                return None
        
        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_rot = np.asarray(data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        hand_positions = np.asarray(data["hand_positions"], dtype=np.float32)
        
        # Validate shapes
        if root_pos.shape[1] != 3 or root_rot.shape[1] != 4:
            print(f"Warning: Invalid root shape in {path}")
            return None
        if hand_positions.shape[1] != 6:
            print(f"Warning: Invalid hand_positions shape in {path}")
            return None
        
        data_proc = {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "hand_positions": hand_positions,
            "seq_name": data.get("seq_name", os.path.splitext(os.path.basename(path))[0]),
            "fps": float(data.get("fps", 30.0)),
        }
        
        if self.preload:
            self._file_cache[file_idx] = data_proc
        
        return data_proc
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def _get_window(
        self,
        idx: int,
        normalized: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get a single windowed sample."""
        file_idx, start = self.windows[idx]
        path = self.file_paths[file_idx]
        data = self._load_file(file_idx, path)
        
        T = self.window_size
        end = start + T
        
        # Extract window
        root_pos = data["root_pos"][start:end]      # (T, 3)
        root_rot = data["root_rot"][start:end]      # (T, 4)
        dof_pos = data["dof_pos"][start:end]        # (T, Dq)
        hand_positions = data["hand_positions"][start:end]  # (T, 6)
        
        # Convert root rotation to 6D
        root_rot_t = torch.from_numpy(root_rot).float()
        root_rot_6d = quat_to_rot6d_xyzw(root_rot_t).numpy()  # (T, 6)
        
        # Build state vector: [root_pos, root_rot_6d, dof_pos]
        state = np.concatenate([root_pos, root_rot_6d, dof_pos], axis=-1)  # (T, D)
        state_t = torch.from_numpy(state).float()
        
        # Conditioning: hand positions
        hand_t = torch.from_numpy(hand_positions).float()  # (T, 6)
        
        meta = {
            "file_idx": file_idx,
            "start": start,
            "fps": data["fps"],
            "seq_name": data["seq_name"],
        }
        
        # Apply normalization
        if normalized:
            if self.state_mean is not None:
                state_t = (state_t - self.state_mean.view(1, -1)) / self.state_std.view(1, -1)
            
            if self.normalize_hands and self.hand_mean is not None:
                hand_t = (hand_t - self.hand_mean.view(1, -1)) / self.hand_std.view(1, -1)
        
        return state_t, hand_t, meta
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get normalized sample for training."""
        state, hand_cond, meta = self._get_window(idx, normalized=True)
        return {
            "state": state,            # (T, D)
            "cond": hand_cond,         # (T, 6) - hand positions
            "file_idx": meta["file_idx"],
            "start": meta["start"],
            "fps": meta["fps"],
            "seq_name": meta["seq_name"],
        }
    
    def get_unnormalized(self, idx: int) -> Dict[str, Any]:
        """Get unnormalized sample."""
        state, hand_cond, meta = self._get_window(idx, normalized=False)
        return {
            "state": state,
            "cond": hand_cond,
            **meta,
        }
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Denormalize state vector."""
        if self.state_mean is None:
            return state
        mean = self.state_mean.to(state.device)
        std = self.state_std.to(state.device)
        if state.ndim == 3:  # (B, T, D)
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:  # (T, D)
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return state * std + mean
    
    def denormalize_hands(self, hands: torch.Tensor) -> torch.Tensor:
        """Denormalize hand positions if they were normalized."""
        if not self.normalize_hands or self.hand_mean is None:
            return hands
        mean = self.hand_mean.to(hands.device)
        std = self.hand_std.to(hands.device)
        if hands.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return hands * std + mean


class UnifiedOmomoDataset(Dataset):
    """
    Unified dataset for OMOMO-style training.
    
    Provides data for both stages:
    - Stage 1: object geometry → hand positions
    - Stage 2: hand positions → full body
    
    Can be used for joint training or separate training of each stage.
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
        # Normalization stats
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        hand_mean: Optional[np.ndarray] = None,
        hand_std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.min_seq_len = min_seq_len
        self.preload = preload
        
        # Normalization stats
        self.state_mean = state_mean
        self.state_std = state_std
        self.hand_mean = hand_mean  
        self.hand_std = hand_std
        
        # Find files
        file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if not file_paths:
            raise RuntimeError(f"No motion .pkl files found in {root_dir}")
        
        # Train/val split
        num_files = len(file_paths)
        self.num_files = num_files
        split_idx = int(num_files * train_split)
        
        if train:
            self.file_paths = file_paths[:split_idx]
        else:
            self.file_paths = file_paths[split_idx:]
        
        # Build windows
        self.windows: List[Tuple[int, int]] = []
        self._file_cache: Dict[int, Dict[str, Any]] = {}
        
        for fi, path in enumerate(self.file_paths):
            data = self._load_file(fi, path)
            if data is None:
                continue
            T = data["root_pos"].shape[0]
            if T < self.min_seq_len or T < self.window_size:
                continue
            for start in range(0, T - self.window_size + 1, self.stride):
                self.windows.append((fi, start))
        
        if not self.windows:
            raise RuntimeError("No windows constructed.")
        
        # Compute normalization stats during training
        if train and self.state_mean is None:
            self._compute_norm_stats()
    
    def _compute_norm_stats(self):
        """Compute normalization statistics for state and hands."""
        all_states = []
        all_hands = []
        
        for idx in range(len(self.windows)):
            sample = self._get_raw_window(idx)
            all_states.append(sample["state"])
            all_hands.append(sample["hand_positions"])
        
        states_arr = np.concatenate(all_states, axis=0)
        hands_arr = np.concatenate(all_hands, axis=0)
        
        state_mean, state_std = compute_mean_std(states_arr)
        hand_mean, hand_std = compute_mean_std(hands_arr)
        
        self.state_mean = torch.from_numpy(state_mean).float()
        self.state_std = torch.from_numpy(state_std).float()
        self.hand_mean = torch.from_numpy(hand_mean).float()
        self.hand_std = torch.from_numpy(hand_std).float()
    
    def _load_file(self, file_idx: int, path: str) -> Optional[Dict[str, Any]]:
        """Load and cache a file."""
        if self.preload and file_idx in self._file_cache:
            return self._file_cache[file_idx]
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None
        
        # Check all required keys
        required = [
            "root_pos", "root_rot", "dof_pos", "hand_positions",
            "bps_encoding", "object_centroid",
        ]
        for key in required:
            if key not in data:
                print(f"Warning: Missing '{key}' in {path}")
                return None
        
        data_proc = {
            "root_pos": np.asarray(data["root_pos"], dtype=np.float32),
            "root_rot": np.asarray(data["root_rot"], dtype=np.float32),
            "dof_pos": np.asarray(data["dof_pos"], dtype=np.float32),
            "hand_positions": np.asarray(data["hand_positions"], dtype=np.float32),
            "bps_encoding": np.asarray(data["bps_encoding"], dtype=np.float32),
            "object_centroid": np.asarray(data["object_centroid"], dtype=np.float32),
            "object_verts": np.asarray(data["object_verts"], dtype=np.float32) if "object_verts" in data else None,
            "object_rotation": np.asarray(data["object_rotation"], dtype=np.float32) if "object_rotation" in data else None,
            "seq_name": data.get("seq_name", os.path.basename(path)),
            "fps": float(data.get("fps", 30.0)),
        }
        
        if self.preload:
            self._file_cache[file_idx] = data_proc
        
        return data_proc
    
    def _get_raw_window(self, idx: int) -> Dict[str, np.ndarray]:
        """Get raw (unnormalized) window data."""
        file_idx, start = self.windows[idx]
        path = self.file_paths[file_idx]
        data = self._load_file(file_idx, path)
        
        T = self.window_size
        end = start + T
        
        # Robot state
        root_pos = data["root_pos"][start:end]
        root_rot = data["root_rot"][start:end]
        dof_pos = data["dof_pos"][start:end]
        
        # Convert to 6D rotation
        root_rot_t = torch.from_numpy(root_rot).float()
        root_rot_6d = quat_to_rot6d_xyzw(root_rot_t).numpy()
        
        # Build state
        state = np.concatenate([root_pos, root_rot_6d, dof_pos], axis=-1)
        
        return {
            "state": state,
            "hand_positions": data["hand_positions"][start:end],
            "bps_encoding": data["bps_encoding"][start:end],
            "object_centroid": data["object_centroid"][start:end],
            "object_verts": data["object_verts"][start:end] if data["object_verts"] is not None else None,
            "object_rotation": data["object_rotation"][start:end] if data["object_rotation"] is not None else None,
            "seq_name": data["seq_name"],
            "fps": data["fps"],
            "file_idx": file_idx,
            "start": start,
        }
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get normalized sample."""
        raw = self._get_raw_window(idx)
        
        # Convert to tensors
        state = torch.from_numpy(raw["state"]).float()
        hands = torch.from_numpy(raw["hand_positions"]).float()
        bps = torch.from_numpy(raw["bps_encoding"].reshape(self.window_size, -1)).float()
        centroid = torch.from_numpy(raw["object_centroid"]).float()
        
        # Normalize
        if self.state_mean is not None:
            state = (state - self.state_mean.view(1, -1)) / self.state_std.view(1, -1)
        if self.hand_mean is not None:
            hands = (hands - self.hand_mean.view(1, -1)) / self.hand_std.view(1, -1)
        
        result = {
            "state": state,                # (T, D) - full body state
            "hand_positions": hands,       # (T, 6) - hand positions (normalized)
            "bps_encoding": bps,           # (T, 3072) - flattened BPS
            "object_centroid": centroid,   # (T, 3) - object centroid
            "seq_name": raw["seq_name"],
            "fps": raw["fps"],
        }
        
        # Optional: object data for contact constraints
        if raw["object_verts"] is not None:
            result["object_verts"] = torch.from_numpy(raw["object_verts"]).float()
        if raw["object_rotation"] is not None:
            result["object_rotation"] = torch.from_numpy(raw["object_rotation"]).float()
        
        return result
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_mean is None:
            return state
        mean = self.state_mean.to(state.device)
        std = self.state_std.to(state.device)
        if state.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return state * std + mean
    
    def denormalize_hands(self, hands: torch.Tensor) -> torch.Tensor:
        if self.hand_mean is None:
            return hands
        mean = self.hand_mean.to(hands.device)
        std = self.hand_std.to(hands.device)
        if hands.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return hands * std + mean
