"""
Dataset classes for HuggingFace retargeted motion data.

Two dataset classes for the two-stage diffusion pipeline:
  - HFHandMotionDataset (Stage 1): Object motion → Hand positions
  - HFFullBodyDataset (Stage 2): Hand positions → Full-body motion

These load preprocessed PKL files produced by preprocess_hf_data.py.
The PKL format has: root_pos, root_rot, dof_pos, hand_positions,
and optionally object_pos, object_rot, object_lin_vel, object_ang_vel.
"""

import os
import glob
import pickle
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.rotation import quat_to_rot6d_xyzw
from utils.normalization import compute_mean_std


# ---------------------------------------------------------------------------
# Stage 1 Dataset: Object Motion Features → Hand Positions
# ---------------------------------------------------------------------------

class HFHandMotionDataset(Dataset):
    """
    Stage 1 dataset: predicts hand positions from object motion features.

    Unlike the original OMOMO dataset which uses BPS encoding (1024×3 = 3072D),
    this dataset uses compact object motion features:
        - object_pos: (T, 3) position
        - object_rot_6d: (T, 6) rotation in 6D representation
        - object_lin_vel: (T, 3) linear velocity
        - object_ang_vel: (T, 3) angular velocity
        Total conditioning: 15D per frame

    Target: hand_positions (T, 6) = [left_xyz, right_xyz]

    For motions without object data, the object features are zeros
    and this effectively becomes unconditional hand generation.
    """

    OBJECT_FEATURE_DIM = 15  # 3 + 6 + 3 + 3

    def __init__(
        self,
        root_dir: str,
        window_size: int,
        stride: int,
        min_seq_len: int = 30,
        train: bool = True,
        train_split: float = 0.9,
        preload: bool = True,
        hand_mean: Optional[torch.Tensor] = None,
        hand_std: Optional[torch.Tensor] = None,
        require_object: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.min_seq_len = min_seq_len
        self.preload = preload
        self.require_object = require_object

        self.hand_mean = hand_mean
        self.hand_std = hand_std

        # Find PKL files
        file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if not file_paths:
            raise RuntimeError(f"No .pkl files found in {root_dir}")

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
            T = data["hand_positions"].shape[0]
            if T < self.min_seq_len or T < self.window_size:
                continue
            for start in range(0, T - self.window_size + 1, self.stride):
                self.windows.append((fi, start))

        if not self.windows:
            raise RuntimeError("No windows constructed. Check data and min_seq_len.")

        print(f"  HFHandMotionDataset: {len(self.windows)} windows from {len(self.file_paths)} files")

        # Compute normalization stats
        if train and self.hand_mean is None:
            self._compute_norm_stats()

    def _load_file(self, file_idx: int, path: str) -> Optional[Dict[str, Any]]:
        if self.preload and file_idx in self._file_cache:
            return self._file_cache[file_idx]

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None

        # Validate required fields
        for key in ["root_pos", "root_rot", "dof_pos", "hand_positions"]:
            if key not in data:
                print(f"Warning: Missing '{key}' in {path}")
                return None

        hand_pos = np.asarray(data["hand_positions"], dtype=np.float32)
        T = hand_pos.shape[0]

        if hand_pos.shape[1] != 6:
            print(f"Warning: Invalid hand_positions shape in {path}")
            return None

        # Check for object data
        has_object = "object_pos" in data and data["object_pos"] is not None
        if self.require_object and not has_object:
            print(f"Warning: No object data in {path}, skipping (require_object=True)")
            return None

        # Build object features: [pos(3), rot_6d(6), lin_vel(3), ang_vel(3)] = 15D
        if has_object:
            obj_pos = np.asarray(data["object_pos"], dtype=np.float32)

            # Object rotation → 6D
            obj_rot_raw = data.get("object_rot")
            if obj_rot_raw is not None:
                obj_rot_raw = np.asarray(obj_rot_raw, dtype=np.float32)
                if obj_rot_raw.shape[1] == 4:
                    obj_rot_6d = quat_to_rot6d_xyzw(
                        torch.from_numpy(obj_rot_raw).float()
                    ).numpy()
                elif obj_rot_raw.shape[1] == 6:
                    obj_rot_6d = obj_rot_raw
                elif obj_rot_raw.shape[1] == 9:
                    # Flattened rotation matrix
                    from utils.rotation import mat_to_quat_xyzw
                    quat = mat_to_quat_xyzw(
                        torch.from_numpy(obj_rot_raw.reshape(-1, 3, 3)).float()
                    )
                    obj_rot_6d = quat_to_rot6d_xyzw(quat).numpy()
                else:
                    obj_rot_6d = np.zeros((T, 6), dtype=np.float32)
            else:
                obj_rot_6d = np.zeros((T, 6), dtype=np.float32)

            obj_lin_vel = np.asarray(data.get("object_lin_vel", np.zeros((T, 3))), dtype=np.float32)
            obj_ang_vel = np.asarray(data.get("object_ang_vel", np.zeros((T, 3))), dtype=np.float32)

            # Ensure shapes match
            for arr_name, arr in [("obj_pos", obj_pos), ("obj_rot_6d", obj_rot_6d),
                                   ("obj_lin_vel", obj_lin_vel), ("obj_ang_vel", obj_ang_vel)]:
                if arr.shape[0] != T:
                    # Truncate or pad
                    if arr.shape[0] > T:
                        arr = arr[:T]
                    else:
                        pad = np.zeros((T - arr.shape[0],) + arr.shape[1:], dtype=np.float32)
                        arr = np.concatenate([arr, pad], axis=0)

            object_features = np.concatenate([obj_pos, obj_rot_6d, obj_lin_vel, obj_ang_vel], axis=-1)
        else:
            object_features = np.zeros((T, self.OBJECT_FEATURE_DIM), dtype=np.float32)

        data_proc = {
            "hand_positions": hand_pos,
            "object_features": object_features,
            "has_object": has_object,
            "seq_name": data.get("seq_name", os.path.splitext(os.path.basename(path))[0]),
            "fps": float(data.get("fps", 30.0)),
        }

        if self.preload:
            self._file_cache[file_idx] = data_proc

        return data_proc

    def _compute_norm_stats(self):
        all_hands = []
        for fi, start in self.windows:
            data = self._file_cache.get(fi)
            if data is None:
                data = self._load_file(fi, self.file_paths[fi])
            T = self.window_size
            hand_window = data["hand_positions"][start:start + T]
            all_hands.append(hand_window)

        hands_arr = np.concatenate(all_hands, axis=0)
        mean, std = compute_mean_std(hands_arr)
        self.hand_mean = torch.from_numpy(mean).float()
        self.hand_std = torch.from_numpy(std).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fi, start = self.windows[idx]
        data = self._file_cache.get(fi)
        if data is None:
            data = self._load_file(fi, self.file_paths[fi])

        T = self.window_size
        end = start + T

        hand_t = torch.from_numpy(data["hand_positions"][start:end]).float()
        obj_t = torch.from_numpy(data["object_features"][start:end]).float()

        # Normalize hands
        if self.hand_mean is not None:
            hand_t = (hand_t - self.hand_mean.view(1, -1)) / self.hand_std.view(1, -1)

        return {
            "hand_positions": hand_t,       # (T, 6)
            "object_features": obj_t,       # (T, 15)
            "seq_name": data["seq_name"],
            "fps": data["fps"],
            "file_idx": fi,
            "start": start,
        }

    def denormalize_hands(self, hand_positions: torch.Tensor) -> torch.Tensor:
        if self.hand_mean is None:
            return hand_positions
        mean = self.hand_mean.to(hand_positions.device)
        std = self.hand_std.to(hand_positions.device)
        if hand_positions.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return hand_positions * std + mean


# ---------------------------------------------------------------------------
# Stage 2 Dataset: Hand Positions → Full-Body Motion
# ---------------------------------------------------------------------------

class HFFullBodyDataset(Dataset):
    """
    Stage 2 dataset: predicts full-body motion conditioned on hand positions.

    State vector: [root_pos(3), root_rot_6d(6), dof_pos(29)] = 38D
    Conditioning: hand_positions(6D) = [left_xyz, right_xyz]

    This is functionally identical to G1MotionDatasetHandCond but loads
    from the HuggingFace preprocessed PKL format.
    """

    def __init__(
        self,
        root_dir: str,
        window_size: int,
        stride: int,
        min_seq_len: int = 30,
        train: bool = True,
        train_split: float = 0.99,
        preload: bool = True,
        state_mean: Optional[torch.Tensor] = None,
        state_std: Optional[torch.Tensor] = None,
        hand_mean: Optional[torch.Tensor] = None,
        hand_std: Optional[torch.Tensor] = None,
        normalize_hands: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.min_seq_len = min_seq_len
        self.preload = preload
        self.normalize_hands = normalize_hands

        self.state_mean = state_mean
        self.state_std = state_std
        self.hand_mean = hand_mean
        self.hand_std = hand_std

        # Find PKL files
        file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if not file_paths:
            raise RuntimeError(f"No .pkl files found in {root_dir}")

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

        print(f"  HFFullBodyDataset: {len(self.windows)} windows from {len(self.file_paths)} files")

        # Compute normalization stats
        if train and self.state_mean is None:
            self._compute_norm_stats()

    def _load_file(self, file_idx: int, path: str) -> Optional[Dict[str, Any]]:
        if self.preload and file_idx in self._file_cache:
            return self._file_cache[file_idx]

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None

        for key in ["root_pos", "root_rot", "dof_pos", "hand_positions"]:
            if key not in data:
                print(f"Warning: Missing '{key}' in {path}")
                return None

        data_proc = {
            "root_pos": np.asarray(data["root_pos"], dtype=np.float32),
            "root_rot": np.asarray(data["root_rot"], dtype=np.float32),
            "dof_pos": np.asarray(data["dof_pos"], dtype=np.float32),
            "hand_positions": np.asarray(data["hand_positions"], dtype=np.float32),
            "seq_name": data.get("seq_name", os.path.splitext(os.path.basename(path))[0]),
            "fps": float(data.get("fps", 30.0)),
        }

        if self.preload:
            self._file_cache[file_idx] = data_proc

        return data_proc

    def _compute_norm_stats(self):
        all_states = []
        all_hands = []

        for fi, start in self.windows:
            data = self._file_cache.get(fi)
            if data is None:
                data = self._load_file(fi, self.file_paths[fi])

            T = self.window_size
            end = start + T

            root_pos = data["root_pos"][start:end]
            root_rot = data["root_rot"][start:end]
            dof_pos = data["dof_pos"][start:end]
            hand_pos = data["hand_positions"][start:end]

            # Convert root_rot to 6D
            root_rot_6d = quat_to_rot6d_xyzw(
                torch.from_numpy(root_rot).float()
            ).numpy()

            state = np.concatenate([root_pos, root_rot_6d, dof_pos], axis=-1)
            all_states.append(state)
            all_hands.append(hand_pos)

        states_arr = np.concatenate(all_states, axis=0)
        hands_arr = np.concatenate(all_hands, axis=0)

        state_mean, state_std = compute_mean_std(states_arr)
        hand_mean, hand_std = compute_mean_std(hands_arr)

        self.state_mean = torch.from_numpy(state_mean).float()
        self.state_std = torch.from_numpy(state_std).float()
        self.hand_mean = torch.from_numpy(hand_mean).float()
        self.hand_std = torch.from_numpy(hand_std).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fi, start = self.windows[idx]
        data = self._file_cache.get(fi)
        if data is None:
            data = self._load_file(fi, self.file_paths[fi])

        T = self.window_size
        end = start + T

        root_pos = data["root_pos"][start:end]
        root_rot = data["root_rot"][start:end]
        dof_pos = data["dof_pos"][start:end]
        hand_pos = data["hand_positions"][start:end]

        # Convert root rotation to 6D
        root_rot_6d = quat_to_rot6d_xyzw(
            torch.from_numpy(root_rot).float()
        ).numpy()

        # Build state vector: [root_pos(3), root_rot_6d(6), dof_pos(D)]
        state = np.concatenate([root_pos, root_rot_6d, dof_pos], axis=-1)
        state_t = torch.from_numpy(state).float()
        hand_t = torch.from_numpy(hand_pos).float()

        # Normalize
        if self.state_mean is not None:
            state_t = (state_t - self.state_mean.view(1, -1)) / self.state_std.view(1, -1)
        if self.normalize_hands and self.hand_mean is not None:
            hand_t = (hand_t - self.hand_mean.view(1, -1)) / self.hand_std.view(1, -1)

        return {
            "state": state_t,      # (T, 38)
            "cond": hand_t,        # (T, 6)
            "seq_name": data["seq_name"],
            "fps": data["fps"],
            "file_idx": fi,
            "start": start,
        }

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
