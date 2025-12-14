import os
import glob
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import sys
import types
import torch
from torch.utils.data import Dataset

from utils.rotation import quat_wxyz_to_xyzw, quat_to_rot6d_xyzw
from utils.normalization import compute_mean_std

# ---------------------------------------------------------------------------
# Compatibility shim for pickles created by NumPy >= 2.0 in envs with older NumPy
# GMR wrote the .pkl files with numpy._core.*; omomo_env has numpy.core.* only.
# ---------------------------------------------------------------------------
if "numpy._core" not in sys.modules:
    core_pkg = types.ModuleType("numpy._core")
    core_pkg.__path__ = []  # mark as a package so 'numpy._core.multiarray' imports work
    sys.modules["numpy._core"] = core_pkg

# Map submodules used in pickled arrays to the old locations
if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

if "numpy._core.numerictypes" not in sys.modules:
    sys.modules["numpy._core.numerictypes"] = np.core.numerictypes

if "numpy._core.umath" not in sys.modules:
    sys.modules["numpy._core.umath"] = np.core.umath
# ---------------------------------------------------------------------------




class ObjectMotionDataset(Dataset):
    """
    Dataset over GMR-process object motion files.

    Each underlying file is a full sequence with keys:
      - object_centroid: (T, 3)
      - bps_encoding: (T, 1024, 3)
    """

    def __init__(
        self,
        root_dir: str,
        window_size: int,
        stride: int,
        min_seq_len: int,
        # normalize: bool,
        train: bool,
        train_split: float,
        preload: bool,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.min_seq_len = min_seq_len
        # self.normalize_flag = normalize
        self.preload = preload
        self.mean = mean
        self.std = std

        file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if not file_paths:
            raise RuntimeError(f"No motion .pkl files found in {root_dir}")

        # simple train/val split at file level
        num_files = len(file_paths)
        self.num_files = num_files
        split_idx = int(num_files * train_split)
        if train:
            self.file_paths = file_paths[:split_idx]
        else:
            self.file_paths = file_paths[split_idx:]

        # cache seq metadata: (file_idx, start_t)
        self.windows: List[Tuple[int, int]] = []
        # self.seq_lengths: List[int] = []
        self._file_cache: Dict[int, Dict[str, Any]] = {}

        for fi, path in enumerate(self.file_paths):
            data = self._load_file(fi, path)
            T = data["root_pos"].shape[0]
            # self.seq_lengths.append(T)
            if T < self.min_seq_len or T < self.window_size:
                continue
            for start in range(0, T - self.window_size + 1, self.stride):
                self.windows.append((fi, start))

        if not self.windows:
            raise RuntimeError("No windows constructed. Check min_seq_len and window_size.")

        # compute normalization over all windows if required
        if self.mean is not None and self.std is not None: assert not train, "Mean and std should only be passed at inference"
        # self.mean = None
        # self.std = None
        # if self.normalize_flag and train:
        if train:
            all_samples = []
            for idx in range(len(self.windows)):
                state_un, _, _ = self._get_window(idx, normalized=False)
                all_samples.append(state_un.numpy())
            all_samples_arr = np.concatenate(all_samples, axis=0)  # (N, D)
            mean, std = compute_mean_std(all_samples_arr)
            self.mean = torch.from_numpy(mean).float()
            self.std = torch.from_numpy(std).float()

    def _load_file(self, file_idx: int, path: str) -> Dict[str, Any]:
        if self.preload and file_idx in self._file_cache:
            return self._file_cache[file_idx]

        with open(path, "rb") as f:
            data = pickle.load(f)

        # basic checks
        root_pos = np.asarray(data["root_pos"], dtype=np.float32)  # (T, 3)
        root_rot = np.asarray(data["root_rot"], dtype=np.float32)  # (T, 4)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)    # (T, Dq)

        assert root_pos.ndim == 2 and root_pos.shape[1] == 3
        assert root_rot.ndim == 2 and root_rot.shape[1] == 4
        assert dof_pos.ndim == 2

        object_pos = data.get("object_pos")
        object_rot = data.get("object_rot")
        # cond_seq = self._build_cond_sequence(root_pos, object_pos, object_rot_6d)
        cond_seq = self._build_cond_sequence(object_pos, object_rot)

        seq_name = data.get(
            "seq_name",
            os.path.splitext(os.path.basename(path))[0],
        )

        data_proc = {
            "fps": float(data.get("fps", 30.0)),
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": np.asarray(data.get("local_body_pos", None)) if "local_body_pos" in data else None,
            "link_body_list": data.get("link_body_list", None),
            "cond": cond_seq,
            "seq_name": seq_name,
        }

        if self.preload:
            self._file_cache[file_idx] = data_proc

        return data_proc

    def __len__(self) -> int:
        return len(self.windows)

    def _get_window(self, idx: int, normalized: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        file_idx, start = self.windows[idx]
        path = self.file_paths[file_idx]
        data = self._load_file(file_idx, path)

        T = self.window_size
        root_pos = data["root_pos"][start:start + T]  # (T, 3)
        root_rot = data["root_rot"][start:start + T]  # (T, 4)
        dof_pos = data["dof_pos"][start:start + T]    # (T, Dq)
        cond_seq = data["cond"][start:start + T]      # (T, 9)

        # build state vector s_t
        components = []

        # if self.use_root_pos:
        components.append(root_pos)  # (T, 3)

        # BUG
        # if self.use_root_rot_6d:
        #     quat_xyzw = quat_wxyz_to_xyzw(root_rot)
        #     quat_xyzw_t = torch.from_numpy(quat_xyzw).float()  # (T, 4)
        #     rot6d = quat_to_rot6d_xyzw(quat_xyzw_t).numpy()    # (T, 6)
        #     components.append(rot6d)

        # if self.use_root_rot_6d:
        root_quat_xyzw = torch.from_numpy(root_rot).float()  # (T, 4)
        root_rot6d = quat_to_rot6d_xyzw(root_quat_xyzw).numpy()    # (T, 6)
        components.append(root_rot6d)

        # if self.use_dof_pos:
        components.append(dof_pos)  # (T, Dq)

        state = np.concatenate(components, axis=-1)  # (T, D)
        state_t = torch.from_numpy(state).float()    # (T, D)

        meta = {
            "file_path": path,
            "file_idx": file_idx,
            "start": start,
            "fps": data["fps"],
            "seq_name": data["seq_name"],
        }

        # if self.normalize_flag and normalized:
        if normalized:
            assert self.mean is not None and self.std is not None
            mean = self.mean.view(1, -1)
            std = self.std.view(1, -1)
            # print(mean.device)
            # print(std.device)
            # print(state_t.device)
            # exit()
            state_t = (state_t - mean) / std

        cond_t = torch.from_numpy(cond_seq).float()    # (T, 9)

        return state_t, cond_t, meta

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        state_t, cond_t, meta = self._get_window(idx, normalized=True)
        sample = {
            "state": state_t,          # (T, D)
            "cond": cond_t,            # (T, 9)
            "file_idx": meta["file_idx"],
            "start": meta["start"],
            "fps": meta["fps"],
            "seq_name": meta["seq_name"],
        }
        return sample

    # def _build_cond_sequence(self, root_pos: np.ndarray, obj_pos, obj_rot_6d) -> np.ndarray:
    #     T = root_pos.shape[0]
    #     pos_block = np.zeros((T, 3), dtype=np.float32)
    #     rot_block = np.zeros((T, 6), dtype=np.float32)

    #     if obj_pos is not None:
    #         obj_pos_arr = np.asarray(obj_pos, dtype=np.float32)
    #         if obj_pos_arr.ndim > 2:
    #             obj_pos_arr = obj_pos_arr.reshape(obj_pos_arr.shape[0], -1)
    #         if obj_pos_arr.shape[0] == T:
    #             pos_block = obj_pos_arr.reshape(T, 3)

    #     if obj_rot_6d is not None:
    #         obj_rot_arr = np.asarray(obj_rot_6d, dtype=np.float32)
    #         if obj_rot_arr.shape[0] == T:
    #             rot_block = obj_rot_arr.reshape(T, 6)

    #     return np.concatenate([pos_block, rot_block], axis=-1)

    def _build_cond_sequence(self, obj_pos, obj_rot) -> np.ndarray:
        obj_quat_xyzw = torch.from_numpy(obj_rot).float()  # (T, 4)
        obj_rot6d = quat_to_rot6d_xyzw(obj_quat_xyzw).numpy()    # (T, 6)
        return np.concatenate([obj_pos, obj_rot6d], axis=-1)