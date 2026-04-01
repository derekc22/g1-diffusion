"""
End-to-End HF Pipeline: Object Motion → Full-Body Motion

Complete DDPM pipeline that:
1. Stage 1: Generates hand positions from object motion features (15D)
2. Apply contact constraints to hand positions
3. Stage 2: Generates full-body motion from hand positions

Usage:
    python scripts/sample_stage2_hf.py
    python scripts/sample_stage2_hf.py --config_path experiments/stage2_hf/my_config.yaml
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm
import types

from models.stage1_hf_diffusion import Stage1HFHandDiffusion, Stage1HFHandDiffusionMLP
from models.stage2_diffusion import Stage2TransformerModel, Stage2MLPModel
from utils.diffusion import DiffusionConfig, DiffusionSchedule
from utils.contact_constraints import apply_contact_constraints, ContactConstraintProcessor
from utils.rotation import rot6d_to_quat_xyzw, mat_to_quat_xyzw
from utils.general import load_config

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


def _build_object_features(data: Dict[str, Any]) -> Optional[np.ndarray]:
    """Build 15D object feature vector from PKL fields."""
    if "object_features" in data:
        return np.asarray(data["object_features"], dtype=np.float32)

    obj_pos = data.get("object_pos")
    if obj_pos is None:
        return None

    obj_pos = np.asarray(obj_pos, dtype=np.float32)
    T = obj_pos.shape[0]

    obj_rot = data.get("object_rot")
    if obj_rot is not None:
        obj_rot = np.asarray(obj_rot, dtype=np.float32)
        if obj_rot.shape[-1] == 4:
            from utils.rotation import quat_to_rot6d_xyzw
            obj_rot_6d = quat_to_rot6d_xyzw(torch.from_numpy(obj_rot)).numpy()
        elif obj_rot.shape[-1] == 6:
            obj_rot_6d = obj_rot
        elif obj_rot.shape[-2:] == (3, 3):
            obj_rot_6d = obj_rot.reshape(T, 9)[:, :6]
        else:
            obj_rot_6d = np.zeros((T, 6), dtype=np.float32)
    else:
        obj_rot_6d = np.zeros((T, 6), dtype=np.float32)

    obj_lin_vel = data.get("object_lin_vel")
    obj_lin_vel = np.asarray(obj_lin_vel, dtype=np.float32) if obj_lin_vel is not None else np.zeros((T, 3), dtype=np.float32)

    obj_ang_vel = data.get("object_ang_vel")
    obj_ang_vel = np.asarray(obj_ang_vel, dtype=np.float32) if obj_ang_vel is not None else np.zeros((T, 3), dtype=np.float32)

    return np.concatenate([obj_pos, obj_rot_6d, obj_lin_vel, obj_ang_vel], axis=-1)


class HFPipeline:
    """
    Complete HF pipeline for object-guided motion synthesis.

    Pipeline:
        Object motion features (15D)
            → Stage 1 (DDPM) → Raw hand positions
            → Contact constraints → Rectified hand positions
            → Stage 2 (DDPM) → Full-body robot motion
    """

    def __init__(
        self,
        stage1_ckpt_path: str,
        stage2_ckpt_path: str,
        device: str = "cuda:0",
        contact_threshold: float = 0.03,
    ):
        self.device = torch.device(device)
        self.contact_threshold = contact_threshold

        # Load Stage 1 (HF model)
        self.stage1_model, self.stage1_schedule, self.stage1_norm, stage1_max_len = self._load_stage1(stage1_ckpt_path)

        # Load Stage 2 (standard model)
        self.stage2_model, self.stage2_schedule, self.stage2_norm, stage2_max_len = self._load_stage2(stage2_ckpt_path)

        # Use minimum max_len for safety
        self.max_len = min(stage1_max_len, stage2_max_len)

        # Contact processor
        self.contact_processor = ContactConstraintProcessor(contact_threshold=contact_threshold)

    def _load_stage1(self, ckpt_path: str) -> Tuple[torch.nn.Module, DiffusionSchedule, dict, int]:
        """Load Stage 1 HF model."""
        print(f"Loading Stage 1 (HF): {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})

        window_size = dataset_cfg.get("window_size", 120)
        max_len = model_cfg.get("max_len", window_size + 100)

        object_feature_input_dim = model_cfg.get("object_feature_input_dim", 15)
        encoder_hidden = model_cfg.get("encoder_hidden", 512)
        encoder_layers = model_cfg.get("encoder_layers", 3)
        object_feature_dim = model_cfg.get("object_feature_dim", 256)
        hand_dim = model_cfg.get("hand_dim", 6)

        if arch == "transformer":
            model = Stage1HFHandDiffusion(
                object_feature_input_dim=object_feature_input_dim,
                encoder_hidden=encoder_hidden,
                object_feature_dim=object_feature_dim,
                encoder_layers=encoder_layers,
                hand_dim=hand_dim,
                d_model=model_cfg.get("d_model", 256),
                nhead=model_cfg.get("nhead", 4),
                num_transformer_layers=model_cfg.get("num_layers", 4),
                dim_feedforward=model_cfg.get("dim_feedforward", 512),
                dropout=model_cfg.get("dropout", 0.1),
                max_len=max_len,
            )
        else:
            model = Stage1HFHandDiffusionMLP(
                object_feature_input_dim=object_feature_input_dim,
                encoder_hidden=encoder_hidden,
                object_feature_dim=object_feature_dim,
                encoder_layers=encoder_layers,
                hand_dim=hand_dim,
            )

        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()

        timesteps = config.get("train", {}).get("timesteps", 1000)
        schedule = DiffusionSchedule(
            DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
        ).to(self.device)

        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "hand_mean": norm_stats.get("hand_mean"),
            "hand_std": norm_stats.get("hand_std"),
        }
        if norm["hand_mean"] is not None:
            norm["hand_mean"] = norm["hand_mean"].to(self.device)
            norm["hand_std"] = norm["hand_std"].to(self.device)

        return model, schedule, norm, max_len

    def _load_stage2(self, ckpt_path: str) -> Tuple[torch.nn.Module, DiffusionSchedule, dict, int]:
        """Load Stage 2 model."""
        print(f"Loading Stage 2: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        config = ckpt["config"]
        arch = config.get("train", {}).get("architecture", "transformer")
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})

        window_size = dataset_cfg.get("window_size", 120)
        max_len = model_cfg.get("max_len", window_size + 100)

        # Infer state_dim from checkpoint
        state_dict = ckpt["model"]
        if "out_proj.weight" in state_dict:
            state_dim = state_dict["out_proj.weight"].shape[0]
        else:
            state_dim = model_cfg.get("state_dim", 38)

        cond_dim = 6  # Hand positions

        if arch == "transformer":
            model = Stage2TransformerModel(
                state_dim=state_dim,
                cond_dim=cond_dim,
                d_model=model_cfg.get("d_model", 256),
                nhead=model_cfg.get("nhead", 4),
                num_layers=model_cfg.get("num_layers", 4),
                dim_feedforward=model_cfg.get("dim_feedforward", 512),
                dropout=model_cfg.get("dropout", 0.1),
                max_len=max_len,
            )
        else:
            model = Stage2MLPModel(
                state_dim=state_dim,
                cond_dim=cond_dim,
            )

        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()

        timesteps = config.get("train", {}).get("timesteps", 1000)
        schedule = DiffusionSchedule(
            DiffusionConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
        ).to(self.device)

        norm_stats = ckpt.get("norm_stats", {})
        norm = {
            "state_mean": norm_stats.get("state_mean"),
            "state_std": norm_stats.get("state_std"),
        }
        if norm["state_mean"] is not None:
            norm["state_mean"] = norm["state_mean"].to(self.device)
            norm["state_std"] = norm["state_std"].to(self.device)

        return model, schedule, norm, max_len

    @torch.no_grad()
    def _sample_ddpm(
        self,
        model: torch.nn.Module,
        schedule: DiffusionSchedule,
        cond: torch.Tensor,
        output_dim: int,
        cond_fn=None,
    ) -> torch.Tensor:
        """Generic DDPM sampling."""
        B, T, _ = cond.shape
        x = torch.randn(B, T, output_dim, device=self.device)

        for n in reversed(range(schedule.timesteps)):
            t = torch.full((B,), n, device=self.device, dtype=torch.long)

            if cond_fn is not None:
                x0_pred = cond_fn(model, x, t, cond)
            else:
                x0_pred = model(x, t, cond)

            if n > 0:
                alpha_bar_t = schedule.alpha_bar[n]
                alpha_bar_t_prev = schedule.alpha_bar[n - 1]
                alpha_t = schedule.alpha[n]

                mean = (
                    torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
                    + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x
                )

                sigma = torch.sqrt(schedule.beta[n])
                x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x

    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if mean is None:
            return x
        if x.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        else:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return x * std + mean

    def generate(
        self,
        object_features: np.ndarray,
        object_verts: Optional[np.ndarray] = None,
        object_rotation: Optional[np.ndarray] = None,
        apply_constraints: bool = True,
        partial_motion_length: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate full-body motion from object motion features.

        Args:
            object_features: (T, 15) object motion features
            object_verts: (T, K, 3) object vertices (for contact constraints)
            object_rotation: (T, 3, 3) object rotations (for contact constraints)
            apply_constraints: Whether to apply contact constraints
            partial_motion_length: If set, generate this many frames instead of full length

        Returns:
            Dict with hands_raw, hands_rectified, state, root_pos, root_rot, dof_pos, etc.
        """
        T_original = object_features.shape[0]
        truncated = False
        partial = False
        target_len = None

        # Handle partial motion
        if partial_motion_length is not None and partial_motion_length > 0:
            if partial_motion_length < T_original:
                object_features = object_features[:partial_motion_length]
                if object_verts is not None:
                    object_verts = object_verts[:partial_motion_length]
                if object_rotation is not None:
                    object_rotation = object_rotation[:partial_motion_length]
                partial = True
                target_len = partial_motion_length

        T_working = object_features.shape[0]

        # Truncate if exceeds max_len
        if T_working > self.max_len:
            print(f"  Warning: Truncating sequence from {T_working} to {self.max_len} frames")
            object_features = object_features[: self.max_len]
            if object_verts is not None:
                object_verts = object_verts[: self.max_len]
            if object_rotation is not None:
                object_rotation = object_rotation[: self.max_len]
            truncated = True

        T_seq = object_features.shape[0]

        # Prepare object features tensor
        obj_feat = torch.from_numpy(object_features).float().unsqueeze(0).to(self.device)  # (1, T, 15)

        # =====================================================================
        # Stage 1: Object features → Hand positions
        # =====================================================================
        def stage1_forward(model, x, t, _):
            return model(x, t, obj_feat)

        hands_norm = self._sample_ddpm(
            self.stage1_model,
            self.stage1_schedule,
            obj_feat,
            output_dim=6,
            cond_fn=stage1_forward,
        )

        hands_raw = self._denormalize(
            hands_norm,
            self.stage1_norm["hand_mean"],
            self.stage1_norm["hand_std"],
        )
        hands_raw_np = hands_raw.squeeze(0).cpu().numpy()

        # =====================================================================
        # Apply Contact Constraints
        # =====================================================================
        if apply_constraints and object_verts is not None and object_rotation is not None:
            hands_rect_np, contact_meta = self.contact_processor.process(
                hands_raw_np, object_verts, object_rotation
            )
        else:
            hands_rect_np = hands_raw_np
            contact_meta = None

        # =====================================================================
        # Stage 2: Hand positions → Full-body motion
        # =====================================================================
        hands_rect = torch.from_numpy(hands_rect_np).float().unsqueeze(0).to(self.device)

        if hasattr(self.stage2_model, "state_dim"):
            state_dim = self.stage2_model.state_dim
        else:
            state_dim = self.stage2_model.out_proj.out_features

        state_norm = self._sample_ddpm(
            self.stage2_model,
            self.stage2_schedule,
            hands_rect,
            output_dim=state_dim,
        )

        state = self._denormalize(
            state_norm,
            self.stage2_norm["state_mean"],
            self.stage2_norm["state_std"],
        )
        state_np = state.squeeze(0).cpu().numpy()

        # Parse state
        root_pos = state_np[:, :3]
        root_rot_6d = state_np[:, 3:9]
        dof_pos = state_np[:, 9:]

        root_rot_6d_t = torch.from_numpy(root_rot_6d).float()
        root_rot_quat = rot6d_to_quat_xyzw(root_rot_6d_t).numpy()

        return {
            "hands_raw": hands_raw_np,
            "hands_rectified": hands_rect_np,
            "contact_metadata": contact_meta,
            "state": state_np,
            "root_pos": root_pos,
            "root_rot": root_rot_quat,
            "dof_pos": dof_pos,
            "truncated": truncated,
            "original_len": T_original,
            "partial": partial,
            "target_len": target_len,
        }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 End-to-End HF Pipeline")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./experiments/stage2_hf/sample_stage2_hf.yaml",
        help="Path to YAML experiment config file",
    )
    args = parser.parse_args()

    yml = load_config(args.config_path)
    sample_yml = yml["sample"]

    root_dir = yml["root_dir"]
    stage1_ckpt_path = sample_yml["stage1_ckpt_path"]
    stage2_ckpt_path = sample_yml["stage2_ckpt_path"]
    device_str = sample_yml.get("device", "cuda")
    apply_constraints = sample_yml.get("apply_constraints", True)
    contact_threshold = sample_yml.get("contact_threshold", 0.03)
    num_samples = sample_yml.get("num_samples", None)
    seed = sample_yml.get("seed", 42)
    timesteps = sample_yml.get("timesteps", 1000)
    partial_motion_length = sample_yml.get("partial_motion_length", None)

    if not stage1_ckpt_path or not stage2_ckpt_path:
        raise ValueError("stage1_ckpt_path and stage2_ckpt_path must be set in config")

    # Derive output directory
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    exp_name = yml.get("exp_name", "")
    suffix = f"_{exp_name}" if exp_name else ""

    ckpt_parts = stage2_ckpt_path.split("/")
    if "logs" in ckpt_parts and "checkpoints" in ckpt_parts:
        logs_idx = ckpt_parts.index("logs")
        log_id = ckpt_parts[logs_idx + 1]
        dataset_yml = yml.get("dataset", {})
        window_size = dataset_yml.get("window_size", 120)
        stride = dataset_yml.get("stride", 10)
        sample_folder = f"ts{timesteps}_w{window_size}_s{stride}_{timestamp}{suffix}"
        output_dir = os.path.join("logs", log_id, "samples", sample_folder)
    else:
        output_dir = os.path.join("out", "stage2_hf", timestamp)

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    # Save config for reproducibility
    import yaml

    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(yml, f, default_flow_style=False)

    # Create pipeline
    pipeline = HFPipeline(
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        device=device_str,
        contact_threshold=contact_threshold,
    )

    # Initialize MuJoCo FK for computing local_body_pos
    mj_fk = None
    xml_path = os.path.join(PROJECT_ROOT, "..", "g1-gmr", "assets", "unitree_g1", "g1_mocap_29dof.xml")
    if not os.path.isfile(xml_path):
        xml_path = os.path.join(PROJECT_ROOT, "assets", "unitree_g1", "g1_mocap_29dof.xml")
    try:
        import mujoco as mj
        if os.path.isfile(xml_path):
            _mj_model = mj.MjModel.from_xml_path(xml_path)
            _mj_data = mj.MjData(_mj_model)
            # Build link_body_list from model
            _link_body_list = []
            for i in range(_mj_model.nbody):
                name = mj.mj_id2name(_mj_model, mj.mjtObj.mjOBJ_BODY, i)
                if name:
                    _link_body_list.append(name)
            # Remove 'world' body (id 0)
            if _link_body_list and _link_body_list[0] == "world":
                _link_body_list = _link_body_list[1:]
            mj_fk = (_mj_model, _mj_data, mj, _link_body_list)
            print(f"MuJoCo FK initialized: {len(_link_body_list)} bodies")
        else:
            print(f"Warning: MuJoCo XML not found at {xml_path}, local_body_pos will be omitted")
    except ImportError:
        print("Warning: mujoco not installed, local_body_pos will be omitted")

    # Find input files
    files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
    if num_samples:
        files = files[:num_samples]

    print(f"\nProcessing {len(files)} files")
    print(f"Output directory: {output_dir}")

    # Performance tracking
    total_time = 0.0
    sample_durations = []
    all_frame_counts = []

    for fpath in tqdm(files, desc="Generating"):
        fname = os.path.basename(fpath)

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        # Build object features
        obj_feat = _build_object_features(data)
        if obj_feat is None:
            T = data.get("hand_positions", data.get("dof_pos")).shape[0]
            obj_feat = np.zeros((T, 15), dtype=np.float32)

        # Generate with timing
        start_time = time.perf_counter()
        result = pipeline.generate(
            object_features=obj_feat,
            object_verts=data.get("object_verts"),
            object_rotation=data.get("object_rotation"),
            apply_constraints=apply_constraints,
            partial_motion_length=partial_motion_length,
        )
        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        # Save
        output_data = {
            "seq_name": data.get("seq_name", fname),
            "fps": data.get("fps", 30.0),
            **result,
        }

        # Number of generated frames
        gen_T = result["root_pos"].shape[0]

        # Visualization compatibility keys
        if "hands_rectified" in result:
            output_data["hand_positions"] = result["hands_rectified"]

        # Object data — truncate to generated length
        if "object_pos" in data:
            output_data["object_pos"] = data["object_pos"][:gen_T]
        if "object_rot" in data:
            # HF data stores object_rot as quaternion (xyzw) directly
            output_data["object_rot"] = data["object_rot"][:gen_T]
        elif "object_rotation" in data:
            # Old data stores rotation matrices
            obj_rot_mat = data["object_rotation"][:gen_T]
            obj_rot_mat_t = torch.from_numpy(obj_rot_mat).float()
            obj_rot_quat = mat_to_quat_xyzw(obj_rot_mat_t).numpy()
            output_data["object_rot"] = obj_rot_quat

        # Compute local_body_pos via MuJoCo FK
        if mj_fk is not None:
            _mj_model, _mj_data, _mj, _link_body_list = mj_fk
            nbody = _mj_model.nbody - 1  # exclude world body
            local_body_pos = np.zeros((gen_T, nbody, 3), dtype=np.float32)
            root_pos_gen = result["root_pos"]
            root_rot_gen = result["root_rot"]  # xyzw quaternion
            dof_pos_gen = result["dof_pos"]
            for t in range(gen_T):
                quat_wxyz = np.array([
                    root_rot_gen[t, 3], root_rot_gen[t, 0],
                    root_rot_gen[t, 1], root_rot_gen[t, 2],
                ])
                qpos = np.zeros(_mj_model.nq, dtype=np.float64)
                qpos[0:3] = root_pos_gen[t]
                qpos[3:7] = quat_wxyz
                qpos[7:7 + dof_pos_gen.shape[1]] = dof_pos_gen[t]
                _mj_data.qpos[:] = qpos
                _mj.mj_forward(_mj_model, _mj_data)
                local_body_pos[t] = _mj_data.xpos[1:]  # skip world body
            output_data["local_body_pos"] = local_body_pos
            output_data["link_body_list"] = _link_body_list
        elif "local_body_pos" in data:
            output_data["local_body_pos"] = data["local_body_pos"][:gen_T]
        if "link_body_list" in data:
            output_data["link_body_list"] = data["link_body_list"]
        output_data["source_start"] = 0

        # GT for comparison
        if "root_pos" in data:
            output_data["gt_root_pos"] = data["root_pos"]
            output_data["gt_root_rot"] = data["root_rot"]
            output_data["gt_dof_pos"] = data["dof_pos"]
        if "hand_positions" in data:
            output_data["gt_hands"] = data["hand_positions"]

        # Track timing
        fps = output_data["fps"]
        num_frames = result.get("root_pos", result.get("dof_pos", np.zeros((1,)))).shape[0]
        if num_frames > 0:
            all_frame_counts.append(num_frames)
            sample_durations.append(num_frames / fps)

        fname_base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{fname_base}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)

    # Print performance summary
    num_files = len(files)
    avg_time = (total_time / num_files * 1000) if num_files > 0 else 0
    throughput = num_files / total_time if total_time > 0 else 0
    avg_duration = sum(sample_durations) / len(sample_durations) if sample_durations else 0

    print(f"\n{'='*50}")
    print("Performance Summary")
    print(f"{'='*50}")
    print(f"Total files: {num_files}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {avg_time:.1f}ms per sample")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Average sample duration: {avg_duration:.2f}s")
    if all_frame_counts:
        print(f"Frames per motion: {int(np.mean(all_frame_counts))} avg, {min(all_frame_counts)} min, {max(all_frame_counts)} max")
        total_frames = sum(all_frame_counts)
        gen_fps = total_frames / total_time if total_time > 0 else 0
        print(f"Generated fps: {gen_fps:.1f}")
    print(f"{'='*50}")

    # Save summary
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Performance Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total files: {num_files}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Average time: {avg_time:.1f}ms per sample\n")
        f.write(f"Throughput: {throughput:.2f} samples/sec\n")
        if all_frame_counts:
            f.write(f"Frames per motion: {int(np.mean(all_frame_counts))} avg\n")
            total_frames = sum(all_frame_counts)
            gen_fps = total_frames / total_time if total_time > 0 else 0
            f.write(f"Generated fps: {gen_fps:.1f}\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
