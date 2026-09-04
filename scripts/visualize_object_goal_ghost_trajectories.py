"""Render a generated object-goal sample with translucent parent trajectories."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.visualize_model_dynamic import (
    DEFAULT_GMR_ROOT,
    DEFAULT_OBJECTS_DIR,
    add_gmr_to_path,
    build_mesh_index,
    find_object_mesh,
    infer_object_mesh_scale,
    infer_object_name,
    load_motion_file,
)


@dataclass
class Motion:
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof_pos: np.ndarray
    object_pos: np.ndarray
    object_rot: np.ndarray


@dataclass
class Ghost:
    name: str
    motion: Motion
    color: tuple[float, float, float, float]
    omit_object_mesh: bool
    parent_path: str
    source_object_name: str
    source_mesh_path: str | None
    source_mesh_scale: float | None


@dataclass
class RenderGeomTemplate:
    """Visual geometry selected by the same MuJoCo scene as the main robot."""

    geom_id: int
    geom_type: int
    size: np.ndarray
    dataid: int
    category: int
    modelrbound: float
    emission: float
    specular: float
    shininess: float
    reflectance: float
    texcoord: int


def _load_sample(path: str) -> dict[str, Any]:
    with open(path, "rb") as file:
        sample = pickle.load(file)
    for key in ("root_pos", "root_rot", "dof_pos", "object_pose", "init_parent"):
        if key not in sample:
            raise KeyError(f"{path} is missing required key {key!r}")
    return sample


def _generated_motion(path: str) -> Motion:
    """Use the known-good visualizer's loader for the generated trajectory too."""
    motion, _data = _parent_motion(path, max_len=None)
    return motion


def _parent_motion(path: str, max_len: int | None) -> tuple[Motion, dict[str, Any]]:
    (
        _data,
        _fps,
        root_pos,
        root_rot,
        dof_pos,
        object_pos,
        object_rot,
        _local_body_pos,
        _link_body_list,
        _hand_positions,
    ) = load_motion_file(path)
    if max_len is not None:
        root_pos = root_pos[:max_len]
        root_rot = root_rot[:max_len]
        dof_pos = dof_pos[:max_len]
        object_pos = object_pos[:max_len]
        object_rot = object_rot[:max_len]
    return Motion(root_pos, root_rot, dof_pos, object_pos, object_rot), _data


def _resample_values(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == target_len:
        return values
    old_time = np.linspace(0.0, 1.0, len(values))
    new_time = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(new_time, old_time, values[:, dim]) for dim in range(values.shape[1])], axis=-1)


def _resample_wxyz(quaternions: np.ndarray, target_len: int) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if len(quaternions) == target_len:
        return quaternions
    xyzw = quaternions[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(xyzw)
    old_time = np.linspace(0.0, 1.0, len(quaternions))
    new_time = np.linspace(0.0, 1.0, target_len)
    result_xyzw = Slerp(old_time, rotations)(new_time).as_quat()
    return result_xyzw[:, [3, 0, 1, 2]]


def _resample_motion(motion: Motion, target_len: int) -> Motion:
    return Motion(
        root_pos=_resample_values(motion.root_pos, target_len),
        root_rot=_resample_wxyz(motion.root_rot, target_len),
        dof_pos=_resample_values(motion.dof_pos, target_len),
        object_pos=_resample_values(motion.object_pos, target_len),
        object_rot=_resample_wxyz(motion.object_rot, target_len),
    )


def _set_pose(data: mj.MjData, motion: Motion, frame: int) -> None:
    data.qpos[:3] = motion.root_pos[frame]
    data.qpos[3:7] = motion.root_rot[frame]
    data.qpos[7:36] = motion.dof_pos[frame]
    data.qpos[36:39] = motion.object_pos[frame]
    data.qpos[39:43] = motion.object_rot[frame]


def _visible_geom_templates(scene: Any, model: mj.MjModel) -> list[RenderGeomTemplate]:
    """Capture exactly the model geoms selected by Renderer.update_scene."""
    templates = []
    for scene_index in range(scene.ngeom):
        source = scene.geoms[scene_index]
        if source.objtype != int(mj.mjtObj.mjOBJ_GEOM) or source.objid < 0:
            continue
        geom_id = int(source.objid)
        if int(model.geom_bodyid[geom_id]) == 0:
            continue
        templates.append(
            RenderGeomTemplate(
                geom_id=geom_id,
                geom_type=int(source.type),
                size=np.asarray(source.size, dtype=np.float64).copy(),
                dataid=int(source.dataid),
                category=int(source.category),
                modelrbound=float(source.modelrbound),
                emission=float(source.emission),
                specular=float(source.specular),
                shininess=float(source.shininess),
                reflectance=float(source.reflectance),
                texcoord=int(source.texcoord),
            )
        )
    return templates


def _append_ghost_geoms(
    scene: Any,
    model: mj.MjModel,
    data: mj.MjData,
    templates: list[RenderGeomTemplate],
    rgba: tuple[float, float, float, float],
    omit_object_mesh: bool,
) -> None:
    object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "object")
    for template in templates:
        geom_id = template.geom_id
        if omit_object_mesh and int(model.geom_bodyid[geom_id]) == object_body_id:
            continue
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo scene has insufficient custom-geometry capacity")
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            type=template.geom_type,
            size=template.size,
            pos=data.geom_xpos[geom_id],
            mat=data.geom_xmat[geom_id].reshape(-1),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        # mjv_initGeom normalizes unused size components for some geom types.
        # Restore the exact values selected by the known-good rendered scene.
        geom.size[:] = template.size
        geom.dataid = template.dataid
        geom.objtype = int(mj.mjtObj.mjOBJ_GEOM)
        geom.objid = geom_id
        geom.category = template.category
        geom.modelrbound = template.modelrbound
        geom.emission = template.emission
        geom.specular = template.specular
        geom.shininess = template.shininess
        geom.reflectance = template.reflectance
        geom.texcoord = template.texcoord
        geom.transparent = 1
        scene.ngeom += 1


def _append_sphere(scene: Any, position: np.ndarray, color: tuple[float, float, float, float], radius: float = 0.035) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo scene has insufficient custom-geometry capacity")
    mj.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1


def _rot6d_matrix(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64).reshape(6)
    first = value[:3] / max(float(np.linalg.norm(value[:3])), 1e-8)
    second_raw = value[3:6] - float(np.dot(first, value[3:6])) * first
    second = second_raw / max(float(np.linalg.norm(second_raw)), 1e-8)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1)


def _append_pose_marker(
    scene: Any,
    pose: np.ndarray,
    color: tuple[float, float, float, float],
    axis_length: float = 0.12,
) -> None:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size < 9:
        raise ValueError(f"Pose marker requires xyz + rotation-6D, got {pose.shape}")
    position = pose[:3]
    rotation = _rot6d_matrix(pose[3:9])
    _append_sphere(scene, position, color)
    axis_colors = ((1.0, 0.1, 0.1, 1.0), (0.1, 1.0, 0.1, 1.0), (0.1, 0.3, 1.0, 1.0))
    for axis, axis_color in enumerate(axis_colors):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo scene has insufficient custom-geometry capacity")
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.005, 0.005, 0.005]),
            pos=position,
            mat=np.eye(3).reshape(-1),
            rgba=np.asarray(axis_color, dtype=np.float32),
        )
        mj.mjv_connector(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=position,
            to=position + axis_length * rotation[:, axis],
        )
        scene.ngeom += 1


def _legend(frame: np.ndarray, ghosts: list[Ghost]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    entries = [("generated", (230, 230, 230, 255))]
    entries.extend((ghost.name, tuple(round(channel * 255) for channel in ghost.color)) for ghost in ghosts)
    entries.extend(
        [
            ("conditioned initial robot root", (255, 215, 0, 255)),
            ("conditioned initial object", (0, 200, 255, 255)),
            ("conditioned final object goal", (0, 230, 80, 255)),
            ("generated final object", (240, 50, 50, 255)),
        ]
    )
    box_height = 14 + 17 * len(entries)
    draw.rectangle((8, 8, 300, box_height), fill=(0, 0, 0, 155))
    for index, (name, color) in enumerate(entries):
        y = 14 + index * 17
        draw.rectangle((16, y + 2, 27, y + 11), fill=color)
        draw.text((34, y), name, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _parent_key(parent: dict[str, Any] | None) -> str | None:
    if not parent:
        return None
    return os.path.abspath(str(parent.get("path", ""))) or None


def _build_ghosts(
    sample: dict[str, Any],
    target_len: int,
    alpha: float,
    mesh_index: dict[str, str],
) -> list[Ghost]:
    sample_object = str(sample.get("object_name", "")).lower()
    parents: list[tuple[str, dict[str, Any] | None, tuple[float, float, float, float]]] = [
        ("init parent", sample.get("init_parent"), (0.1, 0.8, 1.0, alpha)),
    ]
    if _parent_key(sample.get("goal_parent")) != _parent_key(sample.get("init_parent")):
        parents.append(("goal parent", sample.get("goal_parent"), (1.0, 0.2, 0.8, alpha)))
    ghosts = []
    for name, parent, color in parents:
        if not parent or not parent.get("path"):
            continue
        path = os.path.abspath(str(parent["path"]))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} trajectory does not exist: {path}")
        recorded_length = parent.get("length")
        max_len = None if recorded_length is None else int(recorded_length)
        parent_motion, parent_data = _parent_motion(path, max_len)
        parent_object = str(
            infer_object_name(parent_data, os.path.basename(path), mesh_index)
            or parent.get("object_name", "")
        ).lower()
        parent_mesh_path = find_object_mesh(parent_object, mesh_index, parent_data)
        parent_mesh_scale = infer_object_mesh_scale(parent_data, parent_mesh_path)
        mismatch = bool(sample_object and parent_object and sample_object != parent_object)
        if mismatch:
            print(
                f"Warning: {name} object {parent_object!r} differs from generated object {sample_object!r}; "
                "omitting its object mesh and retaining its robot ghost and object pose markers."
            )
        ghosts.append(
            Ghost(
                name=name,
                motion=_resample_motion(parent_motion, target_len),
                color=color,
                omit_object_mesh=mismatch,
                parent_path=path,
                source_object_name=parent_object,
                source_mesh_path=parent_mesh_path,
                source_mesh_scale=parent_mesh_scale,
            )
        )
    if not ghosts:
        raise RuntimeError("Sample has no loadable init_parent or goal_parent trajectory")
    return ghosts


def _debug_assets(
    *,
    args: argparse.Namespace,
    env: Any,
    model: mj.MjModel,
    base_robot_xml: str,
    mesh_path: str | None,
    mesh_scale: float,
    ghosts: list[Ghost],
) -> None:
    hinge_joint_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mj.mjtJoint.mjJNT_HINGE)
    ]
    runtime_xml = os.path.abspath(str(env.xml_path))
    centroid_offset = bool(mesh_path is not None and getattr(env, "_object_mesh_tmpdir", None) is not None)
    print("=== Ghost asset/convention report ===")
    print(f"robot_type: {args.robot}")
    print(f"base_robot_xml: {base_robot_xml}")
    print(f"generated_robot_xml: {runtime_xml}")
    print(f"ghost_robot_xml: {runtime_xml} (shared MjModel)")
    print(f"generated_object_mesh: {mesh_path or '<placeholder box>'}")
    print(f"generated_object_scale: {mesh_scale:.9g}")
    print(f"centroid_offset_applied: {centroid_offset}")
    if centroid_offset:
        print("centroid_offset_rule: OBJ vertex centroid is subtracted; object_pos is the centered mesh body origin")
    print("generated_root_rotation: stored xyzw quaternion -> MuJoCo wxyz via visualize_model_dynamic.as_wxyz_quat")
    print("ghost_root_rotation: parent xyzw/matrix -> MuJoCo wxyz via visualize_model_dynamic.load_motion_file")
    print("generated_object_rotation: rotation-6D -> matrix -> MuJoCo wxyz via visualize_model_dynamic")
    print("ghost_object_rotation: parent xyzw/matrix -> MuJoCo wxyz via visualize_model_dynamic.load_motion_file")
    print(f"applied_dof_count: {len(hinge_joint_names)}")
    print(f"applied_dof_order: {hinge_joint_names}")
    for index, ghost in enumerate(ghosts):
        applied_mesh = "<omitted: object identity mismatch>" if ghost.omit_object_mesh else (mesh_path or "<placeholder box>")
        print(f"ghost[{index}].name: {ghost.name}")
        print(f"ghost[{index}].parent_path: {ghost.parent_path}")
        print(f"ghost[{index}].source_object_name: {ghost.source_object_name}")
        print(f"ghost[{index}].source_object_mesh: {ghost.source_mesh_path or '<unresolved>'}")
        print(f"ghost[{index}].source_object_scale: {ghost.source_mesh_scale}")
        print(f"ghost[{index}].rendered_object_mesh: {applied_mesh}")
        print(f"ghost[{index}].rendered_object_scale: {mesh_scale if not ghost.omit_object_mesh else '<not rendered>'}")
        print(f"ghost[{index}].robot_xml: {runtime_xml} (shared MjModel)")
    print("=== End ghost asset/convention report ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render generated and parent object-goal trajectories")
    parser.add_argument("--sample_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gmr_root", default=DEFAULT_GMR_ROOT)
    parser.add_argument("--objects_dir", default=DEFAULT_OBJECTS_DIR)
    parser.add_argument("--robot", default="unitree_g1_with_object")
    parser.add_argument("--ghost_alpha", type=float, default=0.25)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument("--debug_assets", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.ghost_alpha < 1.0:
        raise ValueError("--ghost_alpha must be strictly between 0 and 1")

    sample_path = os.path.abspath(args.sample_path)
    sample = _load_sample(sample_path)
    generated = _generated_motion(sample_path)
    frame_count = min(len(generated.root_pos), len(generated.object_pos))
    mesh_index = build_mesh_index(args.objects_dir)
    object_name = infer_object_name(sample, os.path.basename(sample_path), mesh_index)
    mesh_path = find_object_mesh(object_name, mesh_index, sample)
    mesh_scale = infer_object_mesh_scale(sample, mesh_path)
    if mesh_scale is None:
        mesh_scale = 1.0
    ghosts = _build_ghosts(sample, frame_count, args.ghost_alpha, mesh_index)

    add_gmr_to_path(args.gmr_root)
    from general_motion_retargeting import ROBOT_XML_DICT, RobotMotionViewerWithObject

    fps = float(sample.get("fps", 30.0))
    env = RobotMotionViewerWithObject(
        robot_type=args.robot,
        motion_fps=fps,
        camera_follow=False,
        record_video=False,
        object_mesh_path=mesh_path,
        object_mesh_scale=mesh_scale,
    )
    renderer = mj.Renderer(env.model, height=args.video_height, width=args.video_width)
    ghost_data = [mj.MjData(env.model) for _ in ghosts]
    if args.debug_assets:
        _debug_assets(
            args=args,
            env=env,
            model=env.model,
            base_robot_xml=str(ROBOT_XML_DICT[args.robot]),
            mesh_path=mesh_path,
            mesh_scale=mesh_scale,
            ghosts=ghosts,
        )
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    writer = imageio.get_writer(args.output_path, fps=fps)
    try:
        for frame in range(frame_count):
            env.step(
                generated.root_pos[frame], generated.root_rot[frame], generated.dof_pos[frame],
                generated.object_pos[frame], generated.object_rot[frame], rate_limit=False,
            )
            renderer.update_scene(env.data, camera=env.viewer.cam)
            visible_templates = _visible_geom_templates(renderer.scene, env.model)
            for ghost, data in zip(ghosts, ghost_data):
                _set_pose(data, ghost.motion, frame)
                mj.mj_forward(env.model, data)
                _append_ghost_geoms(
                    renderer.scene,
                    env.model,
                    data,
                    visible_templates,
                    ghost.color,
                    ghost.omit_object_mesh,
                )
                if ghost.omit_object_mesh:
                    _append_sphere(renderer.scene, ghost.motion.object_pos[frame], ghost.color, radius=0.025)
            _append_pose_marker(renderer.scene, np.asarray(sample["initial_robot_frame"]), (1.0, 0.84, 0.0, 1.0))
            _append_pose_marker(renderer.scene, np.asarray(sample["initial_object_frame"]), (0.0, 0.78, 1.0, 1.0))
            _append_pose_marker(renderer.scene, np.asarray(sample["final_object_frame"]), (0.0, 0.9, 0.3, 1.0))
            _append_pose_marker(renderer.scene, np.asarray(sample["object_pose"])[-1], (0.94, 0.2, 0.2, 1.0))
            writer.append_data(_legend(renderer.render(), ghosts))
    finally:
        writer.close()
        renderer.close()
        env.close()
    print(f"Ghost video saved to {args.output_path}")


if __name__ == "__main__":
    main()
