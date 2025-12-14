import mujoco
import numpy as np

def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d

def reset(m: mujoco.MjModel, 
          d: mujoco.MjData, 
          keyframe: str) -> None:
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

def set_cam(viewer,
            track: bool = False,
            presets: dict = None,
            show_world_csys: bool = False,
            show_body_csys: bool = False) -> None:
    
    if track:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 0
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    if presets is None:
        presets = {
                    "lookat": [0.0, 0.0, 0.0], 
                    "distance": 2.0, 
                    "azimuth": 90,
                    "elevation": -45
                }

    viewer.cam.lookat[:] = presets["lookat"]        # Point the camera is looking at
    viewer.cam.distance = presets["distance"]       # Distance from lookat point
    viewer.cam.azimuth = presets["azimuth"]         # Horizontal rotation angle [deg]
    viewer.cam.elevation = presets["elevation"]     # Vertical rotation angle [deg]
    
    if show_body_csys:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    if show_world_csys:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD