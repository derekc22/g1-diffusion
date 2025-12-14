import mujoco
import mujoco.viewer
from utils import *

m, d = load_model("mujoco/assets/g1/scene.xml")
viewer = mujoco.viewer.launch_passive(m, d)
camera_presets = {
                "lookat": [0.0, 0.0, 0.55], 
                "distance": 2, 
                "azimuth": 90, 
                "elevation": -10
            }  
set_cam(viewer, track=False, presets=camera_presets, show_world_csys=False, show_body_csys=False)

while True:        

    mujoco.mj_step(m, d)
    viewer.sync()
    
viewer.close()