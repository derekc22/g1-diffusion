# G1 Diffusion

This repo trains a two-stage diffusion model directly in Unitree G1 joint space.

Data source:
- GMR-retargeted trajectories in `../GMR-master/export_smplx_retargeted/`.
- Each file is a `.pkl` with keys: `fps`, `root_pos`, `root_rot`, `dof_pos`, `local_body_pos`, `link_body_list`.

This repo does **not** call OMOMO or GMR. It only consumes their outputs.
