# Handoff: preserve i020 contact while fixing root/body jitter and foot-floor penetration

This file is for the next agent working in `/home/learning/Documents/g1-diffusion`.
Read it before changing code. The user has built up a lot of context in this chat,
and losing that context will almost certainly produce a "fix" that makes the hands
look worse.

## Current user-facing problem

The current best-looking samples are from:

```text
/home/learning/Documents/g1-diffusion/logs/stage2_hf_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026May07_22-33-42/samples/ts1000_2026May14_17-35-45_agentic_i020_fallback060_refine_smooth_wide_full_length
```

The user says the hands/contact in this i020 full-length run are "sorta good" if
being generous. The remaining obvious failure is not mainly hand contact anymore.
It is whole-body/root quality:

- Root jerks back and forth.
- Body motion is jittery/jerky, not only root rotation.
- Feet now phase through the ground.
- Another attempted fix smoothed/improved root but destroyed the hands/contact.

The next implementation must therefore be **contact-preserving motion
stabilization**, not another contact rewrite and not a blanket low-pass filter.

## Non-negotiable constraints from the user

- Do not use any data except the local Hugging Face data or the derived local
  HF-BPS data. No external training data.
- Stay inside the diffusion/motion-generation/object-interaction lane. Do not
  introduce reinforcement learning, MPC, online control, or a controller stack.
- If training any one model, stop within 30 minutes wall-clock. However, for this
  next task, prefer **post-sampling optimization/refinement** first; it directly
  targets the failure and is lower risk than retraining.
- Training loss is not the success metric. The user cares about visual realism,
  root/body smoothness, hand-object contact, and foot-ground behavior.
- Do not make broad unsolicited edits. The user is sensitive to code being touched
  without intent. Keep changes scoped and explain them.

## Big picture: what exists now

The repo implements an OMOMO-like two-stage pipeline:

1. Stage 1 predicts hand/contact-link trajectories from object motion/geometry.
2. Stage 2 predicts full robot motion conditioned on those hand trajectories.
3. Inference-time contact processing rectifies Stage 1 hand targets against object
   geometry.
4. Later additions then use robot FK to reduce drift between generated robot hands
   and Stage 1 contact anchors.

Important code paths:

- `scripts/sample_stage2_optimized.py`
  - Main end-to-end sampler.
  - `OptimizedOmomoPipeline.generate(...)` runs Stage 1, contact constraints,
    Stage 2, body smoothing, root contact correction, upper-body contact
    refinement, then saves samples.
  - Current save path includes `hands_raw`, `hands_rectified`, `robot_hands`,
    `root_pos`, `root_rot`, `dof_pos`, metadata, and GT fields copied from source
    only for evaluation/plotting.
- `utils/contact_constraints.py`
  - Implements OMOMO-style hand contact constraints and Stage 1 fallback
    object-surface anchoring.
- `utils/robot_kinematics.py`
  - Pure Python G1 FK parser and differentiable torch FK for hands.
  - `robot_hand_positions(...)`
  - `robot_hand_positions_from_state_torch(...)`
  - `apply_robot_contact_root_correction(...)`
  - `apply_robot_contact_state_refinement(...)`
- `scripts/evaluate_hf_bps_samples.py`
  - Existing sample evaluator.
  - Computes target hand metrics and actual robot FK hand/contact metrics.
  - It does **not yet** compute root jerk, joint jerk, foot penetration, or foot
    sliding metrics. Add those.

## Current i020 flow

For the known-good May14 i020 sample, the relevant flow is:

```text
object BPS/centroid
  -> Stage 1 predicted hands
  -> OMOMO contact constraints and fallback anchoring
  -> hands_rectified
  -> Stage 2 robot motion conditioned on hands_rectified
  -> body_smooth_strength/window smoothing
  -> root-only FK contact correction
  -> upper-body FK contact refinement
  -> saved root_pos/root_rot/dof_pos/robot_hands
```

i020 config of interest:

```text
experiments/agentic_loop/iter020/sample_stage2_hf_bps_fallback060_refine_smooth_wide.yaml
```

Key settings there:

```yaml
body_smooth_strength: 0.4
body_smooth_window: 7
body_smooth_iterations: 1
robot_contact_correction: true
robot_contact_max_translation: 0.50
robot_contact_refinement: true
robot_contact_refinement_steps: 18
robot_contact_refinement_lr: 0.025
robot_contact_refinement_pose_reg_weight: 0.004
robot_contact_refinement_velocity_reg_weight: 0.12
robot_contact_refinement_acceleration_reg_weight: 0.50
robot_contact_refinement_max_joint_delta: 0.25
robot_contact_refinement_mode: "upper"
```

Important: i020 preserves hands by refining only upper-body/arms after root
translation correction. That helps contact, but it does not adequately constrain
feet/floor and does not fully remove root/body jerk.

## Do not trust iter022 blindly

There is an `experiments/agentic_loop/iter022/` config in this checkout:

```text
experiments/agentic_loop/iter022/sample_stage2_hf_bps_may15_full_length.yaml
```

It contains extra flags such as:

```yaml
robot_contact_refinement_anchor_tracking: true
robot_contact_refinement_contact_velocity_weight: 0.20
robot_contact_refinement_jerk_reg_weight: 0.08
robot_contact_refinement_residual_stride: 8
robot_contact_refinement_soft_activation_margin: 0.04
```

The user says another agent's version harmed the hands while trying to fix root.
Treat iter022 as untrusted until independently verified. Do not replace the May14
i020 reference with this config just because it is newer.

## Why naive root/body smoothing breaks hands

World-space hand positions are a function of:

```text
root_pos + root_rot + torso/arm joints
```

If you smooth root position or root rotation after contact has been made, you move
the hands in world space. Unless you compensate with arm/upper-body joints, the
hands slide off the object. This is exactly the failure mode the user observed
from the other attempted fix.

Therefore the correct objective is not:

```text
smooth root/body
```

It is:

```text
smooth root/body while preserving FK hands near i020 contact anchors
and keeping feet above/on the ground
```

## What the next implementation should do

Implement a post-sampling **contact-preserving whole-body stabilizer** that runs
after the current i020 contact refinement. It should optimize the generated
sequence itself, not train a new model first.

Suggested function name:

```python
apply_contact_preserving_motion_stabilization(...)
```

Suggested location:

```text
utils/robot_kinematics.py
```

or, if the file is getting too large:

```text
utils/robot_motion_refinement.py
```

Then wire it into:

```text
scripts/sample_stage2_optimized.py
```

immediately after `apply_robot_contact_state_refinement(...)` and before
`robot_hands = robot_hand_positions(...)`.

### Inputs

The stabilizer should take:

```python
root_pos: np.ndarray          # (T, 3), generated i020 root
root_rot_xyzw: np.ndarray     # (T, 4), generated i020 root quat
dof_pos: np.ndarray           # (T, 29), generated i020 joints
target_hands: np.ndarray      # (T, 6), i020 hands_rectified contact anchors
object_verts: Optional[np.ndarray]
floor_height: float = 0.0
```

It should not require GT root/joints/hands at inference. Source/GT data can be
used only for evaluation.

### Outputs

Return:

```python
root_pos_out, root_rot_xyzw_out, dof_pos_out, metadata
```

Also update `state_np` in `scripts/sample_stage2_optimized.py` after calling it,
the same way the current contact refinement updates `state_np[:, :3]`,
`state_np[:, 3:9]`, and `state_np[:, 9:]`.

## The objective to optimize

Optimize a torch state:

```text
state = [root_pos(3), root_rot_6d(6), dof_pos(29)]
```

Use differentiable FK to compute hands and feet. Minimize:

```text
total =
  w_hand_contact * hand_contact_loss
  + w_floor_penetration * foot_penetration_loss
  + w_foot_slide * foot_sliding_loss
  + w_root_jerk * root_jerk_loss
  + w_body_jerk * state_jerk_loss
  + w_root_acc * root_acc_loss
  + w_state_acc * state_acc_loss
  + w_pose_reg * pose_reg_to_i020
  + w_velocity_reg * velocity_reg_to_i020
```

The purpose of each term:

- `hand_contact_loss`: keeps FK hands close to `hands_rectified` on active
  contact frames. This is the guardrail that prevents root smoothing from ruining
  object contact.
- `foot_penetration_loss`: penalizes foot/sole proxy points below the floor.
- `foot_sliding_loss`: during stance, penalizes horizontal foot velocity.
- `root_jerk_loss`: directly targets visible back-and-forth root twitch.
- `state_jerk_loss`: catches non-root joint jitter.
- `root_acc_loss` and `state_acc_loss`: smoother than jerk alone, usually helps
  with visible oscillation.
- `pose_reg_to_i020`: keeps the optimizer from inventing a new motion.
- `velocity_reg_to_i020`: preserves intended motion timing and prevents over-
  smoothing.

Start as a postprocess optimizer with 30-80 Adam steps. This is inference-time
sampling refinement, not a separate trained controller.

## Contact frames: use the current i020 anchors

Use the current i020 target hands as the reference:

```python
target_hands = hands_rect_np  # generated Stage 1 rectified anchors
```

Detect active hand-object frames exactly as the existing code does:

```python
left_contact, right_contact = detect_contact_frames(
    target_hands[:T],
    object_verts[:T],
    contact_threshold=activation_threshold,
)
```

Where this exists now:

```text
utils/robot_kinematics.py: apply_robot_contact_state_refinement(...)
```

Do not use GT hand positions for the stabilizer. GT/source hands are evaluation
only. The stabilizer should be deployable with generated Stage 1 targets, object
geometry, robot FK, and a floor plane.

## Foot/floor handling

The current FK helper only exposes hand positions cleanly. Extend it.

Current useful body names from the G1 XML:

```text
left_ankle_roll_link
right_ankle_roll_link
left_ankle_pitch_link
right_ankle_pitch_link
```

Joint order in `dof_pos`:

```text
0  left_hip_pitch_joint
1  left_hip_roll_joint
2  left_hip_yaw_joint
3  left_knee_joint
4  left_ankle_pitch_joint
5  left_ankle_roll_joint
6  right_hip_pitch_joint
7  right_hip_roll_joint
8  right_hip_yaw_joint
9  right_knee_joint
10 right_ankle_pitch_joint
11 right_ankle_roll_joint
12 waist_yaw_joint
13 waist_roll_joint
14 waist_pitch_joint
15 left_shoulder_pitch_joint
16 left_shoulder_roll_joint
17 left_shoulder_yaw_joint
18 left_elbow_joint
19 left_wrist_roll_joint
20 left_wrist_pitch_joint
21 left_wrist_yaw_joint
22 right_shoulder_pitch_joint
23 right_shoulder_roll_joint
24 right_shoulder_yaw_joint
25 right_elbow_joint
26 right_wrist_roll_joint
27 right_wrist_pitch_joint
28 right_wrist_yaw_joint
```

Existing `_state_refinement_mask("upper")` only allows root rotation and upper
body/arms:

```text
state[3:9] and dof indices 12:28
```

That is why feet/floor can be ignored. For the stabilizer, add a new mode such
as:

```text
stabilize_all
```

or:

```text
root_legs_upper
```

that allows:

- root translation and root rotation, but with tight deviation limits;
- leg joints 0:12 for foot-floor repair;
- waist and arms 12:29 for hand compensation.

Do not let root move freely without hand and foot constraints.

### Foot proxy points

Best implementation: extend differentiable FK to return body transforms, not just
positions, then transform a few local sole proxy offsets on each ankle roll link.

Suggested API:

```python
robot_body_transforms_from_state_torch(state, xml_path=DEFAULT_G1_XML)
```

returning body positions and rotations for all bodies, or at least selected
bodies.

Then define local foot proxy offsets relative to `left_ankle_roll_link` and
`right_ankle_roll_link`. Start simple:

```python
FOOT_PROXY_OFFSETS = torch.tensor([
    [ 0.08,  0.04, -0.035],  # toe outer-ish
    [ 0.08, -0.04, -0.035],  # toe inner-ish
    [-0.08,  0.04, -0.035],  # heel outer-ish
    [-0.08, -0.04, -0.035],  # heel inner-ish
])
```

These offsets are approximate. Calibrate visually/with source data if needed. In
the local HF-BPS largebox source, ankle roll z is often about 3-4 cm above floor,
so a `-0.03` to `-0.04` local z sole proxy is plausible.

If transform support is too much for the first pass, use ankle roll body origins
with a `foot_clearance` target around 0.03 m. This is less correct but still
better than allowing visible ground penetration. The transform/proxy approach is
preferred.

### Floor height

Use `floor_height = 0.0` by default. The local retargeted data appears to use a
ground plane around z=0; source ankle-roll origins are around 3-4 cm above it.

Do not infer floor height from GT at inference. You can expose YAML config:

```yaml
motion_stabilization_floor_height: 0.0
motion_stabilization_foot_clearance: 0.0
```

If using ankle origin instead of proxy sole points:

```yaml
motion_stabilization_ankle_min_height: 0.03
```

## Stance detection

For foot sliding, define stance from the generated motion itself, not GT:

1. Compute foot/sole proxy z.
2. Compute horizontal foot speed.
3. A foot is stance when:

```text
foot_z < floor_height + stance_height_threshold
and horizontal_speed < stance_speed_threshold
```

Start thresholds:

```text
stance_height_threshold = 0.06 m
stance_speed_threshold = 0.04 m/frame
```

Then smooth/dilate stance masks for temporal stability, e.g. a 3-5 frame window.

The sliding loss should be applied only on stance frames:

```text
mean(||foot_xy[t+1] - foot_xy[t]||^2 over stance frames)
```

Foot penetration loss should apply everywhere:

```text
relu(floor_height - sole_z)^2
```

If using ankle origin rather than sole proxies:

```text
relu(ankle_min_height - ankle_z)^2
```

## Suggested first implementation settings

Add YAML keys under `sample:`:

```yaml
motion_stabilization: true
motion_stabilization_steps: 50
motion_stabilization_lr: 0.01
motion_stabilization_mode: "root_legs_upper"
motion_stabilization_hand_weight: 80.0
motion_stabilization_floor_weight: 120.0
motion_stabilization_foot_slide_weight: 10.0
motion_stabilization_root_acc_weight: 2.0
motion_stabilization_root_jerk_weight: 8.0
motion_stabilization_state_acc_weight: 0.25
motion_stabilization_state_jerk_weight: 1.0
motion_stabilization_pose_reg_weight: 0.02
motion_stabilization_velocity_reg_weight: 0.05
motion_stabilization_max_root_delta: 0.08
motion_stabilization_max_joint_delta: 0.20
motion_stabilization_floor_height: 0.0
motion_stabilization_ankle_min_height: 0.03
motion_stabilization_stance_height_threshold: 0.06
motion_stabilization_stance_speed_threshold: 0.04
```

These are starting points. Tune with actual sample metrics and visualization.

Most important tuning rule:

- If hands degrade, raise `hand_weight`, reduce `max_root_delta`, reduce LR, or
  apply more arm/upper-body compensation.
- If feet still penetrate, raise `floor_weight` or allow leg joints.
- If root still jitters, raise root jerk/acc weights gradually.
- If motion becomes frozen/robotic, lower pose/velocity/smoothness weights.

## Acceptance gates

The new stabilizer must be accepted only if it improves smoothness/foot metrics
without meaningfully degrading hand contact. Do not judge by training loss.

Add these metrics to `scripts/evaluate_hf_bps_samples.py`:

- `root_acc_rms_cm`
- `root_jerk_rms_cm`
- `state_acc_rms`
- `state_jerk_rms`
- `dof_acc_rms`
- `dof_jerk_rms`
- `foot_penetration_mean_cm`
- `foot_penetration_max_cm`
- `foot_below_floor_frac`
- `foot_slide_cm`
- existing hand/contact metrics should remain.

For the user's current May14 i020 reference, compute metrics on:

```text
logs/stage2_hf_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026May07_22-33-42/samples/ts1000_2026May14_17-35-45_agentic_i020_fallback060_refine_smooth_wide_full_length
```

Then compare the new run against that reference.

Suggested acceptance criteria:

- `robot_to_target_hand_jpe_cm` does not worsen by more than 0.5-1.0 cm.
- `robot_contact_f1` does not worsen by more than 0.01-0.02 absolute.
- `robot_surface_p90_cm` does not worsen materially.
- `root_jerk_rms_cm` improves clearly, ideally at least 15-25%.
- `dof_jerk_rms` improves or stays flat.
- `foot_penetration_max_cm` and `foot_below_floor_frac` improve clearly.
- Visual check confirms less root back-and-forth and no obvious hand sliding.

If root improves but hands worsen, reject the change. The core requirement is
preserving the i020 contact quality.

## Where to wire config

In `scripts/sample_stage2_optimized.py`:

1. Add constructor args to `OptimizedOmomoPipeline.__init__`.
2. Store them on `self`.
3. Read YAML keys in `main()` near the existing `robot_contact_refinement_*`
   parameters.
4. In `generate(...)`, after the current block:

```python
if self.robot_contact_refinement:
    root_pos, root_rot_quat, dof_pos, robot_contact_refinement_metadata = ...
```

add:

```python
motion_stabilization_metadata = None
if self.motion_stabilization:
    root_pos, root_rot_quat, dof_pos, motion_stabilization_metadata = (
        apply_contact_preserving_motion_stabilization(...)
    )
    state_np[:, :3] = root_pos
    state_np[:, 3:9] = quat_to_rot6d_xyzw(torch.from_numpy(root_rot_quat).float()).numpy()
    state_np[:, 9:] = dof_pos
```

5. Include metadata in returned dict:

```python
"motion_stabilization_metadata": motion_stabilization_metadata
```

6. Recompute `robot_hands` after all refinements, as currently done.

## Avoid these traps

Do not simply low-pass `root_pos`, `root_rot`, or all joints after i020. That
will move FK hands off the object.

Do not use GT root/joints/hands inside inference. GT copied into sample files is
for evaluation/plotting only.

Do not optimize only the root. If root changes, arms and sometimes legs must
compensate.

Do not optimize all joints with weak regularization. The model can satisfy a
smoothness objective by inventing unrealistic posture or destroying contact.

Do not judge success from one scalar. Contact, surface distance, root jerk, joint
jerk, and foot penetration all matter.

Do not assume the newest experiment folder is best. The user's trusted reference
is the May14 i020 full-length sample path above.

## Data notes relevant to this issue

The user is comparing HF-BPS largebox behavior against OMOMO-style samples, and
corrected an earlier bad framing:

- HF-BPS largebox has about 202 files/samples.
- OMOMO sub1 clothesstand has about 70 files.
- The problem is not simply "HF has fewer samples."

Local observations from the repo:

- HF-BPS largebox samples are mostly `omomo_sub3_largebox_003_sample*`, i.e. many
  samples from one base motion/action rather than many independent motion IDs.
- BPS improves Stage 1 object geometry conditioning but does not directly solve
  Stage 2 root dynamics.
- Stage 2 is still mainly conditioned on 6D hand trajectories. Root behavior is
  inferred through a small bottleneck and is therefore vulnerable to jitter.

This reinforces the need for inference-time whole-body stabilization with hand
and foot constraints.

## Literature grounding

This proposed stabilizer is still within the motion-generation/diffusion
literature lane. It is analogous to analytical guidance/refinement used in
human-object interaction diffusion papers:

- OMOMO: explicit intermediate hand/contact-link trajectory before full-body
  motion.
- CHOIS/CG-HOI-style sampling guidance: use geometry/contact losses on predicted
  clean samples to reduce floating, penetration, and bad contact.
- Contact-guided HOI methods: treat contact anchors as constraints/guidance, not
  as emergent side effects.

This is not RL, MPC, or a controller. It is a post-denoising differentiable
projection/refinement of the generated motion under FK, contact, smoothness, and
floor constraints.

## Suggested implementation skeleton

Pseudo-code for the stabilizer:

```python
def apply_contact_preserving_motion_stabilization(
    root_pos,
    root_rot_xyzw,
    dof_pos,
    target_hands,
    object_verts=None,
    activation_threshold=0.16,
    floor_height=0.0,
    ankle_min_height=0.03,
    steps=50,
    lr=0.01,
    hand_weight=80.0,
    floor_weight=120.0,
    foot_slide_weight=10.0,
    root_acc_weight=2.0,
    root_jerk_weight=8.0,
    state_acc_weight=0.25,
    state_jerk_weight=1.0,
    pose_reg_weight=0.02,
    velocity_reg_weight=0.05,
    max_root_delta=0.08,
    max_joint_delta=0.20,
    mode="root_legs_upper",
    device="cuda",
    xml_path=DEFAULT_G1_XML,
):
    # 1. Build state0 = [root, root_rot6d, dof].
    # 2. Create state = state0.clone().requires_grad_(True).
    # 3. Build hand active mask from target_hands/object_verts.
    # 4. Build foot stance mask from initial generated feet, not GT.
    # 5. For each Adam step:
    #    - compute FK hands
    #    - compute FK feet/proxy sole points
    #    - hand_loss on active frames
    #    - foot penetration loss
    #    - foot sliding loss on stance frames
    #    - root/state acceleration and jerk losses
    #    - pose/velocity regularization to state0
    #    - backprop through allowed mask only
    #    - clamp root/joint deltas
    # 6. Convert refined rot6d back to xyzw quat.
    # 7. Return refined arrays + before/after metadata.
```

Loss details:

```python
hands = robot_hand_positions_from_state_torch(state.unsqueeze(0))[0].view(T, 2, 3)
hand_err = torch.linalg.norm(hands - target_t, dim=-1)
hand_loss = ((hands - target_t).pow(2).sum(-1) * active_t).sum() / active_t.sum().clamp_min(1.0)

root_acc = state[2:, :3] - 2 * state[1:-1, :3] + state[:-2, :3]
root_jerk = state[3:, :3] - 3 * state[2:-1, :3] + 3 * state[1:-2, :3] - state[:-3, :3]

state_acc = state[2:] - 2 * state[1:-1] + state[:-2]
state_jerk = state[3:] - 3 * state[2:-1] + 3 * state[1:-2] - state[:-3]

pose_reg = ((state - state0).pow(2) * state_mask).sum() / state_mask.sum().clamp_min(1.0)
vel_reg = (((state[1:] - state[:-1]) - (state0[1:] - state0[:-1])).pow(2) * state_mask[1:]).sum() / state_mask[1:].sum().clamp_min(1.0)
```

Foot loss depends on whether proxy sole points are implemented. With proxy points:

```python
penetration = torch.relu(floor_height - sole_points[..., 2])
floor_loss = penetration.pow(2).mean()
```

With ankle-origin fallback:

```python
penetration = torch.relu(ankle_min_height - ankle_positions[..., 2])
floor_loss = penetration.pow(2).mean()
```

## Verification commands

After implementation, sample a small set first. Use the user's normal shell
scripts if available; do not invent a totally new workflow unless necessary.
The direct command shape is:

```bash
/home/learning/miniconda3/envs/g1-diffusion/bin/python scripts/sample_stage2_optimized.py --config_path experiments/agentic_loop/<new_iter>/sample_stage2_hf_bps_<name>.yaml
```

Evaluate against HF-BPS source:

```bash
/home/learning/miniconda3/envs/g1-diffusion/bin/python scripts/evaluate_hf_bps_samples.py --sample_dir <new_sample_dir> --source_dir ./data/hf_bps_preprocessed --contact_threshold 0.05
```

Also evaluate the May14 i020 reference with the same updated evaluator:

```bash
/home/learning/miniconda3/envs/g1-diffusion/bin/python scripts/evaluate_hf_bps_samples.py --sample_dir logs/stage2_hf_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026May07_22-33-42/samples/ts1000_2026May14_17-35-45_agentic_i020_fallback060_refine_smooth_wide_full_length --source_dir ./data/hf_bps_preprocessed --contact_threshold 0.05
```

Visualization is mandatory before declaring success. The user already has scripts
for visualization/plotting; do not replace them unless asked.

## Final message expectations

When reporting back to the user, be direct:

- State exactly whether root/body jerk improved.
- State whether hand/contact metrics regressed.
- State whether foot penetration improved.
- Mention the sample folder and config used.
- Do not claim success from training loss.
- Do not bury the tradeoff. If hands worsened, say so and reject that variant.
