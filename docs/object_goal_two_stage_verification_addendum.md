# Object-Goal Two-Stage Verification Addendum

This addendum answers the reviewer follow-up for the corrected two-stage
object-goal implementation.

## 1. Object Pose Semantics

The object pose used by the object-goal code is:

```text
object_pose = object_pos(3) + object_rot_6d(6)
```

For HF-BPS preprocessed data, `object_pos` is the object mesh centroid, not the
original mesh origin or a simulator body-frame origin.

Exact source:

- `scripts/preprocess_hf_bps_data.py::add_bps_geometry`
  - transforms mesh vertices with the raw object pose
  - computes `object_centroid = object_verts.mean(axis=1)`
  - stores `motion_data["object_centroid"] = object_centroid`
  - stores `motion_data["object_pos"] = object_centroid`
  - stores `motion_data["object_rotation"] = object_rotation`
  - stores `motion_data["object_rot"]` as the quaternion converted from `object_rotation`

Exact consumer:

- `utils/object_goal_features.py::object_pose_from_data`
  - prefers `object_pos` when present
  - falls back to `object_centroid`
  - prefers `object_rot` when present
  - falls back to `object_rotation`
  - converts quaternion/matrix rotation to the repo's first-two-columns `rot6d`

This matches the existing HF-BPS sampling path because the BPS preprocessing
already overwrites `object_pos` with the mesh centroid and uses that same
centroid to compute BPS basis positions:

```text
basis_world = basis * radius + object_centroid[t]
bps[t] = basis_world - nearest_object_vertex
```

So in the current HF-BPS data, object position, object centroid, BPS, and object
vertices are all aligned around the mesh centroid convention.

## 2. Stage 2 Target Versus Condition

Stage 2 target is:

```text
state[:, :, 0:38]  = robot_state
state[:, :, 38:47] = object_pose
```

The layout is defined in:

- `utils/object_goal_features.py::robot_object_layout`

Field slices:

```text
robot_state:
  root_pos:    [0, 3]
  root_rot_6d: [3, 9]
  dof_pos:     [9, 38]

object_pose:
  object_pos:    [38, 41]
  object_rot_6d: [41, 47]
```

Reviewer concern accepted: the previous Stage 2 condition did include the clean
per-frame object pose trajectory and centroid while denoising that same object
pose. That was target leakage.

This has been fixed.

Current Stage 2 per-frame condition is:

```text
cond = hands(6) + static_bps_context(3072) = 3078D
```

Current Stage 2 global condition is:

```text
global_cond = final_object_pose(9)
```

The clean per-frame object pose trajectory is no longer concatenated into
`cond`. The clean object centroid trajectory is also not concatenated into
`cond`.

Exact code:

- `datasets/hf_motion_dataset.py::HFFullBodyDataset.__getitem__`
  - target appends `object_pose_window` to robot state
  - condition starts with hand conditioning
  - BPS is flattened
  - `static_bps_context` repeats the first BPS frame over the window
  - condition appends only that static BPS context
- `scripts/train_object_goal_stage2_hf_bps.py`
  - prints `Condition dim: 3078`
  - saves checkpoint metadata: `per_frame: "hands + static_bps_context"`
- `scripts/sample_object_goal_two_stage.py`
  - builds Stage 2 condition with `torch.cat([hands_cond, static_bps], dim=-1)`

Distinction:

- Per-frame object pose trajectory `O_{0:T}`: Stage 2 target only.
- Static BPS geometry context: Stage 2 per-frame condition, repeated across time.
- Final object goal `g`: Stage 1 and Stage 2 global condition.

## 3. Stage 2 Losses

Stage 2 training currently uses these losses:

1. `base_loss = F.mse_loss(state_pred, state)`
   - Applies to both robot slice `[0:38]` and object slice `[38:47]`.
   - This is the primary reconstruction supervision for object pose.
   - Logging now also reports `robot_base` and `object_base`.

2. `temporal_reconstruction_loss(state_pred_phys, state_phys, loss_cfg)`
   - Applies to all `47D` dimensions.
   - This supervises temporal consistency of both robot state and object pose.
   - The active terms depend on config:
     - `velocity_weight`
     - `acceleration_weight`
     - `jerk_weight`
     - `smooth_acceleration_weight`
     - `smooth_jerk_weight`

3. `robot_fk_hand_loss(robot_pred_phys, cond_hands, loss_cfg)`
   - Applies only to the robot slice.
   - Code passes `robot_pred_phys = state_pred_phys[..., :38]`.
   - It does not see or index the object slice.

4. `full_body_contact_loss(robot_pred_phys, contact_logits, batch, loss_cfg, ...)`
   - Applies only to the robot slice.
   - Code passes `robot_pred_phys = state_pred_phys[..., :38]`.
   - Contact logits are predicted by the Stage 2 contact head when enabled, but
     geometry/FK contact losses use only the robot state slice.

Object pose slice supervision:

- Object pose is supervised by the full-state base MSE.
- Object pose is supervised by temporal reconstruction losses when the temporal
  weights are nonzero.
- There is no separate object-only contact/FK loss, because object pose is not
  a robot kinematic chain.

## 4. Stage 1 Contact Rectification

The normal HF-BPS object-goal sampling path requires object geometry by default.

HF-BPS preprocessing stores:

- `object_verts`
- `object_rotation`

unless preprocessing was explicitly run with `--no_object_verts`.

The object-goal sampler now defaults to:

```yaml
require_contact_geometry: true
```

Exact call site:

- `scripts/sample_object_goal_two_stage.py`
  - loads `object_verts = data.get("object_verts")`
  - loads `object_rotation = data.get("object_rotation")`
  - calls `ContactConstraintProcessor.process(...)`
  - raises if geometry is missing and `require_contact_geometry` is true

The rectifier implementation is:

- `utils/contact_constraints.py::ContactConstraintProcessor.process`

The verified 300-frame sample produced contact metadata containing rectifier
fields such as:

- `contact_search_threshold`
- `left_contact_frames`
- `right_contact_frames`
- `max_contact_offset`
- `max_contact_correction`

That confirms the rectifier path ran for the tested HF-BPS sample.

## 5. Horizon

The earlier smoke sample length of `132` came from the first sampled source
sequence being only 132 frames long. The sampler uses:

```text
T = min(source_length, max_len)
```

so it does not pad a short source sequence to 300 frames.

Full 300-frame paths were tested:

- dataset smoke loaded `(B, 300, ...)` windows
- Stage 1 dry training used `window_size: 300`
- Stage 2 dry training used `window_size: 300`
- full 300-frame sample/decode was run on:
  - `data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl`
  - source length: 326 frames

The full 300-frame sample output shapes were:

```text
hands_raw       (300, 6)
hands_rectified (300, 6)
state           (300, 47)
robot_state     (300, 38)
object_pose     (300, 9)
root_pos        (300, 3)
root_rot        (300, 4)
dof_pos         (300, 29)
goal            (9,)
```

Default train configs still use a 300-frame horizon:

- `config/train_object_goal_stage1_hf_bps.yaml`
- `config/train_object_goal_stage2_hf_bps.yaml`
- both smoke configs

## 6. Files Changed

### g1-diffusion Modified

- `datasets/hand_motion_dataset.py`
- `datasets/hf_motion_dataset.py`
- `models/stage1_diffusion.py`
- `models/stage2_diffusion.py`

### g1-diffusion Added

- `utils/object_goal_features.py`
- `scripts/train_object_goal_stage1_hf_bps.py`
- `scripts/train_object_goal_stage2_hf_bps.py`
- `scripts/sample_object_goal_two_stage.py`
- `scripts/smoke_object_goal_two_stage.py`
- `config/train_object_goal_stage1_hf_bps.yaml`
- `config/train_object_goal_stage1_hf_bps_smoke.yaml`
- `config/train_object_goal_stage2_hf_bps.yaml`
- `config/train_object_goal_stage2_hf_bps_smoke.yaml`
- `experiments/object_goal/sample_object_goal_two_stage.yaml`
- `docs/object_goal_smp.md`
- `docs/object_goal_two_stage_implementation_summary.md`
- `docs/object_goal_two_stage_verification_addendum.md`

### Removed From The Previous Incorrect Pass

These were created in the earlier one-stage attempt and then removed:

- `HFObjectGoalDataset`
- `scripts/train_object_goal_prior_hf.py`
- `scripts/smoke_object_goal_smp.py`
- `config/train_object_goal_prior_hf.yaml`
- `config/train_object_goal_prior_hf_smoke.yaml`
- SMP object-goal prior loader/reward path
- `Smp-ObjectGoal-G1`
- free-box SMP task package

### smp

The `smp` repo has no current diff. The earlier free-box object-goal task and
object-goal reward/loader edits were reverted/removed from the main path.

## 7. Commands

Train Stage 1:

```bash
python scripts/train_object_goal_stage1_hf_bps.py \
  --config_path config/train_object_goal_stage1_hf_bps.yaml
```

Train Stage 2:

```bash
python scripts/train_object_goal_stage2_hf_bps.py \
  --config_path config/train_object_goal_stage2_hf_bps.yaml
```

Sample from trained checkpoints:

1. Set `stage1_ckpt_path` and `stage2_ckpt_path` in:

```text
experiments/object_goal/sample_object_goal_two_stage.yaml
```

2. Run:

```bash
python scripts/sample_object_goal_two_stage.py \
  --config_path experiments/object_goal/sample_object_goal_two_stage.yaml
```

Smoke-test the pipeline:

```bash
python scripts/smoke_object_goal_two_stage.py \
  --device cpu \
  --batch_size 2 \
  --window_size 300
```

CPU dry-run Stage 1:

```bash
python scripts/train_object_goal_stage1_hf_bps.py \
  --config_path config/train_object_goal_stage1_hf_bps_smoke.yaml
```

CPU dry-run Stage 2:

```bash
python scripts/train_object_goal_stage2_hf_bps.py \
  --config_path config/train_object_goal_stage2_hf_bps_smoke.yaml
```
