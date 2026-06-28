# Object-Goal Two-Stage Diffusion

This document describes the corrected object-goal diffusion path. The active
deliverable is diffusion training and sampling in `g1-diffusion`; PPO/RL and
the free-box SMP task are intentionally out of the main path for now.

## Factorization

The pipeline preserves the OMOMO-style hand intermediate:

```text
p(R, O, H | G, g)
  = p(H | O, G, g) * p(R, O | H_hat, G, g)
```

where:

- `R`: robot state trajectory
- `O`: object pose trajectory
- `H`: left/right hand trajectory
- `H_hat`: Stage 1 hands after contact rectification
- `G`: HF-BPS object geometry, centroid, object metadata, and BPS semantics
- `g`: final full object pose

## Representation

- Robot state: `root_pos(3), root_rot_6d(6), dof_pos(29)` = `38D`
- Object pose: `object_pos/object_centroid(3), object_rot_6d(6)` = `9D`
- Stage 1 target: hand trajectory = `6D`
- Stage 2 target: `[robot_state, object_pose]` = `47D`
- Stage 2 per-frame condition: rectified/training hands + static BPS geometry context = `3078D`
- Global goal condition: final full object pose = `9D`
- Default horizon: `300` frames

## Implemented Training Path

Stage 1:

```text
target:    H_{0:T}
condition: HF-BPS object geometry/centroid + O_{0:T} + g
script:    scripts/train_object_goal_stage1_hf_bps.py
config:    config/train_object_goal_stage1_hf_bps.yaml
```

Stage 2:

```text
target:    [R_{0:T}, O_{0:T}]
condition: H_hat or training hand condition + static BPS geometry context + g
script:    scripts/train_object_goal_stage2_hf_bps.py
config:    config/train_object_goal_stage2_hf_bps.yaml
```

Stage 2 intentionally does not receive the clean per-frame object pose
trajectory as condition. It denoises that object pose as part of the `47D`
target. The only object-pose value used as a Stage 2 condition is the final
goal pose `g`.

Smoke configs:

- `config/train_object_goal_stage1_hf_bps_smoke.yaml`
- `config/train_object_goal_stage2_hf_bps_smoke.yaml`

Sampling:

```text
HF-BPS object context
  -> Stage 1 hand diffusion
  -> contact rectification
  -> Stage 2 robot-plus-object diffusion
  -> decoded sample pickle
```

Sampling script/config:

- `scripts/sample_object_goal_two_stage.py`
- `experiments/object_goal/sample_object_goal_two_stage.yaml`

## Explicit Corrections

- Stage 1 remains active.
- Stage 1 predicts hands.
- Stage 1 uses the current repo's HF-BPS object geometry, centroid, pose, and metadata semantics.
- Hand contact rectification remains active before Stage 2 sampling.
- Stage 2 does not bypass Stage 1.
- Stage 2's diffusion target includes both robot state and object pose, matching the block diagram.
- The previous single-stage `p(R, O | g)` 47D shortcut was removed.
- The free-box SMP task was removed from the main path.
