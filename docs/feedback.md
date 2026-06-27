# Feedback on active motion-stabilization changes

I reviewed the active working-tree changes after the other agent's implementation.
The approach is directionally correct: it adds a post-sampling optimizer that
uses FK hand anchors, foot/floor proxies, and root/body smoothness terms. A small
smoke test on one 60-frame largebox segment showed promising behavior:

```text
robot_to_target_hand_jpe_cm: 1.575 -> 0.152
root_jerk_rms_cm:           0.319 -> 0.111
foot_penetration_max_cm:    0.443 -> 0.000
foot_slide_cm:              0.519 -> 0.122
```

That said, I have concerns before trusting it as the new default.

## High-priority concerns

### 1. No acceptance/rollback gate

`scripts/sample_stage2_optimized.py:1010-1043` always applies
`apply_contact_preserving_motion_stabilization(...)` when the YAML flag is on.
`utils/robot_kinematics.py:936-965` records before/after metrics in metadata, but
there is no automatic rejection if the optimizer damages contact or surface
distance.

This is the biggest risk because the user's exact failure mode was: another
version improved root but wrecked hands.

Recommended fix:

- Add config tolerances such as:
  - `motion_stabilization_max_hand_regress_cm`
  - `motion_stabilization_max_surface_p90_regress_cm`
  - `motion_stabilization_require_root_jerk_improvement`
  - `motion_stabilization_require_foot_penetration_improvement`
- Compute before/after hand-to-target and mesh-surface metrics.
- If contact/surface gets worse beyond tolerance, return the original i020
  `root_pos/root_rot/dof_pos` and mark metadata as rejected.

### 2. Root rotation is effectively unclamped

`utils/robot_kinematics.py:594-595` makes `root_legs_upper` optimize the full
state. Later, `utils/robot_kinematics.py:887-898` clamps root translation and
joint deltas, but it does not clamp root rotation / 6D root orientation deltas.

That means the optimizer can use root orientation as an unbounded escape hatch to
satisfy floor/smoothness/contact losses. Pose regularization helps, but it is a
soft term and may not be enough.

Recommended fix:

- Add `motion_stabilization_max_root_rot_delta` or lock root rotation for the
  first pass.
- Alternatively allow yaw-only smoothing and keep roll/pitch tightly bounded.
- Report max root rotation delta in metadata.

### 3. Surface contact can worsen while hand-anchor error improves

The stabilizer uses `hands_rectified` as the hand target. That is reasonable, but
it does not directly penalize robot-hand distance to the object surface.

In the smoke test, hand-anchor error improved strongly, but robot/object surface
distance got slightly worse:

```text
robot_surface_mean_cm: 2.280 -> 3.006
robot_surface_p90_cm:  2.651 -> 3.541
```

This is not catastrophic in that sample, but it proves the optimizer can improve
one contact proxy while degrading another. Since the user cares about object
interaction realism, surface metrics need to be part of the acceptance gate.

Recommended fix:

- Add before/after surface distance checks when `object_verts` is available.
- Consider a light object-surface term on active hand frames, not just hand-anchor
  tracking.

### 4. Foot sliding loss ignores feet that are already sliding

Stance is computed from the initial generated motion at
`utils/robot_kinematics.py:817-827`. The mask requires:

```text
foot near floor AND initial horizontal speed < stance_speed_threshold
```

Then `utils/robot_kinematics.py:853-858` applies foot sliding loss only on that
mask. If a foot is visibly skating in the initial motion, its speed may exceed the
threshold, so it gets excluded from the sliding loss and is never fixed.

Recommended fix:

- Detect stance primarily from low foot height / penetration, not already-low
  speed.
- Or use a soft weight instead of hard-dropping high-speed feet.
- At minimum, add metadata for stance coverage per foot so it is obvious when
  the slide loss is inactive.

### 5. Default scripts now point at untrusted May15/i022 paths

`experiments/stage2_hf_bps_optimized/sample_stage2_hf_bps_optimized.yaml` was
changed from the trusted May07 checkpoints to May15 checkpoints.

`src/visualize_model_hf_dynamic.sh` now points at:

```text
ts1000_2026May15_16-44-51_hf_bps_contact_refine_i022_full_length
```

The user explicitly identified the trusted reference as the May14 i020 full-length
sample:

```text
ts1000_2026May14_17-35-45_agentic_i020_fallback060_refine_smooth_wide_full_length
```

Recommended fix:

- Do not overwrite the default optimized config or visualization script with
  untrusted experiment paths.
- Keep i020/Mai14 as the comparison reference.
- Put new variants in separate iter023 scripts/configs until accepted.

### 6. iter022 contains YAML flags that are silently ignored

`experiments/agentic_loop/iter022/sample_stage2_hf_bps_may15_full_length.yaml`
contains keys like:

```yaml
robot_contact_refinement_anchor_tracking: true
robot_contact_refinement_contact_velocity_weight: 0.20
robot_contact_refinement_jerk_reg_weight: 0.08
robot_contact_refinement_residual_stride: 8
robot_contact_refinement_soft_activation_margin: 0.04
```

But `rg` only finds those keys in the YAML; `scripts/sample_stage2_optimized.py`
does not parse or use them. That creates false confidence that i022 includes
features that are not actually active.

Recommended fix:

- Remove ignored keys, or implement and wire them properly.
- Prefer failing on unknown experimental keys for critical configs.

## Medium-priority concerns

### 7. Foot proxy offsets are plausible but not calibrated

The new foot proxy offsets in `utils/robot_kinematics.py:277-282` are reasonable
as a first pass, and torch/numpy FK foot proxy positions agree in a smoke check.
Still, the offsets are approximate and may not correspond to the visible sole
mesh.

Recommended fix:

- Validate proxy points visually against the rendered G1 feet.
- Add metadata for min/max sole z before and after.
- If possible, derive the sole z offset from mesh bounds rather than hardcoding.

### 8. Evaluator has useful new metrics but no reference comparison mode

`scripts/evaluate_hf_bps_samples.py` now reports root/body/foot metrics, which is
good. But the user needs to compare a new run against the trusted i020 reference
without manually diffing two reports.

Recommended fix:

- Add optional `--baseline_sample_dir`.
- Print deltas for contact, surface, root jerk, foot penetration, and foot slide.
- Mark PASS/FAIL against explicit thresholds.

## Bottom line

The implementation is worth continuing. It is not obviously broken, and the core
postprocess idea is aligned with the desired contact-preserving root/body fix.

But I would not make it the default yet. The missing rollback gate, unclamped root
rotation, ignored iter022 config keys, and overwritten visualization/default paths
are enough to produce misleading results or accidentally ship a variant that
improves root while damaging contact.
