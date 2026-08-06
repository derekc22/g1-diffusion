# G1 Diffusion

Two-stage diffusion training for Unitree G1 object-manipulation motion. The
current object-goal configs use the OMOMO subset of
[`l-k-r/retargeted_motions`](https://huggingface.co/datasets/l-k-r/retargeted_motions),
preprocessed into local BPS pickle files.

This guide documents the complete setup validated on the
`/home/learning/Documents` installation, including the object-scale data and
MuJoCo viewer asset that are not stored in the Git repositories.

## Expected layout

```text
~/Documents/
  g1-diffusion/
  g1-gmr/
    assets/unitree_g1/
      g1_mocap_29dof.xml
      g1_mocap_29dof_with_object.xml
      meshes/
  omomo_release/
    data/
      captured_objects/
        *_cleaned_simplified.obj
      train_diffusion_manip_seq_joints24.p
      test_diffusion_manip_seq_joints24.p
```

Generated data inside `g1-diffusion`:

```text
data/
  hf_dataset/data/unitree_g1/omomo/...
  omomo_scale_references/*.pkl
  hf_bps_preprocessed/*.pkl
```

`data/` and `logs/` are ignored by Git. The `g1-gmr` repository also ignores
`assets/*`, so pulling or merging a branch does not restore the robot XMLs.

## 1. Environments

Training and preprocessing:

```bash
cd ~/Documents/g1-diffusion
conda activate g1-diffusion
python -m pip install huggingface_hub mujoco scipy joblib
```

The committed `environment.yml` records the original environment, including
Python 3.8, PyTorch 1.11, and CUDA 11.3. Use a CUDA/PyTorch build compatible
with the GPU and driver on the current machine.

Visualization uses the `g1-gmr` environment:

```bash
conda activate g1-gmr
```

## 2. Hugging Face authentication and download

Check authentication:

```bash
hf auth whoami
```

If necessary:

```bash
hf auth login
```

The project helper downloads all selected robot data and may request more than
14,000 individual files:

```bash
python scripts/download_hf_dataset.py \
  --repo_id l-k-r/retargeted_motions \
  --output_dir ./data/hf_dataset \
  --robot unitree_g1
```

If only OMOMO is needed, the following filtered download avoids most of those
requests while retaining the directory layout expected by the preprocessor:

```bash
HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="l-k-r/retargeted_motions",
    repo_type="dataset",
    local_dir="./data/hf_dataset",
    allow_patterns=[
        "data/unitree_g1/omomo/**/motion.npz",
        "data/unitree_g1/omomo/**/motion.csv",
        "data/unitree_g1/omomo/**/object_motion.npz",
        "data/unitree_g1/omomo/**/metadata.json",
    ],
    max_workers=1,
)
PY
```

Do not use `datasets.load_dataset()` for this pipeline. The preprocessor needs
the repository's original nested file layout.

Verify the OMOMO download:

```bash
for name in motion.npz motion.csv object_motion.npz metadata.json; do
  printf '%-20s' "$name"
  find data/hf_dataset/data/unitree_g1/omomo -name "$name" | wc -l
done
```

The validated dataset has `215` of each file. `motion.csv` is required: current
`motion.npz` files contain 29 joint values but no separate `root_pos`, while
the compatible loader obtains the combined root and joint state from the CSV.

### Hugging Face 429 errors

If the Hub reports `429 Too Many Requests`, do not delete the partial download.
Wait at least five minutes for the request window to reset, verify
`hf auth whoami`, and rerun the same command. The download resumes into the
same local directory.

## 3. OMOMO meshes and scale records

The Hugging Face motion repository does not contain enough information by
itself to reconstruct the fitted OMOMO mesh scale. The following local OMOMO
files are required:

```text
~/Documents/omomo_release/data/captured_objects/
~/Documents/omomo_release/data/train_diffusion_manip_seq_joints24.p
~/Documents/omomo_release/data/test_diffusion_manip_seq_joints24.p
```

Cloning the OMOMO source repository is not required if this data directory is
already available.

Verify the meshes and scale records:

```bash
test -d ~/Documents/omomo_release/data/captured_objects \
  && echo "OMOMO meshes: FOUND" \
  || echo "OMOMO meshes: MISSING"

find ~/Documents/omomo_release/data/captured_objects \
  -maxdepth 1 -name '*.obj' | wc -l

for name in \
  train_diffusion_manip_seq_joints24.p \
  test_diffusion_manip_seq_joints24.p; do
  test -f ~/Documents/omomo_release/data/$name \
    && echo "$name: FOUND" \
    || echo "$name: MISSING"
done
```

The validated setup has 19 OBJ files.

### Build the scale-reference directory

`preprocess_hf_bps_data.py` expects one reference pickle per motion. Convert
the two aggregate OMOMO files into that format:

```bash
cd ~/Documents/g1-diffusion
conda activate g1-diffusion

python - <<'PY'
from pathlib import Path
import pickle

import joblib
import numpy as np

omomo = Path.home() / "Documents/omomo_release/data"
hf_motions = Path("data/hf_dataset/data/unitree_g1/omomo")
output = Path("data/omomo_scale_references")
output.mkdir(parents=True, exist_ok=True)

wanted = {path.name for path in hf_motions.iterdir() if path.is_dir()}
written = set()

for filename in [
    "train_diffusion_manip_seq_joints24.p",
    "test_diffusion_manip_seq_joints24.p",
]:
    data = joblib.load(omomo / filename)
    for item in data.values():
        name = item.get("seq_name")
        if name not in wanted or "obj_scale" not in item:
            continue

        scale = np.asarray(item["obj_scale"]).reshape(-1)
        reference = {"object_mesh_scale": float(scale[0])}
        with open(output / f"{name}.pkl", "wb") as f:
            pickle.dump(reference, f, protocol=pickle.HIGHEST_PROTOCOL)
        written.add(name)

missing = sorted(wanted - written)
if missing:
    raise RuntimeError(f"Missing OMOMO scales for: {missing}")

print("Reference files written:", len(written))
print("Output:", output.resolve())
PY
```

Expected result:

```text
Reference files written: 15
```

Important: **do not pass `--reference_dir ""` for OMOMO BPS preprocessing.**
An empty reference directory makes the script use its fallback mesh scale of
`1.0`. The raw meshes are not in metres, so that produces enormous objects and
also corrupts the BPS geometry and contact labels used by two-stage training.

For example, the validated `sub1_largetable_053` scale is approximately
`0.02589392`, not `1.0`.

## 4. G1 MuJoCo assets

The BPS preprocessor needs the 29-DOF G1 XML and all adjacent mesh files:

```text
~/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml
~/Documents/g1-gmr/assets/unitree_g1/meshes/
```

If they are missing, download the matching asset bundle:

```bash
hf download retarget/retarget_example \
  --repo-type dataset \
  --revision ea218b069c6b81ee0b14b4a24cb838b0fcd67975 \
  --include 'processed/amass/unitree_g1/assets/robots/unitree_g1/**' \
  --local-dir /tmp/g1_assets_download

mkdir -p ~/Documents/g1-gmr/assets/unitree_g1

cp -a \
  /tmp/g1_assets_download/processed/amass/unitree_g1/assets/robots/unitree_g1/. \
  ~/Documents/g1-gmr/assets/unitree_g1/
```

Verify:

```bash
test -f ~/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml \
  && echo "G1 XML: FOUND" \
  || echo "G1 XML: MISSING"

find ~/Documents/g1-gmr/assets/unitree_g1/meshes -type f | wc -l
```

The validated bundle has 64 mesh files.

## 5. Object-mesh viewer support in `g1-gmr`

`scripts/visualize_model_dynamic.py` passes `object_mesh_path` and
`object_mesh_scale` to `RobotMotionViewerWithObject`. The installed `g1-gmr`
`main` branch must contain the former `two_stage` viewer changes:

```bash
git -C ~/Documents/g1-gmr pull

rg -n 'object_mesh_path|object_mesh_scale' \
  ~/Documents/g1-gmr/general_motion_retargeting/robot_motion_viewer_w_object.py
```

If those constructor parameters are absent, the viewer fails with:

```text
TypeError: RobotMotionViewerWithObject.__init__() got an unexpected keyword argument 'object_mesh_path'
```

### Create `g1_mocap_29dof_with_object.xml`

The viewer code references an additional XML that is not included in the asset
bundle and is ignored by `g1-gmr/.gitignore`. Create it from the existing
29-DOF XML by adding one free-joint placeholder body. At runtime, the viewer
replaces this placeholder box with the selected OMOMO mesh.

```bash
cd ~/Documents/g1-gmr

python - <<'PY'
from pathlib import Path

src = Path("assets/unitree_g1/g1_mocap_29dof.xml")
dst = Path("assets/unitree_g1/g1_mocap_29dof_with_object.xml")

xml = src.read_text()
marker = "</worldbody>"
index = xml.rfind(marker)
if index == -1:
    raise RuntimeError(f"No {marker} found in {src}")

# Keep name="object" and type="box" together: the current viewer matcher
# expects this exact attribute order.
placeholder = """    <body name="object">
      <joint name="object" type="free"/>
      <geom name="object" type="box" size="0.1 0.1 0.1" mass="0.01" contype="0" conaffinity="0" rgba="0.72 0.58 0.42 1"/>
    </body>
  """

dst.write_text(xml[:index] + placeholder + xml[index:])
print(f"Created: {dst}")
PY
```

Validate the model layout in the `g1-gmr` environment:

```bash
conda activate g1-gmr

python - <<'PY'
import mujoco

path = "assets/unitree_g1/g1_mocap_29dof_with_object.xml"
model = mujoco.MjModel.from_xml_path(path)
joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "object",
)

print("nq:", model.nq)
print("object qpos start:", model.jnt_qposadr[joint_id])
PY
```

Expected:

```text
nq: 43
object qpos start: 36
```

If the viewer says `Could not replace placeholder object body`, inspect the
block and confirm that the geom begins exactly with
`<geom name="object" type="box"`.

## 6. Test scale-aware preprocessing on one motion

Always test one item before building all 215:

```bash
cd ~/Documents/g1-diffusion
conda activate g1-diffusion

python scripts/preprocess_hf_bps_data.py \
  --input_dir ./data/hf_dataset/data/unitree_g1 \
  --output_dir /tmp/hf_bps_scaled_test \
  --objects_dir ~/Documents/omomo_release/data/captured_objects \
  --reference_dir ./data/omomo_scale_references \
  --robot_xml ~/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml \
  --datasets omomo \
  --num_motions 1 \
  --skip_missing_mesh \
  --min_length 30
```

Expected summary:

```text
Successful: 1
Skipped: 0
Failed: 0
```

Verify the first sample's scale and geometry:

```bash
python - <<'PY'
import pickle
import numpy as np

path = "/tmp/hf_bps_scaled_test/omomo_sub1_largetable_053_sample1.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print("stored mesh scale:", data["object_mesh_scale"])
print("stored object dimensions:", np.ptp(data["object_verts"][0], axis=0))
PY
```

Validated output:

```text
stored mesh scale: 0.02589392475783825
stored object dimensions: [0.4891137  0.5660041  0.57940805]
```

## 7. Generate `hf_bps_preprocessed`

On a fresh machine, write the correct data directly to the canonical path used
by the training and sampling configs:

```bash
python scripts/preprocess_hf_bps_data.py \
  --input_dir ./data/hf_dataset/data/unitree_g1 \
  --output_dir ./data/hf_bps_preprocessed \
  --objects_dir ~/Documents/omomo_release/data/captured_objects \
  --reference_dir ./data/omomo_scale_references \
  --robot_xml ~/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml \
  --datasets omomo \
  --skip_missing_mesh \
  --min_length 30
```

Expected result:

```text
Successful: 215
Skipped: 0
Failed: 0
```

Verify both the files and metadata:

```bash
echo "PKL files:"
find data/hf_bps_preprocessed -maxdepth 1 -name '*.pkl' | wc -l

python - <<'PY'
import json

with open("data/hf_bps_preprocessed/preprocessing_meta.json") as f:
    meta = json.load(f)

print("successful sequences:", meta["num_sequences"])
print("failed:", meta["failed"])
print("inferred scales:", len(meta["inferred_reference_scales"]))
PY
```

Expected:

```text
PKL files:
215
successful sequences: 215
failed: 0
inferred scales: 15
```

### Correct an existing scale-1.0 build safely

Build the corrected data alongside the old directory. Reuse the old BPS basis
so only the object scale changes:

```bash
python scripts/preprocess_hf_bps_data.py \
  --input_dir ./data/hf_dataset/data/unitree_g1 \
  --output_dir ./data/hf_bps_preprocessed_corrected \
  --objects_dir ~/Documents/omomo_release/data/captured_objects \
  --reference_dir ./data/omomo_scale_references \
  --robot_xml ~/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml \
  --bps_basis_path ./data/hf_bps_preprocessed/bps_basis_points.npy \
  --datasets omomo \
  --skip_missing_mesh \
  --min_length 30
```

After verifying the corrected build has 215 PKLs, make it active without
deleting the old data:

```bash
mv data/hf_bps_preprocessed \
   data/hf_bps_preprocessed_incorrect_scale

mv data/hf_bps_preprocessed_corrected \
   data/hf_bps_preprocessed
```

These commands only rename directories:

```text
Before:
  hf_bps_preprocessed            incorrect scale
  hf_bps_preprocessed_corrected  correct scale

After:
  hf_bps_preprocessed_incorrect_scale  incorrect backup
  hf_bps_preprocessed                 correct active dataset
```

## 8. Train and sample

Object-goal Stage 1:

```bash
./src/train_object_goal_stage1_hf_bps.sh
```

Object-goal Stage 2:

```bash
./src/train_object_goal_stage2_hf_bps.sh
```

Single-stage goal-only baseline:

```bash
./src/train_object_goal_single_stage_hf_bps.sh
```

Sample the single-stage baseline after placing its checkpoint in
`experiments/object_goal/sample_object_goal_single_stage_hf_bps.yaml`:

```bash
./src/sample_object_goal_single_stage_hf_bps.sh
```

Sample the two-stage model after configuring both checkpoints in
`experiments/object_goal/sample_object_goal_stage2_hf_bps_optimized.yaml`:

```bash
./src/sample_object_goal_stage2_hf_bps_optimized.sh
```

Training outputs and timestamped sample directories are written under `logs/`.
The shell wrappers contain paths for the original
`/home/learning/Documents` installation; update those paths if the repositories
are installed elsewhere.

### After correcting an old scale-1.0 dataset

- **Single-stage goal-only checkpoint:** no retraining is required. That model
  consumes only robot/object poses and the final object goal; its training path
  explicitly disables BPS, object geometry, and contact inputs. Resample the
  existing checkpoint so new output PKLs copy the corrected
  `object_mesh_scale` metadata.
- **Two-stage pipeline:** retrain both object-goal stages. Stage 1 uses BPS
  geometry and object contact information, while Stage 2 uses static BPS
  context and geometry-derived contact labels/losses.
- Existing sample PKLs are not changed by replacing the dataset directory. Old
  samples that already contain `object_mesh_scale: 1.0` remain oversized.

## 9. Visualize generated samples

`src/visualize_object_goal_two_stage_dynamic.sh` works for both two-stage and
single-stage generated PKLs despite its filename.

Open the script and set `ROBOT_MOTION_FOLDER_ALL` to the exact newly generated
sample directory, for example:

```bash
ROBOT_MOTION_FOLDER_ALL="/home/learning/Documents/g1-diffusion/logs/<experiment>/samples/ddim_<timestamp>_object_goal_single_stage_goal_only"
```

Also verify these paths in the script:

```bash
OBJECTS_DIR="/home/learning/Documents/omomo_release/data/captured_objects"
GMR_ROOT="/home/learning/Documents/g1-gmr"
```

Then render:

```bash
./src/visualize_object_goal_two_stage_dynamic.sh
```

The script prints the final `render_dynamic.mp4` path. Pointing it at a parent
`samples/` directory causes it to render every child sample directory,
including any older scale-1.0 samples; use the exact new directory when only
the corrected run is desired.

## Troubleshooting

### `No root_pos found`

Confirm all 215 `motion.csv` files were downloaded:

```bash
find data/hf_dataset/data/unitree_g1/omomo -name motion.csv | wc -l
```

### `No .pkl files found in .../data/hf_bps_preprocessed`

Complete the scale-aware preprocessing in Step 7 and confirm the directory
contains 215 motion PKLs.

### Objects are enormous

Inspect a source PKL:

```bash
python - <<'PY'
import pickle

path = "data/hf_bps_preprocessed/omomo_sub1_largetable_053_sample1.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)
print(data.get("object_mesh_scale"))
PY
```

If it prints `1.0`, the dataset was built without OMOMO scale references.
Rebuild it using Steps 3, 6, and 7. For the largetable smoke test, the expected
scale is approximately `0.02589392`.

### `unexpected keyword argument 'object_mesh_path'`

The installed `g1-gmr` viewer is older than the dynamic object-mesh caller.
Update `g1-gmr/main` to include the former `two_stage` viewer implementation
described in Step 5.

### `g1_mocap_29dof_with_object.xml` is missing

This file is a local generated asset and is ignored by the `g1-gmr` repository.
Create and validate it using Step 5.

### `Could not replace placeholder object body`

In `g1_mocap_29dof_with_object.xml`, the placeholder geom must begin exactly:

```xml
<geom name="object" type="box"
```

Do not place a newline between `name="object"` and `type="box"` with the
current viewer matcher.

### NumPy `numpy.core.numeric` deprecation warning

This warning can appear while unpickling older NumPy data. It is not the cause
of viewer failure and can be ignored during this setup.

### MuJoCo cannot load the G1 robot

Check both the XML and the adjacent `meshes/` directory. Copying only the XML
is insufficient.
