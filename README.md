# G1 Diffusion

Two-stage diffusion training for Unitree G1 object-manipulation motion. The
main data path used by the current training configs is the Hugging Face OMOMO
subset preprocessed into local BPS pickle files.

## Expected layout

The commands below assume the repositories and assets are arranged as:

```text
~/Documents/
  g1-diffusion/
  g1-gmr/
    assets/unitree_g1/
      g1_mocap_29dof.xml
      meshes/
  omomo_release/
    data/captured_objects/
      *_cleaned_simplified.obj
```

Inside `g1-diffusion`, generated data will be stored as:

```text
data/
  hf_dataset/data/unitree_g1/omomo/...
  hf_bps_preprocessed/*.pkl
```

`data/` and `logs/` are ignored by Git.

## 1. Activate the environment

```bash
cd ~/Documents/g1-diffusion
conda activate g1-diffusion
pip install huggingface_hub mujoco scipy
```

The committed `environment.yml` records the original environment, including
Python 3.8, PyTorch 1.11, and CUDA 11.3. Use a CUDA/PyTorch build compatible
with the GPU and driver on the training machine.

## 2. Check Hugging Face authentication

```bash
hf auth whoami
```

If necessary:

```bash
hf auth login
```

The source dataset is:

```text
l-k-r/retargeted_motions
```

## 3. Download the required OMOMO files

Do not use `datasets.load_dataset()` for this pipeline. The preprocessor needs
the original nested directory structure. Download only the OMOMO files used by
the BPS pipeline to avoid making more than 14,000 Hub requests:

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

The CSV files are required with the current dataset. Current `motion.npz`
files store 29 joint values separately from the root pose, while the repository
loader's compatible fallback reads the combined root and joint state from
`motion.csv`.

Verify the download:

```bash
find data/hf_dataset/data/unitree_g1/omomo -name motion.npz | wc -l
find data/hf_dataset/data/unitree_g1/omomo -name motion.csv | wc -l
find data/hf_dataset/data/unitree_g1/omomo -name object_motion.npz | wc -l
find data/hf_dataset/data/unitree_g1/omomo -name metadata.json | wc -l
```

At the time this setup was validated, each command printed `215`.

### Hugging Face 429 errors

If the Hub reports `429 Too Many Requests`, keep the partial download, wait at
least five minutes for the request window to reset, confirm `hf auth whoami`,
and rerun the filtered command above. `snapshot_download` resumes into the
same local directory.

## 4. Install the OMOMO object meshes

BPS encodes object geometry, so the Hugging Face motion files alone are not
enough. Obtain the OMOMO data and place its `captured_objects` directory at:

```text
/home/learning/Documents/omomo_release/data/captured_objects
```

Only the data directory is required; cloning the OMOMO source repository is
not necessary. Verify it with:

```bash
test -d ~/Documents/omomo_release/data/captured_objects \
  && echo "OMOMO meshes: FOUND" \
  || echo "OMOMO meshes: MISSING"

find ~/Documents/omomo_release/data/captured_objects \
  -maxdepth 1 -name '*.obj' | wc -l
```

The validated setup contained 19 OBJ files.

## 5. Install the G1 MuJoCo assets

The BPS preprocessor uses the G1 model for forward kinematics. If the following
file is already available, this step can be skipped:

```text
/home/learning/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml
```

Otherwise, download the matching asset bundle:

```bash
hf download retarget/retarget_example \
  --repo-type dataset \
  --revision ea218b069c6b81ee0b14b4a24cb838b0fcd67975 \
  --include 'processed/amass/unitree_g1/assets/robots/unitree_g1/**' \
  --local-dir /tmp/g1_assets_download
```

Copy it into the expected location:

```bash
mkdir -p ~/Documents/g1-gmr/assets/unitree_g1

cp -a \
  /tmp/g1_assets_download/processed/amass/unitree_g1/assets/robots/unitree_g1/. \
  ~/Documents/g1-gmr/assets/unitree_g1/
```

Verify the installation:

```bash
test -f ~/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml \
  && echo "G1 XML: FOUND" \
  || echo "G1 XML: MISSING"

find ~/Documents/g1-gmr/assets/unitree_g1/meshes -type f | wc -l
```

The validated bundle contained 64 mesh files.

## 6. Test preprocessing on one motion

Run one sample before processing the entire dataset:

```bash
python scripts/preprocess_hf_bps_data.py \
  --input_dir ./data/hf_dataset/data/unitree_g1 \
  --output_dir /tmp/hf_bps_test \
  --objects_dir ~/Documents/omomo_release/data/captured_objects \
  --reference_dir "" \
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

Passing an empty `--reference_dir` makes preprocessing use the configured
fallback object mesh scale of `1.0`. Recover the old reference PKLs and supply
their directory instead if exact numerical reproduction of an earlier BPS
build is required.

## 7. Generate `hf_bps_preprocessed`

```bash
python scripts/preprocess_hf_bps_data.py \
  --input_dir ./data/hf_dataset/data/unitree_g1 \
  --output_dir ./data/hf_bps_preprocessed \
  --objects_dir ~/Documents/omomo_release/data/captured_objects \
  --reference_dir "" \
  --robot_xml ~/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml \
  --datasets omomo \
  --skip_missing_mesh \
  --min_length 30
```

The validated run completed with:

```text
Successful: 215
Skipped: 0
Failed: 0
```

Confirm the generated files:

```bash
find data/hf_bps_preprocessed -maxdepth 1 -name '*.pkl' | wc -l
```

## 8. Train

Object-goal Stage 1:

```bash
./src/train_object_goal_stage1_hf_bps.sh
```

Object-goal Stage 2:

```bash
./src/train_object_goal_stage2_hf_bps.sh
```

Baseline Stage 1 HF-BPS variant0:

```bash
python scripts/train_stage1.py \
  --config_path ./config/train_stage1_hf_bps_variant0.yaml
```

Baseline Stage 2:

```bash
python scripts/train_stage2_hf.py \
  --config_path ./config/train_stage2_hf.yaml
```

Training outputs are written under `logs/`. The shell wrappers contain paths
for the original `/home/learning/Documents` installation, so use the direct
Python commands or update those paths if the repository is installed
elsewhere.

## Common preprocessing failures

### `No root_pos found`

Make sure all `motion.csv` files were downloaded. The expected count is 215:

```bash
find data/hf_dataset/data/unitree_g1/omomo -name motion.csv | wc -l
```

### `Object mesh directory not found`

Check that this exact directory exists:

```text
~/Documents/omomo_release/data/captured_objects
```

### MuJoCo cannot load the robot

Check both the XML and its adjacent `meshes/` directory. Copying only the XML
is insufficient for MuJoCo model loading.

### `No .pkl files found in .../data/hf_bps_preprocessed`

Complete Step 7 and confirm that the directory contains the expected 215 PKL
files before starting training.
