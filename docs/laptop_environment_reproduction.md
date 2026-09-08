# ForceVLA Laptop / Second-Machine Setup

This document records the validated thesis ForceVLA environment and the
workflow for reproducing the source/development setup on another Linux
machine.

## Validated lab reference

The lab workstation currently uses:

- Python 3.11.15
- JAX 0.5.3
- two NVIDIA RTX A5000 GPUs
- the Git commit recorded in
  `reproducibility/lab_runtime_reference.txt`

The exact installed package references are captured in:

- `reproducibility/lab_pip_versions.txt`
- `reproducibility/lab_editable_packages.txt`
- `reproducibility/lab_conda_packages.txt`
- `reproducibility/lab_submodules.txt`
- `reproducibility/lab_runtime_reference.txt`

The repository's own `.python-version`, `pyproject.toml`, `uv.lock`,
`.gitmodules`, and Git commit history remain the primary source-controlled
dependency contract.

## Clone on the laptop

Clone the thesis repository:

```bash
git clone \
  https://github.com/Raunak-Chakraborty/forcevla_thesis.git

cd forcevla_thesis
```

Synchronize the submodule configuration:

```bash
git submodule sync --recursive
```

The currently validated lab tree has `dlimp` and `lerobot` initialized:

```bash
git submodule update --init dlimp lerobot
```

Check against the stored lab reference:

```bash
git submodule status --recursive
cat reproducibility/lab_submodules.txt
```

`third_party/aloha` and `third_party/libero` are currently uninitialized
in the validated lab tree. They do not need to be initialized merely to
match the current thesis environment.

## Create Python environment

Using Miniforge/Conda:

```bash
conda create \
  -n forcevla_lab_tf_test \
  python=3.11 \
  -y

conda activate forcevla_lab_tf_test

python -m pip install \
  --upgrade \
  pip \
  setuptools \
  wheel
```

For a compatible NVIDIA Linux laptop, the ForceVLA baseline installation
uses CUDA toolkit 12.8:

```bash
conda install \
  -c nvidia \
  cuda-toolkit=12.8
```

Install ffmpeg:

```bash
conda install \
  -c conda-forge \
  ffmpeg=7.1.1
```

The laptop's NVIDIA host driver is machine-specific and must be compatible
with the laptop GPU. Do not copy the workstation driver installation
blindly.

## Install thesis source components

Install the pinned LeRobot checkout:

```bash
python -m pip install -e ./lerobot
```

Install the main ForceVLA/OpenPI project:

```bash
python -m pip install -e .
```

Install dlimp:

```bash
python -m pip install -e ./dlimp
```

Install OpenPI client:

```bash
python -m pip install -e ./packages/openpi-client
```

Install flaxformer:

```bash
python -m pip install -e ./flaxformer
```

## Compare with the validated workstation

Inspect the exact lab reference:

```bash
cat reproducibility/lab_runtime_reference.txt

cat reproducibility/lab_pip_versions.txt

cat reproducibility/lab_editable_packages.txt

cat reproducibility/lab_conda_packages.txt
```

These snapshots are primarily for verification and debugging.

Do not blindly replace a laptop-compatible CUDA/driver stack merely to
match every workstation package byte-for-byte.

## Runtime validation

Run:

```bash
python - <<'PY'
import sys
import jax

print("Python:", sys.version)
print("JAX:", jax.__version__)
print("Devices:", jax.devices())

assert sys.version_info[:2] == (3, 11)
assert jax.__version__ == "0.5.3"
PY
```

The validated lab reference uses two GPUs, but source development on the
laptop does not require two GPUs.

Training and inference capability depend on the laptop GPU and available
VRAM.

## Large artifacts are not stored in Git

The Git repository intentionally does not contain:

- OpenPI model weights
- Hugging Face caches
- training datasets
- experiment checkpoints
- generated training outputs

Those large artifacts must be transferred or downloaded separately when
needed.

## Working from both lab workstation and laptop

Before beginning work on either computer:

```bash
git status
git fetch
git pull --ff-only
```

After a validated change:

```bash
git add <files>
git diff --cached --check
git commit -m "<description>"
git push
```

Before continuing from the other computer, pull the new commit first.

Avoid keeping conflicting uncommitted versions of the same source changes
on the workstation and laptop.
