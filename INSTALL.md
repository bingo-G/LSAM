# LSAM Installation Guide

> Python 3.10 | CUDA 12.1 | PyTorch 2.1.0

---

## Environment Overview

| Item          | Version                     |
| ------------- | --------------------------- |
| OS            | Linux (x86_64)              |
| Python        | 3.10.19                     |
| CUDA Toolkit  | 12.1.66                     |
| cuDNN         | 8.9.7                       |
| PyTorch       | 2.1.0+cu121                 |
| torchvision   | 0.16.0+cu121                |
| torchaudio    | 2.1.0+cu121                 |
| Conda env     | `lsam`                      |

---

## Quick Install

### Step 1: Create conda environment

```bash
conda create -n lsam python=3.10 -y
conda activate lsam
```

### Step 2: Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Python dependencies

```bash
cd LSAM
pip install -r requirements.txt
```

### Step 4: Install system dependencies

```bash
# Ubuntu / Debian
sudo apt install -y ffmpeg

# Or via conda
conda install -c conda-forge ffmpeg -y
```

### Step 5: Verify

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

import numpy, cv2, av, decord, timm, einops, pandas
print('All core packages OK')
"
```

---

## One-Liner Install Script

Save the following as `install.sh` and run:

```bash
#!/bin/bash
set -e

ENV_NAME="lsam"
PYTHON_VER="3.10"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n ${ENV_NAME} python=${PYTHON_VER} -y

echo "=== Activating environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing project dependencies ==="
pip install -r requirements.txt

echo "=== Installing ffmpeg (if missing) ==="
if ! command -v ffmpeg &> /dev/null; then
    conda install -c conda-forge ffmpeg -y
fi

echo "=== Verifying installation ==="
python -c "
import torch; import torchvision; import numpy; import cv2
import av; import decord; import timm; import einops; import pandas
print('Installation successful!')
print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU available: {torch.cuda.is_available()}')
"

echo "=== Done! Activate with: conda activate ${ENV_NAME} ==="
```

---

## Dependency Overview

### Core

| Package         | Version   | Purpose                          |
| --------------- | --------- | -------------------------------- |
| torch           | 2.1.0     | Deep learning framework          |
| torchvision     | 0.16.0    | Image transforms / pretrained    |
| numpy           | 1.26.4    | Numerical computing              |
| scipy           | 1.15.3    | Scientific computing             |
| scikit-learn    | 1.3.2     | SVR / evaluation metrics         |
| pandas          | 2.3.3     | Data manipulation                |

### Video / Image I/O

| Package         | Version   | Purpose                          |
| --------------- | --------- | -------------------------------- |
| opencv-python   | 4.8.1.78  | Image read / color conversion    |
| av              | 17.0.0    | PyAV video codec                 |
| decord          | 0.6.0     | Fast video frame reading         |
| Pillow          | 10.1.0    | Image processing                 |

### Models & Utilities

| Package         | Version   | Purpose                          |
| --------------- | --------- | -------------------------------- |
| timm            | 0.9.12    | Pretrained vision backbones      |
| einops          | 0.8.2     | Tensor reshaping                 |
| huggingface_hub | 1.3.5     | Model download / HF integration  |
| safetensors     | 0.7.0     | Safe model weight format         |

### Visualization & Logging

| Package         | Version   | Purpose                          |
| --------------- | --------- | -------------------------------- |
| matplotlib      | 3.8.2     | Plotting                         |
| tensorboard     | 2.15.1    | Training visualization           |
| openpyxl        | 3.1.5     | Excel label file reading (optional) |
