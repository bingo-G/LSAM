# LSAM — Layer- and Spatially Adaptive Degradation Modeling for Transformer-based Compressed Video Quality Assessment

LSAM is a full-reference video quality assessment (VQA) model. Given a
distorted video and its pristine reference, it predicts a video quality
score.

---

## 1. Install

See [`INSTALL.md`](INSTALL.md) for the full guide. TL;DR:

```bash
conda create -n lsam python=3.10 -y
conda activate lsam
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
sudo apt install -y ffmpeg   # or: conda install -c conda-forge ffmpeg
```

Target environment: **Python 3.10 · CUDA 12.1 · PyTorch 2.1.0**.

Download the released checkpoint from Google Drive and place it in
`weights/`:

```
weights/
└── lsam.pth              # full model state_dict (~400 MB)
```

Download link: <https://drive.google.com/file/d/1kSbLw6WzfL5YHTiPAPsTfyyTKfz1uUoU/view?usp=sharing>

`lsam.pth` is a complete state_dict — no separate Perception-Encoder download
is required.

---

## 2. Quick start

### 2.1 Score a single (reference, distorted) pair

Raw YUV — pass geometry explicitly (width / height / bit depth / pixel
format):

```bash
python infer.py \
    --eval_ckpt weights/lsam.pth \
    --ref /data/refs/BasketballDrive_1920x1080_50fps_10bit.yuv \
    --dis /data/dis/HM/BasketballDrive_QP22.yuv \
    --width 1920 --height 1080 --bitdepth 10 --pixel_format yuv420p10le \
    --output outputs/result.csv
```

4K example:

```bash
python infer.py \
    --eval_ckpt weights/lsam.pth \
    --ref /data/refs/TiergartenParkway_3840x2160_60fps_10bit.yuv \
    --dis /data/dis/VTM/TiergartenParkway_QP37.yuv \
    --width 3840 --height 2160 --bitdepth 10 --pixel_format yuv420p10le \
    --output outputs/result.csv
```

Container video (mp4 / mkv / mov / y4m / webm) — geometry is auto-probed, no
manual `--width` / `--height` needed:

```bash
python infer.py --eval_ckpt weights/lsam.pth \
    --ref /data/refs/BasketballDrive.mp4 \
    --dis /data/dis/BasketballDrive_QP22.mp4 \
    --output outputs/result.csv
```

### 2.2 Batch mode — two directories

When you have parallel `ref` and `dis` directories, LSAM auto-pairs by
sequence name (the trailing `_QP<NN>` suffix on the distorted filename is
stripped to look up the reference):

```bash
python infer.py \
    --eval_ckpt weights/lsam.pth \
    --ref_dir /data/refs --dis_dir /data/dis \
    --output outputs/scores.csv
```

Pairing precedence (first hit wins):

1. **Exact stem match** — `dis: HM/BasketballDrive_QP22.yuv` → strip `_QP22`
   → `BasketballDrive` → look up `refs/BasketballDrive*` by stem.
2. **Sequence prefix match** — try `BasketballDrive` against any ref whose
   stem starts with `BasketballDrive`.

Distorted files whose reference cannot be located emit a warning and are
skipped (the count is reported at the end).

### 2.3 Batch mode — explicit manifest CSV

For full control, provide a manifest:

```csv
dis_path,ref_path,width,height,bitdepth,pixel_format,video_id,mos
/data/dis/HM/BasketballDrive_QP22.yuv,/data/refs/BasketballDrive_1920x1080_50fps_10bit.yuv,1920,1080,10,yuv420p10le,bd_qp22,9.5
/data/dis/VTM/Tango_QP37.yuv,/data/refs/Tango_3840x2160_60fps_10bit.yuv,3840,2160,10,yuv420p10le,tango_qp37,5.8
```

Required column: `dis_path`. Everything else is optional.

```bash
python infer.py --eval_ckpt weights/lsam.pth \
    --manifest pairs.csv --output outputs/scores.csv
```

### 2.4 Batch driver script

`scripts/run_batch.sh` shows one common batch-run recipe (manifest mode with
sensible defaults for `num_workers` and `batch_size`). Copy and edit for your
own layout:

```bash
bash scripts/run_batch.sh weights/lsam.pth pairs.csv outputs/scores.csv
```

### 2.5 Multi-GPU

All three modes work under `torchrun`:

```bash
torchrun --nproc_per_node=4 infer.py --manifest pairs.csv \
    --eval_ckpt weights/lsam.pth --output outputs/scores.csv
```

---

## 3. Output

`infer.py` writes a per-video CSV at `--output` (default:
`outputs/inference_results.csv`) with the following columns:

| column       | meaning                                            |
|--------------|----------------------------------------------------|
| `video_id`   | as supplied (or `basename(dis_path)`)              |
| `mos`        | from the manifest, or `0` if not supplied          |
| `pred`       | model quality prediction                           |
| `height` / `width` | resolution metadata                          |

---

## 4. Repository layout

```
LSAM/
├── README.md                  # this file
├── INSTALL.md                 # install guide
├── requirements.txt
├── infer.py                   # main CLI (single / dir-pair / manifest)
├── configs/
│   └── lsam.yaml              # released model configuration
├── scripts/
│   └── run_batch.sh           # example batch runner
├── src/                       # data pipeline, models, engine, utils
├── perception_models/         # Meta Perception Encoder (own LICENSE)
├── weights/                   # drop lsam.pth here (see weights/README.md)
└── outputs/                   # default output directory
```

---

## 5. License and acknowledgements

* The LSAM code in this repository is released for research use.
* `perception_models/` bundles Meta's Perception Encoder implementation and
  is governed by its own `LICENSE.PE` / `LICENSE.PLM` files.
