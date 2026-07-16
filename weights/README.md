# Model weights

Download the released LSAM checkpoint from Google Drive and place it here
as `lsam.pth`:

- Download link: <https://drive.google.com/file/d/1kSbLw6WzfL5YHTiPAPsTfyyTKfz1uUoU/view?usp=sharing>

```
weights/
└── lsam.pth              # full model state_dict (~400 MB)
```

`lsam.pth` is a complete state_dict — it already contains the PE backbone
weights fused with the FR interaction head and fusion module. No separate
Perception-Encoder download is required.

The `infer.py` CLI expects the file at `weights/lsam.pth` by default
(overridable with `--eval_ckpt`).
