# YOLOv11 from Scratch — PyTorch

A clean PyTorch implementation of YOLOv11-nano built from scratch, without relying on Ultralytics. Includes the full training pipeline: C3K2 + C2PSA backbone, FPN neck, DFL head, TAL assignment, mosaic/MixUp augmentation, EMA, and AMP training.

## Architecture

```
Input → Backbone (C3K2 + SPPF + C2PSA) → Neck (FPN) → Head (DFL) → Detections
```

| Component | Details |
|-----------|---------|
| **Backbone** | Conv stem → C3K2 blocks (P2–P4) → SPPF → C2PSA (P5) |
| **Neck** | FPN with upsample top-down path + downsample bottom-up path |
| **Head** | Decoupled box (DFL, 16 bins) + classification branch |
| **Assignment** | Task-Aligned Learning (TAL) with top-k anchor selection |

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Dataset

The dataset must follow standard YOLO format:

```
<dataset>/
├── train/
│   ├── images/       # *.jpg
│   └── labels/       # *.txt — one line per object: <class> <cx> <cy> <w> <h> (normalized)
└── valid/
    ├── images/
    └── labels/
```

Place the dataset folder in the project root. Class names are declared in `utils/args.yaml` under the `names` field.

## Training

```bash
# Single GPU
python main.py --train --epochs 600 --batch-size 32 --input-size 640

# Multi-GPU
bash main.sh <num_gpus> --train --epochs 600 --batch-size 32 --input-size 640
```

Training logs (loss, mAP, precision, recall) are written to `weights/step.csv` each epoch.

## Validation

```bash
python main.py --test --input-size 640
```

Prints precision, recall, mAP@0.5, and mAP@0.5:0.95 against `best.pt`.

## Export to ONNX

```python
from utils.util import export_onnx
import argparse

export_onnx(argparse.Namespace(input_size=640))
# Output: weights/best.onnx
```

## Project Structure

```
├── main.py              # Training & validation entry point
├── main.sh              # Multi-GPU launcher (torchrun)
├── requirements.txt
├── nets/
│   └── nn.py            # Backbone, Neck, Head, YOLO, DFL, C3K2, C2PSA, SPPF, Attention
└── utils/
    ├── args.yaml        # Hyperparameters & class names
    ├── dataset.py       # Dataset loader, mosaic, MixUp, HSV, random perspective
    └── util.py          # Loss (BoxLoss, QFL, VFL), metrics, EMA, LR schedulers
```

## Hyperparameters

All hyperparameters are configured in `utils/args.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_lr` | `0.01` | Peak learning rate |
| `min_lr` | `0.0001` | Minimum learning rate (warmup start & cosine end) |
| `warmup_epochs` | `3` | Number of linear warmup epochs |
| `box` | `7.5` | Box regression loss weight |
| `cls` | `0.5` | Classification loss weight |
| `dfl` | `1.5` | Distribution focal loss weight |
| `mosaic` | `1.0` | Mosaic augmentation probability |
| `mix_up` | `0.0` | MixUp augmentation probability |
| `flip_lr` | `0.5` | Horizontal flip probability |
| `scale` | `0.5` | Random scale range (±gain) |
