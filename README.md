# YOLOv11 from Scratch — PyTorch

YOLOv11-nano được implement từ đầu bằng PyTorch, không dùng Ultralytics. Bao gồm backbone C3K2 + C2PSA, neck FPN, head DFL, và training pipeline đầy đủ.

## Kiến trúc

```
Input → Backbone (C3K2 + SPPF + C2PSA) → Neck (FPN) → Head (DFL) → Detections
```

- **Backbone**: Conv → C3K2 (p2–p4) → SPPF → C2PSA (p5)
- **Neck**: FPN upsample + downsample path
- **Head**: Decoupled box (DFL) + class branch, TAL assignment

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị dataset

Dataset cần theo cấu trúc YOLO chuẩn:

```
food-ingredients-5/
├── train/
│   ├── images/   # *.jpg
│   └── labels/   # *.txt  (class cx cy w h, normalized)
└── valid/
    ├── images/
    └── labels/
```

Các class được khai báo trong `utils/args.yaml` (field `names`).

## Training

```bash
# Single GPU
python main.py --train --epochs 600 --batch-size 32 --input-size 640

# Multi-GPU (torchrun)
bash main.sh <num_gpus> --train --epochs 600 --batch-size 32
```

Log được ghi vào `weights/step.csv`.

## Validation

```bash
python main.py --test --input-size 640
```

Kết quả in ra precision, recall, mAP@0.5, mAP@0.5:0.95 và validation loss.

## Export ONNX

```python
from utils.util import export_onnx
import argparse

args = argparse.Namespace(input_size=640)
export_onnx(args)
# Output: weights/best.onnx
```

## Cấu trúc project

```
├── main.py              # Training & validation entry point
├── main.sh              # Multi-GPU launcher (torchrun)
├── nets/
│   └── nn.py            # Model: Backbone, Neck, Head, YOLO
├── utils/
│   ├── args.yaml        # Hyperparameters & class names
│   ├── dataset.py       # Dataset, augmentation (mosaic, MixUp, HSV, ...)
│   └── util.py          # Loss, metrics, LR scheduler, EMA, ...
└── weights/             # Checkpoints (best.pt, last.pt)
```

## Hyperparameters

Chỉnh trong `utils/args.yaml`:

| Param | Default | Mô tả |
|-------|---------|-------|
| `max_lr` | 0.01 | Learning rate tối đa |
| `min_lr` | 0.0001 | Learning rate tối thiểu |
| `epochs` | 600 | Số epoch |
| `box` | 7.5 | Box loss weight |
| `cls` | 0.5 | Classification loss weight |
| `dfl` | 1.5 | DFL loss weight |
| `mosaic` | 1.0 | Xác suất mosaic augmentation |
| `mix_up` | 0.0 | Xác suất MixUp augmentation |
