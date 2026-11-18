# Trained detection models

These are the submitted checkpoints for Assignment 1. All were trained on the
fixed split in `data/split/manifest.csv` (seed 42; 7,325 / 1,570 / 1,569 images)
and selected on validation; each was evaluated once on the test split.

| File | Role | Training run | Framework | Size | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| `yolo26s-strongaug-best.pt` | **Final model** (used in Tasks 2 and 3) | `yolo26s_strongaug`, best of 60 epochs (epoch 50), stronger training recipe ([Task 2](../task-2/README.md#final-model-stronger-training-recipe)) | Ultralytics 8.4.208 | 20.3 MB | `c280525fa1e38bd72a29d77fa9365a9c471fdaba7a68f62ce9897be833241672` |
| `yolo26s-best.pt` | Baseline (first recipe) | `yolo26s_baseline-2`, best of 60 epochs (epoch 57) | Ultralytics 8.4.208 | 20.3 MB | `98e5eb47f4e15bca6fd99b1402f2ab58954755e9caf7c3a27945ef50e23b2927` |
| `yolov5s-best.pt` | Lecture-based baseline for comparison | `yolov5s_20261113_135834_037492`, best logged epoch 58 | YOLOv5 v7.0 (`yolov5/`) | 56.9 MB | `f3d3d16665fb2ba6e5dabe440bbed286c74c6b34aecdba06aa8f381c83296fbb` |

Held-out test results (1,569 images, 11,098 boxes, 640-pixel input, up to
400 detections per image; details in [Task 2](../task-2/README.md)):

| Model | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| **YOLO26s, final** | **78.6%** | **63.1%** | **72.3%** | **51.8%** |
| YOLO26s, baseline | 74.9% | 54.6% | 61.9% | 44.5% |
| YOLOv5s | 67.7% | 48.9% | 55.0% | 36.5% |

Classes (ids 0 to 5): biodegradable, cardboard, glass, metal, paper, plastic.

## Verify

```powershell
Get-FileHash assignment-1/models/yolo26s-strongaug-best.pt -Algorithm SHA256
```

On macOS or Linux: `shasum -a 256 assignment-1/models/*.pt`.

## Use

YOLO26s, from the repository root:

```python
from ultralytics import YOLO
model = YOLO("assignment-1/models/yolo26s-strongaug-best.pt")
results = model.predict("path/to/image.jpg", imgsz=640, conf=0.25, max_det=400)
```

The Task 3 analysis uses this file by default:
`python assignment-1/task-3/error_analysis.py`.

The YOLOv5s checkpoint is a YOLOv5 v7.0 model and loads with the YOLOv5
code described in [yolov5/README.md](../yolov5/README.md), not with the
Ultralytics `YOLO` class.

Only load checkpoints from a trusted source: `.pt` files are pickles and can
run code when loaded.
