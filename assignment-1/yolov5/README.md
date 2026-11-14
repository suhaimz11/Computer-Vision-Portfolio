# Original YOLOv5s experiment

Uses the official anchor-based ultralytics/yolov5 repository, revision
`cb2f17fd5d790cfdd0c07c25351904df9affbe6e`, and official v7.0 COCO-pretrained
`yolov5s.pt`. This is not the newer anchor-free YOLOv5u model. No YOLO26 weights
are loaded or overwritten.

From the `assignment-1` PowerShell directory:

```powershell
..\.venv\Scripts\python.exe yolov5/train.py --dry-run
..\.venv\Scripts\python.exe yolov5/train.py
```

The full run uses 60 epochs, patience 20, batch 4, 640-pixel inputs, seed 42,
SGD and cosine learning-rate decay. Augmentation settings match the prior
YOLO26 baseline where applicable. YOLOv5-specific loss defaults and AutoAnchor
remain enabled. Model selection uses upstream YOLOv5 fitness (0.1*mAP50 +
0.9*mAP50-95), unlike YOLO26's mAP50-95-only fitness. All validation calls
use max_det=400 via the small worker adapter. Compare validation metrics
directly, not training loss values across architectures.
The worker also applies a one-line plotting compatibility fix: convert the
current Ultralytics Annotator's NumPy result to PIL before saving batch images.
This leaves model computation and optimization unchanged and is recorded per run.

First execution copies the existing training and validation image/label pairs
into `data/full`, preserving the split and isolating YOLOv5 label caches.
The test subset is not used during training. Do not edit this copied dataset
independently of the fixed source; recreate it deliberately if labels change.

Each invocation creates a unique timestamped folder under `runs/`, containing:
- `weights/best.pt` and `weights/last.pt`
- `results.csv`, training curves, confusion matrix and batch previews
- upstream `opt.yaml` and `hyp.yaml`
- `console.log` and `exit-status.json`
- launch settings, dataset YAML, original split manifest, environment package
  versions, source revision and initialization checkpoint SHA-256

The manifest snapshot describes the original complete split. Smoke runs use
only the first 24 sorted images of train and validation and are not meaningful
accuracy evaluations. To repeat the installation check:

```powershell
..\.venv\Scripts\python.exe yolov5/train.py --smoke --workers 0
```

Dependencies share the existing CUDA-enabled virtual environment. Missing
plotting dependencies installed for this experiment: pandas 3.1.1 and
seaborn 0.13.2. PyTorch and Ultralytics were not upgraded. Source, dataset copies,
downloaded weights and run artifacts stay local via this folder's `.gitignore`.

To recreate the upstream checkout, clone https://github.com/ultralytics/yolov5
into `vendor/yolov5` and check out the revision above. Install its requirements
without changing the project's pinned torch/torchvision/ultralytics versions.
Download https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
to `weights/yolov5s.pt`. The launcher verifies the source commit.

Reference: https://docs.ultralytics.com/yolov5/tutorials/train-custom-data/
