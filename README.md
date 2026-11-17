# Computer Vision Portfolio

The [portfolio brief](WS2026_ComputerVision_Portfolio-1.pdf) defines five tasks across two assignments.

## Assignment 1 · Waste object detection

| Task | Deliverable | Implementation and evidence |
| --- | --- | --- |
| 1 · Dataset analysis and pipeline | [Dataset and pipeline report](assignment-1/task-1/README.md) | [Audit, duplicate review, split and export instructions](assignment-1/README.md#dataset-audit) |
| 2 · Training and evaluation | [Held-out results](assignment-1/task-2/README.md) | [YOLOv5s training](assignment-1/yolov5/README.md), [YOLO26s training](assignment-1/README.md#yolo26s-training-baseline), and evaluator logs in Task 2 |
| 3 · Visualization and error analysis | [Report with nine annotated examples](assignment-1/task-3/README.md) | [Analysis script](assignment-1/task-3/error_analysis.py) and [metrics](assignment-1/task-3/results/metrics/) |

We evaluated YOLOv5s as our lecture-based baseline and compared it with YOLO26s using the same dataset split. YOLO26s achieved higher test mAP50–95, 44.5% compared with 36.5% for YOLOv5s. We selected YOLO26s on validation performance and used it for the final visualization and error analysis in Task 3. The [Task 2 report](assignment-1/task-2/README.md) documents both models and their results.

### Run Task 3 locally

With the virtual environment active, run this from the repository root.

```powershell
python .\assignment-1\task-3\error_analysis.py
```

The script uses the prepared test split and YOLO26s checkpoint by default.
It writes examples and metrics to `runs/task3/`. Add `--device cpu` to run
without CUDA. See the [Task 3 README](assignment-1/task-3/README.md) for
the checkpoint path.

## Assignment 2 · Deep metric learning

1. [Loss function and sampling strategy](assignment-2/task-1/README.md)
2. [Analysis and use of the learned embedding space](assignment-2/task-2/README.md)

Both tasks are theoretical.

## Reproduce Assignment 1

Every command runs from the repository root (PowerShell shown; on macOS or
Linux use `source .venv/bin/activate` and forward slashes). Steps 1 to 4 need
no GPU; step 5 needs a CUDA GPU (about 8 GB); steps 6 and 7 run on CPU with
`device=cpu` / `--device cpu`, only slower.

| Step | Command | Output |
| --- | --- | --- |
| 0. Environment (once) | `py -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, `python -m pip install -r requirements.txt` | Python 3.14.4, PyTorch 2.11.0+cu128, Ultralytics 8.4.208 |
| 1. Dataset | Put the course's Garbage Detection Dataset (YOLO format) in `assignment-1/GarbageDataset/images/` and `assignment-1/GarbageDataset/labels/` | 10,464 images, 416 x 416 |
| 2. Tests and audit | `python -m unittest discover -s assignment-1/tests -v`, then `python assignment-1/audit_dataset.py` | `assignment-1/reports/audit.json` |
| 3. Cleaned labels | `python assignment-1/rebuild_prepared_labels.py` | `assignment-1/data/prepared/labels/`: the originals minus the 17 zero-area rows in `removed_rows.txt` |
| 3b. Check the split | `python assignment-1/prepare_split.py --output assignment-1/data/split-check` | Same rows as the committed `assignment-1/data/split/manifest.csv` (seed 42); only line endings may differ |
| 4. Export for YOLO | `python assignment-1/export_yolo.py` | `assignment-1/data/processed/` with `data.yaml` (train / val / test) |
| 5. Train (optional) | Download `yolo26s.pt` into `assignment-1/`, then `cd assignment-1` and `python train_detector.py` | `assignment-1/runs/train/yolo26s_baseline/` |
| 6. Test evaluation (Task 2) | `yolo detect val model=assignment-1/models/yolo26s-best.pt data=assignment-1/data/processed/data.yaml split=test imgsz=640 max_det=400` | Test precision, recall, mAP50, mAP50-95 and per-class AP |
| 7. Error analysis (Task 3) | `python assignment-1/task-3/error_analysis.py` | `runs/task3/` with example images and metrics |

Skip step 5 to reproduce the reported results with the submitted checkpoint.
Step 6 should print test mAP50 61.9% and mAP50-95 44.5% (see
[Task 2](assignment-1/task-2/README.md)); small differences can come from a
different GPU or PyTorch build. The YOLOv5s baseline is trained and validated
with its own launcher, described in [yolov5/README.md](assignment-1/yolov5/README.md).

Without CUDA, install PyTorch for your platform from https://pytorch.org first,
then `python -m pip install -r assignment-1/requirements-audit.txt ultralytics==8.4.208`.

## Local setup and submission

The trained checkpoints are in [`assignment-1/models/`](assignment-1/models/README.md) (final model: YOLO26s). The dataset and virtual environment are stored locally. See the [Assignment 1 setup](assignment-1/README.md#setup) for dataset placement and dependencies. Run the local tests from the repository root with `python -m unittest discover -s assignment-1/tests -v`.

The submission also requires a short presentation covering both assignments, and a statement identifying each group member's contribution.
