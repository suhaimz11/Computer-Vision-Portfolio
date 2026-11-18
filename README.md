# Computer Vision Portfolio

The [portfolio brief](WS2026_ComputerVision_Portfolio-1.pdf) defines five tasks across two assignments.

## Assignment 1 · Waste object detection

| Task | Deliverable | Implementation and evidence |
| --- | --- | --- |
| 1 · Dataset analysis and pipeline | [Dataset and pipeline report](assignment-1/task-1/README.md) | [Audit, duplicate review, split and export instructions](assignment-1/README.md#dataset-audit) |
| 2 · Training and evaluation | [Held-out results](assignment-1/task-2/README.md) | [YOLOv5s training](assignment-1/yolov5/README.md), [YOLO26s training](assignment-1/README.md#yolo26s-training-baseline), and evaluator logs in Task 2 |
| 3 · Visualization and error analysis | [Report with nine annotated examples](assignment-1/task-3/README.md) | [Analysis script](assignment-1/task-3/error_analysis.py) and [metrics](assignment-1/task-3/results/metrics/) |

We evaluated YOLOv5s as our lecture-based baseline and compared it with YOLO26s using the same dataset split. YOLO26s achieved higher test mAP50–95, 44.5% compared with 36.5% for YOLOv5s, and was selected on validation. A second YOLO26s run with a stronger training recipe (batch 16, mosaic and stronger augmentation) raised validation mAP50–95 from 44.2% to 51.2%; evaluated once on test it reaches **51.8% mAP50–95** (51.9% without the 23 leaked duplicate images). This is the final model, used for the error analysis in Task 3. The [Task 2 report](assignment-1/task-2/README.md) documents all runs and their results.

### Run Task 3 locally

With the virtual environment active, run this from the repository root.

```powershell
python .\assignment-1\task-3\error_analysis.py
```

The script uses the prepared test split and the final YOLO26s checkpoint by default.
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
no GPU; step 5 needs a GPU (CUDA, or `--device mps` on Apple silicon); scripts
default to `--device 0`, so pass `--device mps` or `--device cpu` without CUDA; steps 6 and 7 run on CPU with
`device=cpu` / `--device cpu`, only slower.

| Step | Command | Output |
| --- | --- | --- |
| 0. Environment (once) | `py -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, `python -m pip install -r requirements.txt` | Python 3.14.4, PyTorch 2.11.0+cu128, Ultralytics 8.4.208 |
| 1. Dataset | Put the course's Garbage Detection Dataset (YOLO format) in `assignment-1/GarbageDataset/images/` and `assignment-1/GarbageDataset/labels/` | 10,464 images, 416 x 416 |
| 2. Tests and audit | `python -m unittest discover -s assignment-1/tests -v`, then `python assignment-1/audit_dataset.py` | `assignment-1/reports/audit.json` |
| 3. Cleaned labels | `python assignment-1/rebuild_prepared_labels.py` | `assignment-1/data/prepared/labels/`: the originals minus the 17 zero-area rows in `removed_rows.txt` |
| 3b. Check the split | `python assignment-1/prepare_split.py --output assignment-1/data/split-check` | Same rows as the committed `assignment-1/data/split/manifest.csv` (seed 42); only line endings may differ |
| 4. Export for YOLO | `python assignment-1/export_yolo.py` | `assignment-1/data/processed/` with `data.yaml` (train / val / test) |
| 5. Train (optional) | Download `yolo26s.pt` into `assignment-1/`, then `cd assignment-1`. Baseline: `python train_detector.py`. Final model: `python train_detector.py --device mps --batch 16 --workers 0 --mosaic 1.0 --close-mosaic 10 --scale 0.5 --flipud 0.5 --degrees 10 --hsv-s 0.7 --hsv-v 0.4 --name yolo26s_strongaug` (use `--device 0` on CUDA) | `assignment-1/runs/train/yolo26s_baseline/`, `assignment-1/runs/train/yolo26s_strongaug/` |
| 6. Test evaluation (Task 2) | `yolo detect val model=assignment-1/models/yolo26s-strongaug-best.pt data=assignment-1/data/processed/data.yaml split=test imgsz=640 max_det=400` | Test precision, recall, mAP50, mAP50-95 and per-class AP |
| 7. Error analysis (Task 3) | `python assignment-1/task-3/error_analysis.py` | `runs/task3/` with example images and metrics |

Skip step 5 to reproduce the reported results with the submitted checkpoint.
Step 6 should print test mAP50 72.3% and mAP50-95 51.8% for the final model
(44.5% with `models/yolo26s-best.pt`, the baseline; see
[Task 2](assignment-1/task-2/README.md)); small differences can come from a
different GPU or PyTorch build. The YOLOv5s baseline is trained and validated
with its own launcher, described in [yolov5/README.md](assignment-1/yolov5/README.md).

Without CUDA, install PyTorch for your platform from https://pytorch.org first,
then `python -m pip install -r assignment-1/requirements-audit.txt ultralytics==8.4.208`.

## Local setup and submission

The trained checkpoints are in [`assignment-1/models/`](assignment-1/models/README.md) (final model: YOLO26s). The dataset and virtual environment are stored locally. See the [Assignment 1 setup](assignment-1/README.md#setup) for dataset placement and dependencies. Run the local tests from the repository root with `python -m unittest discover -s assignment-1/tests -v`.

The submission also requires a short presentation covering both assignments.

## Contribution statement

Suhaim and Lorena share responsibility for Assignment 1 through an equal allocation of 1.5 tasks each. Suhaim covers Task 1 and model training in Task 2. Lorena covers test evaluation in Task 2 and Task 3.

| Member | Primary responsibility | Main contributions |
| --- | --- | --- |
| Suhaim Manna | Assignment 1, Task 1 and the training part of Task 2 | Dataset analysis, annotation cleaning, duplicate handling, train/validation/test split and data export. Detector selection, preprocessing, augmentation, training settings, model training and validation-based checkpoint selection ([Task 1](assignment-1/task-1/README.md), [Task 2](assignment-1/task-2/README.md)). |
| Lorena Rama | Assignment 1, the evaluation part of Task 2 and Task 3 | Test evaluation and interpretation of precision, recall, mAP and per-class performance. Bounding-box visualizations of correct, missed and incorrect detections, analysis of observed errors, model strengths and limitations, and proposed improvements ([Task 2](assignment-1/task-2/README.md), [Task 3](assignment-1/task-3/README.md)). |
| Pavin Sumathi Palanichamy | Assignment 2, Task 1: loss function and sampling strategy | Chose and justified the triplet loss with a margin on L2-normalised crop embeddings; designed the online, class-balanced 6 x K batch sampling, the image-first crop drawing, the 24-pixel minimum crop size, the duplicate-aware triplet rules and the warm-up plus semi-hard negative mining; analysed the dataset properties behind each choice and discussed how sampling shapes training and the embedding space ([Task 1](assignment-2/task-1/README.md)). |
| Prethebha Muthukumaran | Assignment 2, Task 2: analysis and use of the learned embedding space | Designed the retrieval of similar waste objects from detector crops, the class-balanced k-nearest-neighbour assignment of unknown samples with "unknown" and "uncertain" answers, the visualisation and structural analysis of the space (PCA, t-SNE, distance and confusion matrices, silhouette), and the quantitative evaluation (Recall@K, MAP@R, macro-F1, chance levels, breakdowns and bootstrap intervals) ([Task 2](assignment-2/task-2/README.md)). Reviewed and merged the Assignment 2 Task 1 pull request. |
