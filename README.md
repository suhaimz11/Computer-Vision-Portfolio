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

## Local setup and submission

The trained checkpoints are in [`assignment-1/models/`](assignment-1/models/README.md) (final model: YOLO26s). The dataset and virtual environment are stored locally. See the [Assignment 1 setup](assignment-1/README.md#setup) for dataset placement and dependencies. Run the local tests from the repository root with `python -m unittest discover -s assignment-1/tests -v`.

The submission also requires a short presentation covering both assignments, and a statement identifying each group member's contribution.
