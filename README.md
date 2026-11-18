# Computer Vision Portfolio

Object detection and theoretical deep metric learning for automated waste sorting.
The [assignment brief](WS2026_ComputerVision_Portfolio-1.pdf) defines the five tasks.

## Submitted model and results

The selected detector is COCO-pretrained YOLO26s fine-tuned with the stronger
training recipe. The checkpoint is
[assignment-1/models/yolo26s-strongaug-best.pt](assignment-1/models/yolo26s-strongaug-best.pt).
Validation selected epoch 50 of 60 at 51.2% mAP50-95.

| Test subset | Images | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full test | 1,569 | 78.6% | 63.1% | 72.3% | 51.8% |
| Excluding all 29 flagged source copies | 1,540 | 78.8% | 63.2% | 72.2% | 51.9% |

The split retains known source overlap, including validation images with training
copies. The filtered result is a sensitivity analysis without retraining.
[Task 2](assignment-1/task-2/README.md) records the exclusions, uncertainty estimates,
earlier runs and evaluation protocol.

## Tasks and submission files

| Assignment | Task | Deliverable |
| --- | --- | --- |
| 1 | 1. Dataset analysis and training pipeline | [Report and implementation links](assignment-1/task-1/README.md) |
| 1 | 2. Training and evaluation | [Metrics, training log and interpretation](assignment-1/task-2/README.md) |
| 1 | 3. Visualization and error analysis | [Nine annotated examples and error analysis](assignment-1/task-3/README.md) |
| 2 | 1. Loss and sampling | [Theoretical proposal](assignment-2/task-1/README.md) |
| 2 | 2. Embedding analysis and use | [Retrieval, classification and evaluation proposal](assignment-2/task-2/README.md) |
| 1 and 2 | Presentation (15 min, both assignments) | [Slides as PDF](presentation/CV-Portfolio-1_Waste-Object-Detection.pdf) and [PowerPoint](presentation/CV-Portfolio-1_Waste-Object-Detection.pptx) |
| 1 and 2 | Contribution statement | [Contribution-Statement.pdf](Contribution-Statement.pdf) and the [section below](#contribution-statement) |

Model hashes and baseline
checkpoints are documented in [models/README.md](assignment-1/models/README.md).
Assignment 2 is theoretical and requires no trained embedding model.

## Project structure

```text
assignment-1/
  task-1/          Dataset report, figures and augmentation analysis
  task-2/          Training and evaluation scripts; results/ holds logs and metrics
  task-3/          Error analysis scripts, figures and prediction examples
  data/            Split manifests, duplicate groups and cleaning records
  duplicates/      Staged duplicate detection pipeline
  models/          Submitted checkpoints and provenance
  reports/         Dataset audit and completed review evidence
  tests/           Pipeline regression tests
  yolov5/          Baseline training launcher and configuration
assignment-2/
  task-1/          Metric learning loss and sampling proposal
  task-2/          Embedding analysis and application proposal
presentation/      Final slides (PDF and PowerPoint)
```

Shared Assignment 1 data preparation scripts stay at its root. Recorded results
and review evidence are versioned; source datasets, environments, caches and
new experiment runs stay local. Task reports link to their supporting artifacts.

## Setup

The dataset and virtual environment stay local. Download the course dataset from
[the course share](https://cloud.fiw.fhws.de/s/P9BGwMj7oAZrQ2R) and place its YOLO
images and labels in `assignment-1/GarbageDataset/images/` and
`assignment-1/GarbageDataset/labels/`.

Create an environment from the repository root and install the pinned Windows
CUDA dependencies.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The recorded Windows environment used Python 3.14.4 and CUDA 12.8. The final
training run used an Apple M4 Pro GPU with the same Ultralytics 8.4.208 release.
For another platform, install the matching PyTorch build first, then the audit
requirements and `ultralytics==8.4.208`. The saved checkpoint runs on CUDA,
MPS or CPU. Use `--device mps` for Apple silicon or `--device cpu` without a GPU.

## Reproduce from the repository root

| Step | Command | Output |
| --- | --- | --- |
| Tests | `python -m unittest discover -s assignment-1/tests -v` | Local regression checks |
| Dataset audit | `python assignment-1/audit_dataset.py` | `assignment-1/reports/audit.json` |
| Clean labels | `python assignment-1/rebuild_prepared_labels.py` | Verified cleaned label copies |
| Verify the recorded split | `python assignment-1/prepare_split.py --output assignment-1/data/split-check` | Fresh split for comparison with the committed manifest |
| Export | `python assignment-1/export_yolo.py` | `assignment-1/data/processed/` |
| Full and filtered test metrics | `python assignment-1/task-2/evaluate_clean_test.py --device 0` | `runs/clean-test/results.json` |
| Detection examples | `python assignment-1/task-3/error_analysis.py` | `runs/task3/predictions/` and `runs/task3/metrics/` |

Cleaning, splitting and export protect existing outputs. Reuse the already
prepared data on subsequent evaluations. The evaluator defaults to the selected
checkpoint and the extended 29-image exclusion list. Pass `--leaked` to
reproduce an earlier exclusion list.

Task 3 uses confidence 0.25, selected on validation, and IoU 0.5 matching.
Green marks dataset annotations, blue correct predictions, yellow missed
annotations and red wrong-class predictions. These fixed-threshold counts use
a different matching and operating point from the standard evaluator metrics.

## Repeat the final training recipe

Training is optional when evaluating the submitted checkpoint. Download the
COCO-pretrained `yolo26s.pt` into `assignment-1/`, then run the following from
the repository root on a GPU with sufficient memory for batch 16.

```powershell
python assignment-1/train_detector.py --device 0 --batch 16 --workers 0 --mosaic 1.0 --close-mosaic 10 --scale 0.5 --flipud 0.5 --degrees 10 --hsv-s 0.7 --hsv-v 0.4 --name yolo26s_strongaug
```

Use `--device mps` for the recorded Apple GPU configuration. The launcher
defaults remain the original baseline settings, so retain the flags above when
repeating the final recipe. Recorded arguments and all 60 training epochs are in
[strong-augmentation-run/](assignment-1/task-2/results/strong-augmentation-run/).

## Contribution statement

Also available as a one-page PDF: [Contribution-Statement.pdf](Contribution-Statement.pdf).

Suhaim and Lorena share responsibility for Assignment 1 through an equal allocation of 1.5 tasks each. Suhaim covers Task 1 and model training in Task 2. Lorena covers test evaluation in Task 2 and Task 3.

| Member | Primary responsibility | Main contributions |
| --- | --- | --- |
| Suhaim Manna | Assignment 1, Task 1 and the training part of Task 2 | Dataset analysis, annotation cleaning, duplicate handling, train/validation/test split and data export. Detector selection, preprocessing, augmentation, training settings, model training and validation-based checkpoint selection ([Task 1](assignment-1/task-1/README.md), [Task 2](assignment-1/task-2/README.md)). |
| Lorena Rama | Assignment 1, the evaluation part of Task 2 and Task 3 | Test evaluation and interpretation of precision, recall, mAP and per-class performance. Bounding-box visualizations of correct, missed and incorrect detections, analysis of observed errors, model strengths and limitations, and proposed improvements ([Task 2](assignment-1/task-2/README.md), [Task 3](assignment-1/task-3/README.md)). |
| Pavin Sumathi Palanichamy | Assignment 2, Task 1: loss function and sampling strategy | Chose and justified the triplet loss with a margin on L2-normalised crop embeddings; designed the online, class-balanced 6 x K batch sampling, the image-first crop drawing, the 24-pixel minimum crop size, the duplicate-aware triplet rules and the warm-up plus semi-hard negative mining; analysed the dataset properties behind each choice and discussed how sampling shapes training and the embedding space ([Task 1](assignment-2/task-1/README.md)). |
| Prethebha Muthukumaran | Assignment 2, Task 2: analysis and use of the learned embedding space | Designed the retrieval of similar waste objects from detector crops, the class-balanced k-nearest-neighbour assignment of unknown samples with "unknown" and "uncertain" answers, the visualisation and structural analysis of the space (PCA, t-SNE, distance and confusion matrices, silhouette), and the quantitative evaluation (Recall@K, MAP@R, macro-F1, chance levels, breakdowns and bootstrap intervals) ([Task 2](assignment-2/task-2/README.md)). Reviewed and merged the Assignment 2 Task 1 pull request. |
