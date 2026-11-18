# Assignment 1: Waste Object Detection

The dataset audit, visual review and duplicate pair review are complete.
The complete [Task 1 report](task-1/README.md) documents the analysis,
cleaning, verified split/export and selected YOLO26s training pipeline.
The reproducible split workflow verifies cleaned labels before assigning data.
The 60-epoch baseline is complete. Best logged validation mAP50 is 62.35% and
mAP50-95 is 44.20% (epoch 57). A second run with a stronger recipe (batch 16,
mosaic, stronger augmentation) reached 72.6% and 51.2% (epoch 50) and is the
final model (test mAP50-95 51.8%; [Task 2](task-2/README.md#final-model-stronger-training-recipe)). See the [training review](task-2/training-review.md)
for the Task 1 assessment and runnable improvement experiments. The [held-out test results](task-2/README.md) and [visual error analysis](task-3/README.md) are available.

## Setup

Run from the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r assignment-1/requirements-audit.txt
```

Create the environment only once. From inside `assignment-1`, activate it with
`..\.venv\Scripts\Activate.ps1`.

## Dataset audit

Place the dataset in `assignment-1/GarbageDataset/`, with `images/` and `labels/`
subfolders. Labels use YOLO format:

```text
class_id center_x center_y width height
```

Coordinates are fractions of the image dimensions. Run:

```powershell
python assignment-1/audit_dataset.py
```

The script checks image decoding, image-label pairs, box coordinates, class
counts, image sizes, and exact duplicates. It saves `reports/audit.json` and
`reports/annotated_examples.jpg`. Rerunning replaces those reports. Source
images and labels are not changed.

Custom paths are supported with `--dataset` and `--output`.

## Annotation review

```powershell
python assignment-1/sample_annotations.py
```

This selects five images per class with seed 42 and saves side-by-side previews
and `review.csv` in `reports/random-review/`. Images containing several classes
can appear in more than one sample. Existing review folders are not overwritten;
use `--output` to choose another folder.

We inspected all 30 sampled images and found that the boxes fit the visible
objects. This checks placement in the sample, not every annotation in the dataset.
Generated image previews and the working checklist are excluded from Git.

See [findings](task-1/README.md) and [raw results](reports/audit.json).

## Duplicate checking

We use the teammate's pipeline in `duplicates/passC/`, with local path setup
and a Windows launcher. Run `python check_duplicates.py` from `assignment-1`.
See [script order, outputs and portability changes](duplicates/README.md).
The matching algorithm and thresholds are unchanged. Status labels reference
the supplied v3 CSVs rather than the missing historical baseline.
Our earlier duplicate checker is archived locally. Existing data and completed
reviews are preserved. All 364 candidate pairs have been reviewed: 341 are
`same_source` and 23 are `different_images`.

The current [pair review](reports/duplicate-review/README.txt) includes 364
previews and a CSV checklist. Edit decision to same_source, different_images,
or uncertain. The current review is complete; historical reference status is not a verdict.
Run `python preview_duplicates.py` only if generating a fresh review is needed.
Dataset images and feature caches are local; this specific preview set is versioned.

## Structure

Every script, in pipeline order, with what it reads and writes. Paths are
relative to `assignment-1/`; run commands from the repository root (see the
[reproduction table](../README.md#reproduce-from-the-repository-root)).

| Stage | Script | Reads | Writes |
| --- | --- | --- | --- |
| Task 1 · audit | `audit_dataset.py` | `GarbageDataset/` | `reports/audit.json` |
| Task 1 · visual check | `sample_annotations.py` | `GarbageDataset/` | local preview images and a review checklist |
| Task 1 · duplicates | `check_duplicates.py` → `duplicates/passC/c1`–`c5` ([stages](duplicates/README.md)) | `GarbageDataset/images/` | `reports/team-duplicates-run*/verified2.csv`, `verified3.csv` |
| Task 1 · duplicate review | `preview_duplicates.py` | latest `verified3.csv` | `reports/duplicate-review/` (`review.csv`, previews) |
| Task 1 · label check | `check_name_label_mismatch.py` | `data/prepared/labels/`, `data/split/manifest.csv` | `reports/name-label-mismatches.csv` |
| Task 1 · cleaned labels | `rebuild_prepared_labels.py` | `GarbageDataset/labels/`, `data/prepared/removed_rows.txt` | `data/prepared/labels/` |
| Task 1 · split | `prepare_split.py` | `GarbageDataset/`, `data/prepared/`, `review.csv` | `data/split/manifest.csv`, `summary.json` |
| Task 1 · export | `export_yolo.py` | `data/split/`, `GarbageDataset/images/`, `data/prepared/labels/` | `data/processed/` with `data.yaml` |
| Task 2 · train YOLO26s | `train_detector.py` | `data/processed/data.yaml`, `yolo26s.pt` | `runs/train/<name>/` (selected copy: `models/yolo26s-strongaug-best.pt`) |
| Task 2 · train YOLOv5s | `yolov5/train.py` (+ `worker.py`) | `data/processed/`, YOLOv5 v7.0 code | `yolov5/runs/<run>/` (submitted copy: `models/yolov5s-best.pt`) |
| Task 2 · full and filtered test | `task-2/evaluate_clean_test.py` | `models/`, `data/processed/`, `task-2/results/leaked-test-images-extended.csv` | `../runs/clean-test/results.json` |
| Task 2 · confidence intervals | `task-2/bootstrap_test_ci.py` | `models/`, `data/processed/data.yaml` | `task-2/results/test-confidence-intervals.json` |
| Task 3 · threshold | `task-3/select_threshold.py` | `models/`, validation split | `task-3/results/metrics/threshold_selection_val.csv` |
| Task 3 · examples and errors | `task-3/error_analysis.py` | `models/`, test split | `../runs/task3/` (checked-in copy: `task-3/results/`) |
| Task 3 · breakdowns | `task-3/breakdowns.py` | `models/`, test split | `task-3/results/metrics/breakdowns_test.csv`, `breakdowns_test.png` |
| Task 3 · error types | `task-3/error_types.py` | `models/`, test split | `task-3/results/metrics/error_types_test.csv` |
| Tests | `tests/` | small synthetic files | — |

Folders: `GarbageDataset/` (original data, local only), `data/` (split,
duplicate lists, removal log), `models/` (submitted checkpoints), `reports/`
(audit, duplicate review, label check), `task-1/` to `task-3/` (reports and
their scripts), `duplicates/passC/` (duplicate-search stages).

Datasets, caches, old/generated review runs and local archives are excluded from Git.
The current 364-pair preview set, code, supplied result CSVs and findings are versioned.

## Final training and evaluation

The selected checkpoint is `models/yolo26s-strongaug-best.pt`. It uses batch 16,
mosaic, stronger colour and scale augmentation, vertical flips and rotation.
The [Task 1 report](task-1/README.md#training-only-augmentation) explains the choices,
and [Task 2](task-2/README.md#final-model-stronger-training-recipe) records the run.

See the [root reproduction commands](../README.md#reproduce-from-the-repository-root)
for setup, export, evaluation and error analysis, and the
[final training command](../README.md#repeat-the-final-training-recipe) to train
again. All commands there run from the repository root.

## Tests

```powershell
python -m unittest discover -s assignment-1/tests -v
```

## Training history

The [training history](task-2/training-review.md) records the original baseline,
the AdamW refinement and the selected stronger recipe. Earlier checkpoints are
retained for reproducibility in [models/](models/README.md).
