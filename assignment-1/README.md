# Assignment 1: Waste Object Detection

## Current status

Task 1 is in progress: the initial dataset audit is complete and the training
pipeline is proposed. The student reviewed box placement in 30 random images.
Cleaning, splitting, and training remain. The lecturer supplied the dataset;
we use an inferred working class mapping. Near-duplicate analysis is out of scope.

## Files

```text
assignment-1/
  audit_dataset.py             Read-only dataset audit
  requirements-audit.txt      Audit dependency (Pillow)
  tests/test_audit_dataset.py  Validation and end-to-end regression checks
  reports/
    audit.json                Full measured results, including issue locations
    task-1-findings.md         Interpretation and pipeline implications
    annotated_examples.jpg    Generated local preview; excluded from Git
  GarbageDataset/             Local images/ and labels/; excluded from Git
```

## Setup and manual execution

From the repository root in PowerShell (Python 3.10 or newer):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r assignment-1/requirements-audit.txt
python assignment-1/audit_dataset.py
```

Create the environment only once. If already inside `assignment-1`, activate it
with `..\.venv\Scripts\Activate.ps1`, then run `python audit_dataset.py`.
The default dataset and output paths are relative to the script, so execution
works from either directory. If PowerShell blocks activation, you can run the
root environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe assignment-1/audit_dataset.py
```

For another dataset or output folder:

```powershell
python assignment-1/audit_dataset.py --dataset C:/path/to/dataset --output C:/path/to/reports
```

The dataset must contain `images/` and `labels/`. Labels must use one YOLO box per
line: `class_id center_x center_y width height`, with normalized coordinates.
A bounding box is the rectangle marking an object's location. Width and height
must be positive, and each rectangle must lie within the image.

The script reads all images and labels, prints progress every 500 images, and
writes `audit.json` and `annotated_examples.jpg` to the output folder. Rerunning
overwrites these reports. It does not modify the dataset, split it, or train a
model. Invalid annotations are reported and excluded from valid-box statistics;
invalid source rows are not deleted.

## Findings

See [the full findings](reports/task-1-findings.md), [raw results](reports/audit.json), and [provenance investigation](reports/dataset-provenance.md).
The scan found 10,464 images (all 416 x 416), 74,073 valid boxes across six IDs,
and 17 zero-area annotations. No missing/empty/orphan labels or exact image
duplicates were found. Class 0 accounts for 61.28% of valid boxes, and 36.45% of
valid boxes occupy less than 1% of an image's area.

Class names remain unverified. Filename prefixes are clues, not ground truth.
Exact duplicate checks do not rule out near-duplicates. Valid box coordinates
do not establish semantic correctness or annotation completeness. The preview
shows the first three images containing each ID, not a random sample. Every box
on each selected image is drawn, so the row heading does not describe every box.

## Proposed training pipeline

| Decision | Starting proposal | Reason / remaining check |
| --- | --- | --- |
| Detector | Small pretrained YOLO detector | Confirm it was covered in lectures; record exact model and package versions |
| Cleaning | Remove 17 zero-area rows in a prepared copy with a log | Preserve originals and all remaining valid annotations |
| Split | Target 70/15/15 train/validation/test, seed 42 | Assign each image once and verify all six classes appear in each split |
| Resolution | 416 baseline; compare 640 on validation if compute permits | All sources are 416 square; upscaling adds no source detail |
| Augmentation | Moderate scaling, translation, horizontal flips, color/brightness | Training only; inspect transformed boxes and avoid damaging small objects |
| Training | Up to 100 epochs, patience 20, hardware-dependent batch size | Starting settings; record hardware and adjust using validation |
| Selection | Best validation mAP50-95 checkpoint | Keep test data out of tuning and threshold selection |
| Evaluation | Test mAP50-95, mAP50, per-class AP, precision and recall | Expose class imbalance effects; review small objects and crowded scenes |

No oversampling is proposed initially. Consider balancing only if validation
results show systematic minority-class failures. Record the split manifest,
seed, dependency versions, initialization weights, and all training parameters.
These are design choices, not experimental results.

## Checks

From the repository root, with the environment active:

```powershell
python -m unittest discover -s assignment-1/tests -v
```

The checks cover malformed/zero-area/out-of-bounds/nonfinite annotations, missing
and orphan labels, corrupt images, exact duplicates, report creation, and source
preservation. A complete dataset audit is the integration check.

## Next steps

1. Remove the 17 zero-area rows in a separate prepared copy and log removals.
2. Create reproducible 70/15/15 splits with seed 42 and check class coverage.
3. Finalize the detector and training settings.

Near-duplicates were not checked, so visually related images across splits
remain a possible evaluation limitation.

## Random annotation review

The initial contact sheet has been reviewed by the student: all displayed boxes
were reported to fit visible objects. This is a placement observation only;
class correctness and dataset-wide quality are not established.

Generate a second, reproducible sample (five images per numeric class):

```powershell
python assignment-1/sample_annotations.py
```

From inside `assignment-1`, run `python sample_annotations.py` instead.
Open the numbered JPG files in `assignment-1/reports/random-review/` and record
observations in `review.csv`. Each panel shows the original on the left and
annotations on the right. The generated README explains the checklist values.
Class names remain unverified. Sampling uses seed 42; images containing multiple
classes may appear in multiple samples. The generator refuses to overwrite an
existing review directory so completed notes are preserved. For another sample,
use `--seed 43 --output assignment-1/reports/random-review/seed-43`.
Generated previews and working notes remain local and are ignored by Git;
the student-reported placement review is recorded in the findings.
