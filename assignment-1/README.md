# Assignment 1: Waste Object Detection

The dataset audit and visual review are complete. Cleaning, splitting, and
model training are next.

## Setup

Run from the repository root in PowerShell (Python 3.10+):

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

See [findings](reports/task-1-findings.md) and [raw results](reports/audit.json).

## Duplicate checking

We use the teammate's pipeline in `duplicates/passC/`, with local path setup
and a Windows launcher. Run `python check_duplicates.py` from `assignment-1`.
See [script order, outputs and portability changes](duplicates/README.md).
The matching algorithm and thresholds are unchanged. Status labels reference
the supplied v3 CSVs rather than the missing historical baseline.
Our earlier duplicate checker is archived locally. Existing data and completed
reviews are preserved. The Windows run completed; its 364 candidate pairs are ready for review.

The current [pair review](reports/duplicate-review/README.txt) includes 364
previews and a CSV checklist. Edit decision to same_source, different_images,
or uncertain. All entries are pending; historical reference status is not a verdict.
Run `python preview_duplicates.py` only if generating a fresh review is needed.
Dataset images and feature caches are local; this specific preview set is versioned.

## Structure

```text
assignment-1/
  audit_dataset.py          Dataset integrity and annotation checks
  sample_annotations.py     Random annotation previews
  check_duplicates.py       Run the teammate's five pipeline stages
  duplicates/passC/         Teammate's pipeline with portable path configuration
  data/team-duplicates/     Teammate's supplied result CSVs
  data/prepared/            Removal log and local cleaned label copies
  GarbageDataset/           Original dataset; local only
  reports/                  Findings and local generated reviews
  tests/                    Dataset audit tests
  local-archive/            Our superseded duplicate checker; local only
```

Datasets, caches, old/generated review runs and local archives are excluded from Git.
The current 364-pair preview set, code, supplied result CSVs and findings are versioned.

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

## Tests

```powershell
python -m unittest discover -s assignment-1/tests -v
```

## Next steps

1. Remove the 17 zero-area rows in a separate copy and save a removal log.
2. Split the cleaned data 70/15/15 with seed 42 and check class coverage.
3. Finalize the model and training settings.
