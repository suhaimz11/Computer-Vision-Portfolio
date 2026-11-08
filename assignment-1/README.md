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

## Near-duplicate scan

```powershell
python assignment-1/check_near_duplicates.py
```

Requires the dependencies in `requirements-audit.txt`. From inside assignment-1,
install them once with `python -m pip install -r requirements-audit.txt`, then
run `python check_near_duplicates.py` without extra arguments.

The read-only scan combines 64-bit DCT perceptual hashes (distance <= 6) with
DINOv2-small image embeddings (cosine similarity >= 0.90). The embedding search
compares every distinct pair in memory-bounded blocks, with no top-k cutoff.
All pairs passing either threshold are exported once, each with a preview.
This does not guarantee every near-duplicate: thresholds can miss matches and
embeddings can confuse similar objects. Both scores and detection methods are
included in the CSV. Pair numbers can change between runs; identify images by
their filenames when transferring previous review decisions.

The first run downloads pretrained weights from Hugging Face. CPU is supported
but can be slow; CUDA is used if available. Images stay local. The scan writes
embeddings, image/hash order, model details, candidate pairs and previews under
`reports/near-duplicates/`, automatically using a numbered folder if it exists.
Every candidate needs visual review. No decisions or groups are inferred from
similarity scores. The latest local run flagged 5,952 candidate pairs, including
32 flagged by both methods. Generated results and the dataset are excluded
from Git.

To review on another PC, clone the repository and place your own copy of the
dataset in `assignment-1/GarbageDataset/`. Install the dependencies and run
`python check_near_duplicates.py` from inside `assignment-1`.

Open the generated `candidates.csv` and matching numbered JPGs in `previews/`.
Pairs flagged by both methods come first. Set `decision` to `same_source`,
`different_images`, or `uncertain`, and add notes as needed. Do not change pair
numbers or filenames. Return the completed CSV to the team. Keep confirmed
related images together in the future split. No source images or labels have
been deleted.

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
