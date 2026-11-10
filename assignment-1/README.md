# Assignment 1: Waste Object Detection

The dataset audit, visual review and duplicate pair review are complete.
The complete [Task 1 report](reports/task-1-findings.md) documents the analysis,
cleaning, verified split/export and selected YOLO26s training pipeline.
The reproducible split workflow verifies cleaned labels before assigning data.
Model training has begun; final evaluation is pending.

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
reviews are preserved. All 364 candidate pairs have been reviewed: 341 are
`same_source` and 23 are `different_images`.

The current [pair review](reports/duplicate-review/README.txt) includes 364
previews and a CSV checklist. Edit decision to same_source, different_images,
or uncertain. The current review is complete; historical reference status is not a verdict.
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

## YOLO26s training baseline

The selected detector is COCO-pretrained `yolo26s.pt`, fine-tuned using the
Ultralytics Python API. Confirm YOLO-family lecture coverage before submission.
The fixed split and verified cleaned labels in `data/processed/` are used.

From `assignment-1`, inspect the settings without training:

```powershell
python train_detector.py --dry-run
```

Install the pinned training packages with
`python -m pip install -r requirements-training.txt` from `assignment-1`.
The local Python 3.14 environment uses PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128 and Ultralytics 8.4.208. CUDA availability, GPU matrix
multiplication and torchvision CUDA NMS were verified on the RTX 5050 Laptop GPU.
Then run a short training check:

```powershell
python train_detector.py --epochs 1 --fraction 0.02 --name yolo26s_smoke
```

For a CUDA illegal-instruction failure, use a fresh Python process and a short
diagnostic run: `python train_detector.py --epochs 1 --fraction 0.02 --no-amp
--cuda-debug --workers 0 --name yolo26s_cuda_check`. This disables mixed
precision and reports CUDA operations synchronously; it is a diagnostic, not
proof that mixed precision caused the failure. Record any changed training
settings with the experiment.

Then inspect the augmented training batch plots before the baseline:

```powershell
python train_detector.py
```

Baseline: 100 epochs maximum, patience 20, seed 42, 640-pixel inputs, batch 4,
GPU 0, AMP enabled and two Windows data workers. The RTX 5050 has 8 GB VRAM;
reduce batch to 2 if crowded scenes exhaust memory. Training speed and peak
memory have not yet been measured. Compare 416 using validation if needed;
upscaling adds no source detail and its accuracy benefit must be measured.
The loader handles resize/letterbox and RGB scaling to [0, 1].

Use SGD with initial LR 0.01, momentum 0.937, weight decay 0.0005, three warmup
epochs, cosine LR decay and final LR factor 0.01. These are baseline choices,
not tuned results or a reproduction of the published YOLO26 training recipe.
Ultralytics records resolved arguments and logs in `runs/train/`; record the
exact package versions, GPU information and initialization weight hash with
experimental results. Small batches may use gradient accumulation internally.

Training-only augmentation: horizontal flips 0.5, translation 0.1, scaling 0.2,
and HSV gains 0.015/0.3/0.2. Rotation, vertical flips, shear, perspective,
mosaic, mixup and copy-paste are disabled for the initial baseline to limit
additional shrinking of small objects. Inspect the resulting boxes before
full training. Validation and test use deterministic preprocessing.

The detection cap is 400 for crowded images (up to 358 annotated objects).
Use the saved best validation checkpoint according to the installed
implementation's fitness; record that criterion with the pinned version.
Use validation for all configuration and confidence-threshold choices.
Report final test mAP50-95, mAP50, per-class AP, precision and recall with
`max_det=400`. Test evaluation is separate from this training script. Review
correct, missed and incorrect detections, including small and crowded objects.
No oversampling initially; consider balancing only if validation supports it.
Class names remain the provisional repository mapping.

References: [YOLO26](https://docs.ultralytics.com/models/yolo26) and
[training arguments](https://docs.ultralytics.com/modes/train).
The CUDA environment and a short full-precision training/validation smoke run
have been verified. After an intermittent CUDA illegal-instruction failure,
the current full-data run uses `--no-amp --workers 0 --cuda-debug`; the user has
reported successful completion of epoch 1. Final training results and test
metrics are pending. See the Task 1 report for the diagnostic record.

## Tests

```powershell
python -m unittest discover -s assignment-1/tests -v
```

## Next steps

Run `python assignment-1/prepare_split.py` from the repository root to verify
the supplied cleaned labels against originals and the removal log, then produce
a group-aware 70/15/15 split with seed 42. Confirmed duplicate groups stay in
one subset. The script refuses to overwrite existing output; use `--output`
for a fresh destination. See [split outputs and training setup](data/split-README.md).

Run `python assignment-1/export_yolo.py` from the repository root to copy the
fixed split with cleaned labels into `data/processed/` and generate `data.yaml`.
Every copied image and label is checked against its source. Existing output
folders are refused. See the split documentation linked above for details.

Next: confirm official class names and lecture coverage, inspect available
hardware, pin YOLO26-compatible Ultralytics and dependencies, save the runnable hyperparameters and
launch command, then inspect augmented batches and run a training smoke check.
