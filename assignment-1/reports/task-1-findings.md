# Task 1: Dataset analysis and training pipeline

## Audit results

- 10,464 images, all 416 x 416 pixels. All opened successfully.
- 10,464 matching label files, with no missing, empty, or orphan labels.
- 74,073 valid bounding boxes across six class IDs.
- 17 additional rows with zero-width or zero-height boxes.
- No exact image duplicates, checked by file contents and decoded pixels.
- 26,997 boxes (36.45%) cover less than 1% of an image. This is a relative-area
  threshold, not the COCO definition of a small object.
- Between 1 and 358 valid objects per image; 5,082 images contain one object.

## Class distribution

| ID | Working class name | Valid boxes | Images containing class |
| --- | --- | ---: | ---: |
| 0 | biodegradable | 45,395 | 2,287 |
| 1 | cardboard | 4,696 | 1,502 |
| 2 | glass | 7,809 | 2,684 |
| 3 | metal | 5,841 | 1,898 |
| 4 | paper | 4,387 | 1,768 |
| 5 | plastic | 5,945 | 1,464 |

Class names are inferred from filename patterns. Images can contain multiple
classes, so image counts overlap. The filename alone does not determine the
class of every object in an image.

## Visual review

We reviewed the initial preview and all 30 random sample images (five per class,
seed 42). The boxes appeared to fit the visible objects. The random sample
contained 30 unique images. No placement corrections were identified.

The images include crowded scenes, overlapping objects, small objects, and some
watermarks. The review covers box placement in these examples; it does not
confirm every class label or establish that all objects have been annotated.
Invalid rows were not drawn in the previews.

## Duplicate review and leakage prevention

The SIFT/RootSIFT loose search produced 364 unique candidate pairs with at least
15 RANSAC inliers. Every pair was manually reviewed. A pair was marked
`same_source` when it showed the same source picture, including rotation,
mirroring, cropping or color edits. Separate photographs were marked
`different_images`, even if they depicted similar objects.

Final decisions in [review.csv](duplicate-review/review.csv): **341 same-source
pairs, 23 different-image pairs, zero pending or uncertain decisions**. These
are pair counts, not counts of removable images. The 23 different pairs are:
56, 91, 173, 189, 197, 208, 220, 221, 228, 233, 235, 255, 256, 267, 273, 282,
309, 326, 328, 350, 351, 353 and 361.

Confirmed links were merged transitively into connected groups. If A matches B
and B matches C, all three stay together. There are 267 non-singleton groups;
the largest contains five images. Including singleton images gives 10,156
independent allocation groups. No confirmed duplicate group crosses subsets.
Historical teammate lists supplied reference labels only; current review
verdicts determined grouping. Unflagged images are not proven unique.

Duplicates were retained together in one subset rather than deleted. This
prevents known train/test source leakage while preserving images and labels.
Related images within a subset can still give repeated sources extra weight;
this remains a limitation of the retained dataset.

## Annotation cleaning

The original dataset remains unchanged. The prepared label copies were checked
against all original labels. Exactly 17 zero-area rows were removed, matching
[the removal log](../data/prepared/removed_rows.txt). Every valid annotation was
preserved in order, with no other annotation changes. The cleaned dataset
contains all 10,464 images and 74,073 valid boxes. Numeric validation does not
establish visual annotation completeness or correctness.

## Reproducible train/validation/test split

The target is 70/15/15 with seed 42. `prepare_split.py` allocates entire duplicate
groups, balancing image counts, per-class image presence and per-class box
counts. It uses all class IDs in an image rather than only its dominant class.
The allocation is greedy, not a guarantee of globally optimal stratification.
Rounded image-count capacities constrain the subset sizes.

| Subset | Images | Actual share | Purpose |
| --- | ---: | ---: | --- |
| Training | 7,325 | 70.002% | Learn model parameters; training augmentation only |
| Validation | 1,570 | 15.004% | Select checkpoint and tune settings |
| Test | 1,569 | 14.994% | Final evaluation after settings are fixed |

### Valid bounding boxes by class and subset

| ID | Working class name | Train | Validation | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | biodegradable | 31,772 | 6,818 | 6,805 | 45,395 |
| 1 | cardboard | 3,289 | 704 | 703 | 4,696 |
| 2 | glass | 5,476 | 1,168 | 1,165 | 7,809 |
| 3 | metal | 4,091 | 875 | 875 | 5,841 |
| 4 | paper | 3,072 | 657 | 658 | 4,387 |
| 5 | plastic | 4,161 | 892 | 892 | 5,945 |

### Images containing each class

These counts overlap because an image can contain multiple classes.

| ID | Working class name | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| 0 | biodegradable | 1,601 | 343 | 343 |
| 1 | cardboard | 1,051 | 226 | 225 |
| 2 | glass | 1,885 | 401 | 398 |
| 3 | metal | 1,328 | 285 | 285 |
| 4 | paper | 1,237 | 266 | 265 |
| 5 | plastic | 1,024 | 220 | 220 |

All six classes occur in every subset. Every image is assigned once; manifest
coverage, list consistency and confirmed duplicate isolation were checked.
The repeated split run produced the same reported counts and input fingerprints.
The fixed [manifest](../data/split/manifest.csv),
[summary](../data/split/summary.json) and filename lists record the result.
The split used NumPy 2.5.4; its version and input SHA-256 fingerprints are in
the summary. The test subset is kept fixed and is not used to select settings.

## YOLO dataset export

`export_yolo.py` uses the fixed manifest without generating new assignments.
It copies source images and verified cleaned labels into parallel directories:

```text
data/processed/
  images/train/    labels/train/
  images/val/      labels/val/
  images/test/     labels/test/
  data.yaml
  manifest.csv
  split-summary.json
  export-report.json
```

All 10,464 image/label pairs were exported. SHA-256 verification confirmed that
every copied image matches its source and every label matches its cleaned copy.
An independent check confirmed exact expected filenames in each subset and an
unchanged manifest. Existing output directories are refused to prevent stale
files from introducing cross-subset leakage. Dataset copies remain local and
are excluded from Git; code, review decisions and split manifests are versioned.

## Selected detector and preprocessing

The selected baseline is **COCO-pretrained YOLO26s (`yolo26s.pt`)**, fine-tuned
using the Ultralytics Python API. It is a practical small-model starting point
for the available Intel i7 / NVIDIA RTX 5050 system with 8 GB GPU memory.
The model's performance on this waste dataset has not yet been measured.

The initial input size is 640 pixels. The loader applies aspect-ratio-preserving
resize/letterbox and scales RGB intensities to [0, 1]; no separate dataset
mean/std normalization is added. Sources are 416 x 416, so upscaling adds no
new source detail. The rationale is to test whether a larger model input grid
helps localization of small objects; any benefit must be established using
validation. A 416-pixel comparison may be made if compute permits.

## Training-only augmentation

| Transformation | Baseline setting | Rationale |
| --- | --- | --- |
| Horizontal flip | Probability 0.5 | Vary object orientation |
| Translation | 0.1 | Reduce reliance on fixed object position |
| Scaling | 0.2 | Moderate size variation while limiting additional shrinking |
| HSV hue/saturation/value | 0.015 / 0.3 / 0.2 | Moderate color and illumination variation |
| Rotation, vertical flip, shear, perspective | Disabled | Keep the initial transformation policy simple |
| Mosaic, mixup, copy-paste | Disabled | Limit extra shrinking and synthetic crowding in the baseline |

These are baseline design choices, not augmentation results. With 36.45% of
boxes below 1% of image area and some highly crowded scenes, aggressive
transformations may damage useful small-object supervision. Training batch
plots are enabled for inspection. Validation and test use deterministic
preprocessing without random training augmentation.

## Training parameters and strategy

The implementation is [train_detector.py](../train_detector.py).

| Setting | Initial value | Reason |
| --- | --- | --- |
| Initialization | COCO-pretrained YOLO26s; fine-tune all layers | Transfer learned visual features |
| Maximum epochs | 100 | Upper training budget, not a required duration |
| Early stopping | Patience 20 | Stop after sustained lack of validation fitness improvement |
| Batch size | 4 | Conservative starting point for 8 GB VRAM; reduce to 2 if necessary |
| GPU / mixed precision | Device 0 / AMP enabled | Use available GPU while limiting memory demand |
| Data workers | 2 | Conservative Windows loading configuration |
| Seed / deterministic option | 42 / enabled | Improve repeatability; not a cross-platform identity guarantee |
| Optimizer | SGD | Explicit, documented baseline optimizer |
| Initial learning rate | 0.01 | Starting value, subject to validation-based adjustment |
| Momentum / weight decay | 0.937 / 0.0005 | Momentum optimization and regularization |
| Warmup | 3 epochs | Gradually introduce the training learning rate |
| LR schedule | Cosine; final factor 0.01 | Decrease the learning rate over training |
| Detection cap | 400 | Allow evaluation of scenes with up to 358 labeled objects |
| Training fraction | 1.0 | Use the full fixed training subset for the baseline |

The detection cap applies to prediction/validation output, not the number of
training annotations loaded. Keep `max_det=400` for final evaluation as well.
Small batches may use gradient accumulation internally; record the resolved
arguments and optimizer behavior from the installed implementation. This is a
custom fine-tuning baseline, not a reproduction of the published YOLO26 recipe.

Class 0 contributes 61.28% of valid objects. No oversampling or custom class
weighting is applied initially: first measure per-class validation performance.
Consider balancing only if those results justify a controlled follow-up.

## Model selection and evaluation design

Use the saved best checkpoint according to the installed Ultralytics validation
fitness. The exact fitness definition must be recorded with the package version;
no YOLOv5-specific weighted formula is assumed for YOLO26. Training uses the
validation subset for checkpoint selection and early stopping. Hyperparameters,
resolution and any operating confidence threshold are selected using validation.

After those choices are fixed, evaluate the best checkpoint on the untouched
test subset. Report mAP50-95, mAP50, per-class AP, precision and recall. Per-class
results are needed because aggregate scores can obscure minority-class failures.
Review small objects, overlaps and crowded scenes. Subsequent assignment tasks
will include examples of correct, missed and incorrect detections and analysis
of plausible failure causes. No test-set results are available yet.

## Reproduction and implementation evidence

From `assignment-1`, with the environment activated:

```powershell
python audit_dataset.py
python check_duplicates.py
python preview_duplicates.py
# Review pair decisions before generating the split.
python prepare_split.py
python export_yolo.py
python train_detector.py --dry-run
```

The audit command replaces its audit report. Preview generation creates a fresh
review on reruns; it is unnecessary for the completed reviewed set. Split and
export commands refuse existing outputs. Use a fresh `--output` destination
when reproducing, and explicitly pass that manifest directory to the exporter
if using an alternate split. Preserve the current completed review.

Completed verification: all seven audit/split/export unit tests passed;
full-data cleaning, split integrity and copy checks passed; the YOLO26s
configuration dry run passed. A dry run does not execute the training API or
verify GPU kernels. The local environment now uses PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128
and Ultralytics 8.4.208, pinned in `requirements-training.txt`. CUDA matrix
multiplication and torchvision CUDA NMS passed on the RTX 5050 Laptop GPU.
An initial full-training attempt failed with a CUDA illegal instruction after
21 batches. Synthetic forward/backward checks subsequently passed with and
without AMP; the original failure was not reproduced. A one-epoch real-data
smoke run on 2% of the training subset completed with AMP disabled, workers 0
and synchronous CUDA reporting, including validation. This is a tested
workaround, not proof that AMP caused the crash.

The user subsequently reported completion of the first full-data epoch:
2.54 GB reported GPU memory, 6 min 18 sec training plus 30.1 sec validation;
validation precision 0.419, recall 0.360, mAP50 0.319 and mAP50-95 0.199.
These are preliminary Task 2 validation observations, not final test results.
The selected baseline remains documented above; the current diagnostic run
overrides AMP to disabled and workers to 0. Record those overrides with the
experiment. Full-training completion and final evaluation remain outstanding.

Run tests from the repository root:

```powershell
python -m unittest discover -s assignment-1/tests -v
```

## Scope and limitations

This document records Task 1's dataset analysis and designed training pipeline.
Full training and quantitative model evaluation belong to Task 2. The sampled
annotation review does not certify the entire dataset. Near-duplicate detection
may miss related sources, and retained repeated sources may affect weighting.
Working class names are inferred; an authoritative mapping and the original
dataset download URL are not recorded in the repository. The assignment asks
for an approach covered in lectures; lecture coverage has not been independently
verified here. No accuracy, speed or memory claim is made for an unrun model.
Individual group contribution statements should identify the actual members
responsible for these activities; names have not been supplied in this record.

## References and supporting files

- [Raw dataset audit](audit.json)
- [Duplicate review instructions](duplicate-review/README.txt) and [decisions](duplicate-review/review.csv)
- [Cleaned label verification notes](../data/prepared/README.md)
- [Split/export documentation](../data/split-README.md)
- [Official YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26)
- [Official training argument documentation](https://docs.ultralytics.com/modes/train)
