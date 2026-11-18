# Assignment 1 Task 2 — Model training and evaluation

**Final model:** YOLO26s trained with the stronger recipe below,
`models/yolo26s-strongaug-best.pt`. Test mAP50–95 **51.8%** (95% interval
48.7% – 56.0%), 51.9% without the leaked duplicates. The first YOLO26s run
and YOLOv5s are kept as baselines in the sections after this one.

## Final model: stronger training recipe

The baseline (next section) trained with batch 4 and mild augmentation without
mosaic. Task 1 later showed that mosaic does not shrink objects, so the reason
for leaving it off no longer held. We trained YOLO26s once more from the same
COCO checkpoint, on the same split, with a stronger recipe, and compared the
two runs on validation only.

| Setting | Baseline | Final model |
| --- | --- | --- |
| Batch size | 4 | 16 |
| Mosaic | off | 1.0, switched off for the last 10 epochs |
| Scale | 0.2 | 0.5 |
| Vertical flip, rotation | off | 0.5, 10 degrees |
| HSV saturation / value | 0.3 / 0.2 | 0.7 / 0.4 |
| Everything else | 60 epochs, SGD, lr 0.01, cosine decay, 640 px, seed 42, `max_det` 400 | same |
| Hardware | RTX 5050 laptop GPU (CUDA) | Apple M4 Pro GPU (MPS) |

Several settings changed at once, so the gain below belongs to the recipe as a
whole; we did not measure the share of each setting. The run was stopped three
times by the machine (one memory kill after epoch 17 and two MPS indexing
errors after epochs 34 and 47) and each time resumed from its last checkpoint
with Ultralytics' `resume`, which restores the weights, optimizer and learning
rate schedule; the epoch in progress was repeated. The full log is in
[strong-augmentation-run/](strong-augmentation-run/) (`results.csv`,
`args.yaml`, curves).

**Validation (1,570 images), used for the decision:**

| Run | Best epoch | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 57 of 60 | 71.5% | 56.6% | 62.4% | 44.2% |
| **Stronger recipe** | **50 of 60** | **79.6%** | **63.0%** | **72.6%** | **51.2%** |

Validation mAP50–95 rose from 31.1% (epoch 10) to 39.3% (20), 45.7% (30),
49.4% (40) and 51.2% (50), and stayed between 50.5% and 51.0% for epochs 51
to 60: the model had converged, so longer training would not help.

**Test (1,569 images, 11,098 boxes), evaluated once after the decision:**

| Run | Precision | Recall | mAP50 | mAP50–95 | 95% interval of mAP50–95 |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 74.9% | 54.6% | 61.9% | 44.5% | 41.6% – 48.8% |
| **Final model** | **78.6%** | **63.1%** | **72.3%** | **51.8%** | **48.7% – 56.0%** |
| Final model, leaked duplicates removed (1,546 images; [clean-test-final.json](clean-test-final.json)) | 78.8% | 63.2% | 72.3% | 51.9% | |
| Final model, only the 23 leaked images | 89.2% | 77.0% | 87.8% | 75.0% | |

| Class | Test boxes | Baseline AP50–95 (M4 Pro re-run) | Final AP50–95 | 95% interval (final) | Change |
| --- | ---: | ---: | ---: | --- | ---: |
| Biodegradable | 6,805 | 31.4% | 37.7% | 33.4% – 42.5% | +6.3 |
| Cardboard | 703 | 42.9% | 52.8% | 45.2% – 62.0% | +10.0 |
| Glass | 1,165 | 55.0% | 60.1% | 49.6% – 72.0% | +5.1 |
| Metal | 875 | 50.8% | 57.9% | 47.3% – 67.9% | +7.1 |
| Paper | 658 | 49.3% | 55.8% | 46.4% – 65.9% | +6.5 |
| Plastic | 892 | 37.7% | 46.5% | 39.5% – 55.8% | +8.8 |

- **Every metric and every class improved.** Recall gained most (+8.5
  points): the stronger recipe finds more objects, and precision rose as well
  (+3.7), so the gain is not a trade of false alarms for hits.
- **The gain is far larger than the sampling error.** A paired bootstrap
  ([paired_bootstrap.py](paired_bootstrap.py), 1,000 resamples of the same
  test images for both models, [paired-bootstrap.json](paired-bootstrap.json))
  puts the difference at **+7.3 points mAP50–95 (95% interval +6.0 to +8.5)**;
  the final model was ahead in all 1,000 resamples. The two separate intervals
  above nearly touch only because each also contains image-to-image variation,
  which cancels when both models are scored on the same images.

  | Difference final − baseline | Value | 95% interval |
  | --- | ---: | --- |
  | Precision | +3.7 | +0.9 to +6.4 |
  | Recall | +8.5 | +6.3 to +10.4 |
  | mAP50 | +10.3 | +8.4 to +12.1 |
  | mAP50–95 | +7.3 | +6.0 to +8.5 |
  | AP50–95 biodegradable / cardboard / glass | +6.3 / +10.0 / +5.1 | +5.0 to +7.9 / +6.6 to +13.4 / +2.5 to +7.4 |
  | AP50–95 metal / paper / plastic | +7.1 / +6.5 / +8.8 | +4.0 to +10.4 / +3.6 to +9.7 / +5.7 to +11.5 |

  Every class improved significantly.
- **Validation and test agree** (51.2% and 51.8%), as for the baseline.
- **The leaked duplicates still do not inflate the score:** removing them
  changes mAP50–95 from 51.8% to 51.9%, although the leaked images alone are
  much easier (75.0%).
- **A third duplicate search, in all 8 orientations,** found 6 more test
  images with a copy (see [below](#search-for-rotated-and-flipped-copies)).
  Removing all 29 known leaked images still gives **51.9%**
  ([clean-test-extended.json](clean-test-extended.json)).
- **Biodegradable stays the weakest class**, its interval below every other
  class's point estimate: small objects in crowded piles remain the main
  limitation (Task 3).
- Speed: 5.8 ms inference plus 11.6 ms postprocessing per image on the Apple
  M4 Pro GPU (640 px).

Reproduce from the repository root (see the [root README](../../README.md#reproduce-assignment-1)
for the training command):

```powershell
python assignment-1/task-2/evaluate_clean_test.py --device 0
python assignment-1/task-2/bootstrap_test_ci.py --device 0
```

Both models' per-class values come from the same Apple M4 Pro pipeline
(`bootstrap_test_ci.py`), so the changes are computed before rounding; the
baseline's original RTX 5050 values differ by at most 0.2 points (see below).
The leak-free rows are in [clean-test-final.json](clean-test-final.json)
(`evaluate_clean_test.py` writes `runs/clean-test/results.json`; this copy was
made from that output).

Both scripts now default to the final model; pass
`--weights assignment-1/models/yolo26s-best.pt` for the baseline. The
baseline's intervals are kept in
[test-confidence-intervals-baseline.json](test-confidence-intervals-baseline.json).

## Search for rotated and flipped copies

The two earlier duplicate searches compared images mostly in their stored
orientation. [orientation_duplicate_check.py](orientation_duplicate_check.py)
hashes every test image in all 8 orientations (4 rotations, each also
mirrored) with a 64-bit perceptual hash, compares it with all 8,895 training
and validation images (at most 10 differing bits), and keeps a pair only if
the two images also correlate strongly in the matching orientation
(normalised cross-correlation >= 0.90). Every new pair was then checked by eye.

- 16 test images matched. 5 were already in the leaked list; 11 were new.
- Of the 11 new ones, **6 are copies** (1 confirmed: the same crushed can,
  rotated; 5 likely: recoloured can and bottle mock-ups, the same bottle with a
  label added, the same box photo with a different print, the same cardboard
  texture photo) and 5 were different bottles that only look alike on a white
  background. The 6 are added in
  [leaked-test-images-extended.csv](leaked-test-images-extended.csv) (29 images).
- The hash missed 18 of the 23 known copies, which are cropped or recoloured;
  each search catches a different kind of copy, so all three are needed.
- Final model, test mAP50–95: full split 51.8%; without all 29 leaked images
  (1,540 images) **51.9%**; on the 29 leaked images alone 70.1%. The leak does
  not inflate the reported score.

Raw matches are in [orientation-duplicate-check.json](orientation-duplicate-check.json).

## Baseline runs: YOLO26s and YOLOv5s

Both baseline detectors used the fixed split of 7,325 training, 1,570 validation and
1,569 test images. Both started from their respective COCO-pretrained small
checkpoints and trained at 640 pixels with batch size 4 and SGD.

| Model | Epochs logged | Best logged epoch | Validation precision | Validation recall | Validation mAP50 | Validation mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26s | 60 | 57 of 60 | 71.45% | 56.59% | 62.35% | 44.20% |
| YOLOv5s | 58 | 58 of 60 | 65.71% | 49.64% | 54.37% | 35.71% |

YOLOv5 indexes epochs from zero, so its CSV epoch 57 is the 58th epoch.
The run ended with a CUDA illegal memory access after logging 58 epochs.
Its saved best checkpoint is from the best logged epoch by YOLOv5 fitness
(10% mAP50 plus 90% mAP50–95). YOLO26s selected by mAP50–95. Its separate
AdamW refinement reached 43.7% validation mAP50–95, so the original SGD
checkpoint was retained. YOLO26s led YOLOv5s by 8.48 percentage points
in validation mAP50–95.

Both selected checkpoints are submitted in [`assignment-1/models/`](../models/README.md) with their SHA-256 checksums:

- YOLO26s: `models/yolo26s-best.pt` (training run `yolo26s_baseline-2`).
- YOLOv5s: `models/yolov5s-best.pt` (training run `yolov5s_20261113_135834_037492`).

### Held-out test results (baselines)

Both checkpoints were evaluated on the same 1,569 test images and 11,098
annotations, with 640-pixel inputs and up to 400 detections per image.
Each model used its native evaluator with default confidence and
postprocessing settings. Scores use the precision printed by the evaluators.

| Model | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO26s | 74.9% | 54.6% | 61.9% | 44.5% |
| YOLOv5s | 67.7% | 48.9% | 55.0% | 36.5% |

| Class | Test boxes | YOLO26s AP50 | YOLO26s AP50–95 | YOLOv5s AP50 | YOLOv5s AP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Biodegradable | 6,805 | 56.9% | 31.4% | 51.5% | 27.8% |
| Cardboard | 703 | 57.9% | 42.9% | 54.3% | 37.5% |
| Glass | 1,165 | 71.5% | 55.0% | 65.1% | 46.0% |
| Metal | 875 | 68.4% | 51.0% | 60.8% | 41.0% |
| Paper | 658 | 61.3% | 49.2% | 49.4% | 36.5% |
| Plastic | 892 | 55.4% | 37.7% | 48.7% | 30.4% |

Validation and test agree closely for YOLO26s (mAP50–95 44.2% and 44.5%,
mAP50 62.35% and 61.9%). The model generalises to unseen images, and the
test split is neither easier nor harder than the validation split it was
selected on.

YOLO26s led by 8.0 percentage points in test mAP50–95 and achieved
higher AP50–95 in all six classes. Glass was strongest for both models.
Biodegradable waste had the lowest AP50–95 in both despite having the
most annotations. Its lower scores under stricter overlap criteria warrant
inspecting box localization and missed detections in Task 3. Plastic also
scored relatively low. YOLO26s was selected on validation and remains the
model the stronger recipe above started from.

### Detection limit for crowded images

The densest test image has 320 annotated objects (validation: 311), above
Ultralytics' default limit of 300 detections. YOLO26 also offers an
NMS-free end-to-end head, which could cap the output. We checked both on the
submitted checkpoint with Ultralytics 8.4.208: the checkpoint uses the
standard head with NMS, the number of returned boxes follows `max_det`
(300, 400 or 1,000 boxes on the densest image), and both heads give
identical test results. Raising the limit from 400 to 500 changes test
mAP50–95 from 44.51% to 44.54% (mAP50 61.92% to 62.01%), so the 400 limit
does not hold the model back. Only one test image (and one validation
image) has more than 300 objects.

### Uncertainty of the test scores (baseline)

The test split is one sample of 1,569 images, so every score above has
sampling error. We estimated 95% confidence intervals for YOLO26s by a
bootstrap over test images: draw 1,569 images with replacement, recompute
the metrics with the Ultralytics evaluator's own matching and AP code,
repeat 1,000 times (seed 42), and take the 2.5th and 97.5th percentiles.
Whole images are resampled, not boxes, because objects in one photo share
lighting, background and crowding.

| YOLO26s on test | Value | 95% interval |
| --- | ---: | --- |
| Precision | 74.8% | 72.2% – 77.6% |
| Recall | 54.6% | 51.2% – 59.4% |
| mAP50 | 61.9% | 58.7% – 66.7% |
| **mAP50–95** | **44.5%** | **41.6% – 48.8%** |

| Class | AP50–95 | 95% interval | Test images with the class |
| --- | ---: | --- | ---: |
| Biodegradable | 31.4% | 27.3% – 35.9% | 343 |
| Cardboard | 42.8% | 35.3% – 52.4% | 225 |
| Glass | 55.0% | 43.7% – 68.1% | 398 |
| Metal | 50.8% | 41.4% – 60.2% | 285 |
| Paper | 49.3% | 41.1% – 59.2% | 265 |
| Plastic | 37.7% | 30.2% – 48.2% | 220 |

How to read this:

- **The overall score is fairly precise:** 2.9 points below to 4.3 points above
  the estimate. The interval is wider upwards, which fits a test set where a
  minority of crowded images holds most of the boxes and pulls the score down.
- **Per-class intervals are wide** (8 to 25 points). Classes whose intervals
  overlap, for example glass, metal and paper, cannot be ranked reliably from
  this test set. The one clear ranking is that **biodegradable is the weakest
  class**: its whole interval lies below every other class's point estimate.
  Glass has the highest point estimate, but its interval overlaps metal and paper.
- **YOLO26s vs YOLOv5s:** the YOLOv5s score of 36.5% lies well below the lower
  end of the YOLO26s interval (41.6%), so the 8-point gap is not a sampling
  accident. A paired bootstrap of the difference was not run, because the
  YOLOv5 v7.0 checkpoint needs the original YOLOv5 code base.
- **Leaked duplicates:** the baseline's clean-test score of 44.7% (next section) lies well
  inside the interval, consistent with the duplicates having no measurable effect.

Reproduce (all values in [test-confidence-intervals-baseline.json](test-confidence-intervals-baseline.json)):

```powershell
python assignment-1/task-2/bootstrap_test_ci.py --device 0 --weights assignment-1/models/yolo26s-best.pt --out assignment-1/task-2/test-confidence-intervals-baseline.json
```

The script first checks that its recomputed mAP50–95 equals the evaluator's
value on the full test split before resampling. Computed on an Apple M4 Pro
GPU; per-class point values are from that re-run (see the next section).

### Test score without leaked duplicates (baseline)

The split keeps the 267 duplicate groups confirmed in our pair review
together. A wider duplicate search (Assignment 2, Task 1, section 4.4) found
44 more groups that cross train, validation and test: 23 test images
(166 boxes, 1.5% of the test images) have a flipped, rotated, cropped or
recoloured copy in training or validation. They are listed in
[leaked-test-images.csv](leaked-test-images.csv) with a visual verdict
(14 confirmed, 5 likely, 2 same object in a different shot, 2 uncertain).
We re-evaluated YOLO26s with and without them; nothing was retrained.

| Test subset | Images | Boxes | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full test split (reported above) | 1,569 | 11,098 | 74.8% | 54.6% | 61.9% | 44.5% |
| **Clean: leaked images removed** | **1,546** | **10,932** | **75.4%** | **54.5%** | **62.0%** | **44.7%** |
| Only the 23 leaked images | 23 | 166 | 76.5% | 82.6% | 83.7% | 72.6% |

| Class | Full AP50–95 (re-run) | Clean AP50–95 | Change |
| --- | ---: | ---: | ---: |
| Biodegradable | 31.4% | 31.3% | −0.1 |
| Cardboard | 42.9% | 42.8% | −0.1 |
| Glass | 55.0% | 54.8% | −0.2 |
| Metal | 50.8% | 50.4% | −0.4 |
| Paper | 49.3% | 48.9% | −0.4 |
| Plastic | 37.7% | 39.9% | +2.2 |

The leaked images are much easier for the model (72.6% mAP50–95, against
44.5% overall), which is the expected sign of having seen a copy in training.
Because they are only 1.5% of the test images, removing them changes the
overall score by +0.2 points: the reported 44.5% is not inflated by the
known duplicates. Five of the six classes drop slightly (by 0.1 to 0.4
points), as expected. Plastic rises by 2.2 points because its leaked images
were hard, not easy (20.1% AP50–95 on those images, a crowded bottle pile
among them). For the baseline the clean figure is 44.7% (51.9% for the
final model, above); duplicates that neither search found may still remain.

Reproduce from the repository root after exporting the split
(`--device mps` on Apple silicon, `cpu` without a GPU):

```powershell
python assignment-1/task-2/evaluate_clean_test.py --device 0 --weights assignment-1/models/yolo26s-best.pt
```

The full-split row was recomputed on an Apple M4 Pro GPU (Python 3.14.7,
PyTorch 2.11.0, Ultralytics 8.4.208) and matches the original RTX 5050 run
to the printed precision (mAP50 61.9%, mAP50–95 44.5%; precision 74.8% here
vs 74.9%). Per-class values differ by at most 0.2 points (metal 50.8% vs
51.0%, paper 49.3% vs 49.2%), so the class table compares re-run with re-run.

The test console summaries are saved beside this report. Evaluation used
Python 3.14.4, PyTorch 2.11.0+cu128, Ultralytics 8.4.208 and an RTX 5050
Laptop GPU. The YOLOv5 evaluator was from revision
`cb2f17fd5d790cfdd0c07c25351904df9affbe6e`.
