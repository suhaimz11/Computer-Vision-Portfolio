# Assignment 1 Task 2 — Model training and evaluation

Both detectors used the fixed split of 7,325 training, 1,570 validation and
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

## Held-out test results

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
model to use for the final error analysis.

## Detection limit for crowded images

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

## Uncertainty of the test scores

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
- **Leaked duplicates:** the clean-test score of 44.7% (next section) lies well
  inside the interval, consistent with the duplicates having no measurable effect.

Reproduce (all values in [test-confidence-intervals.json](test-confidence-intervals.json)):

```powershell
python assignment-1/task-2/bootstrap_test_ci.py --device 0
```

The script first checks that its recomputed mAP50–95 equals the evaluator's
value on the full test split before resampling. Computed on an Apple M4 Pro
GPU; per-class point values are from that re-run (see the next section).

## Test score without leaked duplicates

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
among them). The clean figure, 44.7%, is the one to quote for the final
model; duplicates that neither search found may still remain.

Reproduce from the repository root after exporting the split
(`--device mps` on Apple silicon, `cpu` without a GPU):

```powershell
python assignment-1/task-2/evaluate_clean_test.py --device 0
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
