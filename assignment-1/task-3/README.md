# Task 3: Visualization and Error Analysis

## Setup

The final model (YOLO26s, selected on validation mAP50â€“95; best checkpoint
`yolo26s_baseline-2`) was run on all 1,569 test images (11,098 annotated
objects) with confidence 0.25, NMS IoU 0.6 and at most 400 detections per
image. Predictions are matched to ground truth with IoU >= 0.5 (greedy, by
confidence). Each prediction is then one of:

- **correct**: matches a ground-truth box with the right class;
- **wrong class**: matches a ground-truth box but with a different class;
- **false alarm**: matches no ground-truth box.

A ground-truth object with no matching prediction is **missed**.

**Why confidence 0.25.** The threshold was chosen on the validation split,
not on test: [select_threshold.py](select_threshold.py) predicts the 1,570
validation images once and scores thresholds from 0.05 to 0.90 with the same
matching rule as below. F1 is highest at 0.25 (precision 72.4%, recall 54.7%,
F1 62.3%) and nearly flat around it (61.9% at 0.20, 61.7% at 0.30); lower
thresholds add many false alarms, higher ones lose many objects
([full sweep](results/metrics/threshold_selection_val.csv)). The test split was
then analysed once at that threshold. The mAP values in Task 2 do not depend on
this threshold.

Colour code in the figures: green = ground truth, blue = correct,
red = wrong class, orange = false alarm, yellow = missed ground truth.

Run from the repository root after preparing the dataset. The script loads the
submitted YOLO26s checkpoint, [`assignment-1/models/yolo26s-best.pt`](../models/README.md), by default.

```powershell
python assignment-1/task-3/error_analysis.py
```

The script uses the prepared test split and GPU 0 by default. Pass
`--device cpu` to use the CPU or `--help` to see the available overrides.

The checked-in examples and metrics are in `results/` beside this report.
The reproduction command writes to `runs/task3/` from the repository root.

## Overall result

| Outcome | Objects |
|---|---:|
| Correct | 5,889 |
| Wrong class | 213 |
| Missed | 4,996 |
| False alarms (predictions) | 2,045 |

| Class | GT objects | Missed | Missed % | Wrong class | False alarms |
|---|---:|---:|---:|---:|---:|
| Biodegradable | 6,805 | 3,445 | 50.6% | 33 | 997 |
| Cardboard | 703 | 287 | 40.8% | 24 | 343 |
| Glass | 1,165 | 365 | 31.3% | 37 | 127 |
| Metal | 875 | 269 | 30.7% | 42 | 162 |
| Paper | 658 | 241 | 36.6% | 27 | 180 |
| Plastic | 892 | 389 | 43.6% | 50 | 236 |

Most frequent class confusions (true class -> predicted class): biodegradable
-> cardboard (18), cardboard -> paper (18), glass -> metal (17), plastic ->
glass (15), metal -> glass (15), plastic -> paper (15).

## Example detections

### Correct detections

![cardboard](results/predictions/correct/cardboard632_jpg.rf.57f52afa4c5f2248b98a882f4e313f43.jpg)
![metal](results/predictions/correct/metal623_jpg.rf.0de5fa2202e76f7beea4bcf9fcccf19f.jpg)
![paper](results/predictions/correct/paper1767_jpeg.rf.14e6a301b93a41b99a091b372445fccc.jpg)

Single, large, well-isolated objects (a flattened box, an aluminium tray,
crumpled paper) detected with confidence of about 0.97. This is the easy case:
one object that fills most of the image and has a distinctive texture or
material.

### Missed detections

The model produced no prediction boxes in these three test images at confidence
0.25. Yellow boxes show the dataset objects that were missed.

![Missed objects](results/predictions/missed/metal383_jpg.rf.dd429689d34de9bdc9af4672094492cc.jpg)

One annotated metal object was missed. The image shows an unusual close-up view inside a can.

![Missed objects](results/predictions/missed/paper1042_jpg.rf.d045f3ab7ee5b9104d5c61eb2d03d814.jpg)

All three annotated objects were missed in this rotated image containing a cup, packaging, and a doughnut.

![Missed objects](results/predictions/missed/metal182_jpg.rf.a3e1a96dbcb2b2dcdfbee430b05cbc6f.jpg)

All nine annotated objects were missed in this rotated, low-contrast scene with overlapping containers.

### Incorrect detections

These examples show predictions with the wrong category. Red boxes mark
predictions that overlap a dataset box at IoU >= 0.5 but have a different class.

![Wrong-class prediction](results/predictions/incorrect/glass956_jpg.rf.3b34a1b57d414600ad580491ae64446b.jpg)

Four objects annotated as glass are predicted as metal, with confidence 0.81–0.91. Their reflective appearance makes the material visually ambiguous, so the annotation should also be checked.

![Wrong-class prediction](results/predictions/incorrect/paper1160_jpg.rf.cdce4a52a1a3fe24ead81115c614c72b.jpg)

Two objects annotated as paper in the takeaway-cup image are predicted as metal, with confidence 0.87 and 0.92.

![Wrong-class prediction](results/predictions/incorrect/plastic256_jpg.rf.08555728a98b826954a24ff45bd73749.jpg)

Two objects annotated as plastic are predicted as glass, with confidence 0.54 and 0.25. Transparent packaging can make these materials difficult to distinguish.

## Analysis of the errors

1. **Missed objects are the dominant error, not class confusion.** 4,996
   objects were missed (45% of all) while only 213 were detected with the
   wrong class (1.9%). Once an object is found, its class is mostly right.
2. **Crowded scenes are much harder.** In the 117 test images with more than
   20 objects, 58.6% of the objects were missed (3,880 of 6,616); in the 1,452
   images with at most 20 objects, 24.9% were missed (1,116 of 4,482). Overlap
   and occlusion make boxes hard to separate, and IoU 0.5 is strict for small
   boxes: 36.45% of all boxes cover less than 1% of the image, and the source
   images are only 416 x 416 pixels, so upscaling to 640 adds no detail.
3. **Biodegradable is the weakest class despite having the most boxes** (test
   AP50-95 31.4%, 50.6% missed, 997 false alarms). It is therefore not a simple
   class-imbalance problem. The class contains many small, irregular and
   visually diverse objects (food scraps, seeds) with unclear boundaries, so
   even correct predictions can fail the IoU criterion. Unmatched predictions can also reflect inconsistent annotation granularity.
   This possibility requires manual checking.
4. **Cardboard has many false alarms relative to its size** (343 false alarms
   for 703 objects), mainly on piles of boxes where individual instances are
   ambiguous.
5. **Class confusions occur between visually similar materials**, e.g.
   biodegradable/cardboard, cardboard/paper, glass/metal, plastic/glass:
   materials that share colour, texture, transparency or reflection, or objects
   whose appearance is dominated by a different material (a can with a paper
   label).

## Limitations of the final model

- Limited recall (about 55%): nearly half of the objects in crowded scenes are
  not found, which would matter in a real sorting system.
- Weak on small, overlapping and unusually coloured objects.
- Results depend on annotation quality: only a sample of 30 images was visually
  reviewed, and incomplete or inconsistent labels appear as false alarms and
  misses.
-Possible duplicate leakage: the team's wider duplicate search (Assignment 2, Task 1, section 4.4) found 44 of its 307 duplicate groups still crossing the train/validation/test split; about 19 of them were judged real copies of a test image (about 1.2% of the 1,569 test images). The test scores may therefore be slightly optimistic. Duplicates found by neither search can also remain, and duplicates kept inside the same subset can give some sources extra weight.
- One model on one split; no confidence intervals were computed.
- The figures show 9 selected examples, not a random sample.

## Possible improvements

- **Annotation review:** check crowded and biodegradable images for missing or
  loose boxes and correct genuine errors only (do not remove hard examples).
- **Small and crowded objects:** test higher input resolution or tiled
  (sliced) inference, a larger model, and moderate mosaic augmentation
  (planned as `--mosaic 0.3`); evaluate each on the validation set first.
- **Post-processing:** tune the NMS IoU threshold and the confidence threshold
  for dense scenes, possibly per class.
- **Training:** a short AdamW refinement did not beat the SGD baseline (43.7%
  vs 44.2% validation mAP50-95), so longer training alone is not a
  demonstrated fix.
- **Evaluation:** report AP separately for small and crowded objects to confirm
  the causes above.
