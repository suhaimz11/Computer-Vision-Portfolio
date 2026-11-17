# Task 1 and completed training review

Reviewed against page 2 of `WS2026_ComputerVision_Portfolio-1.pdf` on 2026-11-12.

## Task 1 verdict

Substantially meets the requirements: dataset characteristics are measured, their
implications are discussed, a train/validation/test split exists, and preprocessing,
augmentation, optimization and checkpoint selection are specified. The PDF explicitly
lists YOLO as an example; confirm that your chosen variant is acceptable in class.
This is a technical review, not a guaranteed mark.

Reverification in this review:
- All seven audit/split/export unit tests passed.
- All 10,464 prepared labels were checked against the originals: exactly the 17
  logged zero-area rows were removed; valid boxes were preserved.
- Regenerating the split in a separate temporary directory produced a byte-identical
  manifest. Counts remain 7,325 train / 1,570 validation / 1,569 test.
- Reviewed duplicate decisions produce 267 non-singleton groups, isolated by split.
  Duplicates found only by the wider search are not isolated: 44 groups cross the
  split (23 test images with a copy in training or validation, 26 validation
  images with a copy in training). Removing the 23
  test images changes test mAP50–95 from 44.5% to 44.7% (see the [Task 2 report](README.md#test-score-without-leaked-duplicates)).

Remaining documentation gaps (class names are now confirmed by the export's `data.yaml`): clear individual group
contributions, and outdated statements that training is unfinished. The dataset link
can be recovered from the PDF: https://cloud.fiw.fhws.de/s/P9BGwMj7oAZrQ2R . This
identifies the assigned source; it does not itself verify the class mapping.
Numeric validation and 30 sampled images cannot certify all annotation semantics.

Task 2 also requires test-set evaluation and interpretation, and Task 3 requires
2-3 examples each of correct, missed and incorrect detections on test images.
The existing `runs/detect/val` pictures alone do not establish which checkpoint,
split or settings produced them. Save those details with final evaluation.

## Completed baseline

Source: `runs/train/yolo26s_baseline-2/results.csv` and `args.yaml`.
60 epochs completed on all training data, with AMP on and two workers. This is
distinct from the one-epoch full-precision and diagnostic runs. Recorded training
time is 10,806 seconds (about three hours).

Best logged epoch: 57. Installed Ultralytics selects by mAP50-95.

| Validation metric | Value |
|---|---:|
| Precision | 71.445% |
| Recall | 56.585% |
| mAP50 | 62.346% |
| mAP50-95 | 44.196% |

These are object-detection metrics, not classification accuracy or final test scores.
The saved final PR plot reports rounded class AP50: biodegradable 57.8%, cardboard
49.4%, glass 70.9%, metal 74.0%, paper 63.6%, plastic 58.5%. Small differences from
the epoch CSV can arise in the final best-checkpoint validation; keep sources clear.
Do not interpret confusion-matrix diagonal values as AP or overall accuracy.

Training and validation losses decrease; validation AP levels off after roughly
40 epochs. There is no obvious late validation collapse. More epochs alone are not
a demonstrated fix. Cardboard is the first class to investigate, followed by plastic
and biodegradable waste. The large gap between mAP50 and mAP50-95 indicates sensitivity
to stricter localization requirements. Small/crowded objects and label quality are
plausible contributors, but size-stratified evaluation is needed to establish that.
Biodegradable boxes dominate counts, while image presence is much less imbalanced;
blind class oversampling is therefore not the first intervention.

## Controlled next experiments

Keep the existing split, seed, baseline weights and metrics. Select experiments on
validation mAP50-95, checking per-class AP and recall too. Do not tune on test data.
No new training has been launched or improvement measured by this review.

First review 20-30 training/validation examples from the weak classes for missing,
loose or inconsistent boxes, especially cardboard versus paper. Correct only genuine
annotation errors and document changes; do not remove difficult examples to raise scores.

The training script now exposes checkpoint, optimizer, learning rate, patience and
mosaic settings while preserving baseline defaults. Run from the repository root
using the existing environment (the bare `python` alias did not work in this session).

1. Try a short, low-learning-rate refinement of the existing best checkpoint. This
   is a new optimization schedule, not resuming the completed 60-epoch run. Compare
   its best result with 0.44196 and retain the original if refinement is worse:

```powershell
.\.venv\Scripts\python.exe assignment-1/train_detector.py --model assignment-1/runs/train/yolo26s_baseline-2/weights/best.pt --epochs 30 --optimizer AdamW --lr0 0.0001 --warmup-bias-lr 0.0001 --patience 10 --name yolo26s_refine_adamw
```

2. If gains are small, compare a fresh COCO-initialized AdamW run against the
   original SGD baseline, keeping the same 60-epoch budget and augmentations:

```powershell
.\.venv\Scripts\python.exe assignment-1/train_detector.py --optimizer AdamW --lr0 0.001 --warmup-bias-lr 0.001 --name yolo26s_adamw
```

3. Separately test moderate mosaic augmentation from the same COCO initialization,
   preserving SGD settings. It may improve scene diversity but can hurt tiny objects;
   do not combine it with the optimizer change until each has been evaluated:

```powershell
.\.venv\Scripts\python.exe assignment-1/train_detector.py --mosaic 0.3 --close-mosaic 10 --name yolo26s_mosaic03
```

Append `--dry-run` to inspect any command without training. If the prior CUDA issue
recurs, restart Python and use the documented `--no-amp --workers 0 --cuda-debug`
diagnostic settings, recording the change. A bigger model or 768-pixel input is a
later experiment, with greater memory cost; original images are only 416 pixels.
Lowering prediction confidence trades precision for recall and does not by itself
demonstrate better AP. Keep the evaluation detection cap and preprocessing consistent.

Official guidance supports explicit AdamW/lower-LR fine-tuning and configurable
augmentation, but does not guarantee improvement on this dataset:
- https://docs.ultralytics.com/guides/yolo26-training-recipe
- https://docs.ultralytics.com/modes/train

After selecting the model on validation, evaluate the fixed test subset once with
the recorded checkpoint, package versions, image size and max_det=400. Save per-class
AP, precision/recall and error examples for Tasks 2 and 3.
