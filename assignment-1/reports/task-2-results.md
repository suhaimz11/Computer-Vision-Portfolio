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

Checkpoint sources relative to Assignment 1:

- YOLO26s: `runs/train/yolo26s_baseline-2/weights/best.pt`.
- YOLOv5s: `yolov5/runs/yolov5s_20261113_135834_037492/weights/best.pt`.

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

YOLO26s led by 8.0 percentage points in test mAP50–95 and achieved
higher AP50–95 in all six classes. Glass was strongest for both models.
Biodegradable waste had the lowest AP50–95 in both despite having the
most annotations. Its lower scores under stricter overlap criteria warrant
inspecting box localization and missed detections in Task 3. Plastic also
scored relatively low. YOLO26s was selected on validation and remains the
model to use for the final error analysis.

The test console summaries are saved beside this report. Evaluation used
Python 3.14.4, PyTorch 2.11.0+cu128, Ultralytics 8.4.208 and an RTX 5050
Laptop GPU. The YOLOv5 evaluator was from revision
`cb2f17fd5d790cfdd0c07c25351904df9affbe6e`.
