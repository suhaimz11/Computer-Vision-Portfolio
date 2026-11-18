# Training history

| Run | Validation mAP50-95 | Outcome |
| --- | ---: | --- |
| Original YOLO26s recipe | 44.2% | Baseline, best epoch 57 of 60 |
| AdamW refinement | 43.7% | Did not improve on the baseline |
| Stronger YOLO26s recipe | 51.2% | Selected model, best epoch 50 of 60 |

The final recipe changes batch size, mosaic, scale, colour augmentation,
rotation and vertical flips together. These results measure their combined
effect. Individual effects require separate controlled experiments.

The full final run and resolved arguments are in
[strong-augmentation-run/](strong-augmentation-run/). The submitted checkpoint
and hashes are listed in [models/README.md](../models/README.md).

Test mAP50-95 is 51.8% on the full split and 51.9% after excluding all 29
flagged test images. See [Task 2](README.md) for per-class results, confidence
intervals and the paired comparison. See [Task 3](../task-3/README.md) for
observed errors and proposed validation experiments.
