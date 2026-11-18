"""Choose the confidence threshold for Task 3 on the VALIDATION split (best F1).

Predictions are made once at a low threshold and then filtered, so every
candidate threshold is scored on the same boxes. Matching uses the Task 3
rule (error_analysis.match, IoU >= 0.5, greedy by confidence): a prediction
is correct only with the right class; wrong-class and unmatched predictions
are false positives; ground-truth boxes without a correct prediction are
false negatives. The test split is not used.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from error_analysis import IMG_EXT, match, predict_all, read_gt


def sweep(predictions, labels_dir, thresholds, iou_thr=0.5):
    cached = []
    for path, boxes, classes, confs in predictions:
        cached.append((read_gt(Path(labels_dir) / f"{Path(path).stem}.txt"), boxes, classes, confs))
    rows = []
    for t in thresholds:
        tp = n_pred = n_gt = 0
        for (gc, gb), pb, pc, pcf in cached:
            keep = pcf >= t
            ps, _, _ = match(gc, gb, pc[keep], pb[keep], pcf[keep], iou_thr)
            tp += ps.count("TP")
            n_pred += int(keep.sum())
            n_gt += len(gc)
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append(dict(conf=round(float(t), 2), precision=round(p, 4), recall=round(r, 4),
                         f1=round(f1, 4), correct=tp, predictions=n_pred, objects=n_gt))
    return rows


def main():
    assignment = Path(__file__).resolve().parents[1]
    data = assignment / "data/processed"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(assignment / "models/yolo26s-strongaug-best.pt"))
    ap.add_argument("--images", default=str(data / "images/val"))
    ap.add_argument("--labels", default=str(data / "labels/val"))
    ap.add_argument("--out", default=str(Path(__file__).with_name("results") / "metrics/threshold_selection_val.csv"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    thresholds = np.round(np.arange(0.05, 0.91, 0.05), 2)
    rows = sweep(predict_all(a.weights, paths, a.imgsz, float(thresholds[0]), a.device), a.labels, thresholds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: r["f1"])
    for r in rows:
        print(("* " if r is best else "  ") + "  ".join(f"{k} {v}" for k, v in r.items()))
    print(f"chosen confidence threshold (best validation F1): {best['conf']}")


if __name__ == "__main__":
    main()
