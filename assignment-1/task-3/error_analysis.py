"""Visualise detections and analyse errors on the TEST set (Assignment 1, Task 3).

Every prediction is matched to ground truth (IoU >= 0.5, greedy by confidence):
  TP   correct class, matches a ground-truth box
  CONF matches a box but with the WRONG class (class confusion)
  FP   matches no ground-truth box (false alarm / background)
  FN   ground-truth box with no matching prediction (missed object)
Saves annotated example images (correct / missed / incorrect) and CSV summaries.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GREEN, YELLOW, BLUE, RED, ORANGE = (0, 200, 0), (0, 220, 255), (255, 120, 0), (0, 0, 255), (0, 140, 255)


def iou_matrix(a, b):
    """IoU between boxes a (N,4) and b (M,4), both normalised xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match(gt_cls, gt_box, p_cls, p_box, p_conf, iou_thr=0.5):
    """Return (pred_status, pred_gt_index, gt_status)."""
    pred_status = ["FP"] * len(p_cls)
    pred_gt = [-1] * len(p_cls)
    gt_status = ["FN"] * len(gt_cls)
    if len(gt_cls) and len(p_cls):
        ious = iou_matrix(p_box, gt_box)
        used = np.zeros(len(gt_cls), bool)
        for pi in np.argsort(-p_conf):
            cand = np.where(~used)[0]
            if len(cand) == 0:
                break
            j = cand[np.argmax(ious[pi, cand])]
            if ious[pi, j] >= iou_thr:
                used[j] = True
                pred_gt[pi] = j
                ok = p_cls[pi] == gt_cls[j]
                pred_status[pi] = "TP" if ok else "CONF"
                gt_status[j] = "TP" if ok else "CONF"
    return pred_status, pred_gt, gt_status


def read_gt(label_path):
    cls, box = [], []
    for line in Path(label_path).read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, x, y, w, h = int(p[0]), *map(float, p[1:])
        if w <= 0 or h <= 0:
            continue
        cls.append(c)
        box.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    return np.array(cls, int), np.array(box, float).reshape(-1, 4)


def pick(items, key, k):
    """Top-k by key, preferring different dominant classes for variety."""
    items = sorted(items, key=key, reverse=True)
    chosen, seen = [], set()
    for it in items:
        if it["dom"] not in seen and len(chosen) < k:
            chosen.append(it); seen.add(it["dom"])
    for it in items:
        if it not in chosen and len(chosen) < k:
            chosen.append(it)
    return chosen


def draw(img_path, gt_cls, gt_box, gt_status, p_cls, p_box, p_conf, p_status, title, scale=2):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]

    def px(b):
        return [int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)]

    for c, b, s in zip(gt_cls, gt_box, gt_status):
        x1, y1, x2, y2 = px(b)
        if s == "FN":
            cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 1)
    for c, b, cf, s in zip(p_cls, p_box, p_conf, p_status):
        x1, y1, x2, y2 = px(b)
        col = {"TP": BLUE, "CONF": RED, "FP": ORANGE}[s]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{CLASSES[int(c)][:5]} {cf:.2f}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    bar = np.full((52, w, 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, "green=GT  yellow=MISSED GT  blue=correct  red=wrong class  orange=false alarm",
                (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def predict_all(weights, paths, imgsz, conf, device):
    """Yield (path, boxes xyxy normalised, classes, confidences) per image."""
    from ultralytics import YOLO
    model = YOLO(weights)
    for p in paths:
        r = model.predict(str(p), imgsz=imgsz, conf=conf, iou=0.6, max_det=400,
                          device=device, verbose=False)[0]
        yield (p, r.boxes.xyxyn.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
               r.boxes.conf.cpu().numpy())


def run(predictions, labels_dir, out_dir, metrics_dir, k=3, iou_thr=0.5):
    records, per_img = [], []
    cls_stats = {c: Counter() for c in range(len(CLASSES))}
    pairs = Counter()
    fp_by_class = Counter()
    for path, pb, pc, pcf in predictions:
        gc, gb = read_gt(Path(labels_dir) / f"{Path(path).stem}.txt")
        ps, pg, gs = match(gc, gb, pc, pb, pcf, iou_thr)
        for c, s in zip(gc, gs):
            cls_stats[int(c)][s] += 1
        for j, (c, s) in enumerate(zip(pc, ps)):
            if s == "CONF":
                pairs[(int(gc[pg[j]]), int(c))] += 1
            if s == "FP":
                fp_by_class[int(c)] += 1
        tp_conf = [cf for cf, s in zip(pcf, ps) if s == "TP"]
        rec = dict(stem=Path(path).stem, path=path, n_gt=len(gc), tp=ps.count("TP"),
                   conf=ps.count("CONF"), fp=ps.count("FP"), fn=gs.count("FN"),
                   dom=int(Counter(gc.tolist()).most_common(1)[0][0]) if len(gc) else -1,
                   mean_tp_conf=float(np.mean(tp_conf)) if tp_conf else 0.0)
        per_img.append((rec, gc, gb, gs, pc, pb, pcf, ps))
        records.append(rec)

    metrics_dir = Path(metrics_dir); metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "error_summary_per_image.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "n_gt", "TP", "wrong_class", "false_pos", "missed"])
        w.writerows([[r["stem"], r["n_gt"], r["tp"], r["conf"], r["fp"], r["fn"]] for r in records])
    with open(metrics_dir / "errors_per_class.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "gt_objects", "detected_correctly", "wrong_class", "missed", "false_alarms"])
        for c, n in enumerate(CLASSES):
            s = cls_stats[c]; tot = sum(s.values())
            w.writerow([n, tot, s["TP"], s["CONF"], s["FN"], fp_by_class[c]])
    with open(metrics_dir / "class_confusions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["true_class", "predicted_as", "count"])
        for (t, p), n in pairs.most_common():
            w.writerow([CLASSES[t], CLASSES[p], n])

    # example selection (readable images only: <= 20 objects)
    small = [x for x in per_img if 1 <= x[0]["n_gt"] <= 20]
    by_stem = {x[0]["stem"]: x for x in small}
    correct = [x[0] for x in small if x[0]["fn"] == 0 and x[0]["fp"] == 0 and x[0]["conf"] == 0]
    missed = [x[0] for x in small if x[0]["fn"] >= 1 and x[0]["tp"] + x[0]["conf"] + x[0]["fp"] == 0]
    wrong = [x[0] for x in small if x[0]["conf"] >= 1]
    chosen = {}
    chosen["correct"] = pick(correct, lambda r: (r["mean_tp_conf"], r["n_gt"]), k)
    taken = {r["stem"] for r in chosen["correct"]}
    chosen["missed"] = pick([r for r in missed if r["stem"] not in taken],
                            lambda r: (-r["n_gt"], r["stem"]), k)
    taken |= {r["stem"] for r in chosen["missed"]}
    chosen["incorrect"] = pick([r for r in wrong if r["stem"] not in taken],
                               lambda r: (r["conf"] / r["n_gt"], -r["fp"], r["conf"]), k)

    out_dir = Path(out_dir)
    for cat, recs in chosen.items():
        (out_dir / cat).mkdir(parents=True, exist_ok=True)
        for r in recs:
            _, gc, gb, gs, pc, pb, pcf, ps = by_stem[r["stem"]]
            title = (f"{cat.upper()}: {r['stem'][:40]} | GT={r['n_gt']} TP={r['tp']} "
                     f"wrong={r['conf']} FP={r['fp']} missed={r['fn']}")
            cv2.imwrite(str(out_dir / cat / f"{r['stem']}.jpg"),
                        draw(r["path"], gc, gb, gs, pc, pb, pcf, ps, title))
    return records, chosen


def main():
    assignment = Path(__file__).resolve().parents[1]
    test_data = assignment / "data/processed"
    output = assignment.parent / "runs/task3"

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(assignment / "models/yolo26s-best.pt"))
    ap.add_argument("--images", default=str(test_data / "images/test"))
    ap.add_argument("--labels", default=str(test_data / "labels/test"))
    ap.add_argument("--out", default=str(output / "predictions"))
    ap.add_argument("--metrics", default=str(output / "metrics"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--examples", type=int, default=3, help="examples per case")
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    records, chosen = run(predict_all(a.weights, paths, a.imgsz, a.conf, a.device),
                          a.labels, a.out, a.metrics, a.examples)
    tot = {k: sum(r[k] for r in records) for k in ("n_gt", "tp", "conf", "fp", "fn")}
    print(f"images: {len(records)}  GT objects: {tot['n_gt']}  correct: {tot['tp']}  "
          f"wrong class: {tot['conf']}  missed: {tot['fn']}  false alarms: {tot['fp']}")
    for cat, recs in chosen.items():
        print(cat, [r["stem"][:25] for r in recs])
    print("examples saved in", a.out, "| tables in", a.metrics)


if __name__ == "__main__":
    main()