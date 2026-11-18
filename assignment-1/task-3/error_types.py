"""Split the Task 3 false alarms and missed objects into finer error types (TEST split).

The Task 3 matching (error_analysis.match, IoU >= 0.5, greedy by confidence) is
kept unchanged; its counts are the starting point. Each unmatched prediction
("false alarm") is then explained by the ground-truth box it overlaps most:

- duplicate:     IoU >= 0.5 with an object that another prediction already matched
- imprecise box: same class as that object, 0.1 <= IoU < 0.5 (box too loose or too tight)
- other class:   overlaps an object of another class, 0.1 <= IoU < 0.5
- background:    IoU < 0.1 with every object (nothing annotated there)

Each missed object is "nearly found" if a prediction of its class overlaps it
with 0.1 <= IoU < 0.5, and "not detected" otherwise.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from error_analysis import CLASSES, IMG_EXT, iou_matrix, match, predict_all, read_gt


def classify(gc, gb, pc, pb, pcf, iou_ok=0.5, iou_poor=0.1):
    ps, pg, gs = match(gc, gb, pc, pb, pcf, iou_ok)
    ious = iou_matrix(pb, gb) if len(pc) and len(gc) else np.zeros((len(pc), len(gc)))
    pred_types, gt_types = [], []
    for i, status in enumerate(ps):
        if status != 'FP':
            continue
        if not len(gc):
            pred_types.append((int(pc[i]), 'background'))
            continue
        j = int(np.argmax(ious[i]))
        iou = ious[i, j]
        if iou >= iou_ok:
            kind = 'duplicate'
        elif iou >= iou_poor:
            kind = 'imprecise box' if pc[i] == gc[j] else 'other class'
        else:
            kind = 'background'
        pred_types.append((int(pc[i]), kind))
    for j, status in enumerate(gs):
        if status != 'FN':
            continue
        same = (pc == gc[j]) if len(pc) else np.zeros(0, bool)
        near = bool(len(pc)) and bool(((ious[:, j] >= iou_poor) & (ious[:, j] < iou_ok) & same).any())
        gt_types.append((int(gc[j]), 'nearly found' if near else 'not detected'))
    return pred_types, gt_types


def main():
    assignment = Path(__file__).resolve().parents[1]
    data = assignment / 'data/processed'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=str(assignment / 'models/yolo26s-strongaug-best.pt'))
    ap.add_argument('--images', default=str(data / 'images/test'))
    ap.add_argument('--labels', default=str(data / 'labels/test'))
    ap.add_argument('--out', default=str(Path(__file__).with_name('results') / 'metrics/error_types_test.csv'))
    ap.add_argument('--conf', type=float, default=0.25, help='chosen on validation by select_threshold.py')
    ap.add_argument('--device', default='0')
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    fp, fn = Counter(), Counter()
    for path, pb, pc, pcf in predict_all(a.weights, paths, 640, a.conf, a.device):
        gc, gb = read_gt(Path(a.labels) / f'{Path(path).stem}.txt')
        pred_types, gt_types = classify(gc, gb, pc, pb, pcf)
        fp.update(pred_types)
        fn.update(gt_types)

    rows = []
    for group, counter, kinds in (('false alarm', fp, ('duplicate', 'imprecise box', 'other class', 'background')),
                                  ('missed object', fn, ('nearly found', 'not detected'))):
        for kind in kinds:
            row = dict(error=group, type=kind, total=sum(n for (c, k), n in counter.items() if k == kind))
            row.update({name: counter[c, kind] for c, name in enumerate(CLASSES)})
            rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print('  '.join(f'{k} {v}' for k, v in r.items()))


if __name__ == '__main__':
    main()
