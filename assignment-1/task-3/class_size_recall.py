"""Recall per class and object size on the TEST split (final model, confidence 0.25).

Same predictions and matching rule as error_analysis.py / breakdowns.py (IoU >=
0.5, greedy by confidence, right class required). Each ground-truth object is
put in a COCO size bin by its own box on the 416-pixel source image (small <
32^2 px, medium < 96^2 px, large otherwise); recall = objects found / objects.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from breakdowns import SIZE_ORDER, size_bin
from error_analysis import CLASSES, IMG_EXT, match, predict_all, read_gt


def main():
    assignment = Path(__file__).resolve().parents[1]
    data = assignment / 'data/processed'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=str(assignment / 'models/yolo26s-strongaug-best.pt'))
    ap.add_argument('--images', default=str(data / 'images/test'))
    ap.add_argument('--labels', default=str(data / 'labels/test'))
    ap.add_argument('--out', default=str(Path(__file__).with_name('results') / 'metrics/recall_class_size_test.csv'))
    ap.add_argument('--conf', type=float, default=0.25, help='chosen on validation by select_threshold.py')
    ap.add_argument('--device', default='0')
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    counts = defaultdict(lambda: [0, 0])  # (class, size) -> objects, found
    for path, pb, pc, pcf in predict_all(a.weights, paths, 640, a.conf, a.device):
        gc, gb = read_gt(Path(a.labels) / f'{Path(path).stem}.txt')
        height, width = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        _, _, gs = match(gc, gb, pc, pb, pcf)
        for c, box, status in zip(gc, gb, gs):
            key = (CLASSES[int(c)], size_bin(box, width, height))
            counts[key][0] += 1
            counts[key][1] += status == 'TP'

    rows = []
    for name in CLASSES:
        row = {'class': name}
        for s in SIZE_ORDER:
            n, found = counts[name, s]
            row[f'{s}_objects'] = n
            row[f'{s}_recall'] = round(found / n, 4) if n else ''
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
