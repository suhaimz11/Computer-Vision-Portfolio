"""Precision and recall on the TEST split by object size, crowding and image type.

Uses the Task 3 predictions and matching rule (error_analysis.match: IoU >= 0.5,
greedy by confidence, a prediction is correct only with the right class) at the
confidence threshold chosen on validation (0.25).

- Recall of a group  = correctly detected ground-truth objects / objects in the group.
- Precision of a group = correct predictions / predictions in the group.

Groups (also written: recall per crowding level x image type, so the two can
be compared at equal crowding):
- size: COCO area bins on the 416 x 416 source image (small < 32^2 px,
  medium < 96^2 px, large otherwise); objects by their ground-truth box,
  predictions by their predicted box;
- crowding: annotated objects in the image (1, 2-5, 6-20, >20);
- image type: "white background" if at least 80% of a border strip
  (5% of the shorter side) is near-white (grey value > 225), else "scene".
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from error_analysis import IMG_EXT, match, predict_all, read_gt

SIZE_ORDER = ('small', 'medium', 'large')
CROWD_ORDER = ('1', '2-5', '6-20', '>20')
TYPE_ORDER = ('white background', 'scene')


def size_bin(box, width, height):
    area = (box[2] - box[0]) * width * (box[3] - box[1]) * height
    return 'small' if area < 32 ** 2 else 'medium' if area < 96 ** 2 else 'large'


def crowd_bin(n):
    return '1' if n <= 1 else '2-5' if n <= 5 else '6-20' if n <= 20 else '>20'


def image_type(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    b = max(2, int(0.05 * min(gray.shape)))
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return 'white background' if (border > 225).mean() >= 0.8 else 'scene'


def breakdown(predictions, labels_dir, cross=None):
    # counts[(kind, group)] = [objects, objects found, predictions, correct predictions]
    # cross[(crowding, image type)] = [images, objects, objects found], filled if given
    counts = defaultdict(lambda: [0, 0, 0, 0])
    images = defaultdict(int)
    for path, pb, pc, pcf in predictions:
        gc, gb = read_gt(Path(labels_dir) / f'{Path(path).stem}.txt')
        height, width = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        ps, _, gs = match(gc, gb, pc, pb, pcf)
        image_groups = {'crowding': crowd_bin(len(gc)), 'image type': image_type(path)}
        for kind, group in image_groups.items():
            images[kind, group] += 1
        if cross is not None:
            cell = cross[image_groups['crowding'], image_groups['image type']]
            cell[0] += 1
            cell[1] += len(gc)
            cell[2] += sum(status == 'TP' for status in gs)
        for box, status in zip(gb, gs):
            for key in [('size', size_bin(box, width, height))] + list(image_groups.items()):
                counts[key][0] += 1
                counts[key][1] += status == 'TP'
        for box, status in zip(pb, ps):
            for key in [('size', size_bin(box, width, height))] + list(image_groups.items()):
                counts[key][2] += 1
                counts[key][3] += status == 'TP'
    rows = []
    for kind, order in (('size', SIZE_ORDER), ('crowding', CROWD_ORDER), ('image type', TYPE_ORDER)):
        for group in order:
            n_obj, found, n_pred, correct = counts[kind, group]
            rows.append(dict(breakdown=kind, group=group, images=images.get((kind, group), ''),
                             objects=n_obj, found=found, recall=round(found / n_obj, 4) if n_obj else '',
                             predictions=n_pred, correct=correct,
                             precision=round(correct / n_pred, 4) if n_pred else ''))
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kinds = ('size', 'crowding', 'image type')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, kind in zip(axes, kinds):
        part = [r for r in rows if r['breakdown'] == kind]
        x = np.arange(len(part))
        ax.bar(x - 0.2, [r['recall'] or 0 for r in part], 0.4, label='recall', color='#2a6f97')
        ax.bar(x + 0.2, [r['precision'] or 0 for r in part], 0.4, label='precision', color='#e9a03b')
        ax.set_xticks(x, [f"{r['group']}\n{r['objects']:,} obj." for r in part], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'by {kind}')
        ax.grid(axis='y', alpha=0.3)
        for i, r in enumerate(part):
            ax.text(i - 0.2, (r['recall'] or 0) + 0.02, f"{r['recall']:.0%}", ha='center', fontsize=8)
    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle('YOLO26s on the test split (confidence 0.25, IoU 0.5)', fontsize=11)
    fig.savefig(path, dpi=150)


def main():
    assignment = Path(__file__).resolve().parents[1]
    data = assignment / 'data/processed'
    results = Path(__file__).with_name('results')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=str(assignment / 'models/yolo26s-strongaug-best.pt'))
    ap.add_argument('--images', default=str(data / 'images/test'))
    ap.add_argument('--labels', default=str(data / 'labels/test'))
    ap.add_argument('--out', default=str(results / 'metrics/breakdowns_test.csv'))
    ap.add_argument('--figure', default=str(results / 'breakdowns_test.png'))
    ap.add_argument('--cross-out', default=str(results / 'metrics/recall_crowding_imagetype_test.csv'))
    ap.add_argument('--conf', type=float, default=0.25, help='chosen on validation by select_threshold.py')
    ap.add_argument('--device', default='0')
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    cross = defaultdict(lambda: [0, 0, 0])
    rows = breakdown(predict_all(a.weights, paths, 640, a.conf, a.device), a.labels, cross)
    with open(a.cross_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['objects_per_image', 'image_type', 'images', 'objects', 'found', 'recall'])
        for crowd in CROWD_ORDER:
            for kind in TYPE_ORDER:
                n_img, n_obj, found = cross[crowd, kind]
                w.writerow([crowd, kind, n_img, n_obj, found, round(found / n_obj, 4) if n_obj else ''])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    plot(rows, a.figure)
    for r in rows:
        print(f"{r['breakdown']:10s} {r['group']:16s} images {str(r['images']):>5s}  objects {r['objects']:6d}  "
              f"recall {r['recall']:.3f}  predictions {r['predictions']:5d}  precision {r['precision']:.3f}")


if __name__ == '__main__':
    main()
