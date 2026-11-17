"""Figures for the Task 1 dataset analysis (all 10,464 images, cleaned labels).

1. dataset_overview.png: boxes and images per class, objects per image,
   object size per class, and where box centres lie in the image.
2. dataset_examples.png: one random labelled image per class plus the most
   crowded image, with the annotated boxes drawn.
"""
import argparse
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = ('biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic')
COLOURS = ('#4c9a2a', '#a0522d', '#2a8fbd', '#7f7f7f', '#d4a017', '#c0392b')
SIDE = 416


def load(labels_dir):
    data = {}
    for f in sorted(Path(labels_dir).glob('*.txt')):
        rows = [line.split() for line in f.read_text(encoding='utf-8').splitlines() if line.strip()]
        data[f.stem] = np.array([[float(v) for v in r] for r in rows]).reshape(-1, 5)
    return data


def overview(data, path):
    boxes = np.concatenate(list(data.values()))
    cls = boxes[:, 0].astype(int)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    n_boxes = np.bincount(cls, minlength=6)
    n_images = [sum((b[:, 0] == c).any() for b in data.values()) for c in range(6)]
    x = np.arange(6)
    a = ax[0, 0]
    a.bar(x - 0.2, n_boxes, 0.4, color=COLOURS, label='boxes')
    a2 = a.twinx()
    a2.bar(x + 0.2, n_images, 0.4, color=COLOURS, alpha=0.45, hatch='//', label='images')
    a.set_xticks(x, NAMES, rotation=20)
    a.set_ylabel('boxes (solid)')
    a2.set_ylabel('images containing the class (hatched)')
    a.set_title(f'Class balance: {n_boxes[0] / n_boxes.sum():.0%} of boxes are biodegradable')

    per_image = np.array([len(b) for b in data.values()])
    a = ax[0, 1]
    a.hist(np.clip(per_image, 1, 51), bins=np.arange(1, 53) - 0.5, color='#2a6f97', log=True)
    a.set_xlabel('objects per image (last bar: more than 50)')
    a.set_ylabel('images (log scale)')
    a.set_title(f'Objects per image: median {int(np.median(per_image))}, maximum {per_image.max()}')

    side = np.sqrt(boxes[:, 3] * boxes[:, 4]) * SIDE
    a = ax[1, 0]
    parts = a.violinplot([side[cls == c] for c in range(6)], showmedians=True)
    for body, colour in zip(parts['bodies'], COLOURS):
        body.set_facecolor(colour)
    a.axhline(32, ls='--', c='k', lw=0.8)
    a.set_yscale('log')
    a.set_xticks(np.arange(1, 7), NAMES, rotation=20)
    a.set_ylabel('box size: sqrt(width x height) in px (log)')
    a.set_title(f'Object size: {np.mean(side < 32):.0%} below 32 px (dashed line)')

    a = ax[1, 1]
    h = a.hist2d(boxes[:, 1], boxes[:, 2], bins=40, range=[[0, 1], [0, 1]], cmap='viridis',
                 norm=matplotlib.colors.LogNorm())
    a.invert_yaxis()
    a.set_aspect('equal')
    a.set_xlabel('box centre x (fraction of width)')
    a.set_ylabel('box centre y')
    a.set_title('Where objects are: box centres')
    fig.colorbar(h[3], ax=a, label='boxes (log)')
    fig.savefig(path, dpi=130)


def examples(data, images_dir, path, seed=42, tile=300):
    rng = random.Random(seed)
    picks = []
    for name in NAMES:
        stems = [s for s in data if s.startswith(name) and len(data[s])]
        picks.append(rng.choice(stems))
    picks.append(max(data, key=lambda s: len(data[s])))
    tiles = []
    for stem in picks:
        image = next(Path(images_dir).glob(stem + '.*'))
        im = cv2.resize(cv2.imread(str(image)), (tile, tile))
        for c, x, y, w, h in data[stem]:
            colour = tuple(int(COLOURS[int(c)][i:i + 2], 16) for i in (5, 3, 1))
            cv2.rectangle(im, (int((x - w / 2) * tile), int((y - h / 2) * tile)),
                          (int((x + w / 2) * tile), int((y + h / 2) * tile)), colour, 2)
        bar = np.full((24, tile, 3), 255, np.uint8)
        cv2.putText(bar, f'{stem.split("_")[0]}: {len(data[stem])} box' + ('es' if len(data[stem]) != 1 else ''), (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, im]))
    rows = [np.hstack(tiles[i:i + 4]) for i in (0, 4)]
    legend = np.full((30, rows[0].shape[1], 3), 255, np.uint8)
    for i, (name, colour) in enumerate(zip(NAMES, COLOURS)):
        bgr = tuple(int(colour[j:j + 2], 16) for j in (5, 3, 1))
        cv2.rectangle(legend, (10 + 200 * i, 8), (26 + 200 * i, 22), bgr, -1)
        cv2.putText(legend, name, (32 + 200 * i, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if rows[1].shape[1] < rows[0].shape[1]:
        pad = np.full((rows[1].shape[0], rows[0].shape[1] - rows[1].shape[1], 3), 255, np.uint8)
        rows[1] = np.hstack([rows[1], pad])
    cv2.imwrite(str(path), np.vstack([legend] + rows))


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', default=str(base / 'data/prepared/labels'))
    ap.add_argument('--images', default=str(base / 'GarbageDataset/images'))
    ap.add_argument('--out', type=Path, default=Path(__file__).with_name('figures'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = load(a.labels)
    overview(data, a.out / 'dataset_overview.png')
    examples(data, a.images, a.out / 'dataset_examples.png')
    print('wrote', a.out / 'dataset_overview.png', 'and', a.out / 'dataset_examples.png')


if __name__ == '__main__':
    main()
