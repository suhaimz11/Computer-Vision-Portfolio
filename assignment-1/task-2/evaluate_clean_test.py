"""Evaluate a checkpoint on the test split with and without the 23 leaked test images.

The 23 images listed in leaked-test-images.csv have a duplicate (flipped, rotated,
cropped or recoloured copy) in training or validation according to the union of
both duplicate searches. This script links the full, clean and leaked subsets of
data/processed/ into a scratch folder and runs the Ultralytics evaluator on each.
"""
import argparse
import csv
import json
from pathlib import Path

NAMES = ('biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic')


def count_boxes(subset):
    return sum(1 for f in (subset / 'labels/test').glob('*.txt')
               for line in f.read_text(encoding='utf-8').splitlines() if line.strip())


def link_subset(source, target, keep):
    for kind in ('images', 'labels'):
        (target / kind / 'test').mkdir(parents=True, exist_ok=True)
    for image in sorted((source / 'images/test').iterdir()):
        if not keep(image.name):
            continue
        label = source / 'labels/test' / (image.stem + '.txt')
        for src, dst in ((image, target / 'images/test' / image.name),
                         (label, target / 'labels/test' / label.name)):
            if not dst.exists():
                dst.symlink_to(src.resolve())
    # Ultralytics requires train and val keys; only the test split is evaluated.
    # Ultralytics requires train and val keys; only the test split is evaluated.
    yaml = (f'path: {json.dumps(target.resolve().as_posix())}\n'
            'train: images/test\nval: images/test\ntest: images/test\nnames:\n'
            + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES)))
    (target / 'data.yaml').write_text(yaml, encoding='utf-8')
    return target / 'data.yaml'


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=str(base / 'models/yolo26s-strongaug-best.pt'))
    ap.add_argument('--data', type=Path, default=base / 'data/processed')
    ap.add_argument('--leaked', type=Path, default=Path(__file__).with_name('leaked-test-images.csv'))
    ap.add_argument('--work', type=Path, default=base.parent / 'runs/clean-test')
    ap.add_argument('--device', default='0', help="'0' for CUDA, 'mps' for Apple GPU, 'cpu'")
    a = ap.parse_args()

    from ultralytics import YOLO
    with a.leaked.open(encoding='utf-8', newline='') as stream:
        leaked = {row['test_image'] for row in csv.DictReader(stream)}
    available = {p.name for p in (a.data / 'images/test').iterdir()}
    if not leaked <= available:
        raise ValueError(f'Leaked images missing from the test split: {sorted(leaked - available)}')
    subsets = {'full': lambda n: True, 'clean': lambda n: n not in leaked, 'leaked': lambda n: n in leaked}
    results = {}
    for name, keep in subsets.items():
        data = link_subset(a.data, a.work / name, keep)
        box = YOLO(a.weights).val(data=str(data), split='test', imgsz=640, max_det=400, device=a.device,
                                  plots=False, verbose=False, project=str(a.work / 'eval'), name=name,
                                  exist_ok=True).box
        results[name] = {'images': sum(keep(n) for n in available), 'boxes': count_boxes(a.work / name),
                         'precision': round(float(box.mp), 4), 'recall': round(float(box.mr), 4),
                         'mAP50': round(float(box.map50), 4), 'mAP50-95': round(float(box.map), 4),
                         'per_class_AP50-95': {NAMES[c]: round(float(box.ap[i]), 4)
                                               for i, c in enumerate(box.ap_class_index)}}
    (a.work / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
