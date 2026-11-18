"""Evaluate a checkpoint before and after excluding flagged source copies.

The default exclusion list contains 29 test images identified by the duplicate
audits. Pass --leaked to reproduce another list. Subset files are rebuilt in
the output directory without changing source images or labels.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

NAMES = ('biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic')


def count_boxes(subset):
    return sum(1 for f in (subset / 'labels/test').glob('*.txt')
               for line in f.read_text(encoding='utf-8').splitlines() if line.strip())


def link_subset(source, target, keep):
    source = Path(source).resolve()
    target = Path(target).resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Source and output dataset directories must be separate")

    images = [image for image in sorted((source / 'images/test').iterdir())
              if image.is_file() and keep(image.name)]
    expected = {
        'images': {image.name for image in images},
        'labels': {image.stem + '.txt' for image in images},
    }
    for image in images:
        if not (source / 'labels/test' / (image.stem + '.txt')).is_file():
            raise FileNotFoundError(f"Missing label for {image.name}")

    for kind, names in expected.items():
        folder = target / kind / 'test'
        if not folder.resolve().is_relative_to(target):
            raise ValueError("Output dataset folders must stay inside the target")
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.iterdir():
            if (old.is_file() or old.is_symlink()) and old.name not in names:
                old.unlink()

    for image in images:
        label = source / 'labels/test' / (image.stem + '.txt')
        for src, dst in ((image, target / 'images/test' / image.name),
                         (label, target / 'labels/test' / label.name)):
            if dst.exists() and dst.samefile(src):
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)

    # Ultralytics requires train and val keys when evaluating the test split.
    yaml = (f'path: {json.dumps(target.as_posix())}\n'
            'train: images/test\nval: images/test\ntest: images/test\nnames:\n'
            + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES)))
    (target / 'data.yaml').write_text(yaml, encoding='utf-8')
    return target / 'data.yaml'


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=str(base / 'models/yolo26s-strongaug-best.pt'))
    ap.add_argument('--data', type=Path, default=base / 'data/processed')
    ap.add_argument('--leaked', type=Path, default=Path(__file__).with_name('leaked-test-images-extended.csv'))
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
