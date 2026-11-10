"""Copy the fixed manifest and verified cleaned labels into YOLO directories.

Run from assignment-1: python export_yolo.py
Existing destinations are refused. No new split is generated.
"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audit_dataset import parse_box


SPLITS = ('train', 'val', 'test')
NAMES = ('biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root, manifest_dir, output):
    root, manifest_dir, output = root.resolve(), manifest_dir.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f'Destination exists; choose a fresh --output: {output}')
    manifest = manifest_dir / 'manifest.csv'
    summary = json.loads((manifest_dir / 'summary.json').read_text(encoding='utf-8'))
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    images = root / 'GarbageDataset/images'
    labels = root / 'data/prepared/labels'
    seen, seen_labels, assignments, groups = set(), set(), {}, defaultdict(set)
    for row in rows:
        image, label, split = row['image'], row['label'], row['split']
        if (Path(image).name != image or Path(label).name != label
                or image in seen or label in seen_labels or split not in SPLITS
                or Path(image).stem != Path(label).stem):
            raise ValueError(f'Invalid or repeated manifest row: {row}')
        if not (images / image).is_file() or not (labels / label).is_file():
            raise ValueError(f'Missing source file: {row}')
        seen.add(image)
        seen_labels.add(label)
        assignments[image] = split
        groups[row['duplicate_group']].add(split)
    source_names = {p.name for p in images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}}
    if seen != source_names or len(rows) != summary['images'] or any(len(g) != 1 for g in groups.values()):
        raise ValueError('Incomplete manifest or duplicate group crosses subsets')
    review = root / 'reports/duplicate-review/review.csv'
    if sha(review) != summary['review_sha256']:
        raise ValueError('Review changed since split creation')
    with review.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['decision'] == 'same_source' and assignments[row['image_a']] != assignments[row['image_b']]:
                raise ValueError(f"Duplicate leakage at pair {row['pair']}")

    def inspect(row):
        data = (labels / row['label']).read_bytes()
        boxes = [parse_box(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
        if any(box[0] not in range(6) for box in boxes):
            raise ValueError(f"Invalid class in {row['label']}")
        return boxes, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=16) as pool:
        inspected = list(pool.map(inspect, rows))
    digest = hashlib.sha256()
    boxes_by_split, presence = defaultdict(Counter), defaultdict(Counter)
    for row, (boxes, _) in zip(rows, inspected):
        digest.update(row['label'].encode() + repr(boxes).encode())
        boxes_by_split[row['split']].update(b[0] for b in boxes)
        presence[row['split']].update({b[0] for b in boxes})
    if digest.hexdigest() != summary['clean_label_content_sha256']:
        raise ValueError('Cleaned labels changed since split creation')
    counts = Counter(row['split'] for row in rows)
    for split in SPLITS:
        expected = summary['splits'][split]
        if (counts[split] != expected['images']
                or [boxes_by_split[split][c] for c in range(6)] != expected['boxes_by_class_id']
                or [presence[split][c] for c in range(6)] != expected['images_by_class_id']):
            raise ValueError(f'Summary mismatch: {split}')
    for split in SPLITS:
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)

    def copy_verify(item):
        row, (_, label_hash) = item
        source_image, source_label = images / row['image'], labels / row['label']
        target_image = output / 'images' / row['split'] / row['image']
        target_label = output / 'labels' / row['split'] / row['label']
        image_hash = sha(source_image)
        shutil.copyfile(source_image, target_image)
        shutil.copyfile(source_label, target_label)
        if sha(target_image) != image_hash or sha(target_label) != label_hash:
            raise ValueError(f"Copy verification failed: {row['image']}")

    print(f'Copying and verifying {len(rows)} image/label pairs...', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(copy_verify, zip(rows, inspected)), 1):
            if index % 1000 == 0:
                print(f'Exported {index}/{len(rows)}', flush=True)
    shutil.copyfile(manifest, output / 'manifest.csv')
    shutil.copyfile(manifest_dir / 'summary.json', output / 'split-summary.json')
    yaml = '# Class names follow the repository working mapping; publisher confirmation pending.\n'
    yaml += f'path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'
    yaml += ''.join(f'  {i}: {name}\n' for i, name in enumerate(NAMES))
    (output / 'data.yaml').write_text(yaml, encoding='utf-8')
    report = {'images_by_split': dict(counts), 'verified_image_label_pairs': len(rows),
              'manifest_sha256': sha(manifest), 'clean_label_content_sha256': digest.hexdigest(),
              'copy_verification': 'SHA-256 of every exported image and cleaned label matches source',
              'class_names_verified': False}
    (output / 'export-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=base)
    parser.add_argument('--manifest-dir', type=Path, default=base / 'data/split')
    parser.add_argument('--output', type=Path, default=base / 'data/processed')
    args = parser.parse_args()
    export(args.root, args.manifest_dir, args.output)
