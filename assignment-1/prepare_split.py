"""Verify cleaned labels and create a reproducible, duplicate-group-aware split.

Run: python assignment-1/prepare_split.py
Requires numpy (already used by the duplicate pipeline). Originals are read-only.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from audit_dataset import parse_box


def make_groups(names, review):
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for row in review:
        if row['decision'] not in {'same_source', 'different_images'}:
            raise ValueError(f"Unresolved pair {row['pair']}")
        a, b = row['image_a'], row['image_b']
        if a not in parent or b not in parent:
            raise ValueError(f"Unknown image in pair {row['pair']}")
        if row['decision'] == 'same_source':
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
    for row in review:
        if row['decision'] == 'different_images' and find(row['image_a']) == find(row['image_b']):
            raise ValueError(f"Contradictory duplicate decisions at pair {row['pair']}")
    groups = defaultdict(list)
    for name in sorted(names):
        groups[find(name)].append(name)
    return list(groups.values())


def allocate(groups, counts, seed):
    """Greedily balance image count, class presence and box count by whole groups."""
    rng = np.random.default_rng(seed)
    features = np.array([
        [len(group), *np.sum([counts[n] > 0 for n in group], axis=0),
         *np.sum([counts[n] for n in group], axis=0)] for group in groups
    ], dtype=float)
    total = features.sum(axis=0)
    ratios = np.array([.70, .15, .15])
    targets = ratios[:, None] * total
    capacities = np.array([round(total[0] * .70), round(total[0] * .15), 0])
    capacities[2] = int(total[0]) - capacities[:2].sum()
    current = np.zeros_like(targets)
    # Rare-class and large groups first; seeded tie breaking.
    order = rng.permutation(len(groups))
    order = sorted(order, key=lambda i: -np.max(features[i] / np.maximum(total, 1)))
    assignments = {}
    for i in order:
        costs = []
        for split in range(3):
            if current[split, 0] + features[i, 0] > capacities[split]:
                costs.append(float('inf'))
                continue
            before = (current[split] - targets[split]) / np.maximum(total, 1)
            after = (current[split] + features[i] - targets[split]) / np.maximum(total, 1)
            costs.append(np.sum(after**2 - before**2) / ratios[split])
        if not np.isfinite(min(costs)):
            raise ValueError('Group sizes cannot fit target counts; choose another allocation strategy')
        chosen = int(np.argmin(costs))
        current[chosen] += features[i]
        for name in groups[i]:
            assignments[name] = ('train', 'val', 'test')[chosen]
    return assignments


def prepare(root, output, seed=42):
    if output.exists():
        raise ValueError(f'Output already exists; choose a fresh --output: {output}')
    dataset = root / 'GarbageDataset'
    images = sorted((dataset / 'images').glob('*'))
    images = [p for p in images if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError('Missing images or ambiguous stems')
    original_labels = dataset / 'labels'
    prepared_labels = root / 'data/prepared/labels'
    expected = {p.stem + '.txt' for p in images}
    for folder in (original_labels, prepared_labels):
        if {p.name for p in folder.glob('*.txt')} != expected:
            raise ValueError(f'Image/label mismatch: {folder}')
    counts, removed, digest = {}, [], hashlib.sha256()
    def read_labels(image):
        name = image.stem + '.txt'
        return ((original_labels / name).read_text(encoding='utf-8-sig'),
                (prepared_labels / name).read_text(encoding='utf-8-sig'))

    with ThreadPoolExecutor(max_workers=16) as pool:
        label_texts = list(pool.map(read_labels, images))
    for index, (image, (original_text, clean_text)) in enumerate(zip(images, label_texts)):
        if index % 1000 == 0:
            print(f'Verifying labels: {index}/{len(images)}', flush=True)
        label_name = image.stem + '.txt'
        valid = []
        boxes = np.zeros(6, dtype=int)
        for line_number, line in enumerate(original_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                box = parse_box(line)
            except ValueError as exc:
                if str(exc) != 'zero-area bounding box':
                    raise
                removed.append(f'{label_name} | line {line_number} | {line.strip()}')
                continue
            if box[0] not in range(6):
                raise ValueError(f'Unexpected class: {label_name}')
            valid.append(box)
            boxes[box[0]] += 1
        clean = [parse_box(line) for line in clean_text.splitlines() if line.strip()]
        if clean != valid:
            raise ValueError(f'Prepared label differs beyond zero-area removal: {label_name}')
        counts[image.name] = boxes
        digest.update(label_name.encode() + repr(valid).encode())
    supplied_log = (root / 'data/prepared/removed_rows.txt').read_text(encoding='utf-8-sig').splitlines()
    if set(removed) != set(line.strip() for line in supplied_log if line.strip()):
        raise ValueError('Removal log does not match source labels')
    review_path = root / 'reports/duplicate-review/review.csv'
    with review_path.open(encoding='utf-8-sig', newline='') as stream:
        review = list(csv.DictReader(stream))
    groups = make_groups(counts, review)
    assignments = allocate(groups, counts, seed)
    group_ids = {name: group[0] for group in groups for name in group}
    summary = {'seed': seed, 'target_ratios': {'train': .7, 'val': .15, 'test': .15},
               'algorithm': 'grouped greedy balance of image counts, per-class image presence and box counts; numpy default_rng',
               'numpy_version': np.__version__, 'images': len(images), 'groups': len(groups),
               'duplicate_groups': sum(len(g) > 1 for g in groups),
               'largest_group': max(map(len, groups)), 'removed_zero_area_rows': len(removed),
               'review_decisions': dict(Counter(r['decision'] for r in review)),
               'review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
               'clean_label_content_sha256': digest.hexdigest(), 'splits': {}}
    for split in ('train', 'val', 'test'):
        members = [name for name in counts if assignments[name] == split]
        box_counts = np.sum([counts[n] for n in members], axis=0)
        presence = np.sum([counts[n] > 0 for n in members], axis=0)
        if not members or np.any(presence == 0):
            raise ValueError(f'Missing class coverage in {split}')
        summary['splits'][split] = {'images': len(members), 'fraction': len(members) / len(images),
                                    'boxes_by_class_id': box_counts.tolist(), 'images_by_class_id': presence.tolist()}
    assert all(len({assignments[n] for n in group}) == 1 for group in groups)
    output.mkdir(parents=True)
    with (output / 'manifest.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'label', 'split', 'duplicate_group'])
        writer.writerows((p.name, p.stem + '.txt', assignments[p.name], group_ids[p.name]) for p in images)
    # Lists use source images. Labels will need materializing beside them for YOLO;
    # do not emit a training YAML that would silently select original dirty labels.
    for split in ('train', 'val', 'test'):
        (output / f'{split}.txt').write_text(''.join(f'{p.name}\n' for p in images if assignments[p.name] == split), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'data/split')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.root, args.output, args.seed)
