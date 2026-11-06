"""Audit a local YOLO detection dataset without changing source files.

Usage: python assignment-1/audit_dataset.py
Requires: Pillow. Output: reports/audit.json and annotated_examples.jpg.
Class names must be confirmed from the dataset publisher before training.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import re
from PIL import Image, ImageDraw, ImageOps


def parse_box(line):
    """Validate one YOLO row and return (class ID, center x/y, width/height)."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError('expected class x_center y_center width height')
    cls = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    if cls < 0 or not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)):
        raise ValueError('invalid class or normalized coordinates')
    if w <= 0 or h <= 0:
        raise ValueError('zero-area bounding box')
    if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
        raise ValueError('box extends outside image')
    return cls, x, y, w, h


def duplicate_groups(groups):
    """Return exact duplicate groups; this does not detect near-duplicates."""
    return [names for names in groups.values() if len(names) > 1]


def audit(root, output):
    """Inspect source data and write reports without changing images or labels."""
    if not (root / 'images').is_dir() or not (root / 'labels').is_dir():
        raise ValueError(f'Dataset must contain images/ and labels/: {root}')
    images = sorted(p for p in (root / 'images').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'})
    if not images:
        raise ValueError(f'No supported images found in {root / "images"}')
    label_paths = sorted((root / 'labels').rglob('*.txt'))
    labels = {p.stem: p for p in label_paths}
    if len(labels) != len(label_paths) or len({p.stem for p in images}) != len(images):
        raise ValueError('Duplicate filename stems make image/label pairing ambiguous')
    sizes, classes, objects_per_image = Counter(), Counter(), Counter()
    class_images, prefixes = defaultdict(set), defaultdict(Counter)
    hashes, pixel_hashes = defaultdict(list), defaultdict(list)
    issues, missing, empty, examples, areas = [], [], [], defaultdict(list), []
    for image_index, path in enumerate(images):
        if image_index % 500 == 0:
            print(f'Checking {image_index}/{len(images)} images', flush=True)
        try:
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.name)
            with Image.open(path) as source:
                source.load()
                width, height = source.size
                sizes[f'{width}x{height}'] += 1
                rgb = source.convert('RGB')
                digest = hashlib.sha256(f'{width}x{height}'.encode() + rgb.tobytes()).hexdigest()
                pixel_hashes[digest].append(path.name)
        except Exception as exc:
            issues.append({'file': path.name, 'error': str(exc)})
            continue
        label = labels.get(path.stem)
        if label is None:
            missing.append(path.name)
            continue
        lines = label.read_text(encoding='utf-8-sig').splitlines()
        if not any(line.strip() for line in lines):
            empty.append(path.name)
        valid_boxes = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                cls, x, y, w, h = parse_box(line)
                classes[cls] += 1
                class_images[cls].add(path.name)
                prefixes[cls][re.split(r'\d', path.stem)[0]] += 1
                areas.append(w*h)
                valid_boxes.append((cls, x, y, w, h))
            except ValueError as exc:
                issues.append({'file': label.name, 'line': number, 'error': str(exc)})
        objects_per_image[len(valid_boxes)] += 1
        for cls in {box[0] for box in valid_boxes}:
            if len(examples[cls]) < 3:
                examples[cls].append((path, valid_boxes))
    summary = {
        'schema_version': 1,
        'annotation_format': 'YOLO normalized center x/y and width/height',
        'class_names_verified': False,
        'images': len(images), 'label_files': len(labels), 'image_dimensions': dict(sizes),
        'objects_by_class_id': dict(sorted(classes.items())),
        'images_by_class_id': {cls: len(names) for cls, names in sorted(class_images.items())},
        'filename_prefixes_by_class_id_NOT_verified_names': {cls: dict(counts) for cls, counts in sorted(prefixes.items())},
        'valid_objects_per_image': dict(sorted(objects_per_image.items())),
        'missing_labels': missing, 'empty_labels': empty,
        'orphan_labels': sorted(set(labels)-{p.stem for p in images}),
        'issues': issues, 'byte_duplicate_groups': duplicate_groups(hashes),
        'decoded_pixel_duplicate_groups': duplicate_groups(pixel_hashes),
        'box_area_fraction': {'minimum': min(areas, default=0), 'mean': sum(areas)/len(areas) if areas else 0,
                              'maximum': max(areas, default=0), 'below_1_percent': sum(a < .01 for a in areas)},
        'limitations': ['No authoritative class-name mapping supplied.',
                        'Exact duplicate checks do not detect near-duplicates.',
                        'Valid coordinates do not prove annotation completeness or correctness.',
                        'Example images are deterministic first examples, not a representative random sample.']
    }
    output.mkdir(parents=True, exist_ok=True)
    (output/'audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    rows = [(cls, example) for cls in sorted(examples) for example in examples[cls]]
    sheet = Image.new('RGB', (900, max(250, 250*((len(rows)+2)//3))), 'white')
    for index, (cls, (path, boxes)) in enumerate(rows):
        with Image.open(path) as source:
            tile = ImageOps.contain(source.convert('RGB'), (290, 215))
        draw = ImageDraw.Draw(tile)
        for box_cls, x, y, w, h in boxes:
            draw.rectangle(((x-w/2)*tile.width, (y-h/2)*tile.height,
                            (x+w/2)*tile.width, (y+h/2)*tile.height), outline='red', width=2)
            draw.text((max(0, (x-w/2)*tile.width), max(0, (y-h/2)*tile.height)), str(box_cls), fill='red')
        left, top = index%3*300, index//3*250
        sheet.paste(tile, (left, top+25))
        ImageDraw.Draw(sheet).text((left+5, top+5), f'Contains ID {cls}: {path.name[:24]}', fill='black')
    if rows:
        sheet.save(output/'annotated_examples.jpg')
    print(json.dumps({k: v for k, v in summary.items() if 'duplicate_groups' not in k and k != 'issues'}, indent=2))
    print(f'Issues: {len(issues)}; byte duplicate groups: {len(duplicate_groups(hashes))}; pixel duplicate groups: {len(duplicate_groups(pixel_hashes))}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).parent/'GarbageDataset')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'reports')
    args = parser.parse_args()
    audit(args.dataset, args.output)
