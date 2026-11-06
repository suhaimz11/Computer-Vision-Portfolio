"""Generate a deterministic random annotation review sample without editing data.

Run from either folder: python assignment-1/sample_annotations.py
Outputs are local previews plus a CSV checklist. Existing output is not replaced.
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random

from PIL import Image, ImageDraw
from audit_dataset import parse_box

COLORS = ('#ff3333', '#00a000', '#0066ff', '#ff9900', '#cc00cc', '#00aaaa')


def generate_review(dataset, output, per_class=5, seed=42):
    """Sample images per numeric class; an image may occur in multiple strata."""
    if per_class < 1:
        raise ValueError('per-class must be positive')
    if not (dataset / 'images').is_dir() or not (dataset / 'labels').is_dir():
        raise ValueError('Dataset must contain images/ and labels/')
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new --output to preserve review notes.')
    image_paths = sorted(p for p in (dataset / 'images').iterdir()
                         if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'})
    if len({p.stem for p in image_paths}) != len(image_paths):
        raise ValueError('Ambiguous image filename stems')
    images = {p.stem: p for p in image_paths}
    candidates = defaultdict(list)
    boxes_by_stem = {}
    invalid_by_stem = defaultdict(int)
    for index, label in enumerate(sorted((dataset / 'labels').glob('*.txt'))):
        if index % 2000 == 0:
            print(f'Reading labels: {index}', flush=True)
        if label.stem not in images:
            continue
        boxes = []
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            try:
                boxes.append(parse_box(line))
            except ValueError:
                invalid_by_stem[label.stem] += 1
        boxes_by_stem[label.stem] = boxes
        for cls in sorted({box[0] for box in boxes}):
            candidates[cls].append(label.stem)
    if not candidates:
        raise ValueError('No valid annotated images found')
    rng = random.Random(seed)
    selected = [(cls, stem) for cls in sorted(candidates)
                for stem in rng.sample(candidates[cls], min(per_class, len(candidates[cls])))]
    output.mkdir(parents=True)
    rows = []
    for index, (cls, stem) in enumerate(selected, 1):
        with Image.open(images[stem]) as source:
            original = source.convert('RGB')
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        for box_cls, x, y, w, h in boxes_by_stem[stem]:
            left, top = (x-w/2)*original.width, (y-h/2)*original.height
            right, bottom = (x+w/2)*original.width, (y+h/2)*original.height
            color = COLORS[box_cls % len(COLORS)]
            draw.rectangle((left, top, right, bottom), outline=color, width=2)
            draw.text((left+2, top+2), str(box_cls), fill='white', stroke_width=1, stroke_fill='black')
        panel = Image.new('RGB', (original.width*2, original.height+40), 'white')
        panel.paste(original, (0, 40))
        panel.paste(annotated, (original.width, 40))
        heading = f'{index:02d} | sample contains ID {cls} | original left, annotations right'
        ImageDraw.Draw(panel).text((8, 10), heading, fill='black')
        preview = f'{index:02d}_class_{cls}.jpg'
        panel.save(output / preview, quality=95)
        rows.append({'sample': index, 'sampling_class_id': cls, 'image': images[stem].name,
                     'preview': preview, 'valid_boxes': len(boxes_by_stem[stem]),
                     'invalid_rows_not_drawn': invalid_by_stem[stem], 'placement': 'pending',
                     'missing_objects': 'pending', 'class_correctness': 'unverified', 'notes': ''})
    with (output / 'review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    instructions = f'''# Random annotation review

Seed: {seed}. Up to {per_class} images sampled without replacement per observed
class ID. An image may appear in multiple class samples; this is not a uniform
sample of the entire dataset. No findings are pre-filled as reviewed.

Open the numbered JPG files. Original image: left. All valid boxes: right.
Numbers on boxes are numeric IDs, not verified material names. Invalid rows are
not drawn; their count is in review.csv. Source images are not modified.

For each CSV row, set placement to pass / concern / uncertain, missing_objects
to none_observed / concern / uncertain, and add concrete notes. Leave
class_correctness as unverified until official class metadata is available.
Do not infer correctness for the entire dataset from this small sample.

Keep these source-image previews local. To preserve review notes, the generator
refuses to overwrite an existing output directory. Use --output for another run.
'''
    (output / 'README.md').write_text(instructions, encoding='utf-8')
    print(f'Created {len(rows)} review panels ({len({stem for _, stem in selected})} unique images) in {output}')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).parent / 'GarbageDataset')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'reports' / 'random-review')
    parser.add_argument('--per-class', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate_review(args.dataset, args.output, args.per_class, args.seed)
