"""Preview the latest SIFT run's >=15-inlier pairs: python preview_duplicates.py."""
import csv
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageOps


def main():
    base = Path(__file__).resolve().parent
    sources = list((base / 'reports').glob('team-duplicates-run*/verified3.csv'))
    if not sources:
        raise SystemExit('No SIFT results found. Run check_duplicates.py first.')
    source = max(sources, key=lambda path: path.stat().st_mtime)
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'a', 'b', 'inliers'} <= set(reader.fieldnames or []):
            raise ValueError(f'Unexpected result columns: {source}')
        candidates = [row for row in reader if int(row['inliers']) >= 15]
    if not candidates:
        raise SystemExit('No pairs meet the 15-inlier threshold.')
    root = (base / 'GarbageDataset' / 'images').resolve()
    rows = []
    seen = set()
    for source_row in candidates:
        key = tuple(sorted((source_row['a'], source_row['b'])))
        if key in seen:
            continue
        seen.add(key)
        for name in key:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Missing image or path outside dataset: {name}')
        number = len(rows)+1
        rows.append(dict(pair=number, image_a=source_row['a'], image_b=source_row['b'],
                         inliers=source_row['inliers'], votes=source_row.get('votes', ''),
                         scale=source_row.get('scale', ''), mirror=source_row.get('mirror', ''),
                         reference_status=source_row.get('status', ''),
                         decision='pending', notes='', preview=f'previews/{number:03d}.jpg'))
    output = base / 'reports' / 'duplicate-review'
    number = 2
    while output.exists():
        output = base / 'reports' / f'duplicate-review-{number}'
        number += 1
    (output / 'previews').mkdir(parents=True)
    with (output / 'review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        panel = Image.new('RGB', (1000, 610), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((10, 8), f"Pair {row['pair']} | {row['inliers']} inliers | candidate, not confirmed", fill='black')
        for column, field in enumerate(('image_a', 'image_b')):
            x = column*500
            draw.text((x+10, 30), 'LEFT' if column == 0 else 'RIGHT', fill='black')
            for line, label in enumerate(textwrap.wrap(row[field], width=62)):
                draw.text((x+10, 48+line*15), label, fill='black')
            with Image.open(root / row[field]) as image:
                tile = ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'), (480, 480))
            panel.paste(tile, (x+10+(480-tile.width)//2, 115+(480-tile.height)//2))
        panel.save(output / row['preview'], quality=95)
        if row['pair'] % 50 == 0:
            print(f"Created {row['pair']}/{len(rows)} previews", flush=True)
    (output / 'README.txt').write_text(
        f'Source: {source.relative_to(base).as_posix()}\nThreshold: inliers >= 15\n'
        'Open previews/ and enter same_source, different_images or uncertain in review.csv.\n'
        'Matching source pictures can be rotated, mirrored, cropped or recolored.\n'
        'Similar objects alone do not prove a duplicate. Leave unclear cases uncertain.\n'
        'Pair numbers apply only to this review; identify images by full filenames.\n'
        'Reference status is historical metadata, not a new confirmation.\n'
        'Unlisted images are not guaranteed unique. No source files were modified.\n', encoding='utf-8')
    print(f'Created {len(rows)} previews from {source}')
    print(f'Open: {output / "previews"}')
    print(f'Record decisions: {output / "review.csv"}')


if __name__ == '__main__':
    main()
