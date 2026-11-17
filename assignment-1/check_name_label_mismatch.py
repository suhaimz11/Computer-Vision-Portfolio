"""List images whose most frequent label class differs from the class in the file name.

The file names start with a class word (for example glass349_jpg...). This
script compares it with the cleaned labels and writes every mismatch with its
split, so the cases can be reviewed. A mismatch is not automatically a label
error: a photo filed under one class can show mostly objects of another.
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

NAMES = ('biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic')


def main():
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', type=Path, default=base / 'data/prepared/labels')
    ap.add_argument('--manifest', type=Path, default=base / 'data/split/manifest.csv')
    ap.add_argument('--out', type=Path, default=base / 'reports/name-label-mismatches.csv')
    a = ap.parse_args()

    rows = []
    with a.manifest.open(encoding='utf-8', newline='') as stream:
        for r in csv.DictReader(stream):
            name_class = re.match(r'([a-z]+)\d+_', r['image']).group(1)
            lines = (a.labels / r['label']).read_text(encoding='utf-8').splitlines()
            counts = Counter(NAMES[int(line.split()[0])] for line in lines if line.strip())
            majority, _ = counts.most_common(1)[0]
            if majority != name_class:
                rows.append(dict(image=r['image'], split=r['split'], file_name_class=name_class,
                                 majority_label=majority, boxes=sum(counts.values()),
                                 boxes_of_file_name_class=counts.get(name_class, 0)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} images; {sum(r['boxes_of_file_name_class'] == 0 for r in rows)} without any box "
          f"of the file-name class; by split {dict(Counter(r['split'] for r in rows))}")
    print('most common:', Counter((r['file_name_class'], r['majority_label']) for r in rows).most_common(5))


if __name__ == '__main__':
    main()
