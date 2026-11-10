import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_yolo import export


class ExportTests(unittest.TestCase):
    def fixture(self, root):
        for folder in ('GarbageDataset/images', 'GarbageDataset/labels',
                       'data/prepared/labels', 'data/split', 'reports/duplicate-review'):
            (root / folder).mkdir(parents=True)
        text = '0 0.5 0.5 0.2 0.2\n'
        digest = hashlib.sha256()
        rows = []
        for i, split in enumerate(('train', 'val', 'test')):
            Image.new('RGB', (8, 8), (i * 50, 0, 0)).save(root / f'GarbageDataset/images/{i}.png')
            (root / f'GarbageDataset/labels/{i}.txt').write_text(text + '0 0.5 0.5 0 0.2\n')
            (root / f'data/prepared/labels/{i}.txt').write_text(text)
            digest.update(f'{i}.txt'.encode() + repr([(0, .5, .5, .2, .2)]).encode())
            rows.append([f'{i}.png', f'{i}.txt', split, f'{i}.png'])
        with (root / 'data/split/manifest.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'label', 'split', 'duplicate_group'])
            writer.writerows(rows)
        review = root / 'reports/duplicate-review/review.csv'
        review.write_text('pair,image_a,image_b,decision\n1,0.png,1.png,different_images\n')
        summary = {'images': 3, 'review_sha256': hashlib.sha256(review.read_bytes()).hexdigest(),
                   'clean_label_content_sha256': digest.hexdigest(),
                   'splits': {s: {'images': 1, 'boxes_by_class_id': [1, 0, 0, 0, 0, 0],
                                   'images_by_class_id': [1, 0, 0, 0, 0, 0]}
                              for s in ('train', 'val', 'test')}}
        (root / 'data/split/summary.json').write_text(json.dumps(summary))

    def test_exports_cleaned_labels_and_preserves_fixed_assignments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            output = root / 'data/processed'
            result = export(root, root / 'data/split', output)
            self.assertEqual(result['verified_image_label_pairs'], 3)
            for i, split in enumerate(('train', 'val', 'test')):
                self.assertEqual((output / f'labels/{split}/{i}.txt').read_bytes(),
                                 (root / f'data/prepared/labels/{i}.txt').read_bytes())
                self.assertTrue((output / f'images/{split}/{i}.png').is_file())
            with self.assertRaises(ValueError):
                export(root, root / 'data/split', output)

    def test_rejects_changed_labels_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / 'data/prepared/labels/0.txt').write_text('0 0.5 0.5 0.3 0.3\n')
            output = root / 'data/processed'
            with self.assertRaisesRegex(ValueError, 'labels changed'):
                export(root, root / 'data/split', output)
            self.assertFalse(output.exists())
