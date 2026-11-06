"""Regression checks for annotation validation and dataset pairing.

Run: python -m unittest discover -s assignment-1/tests -v
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'audit_dataset.py'
SPEC = importlib.util.spec_from_file_location('audit_dataset', SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class DatasetAuditTests(unittest.TestCase):
    def test_invalid_annotations_are_rejected(self):
        for row in ('0 0.5 0.5 0 0.2', '0 0.5 0.5 0.2 0',
                    '0 nan 0.5 0.2 0.2', '0 0.95 0.5 0.2 0.2',
                    '-1 0.5 0.5 0.2 0.2', '0 0.5 0.5'):
            with self.subTest(row=row), self.assertRaises(ValueError):
                AUDIT.parse_box(row)
        self.assertEqual(AUDIT.parse_box('2 0.5 0.5 1 1'), (2, .5, .5, 1., 1.))

    def test_audit_reports_problems_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'dataset'
            images, labels = root / 'images', root / 'labels'
            images.mkdir(parents=True)
            labels.mkdir()
            for name in ('a', 'b', 'missing'):
                Image.new('RGB', (32, 32), 'blue').save(images / f'{name}.png')
            (images / 'broken.png').write_bytes(b'not an image')
            (labels / 'a.txt').write_text('0 0.5 0.5 0.5 0.5\n0 0.5 0.5 0 0.1\n')
            (labels / 'b.txt').write_text('')
            (labels / 'orphan.txt').write_text('1 0.5 0.5 0.5 0.5')
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            output = Path(temporary) / 'reports'
            with contextlib.redirect_stdout(io.StringIO()):
                report = AUDIT.audit(root, output)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(report['objects_by_class_id'], {0: 1})
            self.assertEqual(report['missing_labels'], ['missing.png'])
            self.assertEqual(report['empty_labels'], ['b.png'])
            self.assertEqual(report['orphan_labels'], ['orphan'])
            self.assertEqual(len(report['issues']), 2)
            self.assertEqual(len(report['decoded_pixel_duplicate_groups'][0]), 3)
            self.assertEqual(json.loads((output / 'audit.json').read_text())['images'], 4)
            self.assertTrue((output / 'annotated_examples.jpg').is_file())

    def test_missing_dataset_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'images/ and labels/'):
                AUDIT.audit(Path(temporary), Path(temporary) / 'reports')


if __name__ == '__main__':
    unittest.main()
