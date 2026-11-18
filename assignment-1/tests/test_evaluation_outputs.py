import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


ASSIGNMENT = Path(__file__).resolve().parents[1]


def load_script(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ASSIGNMENT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_script('clean_evaluation', 'task-2/evaluate_clean_test.py')
analysis = load_script('detection_analysis', 'task-3/error_analysis.py')


class EvaluationOutputTests(unittest.TestCase):
    def test_subset_rerun_removes_excluded_files_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            for kind in ('images', 'labels'):
                (source / kind / 'test').mkdir(parents=True)
            for name in ('a', 'b'):
                (source / 'images/test' / f'{name}.jpg').write_bytes(name.encode())
                (source / 'labels/test' / f'{name}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
            output = root / 'output'

            evaluation.link_subset(source, output, lambda name: True)
            evaluation.link_subset(source, output, lambda name: name == 'a.jpg')

            self.assertEqual({p.name for p in (output / 'images/test').iterdir()}, {'a.jpg'})
            self.assertEqual({p.name for p in (output / 'labels/test').iterdir()}, {'a.txt'})
            self.assertEqual(evaluation.count_boxes(output), 1)
            self.assertEqual((source / 'images/test/b.jpg').read_bytes(), b'b')
            self.assertTrue((source / 'labels/test/b.txt').exists())

    def test_subset_falls_back_to_copy_when_hardlinks_are_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            for kind in ('images', 'labels'):
                (source / kind / 'test').mkdir(parents=True)
            (source / 'images/test/a.jpg').write_bytes(b'image')
            (source / 'labels/test/a.txt').write_text('0 0.5 0.5 0.2 0.2\n')
            output = root / 'output'

            with patch.object(Path, 'hardlink_to', side_effect=OSError('unsupported')):
                evaluation.link_subset(source, output, lambda name: True)

            self.assertEqual((output / 'images/test/a.jpg').read_bytes(), b'image')
            self.assertEqual(evaluation.count_boxes(output), 1)

    def test_subset_rejects_overlapping_source_and_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            with self.assertRaises(ValueError):
                evaluation.link_subset(source, source / 'output', lambda name: True)

    def test_example_rerun_replaces_previous_case_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'sample.jpg'
            cv2.imwrite(str(image), np.zeros((80, 80, 3), np.uint8))
            labels = root / 'labels'
            labels.mkdir()
            (labels / 'sample.txt').write_text('0 0.5 0.5 0.5 0.5\n')
            boxes = np.array([[0.25, 0.25, 0.75, 0.75]])
            classes = np.array([0])
            predictions = [(image, boxes, classes, np.array([0.9]))]
            output = root / 'predictions'
            metrics = root / 'metrics'
            analysis.run(predictions, labels, output, metrics)
            self.assertTrue((output / 'correct/sample.jpg').exists())

            analysis.run([(image, np.empty((0, 4)), np.array([], int), np.array([]))],
                         labels, output, metrics)

            self.assertEqual(list((output / 'correct').glob('*.jpg')), [])
            self.assertTrue((output / 'missed/sample.jpg').exists())


if __name__ == '__main__':
    unittest.main()
