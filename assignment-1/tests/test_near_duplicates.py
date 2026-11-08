"""Check radius-search completeness and hash behavior on controlled images."""
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import csv

import numpy as np
from PIL import Image

SPEC = importlib.util.spec_from_file_location('near_duplicates', Path(__file__).resolve().parents[1] / 'check_near_duplicates.py')
SCAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN)


class NearDuplicateTests(unittest.TestCase):
    def test_embedding_search_matches_exhaustive_cosine(self):
        values = np.random.default_rng(42).normal(size=(19, 12)).astype(np.float32)
        values[1] = values[0] * 2
        unit = values / np.linalg.norm(values, axis=1, keepdims=True)
        expected = {(i, j) for i in range(len(values)) for j in range(i+1, len(values))
                    if unit[i] @ unit[j] >= 0.5}
        for block_size in (1, 7, 100):
            found = SCAN.embedding_pairs(values, 0.5, block_size)
            self.assertEqual(set(found), expected)
            self.assertAlmostEqual(found[(0, 1)], 1, places=5)

    def test_all_candidates_get_previews_and_rerun_preserves_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'dataset' / 'images'
            images.mkdir(parents=True)
            for index in range(15):
                Image.new('RGB', (32, 32), 'red').save(images / f'{index}.png')
            output = root / 'results'
            output.mkdir()
            marker = output / 'keep.txt'
            marker.write_text('review notes')
            with patch.object(SCAN, 'extract_embeddings', return_value=(np.ones((15, 4), dtype=np.float32), {})):
                summary = SCAN.scan(root / 'dataset', output)
            self.assertEqual(summary['candidate_pairs'], 105)
            self.assertEqual(summary['previews'], 105)
            result = root / 'results-2'
            with (result / 'candidates.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 105)
            self.assertTrue(all((result / row['preview']).is_file() for row in rows))
            self.assertTrue(all(row['methods'] == 'phash+embedding' for row in rows))
            self.assertEqual(marker.read_text(), 'review notes')

    def test_tree_matches_exhaustive_search(self):
        rng = random.Random(42)
        hashes = [rng.getrandbits(64) for _ in range(100)]
        hashes += [hashes[0], hashes[0] ^ 63, hashes[3] ^ 127]
        for radius in (0, 6, 12, 64):
            expected = {(i, j, (a ^ b).bit_count()) for j, b in enumerate(hashes)
                        for i, a in enumerate(hashes[:j]) if (a ^ b).bit_count() <= radius}
            self.assertEqual(set(SCAN.find_pairs(hashes, radius)), expected)

    def test_same_image_and_mild_brightness_change(self):
        pixels = np.random.default_rng(42).integers(30, 200, (128, 128), dtype=np.uint8)
        image = Image.fromarray(pixels)
        value = SCAN.phash(image)
        self.assertEqual(value, SCAN.phash(image.copy()))
        brighter = SCAN.phash(Image.fromarray(pixels + 10))
        self.assertLessEqual((value ^ brighter).bit_count(), 6)


if __name__ == '__main__':
    unittest.main()
