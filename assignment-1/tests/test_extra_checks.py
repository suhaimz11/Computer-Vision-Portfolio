import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task-2'))
from orientation_duplicate_check import ncc, orientations, phash, popcount


def pattern(seed=0, size=96):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    img[10:60, 20:50] = 200
    img[50:80, 55:90] = 90
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)
    return img


class OrientationDuplicateTests(unittest.TestCase):
    def test_eight_distinct_orientations(self):
        img = pattern()
        outs = orientations(img)
        self.assertEqual(len(outs), 8)
        self.assertEqual(len({o.tobytes() for o in outs}), 8)

    def test_rotated_copy_found_in_some_orientation(self):
        img = pattern()
        copy = np.ascontiguousarray(np.fliplr(np.rot90(img, 1)))
        best = min(int(popcount(np.array([phash(o) ^ phash(copy)], dtype=np.uint64))[0])
                   for o in orientations(img))
        self.assertEqual(best, 0)

    def test_different_image_is_far(self):
        a, b = pattern(0), np.ascontiguousarray(pattern(1)[::-1, :].T)
        self.assertGreater(int(popcount(np.array([phash(a) ^ phash(b)], dtype=np.uint64))[0]), 0)

    def test_popcount(self):
        x = np.array([0, 1, 0xFF, 2 ** 64 - 1], dtype=np.uint64)
        self.assertEqual(popcount(x).tolist(), [0, 1, 8, 64])

    def test_ncc(self):
        img = pattern()
        self.assertAlmostEqual(ncc(img, img), 1.0, places=5)
        self.assertLess(ncc(img, 255 - img), 0)


if __name__ == '__main__':
    unittest.main()
