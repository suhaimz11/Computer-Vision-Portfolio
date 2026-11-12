import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_split import allocate, make_groups


class SplitTests(unittest.TestCase):
    def test_transitive_groups_and_contradiction(self):
        review = [dict(pair=1, image_a='a', image_b='b', decision='same_source'),
                  dict(pair=2, image_a='b', image_b='c', decision='same_source')]
        self.assertEqual(make_groups(['a', 'b', 'c', 'd'], review), [['a', 'b', 'c'], ['d']])
        review.append(dict(pair=3, image_a='a', image_b='c', decision='different_images'))
        with self.assertRaises(ValueError):
            make_groups(['a', 'b', 'c', 'd'], review)

    def test_complete_deterministic_split_without_group_leakage(self):
        groups = [[f'{i}a', f'{i}b'] for i in range(100)]
        counts = {n: np.ones(6, dtype=int) for g in groups for n in g}
        result = allocate(groups, counts, 42)
        self.assertEqual(result, allocate(groups, counts, 42))
        self.assertEqual(set(result), set(counts))
        for group in groups:
            self.assertEqual(len({result[n] for n in group}), 1)
        self.assertEqual([sum(v == s for v in result.values()) for s in ('train', 'val', 'test')], [140, 30, 30])


if __name__ == '__main__':
    unittest.main()
