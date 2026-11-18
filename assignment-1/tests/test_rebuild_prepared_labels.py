import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebuild_prepared_labels import rebuild


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dataset = root / 'GarbageDataset'
        (self.dataset / 'labels').mkdir(parents=True)
        (self.dataset / 'labels/a.txt').write_text('0 0.5 0.5 0.1 0.1\n0 0.4 0.4 0 0.1\n1 0.2 0.2 0.1 0.1\n')
        (self.dataset / 'labels/b.txt').write_text('2 0.5 0.5 0.3 0.3\n')
        self.log = root / 'removed_rows.txt'
        self.out = root / 'out'

    def tearDown(self):
        self.tmp.cleanup()

    def test_removes_only_logged_rows(self):
        self.log.write_text('a.txt | line 2 | 0 0.4 0.4 0 0.1\n')
        self.assertEqual(rebuild(self.dataset, self.log, self.out), (2, 1))
        self.assertEqual((self.out / 'a.txt').read_text(), '0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n')
        self.assertEqual((self.out / 'b.txt').read_text(), '2 0.5 0.5 0.3 0.3\n')

    def test_refuses_mismatched_log_row(self):
        self.log.write_text('a.txt | line 1 | 0 0.4 0.4 0 0.1\n')
        with self.assertRaises(ValueError):
            rebuild(self.dataset, self.log, self.out)

    def test_refuses_non_empty_output(self):
        self.log.write_text('')
        self.out.mkdir()
        (self.out / 'old.txt').write_text('x')
        with self.assertRaises(ValueError):
            rebuild(self.dataset, self.log, self.out)


if __name__ == '__main__':
    unittest.main()
