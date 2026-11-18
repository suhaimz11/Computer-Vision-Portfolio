"""Paired bootstrap of the test-score difference between two checkpoints.

Both models are evaluated once on the same test split with the Ultralytics
validator (per-image matching records are kept, as in bootstrap_test_ci.py).
In every bootstrap round the SAME resampled set of test images is scored for
both models, and the difference final - baseline is recorded. Because both
models see identical images, image-to-image variation cancels out and the
interval of the difference is much narrower than two separate intervals.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap_test_ci import collect, metrics


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', default=str(base / 'models/yolo26s-best.pt'))
    ap.add_argument('--final', default=str(base / 'models/yolo26s-strongaug-best.pt'))
    ap.add_argument('--data', type=Path, default=base / 'data/processed/data.yaml')
    ap.add_argument('--device', default='0', help="'0' for CUDA, 'mps' for Apple GPU, 'cpu'")
    ap.add_argument('--resamples', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'results' / 'paired-bootstrap.json')
    a = ap.parse_args()

    stats = {}
    for name, weights in (('baseline', a.baseline), ('final', a.final)):
        stats[name], box = collect(weights, a.data, a.device)
        print(name, 'mAP50-95', round(float(box.map), 4))
    b, f = stats['baseline'], stats['final']
    if len(b) != len(f) or any(not np.array_equal(x['target_cls'], y['target_cls']) for x, y in zip(b, f)):
        raise RuntimeError('the two runs did not see the same images in the same order')

    n = len(b)
    keys = list(metrics(b, range(n)))
    point = {k: metrics(f, range(n))[k] - metrics(b, range(n))[k] for k in keys}
    rng = np.random.default_rng(a.seed)
    draws = {k: [] for k in keys}
    for _ in range(a.resamples):
        idx = rng.integers(0, n, n)
        mb, mf = metrics(b, idx), metrics(f, idx)
        for k in keys:
            draws[k].append(mf.get(k, np.nan) - mb.get(k, np.nan))
    out = {'baseline': Path(a.baseline).name, 'final': Path(a.final).name, 'test_images': n,
           'resamples': a.resamples, 'seed': a.seed, 'difference_final_minus_baseline': {}}
    for k in keys:
        d = np.array(draws[k], dtype=float)
        out['difference_final_minus_baseline'][k] = {
            'value': round(float(point[k]), 4),
            'ci95': [round(float(np.nanpercentile(d, 2.5)), 4), round(float(np.nanpercentile(d, 97.5)), 4)],
            'share_of_resamples_above_0': round(float(np.nanmean(d > 0)), 4)}
    a.out.write_text(json.dumps(out, indent=1))
    for k, v in out['difference_final_minus_baseline'].items():
        print(f"{k:24s} {v['value']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  P(>0)={v['share_of_resamples_above_0']}")


if __name__ == '__main__':
    main()
