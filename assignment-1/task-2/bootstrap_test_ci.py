"""95% confidence intervals for the test metrics by bootstrapping test images.

The Ultralytics validator is run once on the test split. Its per-image matching
records (which predictions are correct at IoU 0.50:0.95, their confidences and
classes, and the ground-truth classes) are kept, and the metrics are recomputed
with Ultralytics' own ap_per_class on resampled sets of images. Whole images
are resampled, not boxes, because the boxes of one photo are not independent.
The point estimate is the metric on the original test split; the interval is
the 2.5th to 97.5th percentile of the bootstrap distribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ('biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic')


def collect(weights, data, device, imgsz=640, max_det=400):
    """Run the test evaluation once and return one stats dict per image."""
    from ultralytics import YOLO
    from ultralytics.utils.metrics import DetMetrics

    per_image = []
    original = DetMetrics.update_stats

    def recording(self, stat):
        per_image.append({k: np.asarray(stat[k]).copy() for k in ('tp', 'conf', 'pred_cls', 'target_cls')})
        return original(self, stat)

    DetMetrics.update_stats = recording
    try:
        box = YOLO(weights).val(data=str(data), split='test', imgsz=imgsz, max_det=max_det, device=device,
                                plots=False, verbose=False, save_json=False).box
    finally:
        DetMetrics.update_stats = original
    return per_image, box


def metrics(per_image, index):
    from ultralytics.utils.metrics import ap_per_class
    parts = [per_image[i] for i in index]
    cat = {k: np.concatenate([p[k] for p in parts], 0) for k in ('tp', 'conf', 'pred_cls', 'target_cls')}
    _, _, p, r, _, ap, classes, *_ = ap_per_class(cat['tp'], cat['conf'], cat['pred_cls'], cat['target_cls'])
    out = {'precision': p.mean(), 'recall': r.mean(), 'mAP50': ap[:, 0].mean(), 'mAP50-95': ap.mean()}
    for i, c in enumerate(classes):
        out[f'AP50-95 {NAMES[int(c)]}'] = ap[i].mean()
    return out


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=str(base / 'models/yolo26s-strongaug-best.pt'))
    ap.add_argument('--data', type=Path, default=base / 'data/processed/data.yaml')
    ap.add_argument('--device', default='0', help="'0' for CUDA, 'mps' for Apple GPU, 'cpu'")
    ap.add_argument('--resamples', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'results' / 'test-confidence-intervals.json')
    a = ap.parse_args()

    per_image, box = collect(a.weights, a.data, a.device)
    n = len(per_image)
    point = metrics(per_image, range(n))
    if abs(point['mAP50-95'] - float(box.map)) > 1e-6:
        raise RuntimeError(f"Recomputed mAP50-95 {point['mAP50-95']:.6f} != validator {float(box.map):.6f}")

    rng = np.random.default_rng(a.seed)
    samples = {k: [] for k in point}
    for _ in range(a.resamples):
        result = metrics(per_image, rng.integers(0, n, n))
        for k in samples:
            if k in result:
                samples[k].append(result[k])
    report = {'weights': Path(a.weights).name, 'test_images': n, 'resamples': a.resamples, 'seed': a.seed,
              'metrics': {k: {'value': round(float(point[k]), 4),
                              'ci95': [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
                          for k, v in samples.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for k, v in report['metrics'].items():
        print(f"{k:24s} {v['value']:.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")


if __name__ == '__main__':
    main()
