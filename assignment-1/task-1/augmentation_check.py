"""Check what the training augmentation does to object sizes, without training.

Training images are passed through the Ultralytics training data loader with
(a) the baseline augmentation of train_detector.py (mosaic off) and (b) the
same settings with mosaic on (probability 1.0, the Ultralytics default). For
every augmented sample the boxes are measured in pixels of the 640-pixel
network input, so the share of small objects the network actually sees can be
compared. A grid of augmented baseline samples is saved for visual inspection.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Baseline training augmentation, as in train_detector.py.
BASELINE = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, scale=0.2, translate=0.1, fliplr=0.5, flipud=0.0,
                degrees=0.0, shear=0.0, perspective=0.0, hsv_h=0.015, hsv_s=0.3, hsv_v=0.2)


def sizes(dataset, n, seed):
    rng = np.random.default_rng(seed)
    side = []
    for i in rng.choice(len(dataset), n, replace=False):
        sample = dataset[int(i)]
        h, w = sample['img'].shape[1:]
        wh = sample['bboxes'][:, 2:].numpy() * np.array([w, h])
        side.append(np.sqrt(wh[:, 0] * wh[:, 1]))
    side = np.concatenate(side)
    return {'boxes': int(len(side)), 'median_px': round(float(np.median(side)), 1),
            'below_16_px': round(float(np.mean(side < 16)), 4),
            'below_32_px': round(float(np.mean(side < 32)), 4)}


def build(images, data_yaml, overrides):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    cfg = get_cfg(overrides=dict(imgsz=640, **overrides))
    return build_yolo_dataset(cfg, images, 16, check_det_dataset(data_yaml), mode='train')


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=str(base / 'data/processed/data.yaml'))
    ap.add_argument('--samples', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=Path, default=Path(__file__).with_name('figures'))
    a = ap.parse_args()

    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.plotting import plot_images
    from ultralytics.data.dataset import YOLODataset
    images = check_det_dataset(a.data)['train']
    results = {}
    for name, overrides in (('no augmentation', {**BASELINE, 'scale': 0.0, 'translate': 0.0, 'fliplr': 0.0,
                                                 'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0}),
                            ('baseline (mosaic off)', BASELINE),
                            ('baseline + mosaic 1.0', {**BASELINE, 'mosaic': 1.0})):
        np.random.seed(a.seed)
        import random
        random.seed(a.seed)
        results[name] = sizes(build(images, a.data, overrides), a.samples, a.seed)
        print(name, results[name])

    np.random.seed(a.seed)
    dataset = build(images, a.data, BASELINE)
    picks = np.random.default_rng(a.seed).choice(len(dataset), 16, replace=False)
    batch = YOLODataset.collate_fn([dataset[int(i)] for i in picks])
    a.out.mkdir(parents=True, exist_ok=True)
    plot_images(labels=batch, images=batch['img'], paths=batch['im_file'],
                fname=a.out / 'augmented_training_batch.jpg',
                names=check_det_dataset(a.data)['names'])
    (a.out / 'augmentation_check.json').write_text(json.dumps(
        {'samples_per_setting': a.samples, 'seed': a.seed, 'input_px': 640, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
