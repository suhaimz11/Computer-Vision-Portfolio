"""Per-class precision, recall, AP50 and AP50-95 of a checkpoint on the TEST split.

Runs the Ultralytics validator once (640 px, max_det 400, as in Task 2) and saves
the per-class values that Assignment 2 Task 2 quotes for the detector.
"""
import argparse
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=str(base / 'models/yolo26s-strongaug-best.pt'))
    ap.add_argument('--data', type=Path, default=base / 'data/processed/data.yaml')
    ap.add_argument('--device', default='0', help="'0' for CUDA, 'mps' for Apple GPU, 'cpu'")
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'results' / 'test-per-class-metrics.json')
    a = ap.parse_args()

    from ultralytics import YOLO
    box = YOLO(a.weights).val(data=str(a.data), split='test', imgsz=640, max_det=400, device=a.device,
                              plots=False, verbose=False).box
    names = ('biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic')
    out = {'weights': Path(a.weights).name, 'split': 'test',
           'overall': {'precision': round(float(box.mp), 4), 'recall': round(float(box.mr), 4),
                       'mAP50': round(float(box.map50), 4), 'mAP50-95': round(float(box.map), 4)},
           'per_class': {}}
    for i, name in enumerate(names):
        p, r, ap50, ap5095 = box.class_result(i)
        out['per_class'][name] = {'precision': round(float(p), 4), 'recall': round(float(r), 4),
                                  'AP50': round(float(ap50), 4), 'AP50-95': round(float(ap5095), 4)}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
