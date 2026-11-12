"""YOLO26s baseline. Use --dry-run to inspect settings without training."""
import argparse
import json
import os
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=base / 'data/processed/data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--name', default='yolo26s_baseline')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision for CUDA troubleshooting')
    parser.add_argument('--cuda-debug', action='store_true', help='Report CUDA failures synchronously')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or not 0 < args.fraction <= 1:
        parser.error('epochs/batch must be positive, imgsz >= 32, fraction in (0, 1]')
    if not args.data.is_file():
        parser.error(f'Dataset YAML not found: {args.data}')
    config = dict(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                  batch=args.batch, device=args.device, workers=args.workers,
                  fraction=args.fraction, project=str(base / 'runs/train'),
                  name=args.name, exist_ok=False, seed=42, deterministic=True,
                  patience=20, amp=not args.no_amp, pretrained=True, optimizer='SGD',
                  lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
                  warmup_epochs=3.0, cos_lr=True, mosaic=0.0, mixup=0.0,
                  copy_paste=0.0, scale=0.2, translate=0.1,
                  fliplr=0.5, flipud=0.0, degrees=0.0, shear=0.0,
                  perspective=0.0, hsv_h=0.015, hsv_s=0.3, hsv_v=0.2,
                  max_det=400, plots=True, val=True)
    print(json.dumps({'model': 'yolo26s.pt', 'train': config}, indent=2))
    if args.dry_run:
        return
    if args.cuda_debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import torch
    from ultralytics import YOLO
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Install GPU-compatible PyTorch or explicitly use --device cpu.')
        # Exercise a CUDA kernel before downloading weights or starting training.
        device = f'cuda:{args.device}'
        (torch.ones(4, device=device) * 2).sum().item()
    print(f'PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}')
    model = YOLO(str(base / 'yolo26s.pt'))
    model.train(**config)
    # Test evaluation is a separate action after validation-based model selection.


if __name__ == '__main__':
    main()
