"""YOLO26s baseline. Use --dry-run to inspect settings without training."""
import argparse
import json
import os
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=base / 'data/processed/data.yaml')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0', help="CUDA index such as '0', 'mps' (Apple GPU) or 'cpu'")
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--name', default='yolo26s_baseline')
    parser.add_argument('--model', type=Path, default=base / 'yolo26s.pt',
                        help='Pretrained weights or a completed best.pt for a new fine-tuning run')
    parser.add_argument('--optimizer', choices=['SGD', 'AdamW', 'MuSGD'], default='SGD')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--mosaic', type=float, default=0.0)
    parser.add_argument('--close-mosaic', type=int, default=10)
    # Augmentation; the defaults are the baseline used for the submitted model.
    parser.add_argument('--scale', type=float, default=0.2)
    parser.add_argument('--flipud', type=float, default=0.0)
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--hsv-s', type=float, default=0.3)
    parser.add_argument('--hsv-v', type=float, default=0.2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision for CUDA troubleshooting')
    parser.add_argument('--cuda-debug', action='store_true', help='Report CUDA failures synchronously')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or not 0 < args.fraction <= 1:
        parser.error('epochs/batch must be positive, imgsz >= 32, fraction in (0, 1]')
    if not args.data.is_file():
        parser.error(f'Dataset YAML not found: {args.data}')
    if (args.lr0 <= 0 or args.warmup_bias_lr < 0 or args.patience < 0
            or not 0 <= args.mosaic <= 1 or args.close_mosaic < 0
            or not 0 <= args.scale < 1 or not 0 <= args.flipud <= 1 or args.degrees < 0
            or not 0 <= args.hsv_s <= 1 or not 0 <= args.hsv_v <= 1):
        parser.error('Invalid learning rate, patience, or mosaic settings')
    if not args.model.is_file():
        parser.error(f'Model checkpoint not found: {args.model}')
    config = dict(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                  batch=args.batch, device=args.device, workers=args.workers,
                  fraction=args.fraction, project=str(base / 'runs/train'),
                  name=args.name, exist_ok=False, seed=42, deterministic=True,
                  patience=args.patience, amp=not args.no_amp, pretrained=True, optimizer=args.optimizer,
                  lr0=args.lr0, lrf=0.01, momentum=0.937, weight_decay=0.0005,
                  warmup_epochs=3.0, warmup_bias_lr=args.warmup_bias_lr,
                  cos_lr=True, mosaic=args.mosaic, close_mosaic=args.close_mosaic, mixup=0.0,
                  copy_paste=0.0, scale=args.scale, translate=0.1,
                  fliplr=0.5, flipud=args.flipud, degrees=args.degrees, shear=0.0,
                  perspective=0.0, hsv_h=0.015, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
                  max_det=400, plots=True, val=True)
    print(json.dumps({'model': str(args.model.resolve()), 'train': config}, indent=2))
    if args.dry_run:
        return
    if args.cuda_debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import torch
    from ultralytics import YOLO
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('Apple MPS unavailable on this machine.')
    elif args.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Install GPU-compatible PyTorch or explicitly use --device cpu.')
        # Exercise a CUDA kernel before downloading weights or starting training.
        device = f'cuda:{args.device}'
        (torch.ones(4, device=device) * 2).sum().item()
    print(f'PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}')
    model = YOLO(str(args.model.resolve()))
    model.train(**config)
    # Test evaluation is a separate action after validation-based model selection.


if __name__ == '__main__':
    main()
