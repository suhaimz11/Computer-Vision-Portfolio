"""Train original YOLOv5s with isolated data, logs and checkpoints."""
import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

BASE = Path(__file__).resolve().parent
REVISION = 'cb2f17fd5d790cfdd0c07c25351904df9affbe6e'


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='0')
    parser.add_argument('--smoke', action='store_true', help='One epoch on 24 train and 24 validation images')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error('Invalid epochs, batch or workers')
    repo = BASE / 'vendor/yolov5'
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise RuntimeError('YOLOv5 source revision differs from the recorded revision')
    name = ('smoke_' if args.smoke else 'yolov5s_') + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run = BASE / 'runs' / name
    data = BASE / 'data' / ('smoke' if args.smoke else 'full')
    config = dict(weights=str(BASE / 'weights/yolov5s.pt'), data=str(data / 'data.yaml'),
                  hyp=str(BASE / 'hyp.yaml'), epochs=1 if args.smoke else args.epochs,
                  batch_size=args.batch, imgsz=640, device=args.device, workers=args.workers,
                  optimizer='SGD', cos_lr=True, patience=20, seed=42,
                  project=str(BASE / 'runs'), name=name, exist_ok=True)
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return
    if not Path(config['weights']).is_file():
        raise FileNotFoundError('Download official v7.0 yolov5s.pt into yolov5/weights first')
    source = BASE.parent / 'data/processed'
    if not (data / 'data.yaml').exists():
        for split in ('train', 'val'):
            images = sorted((source / 'images' / split).glob('*'))
            if args.smoke:
                images = images[:24]
            if not images:
                raise ValueError(f'No source images for {split}')
            for kind in ('images', 'labels'):
                (data / kind / split).mkdir(parents=True, exist_ok=True)
            for image in images:
                shutil.copy2(image, data / 'images' / split / image.name)
                label = image.stem + '.txt'
                shutil.copy2(source / 'labels' / split / label, data / 'labels' / split / label)
        import yaml
        original = yaml.safe_load((source / 'data.yaml').read_text())
        dataset = dict(path=str(data), train='images/train', val='images/val',
                       nc=6, names=original['names'])
        (data / 'data.yaml').write_text(yaml.safe_dump(dataset), encoding='utf-8')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'launch.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    shutil.copy2(BASE / 'hyp.yaml', run / 'requested-hyp.yaml')
    shutil.copy2(data / 'data.yaml', run / 'dataset.yaml')
    shutil.copy2(BASE.parent / 'data/split/manifest.csv', run / 'source-manifest.csv')
    metadata = dict(source_revision=actual, smoke=args.smoke,
                    weights_sha256=hashlib.sha256(Path(config['weights']).read_bytes()).hexdigest(),
                    max_det=400, executable=sys.executable)
    (run / 'provenance.json').write_text(json.dumps(metadata, indent=2))
    (run / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', YOLOv5_AUTOINSTALL='false')
    with (run / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(BASE / 'worker.py'),
                                    str(run / 'launch.json')], cwd=repo, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            # Text-mode pipes turn progress-bar carriage returns into newlines.
            # Decode binary chunks instead, preserving terminal redraw controls.
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            while chunk := process.stdout.read1(8192):
                output = decoder.decode(chunk)
                log.write(output)
                log.flush()
                sys.stdout.write(output)
                sys.stdout.flush()
            tail = decoder.decode(b'', final=True)
            log.write(tail)
            sys.stdout.write(tail)
            sys.stdout.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    (run / 'exit-status.json').write_text(json.dumps({'exit_code': code}))
    print(f'Outputs: {run}')
    sys.exit(code)


if __name__ == '__main__':
    main()
