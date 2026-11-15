"""Adapter for pinned upstream YOLOv5; validation cap matches the prior baseline."""
import functools
import json
from pathlib import Path
import sys


if __name__ == '__main__':
    repo = Path(__file__).resolve().parent / 'vendor/yolov5'
    # Current Ultralytics Annotator stores a NumPy image, not a PIL image.
    # Keep the compatibility edit reproducible without downgrading YOLO26's environment.
    plots = repo / 'utils/plots.py'
    old = 'annotator.im.save(fname)  # save'
    new = 'Image.fromarray(annotator.result()).save(fname)  # compatibility: NumPy result'
    source = plots.read_text(encoding='utf-8')
    if old in source:
        plots.write_text(source.replace(old, new), encoding='utf-8')
    elif new not in source:
        raise RuntimeError('Unexpected YOLOv5 plotting source')
    sys.path.insert(0, str(repo))
    import train
    config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    (Path(config['project']) / config['name'] / 'compatibility.txt').write_text(
        'Plot save uses PIL Image.fromarray(Annotator.result()); validation max_det=400.\n')
    train.validate.run = functools.partial(train.validate.run, max_det=400)
    train.run(**config)
