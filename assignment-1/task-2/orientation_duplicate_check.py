"""Search the test split for copies of training/validation images in all 8 orientations.

The Roboflow export contains flipped and rotated copies of the same photo. A
plain image hash only finds copies in the same orientation, so every test image
is hashed in all 8 orientations (4 rotations, each also mirrored) and compared
with the hash of every training and validation image.

1. Perceptual hash (64-bit DCT hash of a 32 x 32 greyscale image).
2. A test image is a candidate copy of a train/val image when one of its 8
   orientations is within --max-bits Hamming distance of that image.
3. Each candidate is verified by the normalised cross-correlation of the two
   64 x 64 greyscale images in the matching orientation (>= --min-ncc).

Verified pairs are compared with leaked-test-images.csv (the 23 test images
already known to have a copy). Recoloured copies are found as long as the
structure is unchanged; strongly cropped copies can be missed.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {'.jpg', '.jpeg', '.png'}


def orientations(gray):
    out = []
    for k in range(4):
        r = np.rot90(gray, k)
        out += [r, np.fliplr(r)]
    return out


def phash(gray):
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return np.packbits(bits).view('>u8')[0]


def popcount(x):
    x = x.astype(np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).reshape(x.shape)


def load(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def ncc(a, b):
    a = cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    b = cv2.resize(b, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, default=base / 'data/processed/images')
    ap.add_argument('--known', type=Path, default=Path(__file__).parent / 'results' / 'leaked-test-images.csv')
    ap.add_argument('--max-bits', type=int, default=10)
    ap.add_argument('--min-ncc', type=float, default=0.90)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'results' / 'orientation-duplicate-check.json')
    a = ap.parse_args()

    files = {s: sorted(p for p in (a.images / s).iterdir() if p.suffix.lower() in IMG_EXT)
             for s in ('train', 'val', 'test')}
    ref = files['train'] + files['val']
    ref_split = ['train'] * len(files['train']) + ['val'] * len(files['val'])
    ref_hash = np.array([phash(load(p)) for p in ref], dtype=np.uint64)

    pairs = []
    for t in files['test']:
        g = load(t)
        best = {}
        for o, og in enumerate(orientations(g)):
            d = popcount(np.bitwise_xor(ref_hash, np.uint64(phash(og))))
            for j in np.flatnonzero(d <= a.max_bits):
                if j not in best or d[j] < best[j][0]:
                    best[j] = (int(d[j]), o)
        for j, (bits, o) in best.items():
            score = ncc(orientations(g)[o], load(ref[j]))
            if score >= a.min_ncc:
                pairs.append({'test_image': t.name, 'copy': ref[j].name, 'copy_split': ref_split[j],
                              'orientation': o, 'hash_bits': bits, 'ncc': round(score, 3)})

    known = {r['test_image'] for r in csv.DictReader(open(a.known, encoding='utf-8'))}
    found = sorted({p['test_image'] for p in pairs})
    new = [x for x in found if x not in known]
    result = {'test_images': len(files['test']), 'reference_images': len(ref),
              'max_bits': a.max_bits, 'min_ncc': a.min_ncc,
              'test_images_with_a_copy': len(found),
              'already_known': len([x for x in found if x in known]),
              'known_not_refound': sorted(known - set(found)),
              'new_test_images': new,
              'orientation_counts': {str(o): sum(p['orientation'] == o for p in pairs) for o in range(8)},
              'pairs': pairs}
    a.out.write_text(json.dumps(result, indent=1))
    print({k: v for k, v in result.items() if k != 'pairs'})


if __name__ == '__main__':
    main()
