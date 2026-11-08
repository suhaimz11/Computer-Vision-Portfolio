"""Read-only pHash + DINOv2 scan. Run: python check_near_duplicates.py."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


# Unnormalized DCT-II, matching the usual pHash low-frequency construction.
DCT = 2 * np.cos(np.pi * np.arange(8)[:, None] *
                 (2 * np.arange(32)[None, :] + 1) / 64)


def phash(image):
    pixels = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = DCT @ pixels @ DCT.T
    bits = (low > np.median(low)).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')


def find_pairs(hashes, threshold):
    """BK-tree radius search, including distinct files with identical hashes."""
    tree = None
    pairs = []
    for index, value in enumerate(hashes):
        if tree is None:
            tree = (value, [index], {})
            continue
        stack = [tree]
        while stack:
            other, indices, children = stack.pop()
            distance = (value ^ other).bit_count()
            if distance <= threshold:
                pairs.extend((previous, index, distance) for previous in indices)
            stack.extend(child for edge, child in children.items()
                         if distance-threshold <= edge <= distance+threshold)
        node = tree
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                break
            if distance not in node[2]:
                node[2][distance] = (value, [index], {})
                break
            node = node[2][distance]
    return sorted(pairs, key=lambda row: (row[2], row[0], row[1]))


def embedding_pairs(embeddings, threshold=0.90, block_size=256):
    """Exhaustively compare all distinct pairs, in bounded-memory blocks.

    No nearest-neighbor top-k cutoff: return EVERY pair above the threshold.
    """
    if not -1 <= threshold <= 1 or block_size < 1:
        raise ValueError('Invalid similarity threshold or block size')
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Embeddings must be a finite 2D matrix')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Zero-length embedding')
    values = values / norms
    pairs = {}
    for start in range(0, len(values), block_size):
        scores = np.clip(values[start:start+block_size] @ values.T, -1, 1)
        for offset, row in enumerate(scores):
            left = start + offset
            for right in np.flatnonzero(row[left+1:] >= threshold) + left + 1:
                pairs[(left, int(right))] = float(row[right])
        print(f'Compared embeddings {min(start+block_size, len(values))}/{len(values)}', flush=True)
    return pairs


def extract_embeddings(paths, batch_size=16):
    try:
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError('Install dependencies first: python -m pip install -r requirements-audit.txt') from exc
    model_id = 'facebook/dinov2-small'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading {model_id} on {device}. First run downloads model weights.', flush=True)
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start+batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            inputs = processor(images=images, return_tensors='pt').to(device)
            # CLS token: one global descriptor for each image.
            features = model(**inputs).last_hidden_state[:, 0, :]
            batches.append(features.cpu().float().numpy())
            print(f'Embedded {min(start+batch_size, len(paths))}/{len(paths)}', flush=True)
    return np.concatenate(batches), dict(model=model_id, device=device,
        model_revision=getattr(model.config, '_commit_hash', None),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        processor=processor.to_dict())


def scan(dataset, output, threshold=6, similarity=0.90):
    if not 0 <= threshold <= 64 or not -1 <= similarity <= 1:
        raise ValueError('Hash threshold must be 0..64; similarity must be -1..1')
    original_output = output
    run_number = 2
    while output.exists():
        output = original_output.with_name(f'{original_output.name}-{run_number}')
        run_number += 1
    print(f'Results folder: {output.resolve()}', flush=True)
    root = dataset / 'images'
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in
                   {'.jpg', '.jpeg', '.png', '.webp', '.bmp'})
    if not paths:
        raise ValueError(f'No images found: {root}')
    hashes = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            hashes.append(phash(image))
        if index % 1000 == 0:
            print(f'Hashed {index}/{len(paths)}', flush=True)
    hash_pairs = {(a, b): d for a, b, d in find_pairs(hashes, threshold)}
    print(f'pHash candidates: {len(hash_pairs)}. Starting embedding scan.', flush=True)
    embeddings, embedding_info = extract_embeddings(paths)
    semantic_pairs = embedding_pairs(embeddings, similarity)
    pairs = sorted(set(hash_pairs) | set(semantic_pairs), key=lambda p:
                   (0 if p in hash_pairs and p in semantic_pairs else 1,
                    -semantic_pairs.get(p, -1), hash_pairs.get(p, 65), p))
    output.mkdir(parents=True)
    previews = output / 'previews'
    previews.mkdir()
    np.save(output / 'embeddings.npy', embeddings)
    rows = []
    for rank, (left, right) in enumerate(pairs, 1):
        distance = (hashes[left] ^ hashes[right]).bit_count()
        cosine = float(np.dot(embeddings[left], embeddings[right]) /
                       (np.linalg.norm(embeddings[left]) * np.linalg.norm(embeddings[right])))
        methods = '+'.join(name for name, found in [('phash', (left, right) in hash_pairs),
                                                   ('embedding', (left, right) in semantic_pairs)] if found)
        preview = f'previews/{rank:05d}.jpg'
        rows.append(dict(pair=rank, image_a=paths[left].relative_to(root).as_posix(),
                         image_b=paths[right].relative_to(root).as_posix(),
                         distance=distance, cosine_similarity=f'{cosine:.6f}',
                         methods=methods, decision='pending', notes='', preview=preview))
        if preview:
            panel = Image.new('RGB', (832, 516), 'white')
            for column, idx in enumerate((left, right)):
                with Image.open(paths[idx]) as image:
                    tile = ImageOps.contain(image.convert('RGB'), (416, 416))
                panel.paste(tile, (column*416, 100))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 8), f'Pair {rank} | pHash {distance}/64 | cosine {cosine:.4f} | {methods}', fill='black')
            for y, label, idx in [(30, 'LEFT', left), (62, 'RIGHT', right)]:
                name = paths[idx].relative_to(root).as_posix()
                draw.text((8, y), f'{label}: {name[:105]}', fill='black')
                if len(name) > 105:
                    draw.text((8, y+13), name[105:210], fill='black')
            panel.save(output / preview, quality=90)
        if rank % 100 == 0:
            print(f'Previewed {rank}/{len(pairs)}', flush=True)
    with (output / 'candidates.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pair', 'image_a', 'image_b', 'distance', 'cosine_similarity', 'methods', 'decision', 'notes', 'preview'])
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'hashes.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'phash64'])
        writer.writerows((p.relative_to(root).as_posix(), f'{h:016x}') for p, h in zip(paths, hashes))
    summary = dict(images=len(paths), algorithm='64-bit grayscale DCT-II pHash, 32x32 resize, 8x8 coefficients, median including DC',
                   threshold=threshold, embedding_similarity_threshold=similarity,
                   embedding=embedding_info, candidate_pairs=len(pairs),
                   phash_pairs=len(hash_pairs), embedding_pairs=len(semantic_pairs),
                   both_methods=len(set(hash_pairs) & set(semantic_pairs)),
                   distance_counts=dict(sorted(Counter(hash_pairs.values()).items())),
                   previews=len(pairs), confirmed_pairs=0,
                   limitations=['Candidates are not confirmed duplicates.',
                                'All pairs compared, but only pairs passing either threshold are exported.',
                                'Thresholds are screening choices, not calibrated guarantees; visually similar objects can be false positives.',
                                'Crops, flips, rotations and changed viewpoints can still be missed.',
                                'No source images or labels were changed; no split was created.'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).parent / 'GarbageDataset')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'reports' / 'near-duplicates')
    parser.add_argument('--threshold', type=int, default=6)
    parser.add_argument('--similarity', type=float, default=0.90)
    args = parser.parse_args()
    scan(args.dataset, args.output, args.threshold, args.similarity)
