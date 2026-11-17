"""Rebuild the cleaned label copies in data/prepared/labels/ from the original dataset.

The cleaned copies are not stored in Git. They are the original label files with
exactly the 17 zero-area rows listed in data/prepared/removed_rows.txt removed.
Each listed row is checked against the original file before it is dropped, and
prepare_split.py verifies the result again.
"""
import argparse
from pathlib import Path


def read_log(log_path):
    """Map label file name -> {line number: row text} from 'file | line N | row' entries."""
    removals = {}
    for entry in log_path.read_text(encoding='utf-8-sig').splitlines():
        if not entry.strip():
            continue
        name, line, row = (part.strip() for part in entry.split('|', 2))
        number = int(line.removeprefix('line').strip())
        removals.setdefault(name, {})[number] = row
    return removals


def rebuild(dataset, log_path, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output is not empty; choose a fresh --output: {output}')
    originals = sorted((dataset / 'labels').glob('*.txt'))
    if not originals:
        raise ValueError(f'No label files found in {dataset / "labels"}')
    removals = read_log(log_path)
    missing = set(removals) - {p.name for p in originals}
    if missing:
        raise ValueError(f'Logged files not in the dataset: {sorted(missing)}')
    output.mkdir(parents=True, exist_ok=True)
    removed = 0
    for label in originals:
        lines = label.read_text(encoding='utf-8-sig').splitlines()
        drop = removals.get(label.name, {})
        for number, row in drop.items():
            if number < 1 or number > len(lines) or lines[number - 1].strip() != row:
                raise ValueError(f'{label.name} line {number} does not match the removal log')
        kept = [line for i, line in enumerate(lines, start=1) if i not in drop]
        (output / label.name).write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
        removed += len(drop)
    return len(originals), removed


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=base / 'GarbageDataset')
    parser.add_argument('--log', type=Path, default=base / 'data/prepared/removed_rows.txt')
    parser.add_argument('--output', type=Path, default=base / 'data/prepared/labels')
    args = parser.parse_args()
    files, removed = rebuild(args.dataset, args.log, args.output)
    print(f'Wrote {files} label files to {args.output}; removed {removed} logged rows.')


if __name__ == '__main__':
    main()
