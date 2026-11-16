"""Copy the TEST split (test.txt from the repo) out of the original dataset.

Usage (from the repo root):
  python assignment-1\make_team_test.py --list assignment-1\data\split\test.txt --src assignment-1\GarbageDataset --dst assignment-1\data\team_test
Creates images and labels folders in --dst (originals are not modified).
Zero-area label rows stay; error_analysis.py ignores them.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="test.txt with image filenames, one per line")
    ap.add_argument("--src", default="data/original/GarbageDataset")
    ap.add_argument("--dst", default="data/team_test")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    names = [l.strip() for l in Path(a.list).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    missing = []
    for n in names:
        img, lab = src / "images" / n, src / "labels" / (Path(n).stem + ".txt")
        if not img.exists() or not lab.exists():
            missing.append(n)
            continue
        shutil.copy2(img, dst / "images" / n)
        shutil.copy2(lab, dst / "labels" / lab.name)
    print(f"listed: {len(names)}  copied: {len(names) - len(missing)}  missing: {len(missing)}")
    if missing:
        print("first missing:", missing[:5])
        raise SystemExit(1)


if __name__ == "__main__":
    main()