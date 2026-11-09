"""Run the teammate's duplicate pipeline locally: python check_duplicates.py."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def main():
    base = Path(__file__).resolve().parent
    missing = [name for name in ('cv2', 'numpy') if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit('Install dependencies first: python -m pip install -r requirements-audit.txt')
    if not (base / 'GarbageDataset' / 'images').is_dir():
        raise SystemExit('Place your images in assignment-1/GarbageDataset/images first.')
    output = base / 'reports' / 'team-duplicates-run'
    number = 2
    while output.exists():
        output = base / 'reports' / f'team-duplicates-run-{number}'
        number += 1
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment['TEAM_DUPLICATE_RUN_DIR'] = str(output)
    environment['PYTHONUNBUFFERED'] = '1'
    scripts = base / 'duplicates' / 'passC'
    stages = ('c1_extract.py', 'c2_match.py', 'c3b_verify.py',
              'c4_match_loose.py', 'c5_verify_loose.py')
    print(f'Results folder: {output}', flush=True)
    for number, name in enumerate(stages, 1):
        print(f'\nStage {number}/{len(stages)}: {name}', flush=True)
        subprocess.run([sys.executable, str(scripts / name)], cwd=output,
                       env=environment, check=True)
    print(f'\nFinished. Results: {output}', flush=True)
    print('verified2.csv = six-vote search; verified3.csv = three-vote search.')
    print('These contain pair scores, not automatically confirmed duplicates.')


if __name__ == '__main__':
    main()
