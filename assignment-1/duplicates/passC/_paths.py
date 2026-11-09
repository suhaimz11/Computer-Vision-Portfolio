"""Local filesystem configuration; no descriptor/matching threshold changes."""
import csv
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
IMAGE_DIR = BASE / 'GarbageDataset' / 'images'
REFERENCE_DIR = BASE / 'data' / 'team-duplicates'
RUN_DIR = Path(os.environ.get('TEAM_DUPLICATE_RUN_DIR', str(BASE / 'reports' / 'team-duplicates-run'))).resolve()
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f'Dataset images folder missing: {IMAGE_DIR}')
RUN_DIR.mkdir(parents=True, exist_ok=True)
# Original scripts use relative filenames for stage-to-stage artifacts.
os.chdir(RUN_DIR)


def load_reference_labels():
    """Supplied v3 metadata affects status labels only, never matching scores."""
    groups, unsure = {}, set()
    path = REFERENCE_DIR / 'groups.csv'
    if path.exists():
        with path.open(newline='', encoding='utf-8-sig') as stream:
            groups = {row['image']: row['group_id'] for row in csv.DictReader(stream)}
    for name in ('unsure.csv', 'unsure_pass_c.csv', 'unsure_pass_d.csv'):
        path = REFERENCE_DIR / name
        if path.exists():
            with path.open(newline='', encoding='utf-8-sig') as stream:
                unsure.update(frozenset((row['image_a'], row['image_b'])) for row in csv.DictReader(stream))
    print('Status labels reference supplied v3 groups/unsure lists, not the missing historical baseline.', flush=True)
    return groups, unsure
