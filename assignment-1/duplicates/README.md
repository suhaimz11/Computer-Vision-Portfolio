# Teammate's duplicate pipeline

From assignment-1, with your environment activated:

```powershell
python -m pip install -r requirements-audit.txt
python check_duplicates.py
python preview_duplicates.py
```

check_duplicates.py runs extraction, strict matching, strict verification, loose
matching and loose verification. Matching logic and thresholds follow the supplied
scripts. Local paths and subprocess startup support Windows.

Images must be 416x416 in GarbageDataset/images/. A fresh numbered folder under
reports/team-duplicates-run/ holds stage outputs. verified2.csv is the strict
search; verified3.csv is the loose search. These are scores, not final verdicts.

preview_duplicates.py selects unique pairs with at least 15 inliers from the
latest verified3.csv. Its first output goes in reports/duplicate-review/; reruns
use numbered folders and preserve existing decisions. The committed review has
364 candidate pairs and previews, with decisions pending.

_paths.py uses supplied v3 groups/unsure lists only for status labels. The
historical reference files were not supplied; status counts therefore differ
from the original baseline. Additional historical steps creating the final group
list were not supplied, so these scripts do not reproduce every historical step.

The original scripts, unused synthetic experiment and old verification CSV are
preserved locally under local-archive/. They are not needed for normal execution.
Do not automatically delete images or treat unflagged images as proven unique.
