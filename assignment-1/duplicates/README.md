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

## What each stage does

The stage files keep the teammate's short names; renaming them would break the
launcher and earlier run folders. All stages run inside one results folder.

| Stage | Script | Method | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 | `c1_extract.py` | Up to 200 SIFT keypoints per image, RootSIFT descriptors, also for the mirrored image (to catch flipped copies) | `GarbageDataset/images/` | `sift.pkl` |
| 2 | `c2_match.py` | Approximate nearest-neighbour search (FLANN) over all descriptors with Lowe's ratio test 0.8; an image pair is a candidate with at least **6** matching keypoints (strict) | `sift.pkl` | `cand.pkl` |
| 3 | `c3b_verify.py` | Brute-force matching of each candidate pair, ratio test, keypoints within 45 px of the border ignored, RANSAC similarity transform (`estimateAffinePartial2D`, 4 px tolerance); score = inliers, set to 0 if they cover less than 12% of the image | `sift.pkl`, `cand.pkl` | `verified2.csv` |
| 4 | `c4_match_loose.py` | As stage 2 with at least **3** matching keypoints (loose), so more candidates | `sift.pkl` | `cand3.pkl` |
| 5 | `c5_verify_loose.py` | As stage 3 on the loose candidates | `sift.pkl`, `cand3.pkl` | `verified3.csv` |
| review | `preview_duplicates.py` | Side-by-side previews of every loose pair with at least 15 inliers for manual review | `verified3.csv` | `reports/duplicate-review/` (364 pairs, decisions in `review.csv`) |

`_paths.py` holds the dataset path and loads the supplied reference labels.
The CSV columns are the two image indices, keypoint votes, verified inliers,
estimated scale, whether the mirrored image matched, and the reference status.

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
