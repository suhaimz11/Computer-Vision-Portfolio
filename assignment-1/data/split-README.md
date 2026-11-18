# Reproducible data split

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe assignment-1/prepare_split.py
```

Outputs are written to `data/split/`. The script refuses to overwrite an existing output directory. To reproduce into
a fresh directory, pass `--output assignment-1/data/split-check`.

`manifest.csv` assigns every source image exactly once. `train.txt`, `val.txt`
and `test.txt` contain image filenames, not YOLO training paths. Images remain
in `GarbageDataset/images/`; verified cleaned annotations remain in
`data/prepared/labels/`. No source files are deleted or changed.

Targets are 70/15/15 with seed 42. A greedy allocation balances image counts,
per-class image presence and box counts. All connected components of reviewed
`same_source` pairs stay together. Separate photographs may still share a split.
Only the completed current review supplies duplicate links; historical status
labels are not used as verdicts. Undetected duplicate sources remain possible.

Final image counts: train 7,325; validation 1,570; test 1,569. All six class IDs
appear in each subset. There are 267 non-singleton duplicate groups (largest:
five images), with no confirmed group crossing subsets. Duplicates found only by
the wider team search are not grouped: 44 such groups cross subsets (23 test images have
a copy in training or validation, 26 validation images a copy in training); their effect on the test score
is reported in [Task 2](../task-2/README.md#test-score-without-leaked-duplicates-baseline). All 10,464 images are
retained. The 74,073 valid boxes are preserved; only 17 zero-area rows are removed.

`summary.json` records actual counts, class coverage, dependency version and
input fingerprints. All six class IDs must occur in every subset. The prepared
labels are checked against the originals and the supplied removal log before
the split is written. Class names still need confirmation from the publisher.

## YOLO export

From `assignment-1`, run `python export_yolo.py`. From the repository root,
run `python assignment-1/export_yolo.py`.

This copies the fixed manifest's images and verified cleaned labels into
`data/processed/images/{train,val,test}` and matching `labels/` folders.
`data/processed/data.yaml` configures those directories. The exporter verifies
the saved input fingerprints, class counts and duplicate isolation, then checks
every copied file using SHA-256. It also saves the manifest, split summary and
an `export-report.json`. Existing destinations are refused; a failed partial
export has no completion report. Use `--output` for a fresh destination.

The dataset copies and local verification split are excluded from Git. Source
data stays unchanged. YAML class names follow the export's own `data.yaml`
(biodegradable, cardboard, glass, metal, paper, plastic for ids 0 to 5). Keep the test subset fixed and use
validation only for tuning and checkpoint selection.
