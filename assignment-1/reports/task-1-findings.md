# Task 1: initial dataset findings

## Measured results

The audit inspected every image and label in the local dataset.

- 10,464 images, all 416 x 416 pixels; all decoded successfully.
- 10,464 matching YOLO-format label files; no missing, empty, or orphan labels.
- 74,073 valid bounding boxes across six class IDs.
- 17 additional annotations have zero width or height, across 17 images.
- No byte-identical or decoded-pixel-identical images were found. Near-duplicates
  and related capture sequences have not yet been checked.
- 26,997 valid boxes (36.45%) occupy less than 1% of the image area. This is a
  descriptive relative-area threshold, not a COCO small-object metric.
- Images contain between 1 and 358 valid objects; 5,082 images contain one.

| Class ID | Valid boxes | Images containing class | Likely name from filename patterns (unverified) |
| --- | ---: | ---: | --- |
| 0 | 45,395 | 2,287 | biodegradable |
| 1 | 4,696 | 1,502 | cardboard |
| 2 | 7,809 | 2,684 | glass |
| 3 | 5,841 | 1,898 | metal |
| 4 | 4,387 | 1,768 | paper |
| 5 | 5,945 | 1,464 | plastic |

Image counts overlap because some images contain multiple classes. Filename
prefixes cannot establish the correct class of each object.

## Visual observations and limitations

The annotated contact sheet shows crowded food scenes, overlapping boxes,
small objects, and some stock-photo watermarks. Some nonzero class IDs appear
in food scenes, including class 2 or 3 boxes around parts of a sandwich scene.
These are candidates for manual review, not automatically proven errors:
confirm the authoritative ID mapping and inspect the original images first.
The sheet uses the first three images containing each class in filename order;
it is deliberately reproducible but not representative of the full dataset.

## Consequences for the pipeline

1. Preserve the original dataset. In a separate prepared copy, remove zero-area
   annotation rows with an explicit cleaning log; retain the other annotations
   in the same image. Do not silently relabel questionable objects.
2. Verify class names against original metadata. Review random samples per class
   and the suspicious mixed-class images before training.
3. Class 0 contributes 61.28% of valid boxes, although image counts are less
   imbalanced. Preserve multi-class coverage during splitting and report per-class
   AP and recall. Start without oversampling; consider it only if validation
   shows systematic minority-class failures.
4. Check perceptual similarity and source groups before a 70/15/15 split.
   Group related images to avoid leakage. Record the split manifest and seed.
5. Use a small pretrained YOLO detector if covered in class. Compare a 416 input
   baseline with 640 on validation data if compute permits. Upscaling adds no
   source detail, but may change model behavior. Small boxes and crowded images
   make aggressive downscaling and heavy cropping risky.
6. Start with moderate training-only geometric and color augmentation; visually
   validate transformed boxes. Validation and test use deterministic preprocessing.
7. Use validation mAP50-95 for checkpoint selection and early stopping. Evaluate
   the final checkpoint once on the held-out test set, with per-class metrics and
   separate examination of crowded scenes and small relative-area boxes.

Training settings and reproducible run commands are outlined in [the assignment README](../README.md).
This is the initial analysis and pipeline proposal. Metadata verification,
annotation review, near-duplicate checks, cleaning, and split creation remain
before the pipeline can be finalized. No model has been trained.

## Initial preview review

The student reported that all boxes displayed in the initial contact sheet fit
the visible objects. This records a visual placement review of those examples
only, not verified class correctness, annotation completeness, or whole-dataset
quality. The subsequent seeded random review is recorded below.

## Random sample placement review

The student confirmed reviewing all 30 panels generated with seed 42 (five
images per observed class ID, 30 unique images) and reported that boxes fit the
visible objects. Placement is recorded as pass for these sampled images in the
local review checklist. This is a student-reported visual assessment, not an
independent verification of every annotation. Class correctness remains
unverified, and annotation completeness was not explicitly assessed. The
review does not establish the quality of images outside the sample. The 17
zero-area rows identified by the full audit remain in the original dataset;
invalid rows were not drawn in the previews.

## Current assignment scope

The dataset is lecturer-provided. Use IDs 0-5 with the inferred working names
listed above; these names are not claimed to come from official metadata.
Near-duplicate analysis and further provenance investigation are outside scope,
superseding those prerequisites in the initial pipeline proposal above.
Next steps are logged removal of the 17 zero-area rows, reproducible dataset
splitting, and finalization of the training pipeline. No semantic corrections
were identified by the student's placement review. Related images across
splits remain a possible evaluation limitation.
