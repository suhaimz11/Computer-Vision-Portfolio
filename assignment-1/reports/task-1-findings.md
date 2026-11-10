# Task 1: Dataset findings

## Audit results

- 10,464 images, all 416 x 416 pixels. All opened successfully.
- 10,464 matching label files, with no missing, empty, or orphan labels.
- 74,073 valid bounding boxes across six class IDs.
- 17 additional rows with zero-width or zero-height boxes.
- No exact image duplicates, checked by file contents and decoded pixels.
- 26,997 boxes (36.45%) cover less than 1% of an image. This is a relative-area
  threshold, not the COCO definition of a small object.
- Between 1 and 358 valid objects per image; 5,082 images contain one object.

## Class distribution

| ID | Working class name | Valid boxes | Images containing class |
| --- | --- | ---: | ---: |
| 0 | biodegradable | 45,395 | 2,287 |
| 1 | cardboard | 4,696 | 1,502 |
| 2 | glass | 7,809 | 2,684 |
| 3 | metal | 5,841 | 1,898 |
| 4 | paper | 4,387 | 1,768 |
| 5 | plastic | 5,945 | 1,464 |

Class names are inferred from filename patterns. Images can contain multiple
classes, so image counts overlap. The filename alone does not determine the
class of every object in an image.

## Visual review

We reviewed the initial preview and all 30 random sample images (five per class,
seed 42). The boxes appeared to fit the visible objects. The random sample
contained 30 unique images. No placement corrections were identified.

The images include crowded scenes, overlapping objects, small objects, and some
watermarks. The review covers box placement in these examples; it does not
confirm every class label or establish that all objects have been annotated.
Invalid rows were not drawn in the previews.

## Training implications

- Remove the 17 zero-area rows in a separate copy, keeping all valid boxes and
  recording the removed rows. The original data has not been changed.
- Class 0 accounts for 61.28% of valid boxes. Check class coverage in each split
  and report per-class AP and recall alongside overall metrics.
- Many objects are small or crowded. Avoid aggressive downscaling and inspect
  augmented images to make sure objects remain recognizable.
- Start with a small pretrained YOLO detector and 416-pixel inputs. A 640-pixel
  validation comparison is optional; upscaling adds no source detail.
- Use training-only augmentation and validation data for checkpoint selection.
  Reserve the test set for final evaluation.

The current teammate-based SIFT/RootSIFT loose search produced 364 candidate
pairs with at least 15 RANSAC inliers. All pairs have been reviewed: 341
`same_source` and 23 `different_images`; see
[review instructions](duplicate-review/README.txt) and the accompanying previews.
Historical team groups are supplied separately as reference metadata. No list
proves that all other images are unique. Keep confirmed related images together
in the split. `prepare_split.py` verifies cleaned labels and assigns connected
duplicate groups together, targeting 70/15/15 with seed 42. Actual counts and
class coverage are recorded in `data/split/summary.json`. Training remains pending.
