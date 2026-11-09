# Duplicate pair review

This review contains 364 unique candidate pairs from the teammate's SIFT/RootSIFT
pipeline, using the loose-search output with at least 15 RANSAC inliers.
These pairs are not automatically confirmed duplicates. All decisions start pending.

Open the numbered JPG files in previews/ and edit review.csv:
- same_source: same source picture, including rotation, mirroring, crop or color edits.
- different_images: separate photographs, even if the objects look similar.
- uncertain: insufficient evidence to decide.

Keep pair numbers and filenames unchanged. Add short notes where helpful.
reference_status indicates membership in supplied historical lists, not confirmation.
The source scores are from reports/team-duplicates-run/verified3.csv (local cache).
Other images are not guaranteed unique. No source images or labels were deleted.
The repo includes these previews so teammates can review without rerunning the scan.
