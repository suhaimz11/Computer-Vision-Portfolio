# Dataset provenance and class-mapping verification

Checked: 2026-11-06.

## Confirmed local provenance

The student confirmed that the lecturer supplied this dataset. The assignment
PDF links to this university-hosted distribution:

https://cloud.fiw.fhws.de/s/P9BGwMj7oAZrQ2R

The link was extracted from the PDF hyperlink annotation. Its remote contents
could not be inspected with the browsing tool, so the archive contents have not
been verified. The supplied local copy has only images/ and labels/; no class
configuration, source README, license file, or split manifests were found.

The audit establishes 10,464 images at 416 x 416 and six observed IDs (0-5).
It does not establish the original export's class ordering or license.

## Candidate upstream source (not a confirmed match)

Material Identification's GARBAGE CLASSIFICATION 3, version 2 (GC1):

https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2

The publisher's version page reports 10,464 images, stretch resizing to
416 x 416, and original splits of 7,324 training / 2,098 validation / 1,042 test
images. These properties make it a plausible upstream source, but matching
counts and dimensions alone do not establish identity.

The project overview lists CC BY 4.0 and seven classes, including CLOTH:

https://universe.roboflow.com/material-identification/garbage-classification-3

Our local data has six observed IDs. This discrepancy means that neither the
website's displayed class list nor its order should be copied into a training
configuration. The candidate project's license has not been established as the
license of the lecturer's exact copy. Record source attribution and applicable
license terms after confirming the connection.

## Provisional mapping from local filename patterns

| ID | Candidate name | Status |
| --- | --- | --- |
| 0 | biodegradable | Unverified |
| 1 | cardboard | Unverified |
| 2 | glass | Unverified |
| 3 | metal | Unverified |
| 4 | paper | Unverified |
| 5 | plastic | Unverified |

These associations are supported by dominant filename prefixes only. An image
can contain objects of several classes. Do not relabel objects based solely on
its filename, and do not mark this mapping verified without export metadata or
lecturer confirmation.

## Assignment scope decision

For this CV assignment, provenance is recorded as lecturer-provided. The team
will use the six-name mapping above as an inferred working mapping, without
claiming authoritative metadata verification. Further source investigation and
near-duplicate analysis are outside the current scope and do not block dataset
preparation. The candidate upstream source is background information only.

No original split membership is available locally. New splits will be made
reproducibly from the cleaned data. Near-duplicates across splits remain a
possible evaluation limitation.
