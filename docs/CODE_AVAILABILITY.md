# Code Availability

This repository releases the components of SciXplain that can be shared without redistributing copyrighted or license-restricted assets.

## Publicly Released

- training code for the structural teacher, visual student, and hierarchical decoder
- inference and evaluation code for caption, description, explanation, and perturbation analysis
- dataset conversion helpers
- public manifest-building utilities
- reporting scripts used to aggregate metrics and diagnostic tables

## Not Redistributed

The following assets are intentionally excluded from the public release:

- project-specific curated benchmark payloads
- figure crops, OCR payloads, or derived annotations extracted from copyrighted papers
- redistributed copies of third-party benchmark assets
- trained project checkpoints and full released weights
- third-party baseline bundles whose redistribution is restricted by upstream licenses or packaging constraints

## Practical Reproduction Boundary

The code release supports local reconstruction of the training and evaluation pipeline. To reproduce the full workflow, a user must:

1. obtain legally accessible source documents locally
2. run local figure extraction, OCR, and any additional annotation or segmentation tooling
3. prepare split files in the expected repository format
4. acquire compatible open backbones locally
5. train and evaluate the released stages with local paths

This repository therefore provides a reproducible **implementation path**, while the benchmark payload and trained artifacts remain outside the release boundary.
