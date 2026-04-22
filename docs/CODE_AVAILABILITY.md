# Code Availability

This public repository releases the parts of SciXplain that can be shared safely:

- training scripts
- inference scripts
- evaluation scripts
- dataset conversion helpers
- data crawling / manifest-building pipeline

The following assets are **not** included in the public release:

- raw SciStruct images or redistributed figure crops
- processed OCR payloads derived from copyrighted figures
- private checkpoints or trained weights
- third-party baseline bundles and their local wrappers

If you want to reproduce the full pipeline locally, prepare the data and checkpoints yourself, then point the scripts to your local paths with CLI flags or environment variables.

