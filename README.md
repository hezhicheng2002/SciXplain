# SciXplain

Public code release for **SciXplain**, with naming aligned to the paper:

- `SciXplain`: the full method and repository name.
- `SciStruct`: the in-house structure-aware benchmark / explanation branch.
- `scixplain/`: the main Python package for training and evaluation code.

This repository contains the public training, inference, evaluation, and data-crawling utilities that can be released safely. It does **not** redistribute copyrighted figure data, pretrained checkpoints, third-party baselines, or model weights.

## Release Scope

Included:

- SciXplain training and inference scripts
- evaluation scripts and table-generation helpers
- public data crawling / manifest-building pipeline
- prompt builders and dataset conversion utilities

Not included:

- raw or processed SciStruct figure data
- PDF-derived figure crops or OCR payloads
- model weights / checkpoints
- bundled third-party baseline code or weights

## Repository Layout

- `scixplain/`: model code and task tools
- `scripts/`: training, evaluation, and reporting entry points
- `pipelines/`: public crawling helpers for manifest creation and PDF download
- `docs/`: release notes and data-availability details

## Naming Notes

Some local paths still use legacy processed dataset folder names such as `scicap_mlbcap_node_diagram_v2`. Those names are kept only where they refer to an existing processed asset layout. The public-facing benchmark name is `SciStruct`, and the old internal `SciFlow` name has been removed from the release code.

## Quick Start

1. Create an environment with the dependencies used by your local training setup.
2. Put private checkpoints under `checkpoints/` or override them with CLI flags.
3. Export `PYTHONPATH` to the repo root.
4. Run the desired script under `scripts/`.

If your private split files contain legacy absolute paths, set environment variables such as `SCISTRUCT_LEGACY_DATASET_ROOT`, `SCICAP_LEGACY_IMAGES`, or `SCICAP_LEGACY_IMAGES_STORE` before training.

## Data Pipeline

The public crawling entry points are documented in [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) and mirrored under [pipelines/README.md](pipelines/README.md).

## Availability

The release boundary is summarized in [docs/CODE_AVAILABILITY.md](docs/CODE_AVAILABILITY.md).

## Images

This repository intentionally ships without paper figures or benchmark images. For the public GitHub version, that is the right default: it avoids copyright redistribution and keeps the repo clean. If we later add a visual asset, it should be a newly redrawn workflow diagram rather than a paper figure or dataset example.
