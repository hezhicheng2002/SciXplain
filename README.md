# SciXplain

Public code release for **SciXplain**, which is currently under review of ICDAR 2026:

This repository contains the public training, inference, evaluation, and data curation pipeline.

## Repository Layout

- `scixplain/`: model code and task tools
- `scripts/`: training, evaluation, and reporting entry points
- `pipelines/`: public crawling helpers for manifest creation and PDF download
- `docs/`: release notes and data-availability details

## Quick Start

1. Create an environment with the dependencies used by your local training setup.
2. Put private checkpoints under `checkpoints/` or override them with CLI flags.
3. Export `PYTHONPATH` to the repo root.
4. Run the desired script under `scripts/`.

## Data Pipeline

The public curation entry points are documented in [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) and mirrored under [pipelines/README.md](pipelines/README.md).

## Availability

The release boundary is summarized in [docs/CODE_AVAILABILITY.md](docs/CODE_AVAILABILITY.md).
