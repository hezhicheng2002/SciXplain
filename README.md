# SciXplain

This repository contains the **SciXplain** training, inference, evaluation, and data-processing code.

## Repository Layout

- `scixplain/`: core model code, prompts, and evaluation/training tools
- `scripts/`: stage-wise training, evaluation, and reporting entry points
- `pipelines/`: manifest-building helpers
- `docs/`: setup and data-pipeline notes

## Quick Start

1. Create an environment with the dependencies used by your local training setup.
2. Put the required checkpoints under your local paths or override them with CLI flags.
3. Export `PYTHONPATH` to the repository root.
4. Run the desired entry point under `scripts/`.

## Reproduction Overview

The repository supports local reconstruction of the training and evaluation workflow from locally prepared data and model assets.

Reproducing the pipeline requires three local ingredients:

1. **Source documents**
   - Assemble a local corpus of scientific papers that can be processed locally.
2. **Local data processing**
   - Extract figures and surrounding text from PDFs.
   - Run OCR and any additional region or segmentation tooling locally.
   - Build split files in the JSON or JSONL layouts expected by the training scripts.
3. **Local model preparation**
   - Obtain compatible open backbones locally.
   - Train the released stages with the scripts provided in this repository.

## Stage-Wise Entry Points

- **Teacher training**
  - `scixplain/tools/train_ai2d_teacher.py`
- **Visual-student training / ablation**
  - `scixplain/tools/train_visual_student.py`
  - `scripts/run_visual_student_ablation_gpu4.sh`
- **Hierarchical text-generation training**
  - `scripts/train_tinyllava_stage4a_scicap.sh`
  - `scripts/train_tinyllava_stage4b_scistruct_explain.sh`
  - `scripts/run_stage4_full_multilevel_unified_gpu4.sh`
- **Evaluation**
  - `scixplain/tools/eval_all_tasks_metrics.py`
  - `scixplain/tools/eval_desc_struct_consistency.py`
  - `scripts/run_explanation_perturb_live.sh`
  - `scripts/generate_multitable_report.py`

## Data Pipeline

The data-construction entry points are documented in:

- [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)
- [pipelines/README.md](pipelines/README.md)

These files describe how to construct local manifests and prepare a compatible local training corpus.

For a quick sanity check of local paths before training or evaluation, use:

```bash
bash scripts/check_scixplain_data_and_models.sh
```
