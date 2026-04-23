# SciXplain

This repository contains the **SciXplain** training, inference, evaluation, and data-processing code.

## Repository Layout

- `scixplain/`: core model code, prompts, and evaluation/training tools
- `scripts/`: stage-wise training, evaluation, and reporting entry points
- `pipelines/`: public manifest-building and PDF download helpers
- `docs/`: setup and data-pipeline notes

## Reproduction Overview

Reproducing the full pipeline requires three local ingredients:

1. **Legally accessible source documents**
   - Assemble a local corpus of scientific papers from sources that permit local download and processing.
2. **Local data processing**
   - Extract figures and surrounding text from PDFs.
   - Run OCR and any additional region/segmentation tooling locally.
   - Build split files in the JSON/JSONL layouts expected by the training scripts.
3. **Local model preparation**
   - Obtain compatible open backbones locally.
   - Train the released stages with the scripts provided in this repository.

The repository supports local reconstruction of the training and evaluation workflow from locally prepared data and model assets.

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

## Local Data Reconstruction

The public curation entry points are documented in:

- [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)
- [pipelines/README.md](pipelines/README.md)

These files describe how to construct local manifests, download public PDFs, and prepare a compatible local training corpus.

For a quick sanity check of local paths before training or evaluation, use:

```bash
bash scripts/check_scixplain_data_and_models.sh
```
