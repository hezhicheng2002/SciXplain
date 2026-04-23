# Data Pipeline

This repository does not publish the benchmark payload itself. Instead, it releases the public part of the data-construction workflow needed to rebuild a compatible local corpus.

## Design Principles

- do not redistribute copyrighted documents, figure crops, or OCR payloads
- keep the released pipeline focused on local preparation steps
- allow local reconstruction from locally prepared sources

## Public Reconstruction Workflow

1. **Build an article manifest**
   - Create a JSONL manifest containing the document identifiers and metadata needed for local collection.
2. **Run local document processing**
   - Extract figures and surrounding text from local documents.
   - Run OCR locally.
   - Run any region proposal, segmentation, or diagram-specific preprocessing locally.
3. **Filter and curate the local corpus**
   - Apply local filtering rules appropriate for the target diagram subset.
   - Perform local quality control and manual verification where required.
4. **Build split files and benchmark JSON/JSONL assets**
   - Export training, validation, and test files in the formats expected by the training and evaluation code in this repository.

## Included Scripts

- `pipelines/build_article_manifest.py`
  - normalizes input article metadata into a JSONL manifest
  - retains fields such as `article_id`, `source`, `external_id`, and `title`

## Notes

- The repository does not package extracted figures.
- OCR results are not redistributed.
- The release does not expose the project-specific article list, exact curation range, or benchmark payload.
- Users should keep article manifests and local benchmark artifacts separate whenever venue or copyright policy requires that separation.
