# Data Pipeline

This repository does not publish the benchmark payload itself. Instead, it releases the public part of the data-construction workflow needed to rebuild a compatible local corpus.

## Design Principles

- do not redistribute copyrighted PDFs, figure crops, or OCR payloads
- keep the downloadable part of the pipeline separate from project-specific curation artifacts
- allow local reconstruction from legally accessible sources

## Public Reconstruction Workflow

1. **Build an article manifest**
   - Create a JSONL manifest containing the document identifiers and public download information needed for local collection.
2. **Download PDFs from public sources**
   - Use the provided helpers for arXiv-compatible sources, or adapt the manifest to other legally accessible providers.
3. **Run local document processing**
   - Extract figures and surrounding text from PDFs.
   - Run OCR locally.
   - Run any region proposal, segmentation, or diagram-specific preprocessing locally.
4. **Filter and curate the local corpus**
   - Apply local filtering rules appropriate for the target diagram subset.
   - Perform local quality control and manual verification where required.
5. **Build split files and benchmark JSON/JSONL assets**
   - Export training, validation, and test files in the formats expected by the training and evaluation code in this repository.

## Included Scripts

- `pipelines/build_article_manifest.py`
  - normalizes input article metadata into a JSONL manifest
  - retains fields such as `article_id`, `source`, `arxiv_id`, `pdf_url`, and `title`

- `pipelines/download_arxiv_pdfs.py`
  - downloads PDFs from arXiv-compatible entries in the manifest
  - stores them locally without redistributing them

## Notes

- The repository does not package extracted figures.
- OCR results are not redistributed.
- The release does not expose the project-specific article list, exact curation range, or benchmark payload.
- Users should keep article manifests and local benchmark artifacts separate whenever venue policy, copyright policy, or source-site terms require that separation.
