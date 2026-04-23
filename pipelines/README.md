# Public Collection Helpers

This directory contains the public part of the SciXplain collection pipeline.

- `build_article_manifest.py`
  - normalizes local article metadata into a JSONL manifest
- `download_arxiv_pdfs.py`
  - downloads PDFs from arXiv-compatible manifest entries into a local directory

These utilities are intended for **local corpus reconstruction** only. They do not redistribute benchmark assets, figure crops, OCR payloads, or project-specific curation outputs.
