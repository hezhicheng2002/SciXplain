# Data Pipeline

Because the source figures and PDFs have copyright restrictions, this repository does not publish the benchmark payload itself. Instead, it provides the public part of the collection pipeline.

## Public Steps

1. Build an article manifest.
2. Download PDFs from public sources such as arXiv.
3. Run your local figure extraction / OCR toolchain.
4. Build task-specific split files for SciXplain and SciStruct training.

## Included Scripts

- `pipelines/build_article_manifest.py`
  - normalizes article metadata into a JSONL manifest
  - keeps fields such as `article_id`, `source`, `arxiv_id`, `pdf_url`, and `title`

- `pipelines/download_arxiv_pdfs.py`
  - downloads PDFs from an input manifest
  - stores them under a local output directory without redistributing them

## Notes

- The public release does not package extracted figures.
- OCR results are not redistributed.
- You should keep article IDs and download manifests separate from the released benchmark payload when a venue or copyright policy requires that.

