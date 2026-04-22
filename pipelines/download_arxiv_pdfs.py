#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dst.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="jsonl manifest from build_article_manifest.py")
    parser.add_argument("--out_dir", required=True, help="directory for downloaded PDFs")
    parser.add_argument("--sleep", type=float, default=1.0, help="delay between requests in seconds")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    count = 0
    skipped = 0

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            article_id = str(row.get("article_id") or row.get("arxiv_id") or "").strip()
            pdf_url = str(row.get("pdf_url") or "").strip()
            if not article_id or not pdf_url:
                skipped += 1
                continue
            dst = out_dir / f"{article_id}.pdf"
            if dst.exists() and not args.overwrite:
                skipped += 1
                continue
            _download(pdf_url, dst)
            count += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(json.dumps({"downloaded": count, "skipped": skipped, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
