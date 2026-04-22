#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def _read_rows(path: Path) -> Iterator[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield row
        elif isinstance(data, dict):
            for row in data.get("items", []):
                if isinstance(row, dict):
                    yield row
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield dict(row)
        return
    raise ValueError(f"unsupported input format: {path}")


def _clean(text: object) -> str:
    return " ".join(str(text or "").split())


def _normalize(row: Dict, source: str) -> Dict:
    article_id = _clean(row.get("article_id") or row.get("paper_id") or row.get("id"))
    arxiv_id = _clean(row.get("arxiv_id") or row.get("arxiv") or row.get("identifier"))
    pdf_url = _clean(row.get("pdf_url"))
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return {
        "article_id": article_id or arxiv_id,
        "source": _clean(row.get("source") or source),
        "arxiv_id": arxiv_id,
        "pdf_url": pdf_url,
        "title": _clean(row.get("title")),
    }


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input .json, .jsonl, or .csv file")
    parser.add_argument("--output", required=True, help="output manifest jsonl")
    parser.add_argument("--source", default="arxiv", help="default source label")
    args = parser.parse_args()

    rows: List[Dict] = []
    for row in _read_rows(Path(args.input)):
        norm = _normalize(row, args.source)
        if norm["article_id"] and (norm["arxiv_id"] or norm["pdf_url"]):
            rows.append(norm)

    count = _write_jsonl(Path(args.output), rows)
    print(json.dumps({"output": args.output, "rows": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()

