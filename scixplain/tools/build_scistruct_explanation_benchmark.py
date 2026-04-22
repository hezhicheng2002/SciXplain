#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return " ".join(parts)
    return " ".join(str(value).split())


def _apply_replacements(path: str, reps: List[str]) -> str:
    out = path
    for r in reps:
        if "=" not in r:
            continue
        a, b = r.split("=", 1)
        out = out.replace(a, b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="SciStruct split json (e.g., dataset/SciStruct/dataset_split_811/test.json)")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument(
        "--path_replace",
        action="append",
        default=[],
        help="from=to replacement for figure/result paths, can be specified multiple times",
    )
    ap.add_argument("--target_field", choices=["figure_des", "figure_caption", "figure_info"], default="figure_des")
    args = ap.parse_args()

    src = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for art in src:
        article_id = str(art.get("article_id") or "")
        title = _flatten_text(art.get("title"))
        abstract = _flatten_text(art.get("abstract"))
        for j, fig in enumerate(art.get("figures", []) or []):
            fig_id = str(fig.get("figure_id") or f"fig{j}")

            target = _flatten_text(fig.get(args.target_field))
            if not target:
                # fallback: explanation text must be non-empty
                target = _flatten_text(fig.get("figure_des")) or _flatten_text(fig.get("figure_caption"))
            if not target:
                continue

            result_path = _flatten_text(fig.get("result_path"))
            figure_path = _flatten_text(fig.get("figure_path"))
            image_path = result_path or figure_path
            image_path = _apply_replacements(image_path, args.path_replace)
            # UID must be unique at sample level; article_id::figure_id can collide.
            image_tag = Path(image_path).stem if image_path else f"idx{j}"
            uid = f"{article_id}::{fig_id}::{image_tag}"

            ocr = _flatten_text(fig.get("figure_ocr"))
            paragraph = _flatten_text(fig.get("figure_info"))
            context = "\n".join(
                x
                for x in [
                    f"Title: {title}" if title else "",
                    f"Abstract: {abstract}" if abstract else "",
                    f"OCR context: {ocr}" if ocr else "",
                    f"Paragraph context: {paragraph}" if paragraph else "",
                ]
                if x
            )

            rows.append(
                {
                    "uid": uid,
                    "article_id": article_id,
                    "figure_id": fig_id,
                    "image_path": image_path,
                    "target_explanation": target,
                    "ocr": ocr,
                    "paragraph": paragraph,
                    "context": context,
                    "source": "SciStruct",
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(out_path), "rows": len(rows), "target_field": args.target_field}, ensure_ascii=False))


if __name__ == "__main__":
    main()
