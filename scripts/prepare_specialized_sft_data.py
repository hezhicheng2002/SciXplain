#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from scixplain.prompts import (
    build_caption_long_prompt,
    build_caption_short_prompt,
    build_description_prompt,
    build_explanation_prompt,
)


def _flatten_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        s = x
    elif isinstance(x, list):
        parts = []
        for it in x:
            t = _flatten_text(it)
            if t:
                parts.append(t)
        s = " ".join(parts)
    else:
        s = str(x)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _flatten_maybe_json_text(x) -> str:
    if not isinstance(x, str):
        return _flatten_text(x)
    s = x.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            return _flatten_text(obj)
        except Exception:
            pass
    return _flatten_text(x)


def _first_sentence(text: str) -> str:
    s = _flatten_text(text)
    if not s:
        return ""
    m = re.search(r"[.!?]\s+", s)
    if not m:
        return s
    out = s[: m.end()].strip()
    return out if len(out) >= 12 else s


def _clean_ocr_items(x, max_items: int = 96) -> List[str]:
    if x is None:
        return []
    raw: List[str] = []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    raw = [str(it) for it in obj]
                else:
                    raw = [s]
            except Exception:
                raw = [s]
        else:
            raw = re.split(r"[;\n]", s)
    elif isinstance(x, list):
        raw = [str(it) for it in x]
    else:
        raw = [str(x)]
    out: List[str] = []
    seen = set()
    for it in raw:
        t = re.sub(r"\s+", " ", str(it or "")).strip()
        if not t:
            continue
        k = re.sub(r"[^a-z0-9]", "", t.lower())
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= max_items:
            break
    return out


def _apply_path_replacements(path: str, replacements: List[Tuple[str, str]]) -> str:
    p = _flatten_text(path)
    if not p:
        return ""
    for src, dst in replacements:
        if p.startswith(src):
            return p.replace(src, dst, 1)
    return p


def _build_prompt(task: str, ex_ctx: Dict) -> str:
    if task == "caption_short":
        return build_caption_short_prompt(ex_ctx)
    if task == "caption_long":
        return build_caption_long_prompt(ex_ctx)
    if task == "description":
        return build_description_prompt(ex_ctx)
    if task == "explanation":
        return build_explanation_prompt(ex_ctx)
    raise ValueError(f"unknown task: {task}")


def _append_sample(
    dst: List[Dict],
    *,
    split: str,
    source: str,
    uid: str,
    image_path: str,
    task: str,
    target: str,
    paragraph: str,
    ocr: str,
) -> None:
    target = _flatten_text(target)
    if not target:
        return
    ex_ctx = {"paragraph": paragraph, "ocr": ocr}
    prompt = _build_prompt(task, ex_ctx)
    dst.append(
        {
            "uid": uid,
            "split": split,
            "source": source,
            "task": task,
            "image_path": image_path,
            "paragraph": paragraph,
            "ocr": ocr,
            "prompt": prompt,
            "target": target,
        }
    )


def _iter_scicap(split_json: Path, split: str, path_replace: List[Tuple[str, str]]) -> List[Dict]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    rows: List[Dict] = []
    for art in data:
        article_id = _flatten_text(art.get("article_id"))
        for i, fig in enumerate(art.get("figures", []) or []):
            raw = (fig.get("metadata") or {}).get("scicap_raw") or {}
            fig_path = _apply_path_replacements(
                fig.get("figure_path") or fig.get("result_path"),
                path_replace,
            )
            if not fig_path or not Path(fig_path).exists():
                continue
            long_cap = _flatten_maybe_json_text(raw.get("mlbcap_long") or fig.get("figure_caption"))
            short_cap = _flatten_maybe_json_text(raw.get("mlbcap_short"))
            if not short_cap and long_cap:
                short_cap = _first_sentence(long_cap)
            desc = _flatten_maybe_json_text(raw.get("figure_description"))
            paragraph = _flatten_maybe_json_text(raw.get("paragraph") or fig.get("figure_info"))
            ocr = " ; ".join(_clean_ocr_items(raw.get("ocr") or fig.get("figure_ocr") or fig.get("ocr"), max_items=96))
            uid = _flatten_text(fig.get("uid")) or f"{article_id}::scicap_fig_{i}"

            _append_sample(
                rows,
                split=split,
                source="SciCap",
                uid=uid,
                image_path=fig_path,
                task="caption_short",
                target=short_cap,
                paragraph=paragraph,
                ocr=ocr,
            )
            _append_sample(
                rows,
                split=split,
                source="SciCap",
                uid=uid,
                image_path=fig_path,
                task="caption_long",
                target=long_cap,
                paragraph=paragraph,
                ocr=ocr,
            )
            _append_sample(
                rows,
                split=split,
                source="SciCap",
                uid=uid,
                image_path=fig_path,
                task="description",
                target=desc,
                paragraph=paragraph,
                ocr=ocr,
            )
    return rows


def _iter_scistruct(split_json: Path, split: str, path_replace: List[Tuple[str, str]]) -> List[Dict]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    rows: List[Dict] = []
    for art in data:
        article_id = _flatten_text(art.get("article_id"))
        for i, fig in enumerate(art.get("figures", []) or []):
            fig_path = _apply_path_replacements(
                fig.get("figure_path") or fig.get("result_path"),
                path_replace,
            )
            if not fig_path or not Path(fig_path).exists():
                continue
            paragraph = _flatten_text(fig.get("figure_info") or fig.get("figure_des"))
            ocr = " ; ".join(_clean_ocr_items(fig.get("ocr_text") or fig.get("figure_ocr") or fig.get("ocr"), max_items=96))
            expl = _flatten_text(fig.get("figure_des") or fig.get("figure_caption"))
            uid = _flatten_text(fig.get("uid")) or f"{article_id}::scistruct_fig_{i}"
            _append_sample(
                rows,
                split=split,
                source="SciStruct",
                uid=uid,
                image_path=fig_path,
                task="explanation",
                target=expl,
                paragraph=paragraph,
                ocr=ocr,
            )
    return rows


def _to_tinychart_rows(rows: Iterable[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for i, r in enumerate(rows):
        prompt = f"<image>\n{r['prompt']}".strip()
        out.append(
            {
                "id": f"{r['uid']}::{r['task']}::{i}",
                "image": r["image_path"],
                "task": r["task"],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": r["target"]},
                ],
            }
        )
    return out


def _to_ureader_rows(rows: Iterable[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for i, r in enumerate(rows):
        prompt = f"<image>\n{r['prompt']}".strip()
        text = f"Human: {prompt}\nAI: {r['target']}".strip()
        out.append(
            {
                "id": f"{r['uid']}::{r['task']}::{i}",
                "image": [r["image_path"]],
                "task_type": "qa_sft",
                "text": text,
                "conversations": [
                    {"from": "user", "value": prompt},
                    {"from": "assistant", "value": r["target"]},
                ],
            }
        )
    return out


def _to_omnicaptioner_rows(rows: Iterable[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for i, r in enumerate(rows):
        prompt = f"<image>\n{r['prompt']}".strip()
        out.append(
            {
                "id": f"{r['uid']}::{r['task']}::{i}",
                "image": [r["image_path"]],
                "task_type": "chart",
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": r["target"]},
                ],
            }
        )
    return out


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scicap_train_json", default="dataset/scicap_mlbcap_node_diagram_v2/dataset_split/train.json")
    ap.add_argument("--scicap_val_json", default="dataset/scicap_mlbcap_node_diagram_v2/dataset_split/val.json")
    ap.add_argument("--scistruct_train_json", default="dataset/SciStruct/dataset_split_811/train.json")
    ap.add_argument("--scistruct_val_json", default="dataset/SciStruct/dataset_split_811/val.json")
    ap.add_argument("--out_dir", default="artifacts/specialized_sft")
    ap.add_argument(
        "--path_replace",
        nargs="*",
        default=[],
        help="optional prefix replacement pairs in src=dst form",
    )
    args = ap.parse_args()

    replace_pairs: List[Tuple[str, str]] = []
    for kv in args.path_replace:
        if "=" not in kv:
            continue
        src, dst = kv.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if src and dst:
            replace_pairs.append((src, dst))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scicap_train = _iter_scicap(Path(args.scicap_train_json), split="train", path_replace=replace_pairs)
    scicap_val = _iter_scicap(Path(args.scicap_val_json), split="val", path_replace=replace_pairs)
    scistruct_train = _iter_scistruct(Path(args.scistruct_train_json), split="train", path_replace=replace_pairs)
    scistruct_val = _iter_scistruct(Path(args.scistruct_val_json), split="val", path_replace=replace_pairs)

    train_rows = scicap_train + scistruct_train
    val_rows = scicap_val + scistruct_val

    _write_jsonl(out_dir / "unified_train.jsonl", train_rows)
    _write_jsonl(out_dir / "unified_val.jsonl", val_rows)

    tiny_train = _to_tinychart_rows(train_rows)
    tiny_val = _to_tinychart_rows(val_rows)
    _write_json(out_dir / "tinychart_train.json", tiny_train)
    _write_json(out_dir / "tinychart_val.json", tiny_val)

    ureader_train = _to_ureader_rows(train_rows)
    ureader_val = _to_ureader_rows(val_rows)
    _write_jsonl(out_dir / "ureader_train.jsonl", ureader_train)
    _write_jsonl(out_dir / "ureader_val.jsonl", ureader_val)

    omni_train = _to_omnicaptioner_rows(train_rows)
    omni_val = _to_omnicaptioner_rows(val_rows)
    _write_jsonl(out_dir / "omnicaptioner_train.jsonl", omni_train)
    _write_jsonl(out_dir / "omnicaptioner_val.jsonl", omni_val)

    stat = {
        "out_dir": str(out_dir),
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "train_by_task": {},
        "val_by_task": {},
        "path_replace": replace_pairs,
        "files": {
            "unified_train_jsonl": str(out_dir / "unified_train.jsonl"),
            "unified_val_jsonl": str(out_dir / "unified_val.jsonl"),
            "tinychart_train_json": str(out_dir / "tinychart_train.json"),
            "tinychart_val_json": str(out_dir / "tinychart_val.json"),
            "ureader_train_jsonl": str(out_dir / "ureader_train.jsonl"),
            "ureader_val_jsonl": str(out_dir / "ureader_val.jsonl"),
            "omnicaptioner_train_jsonl": str(out_dir / "omnicaptioner_train.jsonl"),
            "omnicaptioner_val_jsonl": str(out_dir / "omnicaptioner_val.jsonl"),
        },
    }
    for r in train_rows:
        stat["train_by_task"][r["task"]] = stat["train_by_task"].get(r["task"], 0) + 1
    for r in val_rows:
        stat["val_by_task"][r["task"]] = stat["val_by_task"].get(r["task"], 0) + 1
    _write_json(out_dir / "stats.json", stat)
    print(json.dumps(stat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

