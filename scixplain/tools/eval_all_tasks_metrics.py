#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

FIG_PREFIX_RE = re.compile(
    r"^\s*(figure|fig\.?)\s*\d+[a-z]?(?:\([a-z0-9]+\))?\s*[:.\-]?\s*",
    re.IGNORECASE,
)


class _NoWordNet:
    """Disable synonym lookup so NLTK METEOR does not require external corpora."""

    @staticmethod
    def synsets(*_args, **_kwargs):
        return []


NO_WORDNET = _NoWordNet()


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return out


def _norm(x: str, strip_figure_prefix: bool = True) -> str:
    if x is None:
        return ""
    s = " ".join(str(x).replace("\u00A0", " ").split())
    if strip_figure_prefix:
        s = FIG_PREFIX_RE.sub("", s)
    return s


def _pairs_caption(
    gt_rows: List[dict],
    pred_rows: List[dict],
    scale: str,
    strip_figure_prefix: bool,
) -> Tuple[List[str], List[str]]:
    gt = {}
    for r in gt_rows:
        uid = r.get("uid")
        if not uid:
            continue
        sc = r.get("scales") or {}
        txt = _norm(sc.get(scale, ""), strip_figure_prefix=strip_figure_prefix)
        if txt:
            gt[uid] = txt

    preds, refs = [], []
    pkey = "pred_short" if scale == "short" else "pred_long"
    for r in pred_rows:
        if "error" in r:
            continue
        uid = r.get("uid")
        if not uid or uid not in gt:
            continue
        p = _norm(r.get(pkey, ""), strip_figure_prefix=strip_figure_prefix)
        if p:
            preds.append(p)
            refs.append(gt[uid])
    return preds, refs


def _pairs_desc(gt_rows: List[dict], pred_rows: List[dict], strip_figure_prefix: bool) -> Tuple[List[str], List[str]]:
    gt_uid = {
        r.get("uid"): _norm(r.get("description", ""), strip_figure_prefix=strip_figure_prefix)
        for r in gt_rows
        if r.get("uid")
    }
    gt_img = {
        r.get("image_path"): _norm(r.get("description", ""), strip_figure_prefix=strip_figure_prefix)
        for r in gt_rows
        if r.get("image_path")
    }

    preds, refs = [], []
    for r in pred_rows:
        if "error" in r:
            continue
        p = _norm(r.get("pred_description", ""), strip_figure_prefix=strip_figure_prefix)
        if not p:
            continue
        uid = r.get("uid")
        ip = r.get("image_path")
        ref = _norm(gt_uid.get(uid, ""), strip_figure_prefix=strip_figure_prefix) if uid else ""
        if not ref and ip:
            ref = _norm(gt_img.get(ip, ""), strip_figure_prefix=strip_figure_prefix)
        if ref:
            preds.append(p)
            refs.append(ref)
    return preds, refs


def _pairs_expl(gt_rows: List[dict], pred_rows: List[dict], strip_figure_prefix: bool) -> Tuple[List[str], List[str]]:
    gt_uid = {}
    gt_img = {}
    for r in gt_rows:
        tgt = _norm(
            r.get("target_explanation", "") or r.get("explanation", "") or r.get("description", ""),
            strip_figure_prefix=strip_figure_prefix,
        )
        if not tgt:
            continue
        if r.get("uid"):
            gt_uid[r["uid"]] = tgt
        if r.get("image_path"):
            gt_img[r["image_path"]] = tgt

    preds, refs = [], []
    for r in pred_rows:
        if "error" in r:
            continue
        p = _norm(
            r.get("pred_explanation", "")
            or r.get("prediction", "")
            or r.get("pred", "")
            or r.get("pred_description", ""),
            strip_figure_prefix=strip_figure_prefix,
        )
        if not p:
            continue
        uid = r.get("uid")
        ip = r.get("image_path")
        ref = _norm(gt_uid.get(uid, ""), strip_figure_prefix=strip_figure_prefix) if uid else ""
        if not ref and ip:
            ref = _norm(gt_img.get(ip, ""), strip_figure_prefix=strip_figure_prefix)
        if ref:
            preds.append(p)
            refs.append(ref)
    return preds, refs


def _metrics(preds: List[str], refs: List[str], bert: BERTScorer) -> Dict[str, float]:
    if not preds:
        return {
            "N": 0,
            "BLEU-4": 0.0,
            "ROUGE-L": 0.0,
            "ROUGE-1": 0.0,
            "ROUGE-2": 0.0,
            "CIDEr": 0.0,
            "BERTScore": 0.0,
            "METEOR": 0.0,
        }

    bleu = BLEU(effective_order=True).corpus_score(preds, [refs]).score / 100.0

    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for p, r in zip(preds, refs):
        s = rs.score(r, p)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(preds)

    gts = {i: [refs[i]] for i in range(n)}
    res = {i: [preds[i]] for i in range(n)}
    cider, _ = Cider().compute_score(gts, res)

    _, _, bf1 = bert.score(preds, refs, batch_size=16)
    bscore = float(bf1.mean().item())

    meteor_vals = []
    for p, r in zip(preds, refs):
        # nltk meteor_score expects tokenized hypothesis/reference.
        meteor_vals.append(meteor_score([r.split()], p.split(), wordnet=NO_WORDNET))
    meteor = float(sum(meteor_vals) / len(meteor_vals)) if meteor_vals else 0.0

    return {
        "N": n,
        "BLEU-4": float(bleu),
        "ROUGE-L": float(rl / n),
        "ROUGE-1": float(r1 / n),
        "ROUGE-2": float(r2 / n),
        "CIDEr": float(cider),
        "BERTScore": bscore,
        "METEOR": meteor,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="json file with gt paths and per-model prediction paths")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--no_strip_figure_prefix", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    strip_figure_prefix = not args.no_strip_figure_prefix

    gt_caption = _read_jsonl(manifest["gt"]["caption"])
    gt_desc = _read_jsonl(manifest["gt"]["description"])
    gt_expl = _read_jsonl(manifest["gt"]["explanation"])

    bert = BERTScorer(lang="en", rescale_with_baseline=False)
    out = {
        "manifest": manifest,
        "normalization": {"strip_figure_prefix": strip_figure_prefix},
        "results": {},
    }
    model_items = list(manifest.get("models", {}).items())
    total_models = len(model_items)
    for idx, (model_name, mp) in enumerate(model_items, start=1):
        cap_rows = _read_jsonl(mp["caption"])
        desc_rows = _read_jsonl(mp["description"])
        expl_rows = _read_jsonl(mp["explanation"])

        p_short, r_short = _pairs_caption(gt_caption, cap_rows, "short", strip_figure_prefix)
        p_long, r_long = _pairs_caption(gt_caption, cap_rows, "long", strip_figure_prefix)
        p_desc, r_desc = _pairs_desc(gt_desc, desc_rows, strip_figure_prefix)
        p_expl, r_expl = _pairs_expl(gt_expl, expl_rows, strip_figure_prefix)

        out["results"][model_name] = {
            "Caption_short": _metrics(p_short, r_short, bert),
            "Caption_long": _metrics(p_long, r_long, bert),
            "Description": _metrics(p_desc, r_desc, bert),
            "Explanation": _metrics(p_expl, r_expl, bert),
        }
        # Write partial checkpoints so progress can be observed while the job is running.
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[progress] model={model_name} done={idx}/{total_models}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
