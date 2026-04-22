#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple


FIG_PREFIX_RE = re.compile(
    r"^\s*(figure|fig\.?)\s*\d+[a-z]?(?:\([a-z0-9]+\))?\s*[:.\-]?\s*",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9_\-/]{1,}\b")
RELATION_RE = re.compile(
    r"\b(connect(?:ed|s|ing)?|link(?:ed|s|ing)?|edge(?:s)?|flow(?:s|ing)?|"
    r"arrow(?:s)?|depend(?:s|ed|ing)?|interaction(?:s)?|route(?:s|d|ing)?|"
    r"path(?:way|ways)?|transfer(?:s|red|ring)?|transmit(?:s|ted|ting)?|between)\b",
    re.IGNORECASE,
)
RELATION_DIR_RE = re.compile(
    r"\b([a-z0-9_\-/]{2,40})\b[^.]{0,40}\b("
    r"inhibits?|suppresses?|blocks?|prevents?|"
    r"activates?|promotes?|enhances?|increases?|decreases?|"
    r"regulates?|controls?|triggers?"
    r")\b[^.]{0,40}\b([a-z0-9_\-/]{2,40})\b",
    re.IGNORECASE,
)
FROM_TO_RE = re.compile(r"\bfrom\b[^.]{0,48}\bto\b", re.IGNORECASE)
LABEL_MENTION_RE = re.compile(
    r"\b(node|module|component|block|stage|layer|state|unit)\s+([a-z0-9_\-/]{1,40})\b",
    re.IGNORECASE,
)
ABSTRACT_EXPR_RE = re.compile(
    r"\b("
    r"node diagram|flowchart|workflow|pipeline|architecture|framework|"
    r"generic (?:diagram|pipeline|architecture)|"
    r"overall (?:process|workflow|pipeline)|"
    r"data flow|information flow|processing pipeline|"
    r"neural network (?:structure|architecture)|"
    r"model (?:architecture|structure)"
    r")\b",
    re.IGNORECASE,
)

GENERIC_MENTION_STOP = {
    "figure",
    "module",
    "component",
    "block",
    "stage",
    "layer",
    "state",
    "unit",
    "node",
    "nodes",
    "system",
    "method",
    "model",
    "network",
    "diagram",
    "pipeline",
    "process",
    "input",
    "output",
}

ALIAS_GENERIC_WORDS = GENERIC_MENTION_STOP | {
    "method",
    "approach",
    "framework",
    "architecture",
}

LABEL_FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "for",
    "with",
    "by",
    "as",
    "via",
    "then",
    "through",
    "has",
    "have",
    "had",
    "uses",
    "using",
    "consists",
    "consist",
    "produces",
    "receives",
    "connects",
    "connected",
    "representing",
    "represents",
    "labeled",
    "indicated",
    "also",
    "not",
    "possibly",
}

ABSTRACT_MENTION_TOKENS = {
    "encoder",
    "decoder",
    "classifier",
    "predictor",
    "detector",
    "module",
    "modules",
    "component",
    "components",
    "block",
    "blocks",
    "stage",
    "stages",
    "layer",
    "layers",
    "node",
    "nodes",
    "unit",
    "units",
    "network",
    "system",
    "model",
    "framework",
    "architecture",
    "pipeline",
    "workflow",
    "diagram",
    "flowchart",
    "process",
    "feature",
    "features",
    "embedding",
    "attention",
    "token",
    "tokens",
    "state",
    "states",
}

ABSTRACT_ACRONYM_HINTS = {
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "gcn",
    "gnn",
    "mlp",
    "bert",
    "gan",
    "vae",
    "unet",
    "resnet",
    "vit",
    "transformer",
}


def norm_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_fig_prefix(text: str) -> str:
    return FIG_PREFIX_RE.sub("", text or "")


def normalize_ocr_entry(entry: str) -> str:
    s = norm_text(entry)
    s = re.sub(r"[^a-z0-9_\-/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_maybe_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [x for x in value if isinstance(x, str)]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            # Prefer json; fallback to literal eval for legacy dumps.
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [x for x in obj if isinstance(x, str)]
            except Exception:
                pass
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, list):
                    return [x for x in obj if isinstance(x, str)]
            except Exception:
                pass
        return [s]
    return []


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(norm_text(text))


def alias_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", norm_text(text))


def alias_variants_from_entry(entry: str, max_alias: int = 10) -> List[str]:
    s0 = normalize_ocr_entry(entry)
    if not s0:
        return []
    out: List[str] = []
    seen = set()

    def _add(x: str) -> None:
        sx = normalize_ocr_entry(x)
        if not sx:
            return
        k = alias_key(sx)
        if len(k) < 2 or k in seen:
            return
        if k in ALIAS_GENERIC_WORDS or k in LABEL_FUNCTION_WORDS:
            return
        seen.add(k)
        out.append(sx)

    _add(s0)
    _add(re.sub(r"[-_/]+", " ", s0))
    _add(re.sub(r"[^a-z0-9 ]+", " ", s0))
    _add(re.sub(r"[^a-z0-9]", "", s0))

    compact = re.sub(r"[^a-z0-9]", "", s0)
    m = re.match(r"^([a-z]+)([0-9]+)$", compact)
    if m:
        _add(f"{m.group(1)} {m.group(2)}")

    toks = [t for t in re.split(r"[^a-z0-9]+", s0) if t]
    if len(toks) >= 2:
        _add(" ".join(toks[:2]))
        _add(" ".join(toks[-2:]))
    if len(toks) == 1:
        t = toks[0]
        if len(t) <= 12:
            _add(f"module {t}")
            _add(f"stage {t}")
            _add(f"block {t}")

    if len(out) > max(1, int(max_alias)):
        out = out[: max(1, int(max_alias))]
    return out


def build_alias_lexicon(ocr_components: Sequence[str], gt_mentions: Set[str], max_alias_per_item: int = 10) -> Set[str]:
    out: Set[str] = set()
    for x in list(ocr_components) + list(gt_mentions):
        for a in alias_variants_from_entry(x, max_alias=max_alias_per_item):
            out.add(a)
    return out


def match_aliases(text: str, aliases: Set[str]) -> Set[str]:
    if not aliases:
        return set()
    txt = norm_text(text)
    toks = set(tokenize(txt))
    matched: Set[str] = set()
    for a in aliases:
        if not a:
            continue
        parts = a.split()
        if not parts:
            continue
        if len(parts) == 1:
            if parts[0] in toks:
                matched.add(a)
        else:
            pat = r"\b" + r"\s+".join(re.escape(t) for t in parts) + r"\b"
            if re.search(pat, txt):
                matched.add(a)
    return matched


def match_ocr_components(text: str, ocr_components: Sequence[str]) -> Set[str]:
    txt = norm_text(text)
    toks = set(tokenize(txt))
    matched: Set[str] = set()
    for comp in ocr_components:
        if not comp:
            continue
        comp_toks = comp.split()
        if not comp_toks:
            continue
        if len(comp_toks) == 1:
            if comp_toks[0] in toks:
                matched.add(comp)
        else:
            # phrase-level soft match with word boundaries
            pat = r"\b" + r"\s+".join(re.escape(t) for t in comp_toks) + r"\b"
            if re.search(pat, txt):
                matched.add(comp)
    return matched


def extract_component_mentions(text: str) -> Set[str]:
    mentions: Set[str] = set()
    raw = text or ""

    for m in ACRONYM_RE.findall(raw):
        mm = normalize_ocr_entry(m)
        if mm and mm not in GENERIC_MENTION_STOP:
            mentions.add(mm)

    low = norm_text(raw)
    for _, label in LABEL_MENTION_RE.findall(low):
        mm = normalize_ocr_entry(label)
        if mm and mm not in GENERIC_MENTION_STOP and mm not in LABEL_FUNCTION_WORDS:
            mentions.add(mm)

    return mentions


def relation_triggered(text: str) -> bool:
    if RELATION_RE.search(text or ""):
        return True
    if FROM_TO_RE.search(text or ""):
        return True
    return False


def extract_directional_relations(text: str) -> Set[Tuple[str, str, str]]:
    out: Set[Tuple[str, str, str]] = set()
    raw = norm_text(text or "")
    for m in RELATION_DIR_RE.finditer(raw):
        a = normalize_ocr_entry(m.group(1))
        v = normalize_ocr_entry(m.group(2))
        b = normalize_ocr_entry(m.group(3))
        if not a or not b or not v:
            continue
        if a in LABEL_FUNCTION_WORDS or b in LABEL_FUNCTION_WORDS:
            continue
        out.add((a, v, b))
    return out


def mention_is_abstract(mention: str) -> bool:
    m = norm_text(mention)
    if not m:
        return False
    toks = [t for t in m.split() if t]
    if not toks:
        return False
    if m in LABEL_FUNCTION_WORDS:
        return True
    if re.fullmatch(r"[a-z]", m):
        return True
    if re.fullmatch(r"[a-z]\d{0,2}", m):
        return True
    if re.fullmatch(r"[a-z]-[a-z]", m):
        return True
    if "/" in m and len(m) >= 12:
        return True
    if all(t in ABSTRACT_MENTION_TOKENS for t in toks):
        return True
    if len(toks) == 1 and toks[0] in ABSTRACT_ACRONYM_HINTS:
        return True
    return False


def text_has_abstract_expr(text: str) -> bool:
    return bool(ABSTRACT_EXPR_RE.search(text or ""))


def build_ref_to_ocr_map(scicap_test_json: str) -> Dict[str, List[str]]:
    data = json.load(open(scicap_test_json))
    ref_to_ocr: Dict[str, List[str]] = {}
    dup = 0
    for art in data:
        for fig in art.get("figures", []):
            sr = fig.get("metadata", {}).get("scicap_raw", {})
            ref = sr.get("figure_description") or fig.get("figure_caption") or ""
            key = norm_text(ref)
            if key in ref_to_ocr:
                dup += 1
            ocr_raw = parse_maybe_list(sr.get("ocr")) or parse_maybe_list(fig.get("figure_ocr"))
            comps: List[str] = []
            for x in ocr_raw:
                if not isinstance(x, str):
                    continue
                nx = normalize_ocr_entry(x)
                if len(nx) < 2:
                    continue
                comps.append(nx)
            # preserve order while dedup
            seen = set()
            comps_u = []
            for c in comps:
                if c in seen:
                    continue
                seen.add(c)
                comps_u.append(c)
            ref_to_ocr[key] = comps_u
    if dup:
        print(f"[warn] duplicated normalized refs in test set: {dup}")
    return ref_to_ocr


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def calc_for_texts(
    texts: Sequence[str],
    ocr_lists: Sequence[Sequence[str]],
    node_lexicons: Sequence[Set[str]],
    alias_lexicons: Sequence[Set[str]],
    ref_texts: Sequence[str] | None = None,
) -> Dict[str, float]:
    n = len(texts)
    node_mention_hits = 0
    relation_hits = 0
    ocr_cov_sum = 0.0
    node_cov_sum = 0.0
    alias_hit_sum = 0
    alias_cov_sum = 0.0
    alias_non_empty = 0
    exact_ocr_hit_sum = 0
    node_non_empty = 0
    ocr_non_empty = 0
    halluc_sample_hits = 0
    halluc_mention_total = 0
    halluc_nonexist_sample_hits = 0
    halluc_nonexist_mention_total = 0
    halluc_abstract_sample_hits = 0
    halluc_abstract_mention_total = 0
    abstract_style_sample_hits = 0
    mention_total = 0
    relation_acc_hits = 0
    relation_acc_total = 0

    for i, (text, ocr, node_lex, alias_lex) in enumerate(zip(texts, ocr_lists, node_lexicons, alias_lexicons)):
        ocr = list(ocr)
        node_lex = set(node_lex)
        alias_lex = set(alias_lex)
        matched = match_ocr_components(text, ocr)
        matched_alias = match_aliases(text, alias_lex)
        mentions = extract_component_mentions(text)
        node_hit = len(mentions & node_lex) > 0 if node_lex else False
        node_mention_hits += 1 if node_hit else 0
        relation_hits += 1 if relation_triggered(text) else 0
        exact_ocr_hit_sum += 1 if matched else 0
        alias_hit_sum += 1 if matched_alias else 0
        if ocr:
            ocr_cov_sum += safe_div(len(matched), len(ocr))
            ocr_non_empty += 1
        if node_lex:
            node_cov_sum += safe_div(len(mentions & node_lex), len(node_lex))
            node_non_empty += 1
        if alias_lex:
            alias_cov_sum += safe_div(len(matched_alias), len(alias_lex))
            alias_non_empty += 1

        alias_keys = {alias_key(a) for a in alias_lex if alias_key(a)}
        halluc = {m for m in mentions if alias_key(m) not in alias_keys and m not in node_lex}
        halluc_nonexist = {m for m in halluc if not mention_is_abstract(m)}
        halluc_abstract = halluc - halluc_nonexist
        if halluc:
            halluc_sample_hits += 1
        if halluc_nonexist:
            halluc_nonexist_sample_hits += 1
        if halluc_abstract:
            halluc_abstract_sample_hits += 1
        if text_has_abstract_expr(text):
            abstract_style_sample_hits += 1
        halluc_mention_total += len(halluc)
        halluc_nonexist_mention_total += len(halluc_nonexist)
        halluc_abstract_mention_total += len(halluc_abstract)
        mention_total += len(mentions)
        if ref_texts is not None and i < len(ref_texts):
            pred_rel = extract_directional_relations(text)
            ref_rel = extract_directional_relations(ref_texts[i])
            if ref_rel:
                relation_acc_total += 1
                relation_acc_hits += 1 if (pred_rel & ref_rel) else 0

    return {
        "n_samples": n,
        "node_coverage_rate": safe_div(node_mention_hits, n),
        "node_coverage_ratio": safe_div(node_cov_sum, node_non_empty),
        "node_coverage_denominator_nonempty": node_non_empty,
        "relation_trigger_rate": safe_div(relation_hits, n),
        "alias_hit_rate_sample": safe_div(alias_hit_sum, n),
        "alias_coverage_ratio": safe_div(alias_cov_sum, alias_non_empty),
        "alias_coverage_denominator_nonempty": alias_non_empty,
        "exact_ocr_hit_rate_sample": safe_div(exact_ocr_hit_sum, n),
        "ocr_grounding_rate": safe_div(ocr_cov_sum, ocr_non_empty),
        "ocr_grounding_denominator_nonempty": ocr_non_empty,
        "hallucination_rate_sample": safe_div(halluc_sample_hits, n),
        "hallucination_rate_mentions": safe_div(halluc_mention_total, mention_total),
        "hallucination_nonexistent_entity_rate_sample": safe_div(halluc_nonexist_sample_hits, n),
        "hallucination_nonexistent_entity_rate_mentions": safe_div(halluc_nonexist_mention_total, mention_total),
        "hallucination_abstract_expression_rate_sample": safe_div(halluc_abstract_sample_hits, n),
        "hallucination_abstract_expression_rate_mentions": safe_div(halluc_abstract_mention_total, mention_total),
        "abstract_style_rate_sample": safe_div(abstract_style_sample_hits, n),
        "avg_component_mentions": safe_div(mention_total, n),
        "relation_accuracy": safe_div(relation_acc_hits, relation_acc_total),
        "relation_accuracy_denominator_nonempty": relation_acc_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--scicap_test_json", required=True)
    ap.add_argument("--scale", default="desc")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    ref_to_ocr = build_ref_to_ocr_map(args.scicap_test_json)

    pred_texts: List[str] = []
    gt_texts: List[str] = []
    ocr_lists: List[List[str]] = []
    node_lexicons: List[Set[str]] = []
    alias_lexicons: List[Set[str]] = []

    with open(args.pairs_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("scale") != args.scale:
                continue
            ref = obj.get("ref", "")
            key = norm_text(ref)
            ocr = ref_to_ocr.get(key)
            if ocr is None:
                # Robust fallback for potential figure prefix variance.
                key2 = norm_text(strip_fig_prefix(ref))
                ocr = ref_to_ocr.get(key2, [])
            pred_texts.append(obj.get("pred", ""))
            gt_texts.append(ref)
            ocr_lists.append(ocr)
            gt_mentions = extract_component_mentions(ref)
            ocr_lex = set(ocr)
            for c in ocr:
                ocr_lex.update(c.split())
            node_lexicons.append(ocr_lex | gt_mentions)
            alias_lexicons.append(build_alias_lexicon(ocr_components=ocr, gt_mentions=gt_mentions, max_alias_per_item=10))

    pred_metrics = calc_for_texts(pred_texts, ocr_lists, node_lexicons, alias_lexicons, ref_texts=gt_texts)
    gt_metrics = calc_for_texts(gt_texts, ocr_lists, node_lexicons, alias_lexicons, ref_texts=gt_texts)

    out = {
        "input": {
            "pairs_jsonl": args.pairs_jsonl,
            "scicap_test_json": args.scicap_test_json,
            "scale": args.scale,
            "matched_samples": len(pred_texts),
        },
        "pred": pred_metrics,
        "gt": gt_metrics,
        "delta_pred_minus_gt": {
            k: pred_metrics[k] - gt_metrics[k]
            for k in pred_metrics.keys()
            if isinstance(pred_metrics[k], (int, float)) and k != "n_samples"
        },
        "notes": {
            "node_coverage_rate": "Sample-level rate: mentions >=1 OCR-derived component.",
            "alias_hit_rate_sample": "Sample-level alias hit: text mentions at least one alias from OCR/GT-derived alias set.",
            "alias_coverage_ratio": "Average per-sample alias coverage ratio.",
            "exact_ocr_hit_rate_sample": "Sample-level exact OCR hit: text contains at least one raw OCR component phrase.",
            "relation_trigger_rate": "Sample-level rate: relation lexical triggers (connect/edge/flow/from-to etc.).",
            "relation_accuracy": "Sample-level directional relation accuracy: overlap with GT directional action tuples (A verb B), computed on samples where GT has directional tuples.",
            "ocr_grounding_rate": "Average per-sample OCR component coverage ratio.",
            "hallucination_rate_sample": "Sample-level rate: mentions at least one component not found in OCR lexicon.",
            "hallucination_rate_mentions": "Mention-level ratio: hallucinated component mentions / all extracted component mentions.",
            "hallucination_nonexistent_entity_rate_sample": "Sample-level strict hallucination: at least one non-abstract mention not in OCR/GT-derived lexicon.",
            "hallucination_nonexistent_entity_rate_mentions": "Mention-level strict hallucination ratio over all extracted mentions.",
            "hallucination_abstract_expression_rate_sample": "Sample-level abstract-expression hallucination: only abstract mentions (e.g., placeholders like A/B/C, generic concept tags) not tied to OCR entities.",
            "hallucination_abstract_expression_rate_mentions": "Mention-level abstract-expression ratio over all extracted mentions.",
            "abstract_style_rate_sample": "Sample-level rate of generic abstract writing style phrases (diagnostic only; not hallucination by itself).",
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
