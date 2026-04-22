#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
import random
import re
import types
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from scixplain.models import CLIPVisionTower
from scixplain.tools.train_text_decoder import SciCapMultiScaleDataset


TASK_TOKENS = {
    "short": "<CAPTION_SHORT>",
    "long": "<CAPTION_LONG>",
    "desc": "<DESCRIPTION>",
    "explain": "<EXPLAIN>",
}

NO_CONTEXT_TOKEN = "<NO_CONTEXT>"
NO_OCR_TOKEN = "<NO_OCR>"
NO_ADESC_TOKEN = "<NO_ADESC>"

TASK_STYLE_TOKENS = {
    "short": "<STYLE_SHORT>",
    "long": "<STYLE_LONG>",
    "desc": "<STYLE_DESC>",
}

SCICAP_TASK_PROMPTS = {
    "short": (
        "Write one concise sentence as a figure caption. "
        "Cover the main components, the purpose, and one key workflow detail. "
        "Avoid phrases like \"this figure shows\"."
    ),
    "long": (
        "Write a detailed figure caption with this structure: "
        "(1) main components/modules, "
        "(2) component relations/interactions, "
        "(3) overall workflow."
    ),
    "desc": (
        "Describe the figure structure in detail: "
        "name concrete components, explain interactions among components, "
        "and summarize end-to-end information flow."
    ),
}

SCICAP_DESC_STRICT_PROMPT = (
    "List and expand the concrete components in the figure, "
    "then explain explicit relations among them (inputs, outputs, dependencies, or control flow), "
    "and finish with a clear end-to-end workflow summary."
)

ZERO_SHOT_CAPTION_PROMPT = "Generate a figure caption based on the image. Be specific and avoid generic statements."

TYPE_KEYWORDS = [
    "state", "pipeline", "flowchart", "circuit", "graph", "dag", "network",
    "architecture", "diagram", "tree", "automaton"
]
STRUCT_KEYWORDS = [
    "node", "edge", "module", "block", "stage", "layer", "component",
    "transition", "branch", "loop", "arrow", "server", "worker", "input", "output"
]
REL_KEYWORDS = [
    "control", "controls", "flow", "flows", "enable", "enables", "depend", "depends",
]

KEEP_WORDS = {
    "because", "therefore", "then", "next", "after", "before", "since", "thus",
    "so", "as", "if", "else", "first", "second", "third", "finally", "when",
    "while", "where", "which", "that", "and", "or", "but"
}

_COVERAGE_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "as", "at", "from", "that", "this", "these",
    "those", "it", "its", "their", "our", "we", "they", "can", "may", "using",
}

DESC_COMPONENT_HINTS = set(STRUCT_KEYWORDS) | {
    "component", "components", "unit", "units", "cell", "cells", "encoder", "decoder",
    "attention", "branch", "path", "stream", "controller", "gateway", "router",
}
DESC_REL_HINTS = set(REL_KEYWORDS) | {
    "connect", "connected", "link", "linked", "route", "routed", "send", "receive",
    "input", "output", "feedback", "dependency", "depends", "interaction", "interacts",
}
DESC_FLOW_HINTS = {
    "first", "next", "then", "finally", "after", "before", "pipeline", "stage",
    "flow", "workflow", "sequence", "step", "process", "through",
}
DESC_FUNC_HINTS = {
    "detect", "predict", "classify", "generate", "optimize", "control", "evaluate",
    "fuse", "align", "aggregate", "select", "retrieve", "encode", "decode",
}

_DESC_SLOT_HINTS = [
    DESC_COMPONENT_HINTS,
    DESC_REL_HINTS,
    DESC_FLOW_HINTS,
    DESC_FUNC_HINTS,
]

METHOD_PATTERNS = [
    r"\\bwe (propose|present|introduce|develop|design|study|investigate|explore|extend)\\b",
    r"\\bour (method|model|approach|framework|architecture)\\b",
    r"\\bthe proposed (method|model|approach|framework|architecture)\\b",
    r"\\bproposed (method|model|approach|framework|architecture)\\b",
    r"\\bthis paper (proposes|presents|introduces|develops|describes)\\b",
]

NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ing", "ism", "ology", "graph", "network", "model", "system", "framework", "architecture", "algorithm")


def _flatten_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (list, tuple)):
        parts = []
        for it in val:
            s = _flatten_text(it)
            if s:
                parts.append(s)
        return " ".join(parts).strip()
    return str(val).strip()


def _tokenize_ids(
    tokenizer: AutoTokenizer,
    text: str,
    *,
    add_special_tokens: bool = False,
    max_length: int = 0,
) -> List[int]:
    kwargs = {"add_special_tokens": add_special_tokens}
    if max_length and int(max_length) > 0:
        kwargs["truncation"] = True
        kwargs["max_length"] = int(max_length)
    return tokenizer(text, **kwargs)["input_ids"]


def _first_sentence(text: str) -> str:
    for sep in [".", "?", "!", "。", "？", "！"]:
        idx = text.find(sep)
        if idx >= 0:
            return text[: idx + 1].strip()
    return text.strip()


def _word_count(text: str) -> int:
    s = _flatten_text(text)
    if not s:
        return 0
    return len([w for w in re.split(r"\s+", s) if w])


def _hash_token_ids(ids: List[int]) -> int:
    if not ids:
        return 0
    data = ",".join(str(int(x)) for x in ids).encode("utf-8")
    return int(hashlib.sha1(data).hexdigest()[:16], 16) & ((1 << 63) - 1)


_CF_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _cf_context_tokens(text: str, max_unique: int = 96) -> set[str]:
    out: set[str] = set()
    for tok in _CF_WORD_RE.findall(str(text or "").lower()):
        if len(tok) < 3:
            continue
        if tok in _COVERAGE_STOPWORDS:
            continue
        out.add(tok)
        if len(out) >= max(8, int(max_unique)):
            break
    return out


def _jaccard_token_set(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    union = len(a | b)
    return float(inter) / float(max(1, union))


def _build_explain_cf_perm(
    exp_idx: List[int],
    contexts: Optional[List[str]],
    mode: str,
) -> List[int]:
    if len(exp_idx) <= 1:
        return list(exp_idx)
    base = list(exp_idx)
    if mode == "random":
        cand = list(base)
        random.shuffle(cand)
        if cand == base:
            cand = cand[1:] + cand[:1]
        return cand
    if mode != "hard_jaccard":
        return base[1:] + base[:1]

    ctx = contexts or []
    tok_map: Dict[int, set[str]] = {}
    len_map: Dict[int, int] = {}
    for i in base:
        c = str(ctx[i] if i < len(ctx) else "")
        tok_map[i] = _cf_context_tokens(c)
        len_map[i] = max(1, len(c.split()))

    out: List[int] = []
    for i in base:
        best_j = -1
        best_score = -1e9
        ti = tok_map.get(i, set())
        li = len_map.get(i, 1)
        for j in base:
            if j == i:
                continue
            tj = tok_map.get(j, set())
            lj = len_map.get(j, 1)
            sim = _jaccard_token_set(ti, tj)
            len_pen = abs(li - lj) / float(max(li, lj))
            score = sim - 0.05 * len_pen + random.random() * 1e-6
            if score > best_score:
                best_score = score
                best_j = j
        if best_j < 0:
            best_j = base[(base.index(i) + 1) % len(base)]
        out.append(best_j)

    if out == base:
        out = base[1:] + base[:1]
    return out


def _is_connector_state_key(key: str) -> bool:
    parts = str(key or "").split(".")
    return "connector" in parts


def _find_lm_matrix_rows(state_dict: Dict[str, torch.Tensor], suffix: str) -> Tuple[Optional[str], Optional[int]]:
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "language_model" not in k:
            continue
        if not k.endswith(suffix):
            continue
        if v.ndim < 2:
            continue
        return k, int(v.shape[0])
    return None, None


def _task_prompt_ids(tokenizer: AutoTokenizer, scale: str, prompt_style: str) -> List[int]:
    if prompt_style == "scicap_metric":
        prompt = SCICAP_TASK_PROMPTS.get(scale, "")
    elif prompt_style == "scicap_metric_desc_strict":
        if scale == "desc":
            prompt = SCICAP_DESC_STRICT_PROMPT
        else:
            prompt = SCICAP_TASK_PROMPTS.get(scale, "")
    else:
        return []
    if not prompt:
        return []
    return _tokenize_ids(tokenizer, "\n" + prompt, add_special_tokens=False, max_length=512)


def _task_style_ids(tokenizer: AutoTokenizer, scale: str, enabled: bool) -> List[int]:
    if not enabled:
        return []
    tok = TASK_STYLE_TOKENS.get(scale, "")
    if not tok:
        return []
    return tokenizer(tok, add_special_tokens=False)["input_ids"]


def _infer_formula_token_ids(tokenizer: AutoTokenizer, max_ids: int = 256) -> List[int]:
    ids: List[int] = []
    vocab_size = int(getattr(tokenizer, "vocab_size", 0)) or len(tokenizer)
    math_re = re.compile(r"[=<>^_{}\\/$%*+|]|\\b(alpha|beta|gamma|delta|theta|lambda|sigma|omega|proof|lemma|theorem|equation|corollary)\\b", re.IGNORECASE)
    for tid in range(vocab_size):
        try:
            piece = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        txt = (piece or "").strip().lower()
        if not txt:
            continue
        if math_re.search(txt):
            ids.append(int(tid))
            if len(ids) >= max_ids:
                break
    return ids


def _infer_desc_prompt_leak_token_ids(tokenizer: AutoTokenizer, max_ids: int = 128) -> List[int]:
    ids: List[int] = []
    vocab_size = int(getattr(tokenizer, "vocab_size", 0)) or len(tokenizer)
    leak_re = re.compile(
        r"(ocr|context|<ocr>|</ocr>|<para>|</para>|ocr[_ ]tokens?|ocr[_ ]context)",
        re.IGNORECASE,
    )
    for tid in range(vocab_size):
        try:
            piece = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        txt = (piece or "").strip().lower()
        if not txt:
            continue
        if leak_re.search(txt):
            ids.append(int(tid))
            if len(ids) >= max_ids:
                break
    return ids


def _coverage_token_ids_from_text(
    tokenizer: AutoTokenizer,
    text: str,
    max_ids: int,
) -> List[int]:
    if not text:
        return []
    toks = _tokenize_ids(tokenizer, "\n" + str(text), add_special_tokens=False, max_length=2048)
    out: List[int] = []
    seen = set()
    for tid in toks:
        tid = int(tid)
        if tid in seen:
            continue
        seen.add(tid)
        try:
            dec = tokenizer.decode([tid], skip_special_tokens=True).strip().lower()
        except Exception:
            continue
        if not dec:
            continue
        core = re.sub(r"[^a-z0-9]", "", dec)
        if len(core) < 2:
            continue
        if core in _COVERAGE_STOPWORDS:
            continue
        if core.isdigit():
            continue
        out.append(tid)
        if len(out) >= max(1, int(max_ids)):
            break
    return out


_ANCHOR_GENERIC_WORDS = {
    "figure",
    "diagram",
    "flowchart",
    "workflow",
    "pipeline",
    "network",
    "system",
    "process",
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
    "branch",
    "path",
    "input",
    "inputs",
    "output",
    "outputs",
}

_DESC_ENTITY_CUE_WORDS = {
    "node",
    "nodes",
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
    "unit",
    "units",
    "state",
    "states",
}


def _infer_piece_token_ids_for_words(
    tokenizer: AutoTokenizer,
    words: set[str],
    max_ids: int = 512,
) -> List[int]:
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        return []
    word_keys = {_anchor_key(w) for w in words if _anchor_key(w)}
    if not word_keys:
        return []
    ids: List[int] = []
    for tid in range(vocab_size):
        try:
            piece = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        txt = (piece or "").strip()
        if not txt:
            continue
        txt = txt.replace("Ġ", "").replace("▁", "").strip()
        if not txt:
            continue
        key = _anchor_key(txt)
        if not key:
            continue
        if key in word_keys:
            ids.append(int(tid))
            if len(ids) >= max(1, int(max_ids)):
                break
    return ids


def _entity_shape_key_from_text(
    text: str,
    allow_digit_only: bool = True,
) -> str:
    txt = (text or "").strip()
    if not txt:
        return ""
        txt = txt.replace("Ġ", "").replace("▁", "").strip()
    if not txt:
        return ""
    core_raw = re.sub(r"[^A-Za-z0-9_/-]", "", txt)
    if len(core_raw) < 1:
        return ""
    core_alnum = re.sub(r"[^A-Za-z0-9]", "", core_raw)
    if len(core_alnum) < 1:
        return ""
    has_alpha = bool(re.search(r"[A-Za-z]", core_raw))
    has_digit = bool(re.search(r"[0-9]", core_raw))
    if (not allow_digit_only) and (not has_alpha):
        return ""
    has_sep = any(ch in core_raw for ch in ("-", "_", "/"))
    is_upper = bool(re.fullmatch(r"[A-Z0-9_/-]+", core_raw)) and bool(re.search(r"[A-Z]", core_raw))
    n = len(core_alnum)
    if n <= 2:
        len_bucket = "s"
    elif n <= 5:
        len_bucket = "m"
    else:
        len_bucket = "l"
    return f"d{int(has_digit)}u{int(is_upper)}s{int(has_sep)}l{len_bucket}"


def _shape_keys_from_alias_text(alias_text: str) -> set[str]:
    keys: set[str] = set()
    for w in re.findall(r"[A-Za-z0-9_/-]+", str(alias_text or "")):
        k = _entity_shape_key_from_text(w, allow_digit_only=True)
        if k:
            keys.add(k)
    return keys


def _infer_entity_like_token_ids(
    tokenizer: AutoTokenizer,
    max_ids: int = 4096,
) -> List[int]:
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        return []
    ids: List[int] = []
    seen = set()
    for tid in range(vocab_size):
        try:
            piece = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        txt = (piece or "").strip()
        if not txt:
            continue
        # Handle GPT/SentencePiece boundaries.
        txt = txt.replace("Ġ", "").replace("▁", "").strip()
        if not txt:
            continue
        core_raw = re.sub(r"[^A-Za-z0-9_/-]", "", txt)
        if len(core_raw) < 2:
            continue
        key = _anchor_key(core_raw)
        if not key:
            continue
        if key in seen:
            continue
        if key in _COVERAGE_STOPWORDS or key in _ANCHOR_GENERIC_WORDS:
            continue
        has_alpha = bool(re.search(r"[A-Za-z]", core_raw))
        has_digit = bool(re.search(r"[0-9]", core_raw))
        has_sep = any(ch in core_raw for ch in ("-", "_", "/"))
        mixed_case = bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]", txt))
        upper_short = bool(re.fullmatch(r"[A-Z0-9]{2,6}", core_raw)) and bool(re.search(r"[A-Z]", core_raw))
        alpha_num_shape = has_alpha and has_digit
        sep_shape = has_sep and len(core_raw) >= 3
        underscore_shape = "_" in core_raw and len(core_raw) >= 3
        slash_shape = "/" in core_raw and len(core_raw) >= 3
        digit_short = bool(re.fullmatch(r"[0-9]{1,3}", core_raw))
        # Keep only "entity-like" surface forms:
        # - alnum IDs: p1, x2, conv3
        # - acronyms: CNN, RNN, GPU
        # - camelCase / snake_case / slash / hyphen variants
        if not (alpha_num_shape or upper_short or mixed_case or sep_shape or underscore_shape or slash_shape or digit_short):
            continue
        ids.append(int(tid))
        seen.add(key)
        if len(ids) >= max(1, int(max_ids)):
            break
    return ids


def _normalize_anchor_surface(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s


def _anchor_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _extract_struct_node_texts_from_meta(
    meta: Optional[Dict],
    max_items: int = 32,
) -> List[str]:
    if not isinstance(meta, dict):
        return []
    vals: List[str] = []
    for key in ("struct_nodes", "node_texts", "nodes", "node_names"):
        cur = meta.get(key)
        if cur is None:
            continue
        if isinstance(cur, (list, tuple)):
            for it in cur:
                vals.append(str(it or ""))
        elif isinstance(cur, str):
            # Reuse OCR cleaner to parse list-like strings robustly.
            parsed = clean_ocr_items(cur, max_items=max(8, int(max_items)))
            if parsed:
                vals.extend(parsed)
            else:
                vals.append(cur)
    out: List[str] = []
    seen = set()
    for v in vals:
        s = _normalize_anchor_surface(v)
        if not s:
            continue
        key = _anchor_key(s)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        if key in _COVERAGE_STOPWORDS or key in _ANCHOR_GENERIC_WORDS:
            continue
        out.append(s)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def _alias_variants_from_surface(
    surface: str,
    max_alias: int = 8,
) -> List[str]:
    s0 = _normalize_anchor_surface(surface)
    if not s0:
        return []
    out: List[str] = []
    seen = set()

    def _push(x: str) -> None:
        sx = _normalize_anchor_surface(x)
        if not sx:
            return
        k = _anchor_key(sx)
        if len(k) < 2:
            return
        if k in seen:
            return
        if k in _COVERAGE_STOPWORDS or k in _ANCHOR_GENERIC_WORDS:
            return
        seen.add(k)
        out.append(sx)

    _push(s0)
    _push(s0.lower())
    _push(re.sub(r"[-_/]+", " ", s0))
    _push(re.sub(r"[^A-Za-z0-9 ]+", " ", s0))
    _push(re.sub(r"[^A-Za-z0-9]", "", s0))

    compact = re.sub(r"[^A-Za-z0-9]", "", s0)
    m = re.match(r"^([A-Za-z]+)([0-9]+)$", compact)
    if m:
        _push(f"{m.group(1)} {m.group(2)}")
        _push(f"{m.group(1)}-{m.group(2)}")

    toks = [t for t in re.split(r"[^A-Za-z0-9]+", s0) if t]
    if len(toks) >= 2:
        _push(" ".join(toks[:2]))
        _push(" ".join(toks[-2:]))
    if len(toks) == 1:
        t = toks[0]
        if len(t) <= 12:
            _push(f"module {t}")
            _push(f"stage {t}")
            _push(f"block {t}")
            _push(f"layer {t}")
            _push(f"component {t}")
            _push(f"node {t}")

    if len(out) > max(1, int(max_alias)):
        out = out[: max(1, int(max_alias))]
    return out


def _collect_entity_anchor_seeds(
    ocr_text: str,
    struct_nodes: List[str],
    max_items: int = 12,
) -> List[str]:
    ocr_items = clean_ocr_items(ocr_text, max_items=max(16, int(max_items) * 6))
    ocr_keys = [(_anchor_key(x), x) for x in ocr_items if _anchor_key(x)]
    out: List[str] = []
    seen = set()

    def _add_seed(x: str) -> None:
        sx = _normalize_anchor_surface(x)
        if not sx:
            return
        k = _anchor_key(sx)
        if len(k) < 2 or k in seen:
            return
        if k in _COVERAGE_STOPWORDS or k in _ANCHOR_GENERIC_WORDS:
            return
        seen.add(k)
        out.append(sx)

    # Prefer structure nodes when available.
    for node in struct_nodes or []:
        _add_seed(node)
        nk = _anchor_key(node)
        if not nk:
            continue
        # Pull OCR-near variants for this node to bridge naming mismatch.
        for ok, ov in ocr_keys:
            if not ok:
                continue
            if nk in ok or ok in nk:
                _add_seed(ov)
                break
        if len(out) >= max(1, int(max_items)):
            return out[: max(1, int(max_items))]

    for _, ov in ocr_keys:
        _add_seed(ov)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def _build_desc_anchor_alias_texts(
    ocr_text: str,
    struct_nodes: List[str],
    max_items: int = 12,
    max_alias_per_item: int = 8,
) -> List[str]:
    seeds = _collect_entity_anchor_seeds(
        ocr_text=ocr_text,
        struct_nodes=struct_nodes,
        max_items=max_items,
    )
    if not seeds:
        return []
    out: List[str] = []
    seen = set()
    max_total = max(4, int(max_items) * max(1, int(max_alias_per_item)))
    for s in seeds:
        aliases = _alias_variants_from_surface(s, max_alias=max_alias_per_item)
        for a in aliases:
            k = _anchor_key(a)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(a)
            if len(out) >= max_total:
                return out
    return out


def _extract_desc_slot_texts(
    text: str,
    max_slots: int = 8,
) -> List[str]:
    if not text:
        return []
    max_slots = max(1, int(max_slots))
    sents = re.split(r"[.!?;:\n]+", str(text))
    cand: List[Tuple[int, str]] = []
    for sent in sents:
        s = re.sub(r"\s+", " ", sent).strip()
        if len(s) < 6:
            continue
        sl = s.lower()
        hit = 0
        for hints in _DESC_SLOT_HINTS:
            if any(w in sl for w in hints):
                hit += 1
        if hit <= 0:
            continue
        slot = " ".join(s.split()[:14]).strip()
        if slot:
            cand.append((hit, slot))

    # fallback: lexical slots from GT description if sentence slots are sparse
    if len(cand) < 2:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_/-]{2,}", str(text))
        seen_w = set()
        for w in words:
            wl = w.lower()
            if wl in seen_w:
                continue
            seen_w.add(wl)
            core = re.sub(r"[^a-z0-9]", "", wl)
            if len(core) < 3 or core in _COVERAGE_STOPWORDS:
                continue
            if (
                wl in DESC_COMPONENT_HINTS
                or wl in DESC_REL_HINTS
                or wl in DESC_FLOW_HINTS
                or wl in DESC_FUNC_HINTS
                or wl.endswith(NOUN_SUFFIXES)
            ):
                cand.append((1, w))
            if len(cand) >= max_slots * 2:
                break

    # dedup with priority by hint count and phrase length
    cand.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    out: List[str] = []
    seen = set()
    for _, slot in cand:
        key = re.sub(r"\s+", " ", slot.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(slot)
        if len(out) >= max_slots:
            break
    return out


def _meta_first_text(meta: Optional[Dict], keys: List[str]) -> str:
    if not isinstance(meta, dict):
        return ""
    for key in keys:
        text = _flatten_text(meta.get(key))
        if text:
            return text
    return ""


def _compact_ocr_context_text(ocr_text: str, max_items: int = 48, max_chars: int = 900) -> str:
    items = clean_ocr_items(ocr_text, max_items=max_items)
    if not items:
        return ""
    kept: List[str] = []
    seen = set()
    for it in items:
        s = re.sub(r"\s+", " ", str(it or "")).strip()
        if not s:
            continue
        key = re.sub(r"[^a-z0-9]", "", s.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(s)
        if len(kept) >= max_items:
            break
    if not kept:
        return ""
    txt = " ; ".join(kept)
    if max_chars > 0 and len(txt) > max_chars:
        txt = txt[:max_chars].rstrip(" ;,")
    return txt.strip()


def _extract_tag_payload(text: str, tag: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", s, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return re.sub(r"\s+", " ", str(m.group(1) or "")).strip()


def _ensure_tagged_segment(text: str, tag: str, placeholder: str) -> str:
    payload = _extract_tag_payload(text, tag)
    if not payload:
        payload = re.sub(rf"</?{tag}>", " ", str(text or ""), flags=re.IGNORECASE)
        payload = re.sub(r"\s+", " ", payload).strip()
    if not payload:
        payload = placeholder
    return f"<{tag}>{payload}</{tag}>"


def _resolve_scicap_task_context(
    scale: str,
    context: str,
    context_meta: Optional[Dict],
    routing_mode: str,
    use_placeholders: bool = False,
) -> str:
    if routing_mode != "caption_para_desc_ocr":
        return (context or "").strip()
    if scale not in ("short", "long", "desc"):
        return (context or "").strip()

    paragraph = _meta_first_text(
        context_meta,
        ["paragraph", "scicap_paragraph", "forbidden_para_text"],
    )
    mention = _meta_first_text(context_meta, ["mention", "scicap_mention"])
    ocr = _meta_first_text(
        context_meta,
        ["ocr", "scicap_ocr", "allowed_ocr_text", "context_ocr"],
    )

    if scale in ("short", "long"):
        cap_ctx = paragraph or mention
        if use_placeholders:
            return _ensure_tagged_segment(cap_ctx, "PARA", NO_CONTEXT_TOKEN).strip()
        if not cap_ctx:
            return ""
        return f"<PARA>{cap_ctx}</PARA>".strip()
    # description branch: paragraph is explicitly forbidden.
    ocr_compact = _compact_ocr_context_text(ocr, max_items=48, max_chars=900)
    if use_placeholders:
        return _ensure_tagged_segment(ocr_compact, "OCR", NO_OCR_TOKEN).strip()
    if not ocr_compact:
        return ""
    return f"<OCR>{ocr_compact}</OCR>".strip()


def _unpack_multimodal_item(item: Tuple) -> Tuple:
    if not isinstance(item, tuple) or len(item) < 4:
        raise RuntimeError(f"unexpected sample format: {type(item)} / len={len(item) if isinstance(item, tuple) else 'NA'}")
    if len(item) >= 6:
        img, text, scale, context, regions, ctx_meta = item[:6]
        return img, text, scale, context, regions, ctx_meta
    if len(item) >= 5:
        img, text, scale, context, fifth = item[:5]
        if isinstance(fifth, dict):
            return img, text, scale, context, [], fifth
        return img, text, scale, context, fifth, None
    img, text, scale, context = item[:4]
    return img, text, scale, context, [], None


class SciCapTextOnlyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_json: str,
        task: str = "short",
        min_len_short: int = 20,
        min_len_long: int = 40,
        min_len_desc: int = 40,
    ) -> None:
        super().__init__()
        with open(split_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for art in data:
            for fig in art.get("figures", []) or []:
                meta = fig.get("metadata") or {}
                raw = meta.get("scicap_raw") or {}
                long_cap = _flatten_text(raw.get("mlbcap_long") or fig.get("figure_caption"))
                short_cap = _flatten_text(raw.get("mlbcap_short"))
                if not short_cap and long_cap:
                    short_cap = _first_sentence(long_cap)
                desc = _flatten_text(raw.get("figure_description"))
                if task == "short" and len(short_cap) >= min_len_short:
                    items.append(short_cap)
                elif task == "long" and len(long_cap) >= min_len_long:
                    items.append(long_cap)
                elif task == "desc" and len(desc) >= min_len_desc:
                    items.append(desc)
        if not items:
            raise RuntimeError(f"text-only dataset empty for task={task}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class StudentVisionTower(nn.Module):
    def __init__(
        self,
        student: CLIPVisionTower,
        pool_size: int = 1,
        pool_mode: str = "avg",
        max_region_tokens: int = 0,
        region_token_scale: float = 1.0,
        enable_explain_region_adapter: bool = False,
        explain_region_adapter_on_unknown: bool = False,
        enable_task_specific_region_adapter: bool = False,
        task_specific_region_adapter_on_unknown: bool = False,
    ):
        super().__init__()
        self.student = student
        self.pool_size = max(1, int(pool_size))
        self.pool_mode = pool_mode
        self.max_region_tokens = max(0, int(max_region_tokens))
        self.region_token_scale = float(region_token_scale)
        self.enable_explain_region_adapter = bool(enable_explain_region_adapter)
        self.explain_region_adapter_on_unknown = bool(explain_region_adapter_on_unknown)
        self.enable_task_specific_region_adapter = bool(enable_task_specific_region_adapter)
        self.task_specific_region_adapter_on_unknown = bool(task_specific_region_adapter_on_unknown)
        model_cfg = getattr(getattr(student, "model", None), "config", None)
        emb_dim = int(getattr(model_cfg, "hidden_size", 768))
        if emb_dim <= 0:
            emb_dim = 768
        # Trainable region type embedding to separate region tokens from patch tokens.
        self.region_type_embedding = nn.Parameter(torch.zeros(1, 1, emb_dim))
        if self.enable_explain_region_adapter:
            self.region_proj_explain = nn.Linear(emb_dim, emb_dim, bias=True)
            with torch.no_grad():
                nn.init.eye_(self.region_proj_explain.weight)
                nn.init.zeros_(self.region_proj_explain.bias)
        else:
            self.region_proj_explain = None
        if self.enable_task_specific_region_adapter:
            self.region_proj_task = nn.ModuleDict({
                k: nn.Linear(emb_dim, emb_dim, bias=True) for k in ("short", "long", "desc", "explain")
            })
            with torch.no_grad():
                for proj in self.region_proj_task.values():
                    nn.init.eye_(proj.weight)
                    nn.init.zeros_(proj.bias)
        else:
            self.region_proj_task = None
        self._regions: Optional[List[List[Dict]]] = None
        self._scales: Optional[List[str]] = None
        self._drop_one_region = False
        self.last_region_valid_mask: Optional[torch.Tensor] = None

    def set_regions(self, regions: Optional[List[List[Dict]]], drop_one_region: bool = False) -> None:
        self._regions = regions
        self._drop_one_region = bool(drop_one_region)

    def set_scales(self, scales: Optional[List[str]]) -> None:
        self._scales = scales

    def _use_explain_adapter_for_sample(self, sample_idx: int) -> bool:
        if self.region_proj_explain is None:
            return False
        if not self._scales:
            return bool(self.explain_region_adapter_on_unknown)
        if sample_idx >= len(self._scales):
            return bool(self.explain_region_adapter_on_unknown)
        return str(self._scales[sample_idx] or "") == "explain"

    def _task_region_adapter_for_sample(self, sample_idx: int) -> Optional[nn.Module]:
        if self.region_proj_task is None:
            return None
        scale_i = ""
        if self._scales and sample_idx < len(self._scales):
            scale_i = str(self._scales[sample_idx] or "")
        if scale_i in self.region_proj_task:
            return self.region_proj_task[scale_i]
        if self.task_specific_region_adapter_on_unknown:
            if "explain" in self.region_proj_task:
                return self.region_proj_task["explain"]
            return next(iter(self.region_proj_task.values()))
        return None

    def clear_regions(self) -> None:
        self._regions = None
        self._scales = None
        self._drop_one_region = False
        self.last_region_valid_mask = None

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool_size <= 1:
            return tokens
        if tokens.dim() != 3:
            return tokens
        bsz, num_tokens, dim = tokens.shape
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            return tokens
        if side % self.pool_size != 0:
            return tokens
        x = tokens.reshape(bsz, side, side, dim).permute(0, 3, 1, 2)
        if self.pool_mode == "max":
            x = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
        else:
            x = F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = x.permute(0, 2, 3, 1).reshape(bsz, -1, dim)
        return x

    def _append_region_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.max_region_tokens <= 0:
            self.last_region_valid_mask = None
            return tokens
        bsz, num_tokens, dim = tokens.shape
        if bsz == 0:
            self.last_region_valid_mask = None
            return tokens
        if not self._regions or len(self._regions) != bsz:
            zeros = tokens.new_zeros((bsz, self.max_region_tokens, dim))
            self.last_region_valid_mask = torch.zeros((bsz, self.max_region_tokens), device=tokens.device, dtype=torch.bool)
            type_emb = self.region_type_embedding.to(device=tokens.device, dtype=tokens.dtype)
            if type_emb.shape[-1] == dim:
                zeros = zeros + type_emb
            # Region tokens are prefixed before patch tokens.
            return torch.cat([zeros, tokens], dim=1)

        region_bank = tokens.new_zeros((bsz, self.max_region_tokens, dim))
        valid_mask = torch.zeros((bsz, self.max_region_tokens), device=tokens.device, dtype=torch.bool)
        for b in range(bsz):
            regs = list(self._regions[b] or [])
            if self._drop_one_region and len(regs) > 0:
                regs.pop(random.randrange(len(regs)))
            slot = 0
            for reg in regs:
                if slot >= self.max_region_tokens:
                    break
                idxs = reg.get("patch_indices", []) if isinstance(reg, dict) else []
                if not idxs:
                    continue
                idxs = sorted(set(int(x) for x in idxs if 0 <= int(x) < num_tokens))
                if not idxs:
                    continue
                idx_t = torch.tensor(idxs, device=tokens.device, dtype=torch.long)
                reg_emb = tokens[b].index_select(0, idx_t).mean(dim=0)
                task_adapter = self._task_region_adapter_for_sample(b)
                if task_adapter is not None:
                    reg_emb = task_adapter(reg_emb.unsqueeze(0)).squeeze(0)
                elif self._use_explain_adapter_for_sample(b):
                    reg_emb = self.region_proj_explain(reg_emb.unsqueeze(0)).squeeze(0)
                if self.region_token_scale != 1.0:
                    reg_emb = reg_emb * self.region_token_scale
                region_bank[b, slot] = reg_emb
                valid_mask[b, slot] = True
                slot += 1
        type_emb = self.region_type_embedding.to(device=tokens.device, dtype=tokens.dtype)
        if type_emb.shape[-1] == dim:
            region_bank = region_bank + type_emb
        self.last_region_valid_mask = valid_mask
        # Region tokens are prefixed before patch tokens.
        return torch.cat([region_bank, tokens], dim=1)

    def forward(self, images, **_kwargs):
        # images: pixel_values tensor (B, C, H, W)
        out = self.student(images)
        tokens = out["tokens"]  # (B, N, 768), includes CLS at 0
        if tokens.dim() == 3 and tokens.size(1) > 1:
            tokens = tokens[:, 1:]
        tokens = self._pool_tokens(tokens)
        tokens = self._append_region_tokens(tokens)
        if tokens.dtype != images.dtype:
            tokens = tokens.to(dtype=images.dtype)
        return tokens


def build_batch(
    tokenizer: AutoTokenizer,
    batch: List[Tuple],
    max_length: int,
    max_target_map: Dict[str, int],
    image_token_index: int,
    add_eos: bool,
    fixed_task: str | None = None,
    context_dropout: float = 0.0,
    paragraph_token_dropout: float = 0.0,
    max_ctx_tokens: int = 0,
    max_ctx_tokens_explain: int = 0,
    explain_ctx_min_adesc_tokens: int = 0,
    explain_ctx_max_ocr_tokens: int = 0,
    bucket_bins: List[int] | None = None,
    image_tokens: int = 0,
    scicap_prompt_style: str = "none",
    scicap_task_context_routing: str = "none",
    enable_task_style_tokens: bool = True,
    use_context_placeholders: bool = False,
):
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id or 0
    images: List = []
    texts: List[str] = []
    scales: List[str] = []
    contexts: List[str] = []
    regions_list: List = []
    context_meta_list: List[Optional[Dict]] = []
    for it in batch:
        img, text, scale, context, regions, ctx_meta = _unpack_multimodal_item(it)
        images.append(img)
        texts.append(text)
        scales.append(scale)
        contexts.append(context)
        regions_list.append(regions if regions is not None else [])
        context_meta_list.append(ctx_meta)
    input_ids_list = []
    labels_list = []
    contexts_used: List[str] = []
    context_allow_tokens: List[int] = []
    contexts_ocr: List[str] = []
    contexts_adesc: List[str] = []
    contexts_para: List[str] = []
    contexts_struct_nodes: List[List[str]] = []
    context_total_tokens: List[int] = []
    context_ocr_tokens: List[int] = []
    context_adesc_tokens: List[int] = []
    context_para_tokens: List[int] = []
    context_ocr_hash: List[int] = []
    context_adesc_hash: List[int] = []
    for idx, (text, scale, ctx) in enumerate(zip(texts, scales, contexts)):
        if fixed_task:
            scale = fixed_task
        task_token = TASK_TOKENS.get(scale, "<CAPTION_LONG>")
        task_ids = tokenizer(task_token, add_special_tokens=False)["input_ids"]
        if not task_ids:
            task_ids = [tokenizer.unk_token_id]
        task_style_ids = _task_style_ids(tokenizer, scale, enable_task_style_tokens)
        task_prompt_ids = _task_prompt_ids(tokenizer, scale, scicap_prompt_style)
        max_target = max_target_map.get(scale, max_length)
        target_cap = int(max_target) if (max_target and max_target > 0) else int(max_length if max_length > 0 else 0)
        target_ids = _tokenize_ids(
            tokenizer,
            text,
            add_special_tokens=False,
            max_length=target_cap,
        )
        if add_eos and tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]
        if max_target > 0 and len(target_ids) > max_target:
            target_ids = target_ids[:max_target]

        ctx_ids: List[int] = []
        ctx_allowed_ids: List[int] = []
        ctx_para_ids: List[int] = []
        ctx_allowed_ocr_n = 0
        ctx_allowed_adesc_n = 0
        ctx_text = (ctx or "").strip()
        ctx_meta = context_meta_list[idx] if idx < len(context_meta_list) else None
        ctx_allowed_text = ""
        ctx_ocr_text = ""
        ctx_adesc_text = ""
        ctx_para_text = ""
        ctx_struct_nodes: List[str] = []
        if isinstance(ctx_meta, dict):
            ctx_allowed_text = _flatten_text(ctx_meta.get("allowed_text"))
            ctx_ocr_text = _flatten_text(ctx_meta.get("allowed_ocr_text"))
            ctx_adesc_text = _flatten_text(ctx_meta.get("allowed_desc_text"))
            ctx_para_text = _flatten_text(ctx_meta.get("forbidden_para_text"))
            ctx_struct_nodes = _extract_struct_node_texts_from_meta(ctx_meta, max_items=64)
        if scicap_task_context_routing != "none":
            routed_ctx = _resolve_scicap_task_context(
                scale=scale,
                context=ctx_text,
                context_meta=ctx_meta if isinstance(ctx_meta, dict) else None,
                routing_mode=scicap_task_context_routing,
                use_placeholders=use_context_placeholders,
            )
            if scale in ("short", "long"):
                ctx_para_text = routed_ctx
                ctx_allowed_text = routed_ctx
                ctx_ocr_text = ""
                ctx_adesc_text = ""
            elif scale == "desc":
                ctx_ocr_text = routed_ctx
                ctx_allowed_text = routed_ctx
                ctx_para_text = ""
                ctx_adesc_text = ""
            ctx_text = routed_ctx

        if use_context_placeholders:
            if scale in ("short", "long"):
                ctx_para_text = _ensure_tagged_segment(ctx_para_text or ctx_text, "PARA", NO_CONTEXT_TOKEN)
                ctx_allowed_text = ctx_para_text
                ctx_ocr_text = ""
                ctx_adesc_text = ""
                ctx_text = ctx_para_text
            elif scale == "desc":
                ctx_ocr_text = _ensure_tagged_segment(ctx_ocr_text or ctx_text, "OCR", NO_OCR_TOKEN)
                ctx_allowed_text = ctx_ocr_text
                ctx_para_text = ""
                ctx_adesc_text = ""
                ctx_text = ctx_ocr_text
            elif scale == "explain":
                ctx_ocr_text = _ensure_tagged_segment(ctx_ocr_text, "OCR", NO_OCR_TOKEN)
                ctx_adesc_text = _ensure_tagged_segment(ctx_adesc_text, "ADESC", NO_ADESC_TOKEN)
                ctx_para_text = _ensure_tagged_segment(ctx_para_text, "PARA", NO_CONTEXT_TOKEN)
                ctx_allowed_text = _flatten_text([ctx_ocr_text, ctx_adesc_text])
                ctx_text = _flatten_text([ctx_allowed_text, ctx_para_text])

        ctx_limit = max_ctx_tokens
        if scale == "explain" and max_ctx_tokens_explain and max_ctx_tokens_explain > 0:
            ctx_limit = max_ctx_tokens_explain
        if ctx_text and context_dropout > 0 and random.random() < context_dropout:
            ctx_text = ""
        apply_para_dropout = paragraph_token_dropout > 0 and not (
            scicap_task_context_routing != "none" and scale == "desc"
        )
        if ctx_text and apply_para_dropout:
            ctx_text = mask_paragraph_text(ctx_text, paragraph_token_dropout)
        if use_context_placeholders and not ctx_text:
            if scale in ("short", "long"):
                ctx_para_text = _ensure_tagged_segment("", "PARA", NO_CONTEXT_TOKEN)
                ctx_allowed_text = ctx_para_text
                ctx_text = ctx_para_text
            elif scale == "desc":
                ctx_ocr_text = _ensure_tagged_segment("", "OCR", NO_OCR_TOKEN)
                ctx_allowed_text = ctx_ocr_text
                ctx_text = ctx_ocr_text
            elif scale == "explain":
                ctx_ocr_text = _ensure_tagged_segment("", "OCR", NO_OCR_TOKEN)
                ctx_adesc_text = _ensure_tagged_segment("", "ADESC", NO_ADESC_TOKEN)
                ctx_para_text = _ensure_tagged_segment("", "PARA", NO_CONTEXT_TOKEN)
                ctx_allowed_text = _flatten_text([ctx_ocr_text, ctx_adesc_text])
                ctx_text = _flatten_text([ctx_allowed_text, ctx_para_text])
        use_segment_budget = bool(
            scale == "explain"
            and ctx_limit
            and ctx_limit > 0
            and (ctx_ocr_text or ctx_adesc_text or ctx_para_text)
        )
        if use_segment_budget:
            seg_cap = int(ctx_limit) if (ctx_limit and ctx_limit > 0) else int(max_length if max_length > 0 else 0)
            ocr_ids_full = _tokenize_ids(tokenizer, "\n" + ctx_ocr_text, add_special_tokens=False, max_length=seg_cap) if ctx_ocr_text else []
            adesc_ids_full = _tokenize_ids(tokenizer, "\n" + ctx_adesc_text, add_special_tokens=False, max_length=seg_cap) if ctx_adesc_text else []
            para_ids_full = _tokenize_ids(tokenizer, "\n" + ctx_para_text, add_special_tokens=False, max_length=seg_cap) if ctx_para_text else []
            if not ctx_text:
                # Context dropout clears every segment for this sample.
                ocr_ids_full = []
                adesc_ids_full = []
                para_ids_full = []
            ctx_budget = int(max(0, ctx_limit))
            min_adesc = int(max(0, explain_ctx_min_adesc_tokens))
            if not adesc_ids_full:
                min_adesc = 0
            min_adesc = min(min_adesc, ctx_budget)
            ocr_cap = max(0, ctx_budget - min_adesc)
            if explain_ctx_max_ocr_tokens and explain_ctx_max_ocr_tokens > 0:
                ocr_cap = min(ocr_cap, int(explain_ctx_max_ocr_tokens))
            ocr_take = min(len(ocr_ids_full), ocr_cap)
            rem_after_ocr = max(0, ctx_budget - ocr_take)
            adesc_take = min(len(adesc_ids_full), rem_after_ocr)
            if min_adesc > 0 and adesc_take < min_adesc and len(adesc_ids_full) >= min_adesc:
                need = min_adesc - adesc_take
                shift = min(need, ocr_take)
                if shift > 0:
                    ocr_take -= shift
                    rem_after_ocr = max(0, ctx_budget - ocr_take)
                    adesc_take = min(len(adesc_ids_full), rem_after_ocr)
            ocr_ids_vis = ocr_ids_full[:ocr_take]
            adesc_ids_vis = adesc_ids_full[:adesc_take]
            ctx_allowed_ids = ocr_ids_vis + adesc_ids_vis
            ctx_allowed_ocr_n = int(len(ocr_ids_vis))
            ctx_allowed_adesc_n = int(len(adesc_ids_vis))
            para_budget = max(0, ctx_budget - len(ctx_allowed_ids))
            para_ids_vis = para_ids_full[:para_budget] if para_budget > 0 else []
            ctx_para_ids = list(para_ids_vis)
            ctx_ids = ctx_allowed_ids + ctx_para_ids
            if ctx_ids:
                ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=True).strip()
            else:
                ctx_text = ""
        else:
            ctx_cap = int(ctx_limit) if (ctx_limit and ctx_limit > 0) else int(max_length if max_length > 0 else 0)
            if ctx_text:
                ctx_ids = _tokenize_ids(tokenizer, "\n" + ctx_text, add_special_tokens=False, max_length=ctx_cap)
            if ctx_allowed_text:
                ctx_allowed_ids = _tokenize_ids(tokenizer, "\n" + ctx_allowed_text, add_special_tokens=False, max_length=ctx_cap)
            if ctx_limit and ctx_ids:
                ctx_ids = ctx_ids[:ctx_limit]
                # approximate used context by truncation
                ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=True).strip()
            if ctx_limit and ctx_allowed_ids:
                ctx_allowed_ids = ctx_allowed_ids[:ctx_limit]
        contexts_used.append(ctx_text)
        contexts_ocr.append(ctx_ocr_text)
        contexts_adesc.append(ctx_adesc_text)
        contexts_para.append(ctx_para_text)
        contexts_struct_nodes.append(ctx_struct_nodes)

        base_prompt = [bos_id] + task_ids + task_style_ids + task_prompt_ids + [image_token_index]
        if max_length > 0:
            # reserve space for expanded image tokens (approx)
            image_tokens = max(0, image_tokens - 1)
            max_ctx = max_length - len(base_prompt) - len(target_ids) - image_tokens
            if max_ctx < 0:
                # target too long, trim target to fit base prompt
                max_tgt = max(0, max_length - len(base_prompt) - image_tokens)
                target_ids = target_ids[:max_tgt]
                max_ctx = 0
            if max_ctx >= 0 and ctx_allowed_ids:
                if use_segment_budget and scale == "explain":
                    if len(ctx_allowed_ids) > max_ctx:
                        keep_adesc = min(ctx_allowed_adesc_n, max_ctx)
                        keep_ocr = min(ctx_allowed_ocr_n, max_ctx - keep_adesc)
                        adesc_start = ctx_allowed_ocr_n
                        ctx_allowed_ids = (
                            ctx_allowed_ids[:keep_ocr]
                            + ctx_allowed_ids[adesc_start : adesc_start + keep_adesc]
                        )
                        ctx_allowed_ocr_n = int(keep_ocr)
                        ctx_allowed_adesc_n = int(keep_adesc)
                    para_room = max(0, max_ctx - len(ctx_allowed_ids))
                    ctx_ids = ctx_allowed_ids + ctx_para_ids[:para_room]
                else:
                    ctx_allowed_ids = ctx_allowed_ids[:max_ctx]
                    if max_ctx >= 0 and ctx_ids:
                        ctx_ids = ctx_ids[:max_ctx]
            elif max_ctx >= 0 and ctx_ids:
                ctx_ids = ctx_ids[:max_ctx]

        prompt_ids = base_prompt + ctx_ids
        input_ids = prompt_ids + target_ids
        if max_length > 0 and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        labels = [-100] * len(prompt_ids) + target_ids
        if max_length > 0 and len(labels) > max_length:
            labels = labels[:max_length]
        if ctx_allowed_ids:
            allow_n = min(len(ctx_ids), len(ctx_allowed_ids))
        else:
            allow_n = 0
        total_ctx_n = len(ctx_ids)
        ocr_n = 0
        adesc_n = 0
        para_n = max(0, total_ctx_n - int(allow_n))
        hash_ocr = 0
        hash_adesc = 0
        if use_segment_budget and scale == "explain":
            ocr_n = min(int(allow_n), int(max(0, ctx_allowed_ocr_n)))
            adesc_n = min(int(max(0, allow_n - ocr_n)), int(max(0, ctx_allowed_adesc_n)))
            ocr_ids_vis = ctx_allowed_ids[: int(ocr_n)] if ocr_n > 0 else []
            adesc_ids_vis = ctx_allowed_ids[int(ocr_n) : int(ocr_n + adesc_n)] if adesc_n > 0 else []
            hash_ocr = _hash_token_ids(ocr_ids_vis)
            hash_adesc = _hash_token_ids(adesc_ids_vis)
        elif allow_n > 0 and (ctx_ocr_text or ctx_adesc_text):
            pref_cap = int(len(ctx_allowed_ids)) if ctx_allowed_ids else int(ctx_limit if (ctx_limit and ctx_limit > 0) else (max_length if max_length > 0 else 0))
            ocr_pref_ids = _tokenize_ids(tokenizer, "\n" + ctx_ocr_text, add_special_tokens=False, max_length=pref_cap) if ctx_ocr_text else []
            ocr_adesc_pref_text = _flatten_text([ctx_ocr_text, ctx_adesc_text])
            ocr_adesc_pref_ids = _tokenize_ids(tokenizer, "\n" + ocr_adesc_pref_text, add_special_tokens=False, max_length=pref_cap) if ocr_adesc_pref_text else []
            ocr_cap = min(len(ocr_pref_ids), len(ctx_allowed_ids))
            ocr_n = min(int(allow_n), int(ocr_cap))
            ocr_adesc_cap = min(len(ocr_adesc_pref_ids), len(ctx_allowed_ids))
            adesc_cap = max(0, int(ocr_adesc_cap) - int(ocr_cap))
            adesc_n = max(0, min(adesc_cap, int(allow_n) - int(ocr_n)))
            if ocr_n <= 0 and not ctx_ocr_text and ctx_adesc_text:
                adesc_n = min(int(allow_n), int(ocr_adesc_cap))
            ocr_ids_vis = ctx_allowed_ids[: int(ocr_n)] if ocr_n > 0 else []
            adesc_ids_vis = ctx_allowed_ids[int(ocr_n) : int(ocr_n + adesc_n)] if adesc_n > 0 else []
            hash_ocr = _hash_token_ids(ocr_ids_vis)
            hash_adesc = _hash_token_ids(adesc_ids_vis)
        context_total_tokens.append(int(total_ctx_n))
        context_ocr_tokens.append(int(ocr_n))
        context_adesc_tokens.append(int(adesc_n))
        context_para_tokens.append(int(para_n))
        context_ocr_hash.append(int(hash_ocr))
        context_adesc_hash.append(int(hash_adesc))
        context_allow_tokens.append(int(max(0, allow_n)))
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(x) for x in input_ids_list) if input_ids_list else 1
    if max_length > 0:
        max_len = min(max_len, max_length)
    if bucket_bins:
        eff_len = max_len + max(0, image_tokens - 1)
        bucket_max = None
        for b in bucket_bins:
            if eff_len <= b:
                bucket_max = b
                break
        if bucket_max is None:
            bucket_max = eff_len
        pad_to = max(1, bucket_max - max(0, image_tokens - 1))
        if max_length > 0:
            pad_to = min(pad_to, max_length)
        max_len = max_len if pad_to < max_len else pad_to

    input_ids_padded = []
    labels_padded = []
    attention_mask = []
    for ids, labs in zip(input_ids_list, labels_list):
        ids = ids[:max_len]
        labs = labs[:max_len]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
            labs = labs + [-100] * pad_len
        mask = [1] * (max_len - pad_len) + [0] * pad_len
        input_ids_padded.append(ids)
        labels_padded.append(labs)
        attention_mask.append(mask)

    batch_out = {
        "images": images,
        "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
        "labels": torch.tensor(labels_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "texts": list(texts),
        "scales": list(scales),
        "contexts": list(contexts),
        "contexts_used": contexts_used,
        "contexts_ocr": contexts_ocr,
        "contexts_adesc": contexts_adesc,
        "contexts_para": contexts_para,
        "contexts_struct_nodes": contexts_struct_nodes,
        "context_total_tokens": torch.tensor(context_total_tokens, dtype=torch.long),
        "context_ocr_tokens": torch.tensor(context_ocr_tokens, dtype=torch.long),
        "context_adesc_tokens": torch.tensor(context_adesc_tokens, dtype=torch.long),
        "context_para_tokens": torch.tensor(context_para_tokens, dtype=torch.long),
        "context_ocr_hash": torch.tensor(context_ocr_hash, dtype=torch.long),
        "context_adesc_hash": torch.tensor(context_adesc_hash, dtype=torch.long),
        "context_allow_tokens": torch.tensor(context_allow_tokens, dtype=torch.long),
        "regions": list(regions_list),
    }
    return batch_out


def build_warmup_batch(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    max_target: int,
    task_token: str,
    add_eos: bool,
):
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id or 0
    task_ids = tokenizer(task_token, add_special_tokens=False)["input_ids"]
    if not task_ids:
        task_ids = [tokenizer.unk_token_id]

    input_ids_list = []
    labels_list = []
    for text in texts:
        target_cap = int(max_target) if (max_target and max_target > 0) else int(max_length if max_length > 0 else 0)
        target_ids = _tokenize_ids(tokenizer, text, add_special_tokens=False, max_length=target_cap)
        if add_eos and tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]
        if max_target > 0 and len(target_ids) > max_target:
            target_ids = target_ids[:max_target]
        prompt_ids = [bos_id] + task_ids
        input_ids = prompt_ids + target_ids
        if max_length > 0 and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        labels = [-100] * len(prompt_ids) + target_ids
        if max_length > 0 and len(labels) > max_length:
            labels = labels[:max_length]
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(x) for x in input_ids_list) if input_ids_list else 1
    if max_length > 0:
        max_len = min(max_len, max_length)

    input_ids_padded = []
    labels_padded = []
    attention_mask = []
    for ids, labs in zip(input_ids_list, labels_list):
        ids = ids[:max_len]
        labs = labs[:max_len]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
            labs = labs + [-100] * pad_len
        mask = [1] * (max_len - pad_len) + [0] * pad_len
        input_ids_padded.append(ids)
        labels_padded.append(labs)
        attention_mask.append(mask)

    return {
        "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
        "labels": torch.tensor(labels_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _force_eager_attn(module: nn.Module, name: str) -> None:
    if module is None:
        return
    try:
        if hasattr(module, "set_attn_implementation"):
            module.set_attn_implementation("eager")
    except Exception as e:
        print(f"[warn] failed set_attn_implementation(eager) for {name}: {e}")
    cfg = getattr(module, "config", None)
    if cfg is not None:
        for attr in ("attn_implementation", "_attn_implementation", "_attn_implementation_internal"):
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, "eager")
                except Exception:
                    pass


def _get_llm_layers(llm: nn.Module):
    # Try common attribute paths first
    for path in ("model.layers", "transformer.layers", "gpt_neox.layers", "layers"):
        cur = llm
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, (nn.ModuleList, list)) and len(cur) > 0:
            return cur
    # Fallback: search module lists matching num_hidden_layers
    target = getattr(getattr(llm, "config", None), "num_hidden_layers", None)
    for _, module in llm.named_modules():
        if isinstance(module, nn.ModuleList) and target and len(module) == target:
            return module
    return None


def build_region_attention_bias_ranges(
    input_ids: torch.Tensor,
    scales: Optional[List[str]],
    region_presence: Optional[List[bool]],
    image_token_index: int,
    image_tokens_total: int,
    region_token_slots: int,
    beta: float,
    task: str = "explain",
) -> Optional[List[Optional[Tuple[int, int, float]]]]:
    if beta <= 0 or region_token_slots <= 0 or image_tokens_total <= 0:
        return None
    if input_ids is None or input_ids.dim() != 2:
        return None
    bsz = input_ids.size(0)
    out: List[Optional[Tuple[int, int, float]]] = [None] * bsz
    any_valid = False
    for i in range(bsz):
        scale_i = ""
        if scales is not None and i < len(scales):
            scale_i = str(scales[i] or "")
        if scale_i and scale_i != task:
            continue
        if region_presence is not None and i < len(region_presence) and not bool(region_presence[i]):
            continue
        pos = (input_ids[i] == image_token_index).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        img_pos = int(pos[0].item())
        # Region token slots are placed at the head of expanded visual tokens.
        start = img_pos
        end = img_pos + region_token_slots
        if end <= start:
            continue
        out[i] = (start, end, float(beta))
        any_valid = True
    if not any_valid:
        return None
    return out


def bind_region_attention_bias(mm_model: nn.Module, num_layers: int) -> int:
    if num_layers <= 0:
        return 0
    llm = getattr(mm_model, "language_model", None)
    if llm is None:
        return 0
    layers = _get_llm_layers(llm)
    if layers is None:
        return 0
    n = min(int(num_layers), len(layers))
    wrapped = 0
    for i in range(n):
        layer = layers[i]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "_region_bias_wrapped", False):
            continue
        attn._region_bias_orig_forward = attn.forward

        def _forward_with_region_bias(
            self_attn,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor],
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            ranges = getattr(mm_model, "_region_attn_bias_ranges", None)
            if attention_mask is not None and ranges:
                try:
                    bsz = int(attention_mask.shape[0])
                    k_len = int(attention_mask.shape[-1])
                    bias = attention_mask.new_zeros((bsz, 1, 1, k_len))
                    any_bias = False
                    for bi in range(min(bsz, len(ranges))):
                        item = ranges[bi]
                        if item is None:
                            continue
                        entries: List[Tuple[int, int, float]]
                        if isinstance(item, tuple) and len(item) == 3:
                            entries = [item]
                        elif isinstance(item, list):
                            entries = []
                            for it in item:
                                if isinstance(it, tuple) and len(it) == 3:
                                    entries.append(it)
                        else:
                            continue
                        for s, e, b in entries:
                            s = max(0, min(k_len, int(s)))
                            e = max(s, min(k_len, int(e)))
                            if e <= s:
                                continue
                            bias[bi, 0, 0, s:e] = bias[bi, 0, 0, s:e] + float(b)
                            any_bias = True
                    if any_bias:
                        attention_mask = attention_mask + bias
                except Exception:
                    pass
            return self_attn._region_bias_orig_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        attn.forward = types.MethodType(_forward_with_region_bias, attn)
        attn._region_bias_wrapped = True
        wrapped += 1
    return wrapped


def _parse_bias_entries(item) -> List[Tuple[int, int, float]]:
    entries: List[Tuple[int, int, float]] = []
    if item is None:
        return entries
    if isinstance(item, tuple) and len(item) == 3:
        entries.append((int(item[0]), int(item[1]), float(item[2])))
        return entries
    if isinstance(item, list):
        for it in item:
            if isinstance(it, tuple) and len(it) == 3:
                entries.append((int(it[0]), int(it[1]), float(it[2])))
    return entries


def compute_explain_attn_metrics(
    attn_probs: Optional[List[torch.Tensor]],
    q_indices: List[torch.Tensor],
    k_region_indices: List[torch.Tensor],
    k_textctx_indices: List[torch.Tensor],
    k_image_indices: List[torch.Tensor],
    k_ocr_indices: Optional[List[torch.Tensor]] = None,
    k_adesc_indices: Optional[List[torch.Tensor]] = None,
    k_para_indices: Optional[List[torch.Tensor]] = None,
    visible_k_mask: Optional[List[torch.Tensor]] = None,
    last_k_layers: int = 4,
    avg_over_heads: bool = True,
) -> Dict[str, float]:
    if not attn_probs:
        return {
            "attn_to_region": 0.0,
            "attn_to_img": 0.0,
            "attn_to_textctx": 0.0,
            "attn_to_ocr": 0.0,
            "attn_to_adesc": 0.0,
            "attn_to_para": 0.0,
            "layers": 0,
            "samples": 0,
        }
    layers = list(attn_probs)
    if last_k_layers > 0:
        layers = layers[-min(len(layers), int(last_k_layers)):]
    total_region = 0.0
    total_img = 0.0
    total_ctx = 0.0
    total_ocr = 0.0
    total_adesc = 0.0
    total_para = 0.0
    total_samples = 0
    used_layers = 0
    eps = 1e-12
    for att in layers:
        if att is None or att.dim() != 4:
            continue
        bsz, num_heads, q_len, k_len = att.shape
        if bsz <= 0 or num_heads <= 0 or q_len <= 0 or k_len <= 0:
            continue
        used_layers += 1
        for bi in range(min(bsz, len(q_indices))):
            q_idx = q_indices[bi]
            if q_idx is None or q_idx.numel() == 0:
                continue
            q_idx = q_idx.to(device=att.device, dtype=torch.long)
            q_idx = q_idx[(q_idx >= 0) & (q_idx < q_len)]
            if q_idx.numel() == 0:
                continue
            q_att = att[bi, :, q_idx, :]  # [H, Qs, K]
            if q_att.numel() == 0:
                continue
            if visible_k_mask is not None and bi < len(visible_k_mask):
                vm = visible_k_mask[bi]
                if vm is not None and vm.numel() > 0:
                    vm = vm.to(device=att.device, dtype=torch.bool)
                    if vm.numel() != k_len:
                        if vm.numel() < k_len:
                            pad = torch.ones((k_len - vm.numel(),), dtype=torch.bool, device=att.device)
                            vm = torch.cat([vm, pad], dim=0)
                        else:
                            vm = vm[:k_len]
                    q_att = q_att * vm.view(1, 1, k_len).to(dtype=q_att.dtype)
            denom = q_att.sum(dim=-1).clamp_min(eps)  # [H, Qs]

            def _mass_for(idx: torch.Tensor) -> torch.Tensor:
                if idx is None or idx.numel() == 0:
                    return torch.zeros_like(denom)
                idx = idx.to(device=att.device, dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < k_len)]
                if idx.numel() == 0:
                    return torch.zeros_like(denom)
                part = q_att.index_select(-1, idx).sum(dim=-1)
                return part / denom

            m_region = _mass_for(k_region_indices[bi] if bi < len(k_region_indices) else torch.empty(0, dtype=torch.long))
            m_img = _mass_for(k_image_indices[bi] if bi < len(k_image_indices) else torch.empty(0, dtype=torch.long))
            m_ctx = _mass_for(k_textctx_indices[bi] if bi < len(k_textctx_indices) else torch.empty(0, dtype=torch.long))
            m_ocr = _mass_for(k_ocr_indices[bi] if (k_ocr_indices is not None and bi < len(k_ocr_indices)) else torch.empty(0, dtype=torch.long))
            m_adesc = _mass_for(k_adesc_indices[bi] if (k_adesc_indices is not None and bi < len(k_adesc_indices)) else torch.empty(0, dtype=torch.long))
            m_para = _mass_for(k_para_indices[bi] if (k_para_indices is not None and bi < len(k_para_indices)) else torch.empty(0, dtype=torch.long))
            if avg_over_heads:
                total_region += float(m_region.mean().item())
                total_img += float(m_img.mean().item())
                total_ctx += float(m_ctx.mean().item())
                total_ocr += float(m_ocr.mean().item())
                total_adesc += float(m_adesc.mean().item())
                total_para += float(m_para.mean().item())
            else:
                total_region += float(m_region.sum(dim=0).mean().item())
                total_img += float(m_img.sum(dim=0).mean().item())
                total_ctx += float(m_ctx.sum(dim=0).mean().item())
                total_ocr += float(m_ocr.sum(dim=0).mean().item())
                total_adesc += float(m_adesc.sum(dim=0).mean().item())
                total_para += float(m_para.sum(dim=0).mean().item())
            total_samples += 1
    if total_samples <= 0:
        return {
            "attn_to_region": 0.0,
            "attn_to_img": 0.0,
            "attn_to_textctx": 0.0,
            "attn_to_ocr": 0.0,
            "attn_to_adesc": 0.0,
            "attn_to_para": 0.0,
            "layers": used_layers,
            "samples": 0,
        }
    return {
        "attn_to_region": total_region / float(total_samples),
        "attn_to_img": total_img / float(total_samples),
        "attn_to_textctx": total_ctx / float(total_samples),
        "attn_to_ocr": total_ocr / float(total_samples),
        "attn_to_adesc": total_adesc / float(total_samples),
        "attn_to_para": total_para / float(total_samples),
        "layers": used_layers,
        "samples": total_samples,
    }


def _mask_embedding_grads(emb: nn.Embedding, keep_ids: List[int]) -> None:
    if emb is None or emb.weight.grad is None or not keep_ids:
        return
    grad = emb.weight.grad
    keep = torch.tensor(keep_ids, device=grad.device, dtype=torch.long)
    kept = grad[keep].clone()
    grad.zero_()
    grad[keep] = kept


def mask_paragraph_text(text: str, drop_prob: float) -> str:
    if not text or drop_prob <= 0:
        return text
    out = text.strip()
    # remove leading figure/title sentence
    sentences = re.split(r"(?<=[\\.!?])\\s+", out)
    if sentences:
        head = sentences[0].strip()
        if re.match(r"^(fig\\.?|figure|table)\\b", head, flags=re.IGNORECASE):
            sentences = sentences[1:]
    out = " ".join(sentences).strip()
    for pat in METHOD_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)

    tokens = re.findall(r"[A-Za-z0-9]+|[^\\w\\s]", out)
    masked = []
    for tok in tokens:
        if not tok.isalnum():
            masked.append(tok)
            continue
        low = tok.lower()
        if low in KEEP_WORDS:
            masked.append(tok)
            continue
        # method/model names (caps, digits)
        if re.search(r"[A-Z]", tok) and len(tok) > 2:
            if random.random() < min(0.9, drop_prob * 2):
                continue
        # model-like tokens (caps/digits)
        if re.fullmatch(r"[A-Z][A-Za-z0-9_-]{2,}", tok) or re.fullmatch(r"[A-Z]{2,}\\d*", tok):
            if random.random() < min(0.9, drop_prob * 2):
                continue
        if any(low.endswith(suf) for suf in NOUN_SUFFIXES):
            if random.random() < drop_prob:
                continue
        if low in ("method", "model", "approach", "framework", "architecture", "algorithm"):
            if random.random() < drop_prob:
                continue
        masked.append(tok)

    # rebuild text with spacing
    rebuilt = []
    for i, tok in enumerate(masked):
        if i == 0:
            rebuilt.append(tok)
            continue
        if re.fullmatch(r"[\\.,;:\\)\\]\\}]", tok):
            rebuilt.append(tok)
        elif re.fullmatch(r"[\\(\\[\\{]", tok):
            rebuilt.append(" " + tok)
        else:
            rebuilt.append(" " + tok)
    return "".join(rebuilt).strip()


def _max_token_repeat_ratio(text: str) -> float:
    toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    if not toks:
        return 1.0
    counts: Dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    max_rep = max(counts.values()) if counts else 0
    return float(max_rep) / float(len(toks))


def _is_garbled_paragraph_line(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 80:
        return False
    alnum_toks = re.findall(r"[A-Za-z0-9]+", s)
    if not alnum_toks:
        return True
    alpha_chars = [ch for ch in s if ch.isalpha()]
    alpha_n = len(alpha_chars)
    upper_n = sum(1 for ch in alpha_chars if ch.isupper())
    vowel_n = sum(1 for ch in alpha_chars if ch.lower() in "aeiou")
    single_tok_ratio = sum(1 for t in alnum_toks if len(t) == 1) / float(max(1, len(alnum_toks)))
    upper_ratio = float(upper_n) / float(max(1, alpha_n))
    vowel_ratio = float(vowel_n) / float(max(1, alpha_n))
    noise_punct = s.count("/") + s.count("\\") + s.count("=") + s.count("|")
    repeat_ratio = _max_token_repeat_ratio(s)
    if len(alnum_toks) >= 40 and single_tok_ratio >= 0.45 and upper_ratio >= 0.70:
        return True
    if alpha_n >= 120 and upper_ratio >= 0.75 and vowel_ratio < 0.18 and noise_punct >= 8:
        return True
    if len(alnum_toks) >= 30 and repeat_ratio >= 0.45:
        return True
    return False


def clean_paragraph_text(text: str) -> str:
    if not text:
        return ""
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    raw = raw.replace("\uFFFD", " ")
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", raw) if ln.strip()]
    if not lines:
        return ""
    kept = [ln for ln in lines if not _is_garbled_paragraph_line(ln)]
    if not kept:
        return ""
    out = " ".join(kept)
    out = re.sub(r"\s+", " ", out).strip()
    # hard cap on raw characters before tokenization to avoid pathological lines
    if len(out) > 4000:
        out = out[:4000].rstrip()
    return out


def _flatten_maybe_json_text(val) -> str:
    obj = val
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
            except Exception:
                obj = val
    return _flatten_text(obj)


def clean_ocr_items(val, max_items: int = 64) -> List[str]:
    items = val
    if isinstance(items, str):
        s = items.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                items = json.loads(s)
            except Exception:
                items = [items]
        else:
            # OCR caches are often stored as a single ';'-joined string.
            # Split before filtering so tokens are not dropped as one long span.
            if ";" in s or "\n" in s or "|" in s:
                chunks = re.split(r"[;\n|]+", s)
                items = [c.strip() for c in chunks if c and c.strip()]
            else:
                items = [items]
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, (list, tuple)):
        items = [items]
    out: List[str] = []
    for it in items:
        s = _flatten_text(it)
        if not s:
            continue
        s = re.sub(r"\s+", " ", s).strip()
        core = re.sub(r"[^A-Za-z0-9]", "", s)
        if len(core) < 2 or len(core) > 60:
            continue
        if len(set(core.lower())) == 1 and len(core) >= 3:
            continue
        out.append(s)
        if max_items > 0 and len(out) >= max_items:
            break
    return out


def clean_desc_text(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    out = out.replace("\uFFFD", " ")
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > 2000:
        out = out[:2000].rstrip()
    return out


def build_explain_context_strings(paragraph: str, auto_desc: str, ocr_text: str) -> Dict[str, str]:
    para = clean_paragraph_text(paragraph or "")
    adesc = clean_desc_text(auto_desc or "")
    ocr = clean_desc_text(ocr_text or "")
    ocr_seg = f"<OCR>{ocr}</OCR>" if ocr else ""
    adesc_seg = f"<ADESC>{adesc}</ADESC>" if adesc else ""
    para_seg = f"<PARA>{para}</PARA>" if para else ""
    allowed = _flatten_text([ocr_seg, adesc_seg])
    full = _flatten_text([allowed, para_seg])
    return {
        "ocr_seg": ocr_seg,
        "adesc_seg": adesc_seg,
        "para_seg": para_seg,
        "allowed_ctx": allowed,
        "full_ctx": full,
    }


def _apply_path_replacements(path: str, replacements: List[Tuple[str, str]]) -> str:
    if not path:
        return path
    for src, dst in replacements:
        if path.startswith(src):
            return path.replace(src, dst, 1)
    return path


class SciStructExplainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_json: str,
        min_caption_len: int = 40,
        max_items: int | None = None,
        context_mode: str = "paragraph",
        max_masks: int = 16,
        mask_min_area_ratio: float = 0.002,
        mask_max_area_ratio: float = 0.85,
        mask_sort: str = "area",
        region_grid_size: int = 0,
        path_replace: List[Tuple[str, str]] | None = None,
        use_ocr_context: bool = True,
        use_adesc_context: bool = True,
    ):
        self.samples: List[Dict] = []
        self.min_caption_len = min_caption_len
        self.context_mode = context_mode
        self.use_ocr_context = bool(use_ocr_context)
        self.use_adesc_context = bool(use_adesc_context)
        self.max_masks = max_masks
        self.mask_min_area_ratio = mask_min_area_ratio
        self.mask_max_area_ratio = mask_max_area_ratio
        self.mask_sort = mask_sort
        self.region_grid_size = max(0, int(region_grid_size))
        self.path_replace = path_replace or []
        if not self.path_replace:
            default_dst = str(Path(split_json).parents[1] / "dataset")
            default_src = os.environ.get("SCISTRUCT_LEGACY_DATASET_ROOT", "").strip()
            if default_src:
                self.path_replace = [(default_src, default_dst)]
        self._anno_cache: Dict[str, Tuple[List[Dict], Dict[str, int]]] = {}
        self._region_cache: Dict[Tuple[str, int, int], List[Dict]] = {}

        with open(split_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            for fig in item.get("figures", []) or []:
                caption = _flatten_text(fig.get("figure_caption"))
                if len(caption) < self.min_caption_len:
                    continue
                fig_path = _apply_path_replacements(fig.get("figure_path", ""), self.path_replace)
                res_path = _apply_path_replacements(fig.get("result_path", ""), self.path_replace)
                if not fig_path or not os.path.isfile(fig_path):
                    continue
                paragraph = _flatten_text(fig.get("figure_info")) or _flatten_text(fig.get("figure_des"))
                desc_auto = _flatten_text(
                    fig.get("desc_auto")
                    or fig.get("auto_description")
                    or fig.get("description_auto")
                    or ""
                )
                ocr_text = " ; ".join(
                    clean_ocr_items(fig.get("ocr_text") or fig.get("figure_ocr") or fig.get("ocr"), max_items=96)
                ).strip()

                ctx_parts = build_explain_context_strings(
                    paragraph=paragraph,
                    auto_desc=desc_auto,
                    ocr_text=ocr_text,
                )
                if not self.use_ocr_context:
                    ctx_parts["ocr_seg"] = ""
                if not self.use_adesc_context:
                    ctx_parts["adesc_seg"] = ""
                ctx_parts["allowed_ctx"] = _flatten_text([ctx_parts["ocr_seg"], ctx_parts["adesc_seg"]])
                ctx_parts["full_ctx"] = _flatten_text([ctx_parts["allowed_ctx"], ctx_parts["para_seg"]])
                ctx = ""
                ctx_allowed = ""
                if self.context_mode == "paragraph":
                    ctx = ctx_parts["para_seg"]
                elif self.context_mode == "ocr_desc":
                    ctx = ctx_parts["allowed_ctx"]
                    ctx_allowed = ctx
                elif self.context_mode == "paragraph_ocr_desc":
                    ctx_allowed = ctx_parts["allowed_ctx"]
                    ctx = ctx_parts["full_ctx"]
                else:
                    ctx = ""
                    ctx_allowed = ""
                rec = {
                    "image": fig_path,
                    "text": caption,
                    "context": ctx,
                    "context_allowed": ctx_allowed,
                    "context_ocr": ctx_parts["ocr_seg"],
                    "context_adesc": ctx_parts["adesc_seg"],
                    "context_para": ctx_parts["para_seg"],
                    "result_path": res_path,
                    "scale": "explain",
                }
                self.samples.append(rec)
                if max_items is not None and max_items > 0 and len(self.samples) >= max_items:
                    break
            if max_items is not None and max_items > 0 and len(self.samples) >= max_items:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def _load_ann(self, result_dir: str) -> Tuple[List[Dict], Dict[str, int]]:
        cached = self._anno_cache.get(result_dir)
        if cached is not None:
            return cached
        ann_path = Path(result_dir) / "annotations.json"
        if not ann_path.exists():
            self._anno_cache[result_dir] = ([], {})
            return self._anno_cache[result_dir]
        try:
            ann = json.load(open(ann_path, "r", encoding="utf-8"))
        except Exception:
            ann = []
        files = [p.name for p in Path(result_dir).iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        files = [f for f in files if f.lower() != "annotations.json"]
        files.sort()
        name_to_id = {name: idx + 1 for idx, name in enumerate(files)}
        self._anno_cache[result_dir] = (ann, name_to_id)
        return ann, name_to_id

    @staticmethod
    def _point_in_poly(px: float, py: float, poly: List[Tuple[float, float]]) -> bool:
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            cross = (yi > py) != (yj > py)
            if cross:
                denom = (yj - yi) if abs(yj - yi) > 1e-9 else 1e-9
                x_int = (xj - xi) * (py - yi) / denom + xi
                if px < x_int:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _bbox_to_patch_indices(bbox: List[float], width: int, height: int, grid: int) -> List[int]:
        if grid <= 0 or width <= 0 or height <= 0 or len(bbox) < 4:
            return []
        x0, y0, w, h = [float(v) for v in bbox[:4]]
        x1 = x0 + max(0.0, w)
        y1 = y0 + max(0.0, h)
        gx0 = max(0, min(grid - 1, int(math.floor((x0 / float(width)) * grid))))
        gy0 = max(0, min(grid - 1, int(math.floor((y0 / float(height)) * grid))))
        gx1 = max(0, min(grid - 1, int(math.ceil((x1 / float(width)) * grid) - 1)))
        gy1 = max(0, min(grid - 1, int(math.ceil((y1 / float(height)) * grid) - 1)))
        if gx1 < gx0 or gy1 < gy0:
            return []
        return [gy * grid + gx for gy in range(gy0, gy1 + 1) for gx in range(gx0, gx1 + 1)]

    def _seg_to_patch_indices(self, seg: List, bbox: List[float], width: int, height: int, grid: int) -> List[int]:
        if grid <= 0 or width <= 0 or height <= 0:
            return []
        polys: List[List[Tuple[float, float]]] = []
        if isinstance(seg, list):
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = []
                for i in range(0, len(poly) - 1, 2):
                    pts.append((float(poly[i]), float(poly[i + 1])))
                if len(pts) >= 3:
                    polys.append(pts)
        if not polys:
            return self._bbox_to_patch_indices(bbox, width, height, grid)
        picks: List[int] = []
        for gy in range(grid):
            cy = (gy + 0.5) / float(grid) * float(height)
            for gx in range(grid):
                cx = (gx + 0.5) / float(grid) * float(width)
                inside = False
                for poly in polys:
                    if self._point_in_poly(cx, cy, poly):
                        inside = True
                        break
                if inside:
                    picks.append(gy * grid + gx)
        if picks:
            return picks
        return self._bbox_to_patch_indices(bbox, width, height, grid)

    def _mask_regions(self, result_path: str, width: int, height: int) -> List[Dict]:
        if not result_path:
            return []
        key = (result_path, width, height)
        cached_regions = self._region_cache.get(key)
        if cached_regions is not None:
            return cached_regions
        res_dir = str(Path(result_path).parent)
        ann, name_to_id = self._load_ann(res_dir)
        if not ann:
            self._region_cache[key] = []
            return []
        fname = Path(result_path).name
        image_id = name_to_id.get(fname)
        if image_id is None:
            self._region_cache[key] = []
            return []
        area_total = max(1.0, float(width * height))
        items = [a for a in ann if a.get("image_id") == image_id]
        # filter by area ratio
        filt = []
        for a in items:
            area = float(a.get("area") or 0.0)
            ratio = area / area_total
            if ratio < self.mask_min_area_ratio or ratio > self.mask_max_area_ratio:
                continue
            filt.append(a)
        if self.mask_sort == "score":
            filt.sort(key=lambda x: float(x.get("score") or x.get("predicted_iou") or 0.0), reverse=True)
        else:
            filt.sort(key=lambda x: float(x.get("area") or 0.0), reverse=True)
        if self.max_masks > 0:
            filt = filt[: self.max_masks]

        regions: List[Dict] = []
        for a in filt:
            bbox = a.get("bbox") or [0, 0, 0, 0]
            if len(bbox) >= 4:
                x0, y0, w, h = [float(v) for v in bbox[:4]]
            else:
                x0 = y0 = w = h = 0.0
            x1 = float(x0) + float(w)
            y1 = float(y0) + float(h)
            patch_indices = []
            if self.region_grid_size > 0:
                patch_indices = self._seg_to_patch_indices(
                    a.get("segmentation") or [],
                    [x0, y0, w, h],
                    width,
                    height,
                    self.region_grid_size,
                )
            regions.append(
                {
                    "bbox_norm": [
                        max(0.0, min(1.0, x0 / max(1.0, float(width)))),
                        max(0.0, min(1.0, y0 / max(1.0, float(height)))),
                        max(0.0, min(1.0, x1 / max(1.0, float(width)))),
                        max(0.0, min(1.0, y1 / max(1.0, float(height)))),
                    ],
                    "patch_indices": patch_indices,
                }
            )
        self._region_cache[key] = regions
        return regions

    def mask_count(self, result_path: str) -> int:
        if not result_path:
            return 0
        res_dir = str(Path(result_path).parent)
        ann, name_to_id = self._load_ann(res_dir)
        if not ann:
            return 0
        fname = Path(result_path).name
        image_id = name_to_id.get(fname)
        if image_id is None:
            return 0
        return sum(1 for a in ann if a.get("image_id") == image_id)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img = Image.open(rec["image"]).convert("RGB")
        width, height = img.size
        regions = self._mask_regions(rec.get("result_path", ""), width, height)
        ctx_meta = {
            "allowed_text": rec.get("context_allowed", ""),
            "allowed_ocr_text": rec.get("context_ocr", ""),
            "allowed_desc_text": rec.get("context_adesc", ""),
            "forbidden_para_text": rec.get("context_para", ""),
        }
        return img, rec["text"], rec.get("scale", "explain"), rec.get("context", ""), regions, ctx_meta


class SciStructCaptionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_json: str,
        include_short: bool = True,
        include_long: bool = True,
        min_len_short: int = 20,
        min_len_long: int = 40,
        split_mode: str = "legacy",
        word_boundary: int = 30,
        max_items: int | None = None,
    ):
        self.samples: List[Dict] = []
        with open(split_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            for fig in item.get("figures", []) or []:
                fig_path = _flatten_text(fig.get("figure_path"))
                if not fig_path or not os.path.isfile(fig_path):
                    continue
                caption_long = _flatten_text(fig.get("figure_caption"))
                paragraph = clean_paragraph_text(_flatten_text(fig.get("figure_info") or fig.get("figure_des")))
                ocr_text = " ; ".join(
                    clean_ocr_items(fig.get("ocr_text") or fig.get("figure_ocr") or fig.get("ocr"), max_items=96)
                ).strip()
                meta = {"paragraph": paragraph, "ocr": ocr_text}
                if split_mode == "word_boundary":
                    cap_words = _word_count(caption_long)
                    if include_short and cap_words > 0 and cap_words < max(1, int(word_boundary)):
                        self.samples.append(
                            {
                                "image": fig_path,
                                "text": caption_long,
                                "scale": "short",
                                "context": paragraph,
                                "meta": meta,
                            }
                        )
                    if include_long and cap_words >= max(1, int(word_boundary)):
                        self.samples.append(
                            {
                                "image": fig_path,
                                "text": caption_long,
                                "scale": "long",
                                "context": paragraph,
                                "meta": meta,
                            }
                        )
                else:
                    caption_short = _first_sentence(caption_long) if caption_long else ""
                    if include_short and len(caption_short) >= min_len_short:
                        self.samples.append(
                            {
                                "image": fig_path,
                                "text": caption_short,
                                "scale": "short",
                                "context": paragraph,
                                "meta": meta,
                            }
                        )
                    if include_long and len(caption_long) >= min_len_long:
                        self.samples.append(
                            {
                                "image": fig_path,
                                "text": caption_long,
                                "scale": "long",
                                "context": paragraph,
                                "meta": meta,
                            }
                        )
                if max_items is not None and max_items > 0 and len(self.samples) >= max_items:
                    break
            if max_items is not None and max_items > 0 and len(self.samples) >= max_items:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img = Image.open(rec["image"]).convert("RGB")
        return img, rec["text"], rec["scale"], rec.get("context", ""), [], (rec.get("meta") or {})


class SciCapExplainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_json: str,
        min_explain_len: int = 80,
        use_long_fallback: bool = True,
        context_mode: str = "paragraph_ocr_desc",
        max_items: int | None = None,
    ):
        self.samples: List[Dict] = []
        self.context_mode = context_mode
        with open(split_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for art in data:
            for fig in art.get("figures", []) or []:
                fig_path = _flatten_text(fig.get("figure_path") or fig.get("result_path"))
                if not fig_path or not os.path.isfile(fig_path):
                    continue
                meta = fig.get("metadata") or {}
                raw = meta.get("scicap_raw") or {}
                desc = clean_desc_text(_flatten_maybe_json_text(raw.get("figure_description")))
                if len(desc) < min_explain_len and use_long_fallback:
                    desc = clean_desc_text(_flatten_maybe_json_text(raw.get("mlbcap_long") or fig.get("figure_caption")))
                if len(desc) < min_explain_len:
                    continue
                paragraph = clean_paragraph_text(_flatten_maybe_json_text(raw.get("paragraph") or fig.get("figure_info")))
                ocr_text = " ; ".join(clean_ocr_items(raw.get("ocr") or fig.get("figure_ocr"), max_items=96)).strip()
                ctx_parts = build_explain_context_strings(
                    paragraph=paragraph,
                    auto_desc="",
                    ocr_text=ocr_text,
                )
                if self.context_mode == "paragraph":
                    ctx = ctx_parts["para_seg"]
                elif self.context_mode == "ocr_desc":
                    ctx = ctx_parts["allowed_ctx"]
                elif self.context_mode == "paragraph_ocr_desc":
                    ctx = ctx_parts["full_ctx"]
                else:
                    ctx = ""
                rec = {
                    "image": fig_path,
                    "text": desc,
                    "scale": "explain",
                    "context": ctx,
                    "context_allowed": ctx_parts["allowed_ctx"],
                    "context_ocr": ctx_parts["ocr_seg"],
                    "context_para": ctx_parts["para_seg"],
                }
                self.samples.append(rec)
                if max_items is not None and max_items > 0 and len(self.samples) >= max_items:
                    break
            if max_items is not None and max_items > 0 and len(self.samples) >= max_items:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img = Image.open(rec["image"]).convert("RGB")
        ctx_meta = {
            "allowed_text": rec.get("context_allowed", ""),
            "allowed_ocr_text": rec.get("context_ocr", ""),
            "allowed_desc_text": "",
            "forbidden_para_text": rec.get("context_para", ""),
        }
        return img, rec["text"], rec["scale"], rec.get("context", ""), [], ctx_meta


class MixedTaskDataset(torch.utils.data.Dataset):
    """
    Unified 5-tuple dataset for stage-4 multi-task:
    returns (image, text, scale, context, regions)
    """

    def __init__(self, datasets: List[torch.utils.data.Dataset], entries: List[Tuple[int, int]], samples: List[Dict]):
        self.datasets = datasets
        self.entries = entries
        self.samples = samples

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        did, sid = self.entries[idx]
        out = self.datasets[did][sid]
        if isinstance(out, tuple) and len(out) >= 4:
            img, text, scale, context, regions, ctx_meta = _unpack_multimodal_item(out)
            if ctx_meta is not None:
                return img, text, scale, context, regions, ctx_meta
            return img, text, scale, context, regions
        raise RuntimeError(f"unexpected sample format from mixed dataset: type={type(out)}")


def _sample_from_pool(pool: List[Tuple[int, int, Dict]], n: int, rng: random.Random) -> List[Tuple[int, int, Dict]]:
    if n <= 0 or not pool:
        return []
    if n <= len(pool):
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        return [pool[i] for i in idxs[:n]]
    out: List[Tuple[int, int, Dict]] = []
    full = n // len(pool)
    rem = n % len(pool)
    for _ in range(full):
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        out.extend(pool[i] for i in idxs)
    if rem > 0:
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        out.extend(pool[i] for i in idxs[:rem])
    return out


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int | None = None, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / float(self.r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        in_dim = base.in_features
        out_dim = base.out_features
        self.lora_A = nn.Parameter(torch.zeros((r, in_dim), dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, r), dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        x_in = self.dropout(x) if self.dropout is not None else x
        lora_out = F.linear(F.linear(x_in, self.lora_A), self.lora_B)
        return result + lora_out * self.scaling


def _replace_with_lora(module: nn.Module, target_names: List[str], r: int, alpha: int | None, dropout: float) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += _replace_with_lora(child, target_names, r, alpha, dropout)
    return replaced


def apply_lora_to_llm(llm: nn.Module, num_layers: int, r: int, alpha: int | None, dropout: float) -> int:
    layers = _get_llm_layers(llm)
    if layers is None:
        print("[warn] could not locate LLM layers; skipping LoRA")
        return 0
    n = min(num_layers, len(layers))
    target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    total = 0
    for i in range(n):
        total += _replace_with_lora(layers[i], target, r=r, alpha=alpha, dropout=dropout)
    return total


def nll_per_sample(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    # logits: (B, T, V), labels: (B, T)
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("unexpected logits/labels shape")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    )
    loss = loss_flat.view(shift_labels.size(0), shift_labels.size(1))
    mask = shift_labels.ne(ignore_index)
    token_counts = mask.sum(dim=1).clamp(min=1)
    loss_sum = (loss * mask).sum(dim=1)
    return loss_sum / token_counts


def mean_logits_per_sample(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("unexpected logits/labels shape")
    mask = labels.ne(ignore_index)
    pooled = []
    for b in range(logits.size(0)):
        if mask[b].any():
            pooled.append(logits[b][mask[b]].mean(dim=0))
        else:
            pooled.append(logits[b].mean(dim=0))
    return torch.stack(pooled, dim=0)


def _estimate_image_tokens(vision_pool: int, image_size: int = 384, patch_size: int = 16) -> int:
    side = image_size // patch_size
    if vision_pool > 1 and side % vision_pool == 0:
        return (side // vision_pool) ** 2
    return side * side


class LengthBucketBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, lengths: List[int], bucket_bins: List[int], batch_size: int, drop_last: bool = False, seed: int = 42):
        self.lengths = lengths
        self.bucket_bins = sorted(bucket_bins)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed

        self.buckets: Dict[int, List[int]] = {}
        for idx, length in enumerate(lengths):
            bucket = self._assign_bucket(length)
            self.buckets.setdefault(bucket, []).append(idx)

    def _assign_bucket(self, length: int) -> int:
        for b in self.bucket_bins:
            if length <= b:
                return b
        return self.bucket_bins[-1]

    def __iter__(self):
        rng = random.Random(self.seed + random.randint(0, 10_000))
        bucket_batches: List[List[int]] = []
        for bucket, indices in self.buckets.items():
            idxs = indices[:]
            rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                bucket_batches.append(batch)
        rng.shuffle(bucket_batches)
        for batch in bucket_batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.buckets.values():
            n = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size:
                n += 1
            total += n
        return total


def _keyword_ids(tokenizer: AutoTokenizer, words: List[str]) -> List[int]:
    ids = []
    for w in words:
        toks = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            ids.append(toks[0])
    return list(dict.fromkeys(ids))


def keyword_presence(logits: torch.Tensor, keyword_ids: List[int]) -> torch.Tensor:
    # logits: (B, T, V) -> log_probs
    if not keyword_ids:
        return torch.zeros(logits.size(0), device=logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    kw = log_probs[..., keyword_ids]
    # max over keywords, then over positions
    return kw.max(dim=-1).values.max(dim=-1).values


def build_generation_prompt(
    tokenizer: AutoTokenizer,
    image_token_index: int,
    scale: str,
    context: str | None,
    max_length: int,
    max_ctx_tokens: int,
    max_ctx_tokens_explain: int = 0,
    scicap_prompt_style: str = "none",
    enable_task_style_tokens: bool = True,
) -> List[int]:
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id or 0
    task_token = TASK_TOKENS.get(scale, TASK_TOKENS["long"])
    task_ids = tokenizer(task_token, add_special_tokens=False)["input_ids"]
    if not task_ids:
        task_ids = [tokenizer.unk_token_id]
    task_style_ids = _task_style_ids(tokenizer, scale, enable_task_style_tokens)
    task_prompt_ids = _task_prompt_ids(tokenizer, scale, scicap_prompt_style)
    prompt_ids = [bos_id] + task_ids + task_style_ids + task_prompt_ids + [image_token_index]
    if max_length > 0:
        prompt_ids = prompt_ids[:max_length]
    ctx_ids: List[int] = []
    ctx_text = (context or "").strip()
    ctx_limit = max_ctx_tokens
    if scale == "explain" and max_ctx_tokens_explain and max_ctx_tokens_explain > 0:
        ctx_limit = max_ctx_tokens_explain
    if ctx_text:
        ctx_ids = tokenizer(
            "\n" + ctx_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length if max_length > 0 else None,
        )["input_ids"]
        if ctx_limit and ctx_limit > 0:
            ctx_ids = ctx_ids[:ctx_limit]
        if max_length > 0:
            max_ctx = max(0, max_length - len(prompt_ids))
            ctx_ids = ctx_ids[:max_ctx]
    return prompt_ids + ctx_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="checkpoints/phi-sig")
    ap.add_argument("--model_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--visual_ckpt", type=str, default="checkpoints/visual_student_scistruct_scicap_full_v2/ckpt_last.pt")
    ap.add_argument("--init_ckpt", type=str, default="")
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="scicap", choices=["scicap", "scistruct_explain", "stage4_multitask"])
    ap.add_argument("--explain_train_json", type=str, default="")
    ap.add_argument("--explain_val_json", type=str, default="")
    ap.add_argument("--public_caption_json", action="append", default=[])
    ap.add_argument("--stage4_explain_ratio", type=float, default=0.3)
    ap.add_argument("--stage4_epoch_size", type=int, default=0)
    ap.add_argument("--stage4_val_epoch_size", type=int, default=0)
    ap.add_argument("--stage4_include_scistruct_caption", action="store_true", help="add SciStruct captions into caption pool")
    ap.add_argument("--stage4_scistruct_caption_json", type=str, default="", help="optional SciStruct json used for caption pool")
    ap.add_argument("--stage4_scistruct_caption_short", action="store_true", help="include synthesized short captions from SciStruct")
    ap.add_argument("--stage4_scistruct_caption_long", action="store_true", help="include long captions from SciStruct")
    ap.add_argument("--stage4_scistruct_caption_min_len_short", type=int, default=20)
    ap.add_argument("--stage4_scistruct_caption_min_len_long", type=int, default=40)
    ap.add_argument(
        "--stage4_scistruct_caption_split_mode",
        type=str,
        default="word_boundary",
        choices=["legacy", "word_boundary"],
        help="SciStruct caption split policy: legacy(first sentence + min lengths) or word boundary on figure_caption",
    )
    ap.add_argument(
        "--stage4_scistruct_caption_word_boundary",
        type=int,
        default=30,
        help="when split_mode=word_boundary: < boundary -> short, >= boundary -> long",
    )
    ap.add_argument("--stage4_include_scicap_explain", action="store_true", help="add SciCap pseudo-explain samples into explain pool")
    ap.add_argument("--stage4_scicap_explain_json", type=str, default="", help="optional SciCap json used for explain pool")
    ap.add_argument("--stage4_scicap_explain_ratio", type=float, default=0.0, help="fraction of explain samples drawn from SciCap pseudo-explain pool")
    ap.add_argument("--stage4_scicap_explain_min_len", type=int, default=80)
    ap.add_argument("--stage4_scicap_explain_use_long_fallback", action="store_true")
    ap.add_argument("--val_json", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="checkpoints/tinyllava_phi_siglip_image_only")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--vision_pool", type=int, default=3)
    ap.add_argument("--vision_pool_mode", type=str, default="avg", choices=["avg", "max"])
    ap.add_argument("--region_token_scale", type=float, default=1.0)
    ap.add_argument("--enable_explain_region_adapter", action="store_true", help="add explain-only adapter on region tokens")
    ap.add_argument("--explain_region_adapter_on_unknown", action="store_true", help="when scale metadata is missing, still apply explain region adapter")
    ap.add_argument("--enable_task_specific_region_adapter", action="store_true", help="add task-specific (short/long/desc/explain) region adapters")
    ap.add_argument("--task_specific_region_adapter_on_unknown", action="store_true", help="when task label is missing, still apply a fallback task-specific region adapter")
    ap.add_argument(
        "--context_mode",
        type=str,
        default="none",
        choices=["none", "paragraph", "para_mention", "para_mention_ocr", "ocr_desc", "paragraph_ocr_desc"],
    )
    ap.add_argument("--disable_explain_ocr", action="store_true", help="for explain dataset, drop OCR segment from context")
    ap.add_argument("--disable_explain_adesc", action="store_true", help="for explain dataset, drop AutoDesc segment from context")
    ap.add_argument("--context_dropout", type=float, default=0.0)
    ap.add_argument("--paragraph_token_dropout", type=float, default=0.0)
    ap.add_argument("--use_context_placeholders", action="store_true", help="force unified context protocol with explicit <NO_*> placeholders")
    ap.add_argument("--max_ctx_tokens", type=int, default=0)
    ap.add_argument("--max_ctx_tokens_explain", type=int, default=0, help="if >0, override max_ctx_tokens only for explain samples")
    ap.add_argument("--explain_ctx_min_adesc_tokens", type=int, default=0, help="reserve at least this many ADESC tokens (if present) within explain context budget")
    ap.add_argument("--explain_ctx_max_ocr_tokens", type=int, default=0, help="cap OCR tokens within explain context budget (0 disables cap)")
    ap.add_argument("--context_mix_mode", type=str, default="none", choices=["none", "stage3a", "stage3b"])
    ap.add_argument("--context_region_only_prob", type=float, default=0.0)
    ap.add_argument("--context_region_ctx_prob", type=float, default=0.0)
    ap.add_argument("--context_paragraph_only_prob", type=float, default=0.0)
    ap.add_argument("--context_shuffle_prob", type=float, default=0.0)
    ap.add_argument("--paragraph_only_weight", type=float, default=0.0)
    ap.add_argument("--anchor_region_only_scale", type=float, default=1.0)
    ap.add_argument(
        "--scicap_prompt_style",
        type=str,
        default="none",
        choices=["none", "scicap_metric", "scicap_metric_desc_strict"],
    )
    ap.add_argument("--scicap_fixed_task_loss", action="store_true", help="optimize short+long+desc as equal-weight task losses")
    ap.add_argument(
        "--task_balance_mode",
        type=str,
        default="none",
        choices=["none", "ema_inverse"],
        help="multi-task conflict mitigation: reweight task losses with inverse EMA",
    )
    ap.add_argument("--task_balance_alpha", type=float, default=0.5, help="inverse-EMA exponent for task reweighting")
    ap.add_argument("--task_balance_ema", type=float, default=0.95, help="EMA momentum for task reweighting")
    ap.add_argument("--task_balance_min_weight", type=float, default=0.5, help="lower clamp for dynamic task weight")
    ap.add_argument("--task_balance_max_weight", type=float, default=2.0, help="upper clamp for dynamic task weight")
    ap.add_argument(
        "--scicap_task_context_routing",
        type=str,
        default="none",
        choices=["none", "caption_para_desc_ocr"],
        help="SciCap-only context routing: caption tasks use paragraph context, description uses OCR context",
    )
    ap.add_argument(
        "--scicap_struct_jsonl",
        type=str,
        default="",
        help="optional structure-cache jsonl for SciCap; when set, struct_nodes are exposed in sample meta",
    )
    ap.add_argument("--scicap_struct_max_nodes", type=int, default=64)
    ap.add_argument("--scicap_loss_w_short", type=float, default=1.0, help="fixed-task loss weight for CAPTION_SHORT")
    ap.add_argument("--scicap_loss_w_long", type=float, default=1.0, help="fixed-task loss weight for CAPTION_LONG")
    ap.add_argument("--scicap_loss_w_desc", type=float, default=1.0, help="fixed-task loss weight for DESCRIPTION")
    ap.add_argument("--enable_task_style_tokens", action="store_true", help="prepend task-style token for short/long/desc")
    ap.add_argument("--caption_formula_penalty_weight", type=float, default=0.0, help="penalize formula-like token mass on caption tasks")
    ap.add_argument("--caption_formula_max_token_ids", type=int, default=256)
    ap.add_argument("--caption_block_formula_in_decode", action="store_true", help="block formula-like tokens during caption decoding")
    ap.add_argument("--caption_context_cov_weight", type=float, default=0.0, help="coverage penalty over paragraph evidence for caption tasks")
    ap.add_argument("--caption_context_cov_min", type=float, default=0.05)
    ap.add_argument("--caption_context_cov_max_ids", type=int, default=48)
    ap.add_argument("--desc_ocr_cov_weight", type=float, default=0.0, help="coverage penalty over OCR evidence for description task")
    ap.add_argument("--desc_ocr_cov_min", type=float, default=0.05)
    ap.add_argument("--desc_ocr_cov_max_ids", type=int, default=64)
    ap.add_argument("--desc_entity_anchor_weight", type=float, default=0.0, help="lightweight per-sample anchor forcing description to hit >=1 real OCR entity")
    ap.add_argument("--desc_entity_anchor_min", type=float, default=0.02)
    ap.add_argument("--desc_entity_anchor_max_items", type=int, default=12)
    ap.add_argument("--desc_entity_anchor_max_ids_per_item", type=int, default=8)
    ap.add_argument("--desc_entity_anchor_alias_max_per_item", type=int, default=8)
    ap.add_argument("--desc_entity_anchor_topk", type=int, default=1)
    ap.add_argument(
        "--desc_entity_nonalias_penalty_weight",
        type=float,
        default=0.0,
        help="light penalty on entity-like token mass that is outside per-sample alias set",
    )
    ap.add_argument(
        "--desc_entity_nonalias_max",
        type=float,
        default=0.08,
        help="target upper bound for non-alias entity-like token mass",
    )
    ap.add_argument(
        "--desc_entity_nonalias_max_token_ids",
        type=int,
        default=4096,
        help="max number of global entity-like token ids used by non-alias penalty",
    )
    ap.add_argument(
        "--desc_entity_nonalias_only_after_cue",
        action="store_true",
        help="apply non-alias penalty only when the previous token is a cue word (node/module/stage/...)",
    )
    ap.add_argument("--desc_struct_cov_weight", type=float, default=0.0, help="weak structural-slot coverage penalty for description task")
    ap.add_argument("--desc_struct_cov_min", type=float, default=0.06)
    ap.add_argument("--desc_struct_cov_max_slots", type=int, default=8)
    ap.add_argument("--desc_struct_cov_max_ids_per_slot", type=int, default=16)
    ap.add_argument("--desc_struct_cov_target_ratio", type=float, default=0.4, help="fraction of GT-derived slots required to be covered")
    ap.add_argument("--desc_prompt_leak_penalty_weight", type=float, default=0.0, help="penalize prompt-leak token mass (e.g., OCR/context markers) on description task")
    ap.add_argument("--desc_prompt_leak_max_token_ids", type=int, default=128)
    ap.add_argument("--desc_block_prompt_leak_in_decode", action="store_true", help="block prompt-leak tokens during description decoding")
    ap.add_argument("--disable_explain_gate_non_explain", action="store_true", help="disable explain-specific gate/bias when batch has no explain samples")
    ap.add_argument("--min_caption_len", type=int, default=40)
    ap.add_argument("--max_masks", type=int, default=16)
    ap.add_argument("--mask_min_area_ratio", type=float, default=0.002)
    ap.add_argument("--mask_max_area_ratio", type=float, default=0.85)
    ap.add_argument("--mask_sort", type=str, default="area", choices=["area", "score"])
    ap.add_argument("--path_replace", action="append", default=[], help="from=to")
    ap.add_argument("--bucket_by_length", action="store_true")
    ap.add_argument("--bucket_bins", type=str, default="")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--image_patch_size", type=int, default=16)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--warmup_task", type=str, default="short", choices=["short", "long", "desc"])
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=0)
    ap.add_argument("--val_batch_size", type=int, default=0)
    ap.add_argument("--val_num_batches", type=int, default=0, help="0 means full validation loader")
    ap.add_argument("--val_max_items", type=int, default=0, help="0 means all validation samples")
    ap.add_argument("--sample_every", type=int, default=0)
    ap.add_argument("--sample_num", type=int, default=4)
    ap.add_argument("--sample_mode", type=str, default="tasks", choices=["tasks", "stage3_modes", "explain_diag"])
    ap.add_argument("--sample_tasks", type=str, default="short,long,desc,explain")
    ap.add_argument("--sample_max_new_tokens", type=int, default=96)
    ap.add_argument("--sample_min_new_tokens", type=int, default=10)
    ap.add_argument("--sample_max_new_tokens_short", type=int, default=0)
    ap.add_argument("--sample_max_new_tokens_long", type=int, default=0)
    ap.add_argument("--sample_max_new_tokens_desc", type=int, default=0)
    ap.add_argument("--eval_num_beams", type=int, default=1)
    ap.add_argument("--eval_length_penalty", type=float, default=1.0)
    ap.add_argument("--eval_repetition_penalty", type=float, default=1.0)
    ap.add_argument("--eval_no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--sample_out_dir", type=str, default="")
    ap.add_argument("--eval_only", action="store_true", help="run validation once and exit without training")
    ap.add_argument("--eval_step", type=int, default=0, help="step id used in eval_only logs")
    ap.add_argument("--log_attn", action="store_true", help="enable attention diagnostics; force exportable attention path")
    ap.add_argument("--explain_attn_last_k_layers", type=int, default=4, help="aggregate explain attention over last k layers (<=0 means all)")
    ap.add_argument("--debug_explain_attn_consistency", action="store_true", help="assert val/sample explain attention metrics are consistent")
    ap.add_argument("--explain_metrics_minimal_only", action="store_true", help="write only key explain metrics to val_log")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--connector_lr", type=float, default=1e-4)
    ap.add_argument("--lora_lr", type=float, default=5e-5)
    ap.add_argument("--contrastive_margin", type=float, default=0.3)
    ap.add_argument("--contrastive_weight", type=float, default=0.5)
    ap.add_argument("--contrastive_type", type=str, default="shuffle", choices=["shuffle", "zero", "mask_drop"])
    ap.add_argument("--contrastive_mode", type=str, default="continuous", choices=["continuous", "hinge", "cosine"])
    ap.add_argument("--explain_counterfactual_weight", type=float, default=0.0, help="counterfactual regularization for explain: wrong image should increase explain NLL")
    ap.add_argument("--explain_counterfactual_margin", type=float, default=0.1, help="target margin for explain counterfactual gap")
    ap.add_argument("--explain_counterfactual_mode", type=str, default="hinge", choices=["hinge", "continuous"])
    ap.add_argument("--explain_counterfactual_weight_schedule_start_step", type=int, default=-1, help="enable linear schedule when >=0 and end>start")
    ap.add_argument("--explain_counterfactual_weight_schedule_end_step", type=int, default=-1, help="linear schedule end step")
    ap.add_argument("--explain_counterfactual_weight_start", type=float, default=-1.0, help="schedule start value; <0 uses --explain_counterfactual_weight")
    ap.add_argument("--explain_counterfactual_weight_end", type=float, default=-1.0, help="schedule end value; <0 uses --explain_counterfactual_weight")
    ap.add_argument(
        "--explain_counterfactual_pairing",
        type=str,
        default="rotate",
        choices=["rotate", "random", "hard_jaccard"],
        help="how to choose wrong-image pairs for explain counterfactual",
    )
    ap.add_argument("--explain_region_required_loss", action="store_true", help="for EXPLAIN only: enforce region-drop loss to be worse than base")
    ap.add_argument("--explain_region_required_lambda", type=float, default=0.5, help="lambda for EXPLAIN region-required contrastive loss")
    ap.add_argument("--explain_region_required_margin", type=float, default=0.0, help="margin m for EXPLAIN region-required hinge: enforce (loss_drop - loss_base) > m")
    ap.add_argument("--batch_contrastive", action="store_true")
    ap.add_argument("--consistency_weight", type=float, default=0.0)
    ap.add_argument("--consistency_margin", type=float, default=0.1)
    ap.add_argument("--consistency_mode", type=str, default="ctx_shuffle", choices=["ctx_shuffle", "anchor"])
    ap.add_argument("--anchor_type", action="store_true", help="anchor type keywords in output")
    ap.add_argument("--anchor_struct", action="store_true", help="anchor structure keywords in output")
    ap.add_argument("--anchor_rel", action="store_true", help="anchor relation keywords in output")
    ap.add_argument("--image_dropout_prob", type=float, default=0.0)
    ap.add_argument("--image_dropout_mode", type=str, default="zero", choices=["zero", "noise"])
    ap.add_argument("--image_dropout_sigma", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_target_short", type=int, default=64)
    ap.add_argument("--max_target_long", type=int, default=256)
    ap.add_argument("--max_target_desc", type=int, default=384)
    ap.add_argument("--min_len_short", type=int, default=20)
    ap.add_argument("--min_len_long", type=int, default=40)
    ap.add_argument("--min_len_desc", type=int, default=40)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--freeze_llm", action="store_true")
    ap.add_argument("--unfreeze_embeddings", action="store_true")
    ap.add_argument("--unfreeze_llm_layers", type=int, default=0)
    ap.add_argument("--unfreeze_llm_after_step", type=int, default=0, help="delay llm layer unfreeze until this global step; <=0 means immediate")
    ap.add_argument("--task_embed_only", action="store_true")
    ap.add_argument("--lora_layers", type=int, default=12)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--train_connector", action="store_true")
    ap.add_argument(
        "--strict_connector_load",
        action="store_true",
        help="fail if connector weights are dropped/missing/unexpected at init checkpoint load",
    )
    ap.add_argument(
        "--strict_vocab_size_match",
        action="store_true",
        help="fail if tokenizer/embedding vocab sizes mismatch between runtime and init checkpoint",
    )
    ap.add_argument("--fixed_task", type=str, default="", choices=["", "short", "long", "desc", "explain"])
    ap.add_argument("--region_attn_bias_layers", type=int, default=0)
    ap.add_argument("--region_attn_bias_beta", type=float, default=0.0)
    ap.add_argument("--region_attn_bias_beta_schedule_start_step", type=int, default=-1, help="enable linear schedule when >=0 and end>start")
    ap.add_argument("--region_attn_bias_beta_schedule_end_step", type=int, default=-1, help="linear schedule end step")
    ap.add_argument("--region_attn_bias_beta_start", type=float, default=-1.0, help="schedule start value; <0 uses --region_attn_bias_beta")
    ap.add_argument("--region_attn_bias_beta_end", type=float, default=-1.0, help="schedule end value; <0 uses --region_attn_bias_beta")
    ap.add_argument("--paragraph_attn_neg_bias_gamma", type=float, default=0.0)
    ap.add_argument("--region_attn_bias_task", type=str, default="explain", choices=["short", "long", "desc", "explain"])
    ap.add_argument("--explain_hard_paragraph_kv_gate", action="store_true", help="hard-mask explain paragraph tokens from K/V attention")
    ap.add_argument("--explain_hard_paragraph_kv_bias", type=float, default=-1e4, help="additive bias for explain paragraph K/V masking")
    ap.add_argument("--explain_allow_context_kv", action="store_true", help="under EXPLAIN hard gate, keep context tokens visible as K/V")
    ap.add_argument("--freeze_region_mechanism", action="store_true")
    ap.add_argument("--unfreeze_llm_from", type=int, default=-1, help="inclusive layer index")
    ap.add_argument("--unfreeze_llm_to", type=int, default=-1, help="exclusive layer index; <=0 means to the end")
    ap.add_argument("--shuffle_images", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def _is_schedule_active(start_step: int, end_step: int) -> bool:
        return int(start_step) >= 0 and int(end_step) > int(start_step)

    def _resolve_sched_endpoint(v: float, fallback: float) -> float:
        return float(fallback) if float(v) < 0 else float(v)

    def _linear_schedule_value(
        cur_step: int,
        start_step: int,
        end_step: int,
        start_value: float,
        end_value: float,
    ) -> float:
        if not _is_schedule_active(start_step, end_step):
            return float(start_value)
        if cur_step <= start_step:
            return float(start_value)
        if cur_step >= end_step:
            return float(end_value)
        denom = max(1, int(end_step) - int(start_step))
        ratio = float(cur_step - int(start_step)) / float(denom)
        return float(start_value) + (float(end_value) - float(start_value)) * ratio

    beta_sched_active = _is_schedule_active(
        args.region_attn_bias_beta_schedule_start_step,
        args.region_attn_bias_beta_schedule_end_step,
    )
    beta_sched_start = _resolve_sched_endpoint(args.region_attn_bias_beta_start, args.region_attn_bias_beta)
    beta_sched_end = _resolve_sched_endpoint(args.region_attn_bias_beta_end, args.region_attn_bias_beta)
    cf_sched_active = _is_schedule_active(
        args.explain_counterfactual_weight_schedule_start_step,
        args.explain_counterfactual_weight_schedule_end_step,
    )
    cf_sched_start = _resolve_sched_endpoint(args.explain_counterfactual_weight_start, args.explain_counterfactual_weight)
    cf_sched_end = _resolve_sched_endpoint(args.explain_counterfactual_weight_end, args.explain_counterfactual_weight)

    region_attn_bias_beta_runtime = _linear_schedule_value(
        0,
        args.region_attn_bias_beta_schedule_start_step,
        args.region_attn_bias_beta_schedule_end_step,
        beta_sched_start,
        beta_sched_end,
    )
    explain_cf_weight_runtime = _linear_schedule_value(
        0,
        args.explain_counterfactual_weight_schedule_start_step,
        args.explain_counterfactual_weight_schedule_end_step,
        cf_sched_start,
        cf_sched_end,
    )
    _last_sched_log_step = -1

    def _update_runtime_schedule(cur_step: int, *, force_log: bool = False) -> Tuple[float, float]:
        nonlocal region_attn_bias_beta_runtime, explain_cf_weight_runtime, _last_sched_log_step
        region_attn_bias_beta_runtime = _linear_schedule_value(
            cur_step,
            args.region_attn_bias_beta_schedule_start_step,
            args.region_attn_bias_beta_schedule_end_step,
            beta_sched_start,
            beta_sched_end,
        )
        explain_cf_weight_runtime = _linear_schedule_value(
            cur_step,
            args.explain_counterfactual_weight_schedule_start_step,
            args.explain_counterfactual_weight_schedule_end_step,
            cf_sched_start,
            cf_sched_end,
        )
        if force_log and cur_step != _last_sched_log_step:
            print(
                f"[sched] step={int(cur_step)} "
                f"region_beta={float(region_attn_bias_beta_runtime):.4f} "
                f"cf_weight={float(explain_cf_weight_runtime):.4f}"
            )
            _last_sched_log_step = int(cur_step)
        return float(region_attn_bias_beta_runtime), float(explain_cf_weight_runtime)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer & model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_dtype = dtype_map.get(args.model_dtype, torch.float16)
    model_load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": load_dtype if torch.cuda.is_available() else torch.float32,
        # TinyLlavaForConditionalGeneration in this repo does not implement SDPA path.
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_load_kwargs)
    if args.log_attn:
        print("[info] log_attn enabled: forcing eager attention path for consistent exported attentions")
        _force_eager_attn(model, "model")
        _force_eager_attn(getattr(model, "language_model", None), "language_model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        model_max_length=model.config.tokenizer_model_max_length,
        padding_side=model.config.tokenizer_padding_side,
    )

    extra_tokens = []
    for t in TASK_TOKENS.values():
        if t not in tokenizer.get_vocab():
            extra_tokens.append(t)
    if args.enable_task_style_tokens:
        for t in TASK_STYLE_TOKENS.values():
            if t not in tokenizer.get_vocab():
                extra_tokens.append(t)
    if extra_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": extra_tokens})
        model.resize_token_embeddings(len(tokenizer))
    task_token_ids: List[int] = []
    for t in TASK_TOKENS.values():
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        task_token_ids.extend(ids)
    if args.enable_task_style_tokens:
        for t in TASK_STYLE_TOKENS.values():
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            task_token_ids.extend(ids)
    task_token_ids = list(dict.fromkeys(task_token_ids))
    caption_formula_token_ids = _infer_formula_token_ids(
        tokenizer,
        max_ids=max(0, int(args.caption_formula_max_token_ids)),
    ) if (args.caption_formula_penalty_weight > 0 or args.caption_block_formula_in_decode) else []
    if caption_formula_token_ids:
        print(f"[info] caption formula token controls enabled: n_ids={len(caption_formula_token_ids)}")
    desc_prompt_leak_token_ids = _infer_desc_prompt_leak_token_ids(
        tokenizer,
        max_ids=max(0, int(args.desc_prompt_leak_max_token_ids)),
    ) if (args.desc_prompt_leak_penalty_weight > 0 or args.desc_block_prompt_leak_in_decode) else []
    if desc_prompt_leak_token_ids:
        print(f"[info] description prompt-leak token controls enabled: n_ids={len(desc_prompt_leak_token_ids)}")
    desc_entity_like_token_ids = _infer_entity_like_token_ids(
        tokenizer,
        max_ids=max(0, int(args.desc_entity_nonalias_max_token_ids)),
    ) if (args.desc_entity_nonalias_penalty_weight > 0) else []
    desc_entity_like_shape_map: Dict[int, str] = {}
    desc_entity_like_ids_by_shape: Dict[str, List[int]] = {}
    if desc_entity_like_token_ids:
        for tid in desc_entity_like_token_ids:
            try:
                piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
            except Exception:
                continue
            sk = _entity_shape_key_from_text(piece, allow_digit_only=True)
            if not sk:
                continue
            desc_entity_like_shape_map[int(tid)] = sk
            desc_entity_like_ids_by_shape.setdefault(sk, []).append(int(tid))
    if desc_entity_like_token_ids:
        print(f"[info] description non-alias entity penalty enabled: n_entity_like_ids={len(desc_entity_like_token_ids)}")
    desc_entity_cue_token_ids = _infer_piece_token_ids_for_words(
        tokenizer,
        _DESC_ENTITY_CUE_WORDS,
        max_ids=512,
    ) if (args.desc_entity_nonalias_penalty_weight > 0 and args.desc_entity_nonalias_only_after_cue) else []
    if desc_entity_cue_token_ids:
        print(f"[info] description non-alias cue filter enabled: n_cue_ids={len(desc_entity_cue_token_ids)}")
    type_keyword_ids = _keyword_ids(tokenizer, TYPE_KEYWORDS)
    struct_keyword_ids = _keyword_ids(tokenizer, STRUCT_KEYWORDS)
    rel_keyword_ids = _keyword_ids(tokenizer, REL_KEYWORDS)

    # Build visual student
    student = CLIPVisionTower(output_attentions=False)
    ckpt = torch.load(args.visual_ckpt, map_location="cpu")
    student.load_state_dict(ckpt.get("model", {}), strict=False)

    if args.freeze_vision:
        set_requires_grad(student, False)

    region_token_slots = args.max_masks if args.dataset in ("scistruct_explain", "stage4_multitask") else 0
    vision_wrapper = StudentVisionTower(
        student,
        pool_size=args.vision_pool,
        pool_mode=args.vision_pool_mode,
        max_region_tokens=region_token_slots,
        region_token_scale=args.region_token_scale,
        enable_explain_region_adapter=args.enable_explain_region_adapter,
        explain_region_adapter_on_unknown=args.explain_region_adapter_on_unknown,
        enable_task_specific_region_adapter=args.enable_task_specific_region_adapter,
        task_specific_region_adapter_on_unknown=args.task_specific_region_adapter_on_unknown,
    )
    if args.enable_task_specific_region_adapter:
        print(
            f"[info] task-specific region adapters enabled: "
            f"on_unknown={bool(args.task_specific_region_adapter_on_unknown)}"
        )

    # Replace vision tower & connector to match 768-dim student tokens
    model.vision_tower = vision_wrapper
    model.config.vision_hidden_size = 768
    if hasattr(model.config, "vision_config"):
        try:
            model.config.vision_config.hidden_size = 768
        except Exception:
            pass
    # Ensure connector is Linear -> GELU -> Linear
    model.config.connector_type = "mlp2x_gelu"
    model.connector = model.connector.__class__(model.config)

    if args.freeze_region_mechanism:
        set_requires_grad(vision_wrapper, False)
    if args.enable_task_specific_region_adapter and vision_wrapper.region_proj_task is not None:
        # Allow lightweight task-routed region adapters even when region mechanism is frozen.
        set_requires_grad(vision_wrapper.region_proj_task, True)
    if args.enable_explain_region_adapter and vision_wrapper.region_proj_explain is not None:
        # Route-B: keep only explain-specific region adapter trainable.
        set_requires_grad(vision_wrapper.region_proj_explain, True)

    if args.train_connector:
        set_requires_grad(model.connector, True)
    else:
        set_requires_grad(model.connector, False)

    def _apply_llm_unfreeze_targets() -> int:
        changed = 0
        layers = None
        if args.unfreeze_llm_layers > 0:
            layers = _get_llm_layers(model.language_model)
            if layers is None:
                print("[warn] could not locate LLM layers; skipping unfreeze_llm_layers")
            else:
                n = min(args.unfreeze_llm_layers, len(layers))
                for i in range(n):
                    for p in layers[i].parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            changed += 1
        if args.unfreeze_llm_from >= 0:
            if layers is None:
                layers = _get_llm_layers(model.language_model)
            if layers is None:
                print("[warn] could not locate LLM layers; skipping unfreeze_llm_from/to")
            else:
                st = max(0, int(args.unfreeze_llm_from))
                ed = len(layers) if int(args.unfreeze_llm_to) <= 0 else min(len(layers), int(args.unfreeze_llm_to))
                if st >= ed:
                    print(f"[warn] invalid unfreeze range [{st}, {ed}); skipping")
                else:
                    for i in range(st, ed):
                        for p in layers[i].parameters():
                            if not p.requires_grad:
                                p.requires_grad = True
                                changed += 1
        return int(changed)

    delayed_llm_unfreeze = bool(
        int(args.unfreeze_llm_after_step) > 0
        and (int(args.unfreeze_llm_layers) > 0 or int(args.unfreeze_llm_from) >= 0)
    )
    if args.freeze_llm or args.unfreeze_embeddings or args.unfreeze_llm_layers > 0 or args.task_embed_only or args.lora_layers > 0:
        set_requires_grad(model.language_model, False)
        if args.lora_layers > 0 and args.lora_r > 0:
            replaced = apply_lora_to_llm(
                model.language_model,
                num_layers=args.lora_layers,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
            print(f"[info] applied LoRA to {replaced} linear modules in first {args.lora_layers} layers")
        if args.unfreeze_embeddings or args.task_embed_only:
            set_requires_grad(model.language_model.get_input_embeddings(), True)
            out_emb = model.language_model.get_output_embeddings()
            if out_emb is not None:
                set_requires_grad(out_emb, True)
        if delayed_llm_unfreeze:
            print(f"[info] delayed llm unfreeze: step={int(args.unfreeze_llm_after_step)}")
        else:
            _ = _apply_llm_unfreeze_targets()

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model_state = model.state_dict()
        if args.strict_vocab_size_match:
            ckpt_tok_len = ckpt.get("tokenizer_len", None) if isinstance(ckpt, dict) else None
            if ckpt_tok_len is not None:
                try:
                    ckpt_tok_len_i = int(ckpt_tok_len)
                except Exception:
                    ckpt_tok_len_i = -1
                cur_tok_len = int(len(tokenizer))
                if ckpt_tok_len_i > 0 and ckpt_tok_len_i != cur_tok_len:
                    raise RuntimeError(
                        f"[strict_vocab] tokenizer length mismatch: ckpt={ckpt_tok_len_i} current={cur_tok_len} "
                        f"init_ckpt={args.init_ckpt}"
                    )
            for suffix in ("embed_tokens.weight", "lm_head.weight"):
                ck_key, ck_rows = _find_lm_matrix_rows(state, suffix)
                cur_key, cur_rows = _find_lm_matrix_rows(model_state, suffix)
                if ck_key and cur_key and ck_rows is not None and cur_rows is not None and ck_rows != cur_rows:
                    raise RuntimeError(
                        f"[strict_vocab] {suffix} row mismatch: ckpt_key={ck_key} ckpt_rows={ck_rows} "
                        f"current_key={cur_key} current_rows={cur_rows} init_ckpt={args.init_ckpt}"
                    )

        drop: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        for k, v in list(state.items()):
            if k in model_state and model_state[k].shape != v.shape:
                drop.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                state.pop(k, None)
        if drop:
            drop_keys = [d[0] for d in drop]
            connector_drop = [k for k in drop_keys if _is_connector_state_key(k)]
            preview = ", ".join(f"{k}:{s1}->{s2}" for k, s1, s2 in drop[:6])
            if args.strict_connector_load and connector_drop:
                raise RuntimeError(
                    f"[strict_connector] dropped connector keys due shape mismatch: n={len(connector_drop)} "
                    f"examples={connector_drop[:6]} init_ckpt={args.init_ckpt}"
                )
            print(
                f"[warn] dropped {len(drop)} keys with shape mismatch"
                f"{' (connector involved)' if connector_drop else ''}: {preview}"
                f"{' ...' if len(drop) > 6 else ''}"
            )
        load_ret = model.load_state_dict(state, strict=False)
        missing_keys = list(getattr(load_ret, "missing_keys", []))
        unexpected_keys = list(getattr(load_ret, "unexpected_keys", []))
        connector_missing = [k for k in missing_keys if _is_connector_state_key(k)]
        connector_unexpected = [k for k in unexpected_keys if _is_connector_state_key(k)]
        if args.strict_connector_load and (connector_missing or connector_unexpected):
            raise RuntimeError(
                f"[strict_connector] load_state mismatch:"
                f" missing={connector_missing[:8]} unexpected={connector_unexpected[:8]}"
                f" init_ckpt={args.init_ckpt}"
            )
        if connector_missing or connector_unexpected:
            print(
                f"[warn] connector load mismatch:"
                f" missing={connector_missing[:6]} unexpected={connector_unexpected[:6]}"
            )
        if args.strict_vocab_size_match:
            vocab_missing = [
                k for k in missing_keys
                if ("embed_tokens.weight" in k) or ("lm_head.weight" in k)
            ]
            if vocab_missing:
                raise RuntimeError(
                    f"[strict_vocab] embedding/lm_head keys missing after load: {vocab_missing[:8]} "
                    f"init_ckpt={args.init_ckpt}"
                )

    model.config.use_cache = False
    if args.grad_checkpoint and not args.freeze_llm:
        try:
            model.language_model.gradient_checkpointing_enable()
        except Exception:
            model.gradient_checkpointing_enable()

    model.to(device)
    student.to(device)
    model_dtype = next(model.parameters()).dtype
    model.connector.to(device=device, dtype=model_dtype)
    desc_entity_like_token_ids_t = (
        torch.tensor(desc_entity_like_token_ids, device=device, dtype=torch.long)
        if desc_entity_like_token_ids
        else None
    )
    desc_entity_like_by_shape_t: Dict[str, torch.Tensor] = {}
    for sk, ids in desc_entity_like_ids_by_shape.items():
        if not ids:
            continue
        desc_entity_like_by_shape_t[sk] = torch.tensor(
            sorted(set(int(x) for x in ids)),
            device=device,
            dtype=torch.long,
        )
    desc_entity_cue_token_ids_t = (
        torch.tensor(desc_entity_cue_token_ids, device=device, dtype=torch.long)
        if desc_entity_cue_token_ids
        else None
    )

    def _bind_encode_images_with_region_mask(mm_model):
        def _encode_images(images):
            kwargs = {}
            kwargs["vision_feature_layer"] = mm_model.config.vision_feature_layer
            kwargs["vision_feature_select_strategy"] = mm_model.config.vision_feature_select_strategy
            images_local = images.to(device=mm_model.device, dtype=mm_model.dtype)
            image_features = mm_model.vision_tower(images_local, **kwargs)
            image_features = mm_model.connector(image_features)
            reg_mask = getattr(mm_model.vision_tower, "last_region_valid_mask", None)
            if reg_mask is not None and reg_mask.numel() > 0:
                reg_mask = reg_mask.to(device=image_features.device, dtype=image_features.dtype)
                k = reg_mask.shape[1]
                # Region tokens are prefixed before patch tokens.
                image_features[:, :k, :] = image_features[:, :k, :] * reg_mask.unsqueeze(-1)
            return image_features

        mm_model.encode_images = _encode_images

    _bind_encode_images_with_region_mask(model)
    model._region_attn_bias_ranges = None
    attn_bias_layers = int(args.region_attn_bias_layers)
    if args.explain_hard_paragraph_kv_gate and attn_bias_layers <= 0:
        # Hard K/V gating needs wrapped attention layers.
        attn_bias_layers = 9999
    _update_runtime_schedule(0, force_log=True)
    if beta_sched_active:
        print(
            f"[info] beta schedule active: "
            f"start_step={int(args.region_attn_bias_beta_schedule_start_step)} "
            f"end_step={int(args.region_attn_bias_beta_schedule_end_step)} "
            f"start={float(beta_sched_start):.4f} end={float(beta_sched_end):.4f}"
        )
    if cf_sched_active:
        print(
            f"[info] explain cf-weight schedule active: "
            f"start_step={int(args.explain_counterfactual_weight_schedule_start_step)} "
            f"end_step={int(args.explain_counterfactual_weight_schedule_end_step)} "
            f"start={float(cf_sched_start):.4f} end={float(cf_sched_end):.4f}"
        )
    wrapped_region_bias = bind_region_attention_bias(model, attn_bias_layers)
    if wrapped_region_bias > 0 and (
        region_attn_bias_beta_runtime > 0
        or beta_sched_active
        or args.paragraph_attn_neg_bias_gamma > 0
        or args.explain_hard_paragraph_kv_gate
    ):
        print(
            f"[info] region attention key-bias active: layers={attn_bias_layers} "
            f"beta={float(region_attn_bias_beta_runtime):.4f} paragraph_gamma={args.paragraph_attn_neg_bias_gamma} "
            f"hard_explain_kv_gate={args.explain_hard_paragraph_kv_gate} "
            f"task={args.region_attn_bias_task}"
        )

    def _infer_runtime_visual_tokens() -> int:
        try:
            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                bos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (tokenizer.pad_token_id or 0)
            explain_ids = tokenizer(TASK_TOKENS["explain"], add_special_tokens=False)["input_ids"]
            task_id = int(explain_ids[0]) if explain_ids else int(tokenizer.unk_token_id or 0)
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else int(task_id)
            input_ids_probe = torch.tensor(
                [[int(bos_id), int(task_id), int(model.config.image_token_index), int(eos_id)]],
                dtype=torch.long,
                device=device,
            )
            labels_probe = input_ids_probe.clone()
            labels_probe[:, :3] = -100
            attn_probe = torch.ones_like(input_ids_probe, dtype=torch.long)
            pixel_probe = torch.zeros(
                (1, 3, int(args.image_size), int(args.image_size)),
                dtype=model_dtype,
                device=device,
            )
            vision_wrapper.clear_regions()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    _, _, _, _, _, labels_mm_probe = model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids_probe,
                        position_ids=None,
                        attention_mask=attn_probe,
                        past_key_values=None,
                        labels=labels_probe,
                        images=pixel_probe,
                        image_sizes=None,
                    )
            pre_nz = (labels_probe[0] != -100).nonzero(as_tuple=False)
            mm_nz = (labels_mm_probe[0] != -100).nonzero(as_tuple=False)
            if pre_nz.numel() == 0 or mm_nz.numel() == 0:
                return 0
            pre_prompt = int(pre_nz[0].item())
            mm_prompt = int(mm_nz[0].item())
            vis_total = int(max(0, mm_prompt - pre_prompt + 1))
            return vis_total
        except Exception as e:
            print(f"[warn] runtime visual token probe failed: {e}")
            return 0

    runtime_visual_tokens_total = _infer_runtime_visual_tokens()
    if runtime_visual_tokens_total > 0:
        runtime_patch_tokens = max(1, int(runtime_visual_tokens_total) - int(region_token_slots))
        print(
            f"[info] runtime visual tokens: total={runtime_visual_tokens_total} "
            f"patch~={runtime_patch_tokens} region_slots={region_token_slots}"
        )
    else:
        runtime_patch_tokens = 0

    # Dataset
    replacements: List[Tuple[str, str]] = []
    for item in args.path_replace:
        if "=" not in item:
            continue
        src, dst = item.split("=", 1)
        replacements.append((src, dst))

    def _resolve_scicap_struct_jsonl(split_json: str) -> str:
        manual = (args.scicap_struct_jsonl or "").strip()
        if manual:
            return manual if os.path.isfile(manual) else ""
        split_stem = Path(split_json).stem.lower()
        cand = Path("outputs/structure_cache") / f"scicap_{split_stem}_struct.qf_v2.jsonl"
        if cand.is_file():
            return str(cand)
        return ""

    def _build_scistruct(split_json: str, ctx_mode: str) -> SciStructExplainDataset:
        region_grid_size = int(math.sqrt(max(1, _estimate_image_tokens(args.vision_pool, args.image_size, args.image_patch_size))))
        return SciStructExplainDataset(
            split_json=split_json,
            min_caption_len=args.min_caption_len,
            max_items=None,
            context_mode=ctx_mode,
            max_masks=args.max_masks,
            mask_min_area_ratio=args.mask_min_area_ratio,
            mask_max_area_ratio=args.mask_max_area_ratio,
            mask_sort=args.mask_sort,
            region_grid_size=region_grid_size,
            path_replace=replacements,
            use_ocr_context=not args.disable_explain_ocr,
            use_adesc_context=not args.disable_explain_adesc,
        )

    def _build_scicap(split_json: str) -> SciCapMultiScaleDataset:
        struct_jsonl = _resolve_scicap_struct_jsonl(split_json)
        return SciCapMultiScaleDataset(
            split_json=split_json,
            images_root=None,
            sample_mode="expand",
            max_items=None,
            min_len_short=args.min_len_short,
            min_len_long=args.min_len_long,
            min_len_desc=args.min_len_desc,
            use_desc=True,
            context_mode=args.context_mode,
            return_meta=(args.scicap_task_context_routing != "none"),
            struct_jsonl=struct_jsonl if struct_jsonl else None,
            max_struct_nodes=max(0, int(args.scicap_struct_max_nodes)) if int(args.scicap_struct_max_nodes) > 0 else None,
            max_image_side=None,
        )

    if args.dataset == "scistruct_explain":
        train_ds = _build_scistruct(args.train_json, args.context_mode)
    elif args.dataset == "stage4_multitask":
        explain_train_json = args.explain_train_json.strip() or args.train_json
        if not explain_train_json:
            raise RuntimeError("stage4_multitask requires --explain_train_json (or train_json as fallback)")
        explain_main_ds = _build_scistruct(explain_train_json, args.context_mode)
        explain_aux_datasets: List[torch.utils.data.Dataset] = []
        if args.stage4_include_scicap_explain:
            scicap_explain_json = args.stage4_scicap_explain_json.strip() or args.train_json
            if scicap_explain_json and os.path.isfile(scicap_explain_json):
                explain_aux_datasets.append(
                    SciCapExplainDataset(
                        split_json=scicap_explain_json,
                        min_explain_len=max(1, int(args.stage4_scicap_explain_min_len)),
                        use_long_fallback=bool(args.stage4_scicap_explain_use_long_fallback),
                        context_mode=args.context_mode,
                        max_items=None,
                    )
                )
            else:
                print(f"[warn] skip SciCap explain pool; missing json: {scicap_explain_json}")

        cap_datasets: List[torch.utils.data.Dataset] = []
        if args.train_json and os.path.isfile(args.train_json):
            cap_datasets.append(_build_scicap(args.train_json))
        for pjson in args.public_caption_json:
            if pjson and os.path.isfile(pjson):
                cap_datasets.append(_build_scicap(pjson))
            else:
                print(f"[warn] skip missing public caption json: {pjson}")
        if args.stage4_include_scistruct_caption:
            scif_cap_json = args.stage4_scistruct_caption_json.strip() or explain_train_json
            include_short = bool(args.stage4_scistruct_caption_short)
            include_long = bool(args.stage4_scistruct_caption_long)
            if not include_short and not include_long:
                include_short = True
                include_long = True
            if scif_cap_json and os.path.isfile(scif_cap_json):
                cap_datasets.append(
                    SciStructCaptionDataset(
                        split_json=scif_cap_json,
                        include_short=include_short,
                        include_long=include_long,
                        min_len_short=max(1, int(args.stage4_scistruct_caption_min_len_short)),
                        min_len_long=max(1, int(args.stage4_scistruct_caption_min_len_long)),
                        split_mode=str(args.stage4_scistruct_caption_split_mode),
                        word_boundary=max(1, int(args.stage4_scistruct_caption_word_boundary)),
                        max_items=None,
                    )
                )
            else:
                print(f"[warn] skip SciStruct caption pool; missing json: {scif_cap_json}")
        if not cap_datasets:
            raise RuntimeError("stage4_multitask requires non-explain caption data (SciCap/public)")

        explain_sets: List[torch.utils.data.Dataset] = [explain_main_ds] + explain_aux_datasets
        all_sets: List[torch.utils.data.Dataset] = explain_sets + cap_datasets
        explain_main_pool: List[Tuple[int, int, Dict]] = []
        explain_aux_pool: List[Tuple[int, int, Dict]] = []
        cap_pool: List[Tuple[int, int, Dict]] = []
        for did, ds_exp in enumerate(explain_sets):
            exp_samples = getattr(ds_exp, "samples", [])
            for i in range(len(exp_samples)):
                s = exp_samples[i]
                item = (did, i, {"text": s.get("text", ""), "scale": "explain", "context": s.get("context", "")})
                if did == 0:
                    explain_main_pool.append(item)
                else:
                    explain_aux_pool.append(item)
        for cap_offset, ds_cap in enumerate(cap_datasets):
            did = len(explain_sets) + cap_offset
            cap_samples = getattr(ds_cap, "samples", [])
            for i in range(len(cap_samples)):
                s = cap_samples[i]
                cap_pool.append((did, i, {"text": s.get("text", ""), "scale": s.get("scale", ""), "context": s.get("context", "")}))
        if not explain_main_pool:
            raise RuntimeError("stage4_multitask explain pool is empty")
        if not cap_pool:
            raise RuntimeError("stage4_multitask caption pool is empty")

        mix_rng = random.Random(args.seed)
        total_target = int(args.stage4_epoch_size) if args.stage4_epoch_size > 0 else (len(explain_main_pool) + len(explain_aux_pool) + len(cap_pool))
        total_target = max(1, total_target)
        er = max(0.0, min(1.0, float(args.stage4_explain_ratio)))
        n_exp = int(round(total_target * er))
        n_cap = max(0, total_target - n_exp)
        if n_exp == 0:
            n_exp = min(1, total_target)
            n_cap = max(0, total_target - n_exp)
        aux_ratio = max(0.0, min(1.0, float(args.stage4_scicap_explain_ratio)))
        n_exp_aux = min(n_exp, int(round(n_exp * aux_ratio))) if explain_aux_pool else 0
        n_exp_main = max(0, n_exp - n_exp_aux)
        chosen = (
            _sample_from_pool(explain_main_pool, n_exp_main, mix_rng)
            + _sample_from_pool(explain_aux_pool, n_exp_aux, mix_rng)
            + _sample_from_pool(cap_pool, n_cap, mix_rng)
        )
        mix_rng.shuffle(chosen)
        entries = [(did, sid) for did, sid, _ in chosen]
        samples = [meta for _, _, meta in chosen]
        train_ds = MixedTaskDataset(all_sets, entries, samples)
        print(
            f"[info] stage4 train mixture: total={len(train_ds)} explain={n_exp} cap={n_cap} "
            f"ratio_explain={er:.3f} pools(explain_main={len(explain_main_pool)}, "
            f"explain_aux={len(explain_aux_pool)}, cap={len(cap_pool)})"
        )
    else:
        train_ds = _build_scicap(args.train_json)
    if args.fixed_task:
        if isinstance(train_ds, MixedTaskDataset):
            print("[warn] fixed_task is ignored for stage4_multitask")
        elif hasattr(train_ds, "samples"):
            train_ds.samples = [s for s in train_ds.samples if s.get("scale") == args.fixed_task]
            if not train_ds.samples:
                raise RuntimeError(f"fixed_task={args.fixed_task} filtered all samples")

    bucket_bins = []
    if args.bucket_by_length and args.bucket_bins:
        bucket_bins = [int(x) for x in args.bucket_bins.split(",") if x.strip()]
        bucket_bins = sorted(set(bucket_bins))

    image_tokens = _estimate_image_tokens(args.vision_pool, args.image_size, args.image_patch_size)
    region_token_slots = args.max_masks if args.dataset in ("scistruct_explain", "stage4_multitask") else 0
    image_tokens_total = int(image_tokens + region_token_slots)
    image_tokens_total_attn = int(runtime_visual_tokens_total) if runtime_visual_tokens_total > 0 else int(image_tokens_total)

    def _set_region_attention_bias_for_inputs(
        input_ids_local: Optional[torch.Tensor],
        scales_local: Optional[List[str]],
        regions_local: Optional[List[List[Dict]]],
        labels_local: Optional[torch.Tensor] = None,
        contexts_local: Optional[List[str]] = None,
        explain_region_visible: Optional[List[bool]] = None,
        context_allow_tokens_local: Optional[torch.Tensor] = None,
        context_total_tokens_local: Optional[torch.Tensor] = None,
        context_ocr_tokens_local: Optional[torch.Tensor] = None,
        context_adesc_tokens_local: Optional[torch.Tensor] = None,
        context_para_tokens_local: Optional[torch.Tensor] = None,
    ) -> None:
        try:
            if hasattr(vision_wrapper, "set_scales"):
                vision_wrapper.set_scales(scales_local)
        except Exception:
            pass
        if (
            args.region_attn_bias_layers <= 0
            and not args.explain_hard_paragraph_kv_gate
        ):
            model._region_attn_bias_ranges = None
            return
        if args.disable_explain_gate_non_explain and scales_local is not None:
            has_explain = any(str(s or "") == "explain" for s in scales_local)
            if not has_explain:
                model._region_attn_bias_ranges = None
                return
        if input_ids_local is None:
            model._region_attn_bias_ranges = None
            return
        bsz = int(input_ids_local.size(0))
        ranges: List[Optional[List[Tuple[int, int, float]]]] = [None] * bsz
        any_bias = False
        if regions_local is not None:
            region_presence = [bool(r and len(r) > 0) for r in regions_local]
        else:
            region_presence = [False] * bsz
        for bi in range(bsz):
            pos = (input_ids_local[bi] == model.config.image_token_index).nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue
            img_pos = int(pos[0].item())
            region_visible_i = True
            if explain_region_visible is not None and bi < len(explain_region_visible):
                region_visible_i = bool(explain_region_visible[bi])
            scale_i = ""
            if scales_local is not None and bi < len(scales_local):
                scale_i = str(scales_local[bi] or "")
            task_match = (not scale_i) or (scale_i == args.region_attn_bias_task)
            entries: List[Tuple[int, int, float]] = []
            if region_attn_bias_beta_runtime > 0 and task_match and bi < len(region_presence) and bool(region_presence[bi]):
                rs = img_pos
                re = img_pos + region_token_slots
                if re > rs:
                    entries.append((rs, re, float(region_attn_bias_beta_runtime)))
            if args.explain_hard_paragraph_kv_gate and task_match:
                # EXPLAIN hard gate:
                # Default: only visual tokens ([region + image] span) can be used as K/V.
                # Optional: keep context tokens visible via --explain_allow_context_kv.
                if region_visible_i:
                    vis_s = max(0, img_pos)
                else:
                    vis_s = max(0, img_pos + region_token_slots)
                vis_e = max(vis_s, img_pos + image_tokens_total_attn)
                ctx_tokens = 0
                ctx_total_tokens = 0
                if context_allow_tokens_local is not None:
                    if isinstance(context_allow_tokens_local, torch.Tensor) and bi < context_allow_tokens_local.size(0):
                        ctx_tokens = int(max(0, int(context_allow_tokens_local[bi].item())))
                if context_total_tokens_local is not None:
                    if isinstance(context_total_tokens_local, torch.Tensor) and bi < context_total_tokens_local.size(0):
                        ctx_total_tokens = int(max(0, int(context_total_tokens_local[bi].item())))
                if ctx_tokens <= 0 and labels_local is not None and labels_local.dim() == 2 and bi < labels_local.size(0):
                    lab = labels_local[bi]
                    nz = (lab != -100).nonzero(as_tuple=False)
                    if nz.numel() > 0:
                        prompt_len = int(nz[0].item())
                        if ctx_total_tokens <= 0:
                            ctx_total_tokens = max(0, prompt_len - (img_pos + image_tokens_total_attn))
                        ctx_tokens = max(0, min(max(0, ctx_total_tokens), ctx_tokens))
                elif contexts_local is not None and bi < len(contexts_local):
                    ctx_text = str(contexts_local[bi] or "").strip()
                    if ctx_text:
                        ctx_limit = 0
                        if scale_i == "explain" and args.max_ctx_tokens_explain > 0:
                            ctx_limit = int(args.max_ctx_tokens_explain)
                        elif args.max_ctx_tokens > 0:
                            ctx_limit = int(args.max_ctx_tokens)
                        tok_kwargs = {"add_special_tokens": False, "truncation": True}
                        if ctx_limit > 0:
                            tok_kwargs["max_length"] = ctx_limit
                        elif args.max_length > 0:
                            tok_kwargs["max_length"] = int(args.max_length)
                        est = len(tokenizer("\n" + ctx_text, **tok_kwargs)["input_ids"])
                        if ctx_total_tokens <= 0:
                            ctx_total_tokens = int(est)
                        if ctx_tokens <= 0:
                            ctx_tokens = int(est)
                ctx_ocr_tokens = 0
                if context_ocr_tokens_local is not None:
                    if isinstance(context_ocr_tokens_local, torch.Tensor) and bi < context_ocr_tokens_local.size(0):
                        ctx_ocr_tokens = int(max(0, int(context_ocr_tokens_local[bi].item())))
                ctx_adesc_tokens = 0
                if context_adesc_tokens_local is not None:
                    if isinstance(context_adesc_tokens_local, torch.Tensor) and bi < context_adesc_tokens_local.size(0):
                        ctx_adesc_tokens = int(max(0, int(context_adesc_tokens_local[bi].item())))
                ctx_para_tokens = 0
                if context_para_tokens_local is not None:
                    if isinstance(context_para_tokens_local, torch.Tensor) and bi < context_para_tokens_local.size(0):
                        ctx_para_tokens = int(max(0, int(context_para_tokens_local[bi].item())))
                ctx_tokens = max(0, int(ctx_tokens))
                if ctx_total_tokens <= 0:
                    ctx_total_tokens = max(ctx_tokens, int(ctx_ocr_tokens + ctx_adesc_tokens + ctx_para_tokens))
                ctx_total_tokens = max(0, int(ctx_total_tokens))
                have_ctx_segments = bool(context_ocr_tokens_local is not None or context_adesc_tokens_local is not None or context_para_tokens_local is not None)
                if vis_s > 0:
                    entries.append((0, vis_s, float(args.explain_hard_paragraph_kv_bias)))
                tail_end = int(input_ids_local.size(1)) + int(image_tokens_total_attn) + 4096
                # Context K/V span in multimodal-expanded space starts right after visual tokens.
                # We anchor to vis_e instead of pre-mm prompt length to avoid offset drift.
                ctx_s = max(vis_e, img_pos + image_tokens_total_attn)
                ctx_span_hint = int(ctx_total_tokens if ctx_total_tokens > 0 else ctx_tokens)
                ctx_e = max(ctx_s, ctx_s + max(0, ctx_span_hint))
                if ctx_e > tail_end:
                    ctx_e = tail_end
                if ctx_s > ctx_e:
                    ctx_s = ctx_e
                if have_ctx_segments:
                    # Build explicit OCR/ADESC/PARA spans and block everything except visual + optionally OCR/ADESC.
                    ctx_span = max(0, int(ctx_e - ctx_s))
                    if ctx_span <= 0:
                        if ctx_s < tail_end:
                            entries.append((ctx_s, tail_end, float(args.explain_hard_paragraph_kv_bias)))
                        ranges[bi] = entries
                        any_bias = True
                        continue
                    ctx_ocr_tokens = max(0, min(ctx_span, int(ctx_ocr_tokens)))
                    rem_after_ocr = max(0, ctx_span - ctx_ocr_tokens)
                    ctx_adesc_tokens = max(0, min(rem_after_ocr, int(ctx_adesc_tokens)))
                    rem_after_allow = max(0, ctx_span - (ctx_ocr_tokens + ctx_adesc_tokens))
                    ctx_para_tokens = max(0, min(rem_after_allow, int(ctx_para_tokens)))
                    allow_n = int(ctx_ocr_tokens + ctx_adesc_tokens)
                    para_n = int(ctx_para_tokens if ctx_para_tokens > 0 else max(0, ctx_span - allow_n))
                    allow_ranges: List[Tuple[int, int]] = []
                    if args.explain_allow_context_kv:
                        o_s = ctx_s
                        o_e = min(ctx_e, o_s + int(ctx_ocr_tokens))
                        if o_e > o_s:
                            allow_ranges.append((o_s, o_e))
                        a_s = o_e
                        a_e = min(ctx_e, a_s + int(ctx_adesc_tokens))
                        if a_e > a_s:
                            allow_ranges.append((a_s, a_e))
                    cursor = vis_e
                    if allow_ranges:
                        for s_a, e_a in allow_ranges:
                            s_blk = max(cursor, min(tail_end, s_a))
                            if s_blk > cursor:
                                entries.append((cursor, s_blk, float(args.explain_hard_paragraph_kv_bias)))
                            cursor = max(cursor, min(tail_end, e_a))
                        if cursor < tail_end:
                                entries.append((cursor, tail_end, float(args.explain_hard_paragraph_kv_bias)))
                    else:
                        entries.append((vis_e, tail_end, float(args.explain_hard_paragraph_kv_bias)))
                    # Final hard rule for EXPLAIN: paragraph K/V must never be visible.
                    if para_n > 0:
                        para_e = int(ctx_e)
                        para_s = max(ctx_s + allow_n, para_e - para_n)
                        if para_e > para_s:
                            entries.append((para_s, para_e, float(args.explain_hard_paragraph_kv_bias)))
                elif args.explain_allow_context_kv and ctx_tokens > 0:
                    # Fallback: keep prefix context span visible while blocking all other text K/V.
                    ctx_span = max(0, int(ctx_e - ctx_s))
                    allow_n = min(max(0, int(ctx_tokens)), ctx_span)
                    ctx_e_allow = ctx_s + allow_n
                    if ctx_s > vis_e:
                        entries.append((vis_e, ctx_s, float(args.explain_hard_paragraph_kv_bias)))
                    entries.append((ctx_e_allow, tail_end, float(args.explain_hard_paragraph_kv_bias)))
                    if ctx_span > allow_n:
                        para_s = ctx_e_allow
                        para_e = ctx_s + ctx_span
                        if para_e > para_s:
                            entries.append((para_s, para_e, float(args.explain_hard_paragraph_kv_bias)))
                else:
                    entries.append((vis_e, tail_end, float(args.explain_hard_paragraph_kv_bias)))
            elif args.paragraph_attn_neg_bias_gamma > 0 and task_match:
                ctx_tokens = 0
                if labels_local is not None and labels_local.dim() == 2 and bi < labels_local.size(0):
                    lab = labels_local[bi]
                    nz = (lab != -100).nonzero(as_tuple=False)
                    if nz.numel() > 0:
                        prompt_len = int(nz[0].item())
                        ctx_tokens = max(0, prompt_len - (img_pos + 1))
                elif contexts_local is not None and bi < len(contexts_local):
                    ctx_text = str(contexts_local[bi] or "").strip()
                    if ctx_text:
                        ctx_limit = 0
                        if scale_i == "explain" and args.max_ctx_tokens_explain > 0:
                            ctx_limit = int(args.max_ctx_tokens_explain)
                        elif args.max_ctx_tokens > 0:
                            ctx_limit = int(args.max_ctx_tokens)
                        tok_kwargs = {"add_special_tokens": False, "truncation": True}
                        if ctx_limit > 0:
                            tok_kwargs["max_length"] = ctx_limit
                        elif args.max_length > 0:
                            tok_kwargs["max_length"] = int(args.max_length)
                        ctx_tokens = len(tokenizer("\n" + ctx_text, **tok_kwargs)["input_ids"])
                if ctx_tokens > 0:
                    ps = img_pos + image_tokens_total_attn
                    pe = ps + int(ctx_tokens)
                    if pe > ps:
                        entries.append((ps, pe, -float(args.paragraph_attn_neg_bias_gamma)))
            if entries:
                ranges[bi] = entries
                any_bias = True
        if not any_bias:
            model._region_attn_bias_ranges = None
            return
        model._region_attn_bias_ranges = ranges

    def _clear_region_attention_bias() -> None:
        model._region_attn_bias_ranges = None

    def _ensure_log_attn_exportable() -> None:
        if not args.log_attn:
            return
        _force_eager_attn(model, "model")
        _force_eager_attn(getattr(model, "language_model", None), "language_model")

    collate = lambda b: build_batch(
        tokenizer,
        b,
        args.max_length,
        {
            "short": args.max_target_short,
            "long": args.max_target_long,
            "desc": args.max_target_desc,
            "explain": args.max_target_long,
        },
        model.config.image_token_index,
        add_eos=True,
        fixed_task=args.fixed_task or None,
        context_dropout=args.context_dropout,
        paragraph_token_dropout=args.paragraph_token_dropout,
        max_ctx_tokens=args.max_ctx_tokens,
        max_ctx_tokens_explain=args.max_ctx_tokens_explain,
        explain_ctx_min_adesc_tokens=args.explain_ctx_min_adesc_tokens,
        explain_ctx_max_ocr_tokens=args.explain_ctx_max_ocr_tokens,
        bucket_bins=bucket_bins if bucket_bins else None,
        image_tokens=image_tokens_total,
        scicap_prompt_style=args.scicap_prompt_style,
        scicap_task_context_routing=args.scicap_task_context_routing,
        enable_task_style_tokens=args.enable_task_style_tokens,
        use_context_placeholders=args.use_context_placeholders,
    )

    if bucket_bins and hasattr(train_ds, "samples"):
        lengths = []
        for s in train_ds.samples:
            text = s.get("text", "")
            scale = s.get("scale", "long")
            ctx = s.get("context", "")
            task_token = TASK_TOKENS.get(scale, "<CAPTION_LONG>")
            task_len = len(tokenizer(task_token, add_special_tokens=False)["input_ids"]) or 1
            task_style_len = len(_task_style_ids(tokenizer, scale, args.enable_task_style_tokens))
            task_prompt_len = len(_task_prompt_ids(tokenizer, scale, args.scicap_prompt_style))
            max_tgt = {
                "short": args.max_target_short,
                "long": args.max_target_long,
                "desc": args.max_target_desc,
                "explain": args.max_target_long,
            }.get(scale, args.max_target_long)
            tgt_cap = int(max_tgt) if (max_tgt and max_tgt > 0) else int(args.max_length if args.max_length > 0 else 0)
            tgt_ids = _tokenize_ids(tokenizer, text, add_special_tokens=False, max_length=tgt_cap)
            if max_tgt > 0 and len(tgt_ids) > max_tgt:
                tgt_ids = tgt_ids[:max_tgt]
            ctx_text = _resolve_scicap_task_context(
                scale=scale,
                context=(ctx or "").strip(),
                context_meta=s.get("meta", {}) if isinstance(s, dict) else None,
                routing_mode=args.scicap_task_context_routing,
                use_placeholders=args.use_context_placeholders,
            )
            ctx_limit = args.max_ctx_tokens
            if scale == "explain" and args.max_ctx_tokens_explain > 0:
                ctx_limit = args.max_ctx_tokens_explain
            ctx_cap = int(ctx_limit) if (ctx_limit and ctx_limit > 0) else int(args.max_length if args.max_length > 0 else 0)
            ctx_ids = _tokenize_ids(tokenizer, "\n" + ctx_text, add_special_tokens=False, max_length=ctx_cap) if ctx_text else []
            if ctx_limit and ctx_ids:
                ctx_ids = ctx_ids[:ctx_limit]
            base_len = 1 + task_len + task_style_len + task_prompt_len + 1 + len(ctx_ids) + len(tgt_ids)
            if args.max_length > 0:
                base_len = min(base_len, args.max_length)
            eff_len = base_len + max(0, image_tokens_total - 1)
            lengths.append(eff_len)

        batch_sampler = LengthBucketBatchSampler(
            lengths=lengths,
            bucket_bins=bucket_bins,
            batch_size=args.batch_size,
            drop_last=False,
            seed=args.seed,
        )
        train_dl = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
    if args.contrastive_weight > 0 and args.batch_size < 2 and args.contrastive_type != "mask_drop":
        print("[warn] contrastive loss enabled but batch_size < 2; margin term will be skipped")
    if args.contrastive_type == "shuffle" and args.batch_size < 2:
        print("[warn] contrastive_type=shuffle but batch_size < 2; will fallback to zero-image contrast")
    if args.batch_contrastive and args.batch_size < 2:
        print("[warn] batch_contrastive enabled but batch_size < 2; disabling batch_contrastive")
        args.batch_contrastive = False
    if args.context_mix_mode != "none":
        if args.context_mix_mode == "stage3a":
            if args.context_region_only_prob <= 0 and args.context_region_ctx_prob <= 0:
                args.context_region_only_prob = 0.9
                args.context_region_ctx_prob = 0.1
                args.context_paragraph_only_prob = 0.0
        elif args.context_mix_mode == "stage3b":
            if args.context_region_only_prob <= 0 and args.context_region_ctx_prob <= 0:
                args.context_region_only_prob = 0.6
                args.context_region_ctx_prob = 0.3
                args.context_paragraph_only_prob = 0.1
        total = args.context_region_only_prob + args.context_region_ctx_prob + args.context_paragraph_only_prob
        if total <= 0:
            print("[warn] context_mix_mode enabled but probabilities sum to 0; disabling mix")
            args.context_mix_mode = "none"
        elif abs(total - 1.0) > 1e-3:
            args.context_region_only_prob /= total
            args.context_region_ctx_prob /= total
            args.context_paragraph_only_prob /= total
            print(f"[info] normalized context mix probs to sum=1.0 (mode={args.context_mix_mode})")

    val_ds = None
    val_dl = None
    val_scicap_ds = None
    val_scicap_dl = None
    val_explain_ds = None
    val_explain_dl = None
    val_batch_size = args.val_batch_size if args.val_batch_size and args.val_batch_size > 0 else args.batch_size

    val_collate_fn = lambda b: build_batch(
        tokenizer,
        b,
        args.max_length,
        {
            "short": args.max_target_short,
            "long": args.max_target_long,
            "desc": args.max_target_desc,
            "explain": args.max_target_long,
        },
        model.config.image_token_index,
        add_eos=True,
        fixed_task=args.fixed_task or None,
        context_dropout=0.0,
        paragraph_token_dropout=0.0,
        max_ctx_tokens=args.max_ctx_tokens,

        max_ctx_tokens_explain=args.max_ctx_tokens_explain,
        explain_ctx_min_adesc_tokens=args.explain_ctx_min_adesc_tokens,
        explain_ctx_max_ocr_tokens=args.explain_ctx_max_ocr_tokens,
        bucket_bins=None,
        image_tokens=image_tokens_total,
        scicap_prompt_style=args.scicap_prompt_style,
        scicap_task_context_routing=args.scicap_task_context_routing,
        enable_task_style_tokens=args.enable_task_style_tokens,
        use_context_placeholders=args.use_context_placeholders,
    )

    if args.val_json:
        if args.dataset == "scistruct_explain":
            val_ds = _build_scistruct(args.val_json, args.context_mode)
            if len(val_ds) == 0:
                raise RuntimeError(f"validation dataset is empty: {args.val_json}")
            val_dl = DataLoader(
                val_ds,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=max(0, args.num_workers // 2),
                collate_fn=val_collate_fn,
            )
            print(f"[info] validation enabled: n={len(val_ds)} every={args.val_every} steps")
        elif args.dataset == "stage4_multitask":
            # Stage-4 corrected validation split:
            # - SciCap only for short/long/desc GT-based metrics
            # - SciStruct only for EXPLAIN behavior diagnostics
            val_scicap_ds = _build_scicap(args.val_json)
            if hasattr(val_scicap_ds, "samples"):
                val_scicap_ds.samples = [s for s in val_scicap_ds.samples if s.get("scale") in ("short", "long", "desc")]
            explain_val_json = args.explain_val_json.strip() or args.val_json
            val_explain_ds = _build_scistruct(explain_val_json, args.context_mode)
            if len(val_scicap_ds) == 0:
                raise RuntimeError(f"stage4 scicap validation dataset is empty: {args.val_json}")
            if len(val_explain_ds) == 0:
                raise RuntimeError(f"stage4 explain validation dataset is empty: {explain_val_json}")
            val_scicap_dl = DataLoader(
                val_scicap_ds,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=max(0, args.num_workers // 2),
                collate_fn=val_collate_fn,
            )
            val_explain_dl = DataLoader(
                val_explain_ds,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=max(0, args.num_workers // 2),
                collate_fn=val_collate_fn,
            )
            print(
                f"[info] stage4 validation split enabled: scicap={len(val_scicap_ds)} "
                f"scistruct_explain={len(val_explain_ds)} every={args.val_every} steps"
            )
        else:
            val_ds = _build_scicap(args.val_json)
            if args.fixed_task and hasattr(val_ds, "samples"):
                val_ds.samples = [s for s in val_ds.samples if s.get("scale") == args.fixed_task]
                if not val_ds.samples:
                    raise RuntimeError(f"fixed_task={args.fixed_task} filtered all val samples")
            if len(val_ds) == 0:
                raise RuntimeError(f"validation dataset is empty: {args.val_json}")
            val_dl = DataLoader(
                val_ds,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=max(0, args.num_workers // 2),
                collate_fn=val_collate_fn,
            )
            print(f"[info] validation enabled: n={len(val_ds)} every={args.val_every} steps")

    sample_ds = val_ds if val_ds is not None else train_ds
    sample_ds_scicap = None
    sample_ds_explain = None
    sample_out_dir = Path(args.sample_out_dir) if args.sample_out_dir else (out_dir / "samples")
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    sample_rng = random.Random(args.seed + 1234)
    sample_indices: List[int] = []
    sample_indices_scicap: List[int] = []
    sample_indices_explain: List[int] = []
    if args.dataset == "stage4_multitask":
        sample_ds_scicap = val_scicap_ds if val_scicap_ds is not None else _build_scicap(args.train_json)
        explain_sample_json = args.explain_val_json.strip() or args.explain_train_json.strip() or args.train_json
        sample_ds_explain = val_explain_ds if val_explain_ds is not None else _build_scistruct(explain_sample_json, args.context_mode)
        if args.sample_num > 0 and sample_ds_scicap is not None and len(sample_ds_scicap) > 0:
            sample_indices_scicap = list(range(len(sample_ds_scicap)))
            sample_rng.shuffle(sample_indices_scicap)
            sample_indices_scicap = sample_indices_scicap[: min(args.sample_num, len(sample_ds_scicap))]
        if args.sample_num > 0 and sample_ds_explain is not None and len(sample_ds_explain) > 0:
            sample_indices_explain = list(range(len(sample_ds_explain)))
            sample_rng.shuffle(sample_indices_explain)
            sample_indices_explain = sample_indices_explain[: min(args.sample_num, len(sample_ds_explain))]
    else:
        if args.sample_num > 0 and len(sample_ds) > 0:
            sample_indices = list(range(len(sample_ds)))
            sample_rng.shuffle(sample_indices)
            sample_indices = sample_indices[: min(args.sample_num, len(sample_ds))]
    sample_tasks = [t.strip() for t in args.sample_tasks.split(",") if t.strip() in TASK_TOKENS]
    if not sample_tasks:
        sample_tasks = [args.fixed_task] if args.fixed_task else ["short", "long", "desc", "explain"]
    sample_eval_task = args.fixed_task if args.fixed_task else "explain"
    if args.dataset == "stage4_multitask":
        print(
            f"[info] stage4 sample split enabled: scicap_n={len(sample_indices_scicap)} "
            f"scistruct_explain_n={len(sample_indices_explain)} every={args.sample_every}"
        )
    elif args.sample_mode == "explain_diag":
        print(
            f"[info] sample enabled: n={len(sample_indices)} mode=explain_diag "
            f"task={sample_eval_task} every={args.sample_every}"
        )
    elif args.sample_mode == "stage3_modes":
        print(
            f"[info] sample enabled: n={len(sample_indices)} mode=stage3_modes "
            f"task={sample_eval_task} every={args.sample_every}"
        )
    else:
        print(f"[info] sample enabled: n={len(sample_indices)} mode=tasks tasks={sample_tasks} every={args.sample_every}")

    warmup_dl = None
    if args.warmup_steps > 0:
        warmup_ds = SciCapTextOnlyDataset(
            split_json=args.train_json,
            task=args.warmup_task,
            min_len_short=args.min_len_short,
            min_len_long=args.min_len_long,
            min_len_desc=args.min_len_desc,
        )
        warmup_dl = DataLoader(
            warmup_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: build_warmup_batch(
                tokenizer,
                b,
                args.max_length,
                {
                    "short": args.max_target_short,
                    "long": args.max_target_long,
                    "desc": args.max_target_desc,
                }.get(args.warmup_task, args.max_target_short),
                TASK_TOKENS.get(args.warmup_task, TASK_TOKENS["short"]),
                add_eos=True,
            ),
        )

    # Optimizer
    lora_params = []
    connector_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "connector" in name:
            connector_params.append(p)
        elif "lora_" in name:
            lora_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lora_lr})
    if connector_params:
        param_groups.append({"params": connector_params, "lr": args.connector_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr})

    if not param_groups:
        raise RuntimeError("no trainable parameters found")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    optimizer_param_ids = {id(p) for g in optimizer.param_groups for p in g.get("params", [])}

    def _add_new_trainables_to_optimizer() -> int:
        new_params: List[torch.nn.Parameter] = []
        for _, p in model.named_parameters():
            if (not p.requires_grad) or (id(p) in optimizer_param_ids):
                continue
            new_params.append(p)
            optimizer_param_ids.add(id(p))
        if not new_params:
            return 0
        optimizer.add_param_group(
            {
                "params": new_params,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            }
        )
        return int(len(new_params))

    use_scaler = args.amp and device.type == "cuda" and model_dtype == torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()

    def _build_explain_attn_inputs(
        input_ids_local: torch.Tensor,
        labels_mm_local: torch.Tensor,
        regions_local: Optional[List[List[Dict]]],
        k_len: int,
        q_len: int,
        bias_ranges_local: Optional[List],
        context_allow_tokens_local: Optional[torch.Tensor] = None,
        context_total_tokens_local: Optional[torch.Tensor] = None,
        context_ocr_tokens_local: Optional[torch.Tensor] = None,
        context_adesc_tokens_local: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        bsz = int(input_ids_local.size(0))
        q_indices: List[torch.Tensor] = []
        k_region_indices: List[torch.Tensor] = []
        k_textctx_indices: List[torch.Tensor] = []
        k_image_indices: List[torch.Tensor] = []
        k_ocr_indices: List[torch.Tensor] = []
        k_adesc_indices: List[torch.Tensor] = []
        k_para_indices: List[torch.Tensor] = []
        visible_k_masks: List[torch.Tensor] = []
        ctx_starts: List[int] = []
        ctx_ends: List[int] = []
        ocr_starts: List[int] = []
        ocr_ends: List[int] = []
        adesc_starts: List[int] = []
        adesc_ends: List[int] = []
        para_starts: List[int] = []
        para_ends: List[int] = []
        prompt_lens: List[int] = []
        ctx_total_hints: List[int] = []
        allow_ns: List[int] = []
        for bi in range(bsz):
            empty = torch.empty(0, dtype=torch.long)
            pos = (input_ids_local[bi] == model.config.image_token_index).nonzero(as_tuple=False)
            if pos.numel() == 0:
                q_indices.append(empty)
                k_region_indices.append(empty)
                k_textctx_indices.append(empty)
                k_image_indices.append(empty)
                k_ocr_indices.append(empty)
                k_adesc_indices.append(empty)
                k_para_indices.append(empty)
                visible_k_masks.append(torch.ones((k_len,), dtype=torch.bool))
                ctx_starts.append(0)
                ctx_ends.append(0)
                ocr_starts.append(0)
                ocr_ends.append(0)
                adesc_starts.append(0)
                adesc_ends.append(0)
                para_starts.append(0)
                para_ends.append(0)
                prompt_lens.append(0)
                ctx_total_hints.append(0)
                allow_ns.append(0)
                continue
            img_pos = int(pos[0].item())
            lab = labels_mm_local[bi]
            q_pos = (lab != -100).nonzero(as_tuple=False).squeeze(-1)
            if q_pos.numel() > 0:
                q_pos = q_pos[(q_pos >= 0) & (q_pos < q_len)]
            if q_pos.numel() == 0:
                q_pos = empty
            prompt_len = int(q_pos[0].item()) if q_pos.numel() > 0 else int(img_pos + image_tokens_total_attn)
            prompt_len = max(0, min(k_len, prompt_len))
            reg_s = max(0, min(k_len, img_pos))
            reg_e = max(reg_s, min(k_len, img_pos + region_token_slots))
            img_s = max(0, min(k_len, img_pos + region_token_slots))
            img_e = max(img_s, min(k_len, img_pos + image_tokens_total_attn))
            ctx_total_hint = 0
            if context_total_tokens_local is not None:
                try:
                    ctx_total_hint = int(max(0, int(context_total_tokens_local[bi].item())))
                except Exception:
                    ctx_total_hint = 0
            ctx_s = max(0, min(k_len, img_pos + image_tokens_total_attn))
            if ctx_total_hint > 0:
                ctx_e = max(ctx_s, min(k_len, ctx_s + ctx_total_hint))
            else:
                ctx_e = max(ctx_s, min(k_len, prompt_len))
            has_region = bool(regions_local and bi < len(regions_local) and regions_local[bi] and len(regions_local[bi]) > 0)
            ctx_total_n = max(0, int(ctx_e - ctx_s))
            allow_n = 0
            if context_allow_tokens_local is not None:
                try:
                    allow_n = int(max(0, int(context_allow_tokens_local[bi].item())))
                except Exception:
                    allow_n = 0
            allow_n = max(0, min(ctx_total_n, int(allow_n)))
            ocr_n = 0
            if context_ocr_tokens_local is not None:
                try:
                    ocr_n = int(max(0, int(context_ocr_tokens_local[bi].item())))
                except Exception:
                    ocr_n = 0
            ocr_n = max(0, min(allow_n, int(ocr_n)))
            adesc_n = 0
            if context_adesc_tokens_local is not None:
                try:
                    adesc_n = int(max(0, int(context_adesc_tokens_local[bi].item())))
                except Exception:
                    adesc_n = 0
            adesc_n = max(0, min(max(0, allow_n - ocr_n), int(adesc_n)))
            para_n = max(0, int(ctx_total_n - allow_n))
            ocr_s = int(ctx_s)
            ocr_e = int(min(ctx_e, ocr_s + ocr_n))
            ad_s = int(ocr_e)
            ad_e = int(min(ctx_e, ad_s + adesc_n))
            pa_s = int(ad_e)
            pa_e = int(min(ctx_e, pa_s + para_n))

            q_indices.append(q_pos.detach().cpu())
            if has_region and reg_e > reg_s:
                k_region_indices.append(torch.arange(reg_s, reg_e, dtype=torch.long))
            else:
                k_region_indices.append(empty)
            if img_e > img_s:
                k_image_indices.append(torch.arange(img_s, img_e, dtype=torch.long))
            else:
                k_image_indices.append(empty)
            text_parts: List[torch.Tensor] = []
            if reg_s > 0:
                text_parts.append(torch.arange(0, reg_s, dtype=torch.long))
            if k_len > img_e:
                text_parts.append(torch.arange(img_e, k_len, dtype=torch.long))
            if text_parts:
                k_text = torch.cat(text_parts, dim=0)
                if k_text.numel() > 0:
                    k_textctx_indices.append(k_text)
                else:
                    k_textctx_indices.append(empty)
            else:
                k_textctx_indices.append(empty)
            if ocr_n > 0:
                k_ocr_indices.append(torch.arange(ocr_s, ocr_e, dtype=torch.long))
            else:
                k_ocr_indices.append(empty)
            if adesc_n > 0:
                k_adesc_indices.append(torch.arange(ad_s, ad_e, dtype=torch.long))
            else:
                k_adesc_indices.append(empty)
            if para_n > 0:
                k_para_indices.append(torch.arange(pa_s, pa_e, dtype=torch.long))
            else:
                k_para_indices.append(empty)
            ctx_starts.append(int(ctx_s))
            ctx_ends.append(int(ctx_e))
            ocr_starts.append(int(ocr_s))
            ocr_ends.append(int(ocr_e))
            adesc_starts.append(int(ad_s))
            adesc_ends.append(int(ad_e))
            para_starts.append(int(pa_s))
            para_ends.append(int(pa_e))
            prompt_lens.append(int(prompt_len))
            ctx_total_hints.append(int(ctx_total_hint))
            allow_ns.append(int(allow_n))

            vm = torch.ones((k_len,), dtype=torch.bool)
            if bias_ranges_local is not None and bi < len(bias_ranges_local):
                entries = _parse_bias_entries(bias_ranges_local[bi])
                for bs, be, bb in entries:
                    if bb <= -1e3:
                        ss = max(0, min(k_len, int(bs)))
                        ee = max(ss, min(k_len, int(be)))
                        if ee > ss:
                            vm[ss:ee] = False
            visible_k_masks.append(vm)
        return {
            "q_indices": q_indices,
            "k_region_indices": k_region_indices,
            "k_textctx_indices": k_textctx_indices,
            "k_image_indices": k_image_indices,
            "k_ocr_indices": k_ocr_indices,
            "k_adesc_indices": k_adesc_indices,
            "k_para_indices": k_para_indices,
            "visible_k_mask": visible_k_masks,
            "ctx_starts": ctx_starts,
            "ctx_ends": ctx_ends,
            "ocr_starts": ocr_starts,
            "ocr_ends": ocr_ends,
            "adesc_starts": adesc_starts,
            "adesc_ends": adesc_ends,
            "para_starts": para_starts,
            "para_ends": para_ends,
            "prompt_lens": prompt_lens,
            "ctx_total_hints": ctx_total_hints,
            "allow_ns": allow_ns,
        }

    def _collect_attn_mass_from_forward(
        attentions: Optional[List[torch.Tensor]],
        input_ids_local: torch.Tensor,
        labels_mm_local: torch.Tensor,
        regions_local: Optional[List[List[Dict]]],
        bias_ranges_local: Optional[List] = None,
        context_allow_tokens_local: Optional[torch.Tensor] = None,
        context_total_tokens_local: Optional[torch.Tensor] = None,
        context_ocr_tokens_local: Optional[torch.Tensor] = None,
        context_adesc_tokens_local: Optional[torch.Tensor] = None,
        context_para_tokens_local: Optional[torch.Tensor] = None,
        context_ocr_hash_local: Optional[torch.Tensor] = None,
        context_adesc_hash_local: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, float]]:
        if not attentions:
            return None
        att0 = None
        for att in attentions:
            if att is not None and att.dim() == 4:
                att0 = att
                break
        if att0 is None:
            return None
        k_len = int(att0.shape[-1])
        q_len = int(att0.shape[-2])
        idx_bundle = _build_explain_attn_inputs(
            input_ids_local=input_ids_local,
            labels_mm_local=labels_mm_local,
            regions_local=regions_local,
            k_len=k_len,
            q_len=q_len,
            bias_ranges_local=bias_ranges_local,
            context_allow_tokens_local=context_allow_tokens_local,
            context_total_tokens_local=context_total_tokens_local,
            context_ocr_tokens_local=context_ocr_tokens_local,
            context_adesc_tokens_local=context_adesc_tokens_local,
        )
        stat = compute_explain_attn_metrics(
            attn_probs=list(attentions),
            q_indices=idx_bundle["q_indices"],
            k_region_indices=idx_bundle["k_region_indices"],
            k_textctx_indices=idx_bundle["k_textctx_indices"],
            k_image_indices=idx_bundle["k_image_indices"],
            k_ocr_indices=idx_bundle["k_ocr_indices"],
            k_adesc_indices=idx_bundle["k_adesc_indices"],
            k_para_indices=idx_bundle["k_para_indices"],
            visible_k_mask=idx_bundle["visible_k_mask"],
            last_k_layers=int(args.explain_attn_last_k_layers),
            avg_over_heads=True,
        )
        if stat.get("samples", 0) <= 0:
            return None
        stat["seq_len_input_ids"] = float(input_ids_local.size(1))
        stat["attn_k_len"] = float(k_len)
        stat["attn_q_len"] = float(q_len)
        stat["q_token_count"] = float(sum(int(q.numel()) for q in idx_bundle["q_indices"]))
        if idx_bundle.get("ctx_starts"):
            stat["span_ctx_start"] = float(idx_bundle["ctx_starts"][0])
            stat["span_ctx_end"] = float(idx_bundle["ctx_ends"][0])
            stat["span_ocr_start"] = float(idx_bundle["ocr_starts"][0])
            stat["span_ocr_end"] = float(idx_bundle["ocr_ends"][0])
            stat["span_adesc_start"] = float(idx_bundle["adesc_starts"][0])
            stat["span_adesc_end"] = float(idx_bundle["adesc_ends"][0])
            stat["span_para_start"] = float(idx_bundle["para_starts"][0])
            stat["span_para_end"] = float(idx_bundle["para_ends"][0])
            stat["prompt_len_mm"] = float(idx_bundle["prompt_lens"][0])
            stat["ctx_total_hint"] = float(idx_bundle["ctx_total_hints"][0])
            stat["allow_n_hint"] = float(idx_bundle["allow_ns"][0])
        stat["q_len"] = float(q_len)
        stat["region_k_len"] = float(
            sum(int(x.numel()) for x in idx_bundle["k_region_indices"]) / max(1, len(idx_bundle["k_region_indices"]))
        )
        stat["image_k_len"] = float(
            sum(int(x.numel()) for x in idx_bundle["k_image_indices"]) / max(1, len(idx_bundle["k_image_indices"]))
        )
        stat["textctx_k_len"] = float(
            sum(int(x.numel()) for x in idx_bundle["k_textctx_indices"]) / max(1, len(idx_bundle["k_textctx_indices"]))
        )
        stat["visible_k_nonzero"] = float(
            sum(int(x.sum().item()) for x in idx_bundle["visible_k_mask"]) / max(1, len(idx_bundle["visible_k_mask"]))
        )
        vis_region = 0
        vis_img = 0
        vis_textctx = 0
        vis_ocr = 0
        vis_adesc = 0
        vis_para = 0
        vis_total = 0
        for i, vm in enumerate(idx_bundle["visible_k_mask"]):
            if vm is None:
                continue
            vm_cpu = vm.to(dtype=torch.bool).cpu()
            vis_total += int(vm_cpu.sum().item())
            if i < len(idx_bundle["k_region_indices"]):
                ridx = idx_bundle["k_region_indices"][i]
                if ridx is not None and ridx.numel() > 0:
                    ridx = ridx[(ridx >= 0) & (ridx < vm_cpu.numel())]
                    if ridx.numel() > 0:
                        vis_region += int(vm_cpu.index_select(0, ridx).sum().item())
            if i < len(idx_bundle["k_image_indices"]):
                iidx = idx_bundle["k_image_indices"][i]
                if iidx is not None and iidx.numel() > 0:
                    iidx = iidx[(iidx >= 0) & (iidx < vm_cpu.numel())]
                    if iidx.numel() > 0:
                        vis_img += int(vm_cpu.index_select(0, iidx).sum().item())
            if i < len(idx_bundle["k_textctx_indices"]):
                cidx = idx_bundle["k_textctx_indices"][i]
                if cidx is not None and cidx.numel() > 0:
                    cidx = cidx[(cidx >= 0) & (cidx < vm_cpu.numel())]
                    if cidx.numel() > 0:
                        vis_textctx += int(vm_cpu.index_select(0, cidx).sum().item())
            if i < len(idx_bundle["k_ocr_indices"]):
                oidx = idx_bundle["k_ocr_indices"][i]
                if oidx is not None and oidx.numel() > 0:
                    oidx = oidx[(oidx >= 0) & (oidx < vm_cpu.numel())]
                    if oidx.numel() > 0:
                        vis_ocr += int(vm_cpu.index_select(0, oidx).sum().item())
            if i < len(idx_bundle["k_adesc_indices"]):
                aidx = idx_bundle["k_adesc_indices"][i]
                if aidx is not None and aidx.numel() > 0:
                    aidx = aidx[(aidx >= 0) & (aidx < vm_cpu.numel())]
                    if aidx.numel() > 0:
                        vis_adesc += int(vm_cpu.index_select(0, aidx).sum().item())
            if i < len(idx_bundle["k_para_indices"]):
                pidx = idx_bundle["k_para_indices"][i]
                if pidx is not None and pidx.numel() > 0:
                    pidx = pidx[(pidx >= 0) & (pidx < vm_cpu.numel())]
                    if pidx.numel() > 0:
                        vis_para += int(vm_cpu.index_select(0, pidx).sum().item())
        denom = max(1, len(idx_bundle["visible_k_mask"]))
        stat["visible_k_count"] = float(vis_total) / float(denom)
        stat["visible_k_region_count"] = float(vis_region) / float(denom)
        stat["visible_k_img_count"] = float(vis_img) / float(denom)
        stat["visible_k_textctx_count"] = float(vis_textctx) / float(denom)
        stat["visible_k_ocr_count"] = float(vis_ocr) / float(denom)
        stat["visible_k_adesc_count"] = float(vis_adesc) / float(denom)
        stat["visible_k_para_count"] = float(vis_para) / float(denom)
        if context_ocr_tokens_local is not None:
            try:
                stat["len_ctx_ocr_tokens"] = float(context_ocr_tokens_local.float().mean().item())
            except Exception:
                stat["len_ctx_ocr_tokens"] = 0.0
        if context_adesc_tokens_local is not None:
            try:
                stat["len_ctx_adesc_tokens"] = float(context_adesc_tokens_local.float().mean().item())
            except Exception:
                stat["len_ctx_adesc_tokens"] = 0.0
        if context_para_tokens_local is not None:
            try:
                stat["len_ctx_para_tokens"] = float(context_para_tokens_local.float().mean().item())
            except Exception:
                stat["len_ctx_para_tokens"] = 0.0
        if context_ocr_hash_local is not None and context_ocr_hash_local.numel() > 0:
            stat["hash_ctx_ocr"] = float(int(context_ocr_hash_local.view(-1)[0].item()))
        if context_adesc_hash_local is not None and context_adesc_hash_local.numel() > 0:
            stat["hash_ctx_adesc"] = float(int(context_adesc_hash_local.view(-1)[0].item()))
        return stat

    def _assert_explain_attn_consistency(
        attn_stat_a: Optional[Dict[str, float]],
        attentions: Optional[List[torch.Tensor]],
        input_ids_local: torch.Tensor,
        labels_mm_local: torch.Tensor,
        regions_local: Optional[List[List[Dict]]],
        bias_ranges_local: Optional[List],
    ) -> None:
        if not args.debug_explain_attn_consistency:
            return
        if attn_stat_a is None:
            return
        attn_stat_b = _collect_attn_mass_from_forward(
            attentions=attentions,
            input_ids_local=input_ids_local,
            labels_mm_local=labels_mm_local,
            regions_local=regions_local,
            bias_ranges_local=bias_ranges_local,
        )
        if attn_stat_b is None:
            return
        diff = max(
            abs(float(attn_stat_a.get("attn_to_region", 0.0)) - float(attn_stat_b.get("attn_to_region", 0.0))),
            abs(float(attn_stat_a.get("attn_to_img", 0.0)) - float(attn_stat_b.get("attn_to_img", 0.0))),
            abs(float(attn_stat_a.get("attn_to_textctx", 0.0)) - float(attn_stat_b.get("attn_to_textctx", 0.0))),
        )
        if diff <= 1e-3:
            return
        llm_cfg = getattr(model.language_model, "config", None)
        attn_impl = (
            getattr(llm_cfg, "_attn_implementation", None)
            or getattr(llm_cfg, "attn_implementation", None)
            or "unknown"
        )
        print(
            "[warn][attn-consistency] mismatch "
            f"diff={diff:.6f} q_len={attn_stat_a.get('q_len')} "
            f"region_k_len={attn_stat_a.get('region_k_len')} textctx_k_len={attn_stat_a.get('textctx_k_len')} "
            f"visible_k_nonzero={attn_stat_a.get('visible_k_nonzero')} "
            f"attn_impl={attn_impl} use_cache=False"
        )
        raise AssertionError("explain attention metric mismatch between val/sample paths")

    def _compose_explain_context(ocr_seg: str, adesc_seg: str, para_seg: str, keep_ocr: bool, keep_adesc: bool) -> Tuple[str, str]:
        ocr_text = ocr_seg if keep_ocr else ""
        adesc_text = adesc_seg if keep_adesc else ""
        allowed = _flatten_text([ocr_text, adesc_text])
        full = _flatten_text([allowed, para_seg])
        return full, allowed

    def _build_explain_variant_batch(
        images_local,
        texts_local: List[str],
        regions_local: List[List[Dict]],
        ocr_segs: List[str],
        adesc_segs: List[str],
        para_segs: List[str],
        keep_ocr: bool,
        keep_adesc: bool,
    ) -> Dict[str, torch.Tensor]:
        records = []
        n = len(images_local)
        for i in range(n):
            txt_i = texts_local[i] if i < len(texts_local) else ""
            regs_i = regions_local[i] if i < len(regions_local) else []
            ocr_i = ocr_segs[i] if i < len(ocr_segs) else ""
            adesc_i = adesc_segs[i] if i < len(adesc_segs) else ""
            para_i = para_segs[i] if i < len(para_segs) else ""
            full_ctx, allowed_ctx = _compose_explain_context(ocr_i, adesc_i, para_i, keep_ocr=keep_ocr, keep_adesc=keep_adesc)
            records.append(
                (
                    images_local[i],
                    txt_i,
                    "explain",
                    full_ctx,
                    regs_i,
                    {
                        "allowed_text": allowed_ctx,
                        "allowed_ocr_text": ocr_i if keep_ocr else "",
                        "allowed_desc_text": adesc_i if keep_adesc else "",
                        "forbidden_para_text": para_i,
                    },
                )
            )
        return build_batch(
            tokenizer,
            records,
            args.max_length,
            {
                "short": args.max_target_short,
                "long": args.max_target_long,
                "desc": args.max_target_desc,
                "explain": args.max_target_long,
            },
            model.config.image_token_index,
            add_eos=True,
            fixed_task="explain",
            context_dropout=0.0,
            paragraph_token_dropout=0.0,
            max_ctx_tokens=args.max_ctx_tokens,
            max_ctx_tokens_explain=args.max_ctx_tokens_explain,
            explain_ctx_min_adesc_tokens=args.explain_ctx_min_adesc_tokens,
            explain_ctx_max_ocr_tokens=args.explain_ctx_max_ocr_tokens,
            bucket_bins=None,
            image_tokens=image_tokens_total,
            scicap_prompt_style=args.scicap_prompt_style,
            scicap_task_context_routing=args.scicap_task_context_routing,
            enable_task_style_tokens=args.enable_task_style_tokens,
            use_context_placeholders=args.use_context_placeholders,
        )

    def run_validation(cur_step: int) -> None:
        if (
            val_dl is None
            and val_scicap_dl is None
            and val_explain_dl is None
        ):
            return
        _update_runtime_schedule(int(cur_step), force_log=False)
        model.eval()
        student.eval()
        with torch.no_grad():
            if args.dataset == "stage4_multitask":
                # Shared cap for expensive attention exports in stage4 validation.
                max_attn_batches = 4 if args.val_num_batches <= 0 else max(1, min(4, args.val_num_batches))
                if val_scicap_dl is not None:
                    losses_scicap: List[float] = []
                    by_scale: Dict[str, List[float]] = {"short": [], "long": [], "desc": []}
                    for bidx, vbatch in enumerate(val_scicap_dl):
                        images_v = vbatch["images"]
                        pixel_values_v = student.preprocess(list(images_v))["pixel_values"].to(device)
                        input_ids_v = vbatch["input_ids"].to(device)
                        labels_v = vbatch["labels"].to(device)
                        attention_mask_v = vbatch["attention_mask"].to(device)
                        regions_v = vbatch.get("regions", None)
                        scales_v = list(vbatch.get("scales", []))
                        vision_wrapper.set_regions(regions_v, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids_v,
                            scales_v,
                            regions_v,
                            labels_local=labels_v,
                            contexts_local=vbatch.get("contexts_used", vbatch.get("contexts", None)),
                            context_allow_tokens_local=vbatch.get("context_allow_tokens", None),
                        )
                        if bidx < max_attn_batches:
                            _ensure_log_attn_exportable()
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_v, attn_mask_v, past_kv_v, inputs_embeds_v, labels_mm_v = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids_v,
                                position_ids=None,
                                attention_mask=attention_mask_v,
                                past_key_values=None,
                                labels=labels_v,
                                images=pixel_values_v,
                                image_sizes=None,
                            )
                            out_v = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_v,
                                position_ids=pos_ids_v,
                                past_key_values=past_kv_v,
                                inputs_embeds=inputs_embeds_v,
                                labels=labels_mm_v,
                                use_cache=False,
                            )
                            nll_ps = nll_per_sample(out_v.logits, labels_mm_v)
                            loss_v = nll_ps.mean()
                        _clear_region_attention_bias()
                        losses_scicap.append(float(loss_v.detach().item()))
                        for i, sv in enumerate(scales_v):
                            if sv in by_scale and i < nll_ps.shape[0]:
                                by_scale[sv].append(float(nll_ps[i].detach().item()))
                        del out_v, inputs_embeds_v, labels_mm_v, attn_mask_v, pos_ids_v, past_kv_v
                        torch.cuda.empty_cache()
                        if args.val_num_batches > 0 and (bidx + 1) >= args.val_num_batches:
                            break
                    if losses_scicap:
                        avg_scicap = sum(losses_scicap) / float(len(losses_scicap))
                        scale_stats = {
                            k: (sum(v) / float(len(v)) if v else None)
                            for k, v in by_scale.items()
                        }
                        print(
                            f"[val_scicap] step={cur_step} loss={avg_scicap:.4f} "
                            f"short={scale_stats['short']} long={scale_stats['long']} desc={scale_stats['desc']} "
                            f"batches={len(losses_scicap)}"
                        )
                        with open(out_dir / "val_log.jsonl", "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "step": cur_step,
                                        "split": "scicap",
                                        "val_loss": avg_scicap,
                                        "short": scale_stats["short"],
                                        "long": scale_stats["long"],
                                        "desc": scale_stats["desc"],
                                        "batches": len(losses_scicap),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                if val_explain_dl is not None:
                    losses_base: List[float] = []
                    losses_region_drop: List[float] = []
                    losses_drop_ocr: List[float] = []
                    losses_drop_adesc: List[float] = []
                    losses_ctx_shuffle: List[float] = []
                    attn_region: List[float] = []
                    attn_img: List[float] = []
                    attn_ctx: List[float] = []
                    attn_batches = 0
                    for bidx, vbatch in enumerate(val_explain_dl):
                        images_v = vbatch["images"]
                        pixel_values_v = student.preprocess(list(images_v))["pixel_values"].to(device)
                        input_ids_v = vbatch["input_ids"].to(device)
                        labels_v = vbatch["labels"].to(device)
                        attention_mask_v = vbatch["attention_mask"].to(device)
                        regions_v = list(vbatch.get("regions", []))
                        scales_v = list(vbatch.get("scales", []))
                        texts_v = list(vbatch.get("texts", []))
                        ocr_segs_v = list(vbatch.get("contexts_ocr", []))
                        adesc_segs_v = list(vbatch.get("contexts_adesc", []))
                        para_segs_v = list(vbatch.get("contexts_para", []))
                        # base: image + region + paragraph
                        vision_wrapper.set_regions(regions_v, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids_v,
                            scales_v,
                            regions_v,
                            labels_local=labels_v,
                            contexts_local=vbatch.get("contexts_used", vbatch.get("contexts", None)),
                            context_allow_tokens_local=vbatch.get("context_allow_tokens", None),
                        )
                        want_attn = bool(args.log_attn and (bidx < max_attn_batches))
                        if want_attn:
                            _ensure_log_attn_exportable()
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_b, attn_mask_b, past_kv_b, inputs_embeds_b, labels_mm_b = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids_v,
                                position_ids=None,
                                attention_mask=attention_mask_v,
                                past_key_values=None,
                                labels=labels_v,
                                images=pixel_values_v,
                                image_sizes=None,
                            )
                            out_b = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_b,
                                position_ids=pos_ids_b,
                                past_key_values=past_kv_b,
                                inputs_embeds=inputs_embeds_b,
                                labels=labels_mm_b,
                                use_cache=False,
                                output_attentions=want_attn,
                            )
                            nll_base = nll_per_sample(out_b.logits, labels_mm_b).mean()
                        bias_ranges_b = model._region_attn_bias_ranges
                        losses_base.append(float(nll_base.detach().item()))
                        if want_attn:
                            attn_stat = _collect_attn_mass_from_forward(
                                out_b.attentions,
                                input_ids_v,
                                labels_mm_b,
                                regions_v,
                                bias_ranges_local=bias_ranges_b,
                            )
                            if attn_stat is not None:
                                _assert_explain_attn_consistency(
                                    attn_stat,
                                    out_b.attentions,
                                    input_ids_v,
                                    labels_mm_b,
                                    regions_v,
                                    bias_ranges_b,
                                )
                                attn_region.append(float(attn_stat["attn_to_region"]))
                                attn_img.append(float(attn_stat["attn_to_img"]))
                                attn_ctx.append(float(attn_stat["attn_to_textctx"]))
                                attn_batches += 1
                        _clear_region_attention_bias()
                        # region-drop: image + paragraph only
                        regions_drop = [[] for _ in regions_v]
                        vision_wrapper.set_regions(regions_drop, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids_v,
                            scales_v,
                            regions_drop,
                            labels_local=labels_v,
                            contexts_local=vbatch.get("contexts_used", vbatch.get("contexts", None)),
                            context_allow_tokens_local=vbatch.get("context_allow_tokens", None),
                        )
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_d, attn_mask_d, past_kv_d, inputs_embeds_d, labels_mm_d = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids_v,
                                position_ids=None,
                                attention_mask=attention_mask_v,
                                past_key_values=None,
                                labels=labels_v,
                                images=pixel_values_v,
                                image_sizes=None,
                            )
                            out_d = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_d,
                                position_ids=pos_ids_d,
                                past_key_values=past_kv_d,
                                inputs_embeds=inputs_embeds_d,
                                labels=labels_mm_d,
                                use_cache=False,
                            )
                            nll_drop = nll_per_sample(out_d.logits, labels_mm_d).mean()
                        _clear_region_attention_bias()
                        losses_region_drop.append(float(nll_drop.detach().item()))
                        # drop-ocr: image(+region) + ADESC only (paragraph still forbidden under gate)
                        if any((str(x or "").strip() for x in ocr_segs_v)):
                            drop_ocr_batch = _build_explain_variant_batch(
                                images_local=images_v,
                                texts_local=texts_v,
                                regions_local=regions_v,
                                ocr_segs=ocr_segs_v,
                                adesc_segs=adesc_segs_v,
                                para_segs=para_segs_v,
                                keep_ocr=False,
                                keep_adesc=True,
                            )
                            input_ids_ocr = drop_ocr_batch["input_ids"].to(device)
                            labels_ocr = drop_ocr_batch["labels"].to(device)
                            attention_mask_ocr = drop_ocr_batch["attention_mask"].to(device)
                            vision_wrapper.set_regions(regions_v, drop_one_region=False)
                            _set_region_attention_bias_for_inputs(
                                input_ids_ocr,
                                ["explain"] * len(regions_v),
                                regions_v,
                                labels_local=labels_ocr,
                                contexts_local=drop_ocr_batch.get("contexts_used", drop_ocr_batch.get("contexts", None)),
                                context_allow_tokens_local=drop_ocr_batch.get("context_allow_tokens", None),
                            )
                            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                                _, pos_ids_ocr, attn_mask_ocr, past_kv_ocr, inputs_embeds_ocr, labels_mm_ocr = model.prepare_inputs_labels_for_multimodal(
                                    input_ids=input_ids_ocr,
                                    position_ids=None,
                                    attention_mask=attention_mask_ocr,
                                    past_key_values=None,
                                    labels=labels_ocr,
                                    images=pixel_values_v,
                                    image_sizes=None,
                                )
                                out_ocr = model.language_model(
                                    input_ids=None,
                                    attention_mask=attn_mask_ocr,
                                    position_ids=pos_ids_ocr,
                                    past_key_values=past_kv_ocr,
                                    inputs_embeds=inputs_embeds_ocr,
                                    labels=labels_mm_ocr,
                                    use_cache=False,
                                )
                                nll_drop_ocr = nll_per_sample(out_ocr.logits, labels_mm_ocr).mean()
                            _clear_region_attention_bias()
                            losses_drop_ocr.append(float(nll_drop_ocr.detach().item()))
                            del out_ocr, inputs_embeds_ocr, labels_mm_ocr, attn_mask_ocr, pos_ids_ocr, past_kv_ocr
                        else:
                            losses_drop_ocr.append(float(nll_base.detach().item()))
                        # drop-adesc: image(+region) + OCR only
                        if any((str(x or "").strip() for x in adesc_segs_v)):
                            drop_adesc_batch = _build_explain_variant_batch(
                                images_local=images_v,
                                texts_local=texts_v,
                                regions_local=regions_v,
                                ocr_segs=ocr_segs_v,
                                adesc_segs=adesc_segs_v,
                                para_segs=para_segs_v,
                                keep_ocr=True,
                                keep_adesc=False,
                            )
                            input_ids_ad = drop_adesc_batch["input_ids"].to(device)
                            labels_ad = drop_adesc_batch["labels"].to(device)
                            attention_mask_ad = drop_adesc_batch["attention_mask"].to(device)
                            vision_wrapper.set_regions(regions_v, drop_one_region=False)
                            _set_region_attention_bias_for_inputs(
                                input_ids_ad,
                                ["explain"] * len(regions_v),
                                regions_v,
                                labels_local=labels_ad,
                                contexts_local=drop_adesc_batch.get("contexts_used", drop_adesc_batch.get("contexts", None)),
                                context_allow_tokens_local=drop_adesc_batch.get("context_allow_tokens", None),
                            )
                            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                                _, pos_ids_ad, attn_mask_ad, past_kv_ad, inputs_embeds_ad, labels_mm_ad = model.prepare_inputs_labels_for_multimodal(
                                    input_ids=input_ids_ad,
                                    position_ids=None,
                                    attention_mask=attention_mask_ad,
                                    past_key_values=None,
                                    labels=labels_ad,
                                    images=pixel_values_v,
                                    image_sizes=None,
                                )
                                out_ad = model.language_model(
                                    input_ids=None,
                                    attention_mask=attn_mask_ad,
                                    position_ids=pos_ids_ad,
                                    past_key_values=past_kv_ad,
                                    inputs_embeds=inputs_embeds_ad,
                                    labels=labels_mm_ad,
                                    use_cache=False,
                                )
                                nll_drop_adesc = nll_per_sample(out_ad.logits, labels_mm_ad).mean()
                            _clear_region_attention_bias()
                            losses_drop_adesc.append(float(nll_drop_adesc.detach().item()))
                            del out_ad, inputs_embeds_ad, labels_mm_ad, attn_mask_ad, pos_ids_ad, past_kv_ad
                        else:
                            losses_drop_adesc.append(float(nll_base.detach().item()))
                        # paragraph-shuffle: image + region + shuffled paragraph
                        contexts_v = list(vbatch.get("contexts", []))
                        if len(contexts_v) > 1 and any((str(c or "").strip() for c in contexts_v)):
                            perm = list(range(len(contexts_v)))
                            random.shuffle(perm)
                            if perm == list(range(len(contexts_v))):
                                perm = perm[1:] + perm[:1]
                            shuf_ctx = [contexts_v[p] for p in perm]
                            records_shuf = []
                            for i in range(len(images_v)):
                                records_shuf.append((images_v[i], texts_v[i], "explain", shuf_ctx[i], regions_v[i]))
                            shuf_batch = build_batch(
                                tokenizer,
                                records_shuf,
                                args.max_length,
                                {
                                    "short": args.max_target_short,
                                    "long": args.max_target_long,
                                    "desc": args.max_target_desc,
                                    "explain": args.max_target_long,
                                },
                                model.config.image_token_index,
                                add_eos=True,
                                fixed_task="explain",
                                context_dropout=0.0,
                                paragraph_token_dropout=0.0,
                                max_ctx_tokens=args.max_ctx_tokens,

                                max_ctx_tokens_explain=args.max_ctx_tokens_explain,
                                explain_ctx_min_adesc_tokens=args.explain_ctx_min_adesc_tokens,
                                explain_ctx_max_ocr_tokens=args.explain_ctx_max_ocr_tokens,
                                bucket_bins=None,
                                image_tokens=image_tokens_total,
                                scicap_prompt_style=args.scicap_prompt_style,
                                scicap_task_context_routing=args.scicap_task_context_routing,
                                enable_task_style_tokens=args.enable_task_style_tokens,
                                use_context_placeholders=args.use_context_placeholders,
                            )
                            input_ids_s = shuf_batch["input_ids"].to(device)
                            labels_s = shuf_batch["labels"].to(device)
                            attention_mask_s = shuf_batch["attention_mask"].to(device)
                            vision_wrapper.set_regions(regions_v, drop_one_region=False)
                            _set_region_attention_bias_for_inputs(
                                input_ids_s,
                                ["explain"] * len(records_shuf),
                                regions_v,
                                labels_local=labels_s,
                                contexts_local=shuf_batch.get("contexts_used", shuf_batch.get("contexts", None)),
                                context_allow_tokens_local=shuf_batch.get("context_allow_tokens", None),
                            )
                            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                                _, pos_ids_s, attn_mask_s, past_kv_s, inputs_embeds_s, labels_mm_s = model.prepare_inputs_labels_for_multimodal(
                                    input_ids=input_ids_s,
                                    position_ids=None,
                                    attention_mask=attention_mask_s,
                                    past_key_values=None,
                                    labels=labels_s,
                                    images=pixel_values_v,
                                    image_sizes=None,
                                )
                                out_s = model.language_model(
                                    input_ids=None,
                                    attention_mask=attn_mask_s,
                                    position_ids=pos_ids_s,
                                    past_key_values=past_kv_s,
                                    inputs_embeds=inputs_embeds_s,
                                    labels=labels_mm_s,
                                    use_cache=False,
                                )
                                nll_shuffle = nll_per_sample(out_s.logits, labels_mm_s).mean()
                            _clear_region_attention_bias()
                            losses_ctx_shuffle.append(float(nll_shuffle.detach().item()))
                            del out_s, inputs_embeds_s, labels_mm_s, attn_mask_s, pos_ids_s, past_kv_s
                        else:
                            losses_ctx_shuffle.append(float(nll_base.detach().item()))
                        del out_b, out_d, inputs_embeds_b, labels_mm_b, attn_mask_b, pos_ids_b, past_kv_b
                        del inputs_embeds_d, labels_mm_d, attn_mask_d, pos_ids_d, past_kv_d
                        torch.cuda.empty_cache()
                        if args.val_num_batches > 0 and (bidx + 1) >= args.val_num_batches:
                            break
                    if losses_base:
                        mean_base = sum(losses_base) / float(len(losses_base))
                        mean_drop = sum(losses_region_drop) / float(len(losses_region_drop)) if losses_region_drop else mean_base
                        mean_drop_ocr = sum(losses_drop_ocr) / float(len(losses_drop_ocr)) if losses_drop_ocr else mean_base
                        mean_drop_adesc = sum(losses_drop_adesc) / float(len(losses_drop_adesc)) if losses_drop_adesc else mean_base
                        mean_shuffle = sum(losses_ctx_shuffle) / float(len(losses_ctx_shuffle)) if losses_ctx_shuffle else mean_base
                        diag = {
                            "attn_to_region": (sum(attn_region) / float(len(attn_region))) if attn_region else 0.0,
                            "attn_to_img": (sum(attn_img) / float(len(attn_img))) if attn_img else 0.0,
                            "attn_to_textctx": (sum(attn_ctx) / float(len(attn_ctx))) if attn_ctx else 0.0,
                            "attn_batches": attn_batches,
                        }
                        print(
                            f"[val_explain] step={cur_step} base={mean_base:.4f} "
                            f"region_drop={mean_drop:.4f} ocr_drop={mean_drop_ocr:.4f} adesc_drop={mean_drop_adesc:.4f} "
                            f"paragraph_shuffle={mean_shuffle:.4f} "
                            f"gap_drop={mean_drop - mean_base:.4f} gap_ocr={mean_drop_ocr - mean_base:.4f} "
                            f"gap_adesc={mean_drop_adesc - mean_base:.4f} gap_shuffle={mean_shuffle - mean_base:.4f} "
                            f"attn_region={diag['attn_to_region']:.4f} attn_img={diag['attn_to_img']:.4f} attn_textctx={diag['attn_to_textctx']:.4f}"
                        )
                        with open(out_dir / "val_log.jsonl", "a", encoding="utf-8") as f:
                            record_full = {
                                "step": cur_step,
                                "split": "scistruct_explain",
                                "loss_base": mean_base,
                                "loss_region_drop": mean_drop,
                                "loss_drop_ocr": mean_drop_ocr,
                                "loss_drop_adesc": mean_drop_adesc,
                                "loss_paragraph_shuffle": mean_shuffle,
                                "loss_context_shuffle": mean_shuffle,
                                "gap_region_drop": mean_drop - mean_base,
                                "gap_drop_ocr": mean_drop_ocr - mean_base,
                                "gap_drop_adesc": mean_drop_adesc - mean_base,
                                "gap_paragraph_shuffle": mean_shuffle - mean_base,
                                "gap_context_shuffle": mean_shuffle - mean_base,
                                "gap_shuffle_ocr": mean_shuffle - mean_base,
                                "attn_to_region": diag["attn_to_region"],
                                "attn_to_img": diag["attn_to_img"],
                                "attn_to_textctx": diag["attn_to_textctx"],
                                "attn_to_textctx_forbidden": diag["attn_to_textctx"],
                                "attn_batches": diag["attn_batches"],
                                "batches": len(losses_base),
                            }
                            if args.explain_metrics_minimal_only:
                                record = {
                                    "step": cur_step,
                                    "split": "scistruct_explain",
                                    "gap_drop_ocr": record_full["gap_drop_ocr"],
                                    "gap_drop_adesc": record_full["gap_drop_adesc"],
                                    "gap_region_drop": record_full["gap_region_drop"],
                                    "gap_shuffle_ocr": record_full["gap_shuffle_ocr"],
                                    "attn_to_textctx_forbidden": record_full["attn_to_textctx_forbidden"],
                                }
                            else:
                                record = record_full
                            f.write(
                                json.dumps(record, ensure_ascii=False)
                                + "\n"
                            )
            elif args.dataset == "scistruct_explain":
                losses_base: List[float] = []
                losses_region_drop: List[float] = []
                losses_drop_ocr: List[float] = []
                losses_drop_adesc: List[float] = []
                losses_shuffle_ocr: List[float] = []
                losses_ctx_shuffle: List[float] = []
                gaps_ocr: List[float] = []
                gaps_adesc: List[float] = []
                gaps_shuffle_ocr: List[float] = []
                gaps_adesc_valid: List[float] = []
                attn_region: List[float] = []
                attn_img: List[float] = []
                attn_ctx: List[float] = []
                attn_ocr: List[float] = []
                attn_adesc: List[float] = []
                base_stats: List[Dict[str, float]] = []
                drop_ocr_stats: List[Dict[str, float]] = []
                drop_adesc_stats: List[Dict[str, float]] = []
                val_sample_count = 0
                ocr_nonempty_count = 0
                adesc_nonempty_count = 0
                ocr_tokens_sum = 0.0
                adesc_tokens_sum = 0.0
                attn_batches = 0
                max_attn_batches = 4 if args.val_num_batches <= 0 else max(1, min(4, args.val_num_batches))
                for bidx, vbatch in enumerate(val_dl):
                    images_v = vbatch["images"]
                    pixel_values_v = student.preprocess(list(images_v))["pixel_values"].to(device)
                    input_ids_v = vbatch["input_ids"].to(device)
                    labels_v = vbatch["labels"].to(device)
                    attention_mask_v = vbatch["attention_mask"].to(device)
                    regions_v = list(vbatch.get("regions", []))
                    scales_v = list(vbatch.get("scales", []))
                    texts_v = list(vbatch.get("texts", []))
                    ocr_segs_v = list(vbatch.get("contexts_ocr", []))
                    adesc_segs_v = list(vbatch.get("contexts_adesc", []))
                    para_segs_v = list(vbatch.get("contexts_para", []))
                    ctx_ocr_tok = vbatch.get("context_ocr_tokens", None)
                    ctx_adesc_tok = vbatch.get("context_adesc_tokens", None)
                    bsz_now = int(input_ids_v.size(0))
                    val_sample_count += bsz_now
                    adesc_nonempty_batch = False
                    if isinstance(ctx_ocr_tok, torch.Tensor):
                        ocr_nonempty_count += int((ctx_ocr_tok > 0).sum().item())
                        ocr_tokens_sum += float(ctx_ocr_tok.float().sum().item())
                    else:
                        ocr_nonempty_count += int(sum(1 for s in ocr_segs_v if str(s or "").strip()))
                    if isinstance(ctx_adesc_tok, torch.Tensor):
                        adesc_nonempty_count += int((ctx_adesc_tok > 0).sum().item())
                        adesc_tokens_sum += float(ctx_adesc_tok.float().sum().item())
                        adesc_nonempty_batch = bool((ctx_adesc_tok > 0).any().item())
                    else:
                        adesc_nonempty_batch = any((str(x or "").strip() for x in adesc_segs_v))
                        adesc_nonempty_count += int(sum(1 for s in adesc_segs_v if str(s or "").strip()))
                    # base: image + region + paragraph
                    vision_wrapper.set_regions(regions_v, drop_one_region=False)
                    _set_region_attention_bias_for_inputs(
                        input_ids_v,
                        scales_v,
                        regions_v,
                        labels_local=labels_v,
                        contexts_local=vbatch.get("contexts_used", vbatch.get("contexts", None)),
                        context_allow_tokens_local=vbatch.get("context_allow_tokens", None),
                        context_total_tokens_local=vbatch.get("context_total_tokens", None),
                        context_ocr_tokens_local=vbatch.get("context_ocr_tokens", None),
                        context_adesc_tokens_local=vbatch.get("context_adesc_tokens", None),
                        context_para_tokens_local=vbatch.get("context_para_tokens", None),
                    )
                    want_attn = bool(args.log_attn and (bidx < max_attn_batches))
                    if want_attn:
                        _ensure_log_attn_exportable()
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_b, attn_mask_b, past_kv_b, inputs_embeds_b, labels_mm_b = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids_v,
                            position_ids=None,
                            attention_mask=attention_mask_v,
                            past_key_values=None,
                            labels=labels_v,
                            images=pixel_values_v,
                            image_sizes=None,
                        )
                        out_b = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_b,
                            position_ids=pos_ids_b,
                            past_key_values=past_kv_b,
                            inputs_embeds=inputs_embeds_b,
                            labels=labels_mm_b,
                            use_cache=False,
                            output_attentions=want_attn,
                        )
                        nll_base = nll_per_sample(out_b.logits, labels_mm_b).mean()
                    bias_ranges_b = model._region_attn_bias_ranges
                    losses_base.append(float(nll_base.detach().item()))
                    if want_attn:
                        attn_stat = _collect_attn_mass_from_forward(
                            out_b.attentions,
                            input_ids_v,
                            labels_mm_b,
                            regions_v,
                            bias_ranges_local=bias_ranges_b,
                            context_allow_tokens_local=vbatch.get("context_allow_tokens", None),
                            context_total_tokens_local=vbatch.get("context_total_tokens", None),
                            context_ocr_tokens_local=vbatch.get("context_ocr_tokens", None),
                            context_adesc_tokens_local=vbatch.get("context_adesc_tokens", None),
                            context_para_tokens_local=vbatch.get("context_para_tokens", None),
                            context_ocr_hash_local=vbatch.get("context_ocr_hash", None),
                            context_adesc_hash_local=vbatch.get("context_adesc_hash", None),
                        )
                        if attn_stat is not None:
                            _assert_explain_attn_consistency(
                                attn_stat,
                                out_b.attentions,
                                input_ids_v,
                                labels_mm_b,
                                regions_v,
                                bias_ranges_b,
                            )
                            attn_region.append(float(attn_stat["attn_to_region"]))
                            attn_img.append(float(attn_stat["attn_to_img"]))
                            attn_ctx.append(float(attn_stat["attn_to_textctx"]))
                            attn_ocr.append(float(attn_stat.get("attn_to_ocr", 0.0)))
                            attn_adesc.append(float(attn_stat.get("attn_to_adesc", 0.0)))
                            base_stats.append(attn_stat)
                            attn_batches += 1
                    _clear_region_attention_bias()
                    # region-drop: image + paragraph only
                    regions_drop = [[] for _ in regions_v]
                    vision_wrapper.set_regions(regions_drop, drop_one_region=False)
                    _set_region_attention_bias_for_inputs(
                        input_ids_v,
                        scales_v,
                        regions_drop,
                        labels_local=labels_v,
                        contexts_local=vbatch.get("contexts_used", vbatch.get("contexts", None)),
                        context_allow_tokens_local=vbatch.get("context_allow_tokens", None),
                        context_total_tokens_local=vbatch.get("context_total_tokens", None),
                        context_ocr_tokens_local=vbatch.get("context_ocr_tokens", None),
                        context_adesc_tokens_local=vbatch.get("context_adesc_tokens", None),
                        context_para_tokens_local=vbatch.get("context_para_tokens", None),
                    )
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_d, attn_mask_d, past_kv_d, inputs_embeds_d, labels_mm_d = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids_v,
                            position_ids=None,
                            attention_mask=attention_mask_v,
                            past_key_values=None,
                            labels=labels_v,
                            images=pixel_values_v,
                            image_sizes=None,
                        )
                        out_d = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_d,
                            position_ids=pos_ids_d,
                            past_key_values=past_kv_d,
                            inputs_embeds=inputs_embeds_d,
                            labels=labels_mm_d,
                            use_cache=False,
                        )
                        nll_drop = nll_per_sample(out_d.logits, labels_mm_d).mean()
                    _clear_region_attention_bias()
                    losses_region_drop.append(float(nll_drop.detach().item()))
                    # drop-ocr: image(+region) + ADESC only (paragraph still forbidden under gate)
                    if any((str(x or "").strip() for x in ocr_segs_v)):
                        drop_ocr_batch = _build_explain_variant_batch(
                            images_local=images_v,
                            texts_local=texts_v,
                            regions_local=regions_v,
                            ocr_segs=ocr_segs_v,
                            adesc_segs=adesc_segs_v,
                            para_segs=para_segs_v,
                            keep_ocr=False,
                            keep_adesc=True,
                        )
                        input_ids_ocr = drop_ocr_batch["input_ids"].to(device)
                        labels_ocr = drop_ocr_batch["labels"].to(device)
                        attention_mask_ocr = drop_ocr_batch["attention_mask"].to(device)
                        vision_wrapper.set_regions(regions_v, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids_ocr,
                            ["explain"] * len(regions_v),
                            regions_v,
                            labels_local=labels_ocr,
                            contexts_local=drop_ocr_batch.get("contexts_used", drop_ocr_batch.get("contexts", None)),
                            context_allow_tokens_local=drop_ocr_batch.get("context_allow_tokens", None),
                            context_total_tokens_local=drop_ocr_batch.get("context_total_tokens", None),
                            context_ocr_tokens_local=drop_ocr_batch.get("context_ocr_tokens", None),
                            context_adesc_tokens_local=drop_ocr_batch.get("context_adesc_tokens", None),
                            context_para_tokens_local=drop_ocr_batch.get("context_para_tokens", None),
                        )
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_ocr, attn_mask_ocr, past_kv_ocr, inputs_embeds_ocr, labels_mm_ocr = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids_ocr,
                                position_ids=None,
                                attention_mask=attention_mask_ocr,
                                past_key_values=None,
                                labels=labels_ocr,
                                images=pixel_values_v,
                                image_sizes=None,
                            )
                            out_ocr = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_ocr,
                                position_ids=pos_ids_ocr,
                                past_key_values=past_kv_ocr,
                                inputs_embeds=inputs_embeds_ocr,
                                labels=labels_mm_ocr,
                                use_cache=False,
                                output_attentions=want_attn,
                            )
                            nll_drop_ocr = nll_per_sample(out_ocr.logits, labels_mm_ocr).mean()
                        bias_ranges_ocr = model._region_attn_bias_ranges
                        if want_attn:
                            attn_stat_drop_ocr = _collect_attn_mass_from_forward(
                                out_ocr.attentions,
                                input_ids_ocr,
                                labels_mm_ocr,
                                regions_v,
                                bias_ranges_local=bias_ranges_ocr,
                                context_allow_tokens_local=drop_ocr_batch.get("context_allow_tokens", None),
                                context_total_tokens_local=drop_ocr_batch.get("context_total_tokens", None),
                                context_ocr_tokens_local=drop_ocr_batch.get("context_ocr_tokens", None),
                                context_adesc_tokens_local=drop_ocr_batch.get("context_adesc_tokens", None),
                                context_para_tokens_local=drop_ocr_batch.get("context_para_tokens", None),
                                context_ocr_hash_local=drop_ocr_batch.get("context_ocr_hash", None),
                                context_adesc_hash_local=drop_ocr_batch.get("context_adesc_hash", None),
                            )
                            if attn_stat_drop_ocr is not None:
                                drop_ocr_stats.append(attn_stat_drop_ocr)
                        _clear_region_attention_bias()
                        losses_drop_ocr.append(float(nll_drop_ocr.detach().item()))
                        del out_ocr, inputs_embeds_ocr, labels_mm_ocr, attn_mask_ocr, pos_ids_ocr, past_kv_ocr
                    else:
                        losses_drop_ocr.append(float(nll_base.detach().item()))
                    # drop-adesc: image(+region) + OCR only
                    if any((str(x or "").strip() for x in adesc_segs_v)):
                        drop_adesc_batch = _build_explain_variant_batch(
                            images_local=images_v,
                            texts_local=texts_v,
                            regions_local=regions_v,
                            ocr_segs=ocr_segs_v,
                            adesc_segs=adesc_segs_v,
                            para_segs=para_segs_v,
                            keep_ocr=True,
                            keep_adesc=False,
                        )
                        input_ids_ad = drop_adesc_batch["input_ids"].to(device)
                        labels_ad = drop_adesc_batch["labels"].to(device)
                        attention_mask_ad = drop_adesc_batch["attention_mask"].to(device)
                        vision_wrapper.set_regions(regions_v, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids_ad,
                            ["explain"] * len(regions_v),
                            regions_v,
                            labels_local=labels_ad,
                            contexts_local=drop_adesc_batch.get("contexts_used", drop_adesc_batch.get("contexts", None)),
                            context_allow_tokens_local=drop_adesc_batch.get("context_allow_tokens", None),
                            context_total_tokens_local=drop_adesc_batch.get("context_total_tokens", None),
                            context_ocr_tokens_local=drop_adesc_batch.get("context_ocr_tokens", None),
                            context_adesc_tokens_local=drop_adesc_batch.get("context_adesc_tokens", None),
                            context_para_tokens_local=drop_adesc_batch.get("context_para_tokens", None),
                        )
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_ad, attn_mask_ad, past_kv_ad, inputs_embeds_ad, labels_mm_ad = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids_ad,
                                position_ids=None,
                                attention_mask=attention_mask_ad,
                                past_key_values=None,
                                labels=labels_ad,
                                images=pixel_values_v,
                                image_sizes=None,
                            )
                            out_ad = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_ad,
                                position_ids=pos_ids_ad,
                                past_key_values=past_kv_ad,
                                inputs_embeds=inputs_embeds_ad,
                                labels=labels_mm_ad,
                                use_cache=False,
                                output_attentions=want_attn,
                            )
                            nll_drop_adesc = nll_per_sample(out_ad.logits, labels_mm_ad).mean()
                        bias_ranges_ad = model._region_attn_bias_ranges
                        if want_attn:
                            attn_stat_drop_adesc = _collect_attn_mass_from_forward(
                                out_ad.attentions,
                                input_ids_ad,
                                labels_mm_ad,
                                regions_v,
                                bias_ranges_local=bias_ranges_ad,
                                context_allow_tokens_local=drop_adesc_batch.get("context_allow_tokens", None),
                                context_total_tokens_local=drop_adesc_batch.get("context_total_tokens", None),
                                context_ocr_tokens_local=drop_adesc_batch.get("context_ocr_tokens", None),
                                context_adesc_tokens_local=drop_adesc_batch.get("context_adesc_tokens", None),
                                context_para_tokens_local=drop_adesc_batch.get("context_para_tokens", None),
                                context_ocr_hash_local=drop_adesc_batch.get("context_ocr_hash", None),
                                context_adesc_hash_local=drop_adesc_batch.get("context_adesc_hash", None),
                            )
                            if attn_stat_drop_adesc is not None:
                                drop_adesc_stats.append(attn_stat_drop_adesc)
                        _clear_region_attention_bias()
                        losses_drop_adesc.append(float(nll_drop_adesc.detach().item()))
                        del out_ad, inputs_embeds_ad, labels_mm_ad, attn_mask_ad, pos_ids_ad, past_kv_ad
                    else:
                        losses_drop_adesc.append(float(nll_base.detach().item()))
                    gaps_ocr.append(float(losses_drop_ocr[-1] - losses_base[-1]))
                    gap_adesc_i = float(losses_drop_adesc[-1] - losses_base[-1])
                    gaps_adesc.append(gap_adesc_i)
                    if adesc_nonempty_batch:
                        gaps_adesc_valid.append(gap_adesc_i)
                    # shuffle-ocr: keep ADESC/paragraph fixed, shuffle OCR across samples
                    if len(ocr_segs_v) > 1 and any((str(x or "").strip() for x in ocr_segs_v)):
                        perm_ocr = list(range(len(ocr_segs_v)))
                        random.shuffle(perm_ocr)
                        if perm_ocr == list(range(len(ocr_segs_v))):
                            perm_ocr = perm_ocr[1:] + perm_ocr[:1]
                        ocr_shuf = [ocr_segs_v[p] for p in perm_ocr]
                        shuf_ocr_batch = _build_explain_variant_batch(
                            images_local=images_v,
                            texts_local=texts_v,
                            regions_local=regions_v,
                            ocr_segs=ocr_shuf,
                            adesc_segs=adesc_segs_v,
                            para_segs=para_segs_v,
                            keep_ocr=True,
                            keep_adesc=True,
                        )
                        input_ids_so = shuf_ocr_batch["input_ids"].to(device)
                        labels_so = shuf_ocr_batch["labels"].to(device)
                        attention_mask_so = shuf_ocr_batch["attention_mask"].to(device)
                        vision_wrapper.set_regions(regions_v, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids_so,
                            ["explain"] * len(regions_v),
                            regions_v,
                            labels_local=labels_so,
                            contexts_local=shuf_ocr_batch.get("contexts_used", shuf_ocr_batch.get("contexts", None)),
                            context_allow_tokens_local=shuf_ocr_batch.get("context_allow_tokens", None),
                            context_total_tokens_local=shuf_ocr_batch.get("context_total_tokens", None),
                            context_ocr_tokens_local=shuf_ocr_batch.get("context_ocr_tokens", None),
                            context_adesc_tokens_local=shuf_ocr_batch.get("context_adesc_tokens", None),
                            context_para_tokens_local=shuf_ocr_batch.get("context_para_tokens", None),
                        )
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_so, attn_mask_so, past_kv_so, inputs_embeds_so, labels_mm_so = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids_so,
                                position_ids=None,
                                attention_mask=attention_mask_so,
                                past_key_values=None,
                                labels=labels_so,
                                images=pixel_values_v,
                                image_sizes=None,
                            )
                            out_so = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_so,
                                position_ids=pos_ids_so,
                                past_key_values=past_kv_so,
                                inputs_embeds=inputs_embeds_so,
                                labels=labels_mm_so,
                                use_cache=False,
                            )
                            nll_shuffle_ocr = nll_per_sample(out_so.logits, labels_mm_so).mean()
                        _clear_region_attention_bias()
                        losses_shuffle_ocr.append(float(nll_shuffle_ocr.detach().item()))
                        del out_so, inputs_embeds_so, labels_mm_so, attn_mask_so, pos_ids_so, past_kv_so
                    else:
                        losses_shuffle_ocr.append(float(nll_base.detach().item()))
                    gaps_shuffle_ocr.append(float(losses_shuffle_ocr[-1] - losses_base[-1]))
                    # paragraph-shuffle: image + region + shuffled paragraph
                    contexts_v = list(vbatch.get("contexts", []))
                    if len(contexts_v) > 1 and any((str(c or "").strip() for c in contexts_v)):
                        perm = list(range(len(contexts_v)))
                        random.shuffle(perm)
                        if perm == list(range(len(contexts_v))):
                            perm = perm[1:] + perm[:1]
                        shuf_ctx = [contexts_v[p] for p in perm]
                        records_shuf = []
                        for i in range(len(images_v)):
                            records_shuf.append((images_v[i], texts_v[i], "explain", shuf_ctx[i], regions_v[i]))
                        shuf_batch = build_batch(
                            tokenizer,
                            records_shuf,
                            args.max_length,
                            {
                                "short": args.max_target_short,
                                "long": args.max_target_long,
                                "desc": args.max_target_desc,
                                "explain": args.max_target_long,
                            },
                            model.config.image_token_index,
                            add_eos=True,
                            fixed_task="explain",
                            context_dropout=0.0,
                            paragraph_token_dropout=0.0,
                            max_ctx_tokens=args.max_ctx_tokens,

                            max_ctx_tokens_explain=args.max_ctx_tokens_explain,
                            explain_ctx_min_adesc_tokens=args.explain_ctx_min_adesc_tokens,
                            explain_ctx_max_ocr_tokens=args.explain_ctx_max_ocr_tokens,
                            bucket_bins=None,
                            image_tokens=image_tokens_total,
                            scicap_prompt_style=args.scicap_prompt_style,
                            scicap_task_context_routing=args.scicap_task_context_routing,
                            enable_task_style_tokens=args.enable_task_style_tokens,
                            use_context_placeholders=args.use_context_placeholders,
                        )
                        input_ids_s = shuf_batch["input_ids"].to(device)
                        labels_s = shuf_batch["labels"].to(device)
                        attention_mask_s = shuf_batch["attention_mask"].to(device)
                        vision_wrapper.set_regions(regions_v, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids_s,
                            ["explain"] * len(records_shuf),
                            regions_v,
                            labels_local=labels_s,
                            contexts_local=shuf_batch.get("contexts_used", shuf_batch.get("contexts", None)),
                            context_allow_tokens_local=shuf_batch.get("context_allow_tokens", None),
                            context_total_tokens_local=shuf_batch.get("context_total_tokens", None),
                            context_ocr_tokens_local=shuf_batch.get("context_ocr_tokens", None),
                            context_adesc_tokens_local=shuf_batch.get("context_adesc_tokens", None),
                            context_para_tokens_local=shuf_batch.get("context_para_tokens", None),
                        )
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_s, attn_mask_s, past_kv_s, inputs_embeds_s, labels_mm_s = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids_s,
                                position_ids=None,
                                attention_mask=attention_mask_s,
                                past_key_values=None,
                                labels=labels_s,
                                images=pixel_values_v,
                                image_sizes=None,
                            )
                            out_s = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_s,
                                position_ids=pos_ids_s,
                                past_key_values=past_kv_s,
                                inputs_embeds=inputs_embeds_s,
                                labels=labels_mm_s,
                                use_cache=False,
                            )
                            nll_shuffle = nll_per_sample(out_s.logits, labels_mm_s).mean()
                        _clear_region_attention_bias()
                        losses_ctx_shuffle.append(float(nll_shuffle.detach().item()))
                        del out_s, inputs_embeds_s, labels_mm_s, attn_mask_s, pos_ids_s, past_kv_s
                    else:
                        losses_ctx_shuffle.append(float(nll_base.detach().item()))
                    del out_b, out_d, inputs_embeds_b, labels_mm_b, attn_mask_b, pos_ids_b, past_kv_b
                    del inputs_embeds_d, labels_mm_d, attn_mask_d, pos_ids_d, past_kv_d
                    torch.cuda.empty_cache()
                    if args.val_num_batches > 0 and (bidx + 1) >= args.val_num_batches:
                        break
                if losses_base:
                    def _mean_stat(stats: List[Dict[str, float]], key: str) -> float:
                        vals = [float(s.get(key, 0.0)) for s in stats if key in s]
                        return (sum(vals) / float(len(vals))) if vals else 0.0
                    mean_base = sum(losses_base) / float(len(losses_base))
                    mean_drop = sum(losses_region_drop) / float(len(losses_region_drop)) if losses_region_drop else mean_base
                    mean_drop_ocr = sum(losses_drop_ocr) / float(len(losses_drop_ocr)) if losses_drop_ocr else mean_base
                    mean_drop_adesc = sum(losses_drop_adesc) / float(len(losses_drop_adesc)) if losses_drop_adesc else mean_base
                    mean_shuffle_ocr = sum(losses_shuffle_ocr) / float(len(losses_shuffle_ocr)) if losses_shuffle_ocr else mean_base
                    mean_shuffle = sum(losses_ctx_shuffle) / float(len(losses_ctx_shuffle)) if losses_ctx_shuffle else mean_base
                    gap_drop_ocr_pos_rate = (
                        sum(1 for g in gaps_ocr if g > 0.0) / float(len(gaps_ocr))
                        if gaps_ocr
                        else 0.0
                    )
                    gap_shuffle_ocr_pos_rate = (
                        sum(1 for g in gaps_shuffle_ocr if g > 0.0) / float(len(gaps_shuffle_ocr))
                        if gaps_shuffle_ocr
                        else 0.0
                    )
                    gap_drop_adesc_pos_rate = (
                        sum(1 for g in gaps_adesc if g > 0.0) / float(len(gaps_adesc))
                        if gaps_adesc
                        else 0.0
                    )
                    gap_drop_adesc_valid_mean = (
                        sum(gaps_adesc_valid) / float(len(gaps_adesc_valid))
                        if gaps_adesc_valid
                        else 0.0
                    )
                    gap_drop_adesc_valid_pos_rate = (
                        sum(1 for g in gaps_adesc_valid if g > 0.0) / float(len(gaps_adesc_valid))
                        if gaps_adesc_valid
                        else 0.0
                    )
                    ocr_nonempty_rate = (
                        float(ocr_nonempty_count) / float(max(1, val_sample_count))
                    )
                    adesc_nonempty_rate = (
                        float(adesc_nonempty_count) / float(max(1, val_sample_count))
                    )
                    ocr_span_len_mean = float(ocr_tokens_sum) / float(max(1, val_sample_count))
                    adesc_span_len_mean = float(adesc_tokens_sum) / float(max(1, val_sample_count))
                    diag = {
                        "attn_to_region": (sum(attn_region) / float(len(attn_region))) if attn_region else 0.0,
                        "attn_to_img": (sum(attn_img) / float(len(attn_img))) if attn_img else 0.0,
                        "attn_to_textctx": (sum(attn_ctx) / float(len(attn_ctx))) if attn_ctx else 0.0,
                        "attn_to_ocr": (sum(attn_ocr) / float(len(attn_ocr))) if attn_ocr else 0.0,
                        "attn_to_adesc": (sum(attn_adesc) / float(len(attn_adesc))) if attn_adesc else 0.0,
                        "attn_to_textctx_forbidden": _mean_stat(base_stats, "attn_to_para"),
                        "attn_batches": attn_batches,
                        "visible_k_ocr_count": _mean_stat(base_stats, "visible_k_ocr_count"),
                        "visible_k_adesc_count": _mean_stat(base_stats, "visible_k_adesc_count"),
                        "visible_k_para_count": _mean_stat(base_stats, "visible_k_para_count"),
                        "visible_k_img_count": _mean_stat(base_stats, "visible_k_img_count"),
                        "visible_k_region_count": _mean_stat(base_stats, "visible_k_region_count"),
                        "len_ctx_ocr_tokens": _mean_stat(base_stats, "len_ctx_ocr_tokens"),
                        "len_ctx_adesc_tokens": _mean_stat(base_stats, "len_ctx_adesc_tokens"),
                        "len_ctx_para_tokens": _mean_stat(base_stats, "len_ctx_para_tokens"),
                        "hash_ctx_ocr": _mean_stat(base_stats, "hash_ctx_ocr"),
                        "hash_ctx_adesc": _mean_stat(base_stats, "hash_ctx_adesc"),
                        "seq_len_input_ids": _mean_stat(base_stats, "seq_len_input_ids"),
                        "attn_k_len": _mean_stat(base_stats, "attn_k_len"),
                        "attn_q_len": _mean_stat(base_stats, "attn_q_len"),
                        "q_token_count": _mean_stat(base_stats, "q_token_count"),
                        "span_ctx_start": _mean_stat(base_stats, "span_ctx_start"),
                        "span_ctx_end": _mean_stat(base_stats, "span_ctx_end"),
                        "span_ocr_start": _mean_stat(base_stats, "span_ocr_start"),
                        "span_ocr_end": _mean_stat(base_stats, "span_ocr_end"),
                        "span_adesc_start": _mean_stat(base_stats, "span_adesc_start"),
                        "span_adesc_end": _mean_stat(base_stats, "span_adesc_end"),
                        "span_para_start": _mean_stat(base_stats, "span_para_start"),
                        "span_para_end": _mean_stat(base_stats, "span_para_end"),
                        "prompt_len_mm": _mean_stat(base_stats, "prompt_len_mm"),
                        "ctx_total_hint": _mean_stat(base_stats, "ctx_total_hint"),
                        "allow_n_hint": _mean_stat(base_stats, "allow_n_hint"),
                        "drop_ocr_visible_k_ocr_count": _mean_stat(drop_ocr_stats, "visible_k_ocr_count"),
                        "drop_ocr_visible_k_adesc_count": _mean_stat(drop_ocr_stats, "visible_k_adesc_count"),
                        "drop_ocr_visible_k_para_count": _mean_stat(drop_ocr_stats, "visible_k_para_count"),
                        "drop_ocr_len_ctx_ocr_tokens": _mean_stat(drop_ocr_stats, "len_ctx_ocr_tokens"),
                        "drop_ocr_len_ctx_adesc_tokens": _mean_stat(drop_ocr_stats, "len_ctx_adesc_tokens"),
                        "drop_ocr_hash_ctx_ocr": _mean_stat(drop_ocr_stats, "hash_ctx_ocr"),
                        "drop_ocr_hash_ctx_adesc": _mean_stat(drop_ocr_stats, "hash_ctx_adesc"),
                        "drop_ocr_attn_to_ocr": _mean_stat(drop_ocr_stats, "attn_to_ocr"),
                        "drop_ocr_attn_to_adesc": _mean_stat(drop_ocr_stats, "attn_to_adesc"),
                        "drop_adesc_visible_k_ocr_count": _mean_stat(drop_adesc_stats, "visible_k_ocr_count"),
                        "drop_adesc_visible_k_adesc_count": _mean_stat(drop_adesc_stats, "visible_k_adesc_count"),
                        "drop_adesc_visible_k_para_count": _mean_stat(drop_adesc_stats, "visible_k_para_count"),
                        "drop_adesc_len_ctx_ocr_tokens": _mean_stat(drop_adesc_stats, "len_ctx_ocr_tokens"),
                        "drop_adesc_len_ctx_adesc_tokens": _mean_stat(drop_adesc_stats, "len_ctx_adesc_tokens"),
                        "drop_adesc_hash_ctx_ocr": _mean_stat(drop_adesc_stats, "hash_ctx_ocr"),
                        "drop_adesc_hash_ctx_adesc": _mean_stat(drop_adesc_stats, "hash_ctx_adesc"),
                        "drop_adesc_attn_to_ocr": _mean_stat(drop_adesc_stats, "attn_to_ocr"),
                        "drop_adesc_attn_to_adesc": _mean_stat(drop_adesc_stats, "attn_to_adesc"),
                        "gap_drop_ocr_pos_rate": gap_drop_ocr_pos_rate,
                        "gap_drop_adesc_pos_rate": gap_drop_adesc_pos_rate,
                        "gap_drop_adesc_valid_mean": gap_drop_adesc_valid_mean,
                        "gap_drop_adesc_valid_pos_rate": gap_drop_adesc_valid_pos_rate,
                        "adesc_nonempty_rate": adesc_nonempty_rate,
                        "ocr_nonempty_rate": ocr_nonempty_rate,
                        "ocr_span_len_mean": ocr_span_len_mean,
                        "adesc_span_len_mean": adesc_span_len_mean,
                        "adesc_valid_count": float(len(gaps_adesc_valid)),
                        "val_sample_count": float(val_sample_count),
                    }
                    if args.debug_explain_attn_consistency:
                        para_checks = {
                            "base": float(diag.get("visible_k_para_count", 0.0)),
                            "drop_ocr": float(diag.get("drop_ocr_visible_k_para_count", 0.0)),
                            "drop_adesc": float(diag.get("drop_adesc_visible_k_para_count", 0.0)),
                        }
                        leaked = {k: v for k, v in para_checks.items() if v > 1e-6}
                        if leaked:
                            raise AssertionError(f"[FATAL] paragraph leaked in explain gate: {leaked}")
                        if float(diag.get("len_ctx_ocr_tokens", 0.0)) > 0.0 and float(diag.get("visible_k_ocr_count", 0.0)) <= 1e-6:
                            raise AssertionError(
                                "[FATAL] OCR tokens exist but none visible: "
                                f"len_ctx_ocr={diag.get('len_ctx_ocr_tokens')} vis_ocr={diag.get('visible_k_ocr_count')} "
                                f"span_ocr=({diag.get('span_ocr_start')},{diag.get('span_ocr_end')}) "
                                f"span_ctx=({diag.get('span_ctx_start')},{diag.get('span_ctx_end')}) "
                                f"prompt_len_mm={diag.get('prompt_len_mm')} k_len={diag.get('attn_k_len')} "
                                f"allow_n_hint={diag.get('allow_n_hint')} ctx_total_hint={diag.get('ctx_total_hint')}"
                            )
                        if float(diag.get("len_ctx_adesc_tokens", 0.0)) > 0.0 and float(diag.get("visible_k_adesc_count", 0.0)) <= 1e-6:
                            raise AssertionError(
                                "[FATAL] ADESC tokens exist but none visible: "
                                f"len_ctx_adesc={diag.get('len_ctx_adesc_tokens')} vis_adesc={diag.get('visible_k_adesc_count')} "
                                f"span_adesc=({diag.get('span_adesc_start')},{diag.get('span_adesc_end')}) "
                                f"span_ctx=({diag.get('span_ctx_start')},{diag.get('span_ctx_end')}) "
                                f"prompt_len_mm={diag.get('prompt_len_mm')} k_len={diag.get('attn_k_len')} "
                                f"allow_n_hint={diag.get('allow_n_hint')} ctx_total_hint={diag.get('ctx_total_hint')}"
                            )
                    print(
                        f"[val_explain] step={cur_step} base={mean_base:.4f} "
                        f"region_drop={mean_drop:.4f} ocr_drop={mean_drop_ocr:.4f} adesc_drop={mean_drop_adesc:.4f} "
                        f"ocr_shuffle={mean_shuffle_ocr:.4f} "
                        f"paragraph_shuffle={mean_shuffle:.4f} "
                        f"gap_drop={mean_drop - mean_base:.4f} gap_ocr={mean_drop_ocr - mean_base:.4f} "
                        f"gap_adesc={mean_drop_adesc - mean_base:.4f} gap_shuffle_ocr={mean_shuffle_ocr - mean_base:.4f} "
                        f"gap_shuffle={mean_shuffle - mean_base:.4f} "
                        f"gap_ocr_pos={diag['gap_drop_ocr_pos_rate']:.3f} gap_ocr_shuffle_pos={gap_shuffle_ocr_pos_rate:.3f} "
                        f"adesc_nonempty={diag['adesc_nonempty_rate']:.3f} "
                        f"attn_region={diag['attn_to_region']:.4f} attn_img={diag['attn_to_img']:.4f} "
                        f"attn_textctx={diag['attn_to_textctx']:.4f} attn_ocr={diag['attn_to_ocr']:.4f} "
                        f"attn_adesc={diag['attn_to_adesc']:.4f} "
                        f"vis_ocr={diag['visible_k_ocr_count']:.1f} vis_adesc={diag['visible_k_adesc_count']:.1f} "
                        f"vis_para={diag['visible_k_para_count']:.1f} dropocr_vis_ocr={diag['drop_ocr_visible_k_ocr_count']:.1f} "
                        f"dropadesc_vis_adesc={diag['drop_adesc_visible_k_adesc_count']:.1f}"
                    )
                    with open(out_dir / "val_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                (
                                    {
                                        "step": cur_step,
                                        "split": "scistruct_explain",
                                        "gap_drop_ocr": mean_drop_ocr - mean_base,
                                        "gap_drop_adesc": mean_drop_adesc - mean_base,
                                        "gap_region_drop": mean_drop - mean_base,
                                        "gap_shuffle_ocr": mean_shuffle_ocr - mean_base,
                                        "attn_to_textctx_forbidden": diag["attn_to_textctx_forbidden"],
                                    }
                                    if args.explain_metrics_minimal_only
                                    else
                                {
                                    "step": cur_step,
                                    "split": "scistruct_explain",
                                    "loss_base": mean_base,
                                    "loss_region_drop": mean_drop,
                                    "loss_drop_ocr": mean_drop_ocr,
                                    "loss_drop_adesc": mean_drop_adesc,
                                    "loss_shuffle_ocr": mean_shuffle_ocr,
                                    "loss_paragraph_shuffle": mean_shuffle,
                                    "loss_context_shuffle": mean_shuffle,
                                    "gap_region_drop": mean_drop - mean_base,
                                    "gap_drop_ocr": mean_drop_ocr - mean_base,
                                    "gap_drop_adesc": mean_drop_adesc - mean_base,
                                    "gap_shuffle_ocr": mean_shuffle_ocr - mean_base,
                                    "gap_paragraph_shuffle": mean_shuffle - mean_base,
                                    "gap_context_shuffle": mean_shuffle - mean_base,
                                    "attn_to_region": diag["attn_to_region"],
                                    "attn_to_img": diag["attn_to_img"],
                                    "attn_to_textctx": diag["attn_to_textctx"],
                                    "attn_to_ocr": diag["attn_to_ocr"],
                                    "attn_to_adesc": diag["attn_to_adesc"],
                                    "attn_to_textctx_forbidden": diag["attn_to_textctx_forbidden"],
                                    "attn_batches": diag["attn_batches"],
                                    "visible_k_ocr_count": diag["visible_k_ocr_count"],
                                    "visible_k_adesc_count": diag["visible_k_adesc_count"],
                                    "visible_k_para_count": diag["visible_k_para_count"],
                                    "visible_k_img_count": diag["visible_k_img_count"],
                                    "visible_k_region_count": diag["visible_k_region_count"],
                                    "len_ctx_ocr_tokens": diag["len_ctx_ocr_tokens"],
                                    "len_ctx_adesc_tokens": diag["len_ctx_adesc_tokens"],
                                    "len_ctx_para_tokens": diag["len_ctx_para_tokens"],
                                    "hash_ctx_ocr": int(diag["hash_ctx_ocr"]),
                                    "hash_ctx_adesc": int(diag["hash_ctx_adesc"]),
                                    "seq_len_input_ids": diag["seq_len_input_ids"],
                                    "attn_k_len": diag["attn_k_len"],
                                    "attn_q_len": diag["attn_q_len"],
                                    "q_token_count": diag["q_token_count"],
                                    "span_ctx_start": diag["span_ctx_start"],
                                    "span_ctx_end": diag["span_ctx_end"],
                                    "span_ocr_start": diag["span_ocr_start"],
                                    "span_ocr_end": diag["span_ocr_end"],
                                    "span_adesc_start": diag["span_adesc_start"],
                                    "span_adesc_end": diag["span_adesc_end"],
                                    "span_para_start": diag["span_para_start"],
                                    "span_para_end": diag["span_para_end"],
                                    "prompt_len_mm": diag["prompt_len_mm"],
                                    "ctx_total_hint": diag["ctx_total_hint"],
                                    "allow_n_hint": diag["allow_n_hint"],
                                    "gap_drop_ocr_pos_rate": diag["gap_drop_ocr_pos_rate"],
                                    "gap_shuffle_ocr_pos_rate": gap_shuffle_ocr_pos_rate,
                                    "gap_drop_adesc_pos_rate": diag["gap_drop_adesc_pos_rate"],
                                    "gap_drop_adesc_valid_mean": diag["gap_drop_adesc_valid_mean"],
                                    "gap_drop_adesc_valid_pos_rate": diag["gap_drop_adesc_valid_pos_rate"],
                                    "adesc_nonempty_rate": diag["adesc_nonempty_rate"],
                                    "ocr_nonempty_rate": diag["ocr_nonempty_rate"],
                                    "ocr_span_len_mean": diag["ocr_span_len_mean"],
                                    "adesc_span_len_mean": diag["adesc_span_len_mean"],
                                    "adesc_valid_count": diag["adesc_valid_count"],
                                    "val_sample_count": diag["val_sample_count"],
                                    "drop_ocr_visible_k_ocr_count": diag["drop_ocr_visible_k_ocr_count"],
                                    "drop_ocr_visible_k_adesc_count": diag["drop_ocr_visible_k_adesc_count"],
                                    "drop_ocr_visible_k_para_count": diag["drop_ocr_visible_k_para_count"],
                                    "drop_ocr_len_ctx_ocr_tokens": diag["drop_ocr_len_ctx_ocr_tokens"],
                                    "drop_ocr_len_ctx_adesc_tokens": diag["drop_ocr_len_ctx_adesc_tokens"],
                                    "drop_ocr_hash_ctx_ocr": int(diag["drop_ocr_hash_ctx_ocr"]),
                                    "drop_ocr_hash_ctx_adesc": int(diag["drop_ocr_hash_ctx_adesc"]),
                                    "drop_ocr_attn_to_ocr": diag["drop_ocr_attn_to_ocr"],
                                    "drop_ocr_attn_to_adesc": diag["drop_ocr_attn_to_adesc"],
                                    "drop_adesc_visible_k_ocr_count": diag["drop_adesc_visible_k_ocr_count"],
                                    "drop_adesc_visible_k_adesc_count": diag["drop_adesc_visible_k_adesc_count"],
                                    "drop_adesc_visible_k_para_count": diag["drop_adesc_visible_k_para_count"],
                                    "drop_adesc_len_ctx_ocr_tokens": diag["drop_adesc_len_ctx_ocr_tokens"],
                                    "drop_adesc_len_ctx_adesc_tokens": diag["drop_adesc_len_ctx_adesc_tokens"],
                                    "drop_adesc_hash_ctx_ocr": int(diag["drop_adesc_hash_ctx_ocr"]),
                                    "drop_adesc_hash_ctx_adesc": int(diag["drop_adesc_hash_ctx_adesc"]),
                                    "drop_adesc_attn_to_ocr": diag["drop_adesc_attn_to_ocr"],
                                    "drop_adesc_attn_to_adesc": diag["drop_adesc_attn_to_adesc"],
                                    "batches": len(losses_base),
                                },
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            elif args.dataset == "scicap":
                losses: List[float] = []
                by_scale: Dict[str, List[float]] = {"short": [], "long": [], "desc": []}
                by_scale_len: Dict[str, List[int]] = {"short": [], "long": [], "desc": []}
                for bidx, vbatch in enumerate(val_dl):
                    images_v = vbatch["images"]
                    pixel_values_v = student.preprocess(list(images_v))["pixel_values"].to(device)
                    input_ids_v = vbatch["input_ids"].to(device)
                    labels_v = vbatch["labels"].to(device)
                    attention_mask_v = vbatch["attention_mask"].to(device)
                    scales_v = list(vbatch.get("scales", []))
                    regions_v = vbatch.get("regions", None)
                    vision_wrapper.set_regions(regions_v, drop_one_region=False)
                    _set_region_attention_bias_for_inputs(
                        input_ids_v,
                        scales_v,
                        regions_v,
                        labels_local=labels_v,
                        contexts_local=vbatch.get("contexts_used", vbatch.get("contexts", None)),
                            context_allow_tokens_local=vbatch.get("context_allow_tokens", None),
                    )
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_v, attn_mask_v, past_kv_v, inputs_embeds_v, labels_mm_v = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids_v,
                            position_ids=None,
                            attention_mask=attention_mask_v,
                            past_key_values=None,
                            labels=labels_v,
                            images=pixel_values_v,
                            image_sizes=None,
                        )
                        out_v = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_v,
                            position_ids=pos_ids_v,
                            past_key_values=past_kv_v,
                            inputs_embeds=inputs_embeds_v,
                            labels=labels_mm_v,
                            use_cache=False,
                        )
                        nll_ps = nll_per_sample(out_v.logits, labels_mm_v)
                        loss_v = nll_ps.mean()
                    _clear_region_attention_bias()
                    losses.append(float(loss_v.detach().item()))
                    for i, sv in enumerate(scales_v):
                        if sv in by_scale and i < nll_ps.shape[0]:
                            by_scale[sv].append(float(nll_ps[i].detach().item()))
                            tgt_len = int((labels_mm_v[i] != -100).sum().item())
                            by_scale_len[sv].append(tgt_len)
                    del out_v, inputs_embeds_v, labels_mm_v, attn_mask_v, pos_ids_v, past_kv_v
                    torch.cuda.empty_cache()
                    if args.val_num_batches > 0 and (bidx + 1) >= args.val_num_batches:
                        break
                if losses:
                    avg_val = sum(losses) / float(len(losses))
                    scale_stats = {k: (sum(v) / float(len(v)) if v else None) for k, v in by_scale.items()}
                    len_stats = {k: (sum(v) / float(len(v)) if v else None) for k, v in by_scale_len.items()}
                    ladder_ok = (
                        len_stats["short"] is not None
                        and len_stats["long"] is not None
                        and len_stats["desc"] is not None
                        and (len_stats["short"] < len_stats["long"] <= len_stats["desc"])
                    )
                    print(
                        f"[val_scicap] step={cur_step} loss={avg_val:.4f} "
                        f"short={scale_stats['short']} long={scale_stats['long']} desc={scale_stats['desc']} "
                        f"len_short={len_stats['short']} len_long={len_stats['long']} len_desc={len_stats['desc']} "
                        f"length_ladder={ladder_ok} batches={len(losses)}"
                    )
                    with open(out_dir / "val_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "step": cur_step,
                                    "split": "scicap",
                                    "val_loss": avg_val,
                                    "short": scale_stats["short"],
                                    "long": scale_stats["long"],
                                    "desc": scale_stats["desc"],
                                    "len_short": len_stats["short"],
                                    "len_long": len_stats["long"],
                                    "len_desc": len_stats["desc"],
                                    "length_ladder": bool(ladder_ok),
                                    "batches": len(losses),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            else:
                losses: List[float] = []
                for bidx, vbatch in enumerate(val_dl):
                    images_v = vbatch["images"]
                    pixel_values_v = student.preprocess(list(images_v))["pixel_values"].to(device)
                    input_ids_v = vbatch["input_ids"].to(device)
                    labels_v = vbatch["labels"].to(device)
                    attention_mask_v = vbatch["attention_mask"].to(device)
                    vision_wrapper.set_regions(vbatch.get("regions", None), drop_one_region=False)
                    _set_region_attention_bias_for_inputs(
                        input_ids_v,
                        vbatch.get("scales", None),
                        vbatch.get("regions", None),
                        labels_local=labels_v,
                        contexts_local=vbatch.get("contexts_used", vbatch.get("contexts", None)),
                            context_allow_tokens_local=vbatch.get("context_allow_tokens", None),
                    )
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_v, attn_mask_v, past_kv_v, inputs_embeds_v, labels_mm_v = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids_v,
                            position_ids=None,
                            attention_mask=attention_mask_v,
                            past_key_values=None,
                            labels=labels_v,
                            images=pixel_values_v,
                            image_sizes=None,
                        )
                        out_v = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_v,
                            position_ids=pos_ids_v,
                            past_key_values=past_kv_v,
                            inputs_embeds=inputs_embeds_v,
                            labels=labels_mm_v,
                            use_cache=False,
                        )
                        loss_v = nll_per_sample(out_v.logits, labels_mm_v).mean()
                    _clear_region_attention_bias()
                    losses.append(float(loss_v.detach().item()))
                    del out_v, inputs_embeds_v, labels_mm_v, attn_mask_v, pos_ids_v, past_kv_v
                    torch.cuda.empty_cache()
                    if args.val_num_batches > 0 and (bidx + 1) >= args.val_num_batches:
                        break
                if losses:
                    avg_val = sum(losses) / float(len(losses))
                    print(f"[val] step={cur_step} loss={avg_val:.4f} batches={len(losses)}")
                    with open(out_dir / "val_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": cur_step, "val_loss": avg_val, "batches": len(losses)}, ensure_ascii=False) + "\n")
        vision_wrapper.clear_regions()
        model.train()
        if args.freeze_vision:
            student.eval()
        else:
            student.train()

    def run_sample_dump(cur_step: int) -> None:
        if args.sample_num <= 0:
            return
        if args.dataset == "stage4_multitask":
            if not sample_indices_scicap and not sample_indices_explain:
                return
        else:
            if not sample_indices:
                return
        _update_runtime_schedule(int(cur_step), force_log=False)
        model.eval()
        student.eval()

        def _task_sample_max_new_tokens(task_name: str) -> int:
            if task_name == "short" and args.sample_max_new_tokens_short > 0:
                return int(args.sample_max_new_tokens_short)
            if task_name == "long" and args.sample_max_new_tokens_long > 0:
                return int(args.sample_max_new_tokens_long)
            if task_name == "desc" and args.sample_max_new_tokens_desc > 0:
                return int(args.sample_max_new_tokens_desc)
            return int(args.sample_max_new_tokens)

        def _sample_generate(
            prompt_ids: List[int],
            images_t: torch.Tensor,
            regions_t: List[Dict],
            task_for_bias: str,
            context_for_bias: str = "",
        ) -> Tuple[str, torch.Tensor]:
            input_ids_s = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attn_s = torch.ones_like(input_ids_s, dtype=torch.long, device=device)
            vision_wrapper.set_regions([regions_t], drop_one_region=False)
            _set_region_attention_bias_for_inputs(
                input_ids_s,
                [task_for_bias],
                [regions_t],
                labels_local=None,
                contexts_local=[context_for_bias if context_for_bias else ""],
                context_allow_tokens_local=None,
            )
            max_new_tokens_now = _task_sample_max_new_tokens(task_for_bias)
            gen_kwargs = {
                "inputs": input_ids_s,
                "attention_mask": attn_s,
                "images": images_t,
                "max_new_tokens": max(1, int(max_new_tokens_now)),
                "min_new_tokens": int(args.sample_min_new_tokens),
                "do_sample": False,
                "num_beams": max(1, int(args.eval_num_beams)),
                "length_penalty": float(args.eval_length_penalty),
                "repetition_penalty": float(args.eval_repetition_penalty),
                "no_repeat_ngram_size": max(0, int(args.eval_no_repeat_ngram_size)),
            }
            # Avoid per-call HF warning spam ("Setting pad_token_id to eos_token_id")
            # and keep generation deterministic by explicitly passing tokenizer ids.
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = int(pad_id)
            if tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = int(tokenizer.eos_token_id)
            bad_ids: List[List[int]] = []
            if (
                args.caption_block_formula_in_decode
                and task_for_bias in ("short", "long")
                and caption_formula_token_ids
            ):
                bad_ids.extend([[int(t)] for t in caption_formula_token_ids[:64]])
            if (
                args.desc_block_prompt_leak_in_decode
                and task_for_bias == "desc"
                and desc_prompt_leak_token_ids
            ):
                bad_ids.extend([[int(t)] for t in desc_prompt_leak_token_ids[:64]])
            if bad_ids:
                uniq = []
                seen = set()
                for x in bad_ids:
                    key = int(x[0]) if x else -1
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append([key])
                if uniq:
                    gen_kwargs["bad_words_ids"] = uniq
            out_ids = model.generate(
                **gen_kwargs,
            )
            _clear_region_attention_bias()
            if out_ids.dim() == 1:
                out_ids = out_ids.unsqueeze(0)
            input_len = int(input_ids_s.shape[1])
            has_prefix = out_ids.shape[1] >= input_len and torch.equal(out_ids[:, :input_len], input_ids_s)
            gen_ids = out_ids[:, input_len:] if has_prefix else out_ids
            full_ids = torch.cat([input_ids_s, gen_ids], dim=1)
            text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            return text, full_ids

        def _collect_attn_mass_single(
            prompt_ids: List[int],
            full_ids: torch.Tensor,
            images_t: torch.Tensor,
            regions_t: List[Dict],
            context_for_bias: str,
        ) -> Dict[str, float]:
            if not args.log_attn:
                return {"attn_to_region": 0.0, "attn_to_img": 0.0, "attn_to_textctx": 0.0, "layers": 0}
            input_ids_f = full_ids.to(device)
            attn_f = torch.ones_like(input_ids_f, dtype=torch.long, device=device)
            labels_f = input_ids_f.clone()
            prompt_raw_len = min(int(len(prompt_ids)), int(labels_f.shape[1]))
            if prompt_raw_len > 0:
                labels_f[:, :prompt_raw_len] = -100
            vision_wrapper.set_regions([regions_t], drop_one_region=False)
            _set_region_attention_bias_for_inputs(
                input_ids_f,
                ["explain"],
                [regions_t],
                labels_local=labels_f,
                contexts_local=[context_for_bias if context_for_bias else ""],
                context_allow_tokens_local=None,
            )
            bias_ranges_f = model._region_attn_bias_ranges
            try:
                _ensure_log_attn_exportable()
                with torch.no_grad():
                    _, pos_ids_f, attn_mask_f, past_kv_f, inputs_embeds_f, labels_mm_f = model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids_f,
                        position_ids=None,
                        attention_mask=attn_f,
                        past_key_values=None,
                        labels=labels_f,
                        images=images_t,
                        image_sizes=None,
                    )
                    out_f = model.language_model(
                        input_ids=None,
                        attention_mask=attn_mask_f,
                        position_ids=pos_ids_f,
                        past_key_values=past_kv_f,
                        inputs_embeds=inputs_embeds_f,
                        labels=labels_mm_f,
                        use_cache=False,
                        output_attentions=True,
                    )
                attn_stat = _collect_attn_mass_from_forward(
                    attentions=out_f.attentions,
                    input_ids_local=input_ids_f,
                    labels_mm_local=labels_mm_f,
                    regions_local=[regions_t],
                    bias_ranges_local=bias_ranges_f,
                )
                if attn_stat is None:
                    attn_stat = {"attn_to_region": 0.0, "attn_to_img": 0.0, "attn_to_textctx": 0.0, "layers": 0}
                _assert_explain_attn_consistency(
                    attn_stat,
                    out_f.attentions,
                    input_ids_f,
                    labels_mm_f,
                    [regions_t],
                    bias_ranges_f,
                )
                del out_f, inputs_embeds_f, labels_mm_f, attn_mask_f, pos_ids_f, past_kv_f
                return attn_stat
            finally:
                _clear_region_attention_bias()

        if args.dataset == "stage4_multitask":
            dumps_scicap: List[Dict] = []
            dumps_explain: List[Dict] = []
            explain_items: List[Tuple[int, object, str, str, List[Dict]]] = []
            if sample_ds_explain is not None:
                for idx in sample_indices_explain:
                    item = sample_ds_explain[idx]
                    if not isinstance(item, tuple) or len(item) < 4:
                        continue
                    if len(item) >= 5:
                        img, target_text, _, ctx, regions = item[:5]
                    else:
                        img, target_text, _, ctx = item[:4]
                        regions = []
                    explain_items.append((idx, img, target_text, ctx, regions or []))
            with torch.no_grad():
                if sample_ds_scicap is not None:
                    for idx in sample_indices_scicap:
                        item = sample_ds_scicap[idx]
                        if not isinstance(item, tuple) or len(item) < 4:
                            continue
                        img, target_text, scale, ctx, regions, ctx_meta = _unpack_multimodal_item(item)
                        pixel_values_s = student.preprocess([img])["pixel_values"].to(device)
                        task_outputs: Dict[str, str] = {}
                        for task in ("short", "long", "desc"):
                            task_ctx = _resolve_scicap_task_context(
                                scale=task,
                                context=ctx,
                                context_meta=ctx_meta if isinstance(ctx_meta, dict) else None,
                                routing_mode=args.scicap_task_context_routing,
                                use_placeholders=args.use_context_placeholders,
                            )
                            prompt_ids = build_generation_prompt(
                                tokenizer=tokenizer,
                                image_token_index=model.config.image_token_index,
                                scale=task,
                                context=task_ctx,
                                max_length=args.max_length,
                                max_ctx_tokens=args.max_ctx_tokens,

                                max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                                scicap_prompt_style=args.scicap_prompt_style,
                                enable_task_style_tokens=args.enable_task_style_tokens,
                            )
                            out_text, _ = _sample_generate(prompt_ids, pixel_values_s, regions or [], task, task_ctx)
                            task_outputs[task] = out_text
                        dumps_scicap.append(
                            {
                                "idx": int(idx),
                                "image": getattr(img, "filename", ""),
                                "scale": scale,
                                "target": target_text,
                                "context_preview": (ctx or "")[:240],
                                "outputs": task_outputs,
                            }
                        )
                explain_contexts = [it[3] for it in explain_items]
                for ii, (idx, img, target_text, ctx, regions) in enumerate(explain_items):
                    pixel_values_s = student.preprocess([img])["pixel_values"].to(device)
                    zero_images_s = torch.zeros_like(pixel_values_s)
                    ctx_shuf = ctx
                    if len(explain_contexts) > 1:
                        ctx_shuf = explain_contexts[(ii + 1) % len(explain_contexts)]
                    prompt_region = build_generation_prompt(
                        tokenizer=tokenizer,
                        image_token_index=model.config.image_token_index,
                        scale="explain",
                        context="",
                        max_length=args.max_length,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                        scicap_prompt_style=args.scicap_prompt_style,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                    )
                    prompt_ctx = build_generation_prompt(
                        tokenizer=tokenizer,
                        image_token_index=model.config.image_token_index,
                        scale="explain",
                        context=ctx,
                        max_length=args.max_length,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                        scicap_prompt_style=args.scicap_prompt_style,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                    )
                    prompt_ctx_shuf = build_generation_prompt(
                        tokenizer=tokenizer,
                        image_token_index=model.config.image_token_index,
                        scale="explain",
                        context=ctx_shuf,
                        max_length=args.max_length,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                        scicap_prompt_style=args.scicap_prompt_style,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                    )
                    out_region, _ = _sample_generate(prompt_region, pixel_values_s, regions, "explain", "")
                    out_region_ctx, full_region_ctx = _sample_generate(prompt_ctx, pixel_values_s, regions, "explain", ctx)
                    out_region_drop, _ = _sample_generate(prompt_ctx, pixel_values_s, [], "explain", ctx)
                    out_ctx_shuf, full_ctx_shuf = _sample_generate(prompt_ctx_shuf, pixel_values_s, regions, "explain", ctx_shuf)
                    out_ctx_only, _ = _sample_generate(prompt_ctx, zero_images_s, [], "explain", ctx)
                    attn_region_ctx = _collect_attn_mass_single(
                        prompt_ids=prompt_ctx,
                        full_ids=full_region_ctx,
                        images_t=pixel_values_s,
                        regions_t=regions,
                        context_for_bias=ctx,
                    )
                    attn_ctx_shuf = _collect_attn_mass_single(
                        prompt_ids=prompt_ctx_shuf,
                        full_ids=full_ctx_shuf,
                        images_t=pixel_values_s,
                        regions_t=regions,
                        context_for_bias=ctx_shuf,
                    )
                    dumps_explain.append(
                        {
                            "idx": int(idx),
                            "image": getattr(img, "filename", ""),
                            "scale": "explain",
                            "mask_count": len(regions or []),
                            "target": target_text,
                            "context_preview": (ctx or "")[:240],
                            "context_shuffle_preview": (ctx_shuf or "")[:240],
                            "outputs": {
                                "image_region": out_region,
                                "image_region_paragraph": out_region_ctx,
                                "image_region_region_drop": out_region_drop,
                                "image_region_paragraph_shuffle": out_ctx_shuf,
                                "paragraph_only": out_ctx_only,
                            },
                            "attn_mass": {
                                "image_region_paragraph": attn_region_ctx,
                                "image_region_paragraph_shuffle": attn_ctx_shuf,
                            },
                        }
                    )
            vision_wrapper.clear_regions()
            model.train()
            if args.freeze_vision:
                student.eval()
            else:
                student.train()
            scicap_path = sample_out_dir / f"samples_scicap_step_{cur_step:06d}.json"
            explain_path = sample_out_dir / f"samples_explain_diag_step_{cur_step:06d}.json"
            with open(scicap_path, "w", encoding="utf-8") as f:
                json.dump(dumps_scicap, f, ensure_ascii=False, indent=2)
            with open(explain_path, "w", encoding="utf-8") as f:
                json.dump(dumps_explain, f, ensure_ascii=False, indent=2)
            print(f"[sample] step={cur_step} scicap_saved={scicap_path} explain_diag_saved={explain_path}")
            return

        dumps: List[Dict] = []
        if args.sample_mode == "explain_diag":
            explain_items: List[Tuple[int, object, str, str, List[Dict]]] = []
            for idx in sample_indices:
                item = sample_ds[idx]
                if not isinstance(item, tuple):
                    continue
                if len(item) >= 5:
                    img, target_text, _, ctx, regions = item[:5]
                elif len(item) == 4:
                    img, target_text, _, ctx = item
                    regions = []
                else:
                    continue
                explain_items.append((idx, img, target_text, ctx, regions or []))
            explain_contexts = [it[3] for it in explain_items]
            with torch.no_grad():
                for ii, (idx, img, target_text, ctx, regions) in enumerate(explain_items):
                    pixel_values_s = student.preprocess([img])["pixel_values"].to(device)
                    zero_images_s = torch.zeros_like(pixel_values_s)
                    ctx_shuf = ctx
                    if len(explain_contexts) > 1:
                        ctx_shuf = explain_contexts[(ii + 1) % len(explain_contexts)]
                    prompt_region = build_generation_prompt(
                        tokenizer=tokenizer,
                        image_token_index=model.config.image_token_index,
                        scale=sample_eval_task,
                        context="",
                        max_length=args.max_length,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                        scicap_prompt_style=args.scicap_prompt_style,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                    )
                    prompt_ctx = build_generation_prompt(
                        tokenizer=tokenizer,
                        image_token_index=model.config.image_token_index,
                        scale=sample_eval_task,
                        context=ctx,
                        max_length=args.max_length,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                        scicap_prompt_style=args.scicap_prompt_style,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                    )
                    prompt_ctx_shuf = build_generation_prompt(
                        tokenizer=tokenizer,
                        image_token_index=model.config.image_token_index,
                        scale=sample_eval_task,
                        context=ctx_shuf,
                        max_length=args.max_length,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                        scicap_prompt_style=args.scicap_prompt_style,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                    )
                    out_region, _ = _sample_generate(prompt_region, pixel_values_s, regions, sample_eval_task, "")
                    out_region_ctx, full_region_ctx = _sample_generate(prompt_ctx, pixel_values_s, regions, sample_eval_task, ctx)
                    out_region_drop, _ = _sample_generate(prompt_ctx, pixel_values_s, [], sample_eval_task, ctx)
                    out_ctx_shuf, full_ctx_shuf = _sample_generate(prompt_ctx_shuf, pixel_values_s, regions, sample_eval_task, ctx_shuf)
                    out_ctx_only, _ = _sample_generate(prompt_ctx, zero_images_s, [], sample_eval_task, ctx)
                    attn_region_ctx = _collect_attn_mass_single(
                        prompt_ids=prompt_ctx,
                        full_ids=full_region_ctx,
                        images_t=pixel_values_s,
                        regions_t=regions,
                        context_for_bias=ctx,
                    )
                    attn_ctx_shuf = _collect_attn_mass_single(
                        prompt_ids=prompt_ctx_shuf,
                        full_ids=full_ctx_shuf,
                        images_t=pixel_values_s,
                        regions_t=regions,
                        context_for_bias=ctx_shuf,
                    )
                    dumps.append(
                        {
                            "idx": int(idx),
                            "image": getattr(img, "filename", ""),
                            "scale": sample_eval_task,
                            "sample_mode": args.sample_mode,
                            "mask_count": len(regions or []),
                            "target": target_text,
                            "context_preview": (ctx or "")[:240],
                            "context_shuffle_preview": (ctx_shuf or "")[:240],
                            "outputs": {
                                "image_region": out_region,
                                "image_region_paragraph": out_region_ctx,
                                "image_paragraph": out_region_drop,
                                "image_region_region_drop": out_region_drop,
                                "image_region_paragraph_shuffle": out_ctx_shuf,
                                "paragraph_only": out_ctx_only,
                            },
                            "attn_mass": {
                                "image_region_paragraph": attn_region_ctx,
                                "image_region_paragraph_shuffle": attn_ctx_shuf,
                            },
                        }
                    )
        else:
            with torch.no_grad():
                for idx in sample_indices:
                    item = sample_ds[idx]
                    if not isinstance(item, tuple):
                        continue
                    img, target_text, scale, ctx, regions, ctx_meta = _unpack_multimodal_item(item)
                    pixel_values_s = student.preprocess([img])["pixel_values"].to(device)
                    zero_images_s = torch.zeros_like(pixel_values_s)

                    task_outputs: Dict[str, str] = {}
                    if args.sample_mode == "stage3_modes":
                        prompt_region_only = build_generation_prompt(
                            tokenizer=tokenizer,
                            image_token_index=model.config.image_token_index,
                            scale=sample_eval_task,
                            context="",
                            max_length=args.max_length,
                            max_ctx_tokens=args.max_ctx_tokens,

                            max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                            scicap_prompt_style=args.scicap_prompt_style,
                            enable_task_style_tokens=args.enable_task_style_tokens,
                        )
                        prompt_with_ctx = build_generation_prompt(
                            tokenizer=tokenizer,
                            image_token_index=model.config.image_token_index,
                            scale=sample_eval_task,
                            context=ctx,
                            max_length=args.max_length,
                            max_ctx_tokens=args.max_ctx_tokens,

                            max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                            scicap_prompt_style=args.scicap_prompt_style,
                            enable_task_style_tokens=args.enable_task_style_tokens,
                        )
                        out_image_region, _ = _sample_generate(prompt_region_only, pixel_values_s, regions or [], sample_eval_task, "")
                        out_image_region_ctx, _ = _sample_generate(prompt_with_ctx, pixel_values_s, regions or [], sample_eval_task, ctx)
                        out_image_ctx, _ = _sample_generate(prompt_with_ctx, pixel_values_s, [], sample_eval_task, ctx)
                        out_ctx_only, _ = _sample_generate(prompt_with_ctx, zero_images_s, [], sample_eval_task, ctx)
                        task_outputs = {
                            "image_region": out_image_region,
                            "image_region_paragraph": out_image_region_ctx,
                            "image_paragraph": out_image_ctx,
                            "paragraph_only": out_ctx_only,
                        }
                    else:
                        for task in sample_tasks:
                            task_ctx = _resolve_scicap_task_context(
                                scale=task,
                                context=ctx,
                                context_meta=ctx_meta if isinstance(ctx_meta, dict) else None,
                                routing_mode=args.scicap_task_context_routing,
                                use_placeholders=args.use_context_placeholders,
                            )
                            prompt_ids = build_generation_prompt(
                                tokenizer=tokenizer,
                                image_token_index=model.config.image_token_index,
                                scale=task,
                                context=task_ctx,
                                max_length=args.max_length,
                                max_ctx_tokens=args.max_ctx_tokens,

                                max_ctx_tokens_explain=args.max_ctx_tokens_explain,

                                scicap_prompt_style=args.scicap_prompt_style,
                                enable_task_style_tokens=args.enable_task_style_tokens,
                            )
                            out_task, _ = _sample_generate(prompt_ids, pixel_values_s, regions or [], task, task_ctx)
                            task_outputs[task] = out_task
                    dumps.append(
                        {
                            "idx": int(idx),
                            "image": getattr(img, "filename", ""),
                            "scale": scale,
                            "sample_mode": args.sample_mode,
                            "mask_count": len(regions or []),
                            "target": target_text,
                            "context_preview": (ctx or "")[:240],
                            "outputs": task_outputs,
                        }
                    )
        vision_wrapper.clear_regions()
        model.train()
        if args.freeze_vision:
            student.eval()
        else:
            student.train()
        if args.sample_mode == "explain_diag":
            sample_path = sample_out_dir / f"samples_explain_diag_step_{cur_step:06d}.json"
        else:
            sample_path = sample_out_dir / f"samples_step_{cur_step:06d}.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(dumps, f, ensure_ascii=False, indent=2)
        print(f"[sample] step={cur_step} saved={sample_path}")

    if args.eval_only:
        run_validation(int(args.eval_step))
        if args.sample_every > 0 and args.sample_num > 0:
            run_sample_dump(int(args.eval_step))
        print("[done][eval_only]")
        return

    if warmup_dl is not None:
        warmup_step = 0
        while warmup_step < args.warmup_steps:
            for batch in warmup_dl:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = out.loss / max(1, args.grad_accum)

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if args.task_embed_only:
                    _mask_embedding_grads(model.language_model.get_input_embeddings(), task_token_ids)
                    out_emb = model.language_model.get_output_embeddings()
                    if out_emb is not None and out_emb is not model.language_model.get_input_embeddings():
                        _mask_embedding_grads(out_emb, task_token_ids)

                if (warmup_step + 1) % args.grad_accum == 0:
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * max(1, args.grad_accum)
                if warmup_step % args.log_every == 0:
                    avg_loss = running_loss / max(1, args.log_every)
                    elapsed = time.time() - start_time
                    print(f"[warmup] step={warmup_step} loss={avg_loss:.4f} elapsed={elapsed:.1f}s")
                    running_loss = 0.0
                    start_time = time.time()

                warmup_step += 1
                if warmup_step >= args.warmup_steps:
                    break

    running_loss = 0.0
    start_time = time.time()
    task_loss_ema: Dict[str, float] = {}
    if args.task_balance_mode != "none":
        print(
            f"[info] task balance enabled: mode={args.task_balance_mode} "
            f"alpha={float(args.task_balance_alpha):.3f} ema={float(args.task_balance_ema):.3f} "
            f"clamp=({float(args.task_balance_min_weight):.3f},{float(args.task_balance_max_weight):.3f})"
        )
    if explain_cf_weight_runtime > 0 or cf_sched_active:
        print(
            f"[info] explain counterfactual: weight={float(explain_cf_weight_runtime):.4f} "
            f"margin={float(args.explain_counterfactual_margin):.4f} "
            f"mode={args.explain_counterfactual_mode} pairing={args.explain_counterfactual_pairing}"
        )
    while step < args.max_steps:
        for batch in train_dl:
            cur_region_beta, cur_cf_weight = _update_runtime_schedule(int(step), force_log=False)
            if delayed_llm_unfreeze and int(step) >= int(args.unfreeze_llm_after_step):
                newly_trainable = _apply_llm_unfreeze_targets()
                newly_added = _add_new_trainables_to_optimizer()
                delayed_llm_unfreeze = False
                print(
                    f"[info] delayed llm unfreeze triggered at step={int(step)} "
                    f"params_unfrozen={int(newly_trainable)} params_added_to_optimizer={int(newly_added)}"
                )
            images = batch["images"]
            pixel_values = student.preprocess(list(images))["pixel_values"].to(device)
            if args.image_dropout_prob > 0:
                drop_mask = torch.rand(pixel_values.size(0), device=pixel_values.device) < args.image_dropout_prob
                if drop_mask.any():
                    if args.image_dropout_mode == "noise":
                        noise = torch.randn_like(pixel_values) * float(args.image_dropout_sigma)
                        pixel_values = torch.where(drop_mask[:, None, None, None], noise, pixel_values)
                    else:
                        zero = torch.zeros_like(pixel_values)
                        pixel_values = torch.where(drop_mask[:, None, None, None], zero, pixel_values)
            if args.context_mix_mode != "none":
                texts = batch.get("texts", None)
                scales = batch.get("scales", None)
                contexts = batch.get("contexts", None)
                regions = batch.get("regions", None)
                if texts is not None and scales is not None and contexts is not None and regions is not None and len(texts) > 0:
                    mixed_records = []
                    mixed_modes: List[str] = []
                    bsz_local = len(texts)
                    for bi in range(bsz_local):
                        u = random.random()
                        if u < args.context_region_only_prob:
                            mode = "region_only"
                        elif u < (args.context_region_only_prob + args.context_region_ctx_prob):
                            mode = "region_ctx"
                        else:
                            mode = "paragraph_only"
                        mixed_modes.append(mode)
                        text_i = texts[bi]
                        scale_i = scales[bi]
                        ctx_i = (contexts[bi] or "").strip()
                        regs_i = list(regions[bi] or [])
                        if mode == "region_only":
                            ctx_i = ""
                        elif mode == "paragraph_only":
                            regs_i = []
                        elif mode == "region_ctx" and args.context_shuffle_prob > 0 and bsz_local > 1 and ctx_i:
                            if random.random() < args.context_shuffle_prob:
                                cand = [j for j in range(bsz_local) if j != bi and (contexts[j] or "").strip()]
                                if cand:
                                    j = random.choice(cand)
                                    ctx_i = (contexts[j] or "").strip()
                        mixed_records.append((images[bi], text_i, scale_i, ctx_i, regs_i))
                    mixed_pack = build_batch(
                        tokenizer,
                        mixed_records,
                        args.max_length,
                        {
                            "short": args.max_target_short,
                            "long": args.max_target_long,
                            "desc": args.max_target_desc,
                            "explain": args.max_target_long,
                        },
                        model.config.image_token_index,
                        add_eos=True,
                        fixed_task=args.fixed_task or None,
                        context_dropout=0.0,
                        paragraph_token_dropout=args.paragraph_token_dropout,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,
                        explain_ctx_min_adesc_tokens=args.explain_ctx_min_adesc_tokens,
                        explain_ctx_max_ocr_tokens=args.explain_ctx_max_ocr_tokens,
                        bucket_bins=bucket_bins if bucket_bins else None,
                        image_tokens=image_tokens_total,
                        scicap_prompt_style=args.scicap_prompt_style,
                        scicap_task_context_routing=args.scicap_task_context_routing,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                        use_context_placeholders=args.use_context_placeholders,
                    )
                    mixed_pack["images"] = images
                    mixed_pack["mix_modes"] = mixed_modes
                    batch = mixed_pack

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_regions = batch.get("regions", None)
            mix_modes = batch.get("mix_modes", None)
            if mix_modes is None:
                mix_modes = ["default"] * int(input_ids.size(0))
            paragraph_only_mask = torch.tensor([m == "paragraph_only" for m in mix_modes], device=device, dtype=torch.bool)
            region_only_mask = torch.tensor([m == "region_only" for m in mix_modes], device=device, dtype=torch.bool)
            sample_loss_weights = torch.ones(int(input_ids.size(0)), device=device, dtype=torch.float32)
            if args.context_mix_mode != "none" and paragraph_only_mask.any():
                sample_loss_weights[paragraph_only_mask] = float(max(0.0, args.paragraph_only_weight))
            vision_wrapper.set_regions(batch_regions, drop_one_region=False)
            scales_now = batch.get("scales", None)
            if scales_now is None:
                explain_mask = torch.zeros(int(input_ids.size(0)), device=device, dtype=torch.bool)
            else:
                explain_mask = torch.tensor(
                    [str(s or "") == "explain" for s in scales_now],
                    device=device,
                    dtype=torch.bool,
                )
            cap_task_masks: Dict[str, torch.Tensor] = {}
            task_masks_all: Dict[str, torch.Tensor] = {}
            if scales_now is not None:
                for _tk in ("short", "long", "desc"):
                    m_tk = torch.tensor(
                        [str(s or "") == _tk for s in scales_now],
                        device=device,
                        dtype=torch.bool,
                    )
                    cap_task_masks[_tk] = m_tk
                    task_masks_all[_tk] = m_tk
                task_masks_all["explain"] = explain_mask
            use_explain_region_required = bool(args.explain_region_required_loss and explain_mask.any())
            explain_base_det = None
            explain_drop_det = None
            explain_gap_det = None
            explain_cf_base_det = None
            explain_cf_wrong_det = None
            explain_cf_pen_det = None
            desc_anchor_loss_det = None
            desc_anchor_active_ratio = None
            desc_anchor_desc_count = 0
            desc_nonalias_loss_det = None
            task_balance_weights_det: Optional[Dict[str, float]] = None

            # 1) loss_i forward -> backward -> release
            mean_logits_i_det = None
            loss_nll_explain = None
            _set_region_attention_bias_for_inputs(
                input_ids,
                batch.get("scales", None),
                batch_regions,
                labels_local=labels,
                contexts_local=batch.get("contexts_used", batch.get("contexts", None)),
                context_allow_tokens_local=batch.get("context_allow_tokens", None),
                context_total_tokens_local=batch.get("context_total_tokens", None),
                context_ocr_tokens_local=batch.get("context_ocr_tokens", None),
                context_adesc_tokens_local=batch.get("context_adesc_tokens", None),
                context_para_tokens_local=batch.get("context_para_tokens", None),
            )
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                _, pos_ids, attn_mask, past_kv, inputs_embeds, labels_mm = model.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=labels,
                    images=pixel_values,
                    image_sizes=None,
                )
                out = model.language_model(
                    input_ids=None,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=past_kv,
                    inputs_embeds=inputs_embeds,
                    labels=labels_mm,
                    use_cache=False,
                )
                nll_ps = nll_per_sample(out.logits, labels_mm)
                if explain_mask.any():
                    loss_nll_explain = nll_ps[explain_mask].mean()
                if args.task_balance_mode != "none" and task_masks_all:
                    task_losses: Dict[str, torch.Tensor] = {}
                    for _tk, _mask in task_masks_all.items():
                        if _mask is None or (not bool(_mask.any())):
                            continue
                        if sample_loss_weights is not None:
                            w_t = sample_loss_weights[_mask]
                            denom_t = w_t.sum().clamp(min=1e-6)
                            task_losses[_tk] = (nll_ps[_mask] * w_t).sum() / denom_t
                        else:
                            task_losses[_tk] = nll_ps[_mask].mean()
                    if task_losses:
                        prior_weights = {
                            "short": max(0.0, float(args.scicap_loss_w_short)) if args.scicap_fixed_task_loss else 1.0,
                            "long": max(0.0, float(args.scicap_loss_w_long)) if args.scicap_fixed_task_loss else 1.0,
                            "desc": max(0.0, float(args.scicap_loss_w_desc)) if args.scicap_fixed_task_loss else 1.0,
                            "explain": 1.0,
                        }
                        ema_m = float(min(max(args.task_balance_ema, 0.0), 0.9999))
                        alpha = float(max(0.0, args.task_balance_alpha))
                        min_w = float(max(0.0, args.task_balance_min_weight))
                        max_w = float(max(min_w, args.task_balance_max_weight))
                        dyn_raw: Dict[str, float] = {}
                        for _tk, _loss_t in task_losses.items():
                            cur_v = float(_loss_t.detach().item())
                            prev_v = task_loss_ema.get(_tk, cur_v)
                            ema_v = (ema_m * float(prev_v)) + ((1.0 - ema_m) * cur_v)
                            task_loss_ema[_tk] = ema_v
                            dyn_raw[_tk] = (1.0 / max(ema_v, 1e-6)) ** alpha if alpha > 0 else 1.0
                        dyn_mean = sum(dyn_raw.values()) / float(max(1, len(dyn_raw)))
                        loss_terms: List[torch.Tensor] = []
                        weight_terms: List[float] = []
                        task_balance_weights_det = {}
                        for _tk, _loss_t in task_losses.items():
                            prior = float(prior_weights.get(_tk, 1.0))
                            if prior <= 0.0:
                                continue
                            dyn = dyn_raw.get(_tk, 1.0)
                            dyn = dyn / max(dyn_mean, 1e-6)
                            dyn = min(max_w, max(min_w, dyn))
                            wt = prior * dyn
                            if wt <= 0.0:
                                continue
                            loss_terms.append(_loss_t * wt)
                            weight_terms.append(float(wt))
                            task_balance_weights_det[_tk] = float(wt)
                        if loss_terms and weight_terms:
                            loss_nll = torch.stack(loss_terms).sum() / max(sum(weight_terms), 1e-6)
                        else:
                            if sample_loss_weights is not None:
                                denom = sample_loss_weights.sum().clamp(min=1e-6)
                                loss_nll = (nll_ps * sample_loss_weights).sum() / denom
                            else:
                                loss_nll = nll_ps.mean()
                        if "explain" in task_losses:
                            loss_nll_explain = task_losses["explain"]
                    else:
                        if sample_loss_weights is not None:
                            denom = sample_loss_weights.sum().clamp(min=1e-6)
                            loss_nll = (nll_ps * sample_loss_weights).sum() / denom
                        else:
                            loss_nll = nll_ps.mean()
                elif args.scicap_fixed_task_loss and cap_task_masks:
                    task_loss_weights = {
                        "short": max(0.0, float(args.scicap_loss_w_short)),
                        "long": max(0.0, float(args.scicap_loss_w_long)),
                        "desc": max(0.0, float(args.scicap_loss_w_desc)),
                    }
                    task_terms: List[torch.Tensor] = []
                    for _tk in ("short", "long", "desc"):
                        m = cap_task_masks.get(_tk)
                        if m is None or (not bool(m.any())):
                            continue
                        w_task = float(task_loss_weights.get(_tk, 1.0))
                        if w_task <= 0.0:
                            continue
                        if sample_loss_weights is not None:
                            w = sample_loss_weights[m]
                            denom_t = w.sum().clamp(min=1e-6)
                            task_terms.append(((nll_ps[m] * w).sum() / denom_t) * w_task)
                        else:
                            task_terms.append(nll_ps[m].mean() * w_task)
                    if task_terms:
                        loss_nll = torch.stack(task_terms).sum()
                        if loss_nll_explain is not None:
                            # Keep explain branch independent when mixed with caption tasks.
                            loss_nll = loss_nll + loss_nll_explain
                    else:
                        if sample_loss_weights is not None:
                            denom = sample_loss_weights.sum().clamp(min=1e-6)
                            loss_nll = (nll_ps * sample_loss_weights).sum() / denom
                        else:
                            loss_nll = nll_ps.mean()
                else:
                    if sample_loss_weights is not None:
                        denom = sample_loss_weights.sum().clamp(min=1e-6)
                        loss_nll = (nll_ps * sample_loss_weights).sum() / denom
                    else:
                        loss_nll = nll_ps.mean()
                loss_i_det = loss_nll.detach()
                if args.contrastive_weight > 0 and args.contrastive_type == "mask_drop":
                    mean_logits_i_det = mean_logits_per_sample(out.logits, labels_mm).detach()
                loss_i = loss_nll
                # Caption/Description generation-space shaping:
                # - caption: suppress formula-like continuation
                # - caption: keep minimal context-evidence coverage
                # - description: keep minimal OCR-evidence coverage
                caption_mask = None
                desc_mask = None
                if cap_task_masks:
                    m_short = cap_task_masks.get("short")
                    m_long = cap_task_masks.get("long")
                    m_desc = cap_task_masks.get("desc")
                    if m_short is not None and m_long is not None:
                        caption_mask = (m_short | m_long)
                    desc_mask = m_desc
                need_prob_space = bool(
                    (args.caption_formula_penalty_weight > 0 and caption_formula_token_ids)
                    or (args.caption_context_cov_weight > 0)
                    or (args.desc_ocr_cov_weight > 0)
                    or (args.desc_entity_anchor_weight > 0)
                    or (args.desc_entity_nonalias_penalty_weight > 0)
                    or (args.desc_struct_cov_weight > 0)
                    or (args.desc_prompt_leak_penalty_weight > 0 and desc_prompt_leak_token_ids)
                )
                if need_prob_space and out.logits.size(1) > 1:
                    shift_logits = out.logits[:, :-1, :]
                    shift_labels = labels_mm[:, 1:]
                    valid_token_mask = shift_labels.ne(-100)
                    probs = torch.softmax(shift_logits, dim=-1)

                    if (
                        args.caption_formula_penalty_weight > 0
                        and caption_formula_token_ids
                        and caption_mask is not None
                        and bool(caption_mask.any())
                    ):
                        cap_tok_mask = valid_token_mask & caption_mask.unsqueeze(1)
                        if bool(cap_tok_mask.any()):
                            formula_ids_t = torch.tensor(
                                caption_formula_token_ids,
                                device=probs.device,
                                dtype=torch.long,
                            )
                            formula_mass = probs.index_select(dim=-1, index=formula_ids_t).sum(dim=-1)
                            formula_pen = (formula_mass * cap_tok_mask.to(formula_mass.dtype)).sum() / cap_tok_mask.sum().clamp(min=1)
                            loss_i = loss_i + float(args.caption_formula_penalty_weight) * formula_pen

                    if args.caption_context_cov_weight > 0 and caption_mask is not None and bool(caption_mask.any()):
                        ctxs = batch.get("contexts_used", batch.get("contexts", None))
                        if isinstance(ctxs, list):
                            cap_cov_terms: List[torch.Tensor] = []
                            for bi in range(int(input_ids.size(0))):
                                if not bool(caption_mask[bi].item()):
                                    continue
                                token_ids = _coverage_token_ids_from_text(
                                    tokenizer,
                                    str(ctxs[bi] if bi < len(ctxs) else ""),
                                    max_ids=int(max(1, args.caption_context_cov_max_ids)),
                                )
                                if not token_ids:
                                    continue
                                pos_mask = valid_token_mask[bi]
                                if not bool(pos_mask.any()):
                                    continue
                                tid = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
                                mass = probs[bi][pos_mask].index_select(dim=-1, index=tid).sum(dim=-1).mean()
                                cap_cov_terms.append(torch.relu(torch.tensor(float(args.caption_context_cov_min), device=probs.device) - mass))
                            if cap_cov_terms:
                                loss_i = loss_i + float(args.caption_context_cov_weight) * torch.stack(cap_cov_terms).mean()

                    if args.desc_ocr_cov_weight > 0 and desc_mask is not None and bool(desc_mask.any()):
                        ctxs_ocr = batch.get("contexts_ocr", None)
                        ctxs_fallback = batch.get("contexts_used", batch.get("contexts", None))
                        desc_cov_terms: List[torch.Tensor] = []
                        for bi in range(int(input_ids.size(0))):
                            if not bool(desc_mask[bi].item()):
                                continue
                            src_text = ""
                            if isinstance(ctxs_ocr, list) and bi < len(ctxs_ocr):
                                src_text = str(ctxs_ocr[bi] or "")
                            if not src_text and isinstance(ctxs_fallback, list) and bi < len(ctxs_fallback):
                                src_text = str(ctxs_fallback[bi] or "")
                            token_ids = _coverage_token_ids_from_text(
                                tokenizer,
                                src_text,
                                max_ids=int(max(1, args.desc_ocr_cov_max_ids)),
                            )
                            if not token_ids:
                                continue
                            pos_mask = valid_token_mask[bi]
                            if not bool(pos_mask.any()):
                                continue
                            tid = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
                            mass = probs[bi][pos_mask].index_select(dim=-1, index=tid).sum(dim=-1).mean()
                            desc_cov_terms.append(torch.relu(torch.tensor(float(args.desc_ocr_cov_min), device=probs.device) - mass))
                        if desc_cov_terms:
                            loss_i = loss_i + float(args.desc_ocr_cov_weight) * torch.stack(desc_cov_terms).mean()

                    if (
                        (args.desc_entity_anchor_weight > 0 or args.desc_entity_nonalias_penalty_weight > 0)
                        and desc_mask is not None
                        and bool(desc_mask.any())
                    ):
                        ctxs_ocr = batch.get("contexts_ocr", None)
                        ctxs_fallback = batch.get("contexts_used", batch.get("contexts", None))
                        ctxs_struct_nodes = batch.get("contexts_struct_nodes", None)
                        desc_ent_terms: List[torch.Tensor] = []
                        desc_nonalias_terms: List[torch.Tensor] = []
                        active_count = 0
                        desc_count = 0
                        for bi in range(int(input_ids.size(0))):
                            if not bool(desc_mask[bi].item()):
                                continue
                            desc_count += 1
                            src_text = ""
                            if isinstance(ctxs_ocr, list) and bi < len(ctxs_ocr):
                                src_text = str(ctxs_ocr[bi] or "")
                            if not src_text and isinstance(ctxs_fallback, list) and bi < len(ctxs_fallback):
                                src_text = str(ctxs_fallback[bi] or "")
                            struct_nodes: List[str] = []
                            if isinstance(ctxs_struct_nodes, list) and bi < len(ctxs_struct_nodes):
                                maybe_nodes = ctxs_struct_nodes[bi]
                                if isinstance(maybe_nodes, (list, tuple)):
                                    struct_nodes = [str(x or "") for x in maybe_nodes]
                            alias_texts = _build_desc_anchor_alias_texts(
                                ocr_text=src_text,
                                struct_nodes=struct_nodes,
                                max_items=int(max(1, args.desc_entity_anchor_max_items)),
                                max_alias_per_item=int(max(1, args.desc_entity_anchor_alias_max_per_item)),
                            )
                            if not alias_texts:
                                continue
                            pos_mask = valid_token_mask[bi]
                            if not bool(pos_mask.any()):
                                continue
                            alias_events: List[torch.Tensor] = []
                            alias_token_ids_set = set()
                            alias_raw_token_ids_set = set()
                            alias_shape_keys: set[str] = set()
                            for ent in alias_texts:
                                alias_shape_keys.update(_shape_keys_from_alias_text(ent))
                                try:
                                    raw_ids = tokenizer("\n" + str(ent), add_special_tokens=False)["input_ids"]
                                except Exception:
                                    raw_ids = []
                                for rid in raw_ids:
                                    alias_raw_token_ids_set.add(int(rid))
                                token_ids = _coverage_token_ids_from_text(
                                    tokenizer,
                                    ent,
                                    max_ids=int(max(1, args.desc_entity_anchor_max_ids_per_item)),
                                )
                                if not token_ids:
                                    continue
                                for tid in token_ids:
                                    alias_token_ids_set.add(int(tid))
                                tid = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
                                # Event-style anchor: require at least one decoding position
                                # to allocate mass to any alias token.
                                event_mass = probs[bi][pos_mask].index_select(dim=-1, index=tid).sum(dim=-1).max()
                                alias_events.append(event_mass)
                            if alias_events and args.desc_entity_anchor_weight > 0:
                                event_scores = torch.stack(alias_events)
                                topk = int(max(1, args.desc_entity_anchor_topk))
                                topk = min(topk, int(event_scores.numel()))
                                best_mass = torch.topk(event_scores, k=topk, largest=True).values.mean()
                                desc_ent_terms.append(
                                    torch.relu(torch.tensor(float(args.desc_entity_anchor_min), device=probs.device) - best_mass)
                                )
                                active_count += 1

                            if (
                                args.desc_entity_nonalias_penalty_weight > 0
                                and desc_entity_like_by_shape_t
                                and alias_shape_keys
                            ):
                                nonalias_pos_mask = pos_mask
                                if (
                                    args.desc_entity_nonalias_only_after_cue
                                    and desc_entity_cue_token_ids_t is not None
                                    and int(desc_entity_cue_token_ids_t.numel()) > 0
                                ):
                                    # Use label-space previous tokens (labels_mm[:, :-1]) so
                                    # the cue mask stays aligned with shift_logits/shift_labels.
                                    prev_ids = labels_mm[bi, :-1]
                                    cue_mask = torch.isin(prev_ids, desc_entity_cue_token_ids_t)
                                    nonalias_pos_mask = nonalias_pos_mask & cue_mask
                                if bool(nonalias_pos_mask.any()):
                                    pools: List[torch.Tensor] = []
                                    for sk in alias_shape_keys:
                                        tid_t = desc_entity_like_by_shape_t.get(sk)
                                        if tid_t is not None and int(tid_t.numel()) > 0:
                                            pools.append(tid_t)
                                    if not pools:
                                        continue
                                    nonalias_ids_t = torch.unique(torch.cat(pools, dim=0))
                                    exclude_ids = set(alias_token_ids_set) | set(alias_raw_token_ids_set)
                                    if exclude_ids:
                                        exclude_t = torch.tensor(
                                            sorted(int(x) for x in exclude_ids),
                                            device=probs.device,
                                            dtype=torch.long,
                                        )
                                        keep_mask = ~torch.isin(nonalias_ids_t, exclude_t)
                                        nonalias_ids_t = nonalias_ids_t[keep_mask]
                                    if int(nonalias_ids_t.numel()) > 0:
                                        nonalias_mass = (
                                            probs[bi][nonalias_pos_mask]
                                            .index_select(dim=-1, index=nonalias_ids_t)
                                            .sum(dim=-1)
                                            .mean()
                                        )
                                        desc_nonalias_terms.append(
                                            torch.relu(
                                                nonalias_mass
                                                - torch.tensor(
                                                    float(args.desc_entity_nonalias_max),
                                                    device=probs.device,
                                                )
                                            )
                                        )
                        desc_anchor_desc_count = int(desc_count)
                        if desc_count > 0:
                            desc_anchor_active_ratio = torch.tensor(
                                float(active_count) / float(desc_count),
                                device=probs.device,
                            ).detach()
                        if desc_ent_terms and args.desc_entity_anchor_weight > 0:
                            desc_anchor_mean = torch.stack(desc_ent_terms).mean()
                            desc_anchor_loss_det = desc_anchor_mean.detach()
                            loss_i = loss_i + float(args.desc_entity_anchor_weight) * desc_anchor_mean
                        if desc_nonalias_terms and args.desc_entity_nonalias_penalty_weight > 0:
                            desc_nonalias_mean = torch.stack(desc_nonalias_terms).mean()
                            desc_nonalias_loss_det = desc_nonalias_mean.detach()
                            loss_i = loss_i + float(args.desc_entity_nonalias_penalty_weight) * desc_nonalias_mean

                    if (
                        args.desc_prompt_leak_penalty_weight > 0
                        and desc_prompt_leak_token_ids
                        and desc_mask is not None
                        and bool(desc_mask.any())
                    ):
                        desc_tok_mask = valid_token_mask & desc_mask.unsqueeze(1)
                        if bool(desc_tok_mask.any()):
                            leak_ids_t = torch.tensor(
                                desc_prompt_leak_token_ids,
                                device=probs.device,
                                dtype=torch.long,
                            )
                            leak_mass = probs.index_select(dim=-1, index=leak_ids_t).sum(dim=-1)
                            leak_pen = (leak_mass * desc_tok_mask.to(leak_mass.dtype)).sum() / desc_tok_mask.sum().clamp(min=1)
                            loss_i = loss_i + float(args.desc_prompt_leak_penalty_weight) * leak_pen

                    if args.desc_struct_cov_weight > 0 and desc_mask is not None and bool(desc_mask.any()):
                        gt_texts = batch.get("texts", None)
                        desc_struct_terms: List[torch.Tensor] = []
                        ratio = max(0.1, min(1.0, float(args.desc_struct_cov_target_ratio)))
                        for bi in range(int(input_ids.size(0))):
                            if not bool(desc_mask[bi].item()):
                                continue
                            gt_text = ""
                            if isinstance(gt_texts, list) and bi < len(gt_texts):
                                gt_text = str(gt_texts[bi] or "")
                            slot_texts = _extract_desc_slot_texts(
                                gt_text,
                                max_slots=int(max(1, args.desc_struct_cov_max_slots)),
                            )
                            if not slot_texts:
                                continue
                            pos_mask = valid_token_mask[bi]
                            if not bool(pos_mask.any()):
                                continue
                            slot_masses: List[torch.Tensor] = []
                            for slot_text in slot_texts:
                                token_ids = _coverage_token_ids_from_text(
                                    tokenizer,
                                    slot_text,
                                    max_ids=int(max(1, args.desc_struct_cov_max_ids_per_slot)),
                                )
                                if not token_ids:
                                    continue
                                tid = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
                                mass = probs[bi][pos_mask].index_select(dim=-1, index=tid).sum(dim=-1).mean()
                                slot_masses.append(mass)
                            if not slot_masses:
                                continue
                            mass_vec = torch.stack(slot_masses)
                            k = max(1, int(math.ceil(float(mass_vec.numel()) * ratio)))
                            k = min(k, int(mass_vec.numel()))
                            topk = torch.topk(mass_vec, k=k, largest=True).values
                            agg_mass = topk.mean()
                            desc_struct_terms.append(
                                torch.relu(torch.tensor(float(args.desc_struct_cov_min), device=probs.device) - agg_mass)
                            )
                        if desc_struct_terms:
                            loss_i = loss_i + float(args.desc_struct_cov_weight) * torch.stack(desc_struct_terms).mean()

                if args.consistency_weight > 0 and args.consistency_mode == "anchor":
                    log_margin = math.log(max(args.consistency_margin, 1e-6))
                    terms = []
                    if args.anchor_type:
                        p_type = keyword_presence(out.logits, type_keyword_ids)
                        terms.append(torch.clamp(log_margin - p_type, min=0.0))
                    if args.anchor_struct:
                        p_struct = keyword_presence(out.logits, struct_keyword_ids)
                        terms.append(torch.clamp(log_margin - p_struct, min=0.0))
                    if args.anchor_rel:
                        p_rel = keyword_presence(out.logits, rel_keyword_ids)
                        terms.append(torch.clamp(log_margin - p_rel, min=0.0))
                    if terms:
                        loss_anchor_vec = torch.stack(terms, dim=0).mean(dim=0)
                        if args.context_mix_mode != "none" and args.anchor_region_only_scale != 1.0 and region_only_mask.any():
                            scale_vec = torch.ones_like(loss_anchor_vec)
                            scale_vec = scale_vec + region_only_mask.to(loss_anchor_vec.dtype) * float(args.anchor_region_only_scale - 1.0)
                            loss_anchor_vec = loss_anchor_vec * scale_vec
                        # Stage-4 correction: anchor loss only supervises EXPLAIN samples.
                        if explain_mask.any():
                            loss_anchor = loss_anchor_vec[explain_mask].mean()
                            loss_i = loss_i + args.consistency_weight * loss_anchor
            _clear_region_attention_bias()

            if (
                cur_cf_weight > 0
                and bool(explain_mask.any())
                and int(explain_mask.sum().item()) > 1
            ):
                exp_idx = torch.nonzero(explain_mask, as_tuple=False).view(-1).detach().cpu().tolist()
                if len(exp_idx) > 1:
                    perm_idx = _build_explain_cf_perm(
                        exp_idx,
                        batch.get("contexts_used", batch.get("contexts", None)),
                        mode=str(args.explain_counterfactual_pairing),
                    )
                    pixel_values_cf = pixel_values.clone()
                    pixel_values_cf[exp_idx] = pixel_values[perm_idx]
                    regions_cf = batch_regions
                    if batch_regions is not None:
                        regions_cf = list(batch_regions)
                        for dst_i, src_i in zip(exp_idx, perm_idx):
                            regions_cf[dst_i] = batch_regions[src_i]
                    vision_wrapper.set_regions(regions_cf, drop_one_region=False)
                    _set_region_attention_bias_for_inputs(
                        input_ids,
                        batch.get("scales", None),
                        regions_cf,
                        labels_local=labels,
                        contexts_local=batch.get("contexts_used", batch.get("contexts", None)),
                        context_allow_tokens_local=batch.get("context_allow_tokens", None),
                        context_total_tokens_local=batch.get("context_total_tokens", None),
                        context_ocr_tokens_local=batch.get("context_ocr_tokens", None),
                        context_adesc_tokens_local=batch.get("context_adesc_tokens", None),
                        context_para_tokens_local=batch.get("context_para_tokens", None),
                    )
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_cf, attn_mask_cf, past_kv_cf, inputs_embeds_cf, labels_mm_cf = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids,
                            position_ids=None,
                            attention_mask=attention_mask,
                            past_key_values=None,
                            labels=labels,
                            images=pixel_values_cf,
                            image_sizes=None,
                        )
                        out_cf = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_cf,
                            position_ids=pos_ids_cf,
                            past_key_values=past_kv_cf,
                            inputs_embeds=inputs_embeds_cf,
                            labels=labels_mm_cf,
                            use_cache=False,
                        )
                        nll_ps_cf = nll_per_sample(out_cf.logits, labels_mm_cf)
                        loss_cf_wrong = nll_ps_cf[explain_mask].mean()
                        base_explain = loss_nll_explain if loss_nll_explain is not None else nll_ps[explain_mask].mean()
                        if args.explain_counterfactual_mode == "continuous":
                            cf_pen = torch.relu(base_explain - loss_cf_wrong)
                        else:
                            cf_margin = torch.tensor(
                                float(args.explain_counterfactual_margin),
                                device=loss_cf_wrong.device,
                                dtype=loss_cf_wrong.dtype,
                            )
                            cf_pen = torch.relu(cf_margin + base_explain - loss_cf_wrong)
                        loss_i = loss_i + float(cur_cf_weight) * cf_pen
                    explain_cf_base_det = base_explain.detach()
                    explain_cf_wrong_det = loss_cf_wrong.detach()
                    explain_cf_pen_det = cf_pen.detach()
                    del out_cf, inputs_embeds_cf, labels_mm_cf, attn_mask_cf, pos_ids_cf, past_kv_cf
                    torch.cuda.empty_cache()
                    _clear_region_attention_bias()
                    vision_wrapper.set_regions(batch_regions, drop_one_region=False)

            if use_explain_region_required and loss_nll_explain is not None:
                bsz_local = int(input_ids.size(0))
                regions_drop = [[] for _ in range(bsz_local)]
                explain_region_visible_drop = [not bool(explain_mask[i].item()) for i in range(bsz_local)]

                # 1) Lightweight hinge activation pass (no grad): avoid keeping two full
                # computation graphs in memory at the same time.
                vision_wrapper.set_regions(regions_drop, drop_one_region=False)
                _set_region_attention_bias_for_inputs(
                    input_ids,
                    batch.get("scales", None),
                    regions_drop,
                    labels_local=labels,
                    contexts_local=batch.get("contexts_used", batch.get("contexts", None)),
                    context_allow_tokens_local=batch.get("context_allow_tokens", None),
                    context_total_tokens_local=batch.get("context_total_tokens", None),
                    context_ocr_tokens_local=batch.get("context_ocr_tokens", None),
                    context_adesc_tokens_local=batch.get("context_adesc_tokens", None),
                    context_para_tokens_local=batch.get("context_para_tokens", None),
                    explain_region_visible=explain_region_visible_drop,
                )
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_drop_d, attn_mask_drop_d, past_kv_drop_d, inputs_embeds_drop_d, labels_mm_drop_d = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids,
                            position_ids=None,
                            attention_mask=attention_mask,
                            past_key_values=None,
                            labels=labels,
                            images=pixel_values,
                            image_sizes=None,
                        )
                        out_drop_d = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_drop_d,
                            position_ids=pos_ids_drop_d,
                            past_key_values=past_kv_drop_d,
                            inputs_embeds=inputs_embeds_drop_d,
                            labels=labels_mm_drop_d,
                            use_cache=False,
                        )
                        nll_ps_drop_d = nll_per_sample(out_drop_d.logits, labels_mm_drop_d)
                        loss_drop_explain_det = nll_ps_drop_d[explain_mask].mean()
                _clear_region_attention_bias()
                vision_wrapper.set_regions(batch_regions, drop_one_region=False)
                explain_base_det = loss_nll_explain.detach()
                explain_drop_det = loss_drop_explain_det.detach()
                explain_gap_det = (loss_drop_explain_det - loss_nll_explain.detach()).detach()
                # Keep pushing until region-drop is worse than base by a margin.
                gap_margin = float(args.explain_region_required_margin)
                gap_active = 1.0 if float(explain_gap_det.item()) <= gap_margin else 0.0
                lambda_eff = float(args.explain_region_required_lambda) * gap_active
                del out_drop_d, inputs_embeds_drop_d, labels_mm_drop_d, attn_mask_drop_d, pos_ids_drop_d, past_kv_drop_d
                torch.cuda.empty_cache()

                # 2) Base backward for: L = L_base + lambda * relu(L_base - L_drop).
                base_term = loss_i + lambda_eff * loss_nll_explain
                base_term_scaled = base_term / max(1, args.grad_accum)
                if use_scaler:
                    scaler.scale(base_term_scaled).backward()
                else:
                    base_term_scaled.backward()
                del out, inputs_embeds, labels_mm, attn_mask, pos_ids, past_kv
                torch.cuda.empty_cache()

                # 3) Region-drop backward only when hinge is active.
                if lambda_eff > 0.0:
                    vision_wrapper.set_regions(regions_drop, drop_one_region=False)
                    _set_region_attention_bias_for_inputs(
                        input_ids,
                        batch.get("scales", None),
                        regions_drop,
                        labels_local=labels,
                        contexts_local=batch.get("contexts_used", batch.get("contexts", None)),
                        context_allow_tokens_local=batch.get("context_allow_tokens", None),
                        context_total_tokens_local=batch.get("context_total_tokens", None),
                        context_ocr_tokens_local=batch.get("context_ocr_tokens", None),
                        context_adesc_tokens_local=batch.get("context_adesc_tokens", None),
                        context_para_tokens_local=batch.get("context_para_tokens", None),
                        explain_region_visible=explain_region_visible_drop,
                    )
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_drop, attn_mask_drop, past_kv_drop, inputs_embeds_drop, labels_mm_drop = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids,
                            position_ids=None,
                            attention_mask=attention_mask,
                            past_key_values=None,
                            labels=labels,
                            images=pixel_values,
                            image_sizes=None,
                        )
                        out_drop = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_drop,
                            position_ids=pos_ids_drop,
                            past_key_values=past_kv_drop,
                            inputs_embeds=inputs_embeds_drop,
                            labels=labels_mm_drop,
                            use_cache=False,
                        )
                        nll_ps_drop = nll_per_sample(out_drop.logits, labels_mm_drop)
                        loss_drop_explain = nll_ps_drop[explain_mask].mean()
                        # Negative sign pushes region-drop NLL up when hinge is active.
                        drop_term_scaled = (-lambda_eff * loss_drop_explain) / max(1, args.grad_accum)
                    if use_scaler:
                        scaler.scale(drop_term_scaled).backward()
                    else:
                        drop_term_scaled.backward()
                    explain_drop_det = loss_drop_explain.detach()
                    explain_gap_det = (loss_drop_explain - loss_nll_explain).detach()
                    del out_drop, inputs_embeds_drop, labels_mm_drop, attn_mask_drop, pos_ids_drop, past_kv_drop
                    torch.cuda.empty_cache()
                    _clear_region_attention_bias()
                    vision_wrapper.set_regions(batch_regions, drop_one_region=False)
            else:
                loss_i_scaled = loss_i / max(1, args.grad_accum)
                if use_scaler:
                    scaler.scale(loss_i_scaled).backward()
                else:
                    loss_i_scaled.backward()
                del out, inputs_embeds, labels_mm, attn_mask, pos_ids, past_kv
                torch.cuda.empty_cache()

            # 2) contrastive forward -> backward -> release
            if args.contrastive_weight > 0 and not use_explain_region_required:
                loss_contrast = None
                if args.contrastive_type == "mask_drop":
                    if mean_logits_i_det is not None and batch_regions is not None:
                        valid_regions = torch.tensor(
                            [1 if (regs and len(regs) > 0) else 0 for regs in batch_regions],
                            device=device,
                            dtype=torch.bool,
                        )
                        if valid_regions.any():
                            vision_wrapper.set_regions(batch_regions, drop_one_region=True)
                            _set_region_attention_bias_for_inputs(
                                input_ids,
                                batch.get("scales", None),
                                batch_regions,
                                labels_local=labels,
                                contexts_local=batch.get("contexts_used", batch.get("contexts", None)),
                                context_allow_tokens_local=batch.get("context_allow_tokens", None),
                                context_total_tokens_local=batch.get("context_total_tokens", None),
                                context_ocr_tokens_local=batch.get("context_ocr_tokens", None),
                                context_adesc_tokens_local=batch.get("context_adesc_tokens", None),
                                context_para_tokens_local=batch.get("context_para_tokens", None),
                            )
                            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                                _, pos_ids_d, attn_mask_d, past_kv_d, inputs_embeds_d, labels_mm_d = model.prepare_inputs_labels_for_multimodal(
                                    input_ids=input_ids,
                                    position_ids=None,
                                    attention_mask=attention_mask,
                                    past_key_values=None,
                                    labels=labels,
                                    images=pixel_values,
                                    image_sizes=None,
                                )
                                out_drop = model.language_model(
                                    input_ids=None,
                                    attention_mask=attn_mask_d,
                                    position_ids=pos_ids_d,
                                    past_key_values=past_kv_d,
                                    inputs_embeds=inputs_embeds_d,
                                    labels=labels_mm_d,
                                    use_cache=False,
                                )
                                mean_logits_drop = mean_logits_per_sample(out_drop.logits, labels_mm_d)
                            _clear_region_attention_bias()
                            cos = F.cosine_similarity(mean_logits_drop, mean_logits_i_det, dim=-1)
                            cos = cos[valid_regions]
                            if args.contrastive_mode == "cosine":
                                contrastive = torch.clamp(cos - args.contrastive_margin, min=0.0)
                            else:
                                contrastive = cos
                            loss_contrast_drop = args.contrastive_weight * contrastive.mean()
                            loss_contrast_scaled = loss_contrast_drop / max(1, args.grad_accum)
                            if use_scaler:
                                scaler.scale(loss_contrast_scaled).backward()
                            else:
                                loss_contrast_scaled.backward()
                            del out_drop, inputs_embeds_d, labels_mm_d, attn_mask_d, pos_ids_d, past_kv_d
                            torch.cuda.empty_cache()
                        vision_wrapper.set_regions(batch_regions, drop_one_region=False)
                elif args.batch_contrastive and pixel_values.size(0) > 1:
                    bsz = pixel_values.size(0)
                    loss_matrix = []
                    for j in range(bsz):
                        pix_j = pixel_values[j].unsqueeze(0).expand(bsz, -1, -1, -1)
                        if batch_regions is not None:
                            vision_wrapper.set_regions([batch_regions[j]] * bsz, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids,
                            batch.get("scales", None),
                            batch_regions,
                            labels_local=labels,
                            contexts_local=batch.get("contexts_used", batch.get("contexts", None)),
                            context_allow_tokens_local=batch.get("context_allow_tokens", None),
                            context_total_tokens_local=batch.get("context_total_tokens", None),
                            context_ocr_tokens_local=batch.get("context_ocr_tokens", None),
                            context_adesc_tokens_local=batch.get("context_adesc_tokens", None),
                            context_para_tokens_local=batch.get("context_para_tokens", None),
                        )
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_n, attn_mask_n, past_kv_n, inputs_embeds_n, labels_mm_n = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids,
                                position_ids=None,
                                attention_mask=attention_mask,
                                past_key_values=None,
                                labels=labels,
                                images=pix_j,
                                image_sizes=None,
                            )
                            out_n = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_n,
                                position_ids=pos_ids_n,
                                past_key_values=past_kv_n,
                                inputs_embeds=inputs_embeds_n,
                                labels=labels_mm_n,
                                use_cache=False,
                            )
                            loss_n = nll_per_sample(out_n.logits, labels_mm_n)
                        _clear_region_attention_bias()
                        loss_matrix.append(loss_n)
                        del out_n, inputs_embeds_n, labels_mm_n, attn_mask_n, pos_ids_n, past_kv_n
                        torch.cuda.empty_cache()
                    loss_matrix = torch.stack(loss_matrix, dim=0)
                    eye = torch.eye(bsz, device=loss_matrix.device, dtype=torch.bool)
                    loss_neg = loss_matrix.masked_select(~eye).view(bsz, bsz - 1).mean(dim=1)
                    contrastive = torch.clamp(args.contrastive_margin + loss_i_det - loss_neg, min=0.0)
                    loss_contrast = args.contrastive_weight * contrastive.mean()
                else:
                    loss_ref = None
                    if args.contrastive_type == "shuffle" and pixel_values.size(0) > 1:
                        perm = torch.randperm(pixel_values.size(0), device=pixel_values.device)
                        if torch.all(perm == torch.arange(pixel_values.size(0), device=pixel_values.device)):
                            perm = torch.roll(perm, shifts=1, dims=0)
                        pixel_values_shuf = pixel_values[perm]
                        if batch_regions is not None:
                            perm_list = perm.detach().cpu().tolist()
                            vision_wrapper.set_regions([batch_regions[i] for i in perm_list], drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids,
                            batch.get("scales", None),
                            batch_regions,
                            labels_local=labels,
                            contexts_local=batch.get("contexts_used", batch.get("contexts", None)),
                            context_allow_tokens_local=batch.get("context_allow_tokens", None),
                            context_total_tokens_local=batch.get("context_total_tokens", None),
                            context_ocr_tokens_local=batch.get("context_ocr_tokens", None),
                            context_adesc_tokens_local=batch.get("context_adesc_tokens", None),
                            context_para_tokens_local=batch.get("context_para_tokens", None),
                        )
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_s, attn_mask_s, past_kv_s, inputs_embeds_s, labels_mm_s = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids,
                                position_ids=None,
                                attention_mask=attention_mask,
                                past_key_values=None,
                                labels=labels,
                                images=pixel_values_shuf,
                                image_sizes=None,
                            )
                            out_shuf = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_s,
                                position_ids=pos_ids_s,
                                past_key_values=past_kv_s,
                                inputs_embeds=inputs_embeds_s,
                                labels=labels_mm_s,
                                use_cache=False,
                            )
                            loss_ref = nll_per_sample(out_shuf.logits, labels_mm_s)
                        _clear_region_attention_bias()
                        del out_shuf, inputs_embeds_s, labels_mm_s, attn_mask_s, pos_ids_s, past_kv_s
                        torch.cuda.empty_cache()
                    else:
                        zero_images = torch.zeros_like(pixel_values)
                        vision_wrapper.set_regions(batch_regions, drop_one_region=False)
                        _set_region_attention_bias_for_inputs(
                            input_ids,
                            batch.get("scales", None),
                            batch_regions,
                            labels_local=labels,
                            contexts_local=batch.get("contexts_used", batch.get("contexts", None)),
                            context_allow_tokens_local=batch.get("context_allow_tokens", None),
                            context_total_tokens_local=batch.get("context_total_tokens", None),
                            context_ocr_tokens_local=batch.get("context_ocr_tokens", None),
                            context_adesc_tokens_local=batch.get("context_adesc_tokens", None),
                            context_para_tokens_local=batch.get("context_para_tokens", None),
                        )
                        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            _, pos_ids_z, attn_mask_z, past_kv_z, inputs_embeds_z, labels_mm_z = model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids,
                                position_ids=None,
                                attention_mask=attention_mask,
                                past_key_values=None,
                                labels=labels,
                                images=zero_images,
                                image_sizes=None,
                            )
                            out_zero = model.language_model(
                                input_ids=None,
                                attention_mask=attn_mask_z,
                                position_ids=pos_ids_z,
                                past_key_values=past_kv_z,
                                inputs_embeds=inputs_embeds_z,
                                labels=labels_mm_z,
                                use_cache=False,
                            )
                            loss_ref = nll_per_sample(out_zero.logits, labels_mm_z)
                        _clear_region_attention_bias()
                        del out_zero, inputs_embeds_z, labels_mm_z, attn_mask_z, pos_ids_z, past_kv_z
                        torch.cuda.empty_cache()

                    if loss_ref is not None:
                        if args.contrastive_mode == "continuous":
                            contrastive = loss_i_det - loss_ref
                        else:
                            contrastive = torch.clamp(args.contrastive_margin + loss_i_det - loss_ref, min=0.0)
                        loss_contrast = args.contrastive_weight * contrastive.mean()
                    else:
                        loss_contrast = None
                    vision_wrapper.set_regions(batch_regions, drop_one_region=False)

                if loss_contrast is not None:
                    loss_contrast_scaled = loss_contrast / max(1, args.grad_accum)
                    if use_scaler:
                        scaler.scale(loss_contrast_scaled).backward()
                    else:
                        loss_contrast_scaled.backward()
                    del loss_contrast
                    torch.cuda.empty_cache()
                loss = loss_i_det.detach()
            else:
                loss = loss_i_det.detach()

            # 3) consistency forward -> backward -> release
            if args.consistency_weight > 0 and args.consistency_mode == "ctx_shuffle" and pixel_values.size(0) > 0:
                contexts = batch.get("contexts_used", None) or batch.get("contexts", None)
                texts = batch.get("texts", None)
                scales = batch.get("scales", None)
                regions = batch.get("regions", None)
                if contexts is not None and texts is not None and scales is not None:
                    if regions is not None:
                        orig_batch = list(zip(images, texts, scales, contexts, regions))
                    else:
                        orig_batch = list(zip(images, texts, scales, contexts))
                    orig_pack = build_batch(
                        tokenizer,
                        orig_batch,
                        args.max_length,
                        {
                            "short": args.max_target_short,
                            "long": args.max_target_long,
                            "desc": args.max_target_desc,
                            "explain": args.max_target_long,
                        },
                        model.config.image_token_index,
                        add_eos=True,
                        fixed_task=args.fixed_task or None,
                        context_dropout=0.0,
                        paragraph_token_dropout=args.paragraph_token_dropout,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,
                        explain_ctx_min_adesc_tokens=args.explain_ctx_min_adesc_tokens,
                        explain_ctx_max_ocr_tokens=args.explain_ctx_max_ocr_tokens,
                        bucket_bins=bucket_bins if bucket_bins else None,
                        image_tokens=image_tokens_total,
                        scicap_prompt_style=args.scicap_prompt_style,
                        scicap_task_context_routing=args.scicap_task_context_routing,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                        use_context_placeholders=args.use_context_placeholders,
                    )
                    input_ids_o = orig_pack["input_ids"].to(device)
                    labels_o = orig_pack["labels"].to(device)
                    attention_mask_o = orig_pack["attention_mask"].to(device)
                    vision_wrapper.set_regions(orig_pack.get("regions", None), drop_one_region=False)
                    _set_region_attention_bias_for_inputs(
                        input_ids_o,
                        orig_pack.get("scales", None),
                        orig_pack.get("regions", None),
                        labels_local=labels_o,
                        contexts_local=orig_pack.get("contexts_used", orig_pack.get("contexts", None)),
                        context_allow_tokens_local=orig_pack.get("context_allow_tokens", None),
                        context_total_tokens_local=orig_pack.get("context_total_tokens", None),
                        context_ocr_tokens_local=orig_pack.get("context_ocr_tokens", None),
                        context_adesc_tokens_local=orig_pack.get("context_adesc_tokens", None),
                        context_para_tokens_local=orig_pack.get("context_para_tokens", None),
                    )
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_o, attn_mask_o, past_kv_o, inputs_embeds_o, labels_mm_o = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids_o,
                            position_ids=None,
                            attention_mask=attention_mask_o,
                            past_key_values=None,
                            labels=labels_o,
                            images=pixel_values,
                            image_sizes=None,
                        )
                        out_o = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_o,
                            position_ids=pos_ids_o,
                            past_key_values=past_kv_o,
                            inputs_embeds=inputs_embeds_o,
                            labels=labels_mm_o,
                            use_cache=False,
                        )
                        p_type = keyword_presence(out_o.logits, type_keyword_ids)
                        p_struct = keyword_presence(out_o.logits, struct_keyword_ids)
                    _clear_region_attention_bias()

                    perm_ctx = torch.randperm(len(contexts), device=device).tolist()
                    if perm_ctx == list(range(len(contexts))) and len(contexts) > 1:
                        perm_ctx = perm_ctx[1:] + perm_ctx[:1]
                    contexts_shuf = [contexts[i] for i in perm_ctx]
                    if regions is not None:
                        shuf_batch = list(zip(images, texts, scales, contexts_shuf, regions))
                    else:
                        shuf_batch = list(zip(images, texts, scales, contexts_shuf))
                    shuf_pack = build_batch(
                        tokenizer,
                        shuf_batch,
                        args.max_length,
                        {
                            "short": args.max_target_short,
                            "long": args.max_target_long,
                            "desc": args.max_target_desc,
                            "explain": args.max_target_long,
                        },
                        model.config.image_token_index,
                        add_eos=True,
                        fixed_task=args.fixed_task or None,
                        context_dropout=0.0,
                        paragraph_token_dropout=args.paragraph_token_dropout,
                        max_ctx_tokens=args.max_ctx_tokens,

                        max_ctx_tokens_explain=args.max_ctx_tokens_explain,
                        explain_ctx_min_adesc_tokens=args.explain_ctx_min_adesc_tokens,
                        explain_ctx_max_ocr_tokens=args.explain_ctx_max_ocr_tokens,
                        bucket_bins=bucket_bins if bucket_bins else None,
                        image_tokens=image_tokens_total,
                        scicap_prompt_style=args.scicap_prompt_style,
                        scicap_task_context_routing=args.scicap_task_context_routing,
                        enable_task_style_tokens=args.enable_task_style_tokens,
                        use_context_placeholders=args.use_context_placeholders,
                    )
                    input_ids_c = shuf_pack["input_ids"].to(device)
                    labels_c = shuf_pack["labels"].to(device)
                    attention_mask_c = shuf_pack["attention_mask"].to(device)
                    vision_wrapper.set_regions(shuf_pack.get("regions", None), drop_one_region=False)
                    _set_region_attention_bias_for_inputs(
                        input_ids_c,
                        shuf_pack.get("scales", None),
                        shuf_pack.get("regions", None),
                        labels_local=labels_c,
                        contexts_local=shuf_pack.get("contexts_used", shuf_pack.get("contexts", None)),
                        context_allow_tokens_local=shuf_pack.get("context_allow_tokens", None),
                        context_total_tokens_local=shuf_pack.get("context_total_tokens", None),
                        context_ocr_tokens_local=shuf_pack.get("context_ocr_tokens", None),
                        context_adesc_tokens_local=shuf_pack.get("context_adesc_tokens", None),
                        context_para_tokens_local=shuf_pack.get("context_para_tokens", None),
                    )
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        _, pos_ids_c, attn_mask_c, past_kv_c, inputs_embeds_c, labels_mm_c = model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids_c,
                            position_ids=None,
                            attention_mask=attention_mask_c,
                            past_key_values=None,
                            labels=labels_c,
                            images=pixel_values,
                            image_sizes=None,
                        )
                        out_c = model.language_model(
                            input_ids=None,
                            attention_mask=attn_mask_c,
                            position_ids=pos_ids_c,
                            past_key_values=past_kv_c,
                            inputs_embeds=inputs_embeds_c,
                            labels=labels_mm_c,
                            use_cache=False,
                        )
                        p_type_s = keyword_presence(out_c.logits, type_keyword_ids)
                        p_struct_s = keyword_presence(out_c.logits, struct_keyword_ids)
                    _clear_region_attention_bias()

                    log_margin = math.log(max(args.consistency_margin, 1e-6))
                    loss_type = torch.clamp(log_margin - torch.min(p_type, p_type_s), min=0.0)
                    loss_struct = torch.clamp(log_margin - torch.min(p_struct, p_struct_s), min=0.0)
                    loss_cons = args.consistency_weight * (loss_type.mean() + loss_struct.mean())
                    loss_cons_scaled = loss_cons / max(1, args.grad_accum)
                    if use_scaler:
                        scaler.scale(loss_cons_scaled).backward()
                    else:
                        loss_cons_scaled.backward()
                    del out_o, out_c, inputs_embeds_o, inputs_embeds_c, labels_mm_o, labels_mm_c
                    vision_wrapper.set_regions(batch_regions, drop_one_region=False)
                    torch.cuda.empty_cache()

            if use_scaler:
                pass
            else:
                pass

            if args.task_embed_only:
                _mask_embedding_grads(model.language_model.get_input_embeddings(), task_token_ids)
                out_emb = model.language_model.get_output_embeddings()
                if out_emb is not None and out_emb is not model.language_model.get_input_embeddings():
                    _mask_embedding_grads(out_emb, task_token_ids)

            if (step + 1) % args.grad_accum == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * max(1, args.grad_accum)
            if step % args.log_every == 0:
                avg_loss = running_loss / max(1, args.log_every)
                elapsed = time.time() - start_time
                print(
                    f"[train] step={step} loss={avg_loss:.4f} elapsed={elapsed:.1f}s "
                    f"beta={float(cur_region_beta):.4f} cf_weight={float(cur_cf_weight):.4f}"
                )
                if explain_gap_det is not None and explain_base_det is not None and explain_drop_det is not None:
                    print(
                        f"[train_explain_gap] step={step} "
                        f"loss_base={float(explain_base_det.item()):.4f} "
                        f"loss_region_drop={float(explain_drop_det.item()):.4f} "
                        f"gap={float(explain_gap_det.item()):.4f}"
                    )
                if (
                    explain_cf_base_det is not None
                    and explain_cf_wrong_det is not None
                    and explain_cf_pen_det is not None
                ):
                    print(
                        f"[train_explain_cf] step={step} "
                        f"base={float(explain_cf_base_det.item()):.4f} "
                        f"wrong_image={float(explain_cf_wrong_det.item()):.4f} "
                        f"penalty={float(explain_cf_pen_det.item()):.4f} "
                        f"weight={float(cur_cf_weight):.4f}"
                    )
                if task_balance_weights_det:
                    wb = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(task_balance_weights_det.items()))
                    print(f"[train_task_balance] step={step} weights={wb}")
                if desc_anchor_desc_count > 0:
                    anchor_raw = float(desc_anchor_loss_det.item()) if desc_anchor_loss_det is not None else 0.0
                    anchor_weighted = float(args.desc_entity_anchor_weight) * anchor_raw
                    nonalias_raw = float(desc_nonalias_loss_det.item()) if desc_nonalias_loss_det is not None else 0.0
                    nonalias_weighted = float(args.desc_entity_nonalias_penalty_weight) * nonalias_raw
                    active_ratio = float(desc_anchor_active_ratio.item()) if desc_anchor_active_ratio is not None else 0.0
                    print(
                        f"[train_desc_anchor] step={step} "
                        f"raw={anchor_raw:.6f} weighted={anchor_weighted:.6f} "
                        f"nonalias_raw={nonalias_raw:.6f} nonalias_weighted={nonalias_weighted:.6f} "
                        f"active_ratio={active_ratio:.3f} desc_in_batch={int(desc_anchor_desc_count)}"
                    )
                running_loss = 0.0
                start_time = time.time()

            if step > 0 and step % args.save_every == 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "tokenizer_len": len(tokenizer),
                }
                torch.save(ckpt, out_dir / f"ckpt_step_{step}.pt")
                tokenizer.save_pretrained(out_dir)

            if args.val_every > 0 and step > 0 and step % args.val_every == 0:
                run_validation(step)
            if args.sample_every > 0 and step > 0 and step % args.sample_every == 0:
                run_sample_dump(step)

            step += 1
            if step >= args.max_steps:
                break

    if args.val_every > 0:
        run_validation(step)
    if args.sample_every > 0:
        run_sample_dump(step)

    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "tokenizer_len": len(tokenizer),
    }
    torch.save(ckpt, out_dir / "ckpt_last.pt")
    tokenizer.save_pretrained(out_dir)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()

