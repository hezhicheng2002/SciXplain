#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertLMHeadModel,
    T5ForConditionalGeneration,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput

from scixplain.models import CLIPVisionTower

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TOKEN_SCHEMAS = {
    "legacy": {
        "task": {
            "short": "<CAPTION>",
            "long": "<SUMMARY>",
            "desc": "<DESCRIPTION>",
        },
        "scale": {
            "short": "<SHORT>",
            "long": "<LONG>",
            "desc": "<DESC>",
        },
    },
    "simple": {
        "task": {
            "short": "<CAPTION_SHORT>",
            "long": "<CAPTION_LONG>",
            "desc": "<DESCRIPTION>",
        },
        "scale": {},
    },
}
TASK_TOKENS = TOKEN_SCHEMAS["legacy"]["task"]
SCALE_TOKENS = TOKEN_SCHEMAS["legacy"]["scale"]


def set_token_schema(schema: str) -> None:
    global TASK_TOKENS, SCALE_TOKENS
    cfg = TOKEN_SCHEMAS.get(schema)
    if not cfg:
        raise ValueError(f"unknown token schema: {schema}")
    TASK_TOKENS = cfg.get("task", {})
    SCALE_TOKENS = cfg.get("scale", {})


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap_state(m):
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()


def _flatten_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (list, tuple)):
        parts: List[str] = []
        for it in val:
            s = _flatten_text(it)
            if s:
                parts.append(s)
        return " ".join(parts).strip()
    return str(val).strip()


def _maybe_json(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return val
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            return val
    return val


def _apply_path_replacements(path: str, replacements: List[Tuple[str, str]]) -> str:
    if not path:
        return path
    for src, dst in replacements:
        if path.startswith(src):
            return path.replace(src, dst, 1)
    return path


def _first_sentence(text: str) -> str:
    for sep in [".", "?", "!", "。", "？", "！"]:
        idx = text.find(sep)
        if idx >= 0:
            return text[: idx + 1].strip()
    return text.strip()


def _clean_ocr_items(val: Any, max_items: int = 64) -> List[str]:
    items = _maybe_json(val)
    if isinstance(items, dict):
        items = list(items.values())
    elif isinstance(items, str):
        items = [items]
    elif not isinstance(items, (list, tuple)):
        items = [items]
    cleaned: List[str] = []
    for it in items:
        s = _flatten_text(it)
        if not s:
            continue
        s = re.sub(r"\s+", " ", s).strip()
        core = re.sub(r"[^A-Za-z0-9]", "", s)
        if len(core) < 2 or len(core) > 30:
            continue
        if len(core) / max(1, len(s)) < 0.5:
            continue
        if len(set(core.lower())) == 1 and len(core) >= 3:
            continue
        digit_ratio = sum(ch.isdigit() for ch in core) / max(1, len(core))
        if digit_ratio > 0.7:
            continue
        cleaned.append(s)
        if max_items and len(cleaned) >= max_items:
            break
    return cleaned


def _extract_scicap_context(obj: Dict[str, Any]) -> Tuple[str, str, str, str]:
    if not isinstance(obj, dict):
        return "", "", "", ""
    meta = obj.get("metadata")
    if not isinstance(meta, dict):
        return "", "", "", ""
    raw = meta.get("scicap_raw")
    if not isinstance(raw, dict):
        return "", "", "", ""
    paragraph = _flatten_text(_maybe_json(raw.get("paragraph")))
    mention = _flatten_text(_maybe_json(raw.get("mention")))
    ocr = _flatten_text(_clean_ocr_items(raw.get("ocr")))
    context = _flatten_text([paragraph, mention])
    if not context:
        context = ocr
    return context, ocr, paragraph, mention


def _build_prefix_text(scale: str, context: str, token_mode: str) -> str:
    parts: List[str] = []
    if token_mode in ("task", "both"):
        parts.append(TASK_TOKENS.get(scale, ""))
    if token_mode in ("scale", "both"):
        parts.append(SCALE_TOKENS.get(scale, ""))
    if context:
        parts.append(context)
    parts = [p for p in parts if p]
    parts = list(dict.fromkeys(parts))
    return " ".join(parts).strip()


def _scale_special_tokens(scale: str, token_mode: str) -> List[str]:
    tokens: List[str] = []
    if token_mode in ("task", "both"):
        tok = TASK_TOKENS.get(scale, "")
        if tok:
            tokens.append(tok)
    if token_mode in ("scale", "both"):
        tok = SCALE_TOKENS.get(scale, "")
        if tok:
            tokens.append(tok)
    return tokens


def _build_prefix_ids(
    tokenizer: AutoTokenizer,
    scale: str,
    context: str,
    token_mode: str,
    max_prefix_len: int,
) -> Tuple[List[int], List[int]]:
    special_tokens = _scale_special_tokens(scale, token_mode)
    special_ids: List[int] = []
    if special_tokens:
        special_text = " ".join(special_tokens)
        special_ids = tokenizer(special_text, add_special_tokens=False)["input_ids"]
    ctx_ids: List[int] = []
    ctx = _flatten_text(_maybe_json(context))
    if ctx:
        ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
    if max_prefix_len <= 0:
        return [], special_ids
    if len(special_ids) > max_prefix_len:
        # If we must truncate, keep the last special tokens.
        special_ids = special_ids[-max_prefix_len:]
        return [], special_ids
    max_ctx = max_prefix_len - len(special_ids)
    if max_ctx <= 0:
        return [], special_ids
    if len(ctx_ids) > max_ctx:
        ctx_ids = ctx_ids[-max_ctx:]
    return ctx_ids, special_ids


def _load_struct_map(
    struct_jsonl: str | None,
    max_struct_nodes: int | None,
    max_struct_roles: int | None,
) -> Dict[str, Dict[str, Any]]:
    if not struct_jsonl:
        return {}
    struct_map: Dict[str, Dict[str, Any]] = {}
    with open(struct_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            path = obj.get("figure_path") or obj.get("image_path")
            if not path:
                continue
            path = os.path.abspath(path)
            nodes = obj.get("nodes") or []
            edges = obj.get("edges") or []
            node_texts: List[str] = []
            roles: List[str] = []
            node_type_counts: Counter[str] = Counter()
            role_counts: Counter[str] = Counter()
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                text = _flatten_text(node.get("text"))
                if text:
                    node_texts.append(text)
                role = _flatten_text(node.get("role")).lower()
                if role:
                    roles.append(role)
                    role_counts[role] += 1
                ntype = _flatten_text(node.get("type") or node.get("type_hint")).lower()
                if ntype:
                    node_type_counts[ntype] += 1
            edge_type_counts: Counter[str] = Counter()
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                etype = _flatten_text(edge.get("type")).lower()
                if etype:
                    edge_type_counts[etype] += 1
            if max_struct_nodes is not None:
                node_texts = node_texts[: max_struct_nodes]
            if max_struct_roles is not None:
                roles = roles[: max_struct_roles]
            struct_map[path] = {
                "node_texts": node_texts,
                "roles": roles,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "node_type_counts": dict(node_type_counts),
                "edge_type_counts": dict(edge_type_counts),
                "role_counts": dict(role_counts),
                "linearized": _flatten_text(obj.get("linearized")),
            }
    return struct_map


class SciCapMultiScaleDataset(Dataset):
    def __init__(
        self,
        split_json: str,
        images_root: str | None = None,
        sample_mode: str = "random",
        max_items: int | None = None,
        min_len_short: int = 20,
        min_len_long: int = 40,
        min_len_desc: int = 40,
        use_desc: bool = True,
        context_mode: str = "para_mention",
        scale_weights: Dict[str, float] | None = None,
        return_meta: bool = False,
        struct_jsonl: str | None = None,
        max_struct_nodes: int | None = None,
        max_struct_roles: int | None = None,
        max_image_side: int | None = None,
        path_replace: List[Tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        with open(split_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.sample_mode = sample_mode
        self.context_mode = context_mode
        self.scale_weights = scale_weights or {"short": 0.3, "long": 0.5, "desc": 0.2}
        self.return_meta = return_meta
        self.struct_map = _load_struct_map(struct_jsonl, max_struct_nodes, max_struct_roles)
        self.max_image_side = max_image_side
        self.path_replace = path_replace or []
        if not self.path_replace:
            scicap_root = Path(split_json).parents[1]
            dst_images_store = str(scicap_root / "images_store")
            dst_images = str(scicap_root / "images")
            legacy_images = os.environ.get("SCICAP_LEGACY_IMAGES", "").strip()
            legacy_images_store = os.environ.get("SCICAP_LEGACY_IMAGES_STORE", "").strip()
            replacements = []
            if legacy_images:
                replacements.append(
                    (legacy_images, dst_images_store if os.path.isdir(dst_images_store) else dst_images)
                )
            if legacy_images_store:
                replacements.append(
                    (legacy_images_store, dst_images_store if os.path.isdir(dst_images_store) else dst_images)
                )
            self.path_replace = replacements

        items = []
        samples = []
        for art in data:
            for fig in art.get("figures", []):
                img_path = fig.get("figure_path") or fig.get("result_path")
                if not img_path:
                    continue
                img_path = _apply_path_replacements(img_path, self.path_replace)
                if images_root and not os.path.isabs(img_path):
                    img_path = os.path.join(images_root, img_path)
                if not os.path.exists(img_path):
                    # Fallback by basename into local images_store/images.
                    bn = os.path.basename(img_path)
                    scicap_root = Path(split_json).parents[1]
                    for local_dir in (scicap_root / "images_store", scicap_root / "images"):
                        cand = local_dir / bn
                        if cand.is_file():
                            img_path = str(cand)
                            break
                if not os.path.exists(img_path):
                    continue

                meta = fig.get("metadata") or {}
                raw = meta.get("scicap_raw") or {}
                long_cap = _flatten_text(raw.get("mlbcap_long") or fig.get("figure_caption"))
                short_cap = _flatten_text(raw.get("mlbcap_short"))
                if not short_cap and long_cap:
                    short_cap = _first_sentence(long_cap)
                desc = _flatten_text(raw.get("figure_description"))
                paragraph = _flatten_text(raw.get("paragraph"))
                mention = _flatten_text(raw.get("mention"))
                ocr = _flatten_text(raw.get("ocr"))

                ctx_parts: List[str] = []
                if self.context_mode == "paragraph":
                    if paragraph:
                        ctx_parts.append(paragraph)
                elif self.context_mode in ("para_mention", "para_mention_ocr"):
                    if paragraph:
                        ctx_parts.append(paragraph)
                    if mention:
                        ctx_parts.append(mention)
                if self.context_mode == "para_mention_ocr" and ocr:
                    ctx_parts.append(ocr)
                context = " \n".join([p for p in ctx_parts if p]).strip()

                scales: Dict[str, str] = {}
                if len(short_cap) >= min_len_short:
                    scales["short"] = short_cap
                if len(long_cap) >= min_len_long:
                    scales["long"] = long_cap
                if use_desc and len(desc) >= min_len_desc:
                    scales["desc"] = desc

                if not scales:
                    continue

                meta = {"image_path": img_path, "ocr": ocr, "paragraph": paragraph, "mention": mention}
                if self.struct_map:
                    struct = self.struct_map.get(os.path.abspath(img_path))
                    if struct:
                        meta["struct_nodes"] = struct.get("node_texts", [])
                        meta["struct_roles"] = struct.get("roles", [])
                        meta["struct_node_count"] = struct.get("node_count")
                        meta["struct_edge_count"] = struct.get("edge_count")
                        meta["struct_node_type_counts"] = struct.get("node_type_counts", {})
                        meta["struct_edge_type_counts"] = struct.get("edge_type_counts", {})
                        meta["struct_role_counts"] = struct.get("role_counts", {})
                        meta["struct_linearized"] = struct.get("linearized", "")
                if sample_mode == "expand":
                    for s, t in scales.items():
                        samples.append(
                            {"image_path": img_path, "scale": s, "text": t, "context": context, "meta": meta}
                        )
                else:
                    items.append({"image_path": img_path, "scales": scales, "context": context, "meta": meta})

                if max_items is not None:
                    if sample_mode == "expand" and len(samples) >= max_items:
                        break
                    if sample_mode != "expand" and len(items) >= max_items:
                        break
            if max_items is not None:
                if sample_mode == "expand" and len(samples) >= max_items:
                    break
                if sample_mode != "expand" and len(items) >= max_items:
                    break

        self.items = items
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples) if self.sample_mode == "expand" else len(self.items)

    def _sample_scale(self, scales: Dict[str, str]) -> Tuple[str, str]:
        keys = list(scales.keys())
        weights = [self.scale_weights.get(k, 1.0) for k in keys]
        tot = sum(weights) if weights else 1.0
        probs = [w / tot for w in weights]
        choice = random.choices(keys, weights=probs, k=1)[0]
        return choice, scales[choice]

    def __getitem__(self, idx: int):
        from PIL import Image, ImageFile, UnidentifiedImageError

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None
        max_pixels = None
        if self.max_image_side:
            max_pixels = self.max_image_side * self.max_image_side * 4

        for _ in range(10):
            if self.sample_mode == "expand":
                rec = self.samples[idx]
                scale = rec["scale"]
                text = rec["text"]
                context = rec.get("context", "")
                img_path = rec["image_path"]
                meta = rec.get("meta", {})
            else:
                rec = self.items[idx]
                scale, text = self._sample_scale(rec["scales"])
                context = rec.get("context", "")
                img_path = rec["image_path"]
                meta = rec.get("meta", {})
            try:
                img = Image.open(img_path)
                if max_pixels is not None:
                    w, h = img.size
                    if w * h > max_pixels:
                        raise ValueError("image too large")
                img = img.convert("RGB")
                if self.max_image_side:
                    w, h = img.size
                    if max(w, h) > self.max_image_side:
                        img.thumbnail((self.max_image_side, self.max_image_side))
                if self.return_meta:
                    return img, text, scale, context, meta
                return img, text, scale, context
            except (OSError, UnidentifiedImageError, ValueError):
                idx = random.randint(0, len(self) - 1)
                continue
        raise RuntimeError("Failed to load a valid image after retries")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class JsonlDescDataset(Dataset):
    """
    Generic JSONL dataset for description-only training.
    Each row should include image_path and description (or desc/text).
    """

    def __init__(
        self,
        split_jsonl: str,
        images_root: str | None = None,
        max_items: int | None = None,
        min_len_desc: int = 40,
        desc_key: str = "description",
        image_key: str = "image_path",
        return_meta: bool = False,
        max_image_side: int | None = None,
        struct_jsonl: str | None = None,
        max_struct_nodes: int | None = None,
        max_struct_roles: int | None = None,
    ) -> None:
        super().__init__()
        self.struct_map = _load_struct_map(struct_jsonl, max_struct_nodes, max_struct_roles)
        items = []
        for obj in _iter_jsonl(split_jsonl):
            img_path = obj.get(image_key) or obj.get("image_path") or obj.get("figure_path")
            if not img_path:
                continue
            if images_root and not os.path.isabs(img_path):
                img_path = os.path.join(images_root, img_path)
            if not os.path.exists(img_path):
                continue
            desc = _flatten_text(obj.get(desc_key) or obj.get("description") or obj.get("desc") or obj.get("text"))
            if len(desc) < min_len_desc:
                continue
            meta = {"image_path": img_path}
            context = ""
            ctx_scicap, scicap_ocr, scicap_paragraph, scicap_mention = _extract_scicap_context(obj)
            if ctx_scicap:
                context = ctx_scicap
            if return_meta:
                meta["raw"] = obj
                if scicap_ocr and "ocr" not in meta:
                    meta["ocr"] = scicap_ocr
                if scicap_paragraph and "paragraph" not in meta:
                    meta["paragraph"] = scicap_paragraph
                if scicap_mention and "mention" not in meta:
                    meta["mention"] = scicap_mention
                if self.struct_map:
                    struct = self.struct_map.get(os.path.abspath(img_path))
                    if struct:
                        meta["struct_nodes"] = struct.get("node_texts", [])
                        meta["struct_roles"] = struct.get("roles", [])
                        meta["struct_node_count"] = struct.get("node_count")
                        meta["struct_edge_count"] = struct.get("edge_count")
                        meta["struct_node_type_counts"] = struct.get("node_type_counts", {})
                        meta["struct_edge_type_counts"] = struct.get("edge_type_counts", {})
                        meta["struct_role_counts"] = struct.get("role_counts", {})
                        meta["struct_linearized"] = struct.get("linearized", "")
            items.append((img_path, desc, context, meta))
            if max_items is not None and len(items) >= max_items:
                break
        self.items = items
        self.return_meta = return_meta
        self.max_image_side = max_image_side

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image, ImageFile, UnidentifiedImageError

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None
        max_pixels = None
        if self.max_image_side:
            max_pixels = self.max_image_side * self.max_image_side * 4

        for _ in range(10):
            img_path, desc, context, meta = self.items[idx]
            try:
                img = Image.open(img_path)
                if max_pixels is not None:
                    w, h = img.size
                    if w * h > max_pixels:
                        raise ValueError("image too large")
                img = img.convert("RGB")
                if self.max_image_side:
                    w, h = img.size
                    if max(w, h) > self.max_image_side:
                        img.thumbnail((self.max_image_side, self.max_image_side))
                if self.return_meta:
                    return img, desc, "desc", context, meta
                return img, desc, "desc", context
            except (OSError, UnidentifiedImageError, ValueError):
                idx = random.randint(0, len(self) - 1)
                continue
        raise RuntimeError("Failed to load a valid image after retries")


class JsonlMultiScaleDataset(Dataset):
    """
    JSONL dataset with multiple outputs per image.
    Expected fields: image_path and either `scales` dict or caption_short/caption_long/description.
    """

    def __init__(
        self,
        split_jsonl: str,
        images_root: str | None = None,
        sample_mode: str = "random",
        max_items: int | None = None,
        min_len_short: int = 20,
        min_len_long: int = 40,
        min_len_desc: int = 40,
        scale_weights: Dict[str, float] | None = None,
        return_meta: bool = False,
        image_key: str = "image_path",
        short_key: str = "caption_short",
        long_key: str = "caption_long",
        desc_key: str = "description",
        scales_key: str = "scales",
        context_key: str = "context",
        struct_jsonl: str | None = None,
        max_struct_nodes: int | None = None,
        max_struct_roles: int | None = None,
        max_image_side: int | None = None,
    ) -> None:
        super().__init__()
        self.sample_mode = sample_mode
        self.scale_weights = scale_weights or {"short": 0.3, "long": 0.5, "desc": 0.2}
        self.return_meta = return_meta
        self.max_image_side = max_image_side
        self.struct_map = _load_struct_map(struct_jsonl, max_struct_nodes, max_struct_roles)
        items = []
        samples = []
        for obj in _iter_jsonl(split_jsonl):
            img_path = obj.get(image_key) or obj.get("image_path") or obj.get("figure_path")
            if not img_path:
                continue
            if images_root and not os.path.isabs(img_path):
                img_path = os.path.join(images_root, img_path)
            if not os.path.exists(img_path):
                continue

            scales: Dict[str, str] = {}
            raw_scales = obj.get(scales_key) if scales_key else None
            if isinstance(raw_scales, dict):
                for k, v in raw_scales.items():
                    text = _flatten_text(v)
                    if k == "short" and len(text) >= min_len_short:
                        scales["short"] = text
                    elif k == "long" and len(text) >= min_len_long:
                        scales["long"] = text
                    elif k == "desc" and len(text) >= min_len_desc:
                        scales["desc"] = text
            else:
                short = _flatten_text(obj.get(short_key) or obj.get("short"))
                long = _flatten_text(obj.get(long_key) or obj.get("long"))
                desc = _flatten_text(obj.get(desc_key) or obj.get("description") or obj.get("desc"))
                if len(short) >= min_len_short:
                    scales["short"] = short
                if len(long) >= min_len_long:
                    scales["long"] = long
                if len(desc) >= min_len_desc:
                    scales["desc"] = desc

            if not scales:
                continue

            context = _flatten_text(obj.get(context_key))
            ctx_scicap, scicap_ocr, scicap_paragraph, scicap_mention = _extract_scicap_context(obj)
            if not context and ctx_scicap:
                context = ctx_scicap
            meta = {"image_path": img_path}
            if return_meta:
                meta["raw"] = obj
                if scicap_ocr and "ocr" not in meta:
                    meta["ocr"] = scicap_ocr
                if scicap_paragraph and "paragraph" not in meta:
                    meta["paragraph"] = scicap_paragraph
                if scicap_mention and "mention" not in meta:
                    meta["mention"] = scicap_mention
                if self.struct_map:
                    struct = self.struct_map.get(os.path.abspath(img_path))
                    if struct:
                        meta["struct_nodes"] = struct.get("node_texts", [])
                        meta["struct_roles"] = struct.get("roles", [])
                        meta["struct_node_count"] = struct.get("node_count")
                        meta["struct_edge_count"] = struct.get("edge_count")
                        meta["struct_node_type_counts"] = struct.get("node_type_counts", {})
                        meta["struct_edge_type_counts"] = struct.get("edge_type_counts", {})
                        meta["struct_role_counts"] = struct.get("role_counts", {})
                        meta["struct_linearized"] = struct.get("linearized", "")
            if sample_mode == "expand":
                for s, t in scales.items():
                    samples.append(
                        {"image_path": img_path, "scale": s, "text": t, "context": context, "meta": meta}
                    )
            else:
                items.append({"image_path": img_path, "scales": scales, "context": context, "meta": meta})

            if max_items is not None:
                if sample_mode == "expand" and len(samples) >= max_items:
                    break
                if sample_mode != "expand" and len(items) >= max_items:
                    break

        self.items = items
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples) if self.sample_mode == "expand" else len(self.items)

    def _sample_scale(self, scales: Dict[str, str]) -> Tuple[str, str]:
        keys = list(scales.keys())
        weights = [self.scale_weights.get(k, 1.0) for k in keys]
        tot = sum(weights) if weights else 1.0
        probs = [w / tot for w in weights]
        choice = random.choices(keys, weights=probs, k=1)[0]
        return choice, scales[choice]

    def __getitem__(self, idx: int):
        from PIL import Image, ImageFile, UnidentifiedImageError

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None
        max_pixels = None
        if self.max_image_side:
            max_pixels = self.max_image_side * self.max_image_side * 4

        for _ in range(10):
            if self.sample_mode == "expand":
                rec = self.samples[idx]
                scale = rec["scale"]
                text = rec["text"]
                context = rec.get("context", "")
                img_path = rec["image_path"]
                meta = rec.get("meta", {})
                if "ocr" in rec:
                    meta["ocr"] = rec.get("ocr")
                if "paragraph" in rec:
                    meta["paragraph"] = rec.get("paragraph")
                if "mention" in rec:
                    meta["mention"] = rec.get("mention")
            else:
                rec = self.items[idx]
                scale, text = self._sample_scale(rec["scales"])
                context = rec.get("context", "")
                img_path = rec["image_path"]
                meta = rec.get("meta", {})
                if "ocr" in rec:
                    meta["ocr"] = rec.get("ocr")
                if "paragraph" in rec:
                    meta["paragraph"] = rec.get("paragraph")
                if "mention" in rec:
                    meta["mention"] = rec.get("mention")
            try:
                img = Image.open(img_path)
                if max_pixels is not None:
                    w, h = img.size
                    if w * h > max_pixels:
                        raise ValueError("image too large")
                img = img.convert("RGB")
                if self.max_image_side:
                    w, h = img.size
                    if max(w, h) > self.max_image_side:
                        img.thumbnail((self.max_image_side, self.max_image_side))
                if self.return_meta:
                    return img, text, scale, context, meta
                return img, text, scale, context
            except (OSError, UnidentifiedImageError, ValueError):
                idx = random.randint(0, len(self) - 1)
                continue
        raise RuntimeError("Failed to load a valid image after retries")


def collate_batch(batch):
    if len(batch[0]) == 5:
        images, texts, scales, contexts, metas = zip(*batch)
        return list(images), list(texts), list(scales), list(contexts), list(metas)
    images, texts, scales, contexts = zip(*batch)
    return list(images), list(texts), list(scales), list(contexts)


def _maybe_allow_unsafe_torch_load(allow: bool) -> None:
    if not allow:
        return
    try:
        from transformers import modeling_utils
        from transformers.utils import import_utils

        import_utils.check_torch_load_is_safe = lambda: None
        modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        return


def build_tokenizer(
    token_mode: str = "task",
    decoder_arch: str = "bert",
    t5_model_name: str = "t5-base",
) -> AutoTokenizer:
    if decoder_arch == "t5":
        tok = AutoTokenizer.from_pretrained(t5_model_name, use_fast=True)
    else:
        tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", use_fast=True)
    extra: List[str] = []
    if token_mode in ("task", "both"):
        extra.extend(list(TASK_TOKENS.values()))
    if token_mode in ("scale", "both"):
        extra.extend(list(SCALE_TOKENS.values()))
    extra = list(dict.fromkeys(extra))
    if extra:
        tok.add_special_tokens({"additional_special_tokens": extra})
    return tok


def build_decoder(
    tokenizer: AutoTokenizer,
    decoder_arch: str = "bert",
    t5_model_name: str = "t5-base",
    allow_unsafe_torch_load: bool = False,
):
    if decoder_arch == "t5":
        dec = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        dec.resize_token_embeddings(len(tokenizer))
        dec.config.pad_token_id = tokenizer.pad_token_id
        dec.config.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id
        dec.config.decoder_start_token_id = tokenizer.pad_token_id
        return dec
    cfg = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased")
    cfg.is_decoder = True
    cfg.add_cross_attention = True
    _maybe_allow_unsafe_torch_load(allow_unsafe_torch_load)
    dec = BertLMHeadModel.from_pretrained("allenai/scibert_scivocab_uncased", config=cfg)
    dec.resize_token_embeddings(len(tokenizer))
    dec.config.decoder_start_token_id = tokenizer.cls_token_id
    dec.config.pad_token_id = tokenizer.pad_token_id
    dec.config.eos_token_id = tokenizer.sep_token_id
    return dec


def build_text_encoder(name: str = "scibert", allow_unsafe_torch_load: bool = False):
    if name == "scibert":
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", use_fast=True)
        _maybe_allow_unsafe_torch_load(allow_unsafe_torch_load)
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        return model, tokenizer
    raise ValueError(f"unknown text encoder: {name}")


def tokenize_batch(
    tokenizer: AutoTokenizer,
    texts: List[str],
    scales: List[str],
    contexts: List[str],
    max_length: int,
    min_target_tokens: int,
    max_target_map: Dict[str, int],
    token_mode: str,
    decoder_arch: str = "bert",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    pad_id = tokenizer.pad_token_id or 0
    cls_id = tokenizer.cls_token_id or pad_id
    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or pad_id
    if decoder_arch == "t5":
        cls_id = pad_id
        max_body = max(1, max_length - 1)
    else:
        max_body = max(1, max_length - 2)
    min_target_tokens = max(1, min(min_target_tokens, max_body))

    input_ids_list = []
    attn_list = []
    labels_list = []

    for t, s, ctx in zip(texts, scales, contexts):
        max_target = max_target_map.get(s, max_body)
        max_target = max(min_target_tokens, min(int(max_target), max_body))
        ctx_ids, special_ids = _build_prefix_ids(tokenizer, s, ctx, token_mode, max_body)
        if special_ids:
            max_target = max(min_target_tokens, min(max_target, max_body - len(special_ids)))
        target_ids = tokenizer(
            t,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target,
        )["input_ids"]
        if not target_ids:
            target_ids = [tokenizer.unk_token_id or sep_id]

        if len(target_ids) >= min_target_tokens:
            target_keep = min(len(target_ids), max_target)
            if target_keep < min_target_tokens:
                target_keep = min_target_tokens
        else:
            target_keep = min(len(target_ids), max_target)
        prefix_keep = max_body - target_keep
        prefix_keep = max(0, prefix_keep)
        if special_ids and prefix_keep < len(special_ids):
            target_keep = max(1, max_body - len(special_ids))
            target_ids = target_ids[:target_keep]
            prefix_keep = max_body - target_keep
        max_ctx = max(0, prefix_keep - len(special_ids))
        if len(ctx_ids) > max_ctx:
            ctx_ids = ctx_ids[-max_ctx:]
        prefix_ids = ctx_ids + special_ids
        target_ids = target_ids[:target_keep]

        if decoder_arch == "t5":
            # Seq2seq: decoder input is shift-right of target, no prefix tokens in decoder.
            input_ids = [cls_id] + target_ids[:-1]
            labels = target_ids
        else:
            input_ids = [cls_id] + prefix_ids + target_ids + [sep_id]
            labels = [-100] * (1 + len(prefix_ids)) + target_ids + [-100]
        attn = [1] * len(input_ids)

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len
            attn += [0] * pad_len
        elif pad_len < 0:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attn = attn[:max_length]

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attn_list.append(attn)

    tok = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attn_list, dtype=torch.long),
    }
    labels = torch.tensor(labels_list, dtype=torch.long)
    return tok, labels


def _token_ids_from_texts(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_items: int | None = None,
) -> List[int]:
    ids: List[int] = []
    for txt in texts:
        txt = _flatten_text(txt)
        if not txt:
            continue
        if max_items is not None:
            token_ids = tokenizer(
                txt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_items,
            )["input_ids"]
        else:
            token_ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
        if token_ids:
            ids.extend(token_ids)
    uniq = list(dict.fromkeys(ids))
    if max_items is not None:
        uniq = uniq[: max_items]
    return uniq


def _filter_copy_token_ids(tokenizer: AutoTokenizer, ids: List[int]) -> List[int]:
    if not ids:
        return ids
    toks = tokenizer.convert_ids_to_tokens(ids)
    keep: List[int] = []
    specials = set(getattr(tokenizer, "all_special_tokens", []) or [])
    for tid, tok in zip(ids, toks):
        if tok in specials:
            continue
        if tok.startswith("##"):
            continue
        core = re.sub(r"[^A-Za-z0-9]", "", tok)
        if len(core) < 2:
            continue
        keep.append(tid)
    return list(dict.fromkeys(keep))


def _repeat_ngram_unlikelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    scales: List[str],
    ngram: int,
    target_scales: set[str] | None = None,
) -> torch.Tensor | None:
    if ngram < 2:
        return None
    bsz = logits.size(0)
    losses: List[torch.Tensor] = []
    for i in range(bsz):
        if target_scales and scales[i] not in target_scales:
            continue
        seq_pos = [j for j, v in enumerate(labels[i].tolist()) if v != -100]
        if len(seq_pos) < ngram:
            continue
        tokens = [int(labels[i, j].item()) for j in seq_pos]
        seen = set()
        rep_positions: List[int] = []
        for t in range(len(tokens)):
            if t + 1 < ngram:
                continue
            ng = tuple(tokens[t - ngram + 1 : t + 1])
            if ng in seen:
                rep_positions.append(t)
            else:
                seen.add(ng)
        if not rep_positions:
            continue
        per_pos = []
        for t in rep_positions:
            pos = seq_pos[t]
            tok_id = tokens[t]
            probs = torch.softmax(logits[i, pos].float(), dim=-1)
            p = probs[tok_id].clamp(max=1.0 - 1e-6)
            per_pos.append(-torch.log1p(-p))
        if per_pos:
            losses.append(torch.stack(per_pos).mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def _cross_repeat_ngram_unlikelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ngram: int,
    target_scales: set | None = None,
    scales: List[str] | None = None,
) -> torch.Tensor | None:
    if ngram <= 0:
        return None
    bsz, seq_len, _ = logits.size()
    if bsz <= 1:
        return None
    losses = []
    for i in range(bsz):
        if target_scales and scales and scales[i] not in target_scales:
            continue
        lab_i = labels[i]
        valid_i = lab_i != -100
        if valid_i.sum().item() == 0:
            continue
        tokens_i = lab_i[valid_i].tolist()
        if len(tokens_i) < ngram:
            continue
        other_ngrams = set()
        for j in range(bsz):
            if i == j:
                continue
            if target_scales and scales and scales[j] not in target_scales:
                continue
            lab_j = labels[j]
            valid_j = lab_j != -100
            tokens_j = lab_j[valid_j].tolist()
            if len(tokens_j) < ngram:
                continue
            for k in range(len(tokens_j) - ngram + 1):
                other_ngrams.add(tuple(tokens_j[k : k + ngram]))
        if not other_ngrams:
            continue
        # Map valid token positions back to label index
        pos_idx = valid_i.nonzero(as_tuple=True)[0].tolist()
        per_pos = []
        for t in range(ngram - 1, len(tokens_i)):
            ng = tuple(tokens_i[t - ngram + 1 : t + 1])
            if ng not in other_ngrams:
                continue
            pos = pos_idx[t]
            tok_id = tokens_i[t]
            probs = torch.softmax(logits[i, pos].float(), dim=-1)
            p = probs[tok_id].clamp(max=1.0 - 1e-6)
            per_pos.append(-torch.log1p(-p))
        if per_pos:
            losses.append(torch.stack(per_pos).mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def _coverage_loss_for_ids(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_ids: List[int],
    top_k: int,
) -> torch.Tensor | None:
    if not token_ids:
        return None
    mask = labels != -100
    if mask.sum().item() == 0:
        return None
    probs = torch.softmax(logits[mask].float(), dim=-1)
    token_ids_t = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    max_probs = probs[:, token_ids_t].max(dim=0).values
    if top_k > 0 and max_probs.numel() > top_k:
        max_probs = max_probs.topk(top_k).values
    score = max_probs.mean()
    return -torch.log(score + 1e-6)


def _build_plan_text(
    meta: Dict[str, Any],
    max_nodes: int = 16,
    max_roles: int = 8,
    include_counts: bool = False,
    include_types: bool = False,
    max_types: int = 6,
    include_linearized: bool = False,
    max_linearized_chars: int = 256,
) -> str:
    nodes = meta.get("struct_nodes") or []
    roles = meta.get("struct_roles") or []
    if max_nodes > 0:
        nodes = nodes[:max_nodes]
    if max_roles > 0:
        roles = roles[:max_roles]
    node_text = " ; ".join([_flatten_text(n) for n in nodes if _flatten_text(n)])
    role_text = " ; ".join([_flatten_text(r) for r in roles if _flatten_text(r)])
    parts: List[str] = []
    if include_counts:
        node_count = meta.get("struct_node_count")
        edge_count = meta.get("struct_edge_count")
        counts: List[str] = []
        if isinstance(node_count, int):
            counts.append(f"nodes={node_count}")
        if isinstance(edge_count, int):
            counts.append(f"edges={edge_count}")
        if include_types:
            node_types = meta.get("struct_node_type_counts") or {}
            edge_types = meta.get("struct_edge_type_counts") or {}
            role_counts = meta.get("struct_role_counts") or {}
            if isinstance(node_types, dict) and node_types:
                top_nodes = sorted(node_types.items(), key=lambda x: (-x[1], x[0]))[:max_types]
                counts.append("node_types=" + ",".join([f"{k}:{v}" for k, v in top_nodes]))
            if isinstance(edge_types, dict) and edge_types:
                top_edges = sorted(edge_types.items(), key=lambda x: (-x[1], x[0]))[:max_types]
                counts.append("edge_types=" + ",".join([f"{k}:{v}" for k, v in top_edges]))
            if isinstance(role_counts, dict) and role_counts:
                top_roles = sorted(role_counts.items(), key=lambda x: (-x[1], x[0]))[:max_types]
                counts.append("roles=" + ",".join([f"{k}:{v}" for k, v in top_roles]))
        if counts:
            parts.append("COUNTS: " + " ".join(counts))
    if node_text:
        parts.append(f"NODES: {node_text}")
    if role_text:
        parts.append(f"ROLES: {role_text}")
    if include_linearized:
        lin = _flatten_text(meta.get("struct_linearized") or "")
        if max_linearized_chars > 0 and len(lin) > max_linearized_chars:
            lin = lin[: max_linearized_chars].rstrip()
        if lin:
            parts.append(f"STRUCT: {lin}")
    return " | ".join(parts).strip()


def _plan_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_nodes": args.max_struct_nodes,
        "max_roles": args.max_struct_roles,
        "include_counts": args.plan_include_counts,
        "include_types": args.plan_include_types,
        "max_types": args.plan_max_types,
        "include_linearized": args.plan_include_linearized,
        "max_linearized_chars": args.plan_max_linearized_chars,
    }


def _build_context_from_meta(
    meta: Dict[str, Any],
    fallback: str,
    drop_mention: float,
    drop_paragraph: float,
    drop_ocr: float,
    shuffle_prob: float,
    plan_kwargs: Dict[str, Any] | None = None,
    plan_in_context: bool = False,
) -> str:
    parts: List[str] = []
    mention = _flatten_text(_maybe_json(meta.get("mention", "")))
    paragraph = _flatten_text(_maybe_json(meta.get("paragraph", "")))
    ocr = _flatten_text(_maybe_json(meta.get("ocr", "")))
    if mention and random.random() >= drop_mention:
        parts.append(mention)
    if paragraph and random.random() >= drop_paragraph:
        parts.append(paragraph)
    if ocr and random.random() >= drop_ocr:
        parts.append(ocr)
    if parts and shuffle_prob > 0 and random.random() < shuffle_prob:
        random.shuffle(parts)
    if parts:
        ctx = " \n".join([p for p in parts if p]).strip()
    else:
        ctx = _flatten_text(_maybe_json(fallback))
    if plan_in_context and meta:
        plan = _build_plan_text(meta, **(plan_kwargs or {}))
        if plan:
            return f"{plan}\n{ctx}".strip() if ctx else plan
    return ctx


def _augment_contexts(
    contexts: List[str],
    metas: List[Dict[str, Any]],
    args: argparse.Namespace,
    plan_kwargs: Dict[str, Any] | None = None,
) -> List[str]:
    if args.context_aug_prob <= 0:
        if args.plan_in_context:
            out = []
            for ctx, meta in zip(contexts, metas):
                out.append(
                    _build_context_from_meta(
                        meta or {},
                        ctx,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        plan_kwargs=plan_kwargs,
                        plan_in_context=True,
                    )
                )
            return out
        return contexts
    out = []
    for ctx, meta in zip(contexts, metas):
        if random.random() >= args.context_aug_prob:
            if args.plan_in_context:
                out.append(
                    _build_context_from_meta(
                        meta or {},
                        ctx,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        plan_kwargs=plan_kwargs,
                        plan_in_context=True,
                    )
                )
            else:
                out.append(ctx)
            continue
        out.append(
            _build_context_from_meta(
                meta or {},
                ctx,
                args.context_drop_mention_prob,
                args.context_drop_paragraph_prob,
                args.context_drop_ocr_prob,
                args.context_shuffle_prob,
                plan_kwargs=plan_kwargs,
                plan_in_context=args.plan_in_context,
            )
        )
    return out


def _scale_text_tokens(text_tokens: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return torch.zeros_like(text_tokens)
    if scale == 1.0:
        return text_tokens
    return text_tokens * float(scale)


def _encode_text_context(
    tokenizer: AutoTokenizer,
    contexts: List[str],
    metas: List[Dict[str, Any]],
    max_length: int,
    use_plan_tokens: bool,
    scales: List[str] | None = None,
    token_mode: str = "task",
    plan_kwargs: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    texts = []
    for i, (ctx, meta) in enumerate(zip(contexts, metas)):
        plan = _build_plan_text(meta, **(plan_kwargs or {})) if use_plan_tokens else ""
        scale = scales[i] if scales and i < len(scales) else ""
        prefix = _build_prefix_text(scale, "", token_mode) if scale else ""
        ctx = _flatten_text(_maybe_json(ctx))
        if plan and ctx:
            text = f"{plan}\n{ctx}"
        elif plan:
            text = plan
        else:
            text = ctx or ""
        if prefix:
            text = f"{prefix} {text}".strip()
        texts.append(text)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def _collect_copy_tokens(meta: Dict[str, Any], context: str, source: str = "ocr") -> str:
    raw = meta.get("raw") if isinstance(meta, dict) else None
    if source == "ocr":
        if isinstance(raw, dict) and "ocr" in raw:
            return _flatten_text(_clean_ocr_items(raw.get("ocr")))
        if "ocr" in meta:
            return _flatten_text(_clean_ocr_items(meta.get("ocr")))
        return ""
    if source == "paragraph":
        return _flatten_text(meta.get("paragraph") or "")
    if source == "mention":
        return _flatten_text(meta.get("mention") or "")
    if source == "struct_nodes":
        nodes = meta.get("struct_nodes") or []
        return _flatten_text(nodes)
    if source == "struct_roles":
        roles = meta.get("struct_roles") or []
        return _flatten_text(roles)
    if source == "struct_linearized":
        return _flatten_text(meta.get("struct_linearized") or "")
    if source == "context":
        return context or ""
    return ""


def _build_copy_token_ids(
    tokenizer: AutoTokenizer,
    contexts: List[str],
    metas: List[Dict[str, Any]],
    max_items: int = 128,
    sources: str = "ocr",
) -> List[List[int]]:
    out: List[List[int]] = []
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    for ctx, meta in zip(contexts, metas):
        parts: List[str] = []
        for src in src_list:
            txt = _collect_copy_tokens(meta, ctx, source=src)
            if txt:
                parts.append(txt)
        txt = " ".join(parts).strip()
        ids = _token_ids_from_texts(tokenizer, [txt], max_items=max_items)
        ids = _filter_copy_token_ids(tokenizer, ids)
        out.append(ids)
    return out


def _apply_copy_bias(logits: torch.Tensor, copy_ids: List[List[int]], bias: float) -> torch.Tensor:
    if bias <= 0:
        return logits
    logits = logits.clone()
    for i, ids in enumerate(copy_ids):
        if not ids:
            continue
        logits[i, :, ids] = logits[i, :, ids] + float(bias)
    return logits


def _forward_decoder(
    decoder,
    tok: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    enc_tokens: torch.Tensor,
    enc_mask: torch.Tensor,
    decoder_arch: str,
    copy_ids: List[List[int]] | None = None,
    copy_bias: float = 0.0,
    amp: bool = False,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(device_type=device.type if device else "cuda", enabled=amp and device and device.type == "cuda"):
        if decoder_arch == "t5":
            out = decoder(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_tokens),
                attention_mask=enc_mask,
                decoder_input_ids=tok["input_ids"],
                decoder_attention_mask=tok["attention_mask"],
                labels=labels,
            )
        else:
            out = decoder(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
                encoder_hidden_states=enc_tokens,
                encoder_attention_mask=enc_mask,
                labels=labels,
            )
        logits = out.logits
    if copy_bias > 0 and copy_ids is not None:
        logits = _apply_copy_bias(logits, copy_ids, copy_bias)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
    else:
        loss = out.loss
    return loss, logits


def _build_prefix_batch_for_decode(
    tokenizer: AutoTokenizer,
    scales: List[str],
    contexts: List[str],
    max_length: int,
    max_target_map: Dict[str, int],
    token_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    pad_id = tokenizer.pad_token_id or 0
    cls_id = tokenizer.cls_token_id or pad_id
    max_body = max(1, max_length - 1)
    input_ids_list: List[List[int]] = []
    prefix_lens: List[int] = []
    max_new_tokens_list: List[int] = []

    for scale, ctx in zip(scales, contexts):
        max_target = max_target_map.get(scale, max_body)
        max_target = max(1, min(int(max_target), max_body))
        ctx_ids, special_ids = _build_prefix_ids(tokenizer, scale, ctx, token_mode, max_body)
        if special_ids:
            max_target = min(max_target, max_body - len(special_ids))
        if max_target < 1:
            max_target = 1
        prefix_keep = max_body - max_target
        if prefix_keep < len(special_ids):
            prefix_keep = len(special_ids)
        max_ctx = max(0, prefix_keep - len(special_ids))
        if len(ctx_ids) > max_ctx:
            ctx_ids = ctx_ids[-max_ctx:]
        prefix_ids = ctx_ids + special_ids
        input_ids = [cls_id] + prefix_ids
        input_ids_list.append(input_ids)
        prefix_lens.append(len(input_ids))
        max_new_tokens_list.append(max_target)

    max_len = max(prefix_lens) if prefix_lens else 1
    max_new_cap = max_length - max_len
    if max_new_cap < 0:
        max_new_cap = 0
    max_new_tokens_list = [min(n, max_new_cap) for n in max_new_tokens_list]
    attn_list: List[List[int]] = []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        attn = [1] * len(ids) + [0] * pad_len
        if pad_len > 0:
            ids.extend([pad_id] * pad_len)
        attn_list.append(attn)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attn_list, dtype=torch.long)
    # Decode should start after the padded prefix length for all samples.
    decode_prefix_lens = [max_len] * len(prefix_lens)
    return input_ids, attention_mask, decode_prefix_lens, max_new_tokens_list


@torch.no_grad()
def _greedy_decode_bert(
    decoder: BertLMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    enc_tokens: torch.Tensor,
    enc_mask: torch.Tensor,
    max_new_tokens_list: List[int],
    eos_id: int,
    pad_id: int,
    max_length: int,
    min_new_tokens: int,
    forbidden_ids: List[int] | None = None,
) -> torch.Tensor:
    device = input_ids.device
    bsz = input_ids.size(0)
    max_new = max(max_new_tokens_list) if max_new_tokens_list else 0
    if max_new <= 0:
        return input_ids
    max_new_t = torch.tensor(max_new_tokens_list, device=device)
    if min_new_tokens < 0:
        min_new_tokens = 0
    if min_new_tokens > 0:
        min_new_t = torch.minimum(max_new_t, torch.full_like(max_new_t, min_new_tokens))
    else:
        min_new_t = None
    forbidden = [i for i in (forbidden_ids or []) if i is not None]

    generated = input_ids
    attn = attention_mask
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    for step in range(max_new):
        if done.all() or generated.size(1) >= max_length:
            break
        out = decoder(
            input_ids=generated,
            attention_mask=attn,
            encoder_hidden_states=enc_tokens,
            encoder_attention_mask=enc_mask,
            use_cache=False,
        )
        logits = out.logits[:, -1, :]
        if forbidden:
            logits = logits.clone()
            logits[:, forbidden] = -1.0e9
        if min_new_t is not None:
            logits = logits.clone()
            force_mask = min_new_t > step
            if force_mask.any():
                idx = force_mask.nonzero(as_tuple=True)[0]
                logits[idx, eos_id] = -1.0e9
        next_token = logits.argmax(dim=-1)
        over = (step + 1) >= max_new_t
        step_done = done | over | (next_token == eos_id)
        next_token = torch.where(step_done, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        attn = torch.cat([attn, (~step_done).long().unsqueeze(-1)], dim=-1)
        done = step_done
    return generated


def _decode_generated(
    tokenizer: AutoTokenizer,
    generated: torch.Tensor,
    prefix_lens: List[int],
    max_new_tokens_list: List[int],
    eos_id: int,
) -> List[str]:
    preds: List[str] = []
    pad_id = tokenizer.pad_token_id
    for i in range(generated.size(0)):
        start = prefix_lens[i]
        max_new = max_new_tokens_list[i]
        seq = generated[i, start : start + max_new].tolist()
        if eos_id in seq:
            eos_idx = seq.index(eos_id)
            seq = seq[:eos_idx]
        if pad_id is not None:
            seq = [t for t in seq if t != pad_id]
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        if not text and seq:
            text = tokenizer.decode(seq, skip_special_tokens=False).strip()
        preds.append(text)
    return preds


def _greedy_decode_t5(
    decoder,
    enc_tokens: torch.Tensor,
    enc_mask: torch.Tensor,
    max_new_tokens_list: List[int],
    eos_id: int,
    pad_id: int,
    max_length: int,
    min_new_tokens: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    bsz = enc_tokens.size(0)
    max_new = max(max_new_tokens_list) if max_new_tokens_list else 0
    if max_new <= 0:
        return torch.full((bsz, 1), pad_id, device=device, dtype=torch.long)
    max_new_t = torch.tensor(max_new_tokens_list, device=device)
    if min_new_tokens < 0:
        min_new_tokens = 0
    if min_new_tokens > 0:
        min_new_t = torch.minimum(max_new_t, torch.full_like(max_new_t, min_new_tokens))
    else:
        min_new_t = None
    generated = torch.full((bsz, 1), pad_id, device=device, dtype=torch.long)
    attn = torch.ones_like(generated)
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    for step in range(max_new):
        if done.all() or generated.size(1) >= max_length:
            break
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = decoder(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_tokens),
                attention_mask=enc_mask,
                decoder_input_ids=generated,
                decoder_attention_mask=attn,
                use_cache=False,
            )
            logits = out.logits[:, -1, :]
        if min_new_t is not None:
            logits = logits.clone()
            force_mask = min_new_t > step
            if force_mask.any():
                idx = force_mask.nonzero(as_tuple=True)[0]
                logits[idx, eos_id] = -1.0e9
        next_token = logits.argmax(dim=-1)
        over = (step + 1) >= max_new_t
        step_done = done | over | (next_token == eos_id)
        next_token = torch.where(step_done, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        attn = torch.cat([attn, (~step_done).long().unsqueeze(-1)], dim=-1)
        done = step_done
    return generated


def _apply_enc_proj(enc_tokens: torch.Tensor, enc_proj: nn.Module | None) -> torch.Tensor:
    if enc_proj is None:
        return enc_tokens
    return enc_proj(enc_tokens)


def evaluate(
    decoder: BertLMHeadModel,
    tokenizer: AutoTokenizer,
    vision: CLIPVisionTower,
    loader: DataLoader,
    device: torch.device,
    max_length: int,
    min_target_tokens: int,
    max_target_map: Dict[str, int],
    token_mode: str,
    amp: bool,
    decoder_arch: str = "bert",
    text_encoder: Any = None,
    text_tokenizer: AutoTokenizer | None = None,
    text_max_length: int = 256,
    use_plan_tokens: bool = False,
    use_multi_source_attn: bool = False,
    enc_proj: nn.Module | None = None,
    use_copy_head: bool = False,
    copy_logit_bias: float = 0.0,
    copy_max_tokens: int = 128,
    copy_sources: str = "ocr",
    text_token_scale: float = 1.0,
    plan_in_context: bool = False,
    plan_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    decoder.eval()
    vision.eval()
    if text_encoder is not None:
        text_encoder.eval()
    if enc_proj is not None:
        enc_proj.eval()
    totals = {"loss": 0.0, "short": 0.0, "long": 0.0, "desc": 0.0}
    counts = {"all": 0, "short": 0, "long": 0, "desc": 0}
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                images, texts, scales, contexts, _metas = batch
            else:
                images, texts, scales, contexts = batch
                _metas = [{} for _ in images]
            if plan_in_context:
                updated_contexts = []
                for meta, ctx in zip(_metas, contexts):
                    plan = _build_plan_text(meta or {}, **(plan_kwargs or {}))
                    if plan:
                        ctx = f"{plan}\n{ctx}".strip() if ctx else plan
                    updated_contexts.append(ctx)
                contexts = updated_contexts
            batch = vision.preprocess(images)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                enc = vision(batch)
                enc_tokens = enc["tokens"]
            enc_mask = torch.ones(enc_tokens.size()[:-1], device=device, dtype=torch.long)
            if text_encoder is not None and text_tokenizer is not None and use_multi_source_attn:
                input_ids, attn = _encode_text_context(
                    text_tokenizer,
                    contexts,
                    _metas,
                    text_max_length,
                    use_plan_tokens,
                    scales=scales,
                    token_mode=token_mode,
                    plan_kwargs=plan_kwargs,
                )
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    text_out = text_encoder(input_ids=input_ids, attention_mask=attn)
                    text_tokens = text_out.last_hidden_state
                text_tokens = _scale_text_tokens(text_tokens, text_token_scale)
                enc_tokens = torch.cat([enc_tokens, text_tokens], dim=1)
                enc_mask = torch.cat([enc_mask, attn], dim=1)
            enc_tokens = _apply_enc_proj(enc_tokens, enc_proj)
            tok, labels = tokenize_batch(
                tokenizer,
                texts,
                scales,
                contexts,
                max_length,
                min_target_tokens,
                max_target_map,
                token_mode,
                decoder_arch=decoder_arch,
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            labels = labels.to(device)
            copy_ids = None
            if use_copy_head:
                copy_ids = _build_copy_token_ids(
                    tokenizer, contexts, _metas, copy_max_tokens, sources=copy_sources
                )
            loss_t, _ = _forward_decoder(
                decoder,
                tok,
                labels,
                enc_tokens,
                enc_mask,
                decoder_arch=decoder_arch,
                copy_ids=copy_ids,
                copy_bias=copy_logit_bias if use_copy_head else 0.0,
                amp=amp,
                device=device,
            )
            loss = float(loss_t.item())
            bsz = len(images)
            totals["loss"] += loss * bsz
            counts["all"] += bsz
            for sc in scales:
                totals[sc] += loss
                counts[sc] += 1
    metrics = {"loss": totals["loss"] / max(1, counts["all"])}
    for sc in ["short", "long", "desc"]:
        if counts[sc] > 0:
            metrics[f"{sc}_loss"] = totals[sc] / counts[sc]
    return metrics


def _split_scale_indices(scales: List[str], caption_scales: set) -> Tuple[List[int], List[int]]:
    cap_idx: List[int] = []
    desc_idx: List[int] = []
    for i, sc in enumerate(scales):
        if sc == "desc":
            desc_idx.append(i)
        elif sc in caption_scales:
            cap_idx.append(i)
    return cap_idx, desc_idx


def evaluate_dual(
    decoder_caption: BertLMHeadModel,
    decoder_desc: BertLMHeadModel,
    tokenizer: AutoTokenizer,
    vision: CLIPVisionTower,
    loader: DataLoader,
    device: torch.device,
    max_length: int,
    min_target_tokens: int,
    max_target_map: Dict[str, int],
    token_mode: str,
    amp: bool,
    caption_scales: set,
    decoder_arch: str = "bert",
    text_encoder: Any = None,
    text_tokenizer: AutoTokenizer | None = None,
    text_max_length: int = 256,
    use_plan_tokens: bool = False,
    use_multi_source_attn: bool = False,
    enc_proj: nn.Module | None = None,
    use_copy_head: bool = False,
    copy_logit_bias: float = 0.0,
    copy_max_tokens: int = 128,
    copy_sources: str = "ocr",
    text_token_scale: float = 1.0,
    plan_in_context: bool = False,
    plan_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    decoder_caption.eval()
    decoder_desc.eval()
    vision.eval()
    if text_encoder is not None:
        text_encoder.eval()
    if enc_proj is not None:
        enc_proj.eval()
    totals = {"loss": 0.0, "short": 0.0, "long": 0.0, "desc": 0.0}
    counts = {"all": 0, "short": 0, "long": 0, "desc": 0}
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                images, texts, scales, contexts, _metas = batch
            else:
                images, texts, scales, contexts = batch
                _metas = [{} for _ in images]
            if plan_in_context:
                updated_contexts = []
                for meta, ctx in zip(_metas, contexts):
                    plan = _build_plan_text(meta or {}, **(plan_kwargs or {}))
                    if plan:
                        ctx = f"{plan}\n{ctx}".strip() if ctx else plan
                    updated_contexts.append(ctx)
                contexts = updated_contexts
            batch = vision.preprocess(images)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                enc = vision(batch)
                enc_tokens = enc["tokens"]
            enc_mask = torch.ones(enc_tokens.size()[:-1], device=device, dtype=torch.long)
            if text_encoder is not None and text_tokenizer is not None and use_multi_source_attn:
                input_ids, attn = _encode_text_context(
                    text_tokenizer,
                    contexts,
                    _metas,
                    text_max_length,
                    use_plan_tokens,
                    scales=scales,
                    token_mode=token_mode,
                    plan_kwargs=plan_kwargs,
                )
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    text_out = text_encoder(input_ids=input_ids, attention_mask=attn)
                    text_tokens = text_out.last_hidden_state
                text_tokens = _scale_text_tokens(text_tokens, text_token_scale)
                enc_tokens = torch.cat([enc_tokens, text_tokens], dim=1)
                enc_mask = torch.cat([enc_mask, attn], dim=1)
            enc_tokens = _apply_enc_proj(enc_tokens, enc_proj)
            copy_ids = None
            if use_copy_head:
                copy_ids = _build_copy_token_ids(
                    tokenizer, contexts, _metas, copy_max_tokens, sources=copy_sources
                )

            cap_idx, desc_idx = _split_scale_indices(scales, caption_scales)
            if cap_idx:
                cap_texts = [texts[i] for i in cap_idx]
                cap_scales = [scales[i] for i in cap_idx]
                cap_contexts = [contexts[i] for i in cap_idx]
                cap_metas = [(_metas or [])[i] for i in cap_idx] if _metas else None
                tok, labels = tokenize_batch(
                    tokenizer,
                    cap_texts,
                    cap_scales,
                    cap_contexts,
                    max_length,
                    min_target_tokens,
                    max_target_map,
                    token_mode,
                    decoder_arch=decoder_arch,
                )
                tok = {k: v.to(device) for k, v in tok.items()}
                labels = labels.to(device)
                cap_copy = [copy_ids[i] for i in cap_idx] if copy_ids is not None else None
                loss_t, _ = _forward_decoder(
                    decoder_caption,
                    tok,
                    labels,
                    enc_tokens[cap_idx],
                    enc_mask[cap_idx],
                    decoder_arch=decoder_arch,
                    copy_ids=cap_copy,
                    copy_bias=copy_logit_bias if use_copy_head else 0.0,
                    amp=amp,
                    device=device,
                )
                loss = float(loss_t.item())
                bsz = len(cap_idx)
                totals["loss"] += loss * bsz
                counts["all"] += bsz
                for sc in cap_scales:
                    totals[sc] += loss
                    counts[sc] += 1
            if desc_idx:
                desc_texts = [texts[i] for i in desc_idx]
                desc_scales = [scales[i] for i in desc_idx]
                desc_contexts = [contexts[i] for i in desc_idx]
                tok, labels = tokenize_batch(
                    tokenizer,
                    desc_texts,
                    desc_scales,
                    desc_contexts,
                    max_length,
                    min_target_tokens,
                    max_target_map,
                    token_mode,
                    decoder_arch=decoder_arch,
                )
                tok = {k: v.to(device) for k, v in tok.items()}
                labels = labels.to(device)
                desc_copy = [copy_ids[i] for i in desc_idx] if copy_ids is not None else None
                loss_t, _ = _forward_decoder(
                    decoder_desc,
                    tok,
                    labels,
                    enc_tokens[desc_idx],
                    enc_mask[desc_idx],
                    decoder_arch=decoder_arch,
                    copy_ids=desc_copy,
                    copy_bias=copy_logit_bias if use_copy_head else 0.0,
                    amp=amp,
                    device=device,
                )
                loss = float(loss_t.item())
                bsz = len(desc_idx)
                totals["loss"] += loss * bsz
                counts["all"] += bsz
                for sc in desc_scales:
                    totals[sc] += loss
                    counts[sc] += 1
    metrics = {"loss": totals["loss"] / max(1, counts["all"])}
    for sc in ["short", "long", "desc"]:
        if counts[sc] > 0:
            metrics[f"{sc}_loss"] = totals[sc] / counts[sc]
    return metrics


def main() -> None:
    import warnings

    warnings.filterwarnings("ignore", message="The pynvml package is deprecated*")
    warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated*")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_format", type=str, default="scicap", choices=["scicap", "jsonl_desc", "jsonl_multi"])
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--val_json", type=str, default="")
    ap.add_argument("--test_json", type=str, default="")
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="checkpoints/text_decoder_scicap")
    ap.add_argument("--visual_ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--val_batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_num_workers", type=int, default=-1)
    ap.add_argument("--max_steps", type=int, default=12000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop if no val improvement for this many steps (0 disables).",
    )
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--min_len_short", type=int, default=20)
    ap.add_argument("--min_len_long", type=int, default=40)
    ap.add_argument("--min_len_desc", type=int, default=40)
    ap.add_argument("--max_image_side", type=int, default=4096)
    ap.add_argument("--t5_model_name", type=str, default="t5-base")
    ap.add_argument(
        "--allow_unsafe_torch_load",
        action="store_true",
        help="Bypass the transformers torch.load safety check (CVE-2025-32434).",
    )
    ap.add_argument("--vision_train_last_n", type=int, default=0)
    ap.add_argument("--ocr_cov_weight", type=float, default=0.0)
    ap.add_argument("--ocr_cov_max_tokens", type=int, default=64)
    ap.add_argument("--cross_repeat_weight", type=float, default=0.0)
    ap.add_argument("--cross_repeat_ngram", type=int, default=4)
    ap.add_argument("--context_aug_prob", type=float, default=0.0)
    ap.add_argument("--context_drop_mention_prob", type=float, default=0.0)
    ap.add_argument("--context_drop_paragraph_prob", type=float, default=0.0)
    ap.add_argument("--context_drop_ocr_prob", type=float, default=0.0)
    ap.add_argument("--context_shuffle_prob", type=float, default=0.0)
    ap.add_argument("--text_warmup_steps", type=int, default=0)
    ap.add_argument("--text_token_scale", type=float, default=1.0)
    ap.add_argument("--text_train_last_n", type=int, default=0)
    ap.add_argument("--min_target_tokens", type=int, default=8)
    ap.add_argument("--max_target_short", type=int, default=64)
    ap.add_argument("--max_target_long", type=int, default=256)
    ap.add_argument("--max_target_desc", type=int, default=384)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--use_desc", action="store_true")
    ap.add_argument("--dual_decoder", action="store_true", help="Use separate caption/description decoders.")
    ap.add_argument(
        "--caption_scales",
        type=str,
        default="short,long",
        help="Comma-separated scales routed to the caption decoder (default: short,long).",
    )
    ap.add_argument("--context_mode", type=str, default="para_mention", choices=["none", "paragraph", "para_mention", "para_mention_ocr"])
    ap.add_argument("--sample_mode", type=str, default="random", choices=["random", "expand"])
    ap.add_argument("--token_mode", type=str, default="task", choices=["task", "scale", "both"])
    ap.add_argument("--token_schema", type=str, default="legacy", choices=["legacy", "simple"])
    ap.add_argument("--desc_key", type=str, default="description", help="JSONL description field name")
    ap.add_argument("--image_key", type=str, default="image_path", help="JSONL image path field name")
    ap.add_argument("--w_short", type=float, default=0.3)
    ap.add_argument("--w_long", type=float, default=0.5)
    ap.add_argument("--w_desc", type=float, default=0.2)
    ap.add_argument("--train_struct_jsonl", type=str, default="")
    ap.add_argument("--val_struct_jsonl", type=str, default="")
    ap.add_argument("--test_struct_jsonl", type=str, default="")
    ap.add_argument("--max_struct_nodes", type=int, default=64)
    ap.add_argument("--max_struct_roles", type=int, default=16)
    ap.add_argument("--desc_repeat_ngram", type=int, default=3)
    ap.add_argument("--desc_repeat_weight", type=float, default=0.2)
    ap.add_argument(
        "--repeat_scales",
        type=str,
        default="",
        help="Comma-separated scales to apply repeat penalty (default: desc).",
    )
    ap.add_argument("--desc_cov_weight", type=float, default=0.1)
    ap.add_argument("--desc_role_weight", type=float, default=0.1)
    ap.add_argument("--desc_cov_k_nodes", type=int, default=0)
    ap.add_argument("--desc_cov_k_roles", type=int, default=0)
    ap.add_argument(
        "--struct_cov_scales",
        type=str,
        default="desc",
        help="Comma-separated scales to apply struct coverage loss (default: desc).",
    )
    ap.add_argument("--p_drop_text", type=float, default=0.0)
    ap.add_argument("--p_drop_image", type=float, default=0.0)
    # Decoder + fusion controls.
    ap.add_argument("--text_encoder", type=str, default="none", choices=["none", "scibert"])
    ap.add_argument("--decoder_arch", type=str, default="bert", choices=["bert", "t5"])
    ap.add_argument("--use_copy_head", action="store_true")
    ap.add_argument(
        "--copy_sources",
        type=str,
        default="ocr",
        help=(
            "Comma-separated sources for copy bias (e.g., ocr,paragraph,mention,"
            "struct_nodes,struct_roles,struct_linearized,context)."
        ),
    )
    ap.add_argument("--use_graph_encoder", action="store_true")
    ap.add_argument("--use_plan_tokens", action="store_true")
    ap.add_argument("--plan_in_context", action="store_true")
    ap.add_argument("--plan_include_counts", action="store_true")
    ap.add_argument("--plan_include_types", action="store_true")
    ap.add_argument("--plan_max_types", type=int, default=6)
    ap.add_argument("--plan_include_linearized", action="store_true")
    ap.add_argument("--plan_max_linearized_chars", type=int, default=256)
    ap.add_argument("--use_multi_source_attn", action="store_true")
    ap.add_argument("--text_max_length", type=int, default=256)
    ap.add_argument("--copy_logit_bias", type=float, default=0.0)
    ap.add_argument("--copy_max_tokens", type=int, default=128)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val_sample_count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume_ckpt", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    set_token_schema(args.token_schema)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    if is_ddp:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    is_main = (not is_ddp) or dist.get_rank() == 0
    if args.dual_decoder and not args.use_desc:
        print("[warn] dual_decoder enabled but --use_desc is false; description decoder will be idle.")
    if args.use_graph_encoder and not args.use_plan_tokens:
        print("[info] enabling plan tokens for graph encoder mode.")
        args.use_plan_tokens = True
    if args.use_plan_tokens and args.text_encoder == "none":
        print("[info] enabling scibert text encoder for plan tokens.")
        args.text_encoder = "scibert"
    if args.text_encoder != "none" and not args.use_multi_source_attn:
        print("[info] enabling multi-source attention because text_encoder is enabled.")
        args.use_multi_source_attn = True
    if args.use_copy_head and args.copy_logit_bias <= 0:
        print("[info] use_copy_head enabled; setting copy_logit_bias=1.0.")
        args.copy_logit_bias = 1.0

    plan_kwargs = _plan_kwargs_from_args(args)

    repeat_scales = [s.strip() for s in args.repeat_scales.split(",") if s.strip()]
    if not repeat_scales:
        repeat_scales = ["desc"]
    repeat_scales_set = set(repeat_scales)

    struct_cov_scales = [s.strip() for s in args.struct_cov_scales.split(",") if s.strip()]
    if not struct_cov_scales:
        struct_cov_scales = ["desc"]
    struct_cov_scales_set = set(struct_cov_scales)

    scale_weights = {"short": args.w_short, "long": args.w_long, "desc": args.w_desc}
    max_target_map = {"short": args.max_target_short, "long": args.max_target_long, "desc": args.max_target_desc}
    if args.dataset_format == "scicap":
        need_train_meta = bool(
            args.train_struct_jsonl
            or args.desc_cov_weight > 0
            or args.desc_role_weight > 0
            or args.use_plan_tokens
            or args.plan_in_context
            or args.use_copy_head
            or args.ocr_cov_weight > 0
            or args.context_aug_prob > 0
        )
        train_ds = SciCapMultiScaleDataset(
            args.train_json,
            images_root=args.images_root,
            sample_mode=args.sample_mode,
            max_items=args.max_items,
            min_len_short=args.min_len_short,
            min_len_long=args.min_len_long,
            min_len_desc=args.min_len_desc,
            use_desc=args.use_desc,
            context_mode=args.context_mode,
            scale_weights=scale_weights,
            return_meta=need_train_meta,
            struct_jsonl=args.train_struct_jsonl or None,
            max_struct_nodes=args.max_struct_nodes,
            max_struct_roles=args.max_struct_roles,
            max_image_side=args.max_image_side,
        )
    elif args.dataset_format == "jsonl_multi":
        need_train_meta = bool(
            args.use_plan_tokens
            or args.plan_in_context
            or args.use_copy_head
            or args.ocr_cov_weight > 0
            or args.desc_cov_weight > 0
            or args.desc_role_weight > 0
            or args.context_aug_prob > 0
            or args.train_struct_jsonl
        )
        train_ds = JsonlMultiScaleDataset(
            args.train_json,
            images_root=args.images_root,
            sample_mode=args.sample_mode,
            max_items=args.max_items,
            min_len_short=args.min_len_short,
            min_len_long=args.min_len_long,
            min_len_desc=args.min_len_desc,
            scale_weights=scale_weights,
            return_meta=need_train_meta,
            image_key=args.image_key,
            desc_key=args.desc_key,
            struct_jsonl=args.train_struct_jsonl or None,
            max_struct_nodes=args.max_struct_nodes,
            max_struct_roles=args.max_struct_roles,
            max_image_side=args.max_image_side,
        )
    else:
        need_train_meta = False
        train_ds = JsonlDescDataset(
            args.train_json,
            images_root=args.images_root,
            max_items=args.max_items,
            min_len_desc=args.min_len_desc,
            desc_key=args.desc_key,
            image_key=args.image_key,
            return_meta=False,
            max_image_side=args.max_image_side,
            struct_jsonl=args.train_struct_jsonl or None,
            max_struct_nodes=args.max_struct_nodes,
            max_struct_roles=args.max_struct_roles,
        )
    train_sampler = None
    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
        )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    val_dl = None
    if args.val_json and is_main:
        if args.dataset_format == "scicap":
            val_ds = SciCapMultiScaleDataset(
                args.val_json,
                images_root=args.images_root,
                sample_mode="expand",
                max_items=args.max_items,
                min_len_short=args.min_len_short,
                min_len_long=args.min_len_long,
                min_len_desc=args.min_len_desc,
                use_desc=args.use_desc,
                context_mode=args.context_mode,
                scale_weights=scale_weights,
                return_meta=bool(
                    args.val_struct_jsonl
                    or args.use_plan_tokens
                    or args.plan_in_context
                    or args.use_copy_head
                    or args.desc_cov_weight > 0
                    or args.desc_role_weight > 0
                ),
                struct_jsonl=args.val_struct_jsonl or None,
                max_struct_nodes=args.max_struct_nodes,
                max_struct_roles=args.max_struct_roles,
                max_image_side=args.max_image_side,
            )
        elif args.dataset_format == "jsonl_multi":
            val_ds = JsonlMultiScaleDataset(
                args.val_json,
                images_root=args.images_root,
                sample_mode="expand",
                max_items=args.max_items,
                min_len_short=args.min_len_short,
                min_len_long=args.min_len_long,
                min_len_desc=args.min_len_desc,
                scale_weights=scale_weights,
                return_meta=bool(
                    args.use_plan_tokens
                    or args.plan_in_context
                    or args.use_copy_head
                    or args.desc_cov_weight > 0
                    or args.desc_role_weight > 0
                    or args.val_struct_jsonl
                ),
                image_key=args.image_key,
                desc_key=args.desc_key,
                struct_jsonl=args.val_struct_jsonl or None,
                max_struct_nodes=args.max_struct_nodes,
                max_struct_roles=args.max_struct_roles,
                max_image_side=args.max_image_side,
            )
        else:
            val_ds = JsonlDescDataset(
                args.val_json,
                images_root=args.images_root,
                max_items=args.max_items,
                min_len_desc=args.min_len_desc,
                desc_key=args.desc_key,
                image_key=args.image_key,
                return_meta=bool(args.use_plan_tokens or args.use_copy_head),
                max_image_side=args.max_image_side,
                struct_jsonl=args.val_struct_jsonl or None,
                max_struct_nodes=args.max_struct_nodes,
                max_struct_roles=args.max_struct_roles,
            )
        val_workers = args.val_num_workers if args.val_num_workers >= 0 else args.num_workers
        val_dl = DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=val_workers,
            collate_fn=collate_batch,
        )

    test_dl = None
    if args.test_json and is_main:
        if args.dataset_format == "scicap":
            test_ds = SciCapMultiScaleDataset(
                args.test_json,
                images_root=args.images_root,
                sample_mode="expand",
                max_items=args.max_items,
                min_len_short=args.min_len_short,
                min_len_long=args.min_len_long,
                min_len_desc=args.min_len_desc,
                use_desc=args.use_desc,
                context_mode=args.context_mode,
                scale_weights=scale_weights,
                return_meta=bool(
                    args.test_struct_jsonl
                    or args.use_plan_tokens
                    or args.plan_in_context
                    or args.use_copy_head
                    or args.desc_cov_weight > 0
                    or args.desc_role_weight > 0
                ),
                struct_jsonl=args.test_struct_jsonl or None,
                max_struct_nodes=args.max_struct_nodes,
                max_struct_roles=args.max_struct_roles,
                max_image_side=args.max_image_side,
            )
        elif args.dataset_format == "jsonl_multi":
            test_ds = JsonlMultiScaleDataset(
                args.test_json,
                images_root=args.images_root,
                sample_mode="expand",
                max_items=args.max_items,
                min_len_short=args.min_len_short,
                min_len_long=args.min_len_long,
                min_len_desc=args.min_len_desc,
                scale_weights=scale_weights,
                return_meta=bool(
                    args.use_plan_tokens
                    or args.plan_in_context
                    or args.use_copy_head
                    or args.desc_cov_weight > 0
                    or args.desc_role_weight > 0
                    or args.test_struct_jsonl
                ),
                image_key=args.image_key,
                desc_key=args.desc_key,
                struct_jsonl=args.test_struct_jsonl or None,
                max_struct_nodes=args.max_struct_nodes,
                max_struct_roles=args.max_struct_roles,
                max_image_side=args.max_image_side,
            )
        else:
            test_ds = JsonlDescDataset(
                args.test_json,
                images_root=args.images_root,
                max_items=args.max_items,
                min_len_desc=args.min_len_desc,
                desc_key=args.desc_key,
                image_key=args.image_key,
                return_meta=bool(args.use_plan_tokens or args.use_copy_head),
                max_image_side=args.max_image_side,
                struct_jsonl=args.test_struct_jsonl or None,
                max_struct_nodes=args.max_struct_nodes,
                max_struct_roles=args.max_struct_roles,
            )
        test_workers = args.val_num_workers if args.val_num_workers >= 0 else args.num_workers
        test_dl = DataLoader(
            test_ds,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=test_workers,
            collate_fn=collate_batch,
        )

    tokenizer = build_tokenizer(
        args.token_mode,
        decoder_arch=args.decoder_arch,
        t5_model_name=args.t5_model_name,
    )
    caption_scales = [s.strip() for s in args.caption_scales.split(",") if s.strip()]
    if "desc" in caption_scales:
        caption_scales = [s for s in caption_scales if s != "desc"]
        print("[warn] removing desc from caption_scales; desc is routed to description decoder.")
    caption_scales_set = set(caption_scales)

    decoder = build_decoder(
        tokenizer,
        decoder_arch=args.decoder_arch,
        t5_model_name=args.t5_model_name,
        allow_unsafe_torch_load=args.allow_unsafe_torch_load,
    ).to(device)
    decoder_desc = None
    if args.dual_decoder:
        decoder_desc = build_decoder(
            tokenizer,
            decoder_arch=args.decoder_arch,
            t5_model_name=args.t5_model_name,
            allow_unsafe_torch_load=args.allow_unsafe_torch_load,
        ).to(device)
    resume_step = 0
    resume_state = None
    if args.resume_ckpt:
        resume_state = torch.load(args.resume_ckpt, map_location="cpu")
        if args.dual_decoder:
            decoder.load_state_dict(
                resume_state.get("decoder_caption", resume_state.get("decoder", {})),
                strict=True,
            )
            if decoder_desc is not None and "decoder_desc" in resume_state:
                decoder_desc.load_state_dict(resume_state.get("decoder_desc", {}), strict=True)
            elif decoder_desc is not None:
                print("[warn] resume_ckpt missing decoder_desc; initializing description decoder from scratch.")
        else:
            decoder.load_state_dict(resume_state.get("decoder", {}), strict=True)
        resume_step = int(resume_state.get("step", 0))

    text_encoder = None
    text_tokenizer = None
    if args.text_encoder != "none":
        text_encoder, text_tokenizer = build_text_encoder(
            args.text_encoder,
            allow_unsafe_torch_load=args.allow_unsafe_torch_load,
        )
        text_encoder = text_encoder.to(device)
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad = False
        if args.text_train_last_n > 0:
            try:
                layers = text_encoder.encoder.layer
                last_n = min(args.text_train_last_n, len(layers))
                for i in range(len(layers) - last_n, len(layers)):
                    for p in layers[i].parameters():
                        p.requires_grad = True
                text_encoder.train()
            except Exception:
                # Fallback: keep frozen if structure differs.
                text_encoder.eval()

    vision = CLIPVisionTower(output_attentions=False).to(device)
    ckpt = torch.load(args.visual_ckpt, map_location="cpu")
    vision.load_state_dict(ckpt.get("model", {}), strict=False)
    for p in vision.parameters():
        p.requires_grad = False
    if args.vision_train_last_n > 0 and hasattr(vision, "siglip"):
        try:
            layers = vision.siglip.vision_model.encoder.layers
            last_n = min(args.vision_train_last_n, len(layers))
            for i in range(len(layers) - last_n, len(layers)):
                for p in layers[i].parameters():
                    p.requires_grad = True
            for p in vision.proj.parameters():
                p.requires_grad = True
            vision.train()
        except Exception:
            vision.eval()
    else:
        vision.eval()

    enc_proj = None
    dec_dim = decoder.config.d_model if args.decoder_arch == "t5" else decoder.config.hidden_size
    enc_dim = getattr(getattr(vision, "proj", None), "out_features", None)
    if enc_dim is None:
        enc_dim = dec_dim
    if enc_dim != dec_dim:
        enc_proj = nn.Linear(enc_dim, dec_dim).to(device)

    if resume_state:
        if enc_proj is not None:
            if "enc_proj" in resume_state and resume_state["enc_proj"] is not None:
                enc_proj.load_state_dict(resume_state["enc_proj"], strict=True)
            elif "enc_proj" in resume_state:
                print("[warn] resume_ckpt missing enc_proj weights; reinitializing.")
        elif "enc_proj" in resume_state and resume_state["enc_proj"] is not None:
            print("[warn] resume_ckpt includes enc_proj but current config does not use it.")

    if is_ddp:
        dec_params = sum(p.numel() for p in decoder.parameters())
        if decoder_desc is not None:
            desc_params = sum(p.numel() for p in decoder_desc.parameters())
        else:
            desc_params = 0
        print(f"[ddp] rank={dist.get_rank()} decoder_params={dec_params} desc_params={desc_params}")
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )
        if decoder_desc is not None:
            decoder_desc = torch.nn.parallel.DistributedDataParallel(
                decoder_desc, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
        if enc_proj is not None:
            enc_proj = torch.nn.parallel.DistributedDataParallel(
                enc_proj, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )

    params = [p for p in decoder.parameters() if p.requires_grad]
    if decoder_desc is not None:
        params += [p for p in decoder_desc.parameters() if p.requires_grad]
    if enc_proj is not None:
        params += list(enc_proj.parameters())
    if args.vision_train_last_n > 0:
        params += [p for p in vision.parameters() if p.requires_grad]
    if args.text_train_last_n > 0 and text_encoder is not None:
        params += [p for p in text_encoder.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    if resume_state:
        if "optim" in resume_state:
            optim.load_state_dict(resume_state["optim"])
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])
        if resume_step > 0 and "scheduler" not in resume_state:
            for _ in range(resume_step):
                scheduler.step()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    best_path = out_dir / "ckpt_best.pt"
    best_val = float("inf")
    best_step = -1

    step = resume_step
    t0 = time.time()
    data_iter = iter(train_dl)
    stop_training = False
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
        if len(batch) == 5:
            images, texts, scales, contexts, metas = batch
        else:
            images, texts, scales, contexts = batch
            metas = [{} for _ in images]
        metas = [m or {} for m in metas]
        if args.text_warmup_steps > 0 and step < args.text_warmup_steps:
            contexts = ["" for _ in contexts]
        contexts = _augment_contexts(contexts, metas, args, plan_kwargs=plan_kwargs)
        decoder.train()
        if decoder_desc is not None:
            decoder_desc.train()
        optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            batch = vision.preprocess(images)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                enc = vision(batch)
            enc_tokens = enc["tokens"]

        enc_mask = torch.ones(enc_tokens.size()[:-1], device=device, dtype=torch.long)
        if args.p_drop_image > 0:
            drop_mask = torch.rand(enc_tokens.size(0), device=device) < float(args.p_drop_image)
            if drop_mask.any():
                enc_tokens[drop_mask] = 0

        if args.p_drop_text > 0:
            contexts = ["" if random.random() < float(args.p_drop_text) else c for c in contexts]

        if text_encoder is not None and text_tokenizer is not None and args.use_multi_source_attn:
            input_ids, attn = _encode_text_context(
                text_tokenizer,
                contexts,
                metas,
                args.text_max_length,
                args.use_plan_tokens,
                scales=scales,
                token_mode=args.token_mode,
                plan_kwargs=plan_kwargs,
            )
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                text_out = text_encoder(input_ids=input_ids, attention_mask=attn)
                text_tokens = text_out.last_hidden_state
            text_tokens = _scale_text_tokens(text_tokens, args.text_token_scale)
            enc_tokens = torch.cat([enc_tokens, text_tokens], dim=1)
            enc_mask = torch.cat([enc_mask, attn], dim=1)

        enc_tokens = _apply_enc_proj(enc_tokens, enc_proj)

        copy_ids = None
        copy_bias = args.copy_logit_bias if args.use_copy_head else 0.0
        if args.use_copy_head:
            copy_ids = _build_copy_token_ids(
                tokenizer, contexts, metas, args.copy_max_tokens, sources=args.copy_sources
            )

        loss = None
        loss_cap = None
        loss_desc = None
        rep_loss = None
        cov_node_loss = None
        cov_role_loss = None
        cap_count = 0
        desc_count = 0

        if decoder_desc is None:
            tok, labels = tokenize_batch(
                tokenizer,
                texts,
                scales,
                contexts,
                args.max_length,
                args.min_target_tokens,
                max_target_map,
                args.token_mode,
                decoder_arch=args.decoder_arch,
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            labels = labels.to(device)
            loss, logits = _forward_decoder(
                decoder,
                tok,
                labels,
                enc_tokens,
                enc_mask,
                decoder_arch=args.decoder_arch,
                copy_ids=copy_ids,
                copy_bias=copy_bias,
                amp=args.amp,
                device=device,
            )

            if args.desc_repeat_weight > 0:
                rep_loss = _repeat_ngram_unlikelihood(
                    logits,
                    labels,
                    scales,
                    args.desc_repeat_ngram,
                    target_scales=repeat_scales_set,
                )
                if rep_loss is not None:
                    loss = loss + args.desc_repeat_weight * rep_loss

            if args.cross_repeat_weight > 0:
                cross_rep = _cross_repeat_ngram_unlikelihood(
                    logits,
                    labels,
                    args.cross_repeat_ngram,
                    target_scales=repeat_scales_set,
                    scales=scales,
                )
                if cross_rep is not None:
                    loss = loss + args.cross_repeat_weight * cross_rep

            if (args.desc_cov_weight > 0 or args.desc_role_weight > 0) and metas:
                node_losses: List[torch.Tensor] = []
                role_losses: List[torch.Tensor] = []
                for i, scale in enumerate(scales):
                    if scale not in struct_cov_scales_set:
                        continue
                    meta = metas[i] or {}
                    if args.desc_cov_weight > 0:
                        node_texts = meta.get("struct_nodes") or []
                        node_ids = _token_ids_from_texts(tokenizer, node_texts, args.max_struct_nodes)
                        node_loss = _coverage_loss_for_ids(
                            logits[i],
                            labels[i],
                            node_ids,
                            args.desc_cov_k_nodes,
                        )
                        if node_loss is not None:
                            node_losses.append(node_loss)
                    if args.desc_role_weight > 0:
                        role_texts = meta.get("struct_roles") or []
                        role_ids = _token_ids_from_texts(tokenizer, role_texts, args.max_struct_roles)
                        role_loss = _coverage_loss_for_ids(
                            logits[i],
                            labels[i],
                            role_ids,
                            args.desc_cov_k_roles,
                        )
                        if role_loss is not None:
                            role_losses.append(role_loss)
                if node_losses:
                    cov_node_loss = torch.stack(node_losses).mean()
                    loss = loss + args.desc_cov_weight * cov_node_loss
                if role_losses:
                    cov_role_loss = torch.stack(role_losses).mean()
                    loss = loss + args.desc_role_weight * cov_role_loss

            if args.ocr_cov_weight > 0 and metas:
                ocr_losses: List[torch.Tensor] = []
                for i, meta in enumerate(metas):
                    ocr_text = meta.get("ocr") or ""
                    if not ocr_text:
                        continue
                    ocr_ids = _token_ids_from_texts(tokenizer, [ocr_text], args.ocr_cov_max_tokens)
                    if not ocr_ids:
                        continue
                    ocr_loss = _coverage_loss_for_ids(
                        logits[i],
                        labels[i],
                        ocr_ids,
                        args.ocr_cov_max_tokens,
                    )
                    if ocr_loss is not None:
                        ocr_losses.append(ocr_loss)
                if ocr_losses:
                    loss = loss + args.ocr_cov_weight * torch.stack(ocr_losses).mean()
        else:
            cap_idx, desc_idx = _split_scale_indices(scales, caption_scales_set)
            total_loss = torch.tensor(0.0, device=device)
            total_count = 0

            if cap_idx:
                cap_texts = [texts[i] for i in cap_idx]
                cap_scales = [scales[i] for i in cap_idx]
                cap_contexts = [contexts[i] for i in cap_idx]
                cap_metas = [metas[i] for i in cap_idx] if metas else None
                tok, labels = tokenize_batch(
                    tokenizer,
                    cap_texts,
                    cap_scales,
                    cap_contexts,
                    args.max_length,
                    args.min_target_tokens,
                    max_target_map,
                    args.token_mode,
                    decoder_arch=args.decoder_arch,
                )
                tok = {k: v.to(device) for k, v in tok.items()}
                labels = labels.to(device)
                cap_copy = [copy_ids[i] for i in cap_idx] if copy_ids is not None else None
                loss_cap, cap_logits = _forward_decoder(
                    decoder,
                    tok,
                    labels,
                    enc_tokens[cap_idx],
                    enc_mask[cap_idx],
                    decoder_arch=args.decoder_arch,
                    copy_ids=cap_copy,
                    copy_bias=copy_bias,
                    amp=args.amp,
                    device=device,
                )
                if args.cross_repeat_weight > 0:
                    cross_rep = _cross_repeat_ngram_unlikelihood(
                        cap_logits,
                        labels,
                        args.cross_repeat_ngram,
                        target_scales=repeat_scales_set,
                        scales=cap_scales,
                    )
                    if cross_rep is not None:
                        loss_cap = loss_cap + args.cross_repeat_weight * cross_rep
                if args.ocr_cov_weight > 0 and cap_metas:
                    ocr_losses = []
                    for i, meta in enumerate(cap_metas):
                        ocr_text = meta.get("ocr") or ""
                        if not ocr_text:
                            continue
                        ocr_ids = _token_ids_from_texts(tokenizer, [ocr_text], args.ocr_cov_max_tokens)
                        if not ocr_ids:
                            continue
                        ocr_loss = _coverage_loss_for_ids(
                            cap_logits[i],
                            labels[i],
                            ocr_ids,
                            args.ocr_cov_max_tokens,
                        )
                        if ocr_loss is not None:
                            ocr_losses.append(ocr_loss)
                    if ocr_losses:
                        loss_cap = loss_cap + args.ocr_cov_weight * torch.stack(ocr_losses).mean()
                if (args.desc_cov_weight > 0 or args.desc_role_weight > 0) and cap_metas:
                    node_losses = []
                    role_losses = []
                    for i, scale in enumerate(cap_scales):
                        if scale not in struct_cov_scales_set:
                            continue
                        meta = cap_metas[i] or {}
                        if args.desc_cov_weight > 0:
                            node_texts = meta.get("struct_nodes") or []
                            node_ids = _token_ids_from_texts(tokenizer, node_texts, args.max_struct_nodes)
                            node_loss = _coverage_loss_for_ids(
                                cap_logits[i],
                                labels[i],
                                node_ids,
                                args.desc_cov_k_nodes,
                            )
                            if node_loss is not None:
                                node_losses.append(node_loss)
                        if args.desc_role_weight > 0:
                            role_texts = meta.get("struct_roles") or []
                            role_ids = _token_ids_from_texts(tokenizer, role_texts, args.max_struct_roles)
                            role_loss = _coverage_loss_for_ids(
                                cap_logits[i],
                                labels[i],
                                role_ids,
                                args.desc_cov_k_roles,
                            )
                            if role_loss is not None:
                                role_losses.append(role_loss)
                    if node_losses:
                        loss_cap = loss_cap + args.desc_cov_weight * torch.stack(node_losses).mean()
                    if role_losses:
                        loss_cap = loss_cap + args.desc_role_weight * torch.stack(role_losses).mean()
                cap_count = len(cap_idx)
                total_loss = total_loss + loss_cap * cap_count
                total_count += cap_count

            if desc_idx:
                desc_texts = [texts[i] for i in desc_idx]
                desc_scales = [scales[i] for i in desc_idx]
                desc_contexts = [contexts[i] for i in desc_idx]
                desc_metas = [metas[i] for i in desc_idx] if metas else None
                tok, labels = tokenize_batch(
                    tokenizer,
                    desc_texts,
                    desc_scales,
                    desc_contexts,
                    args.max_length,
                    args.min_target_tokens,
                    max_target_map,
                    args.token_mode,
                    decoder_arch=args.decoder_arch,
                )
                tok = {k: v.to(device) for k, v in tok.items()}
                labels = labels.to(device)
                desc_copy = [copy_ids[i] for i in desc_idx] if copy_ids is not None else None
                loss_desc, desc_logits = _forward_decoder(
                    decoder_desc,
                    tok,
                    labels,
                    enc_tokens[desc_idx],
                    enc_mask[desc_idx],
                    decoder_arch=args.decoder_arch,
                    copy_ids=desc_copy,
                    copy_bias=copy_bias,
                    amp=args.amp,
                    device=device,
                )
                loss_desc_total = loss_desc

                if args.desc_repeat_weight > 0:
                    rep_loss = _repeat_ngram_unlikelihood(
                        desc_logits,
                        labels,
                        desc_scales,
                        args.desc_repeat_ngram,
                        target_scales=repeat_scales_set,
                    )
                    if rep_loss is not None:
                        loss_desc_total = loss_desc_total + args.desc_repeat_weight * rep_loss
                if args.cross_repeat_weight > 0:
                    cross_rep = _cross_repeat_ngram_unlikelihood(
                        desc_logits,
                        labels,
                        args.cross_repeat_ngram,
                        target_scales=repeat_scales_set,
                        scales=desc_scales,
                    )
                    if cross_rep is not None:
                        loss_desc_total = loss_desc_total + args.cross_repeat_weight * cross_rep

                if (args.desc_cov_weight > 0 or args.desc_role_weight > 0) and desc_metas:
                    node_losses = []
                    role_losses = []
                    for i, meta in enumerate(desc_metas):
                        if desc_scales[i] not in struct_cov_scales_set:
                            continue
                        meta = meta or {}
                        if args.desc_cov_weight > 0:
                            node_texts = meta.get("struct_nodes") or []
                            node_ids = _token_ids_from_texts(tokenizer, node_texts, args.max_struct_nodes)
                            node_loss = _coverage_loss_for_ids(
                                desc_logits[i],
                                labels[i],
                                node_ids,
                                args.desc_cov_k_nodes,
                            )
                            if node_loss is not None:
                                node_losses.append(node_loss)
                        if args.desc_role_weight > 0:
                            role_texts = meta.get("struct_roles") or []
                            role_ids = _token_ids_from_texts(tokenizer, role_texts, args.max_struct_roles)
                            role_loss = _coverage_loss_for_ids(
                                desc_logits[i],
                                labels[i],
                                role_ids,
                                args.desc_cov_k_roles,
                            )
                            if role_loss is not None:
                                role_losses.append(role_loss)
                    if node_losses:
                        cov_node_loss = torch.stack(node_losses).mean()
                        loss_desc_total = loss_desc_total + args.desc_cov_weight * cov_node_loss
                    if role_losses:
                        cov_role_loss = torch.stack(role_losses).mean()
                        loss_desc_total = loss_desc_total + args.desc_role_weight * cov_role_loss
                if args.ocr_cov_weight > 0 and desc_metas:
                    ocr_losses = []
                    for i, meta in enumerate(desc_metas):
                        ocr_text = meta.get("ocr") or ""
                        if not ocr_text:
                            continue
                        ocr_ids = _token_ids_from_texts(tokenizer, [ocr_text], args.ocr_cov_max_tokens)
                        if not ocr_ids:
                            continue
                        ocr_loss = _coverage_loss_for_ids(
                            desc_logits[i],
                            labels[i],
                            ocr_ids,
                            args.ocr_cov_max_tokens,
                        )
                        if ocr_loss is not None:
                            ocr_losses.append(ocr_loss)
                    if ocr_losses:
                        loss_desc_total = loss_desc_total + args.ocr_cov_weight * torch.stack(ocr_losses).mean()

                desc_count = len(desc_idx)
                total_loss = total_loss + loss_desc_total * desc_count
                total_count += desc_count

            if total_count > 0:
                loss = total_loss / float(total_count)
            else:
                raise ValueError("empty batch after scale split; check caption_scales/use_desc settings")

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        if is_main and step % args.log_every == 0:
            elapsed = time.time() - t0
            msg = {
                "step": step,
                "loss": float(loss.item()),
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_sec": int(elapsed),
            }
            if loss_cap is not None:
                msg["loss_cap"] = float(loss_cap.item())
                msg["cap_count"] = cap_count
            if loss_desc is not None:
                msg["loss_desc"] = float(loss_desc.item())
                msg["desc_count"] = desc_count
            if rep_loss is not None:
                msg["loss_repeat"] = float(rep_loss.item())
            if cov_node_loss is not None:
                msg["loss_cov_nodes"] = float(cov_node_loss.item())
            if cov_role_loss is not None:
                msg["loss_cov_roles"] = float(cov_role_loss.item())
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            print(f"[train] step={step} loss={msg['loss']:.6f} lr={msg['lr']:.2e}")

        if is_main and val_dl and args.val_every > 0 and step > 0 and step % args.val_every == 0:
            if decoder_desc is not None:
                metrics = evaluate_dual(
                    decoder,
                    decoder_desc,
                    tokenizer,
                    vision,
                    val_dl,
                    device,
                    args.max_length,
                    args.min_target_tokens,
                    max_target_map,
                    args.token_mode,
                    args.amp,
                    caption_scales_set,
                    decoder_arch=args.decoder_arch,
                    text_encoder=text_encoder,
                    text_tokenizer=text_tokenizer,
                    text_max_length=args.text_max_length,
                    use_plan_tokens=args.use_plan_tokens,
                    use_multi_source_attn=args.use_multi_source_attn,
                    enc_proj=enc_proj,
                    use_copy_head=args.use_copy_head,
                    copy_logit_bias=args.copy_logit_bias,
                    copy_max_tokens=args.copy_max_tokens,
                    copy_sources=args.copy_sources,
                    text_token_scale=args.text_token_scale,
                    plan_in_context=args.plan_in_context,
                    plan_kwargs=plan_kwargs,
                )
            else:
                metrics = evaluate(
                    decoder,
                    tokenizer,
                    vision,
                    val_dl,
                    device,
                    args.max_length,
                    args.min_target_tokens,
                    max_target_map,
                    args.token_mode,
                    args.amp,
                    decoder_arch=args.decoder_arch,
                    text_encoder=text_encoder,
                    text_tokenizer=text_tokenizer,
                    text_max_length=args.text_max_length,
                    use_plan_tokens=args.use_plan_tokens,
                    use_multi_source_attn=args.use_multi_source_attn,
                    enc_proj=enc_proj,
                    use_copy_head=args.use_copy_head,
                    copy_logit_bias=args.copy_logit_bias,
                    copy_max_tokens=args.copy_max_tokens,
                    copy_sources=args.copy_sources,
                    text_token_scale=args.text_token_scale,
                    plan_in_context=args.plan_in_context,
                    plan_kwargs=plan_kwargs,
                )
            metrics["step"] = step
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"val": metrics}, ensure_ascii=False) + "\n")
            print(f"[val] step={step} loss={metrics['loss']:.6f}")
            if args.val_sample_count > 0:
                sample_count = min(args.val_sample_count, 3)
                with torch.no_grad():
                    batch = None
                    if val_ds is not None and len(val_ds) > 0:
                        desired = ["short", "long"]
                        if args.use_desc:
                            desired.append("desc")
                        picked = []
                        picked_scales = set()
                        indices = list(range(len(val_ds)))
                        random.shuffle(indices)
                        for idx in indices:
                            item = val_ds[idx]
                            if len(item) < 3:
                                continue
                            scale = item[2]
                            if scale in desired and scale not in picked_scales:
                                picked.append(item)
                                picked_scales.add(scale)
                            if len(picked_scales) >= min(sample_count, len(desired)):
                                break
                        if not picked:
                            picked = [val_ds[random.randint(0, len(val_ds) - 1)]]
                        if picked:
                            if len(picked[0]) == 5:
                                images, texts, scales, contexts, metas = zip(*picked)
                                images, texts, scales, contexts, metas = (
                                    list(images),
                                    list(texts),
                                    list(scales),
                                    list(contexts),
                                    list(metas),
                                )
                            else:
                                images, texts, scales, contexts = zip(*picked)
                                images, texts, scales, contexts = (
                                    list(images),
                                    list(texts),
                                    list(scales),
                                    list(contexts),
                                )
                                metas = [{} for _ in images]
                            batch = (images, texts, scales, contexts, metas)
                    if batch:
                        images, texts, scales, contexts, metas = batch
                    metas = [m or {} for m in metas]
                    batch_v = vision.preprocess(images)
                    batch_v = {k: v.to(device) for k, v in batch_v.items()}
                    with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                        enc = vision(batch_v)
                        enc_tokens = enc["tokens"]
                    enc_mask = torch.ones(enc_tokens.size()[:-1], device=device, dtype=torch.long)
                    if text_encoder is not None and text_tokenizer is not None and args.use_multi_source_attn:
                        input_ids, attn = _encode_text_context(
                            text_tokenizer,
                            contexts,
                            metas,
                            args.text_max_length,
                            args.use_plan_tokens,
                            scales=scales,
                            token_mode=args.token_mode,
                            plan_kwargs=plan_kwargs,
                        )
                        input_ids = input_ids.to(device)
                        attn = attn.to(device)
                        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                            text_out = text_encoder(input_ids=input_ids, attention_mask=attn)
                            text_tokens = text_out.last_hidden_state
                        text_tokens = _scale_text_tokens(text_tokens, args.text_token_scale)
                        enc_tokens = torch.cat([enc_tokens, text_tokens], dim=1)
                        enc_mask = torch.cat([enc_mask, attn], dim=1)
                    enc_tokens = _apply_enc_proj(enc_tokens, enc_proj)
                    preds = []
                    if args.decoder_arch == "t5":
                        for i in range(min(sample_count, len(scales))):
                            use_desc = scales[i] == "desc"
                            dec = decoder_desc if (use_desc and decoder_desc is not None) else decoder
                            gen = _greedy_decode_t5(
                                dec,
                                enc_tokens[i : i + 1],
                                enc_mask[i : i + 1],
                                [max_target_map.get(scales[i], 64)],
                                tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id or 0,
                                tokenizer.pad_token_id or 0,
                                args.max_length,
                                args.min_target_tokens,
                                device,
                                args.amp,
                            )
                            preds.append(tokenizer.decode(gen[0].tolist(), skip_special_tokens=True).strip())
                    else:
                        sample_contexts = contexts
                        if args.plan_in_context:
                            updated = []
                            for meta, ctx in zip(metas, contexts):
                                plan = _build_plan_text(meta or {}, **plan_kwargs)
                                if plan:
                                    ctx = f"{plan}\n{ctx}".strip() if ctx else plan
                                updated.append(ctx)
                            sample_contexts = updated
                        preds = [""] * len(scales)
                        forbidden_ids = [tokenizer.pad_token_id, tokenizer.cls_token_id]
                        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id or 0
                        pad_id = tokenizer.pad_token_id or 0
                        if decoder_desc is not None:
                            cap_idx, desc_idx = _split_scale_indices(scales, caption_scales_set)
                            if cap_idx:
                                cap_scales = [scales[i] for i in cap_idx]
                                cap_ctx = [sample_contexts[i] for i in cap_idx]
                                inp, attn, prefix_lens, max_new = _build_prefix_batch_for_decode(
                                    tokenizer,
                                    cap_scales,
                                    cap_ctx,
                                    args.max_length,
                                    max_target_map,
                                    args.token_mode,
                                )
                                inp = inp.to(device)
                                attn = attn.to(device)
                                idx_t = torch.tensor(cap_idx, device=device, dtype=torch.long)
                                gen = _greedy_decode_bert(
                                    decoder,
                                    inp,
                                    attn,
                                    enc_tokens.index_select(0, idx_t),
                                    enc_mask.index_select(0, idx_t),
                                    max_new,
                                    eos_id,
                                    pad_id,
                                    args.max_length,
                                    args.min_target_tokens,
                                    forbidden_ids,
                                )
                                out = _decode_generated(tokenizer, gen, prefix_lens, max_new, eos_id)
                                for i, pred in zip(cap_idx, out):
                                    preds[i] = pred
                            if desc_idx:
                                desc_scales = [scales[i] for i in desc_idx]
                                desc_ctx = [sample_contexts[i] for i in desc_idx]
                                inp, attn, prefix_lens, max_new = _build_prefix_batch_for_decode(
                                    tokenizer,
                                    desc_scales,
                                    desc_ctx,
                                    args.max_length,
                                    max_target_map,
                                    args.token_mode,
                                )
                                inp = inp.to(device)
                                attn = attn.to(device)
                                idx_t = torch.tensor(desc_idx, device=device, dtype=torch.long)
                                gen = _greedy_decode_bert(
                                    decoder_desc,
                                    inp,
                                    attn,
                                    enc_tokens.index_select(0, idx_t),
                                    enc_mask.index_select(0, idx_t),
                                    max_new,
                                    eos_id,
                                    pad_id,
                                    args.max_length,
                                    args.min_target_tokens,
                                    forbidden_ids,
                                )
                                out = _decode_generated(tokenizer, gen, prefix_lens, max_new, eos_id)
                                for i, pred in zip(desc_idx, out):
                                    preds[i] = pred
                        else:
                            inp, attn, prefix_lens, max_new = _build_prefix_batch_for_decode(
                                tokenizer,
                                scales,
                                sample_contexts,
                                args.max_length,
                                max_target_map,
                                args.token_mode,
                            )
                            inp = inp.to(device)
                            attn = attn.to(device)
                            gen = _greedy_decode_bert(
                                decoder,
                                inp,
                                attn,
                                enc_tokens,
                                enc_mask,
                                max_new,
                                eos_id,
                                pad_id,
                                args.max_length,
                                args.min_target_tokens,
                                forbidden_ids,
                            )
                            preds = _decode_generated(tokenizer, gen, prefix_lens, max_new, eos_id)
                    for i in range(min(sample_count, len(preds))):
                        print(f"[val_sample] scale={scales[i]} pred={preds[i][:160]} ref={texts[i][:160]}")
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                best_step = step
                if decoder_desc is not None:
                    torch.save(
                        {
                            "step": step,
                            "decoder_caption": _unwrap_state(decoder),
                            "decoder_desc": _unwrap_state(decoder_desc),
                            "tokenizer": tokenizer.get_vocab(),
                            "enc_proj": _unwrap_state(enc_proj) if enc_proj is not None else None,
                        },
                        best_path,
                    )
                else:
                    torch.save(
                        {
                            "step": step,
                            "decoder": _unwrap_state(decoder),
                            "tokenizer": tokenizer.get_vocab(),
                            "enc_proj": _unwrap_state(enc_proj) if enc_proj is not None else None,
                        },
                        best_path,
                    )
            if args.early_stop_patience > 0 and best_step >= 0:
                if (step - best_step) >= args.early_stop_patience:
                    print(
                        f"[early_stop] step={step} best_step={best_step} "
                        f"patience={args.early_stop_patience}"
                    )
                    stop_training = True

        if is_main and step > 0 and step % args.save_every == 0:
            if decoder_desc is not None:
                torch.save(
                    {
                        "step": step,
                        "decoder_caption": _unwrap_state(decoder),
                        "decoder_desc": _unwrap_state(decoder_desc),
                        "tokenizer": tokenizer.get_vocab(),
                        "enc_proj": _unwrap_state(enc_proj) if enc_proj is not None else None,
                        "optim": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    out_dir / f"ckpt_step_{step}.pt",
                )
            else:
                torch.save(
                    {
                        "step": step,
                        "decoder": _unwrap_state(decoder),
                        "tokenizer": tokenizer.get_vocab(),
                        "enc_proj": _unwrap_state(enc_proj) if enc_proj is not None else None,
                        "optim": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    out_dir / f"ckpt_step_{step}.pt",
                )

        step += 1
        if is_ddp:
            flag = torch.tensor(1 if stop_training else 0, device=device)
            dist.broadcast(flag, src=0)
            stop_training = bool(flag.item())
        if stop_training:
            break

    if is_main and decoder_desc is not None:
        torch.save(
            {
                "step": step,
                "decoder_caption": _unwrap_state(decoder),
                "decoder_desc": _unwrap_state(decoder_desc),
                "tokenizer": tokenizer.get_vocab(),
                "enc_proj": _unwrap_state(enc_proj) if enc_proj is not None else None,
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            out_dir / "ckpt_last.pt",
        )
    elif is_main:
        torch.save(
            {
                "step": step,
                "decoder": _unwrap_state(decoder),
                "tokenizer": tokenizer.get_vocab(),
                "enc_proj": _unwrap_state(enc_proj) if enc_proj is not None else None,
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            out_dir / "ckpt_last.pt",
        )

    if is_main and val_dl:
        if decoder_desc is not None:
            metrics = evaluate_dual(
                decoder,
                decoder_desc,
                tokenizer,
                vision,
                val_dl,
                device,
                args.max_length,
                args.min_target_tokens,
                max_target_map,
                args.token_mode,
                args.amp,
                caption_scales_set,
                decoder_arch=args.decoder_arch,
                text_encoder=text_encoder,
                text_tokenizer=text_tokenizer,
                text_max_length=args.text_max_length,
                use_plan_tokens=args.use_plan_tokens,
                use_multi_source_attn=args.use_multi_source_attn,
                enc_proj=enc_proj,
                use_copy_head=args.use_copy_head,
                copy_logit_bias=args.copy_logit_bias,
                copy_max_tokens=args.copy_max_tokens,
                copy_sources=args.copy_sources,
                text_token_scale=args.text_token_scale,
                plan_in_context=args.plan_in_context,
                plan_kwargs=plan_kwargs,
            )
        else:
            metrics = evaluate(
                decoder,
                tokenizer,
                vision,
                val_dl,
                device,
                args.max_length,
                args.min_target_tokens,
                max_target_map,
                args.token_mode,
                args.amp,
                decoder_arch=args.decoder_arch,
                text_encoder=text_encoder,
                text_tokenizer=text_tokenizer,
                text_max_length=args.text_max_length,
                use_plan_tokens=args.use_plan_tokens,
                use_multi_source_attn=args.use_multi_source_attn,
                enc_proj=enc_proj,
                use_copy_head=args.use_copy_head,
                copy_logit_bias=args.copy_logit_bias,
                copy_max_tokens=args.copy_max_tokens,
                copy_sources=args.copy_sources,
                text_token_scale=args.text_token_scale,
                plan_in_context=args.plan_in_context,
                plan_kwargs=plan_kwargs,
            )
        metrics["step"] = step
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"val_final": metrics}, ensure_ascii=False) + "\n")
        print(f"[val_final] step={step} loss={metrics['loss']:.6f}")

    if is_main and test_dl:
        if best_path.exists():
            best_state = torch.load(best_path, map_location="cpu")
            if decoder_desc is not None:
                decoder.load_state_dict(
                    best_state.get("decoder_caption", best_state.get("decoder", {})),
                    strict=True,
                )
                if "decoder_desc" in best_state:
                    decoder_desc.load_state_dict(best_state.get("decoder_desc", {}), strict=True)
            else:
                decoder.load_state_dict(best_state.get("decoder", {}), strict=True)
            if enc_proj is not None and best_state.get("enc_proj") is not None:
                enc_proj.load_state_dict(best_state.get("enc_proj", {}), strict=True)
            best_step = int(best_state.get("step", best_step))
        if decoder_desc is not None:
            test_metrics = evaluate_dual(
                decoder,
                decoder_desc,
                tokenizer,
                vision,
                test_dl,
                device,
                args.max_length,
                args.min_target_tokens,
                max_target_map,
                args.token_mode,
                args.amp,
                caption_scales_set,
                decoder_arch=args.decoder_arch,
                text_encoder=text_encoder,
                text_tokenizer=text_tokenizer,
                text_max_length=args.text_max_length,
                use_plan_tokens=args.use_plan_tokens,
                use_multi_source_attn=args.use_multi_source_attn,
                enc_proj=enc_proj,
                use_copy_head=args.use_copy_head,
                copy_logit_bias=args.copy_logit_bias,
                copy_max_tokens=args.copy_max_tokens,
                copy_sources=args.copy_sources,
                text_token_scale=args.text_token_scale,
                plan_in_context=args.plan_in_context,
                plan_kwargs=plan_kwargs,
            )
        else:
            test_metrics = evaluate(
                decoder,
                tokenizer,
                vision,
                test_dl,
                device,
                args.max_length,
                args.min_target_tokens,
                max_target_map,
                args.token_mode,
                args.amp,
                decoder_arch=args.decoder_arch,
                text_encoder=text_encoder,
                text_tokenizer=text_tokenizer,
                text_max_length=args.text_max_length,
                use_plan_tokens=args.use_plan_tokens,
                use_multi_source_attn=args.use_multi_source_attn,
                enc_proj=enc_proj,
                use_copy_head=args.use_copy_head,
                copy_logit_bias=args.copy_logit_bias,
                copy_max_tokens=args.copy_max_tokens,
                copy_sources=args.copy_sources,
                text_token_scale=args.text_token_scale,
                plan_in_context=args.plan_in_context,
                plan_kwargs=plan_kwargs,
            )
        test_metrics["best_step"] = best_step
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"test": test_metrics}, ensure_ascii=False) + "\n")
        print(f"[test] best_step={best_step} loss={test_metrics['loss']:.6f}")


if __name__ == "__main__":
    main()

