#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput

from scixplain.models import CLIPVisionTower
import scixplain.tools.train_text_decoder as ttd
from scixplain.tools.train_text_decoder import (
    SciCapMultiScaleDataset,
    JsonlDescDataset,
    JsonlMultiScaleDataset,
    collate_batch,
    build_tokenizer,
    build_decoder,
    build_text_encoder,
    _build_plan_text,
    _plan_kwargs_from_args,
    _encode_text_context,
    _flatten_text,
    _build_copy_token_ids,
    _apply_copy_bias,
    _build_prefix_ids,
    set_token_schema,
)

STRUCT_KEYWORDS = {
    "node",
    "nodes",
    "edge",
    "edges",
    "arrow",
    "arrows",
    "flow",
    "flowchart",
    "diagram",
    "graph",
    "tree",
    "hierarchy",
    "layer",
    "module",
    "block",
    "box",
    "boxes",
    "circle",
    "circles",
    "ellipse",
    "ellipses",
    "rectangle",
    "rectangles",
    "diamond",
    "diamonds",
    "triangle",
    "triangles",
    "connection",
    "connections",
    "link",
    "links",
    "pipeline",
    "state",
    "states",
    "transition",
    "transitions",
    "input",
    "output",
    "branch",
    "branches",
    "loop",
    "loops",
    "gate",
    "gates",
    "encoder",
    "decoder",
    "attention",
    "cell",
    "network",
    "wire",
    "wires",
    "bbox",
    "boundary",
    "cluster",
}


def _tokenize_words(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 1]


_LEADING_FIGURE_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?:
        fig(?:ure)?\.?
    )
    \s*
    (?:
        [A-Za-z]?\d+[A-Za-z]?
        |
        [IVXLCDMivxlcdm]+
    )
    (?:
        \s*
        (?:
            \([A-Za-z0-9]{1,3}\)
            |
            \[[A-Za-z0-9]{1,3}\]
            |
            [A-Za-z]
        )
    )?
    \s*
    (?:
        [:.\-)\]]
        \s*
        |
        \s+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _strip_leading_figure_prefix(text: str) -> Tuple[str, bool, str]:
    if not text:
        return text, False, ""
    out = text
    removed: List[str] = []
    for _ in range(2):
        m = _LEADING_FIGURE_PREFIX_RE.match(out)
        if not m:
            break
        removed.append(out[m.start() : m.end()])
        out = out[m.end() :].lstrip()
    changed = out != text
    return out, changed, " | ".join(r.strip() for r in removed if r.strip())


def _init_strip_stats() -> Dict[str, Any]:
    return {
        "pairs": 0,
        "pred_changed": 0,
        "ref_changed": 0,
        "both_changed": 0,
        "examples": [],
    }


def _repeat_ngram_ratio(tokens: List[str], n: int) -> float:
    if n < 2 or len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    uniq = len(set(ngrams))
    total = len(ngrams)
    return (total - uniq) / max(1, total)


def _build_prefix_text(scale: str, context: str, token_mode: str) -> str:
    parts: List[str] = []
    if token_mode in ("task", "both"):
        parts.append(ttd.TASK_TOKENS.get(scale, ""))
    if token_mode in ("scale", "both"):
        parts.append(ttd.SCALE_TOKENS.get(scale, ""))
    if context:
        parts.append(context)
    parts = [p for p in parts if p]
    parts = list(dict.fromkeys(parts))
    return " ".join(parts).strip()


def _coverage_metrics(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float]:
    if not ref_tokens:
        return 0.0, 0.0, 0.0
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    inter = len(pred_set & ref_set)
    rec = inter / max(1, len(ref_set))
    prec = inter / max(1, len(pred_set))
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return rec, prec, f1


def _contains_struct_keywords(tokens: List[str]) -> bool:
    return any(t in STRUCT_KEYWORDS for t in tokens)


def _struct_ref_tokens(meta: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    if not isinstance(meta, dict):
        return tokens
    nodes = meta.get("struct_nodes") or []
    roles = meta.get("struct_roles") or []
    if nodes:
        tokens.extend(_tokenize_words(" ".join(nodes)))
    if roles:
        tokens.extend(_tokenize_words(" ".join(roles)))
    node_types = meta.get("struct_node_type_counts") or {}
    edge_types = meta.get("struct_edge_type_counts") or {}
    if isinstance(node_types, dict):
        for k in node_types.keys():
            tokens.extend(_tokenize_words(str(k)))
            if "arrow" in str(k).lower():
                tokens.append("arrow")
            if "box" in str(k).lower():
                tokens.append("box")
            if "circle" in str(k).lower():
                tokens.append("circle")
            if "ellipse" in str(k).lower():
                tokens.append("ellipse")
            if "rect" in str(k).lower():
                tokens.append("rectangle")
            if "diamond" in str(k).lower():
                tokens.append("diamond")
    if isinstance(edge_types, dict):
        for k in edge_types.keys():
            tokens.extend(_tokenize_words(str(k)))
            if "arrow" in str(k).lower():
                tokens.append("arrow")
            if "connect" in str(k).lower():
                tokens.append("connection")
    lin = meta.get("struct_linearized") or ""
    if lin:
        tokens.extend(_tokenize_words(str(lin)))
    return tokens


def _build_prefix_batch(
    tokenizer,
    scales: List[str],
    contexts: List[str],
    max_length: int,
    max_target_map: Dict[str, int],
    token_mode: str,
    decoder_arch: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    pad_id = tokenizer.pad_token_id or 0
    if decoder_arch == "t5":
        cls_id = pad_id
        max_body = max(1, max_length - 1)
    else:
        cls_id = tokenizer.cls_token_id or pad_id
        max_body = max(1, max_length - 1)

    input_ids_list: List[List[int]] = []
    prefix_lens: List[int] = []
    max_new_tokens_list: List[int] = []

    for scale, ctx in zip(scales, contexts):
        max_target = max_target_map.get(scale, max_body)
        max_target = max(1, min(int(max_target), max_body))
        if decoder_arch == "t5":
            input_ids = [cls_id]
            prefix_lens.append(1)
        else:
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
            prefix_lens.append(len(input_ids))
        input_ids_list.append(input_ids)
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


def _split_scale_indices(scales: List[str], caption_scales: set) -> Tuple[List[int], List[int]]:
    cap_idx: List[int] = []
    desc_idx: List[int] = []
    for i, sc in enumerate(scales):
        if sc == "desc":
            desc_idx.append(i)
        elif sc in caption_scales or not caption_scales:
            cap_idx.append(i)
        else:
            cap_idx.append(i)
    return cap_idx, desc_idx


@torch.no_grad()
def _greedy_decode(
    decoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    enc_tokens: torch.Tensor,
    enc_mask: torch.Tensor,
    max_new_tokens_list: List[int],
    eos_id: int,
    pad_id: int,
    max_length: int,
    min_new_tokens: int | List[int],
    forbidden_ids: List[int],
    decoder_arch: str,
    copy_ids: List[List[int]] | None = None,
    copy_bias: float = 0.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    device = input_ids.device
    bsz = input_ids.size(0)
    max_new = max(max_new_tokens_list) if max_new_tokens_list else 0
    if max_new <= 0:
        return input_ids
    max_new_t = torch.tensor(max_new_tokens_list, device=device)
    if isinstance(min_new_tokens, (list, tuple)):
        min_vals = [max(0, int(v)) for v in min_new_tokens]
        if len(min_vals) != bsz:
            min_vals = [min_vals[0]] * bsz if min_vals else [0] * bsz
        min_new_t = torch.tensor(min_vals, device=device, dtype=torch.long)
        min_new_t = torch.minimum(max_new_t, min_new_t)
    else:
        if min_new_tokens < 0:
            min_new_tokens = 0
        if min_new_tokens > 0:
            min_new_t = torch.minimum(max_new_t, torch.full_like(max_new_t, min_new_tokens))
        else:
            min_new_t = None
    always_forbidden = [i for i in forbidden_ids if i is not None]

    generated = input_ids
    attn = attention_mask
    done = torch.zeros(bsz, dtype=torch.bool, device=device)

    for step in range(max_new):
        if done.all():
            break
        if generated.size(1) >= max_length:
            break
        if decoder_arch == "t5":
            out = decoder(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_tokens),
                attention_mask=enc_mask,
                decoder_input_ids=generated,
                decoder_attention_mask=attn,
                use_cache=False,
            )
        else:
            out = decoder(
                input_ids=generated,
                attention_mask=attn,
                encoder_hidden_states=enc_tokens,
                encoder_attention_mask=enc_mask,
                use_cache=False,
            )
        logits = out.logits[:, -1, :]
        if copy_bias > 0 and copy_ids is not None:
            logits = _apply_copy_bias(logits.unsqueeze(1), copy_ids, copy_bias).squeeze(1)
        if always_forbidden or (min_new_t is not None):
            logits = logits.clone()
            if always_forbidden:
                logits[:, always_forbidden] = -1.0e9
            if min_new_t is not None:
                force_mask = min_new_t > step
                if force_mask.any():
                    idx = force_mask.nonzero(as_tuple=True)[0]
                    logits[idx, eos_id] = -1.0e9
        if no_repeat_ngram_size and no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
            logits = logits.clone()
            gen_list = generated.tolist()
            for i, seq in enumerate(gen_list):
                if len(seq) < no_repeat_ngram_size:
                    continue
                n = no_repeat_ngram_size
                prefix = tuple(seq[-(n - 1) :]) if n > 1 else tuple()
                banned = set()
                if n == 1:
                    banned.update(seq)
                else:
                    for j in range(len(seq) - n + 1):
                        ng = tuple(seq[j : j + n])
                        if ng[:-1] == prefix:
                            banned.add(ng[-1])
                if banned:
                    logits[i, list(banned)] = -1.0e9
        next_token = logits.argmax(dim=-1)
        over = (step + 1) >= max_new_t
        step_done = done | over | (next_token == eos_id)
        next_token = torch.where(step_done, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        attn = torch.cat([attn, (~step_done).long().unsqueeze(-1)], dim=-1)
        done = step_done
    return generated


def _decode_batch(
    tokenizer,
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


def _load_metrics(bert_model: str, rescale_with_baseline: bool):
    try:
        import evaluate
        from pycocoevalcap.cider.cider import Cider

        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        bert = evaluate.load("bertscore")

        def compute(preds: List[str], refs: List[str]) -> Dict[str, float]:
            if not preds:
                return {}
            refs_list = [[r] for r in refs]
            out = {}
            out["bleu4"] = float(bleu.compute(predictions=preds, references=refs_list)["bleu"])
            out["rougeL"] = float(rouge.compute(predictions=preds, references=refs)["rougeL"])
            gts = {i: [r] for i, r in enumerate(refs)}
            res = {i: [p] for i, p in enumerate(preds)}
            out["cider"], _ = Cider().compute_score(gts, res)
            try:
                bs = bert.compute(
                    predictions=preds,
                    references=refs,
                    lang="en",
                    model_type=bert_model,
                    rescale_with_baseline=rescale_with_baseline,
                )
                out["bertscore_f1"] = float(sum(bs["f1"]) / max(1, len(bs["f1"])))
            except Exception as exc:
                print(f"[warn] bertscore failed: {exc}")
            return out

        return compute
    except Exception as exc:
        print(f"[warn] evaluate unavailable; fallback to pycocoevalcap metrics: {exc}")
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        try:
            from bert_score import score as bert_score

            has_bert = True
        except Exception:
            has_bert = False

        def compute(preds: List[str], refs: List[str]) -> Dict[str, float]:
            if not preds:
                return {}
            gts = {i: [r] for i, r in enumerate(refs)}
            res = {i: [p] for i, p in enumerate(preds)}
            out = {}
            bleu_scores, _ = Bleu(4).compute_score(gts, res)
            out["bleu4"] = float(bleu_scores[3])
            out["rougeL"], _ = Rouge().compute_score(gts, res)
            out["cider"], _ = Cider().compute_score(gts, res)
            if has_bert:
                try:
                    _, _, f1 = bert_score(
                        preds,
                        refs,
                        lang="en",
                        model_type=bert_model,
                        rescale_with_baseline=rescale_with_baseline,
                    )
                    out["bertscore_f1"] = float(f1.mean().item())
                except Exception as exc2:
                    print(f"[warn] bertscore failed: {exc2}")
            return out

        return compute


def _sanitize_token(tok: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", tok)[:32]


def _reshape_attn_grid(attn_vec: torch.Tensor) -> torch.Tensor | None:
    if attn_vec.numel() <= 1:
        return None
    n = attn_vec.numel()
    g = int(math.sqrt(n))
    if g * g != n:
        return None
    return attn_vec.view(g, g)


def _dump_cross_attn_images(
    decoder,
    tokenizer,
    generated: torch.Tensor,
    prefix_lens: List[int],
    enc_tokens: torch.Tensor,
    enc_mask: torch.Tensor,
    images: List,
    metas: List[Dict],
    scales: List[str],
    vision_len: int,
    dump_dir: Path,
    token_regex: re.Pattern,
    dump_state: Dict[str, int],
    pad_id: int,
    max_items: int,
) -> None:
    if dump_state.get("count", 0) >= max_items:
        return
    if not hasattr(decoder, "forward"):
        return
    if not hasattr(decoder, "config") or not getattr(decoder.config, "is_decoder", False):
        return
    try:
        from PIL import Image
    except Exception:
        return
    with torch.no_grad():
        attn_mask = (generated != pad_id).long()
        out = decoder(
            input_ids=generated,
            attention_mask=attn_mask,
            encoder_hidden_states=enc_tokens,
            encoder_attention_mask=enc_mask,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        cross = out.cross_attentions
        if not cross:
            return
        cross_t = torch.stack(cross, dim=0).mean(dim=0).mean(dim=1)  # [B, T, S]
    dump_dir.mkdir(parents=True, exist_ok=True)
    bsz = generated.size(0)
    for i in range(bsz):
        if dump_state.get("count", 0) >= max_items:
            break
        toks = tokenizer.convert_ids_to_tokens(generated[i].tolist())
        start = prefix_lens[i] if i < len(prefix_lens) else 0
        selected = []
        for pos in range(start, len(toks)):
            tok = toks[pos]
            if tok in ("[PAD]", "[CLS]", "[SEP]"):
                continue
            tok_norm = tok.replace("##", "").lower()
            if token_regex.search(tok_norm):
                selected.append((pos, tok_norm))
        if not selected:
            continue
        img = images[i].convert("RGBA") if hasattr(images[i], "convert") else None
        if img is None:
            continue
        base = os.path.basename(metas[i].get("image_path", f"item{i}")) if isinstance(metas[i], dict) else f"item{i}"
        for pos, tok_norm in selected:
            attn_vec = cross_t[i, pos]
            vis_attn = attn_vec[:vision_len]
            grid = None
            if vision_len > 1:
                grid = _reshape_attn_grid(vis_attn[1:])
            if grid is None:
                grid = _reshape_attn_grid(vis_attn)
            if grid is None:
                continue
            heat = grid - grid.min()
            if heat.max() > 0:
                heat = heat / heat.max()
            heat = (heat * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            heat_img = Image.fromarray(heat, mode="L").resize(img.size, resample=Image.BILINEAR)
            overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
            overlay.putalpha(heat_img)
            out = Image.alpha_composite(img, overlay)
            name = f"{Path(base).stem}_scale-{scales[i]}_tok-{_sanitize_token(tok_norm)}_pos{pos}.png"
            out.save(dump_dir / name)
            dump_state["count"] = dump_state.get("count", 0) + 1
            if dump_state["count"] >= max_items:
                break


def _init_custom_metrics() -> Dict[str, float]:
    return {
        "ocr_recall": 0.0,
        "ocr_precision": 0.0,
        "ocr_f1": 0.0,
        "ocr_count": 0,
        "ctx_recall": 0.0,
        "ctx_precision": 0.0,
        "ctx_f1": 0.0,
        "ctx_count": 0,
        "node_recall": 0.0,
        "node_precision": 0.0,
        "node_f1": 0.0,
        "node_count": 0,
        "role_recall": 0.0,
        "role_precision": 0.0,
        "role_f1": 0.0,
        "role_count": 0,
        "struct_recall": 0.0,
        "struct_precision": 0.0,
        "struct_f1": 0.0,
        "struct_count": 0,
        "novel_struct_ratio": 0.0,
        "novel_struct_count": 0,
        "novel_struct_recall": 0.0,
        "novel_struct_precision": 0.0,
        "novel_struct_f1": 0.0,
        "novel_struct_cov_count": 0,
        "gt_struct_count": 0,
        "gt_struct_total": 0,
        "len_ratio": 0.0,
        "len_count": 0,
        "rep3_sum": 0.0,
        "rep3_count": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_format", type=str, default="scicap", choices=["scicap", "jsonl_desc", "jsonl_multi"])
    ap.add_argument("--test_json", type=str, required=True)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--visual_ckpt", type=str, required=True)
    ap.add_argument("--decoder_ckpt", type=str, required=True)
    ap.add_argument("--dual_decoder", action="store_true")
    ap.add_argument("--decoder_desc_ckpt", type=str, default="")
    ap.add_argument("--decoder_arch", type=str, default="bert", choices=["bert", "t5"])
    ap.add_argument("--t5_model_name", type=str, default="t5-base")
    ap.add_argument(
        "--allow_unsafe_torch_load",
        action="store_true",
        help="Bypass the transformers torch.load safety check (CVE-2025-32434).",
    )
    ap.add_argument("--caption_scales", type=str, default="short,long")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--image_size", type=int, default=0)
    ap.add_argument("--max_target_short", type=int, default=64)
    ap.add_argument("--max_target_long", type=int, default=256)
    ap.add_argument("--max_target_desc", type=int, default=384)
    ap.add_argument("--min_new_tokens", type=int, default=8)
    ap.add_argument("--min_new_short", type=int, default=0)
    ap.add_argument("--min_new_long", type=int, default=0)
    ap.add_argument("--min_new_desc", type=int, default=0)
    ap.add_argument("--min_len_short", type=int, default=20)
    ap.add_argument("--min_len_long", type=int, default=40)
    ap.add_argument("--min_len_desc", type=int, default=40)
    ap.add_argument("--use_desc", action="store_true")
    ap.add_argument("--context_mode", type=str, default="para_mention", choices=["none", "paragraph", "para_mention", "para_mention_ocr"])
    ap.add_argument("--token_mode", type=str, default="task", choices=["task", "scale", "both"])
    ap.add_argument("--token_schema", type=str, default="legacy", choices=["legacy", "simple"])
    ap.add_argument("--desc_key", type=str, default="description")
    ap.add_argument("--image_key", type=str, default="image_path")
    ap.add_argument("--struct_jsonl", type=str, default="")
    ap.add_argument("--max_struct_nodes", type=int, default=64)
    ap.add_argument("--max_struct_roles", type=int, default=16)
    ap.add_argument("--text_encoder", type=str, default="none", choices=["none", "scibert"])
    ap.add_argument("--use_multi_source_attn", action="store_true")
    ap.add_argument("--use_plan_tokens", action="store_true")
    ap.add_argument("--plan_in_context", action="store_true")
    ap.add_argument("--plan_include_counts", action="store_true")
    ap.add_argument("--plan_include_types", action="store_true")
    ap.add_argument("--plan_max_types", type=int, default=6)
    ap.add_argument("--plan_include_linearized", action="store_true")
    ap.add_argument("--plan_max_linearized_chars", type=int, default=256)
    ap.add_argument("--text_max_length", type=int, default=256)
    ap.add_argument("--text_token_scale", type=float, default=1.0)
    ap.add_argument("--ocr_jsonl", type=str, default="")
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
    ap.add_argument("--copy_logit_bias", type=float, default=1.0)
    ap.add_argument("--copy_max_tokens", type=int, default=128)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--bert_model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--bertscore_no_rescale", action="store_true")
    ap.add_argument("--dump_attn_dir", type=str, default="")
    ap.add_argument("--dump_attn_max_items", type=int, default=16)
    ap.add_argument(
        "--dump_attn_token_regex",
        type=str,
        default="node|arrow|edge|flow|flowchart|graph|diagram|state|transition|box|circle|layer|module",
    )
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pred_only", action="store_true")
    ap.add_argument(
        "--strip_leading_figure_prefix",
        action="store_true",
        help="strip leading 'Figure/Fig <id>' prefix from both pred/ref before scoring",
    )
    ap.add_argument(
        "--strip_prefix_max_examples",
        type=int,
        default=8,
        help="max changed pred/ref examples per split to log in output json",
    )
    ap.add_argument("--out_json", type=str, default="logs/text_decoder/eval_text_decoder_metrics.json")
    ap.add_argument("--pred_jsonl", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_token_schema(args.token_schema)
    # Ensure local access to the updated token maps.
    # TASK_TOKENS/SCALE_TOKENS are referenced via the train_text_decoder module.
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
    attn_dir = Path(args.dump_attn_dir) if args.dump_attn_dir else None
    attn_regex = re.compile(args.dump_attn_token_regex, re.IGNORECASE)
    dump_state = {"count": 0}
    max_target_map = {
        "short": args.max_target_short,
        "long": args.max_target_long,
        "desc": args.max_target_desc,
    }
    min_new_map = {
        "short": args.min_new_short if args.min_new_short > 0 else args.min_new_tokens,
        "long": args.min_new_long if args.min_new_long > 0 else args.min_new_tokens,
        "desc": args.min_new_desc if args.min_new_desc > 0 else args.min_new_tokens,
    }

    tokenizer = build_tokenizer(args.token_mode, decoder_arch=args.decoder_arch)
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
    model_max = getattr(tokenizer, "model_max_length", args.max_length)
    if isinstance(model_max, int) and 0 < model_max < 1_000_000:
        max_length = min(args.max_length, model_max)
    else:
        max_length = args.max_length
    ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    if args.dual_decoder:
        if "decoder_caption" in ckpt:
            decoder.load_state_dict(ckpt["decoder_caption"], strict=True)
        elif "decoder" in ckpt:
            decoder.load_state_dict(ckpt["decoder"], strict=True)
        else:
            decoder.load_state_dict(ckpt, strict=False)
            print("[warn] decoder_ckpt missing decoder_caption; loaded with strict=False.")
        desc_state = ckpt
        if args.decoder_desc_ckpt:
            desc_state = torch.load(args.decoder_desc_ckpt, map_location="cpu")
        if decoder_desc is not None:
            if "decoder_desc" in desc_state:
                decoder_desc.load_state_dict(desc_state["decoder_desc"], strict=True)
            elif "decoder" in desc_state:
                decoder_desc.load_state_dict(desc_state["decoder"], strict=True)
            elif "decoder_caption" in desc_state:
                decoder_desc.load_state_dict(desc_state["decoder_caption"], strict=True)
                print("[warn] decoder_desc_ckpt missing decoder_desc; using decoder_caption weights.")
            else:
                decoder_desc.load_state_dict(desc_state, strict=False)
                print("[warn] decoder_desc_ckpt missing decoder_desc; loaded with strict=False.")
    else:
        if "decoder" in ckpt:
            decoder.load_state_dict(ckpt["decoder"], strict=True)
        elif "decoder_caption" in ckpt:
            decoder.load_state_dict(ckpt["decoder_caption"], strict=True)
        else:
            decoder.load_state_dict(ckpt, strict=False)
    decoder.eval()
    if decoder_desc is not None:
        decoder_desc.eval()

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

    vision = CLIPVisionTower(output_attentions=False).to(device)
    vckpt = torch.load(args.visual_ckpt, map_location="cpu")
    vision.load_state_dict(vckpt.get("model", {}), strict=False)
    vision.eval()

    enc_proj = None
    if isinstance(ckpt, dict) and ckpt.get("enc_proj") is not None:
        weight = ckpt["enc_proj"].get("weight")
        bias = ckpt["enc_proj"].get("bias")
        if weight is not None:
            out_dim, in_dim = weight.shape
            enc_proj = nn.Linear(in_dim, out_dim).to(device)
            enc_proj.load_state_dict(ckpt["enc_proj"], strict=True)
            enc_proj.eval()

    if args.dataset_format == "scicap":
        dataset = SciCapMultiScaleDataset(
            args.test_json,
            images_root=args.images_root,
            sample_mode="expand",
            min_len_short=args.min_len_short,
            min_len_long=args.min_len_long,
            min_len_desc=args.min_len_desc,
            use_desc=args.use_desc,
            context_mode=args.context_mode,
            return_meta=True,
            struct_jsonl=args.struct_jsonl or None,
            max_struct_nodes=args.max_struct_nodes,
            max_struct_roles=args.max_struct_roles,
        )
    elif args.dataset_format == "jsonl_multi":
        dataset = JsonlMultiScaleDataset(
            args.test_json,
            images_root=args.images_root,
            sample_mode="expand",
            min_len_short=args.min_len_short,
            min_len_long=args.min_len_long,
            min_len_desc=args.min_len_desc,
            return_meta=True,
            image_key=args.image_key,
            desc_key=args.desc_key,
            struct_jsonl=args.struct_jsonl or None,
            max_struct_nodes=args.max_struct_nodes,
            max_struct_roles=args.max_struct_roles,
        )
    else:
        dataset = JsonlDescDataset(
            args.test_json,
            images_root=args.images_root,
            min_len_desc=args.min_len_desc,
            desc_key=args.desc_key,
            image_key=args.image_key,
            return_meta=True,
            struct_jsonl=args.struct_jsonl or None,
            max_struct_nodes=args.max_struct_nodes,
            max_struct_roles=args.max_struct_roles,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    ocr_map = {}
    def _maybe_json_local(val: str):
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

    def _clean_field(val: str) -> str:
        return _flatten_text(_maybe_json_local(val))

    if args.ocr_jsonl:
        with open(args.ocr_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img = obj.get("image_path")
                if not img:
                    continue
                payload = {
                    "ocr": _clean_field(obj.get("ocr") or ""),
                    "paragraph": _clean_field(obj.get("paragraph") or ""),
                    "mention": _clean_field(obj.get("mention") or ""),
                }
                ocr_map[img] = payload
                ocr_map[os.path.basename(img)] = payload

    if not args.pred_jsonl:
        if args.out_json:
            out_path = Path(args.out_json)
            name = out_path.name
            if name.startswith("eval_"):
                name = "preds_" + name[len("eval_"):]
            elif name.endswith(".json"):
                name = name[:-5] + ".jsonl"
            else:
                name = name + ".jsonl"
            if not name.endswith(".jsonl"):
                name = name + ".jsonl"
            args.pred_jsonl = str(out_path.parent / name)
        else:
            ts = time.strftime("%m%d_%H%M%S")
            base = Path(args.test_json).stem if args.test_json else "preds"
            args.pred_jsonl = str(Path("logs/text_decoder") / f"preds_{base}_{ts}.jsonl")
        print(f"[info] --pred_jsonl not set; defaulting to {args.pred_jsonl}")
    compute_metrics = None if args.pred_only else _load_metrics(args.bert_model, not args.bertscore_no_rescale)
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id or 0
    pad_id = tokenizer.pad_token_id or 0
    # Always forbid PAD/CLS. EOS/SEP is allowed only after min_new_tokens.
    forbidden_ids = [pad_id, tokenizer.cls_token_id]
    caption_scales = [s.strip() for s in args.caption_scales.split(",") if s.strip()]
    if "desc" in caption_scales:
        caption_scales = [s for s in caption_scales if s != "desc"]
        print("[warn] removing desc from caption_scales; desc is routed to description decoder.")
    caption_scales_set = set(caption_scales)

    preds_by_scale: Dict[str, List[str]] = {}
    refs_by_scale: Dict[str, List[str]] = {}
    custom_by_scale: Dict[str, Dict[str, float]] = {}
    strip_stats_by_scale: Dict[str, Dict[str, Any]] = {}
    counts_by_scale: Dict[str, int] = {"short": 0, "long": 0, "desc": 0, "all": 0}
    if not args.pred_only:
        preds_by_scale = {"short": [], "long": [], "desc": []}
        refs_by_scale = {"short": [], "long": [], "desc": []}
        custom_by_scale = {
            "short": _init_custom_metrics(),
            "long": _init_custom_metrics(),
            "desc": _init_custom_metrics(),
            "all": _init_custom_metrics(),
        }
        strip_stats_by_scale = {
            "short": _init_strip_stats(),
            "long": _init_strip_stats(),
            "desc": _init_strip_stats(),
            "all": _init_strip_stats(),
        }

    pred_out = Path(args.pred_jsonl) if args.pred_jsonl else None
    if pred_out:
        pred_out.parent.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        if len(batch) == 5:
            images, texts, scales, contexts, metas = batch
        else:
            images, texts, scales, contexts = batch
            metas = [{} for _ in images]
        metas = [m or {} for m in metas]
        if ocr_map:
            updated_contexts = []
            for meta, ctx in zip(metas, contexts):
                img_path = meta.get("image_path") or ""
                ocr_info = ocr_map.get(img_path)
                if ocr_info:
                    ctx_parts = [
                        str(ocr_info.get("paragraph") or ""),
                        str(ocr_info.get("mention") or ""),
                        str(ocr_info.get("ocr") or ""),
                    ]
                    ctx = " \n".join([c for c in ctx_parts if c]).strip()
                    for k in ("ocr", "paragraph", "mention"):
                        if k not in meta or not meta.get(k):
                            meta[k] = ocr_info.get(k) or ""
                updated_contexts.append(ctx)
            contexts = updated_contexts
        if args.plan_in_context:
            updated_contexts = []
            for meta, ctx in zip(metas, contexts):
                plan = _build_plan_text(meta or {}, **plan_kwargs)
                if plan:
                    ctx = f"{plan}\n{ctx}".strip() if ctx else plan
                updated_contexts.append(ctx)
            contexts = updated_contexts
        batch = vision.preprocess(images, image_size=(args.image_size if args.image_size > 0 else None))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            enc = vision(batch)
        enc_tokens = enc["tokens"]
        vision_len = enc_tokens.size(1)
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
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                text_out = text_encoder(input_ids=input_ids, attention_mask=attn)
                text_tokens = text_out.last_hidden_state
            if args.text_token_scale != 1.0:
                text_tokens = text_tokens * args.text_token_scale
            enc_tokens = torch.cat([enc_tokens, text_tokens], dim=1)
            enc_mask = torch.cat([enc_mask, attn], dim=1)
        if enc_proj is not None:
            enc_tokens = enc_proj(enc_tokens)

        copy_ids = None
        if args.use_copy_head:
            copy_ids = _build_copy_token_ids(
                tokenizer, contexts, metas, args.copy_max_tokens, sources=args.copy_sources
            )

        if decoder_desc is not None:
            batch_preds = [""] * len(scales)
            cap_idx, desc_idx = _split_scale_indices(scales, caption_scales_set)
            if cap_idx:
                cap_scales = [scales[i] for i in cap_idx]
                cap_ctx = [contexts[i] for i in cap_idx]
                cap_min_new = [min_new_map.get(sc, args.min_new_tokens) for sc in cap_scales]
                input_ids, attn, prefix_lens, max_new_tokens_list = _build_prefix_batch(
                    tokenizer,
                    cap_scales,
                    cap_ctx,
                    max_length,
                    max_target_map,
                    args.token_mode,
                    args.decoder_arch,
                )
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                idx_t = torch.tensor(cap_idx, device=device, dtype=torch.long)
                enc_tokens_sub = enc_tokens.index_select(0, idx_t)
                enc_mask_sub = enc_mask.index_select(0, idx_t)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    generated = _greedy_decode(
                        decoder,
                        input_ids,
                        attn,
                        enc_tokens_sub,
                        enc_mask_sub,
                        max_new_tokens_list,
                        eos_id,
                        pad_id,
                        max_length,
                        cap_min_new,
                        forbidden_ids,
                        args.decoder_arch,
                        copy_ids=[copy_ids[i] for i in cap_idx] if copy_ids is not None else None,
                        copy_bias=args.copy_logit_bias if args.use_copy_head else 0.0,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                    )
                preds = _decode_batch(tokenizer, generated, prefix_lens, max_new_tokens_list, eos_id)
                for i, pred in zip(cap_idx, preds):
                    batch_preds[i] = pred
                if attn_dir and args.decoder_arch == "bert":
                    cap_images = [images[i] for i in cap_idx]
                    cap_metas = [metas[i] for i in cap_idx]
                    cap_scales = [scales[i] for i in cap_idx]
                    _dump_cross_attn_images(
                        decoder,
                        tokenizer,
                        generated,
                        prefix_lens,
                        enc_tokens_sub,
                        enc_mask_sub,
                        cap_images,
                        cap_metas,
                        cap_scales,
                        vision_len,
                        attn_dir,
                        attn_regex,
                        dump_state,
                        pad_id,
                        args.dump_attn_max_items,
                    )
            if desc_idx:
                desc_scales = [scales[i] for i in desc_idx]
                desc_ctx = [contexts[i] for i in desc_idx]
                desc_min_new = [min_new_map.get(sc, args.min_new_tokens) for sc in desc_scales]
                input_ids, attn, prefix_lens, max_new_tokens_list = _build_prefix_batch(
                    tokenizer,
                    desc_scales,
                    desc_ctx,
                    max_length,
                    max_target_map,
                    args.token_mode,
                    args.decoder_arch,
                )
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                idx_t = torch.tensor(desc_idx, device=device, dtype=torch.long)
                enc_tokens_sub = enc_tokens.index_select(0, idx_t)
                enc_mask_sub = enc_mask.index_select(0, idx_t)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    generated = _greedy_decode(
                        decoder_desc,
                        input_ids,
                        attn,
                        enc_tokens_sub,
                        enc_mask_sub,
                        max_new_tokens_list,
                        eos_id,
                        pad_id,
                        max_length,
                        desc_min_new,
                        forbidden_ids,
                        args.decoder_arch,
                        copy_ids=[copy_ids[i] for i in desc_idx] if copy_ids is not None else None,
                        copy_bias=args.copy_logit_bias if args.use_copy_head else 0.0,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                    )
                preds = _decode_batch(tokenizer, generated, prefix_lens, max_new_tokens_list, eos_id)
                for i, pred in zip(desc_idx, preds):
                    batch_preds[i] = pred
                if attn_dir and args.decoder_arch == "bert":
                    desc_images = [images[i] for i in desc_idx]
                    desc_metas = [metas[i] for i in desc_idx]
                    desc_scales = [scales[i] for i in desc_idx]
                    _dump_cross_attn_images(
                        decoder_desc,
                        tokenizer,
                        generated,
                        prefix_lens,
                        enc_tokens_sub,
                        enc_mask_sub,
                        desc_images,
                        desc_metas,
                        desc_scales,
                        vision_len,
                        attn_dir,
                        attn_regex,
                        dump_state,
                        pad_id,
                        args.dump_attn_max_items,
                    )
        else:
            all_min_new = [min_new_map.get(sc, args.min_new_tokens) for sc in scales]
            input_ids, attn, prefix_lens, max_new_tokens_list = _build_prefix_batch(
                tokenizer,
                scales,
                contexts,
                max_length,
                max_target_map,
                args.token_mode,
                args.decoder_arch,
            )
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                generated = _greedy_decode(
                    decoder,
                    input_ids,
                    attn,
                    enc_tokens,
                    enc_mask,
                    max_new_tokens_list,
                    eos_id,
                    pad_id,
                    max_length,
                    all_min_new,
                    forbidden_ids,
                    args.decoder_arch,
                    copy_ids=copy_ids,
                    copy_bias=args.copy_logit_bias if args.use_copy_head else 0.0,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
            batch_preds = _decode_batch(tokenizer, generated, prefix_lens, max_new_tokens_list, eos_id)
            if attn_dir and args.decoder_arch == "bert":
                _dump_cross_attn_images(
                    decoder,
                    tokenizer,
                    generated,
                    prefix_lens,
                    enc_tokens,
                    enc_mask,
                    images,
                    metas,
                    scales,
                    vision_len,
                    attn_dir,
                    attn_regex,
                    dump_state,
                    pad_id,
                    args.dump_attn_max_items,
                )
        for pred, ref, scale, ctx, meta in zip(batch_preds, texts, scales, contexts, metas):
            if args.pred_only:
                if scale in counts_by_scale:
                    counts_by_scale[scale] += 1
                counts_by_scale["all"] += 1
                if pred_out:
                    with pred_out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"scale": scale, "pred": pred, "ref": ref}, ensure_ascii=False) + "\n")
                continue

            pred_eval = pred
            ref_eval = ref
            pred_strip_prefix = ""
            ref_strip_prefix = ""
            pred_changed = False
            ref_changed = False
            if args.strip_leading_figure_prefix:
                pred_eval, pred_changed, pred_strip_prefix = _strip_leading_figure_prefix(pred)
                ref_eval, ref_changed, ref_strip_prefix = _strip_leading_figure_prefix(ref)
                if scale not in strip_stats_by_scale:
                    strip_stats_by_scale[scale] = _init_strip_stats()
                for sk in (scale, "all"):
                    stat = strip_stats_by_scale.setdefault(sk, _init_strip_stats())
                    stat["pairs"] += 1
                    stat["pred_changed"] += int(pred_changed)
                    stat["ref_changed"] += int(ref_changed)
                    stat["both_changed"] += int(pred_changed and ref_changed)
                    if (
                        (pred_changed or ref_changed)
                        and len(stat["examples"]) < max(0, args.strip_prefix_max_examples)
                    ):
                        stat["examples"].append(
                            {
                                "scale": scale,
                                "pred_before": pred,
                                "pred_after": pred_eval,
                                "pred_prefix": pred_strip_prefix,
                                "ref_before": ref,
                                "ref_after": ref_eval,
                                "ref_prefix": ref_strip_prefix,
                            }
                        )

            if scale not in preds_by_scale:
                preds_by_scale[scale] = []
                refs_by_scale[scale] = []
            preds_by_scale[scale].append(pred_eval)
            refs_by_scale[scale].append(ref_eval)
            pred_tokens = _tokenize_words(pred_eval)
            ref_tokens = _tokenize_words(ref_eval)
            if ref_tokens:
                custom_by_scale[scale]["len_ratio"] += len(pred_tokens) / max(1, len(ref_tokens))
                custom_by_scale[scale]["len_count"] += 1
                custom_by_scale["all"]["len_ratio"] += len(pred_tokens) / max(1, len(ref_tokens))
                custom_by_scale["all"]["len_count"] += 1
            rep3 = _repeat_ngram_ratio(pred_tokens, 3)
            custom_by_scale[scale]["rep3_sum"] += rep3
            custom_by_scale[scale]["rep3_count"] += 1
            custom_by_scale["all"]["rep3_sum"] += rep3
            custom_by_scale["all"]["rep3_count"] += 1

            ocr_text = ""
            para_text = ""
            mention_text = ""
            if isinstance(meta, dict):
                ocr_text = meta.get("ocr") or ""
                para_text = meta.get("paragraph") or ""
                mention_text = meta.get("mention") or ""
            ctx_text = f"{para_text} {mention_text}".strip()
            if not ctx_text:
                ctx_text = ctx or ""

            ocr_tokens = _tokenize_words(ocr_text)
            if ocr_tokens:
                rec, prec, f1 = _coverage_metrics(pred_tokens, ocr_tokens)
                custom_by_scale[scale]["ocr_recall"] += rec
                custom_by_scale[scale]["ocr_precision"] += prec
                custom_by_scale[scale]["ocr_f1"] += f1
                custom_by_scale[scale]["ocr_count"] += 1
                custom_by_scale["all"]["ocr_recall"] += rec
                custom_by_scale["all"]["ocr_precision"] += prec
                custom_by_scale["all"]["ocr_f1"] += f1
                custom_by_scale["all"]["ocr_count"] += 1

            ctx_tokens = _tokenize_words(ctx_text)
            if ctx_tokens:
                rec, prec, f1 = _coverage_metrics(pred_tokens, ctx_tokens)
                custom_by_scale[scale]["ctx_recall"] += rec
                custom_by_scale[scale]["ctx_precision"] += prec
                custom_by_scale[scale]["ctx_f1"] += f1
                custom_by_scale[scale]["ctx_count"] += 1
                custom_by_scale["all"]["ctx_recall"] += rec
                custom_by_scale["all"]["ctx_precision"] += prec
                custom_by_scale["all"]["ctx_f1"] += f1
                custom_by_scale["all"]["ctx_count"] += 1

            node_tokens: List[str] = []
            role_tokens: List[str] = []
            if isinstance(meta, dict):
                node_texts = meta.get("struct_nodes") or []
                role_texts = meta.get("struct_roles") or []
                if node_texts:
                    node_tokens = _tokenize_words(" ".join(node_texts))
                if role_texts:
                    role_tokens = _tokenize_words(" ".join(role_texts))
            if node_tokens:
                rec, prec, f1 = _coverage_metrics(pred_tokens, node_tokens)
                custom_by_scale[scale]["node_recall"] += rec
                custom_by_scale[scale]["node_precision"] += prec
                custom_by_scale[scale]["node_f1"] += f1
                custom_by_scale[scale]["node_count"] += 1
                custom_by_scale["all"]["node_recall"] += rec
                custom_by_scale["all"]["node_precision"] += prec
                custom_by_scale["all"]["node_f1"] += f1
                custom_by_scale["all"]["node_count"] += 1
            if role_tokens:
                rec, prec, f1 = _coverage_metrics(pred_tokens, role_tokens)
                custom_by_scale[scale]["role_recall"] += rec
                custom_by_scale[scale]["role_precision"] += prec
                custom_by_scale[scale]["role_f1"] += f1
                custom_by_scale[scale]["role_count"] += 1
                custom_by_scale["all"]["role_recall"] += rec
                custom_by_scale["all"]["role_precision"] += prec
                custom_by_scale["all"]["role_f1"] += f1
                custom_by_scale["all"]["role_count"] += 1
            struct_ref = _struct_ref_tokens(meta) if isinstance(meta, dict) else []
            if struct_ref:
                rec, prec, f1 = _coverage_metrics(pred_tokens, struct_ref)
                custom_by_scale[scale]["struct_recall"] += rec
                custom_by_scale[scale]["struct_precision"] += prec
                custom_by_scale[scale]["struct_f1"] += f1
                custom_by_scale[scale]["struct_count"] += 1
                custom_by_scale["all"]["struct_recall"] += rec
                custom_by_scale["all"]["struct_precision"] += prec
                custom_by_scale["all"]["struct_f1"] += f1
                custom_by_scale["all"]["struct_count"] += 1

            gt_has_struct = _contains_struct_keywords(ref_tokens)
            custom_by_scale[scale]["gt_struct_total"] += 1
            custom_by_scale["all"]["gt_struct_total"] += 1
            if gt_has_struct:
                custom_by_scale[scale]["gt_struct_count"] += 1
                custom_by_scale["all"]["gt_struct_count"] += 1
            else:
                struct_kw_count = sum(1 for t in pred_tokens if t in STRUCT_KEYWORDS)
                ratio = struct_kw_count / max(1, len(pred_tokens))
                custom_by_scale[scale]["novel_struct_ratio"] += ratio
                custom_by_scale[scale]["novel_struct_count"] += 1
                custom_by_scale["all"]["novel_struct_ratio"] += ratio
                custom_by_scale["all"]["novel_struct_count"] += 1
                if struct_ref:
                    rec, prec, f1 = _coverage_metrics(pred_tokens, struct_ref)
                    custom_by_scale[scale]["novel_struct_recall"] += rec
                    custom_by_scale[scale]["novel_struct_precision"] += prec
                    custom_by_scale[scale]["novel_struct_f1"] += f1
                    custom_by_scale[scale]["novel_struct_cov_count"] += 1
                    custom_by_scale["all"]["novel_struct_recall"] += rec
                    custom_by_scale["all"]["novel_struct_precision"] += prec
                    custom_by_scale["all"]["novel_struct_f1"] += f1
                    custom_by_scale["all"]["novel_struct_cov_count"] += 1
            if pred_out:
                with pred_out.open("a", encoding="utf-8") as f:
                    obj = {"scale": scale, "pred": pred, "ref": ref}
                    if args.strip_leading_figure_prefix:
                        obj["pred_norm"] = pred_eval
                        obj["ref_norm"] = ref_eval
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if not args.pred_only:
        results: Dict[str, Any] = {}
        all_preds: List[str] = []
        all_refs: List[str] = []
        for scale in ["short", "long", "desc"]:
            preds = preds_by_scale.get(scale, [])
            refs = refs_by_scale.get(scale, [])
            if preds:
                results[scale] = compute_metrics(preds, refs)
                all_preds.extend(preds)
                all_refs.extend(refs)
        if all_preds:
            results["all"] = compute_metrics(all_preds, all_refs)

        for scale in ["short", "long", "desc", "all"]:
            if scale not in results:
                results[scale] = {}
            custom = custom_by_scale.get(scale, {})
            if custom.get("ocr_count", 0) > 0:
                results[scale]["ocr_recall"] = custom["ocr_recall"] / custom["ocr_count"]
                results[scale]["ocr_precision"] = custom["ocr_precision"] / custom["ocr_count"]
                results[scale]["ocr_f1"] = custom["ocr_f1"] / custom["ocr_count"]
            if custom.get("ctx_count", 0) > 0:
                results[scale]["ctx_recall"] = custom["ctx_recall"] / custom["ctx_count"]
                results[scale]["ctx_precision"] = custom["ctx_precision"] / custom["ctx_count"]
                results[scale]["ctx_f1"] = custom["ctx_f1"] / custom["ctx_count"]
            if custom.get("node_count", 0) > 0:
                results[scale]["node_recall"] = custom["node_recall"] / custom["node_count"]
                results[scale]["node_precision"] = custom["node_precision"] / custom["node_count"]
                results[scale]["node_f1"] = custom["node_f1"] / custom["node_count"]
            if custom.get("role_count", 0) > 0:
                results[scale]["role_recall"] = custom["role_recall"] / custom["role_count"]
                results[scale]["role_precision"] = custom["role_precision"] / custom["role_count"]
                results[scale]["role_f1"] = custom["role_f1"] / custom["role_count"]
            if custom.get("struct_count", 0) > 0:
                results[scale]["struct_recall"] = custom["struct_recall"] / custom["struct_count"]
                results[scale]["struct_precision"] = custom["struct_precision"] / custom["struct_count"]
                results[scale]["struct_f1"] = custom["struct_f1"] / custom["struct_count"]
            if custom.get("novel_struct_count", 0) > 0:
                results[scale]["novel_struct_ratio"] = custom["novel_struct_ratio"] / custom["novel_struct_count"]
            if custom.get("novel_struct_cov_count", 0) > 0:
                results[scale]["novel_struct_recall"] = custom["novel_struct_recall"] / custom["novel_struct_cov_count"]
                results[scale]["novel_struct_precision"] = custom["novel_struct_precision"] / custom["novel_struct_cov_count"]
                results[scale]["novel_struct_f1"] = custom["novel_struct_f1"] / custom["novel_struct_cov_count"]
            if custom.get("gt_struct_total", 0) > 0:
                results[scale]["gt_struct_ratio"] = custom["gt_struct_count"] / custom["gt_struct_total"]
            if custom.get("len_count", 0) > 0:
                results[scale]["len_ratio"] = custom["len_ratio"] / custom["len_count"]
            if custom.get("rep3_count", 0) > 0:
                results[scale]["rep3_ratio"] = custom["rep3_sum"] / custom["rep3_count"]

        if args.strip_leading_figure_prefix:
            strip_summary: Dict[str, Any] = {"enabled": True}
            for scale in ["short", "long", "desc", "all"]:
                st = strip_stats_by_scale.get(scale) or _init_strip_stats()
                pairs = int(st.get("pairs", 0))
                pred_changed = int(st.get("pred_changed", 0))
                ref_changed = int(st.get("ref_changed", 0))
                both_changed = int(st.get("both_changed", 0))
                strip_summary[scale] = {
                    "pairs": pairs,
                    "pred_changed": pred_changed,
                    "ref_changed": ref_changed,
                    "both_changed": both_changed,
                    "pred_changed_rate": (pred_changed / pairs) if pairs > 0 else 0.0,
                    "ref_changed_rate": (ref_changed / pairs) if pairs > 0 else 0.0,
                    "both_changed_rate": (both_changed / pairs) if pairs > 0 else 0.0,
                    "examples": st.get("examples", []),
                }
            results["_normalization"] = strip_summary

        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {"pred_only": True, "counts": counts_by_scale}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

