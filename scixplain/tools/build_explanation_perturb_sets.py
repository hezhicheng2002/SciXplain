#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            n += 1
    return n


def _mask_rect(w: int, h: int, rng: random.Random) -> tuple[int, int, int, int]:
    rw = max(8, int(w * rng.uniform(0.2, 0.35)))
    rh = max(8, int(h * rng.uniform(0.2, 0.35)))
    x0 = rng.randint(0, max(0, w - rw))
    y0 = rng.randint(0, max(0, h - rh))
    return x0, y0, x0 + rw, y0 + rh


def _shuffle_ocr_text(s: str, rng: random.Random) -> str:
    toks = [t for t in str(s or '').split() if t]
    if len(toks) <= 1:
        return str(s or '')
    rng.shuffle(toks)
    return ' '.join(toks)


def _deranged_indices(n: int, rng: random.Random) -> List[int]:
    idx = list(range(n))
    if n <= 1:
        return idx
    rng.shuffle(idx)
    for i in range(n):
        if idx[i] == i:
            j = (i + 1) % n
            idx[i], idx[j] = idx[j], idx[i]
    return idx


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "their",
    "our",
    "we",
    "they",
    "can",
    "may",
    "using",
}


def _lexical_tokens(text: str, max_unique: int = 160) -> set[str]:
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(str(text or "").lower()):
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        out.add(tok)
        if len(out) >= max(8, int(max_unique)):
            break
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    union = len(a | b)
    return float(inter) / float(max(1, union))


def _build_hard_shuffle_indices(rows: List[dict], rng: random.Random) -> tuple[List[int], List[float]]:
    n = len(rows)
    if n <= 1:
        return list(range(n)), [0.0 for _ in range(n)]

    fallback = _deranged_indices(n, rng)

    token_sets: List[set[str]] = []
    ocr_lens: List[int] = []
    img_paths: List[str] = []
    src_tags: List[str] = []
    article_ids: List[str] = []
    for r in rows:
        ctx = str(r.get("context") or "")[:5000]
        para = str(r.get("paragraph") or "")[:2000]
        ocr = str(r.get("ocr") or "")[:2000]
        token_sets.append(_lexical_tokens(f"{ctx} {para} {ocr}"))
        ocr_lens.append(len(str(r.get("ocr") or "").split()))
        img_paths.append(str(r.get("image_path") or ""))
        src_tags.append(str(r.get("source") or ""))
        article_ids.append(str(r.get("article_id") or ""))

    picked = fallback[:]
    picked_score = [0.0 for _ in range(n)]
    for i in range(n):
        best_j = -1
        best_score = -1e9
        set_i = token_sets[i]
        len_i = max(1, ocr_lens[i])
        for j in range(n):
            if j == i:
                continue
            if img_paths[i] and img_paths[j] and img_paths[i] == img_paths[j]:
                continue
            sim = _jaccard(set_i, token_sets[j])
            len_pen = abs(len_i - max(1, ocr_lens[j])) / float(max(len_i, max(1, ocr_lens[j])))
            score = sim - 0.08 * len_pen
            if src_tags[i] and src_tags[i] == src_tags[j]:
                score += 0.02
            if article_ids[i] and article_ids[i] == article_ids[j]:
                score += 0.05
            score += rng.random() * 1e-6
            if score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            picked[i] = best_j
            picked_score[i] = float(best_score)
        else:
            picked_score[i] = 0.0
    return picked, picked_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_path = Path(args.input_jsonl)
    out_root = Path(args.out_root)
    img_dir = out_root / 'region_drop_images'
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for r in load_jsonl(in_path):
        if args.limit > 0 and len(rows) >= args.limit:
            break
        rows.append(r)
    n = len(rows)

    region_rows = []
    shuffle_rows = []
    mask_rows = []
    image_shuffle_rows = []
    hard_image_shuffle_rows = []
    visual_zero_rows = []
    visual_mean_rows = []
    visual_noise_rows = []

    shuffle_idx = _deranged_indices(n, rng)
    hard_shuffle_idx, hard_shuffle_scores = _build_hard_shuffle_indices(rows, rng)

    for i, r in enumerate(rows):
        uid = str(r.get('uid') or f'row{i + 1}')
        img_path = str(r.get('image_path') or '')

        # region drop: black rectangle on copied image
        rr = dict(r)
        try:
            ip = Path(img_path)
            if ip.exists():
                im = Image.open(ip).convert('RGB')
                w, h = im.size
                box = _mask_rect(w, h, rng)
                draw = ImageDraw.Draw(im)
                draw.rectangle(box, fill=(0, 0, 0))
                out_img = img_dir / f'{uid}.png'
                im.save(out_img)
                rr['image_path'] = str(out_img)
                rr['perturbation'] = 'region_drop'
                rr['region_drop_box'] = box
            else:
                rr['perturbation'] = 'region_drop'
                rr['region_drop_box'] = None
        except Exception:
            rr['perturbation'] = 'region_drop'
            rr['region_drop_box'] = None
        region_rows.append(rr)

        # shuffle OCR: keep image same, shuffle OCR/context tokens
        sr = dict(r)
        sr['ocr'] = _shuffle_ocr_text(sr.get('ocr', ''), rng)
        if sr.get('context'):
            sr['context'] = _shuffle_ocr_text(sr['context'], rng)
        sr['perturbation'] = 'shuffle_ocr'
        shuffle_rows.append(sr)

        # context masking: cut paragraph/context
        mr = dict(r)
        mr['paragraph'] = ''
        mr['context'] = ''
        mr['perturbation'] = 'context_masking'
        mask_rows.append(mr)

        # image shuffle: keep text/context, replace image path with another sample image
        ir = dict(r)
        if n > 1:
            alt = rows[shuffle_idx[i]]
            alt_img = str(alt.get('image_path') or '')
            if alt_img:
                ir['image_path'] = alt_img
                ir['image_shuffle_from_uid'] = alt.get('uid')
        ir['perturbation'] = 'image_shuffle'
        image_shuffle_rows.append(ir)

        # hard image shuffle: choose a lexically similar context sample as counterfactual.
        hr = dict(r)
        if n > 1:
            alt_h = rows[hard_shuffle_idx[i]]
            alt_h_img = str(alt_h.get('image_path') or '')
            if alt_h_img:
                hr['image_path'] = alt_h_img
                hr['hard_image_shuffle_from_uid'] = alt_h.get('uid')
                hr['hard_image_shuffle_score'] = float(hard_shuffle_scores[i])
        hr['perturbation'] = 'hard_image_shuffle'
        hard_image_shuffle_rows.append(hr)

        # visual token zero: keep same input row, visual ablation is done at model side.
        vr = dict(r)
        vr['perturbation'] = 'visual_token_zero'
        visual_zero_rows.append(vr)

        vm = dict(r)
        vm['perturbation'] = 'visual_token_mean'
        visual_mean_rows.append(vm)

        vn = dict(r)
        vn['perturbation'] = 'visual_token_noise'
        visual_noise_rows.append(vn)

    paths: Dict[str, Path] = {
        'region_drop': out_root / 'explanation_region_drop.jsonl',
        'shuffle_ocr': out_root / 'explanation_shuffle_ocr.jsonl',
        'context_masking': out_root / 'explanation_context_masking.jsonl',
        'image_shuffle': out_root / 'explanation_image_shuffle.jsonl',
        'hard_image_shuffle': out_root / 'explanation_hard_image_shuffle.jsonl',
        'visual_token_zero': out_root / 'explanation_visual_token_zero.jsonl',
        'visual_token_mean': out_root / 'explanation_visual_token_mean.jsonl',
        'visual_token_noise': out_root / 'explanation_visual_token_noise.jsonl',
    }

    stats = {
        'region_drop': write_jsonl(paths['region_drop'], region_rows),
        'shuffle_ocr': write_jsonl(paths['shuffle_ocr'], shuffle_rows),
        'context_masking': write_jsonl(paths['context_masking'], mask_rows),
        'image_shuffle': write_jsonl(paths['image_shuffle'], image_shuffle_rows),
        'hard_image_shuffle': write_jsonl(paths['hard_image_shuffle'], hard_image_shuffle_rows),
        'visual_token_zero': write_jsonl(paths['visual_token_zero'], visual_zero_rows),
        'visual_token_mean': write_jsonl(paths['visual_token_mean'], visual_mean_rows),
        'visual_token_noise': write_jsonl(paths['visual_token_noise'], visual_noise_rows),
    }
    print(json.dumps({'rows_in': n, 'rows_out': stats, 'out_root': str(out_root)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
