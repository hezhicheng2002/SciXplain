#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ijson
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scixplain.models.towers import CLIPVisionTower, RexOmniWrapper
from transformers import AutoTokenizer
from scixplain.models.lora import apply_lora
from scixplain.tools.train_ai2d_teacher import NODE_TYPES, ROLE_TYPES, REL_TYPES, pool_roi_features

try:
    from pycocotools import mask as coco_mask
    _HAS_COCO = True
except Exception:
    coco_mask = None
    _HAS_COCO = False


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


NODE2IDX = {t: i for i, t in enumerate(NODE_TYPES)}
ROLE2IDX = {t: i for i, t in enumerate(ROLE_TYPES)}
REL2IDX = {t: i for i, t in enumerate(REL_TYPES)}


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "__meta__" in rec:
                continue
            yield rec


def _read_cache_meta(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            meta = rec.get("__meta__")
            if isinstance(meta, dict):
                return meta
            return None
    return None


def _load_struct_cache(path: str) -> Tuple[Optional[Dict], Dict[str, Dict]]:
    meta = _read_cache_meta(path)
    struct_map: Dict[str, Dict] = {}
    for rec in _iter_jsonl(path):
        fig_path = rec.get("figure_path")
        if not fig_path:
            continue
        struct_map[fig_path] = {
            "figure_path": fig_path,
            "width": rec.get("width"),
            "height": rec.get("height"),
            "nodes": rec.get("nodes") or [],
            "edges": rec.get("edges") or [],
            "quality_flags": rec.get("quality_flags"),
            "mask_stats": rec.get("mask_stats"),
            "ocr_stats": rec.get("ocr_stats"),
        }
    return meta, struct_map


def _decode_rle(seg: Dict) -> Optional[np.ndarray]:
    if not _HAS_COCO:
        return None
    if not seg:
        return None
    counts = seg.get("counts")
    size = seg.get("size")
    if counts is None or size is None:
        return None
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    try:
        rle = {"counts": counts, "size": size}
        mask = coco_mask.decode(rle)
    except Exception:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask

def _fig_count_from_stats(stats_path: str, split: str) -> int:
    try:
        stats = _load_json(stats_path)
        return int(stats.get("figure_counts", {}).get(split, 0))
    except Exception:
        return 0


def _safe_open_image(path: str) -> Image.Image | None:
    resolved = _remap_legacy_figure_path(path)
    try:
        img = Image.open(resolved).convert("RGB")
        return img
    except Exception:
        return None


def _remap_legacy_figure_path(path: str) -> str:
    if not path:
        return path
    p = Path(path)
    if p.exists():
        return str(p)

    repo_root = Path(__file__).resolve().parents[2]
    path_str = str(path)
    path_lower = path_str.lower()
    candidates = []

    if "/scistruct/" in path_lower:
        parts = p.parts
        if "SciStruct" in parts:
            idx = parts.index("SciStruct")
            rel = Path(*parts[idx + 1 :])
            candidates.append(repo_root / "dataset" / "SciStruct" / rel)
        if len(parts) >= 3:
            paper_id = parts[-3]
            candidates.append(repo_root / "Benchmark" / "datasets" / "images" / "scistruct" / paper_id / p.name)

    if "scicap" in path_lower:
        candidates.append(repo_root / "Benchmark" / "datasets" / "images" / "scicap" / p.name)
        candidates.append(repo_root / "dataset" / "scicap_mlbcap_node_diagram_v2" / "images_store" / p.name)

    for cand in candidates:
        if cand.exists():
            return str(cand)
    return path


def _collect_ocr_text(ocr_list: List[Dict], max_items: int, min_chars: int) -> str:
    texts: List[str] = []
    if not ocr_list:
        return ""
    for item in ocr_list:
        if not isinstance(item, dict):
            continue
        txt = str(item.get("text") or "").strip()
        if len(txt) < min_chars:
            continue
        texts.append(txt)
        if len(texts) >= max_items:
            break
    return " ".join(texts).strip()


def _normalize_sam2_cfg(cfg_path: str) -> str:
    path = Path(cfg_path)
    if path.is_file():
        parts = path.parts
        if "configs" in parts:
            idx = parts.index("configs")
            return "/".join(parts[idx:])
        return path.name
    return cfg_path


class FigurePathStream(IterableDataset):
    def __init__(self, json_path: str, split: str = "train", repeat: bool = True, max_items: int | None = None):
        self.json_path = str(json_path)
        self.split = split
        self.repeat = repeat
        self.max_items = max_items

    def __iter__(self) -> Iterable[Dict]:
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        while True:
            produced = 0
            with open(self.json_path, "r", encoding="utf-8") as f:
                for item in ijson.items(f, "item"):
                    figures = item.get("figures") or []
                    for fig in figures:
                        path = fig.get("figure_path")
                        if not path:
                            continue
                        if produced % num_workers != worker_id:
                            produced += 1
                            continue
                        if self.max_items is not None and produced >= self.max_items:
                            return
                        produced += 1
                        yield {"figure_path": path}
            if not self.repeat:
                break


class StructCacheStream(IterableDataset):
    def __init__(self, jsonl_path: str, repeat: bool = True, max_items: int | None = None):
        self.jsonl_path = str(jsonl_path)
        self.repeat = repeat
        self.max_items = max_items

    def __iter__(self) -> Iterable[Dict]:
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        while True:
            produced = 0
            for rec in _iter_jsonl(self.jsonl_path):
                fig_path = rec.get("figure_path")
                if not fig_path:
                    continue
                if produced % num_workers != worker_id:
                    produced += 1
                    continue
                if self.max_items is not None and produced >= self.max_items:
                    return
                produced += 1
                yield {"figure_path": fig_path, "struct": rec}
            if not self.repeat:
                break


class GeomCacheStream(IterableDataset):
    def __init__(
        self,
        jsonl_path: str,
        struct_map: Optional[Dict[str, Dict]] = None,
        require_struct: bool = False,
        drop_empty: bool = True,
        repeat: bool = True,
        max_items: int | None = None,
    ):
        self.jsonl_path = str(jsonl_path)
        self.struct_map = struct_map or {}
        self.require_struct = require_struct
        self.drop_empty = drop_empty
        self.repeat = repeat
        self.max_items = max_items

    def __iter__(self) -> Iterable[Dict]:
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        while True:
            produced = 0
            for rec in _iter_jsonl(self.jsonl_path):
                fig_path = rec.get("figure_path")
                if not fig_path:
                    continue
                if produced % num_workers != worker_id:
                    produced += 1
                    continue
                if self.max_items is not None and produced >= self.max_items:
                    return
                produced += 1
                struct = self.struct_map.get(fig_path)
                if self.require_struct and not struct:
                    continue
                if struct:
                    out_struct = dict(struct)
                else:
                    out_struct = {"figure_path": fig_path}
                if rec.get("quality_flags") is not None:
                    out_struct["quality_flags"] = rec.get("quality_flags")
                if rec.get("mask_stats") is not None:
                    out_struct["mask_stats"] = rec.get("mask_stats")
                if rec.get("ocr_stats") is not None:
                    out_struct["ocr_stats"] = rec.get("ocr_stats")
                if rec.get("ocr") is not None:
                    out_struct["ocr"] = rec.get("ocr")
                if rec.get("width") is not None:
                    out_struct["width"] = rec.get("width")
                if rec.get("height") is not None:
                    out_struct["height"] = rec.get("height")
                if self.drop_empty:
                    quality = out_struct.get("quality_flags") or {}
                    has_any = quality.get("has_any_structure")
                    if has_any is None:
                        ocr_stats = out_struct.get("ocr_stats") or {}
                        mask_stats = out_struct.get("mask_stats") or {}
                        ocr_list = out_struct.get("ocr") or []
                        has_any = bool(
                            (ocr_stats.get("num_boxes", 0) > 0)
                            or (mask_stats.get("num_masks", 0) > 0)
                            or (len(ocr_list) > 0)
                        )
                    if not has_any:
                        continue
                yield {
                    "figure_path": fig_path,
                    "struct": out_struct,
                    "sam2": rec.get("sam2") or [],
                }
            if not self.repeat:
                break


class MultiSourceImageDataset(IterableDataset):
    def __init__(self, sources: List[FigurePathStream], weights: List[float], seed: int = 42):
        super().__init__()
        self.sources = sources
        self.weights = weights
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        rng = random.Random(self.seed + worker_id)
        iters = [iter(src) for src in self.sources]
        while True:
            idx = rng.choices(range(len(iters)), weights=self.weights, k=1)[0]
            try:
                path = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self.sources[idx])
                path = next(iters[idx])
            figure_path = path.get("figure_path") if isinstance(path, dict) else path
            img = _safe_open_image(figure_path)
            if img is None:
                continue
            yield img, idx, (path.get("struct") if isinstance(path, dict) else None), (path.get("sam2") if isinstance(path, dict) else None)


class SingleSourceImageDataset(IterableDataset):
    def __init__(self, source: IterableDataset):
        super().__init__()
        self.source = source

    def __iter__(self):
        for rec in self.source:
            if isinstance(rec, tuple):
                yield rec
                continue
            if not isinstance(rec, dict):
                continue
            fig_path = rec.get("figure_path")
            if not fig_path:
                continue
            img = _safe_open_image(fig_path)
            if img is None:
                continue
            yield img, 0, rec.get("struct"), rec.get("sam2")


def collate_fn(batch):
    images = [b[0] for b in batch]
    sources = [b[1] for b in batch]
    structs = [b[2] for b in batch]
    sam2 = [b[3] for b in batch]
    return images, sources, structs, sam2


def tokens_to_grid(tokens, grid_thw=None):
    if isinstance(tokens, (list, tuple)):
        b = len(tokens)
        token_list = tokens
    else:
        b = tokens.shape[0]
        token_list = [tokens[i] for i in range(b)]

    grids = []
    shapes = []
    grid_thw_list = None
    if grid_thw is not None:
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_list = grid_thw.detach().cpu().tolist()
        else:
            grid_thw_list = grid_thw
        if len(grid_thw_list) == 3 and isinstance(grid_thw_list[0], (int, float)):
            grid_thw_list = [grid_thw_list for _ in range(b)]

    for i, tok in enumerate(token_list):
        n, c = tok.shape
        if grid_thw_list is not None:
            t, h, w = [int(x) for x in grid_thw_list[i]]
            h = h * max(1, t)
            expected = h * w
            if n == expected + 1:
                tok = tok[1:, :]
                n = tok.shape[0]
            if n != expected:
                ratio = (h * w) / float(n)
                scale = int(round((ratio) ** 0.5))
                if scale > 1 and (h // scale) * (w // scale) == n:
                    h = h // scale
                    w = w // scale
                else:
                    raise ValueError(f"Unexpected token count {n} vs grid {h}x{w}")
            grid = tok.reshape(h, w, c)
            grids.append(grid)
            shapes.append((h, w))
            continue

        grid = int((n - 1) ** 0.5)
        if grid * grid + 1 == n:
            tok = tok[1:, :]
        else:
            grid = int(n ** 0.5)
            if grid * grid != n:
                raise ValueError(f"Unexpected token count {n}; cannot form grid")
        grids.append(tok.reshape(grid, grid, c))
        shapes.append((grid, grid))

    return grids, shapes


def _safe_feat(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)


def normalize_feat(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = _safe_feat(x).float()
    denom = torch.linalg.norm(x, dim=1, keepdim=True)
    denom = torch.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)
    return x / (denom + eps)


def _edge_iou(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    ix1 = torch.max(src[:, 0], dst[:, 0])
    iy1 = torch.max(src[:, 1], dst[:, 1])
    ix2 = torch.min(src[:, 2], dst[:, 2])
    iy2 = torch.min(src[:, 3], dst[:, 3])
    iw = (ix2 - ix1).clamp_min(0.0)
    ih = (iy2 - iy1).clamp_min(0.0)
    inter = iw * ih
    src_area = (src[:, 2] - src[:, 0]).clamp_min(0.0) * (src[:, 3] - src[:, 1]).clamp_min(0.0)
    dst_area = (dst[:, 2] - dst[:, 0]).clamp_min(0.0) * (dst[:, 3] - dst[:, 1]).clamp_min(0.0)
    union = src_area + dst_area - inter
    return inter / (union + 1e-6)


def _edge_geom_features(bboxes: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    src = bboxes[pairs[:, 0]]
    dst = bboxes[pairs[:, 1]]
    src_cx = (src[:, 0] + src[:, 2]) * 0.5
    src_cy = (src[:, 1] + src[:, 3]) * 0.5
    dst_cx = (dst[:, 0] + dst[:, 2]) * 0.5
    dst_cy = (dst[:, 1] + dst[:, 3]) * 0.5
    dx = dst_cx - src_cx
    dy = dst_cy - src_cy
    dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
    iou = _edge_iou(src, dst)
    src_w = (src[:, 2] - src[:, 0]).clamp_min(1e-6)
    src_h = (src[:, 3] - src[:, 1]).clamp_min(1e-6)
    dst_w = (dst[:, 2] - dst[:, 0]).clamp_min(1e-6)
    dst_h = (dst[:, 3] - dst[:, 1]).clamp_min(1e-6)
    w_ratio = dst_w / src_w
    h_ratio = dst_h / src_h
    angle = torch.atan2(dy, dx)
    geom = torch.stack([dx, dy, dist, iou, w_ratio, h_ratio, torch.sin(angle), torch.cos(angle)], dim=-1)
    return geom


def _prepare_struct_rec(rec: Dict, edge_score_thresh: float = 0.0) -> Tuple[List[List[float]], List[int], List[int], List[Tuple[int, int, int]]]:
    nodes = rec.get("nodes") or []
    edges = rec.get("edges") or []
    bboxes: List[List[float]] = []
    node_types: List[int] = []
    node_roles: List[int] = []
    keep = []
    for idx, node in enumerate(nodes):
        ntype = node.get("type")
        if isinstance(ntype, str):
            if ntype not in NODE2IDX:
                continue
            type_idx = NODE2IDX[ntype]
        else:
            type_idx = int(ntype) if ntype is not None else None
            if type_idx is None or type_idx < 0 or type_idx >= len(NODE_TYPES):
                continue
        role = node.get("role") or "other"
        if isinstance(role, str):
            role_idx = ROLE2IDX.get(role, ROLE2IDX["other"])
        else:
            role_idx = int(role) if role is not None else ROLE2IDX["other"]
        bbox = node.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        keep.append(idx)
        bboxes.append([float(x) for x in bbox])
        node_types.append(type_idx)
        node_roles.append(role_idx)
    idx_map = {old: new for new, old in enumerate(keep)}
    edge_list: List[Tuple[int, int, int]] = []
    for e in edges:
        if isinstance(e, dict):
            if edge_score_thresh > 0 and float(e.get("score") or 0.0) < edge_score_thresh:
                continue
            o = e.get("src")
            d = e.get("dst")
            rel = e.get("type")
        else:
            continue
        if o not in idx_map or d not in idx_map:
            continue
        if isinstance(rel, str):
            if rel not in REL2IDX:
                continue
            rel_idx = REL2IDX[rel]
        else:
            rel_idx = int(rel) if rel is not None else None
            if rel_idx is None or rel_idx < 0 or rel_idx >= len(REL_TYPES):
                continue
        edge_list.append((idx_map[o], idx_map[d], rel_idx))
    return bboxes, node_types, node_roles, edge_list


def _sample_edge_pairs(
    num_nodes: int,
    edges: List[Tuple[int, int, int]],
    node_types: List[int],
    neg_ratio: float,
    same_src_neg: int,
    use_rev_neg: bool,
    rng: random.Random,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    pos_pairs = [(o, d) for o, d, _ in edges]
    pos_labels = [rel for _, _, rel in edges]
    pos_set = set(pos_pairs)

    rev_pairs = []
    if use_rev_neg:
        for o, d in pos_pairs:
            if (d, o) not in pos_set:
                rev_pairs.append((d, o))

    same_src_pairs = []
    if same_src_neg > 0:
        for o, d in pos_pairs:
            cand = [i for i, t in enumerate(node_types) if i not in (o, d) and t == node_types[d]]
            if not cand:
                cand = [i for i in range(num_nodes) if i not in (o, d)]
            if cand:
                v = rng.choice(cand)
                if (o, v) not in pos_set:
                    same_src_pairs.append((o, v))

    neg_pairs = []
    num_neg = int(len(pos_pairs) * neg_ratio)
    num_extra = max(0, num_neg - len(rev_pairs) - len(same_src_pairs))
    tries = 0
    while len(neg_pairs) < num_extra and tries < num_extra * 10:
        i = rng.randrange(num_nodes)
        j = rng.randrange(num_nodes)
        if i == j or (i, j) in pos_set:
            tries += 1
            continue
        if (i, j) in rev_pairs or (i, j) in same_src_pairs:
            tries += 1
            continue
        neg_pairs.append((i, j))

    all_pairs = pos_pairs + rev_pairs + same_src_pairs + neg_pairs
    all_labels = pos_labels + [REL2IDX["none"]] * (len(rev_pairs) + len(same_src_pairs) + len(neg_pairs))
    return all_pairs, all_labels


def _sam2_union_mask(sam2_list: List[Dict], max_masks: int = 0) -> Optional[np.ndarray]:
    if not sam2_list or not _HAS_COCO:
        return None
    items = sam2_list
    if max_masks > 0 and len(items) > max_masks:
        items = sorted(items, key=lambda x: float(x.get("predicted_iou") or 0.0), reverse=True)[:max_masks]
    union = None
    for ann in items:
        seg = ann.get("segmentation")
        mask = _decode_rle(seg) if seg else None
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8)
        union = mask if union is None else np.maximum(union, mask)
    return union


class RexOmniTeacher(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_filters: List[str],
        device: torch.device,
    ):
        super().__init__()
        self.rex = RexOmniWrapper()
        apply_lora(
            [("rex", self.rex.model)],
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_base=True,
            name_filter=lora_filters,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(ckpt.get("model", {}), strict=False)
        self.rex.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.device = device
        self.to(self.device)

    @torch.no_grad()
    def forward(self, images):
        batch = self.rex.preprocess(images)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.rex(batch)
        return out["tokens"], out.get("grid_thw")


class SAM2GeometryTeacher(nn.Module):
    def __init__(self, ckpt_path: str, cfg_path: str, device: torch.device):
        super().__init__()
        sssi_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SSSI"))
        if sssi_root not in sys.path:
            sys.path.append(sssi_root)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        cfg_name = _normalize_sam2_cfg(cfg_path)
        self.model = build_sam2(cfg_name, ckpt_path, device=str(device))
        self.model.eval()
        self.predictor = SAM2ImagePredictor(self.model)
        self.device = device

    @torch.no_grad()
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        np_imgs = [np.array(im.convert("RGB")) for im in images]
        self.predictor.set_image_batch(np_imgs)
        return self.predictor.get_image_embedding()


class DINOv3StyleTeacher(nn.Module):
    def __init__(self, weights_path: str, device: torch.device):
        super().__init__()
        dino_roots = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "DINOv3FD", "dinov3")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "DINOv3-FD", "dinov3")),
        ]
        env_dino_root = os.environ.get("DINOV3_ROOT", "").strip()
        if env_dino_root:
            dino_roots.extend([env_dino_root, os.path.join(env_dino_root, "dinov3")])
        if weights_path:
            w_dir = os.path.abspath(os.path.dirname(weights_path))
            dino_roots.extend([
                w_dir,
                os.path.join(w_dir, "dinov3"),
                os.path.dirname(w_dir),
            ])
        import_err = None
        checked = []
        for root in dino_roots:
            if not root:
                continue
            root = os.path.abspath(root)
            if root in checked:
                continue
            checked.append(root)
            if os.path.isdir(root) and root not in sys.path:
                sys.path.append(root)
            try:
                from dinov3.hub.backbones import dinov3_vitl16
                from dinov3.data.transforms import make_eval_transform
                break
            except Exception as exc:
                import_err = exc
        else:
            raise RuntimeError(
                f"failed to import dinov3 package; checked roots: {checked}"
            ) from import_err

        self.model = dinov3_vitl16(weights=weights_path, pretrained=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.transform = make_eval_transform(resize_size=256, crop_size=224)
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(im) for im in images]).to(self.device)
        feats = self.model.forward_features(batch)
        return feats["x_norm_clstoken"]


class StructNodeHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.type_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(NODE_TYPES)),
        )
        self.role_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(ROLE_TYPES)),
        )

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.type_head(feats), self.role_head(feats)


class StructEdgeHead(nn.Module):
    def __init__(self, in_dim: int, geom_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 4 + geom_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(REL_TYPES)),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat)


class GeomMaskHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)


def trainable_state_dict(model: nn.Module) -> dict:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    return {k: v for k, v in sd.items() if k in trainable}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scistruct_json", type=str, default="dataset/SciStruct/dataset_split_811/train.json")
    ap.add_argument(
        "--scicap_json",
        type=str,
        default="dataset/scicap_mlbcap_node_diagram_v2/dataset_split/train.json",
    )
    ap.add_argument(
        "--scistruct_stats",
        type=str,
        default="dataset/SciStruct/dataset_split_811/dataset_statistics.json",
    )
    ap.add_argument(
        "--scicap_stats",
        type=str,
        default="dataset/scicap_mlbcap_node_diagram_v2/dataset_split/dataset_statistics.json",
    )
    ap.add_argument("--scistruct_struct_cache", type=str, default="outputs/structure_cache/scistruct_train_struct.jsonl")
    ap.add_argument("--scicap_struct_cache", type=str, default="outputs/structure_cache/scicap_train_struct.jsonl")
    ap.add_argument("--scistruct_geom_cache", type=str, default="outputs/geom_ocr/scistruct_train_vl.jsonl")
    ap.add_argument("--scicap_geom_cache", type=str, default="outputs/geom_ocr/scicap_train_vl.jsonl")
    ap.add_argument("--scistruct_val_struct_cache", type=str, default="outputs/structure_cache/scistruct_val_struct.jsonl")
    ap.add_argument("--scicap_val_struct_cache", type=str, default="outputs/structure_cache/scicap_val_struct.jsonl")
    ap.add_argument("--scistruct_val_geom_cache", type=str, default="outputs/geom_ocr/scistruct_val_vl.jsonl")
    ap.add_argument("--scicap_val_geom_cache", type=str, default="outputs/geom_ocr/scicap_val_vl.jsonl")
    ap.add_argument("--scistruct_test_struct_cache", type=str, default="")
    ap.add_argument("--scicap_test_struct_cache", type=str, default="")
    ap.add_argument("--scistruct_test_geom_cache", type=str, default="")
    ap.add_argument("--scicap_test_geom_cache", type=str, default="")
    ap.add_argument("--struct_schema", type=str, default="struct_v2_with_arrow")
    ap.add_argument("--require_struct_schema", action="store_true")
    ap.add_argument("--allow_missing_struct_meta", action="store_true")
    ap.add_argument("--struct_required", action="store_true")
    ap.add_argument("--struct_edge_thresh", type=float, default=0.0)
    ap.add_argument("--out_dir", type=str, default="checkpoints/visual_student")
    ap.add_argument("--init_ckpt", type=str, default="")
    ap.add_argument("--init_step", type=int, default=-1)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_batch_size", type=int, default=0)
    ap.add_argument("--val_num_workers", type=int, default=-1)
    ap.add_argument("--val_max_items", type=int, default=0)
    ap.add_argument("--val_log", type=str, default="")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--image_size", type=int, default=384)

    ap.add_argument(
        "--rex_ckpt",
        type=str,
        default="checkpoints/ai2d_teacher_v3_edge/ckpt_best.pt",
    )
    ap.add_argument(
        "--sam2_ckpt",
        type=str,
        default="SSSI/checkpoint/sam2.1_hiera_large.pt",
    )
    ap.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    ap.add_argument(
        "--dino_ckpt",
        type=str,
        default="checkpoints/dinov3_vitl16_pretrain_lvd1689m.pth",
    )

    ap.add_argument("--lambda_struct", type=float, default=2.0)
    ap.add_argument("--lambda_geom", type=float, default=5.0)
    ap.add_argument("--lambda_style", type=float, default=0.01)
    ap.add_argument("--lambda_graph", type=float, default=5.0)
    ap.add_argument("--lambda_node", type=float, default=1.0)
    ap.add_argument("--lambda_role", type=float, default=0.5)
    ap.add_argument("--lambda_edge", type=float, default=1.0)
    ap.add_argument("--lambda_ocr", type=float, default=0.0)
    ap.add_argument("--feat_warmup_steps", type=int, default=2000)
    ap.add_argument("--feat_ramp_steps", type=int, default=4000)
    ap.add_argument("--ocr_density_ref", type=float, default=0.1)
    ap.add_argument("--ocr_weight_min", type=float, default=0.2)
    ap.add_argument("--ocr_mode", type=str, default="global", choices=["global", "roi"])
    ap.add_argument("--ocr_max_items", type=int, default=32)
    ap.add_argument("--ocr_min_chars", type=int, default=1)
    ap.add_argument("--ocr_max_length", type=int, default=64)
    ap.add_argument("--ocr_adapter", action="store_true")
    ap.add_argument("--ocr_detach", action="store_true")
    ap.add_argument("--ocr_warmup_steps", type=int, default=0)
    ap.add_argument("--ocr_ramp_steps", type=int, default=0)
    ap.add_argument(
        "--ocr_encoder_ckpt",
        type=str,
        default="",
        help="Optional separate encoder checkpoint used only for OCR features during eval.",
    )
    ap.add_argument(
        "--ocr_logit_scale",
        type=float,
        default=0.0,
        help="Use fixed logit scale for OCR contrastive loss; <=0 uses model logit_scale.",
    )
    ap.add_argument("--mask_area_ref", type=float, default=0.2)
    ap.add_argument("--mask_weight_min", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--student_device", type=str, default="cuda:0")
    ap.add_argument("--teacher_device", type=str, default="cuda:0")
    ap.add_argument("--rex_device", type=str, default="")
    ap.add_argument("--sam2_device", type=str, default="")
    ap.add_argument("--dino_device", type=str, default="")

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_filters",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,out_proj,fc1,fc2,proj",
    )
    ap.add_argument("--geom_mode", type=str, default="", choices=["", "sam2_feat", "sam2_mask", "none"])
    ap.add_argument("--geom_mask_topk", type=int, default=16)
    ap.add_argument("--edge_neg_ratio", type=float, default=1.0)
    ap.add_argument("--edge_same_src_neg", type=int, default=1)
    ap.add_argument("--no_edge_rev_neg", action="store_true")
    ap.add_argument("--stage", type=str, default="full", choices=["struct_only", "struct_geom", "struct_style", "full"])
    ap.add_argument("--disable_rex_teacher", action="store_true")
    ap.add_argument(
        "--disable_rex_distill_only",
        action="store_true",
        help="disable Rex distillation teacher but keep structural heads/losses enabled",
    )
    ap.add_argument("--disable_geom_teacher", action="store_true")
    ap.add_argument("--disable_style_teacher", action="store_true")
    ap.add_argument("--style_warmup_frac", type=float, default=0.2)
    ap.add_argument("--style_decay_frac", type=float, default=0.3)
    ap.add_argument("--style_min_ratio", type=float, default=0.3)
    ap.add_argument("--res_weight_min", type=float, default=0.0)
    ap.add_argument("--feat_weight_min", type=float, default=0.0)
    ap.add_argument("--style_weight_min", type=float, default=0.0)

    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    student_device = torch.device(args.student_device)
    teacher_device = torch.device(args.teacher_device)
    rex_device = torch.device(args.rex_device) if args.rex_device else teacher_device
    sam2_device = torch.device(args.sam2_device) if args.sam2_device else teacher_device
    dino_device = torch.device(args.dino_device) if args.dino_device else teacher_device

    stage_cfg = {
        "struct_only": {"struct": True, "geom": False, "style": False},
        "struct_geom": {"struct": True, "geom": True, "style": False},
        "struct_style": {"struct": True, "geom": True, "style": True},
        "full": {"struct": True, "geom": True, "style": True},
    }[args.stage]
    if args.disable_rex_teacher and args.disable_rex_distill_only:
        print("[warn] both --disable_rex_teacher and --disable_rex_distill_only set; full struct disable takes priority.")
    if args.disable_rex_teacher:
        print("[info] disabling RexOmni structural teacher (ablation).")
        stage_cfg["struct"] = False
    elif args.disable_rex_distill_only:
        print("[info] disabling RexOmni distillation only (struct heads/losses remain enabled).")
    if args.disable_geom_teacher:
        print("[info] disabling SAM2 geometry teacher (ablation).")
        stage_cfg["geom"] = False
    if args.disable_style_teacher:
        print("[info] disabling DINOv3 style teacher (ablation).")
        stage_cfg["style"] = False

    geom_mode = args.geom_mode
    geom_cache_any = any([
        args.scistruct_geom_cache and os.path.exists(args.scistruct_geom_cache),
        args.scicap_geom_cache and os.path.exists(args.scicap_geom_cache),
    ])
    if not geom_mode:
        geom_mode = "sam2_mask" if geom_cache_any else "sam2_feat"
    if geom_mode == "sam2_mask" and not _HAS_COCO:
        print("[warn] pycocotools not available; disabling sam2_mask geometry loss.")
        geom_mode = "none"
    if geom_mode == "sam2_mask" and not geom_cache_any:
        print("[warn] geom_mode=sam2_mask but no geom cache found; geometry loss will be zero.")
    if not stage_cfg["geom"]:
        geom_mode = "none"

    val_batch_size = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    val_num_workers = args.val_num_workers if args.val_num_workers >= 0 else args.num_workers
    val_max_items = args.val_max_items if args.val_max_items > 0 else None

    # Dataset weights by scale
    w_scistruct = _fig_count_from_stats(args.scistruct_stats, "train")
    w_scicap = _fig_count_from_stats(args.scicap_stats, "train")
    weights = [w_scistruct, w_scicap]
    if sum(weights) == 0:
        weights = [1.0, 1.0]

    def _validate_meta(meta: Optional[Dict], path: str):
        if not meta:
            if args.require_struct_schema and not args.allow_missing_struct_meta:
                raise ValueError(f"missing cache metadata for {path}")
            if not args.allow_missing_struct_meta:
                print(f"[warn] missing cache metadata for {path}")
            return
        if args.require_struct_schema and meta.get("schema_version") != args.struct_schema:
            raise ValueError(
                f"struct schema mismatch for {path}: {meta.get('schema_version')} vs {args.struct_schema}"
            )

    struct_meta_scistruct, struct_map_scistruct = (None, {})
    struct_meta_scicap, struct_map_scicap = (None, {})
    if args.scistruct_struct_cache and os.path.exists(args.scistruct_struct_cache):
        struct_meta_scistruct, struct_map_scistruct = _load_struct_cache(args.scistruct_struct_cache)
        _validate_meta(struct_meta_scistruct, args.scistruct_struct_cache)
    if args.scicap_struct_cache and os.path.exists(args.scicap_struct_cache):
        struct_meta_scicap, struct_map_scicap = _load_struct_cache(args.scicap_struct_cache)
        _validate_meta(struct_meta_scicap, args.scicap_struct_cache)
    val_struct_meta_scistruct, val_struct_map_scistruct = (None, {})
    val_struct_meta_scicap, val_struct_map_scicap = (None, {})
    if args.scistruct_val_struct_cache and os.path.exists(args.scistruct_val_struct_cache):
        val_struct_meta_scistruct, val_struct_map_scistruct = _load_struct_cache(args.scistruct_val_struct_cache)
        _validate_meta(val_struct_meta_scistruct, args.scistruct_val_struct_cache)
    if args.scicap_val_struct_cache and os.path.exists(args.scicap_val_struct_cache):
        val_struct_meta_scicap, val_struct_map_scicap = _load_struct_cache(args.scicap_val_struct_cache)
        _validate_meta(val_struct_meta_scicap, args.scicap_val_struct_cache)
    test_struct_meta_scistruct, test_struct_map_scistruct = (None, {})
    test_struct_meta_scicap, test_struct_map_scicap = (None, {})
    if args.scistruct_test_struct_cache and os.path.exists(args.scistruct_test_struct_cache):
        test_struct_meta_scistruct, test_struct_map_scistruct = _load_struct_cache(args.scistruct_test_struct_cache)
        _validate_meta(test_struct_meta_scistruct, args.scistruct_test_struct_cache)
    if args.scicap_test_struct_cache and os.path.exists(args.scicap_test_struct_cache):
        test_struct_meta_scicap, test_struct_map_scicap = _load_struct_cache(args.scicap_test_struct_cache)
        _validate_meta(test_struct_meta_scicap, args.scicap_test_struct_cache)

    sources = []
    if geom_mode == "sam2_mask" and args.scistruct_geom_cache and os.path.exists(args.scistruct_geom_cache):
        if args.struct_required and not struct_map_scistruct:
            raise ValueError("struct_required but scistruct struct cache missing.")
        sources.append(GeomCacheStream(
            args.scistruct_geom_cache,
            struct_map=struct_map_scistruct,
            require_struct=args.struct_required,
            drop_empty=True,
        ))
    elif args.scistruct_struct_cache and os.path.exists(args.scistruct_struct_cache):
        sources.append(StructCacheStream(args.scistruct_struct_cache))
    else:
        sources.append(FigurePathStream(args.scistruct_json, split="train"))

    if geom_mode == "sam2_mask" and args.scicap_geom_cache and os.path.exists(args.scicap_geom_cache):
        if args.struct_required and not struct_map_scicap:
            raise ValueError("struct_required but scicap struct cache missing.")
        sources.append(GeomCacheStream(
            args.scicap_geom_cache,
            struct_map=struct_map_scicap,
            require_struct=args.struct_required,
            drop_empty=True,
        ))
    elif args.scicap_struct_cache and os.path.exists(args.scicap_struct_cache):
        sources.append(StructCacheStream(args.scicap_struct_cache))
    else:
        sources.append(FigurePathStream(args.scicap_json, split="train"))
    dataset = MultiSourceImageDataset(sources, weights=weights, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Student
    student = CLIPVisionTower(output_attentions=False).to(student_device)
    for p in student.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    ocr_encoder = None
    if args.ocr_encoder_ckpt:
        if not args.eval_only:
            print("[warn] ocr_encoder_ckpt is intended for eval_only; ignoring in training.")
        elif not os.path.exists(args.ocr_encoder_ckpt):
            print(f"[warn] ocr_encoder_ckpt not found: {args.ocr_encoder_ckpt}")
        else:
            ocr_encoder = CLIPVisionTower(output_attentions=False).to(student_device)
            for p in ocr_encoder.parameters():
                p.requires_grad = False
            ocr_ckpt = torch.load(args.ocr_encoder_ckpt, map_location="cpu")
            ocr_encoder.load_state_dict(ocr_ckpt.get("model", {}), strict=False)
            ocr_encoder.eval()

    ocr_tokenizer = None
    if args.lambda_ocr > 0:
        ocr_tokenizer = AutoTokenizer.from_pretrained("google/siglip2-large-patch16-384", use_fast=True)

    # Teachers
    rex_teacher = None
    if stage_cfg["struct"] and not args.disable_rex_distill_only:
        rex_teacher = RexOmniTeacher(
            ckpt_path=args.rex_ckpt,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_filters=[s.strip() for s in args.lora_filters.split(",") if s.strip()],
            device=rex_device,
        )

    geom_teacher = None
    geom_proj = None
    geom_head = None
    if stage_cfg["geom"] and geom_mode == "sam2_feat":
        geom_teacher = SAM2GeometryTeacher(
            ckpt_path=args.sam2_ckpt,
            cfg_path=args.sam2_cfg,
            device=sam2_device,
        )
        geom_proj = nn.Conv2d(256, 768, kernel_size=1).to(student_device)
    elif stage_cfg["geom"] and geom_mode == "sam2_mask":
        geom_head = GeomMaskHead(768).to(student_device)

    style_teacher = None
    style_proj = None
    if stage_cfg["style"]:
        try:
            style_teacher = DINOv3StyleTeacher(
                weights_path=args.dino_ckpt,
                device=dino_device,
            )
            style_proj = nn.Linear(1024, 768).to(student_device)
        except Exception as exc:
            print(f"[warn] style teacher init failed ({exc}); disabling style loss.", file=sys.stderr)
            stage_cfg["style"] = False

    node_head = StructNodeHead(768).to(student_device) if stage_cfg["struct"] and args.lambda_graph > 0 else None
    edge_head = StructEdgeHead(768, geom_dim=8).to(student_device) if stage_cfg["struct"] and args.lambda_graph > 0 else None
    ocr_adapter = None
    if args.lambda_ocr > 0 and args.ocr_adapter:
        ocr_adapter = nn.Sequential(
            nn.Linear(student.vision_hidden, student.vision_hidden),
            nn.ReLU(),
            nn.Linear(student.vision_hidden, student.vision_hidden),
        ).to(student_device)

    start_step = 0
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        student.load_state_dict(ckpt.get("model", {}), strict=False)
        if geom_proj is not None and "geom_proj" in ckpt:
            geom_proj.load_state_dict(ckpt["geom_proj"], strict=False)
        if style_proj is not None and "style_proj" in ckpt:
            style_proj.load_state_dict(ckpt["style_proj"], strict=False)
        if geom_head is not None and "geom_head" in ckpt:
            geom_head.load_state_dict(ckpt["geom_head"], strict=False)
        if node_head is not None and "node_head" in ckpt:
            node_head.load_state_dict(ckpt["node_head"], strict=False)
        if edge_head is not None and "edge_head" in ckpt:
            edge_head.load_state_dict(ckpt["edge_head"], strict=False)
        if ocr_adapter is not None and "ocr_adapter" in ckpt:
            ocr_adapter.load_state_dict(ckpt["ocr_adapter"], strict=False)
        if args.init_step >= 0:
            start_step = args.init_step
        else:
            start_step = int(ckpt.get("step", 0))

    params = [p for p in student.parameters() if p.requires_grad]
    if geom_proj is not None:
        params += list(geom_proj.parameters())
    if style_proj is not None:
        params += list(style_proj.parameters())
    if geom_head is not None:
        params += list(geom_head.parameters())
    if node_head is not None:
        params += list(node_head.parameters())
    if edge_head is not None:
        params += list(edge_head.parameters())
    if ocr_adapter is not None:
        params += list(ocr_adapter.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and student_device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _style_weight(step_idx: int) -> float:
        if not stage_cfg["style"]:
            return 0.0
        if args.max_steps <= 0:
            return max(1.0, args.style_weight_min)
        warmup = int(args.max_steps * args.style_warmup_frac)
        decay_start = int(args.max_steps * (1.0 - args.style_decay_frac))
        if step_idx < warmup:
            return args.style_weight_min
        if step_idx < decay_start:
            span = max(1, decay_start - warmup)
            return max(args.style_weight_min, float(step_idx - warmup) / float(span))
        span = max(1, args.max_steps - decay_start)
        t = float(step_idx - decay_start) / float(span)
        return max(args.style_weight_min, 1.0 - (1.0 - args.style_min_ratio) * t)

    def _feat_weight(step_idx: int) -> float:
        if not stage_cfg["struct"]:
            return 0.0
        warm = max(0, int(args.feat_warmup_steps))
        ramp = max(1, int(args.feat_ramp_steps))
        if step_idx < warm:
            return args.feat_weight_min
        return max(args.feat_weight_min, min(1.0, float(step_idx - warm) / float(ramp)))

    def _set_train_mode(is_train: bool) -> None:
        student.train(is_train)
        for mod in (geom_proj, style_proj, geom_head, node_head, edge_head):
            if mod is not None:
                mod.train(is_train)

    def _compute_losses(
        images: List[Image.Image],
        structs: List[Optional[Dict]],
        sam2_batch: List[Optional[List[Dict]]],
        st_grids: List[torch.Tensor],
        st_grids_raw: Optional[List[torch.Tensor]],
        st_shapes: List[Tuple[int, int]],
        st_pooled: torch.Tensor,
        st_pooled_raw: Optional[torch.Tensor],
        rex_grids: Optional[List[torch.Tensor]],
        geom_embed: Optional[torch.Tensor],
        dino_cls: Optional[torch.Tensor],
        step_idx: int,
        collect_metrics: bool = False,
    ) -> Dict[str, torch.Tensor | float]:
        loss_struct = torch.tensor(0.0, device=student_device)
        loss_geom = torch.tensor(0.0, device=student_device)
        loss_graph = torch.tensor(0.0, device=student_device)
        loss_style = torch.tensor(0.0, device=student_device)
        loss_ocr = torch.tensor(0.0, device=student_device)
        struct_weight = 0.0
        geom_weight = 0.0
        node_loss_sum = torch.tensor(0.0, device=student_device)
        role_loss_sum = torch.tensor(0.0, device=student_device)
        edge_loss_sum = torch.tensor(0.0, device=student_device)
        node_count = 0.0
        edge_count = 0.0
        node_total_raw = 0.0
        edge_total_raw = 0.0
        struct_samples = 0.0
        geom_samples = 0.0
        node_samples = 0.0
        edge_samples = 0.0
        ocr_samples = 0.0
        node_correct = 0.0
        role_correct = 0.0
        edge_correct = 0.0
        edge_tp = 0.0
        edge_fp = 0.0
        edge_fn = 0.0
        edge_pos_correct = 0.0
        edge_pos_total = 0.0
        bsz = len(images)
        ocr_texts: List[str] = []
        ocr_indices: List[int] = []
        ocr_weights: List[float] = []
        ocr_roi_feats: List[torch.Tensor] = []
        ocr_roi_texts: List[str] = []
        ocr_roi_weights: List[float] = []
        for i in range(bsz):
            st_grid = _safe_feat(st_grids[i].to(student_device))
            st_grid_raw = None
            if st_grids_raw is not None:
                st_grid_raw = _safe_feat(st_grids_raw[i].to(student_device))
            h_s, w_s = st_shapes[i]
            st_feat = st_grid.permute(2, 0, 1).unsqueeze(0)
            struct_rec = structs[i] or {}
            quality = struct_rec.get("quality_flags") or {}
            mask_stats = struct_rec.get("mask_stats") or {}
            ocr_stats = struct_rec.get("ocr_stats") or {}

            valid_res = bool(quality.get("valid_resolution", True))
            has_ocr = quality.get("has_ocr")
            if has_ocr is None:
                has_ocr = bool(ocr_stats.get("num_boxes", 0) > 0)
            has_mask = quality.get("has_mask")
            if has_mask is None:
                has_mask = bool(mask_stats.get("num_masks", 0) > 0) or bool(sam2_batch[i])
            has_det = bool(quality.get("has_det", False))
            has_any = quality.get("has_any_structure")
            if has_any is None:
                has_any = bool(has_ocr or has_mask or has_det)
            # Treat explicit node/edge labels as structural signal even if quality flags omit it.
            if struct_rec.get("nodes") or struct_rec.get("edges"):
                has_any = True

            w_res = 1.0 if valid_res else float(args.res_weight_min)
            use_struct = bool(has_any)
            text_density = quality.get("text_density")
            if text_density is None:
                text_density = ocr_stats.get("text_density", 0.0)
            if has_ocr:
                w_ocr = min(1.0, max(args.ocr_weight_min, float(text_density) / max(1e-6, args.ocr_density_ref)))
            else:
                w_ocr = 1.0

            if args.lambda_ocr > 0 and ocr_tokenizer is not None:
                ocr_list = struct_rec.get("ocr") or []
                if args.ocr_mode == "roi":
                    if ocr_list:
                        width = struct_rec.get("width")
                        height = struct_rec.get("height")
                        if not width or not height:
                            try:
                                width, height = images[i].size
                            except Exception:
                                width = None
                                height = None
                        if width and height:
                            ocr_boxes: List[List[float]] = []
                            ocr_texts_img: List[str] = []
                            for item in ocr_list:
                                if not isinstance(item, dict):
                                    continue
                                txt = str(item.get("text") or "").strip()
                                if len(txt) < args.ocr_min_chars:
                                    continue
                                bbox = item.get("bbox") or item.get("box") or item.get("rect")
                                if not bbox or len(bbox) != 4:
                                    continue
                                x1, y1, x2, y2 = [float(v) for v in bbox]
                                if x2 < x1:
                                    x1, x2 = x2, x1
                                if y2 < y1:
                                    y1, y2 = y2, y1
                                x1n = max(0.0, min(1.0, x1 / float(width)))
                                y1n = max(0.0, min(1.0, y1 / float(height)))
                                x2n = max(0.0, min(1.0, x2 / float(width)))
                                y2n = max(0.0, min(1.0, y2 / float(height)))
                                if x2n <= x1n or y2n <= y1n:
                                    continue
                                ocr_boxes.append([x1n, y1n, x2n, y2n])
                                ocr_texts_img.append(txt)
                                if len(ocr_boxes) >= args.ocr_max_items:
                                    break
                            if ocr_boxes:
                                grid_for_ocr = st_grid_raw if st_grid_raw is not None else st_grid
                                feats = pool_roi_features(grid_for_ocr, ocr_boxes, h_s, w_s).to(student_device)
                                ocr_roi_feats.append(feats)
                                ocr_roi_texts.extend(ocr_texts_img)
                                ocr_roi_weights.extend([float(w_ocr) * float(w_res)] * len(ocr_boxes))
                                ocr_samples += float(len(ocr_boxes))
                else:
                    ocr_text = _collect_ocr_text(ocr_list, args.ocr_max_items, args.ocr_min_chars)
                    if ocr_text:
                        ocr_texts.append(ocr_text)
                        ocr_indices.append(i)
                        ocr_weights.append(float(w_ocr) * float(w_res))
                        ocr_samples += 1.0

            if rex_grids is not None and use_struct:
                rx_grid = _safe_feat(rex_grids[i].to(student_device))
                rx_feat = rx_grid.permute(2, 0, 1).unsqueeze(0)
                rx_feat = F.interpolate(rx_feat, size=(h_s, w_s), mode="bilinear", align_corners=False)
                loss_struct = loss_struct + w_res * F.mse_loss(normalize_feat(st_feat), normalize_feat(rx_feat))
                struct_weight += w_res
                if w_res > 0:
                    struct_samples += 1.0

            if geom_mode == "sam2_feat" and geom_embed is not None and geom_proj is not None and has_mask:
                gm_feat = _safe_feat(geom_embed[i].to(student_device).unsqueeze(0))
                gm_feat = geom_proj(gm_feat)
                gm_feat = F.interpolate(gm_feat, size=(h_s, w_s), mode="bilinear", align_corners=False)
                loss_geom = loss_geom + w_res * F.mse_loss(normalize_feat(st_feat), normalize_feat(gm_feat))
                geom_weight += w_res
                if w_res > 0:
                    geom_samples += 1.0
            elif geom_mode == "sam2_mask" and geom_head is not None and has_mask:
                mask = _sam2_union_mask(sam2_batch[i] or [], max_masks=args.geom_mask_topk)
                if mask is not None:
                    mask_t = torch.from_numpy(mask).to(student_device).float()
                    if mask_t.sum() > 0:
                        mask_t = mask_t.unsqueeze(0).unsqueeze(0)
                        mask_small = F.interpolate(mask_t, size=(h_s, w_s), mode="nearest")
                        logits = geom_head(st_feat)
                        pos = mask_small.sum()
                        neg = mask_small.numel() - pos
                        pos_weight = (neg / (pos + 1e-6)).clamp(max=50.0)
                        mask_ratio = float(mask_small.mean().item())
                        w_mask = min(1.0, max(args.mask_weight_min, mask_ratio / max(1e-6, args.mask_area_ref)))
                        loss_geom = loss_geom + w_res * w_mask * F.binary_cross_entropy_with_logits(
                            logits, mask_small, pos_weight=pos_weight
                        )
                        geom_weight += w_res
                        if w_res > 0:
                            geom_samples += 1.0

            if node_head is not None and edge_head is not None and use_struct:
                if struct_rec:
                    bboxes, node_types, node_roles, edges = _prepare_struct_rec(
                        struct_rec, edge_score_thresh=args.struct_edge_thresh
                    )
                    if bboxes:
                        bboxes_t = torch.tensor(bboxes, device=student_device)
                        feats = pool_roi_features(st_grid, bboxes, h_s, w_s).to(student_device)
                        type_logits, role_logits = node_head(feats)
                        node_labels = torch.tensor(node_types, device=student_device)
                        role_labels = torch.tensor(node_roles, device=student_device)
                        l_node = F.cross_entropy(type_logits, node_labels, reduction="sum")
                        l_role = F.cross_entropy(role_logits, role_labels, reduction="sum")
                        l_edge = torch.tensor(0.0, device=student_device)
                        weight = w_ocr * w_res
                        node_loss_sum = node_loss_sum + weight * l_node
                        role_loss_sum = role_loss_sum + weight * l_role
                        node_count += weight * float(len(bboxes))
                        node_total_raw += float(len(bboxes))
                        node_samples += 1.0
                        if collect_metrics:
                            node_pred = type_logits.argmax(dim=-1)
                            role_pred = role_logits.argmax(dim=-1)
                            node_correct += float((node_pred == node_labels).sum().item())
                            role_correct += float((role_pred == role_labels).sum().item())
                        if edges and len(bboxes) > 1:
                            edge_samples += 1.0
                            rng = random.Random(args.seed + step_idx * 1000 + i)
                            pairs, labels = _sample_edge_pairs(
                                len(bboxes),
                                edges,
                                node_types,
                                args.edge_neg_ratio,
                                args.edge_same_src_neg,
                                not args.no_edge_rev_neg,
                                rng,
                            )
                            if pairs:
                                pair_t = torch.tensor(pairs, device=student_device, dtype=torch.long)
                                label_t = torch.tensor(labels, device=student_device)
                                geom_feat = _edge_geom_features(bboxes_t, pair_t)
                                f_src = feats[pair_t[:, 0]]
                                f_dst = feats[pair_t[:, 1]]
                                edge_feat = torch.cat(
                                    [f_src, f_dst, f_src * f_dst, (f_src - f_dst).abs(), geom_feat],
                                    dim=-1,
                                )
                                rel_logits = edge_head(edge_feat)
                                l_edge = F.cross_entropy(rel_logits, label_t, reduction="sum")
                                edge_loss_sum = edge_loss_sum + weight * l_edge
                                edge_count += weight * float(len(labels))
                                edge_total_raw += float(len(labels))
                                if collect_metrics:
                                    pred_rel = rel_logits.argmax(dim=-1)
                                    edge_correct += float((pred_rel == label_t).sum().item())
                                    pos_true = label_t != 0
                                    pos_pred = pred_rel != 0
                                    edge_tp += float((pos_true & pos_pred).sum().item())
                                    edge_fp += float((~pos_true & pos_pred).sum().item())
                                    edge_fn += float((pos_true & ~pos_pred).sum().item())
                                    edge_pos_correct += float(((pred_rel == label_t) & pos_true).sum().item())
                                    edge_pos_total += float(pos_true.sum().item())

        if struct_weight > 0:
            loss_struct = loss_struct / struct_weight
        if geom_weight > 0:
            loss_geom = loss_geom / geom_weight
        if node_count > 0:
            loss_node = node_loss_sum / max(node_count, 1.0)
            loss_role = role_loss_sum / max(node_count, 1.0)
        else:
            loss_node = torch.tensor(0.0, device=student_device)
            loss_role = torch.tensor(0.0, device=student_device)
        if edge_count > 0:
            loss_edge = edge_loss_sum / max(edge_count, 1.0)
        else:
            loss_edge = torch.tensor(0.0, device=student_device)
        loss_graph = args.lambda_node * loss_node + args.lambda_role * loss_role + args.lambda_edge * loss_edge

        if args.lambda_ocr > 0 and ocr_tokenizer is not None:
            if args.ocr_mode == "roi":
                if len(ocr_roi_texts) >= 2 and ocr_roi_feats:
                    max_len = args.ocr_max_length
                    text_cfg = getattr(getattr(student.siglip, "text_model", None), "config", None)
                    if text_cfg is None:
                        text_cfg = getattr(getattr(student.siglip, "config", None), "text_config", None)
                    max_pos = getattr(text_cfg, "max_position_embeddings", None)
                    if isinstance(max_pos, int) and max_pos > 0:
                        max_len = min(max_len, max_pos)
                    tok_max = getattr(ocr_tokenizer, "model_max_length", None)
                    if isinstance(tok_max, int) and 0 < tok_max < 10**6:
                        max_len = min(max_len, tok_max)
                    ocr_inputs = ocr_tokenizer(
                        ocr_roi_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt",
                    )
                    ocr_inputs = {k: v.to(student_device) for k, v in ocr_inputs.items()}
                    with torch.no_grad():
                        txt_feat = student.siglip.get_text_features(**ocr_inputs)
                    img_feat = torch.cat(ocr_roi_feats, dim=0).to(student_device)
                    if args.ocr_detach:
                        img_feat = img_feat.detach()
                    if ocr_adapter is not None:
                        img_feat = ocr_adapter(img_feat.float())
                    img_feat = F.normalize(img_feat.float(), dim=-1)
                    txt_feat = F.normalize(txt_feat.float(), dim=-1)
                    logits = img_feat @ txt_feat.t()
                    if args.ocr_logit_scale > 0:
                        logits = logits * float(args.ocr_logit_scale)
                    elif hasattr(student.siglip, "logit_scale"):
                        logits = logits * student.siglip.logit_scale.exp().clamp(max=100.0)
                    labels = torch.arange(len(ocr_roi_texts), device=student_device)
                    loss_i = F.cross_entropy(logits, labels)
                    loss_t = F.cross_entropy(logits.t(), labels)
                    loss_ocr = 0.5 * (loss_i + loss_t)
                    if ocr_roi_weights:
                        loss_ocr = loss_ocr * (sum(ocr_roi_weights) / max(1.0, float(len(ocr_roi_weights))))
            elif st_pooled_raw is not None and len(ocr_indices) >= 2:
                max_len = args.ocr_max_length
                text_cfg = getattr(getattr(student.siglip, "text_model", None), "config", None)
                if text_cfg is None:
                    text_cfg = getattr(getattr(student.siglip, "config", None), "text_config", None)
                max_pos = getattr(text_cfg, "max_position_embeddings", None)
                if isinstance(max_pos, int) and max_pos > 0:
                    max_len = min(max_len, max_pos)
                tok_max = getattr(ocr_tokenizer, "model_max_length", None)
                if isinstance(tok_max, int) and 0 < tok_max < 10**6:
                    max_len = min(max_len, tok_max)
                ocr_inputs = ocr_tokenizer(
                    ocr_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                ocr_inputs = {k: v.to(student_device) for k, v in ocr_inputs.items()}
                with torch.no_grad():
                    txt_feat = student.siglip.get_text_features(**ocr_inputs)
                img_feat = st_pooled_raw[ocr_indices].to(student_device)
                if args.ocr_detach:
                    img_feat = img_feat.detach()
                if ocr_adapter is not None:
                    img_feat = ocr_adapter(img_feat.float())
                img_feat = F.normalize(img_feat.float(), dim=-1)
                txt_feat = F.normalize(txt_feat.float(), dim=-1)
                logits = img_feat @ txt_feat.t()
                if args.ocr_logit_scale > 0:
                    logits = logits * float(args.ocr_logit_scale)
                elif hasattr(student.siglip, "logit_scale"):
                    logits = logits * student.siglip.logit_scale.exp().clamp(max=100.0)
                labels = torch.arange(len(ocr_indices), device=student_device)
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.t(), labels)
                loss_ocr = 0.5 * (loss_i + loss_t)
                if ocr_weights:
                    loss_ocr = loss_ocr * (sum(ocr_weights) / max(1.0, float(len(ocr_weights))))

        if style_teacher is not None and style_proj is not None and dino_cls is not None:
            dino_cls = _safe_feat(dino_cls.to(student_device))
            style_tgt = style_proj(dino_cls)
            st_pooled_safe = _safe_feat(st_pooled)
            style_tgt = _safe_feat(style_tgt)
            loss_style = 1.0 - F.cosine_similarity(st_pooled_safe.float(), style_tgt.float(), dim=-1).mean()

        style_w = _style_weight(step_idx)
        feat_w = _feat_weight(step_idx)
        ocr_w = 1.0
        if args.ocr_warmup_steps > 0:
            if step_idx < args.ocr_warmup_steps:
                ocr_w = 0.0
            elif args.ocr_ramp_steps > 0:
                ocr_w = min(1.0, float(step_idx - args.ocr_warmup_steps) / float(args.ocr_ramp_steps))
        loss = (
            args.lambda_struct * feat_w * loss_struct
            + args.lambda_graph * loss_graph
            + args.lambda_geom * loss_geom
            + args.lambda_style * style_w * loss_style
            + args.lambda_ocr * ocr_w * loss_ocr
        )

        return {
            "loss": loss,
            "loss_struct": loss_struct,
            "loss_graph": loss_graph,
            "loss_geom": loss_geom,
            "loss_style": loss_style,
            "loss_ocr": loss_ocr,
            "ocr_w": ocr_w,
            "style_w": style_w,
            "feat_w": feat_w,
            "struct_weight": struct_weight,
            "geom_weight": geom_weight,
            "node_count": node_count,
            "edge_count": edge_count,
            "node_correct": node_correct,
            "role_correct": role_correct,
            "edge_correct": edge_correct,
            "edge_tp": edge_tp,
            "edge_fp": edge_fp,
            "edge_fn": edge_fn,
            "edge_pos_correct": edge_pos_correct,
            "edge_pos_total": edge_pos_total,
            "node_total_raw": node_total_raw,
            "edge_total_raw": edge_total_raw,
            "struct_samples": struct_samples,
            "geom_samples": geom_samples,
            "node_samples": node_samples,
            "edge_samples": edge_samples,
            "ocr_samples": ocr_samples,
        }

    def _make_val_loader(
        geom_cache: str,
        struct_cache: str,
        struct_map: Dict[str, Dict],
    ) -> Optional[DataLoader]:
        source = None
        if geom_mode == "sam2_mask" and geom_cache and os.path.exists(geom_cache):
            require_struct = bool(struct_cache and struct_map)
            source = GeomCacheStream(
                geom_cache,
                struct_map=struct_map,
                require_struct=require_struct,
                drop_empty=False,
                repeat=False,
                max_items=val_max_items,
            )
        elif struct_cache and os.path.exists(struct_cache):
            source = StructCacheStream(struct_cache, repeat=False, max_items=val_max_items)
        elif geom_cache and os.path.exists(geom_cache):
            source = GeomCacheStream(
                geom_cache,
                struct_map=struct_map,
                require_struct=False,
                drop_empty=True,
                repeat=False,
                max_items=val_max_items,
            )
        if source is None:
            return None
        dataset = SingleSourceImageDataset(source)
        return DataLoader(
            dataset,
            batch_size=val_batch_size,
            num_workers=val_num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    val_log_path = Path(args.val_log) if args.val_log else (out_dir / "val_metrics.jsonl")

    def _run_validation(step_idx: int, force: bool = False) -> None:
        if args.val_every <= 0 and not force:
            return
        val_sets = [
            ("scistruct_val", args.scistruct_val_geom_cache, args.scistruct_val_struct_cache, val_struct_map_scistruct),
            ("scicap_val", args.scicap_val_geom_cache, args.scicap_val_struct_cache, val_struct_map_scicap),
        ]
        if args.scistruct_test_struct_cache or args.scistruct_test_geom_cache:
            val_sets.append((
                "scistruct_test",
                args.scistruct_test_geom_cache,
                args.scistruct_test_struct_cache,
                test_struct_map_scistruct,
            ))
        if args.scicap_test_struct_cache or args.scicap_test_geom_cache:
            val_sets.append((
                "scicap_test",
                args.scicap_test_geom_cache,
                args.scicap_test_struct_cache,
                test_struct_map_scicap,
            ))
        _set_train_mode(False)
        with torch.no_grad():
            for name, geom_cache, struct_cache, struct_map in val_sets:
                loader = _make_val_loader(geom_cache, struct_cache, struct_map)
                if loader is None:
                    continue
                totals = {
                    "loss": 0.0,
                    "struct": 0.0,
                    "graph": 0.0,
                    "geom": 0.0,
                    "style": 0.0,
                    "ocr": 0.0,
                    "node_correct": 0.0,
                    "role_correct": 0.0,
                    "node_total_raw": 0.0,
                    "edge_correct": 0.0,
                    "edge_total_raw": 0.0,
                    "edge_tp": 0.0,
                    "edge_fp": 0.0,
                    "edge_fn": 0.0,
                    "edge_pos_correct": 0.0,
                    "edge_pos_total": 0.0,
                    "struct_samples": 0.0,
                    "geom_samples": 0.0,
                    "node_samples": 0.0,
                    "edge_samples": 0.0,
                }
                count = 0
                style_w = 0.0
                feat_w = 0.0
                for images, _, structs, sam2_batch in loader:
                    batch = student.preprocess(images, image_size=args.image_size)
                    batch = {k: v.to(student_device) for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=args.amp and student_device.type == "cuda"):
                        student_out = student(batch)
                        st_tokens = student_out["tokens"]
                        st_pooled = student_out["pooled"]
                        st_pooled_raw = student_out.get("pooled_raw")
                        st_tokens_raw = student_out.get("tokens_raw")
                    st_grids, st_shapes = tokens_to_grid(st_tokens)
                    st_grids_raw = None
                    if st_tokens_raw is not None:
                        st_grids_raw, _ = tokens_to_grid(st_tokens_raw)
                    ocr_grids_raw = st_grids_raw
                    ocr_pooled_raw = st_pooled_raw
                    if ocr_encoder is not None:
                        ocr_batch = ocr_encoder.preprocess(images, image_size=args.image_size)
                        ocr_batch = {k: v.to(student_device) for k, v in ocr_batch.items()}
                        with torch.cuda.amp.autocast(enabled=args.amp and student_device.type == "cuda"):
                            ocr_out = ocr_encoder(ocr_batch)
                        ocr_tokens_raw = ocr_out.get("tokens_raw")
                        if ocr_tokens_raw is None:
                            ocr_tokens_raw = ocr_out.get("tokens")
                        ocr_pooled_raw = ocr_out.get("pooled_raw")
                        if ocr_pooled_raw is None:
                            ocr_pooled_raw = ocr_out.get("pooled")
                        if ocr_tokens_raw is not None:
                            ocr_grids_raw, _ = tokens_to_grid(ocr_tokens_raw)

                    rex_grids = None
                    rex_grid_thw = None
                    geom_embed = None
                    dino_cls = None
                    if rex_teacher is not None:
                        rex_tokens, rex_grid_thw = rex_teacher(images)
                        rex_grids, _ = tokens_to_grid(rex_tokens, rex_grid_thw)
                    if geom_teacher is not None:
                        geom_embed = geom_teacher(images)
                    if style_teacher is not None:
                        dino_cls = style_teacher(images)

                    losses = _compute_losses(
                        images,
                        structs,
                        sam2_batch,
                        st_grids,
                        ocr_grids_raw,
                        st_shapes,
                        st_pooled,
                        ocr_pooled_raw,
                        rex_grids,
                        geom_embed,
                        dino_cls,
                        step_idx,
                        collect_metrics=True,
                    )
                    bsz = len(images)
                    count += bsz
                    totals["loss"] += float(losses["loss"].item()) * bsz
                    totals["struct"] += float(losses["loss_struct"].item()) * bsz
                    totals["graph"] += float(losses["loss_graph"].item()) * bsz
                    totals["geom"] += float(losses["loss_geom"].item()) * bsz
                    totals["style"] += float(losses["loss_style"].item()) * bsz
                    totals["ocr"] += float(losses["loss_ocr"].item()) * bsz
                    totals["node_correct"] += float(losses.get("node_correct", 0.0))
                    totals["role_correct"] += float(losses.get("role_correct", 0.0))
                    totals["node_total_raw"] += float(losses.get("node_total_raw", 0.0))
                    totals["edge_correct"] += float(losses.get("edge_correct", 0.0))
                    totals["edge_total_raw"] += float(losses.get("edge_total_raw", 0.0))
                    totals["edge_tp"] += float(losses.get("edge_tp", 0.0))
                    totals["edge_fp"] += float(losses.get("edge_fp", 0.0))
                    totals["edge_fn"] += float(losses.get("edge_fn", 0.0))
                    totals["edge_pos_correct"] += float(losses.get("edge_pos_correct", 0.0))
                    totals["edge_pos_total"] += float(losses.get("edge_pos_total", 0.0))
                    totals["struct_samples"] += float(losses.get("struct_samples", 0.0))
                    totals["geom_samples"] += float(losses.get("geom_samples", 0.0))
                    totals["node_samples"] += float(losses.get("node_samples", 0.0))
                    totals["edge_samples"] += float(losses.get("edge_samples", 0.0))
                    style_w = float(losses["style_w"])
                    feat_w = float(losses["feat_w"])

                if count <= 0:
                    continue
                metrics = {
                    "step": step_idx,
                    "split": name,
                    "count": count,
                    "loss": totals["loss"] / count,
                    "struct": totals["struct"] / count,
                    "graph": totals["graph"] / count,
                    "geom": totals["geom"] / count,
                    "style": totals["style"] / count,
                    "ocr": totals["ocr"] / count,
                    "style_w": style_w,
                    "feat_w": feat_w,
                    "node_acc": totals["node_correct"] / max(1.0, totals["node_total_raw"]),
                    "role_acc": totals["role_correct"] / max(1.0, totals["node_total_raw"]),
                    "edge_acc": totals["edge_correct"] / max(1.0, totals["edge_total_raw"]),
                    "struct_ratio": totals["struct_samples"] / max(1.0, count),
                    "geom_ratio": totals["geom_samples"] / max(1.0, count),
                    "node_ratio": totals["node_samples"] / max(1.0, count),
                    "edge_ratio": totals["edge_samples"] / max(1.0, count),
                }
                edge_prec = totals["edge_tp"] / max(1.0, (totals["edge_tp"] + totals["edge_fp"]))
                edge_rec = totals["edge_tp"] / max(1.0, (totals["edge_tp"] + totals["edge_fn"]))
                if edge_prec + edge_rec > 0:
                    edge_f1 = 2.0 * edge_prec * edge_rec / (edge_prec + edge_rec)
                else:
                    edge_f1 = 0.0
                metrics["edge_f1"] = edge_f1
                metrics["edge_pos_acc"] = totals["edge_pos_correct"] / max(1.0, totals["edge_pos_total"])
                print(
                    f"[val:{name}] step={step_idx} loss={metrics['loss']:.6f} "
                    f"struct={metrics['struct']:.6f} graph={metrics['graph']:.6f} "
                    f"geom={metrics['geom']:.6f} style={metrics['style']:.6f} "
                    f"ocr={metrics['ocr']:.6f} "
                    f"node_acc={metrics['node_acc']:.4f} edge_f1={metrics['edge_f1']:.4f} "
                    f"struct_ratio={metrics['struct_ratio']:.2f} edge_ratio={metrics['edge_ratio']:.2f}"
                )
                with val_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        _set_train_mode(True)

    if args.eval_only:
        _run_validation(start_step if start_step > 0 else 0, force=True)
        return

    step = start_step
    pbar = tqdm(loader, total=args.max_steps, desc="train", initial=step)
    for images, sources, structs, sam2_batch in pbar:
        if step >= args.max_steps:
            break
        optim.zero_grad(set_to_none=True)

        # Student forward (with grad)
        batch = student.preprocess(images, image_size=args.image_size)
        batch = {k: v.to(student_device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=args.amp and student_device.type == "cuda"):
            student_out = student(batch)
            st_tokens = student_out["tokens"]  # (B, N, 768)
            st_pooled = student_out["pooled"]  # (B, 768)
            st_pooled_raw = student_out.get("pooled_raw")
            st_tokens_raw = student_out.get("tokens_raw")

        st_grids, st_shapes = tokens_to_grid(st_tokens)
        st_grids_raw = None
        if st_tokens_raw is not None:
            st_grids_raw, _ = tokens_to_grid(st_tokens_raw)

        # Teacher features (no grad)
        rex_grids = None
        rex_grid_thw = None
        geom_embed = None
        dino_cls = None
        with torch.no_grad():
            if rex_teacher is not None:
                rex_tokens, rex_grid_thw = rex_teacher(images)
                rex_grids, _ = tokens_to_grid(rex_tokens, rex_grid_thw)
            if geom_teacher is not None:
                geom_embed = geom_teacher(images)  # (B, C, H, W)
            if style_teacher is not None:
                dino_cls = style_teacher(images)   # (B, 1024)

        losses = _compute_losses(
            images,
            structs,
            sam2_batch,
            st_grids,
            st_grids_raw,
            st_shapes,
            st_pooled,
            st_pooled_raw,
            rex_grids,
            geom_embed,
            dino_cls,
            step,
        )
        loss = losses["loss"]
        loss_struct = losses["loss_struct"]
        loss_graph = losses["loss_graph"]
        loss_geom = losses["loss_geom"]
        loss_style = losses["loss_style"]
        loss_ocr = losses["loss_ocr"]
        ocr_w = float(losses.get("ocr_w", 1.0))
        style_w = float(losses["style_w"])
        feat_w = float(losses["feat_w"])
        struct_count = float(losses["struct_weight"])
        graph_count = float(losses["node_count"]) + float(losses["edge_count"])
        geom_count = float(losses["geom_weight"])
        ocr_count = float(losses.get("ocr_samples", 0.0)) * ocr_w

        has_supervision = False
        if args.lambda_struct > 0 and struct_count > 0 and feat_w > 0:
            has_supervision = True
        if args.lambda_graph > 0 and graph_count > 0:
            has_supervision = True
        if args.lambda_geom > 0 and geom_count > 0:
            has_supervision = True
        if args.lambda_style > 0 and style_w > 0:
            has_supervision = True
        if args.lambda_ocr > 0 and ocr_count > 0:
            has_supervision = True
        if not has_supervision:
            if step % args.log_every == 0:
                print(
                    f"[warn] step {step}: no active supervision "
                    f"(struct={struct_count}, graph={graph_count}, geom={geom_count}, style_w={style_w}, "
                    f"feat_w={feat_w}, ocr={ocr_count})",
                    file=sys.stderr,
                )
            step += 1
            continue

        if not loss.requires_grad:
            if step % args.log_every == 0:
                print(
                    f"[warn] step {step}: loss has no grad "
                    f"(struct_count={struct_count}, graph_count={graph_count}, geom_count={geom_count})",
                    file=sys.stderr,
                )
            step += 1
            continue

        if not torch.isfinite(loss):
            print(
                f"[warn] step {step}: non-finite loss (struct={loss_struct}, graph={loss_graph}, geom={loss_geom}, style={loss_style}, ocr={loss_ocr})",
                file=sys.stderr,
            )
            step += 1
            continue

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(optim)
        scaler.update()

        if step % args.log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "struct": f"{loss_struct.item():.6f}",
                "graph": f"{loss_graph.item():.6f}",
                "geom": f"{loss_geom.item():.6f}",
                "style": f"{loss_style.item():.6f}",
                "ocr": f"{loss_ocr.item():.6f}",
                "ocr_w": f"{ocr_w:.2f}",
                "style_w": f"{style_w:.2f}",
                "feat_w": f"{feat_w:.2f}",
            })

        if step > 0 and step % args.save_every == 0:
            ckpt = {
                "step": step,
                "model": trainable_state_dict(student),
                "args": vars(args),
            }
            if geom_proj is not None:
                ckpt["geom_proj"] = geom_proj.state_dict()
            if style_proj is not None:
                ckpt["style_proj"] = style_proj.state_dict()
            if geom_head is not None:
                ckpt["geom_head"] = geom_head.state_dict()
            if node_head is not None:
                ckpt["node_head"] = node_head.state_dict()
            if edge_head is not None:
                ckpt["edge_head"] = edge_head.state_dict()
            if ocr_adapter is not None:
                ckpt["ocr_adapter"] = ocr_adapter.state_dict()
            torch.save(ckpt, out_dir / f"ckpt_step_{step}.pt")

        if step > 0 and step % args.val_every == 0:
            _run_validation(step)

        step += 1

    ckpt = {
        "step": step,
        "model": trainable_state_dict(student),
        "args": vars(args),
    }
    if geom_proj is not None:
        ckpt["geom_proj"] = geom_proj.state_dict()
    if style_proj is not None:
        ckpt["style_proj"] = style_proj.state_dict()
    if geom_head is not None:
        ckpt["geom_head"] = geom_head.state_dict()
    if node_head is not None:
        ckpt["node_head"] = node_head.state_dict()
    if edge_head is not None:
        ckpt["edge_head"] = edge_head.state_dict()
    if ocr_adapter is not None:
        ckpt["ocr_adapter"] = ocr_adapter.state_dict()
    torch.save(ckpt, out_dir / "ckpt_last.pt")


if __name__ == "__main__":
    main()

