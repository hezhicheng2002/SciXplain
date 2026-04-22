#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from scixplain.models.towers import RexOmniWrapper
from scixplain.models.lora import apply_lora


NODE_TYPES = ["text", "blob", "arrow", "arrowHead", "image"]
ROLE_TYPES = ["other", "label", "caption", "title"]
REL_TYPES = [
    "none",
    "arrowDescriptor",
    "arrowHeadTail",
    "imageCaption",
    "imageTitle",
    "interObjectLinkage",
    "intraObjectLabel",
    "intraObjectLinkage",
    "intraObjectRegionLabel",
    "intraObjectTextLinkage",
    "misc",
    "sectionTitle",
]

NODE2IDX = {t: i for i, t in enumerate(NODE_TYPES)}
ROLE2IDX = {t: i for i, t in enumerate(ROLE_TYPES)}
REL2IDX = {t: i for i, t in enumerate(REL_TYPES)}


def _bbox_from_polygon(poly: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _bbox_from_rectangle(rect: List[List[float]]) -> Tuple[float, float, float, float]:
    x1, y1 = rect[0]
    x2, y2 = rect[1]
    return float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))


def _norm_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> List[float]:
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    nx1 = max(0.0, min(1.0, x1 / w))
    ny1 = max(0.0, min(1.0, y1 / h))
    nx2 = max(0.0, min(1.0, x2 / w))
    ny2 = max(0.0, min(1.0, y2 / h))
    return [min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)]


def _role_from_rel_categories(cats: List[str]) -> int:
    if any(c in ["imageTitle", "sectionTitle"] for c in cats):
        return ROLE2IDX["title"]
    if any(c in ["imageCaption"] for c in cats):
        return ROLE2IDX["caption"]
    if any(c in [
        "intraObjectLabel",
        "intraObjectRegionLabel",
        "intraObjectTextLinkage",
        "intraObjectLinkage",
        "interObjectLinkage",
        "arrowDescriptor",
    ] for c in cats):
        return ROLE2IDX["label"]
    return ROLE2IDX["other"]


class AI2DStructureDataset(Dataset):
    def __init__(self, ai2d_root: str, split_dir: str, split: str):
        self.ai2d_root = Path(ai2d_root)
        self.split_dir = Path(split_dir)
        split_ids = self.split_dir / f"{split}_ids.txt"
        ids = [line.strip() for line in split_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        items = []
        for rid in ids:
            img = self.ai2d_root / "images" / f"{rid}.png"
            ann = self.ai2d_root / "annotations" / f"{rid}.png.json"
            if img.exists() and ann.exists():
                items.append((rid, img, ann))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rid, img_path, ann_path = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))

        nodes = []
        node_id_to_idx: Dict[str, int] = {}

        # Image const node(s)
        for iid in (ann.get("imageConsts") or {}).keys():
            bbox = [0.0, 0.0, 1.0, 1.0]
            node_id_to_idx[iid] = len(nodes)
            nodes.append({
                "id": iid,
                "type": NODE2IDX["image"],
                "bbox": bbox,
                "role": -1,
                "value": "",
            })

        # Text nodes
        for tid, obj in (ann.get("text") or {}).items():
            rect = obj.get("rectangle")
            if not rect:
                continue
            x1, y1, x2, y2 = _bbox_from_rectangle(rect)
            bbox = _norm_bbox(x1, y1, x2, y2, w, h)
            node_id_to_idx[tid] = len(nodes)
            nodes.append({
                "id": tid,
                "type": NODE2IDX["text"],
                "bbox": bbox,
                "role": -1,  # filled later
                "value": str(obj.get("value") or ""),
            })

        # Blob nodes
        for bid, obj in (ann.get("blobs") or {}).items():
            poly = obj.get("polygon")
            if not poly:
                continue
            x1, y1, x2, y2 = _bbox_from_polygon(poly)
            bbox = _norm_bbox(x1, y1, x2, y2, w, h)
            node_id_to_idx[bid] = len(nodes)
            nodes.append({
                "id": bid,
                "type": NODE2IDX["blob"],
                "bbox": bbox,
                "role": -1,
                "value": "",
            })

        # Arrow nodes
        for aid, obj in (ann.get("arrows") or {}).items():
            poly = obj.get("polygon")
            if not poly:
                continue
            x1, y1, x2, y2 = _bbox_from_polygon(poly)
            bbox = _norm_bbox(x1, y1, x2, y2, w, h)
            node_id_to_idx[aid] = len(nodes)
            nodes.append({
                "id": aid,
                "type": NODE2IDX["arrow"],
                "bbox": bbox,
                "role": -1,
                "value": "",
            })

        # ArrowHead nodes
        for hid, obj in (ann.get("arrowHeads") or {}).items():
            rect = obj.get("rectangle")
            if not rect:
                continue
            x1, y1, x2, y2 = _bbox_from_rectangle(rect)
            bbox = _norm_bbox(x1, y1, x2, y2, w, h)
            node_id_to_idx[hid] = len(nodes)
            nodes.append({
                "id": hid,
                "type": NODE2IDX["arrowHead"],
                "bbox": bbox,
                "role": -1,
                "value": "",
            })

        # Relationships / edges
        edges = []
        text_rel_cats: Dict[str, List[str]] = {}
        for rel in (ann.get("relationships") or {}).values():
            cat = rel.get("category") or "misc"
            if cat not in REL2IDX:
                cat = "misc"
            origin = rel.get("origin")
            dest = rel.get("destination")
            if origin is None or dest is None:
                continue
            if origin not in node_id_to_idx or dest not in node_id_to_idx:
                continue
            o_idx = node_id_to_idx[origin]
            d_idx = node_id_to_idx[dest]
            edges.append((o_idx, d_idx, REL2IDX[cat]))
            if origin.startswith("T"):
                text_rel_cats.setdefault(origin, []).append(cat)

        # Assign roles for text nodes
        for node in nodes:
            if node["type"] != NODE2IDX["text"]:
                continue
            cats = text_rel_cats.get(node["id"], [])
            node["role"] = _role_from_rel_categories(cats)

        return {
            "image": image,
            "nodes": nodes,
            "edges": edges,
        }


def collate_fn(batch):
    images = [b["image"] for b in batch]
    nodes = [b["nodes"] for b in batch]
    edges = [b["edges"] for b in batch]
    return images, nodes, edges


def tokens_to_grid(tokens, grid_thw=None):
    # tokens: (B, N, C) or list of (N, C)
    if isinstance(tokens, list):
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
            # fold temporal into height if present
            h = h * max(1, t)
            expected = h * w
            if n == expected + 1:
                tok = tok[1:, :]
                n = tok.shape[0]
            if n != expected:
                ratio = (h * w) / float(n)
                scale = int(round(math.sqrt(ratio)))
                if scale > 1 and (h // scale) * (w // scale) == n:
                    h = h // scale
                    w = w // scale
                else:
                    raise ValueError(f"Unexpected token count {n} vs grid {h}x{w}")
            grid = tok.reshape(h, w, c)
            grids.append(grid)
            shapes.append((h, w))
            continue

        grid = int(math.sqrt(n - 1))
        if grid * grid + 1 == n:
            tok = tok[1:, :]
        else:
            grid = int(math.sqrt(n))
            if grid * grid != n:
                raise ValueError(f"Unexpected token count {n}; cannot form grid")
        grids.append(tok.reshape(grid, grid, c))
        shapes.append((grid, grid))

    return grids, shapes


def pool_roi_features(tokens_grid: torch.Tensor, bboxes: List[List[float]], grid_h: int, grid_w: int) -> torch.Tensor:
    feats = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        gx1 = max(0, min(grid_w - 1, int(math.floor(x1 * grid_w))))
        gy1 = max(0, min(grid_h - 1, int(math.floor(y1 * grid_h))))
        gx2 = max(gx1 + 1, min(grid_w, int(math.ceil(x2 * grid_w))))
        gy2 = max(gy1 + 1, min(grid_h, int(math.ceil(y2 * grid_h))))
        patch = tokens_grid[gy1:gy2, gx1:gx2, :]
        feats.append(patch.mean(dim=(0, 1)))
    return torch.stack(feats, dim=0)


class AI2DTeacher(nn.Module):
    def __init__(self, use_lora: bool, lora_r: int, lora_alpha: float, lora_dropout: float, lora_filters: List[str]):
        super().__init__()
        self.rex = RexOmniWrapper()
        if use_lora:
            apply_lora(
                [("rex", self.rex.model)],
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                freeze_base=True,
                name_filter=lora_filters,
            )
        self.node_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, len(NODE_TYPES)),
        )
        self.box_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        self.role_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, len(ROLE_TYPES)),
        )
        self.rel_head = nn.Sequential(
            nn.Linear(768 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, len(REL_TYPES)),
        )

    def forward(self, images):
        batch = self.rex.preprocess(images)
        batch = {k: v.to(next(self.parameters()).device) for k, v in batch.items()}
        out = self.rex(batch)
        return out["tokens"], out.get("grid_thw")


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    return {k: v for k, v in sd.items() if k in trainable}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai2d_root", type=str, default="dataset/AI2D-Official")
    ap.add_argument("--split_dir", type=str, default="dataset/AI2D-Official/splits_91")
    ap.add_argument("--out_dir", type=str, default="checkpoints/ai2d_teacher")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="negative edge samples per positive edge")
    ap.add_argument("--same_src_neg", type=int, default=1, help="extra negatives per positive from same source node")
    ap.add_argument("--no_rev_neg", action="store_true", help="disable reverse-edge hard negatives")
    ap.add_argument("--lambda_node", type=float, default=1.0)
    ap.add_argument("--lambda_box", type=float, default=1.0)
    ap.add_argument("--lambda_rel", type=float, default=1.0)
    ap.add_argument("--lambda_rel_dir", type=float, default=0.2)
    ap.add_argument("--rel_dir_margin", type=float, default=0.2)
    ap.add_argument("--lambda_role", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_filters",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,out_proj,fc1,fc2,proj",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = AI2DStructureDataset(args.ai2d_root, args.split_dir, "train")
    test_ds = AI2DStructureDataset(args.ai2d_root, args.split_dir, "test")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = AI2DTeacher(
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_filters=[s.strip() for s in args.lora_filters.split(",") if s.strip()],
    ).to(device)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    def run_epoch(loader, train: bool):
        model.train(train)
        total = 0.0
        count = 0
        for images, nodes_batch, edges_batch in tqdm(loader, desc="train" if train else "eval"):
            if train:
                optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                tokens, grid_thw = model(images)
                tokens_grids, grid_shapes = tokens_to_grid(tokens, grid_thw)
                loss = torch.tensor(0.0, device=device)

                for b in range(len(images)):
                    nodes = nodes_batch[b]
                    edges = edges_batch[b]
                    if not nodes:
                        continue
                    bboxes = [n["bbox"] for n in nodes]
                    tokens_grid = tokens_grids[b]
                    grid_h, grid_w = grid_shapes[b]
                    feats = pool_roi_features(tokens_grid, bboxes, grid_h, grid_w)
                    node_types = torch.tensor([n["type"] for n in nodes], device=device)
                    node_roles = torch.tensor([n["role"] for n in nodes], device=device)
                    gt_boxes = torch.tensor(bboxes, device=device)

                    node_logits = model.node_head(feats)
                    box_pred = model.box_head(feats)
                    l_node = F.cross_entropy(node_logits, node_types)
                    l_box = F.smooth_l1_loss(box_pred, gt_boxes)
                    loss = loss + args.lambda_node * l_node + args.lambda_box * l_box

                    # role loss (text nodes only)
                    role_mask = node_roles >= 0
                    if role_mask.any():
                        role_logits = model.role_head(feats[role_mask])
                        l_role = F.cross_entropy(role_logits, node_roles[role_mask])
                        loss = loss + args.lambda_role * l_role

                    # relation loss
                    if edges:
                        pos_pairs = [(e[0], e[1]) for e in edges]
                        pos_labels = [e[2] for e in edges]
                        pos_set = set(pos_pairs)
                        node_type_list = [n["type"] for n in nodes]
                        num_neg = int(len(pos_pairs) * args.neg_ratio)
                        rng = random.Random(args.seed + b)

                        rev_pairs = []
                        if not args.no_rev_neg:
                            for o, d in pos_pairs:
                                if (d, o) not in pos_set:
                                    rev_pairs.append((d, o))

                        same_src_pairs = []
                        if args.same_src_neg > 0:
                            for o, d in pos_pairs:
                                cand = [
                                    i for i, t in enumerate(node_type_list)
                                    if i not in (o, d) and t == node_type_list[d]
                                ]
                                if not cand:
                                    cand = [i for i in range(len(nodes)) if i not in (o, d)]
                                if cand:
                                    v = rng.choice(cand)
                                    if (o, v) not in pos_set:
                                        same_src_pairs.append((o, v))

                        neg_pairs = []
                        num_extra = max(0, num_neg - len(rev_pairs) - len(same_src_pairs))
                        tries = 0
                        while len(neg_pairs) < num_extra and tries < num_extra * 10:
                            i = rng.randrange(len(nodes))
                            j = rng.randrange(len(nodes))
                            if i == j or (i, j) in pos_set:
                                tries += 1
                                continue
                            if (i, j) in rev_pairs or (i, j) in same_src_pairs:
                                tries += 1
                                continue
                            neg_pairs.append((i, j))
                        neg_labels = [REL2IDX["none"]] * (len(rev_pairs) + len(same_src_pairs) + len(neg_pairs))

                        all_pairs = pos_pairs + rev_pairs + same_src_pairs + neg_pairs
                        all_labels = torch.tensor(pos_labels + neg_labels, device=device)
                        if all_pairs:
                            f_src = feats[[p[0] for p in all_pairs]]
                            f_dst = feats[[p[1] for p in all_pairs]]
                            rel_feat = torch.cat([f_src, f_dst, f_src * f_dst, (f_src - f_dst).abs()], dim=-1)
                            rel_logits = model.rel_head(rel_feat)
                            l_rel = F.cross_entropy(rel_logits, all_labels)
                            loss = loss + args.lambda_rel * l_rel

                            if args.lambda_rel_dir > 0 and rev_pairs:
                                pair_to_idx = {pair: idx for idx, pair in enumerate(all_pairs)}
                                dir_losses = []
                                for (o, d), rel_idx in zip(pos_pairs, pos_labels):
                                    pos_idx = pair_to_idx.get((o, d))
                                    rev_idx = pair_to_idx.get((d, o))
                                    if pos_idx is None or rev_idx is None:
                                        continue
                                    pos_score = rel_logits[pos_idx, rel_idx]
                                    rev_score = rel_logits[rev_idx, 1:].max()
                                    dir_losses.append(F.relu(args.rel_dir_margin - (pos_score - rev_score)))
                                if dir_losses:
                                    l_dir = torch.stack(dir_losses).mean()
                                    loss = loss + args.lambda_rel_dir * l_dir

            if train:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

            total += float(loss.item())
            count += 1

        return total / max(1, count)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(test_loader, train=False)
        print(f"[epoch {ep}] train {train_loss:.4f} val {val_loss:.4f}")

        ckpt = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model": trainable_state_dict(model),
            "node_types": NODE_TYPES,
            "role_types": ROLE_TYPES,
            "rel_types": REL_TYPES,
        }
        torch.save(ckpt, Path(args.out_dir) / f"ckpt_epoch_{ep}.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, Path(args.out_dir) / "ckpt_best.pt")


if __name__ == "__main__":
    main()

