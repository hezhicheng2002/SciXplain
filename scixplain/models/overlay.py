from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class OverlayHead(nn.Module):
    """
    Predicts up to `num_queries` overlay objects (textbox/arrow/rect/ellipse/none) from latent z_G and style.
    Outputs per-object: class logits (num_classes) and bbox (cx, cy, w, h) in [0,1].
    This is a lightweight, set-prediction-like head meant to be trained with weak coverage losses.
    """

    def __init__(
        self,
        z_dim: int = 768,
        style_dim: int = 256,
        d_model: int = 384,
        num_queries: int = 16,
        num_classes: int = 5,  # 0:none, 1:textbox, 2:rect, 3:ellipse, 4:arrow
        n_layers: int = 3,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.style_proj = nn.Linear(style_dim, d_model)
        self.blocks = nn.ModuleList([_Block(d_model=d_model, n_heads=max(1, d_model // 64)) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, num_classes)
        )
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4), nn.Sigmoid()
        )  # (cx, cy, w, h) normalized

    def forward(self, z_g: torch.Tensor, style: torch.Tensor):
        b = z_g.size(0)
        base = self.z_proj(z_g) + self.style_proj(style)  # (B, d)
        q = self.query_embed.expand(b, -1, -1) + base.unsqueeze(1)
        for blk in self.blocks:
            q = blk(q)
        q = self.norm(q)
        logits = self.cls_head(q)  # (B,Q,C)
        boxes = self.box_head(q)   # (B,Q,4) in [0,1]
        return {"logits": logits, "boxes": boxes}

