"""
NextScaleVisualHead: predict VQ token logits for a fixed grid scale from a
shared latent (z_G) and style vector, with optional token context for
coarse-to-fine conditioning.

Initial version: top-scale only (e.g., 16x16). The module produces a sequence
of grid queries with positional encodings, processes them with several
Transformer blocks, and projects to `vocab_size` logits per position.

API
- forward(z_g, style, token_context=None) -> logits [B, H*W, V]
- infer(z_g, style, temperature=1.0) -> token_ids [B, H, W]
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, mlp_ratio: int = 4):
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


class NextScaleVisualHead(nn.Module):
    """Top-scale visual token predictor (coarse-to-fine seed).

    Parameters
    - vocab_size: size of VQ codebook (e.g., 4096)
    - z_dim: latent feature dimension (default 768)
    - style_dim: style embedding dimension (default 256)
    - grid: spatial grid size at top scale (default 16)
    - d_model: token hidden dim
    - n_layers: number of transformer blocks
    - n_heads: attention heads
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        z_dim: int = 768,
        style_dim: int = 256,
        grid: int = 16,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.grid = int(grid)
        self.d_model = int(d_model)
        self.pos = nn.Parameter(torch.randn(1, self.grid * self.grid, d_model) * 0.02)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.style_proj = nn.Linear(style_dim, d_model)
        self.blocks = nn.ModuleList([_Block(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size)

    def forward(
        self,
        z_g: torch.Tensor,
        style: torch.Tensor,
        token_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute top-scale token logits.

        Inputs
        - z_g: [B, z_dim]
        - style: [B, style_dim]
        - token_context: optional [B, N_ctx, d_model] for future conditioning

        Returns
        - logits: [B, H*W, vocab_size]
        """
        b = z_g.size(0)
        base = self.z_proj(z_g) + self.style_proj(style)  # [B, d]
        x = base.unsqueeze(1).expand(b, self.grid * self.grid, base.size(-1)) + self.pos
        # Simple self-attention stack (coarse-to-fine context can be injected later)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits  # [B, H*W, V]

    @torch.no_grad()
    def infer(self, z_g: torch.Tensor, style: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Greedy/soft inference of token IDs at top scale. Returns [B,H,W]."""
        logits = self.forward(z_g, style)  # [B, HW, V]
        if temperature and temperature > 0 and abs(temperature - 1.0) > 1e-5:
            logits = logits / float(temperature)
        ids = torch.argmax(logits, dim=-1)  # [B, HW]
        h = w = self.grid
        return ids.view(ids.size(0), h, w)

