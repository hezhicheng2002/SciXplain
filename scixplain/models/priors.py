from typing import Optional, Dict, Any
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutPriorAdapter(nn.Module):
    """
    Build a token grid prior for the image decoder from z_G and optional layout parser outputs.
    - If VAR repo/ckpt available, attempt to parse layout and embed to grid tokens.
    - Otherwise fallback: MLP(z_G) broadcasted to grid with learned positional embeddings.
    Output shape: (B, grid*grid, d_model)
    """

    def __init__(self, grid: int = 8, z_dim: int = 768, d_model: int = 512, var_repo: Optional[str] = None, var_ckpt: Optional[str] = None):
        super().__init__()
        self.grid = grid
        self.z_proj = nn.Linear(z_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, d_model) * 0.02)
        self.var_repo = var_repo
        self.var_ckpt = var_ckpt
        self._var_model = None
        # try lazy load VAR on first forward

    def _maybe_load_var(self):
        if self._var_model is not None:
            return
        if self.var_repo and os.path.isdir(self.var_repo):
            try:
                if self.var_repo not in sys.path:
                    sys.path.append(self.var_repo)
                # Placeholder: user will place real model here; we look for a callable build_model
                from importlib import import_module
                try:
                    vm = import_module('VAR')  # if VAR/__init__.py exists
                except Exception:
                    vm = import_module('VAR_main')  # alternative name
                build = getattr(vm, 'build_model', None)
                if build is not None:
                    self._var_model = build(self.var_ckpt)
                else:
                    self._var_model = None
            except Exception:
                self._var_model = None

    def parse_layout(self, images: Optional[list] = None, texts: Optional[list[str]] = None, z_g: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Return a (B, K, d) layout embedding if available; otherwise None."""
        self._maybe_load_var()
        if self._var_model is None:
            return None
        try:
            # Placeholder: assume model has .encode(...) returning (B,K,D)
            with torch.no_grad():
                le = self._var_model.encode(images=images, texts=texts, z=z_g)
            return le
        except Exception:
            return None

    def forward(self, z_g: torch.Tensor, images: Optional[list] = None, texts: Optional[list[str]] = None) -> torch.Tensor:
        b = z_g.size(0)
        base = self.z_proj(z_g).unsqueeze(1)  # (B,1,d_model)
        # use VAR layout if available to modulate base
        le = self.parse_layout(images=images, texts=texts, z_g=z_g)
        if le is not None:
            # Reduce K dimension by mean and add as bias
            try:
                bias = le.mean(dim=1, keepdim=True)
                base = base + nn.Linear(bias.size(-1), base.size(-1), bias=False).to(base.device)(bias)
            except Exception:
                pass
        tokens = base.expand(b, self.grid * self.grid, base.size(-1)) + self.pos  # (B,N,d)
        return tokens

