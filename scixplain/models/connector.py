from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Connector(nn.Module):
    """
    Concatenate CLIP and Rex tokens (already projected to 768),
    cross-attend with SciBERT mean pooled token to produce latent z_G in R^768.
    Includes FiLM modulators set on towers (applied before projection) via external wiring.
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        h_clip_tokens: torch.Tensor,
        h_rex_tokens: torch.Tensor,
        h_txt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # h_clip_tokens, h_rex_tokens: (B, N, 768)
        # h_txt_tokens: (B, L, 768)
        hv = torch.cat([h_clip_tokens, h_rex_tokens], dim=1)  # (B, N_total, 768)
        ht = h_txt_tokens.mean(dim=1, keepdim=True)  # (B, 1, 768)
        z, _ = self.attn(query=ht, key=hv, value=hv)
        z = self.mlp(z)
        return z.squeeze(1)  # (B, 768)

