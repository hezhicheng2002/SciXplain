from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class FusionOutputs:
    tokens: torch.Tensor
    mask: torch.Tensor


class SciBertTextEncoder(nn.Module):
    """
    Skeleton text encoder. Intended to encode OCR/paragraph/mention context with SciBERT.
    Not wired into training yet.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SciBertTextEncoder is a skeleton placeholder.")


class GraphEncoder(nn.Module):
    """
    Skeleton graph encoder for structure (nodes/edges) tokens.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("GraphEncoder is a skeleton placeholder.")


class MultiSourceFusion(nn.Module):
    """
    Skeleton multi-source fusion (vision + text + graph).
    Intended to be used before cross-attention in decoders.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        graph_tokens: Optional[torch.Tensor] = None,
        graph_mask: Optional[torch.Tensor] = None,
    ) -> FusionOutputs:
        raise NotImplementedError("MultiSourceFusion is a skeleton placeholder.")


class CopyHead(nn.Module):
    """
    Skeleton copy head for OCR alignment.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, copy_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("CopyHead is a skeleton placeholder.")
