"""
VarVQTokenizer: lightweight wrapper around a VAR/VQ-VAE checkpoint providing
discrete visual tokenization (encode) and reconstruction (decode).

This class aims to be resilient to common VQ-VAE/VQGAN APIs found in VAR repos:
- encode(x) -> returns structure containing code indices or quantized embeddings
- get_code_indices(x) / encode_to_indices(x)
- embed_code(indices) / quantize.embedding.weight for index->embedding
- fhat_to_img(fhat) or decode(z_q) / decoder(z_q) to map features to image

Notes
- We normalize inputs to [-1, 1] as most VQ encoders expect this range.
- Multi-scale outputs are supported at the API level; initial implementation
  guarantees top scale (e.g., 16x16). Larger scales are optionally produced by
  nearest-neighbor upsampling of the top-scale token map when true multi-scale
  quantizers are unavailable (with a one-time warning).

Usage
- Instantiate with local VAR repo path and VAE checkpoint path.
- Call encode(list[PIL.Image] | Tensor[B,3,H,W]) -> {"16": LongTensor[B,16,16], ...}
- Call decode({"16": LongTensor[B,16,16], ...}) -> FloatTensor[B,3,256,256] in [0,1]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarVQTokenizer(nn.Module):
    """VAR-style VQ-VAE tokenizer.

    Parameters
    - var_repo: path to a local VAR repository that provides `models.vqvae.VQVAE`
    - vae_ckpt: path to VAE weights (expected to match vocab_size/z_channels)
    - img_size: input/output spatial size (default 256)
    - vocab_size: codebook size (default 4096)
    - z_channels: latent channels for top scale (default 32)
    - scales: list of string/int scales, e.g., ["16","32"]. Top scale must be first.
    """

    def __init__(
        self,
        var_repo: str,
        vae_ckpt: str,
        img_size: int = 256,
        vocab_size: int = 4096,
        z_channels: int = 32,
        scales: Optional[Sequence[Union[str, int]]] = ("16",),
    ) -> None:
        super().__init__()
        if var_repo and var_repo not in sys.path:
            sys.path.append(var_repo)
        try:
            from models.vqvae import VQVAE  # type: ignore
        except Exception as e:  # pragma: no cover - external runtime
            raise RuntimeError(
                f"Failed to import VAR VQVAE from {var_repo}: {e}. "
                "Ensure VAR repo is available and provides models.vqvae.VQVAE."
            )
        self.img_size = int(img_size)
        self.vocab_size = int(vocab_size)
        self.z_channels = int(z_channels)
        # normalize scales to strings
        self.scales: List[str] = [str(s) for s in (list(scales) if scales else ["16"])]
        try:
            self.vae = VQVAE(vocab_size=self.vocab_size, z_channels=self.z_channels, ch=160, test_mode=True)
        except Exception:
            # Fallback constructor signature
            self.vae = VQVAE()
        # Load checkpoint (non-strict to tolerate small diffs)
        try:
            sd = torch.load(vae_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            self.vae.load_state_dict(sd, strict=False)
        except Exception as e:  # pragma: no cover - external runtime
            print(f"[warn] VarVQTokenizer: could not load VAE ckpt {vae_ckpt}: {e}")
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
        # one-time warning flag for fake multi-scale upsampling
        self._warned_upsample = False

        # image pre/post transforms
        self._resize = nn.Upsample(size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def _to_pixels(self, imgs: Union[List["PIL.Image.Image"], torch.Tensor]) -> torch.Tensor:
        """Convert PIL list or float tensor in [0,1] to Tensor[B,3,H,W] in [-1,1]."""
        if isinstance(imgs, torch.Tensor):
            x = imgs
            if x.dim() == 3:
                x = x.unsqueeze(0)
            # assume input in [0,1]
            if x.max() > 1.5:  # uint8
                x = x.float() / 255.0
        else:
            from torchvision import transforms
            tfm = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
            x = torch.stack([tfm(im) for im in imgs], dim=0)
        x = x.to(next(self.vae.parameters()).device)
        x = (x * 2.0) - 1.0
        return x

    @torch.no_grad()
    def encode(self, imgs: Union[List["PIL.Image.Image"], torch.Tensor]) -> Dict[str, torch.LongTensor]:
        """Encode images to discrete token IDs per configured scale.

        Tries a series of common VQ-VAE APIs to extract top-scale code indices:
        - get_code_indices(x)
        - encode_to_indices(x)
        - encode(x) returning (z_q, indices, ...), or dict/object with .indices/.codes
        - quantize.encode(x) returning (..., indices, ...)

        Returns a dict mapping scale name to LongTensor[B,H,W] (top scale first).
        At minimum, top scale is provided. Additional configured scales may be
        synthesized via nearest resize if no true multi-scale is exposed.
        """
        x = self._to_pixels(imgs)
        idx: Optional[torch.Tensor] = None
        # 0) VAR-main multi-scale API: img_to_idxBl -> List[Bl]
        try:
            if hasattr(self.vae, "img_to_idxBl"):
                ls_idx = self.vae.img_to_idxBl(x, v_patch_nums=None)  # type: ignore[attr-defined]
                if isinstance(ls_idx, (list, tuple)) and len(ls_idx) > 0 and torch.is_tensor(ls_idx[-1]):
                    top_idx = ls_idx[-1]  # last is largest scale according to VAR-main
                    # shape [B, Bl] -> [B,pn,pn]
                    if top_idx.dim() == 2:
                        b, l = top_idx.shape
                        pn = int(round(l ** 0.5))
                        idx = top_idx.view(b, pn, pn)
        except Exception:
            idx = None
        # 1) direct API
        try:
            if hasattr(self.vae, "get_code_indices"):
                out = self.vae.get_code_indices(x)  # type: ignore[attr-defined]
                if torch.is_tensor(out):
                    idx = out
        except Exception:
            idx = None
        # 2) encode_to_indices
        if idx is None:
            try:
                if hasattr(self.vae, "encode_to_indices"):
                    out = self.vae.encode_to_indices(x)  # type: ignore[attr-defined]
                    if torch.is_tensor(out):
                        idx = out
                    elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                        idx = out[0]
            except Exception:
                idx = None
        # 3) generic encode
        if idx is None:
            try:
                if hasattr(self.vae, "encode"):
                    enc_out = self.vae.encode(x)  # type: ignore[attr-defined]
                    # tuple: assume (z_q, indices, ...)
                    if isinstance(enc_out, (list, tuple)):
                        # pick the first LongTensor-like among slots 1..n
                        for t in enc_out[1:]:
                            if torch.is_tensor(t) and t.dtype in (torch.int64, torch.int32, torch.int16):
                                idx = t
                                break
                            if torch.is_tensor(t) and t.dtype.is_floating_point and t.dim() in (2, 3):
                                # sometimes indices are provided as float; cast later
                                idx = t
                                break
                    # dict-like
                    if idx is None and isinstance(enc_out, dict):
                        for k in ["indices", "codes", "code_indices", "ids"]:
                            v = enc_out.get(k)
                            if torch.is_tensor(v):
                                idx = v
                                break
                    # object with attribute
                    if idx is None and hasattr(enc_out, "indices") and torch.is_tensor(getattr(enc_out, "indices")):
                        idx = getattr(enc_out, "indices")
            except Exception:
                idx = None
        # 4) quantizer encode
        if idx is None:
            try:
                q = getattr(self.vae, "quantize", None)
                if q is not None and hasattr(q, "encode"):
                    q_out = q.encode(x)
                    if torch.is_tensor(q_out):
                        idx = q_out
                    elif isinstance(q_out, (list, tuple)):
                        for t in q_out:
                            if torch.is_tensor(t) and t.dtype in (torch.int64, torch.int32, torch.int16):
                                idx = t
                                break
            except Exception:
                idx = None
        if idx is None:
            raise RuntimeError(
                "VarVQTokenizer.encode: could not obtain code indices from VAE. "
                "Tried get_code_indices/encode_to_indices/encode/quantize.encode."
            )
        # cast/reshape
        if idx.dtype.is_floating_point:
            idx = idx.round().long()
        # idx: expected shape [B, H*W] or [B,H,W]
        if idx.dim() == 2:
            b, hw = idx.shape
            h = w = int(hw ** 0.5)
            idx = idx.view(b, h, w)
        top = idx.long().contiguous()
        out: Dict[str, torch.LongTensor] = {self.scales[0]: top}
        # synthesize additional scales if requested beyond the top
        for sc in self.scales[1:]:
            try:
                s = int(sc)
                if s <= top.size(1):
                    # downsample by nearest
                    out[str(sc)] = F.interpolate(
                        top.unsqueeze(1).float(), size=(s, s), mode="nearest"
                    ).squeeze(1).long()
                else:
                    # upsample by nearest
                    out[str(sc)] = F.interpolate(
                        top.unsqueeze(1).float(), size=(s, s), mode="nearest"
                    ).squeeze(1).long()
                if not self._warned_upsample:
                    print(
                        "[warn] VarVQTokenizer: multi-scale indices created via nearest resize of top scale; "
                        "consider adding true multi-scale tokenizers for higher fidelity."
                    )
                    self._warned_upsample = True
            except Exception:
                continue
        return out

    @torch.no_grad()
    def _indices_to_fhat(self, idx: torch.Tensor) -> torch.Tensor:
        """Map code indices [B,H,W] to latent feature map [B,C,H,W] using VAE's codebook.
        Tries embed_code, quantize.embedding, or codebook.embeddings patterns.
        """
        # embed_code path
        if hasattr(self.vae, "embed_code"):
            try:
                z = self.vae.embed_code(idx)  # type: ignore[attr-defined]
                return z
            except Exception:
                pass
        # Generic gather from embedding weight
        emb = None
        # common names across repos
        for attr in [
            "quantize",
            "index_quantizer",
            "codebook",
        ]:
            m = getattr(self.vae, attr, None)
            if m is None:
                continue
            # table attributes
            for we in ["_embedding", "embedding", "embed", "codebook"]:
                table = getattr(m, we, None)
                if table is None:
                    continue
                w = getattr(table, "weight", None)
                if torch.is_tensor(w):
                    emb = w
                    break
            if emb is not None:
                break
        if emb is None:
            raise RuntimeError("VarVQTokenizer: cannot locate codebook embedding weight to map indices→latent.")
        b, h, w = idx.shape
        flat = idx.view(b, -1)
        z = F.embedding(flat, emb)  # [B,HW,C]
        z = z.view(b, h, w, emb.size(1)).permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        return z

    @torch.no_grad()
    def decode(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode token maps into images in [0,1].

        Requires the top scale present in `tokens`. We map indices to latent f̂
        via codebook embedding, then call an available decode path:
        - fhat_to_img(fhat) -> [-1,1]
        - decode(fhat) / decoder(fhat) -> [-1,1] or feature map
        """
        if not tokens:
            raise ValueError("decode(tokens): tokens dict is empty")
        # choose the top scale entry (first configured)
        top_key = self.scales[0]
        if top_key not in tokens:
            # fallback: pick any present key
            top_key = sorted(tokens.keys(), key=lambda x: int(x))[0]
        idx = tokens[top_key].to(next(self.vae.parameters()).device)
        if idx.dim() != 3:
            raise ValueError("decode expects token map [B,H,W] for top scale")
        # Prefer VAR-main idx decode if available
        img_m11 = None
        try:
            if hasattr(self.vae, "idxBl_to_img"):
                # expects List[Bl] flattened tokens
                b, h, w = idx.shape
                flat = idx.view(b, h * w)
                y = self.vae.idxBl_to_img([flat], same_shape=True, last_one=True)  # type: ignore[attr-defined]
                img_m11 = y
        except Exception:
            img_m11 = None
        if img_m11 is None:
            # Fall back to embedding -> fhat -> decode
            fhat = self._indices_to_fhat(idx)
            # Try decoder APIs
            for method in ["fhat_to_img", "decode"]:
                fn = getattr(self.vae, method, None)
                if fn is None:
                    continue
                try:
                    y = fn(fhat)
                    img_m11 = y
                    break
                except Exception:
                    continue
            if img_m11 is None:
                # Try direct decoder module
                dec = getattr(self.vae, "decoder", None)
                if dec is None:
                    raise RuntimeError("VarVQTokenizer: VAE has no decode path (fhat_to_img/decode/decoder)")
                try:
                    img_m11 = dec(fhat)
                except Exception as e:
                    raise RuntimeError(f"VarVQTokenizer: decoder(fhat) failed: {e}")
        # Map from [-1,1] to [0,1] if needed
        img = img_m11
        if torch.is_floating_point(img):
            if img.min() < -0.1 or img.max() > 1.1:
                img = (img + 1.0) * 0.5
        img = img.clamp(0, 1)
        # ensure spatial size
        if img.size(-1) != self.img_size or img.size(-2) != self.img_size:
            img = F.interpolate(img, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return img

    @property
    def codebook_size(self) -> int:
        return int(self.vocab_size)

    @property
    def top_grid(self) -> int:
        try:
            return int(self.scales[0])
        except Exception:
            return 16
