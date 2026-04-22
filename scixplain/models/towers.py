from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel,
    AutoImageProcessor,
)
from torchvision import transforms
import timm


class FiLMModulator(nn.Module):
    def __init__(self, dim: int, style_dim: int):
        super().__init__()
        self.to_gamma = nn.Sequential(nn.Linear(style_dim, dim), nn.Tanh())
        self.to_beta = nn.Sequential(nn.Linear(style_dim, dim), nn.Tanh())

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) or (B, D)
        if x.dim() == 2:
            gamma = self.to_gamma(s)
            beta = self.to_beta(s)
            return gamma * x + beta
        elif x.dim() == 3:
            gamma = self.to_gamma(s).unsqueeze(1)
            beta = self.to_beta(s).unsqueeze(1)
            return gamma * x + beta
        else:
            raise ValueError("Unexpected tensor rank for FiLMModulator")


class CLIPVisionTower(nn.Module):
    """
    CLIP ViT-L/14-336 vision tower, freeze all layers except last 2 attention blocks.
    Returns patch tokens and pooled features, with optional attentions.
    Output tokens are projected from 1024 -> 768.
    """

    def __init__(self, device: Optional[torch.device] = None, output_attentions: bool = False):
        super().__init__()
        # Replace CLIP with SigLIP2 Large Patch16-384
        self.siglip = AutoModel.from_pretrained(
            "google/siglip2-large-patch16-384",
            torch_dtype="auto",
        )
        self.processor = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-384", use_fast=True)
        self.output_attentions = output_attentions

        # Freeze all layers except last 2 vision blocks; keep text part frozen
        if hasattr(self.siglip, "text_model"):
            for p in self.siglip.text_model.parameters():
                p.requires_grad = False
        # Force safe attention impl to avoid SDPA NaNs under AMP
        try:
            cfg = self.siglip.vision_model.config
            if getattr(cfg, "attn_implementation", None) != "eager":
                setattr(cfg, "attn_implementation", "eager")
        except Exception:
            pass
        try:
            vcfg = getattr(self.siglip, "config", None)
            vcfg = getattr(vcfg, "vision_config", None)
            if vcfg is not None and getattr(vcfg, "attn_implementation", None) != "eager":
                setattr(vcfg, "attn_implementation", "eager")
        except Exception:
            pass
        self.siglip.vision_model.config.output_attentions = output_attentions
        self.siglip.vision_model.config.output_hidden_states = True
        layers = self.siglip.vision_model.encoder.layers
        for i, blk in enumerate(layers):
            req = i >= (len(layers) - 2)
            for p in blk.parameters():
                p.requires_grad = req

        self.vision_hidden = int(self.siglip.vision_model.config.hidden_size)
        self.proj = nn.Linear(self.vision_hidden, 768)
        self.film: Optional[FiLMModulator] = None  # set externally

    @torch.no_grad()
    def preprocess(self, images, image_size: int | None = None):
        if image_size:
            return self.processor(
                images=images,
                return_tensors="pt",
                size={"height": int(image_size), "width": int(image_size)},
            )
        return self.processor(images=images, return_tensors="pt")

    def forward(self, images, style_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # images: dict with preprocessed fields or pixel tensor [B,C,H,W]
        if isinstance(images, dict):
            batch = images
            outputs = self.siglip.vision_model(
                pixel_values=batch.get('pixel_values'),
                output_attentions=self.output_attentions,
                output_hidden_states=True,
                return_dict=True,
                interpolate_pos_encoding=True,
            )
        else:
            outputs = self.siglip.vision_model(
                pixel_values=images,
                output_attentions=self.output_attentions,
                output_hidden_states=True,
                return_dict=True,
                interpolate_pos_encoding=True,
            )
        tokens = outputs.last_hidden_state  # (B, N, 1024)
        pooled_raw = getattr(outputs, "pooler_output", None)
        if pooled_raw is None and tokens.dim() == 3:
            pooled_raw = tokens[:, 0]
        if self.film is not None and style_vec is not None:
            tokens = self.film(tokens, style_vec)
        tokens_proj = self.proj(tokens).to(dtype=torch.float32)  # (B, N, 768)
        pooled = tokens_proj[:, 0]  # CLS after projection
        out = {
            "tokens": tokens_proj,
            "pooled": pooled,
            "tokens_raw": tokens,
            "pooled_raw": pooled_raw if pooled_raw is not None else pooled,
        }
        if self.output_attentions and outputs.attentions is not None:
            out["attentions"] = outputs.attentions  # tuple of (num_layers, B, heads, N, N)
        return out


class RexOmniWrapper(nn.Module):
    """
    Wrapper for IDEA-Research/Rex-Omni. We disable detection decoding and only extract
    intermediate patch tokens and attentions from a middle layer (e.g., 12).
    If loading fails, we fallback to a ViT-S from timm-like HF checkpoint.
    """

    def __init__(self, layer_index: int = 12, device: Optional[torch.device] = None):
        super().__init__()
        self.layer_index = layer_index
        self.model = None
        self.image_processor = None
        self.hidden_size = 1024
        self.attn_supported = False
        self.fallback = False
        try:
            self.model = AutoModel.from_pretrained(
                "IDEA-Research/Rex-Omni", trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    "IDEA-Research/Rex-Omni", trust_remote_code=True
                )
            except Exception:
                self.image_processor = None
            if hasattr(self.model, "config") and getattr(self.model.config, "hidden_size", None):
                self.hidden_size = int(self.model.config.hidden_size)
            # Try to enable hidden states; attentions are optional and may be unsupported under SDPA
            if hasattr(self.model, "config"):
                self.model.config.output_hidden_states = True
                if hasattr(self.model.config, "output_attentions"):
                    try:
                        self.model.config.output_attentions = True
                        self.attn_supported = True
                    except Exception:
                        self.attn_supported = False
        except Exception:
            # Fallback to HF ViT as geometry proxy
            self.fallback = True
            self.model = AutoModel.from_pretrained(
                "google/vit-base-patch16-224",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
            self.hidden_size = int(self.model.config.hidden_size)

        for p in self.model.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(self.hidden_size, 768)
        self.film: Optional[FiLMModulator] = None

    @torch.no_grad()
    def preprocess(self, images):
        if self.image_processor is not None:
            return self.image_processor(images=images, return_tensors="pt")
        else:
            # Fallback basic normalization to 512
            from torchvision import transforms
            tfm = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            if isinstance(images, (list, tuple)):
                imgs = [tfm(im) for im in images]
                return {"pixel_values": torch.stack(imgs)}
            else:
                return {"pixel_values": tfm(images).unsqueeze(0)}

    def forward(self, images, style_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # images can be a tensor or a processor dict with pixel_values + grid info
        if isinstance(images, dict):
            inputs = images
        else:
            inputs = {"pixel_values": images}

        grid_thw = None
        if hasattr(self.model, "get_image_features"):
            feats = self.model.get_image_features(**inputs)
            if isinstance(feats, (tuple, list)):
                if len(feats) == 0:
                    raise ValueError("Empty image features from Rex-Omni")
                if isinstance(feats[0], (tuple, list)):
                    token_list = []
                    grid_list = []
                    for f in feats:
                        if len(f) > 0 and isinstance(f[0], torch.Tensor):
                            token_list.append(f[0])
                        if len(f) > 1:
                            grid_list.append(f[1])
                    if token_list:
                        tokens = token_list
                        if grid_list:
                            grid_thw = grid_list
                elif all(isinstance(f, torch.Tensor) for f in feats):
                    # If variable lengths, keep list; otherwise stack
                    shapes = {tuple(f.shape) for f in feats}
                    tokens = torch.stack(feats, dim=0) if len(shapes) == 1 else feats
                elif isinstance(feats[0], torch.Tensor):
                    tokens = feats[0]
                    if len(feats) > 1:
                        grid_thw = feats[1]
                else:
                    tokens = None
                    for f in feats:
                        if isinstance(f, torch.Tensor):
                            tokens = f
                            break
                        if isinstance(f, (tuple, list)) and f and isinstance(f[0], torch.Tensor):
                            tokens = list(f)
                            break
                    if tokens is None:
                        raise TypeError(f"Unexpected image feature tuple: {type(feats)}")
            elif isinstance(feats, torch.Tensor):
                tokens = feats.unsqueeze(0) if feats.dim() == 2 else feats
            else:
                raise TypeError(f"Unexpected image feature type: {type(feats)}")
        else:
            outputs = self.model(pixel_values=inputs["pixel_values"], output_hidden_states=True, return_dict=True)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                idx = min(self.layer_index, len(outputs.hidden_states) - 1)
                tokens = outputs.hidden_states[idx]  # (B, N, C)
            else:
                tokens = outputs.last_hidden_state  # best effort
        if self.film is not None and style_vec is not None:
            tokens = self.film(tokens, style_vec)
        if isinstance(tokens, (list, tuple)):
            tokens_proj = []
            pooled = []
            for t in tokens:
                t = t.to(self.proj.weight.dtype)
                tp = self.proj(t).to(dtype=torch.float32)
                tokens_proj.append(tp)
                pooled.append(tp.mean(dim=0))
            out = {
                "tokens": tokens_proj,
                "pooled": torch.stack(pooled, dim=0),
            }
        else:
            tokens = tokens.to(self.proj.weight.dtype)
            tokens_proj = self.proj(tokens).to(dtype=torch.float32)
            pooled = tokens_proj.mean(dim=1)
            out = {
                "tokens": tokens_proj,
                "pooled": pooled,
            }
        if self.attn_supported and hasattr(outputs, "attentions") and outputs.attentions is not None:
            out["attentions"] = outputs.attentions
        if isinstance(images, dict) and "image_grid_thw" in images:
            out["grid_thw"] = images["image_grid_thw"]
        elif grid_thw is not None:
            out["grid_thw"] = grid_thw
        return out


class StyleEncoder(nn.Module):
    """
    ViT-S/16 ImageNet pretrained style encoder; outputs 256-dim style vector.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("google/vit-base-patch16-224")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
        hid = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(hid, out_dim)

    @torch.no_grad()
    def preprocess(self, images):
        return self.image_processor(images=images, return_tensors="pt")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=images, return_dict=True)
        pooled = out.last_hidden_state[:, 0]
        return self.proj(pooled)


class DinoVisionTower(nn.Module):
    """
    DINOV2 base vision tower, frozen; outputs projected tokens and pooled features.
    """

    def __init__(self, device: Optional[torch.device] = None, output_attentions: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.output_attentions = output_attentions
        if hasattr(self.backbone, "config"):
            try:
                self.backbone.config.output_hidden_states = True
                if hasattr(self.backbone.config, "output_attentions"):
                    self.backbone.config.output_attentions = output_attentions
            except Exception:
                pass
        self.hidden = int(getattr(self.backbone.config, "hidden_size", 768))
        self.proj = nn.Linear(self.hidden, 768)
        self.film: Optional[FiLMModulator] = None

    @torch.no_grad()
    def preprocess(self, images):
        return self.processor(images=images, return_tensors="pt")

    def forward(self, images: torch.Tensor, style_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.backbone(pixel_values=images, output_hidden_states=True, return_dict=True)
        tokens = out.last_hidden_state  # (B,N,C)
        if self.film is not None and style_vec is not None:
            tokens = self.film(tokens, style_vec)
        tproj = self.proj(tokens).to(dtype=torch.float32)
        pooled = tproj[:, 0]
        return {"tokens": tproj, "pooled": pooled}


class DinoV3VisionTower(nn.Module):
    """
    DINOv3 ViT-L/16 vision tower loaded from local checkpoint (timm backbone), frozen.
    Expects a .pth checkpoint; we load with strict=False for robustness.
    """

    def __init__(self, ckpt_path: str, image_size: int = 224, device: Optional[torch.device] = None):
        super().__init__()
        self.image_size = image_size
        self.backbone = timm.create_model('vit_large_patch16_224', pretrained=False)
        self.embed_dim = getattr(self.backbone, 'embed_dim', 1024)
        # try load ckpt
        try:
            import torch
            sd = torch.load(ckpt_path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            # strip common prefixes
            new_sd = {}
            for k, v in sd.items():
                nk = k
                for pref in ['module.', 'backbone.', 'model.']:
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                new_sd[nk] = v
            missing, unexpected = self.backbone.load_state_dict(new_sd, strict=False)
        except Exception:
            missing = unexpected = []
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.embed_dim, 768)
        self.film: Optional[FiLMModulator] = None
        self.preproc = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def preprocess(self, images):
        if isinstance(images, (list, tuple)):
            pv = torch.stack([self.preproc(im) for im in images], dim=0)
        else:
            pv = self.preproc(images).unsqueeze(0)
        return {"pixel_values": pv}

    def forward(self, images: torch.Tensor, style_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # timm vit forward_outputs last token is CLS in .forward_features
        feats = self.backbone.forward_features(images)  # (B, C) pooled or (B,N,C) depending timm
        if feats.dim() == 2:
            pooled_tok = feats
            tokens = None
        else:
            tokens = feats
            pooled_tok = feats[:, 0]
        if tokens is None:
            tokens = pooled_tok.unsqueeze(1)
        if self.film is not None and style_vec is not None:
            tokens = self.film(tokens, style_vec)
        tproj = self.proj(tokens)
        pooled = tproj[:, 0]
        return {"tokens": tproj, "pooled": pooled}


class QwenVLVisionTower(nn.Module):
    """
    Wrapper for Qwen3-VL-8B-Thinking local checkpoint to extract vision tokens.
    Attempts to locate a vision tower inside the model; falls back gracefully.
    """

    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None, output_attentions: bool = False):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.output_attentions = output_attentions
        self.film: Optional[FiLMModulator] = None
        self.model = None
        self.processor = None
        self.vision_model = None
        self.hidden_size = 1024
        self._vision_entry = None
        self._merge = 2
        self._uses_qwen3_vision = False

        try:
            # Prefer Qwen3 vision-only tower + Qwen2VL image processor
            from transformers import Qwen2VLImageProcessor
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            self.processor = Qwen2VLImageProcessor.from_pretrained(ckpt_path)
            self.vision_model = Qwen3VLVisionModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.model = self.vision_model
            self._vision_entry = self.vision_model
            cfg = getattr(self.vision_model, "config", None)
            if cfg is not None:
                self.hidden_size = int(getattr(cfg, "out_hidden_size", getattr(cfg, "hidden_size", 1024)))
                self._merge = int(getattr(cfg, "spatial_merge_size", 2))
            self._uses_qwen3_vision = True
        except Exception:
            try:
                from transformers import AutoModel, AutoProcessor
                self.model = AutoModel.from_pretrained(
                    ckpt_path, trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                try:
                    self.processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
                except Exception:
                    self.processor = None
                # Try to find a vision tower
                for attr in ["vision_tower", "visual", "vision_model", "vision"]:
                    if hasattr(self.model, attr):
                        self._vision_entry = getattr(self.model, attr)
                        break
                # hidden size inference
                hs = None
                for c in [getattr(self._vision_entry, "config", None), getattr(self.model, "config", None)]:
                    if c is not None and hasattr(c, "hidden_size"):
                        hs = int(getattr(c, "hidden_size"))
                        break
                self.hidden_size = hs or 1024
            except Exception:
                # Best-effort fallback to ViT-Base if local ckpt not available
                self.model = AutoModel.from_pretrained(
                    "google/vit-base-patch16-224",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
                self._vision_entry = self.model
                self.hidden_size = int(self.model.config.hidden_size)

        for p in self.model.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(self.hidden_size, 768)

    @torch.no_grad()
    def preprocess(self, images):
        if self.processor is not None:
            try:
                return self.processor(images=images, return_tensors="pt")
            except Exception:
                pass
        # fallback basic preprocess
        tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        if isinstance(images, (list, tuple)):
            imgs = [tfm(im) for im in images]
            return {"pixel_values": torch.stack(imgs)}
        return {"pixel_values": tfm(images).unsqueeze(0)}

    def forward(self, images, style_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # images: dict of processor outputs or pixel tensor
        tokens = None
        # Qwen3 vision-only path (expects hidden_states + grid_thw)
        if self._uses_qwen3_vision and self.vision_model is not None and isinstance(images, dict):
            pv = images.get("pixel_values")
            grid = images.get("image_grid_thw")
            if grid is None:
                grid = images.get("grid_thw")
            if pv is not None and grid is not None:
                out = self.vision_model(hidden_states=pv, grid_thw=grid)
                tok_flat = out[0] if isinstance(out, (tuple, list)) else out
                grid = grid.to(pv.device)
                if grid.dim() == 1:
                    grid = grid.unsqueeze(0)
                sizes = []
                for t, h, w in grid.tolist():
                    mh = max(1, int(h) // max(1, int(self._merge)))
                    mw = max(1, int(w) // max(1, int(self._merge)))
                    sizes.append(int(t) * mh * mw)
                max_len = max(sizes) if sizes else 0
                if max_len > 0:
                    b = len(sizes)
                    tokens = torch.zeros(b, max_len, tok_flat.size(-1), device=tok_flat.device, dtype=tok_flat.dtype)
                    idx = 0
                    for bi, n in enumerate(sizes):
                        if n <= 0:
                            continue
                        tokens[bi, :n] = tok_flat[idx: idx + n]
                        idx += n
        if tokens is None:
            try:
                out = None
                if isinstance(images, dict):
                    # ensure required flags for some chat models
                    try:
                        if 'image_flags' not in images and 'pixel_values' in images:
                            pv = images['pixel_values']
                            images = {**images, 'image_flags': torch.ones(pv.size(0), 1, dtype=torch.long, device=pv.device)}
                    except Exception:
                        pass
                    if self._vision_entry is not None and hasattr(self._vision_entry, 'forward'):
                        out = self._vision_entry(**images, output_hidden_states=True, return_dict=True)
                    elif hasattr(self.model, 'forward_vision'):
                        out = self.model.forward_vision(**images, output_hidden_states=True, return_dict=True)
                    elif hasattr(self.model, 'vision_model'):
                        out = self.model.vision_model(**images, output_hidden_states=True, return_dict=True)
                else:
                    if self._vision_entry is not None and hasattr(self._vision_entry, 'forward'):
                        out = self._vision_entry(pixel_values=images, output_hidden_states=True, return_dict=True)
                    elif hasattr(self.model, "forward_vision"):
                        out = self.model.forward_vision(pixel_values=images, output_hidden_states=True, return_dict=True)
                    elif hasattr(self.model, "vision_model"):
                        out = self.model.vision_model(pixel_values=images, output_hidden_states=True, return_dict=True)
                if out is not None:
                    if hasattr(out, "last_hidden_state"):
                        tokens = out.last_hidden_state
                    elif hasattr(out, "hidden_states") and out.hidden_states is not None:
                        tokens = out.hidden_states[-1]
            except Exception:
                tokens = None
        if tokens is None:
            # final fallback
            if isinstance(images, dict):
                out = self.model(**images, output_hidden_states=True, return_dict=True)
            else:
                out = self.model(pixel_values=images, output_hidden_states=True, return_dict=True)
            tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]

        if self.film is not None and style_vec is not None:
            tokens = self.film(tokens, style_vec)
        tproj = self.proj(tokens).to(dtype=torch.float32)
        pooled = tproj[:, 0]
        return {"tokens": tproj, "pooled": pooled}


class InternVLVisionTower(nn.Module):
    """
    Wrapper for InternVL 3.5-8B local checkpoint vision tower.
    Similar best-effort approach as QwenVLVisionTower.
    """

    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None, output_attentions: bool = False):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.output_attentions = output_attentions
        self.film: Optional[FiLMModulator] = None
        self.model = None
        self.processor = None
        self.hidden_size = 1024
        self._vision_entry = None
        try:
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(
                ckpt_path, trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            try:
                self.processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
            except Exception:
                self.processor = None
            for attr in ["vision_tower", "visual", "vision_model", "vision"]:
                if hasattr(self.model, attr):
                    self._vision_entry = getattr(self.model, attr)
                    break
            hs = None
            for c in [getattr(self._vision_entry, "config", None), getattr(self.model, "config", None)]:
                if c is not None and hasattr(c, "hidden_size"):
                    hs = int(getattr(c, "hidden_size"))
                    break
            self.hidden_size = hs or 1024
        except Exception:
            self.model = AutoModel.from_pretrained(
                "google/vit-base-patch16-224",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
            self._vision_entry = self.model
            self.hidden_size = int(self.model.config.hidden_size)

        for p in self.model.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(self.hidden_size, 768)

    @torch.no_grad()
    def preprocess(self, images):
        if self.processor is not None:
            try:
                return self.processor(images=images, return_tensors="pt")
            except Exception:
                pass
        tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        if isinstance(images, (list, tuple)):
            imgs = [tfm(im) for im in images]
            return {"pixel_values": torch.stack(imgs)}
        return {"pixel_values": tfm(images).unsqueeze(0)}

    def forward(self, images, style_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        tokens = None
        try:
            out = None
            if isinstance(images, dict):
                try:
                    if 'image_flags' not in images and 'pixel_values' in images:
                        pv = images['pixel_values']
                        images = {**images, 'image_flags': torch.ones(pv.size(0), 1, dtype=torch.long, device=pv.device)}
                except Exception:
                    pass
                if self._vision_entry is not None and hasattr(self._vision_entry, 'forward'):
                    out = self._vision_entry(**images, output_hidden_states=True, return_dict=True)
                elif hasattr(self.model, 'forward_vision'):
                    out = self.model.forward_vision(**images, output_hidden_states=True, return_dict=True)
                elif hasattr(self.model, 'vision_model'):
                    out = self.model.vision_model(**images, output_hidden_states=True, return_dict=True)
            else:
                if self._vision_entry is not None and hasattr(self._vision_entry, 'forward'):
                    out = self._vision_entry(pixel_values=images, output_hidden_states=True, return_dict=True)
                elif hasattr(self.model, "forward_vision"):
                    out = self.model.forward_vision(pixel_values=images, output_hidden_states=True, return_dict=True)
                elif hasattr(self.model, "vision_model"):
                    out = self.model.vision_model(pixel_values=images, output_hidden_states=True, return_dict=True)
            if out is not None:
                if hasattr(out, "last_hidden_state"):
                    tokens = out.last_hidden_state
                elif hasattr(out, "hidden_states") and out.hidden_states is not None:
                    tokens = out.hidden_states[-1]
        except Exception:
            pass
        if tokens is None:
            if isinstance(images, dict):
                out = self.model(**images, output_hidden_states=True, return_dict=True)
            else:
                out = self.model(pixel_values=images, output_hidden_states=True, return_dict=True)
            tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]

        if self.film is not None and style_vec is not None:
            tokens = self.film(tokens, style_vec)
        tproj = self.proj(tokens).to(dtype=torch.float32)
        pooled = tproj[:, 0]
        return {"tokens": tproj, "pooled": pooled}
