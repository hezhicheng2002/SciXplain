from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys, os

T5ForConditionalGeneration = None
T5Tokenizer = None


class _TransformerBlock(nn.Module):
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


class ToyDiTDecoder(nn.Module):
    """
    Lightweight DiT-like decoder that maps z_G (768) + style (256) to a 256x256 image.
    Uses a small Transformer over an 8x8 latent grid then up-convolves.
    """

    def __init__(self, z_dim: int = 768, style_dim: int = 256, grid: int = 8, d_model: int = 512, n_layers: int = 4):
        super().__init__()
        self.grid = grid
        self.d_model = d_model
        self.pos = nn.Parameter(torch.randn(1, grid * grid, d_model) * 0.02)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.style_proj = nn.Linear(style_dim, d_model)
        self.blocks = nn.ModuleList([_TransformerBlock(d_model=d_model, n_heads=8) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        # Up-convolutional decoder to 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 256, 4, 2, 1),  # 8->16
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),      # 16->32
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),       # 32->64
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),        # 64->128
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),        # 128->256
            nn.GELU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_g: torch.Tensor, style: torch.Tensor, tokens_init: 'torch.Tensor | None' = None) -> torch.Tensor:
        # z_g: (B, 768), style: (B, 256), tokens_init: optional (B, N, d_model) or (B, d_model, H, W)
        b = z_g.size(0)
        base = self.z_proj(z_g) + self.style_proj(style)  # (B, d_model)
        if tokens_init is not None:
            if tokens_init.dim() == 4:
                # (B, d_model, H, W) -> (B, N, d_model)
                tokens = rearrange(tokens_init, 'b c h w -> b (h w) c')
            else:
                tokens = tokens_init
            tokens = tokens + base.unsqueeze(1) + self.pos
        else:
            tokens = base.unsqueeze(1).expand(b, self.grid * self.grid, base.size(-1)) + self.pos  # (B, N, d)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        feat = rearrange(tokens, "b (h w) c -> b c h w", h=self.grid, w=self.grid)
        img = self.decoder(feat)
        return img  # (B, 3, 256, 256) in [0,1]


class WarmupGridAdapter(nn.Module):
    """
    Map vision tokens to a low-res latent grid to initialize DiT tokens during warmup.
    - Accepts CLIP and/or Rex tokens (B, N, 768) including CLS at position 0.
    - Reshapes spatial tokens to HxW (best square), projects to d_model, resizes to (grid, grid).
    - Fuses multiple sources by averaging.
    Output: (B, N=grid*grid, d_model)
    """

    def __init__(self, grid: int = 8, in_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.grid = grid
        self.proj = nn.Linear(in_dim, out_dim)

    def _to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D), including CLS at index 0
        b, n, d = tokens.shape
        if n <= 1:
            return torch.zeros(b, d, self.grid, self.grid, device=tokens.device, dtype=tokens.dtype)
        x = tokens[:, 1:, :]  # drop CLS
        s = int((x.size(1)) ** 0.5)
        s = max(1, s)
        x = x[:, : s * s, :]
        x = x.view(b, s, s, d)  # (B, H, W, D)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        x = F.interpolate(x, size=(self.grid, self.grid), mode='bilinear', align_corners=False)
        return x  # (B, D, grid, grid)

    def forward(self, clip_tokens: torch.Tensor, rex_tokens: 'torch.Tensor | None' = None) -> torch.Tensor:
        maps = []
        if clip_tokens is not None:
            m = self._to_map(clip_tokens)  # (B, D, g, g)
            m = self.proj(m.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # project channels
            maps.append(m)
        if rex_tokens is not None:
            m = self._to_map(rex_tokens)
            m = self.proj(m.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            maps.append(m)
        if not maps:
            return torch.zeros(clip_tokens.size(0), self.grid * self.grid, self.proj.out_features, device=clip_tokens.device, dtype=clip_tokens.dtype)
        fused = torch.stack(maps, dim=0).mean(dim=0)  # average sources -> (B, C, g, g)
        tokens = rearrange(fused, 'b c h w -> b (h w) c')  # (B, N, C)
        return tokens


class T5PromptDecoder(nn.Module):
    """
    T5-small conditioned via learned prompt embeddings derived from z_G.
    - Training: autoencoding by reconstructing input text with prompt + text as encoder input.
    - Inference: generation from prompt only.
    """

    def __init__(self, prompt_len: int = 10, z_dim: int = 768, layout_in_dim: int = 512):
        super().__init__()
        global T5ForConditionalGeneration, T5Tokenizer
        if T5ForConditionalGeneration is None or T5Tokenizer is None:
            try:
                from transformers import T5ForConditionalGeneration as _T5Model, T5Tokenizer as _T5Tokenizer
                T5ForConditionalGeneration = _T5Model
                T5Tokenizer = _T5Tokenizer
            except Exception as e:
                raise RuntimeError(f"T5 could not be imported: {e}")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        d_model = self.model.config.d_model
        self.prompt_len = prompt_len
        self.z_to_prompt = nn.Sequential(
            nn.Linear(z_dim, d_model * prompt_len),
        )
        # Optional layout tokens mapping to T5 d_model
        self.layout_proj = nn.Linear(layout_in_dim, d_model)

    def make_prompt(self, z_g: torch.Tensor) -> torch.Tensor:
        b, _ = z_g.shape
        d_model = self.model.config.d_model
        prompt = self.z_to_prompt(z_g).view(b, self.prompt_len, d_model)
        return prompt

    def forward(self, z_g: torch.Tensor, texts: list[str], layout_tokens: 'torch.Tensor | None' = None, prompt_only: bool = False):
        # Build encoder inputs
        device = next(self.parameters()).device
        prompt = self.make_prompt(z_g).to(device)
        # Optional layout context
        layout_emb = None
        if layout_tokens is not None:
            layout_emb = self.layout_proj(layout_tokens.to(device))  # (B, N, d)

        if prompt_only:
            # No text fed to encoder; use only prompt (+ layout) as condition, predict caption
            att = [prompt]
            if layout_emb is not None:
                att.append(layout_emb)
            inputs_embeds = torch.cat(att, dim=1)
            tok = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            tok = {k: v.to(device) for k, v in tok.items()}
            labels = tok["input_ids"].clone()
            out = self.model(encoder_outputs=None, inputs_embeds=inputs_embeds, labels=labels, return_dict=True)
        else:
            tok = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            tok = {k: v.to(device) for k, v in tok.items()}
            embeds = self.model.encoder.embed_tokens(tok["input_ids"])  # (B, L, d)
            att = [prompt, embeds]
            if layout_emb is not None:
                att.insert(1, layout_emb)  # [prompt, layout, text]
            inputs_embeds = torch.cat(att, dim=1)
            labels = tok["input_ids"].clone()
            out = self.model(encoder_outputs=None, inputs_embeds=inputs_embeds, labels=labels, return_dict=True)
        return out.loss, {"logits": out.logits}

    @torch.no_grad()
    def generate(self, z_g: torch.Tensor, max_new_tokens: int = 128, layout_tokens: 'torch.Tensor | None' = None):
        device = next(self.parameters()).device
        prompt = self.make_prompt(z_g).to(device)
        b = prompt.size(0)
        # Concatenate layout context if provided
        if layout_tokens is not None:
            layout_emb = self.layout_proj(layout_tokens.to(device))
            inputs = torch.cat([prompt, layout_emb], dim=1)
        else:
            inputs = prompt
        attn_mask = torch.ones(b, inputs.size(1), device=device, dtype=torch.long)
        gen_ids = self.model.generate(inputs_embeds=inputs, attention_mask=attn_mask, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


class MaskDecoder(nn.Module):
    """
    Segmentation head predicting a 1x256x256 mask from z_G and style.
    Architecture mirrors ToyDiTDecoder but ends with a single-channel logits map (no Sigmoid).
    """

    def __init__(self, z_dim: int = 768, style_dim: int = 256, grid: int = 8, d_model: int = 256, n_layers: int = 2):
        super().__init__()
        self.grid = grid
        self.pos = nn.Parameter(torch.randn(1, grid * grid, d_model) * 0.02)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.style_proj = nn.Linear(style_dim, d_model)
        self.blocks = nn.ModuleList([_TransformerBlock(d_model=d_model, n_heads=8) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        # Up-convolutional path to 256x256 mask logits
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, 4, 2, 1),  # 8->16
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),       # 16->32
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),        # 32->64
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),        # 64->128
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),         # 128->256
            nn.GELU(),
            nn.Conv2d(8, 1, 1),  # logits
        )

    def forward(self, z_g: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b = z_g.size(0)
        base = self.z_proj(z_g) + self.style_proj(style)
        tokens = base.unsqueeze(1).expand(b, self.grid * self.grid, base.size(-1)) + self.pos
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        feat = rearrange(tokens, "b (h w) c -> b c h w", h=self.grid, w=self.grid)
        logits = self.decoder(feat)
        return logits  # (B,1,256,256)


class VarVaeDecoder(nn.Module):
    """
    Wrap VAR-main's VQVAE as an image decoder. Maps z_G (+ optional style/prior) to a top-scale VAE feature map f_hat (Cvae, 16, 16),
    then decodes to image via VAE decoder. This provides a stronger decoder than ToyDiT for structure learning.
    """

    def __init__(self, var_repo: str, vae_ckpt: str, top_grid: int = 16, z_dim: int = 768, style_dim: int = 256, cvae: int = 32):
        super().__init__()
        if var_repo not in sys.path:
            sys.path.append(var_repo)
        try:
            from models.vqvae import VQVAE  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import VAR VQVAE from {var_repo}: {e}")
        self.top_grid = top_grid
        self.cvae = cvae
        self.vae = VQVAE(vocab_size=4096, z_channels=cvae, ch=160, test_mode=True)
        # load VAE weights
        try:
            sd = torch.load(vae_ckpt, map_location='cpu')
            self.vae.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"[warn] Could not load VAE ckpt ({vae_ckpt}): {e}")
        # simple head: map z/style/prior to f_hat (B, Cvae, 16,16)
        in_dim = z_dim + style_dim
        self.to_fhat = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(),
            nn.Linear(512, self.cvae * self.top_grid * self.top_grid)
        )
        # optional projection for layout prior tokens
        self.prior_proj = nn.Linear(512, self.cvae)

    def forward(self, z_g: torch.Tensor, style: torch.Tensor, prior_tokens: 'torch.Tensor | None' = None) -> torch.Tensor:
        b = z_g.size(0)
        h = torch.cat([z_g, style], dim=-1)
        fhat = self.to_fhat(h).view(b, self.cvae, self.top_grid, self.top_grid)
        if prior_tokens is not None:
            # prior_tokens: (B, N, d_model=512) -> mean over N then map to Cvae and add
            if prior_tokens.dim() == 3:
                p = prior_tokens.mean(dim=1)
                p = self.prior_proj(p).view(b, self.cvae, 1, 1)
                fhat = fhat + p
        img_m1_1 = self.vae.fhat_to_img(fhat)  # in [-1,1]
        img = (img_m1_1 + 1.0) * 0.5
        return img.clamp(0, 1)
