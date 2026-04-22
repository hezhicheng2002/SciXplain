#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scixplain.models import CLIPVisionTower
from scixplain.tools.train_tinyllava_image_only import (
    TASK_TOKENS,
    StudentVisionTower,
    apply_lora_to_llm,
    SciStructExplainDataset,
    _estimate_image_tokens,
    bind_region_attention_bias,
)


def _force_eager_attn(module, name: str) -> None:
    if module is None:
        return
    try:
        if hasattr(module, "set_attn_implementation"):
            module.set_attn_implementation("eager")
    except Exception as e:
        print(f"[warn] failed set_attn_implementation(eager) for {name}: {e}")
    cfg = getattr(module, "config", None)
    if cfg is not None:
        for attr in ("attn_implementation", "_attn_implementation", "_attn_implementation_internal"):
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, "eager")
                except Exception:
                    pass


def build_prompt(
    tokenizer,
    image_token_index: int,
    task: str,
    context: str | None,
    max_length: int,
):
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id or 0
    task_token = TASK_TOKENS.get(task, "<CAPTION_LONG>")
    task_ids = tokenizer(task_token, add_special_tokens=False)["input_ids"]
    if not task_ids:
        task_ids = [tokenizer.unk_token_id]
    prompt_ids = [bos_id] + task_ids + [image_token_index]
    if max_length > 0:
        prompt_ids = prompt_ids[:max_length]
    ctx_ids: List[int] = []
    ctx_text = (context or "").strip()
    if ctx_text:
        # Enforce paragraph truncation at eval time so long context cannot dominate visual tokens.
        ctx_ids = tokenizer(
            "\n" + ctx_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        if max_length > 0:
            max_ctx = max(0, max_length - len(prompt_ids))
            ctx_ids = ctx_ids[:max_ctx]
    return prompt_ids + ctx_ids


def generate_one(model, tokenizer, input_ids, images, max_new_tokens, min_new_tokens):
    attn = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attn,
            images=images,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
        )
    if out.dim() == 1:
        out = out.unsqueeze(0)
    input_len = int(input_ids.shape[1])
    has_prefix = out.shape[1] >= input_len and torch.equal(out[:, :input_len], input_ids)
    gen_ids = out[:, input_len:] if has_prefix else out
    full_ids = torch.cat([input_ids, gen_ids], dim=1)
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0], full_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="checkpoints/phi-sig")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--visual_ckpt", type=str, default="checkpoints/visual_student_scistruct_scicap_full_v2/ckpt_last.pt")
    ap.add_argument("--data_json", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--min_mask_count", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--min_new_tokens", type=int, default=10)
    ap.add_argument("--input_max_length", type=int, default=384)
    ap.add_argument("--vision_pool", type=int, default=3)
    ap.add_argument("--vision_pool_mode", type=str, default="avg", choices=["avg", "max"])
    ap.add_argument("--region_token_scale", type=float, default=1.0)
    ap.add_argument("--region_attn_bias_layers", type=int, default=0)
    ap.add_argument("--region_attn_bias_beta", type=float, default=0.0)
    ap.add_argument("--region_attn_bias_task", type=str, default="explain", choices=["short", "long", "desc", "explain"])
    ap.add_argument("--paragraph_attn_neg_bias_gamma", type=float, default=0.0)
    ap.add_argument("--attn_probe_layers", type=int, default=8)
    ap.add_argument("--attn_probe_gen_steps", type=int, default=32)
    ap.add_argument("--lora_layers", type=int, default=12)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="outputs/stage3_explain_eval_8.json")
    ap.add_argument("--model_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--min_caption_len", type=int, default=40)
    ap.add_argument("--max_masks", type=int, default=16)
    ap.add_argument("--mask_min_area_ratio", type=float, default=0.002)
    ap.add_argument("--mask_max_area_ratio", type=float, default=0.85)
    ap.add_argument("--path_replace", action="append", default=[], help="from=to")
    args = ap.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    load_dtype = dtype_map.get(args.model_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=load_dtype if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )
    _force_eager_attn(model, "model")
    _force_eager_attn(getattr(model, "language_model", None), "language_model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        model_max_length=model.config.tokenizer_model_max_length,
        padding_side=model.config.tokenizer_padding_side,
    )

    extra_tokens = []
    for t in TASK_TOKENS.values():
        if t not in tokenizer.get_vocab():
            extra_tokens.append(t)
    if extra_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": extra_tokens})
        model.resize_token_embeddings(len(tokenizer))

    if args.lora_layers > 0 and args.lora_r > 0:
        apply_lora_to_llm(
            model.language_model,
            num_layers=args.lora_layers,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    student = CLIPVisionTower(output_attentions=False)
    ckpt = torch.load(args.visual_ckpt, map_location="cpu")
    student.load_state_dict(ckpt.get("model", {}), strict=False)
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    model.vision_tower = StudentVisionTower(
        student,
        pool_size=args.vision_pool,
        pool_mode=args.vision_pool_mode,
        max_region_tokens=args.max_masks,
        region_token_scale=args.region_token_scale,
    )
    model.config.vision_hidden_size = 768
    if hasattr(model.config, "vision_config"):
        try:
            model.config.vision_config.hidden_size = 768
        except Exception:
            pass
    # ensure connector matches 768-dim vision tokens
    model.config.connector_type = "mlp2x_gelu"
    model.connector = model.connector.__class__(model.config)
    model._region_attn_bias_ranges = None
    wrapped_region_bias = bind_region_attention_bias(model, args.region_attn_bias_layers)
    if wrapped_region_bias > 0 and (args.region_attn_bias_beta > 0 or args.paragraph_attn_neg_bias_gamma > 0):
        print(
            f"[info] eval region attention key-bias active: layers={args.region_attn_bias_layers} "
            f"beta={args.region_attn_bias_beta} paragraph_gamma={args.paragraph_attn_neg_bias_gamma} "
            f"task={args.region_attn_bias_task}"
        )

    def _bind_encode_images_with_region_mask(mm_model):
        def _encode_images(images):
            kwargs = {}
            kwargs["vision_feature_layer"] = mm_model.config.vision_feature_layer
            kwargs["vision_feature_select_strategy"] = mm_model.config.vision_feature_select_strategy
            images_local = images.to(device=mm_model.device, dtype=mm_model.dtype)
            image_features = mm_model.vision_tower(images_local, **kwargs)
            image_features = mm_model.connector(image_features)
            reg_mask = getattr(mm_model.vision_tower, "last_region_valid_mask", None)
            if reg_mask is not None and reg_mask.numel() > 0:
                reg_mask = reg_mask.to(device=image_features.device, dtype=image_features.dtype)
                k = reg_mask.shape[1]
                # Region tokens are prefixed before patch tokens.
                image_features[:, :k, :] = image_features[:, :k, :] * reg_mask.unsqueeze(-1)
            return image_features

        mm_model.encode_images = _encode_images

    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    model_state = model.state_dict()
    drop = []
    for k, v in list(sd.items()):
        if k in model_state and model_state[k].shape != v.shape:
            drop.append(k)
            sd.pop(k, None)
    if drop:
        print(f"[warn] dropped {len(drop)} keys with shape mismatch: {drop[:6]}{'...' if len(drop) > 6 else ''}")
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model_dtype = next(model.parameters()).dtype
    model.connector.to(device=device, dtype=model_dtype)
    _bind_encode_images_with_region_mask(model)
    model.eval()
    image_tokens = _estimate_image_tokens(args.vision_pool)
    image_tokens_total = image_tokens + args.max_masks

    def _set_region_attention_bias_for_inputs(
        input_ids_local: torch.Tensor,
        scale_local: str,
        has_region: bool,
        paragraph_tokens=0,
    ) -> None:
        if args.region_attn_bias_layers <= 0:
            model._region_attn_bias_ranges = None
            return
        bsz = int(input_ids_local.size(0))
        ranges: List[Optional[List[tuple[int, int, float]]]] = [None] * bsz
        any_bias = False
        for bi in range(bsz):
            pos = (input_ids_local[bi] == model.config.image_token_index).nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue
            img_pos = int(pos[0].item())
            entries: List[tuple[int, int, float]] = []
            if (
                args.region_attn_bias_beta > 0
                and bool(has_region)
                and (not scale_local or scale_local == args.region_attn_bias_task)
            ):
                rs = img_pos
                re = img_pos + args.max_masks
                if re > rs:
                    entries.append((rs, re, float(args.region_attn_bias_beta)))
            if (
                args.paragraph_attn_neg_bias_gamma > 0
                and (not scale_local or scale_local == args.region_attn_bias_task)
            ):
                if isinstance(paragraph_tokens, list):
                    p_tok = int(paragraph_tokens[bi]) if bi < len(paragraph_tokens) else 0
                else:
                    p_tok = int(paragraph_tokens)
                if p_tok > 0:
                    ps = img_pos + image_tokens_total
                    pe = ps + p_tok
                    if pe > ps:
                        entries.append((ps, pe, -float(args.paragraph_attn_neg_bias_gamma)))
            if entries:
                ranges[bi] = entries
                any_bias = True
        if not any_bias:
            model._region_attn_bias_ranges = None
            return
        model._region_attn_bias_ranges = ranges

    def _clear_region_attention_bias() -> None:
        model._region_attn_bias_ranges = None

    def _safe_range(start: int, end: int, max_len: int) -> Optional[tuple[int, int]]:
        s = max(0, min(int(start), max_len))
        e = max(0, min(int(end), max_len))
        if e <= s:
            return None
        return s, e

    def _collect_attn_mass(
        prompt_ids: List[int],
        full_ids: torch.Tensor,
        images_t: torch.Tensor,
        regions_t: List[Dict],
        has_region: bool,
        has_ctx: bool,
        paragraph_tokens: int = 0,
    ) -> Dict:
        if full_ids.dim() == 2:
            full_ids = full_ids[0]
        input_ids_f = full_ids.unsqueeze(0).to(device)
        attn_f = torch.ones_like(input_ids_f, dtype=torch.long, device=device)
        model.vision_tower.set_regions([regions_t], drop_one_region=False)
        _set_region_attention_bias_for_inputs(input_ids_f, "explain", has_region, paragraph_tokens=paragraph_tokens)
        with torch.no_grad():
            out_f = model(
                input_ids=input_ids_f,
                attention_mask=attn_f,
                images=images_t,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        _clear_region_attention_bias()
        attentions = list(out_f.attentions or [])
        if not attentions:
            return {"attn_to_region": 0.0, "attn_to_img": 0.0, "attn_to_textctx": 0.0, "layers": 0}
        try:
            img_pos = prompt_ids.index(model.config.image_token_index)
        except ValueError:
            return {"attn_to_region": 0.0, "attn_to_img": 0.0, "attn_to_textctx": 0.0, "layers": 0}
        ctx_tokens = max(0, len(prompt_ids) - (img_pos + 1))
        prompt_expanded_len = len(prompt_ids) - 1 + image_tokens_total
        q_start = prompt_expanded_len

        layer_stats = []
        n_layers = min(args.attn_probe_layers, len(attentions))
        for li in range(n_layers):
            att = attentions[li]
            if att is None or att.dim() != 4 or att.size(0) < 1:
                continue
            a = att[0]  # [heads, q_len, k_len]
            q_len = int(a.shape[1])
            k_len = int(a.shape[2])
            q_end = min(q_len, q_start + max(1, int(args.attn_probe_gen_steps)))
            q_rng = _safe_range(q_start, q_end, q_len)
            if q_rng is None:
                continue
            q0, q1 = q_rng
            q_att = a[:, q0:q1, :]
            if q_att.numel() == 0:
                continue
            img_end = img_pos + image_tokens_total
            reg_start = img_pos
            reg_end = img_pos + args.max_masks
            img_patch_start = reg_end
            img_patch_end = img_end
            ctx_start = img_end
            ctx_end = ctx_start + ctx_tokens
            r_rng = _safe_range(reg_start, reg_end, k_len) if has_region else None
            i_rng = _safe_range(img_patch_start, img_patch_end, k_len)
            t_rng = _safe_range(ctx_start, ctx_end, k_len) if has_ctx else None
            m_region = float(q_att[:, :, r_rng[0]:r_rng[1]].sum(dim=-1).mean().item()) if r_rng is not None else 0.0
            m_img = float(q_att[:, :, i_rng[0]:i_rng[1]].sum(dim=-1).mean().item()) if i_rng is not None else 0.0
            m_ctx = float(q_att[:, :, t_rng[0]:t_rng[1]].sum(dim=-1).mean().item()) if t_rng is not None else 0.0
            layer_stats.append({"layer": li, "attn_to_region": m_region, "attn_to_img": m_img, "attn_to_textctx": m_ctx})

        if not layer_stats:
            return {"attn_to_region": 0.0, "attn_to_img": 0.0, "attn_to_textctx": 0.0, "layers": 0}
        return {
            "attn_to_region": float(sum(x["attn_to_region"] for x in layer_stats) / len(layer_stats)),
            "attn_to_img": float(sum(x["attn_to_img"] for x in layer_stats) / len(layer_stats)),
            "attn_to_textctx": float(sum(x["attn_to_textctx"] for x in layer_stats) / len(layer_stats)),
            "layers": len(layer_stats),
            "per_layer": layer_stats,
        }

    replacements = []
    for item in args.path_replace:
        if "=" not in item:
            continue
        src, dst = item.split("=", 1)
        replacements.append((src, dst))

    ds = SciStructExplainDataset(
        split_json=args.data_json,
        min_caption_len=args.min_caption_len,
        max_items=None,
        context_mode="paragraph",
        max_masks=args.max_masks,
        mask_min_area_ratio=args.mask_min_area_ratio,
        mask_max_area_ratio=args.mask_max_area_ratio,
        region_grid_size=int((max(1, _estimate_image_tokens(args.vision_pool))) ** 0.5),
        path_replace=replacements,
    )
    if len(ds) == 0:
        raise RuntimeError("no samples found")

    indices = list(range(len(ds)))
    random.shuffle(indices)
    def _unpack_item(it):
        if isinstance(it, tuple):
            if len(it) >= 6:
                img, text, scale, context, regions, _ctx_meta = it[:6]
                return img, text, scale, context, regions
            if len(it) >= 5:
                img, text, scale, context, regions = it[:5]
                return img, text, scale, context, regions
            if len(it) >= 4:
                img, text, scale, context = it[:4]
                return img, text, scale, context, []
        raise RuntimeError(f"unexpected dataset item format: {type(it)}")

    picked = []
    for idx in indices:
        _, _, _, _, regions = _unpack_item(ds[idx])
        if len(regions or []) < args.min_mask_count:
            continue
        picked.append(idx)
        if len(picked) >= args.num_samples:
            break
    if not picked:
        raise RuntimeError(f"no samples found with mask_count >= {args.min_mask_count}")

    results = []
    for idx in picked:
        img, caption, scale, context, regions = _unpack_item(ds[idx])
        pixel_values = student.preprocess([img])["pixel_values"].to(device)
        zero_images = torch.zeros_like(pixel_values)

        # prompts
        ids_region_only = build_prompt(
            tokenizer,
            model.config.image_token_index,
            "explain",
            None,
            args.input_max_length,
        )
        ids_img_region_ctx = build_prompt(
            tokenizer,
            model.config.image_token_index,
            "explain",
            context,
            args.input_max_length,
        )
        ids_img_ctx = build_prompt(
            tokenizer,
            model.config.image_token_index,
            "explain",
            context,
            args.input_max_length,
        )
        ids_ctx_only = build_prompt(
            tokenizer,
            model.config.image_token_index,
            "explain",
            context,
            args.input_max_length,
        )
        def _prompt_ctx_tokens(ids_local: List[int]) -> int:
            try:
                ip = ids_local.index(model.config.image_token_index)
            except ValueError:
                return 0
            return max(0, len(ids_local) - (ip + 1))
        ctx_tok_region_only = _prompt_ctx_tokens(ids_region_only)
        ctx_tok_img_region_ctx = _prompt_ctx_tokens(ids_img_region_ctx)
        ctx_tok_img_ctx = _prompt_ctx_tokens(ids_img_ctx)
        ctx_tok_ctx_only = _prompt_ctx_tokens(ids_ctx_only)

        print(
            f"[debug] sample={idx} len(region_only)={len(ids_region_only)} "
            f"len(img_region_ctx)={len(ids_img_region_ctx)} len(img_ctx)={len(ids_img_ctx)} len(ctx_only)={len(ids_ctx_only)} "
            f"region_count={len(regions or [])} expected~{min(len(regions or []), args.max_masks)}"
        )

        model.vision_tower.set_regions([regions], drop_one_region=False)
        input_ids = torch.tensor([ids_region_only], device=device)
        _set_region_attention_bias_for_inputs(
            input_ids,
            "explain",
            has_region=bool(regions),
            paragraph_tokens=ctx_tok_region_only,
        )
        out_region_only, ids_region_only_out = generate_one(model, tokenizer, input_ids, pixel_values, args.max_new_tokens, args.min_new_tokens)
        _clear_region_attention_bias()
        attn_region_only = _collect_attn_mass(
            ids_region_only,
            ids_region_only_out,
            pixel_values,
            regions or [],
            has_region=bool(regions),
            has_ctx=False,
            paragraph_tokens=ctx_tok_region_only,
        )

        model.vision_tower.set_regions([regions], drop_one_region=False)
        input_ids = torch.tensor([ids_img_region_ctx], device=device)
        _set_region_attention_bias_for_inputs(
            input_ids,
            "explain",
            has_region=bool(regions),
            paragraph_tokens=ctx_tok_img_region_ctx,
        )
        out_img_region_ctx, ids_img_region_ctx_out = generate_one(model, tokenizer, input_ids, pixel_values, args.max_new_tokens, args.min_new_tokens)
        _clear_region_attention_bias()
        attn_img_region_ctx = _collect_attn_mass(
            ids_img_region_ctx,
            ids_img_region_ctx_out,
            pixel_values,
            regions or [],
            has_region=bool(regions),
            has_ctx=bool(context and context.strip()),
            paragraph_tokens=ctx_tok_img_region_ctx,
        )

        model.vision_tower.set_regions([[]], drop_one_region=False)
        input_ids = torch.tensor([ids_img_ctx], device=device)
        _set_region_attention_bias_for_inputs(
            input_ids,
            "explain",
            has_region=False,
            paragraph_tokens=ctx_tok_img_ctx,
        )
        out_img_ctx, ids_img_ctx_out = generate_one(model, tokenizer, input_ids, pixel_values, args.max_new_tokens, args.min_new_tokens)
        _clear_region_attention_bias()
        attn_img_ctx = _collect_attn_mass(
            ids_img_ctx,
            ids_img_ctx_out,
            pixel_values,
            [],
            has_region=False,
            has_ctx=bool(context and context.strip()),
            paragraph_tokens=ctx_tok_img_ctx,
        )

        model.vision_tower.set_regions([[]], drop_one_region=False)
        input_ids = torch.tensor([ids_ctx_only], device=device)
        _set_region_attention_bias_for_inputs(
            input_ids,
            "explain",
            has_region=False,
            paragraph_tokens=ctx_tok_ctx_only,
        )
        out_ctx_only, ids_ctx_only_out = generate_one(model, tokenizer, input_ids, zero_images, args.max_new_tokens, args.min_new_tokens)
        _clear_region_attention_bias()
        attn_ctx_only = _collect_attn_mass(
            ids_ctx_only,
            ids_ctx_only_out,
            zero_images,
            [],
            has_region=False,
            has_ctx=bool(context and context.strip()),
            paragraph_tokens=ctx_tok_ctx_only,
        )

        results.append({
            "image": getattr(img, "filename", ""),
            "caption": caption,
            "paragraph": context,
            "mask_count": len(regions or []),
            "region_only": out_region_only,
            "image_region": out_region_only,
            "image_region_paragraph": out_img_region_ctx,
            "image_paragraph": out_img_ctx,
            "paragraph_only": out_ctx_only,
            "attn_mass": {
                "region_only": attn_region_only,
                "image_region_paragraph": attn_img_region_ctx,
                "image_paragraph": attn_img_ctx,
                "paragraph_only": attn_ctx_only,
            },
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

