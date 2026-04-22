#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCISTRUCT_TRAIN_JSON="${SCISTRUCT_TRAIN_JSON:-$ROOT/dataset/SciStruct/dataset_split_811/train.json}"
SCISTRUCT_VAL_JSON="${SCISTRUCT_VAL_JSON:-$ROOT/dataset/SciStruct/dataset_split_811/val.json}"

OUT_DIR="${OUT_DIR:-$ROOT/checkpoints/tinyllava_phi_siglip_stage4b_scistruct_explain}"
INIT_CKPT_DEFAULT="${INIT_CKPT_DEFAULT:-$ROOT/checkpoints/tinyllava_phi_siglip_stage4a_scicap/ckpt_last.pt}"
INIT_CKPT="${INIT_CKPT:-$INIT_CKPT_DEFAULT}"
VISUAL_CKPT="${VISUAL_CKPT:-$ROOT/checkpoints/visual_student_scistruct_scicap_full_v2_ablate_no_dino_0209_155830/ckpt_last.pt}"

MAX_STEPS="${MAX_STEPS:-1200}"
VAL_EVERY="${VAL_EVERY:-200}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
SAMPLE_EVERY="${SAMPLE_EVERY:-200}"
SAMPLE_OUT_DIR="${SAMPLE_OUT_DIR:-$ROOT/outputs/stage4b_samples_explain}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_LENGTH="${MAX_LENGTH:-768}"
BUCKET_BINS="${BUCKET_BINS:-640,704,768}"
MAX_MASKS="${MAX_MASKS:-8}"
EXPLAIN_REGION_REQUIRED_LAMBDA="${EXPLAIN_REGION_REQUIRED_LAMBDA:-0.5}"
EXPLAIN_REGION_REQUIRED_MARGIN="${EXPLAIN_REGION_REQUIRED_MARGIN:-0.0}"
LOG_EVERY="${LOG_EVERY:-20}"
LOG_ATTN="${LOG_ATTN:-0}"
DEBUG_EXPLAIN_ATTN_CONSISTENCY="${DEBUG_EXPLAIN_ATTN_CONSISTENCY:-0}"
EXPLAIN_ATTN_LAST_K_LAYERS="${EXPLAIN_ATTN_LAST_K_LAYERS:-4}"
ENABLE_EXPLAIN_REGION_ADAPTER="${ENABLE_EXPLAIN_REGION_ADAPTER:-1}"
CONTEXT_MODE="${CONTEXT_MODE:-paragraph_ocr_desc}"
EXPLAIN_ALLOW_CONTEXT_KV="${EXPLAIN_ALLOW_CONTEXT_KV:-1}"
FORBID_PARAGRAPH_KV="${FORBID_PARAGRAPH_KV:-1}"
USE_AUTO_DESC="${USE_AUTO_DESC:-1}"
USE_OCR="${USE_OCR:-1}"
EXPLAIN_CTX_MIN_ADESC_TOKENS="${EXPLAIN_CTX_MIN_ADESC_TOKENS:-64}"
EXPLAIN_CTX_MAX_OCR_TOKENS="${EXPLAIN_CTX_MAX_OCR_TOKENS:-128}"
MAX_CTX_TOKENS_EXPLAIN="${MAX_CTX_TOKENS_EXPLAIN:-192}"
CONTEXT_DROPOUT="${CONTEXT_DROPOUT:-0.65}"
PARAGRAPH_TOKEN_DROPOUT="${PARAGRAPH_TOKEN_DROPOUT:-0.3}"
REGION_ATTN_BIAS_LAYERS="${REGION_ATTN_BIAS_LAYERS:-8}"
REGION_ATTN_BIAS_BETA="${REGION_ATTN_BIAS_BETA:-0.0}"
PARAGRAPH_ATTN_NEG_BIAS_GAMMA="${PARAGRAPH_ATTN_NEG_BIAS_GAMMA:-0.0}"
EXPLAIN_HARD_PARAGRAPH_KV_BIAS="${EXPLAIN_HARD_PARAGRAPH_KV_BIAS:--10000}"
EXPLAIN_METRICS_MINIMAL_ONLY="${EXPLAIN_METRICS_MINIMAL_ONLY:-1}"
PATH_REPLACE="${PATH_REPLACE:-}"

UNFREEZE_LLM_FROM="${UNFREEZE_LLM_FROM:-16}"
UNFREEZE_LLM_TO="${UNFREEZE_LLM_TO:-32}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

if [[ -f "$OUT_DIR/ckpt_last.pt" ]]; then
  INIT_CKPT="$OUT_DIR/ckpt_last.pt"
fi
echo "[info][stage4b] init_ckpt=$INIT_CKPT"
echo "[info][stage4b] out_dir=$OUT_DIR"
echo "[info][stage4b] visual_ckpt=$VISUAL_CKPT"

EXTRA_ARGS=()
if [[ "$LOG_ATTN" == "1" ]]; then
  EXTRA_ARGS+=(--log_attn)
fi
if [[ "$DEBUG_EXPLAIN_ATTN_CONSISTENCY" == "1" ]]; then
  EXTRA_ARGS+=(--debug_explain_attn_consistency)
fi
if [[ "$ENABLE_EXPLAIN_REGION_ADAPTER" == "1" ]]; then
  EXTRA_ARGS+=(--enable_explain_region_adapter)
fi
if [[ "$EXPLAIN_ALLOW_CONTEXT_KV" == "1" ]]; then
  EXTRA_ARGS+=(--explain_allow_context_kv)
fi
if [[ "$FORBID_PARAGRAPH_KV" == "1" ]]; then
  EXTRA_ARGS+=(--explain_hard_paragraph_kv_gate --explain_hard_paragraph_kv_bias "$EXPLAIN_HARD_PARAGRAPH_KV_BIAS")
fi
if [[ "$USE_AUTO_DESC" != "1" ]]; then
  EXTRA_ARGS+=(--disable_explain_adesc)
fi
if [[ "$USE_OCR" != "1" ]]; then
  EXTRA_ARGS+=(--disable_explain_ocr)
fi
if [[ "$EXPLAIN_METRICS_MINIMAL_ONLY" == "1" ]]; then
  EXTRA_ARGS+=(--explain_metrics_minimal_only)
fi
if [[ -n "$PATH_REPLACE" ]]; then
  EXTRA_ARGS+=(--path_replace "$PATH_REPLACE")
fi

python "$ROOT/scixplain/tools/train_tinyllava_image_only.py" \
  --model_path "$ROOT/checkpoints/phi-sig" \
  --model_dtype bfloat16 \
  --visual_ckpt "$VISUAL_CKPT" \
  --init_ckpt "$INIT_CKPT" \
  --strict_connector_load \
  --strict_vocab_size_match \
  --dataset scistruct_explain \
  --train_json "$SCISTRUCT_TRAIN_JSON" \
  --val_json "$SCISTRUCT_VAL_JSON" \
  --fixed_task explain \
  --out_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --warmup_steps 0 \
  --max_steps "$MAX_STEPS" \
  --lr 1e-5 \
  --connector_lr 1e-5 \
  --lora_lr 1e-5 \
  --min_caption_len 40 \
  --max_masks "$MAX_MASKS" \
  --mask_min_area_ratio 0.002 \
  --mask_max_area_ratio 0.85 \
  --contrastive_type mask_drop \
  --contrastive_mode cosine \
  --contrastive_margin 0.3 \
  --contrastive_weight 0.5 \
  --explain_region_required_loss \
  --explain_region_required_lambda "$EXPLAIN_REGION_REQUIRED_LAMBDA" \
  --explain_region_required_margin "$EXPLAIN_REGION_REQUIRED_MARGIN" \
  --image_dropout_prob 0.6 \
  --image_dropout_mode zero \
  --context_mode "$CONTEXT_MODE" \
  --context_dropout "$CONTEXT_DROPOUT" \
  --paragraph_token_dropout "$PARAGRAPH_TOKEN_DROPOUT" \
  --enable_task_style_tokens \
  --max_ctx_tokens 128 \
  --max_ctx_tokens_explain "$MAX_CTX_TOKENS_EXPLAIN" \
  --explain_ctx_min_adesc_tokens "$EXPLAIN_CTX_MIN_ADESC_TOKENS" \
  --explain_ctx_max_ocr_tokens "$EXPLAIN_CTX_MAX_OCR_TOKENS" \
  --consistency_mode anchor \
  --consistency_weight 0.2 \
  --consistency_margin 0.1 \
  --anchor_struct \
  --anchor_rel \
  --max_length "$MAX_LENGTH" \
  --bucket_by_length \
  --bucket_bins "$BUCKET_BINS" \
  --val_every "$VAL_EVERY" \
  --val_batch_size "$VAL_BATCH_SIZE" \
  --val_num_batches "$VAL_NUM_BATCHES" \
  --sample_every "$SAMPLE_EVERY" \
  --sample_num 8 \
  --sample_mode explain_diag \
  --sample_tasks explain \
  --sample_max_new_tokens 96 \
  --sample_min_new_tokens 10 \
  --explain_attn_last_k_layers "$EXPLAIN_ATTN_LAST_K_LAYERS" \
  --eval_num_beams 5 \
  --eval_length_penalty 1.0 \
  --eval_repetition_penalty 1.1 \
  --eval_no_repeat_ngram_size 3 \
  --sample_out_dir "$SAMPLE_OUT_DIR" \
  --vision_pool 3 \
  --region_token_scale 1.5 \
  --region_attn_bias_layers "$REGION_ATTN_BIAS_LAYERS" \
  --region_attn_bias_beta "$REGION_ATTN_BIAS_BETA" \
  --paragraph_attn_neg_bias_gamma "$PARAGRAPH_ATTN_NEG_BIAS_GAMMA" \
  --region_attn_bias_task explain \
  --freeze_vision \
  --freeze_region_mechanism \
  --freeze_llm \
  --unfreeze_embeddings \
  --task_embed_only \
  --unfreeze_llm_from "$UNFREEZE_LLM_FROM" \
  --unfreeze_llm_to "$UNFREEZE_LLM_TO" \
  --lora_layers 0 \
  --lora_r 8 \
  --lora_alpha 8 \
  --grad_checkpoint \
  --log_every "$LOG_EVERY" \
  --save_every 200 \
  "${EXTRA_ARGS[@]}"

