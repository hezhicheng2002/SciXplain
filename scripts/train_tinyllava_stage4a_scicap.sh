#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCICAP_ROOT="${SCICAP_ROOT:-${ROOT}/dataset/scicap_mlbcap_node_diagram_v2}"
SCICAP_TRAIN_JSON="${SCICAP_TRAIN_JSON:-$SCICAP_ROOT/dataset_split/train.json}"
SCICAP_VAL_JSON="${SCICAP_VAL_JSON:-$SCICAP_ROOT/dataset_split/val.json}"

OUT_DIR="${OUT_DIR:-$ROOT/checkpoints/tinyllava_phi_siglip_stage4a_scicap}"
INIT_CKPT_DEFAULT="${INIT_CKPT_DEFAULT:-$ROOT/checkpoints/tinyllava_phi_siglip_stage4_multitask/ckpt_last.pt}"
INIT_CKPT="${INIT_CKPT:-$INIT_CKPT_DEFAULT}"
VISUAL_CKPT="${VISUAL_CKPT:-$ROOT/checkpoints/visual_student_scistruct_scicap_full_v2/ckpt_last.pt}"

MAX_STEPS="${MAX_STEPS:-1200}"
VAL_EVERY="${VAL_EVERY:-200}"
SAMPLE_EVERY="${SAMPLE_EVERY:-200}"
SAMPLE_OUT_DIR="${SAMPLE_OUT_DIR:-$ROOT/outputs/stage4a_samples_scicap}"

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
echo "[info][stage4a] init_ckpt=$INIT_CKPT"
echo "[info][stage4a] out_dir=$OUT_DIR"
echo "[info][stage4a] visual_ckpt=$VISUAL_CKPT"

python "$ROOT/scixplain/tools/train_tinyllava_image_only.py" \
  --model_path "$ROOT/checkpoints/phi-sig" \
  --model_dtype bfloat16 \
  --visual_ckpt "$VISUAL_CKPT" \
  --init_ckpt "$INIT_CKPT" \
  --strict_connector_load \
  --strict_vocab_size_match \
  --dataset scicap \
  --train_json "$SCICAP_TRAIN_JSON" \
  --val_json "$SCICAP_VAL_JSON" \
  --out_dir "$OUT_DIR" \
  --batch_size 2 \
  --grad_accum 2 \
  --warmup_steps 0 \
  --max_steps "$MAX_STEPS" \
  --lr 1e-5 \
  --connector_lr 1e-5 \
  --lora_lr 1e-5 \
  --context_mode none \
  --context_dropout 0.0 \
  --paragraph_token_dropout 0.0 \
  --scicap_prompt_style scicap_metric \
  --scicap_fixed_task_loss \
  --scicap_task_context_routing caption_para_desc_ocr \
  --scicap_loss_w_short 1.5 \
  --scicap_loss_w_long 1.5 \
  --scicap_loss_w_desc 1.0 \
  --enable_task_style_tokens \
  --caption_formula_penalty_weight 0.15 \
  --caption_block_formula_in_decode \
  --caption_context_cov_weight 0.15 \
  --caption_context_cov_min 0.06 \
  --desc_ocr_cov_weight 0.2 \
  --desc_ocr_cov_min 0.08 \
  --disable_explain_gate_non_explain \
  --max_ctx_tokens 128 \
  --max_length 384 \
  --max_target_short 48 \
  --max_target_long 128 \
  --max_target_desc 320 \
  --bucket_by_length \
  --bucket_bins 256,288,320,352,384 \
  --val_every "$VAL_EVERY" \
  --val_batch_size 2 \
  --val_num_batches 20 \
  --sample_every "$SAMPLE_EVERY" \
  --sample_num 8 \
  --sample_mode tasks \
  --sample_tasks short,long,desc \
  --sample_max_new_tokens 96 \
  --sample_max_new_tokens_short 48 \
  --sample_max_new_tokens_long 96 \
  --sample_max_new_tokens_desc 160 \
  --sample_min_new_tokens 10 \
  --eval_num_beams 5 \
  --eval_length_penalty 1.0 \
  --eval_repetition_penalty 1.1 \
  --eval_no_repeat_ngram_size 3 \
  --sample_out_dir "$SAMPLE_OUT_DIR" \
  --vision_pool 3 \
  --max_masks 8 \
  --region_token_scale 1.5 \
  --region_attn_bias_layers 0 \
  --region_attn_bias_beta 0.0 \
  --paragraph_attn_neg_bias_gamma 0.0 \
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
  --log_every 20 \
  --save_every 200

