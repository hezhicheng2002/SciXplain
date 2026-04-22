#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%m%d_%H%M%S)}"
GPU="${GPU:-4}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/tinyllava}"
mkdir -p "$LOG_DIR"

SCICAP_ROOT="${SCICAP_ROOT:-${ROOT}/dataset/scicap_mlbcap_node_diagram_v2}"
SCICAP_TRAIN_JSON="${SCICAP_TRAIN_JSON:-$SCICAP_ROOT/dataset_split/train.json}"
SCICAP_VAL_JSON="${SCICAP_VAL_JSON:-$SCICAP_ROOT/dataset_split/val.json}"
SCICAP_TEST_JSON="${SCICAP_TEST_JSON:-$SCICAP_ROOT/dataset_split/test.json}"
SCISTRUCT_TRAIN_JSON="${SCISTRUCT_TRAIN_JSON:-$ROOT/dataset/SciStruct/dataset_split_811/train.json}"
SCISTRUCT_VAL_JSON="${SCISTRUCT_VAL_JSON:-$ROOT/dataset/SciStruct/dataset_split_811/val.json}"
SCISTRUCT_TEST_JSON="${SCISTRUCT_TEST_JSON:-$ROOT/dataset/SciStruct/dataset_split_811/test.json}"

VISUAL_CKPT="${VISUAL_CKPT:-$ROOT/checkpoints/visual_student_scistruct_scicap_full_v2_ablate_no_dino_0209_155830/ckpt_last.pt}"
STAGE4A_INIT_CKPT="${STAGE4A_INIT_CKPT:-$ROOT/checkpoints/tinyllava_phi_siglip_imgonly_scicap/ckpt_last.pt}"
if [[ ! -f "$STAGE4A_INIT_CKPT" ]]; then
  STAGE4A_INIT_CKPT="$ROOT/checkpoints/tinyllava_phi_siglip_stage4a_scicap_full_no_dino_0210_233417/ckpt_last.pt"
fi

STAGE4A_STEPS="${STAGE4A_STEPS:-2200}"
STAGE4B_STEPS="${STAGE4B_STEPS:-1400}"
STAGE4B_EXPLAIN_RATIO="${STAGE4B_EXPLAIN_RATIO:-0.24}"
STAGE4B_SCICAP_EXPLAIN_RATIO="${STAGE4B_SCICAP_EXPLAIN_RATIO:-0.15}"
STAGE4B_EPOCH_SIZE="${STAGE4B_EPOCH_SIZE:-12000}"

OUT_STAGE4A="${OUT_STAGE4A:-$ROOT/checkpoints/tinyllava_phi_siglip_stage4a_scicap_unified_${TS}}"
OUT_STAGE4B="${OUT_STAGE4B:-$ROOT/checkpoints/tinyllava_phi_siglip_stage4b_unified_gpu4_${TS}}"
SAMPLE_STAGE4A="${SAMPLE_STAGE4A:-$ROOT/outputs/stage4a_samples_unified_${TS}}"
SAMPLE_STAGE4B="${SAMPLE_STAGE4B:-$ROOT/outputs/stage4b_samples_unified_${TS}}"

BENCH_TMP_OUT="${BENCH_TMP_OUT:-$ROOT/checkpoints/stage4_unified_evaltmp_${TS}}"
BENCH_SAMPLE_DIR="${BENCH_SAMPLE_DIR:-$ROOT/outputs/stage4_unified_test_preds_${TS}}"
BENCH_JSON="${BENCH_JSON:-$ROOT/outputs/stage4_unified_test_benchmark_${TS}.json}"
BENCH_PAIR_JSONL="${BENCH_PAIR_JSONL:-$ROOT/outputs/stage4_unified_test_pairs_${TS}.jsonl}"
BENCH_EXTRA_JSON="${BENCH_EXTRA_JSON:-$ROOT/outputs/stage4_unified_test_benchmark_${TS}_extra_metrics.json}"
EXPLAIN_EVAL_JSON="${EXPLAIN_EVAL_JSON:-$ROOT/outputs/stage4_unified_explain_eval_${TS}.json}"
EXPLAIN_SUMMARY_JSON="${EXPLAIN_SUMMARY_JSON:-$ROOT/outputs/stage4_unified_explain_summary_${TS}.json}"

LOG_STAGE4A="${LOG_STAGE4A:-$LOG_DIR/train_stage4a_unified_${TS}.log}"
LOG_STAGE4B="${LOG_STAGE4B:-$LOG_DIR/train_stage4b_unified_gpu4_${TS}.log}"
LOG_BENCH="${LOG_BENCH:-$LOG_DIR/eval_stage4_unified_gpu4_${TS}.log}"
LOG_EXPLAIN="${LOG_EXPLAIN:-$LOG_DIR/eval_stage4_unified_explain_gpu4_${TS}.log}"

if [[ ! -f "$VISUAL_CKPT" ]]; then
  echo "[fatal] visual ckpt missing: $VISUAL_CKPT" >&2
  exit 2
fi
if [[ ! -f "$STAGE4A_INIT_CKPT" ]]; then
  echo "[fatal] stage4A init ckpt missing: $STAGE4A_INIT_CKPT" >&2
  exit 2
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

echo "[start] ts=$TS gpu=$GPU"
echo "[cfg] visual_ckpt=$VISUAL_CKPT"
echo "[cfg] stage4A_init=$STAGE4A_INIT_CKPT"

# ------------------
# Stage-4A: SciCap
# ------------------
python "$ROOT/scixplain/tools/train_tinyllava_image_only.py" \
  --model_path "$ROOT/checkpoints/phi-sig" \
  --model_dtype bfloat16 \
  --visual_ckpt "$VISUAL_CKPT" \
  --init_ckpt "$STAGE4A_INIT_CKPT" \
  --dataset scicap \
  --train_json "$SCICAP_TRAIN_JSON" \
  --val_json "$SCICAP_VAL_JSON" \
  --out_dir "$OUT_STAGE4A" \
  --batch_size 2 \
  --grad_accum 2 \
  --warmup_steps 0 \
  --max_steps "$STAGE4A_STEPS" \
  --lr 1e-5 \
  --connector_lr 1e-5 \
  --lora_lr 1e-5 \
  --context_mode none \
  --context_dropout 0.0 \
  --paragraph_token_dropout 0.0 \
  --scicap_prompt_style scicap_metric_desc_strict \
  --scicap_fixed_task_loss \
  --scicap_task_context_routing caption_para_desc_ocr \
  --scicap_loss_w_short 1.5 \
  --scicap_loss_w_long 1.5 \
  --scicap_loss_w_desc 1.2 \
  --enable_task_style_tokens \
  --caption_formula_penalty_weight 0.15 \
  --caption_block_formula_in_decode \
  --caption_context_cov_weight 0.15 \
  --caption_context_cov_min 0.06 \
  --desc_ocr_cov_weight 0.35 \
  --desc_ocr_cov_min 0.10 \
  --desc_struct_cov_weight 0.14 \
  --desc_struct_cov_min 0.06 \
  --desc_struct_cov_max_slots 8 \
  --desc_struct_cov_target_ratio 0.40 \
  --disable_explain_gate_non_explain \
  --max_ctx_tokens 128 \
  --max_length 384 \
  --max_target_short 48 \
  --max_target_long 128 \
  --max_target_desc 320 \
  --bucket_by_length \
  --bucket_bins 256,288,320,352,384 \
  --val_every 200 \
  --val_batch_size 2 \
  --val_num_batches 20 \
  --sample_every 200 \
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
  --sample_out_dir "$SAMPLE_STAGE4A" \
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
  --unfreeze_llm_from 16 \
  --unfreeze_llm_to 32 \
  --lora_layers 0 \
  --lora_r 8 \
  --lora_alpha 8 \
  --grad_checkpoint \
  --log_every 20 \
  --save_every 200 \
  2>&1 | tee "$LOG_STAGE4A"

if [[ ! -f "$OUT_STAGE4A/ckpt_last.pt" ]]; then
  echo "[fatal] missing stage4A ckpt: $OUT_STAGE4A/ckpt_last.pt" >&2
  exit 2
fi

# -----------------------------------
# Stage-4B: unified mixed multi-task
# -----------------------------------
python "$ROOT/scixplain/tools/train_tinyllava_image_only.py" \
  --model_path "$ROOT/checkpoints/phi-sig" \
  --model_dtype bfloat16 \
  --visual_ckpt "$VISUAL_CKPT" \
  --init_ckpt "$OUT_STAGE4A/ckpt_last.pt" \
  --dataset stage4_multitask \
  --train_json "$SCICAP_TRAIN_JSON" \
  --val_json "$SCICAP_VAL_JSON" \
  --explain_train_json "$SCISTRUCT_TRAIN_JSON" \
  --explain_val_json "$SCISTRUCT_VAL_JSON" \
  --stage4_explain_ratio "$STAGE4B_EXPLAIN_RATIO" \
  --stage4_epoch_size "$STAGE4B_EPOCH_SIZE" \
  --stage4_include_scistruct_caption \
  --stage4_scistruct_caption_json "$SCISTRUCT_TRAIN_JSON" \
  --stage4_scistruct_caption_short \
  --stage4_scistruct_caption_long \
  --stage4_scistruct_caption_min_len_short 20 \
  --stage4_scistruct_caption_min_len_long 40 \
  --stage4_include_scicap_explain \
  --stage4_scicap_explain_json "$SCICAP_TRAIN_JSON" \
  --stage4_scicap_explain_ratio "$STAGE4B_SCICAP_EXPLAIN_RATIO" \
  --stage4_scicap_explain_min_len 80 \
  --stage4_scicap_explain_use_long_fallback \
  --out_dir "$OUT_STAGE4B" \
  --batch_size 2 \
  --grad_accum 2 \
  --warmup_steps 0 \
  --max_steps "$STAGE4B_STEPS" \
  --lr 1e-5 \
  --connector_lr 1e-5 \
  --lora_lr 1e-5 \
  --context_mode paragraph_ocr_desc \
  --context_dropout 0.65 \
  --paragraph_token_dropout 0.30 \
  --max_ctx_tokens 128 \
  --max_ctx_tokens_explain 192 \
  --explain_ctx_min_adesc_tokens 64 \
  --explain_ctx_max_ocr_tokens 128 \
  --scicap_prompt_style scicap_metric_desc_strict \
  --scicap_fixed_task_loss \
  --scicap_task_context_routing caption_para_desc_ocr \
  --scicap_loss_w_short 1.5 \
  --scicap_loss_w_long 1.5 \
  --scicap_loss_w_desc 1.3 \
  --enable_task_style_tokens \
  --caption_formula_penalty_weight 0.15 \
  --caption_block_formula_in_decode \
  --caption_context_cov_weight 0.15 \
  --caption_context_cov_min 0.06 \
  --desc_ocr_cov_weight 0.38 \
  --desc_ocr_cov_min 0.10 \
  --desc_struct_cov_weight 0.14 \
  --desc_struct_cov_min 0.06 \
  --desc_struct_cov_max_slots 8 \
  --desc_struct_cov_target_ratio 0.40 \
  --disable_explain_gate_non_explain \
  --explain_region_required_loss \
  --explain_region_required_lambda 0.7 \
  --explain_region_required_margin 0.0 \
  --contrastive_type mask_drop \
  --contrastive_mode cosine \
  --contrastive_weight 0.5 \
  --image_dropout_prob 0.6 \
  --image_dropout_mode zero \
  --consistency_mode anchor \
  --consistency_weight 0.2 \
  --consistency_margin 0.1 \
  --anchor_struct \
  --anchor_rel \
  --max_length 768 \
  --bucket_by_length \
  --bucket_bins 640,704,768 \
  --val_every 200 \
  --val_batch_size 2 \
  --val_num_batches 20 \
  --sample_every 200 \
  --sample_num 8 \
  --sample_mode tasks \
  --sample_tasks short,long,desc,explain \
  --sample_max_new_tokens 96 \
  --sample_max_new_tokens_short 48 \
  --sample_max_new_tokens_long 96 \
  --sample_max_new_tokens_desc 160 \
  --sample_min_new_tokens 10 \
  --eval_num_beams 5 \
  --eval_length_penalty 1.0 \
  --eval_repetition_penalty 1.1 \
  --eval_no_repeat_ngram_size 3 \
  --sample_out_dir "$SAMPLE_STAGE4B" \
  --vision_pool 3 \
  --max_masks 8 \
  --region_token_scale 1.5 \
  --region_attn_bias_layers 8 \
  --region_attn_bias_beta 0.0 \
  --paragraph_attn_neg_bias_gamma 0.0 \
  --region_attn_bias_task explain \
  --explain_allow_context_kv \
  --explain_hard_paragraph_kv_gate \
  --explain_hard_paragraph_kv_bias -10000 \
  --freeze_vision \
  --freeze_region_mechanism \
  --freeze_llm \
  --unfreeze_embeddings \
  --task_embed_only \
  --unfreeze_llm_from 16 \
  --unfreeze_llm_to 32 \
  --lora_layers 0 \
  --lora_r 8 \
  --lora_alpha 8 \
  --grad_checkpoint \
  --log_every 20 \
  --save_every 200 \
  2>&1 | tee "$LOG_STAGE4B"

if [[ ! -f "$OUT_STAGE4B/ckpt_last.pt" ]]; then
  echo "[fatal] missing stage4B ckpt: $OUT_STAGE4B/ckpt_last.pt" >&2
  exit 2
fi

# --------------------------
# SciCap benchmark (full)
# --------------------------
GPU="$GPU" \
VISUAL_CKPT="$VISUAL_CKPT" \
CKPT_DIR="$OUT_STAGE4B" \
INIT_CKPT="$OUT_STAGE4B/ckpt_last.pt" \
EVAL_STEP="$STAGE4B_STEPS" \
SCICAP_TRAIN_JSON="$SCICAP_TRAIN_JSON" \
SCICAP_TEST_JSON="$SCICAP_TEST_JSON" \
OUT_DIR="$BENCH_TMP_OUT" \
SAMPLE_DIR="$BENCH_SAMPLE_DIR" \
METRIC_JSON="$BENCH_JSON" \
PAIR_JSONL="$BENCH_PAIR_JSONL" \
METRIC_EXTRA_JSON="$BENCH_EXTRA_JSON" \
bash "$ROOT/scripts/run_stage4a_test_benchmark_gpu4.sh" \
  2>&1 | tee "$LOG_BENCH"

# --------------------------
# Explain diagnostics (full)
# --------------------------
python "$ROOT/scixplain/tools/eval_stage3_explain_samples.py" \
  --model_path "$ROOT/checkpoints/phi-sig" \
  --ckpt "$OUT_STAGE4B/ckpt_last.pt" \
  --visual_ckpt "$VISUAL_CKPT" \
  --data_json "$SCISTRUCT_TEST_JSON" \
  --num_samples 20 \
  --min_mask_count 8 \
  --max_new_tokens 96 \
  --min_new_tokens 10 \
  --input_max_length 384 \
  --vision_pool 3 \
  --max_masks 8 \
  --region_token_scale 1.5 \
  --output "$EXPLAIN_EVAL_JSON" \
  >"$LOG_EXPLAIN" 2>&1

python - "$OUT_STAGE4B/val_log.jsonl" "$EXPLAIN_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

val_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
if val_path.exists():
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("split") == "scistruct_explain":
                rows.append(d)

summary = {"count": len(rows), "last": None, "best_gap_region_drop": None, "best_gap_drop_ocr": None}
if rows:
    summary["last"] = rows[-1]
    gap_rows = [r for r in rows if "gap_region_drop" in r]
    if gap_rows:
        summary["best_gap_region_drop"] = max(gap_rows, key=lambda x: float(x.get("gap_region_drop", -1e9)))
    ocr_rows = [r for r in rows if "gap_drop_ocr" in r]
    if ocr_rows:
        summary["best_gap_drop_ocr"] = max(ocr_rows, key=lambda x: float(x.get("gap_drop_ocr", -1e9)))

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[saved] {out_path}")
PY

echo "[done] unified full multilevel run complete"
echo "[out] stage4A_ckpt=$OUT_STAGE4A/ckpt_last.pt"
echo "[out] stage4B_ckpt=$OUT_STAGE4B/ckpt_last.pt"
echo "[out] benchmark=$BENCH_JSON"
echo "[out] benchmark_extra=$BENCH_EXTRA_JSON"
echo "[out] explain_eval=$EXPLAIN_EVAL_JSON"
echo "[out] explain_summary=$EXPLAIN_SUMMARY_JSON"

