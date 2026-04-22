#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-4}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
ENV_NAME="${ENV_NAME:-scixplain}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=("python")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=("python3")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=("conda" "run" "-n" "$ENV_NAME" "python")
else
  echo "[fatal] no python interpreter found; set PYTHON_BIN or install python/conda" >&2
  exit 2
fi

mkdir -p logs/visual_student

common_args=(
  --scistruct_struct_cache outputs/structure_cache/scistruct_train_struct.qf_v2.jsonl
  --scicap_struct_cache outputs/structure_cache/scicap_train_struct.qf_v2.jsonl
  --scistruct_geom_cache outputs/geom_ocr/scistruct_train_vl.qf_v2.jsonl
  --scicap_geom_cache outputs/geom_ocr/scicap_train_vl.qf_v2.jsonl
  --scistruct_val_struct_cache outputs/structure_cache/scistruct_val_struct.qf_v2.jsonl
  --scicap_val_struct_cache outputs/structure_cache/scicap_val_struct.qf_v2.jsonl
  --scistruct_val_geom_cache outputs/geom_ocr/scistruct_val_vl.qf_v2.jsonl
  --scicap_val_geom_cache outputs/geom_ocr/scicap_val_vl.qf_v2.jsonl
  --max_steps 20000
  --val_every 500
  --lambda_struct 1.0
  --lambda_graph 8.0
  --lambda_node 1.0
  --lambda_role 0.5
  --lambda_edge 2.0
  --lambda_geom 5.0
  --lambda_style 0.01
  --feat_warmup_steps 4000
  --feat_ramp_steps 4000
  --amp
)

eval_args=(
  --scistruct_val_struct_cache outputs/structure_cache/scistruct_val_struct.qf_v2.jsonl
  --scicap_val_struct_cache outputs/structure_cache/scicap_val_struct.qf_v2.jsonl
  --scistruct_val_geom_cache outputs/geom_ocr/scistruct_val_vl.qf_v2.jsonl
  --scicap_val_geom_cache outputs/geom_ocr/scicap_val_vl.qf_v2.jsonl
  --scistruct_test_struct_cache outputs/structure_cache/scistruct_test_struct.qf_v2.jsonl
  --scicap_test_struct_cache outputs/structure_cache/scicap_test_struct.qf_v2.jsonl
  --scistruct_test_geom_cache outputs/geom_ocr/scistruct_test_vl.qf_v2.jsonl
  --scicap_test_geom_cache outputs/geom_ocr/scicap_test_vl.qf_v2.jsonl
  --eval_only
  --amp
)

run_variant() {
  local name="$1"
  shift
  local -a variant_args=("$@")
  local out_dir="checkpoints/visual_student_scistruct_scicap_full_v2_ablate_${name}_${TS}"
  local train_log="logs/visual_student/train_visual_student_full_v2_ablate_${name}_${TS}.log"
  local eval_log="logs/visual_student/eval_visual_student_full_v2_ablate_${name}_${TS}.log"
  local eval_jsonl="logs/visual_student/eval_visual_student_full_v2_ablate_${name}_${TS}.jsonl"

  echo "[start][${name}] train -> ${train_log}"
  CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON_CMD[@]}" scixplain/tools/train_visual_student.py \
    --out_dir "$out_dir" \
    "${common_args[@]}" \
    "${variant_args[@]}" \
    >"$train_log" 2>&1

  echo "[start][${name}] eval(test) -> ${eval_log}"
  CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON_CMD[@]}" scixplain/tools/train_visual_student.py \
    --out_dir "$out_dir" \
    --init_ckpt "$out_dir/ckpt_last.pt" \
    --val_log "$eval_jsonl" \
    "${eval_args[@]}" \
    "${variant_args[@]}" \
    >"$eval_log" 2>&1

  echo "[done][${name}] out_dir=${out_dir} eval_jsonl=${eval_jsonl}"
}

run_variant "no_dino" \
  --disable_style_teacher \
  --lambda_style 0.0

run_variant "no_rex" \
  --disable_rex_teacher \
  --lambda_struct 0.0 \
  --lambda_graph 0.0 \
  --lambda_node 0.0 \
  --lambda_role 0.0 \
  --lambda_edge 0.0

run_variant "no_sam2" \
  --disable_geom_teacher \
  --geom_mode none \
  --lambda_geom 0.0

echo "[done] all ablations finished, ts=${TS}"

