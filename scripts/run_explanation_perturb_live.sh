#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs"
PERT_SET_ROOT="${PERT_SET_ROOT:-$OUT/exp_runtime_0218_112150/perturb}"
EXPL_GT="${EXPL_GT:-$OUT/exp_plan_0218_104500/explanation_test.jsonl}"
PERT_OUT_ROOT="${PERT_OUT_ROOT:-$OUT/explanation_perturb_jobs}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
SLEEP_SEC="${SLEEP_SEC:-20}"

mkdir -p "$PERT_OUT_ROOT" "$OUT/run_state"

need_build=0
for fn in \
  explanation_region_drop.jsonl \
  explanation_shuffle_ocr.jsonl \
  explanation_context_masking.jsonl \
  explanation_image_shuffle.jsonl \
  explanation_hard_image_shuffle.jsonl \
  explanation_visual_token_zero.jsonl \
  explanation_visual_token_mean.jsonl \
  explanation_visual_token_noise.jsonl
do
  if [[ ! -f "$PERT_SET_ROOT/$fn" ]]; then
    need_build=1
    break
  fi
done
if [[ "$need_build" == "1" ]]; then
  mkdir -p "$PERT_SET_ROOT"
  python "$ROOT/scixplain/tools/build_explanation_perturb_sets.py" \
    --input_jsonl "$EXPL_GT" \
    --out_root "$PERT_SET_ROOT"
fi

SELECT_JSON="$OUT/run_state/explanation_perturb_selected_models.json"
python - "$OUT/all_models_metrics_live.json" "$SELECT_JSON" <<'PY'
import json, pathlib, sys
metrics_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
obj = json.load(open(metrics_path, 'r', encoding='utf-8')) if metrics_path.exists() else {}
res = obj.get('results', {}) if isinstance(obj, dict) else {}
foundation = ['llava','deepseekvl2','qwen3vl','internvl35']
specialized = ['ureader','mplug_owl3','tinychart','docowl2','omnicaptioner','metacaptioner']

def top2(arr):
    rows=[]
    for m in arr:
        exp = res.get(m, {}).get('Explanation', {}) if isinstance(res.get(m, {}), dict) else {}
        n = int(exp.get('N') or 0)
        c = float(exp.get('CIDEr', -1.0)) if n > 0 else -1.0
        b = float(exp.get('BERTScore', -1.0)) if n > 0 else -1.0
        if n > 0:
            rows.append((c, b, n, m))
    rows.sort(reverse=True)
    return [m for _, _, _, m in rows[:2]]

f2 = top2(foundation)
s2 = top2(specialized)
sel = f2 + s2
if "scixplain" not in sel:
    sel.append("scixplain")
out = {
  'foundation_top2': f2,
  'specialized_top2': s2,
  'selected_for_perturb': sel,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(','.join(sel))
PY

MODEL_TAGS="$(python - "$SELECT_JSON" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
obj = json.load(open(p, 'r', encoding='utf-8')) if p.exists() else {}
arr = obj.get('selected_for_perturb', [])
print(','.join(arr))
PY
)"

if [[ -z "$MODEL_TAGS" ]]; then
  echo "[fatal] no selected models for perturbation"
  exit 2
fi

echo "[info] selected model tags: $MODEL_TAGS"

dispatch_one() {
  local name="$1"
  local test_json="$2"
  local out_root="$3"
  local qsub_extra_vars="${4:-}"

  mkdir -p "$out_root"
  echo "[dispatch] perturb=$name test_json=$test_json out_root=$out_root qsub_extra_vars=${qsub_extra_vars:-<none>}"
  EXPLAIN_TEST_JSON="$test_json" \
  OUT_ROOT="$out_root" \
  TASKS="explanation" \
  MODEL_TAGS="$MODEL_TAGS" \
  MAX_PARALLEL="$MAX_PARALLEL" \
  WAIT_IDLE_FIRST=0 \
  SKIP_COMPLETED=1 \
  SLEEP_SEC="$SLEEP_SEC" \
  MAX_RETRY_PER_TASK=3 \
  RETRY_COOLDOWN_SEC=180 \
  QSUB_EXTRA_VARS="$qsub_extra_vars" \
  bash "$ROOT/scripts/dispatch_infer_batched.sh"
}

refresh_diag_tables() {
  python "$ROOT/scripts/build_explanation_diag_live.py" \
    --gt_jsonl "$EXPL_GT" \
    --clean_metrics_json "$OUT/all_models_metrics_live.json" \
    --region_root "$PERT_OUT_ROOT/region_drop" \
    --shuffle_root "$PERT_OUT_ROOT/shuffle_ocr" \
    --context_root "$PERT_OUT_ROOT/context_masking" \
    --image_shuffle_root "$PERT_OUT_ROOT/image_shuffle" \
    --hard_shuffle_root "$PERT_OUT_ROOT/hard_image_shuffle" \
    --visual_zero_root "$PERT_OUT_ROOT/visual_token_zero" \
    --visual_mean_root "$PERT_OUT_ROOT/visual_token_mean" \
    --visual_noise_root "$PERT_OUT_ROOT/visual_token_noise" \
    --out_json "$OUT/all_models_explanation_diag_live.json"
  python "$ROOT/scripts/generate_multitable_report.py"
}

dispatch_one "region_drop" "$PERT_SET_ROOT/explanation_region_drop.jsonl" "$PERT_OUT_ROOT/region_drop"
refresh_diag_tables
dispatch_one "shuffle_ocr" "$PERT_SET_ROOT/explanation_shuffle_ocr.jsonl" "$PERT_OUT_ROOT/shuffle_ocr"
refresh_diag_tables
dispatch_one "context_masking" "$PERT_SET_ROOT/explanation_context_masking.jsonl" "$PERT_OUT_ROOT/context_masking"
refresh_diag_tables
dispatch_one "image_shuffle" "$PERT_SET_ROOT/explanation_image_shuffle.jsonl" "$PERT_OUT_ROOT/image_shuffle"
refresh_diag_tables
dispatch_one "hard_image_shuffle" "$PERT_SET_ROOT/explanation_hard_image_shuffle.jsonl" "$PERT_OUT_ROOT/hard_image_shuffle"
refresh_diag_tables
dispatch_one "visual_token_zero" "$PERT_SET_ROOT/explanation_visual_token_zero.jsonl" "$PERT_OUT_ROOT/visual_token_zero" "SCIXPLAIN_EXTRA_ARGS_APPEND=--visual_token_ablation=zero"
refresh_diag_tables
dispatch_one "visual_token_mean" "$PERT_SET_ROOT/explanation_visual_token_mean.jsonl" "$PERT_OUT_ROOT/visual_token_mean" "SCIXPLAIN_EXTRA_ARGS_APPEND=--visual_token_ablation=mean"
refresh_diag_tables
dispatch_one "visual_token_noise" "$PERT_SET_ROOT/explanation_visual_token_noise.jsonl" "$PERT_OUT_ROOT/visual_token_noise" "SCIXPLAIN_EXTRA_ARGS_APPEND=--visual_token_ablation=noise"
refresh_diag_tables

echo "[done] explanation perturb pipeline finished"

