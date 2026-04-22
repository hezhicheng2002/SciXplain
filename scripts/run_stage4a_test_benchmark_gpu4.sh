#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TS="${TS:-$(date +%m%d_%H%M%S)}"
GPU="${GPU:-4}"
TMP_ROOT="${TMP_ROOT:-$ROOT/tmp/run_tmp}"

SCICAP_ROOT="${SCICAP_ROOT:-${ROOT}/dataset/scicap_mlbcap_node_diagram_v2}"
SCICAP_TRAIN_JSON="${SCICAP_TRAIN_JSON:-$SCICAP_ROOT/dataset_split/train.json}"
SCICAP_TEST_JSON="${SCICAP_TEST_JSON:-$SCICAP_ROOT/dataset_split/test.json}"

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tinyllava_phi_siglip_stage4a_scicap_metric_0208_202116}"
INIT_CKPT="${INIT_CKPT:-$CKPT_DIR/ckpt_last.pt}"
EVAL_STEP="${EVAL_STEP:-1200}"
VISUAL_CKPT="${VISUAL_CKPT:-$ROOT/checkpoints/visual_student_scistruct_scicap_full_v2/ckpt_last.pt}"

OUT_DIR="${OUT_DIR:-$ROOT/checkpoints/tinyllava_phi_siglip_stage4a_scicap_metric_0208_202116_evaltest_${TS}}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT/outputs/stage4a_test_preds_${TS}}"
METRIC_JSON="${METRIC_JSON:-$ROOT/outputs/stage4a_test_benchmark_${TS}.json}"
PAIR_JSONL="${PAIR_JSONL:-$ROOT/outputs/stage4a_test_benchmark_pairs_${TS}.jsonl}"
METRIC_EXTRA_JSON="${METRIC_EXTRA_JSON:-$ROOT/outputs/stage4a_test_benchmark_${TS}_extra_metrics.json}"
CAPTION_BLOCK_FORMULA_IN_DECODE="${CAPTION_BLOCK_FORMULA_IN_DECODE:-1}"
DESC_BLOCK_PROMPT_LEAK_IN_DECODE="${DESC_BLOCK_PROMPT_LEAK_IN_DECODE:-1}"

mkdir -p "$TMP_ROOT" "$OUT_DIR" "$SAMPLE_DIR" "$ROOT/logs/tinyllava"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TMPDIR="$TMP_ROOT"
export TMP="$TMP_ROOT"
export TEMP="$TMP_ROOT"

DECODE_ARGS=()
if [[ "$CAPTION_BLOCK_FORMULA_IN_DECODE" == "1" ]]; then
  DECODE_ARGS+=(--caption_block_formula_in_decode)
fi
if [[ "$DESC_BLOCK_PROMPT_LEAK_IN_DECODE" == "1" ]]; then
  DECODE_ARGS+=(--desc_block_prompt_leak_in_decode)
fi

python "$ROOT/scixplain/tools/train_tinyllava_image_only.py" \
  --model_path "$ROOT/checkpoints/phi-sig" \
  --model_dtype bfloat16 \
  --visual_ckpt "$VISUAL_CKPT" \
  --init_ckpt "$INIT_CKPT" \
  --dataset scicap \
  --train_json "$SCICAP_TRAIN_JSON" \
  --val_json "$SCICAP_TEST_JSON" \
  --out_dir "$OUT_DIR" \
  --batch_size 2 \
  --grad_accum 2 \
  --warmup_steps 0 \
  --max_steps 1200 \
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
  "${DECODE_ARGS[@]}" \
  --disable_explain_gate_non_explain \
  --max_ctx_tokens 128 \
  --max_length 384 \
  --max_target_short 48 \
  --max_target_long 128 \
  --max_target_desc 320 \
  --bucket_by_length \
  --bucket_bins 256,288,320,352,384 \
  --val_every 0 \
  --val_batch_size 2 \
  --val_num_batches 20 \
  --sample_every 1 \
  --sample_num 2202 \
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
  --sample_out_dir "$SAMPLE_DIR" \
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
  --eval_only \
  --eval_step "$EVAL_STEP"

SAMPLE_FILE="$SAMPLE_DIR/samples_step_$(printf '%06d' "$EVAL_STEP").json"
if [[ ! -f "$SAMPLE_FILE" ]]; then
  echo "[fatal] sample file missing: $SAMPLE_FILE" >&2
  exit 2
fi

python - "$SAMPLE_FILE" "$METRIC_JSON" "$PAIR_JSONL" "$METRIC_EXTRA_JSON" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from scixplain.tools.eval_text_decoder import _strip_leading_figure_prefix

sample_file = Path(sys.argv[1])
metric_json = Path(sys.argv[2])
pair_jsonl = Path(sys.argv[3])
metric_extra_json = Path(sys.argv[4])

rows = json.loads(sample_file.read_text(encoding="utf-8"))
pairs_raw = defaultdict(list)
pairs_strip = defaultdict(list)
norm = {
    "pairs": 0,
    "pred_changed": 0,
    "ref_changed": 0,
    "both_changed": 0,
    "examples": [],
}

with pair_jsonl.open("w", encoding="utf-8") as fw:
    for r in rows:
        scale = str(r.get("scale") or "").strip().lower()
        outputs = r.get("outputs") if isinstance(r.get("outputs"), dict) else {}
        pred = str(outputs.get(scale) or "").strip()
        ref = str(r.get("target") or "").strip()
        if scale not in {"short", "long", "desc"}:
            continue
        if not pred or not ref:
            continue
        fw.write(json.dumps({"idx": r.get("idx"), "scale": scale, "pred": pred, "ref": ref}, ensure_ascii=False) + "\n")
        pairs_raw[scale].append((pred, ref))
        p2, pchg, pp = _strip_leading_figure_prefix(pred)
        r2, rchg, rp = _strip_leading_figure_prefix(ref)
        pairs_strip[scale].append((p2, r2))
        norm["pairs"] += 1
        norm["pred_changed"] += int(pchg)
        norm["ref_changed"] += int(rchg)
        norm["both_changed"] += int(pchg and rchg)
        if (pchg or rchg) and len(norm["examples"]) < 20:
            norm["examples"].append({
                "idx": r.get("idx"),
                "scale": scale,
                "pred_prefix": pp,
                "ref_prefix": rp,
                "pred_before": pred,
                "pred_after": p2,
                "ref_before": ref,
                "ref_after": r2,
            })


def _score(pairs):
    if not pairs:
        return {"bleu4": 0.0, "rougeL": 0.0, "cider": 0.0, "n": 0}
    preds = [p for p, _ in pairs]
    refs = [r for _, r in pairs]
    gts = {i: [r] for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    rougeL, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return {
        "bleu4": float(bleu_scores[3]),
        "rougeL": float(rougeL),
        "cider": float(cider),
        "n": len(pairs),
    }


def _merge_all(dct):
    all_pairs = []
    for k in ("short", "long", "desc"):
        all_pairs.extend(dct.get(k, []))
    return all_pairs


raw = {k: _score(pairs_raw.get(k, [])) for k in ("short", "long", "desc")}
raw["all"] = _score(_merge_all(pairs_raw))
stripd = {k: _score(pairs_strip.get(k, [])) for k in ("short", "long", "desc")}
stripd["all"] = _score(_merge_all(pairs_strip))

formula_pat = re.compile(r"(\\[A-Za-z]+|[_^]\{?|[{}$]|[鈭埪琞|[伪-蠅螒-惟]|\\times|\\sum|\\frac)")


def _is_formula_like(text: str) -> bool:
    if not text:
        return False
    return bool(formula_pat.search(text))


def _formula_stats(pairs_dict):
    out = {}
    for k in ("short", "long"):
        preds = [p for p, _ in pairs_dict.get(k, [])]
        cnt = sum(1 for p in preds if _is_formula_like(p))
        out[k] = {
            "n": len(preds),
            "contaminated": int(cnt),
            "rate": (float(cnt) / float(len(preds))) if preds else 0.0,
        }
    return out


def _load_extra_metric_fn():
    try:
        import evaluate

        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        bert = evaluate.load("bertscore")

        def _compute(preds, refs):
            if not preds:
                return {
                    "n": 0,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "meteor": 0.0,
                    "bertscore_f1": 0.0,
                }
            rg = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
            mt = meteor.compute(predictions=preds, references=refs)
            out = {
                "n": len(preds),
                "rouge1": float(rg.get("rouge1", 0.0)),
                "rouge2": float(rg.get("rouge2", 0.0)),
                "meteor": float(mt.get("meteor", 0.0)),
                "bertscore_f1": 0.0,
            }
            try:
                bs = bert.compute(
                    predictions=preds,
                    references=refs,
                    lang="en",
                    rescale_with_baseline=False,
                )
                out["bertscore_f1"] = float(sum(bs["f1"]) / max(1, len(bs["f1"])))
            except Exception as exc:
                print(f"[warn] bertscore failed: {exc}", file=sys.stderr)
            return out

        return _compute
    except Exception as exc:
        print(f"[warn] extra metrics unavailable: {exc}", file=sys.stderr)

        def _compute(preds, refs):
            return {
                "n": len(preds),
                "rouge1": 0.0,
                "rouge2": 0.0,
                "meteor": 0.0,
                "bertscore_f1": 0.0,
            }

        return _compute


extra_metric_fn = _load_extra_metric_fn()


def _extra_score(pairs):
    preds = [p for p, _ in pairs]
    refs = [r for _, r in pairs]
    return extra_metric_fn(preds, refs)


extra_raw = {k: _extra_score(pairs_raw.get(k, [])) for k in ("short", "long", "desc")}
extra_raw["all"] = _extra_score(_merge_all(pairs_raw))
extra_strip = {k: _extra_score(pairs_strip.get(k, [])) for k in ("short", "long", "desc")}
extra_strip["all"] = _extra_score(_merge_all(pairs_strip))

# Merge pycocoeval metrics and extra metrics so metric_json is complete in one file.
for k in ("short", "long", "desc", "all"):
    raw[k].update({
        "rouge1": extra_raw[k]["rouge1"],
        "rouge2": extra_raw[k]["rouge2"],
        "meteor": extra_raw[k]["meteor"],
        "bertscore_f1": extra_raw[k]["bertscore_f1"],
    })
    stripd[k].update({
        "rouge1": extra_strip[k]["rouge1"],
        "rouge2": extra_strip[k]["rouge2"],
        "meteor": extra_strip[k]["meteor"],
        "bertscore_f1": extra_strip[k]["bertscore_f1"],
    })

delta = {}
for k in ("short", "long", "desc", "all"):
    delta[k] = {
        "bleu4": stripd[k]["bleu4"] - raw[k]["bleu4"],
        "rougeL": stripd[k]["rougeL"] - raw[k]["rougeL"],
        "cider": stripd[k]["cider"] - raw[k]["cider"],
        "rouge1": stripd[k]["rouge1"] - raw[k]["rouge1"],
        "rouge2": stripd[k]["rouge2"] - raw[k]["rouge2"],
        "meteor": stripd[k]["meteor"] - raw[k]["meteor"],
        "bertscore_f1": stripd[k]["bertscore_f1"] - raw[k]["bertscore_f1"],
        "n": raw[k]["n"],
    }

out = {
    "input": {"sample_file": str(sample_file), "n_pairs": norm["pairs"]},
    "normalization": {
        "pred_changed": norm["pred_changed"],
        "ref_changed": norm["ref_changed"],
        "both_changed": norm["both_changed"],
        "pred_changed_rate": (norm["pred_changed"] / norm["pairs"]) if norm["pairs"] else 0.0,
        "ref_changed_rate": (norm["ref_changed"] / norm["pairs"]) if norm["pairs"] else 0.0,
        "both_changed_rate": (norm["both_changed"] / norm["pairs"]) if norm["pairs"] else 0.0,
        "examples": norm["examples"],
    },
    "raw": raw,
    "strip_figure_prefix": stripd,
    "delta_strip_minus_raw": delta,
    "diagnostics": {
        "caption_formula_contamination_raw": _formula_stats(pairs_raw),
        "caption_formula_contamination_strip_figure_prefix": _formula_stats(pairs_strip),
    },
}

metric_json.parent.mkdir(parents=True, exist_ok=True)
metric_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
metric_extra_json.parent.mkdir(parents=True, exist_ok=True)
metric_extra_json.write_text(
    json.dumps(
        {
            "input": {"sample_file": str(sample_file), "n_raw_pairs": norm["pairs"]},
            "raw": extra_raw,
            "strip_figure_prefix": extra_strip,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

echo "[done] sample_file=$SAMPLE_FILE"
echo "[done] metric_json=$METRIC_JSON"
echo "[done] pair_jsonl=$PAIR_JSONL"
echo "[done] metric_extra_json=$METRIC_EXTRA_JSON"

if [[ "${AUTO_RECLAIM_GPU:-0}" == "1" ]]; then
  if [[ -n "${GPU_RECLAIM_CMD:-}" ]]; then
    nohup bash -lc "$GPU_RECLAIM_CMD" >/dev/null 2>&1 &
    HPID=$!
    mkdir -p "$ROOT/logs/tinyllava/pids"
    echo "$HPID" > "$ROOT/logs/tinyllava/pids/gpu_reclaim_gpu${GPU}.pid"
    echo "$(date '+%F %T') restarted_gpu_reclaim pid=$HPID gpu=$GPU after_stage4a_test=$TS" >> "$ROOT/logs/tinyllava/gpu_reclaim.log"
    echo "[done] gpu_reclaim_pid=$HPID gpu=$GPU"
  else
    echo "[warn] AUTO_RECLAIM_GPU=1 but GPU_RECLAIM_CMD is unset; skip reclaim"
  fi
fi

