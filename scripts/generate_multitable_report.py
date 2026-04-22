#!/usr/bin/env python3
"""Generate split progress/metrics CSV tables for the SciXplain main report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLE_DIR = OUT / "tables_0219_live"

PROGRESS_CSV = OUT / "main_table_progress_0219_live.csv"
SCIX_METRICS_JSON = OUT / "scixplain_metrics_filled_0218.json"
ALL_MODELS_METRICS_JSON = OUT / "all_models_metrics_live.json"
ALL_MODELS_EFF_JSON = OUT / "all_models_efficiency_live.json"
ALL_MODELS_EXPL_DIAG_JSON = OUT / "all_models_explanation_diag_live.json"

MODELS = [
    "scixplain",
    "ureader",
    "mplug_owl3",
    "tinychart",
    "docowl2",
    "omnicaptioner",
    "metacaptioner",
    "llava",
    "deepseekvl2",
    "qwen3vl",
    "internvl35",
]

TASKS_TEXT = ["Caption_short", "Caption_long", "Description", "Explanation"]
TASKS_TEXT_MAIN = ["Caption_short", "Caption_long", "Description"]
TEXT7 = ["BLEU-4", "ROUGE-L", "ROUGE-1", "ROUGE-2", "CIDEr", "BERTScore", "METEOR", "N"]


def fmt_metric(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            out_row = {k: fmt_metric(row.get(k, "")) for k in writer.fieldnames}
            writer.writerow(out_row)


def load_scix_metrics(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_obj(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_tmp_metrics_results(base: Dict[str, object], out_root: Path) -> Dict[str, object]:
    """Backfill model results from tmp single-model metrics files.

    This keeps live metrics as the primary source and only fills missing
    model/task cells from cached tmp metrics, so tables do not miss already
    computed results.
    """
    merged = dict(base) if isinstance(base, dict) else {}
    base_results = merged.get("results")
    if not isinstance(base_results, dict):
        base_results = {}
        merged["results"] = base_results

    tmp_files = sorted(out_root.glob("tmp_metrics_*.json"))
    for p in tmp_files:
        # Skip manifest-like or legacy special files; only merge task metrics.
        if "manifest" in p.name:
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        tmp_results = obj.get("results")
        if not isinstance(tmp_results, dict):
            continue
        for model, model_obj in tmp_results.items():
            if not isinstance(model_obj, dict) or not model_obj:
                continue
            dst_model = base_results.get(model)
            if not isinstance(dst_model, dict):
                base_results[model] = dict(model_obj)
                continue
            for task, task_obj in model_obj.items():
                if task not in dst_model and isinstance(task_obj, dict):
                    dst_model[task] = task_obj
    return merged


def build_progress_split(progress_rows: List[Dict[str, str]]) -> None:
    # 1) model-level compact progress
    fieldnames = [
        "model",
        "group",
        "strategy",
        "train_status",
        "caption_short_valid",
        "caption_short_need",
        "caption_short_error",
        "caption_short_status",
        "caption_long_valid",
        "caption_long_need",
        "caption_long_error",
        "caption_long_status",
        "description_valid",
        "description_need",
        "description_error",
        "description_status",
        "explanation_valid",
        "explanation_need",
        "explanation_error",
        "explanation_status",
        "infer_done_tasks",
        "metrics_ready",
    ]
    rows = []
    for model in MODELS:
        rec = next((r for r in progress_rows if r.get("model") == model), None)
        if rec is None:
            rec = {k: "" for k in fieldnames}
            rec["model"] = model
        rows.append({k: rec.get(k, "") for k in fieldnames})
    write_csv(TABLE_DIR / "table_A_progress.csv", fieldnames, rows)


def build_text_metric_tables(
    scix: Dict[str, object],
    progress_rows: List[Dict[str, str]],
    all_models_metrics: Dict[str, object],
    all_models_eff: Dict[str, object],
) -> None:
    # Model metadata from progress
    group_by_model = {r.get("model", ""): r.get("group", "") for r in progress_rows}

    # 2) full 7 metrics (prefer all-model live metrics; fallback to legacy scixplain-only)
    full7_by_model_task: Dict[str, Dict[str, Dict[str, object]]] = {}
    results = all_models_metrics.get("results", {}) if isinstance(all_models_metrics, dict) else {}
    if isinstance(results, dict) and results:
        for model, mobj in results.items():
            if not isinstance(mobj, dict):
                continue
            full7_by_model_task[model] = {}
            for task in TASKS_TEXT:
                tob = mobj.get(task, {})
                full7_by_model_task[model][task] = tob if isinstance(tob, dict) else {}
    else:
        full7_legacy: Dict[str, Dict[str, object]] = scix.get("metrics_full_7", {}) if isinstance(scix, dict) else {}
        full7_by_model_task["scixplain"] = {}
        for task in TASKS_TEXT:
            full7_by_model_task["scixplain"][task] = full7_legacy.get(task, {})

    rows_7 = []
    for model in MODELS:
        for task in TASKS_TEXT_MAIN:
            task_obj: Dict[str, object] = full7_by_model_task.get(model, {}).get(task, {})
            row = {
                "model": model,
                "group": group_by_model.get(model, ""),
                "task": task,
            }
            for metric in TEXT7:
                row[metric] = task_obj.get(metric, "")
            rows_7.append(row)
    write_csv(
        TABLE_DIR / "table_B_text_metrics_full7.csv",
        ["model", "group", "task", *TEXT7],
        rows_7,
    )

    # Keep explanation text metrics only as an audit table (not for main judging).
    rows_7_expl = []
    for model in MODELS:
        task_obj = full7_by_model_task.get(model, {}).get("Explanation", {})
        row = {
            "model": model,
            "group": group_by_model.get(model, ""),
            "task": "Explanation",
        }
        for metric in TEXT7:
            row[metric] = task_obj.get(metric, "")
        rows_7_expl.append(row)
    write_csv(
        TABLE_DIR / "table_Bx_explanation_legacy7_audit.csv",
        ["model", "group", "task", *TEXT7],
        rows_7_expl,
    )

    # 3) replace legacy main-5 table with efficiency summary table.
    eff_by_model = all_models_eff.get("by_model", {}) if isinstance(all_models_eff, dict) else {}
    rows_eff = []
    for model in MODELS:
        e = eff_by_model.get(model, {}) if isinstance(eff_by_model, dict) else {}
        row = {
            "model": model,
            "group": group_by_model.get(model, ""),
            "Param_B": e.get("param_billion", ""),
            "Peak_VRAM_GB": e.get("peak_vram_allocated_gb", ""),
            "Latency_ms_Caption": e.get("caption_latency_ms_per_sample", ""),
            "Latency_ms_Description": e.get("description_latency_ms_per_sample", ""),
            "Latency_ms_Explanation": e.get("explanation_latency_ms_per_sample", ""),
            "TPOT_ms_per_token": e.get("caption_tpot_ms_per_output_token", ""),
            "N_caption": e.get("caption_valid_count", ""),
            "N_description": e.get("description_valid_count", ""),
            "N_explanation": e.get("explanation_valid_count", ""),
            "eff_probe_source": e.get("eff_probe_source", ""),
        }
        rows_eff.append(row)
    write_csv(
        TABLE_DIR / "table_C_text_metrics_main5.csv",
        [
            "model",
            "group",
            "Param_B",
            "Peak_VRAM_GB",
            "Latency_ms_Caption",
            "Latency_ms_Description",
            "Latency_ms_Explanation",
            "TPOT_ms_per_token",
            "N_caption",
            "N_description",
            "N_explanation",
            "eff_probe_source",
        ],
        rows_eff,
    )


def build_description_tables(scix: Dict[str, object], progress_rows: List[Dict[str, str]]) -> None:
    group_by_model = {r.get("model", ""): r.get("group", "") for r in progress_rows}
    desc = scix.get("description_structured_metrics", {}) if isinstance(scix, dict) else {}
    core3 = desc.get("core_3", {}) if isinstance(desc, dict) else {}
    full = desc.get("full_pred_metrics", {}) if isinstance(desc, dict) else {}

    # 4) Description core 3
    rows_core = []
    for model in MODELS:
        row = {
            "model": model,
            "group": group_by_model.get(model, ""),
            "Alias_Hit_Rate": core3.get("Alias_Hit_Rate", "") if model == "scixplain" else "",
            "Strict_Hallucination_Rate": core3.get("Strict_Hallucination_Rate", "") if model == "scixplain" else "",
            "Relation_Accuracy": core3.get("Relation_Accuracy", "") if model == "scixplain" else "",
        }
        rows_core.append(row)
    write_csv(
        TABLE_DIR / "table_D_description_core3.csv",
        ["model", "group", "Alias_Hit_Rate", "Strict_Hallucination_Rate", "Relation_Accuracy"],
        rows_core,
    )

    # 5) Description full diagnostics (single-row source details for current method)
    rows_full = []
    metric_cols = sorted(full.keys())
    for model in MODELS:
        row = {"model": model, "group": group_by_model.get(model, "")}
        for c in metric_cols:
            row[c] = full.get(c, "") if model == "scixplain" else ""
        rows_full.append(row)
    write_csv(TABLE_DIR / "table_E_description_full_diag.csv", ["model", "group", *metric_cols], rows_full)


def build_explanation_tables(
    scix: Dict[str, object],
    progress_rows: List[Dict[str, str]],
    all_models_expl_diag: Dict[str, object],
) -> None:
    group_by_model = {r.get("model", ""): r.get("group", "") for r in progress_rows}
    by_model = all_models_expl_diag.get("by_model", {}) if isinstance(all_models_expl_diag, dict) else {}
    if not isinstance(by_model, dict):
        by_model = {}

    sel = all_models_expl_diag.get("selected_models", {}) if isinstance(all_models_expl_diag, dict) else {}
    final_set = sel.get("final_set_for_perturb", []) if isinstance(sel, dict) else []
    final_set = final_set if isinstance(final_set, list) else []
    final_set = set(str(x) for x in final_set)

    # Keep legacy scixplain diagnostic numbers as supplementary references.
    scix_legacy = all_models_expl_diag.get("scixplain_legacy_diag", {}) if isinstance(all_models_expl_diag, dict) else {}
    scix_legacy_cf = scix_legacy.get("counterfactual_drop", {}) if isinstance(scix_legacy, dict) else {}
    scix_legacy_shuffle = scix_legacy.get("shuffle_consistency", {}) if isinstance(scix_legacy, dict) else {}

    fieldnames = [
        "model",
        "group",
        "selected_for_perturb",
        "clean_N",
        "clean_CIDEr",
        "clean_BERTScore",
        "region_drop_N",
        "region_drop_CIDEr",
        "region_drop_BERTScore",
        "region_drop_drop_CIDEr_abs",
        "region_drop_drop_BERTScore_abs",
        "shuffle_ocr_N",
        "shuffle_ocr_CIDEr",
        "shuffle_ocr_BERTScore",
        "shuffle_ocr_drop_CIDEr_abs",
        "shuffle_ocr_drop_BERTScore_abs",
        "context_masking_N",
        "context_masking_CIDEr",
        "context_masking_BERTScore",
        "context_masking_drop_CIDEr_abs",
        "context_masking_drop_BERTScore_abs",
        "drop_CIDEr_abs_mean",
        "drop_BERTScore_abs_mean",
        "drop_CIDEr_rel_mean",
        "drop_BERTScore_rel_mean",
        "available_perturbations",
        "perturb_coverage_ratio",
        "diag_status",
        "scix_legacy_region_gap_last",
        "scix_legacy_shuffle_gap_last",
    ]

    rows = []
    for model in MODELS:
        mobj = by_model.get(model, {}) if isinstance(by_model.get(model), dict) else {}
        clean = mobj.get("clean", {}) if isinstance(mobj.get("clean"), dict) else {}
        rd = mobj.get("region_drop", {}) if isinstance(mobj.get("region_drop"), dict) else {}
        so = mobj.get("shuffle_ocr", {}) if isinstance(mobj.get("shuffle_ocr"), dict) else {}
        cm = mobj.get("context_masking", {}) if isinstance(mobj.get("context_masking"), dict) else {}
        ds = mobj.get("drop_summary", {}) if isinstance(mobj.get("drop_summary"), dict) else {}

        diag_status = ""
        if mobj:
            ok_cnt = 0
            for p in (rd, so, cm):
                if p.get("status") == "ok":
                    ok_cnt += 1
            diag_status = f"{ok_cnt}/3"

        row = {
            "model": model,
            "group": group_by_model.get(model, ""),
            "selected_for_perturb": "yes" if (model in final_set or mobj.get("selected")) else "",
            "clean_N": clean.get("N", ""),
            "clean_CIDEr": clean.get("CIDEr", ""),
            "clean_BERTScore": clean.get("BERTScore", ""),
            "region_drop_N": rd.get("N", ""),
            "region_drop_CIDEr": rd.get("CIDEr", ""),
            "region_drop_BERTScore": rd.get("BERTScore", ""),
            "region_drop_drop_CIDEr_abs": rd.get("drop_CIDEr_abs", ""),
            "region_drop_drop_BERTScore_abs": rd.get("drop_BERTScore_abs", ""),
            "shuffle_ocr_N": so.get("N", ""),
            "shuffle_ocr_CIDEr": so.get("CIDEr", ""),
            "shuffle_ocr_BERTScore": so.get("BERTScore", ""),
            "shuffle_ocr_drop_CIDEr_abs": so.get("drop_CIDEr_abs", ""),
            "shuffle_ocr_drop_BERTScore_abs": so.get("drop_BERTScore_abs", ""),
            "context_masking_N": cm.get("N", ""),
            "context_masking_CIDEr": cm.get("CIDEr", ""),
            "context_masking_BERTScore": cm.get("BERTScore", ""),
            "context_masking_drop_CIDEr_abs": cm.get("drop_CIDEr_abs", ""),
            "context_masking_drop_BERTScore_abs": cm.get("drop_BERTScore_abs", ""),
            "drop_CIDEr_abs_mean": ds.get("drop_CIDEr_abs_mean", ""),
            "drop_BERTScore_abs_mean": ds.get("drop_BERTScore_abs_mean", ""),
            "drop_CIDEr_rel_mean": ds.get("drop_CIDEr_rel_mean", ""),
            "drop_BERTScore_rel_mean": ds.get("drop_BERTScore_rel_mean", ""),
            "available_perturbations": ds.get("available_perturbations", ""),
            "perturb_coverage_ratio": ds.get("perturb_coverage_ratio", ""),
            "diag_status": diag_status,
            "scix_legacy_region_gap_last": scix_legacy_cf.get("val_gap_region_drop_last", "") if model == "scixplain" else "",
            "scix_legacy_shuffle_gap_last": scix_legacy_shuffle.get("val_gap_shuffle_ocr_last", "") if model == "scixplain" else "",
        }
        rows.append(row)

    # Fallback: if no live diag exists, keep legacy scix-only diagnostics.
    if not any(isinstance(by_model.get(m), dict) and by_model.get(m) for m in MODELS):
        exp = scix.get("explanation_diagnostic_metrics", {}) if isinstance(scix, dict) else {}
        flat_cols: List[str] = []
        flat_scix: Dict[str, object] = {}
        if isinstance(exp, dict):
            for section, value in exp.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        col = f"{section}.{k}"
                        flat_cols.append(col)
                        flat_scix[col] = v
                else:
                    flat_cols.append(section)
                    flat_scix[section] = value
        flat_cols = sorted(set(flat_cols))
        rows = []
        for model in MODELS:
            row = {"model": model, "group": group_by_model.get(model, "")}
            for c in flat_cols:
                row[c] = flat_scix.get(c, "") if model == "scixplain" else ""
            rows.append(row)
        write_csv(TABLE_DIR / "table_F_explanation_diag.csv", ["model", "group", *flat_cols], rows)
        return

    write_csv(TABLE_DIR / "table_F_explanation_diag.csv", fieldnames, rows)


def build_sources_table(scix: Dict[str, object]) -> None:
    rows = []
    rows.append(
        {
            "table": "A_progress",
            "source_file": str(PROGRESS_CSV.relative_to(ROOT)) if PROGRESS_CSV.exists() else "",
            "note": "live progress by model/task",
        }
    )
    rows.append(
        {
            "table": "B/C/D/E/F_metrics",
            "source_file": str(SCIX_METRICS_JSON.relative_to(ROOT)) if SCIX_METRICS_JSON.exists() else "",
            "note": "legacy metric source (scixplain)",
        }
    )
    rows.append(
        {
            "table": "B_text_metrics_full7_live",
            "source_file": str(ALL_MODELS_METRICS_JSON.relative_to(ROOT)) if ALL_MODELS_METRICS_JSON.exists() else "",
            "note": "all-model text metrics (caption/description main tasks)",
        }
    )
    tmp_metric_files = sorted(OUT.glob("tmp_metrics_*.json"))
    tmp_metric_files = [p for p in tmp_metric_files if "manifest" not in p.name]
    if tmp_metric_files:
        rows.append(
            {
                "table": "B_text_metrics_tmp_fallback",
                "source_file": ",".join(str(p.relative_to(ROOT)) for p in tmp_metric_files),
                "note": "fallback sources used only when live metrics miss model/task",
            }
        )
    rows.append(
        {
            "table": "Bx_explanation_legacy7_audit",
            "source_file": str(ALL_MODELS_METRICS_JSON.relative_to(ROOT)) if ALL_MODELS_METRICS_JSON.exists() else "",
            "note": "explanation legacy text metrics for audit only",
        }
    )
    rows.append(
        {
            "table": "C_efficiency",
            "source_file": str(ALL_MODELS_EFF_JSON.relative_to(ROOT)) if ALL_MODELS_EFF_JSON.exists() else "",
            "note": "all-model efficiency summary",
        }
    )
    rows.append(
        {
            "table": "F_explanation_diag",
            "source_file": str(ALL_MODELS_EXPL_DIAG_JSON.relative_to(ROOT)) if ALL_MODELS_EXPL_DIAG_JSON.exists() else "",
            "note": "explanation perturbation drop diagnostics (CIDEr/BERTScore)",
        }
    )
    if isinstance(scix, dict):
        for key in ["source_benchmark", "updated_at", "generated_at"]:
            rows.append({"table": "scixplain_metrics_meta", "source_file": key, "note": scix.get(key, "")})
    write_csv(TABLE_DIR / "table_Z_sources.csv", ["table", "source_file", "note"], rows)


def main() -> None:
    progress_rows = read_csv_rows(PROGRESS_CSV)
    scix = load_scix_metrics(SCIX_METRICS_JSON)
    all_models_metrics = merge_tmp_metrics_results(load_json_obj(ALL_MODELS_METRICS_JSON), OUT)
    all_models_eff = load_json_obj(ALL_MODELS_EFF_JSON)
    all_models_expl_diag = load_json_obj(ALL_MODELS_EXPL_DIAG_JSON)

    build_progress_split(progress_rows)
    build_text_metric_tables(scix, progress_rows, all_models_metrics, all_models_eff)
    build_description_tables(scix, progress_rows)
    build_explanation_tables(scix, progress_rows, all_models_expl_diag)
    build_sources_table(scix)

    print("Generated tables in:", TABLE_DIR)


if __name__ == "__main__":
    main()
