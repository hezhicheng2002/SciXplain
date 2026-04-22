#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PRED_ROOT = OUT / "unified_infer_jobs"
EFF_ROOT = PRED_ROOT / "efficiency"
LOG_ROOT = ROOT / "logs" / "pbs"
OUT_JSON = OUT / "all_models_efficiency_live.json"

MODEL_ORDER = [
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
TASKS = ["caption", "description", "explanation"]

START_RE = re.compile(r"^\[start\]\s+(.+?)\s+job=")
END_RE = re.compile(r"^\[end\]\s+(.+?)\s+rc=(\d+)")
TIME_FMT = "%a %b %d %I:%M:%S %p %z %Y"


def _count_valid_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    ok = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("error"):
                continue
            ok += 1
    return ok


def _parse_runtime_seconds(log_path: Path) -> Optional[float]:
    try:
        txt = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    start_dt = None
    end_dt = None
    end_rc = None
    for ln in txt:
        m = START_RE.search(ln)
        if m:
            try:
                start_dt = datetime.strptime(m.group(1), TIME_FMT)
            except Exception:
                pass
        m = END_RE.search(ln)
        if m:
            try:
                end_dt = datetime.strptime(m.group(1), TIME_FMT)
                end_rc = int(m.group(2))
            except Exception:
                pass
    if start_dt is None or end_dt is None:
        return None
    if end_rc != 0:
        return None
    sec = (end_dt - start_dt).total_seconds()
    return float(sec) if sec >= 0 else None


def _latest_success_log(model: str, task: str) -> Optional[Path]:
    cands = sorted(
        LOG_ROOT.glob(f"scixp_inf_{model}_{task}_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in cands:
        sec = _parse_runtime_seconds(p)
        if sec is not None:
            return p
    return None


def _load_eff_probe(model: str, task: str) -> Dict[str, object]:
    p = EFF_ROOT / f"{model}_{task}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_first(d: Dict[str, object], keys: Tuple[str, ...]) -> object:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def main() -> None:
    by_model_task: Dict[str, Dict[str, object]] = {}
    by_model: Dict[str, Dict[str, object]] = {}

    for m in MODEL_ORDER:
        mt = {}
        best_param_count = None
        best_param_b = None
        best_peak_alloc = None
        best_peak_resv = None
        best_tpot = None
        best_eff_source = None

        for t in TASKS:
            pred_path = PRED_ROOT / f"{m}_{t}.jsonl"
            valid_n = _count_valid_jsonl(pred_path)
            log_path = _latest_success_log(m, t)
            runtime_s = _parse_runtime_seconds(log_path) if log_path is not None else None
            avg_ms = (runtime_s * 1000.0 / valid_n) if (runtime_s is not None and valid_n > 0) else None

            eff = _load_eff_probe(m, t)
            param_count = _pick_first(eff, ("param_count",))
            param_b = _pick_first(eff, ("param_billion",))
            peak_alloc = _pick_first(eff, ("peak_vram_allocated_gb",))
            peak_resv = _pick_first(eff, ("peak_vram_reserved_gb",))
            tpot = _pick_first(eff, ("tpot_ms_per_output_token",))

            if param_count is not None:
                best_param_count = param_count
            if param_b is not None:
                best_param_b = param_b
            if peak_alloc is not None:
                best_peak_alloc = peak_alloc if best_peak_alloc is None else max(best_peak_alloc, peak_alloc)
            if peak_resv is not None:
                best_peak_resv = peak_resv if best_peak_resv is None else max(best_peak_resv, peak_resv)
            if tpot is not None and t == "caption":
                best_tpot = tpot
            if eff:
                best_eff_source = str(EFF_ROOT / f"{m}_{t}.json")

            mt[t] = {
                "prediction_file": str(pred_path) if pred_path.exists() else "",
                "valid_count": valid_n,
                "runtime_seconds": runtime_s,
                "avg_latency_ms_per_sample": avg_ms,
                "log_file": str(log_path) if log_path is not None else "",
                "eff_probe_file": str(EFF_ROOT / f"{m}_{t}.json") if eff else "",
                "eff_probe": eff if eff else {},
            }

        by_model_task[m] = mt
        by_model[m] = {
            "param_count": best_param_count,
            "param_billion": best_param_b,
            "peak_vram_allocated_gb": best_peak_alloc,
            "peak_vram_reserved_gb": best_peak_resv,
            "caption_tpot_ms_per_output_token": best_tpot,
            "eff_probe_source": best_eff_source,
            "caption_latency_ms_per_sample": mt.get("caption", {}).get("avg_latency_ms_per_sample"),
            "description_latency_ms_per_sample": mt.get("description", {}).get("avg_latency_ms_per_sample"),
            "explanation_latency_ms_per_sample": mt.get("explanation", {}).get("avg_latency_ms_per_sample"),
            "caption_valid_count": mt.get("caption", {}).get("valid_count"),
            "description_valid_count": mt.get("description", {}).get("valid_count"),
            "explanation_valid_count": mt.get("explanation", {}).get("valid_count"),
        }

    out = {
        "model_order": MODEL_ORDER,
        "tasks": TASKS,
        "by_model": by_model,
        "by_model_task": by_model_task,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
