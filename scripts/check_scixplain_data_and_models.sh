#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCICAP_ROOT="${SCICAP_ROOT:-$ROOT/dataset/scicap_mlbcap_node_diagram_v2}"
SCISTRUCT_SPLIT="${SCISTRUCT_SPLIT:-$ROOT/dataset/SciStruct/dataset_split_811}"
BENCH_ROOT="${BENCH_ROOT:-$ROOT/Benchmark/datasets}"

req=(
  "$SCICAP_ROOT/dataset_split/train.json"
  "$SCICAP_ROOT/dataset_split/val.json"
  "$SCICAP_ROOT/dataset_split/test.json"
  "$SCICAP_ROOT/images_store"
  "$SCISTRUCT_SPLIT/train.json"
  "$SCISTRUCT_SPLIT/val.json"
  "$SCISTRUCT_SPLIT/test.json"
  "$ROOT/dataset/SciStruct/dataset"
  "$BENCH_ROOT/caption/test.jsonl"
  "$BENCH_ROOT/description/test.jsonl"
  "$BENCH_ROOT/images/scicap"
  "$BENCH_ROOT/images/scistruct"
)

echo "[check] data paths"
miss=0
for p in "${req[@]}"; do
  if [[ -e "$p" ]]; then
    echo "OK $p"
  else
    echo "MISSING $p"
    miss=$((miss+1))
  fi
done

echo "[check] optional checkpoints"
optional_ckpts=(
  "$ROOT/checkpoints/phi-sig/config.json"
  "$ROOT/checkpoints/ai2d_teacher_v3_edge/ckpt_best.pt"
  "$ROOT/checkpoints/visual_student_scistruct_scicap_full_v2/ckpt_last.pt"
)
for p in "${optional_ckpts[@]}"; do
  if [[ -e "$p" ]]; then
    echo "OPTIONAL_OK $p"
  else
    echo "OPTIONAL_MISSING $p"
  fi
done

echo "[check] quick counts"
python - <<'PY'
import json
from pathlib import Path
for p in [
  Path('dataset/scicap_mlbcap_node_diagram_v2/dataset_split/train.json'),
  Path('dataset/scicap_mlbcap_node_diagram_v2/dataset_split/val.json'),
  Path('dataset/scicap_mlbcap_node_diagram_v2/dataset_split/test.json'),
  Path('dataset/SciStruct/dataset_split_811/train.json'),
  Path('dataset/SciStruct/dataset_split_811/val.json'),
  Path('dataset/SciStruct/dataset_split_811/test.json'),
]:
    if not p.exists():
      print('MISS', p)
      continue
    d=json.loads(p.read_text(encoding='utf-8'))
    n=len(d) if isinstance(d,list) else len(d.get('data',[])) if isinstance(d,dict) else -1
    print('COUNT', p, n)
PY

if [[ "$miss" -gt 0 ]]; then
  echo "[summary] missing=$miss"
  exit 2
fi

echo "[summary] all required public data paths found"
