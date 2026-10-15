#!/bin/bash
# Score every cell of a run-matrix results dir against a GT dir.
# OFFLINE and FREE — no API. Run this after any run_matrix campaign.
#
#   ./scripts/score_matrix.sh pilot30_protocol10 results/pilot_matrix
#   ./scripts/score_matrix.sh output_nuscenes_protocol10 results/matrix_nuscenes
#
# Writes a primary coherence report and a secondary reference-agreement report
# per cell, then prints both without merging them into one score.
set -euo pipefail

GT_DIR="${1:?usage: score_matrix.sh <gt_dir> <matrix_dir>}"
MATRIX="${2:?usage: score_matrix.sh <gt_dir> <matrix_dir>}"
PY=".venv/bin/python"

printf '%-40s %7s %7s %7s | %7s %7s %7s\n' \
  "cell" "Mono" "Resp" "Local" "Ref-F1" "Ref-CAS" "Ref-CTMS"
for d in "$MATRIX"/*/; do
  d="${d%/}"
  # only real cell dirs (contain per-scenario predictions), skip *_report.json etc.
  ls "$d"/scenario_*/l3_pre.json >/dev/null 2>&1 || continue
  n="$(basename "$d")"
  $PY scripts/assemble_dataset.py --gt-dir "$GT_DIR" --pred-dir "$d" \
      --out-gt "$d/gt.json" --out-pred "$d/pred.json" >/dev/null
  $PY run_benchmark.py --gt "$d/gt.json" --pred "$d/pred.json" \
      --output "$d/report.json" >/dev/null
  $PY -m analysis.causal_coherence --scenario-dir "$GT_DIR" --pred-dir "$d" \
      --out "$d/coherence.json" >/dev/null
  $PY - "$d/report.json" "$d/coherence.json" "$n" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
c = json.load(open(sys.argv[2]))["axes"]
def axis(name):
    value = c[name]["mean"]
    return "--" if value is None else f"{value:.3f}"
print(f'{sys.argv[3]:<40} '
      f'{axis("monotonicity"):>7} {axis("responsiveness"):>7} '
      f'{axis("locality"):>7} | '
      f'{s.get("Causal_Foresight_F1_mean",0):7.3f} '
      f'{s.get("Counterfactual_Alignment_Score_mean",0):7.3f} '
      f'{s.get("CTMS_mean",0):7.3f}')
PY
done
echo "primary: ${MATRIX}/<cell>/coherence.json"
echo "secondary reference agreement: ${MATRIX}/<cell>/report.json"
