#!/usr/bin/env bash
# QQQ 0DTE 官方重建：dynamic map + prefer_primary_gapfill
# 契约: qqq_btc/CONFIG/feature_contract_0dte_prefer_primary.json
#
# 用法:
#   bash qqq_btc/tools/rebuild_0dte_prefer_primary.sh
#   DATE_FROM=2023-03-01 DATE_TO=2026-06-30 bash qqq_btc/tools/rebuild_0dte_prefer_primary.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

DATE_FROM="${DATE_FROM:-2025-07-01}"
DATE_TO="${DATE_TO:-2025-12-31}"
OUT_ROOT="${OUT_ROOT:-$HOME/train_data/builds/0dte_prefer_primary}"
TOOL="$REPO/qqq_btc/tools/reproduce_bak_lineage.py"
CONTRACT="$REPO/qqq_btc/CONFIG/feature_contract_0dte_prefer_primary.json"
AB_OUT="$REPO/qqq_btc/results/feature_drift_rootcause_2025h2/ab_prefer_primary_vs_single.json"
MANIFEST="$OUT_ROOT/rebuild_manifest.json"

MONTHS="$($PY - <<PY
import pandas as pd
idx = pd.period_range("${DATE_FROM}"[:7], "${DATE_TO}"[:7], freq="M")
print(",".join(str(p) for p in idx))
PY
)"

echo "=== contract ==="
"$PY" - <<PY
import json
c=json.load(open("${CONTRACT}"))
print(c["name"], c["version"], c["status"])
print("map:", c["lock_map"]["path"])
print("logic:", c["assemble_1m"]["logic"])
PY

mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "DATE_FROM=$DATE_FROM DATE_TO=$DATE_TO"
echo "MONTHS=$MONTHS"

"$PY" - <<PY
import json, time
from pathlib import Path
Path("${MANIFEST}").write_text(json.dumps({
  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
  "date_from": "${DATE_FROM}",
  "date_to": "${DATE_TO}",
  "months": "${MONTHS}".split(","),
  "out_root": "${OUT_ROOT}",
  "contract": "${CONTRACT}",
  "status": "running",
}, indent=2, ensure_ascii=False))
PY

echo "=== [1/6] assemble prefer_primary 1m ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" assemble --date-from "$DATE_FROM" --date-to "$DATE_TO"

echo "=== [2/6] day-iv ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" day-iv --date-from "$DATE_FROM" --date-to "$DATE_TO" --force

echo "=== [3/6] monthly + bucketed ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" monthly-bucketed --months "$MONTHS"

echo "=== [4/6] feature-merge ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" feature-merge --months "$MONTHS"

echo "=== [5/6] split + rolling-norm (per stage) ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" split-norm

echo "=== [6/6] label train/val/test ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" label-stages

echo "=== validate sample months vs bak (if bak present) ==="
for mo in 2025-07 2025-08 2025-12; do
  if [[ -f "$HOME/train_data/_bak_pre4c/quote_options_monthly_iv_QQQ/standard/${mo}.parquet" ]]; then
    "$PY" "$TOOL" --out-root "$OUT_ROOT" validate-monthly --month "$mo" \
      > "$OUT_ROOT/validate_monthly_${mo}.stdout.json" || true
  fi
done

# A/B still compares bak_lineage_reproduce by default; also write OUT_ROOT note
"$PY" - <<PY
import json, time
from pathlib import Path
p=Path("${MANIFEST}")
m=json.loads(p.read_text())
m["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%S")
m["status"]="done"
p.write_text(json.dumps(m, indent=2, ensure_ascii=False))
PY

echo "DONE. Artifacts:"
echo "  features: $OUT_ROOT"
echo "  manifest: $MANIFEST"
echo "  contract: $CONTRACT"
