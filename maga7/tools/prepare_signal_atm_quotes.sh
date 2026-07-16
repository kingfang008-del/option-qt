#!/usr/bin/env bash
# Download 1s quotes for Mag7 signal-time ATM lock map (step2 sniper).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MAP="${SIGNAL_ATM_MAP:-$HOME/train_data/locked_targets_map_maga7_signal_atm_jan_jul.parquet}"
OUT="${SIGNAL_ATM_QUOTE_DIR:-/mnt/s990/data/raw_1s/maga7_mf10_signal_atm}"
STOCK="${STOCK_1S_DIR:-/mnt/s990/data/raw_1s/stocks}"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
WORKERS="${MAX_WORKERS:-16}"
CONTRACT_WORKERS="${CONTRACT_WORKERS:-4}"

if [[ -z "${MASSIVE_API_KEY:-}" && -z "${POLYGON_API_KEY:-}" ]]; then
  echo "set MASSIVE_API_KEY or POLYGON_API_KEY" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "map=$MAP"
echo "out=$OUT"
echo "day_workers=$WORKERS contract_workers=$CONTRACT_WORKERS"
"$PY" -u "$ROOT/preprocess/download/step2_polygon_second_sniper_v1.py" \
  --target-map "$MAP" \
  --output-dir "$OUT" \
  --stock-output-dir "$STOCK" \
  --start-date "${START_DATE:-2026-01-02}" \
  --end-date "${END_DATE:-2026-07-13}" \
  --max-workers "$WORKERS" \
  --contract-workers "$CONTRACT_WORKERS" \
  --window-start "${WINDOW_START:-10:00}" \
  --window-end "${WINDOW_END:-15:00}" \
  --no-download-stock \
  --allow-partial \
  "$@"
