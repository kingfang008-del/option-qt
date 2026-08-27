#!/usr/bin/env bash
# Build Mag7 (+QQQ) option Greeks / locked surface for path-hold research.
#
# Stage A: day_iv + thin high_features via option_cac_day_vectorized.py
# Stage B: locked bucketed_v7 via options_locked_feature.py
#           (needs quote_options_monthly_iv/{SYM}/standard with bucket_id)
#
# Usage:
#   bash maga7/SHELL/run_maga7_option_surface_build.sh --start 2026-05-01 --end 2026-07-20
#   bash maga7/SHELL/run_maga7_option_surface_build.sh --stage a --symbols mag7,QQQ
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
START="2026-05-01"
END="2026-07-20"
STAGE="ab"
SYMBOLS="mag7,QQQ"
STOCK_WORKERS=2
DAY_WORKERS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --symbols) SYMBOLS="$2"; shift 2 ;;
    --stock-workers) STOCK_WORKERS="$2"; shift 2 ;;
    --day-workers) DAY_WORKERS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
echo "[surface] root=$ROOT start=$START end=$END stage=$STAGE symbols=$SYMBOLS"

# Expand mag7 token for locked-feature loop
expand_syms() {
  local raw="$1"
  raw="${raw//mag7/NVDA,TSLA,AAPL,AMZN,META,MSFT,AMD,GOOGL}"
  echo "$raw" | tr ',' '\n' | sed '/^$/d' | sort -u | tr '\n' ',' | sed 's/,$//'
}
SYMS_CSV="$(expand_syms "$SYMBOLS")"

if [[ "$STAGE" == "a" || "$STAGE" == "ab" || "$STAGE" == "A" || "$STAGE" == "AB" ]]; then
  echo "[surface] Stage A: option_cac_day_vectorized"
  # CLI accepts mag7 shortcut; also pass QQQ explicitly when requested
  CAC_SYM="$SYMBOLS"
  if [[ "$SYMBOLS" == *QQQ* && "$SYMBOLS" == *mag7* ]]; then
    CAC_SYM="mag7"
    EXTRA_QQQ=1
  else
    EXTRA_QQQ=0
  fi
  "$PY" preprocess/raw_data_deal/option_cac_day_vectorized.py \
    --symbols "$CAC_SYM" \
    --start "$START" \
    --end "$END" \
    --stock-workers "$STOCK_WORKERS" \
    --day-workers "$DAY_WORKERS"
  if [[ "$EXTRA_QQQ" == "1" || "$SYMBOLS" == "QQQ" || "$SYMBOLS" == *",QQQ"* ]]; then
    "$PY" preprocess/raw_data_deal/option_cac_day_vectorized.py \
      --symbols QQQ \
      --start "$START" \
      --end "$END" \
      --stock-workers 1 \
      --day-workers "$DAY_WORKERS"
  fi
fi

if [[ "$STAGE" == "b" || "$STAGE" == "ab" || "$STAGE" == "B" || "$STAGE" == "AB" ]]; then
  echo "[surface] Stage B: options_locked_feature (monthly_iv/standard → bucketed_v7)"
  echo "[surface] NOTE: Mag7 needs quote_options_monthly_iv/{SYM}/standard with bucket_id."
  echo "[surface]       QQQ already has this; Mag7 may need feature_merge / sniper monthly build first."
  "$PY" - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "$ROOT")
from preprocess.ask_bid.options_locked_feature import process_single_file

raw_root = Path.home() / "train_data/quote_options_monthly_iv"
out_root = Path.home() / "train_data/quote_options_bucketed_v7"
syms = [s.strip().upper() for s in "$SYMS_CSV".split(",") if s.strip()]
start, end = "$START", "$END"
tasks = []
for sym in syms:
    src = raw_root / sym / "standard"
    if not src.is_dir():
        print(f"[skip] no monthly_iv/standard for {sym}: {src}")
        continue
    for p in sorted(src.glob("*.parquet")):
        # month files like 2026-05.parquet
        ym = p.stem
        if len(ym) >= 7:
            # include month if overlaps [start,end]
            if ym > end[:7] or ym < start[:7]:
                continue
        tasks.append((p, out_root, sym))
print(f"[surface] locked tasks={len(tasks)} syms={syms}")
ok = err = 0
for t in tasks:
    res = process_single_file(t)
    if res:
        print("[warn]", res)
        err += 1
    else:
        ok += 1
print(f"[surface] Stage B done ok={ok} warn={err}")
PY
fi

echo "[surface] done. Inspect:"
echo "  ~/train_data/nq_options_day_iv/{SYM}/"
echo "  ~/train_data/quote_options_bucketed_v7/{SYM}/"
