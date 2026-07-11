#!/usr/bin/env bash
# July week1 (2026-07-01..09 trading days) V4 feature build + replay
# Uses freshly downloaded Massive/Polygon 1DTE quotes under raw_1s/dte1_options
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
PY="/home/kingfang007/anaconda3/envs/ibkr/bin/python"
EXP="$HOME/train_data/july_w1_v4_experiment"
OPT1M_SRC="/mnt/s990/data/raw_1m/dte1_options"
OPT1M="$EXP/options_1m_july_w1"
CKPT="$REPO/checkpoint/checkpoints_qqq_v4/best.pth"
CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
CFG_MERGE="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
ANCHOR="$REPO/qqq_btc/CONFIG/anchor_qqq_1dte.json"
OUT="$REPO/qqq_btc/results/v4_original_jul2026_w1_dte1"
LOG="$EXP/pipeline.log"
STOCK_1S="/mnt/s990/data/raw_1s/stocks"
STOCK_RESAMP="$HOME/train_data/spnq_train_resampled"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export FEATURE_CONFIG="$CFG_MERGE"
export LOCKED_TARGETS_MAP="$HOME/train_data/locked_targets_map_1dte_jul2026_w1.parquet"

mkdir -p "$EXP" "$OPT1M/QQQ" "$OUT"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
cd "$REPO"

log "=== July W1 V4 pipeline start ==="

log "[0] stage July option 1m only"
rm -rf "$OPT1M/QQQ"
mkdir -p "$OPT1M/QQQ"
for d in 2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09; do
  cp "$OPT1M_SRC/QQQ/QQQ_${d}.parquet" "$OPT1M/QQQ/"
done
ls "$OPT1M/QQQ" | wc -l

log "[0b] resample QQQ stock 1s → 1min/5min for 2026-07"
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

src = Path("/mnt/s990/data/raw_1s/stocks/QQQ")
out_base = Path.home() / "train_data/spnq_train_resampled/QQQ/regular/09:30-16:00"
days = ["2026-07-01","2026-07-02","2026-07-06","2026-07-07","2026-07-08","2026-07-09"]
parts = []
for d in days:
    p = src / f"QQQ_{d}.parquet"
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    # RTH only
    t = df["timestamp"].dt.time
    import datetime as dt
    df = df[(t >= dt.time(9,30)) & (t < dt.time(16,0))].copy()
    parts.append(df)
full = pd.concat(parts, ignore_index=True).sort_values("timestamp")
full = full.set_index("timestamp")

def agg(freq):
    g = full.resample(freq, label="right", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["close"]).reset_index()
    return g

for freq, name in [("1min", "1min"), ("5min", "5min")]:
    out_dir = out_base / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = agg(freq)
    out_path = out_dir / "2026-07.parquet"
    out.to_parquet(out_path, index=False)
    print(name, "rows", len(out), "->", out_path)
PY

# VIXY warm: if missing July, leave as-is (feature_merge may degrade vix)

log "[1] day_iv"
rm -rf "$EXP/quote_options_day_iv"
"$PY" - <<PY
import multiprocessing
from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass
calc = OptionIVCalculator(
    db_path="/home/kingfang007/notebook/stocks.db",
    option_root="$OPT1M",
    data_root="/home/kingfang007/train_data/spnq_train_resampled",
    iv_option_root="$EXP/quote_options_day_iv",
)
calc.run(max_concurrent_stocks=4)
PY

log "[2] iv_day2month"
"$PY" - <<PY
import glob
from preprocess.ask_bid.iv_day2month import process_single_symbol
inp = "$EXP/quote_options_day_iv"
out = "$EXP/quote_options_monthly_iv"
files = sorted(glob.glob(f"{inp}/QQQ/**/*.parquet", recursive=True))
print("day_iv files", len(files))
print(process_single_symbol(("QQQ", files, out)))
PY

log "[3] options_locked_feature"
"$PY" - <<PY
from pathlib import Path
from preprocess.ask_bid.options_locked_feature import process_single_file
raw = Path("$EXP/quote_options_monthly_iv/QQQ/standard/2026-07.parquet")
out = Path("$EXP/quote_options_bucketed_v7")
print(process_single_file((raw, out, "QQQ")) or "bucketed ok")
print("exists", (out/"QQQ/standard/2026-07.parquet").exists() or list(out.rglob("2026-07.parquet")))
PY

log "[4] feature_merge QQQ 2026-07"
"$PY" - <<'PY'
import json
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm

exp = Path.home() / "train_data/july_w1_v4_experiment"
fm.OPTION_MONTHLY_DIR = exp / "quote_options_monthly_iv"
fm.AGG_OPTION_MONTHLY_DIR = exp / "quote_options_bucketed_v7"
fm.OUTPUT_FEATURES_DIR = exp / "quote_features_raw"

cfg_path = Path("/home/kingfang007/文档/GitHub/option-qt/qqq_btc/CONFIG/slow_feature_qqq_v2.json")
with open(cfg_path) as f:
    config = json.load(f)
print(fm.process_stock_month("QQQ", "2026-07", config))
for res in ("1min", "5min"):
    p = fm.OUTPUT_FEATURES_DIR / f"QQQ/regular/09:30-16:00/{res}/2026-07.parquet"
    print(res, "exists", p.exists(), "rows", __import__("pandas").read_parquet(p).shape[0] if p.exists() else 0)
PY

log "[5] assemble + rolling_norm + label"
TEST1="$EXP/quote_features_test/QQQ/regular/09:30-16:00/1min"
TEST5="$EXP/quote_features_test/QQQ/regular/09:30-16:00/5min"
mkdir -p "$TEST1" "$TEST5"
# warm-start with June from bak or current test
if [[ -f "$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/1min/2026-06.parquet" ]]; then
  cp "$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/1min/2026-06.parquet" "$TEST1/"
  cp "$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/5min/2026-06.parquet" "$TEST5/" 2>/dev/null || true
elif [[ -f "$HOME/train_data/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-06.parquet" ]]; then
  cp "$HOME/train_data/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-06.parquet" "$TEST1/"
  cp "$HOME/train_data/quote_features_test/QQQ/regular/09:30-16:00/5min/2026-06.parquet" "$TEST5/" 2>/dev/null || true
fi
cp "$EXP/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet" "$TEST1/"
cp "$EXP/quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-07.parquet" "$TEST5/" 2>/dev/null || true

"$PY" - <<'PY'
import os
from pathlib import Path
import preprocess.ask_bid.apply_rolling_norm_standalone as norm
exp = Path.home() / "train_data/july_w1_v4_experiment/quote_features_test/QQQ/regular/09:30-16:00/1min"
cfg = Path(os.environ["FEATURE_CONFIG"])
norm_cols = norm.load_target_features(cfg)
print("norm:", norm.process_single_directory((exp, norm_cols)))
PY

"$PY" qqq_btc/tools/label_pipeline.py \
  --input "$TEST1" \
  --output "$TEST1" \
  --symbol QQQ \
  --anchor-config "$ANCHOR" \
  --report "$EXP/label_report_july_w1.json"

log "[6] V4 eval July only"
TMP="$EXP/eval_feat_july_only"
rm -rf "$TMP"
mkdir -p "$TMP/QQQ/regular/09:30-16:00/1min" "$TMP/QQQ/regular/09:30-16:00/5min"
cp "$TEST1/2026-07.parquet" "$TMP/QQQ/regular/09:30-16:00/1min/"
cp "$TEST5/2026-07.parquet" "$TMP/QQQ/regular/09:30-16:00/5min/" 2>/dev/null || true

"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --feature-root "$TMP" \
  --option-1m-root "$OPT1M" \
  --output-dir "$OUT" \
  --device cuda

log "[7] summarize"
"$PY" - <<PY
import json
from pathlib import Path
import pandas as pd
out = Path("$OUT")
s = json.loads((out/"replay_summary.json").read_text())
tr = pd.read_parquet(out/"replay_trades.parquet") if (out/"replay_trades.parquet").exists() else pd.DataFrame()
print("=== V4 July W1 replay ===")
print(json.dumps({k:s[k] for k in ["trades","total_net_return","hit_rate","position_frac","trades_by_leg","label_metrics"] if k in s}, indent=2, default=str))
if len(tr):
    tr["entry_ts"]=pd.to_datetime(tr["entry_ts"]).dt.tz_convert("America/New_York")
    tr["date"]=tr["entry_ts"].dt.strftime("%Y-%m-%d")
    print(tr.groupby("date").agg(n=("net_return","size"), sum_ret=("net_return","sum"), hit=("net_return", lambda x:(x>0).mean())).to_string())
PY

log "=== DONE out=$OUT log=$LOG ==="
