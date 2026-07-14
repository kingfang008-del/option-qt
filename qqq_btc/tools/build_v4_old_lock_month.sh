#!/usr/bin/env bash
# 从 dte1_options_old_lock 构建单月特征 + V4 infer（供 regime profile 对照）
# 用法: bash qqq_btc/tools/build_v4_old_lock_month.sh 2026-05
set -euo pipefail
YM="${1:?YYYY-MM}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export FEATURE_CONFIG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
export LOCKED_TARGETS_MAP="${LOCK_MAP:-$HOME/train_data/locked_targets_map_old_style_trading_1dte.parquet}"

Y="${YM:0:4}"; M="${YM:5:2}"
START="$YM-01"
# end of month
END="$("$PY" - <<PY
from calendar import monthrange
print(f"$Y-$M-{monthrange(int('$Y'), int('$M'))[1]:02d}")
PY
)"
# warmstart prior month
PREV="$("$PY" - <<PY
y, m = int("$Y"), int("$M")
if m == 1:
    print(f"{y-1:04d}-12")
else:
    print(f"{y:04d}-{m-1:02d}")
PY
)"

EXP="${EXP:-$HOME/train_data/${YM//-/}_v4_old_lock}"
# nicer name: april_v4 style → use month name folder
case "$M" in
  04) EXP="${EXP_OVERRIDE:-$HOME/train_data/april_v4_old_lock}" ;;
  05) EXP="${EXP_OVERRIDE:-$HOME/train_data/may_v4_old_lock}" ;;
  06) EXP="${EXP_OVERRIDE:-$HOME/train_data/june_v4_old_lock}" ;;
  07) EXP="${EXP_OVERRIDE:-$HOME/train_data/july_v4_old_lock}" ;;
  *) EXP="${EXP_OVERRIDE:-$HOME/train_data/${YM}_v4_old_lock}" ;;
esac

RAW1S="${RAW1S:-/mnt/s990/data/raw_1s/dte1_options_old_lock}"
OPT1M="$EXP/options_1m_${YM}"
CKPT="${CKPT:-checkpoint/checkpoints_qqq_v4/best.pth}"
CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
ANCHOR="$REPO/qqq_btc/CONFIG/anchor_qqq_1dte.json"
OUT="${OUT:-qqq_btc/results/v4_${YM}_old_lock}"
SEED="${SEED:-42}"

mkdir -p "$EXP" "$OPT1M/QQQ" "$OUT"
echo "[build] ym=$YM start=$START end=$END warm=$PREV exp=$EXP"

"$PY" preprocess/download/step3_databento_aggregate_1s_to_1m.py \
  --input-dir "$RAW1S" --output-dir "$OPT1M" --symbol QQQ \
  --date-from "$START" --date-to "$END" --force

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

"$PY" - <<PY
import glob
from preprocess.ask_bid.iv_day2month import process_single_symbol
inp = "$EXP/quote_options_day_iv"
out = "$EXP/quote_options_monthly_iv"
files = sorted(glob.glob(f"{inp}/QQQ/**/*.parquet", recursive=True))
print("day_iv", len(files), process_single_symbol(("QQQ", files, out)))
PY

"$PY" - <<PY
from pathlib import Path
from preprocess.ask_bid.options_locked_feature import process_single_file
raw = Path("$EXP/quote_options_monthly_iv/QQQ/standard/$YM.parquet")
out = Path("$EXP/quote_options_bucketed_v7")
print(process_single_file((raw, out, "QQQ")) or "bucketed ok")
PY

export EXP YM FEATURE_CONFIG
"$PY" - <<'PY'
import json, os
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm
exp = Path(os.environ["EXP"])
ym = os.environ["YM"]
fm.OPTION_MONTHLY_DIR = exp / "quote_options_monthly_iv"
fm.AGG_OPTION_MONTHLY_DIR = exp / "quote_options_bucketed_v7"
fm.OUTPUT_FEATURES_DIR = exp / "quote_features_raw"
cfg = json.loads(Path(os.environ["FEATURE_CONFIG"]).read_text())
print(fm.process_stock_month("QQQ", ym, cfg))
PY

TEST1="$EXP/quote_features_test/QQQ/regular/09:30-16:00/1min"
TEST5="$EXP/quote_features_test/QQQ/regular/09:30-16:00/5min"
mkdir -p "$TEST1" "$TEST5"
BAK1="$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00"
if [[ -f "$BAK1/1min/${PREV}.parquet" ]]; then
  cp "$BAK1/1min/${PREV}.parquet" "$TEST1/"
  cp "$BAK1/5min/${PREV}.parquet" "$TEST5/" 2>/dev/null || true
fi
cp "$EXP/quote_features_raw/QQQ/regular/09:30-16:00/1min/${YM}.parquet" "$TEST1/"
cp "$EXP/quote_features_raw/QQQ/regular/09:30-16:00/5min/${YM}.parquet" "$TEST5/" 2>/dev/null || true

"$PY" - <<PY
import os
from pathlib import Path
import preprocess.ask_bid.apply_rolling_norm_standalone as norm
exp = Path("$TEST1")
cfg = Path(os.environ["FEATURE_CONFIG"])
print("norm:", norm.process_single_directory((exp, norm.load_target_features(cfg))))
PY

ANCHOR_OL="$EXP/anchor_old_lock_label.json"
"$PY" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$ANCHOR").read_text())
cfg["paths"] = {
  "raw_iv_dir": str(Path.home() / "train_data/nq_options_day_iv"),
  "sniper_option_dir": "$RAW1S",
  "day_iv_dir": "$EXP/quote_options_day_iv",
  "locked_targets_output": "$LOCKED_TARGETS_MAP",
}
Path("$ANCHOR_OL").write_text(json.dumps(cfg, indent=2) + "\n")
print("anchor", "$ANCHOR_OL")
PY

"$PY" qqq_btc/tools/label_pipeline.py \
  --input "$TEST1" --output "$TEST1" --symbol QQQ \
  --anchor-config "$ANCHOR_OL" --report "$EXP/label_report_${YM}.json"

TMP="$EXP/eval_feat_${YM}_only"
rm -rf "$TMP"
mkdir -p "$TMP/QQQ/regular/09:30-16:00/1min" "$TMP/QQQ/regular/09:30-16:00/5min"
cp "$TEST1/${YM}.parquet" "$TMP/QQQ/regular/09:30-16:00/1min/"
cp "$TEST5/${YM}.parquet" "$TMP/QQQ/regular/09:30-16:00/5min/" 2>/dev/null || true

EVAL_DIR="$OUT/infer"
mkdir -p "$EVAL_DIR"
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT" --config "$CFG" \
  --feature-root "$TMP" --option-1m-root "$OPT1M" \
  --call-bucket 2 --put-bucket 0 \
  --output-dir "$EVAL_DIR" --seed "$SEED" --device "${DEVICE:-cuda}" --live-replay

echo "DONE infer=$EVAL_DIR/test_infer.parquet raw1=$EXP/quote_features_raw/QQQ/regular/09:30-16:00/1min/${YM}.parquet"
