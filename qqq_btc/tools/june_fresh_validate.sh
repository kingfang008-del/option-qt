#!/usr/bin/env bash
# 6 月 fresh 下载验证：day_iv → monthly → bucketed → merge → norm → label → V4 infer
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
PY="/home/kingfang007/anaconda3/envs/ibkr/bin/python"
CODE_REF="a170dc6"
EXP="$HOME/train_data/june_fresh_experiment"
OPT1M="/mnt/s990/data/raw_1m/options_databento_june_fresh"
LOG="/tmp/june_fresh_pipeline.log"
CKPT="$REPO/checkpoints_qqq_v4/best.pth"
CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
ANCHOR="$REPO/qqq_btc/CONFIG/anchor_qqq_0dte.json"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export FEATURE_CONFIG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"

exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

cd "$REPO"
git show "${CODE_REF}:preprocess/ask_bid/feature_merge_option_raw.py" > preprocess/ask_bid/feature_merge_option_raw.py
git show "${CODE_REF}:qqq_btc/tools/label_pipeline.py" > qqq_btc/tools/label_pipeline.py

mkdir -p "$EXP"
log "=== June fresh pipeline start ==="

log "[1/7] day_iv (force: purge experiment day_iv)"
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

log "[2/7] iv_day2month → 2026-06"
"$PY" - <<PY
import glob, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from preprocess.ask_bid.iv_day2month import process_single_symbol

inp = "$EXP/quote_options_day_iv"
out = "$EXP/quote_options_monthly_iv"
files = sorted(glob.glob(f"{inp}/QQQ/standard/QQQ_*.parquet"))
print(f"day_iv files: {len(files)}")
res = process_single_symbol(("QQQ", files, out))
print(res)
PY

log "[3/7] options_locked_feature → bucketed 2026-06"
"$PY" - <<PY
from pathlib import Path
from preprocess.ask_bid.options_locked_feature import process_single_file
raw = Path("$EXP/quote_options_monthly_iv/QQQ/standard/2026-06.parquet")
out = Path("$EXP/quote_options_bucketed_v7")
res = process_single_file((raw, out, "QQQ"))
print(res or "bucketed ok")
PY

log "[4/7] feature_merge QQQ 2026-06 only"
"$PY" - <<'PY'
import json
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm

exp = Path.home() / "train_data/june_fresh_experiment"
fm.OPTION_MONTHLY_DIR = exp / "quote_options_monthly_iv"
fm.AGG_OPTION_MONTHLY_DIR = exp / "quote_options_bucketed_v7"
fm.OUTPUT_FEATURES_DIR = exp / "quote_features_raw"

cfg_path = Path("/home/kingfang007/文档/GitHub/option-qt/qqq_btc/CONFIG/slow_feature_qqq_v2.json")
with open(cfg_path) as f:
    config = json.load(f)
res = fm.process_stock_month("QQQ", "2026-06", config)
print(res)
out = fm.OUTPUT_FEATURES_DIR / "QQQ/regular/09:30-16:00/1min/2026-06.parquet"
print("raw out exists:", out.exists(), "rows check pending")
PY

log "[5/7] assemble test dir (Apr/May raw from main + fresh Jun) → norm → label"
RAW_MAIN="$HOME/train_data/quote_features_raw/QQQ/regular/09:30-16:00"
TEST_DIR="$EXP/quote_features_test/QQQ/regular/09:30-16:00/1min"
TEST5="$EXP/quote_features_test/QQQ/regular/09:30-16:00/5min"
mkdir -p "$TEST_DIR" "$TEST5"
for m in 2026-04 2026-05; do
  cp "$RAW_MAIN/1min/${m}.parquet" "$TEST_DIR/"
  cp "$RAW_MAIN/5min/${m}.parquet" "$TEST5/" 2>/dev/null || true
done
cp "$EXP/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-06.parquet" "$TEST_DIR/"
cp "$EXP/quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-06.parquet" "$TEST5/" 2>/dev/null || \
  cp "$RAW_MAIN/5min/2026-06.parquet" "$TEST5/" 2>/dev/null || true

"$PY" - <<'PY'
import json, os
from pathlib import Path
import preprocess.ask_bid.apply_rolling_norm_standalone as norm

exp = Path.home() / "train_data/june_fresh_experiment/quote_features_test/QQQ/regular/09:30-16:00/1min"
cfg = Path(os.environ["FEATURE_CONFIG"])
with open(cfg) as f:
    config = json.load(f)
norm_cols = norm.load_target_features(cfg)
res = norm.process_single_directory((exp, norm_cols))
print("norm:", res)
PY

"$PY" qqq_btc/tools/label_pipeline.py \
  --input "$EXP/quote_features_test/QQQ/regular/09:30-16:00/1min" \
  --output "$EXP/quote_features_test/QQQ/regular/09:30-16:00/1min" \
  --symbol QQQ \
  --anchor-config "$ANCHOR" \
  --report "/tmp/june_fresh_label_report.json"

log "[6/7] V4 infer June — fresh vs gold"
OUT_FRESH="/tmp/v4_june_fresh_redownload"
OUT_GOLD="/tmp/v4_june_gold"

run_eval() {
  local name="$1" feat_base="$2" out="$3"
  local tmp="$out/_feat"
  rm -rf "$tmp"
  mkdir -p "$tmp/QQQ/regular/09:30-16:00/1min" "$tmp/QQQ/regular/09:30-16:00/5min"
  if [[ "$feat_base" == *"_bak_pre4c"* ]]; then
    cp "$feat_base/regular/09:30-16:00/1min/2026-06.parquet" "$tmp/QQQ/regular/09:30-16:00/1min/"
    cp "$feat_base/regular/09:30-16:00/5min/2026-06.parquet" "$tmp/QQQ/regular/09:30-16:00/5min/"
  else
    cp "$feat_base/QQQ/regular/09:30-16:00/1min/2026-06.parquet" "$tmp/QQQ/regular/09:30-16:00/1min/"
    cp "$feat_base/QQQ/regular/09:30-16:00/5min/2026-06.parquet" "$tmp/QQQ/regular/09:30-16:00/5min/" 2>/dev/null || true
  fi
  mkdir -p "$out"
  "$PY" qqq_btc/tools/eval_test_set.py \
    --checkpoint "$CKPT" --config "$CFG" \
    --feature-root "$tmp" \
    --option-1m-root "$OPT1M" \
    --output-dir "$out" --device cuda 2>&1 | tail -4
}

run_eval fresh "$EXP/quote_features_test" "$OUT_FRESH"
run_eval gold "$HOME/train_data/_bak_pre4c/quote_features_test_QQQ" "$OUT_GOLD"

log "[7/7] compare"
"$PY" - <<'PY'
import json
import pandas as pd
from pathlib import Path

def load_summary(p):
    return json.loads(Path(p).read_text())

rows = []
for tag, root in [("gold_7/5", "/tmp/v4_june_gold"), ("june_fresh", "/tmp/v4_june_fresh_redownload")]:
    s = load_summary(f"{root}/replay_summary.json")
    m = s.get("label_metrics", {})
    rows.append({
        "tag": tag,
        "ic": m.get("ic", 0),
        "trades": s.get("trades", 0),
        "return_pct": s.get("total_net_return", 0) * 100,
        "legs": s.get("trades_by_leg", {}),
    })

gold = pd.read_parquet("/tmp/v4_june_gold/test_infer.parquet")
fresh = pd.read_parquet("/tmp/v4_june_fresh_redownload/test_infer.parquet")
mm = gold.merge(fresh, on="timestamp", suffixes=("_gold", "_fresh"))

bak = pd.read_parquet(Path.home()/ "train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/1min/2026-06.parquet")
exp = pd.read_parquet(Path.home()/ "train_data/june_fresh_experiment/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-06.parquet")
for df in (bak, exp):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
fm = bak.merge(exp, on="timestamp", suffixes=("_g", "_f"))

print("\n=== V4 · 2026-06 重下验证 ===")
for r in rows:
    print(f"{r['tag']:14} IC={r['ic']:.4f} trades={r['trades']:3d} return={r['return_pct']:+7.2f}% legs={r['legs']}")
print(f"\ninfer net_edge corr (fresh vs gold): {mm.net_edge_gold.corr(mm.net_edge_fresh):.4f}")
print(f"feature options_vw_delta corr (fresh vs gold): {fm.options_vw_delta_g.corr(fm.options_vw_delta_f):.4f}")

out = Path("/home/kingfang007/文档/GitHub/option-qt/qqq_btc/results/june_fresh_redownload_summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
import json as js
out.write_text(js.dumps({"rows": rows, "infer_corr": float(mm.net_edge_gold.corr(mm.net_edge_fresh)), "delta_corr": float(fm.options_vw_delta_g.corr(fm.options_vw_delta_f))}, indent=2))
print(f"wrote {out}")
PY

log "=== DONE log=$LOG ==="
