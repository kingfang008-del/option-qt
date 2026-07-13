#!/usr/bin/env bash
# July W1 诚实离线重建：开盘窗 step1 锁约 → Databento 1s → 1m → day_iv → features → frozen_norm
#
# 与 july_w1_v4_databento（API ladder 开盘 09:40 近 ATM）不同：
#   本脚本用 step1_build_target_map.py + anchor_qqq_1dte_4bucket（开盘窗 10min δ 锁）。
# Jul1/2 已有 nq_options_day_iv；其余日缺链则跳过（见 report）。
#
# 用法:
#   DAYS="2026-07-01" bash qqq_btc/tools/rebuild_july_w1_honest_openwin_features.sh
#   bash qqq_btc/tools/rebuild_july_w1_honest_openwin_features.sh   # 有 nq 的全部日
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

EXP="${EXP:-$HOME/train_data/july_w1_v4_honest_openwin}"
MAP_OUT="${MAP_OUT:-$HOME/train_data/locked_targets_map_1dte_jul2026_openwin.parquet}"
ANCHOR="$REPO/preprocess/CONFIG/anchor_qqq_1dte_4bucket.json"
RAW_IV="${RAW_IV:-$HOME/train_data/nq_options_day_iv}"
OPT_1S="${OPT_1S:-/mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s}"
OPT_1M="$EXP/options_1m"
CFG_MERGE="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
CFG_V4="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
FROZEN_NORM="${FROZEN_NORM:-$REPO/qqq_btc/CONFIG/frozen_norm_qqq_daily.npz}"
STOCK_RESAMP="${STOCK_RESAMP:-$HOME/train_data/spnq_train_resampled}"
REPORT_DIR="${REPORT_DIR:-$REPO/qqq_btc/results/july_w1_honest_openwin_rebuild}"
API_LADDER_MAP="${API_LADDER_MAP:-$HOME/train_data/locked_targets_map_1dte_jul2026_w1.parquet}"

if [[ -n "${DAYS:-}" ]]; then
  # shellcheck disable=SC2206
  DAY_ARR=($DAYS)
else
  # 仅跑 nq 已有链的日子（当前 07-01/02）；其余需先补 nq_options_day_iv
  DAY_ARR=(2026-07-01 2026-07-02)
fi
DATE_FROM="${DAY_ARR[0]}"
DATE_TO="${DAY_ARR[-1]}"

mkdir -p "$EXP" "$OPT_1M/QQQ" "$REPORT_DIR" "$(dirname "$MAP_OUT")"
LOG="$REPORT_DIR/rebuild.log"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

log "=== honest openwin rebuild ==="
log "days=${DAY_ARR[*]} exp=$EXP map=$MAP_OUT"

log "[1] step1 open-window lock (1dte_4bucket)"
"$PY" preprocess/download/step1_build_target_map.py \
  --config "$ANCHOR" \
  --start-date "$DATE_FROM" \
  --end-date "$DATE_TO" \
  --raw-dir "$RAW_IV" \
  --output "$MAP_OUT"

log "[1b] compare vs API ladder map"
export MAP_OUT API_LADDER_MAP REPORT_DIR
"$PY" - <<'PY'
import json
from pathlib import Path
import pandas as pd
import os
ow = pd.read_parquet(os.environ["MAP_OUT"])
api_p = Path(os.environ["API_LADDER_MAP"])
rep = {"openwin_map": os.environ["MAP_OUT"], "days": sorted(ow["date_str"].astype(str).unique().tolist()), "per_day": {}}
if api_p.exists():
    api = pd.read_parquet(api_p)
    for d in sorted(ow["date_str"].astype(str).unique()):
        a = set(api.loc[api.date_str.astype(str)==d, "contract_symbol"].astype(str))
        o = set(ow.loc[ow.date_str.astype(str)==d, "contract_symbol"].astype(str))
        rep["per_day"][d] = {
            "same": a == o,
            "n_openwin": len(o),
            "n_api": len(a),
            "only_openwin": sorted(o - a),
            "only_api": sorted(a - o),
            "openwin": sorted(o),
            "api": sorted(a),
        }
else:
    rep["api_ladder_missing"] = str(api_p)
Path(os.environ["REPORT_DIR"], "lock_map_compare.json").write_text(
    json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(rep, indent=2, ensure_ascii=False))
PY

export LOCKED_TARGETS_MAP="$MAP_OUT"
export FEATURE_CONFIG="$CFG_MERGE"

log "[2] Databento sniper 1s for openwin contracts"
"$PY" preprocess/download/step2_databento_second_sniper_v1.py \
  --target-map "$MAP_OUT" \
  --output-dir "$OPT_1S" \
  --symbol QQQ \
  --date-from "$DATE_FROM" \
  --date-to "$DATE_TO" \
  --force

log "[3] aggregate 1s → 1m"
"$PY" preprocess/download/step3_databento_aggregate_1s_to_1m.py \
  --input-dir "$OPT_1S" \
  --output-dir "$OPT_1M" \
  --symbol QQQ \
  --date-from "$DATE_FROM" \
  --date-to "$DATE_TO"

# stage into EXP/options_1m/QQQ flat names expected by option_cac
mkdir -p "$OPT_1M/QQQ"
# step3 may write OPT_1M/QQQ already; ensure files present
ls "$OPT_1M/QQQ" | head || true

log "[4] day_iv (option_cac from 1m quotes — not stream greek inject)"
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
    option_root="$OPT_1M",
    data_root="$STOCK_RESAMP",
    iv_option_root="$EXP/quote_options_day_iv",
)
calc.run(max_concurrent_stocks=4)
PY

log "[5] iv_day2month"
"$PY" - <<PY
import glob
from preprocess.ask_bid.iv_day2month import process_single_symbol
inp = "$EXP/quote_options_day_iv"
out = "$EXP/quote_options_monthly_iv"
files = sorted(glob.glob(f"{inp}/QQQ/**/*.parquet", recursive=True))
print("day_iv files", len(files))
print(process_single_symbol(("QQQ", files, out)))
PY

log "[6] options_locked_feature"
"$PY" - <<PY
from pathlib import Path
from preprocess.ask_bid.options_locked_feature import process_single_file
raw = Path("$EXP/quote_options_monthly_iv/QQQ/standard/2026-07.parquet")
out = Path("$EXP/quote_options_bucketed_v7")
print(process_single_file((raw, out, "QQQ")) or "bucketed ok")
print("bucketed", list(out.rglob("2026-07.parquet")))
PY

log "[7] feature_merge → quote_features_raw"
"$PY" - <<PY
import json
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm

exp = Path("$EXP")
fm.OPTION_MONTHLY_DIR = exp / "quote_options_monthly_iv"
fm.AGG_OPTION_MONTHLY_DIR = exp / "quote_options_bucketed_v7"
fm.OUTPUT_FEATURES_DIR = exp / "quote_features_raw"

cfg_path = Path("$CFG_MERGE")
with open(cfg_path) as f:
    config = json.load(f)
print(fm.process_stock_month("QQQ", "2026-07", config))
for res in ("1min", "5min"):
    p = fm.OUTPUT_FEATURES_DIR / f"QQQ/regular/09:30-16:00/{res}/2026-07.parquet"
    import pandas as pd
    print(res, "exists", p.exists(), "rows", pd.read_parquet(p).shape[0] if p.exists() else 0)
PY

log "[8] frozen_norm (deploy 同款) → quote_features_test"
"$PY" - <<PY
from pathlib import Path
import json
import pandas as pd
from qqq_btc.common.frozen_norm import apply_frozen_norm_df

exp = Path("$EXP")
raw_p = exp / "quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
out_dir = exp / "quote_features_test/QQQ/regular/09:30-16:00/1min"
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(raw_p)
cfg = json.loads(Path("$CFG_V4").read_text())
# feature names from v4 config if present
names = None
if isinstance(cfg, dict):
    feats = cfg.get("features") or cfg.get("slow_features") or []
    if isinstance(feats, list) and feats and isinstance(feats[0], dict):
        names = [f.get("name") for f in feats if f.get("name")]
    elif isinstance(feats, list):
        names = [str(x) for x in feats]
normed = apply_frozen_norm_df(df, "$FROZEN_NORM", feature_names=names)
out_p = out_dir / "2026-07.parquet"
normed.to_parquet(out_p, index=False)
print("wrote", out_p, "rows", len(normed), "cols", len(normed.columns))
# also copy 5min raw→frozen if exists
raw5 = exp / "quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-07.parquet"
if raw5.exists():
    out5 = exp / "quote_features_test/QQQ/regular/09:30-16:00/5min"
    out5.mkdir(parents=True, exist_ok=True)
    apply_frozen_norm_df(pd.read_parquet(raw5), "$FROZEN_NORM", feature_names=names).to_parquet(
        out5 / "2026-07.parquet", index=False
    )
    print("wrote 5min", out5 / "2026-07.parquet")
PY

cat > "$REPORT_DIR/manifest.json" <<EOF
{
  "mode": "honest_openwin_offline_rebuild",
  "days": "$(IFS=,; echo "${DAY_ARR[*]}")",
  "lock_map": "$MAP_OUT",
  "anchor": "$ANCHOR",
  "raw_1s": "$OPT_1S",
  "exp": "$EXP",
  "offline_1min": "$EXP/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet",
  "frozen_norm": "$FROZEN_NORM",
  "note": "step1 open-window delta lock; contracts differ from API ladder; stream must use same OPT_1S"
}
EOF

log "=== DONE ==="
log "map=$MAP_OUT"
log "offline=$EXP/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
log "raw_1s=$OPT_1S"
log "next: OPT_ROOT=$OPT_1S OFFLINE_CLEAN=<offline> DAYS=... bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh"
