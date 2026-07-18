#!/usr/bin/env bash
# V4 × 2026-06：从 dte1_options_old_lock 重建特征并 replay
#
# 锁约：locked_targets_map_old_style_trading_1dte.parquet
# 1s： /mnt/s990/data/raw_1s/dte1_options_old_lock
#
# 用法:
#   bash qqq_btc/tools/replay_v4_june2026_old_lock.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
SEED="${SEED:-42}"
export QQQ_BTC_SEED="$SEED"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

EXP="${EXP:-$HOME/train_data/june_v4_old_lock}"
RAW1S="${RAW1S:-/mnt/s990/data/raw_1s/dte1_options_old_lock}"
OPT1M="${OPT1M:-$EXP/options_1m_june}"
LOCK_MAP="${LOCK_MAP:-$HOME/train_data/locked_targets_map_old_style_trading_1dte.parquet}"
CKPT="${CKPT:-checkpoint/checkpoints_qqq_v4/best.pth}"
CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
CFG_MERGE="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
ANCHOR="$REPO/qqq_btc/CONFIG/anchor_qqq_1dte.json"
OUT="${OUT:-qqq_btc/results/v4_june2026_old_lock}"
LOG="$EXP/pipeline.log"
EDGE_Q10_FLOOR="${EDGE_Q10_FLOOR:--0.2}"

export FEATURE_CONFIG="$CFG_MERGE"
export LOCKED_TARGETS_MAP="$LOCK_MAP"

mkdir -p "$EXP" "$OPT1M/QQQ" "$OUT"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

[[ -f "$CKPT" ]] || { echo "missing ckpt $CKPT"; exit 1; }
[[ -f "$LOCK_MAP" ]] || { echo "missing lock map $LOCK_MAP"; exit 1; }
[[ -d "$RAW1S/QQQ" ]] || { echo "missing raw1s $RAW1S/QQQ"; exit 1; }

log "=== V4 June old_lock pipeline start ==="
log "raw1s=$RAW1S lock=$LOCK_MAP out=$OUT"

log "[0] 1s → 1m (June)"
"$PY" preprocess/download/step3_databento_aggregate_1s_to_1m.py \
  --input-dir "$RAW1S" \
  --output-dir "$OPT1M" \
  --symbol QQQ \
  --date-from 2026-06-01 \
  --date-to 2026-06-30 \
  --force
ls "$OPT1M/QQQ"/QQQ_2026-06-*.parquet | wc -l

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
raw = Path("$EXP/quote_options_monthly_iv/QQQ/standard/2026-06.parquet")
out = Path("$EXP/quote_options_bucketed_v7")
print(process_single_file((raw, out, "QQQ")) or "bucketed ok")
print("bucketed", list(out.rglob("2026-06.parquet")))
PY

log "[4] feature_merge QQQ 2026-06"
"$PY" - <<'PY'
import json
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm

exp = Path.home() / "train_data/june_v4_old_lock"
fm.OPTION_MONTHLY_DIR = exp / "quote_options_monthly_iv"
fm.AGG_OPTION_MONTHLY_DIR = exp / "quote_options_bucketed_v7"
fm.OUTPUT_FEATURES_DIR = exp / "quote_features_raw"

cfg_path = Path("/home/kingfang007/文档/GitHub/option-qt/qqq_btc/CONFIG/slow_feature_qqq_v2.json")
with open(cfg_path) as f:
    config = json.load(f)
print(fm.process_stock_month("QQQ", "2026-06", config))
for res in ("1min", "5min"):
    p = fm.OUTPUT_FEATURES_DIR / f"QQQ/regular/09:30-16:00/{res}/2026-06.parquet"
    import pandas as pd
    print(res, "exists", p.exists(), "rows", pd.read_parquet(p).shape[0] if p.exists() else 0)
PY

log "[5] assemble + rolling_norm + label (May bak warmstart)"
TEST1="$EXP/quote_features_test/QQQ/regular/09:30-16:00/1min"
TEST5="$EXP/quote_features_test/QQQ/regular/09:30-16:00/5min"
mkdir -p "$TEST1" "$TEST5"
BAK1="$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00"
if [[ -f "$BAK1/1min/2026-05.parquet" ]]; then
  cp "$BAK1/1min/2026-05.parquet" "$TEST1/"
  cp "$BAK1/5min/2026-05.parquet" "$TEST5/" 2>/dev/null || true
fi
cp "$EXP/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-06.parquet" "$TEST1/"
cp "$EXP/quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-06.parquet" "$TEST5/" 2>/dev/null || true

"$PY" - <<'PY'
import os
from pathlib import Path
import preprocess.ask_bid.apply_rolling_norm_standalone as norm
exp = Path.home() / "train_data/june_v4_old_lock/quote_features_test/QQQ/regular/09:30-16:00/1min"
cfg = Path(os.environ["FEATURE_CONFIG"])
norm_cols = norm.load_target_features(cfg)
print("norm:", norm.process_single_directory((exp, norm_cols)))
PY

"$PY" qqq_btc/tools/label_pipeline.py \
  --input "$TEST1" \
  --output "$TEST1" \
  --symbol QQQ \
  --anchor-config "$ANCHOR" \
  --report "$EXP/label_report_june_old_lock.json"

log "[6] V4 infer June-only + live-replay"
TMP="$EXP/eval_feat_june_only"
rm -rf "$TMP"
mkdir -p "$TMP/QQQ/regular/09:30-16:00/1min" "$TMP/QQQ/regular/09:30-16:00/5min"
cp "$TEST1/2026-06.parquet" "$TMP/QQQ/regular/09:30-16:00/1min/"
cp "$TEST5/2026-06.parquet" "$TMP/QQQ/regular/09:30-16:00/5min/" 2>/dev/null || true

EVAL_DIR="$OUT/infer"
mkdir -p "$EVAL_DIR"
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --feature-root "$TMP" \
  --option-1m-root "$OPT1M" \
  --call-bucket 2 --put-bucket 0 \
  --output-dir "$EVAL_DIR" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda}" \
  --live-replay

log "[7] honest KPI: 1min causal put_gate + LIVE q10=${EDGE_Q10_FLOOR}"
export EVAL_DIR OUT CKPT EDGE_Q10_FLOOR SEED EXP OPT1M LOCK_MAP RAW1S
"$PY" - <<'PY'
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd

from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

OUT = Path(os.environ["OUT"])
EVAL = Path(os.environ["EVAL_DIR"])
EXP = Path(os.environ["EXP"])
q10 = float(os.environ["EDGE_Q10_FLOOR"])
seed = int(os.environ["SEED"])

inf = pd.read_parquet(EVAL / "test_infer.parquet").copy()
inf["timestamp"] = pd.to_datetime(inf["timestamp"], utc=True)
inf = inf.drop(columns=[c for c in inf.columns if c == "put_gate"], errors="ignore")

raw1 = pd.read_parquet(
    EXP / "quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-06.parquet",
    columns=["timestamp", "vix_level"],
)
raw1["timestamp"] = pd.to_datetime(raw1["timestamp"], utc=True)
raw1 = raw1.sort_values("timestamp").drop_duplicates("timestamp")
s = raw1.copy()
s["timestamp"] = s["timestamp"] + pd.Timedelta(minutes=1)
m = pd.merge_asof(
    inf[["timestamp"]].reset_index(drop=True),
    s.rename(columns={"vix_level": "put_gate"}),
    on="timestamp",
    direction="backward",
)
inf["put_gate"] = m["put_gate"].to_numpy()

cfg = replace(qcfg.LIVE_REPLAY, edge_q10_floor=q10)
res = run_strict_replay(
    inf,
    qcfg.FILL_MODEL,
    cfg,
    qcfg.EXIT_RAILS,
    edge_col="net_edge",
    edge_q10_col="net_edge_q10",
    call_edge_col="call_net_edge",
    put_edge_col="put_net_edge",
    put_gate_col="put_gate",
)
sry = res.summary(position_frac=0.25)
trades = res.trades_frame()
trades["entry_ny"] = pd.to_datetime(trades["entry_ts"]).dt.tz_convert("America/New_York")
trades["date"] = trades["entry_ny"].dt.strftime("%Y-%m-%d")
trades.to_parquet(OUT / "v4_june2026_old_lock_trades.parquet", index=False)

daily = []
eq = 1.0
for date, g in trades.groupby("date", sort=True):
    day_eq = 1.0
    for r in g["net_return"]:
        day_eq *= 1 + 0.25 * float(r)
        eq *= 1 + 0.25 * float(r)
    daily.append(
        {
            "date": date,
            "n": int(len(g)),
            "day_acct25": float(day_eq - 1),
            "cum_acct25": float(eq - 1),
            "sum_net": float(g["net_return"].sum()),
            "hit": float((g["net_return"] > 0).mean()),
            "legs": g["leg"].value_counts().to_dict() if "leg" in g.columns else {},
        }
    )

# also keep eval_test_set built-in summary for reference
builtin = {}
bsp = EVAL / "replay_summary.json"
if bsp.exists():
    builtin = json.loads(bsp.read_text())

out = {
    "seed": seed,
    "period": "2026-06",
    "model": "v4",
    "data": "dte1_options_old_lock + old_style_trading_1dte map",
    "recipe": "old_lock_features + 1min_causal(+1m) put_gate + LIVE_REPLAY + q10=-0.2",
    "paths": {
        "ckpt": os.environ["CKPT"],
        "raw_1s": os.environ["RAW1S"],
        "option_1m": os.environ["OPT1M"],
        "lock_map": os.environ["LOCK_MAP"],
        "feat_root": str(EXP / "eval_feat_june_only"),
        "infer": str(EVAL / "test_infer.parquet"),
    },
    "gates": {
        "edge_q10_floor": q10,
        "session_entry_end_bar": cfg.session_entry_end_bar,
        "max_hold_bars": qcfg.EXIT_RAILS.max_hold_bars,
        "put_gate": "raw1 vix_level causal +1m",
    },
    "eval_builtin_live_feat_vix": {
        "acct25": builtin.get("total_net_return"),
        "trades": builtin.get("trades"),
        "hit": builtin.get("hit_rate"),
        "legs": builtin.get("trades_by_leg"),
    },
    "result": {
        "acct25": float(sry["total_net_return"]),
        "trades": int(sry["trades"]),
        "hit": float(sry.get("hit_rate") or 0),
        "pf": float(sry.get("profit_factor") or 0) if sry.get("profit_factor") is not None else None,
        "legs": sry.get("trades_by_leg"),
        "daily": daily,
    },
}
(OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
r = out["result"]
print()
print(
    f"V4 June old_lock honest KPI: acct25={r['acct25']*100:+.2f}% "
    f"trades={r['trades']} hit={r['hit']*100:.1f}%"
)
if out["eval_builtin_live_feat_vix"].get("acct25") is not None:
    b = out["eval_builtin_live_feat_vix"]
    print(
        f"V4 June old_lock builtin LIVE(feat vix): acct25={b['acct25']*100:+.2f}% "
        f"trades={b['trades']}"
    )
PY

log "=== DONE summary=$OUT/summary.json ==="
