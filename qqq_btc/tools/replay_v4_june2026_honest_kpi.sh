#!/usr/bin/env bash
# V4 × 2026-06「诚实 KPI」离线 replay（不对拍 FT56）
#
# 配方与 July W1 诚实 KPI 一致：
#   1) 诚实特征：june_v4_massive_redownload/eval_feat_june_only
#   2) 因果 put_gate：raw 1min vix_level，timestamp+1m 后 merge_asof backward
#   3) LIVE 门控：LIVE_REPLAY + edge_q10_floor=-0.2
#   4) rails：qqq.config EXIT_RAILS + session_entry_end_bar
#
# 用法:
#   bash qqq_btc/tools/replay_v4_june2026_honest_kpi.sh
#   SKIP_INFER=1 bash ...   # 复用已有 infer parquet
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
SEED="${SEED:-42}"
export QQQ_BTC_SEED="$SEED"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="qqq_btc/CONFIG/slow_feature_qqq_v4.json"
SYM="qqq_btc/CONFIG/symbol_map.json"
CKPT_V4="${CKPT_V4:-checkpoint/checkpoints_qqq_v4/best.pth}"

HONEST_ROOT="${HONEST_ROOT:-$HOME/train_data/june_v4_massive_redownload}"
FEAT_JUN="${FEAT_JUN:-$HONEST_ROOT/eval_feat_june_only}"
OPT1M_JUN="${OPT1M_JUN:-$HONEST_ROOT/options_1m_june}"
RAW1_JUN="${RAW1_JUN:-$HONEST_ROOT/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-06.parquet}"

EVAL_V4="${EVAL_V4:-qqq_btc/results/v4_june2026_honest_infer}"
RESULTS="${RESULTS:-qqq_btc/results/v4_june2026_honest_kpi}"
EDGE_Q10_FLOOR="${EDGE_Q10_FLOOR:--0.2}"

echo "=== [0] check inputs ==="
[[ -f "$CKPT_V4" ]] || { echo "missing V4: $CKPT_V4"; exit 1; }
[[ -f "$FEAT_JUN/QQQ/regular/09:30-16:00/1min/2026-06.parquet" ]] || { echo "missing june feat"; exit 1; }
[[ -d "$OPT1M_JUN/QQQ" ]] || { echo "missing june option 1m: $OPT1M_JUN/QQQ"; exit 1; }
[[ -f "$RAW1_JUN" ]] || { echo "missing raw1 for causal put_gate: $RAW1_JUN"; exit 1; }

mkdir -p "$EVAL_V4" "$RESULTS"

if [[ "${SKIP_INFER:-0}" != "1" ]]; then
  echo "=== [1] June honest fixed5m infer (V4) ==="
  "$PY" qqq_btc/tools/eval_test_set.py \
    --checkpoint "$CKPT_V4" --config "$CONFIG" --symbol-map "$SYM" \
    --feature-root "$FEAT_JUN" --option-1m-root "$OPT1M_JUN" \
    --call-bucket 2 --put-bucket 0 \
    --output-dir "$EVAL_V4" --seed "$SEED" --device "${DEVICE:-cuda}" --live-replay
else
  echo "=== [1] SKIP_INFER=1 reuse $EVAL_V4/test_infer.parquet ==="
  [[ -f "$EVAL_V4/test_infer.parquet" ]] || { echo "missing infer"; exit 1; }
fi

echo "=== [2] honest KPI replay: 1min causal put_gate + LIVE + q10=${EDGE_Q10_FLOOR} ==="
export EVAL_V4 RESULTS CKPT_V4 RAW1_JUN EDGE_Q10_FLOOR SEED FEAT_JUN OPT1M_JUN
"$PY" - <<'PY'
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd

from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

OUT = Path(os.environ["RESULTS"])
OUT.mkdir(parents=True, exist_ok=True)
raw1_path = Path(os.environ["RAW1_JUN"])
q10 = float(os.environ["EDGE_Q10_FLOOR"])
seed = int(os.environ["SEED"])
infer_path = Path(os.environ["EVAL_V4"]) / "test_infer.parquet"

inf = pd.read_parquet(infer_path).copy()
inf["timestamp"] = pd.to_datetime(inf["timestamp"], utc=True)
inf = inf.drop(columns=[c for c in inf.columns if c == "put_gate"], errors="ignore")

raw1 = pd.read_parquet(raw1_path, columns=["timestamp", "vix_level"])
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
trades.to_parquet(OUT / "v4_june2026_honest_kpi_trades.parquet", index=False)

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

out = {
    "seed": seed,
    "period": "2026-06",
    "model": "v4",
    "recipe": "honest_features + 1min_causal(+1m) put_gate + LIVE_REPLAY + q10=-0.2",
    "closer_to_live": True,
    "paths": {
        "ckpt": os.environ["CKPT_V4"],
        "honest_feat": os.environ["FEAT_JUN"],
        "option_1m": os.environ["OPT1M_JUN"],
        "raw1_put_gate": str(raw1_path),
        "infer": str(infer_path),
        "config": "qqq_btc/CONFIG/slow_feature_qqq_v4.json",
        "strategy": "qqq_btc.qqq.config LIVE_REPLAY / EXIT_RAILS",
    },
    "gates": {
        "edge_q10_floor": q10,
        "session_entry_end_bar": cfg.session_entry_end_bar,
        "max_hold_bars": qcfg.EXIT_RAILS.max_hold_bars,
        "apply_put_entry_quantile": getattr(cfg, "apply_put_entry_quantile", None),
        "put_gate": "raw1 vix_level causal +1m",
    },
    "data_note": (
        "june_v4_massive_redownload：20 个交易日（缺 2026-06-18）；"
        "与 bak/june_fresh 的 21 日不完全对齐。"
    ),
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
print(f"V4 June2026 honest KPI: acct25={r['acct25']*100:+.2f}% trades={r['trades']} hit={r['hit']*100:.1f}%")
PY

echo "done"
echo "  summary=$RESULTS/summary.json"
echo "  trades=$RESULTS/v4_june2026_honest_kpi_trades.parquet"
