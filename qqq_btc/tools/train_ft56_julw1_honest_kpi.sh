#!/usr/bin/env bash
# V4 → FT56（bak 5–6 月微调）→ July W1「诚实 KPI」离线验收
#
# 诚实 KPI 配方（更贴近实盘，非 databento 开卷对照）：
#   1) 诚实特征：july_w1_v4_honest_openwin/quote_features_test
#   2) 因果 put_gate：raw 1min vix_level，timestamp+1m 后 merge_asof backward
#   3) LIVE 门控：LIVE_REPLAY + edge_q10_floor=-0.2
#   4) rails：沿用 qqq.config EXIT_RAILS（当前 max_hold=55）+ session_entry_end_bar=240
#
# 对照陷阱：
#   eval_test_set 默认走 REPLAY + 特征列 vix_level，在 databento 根上会得到另一套数字
#   （例如曾见 V4 +60% / FT56 +27%），不能与本 KPI / hardcap +49% 横比。
#
# 用法:
#   bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh
#   SKIP_TRAIN=1 bash ...   # 复用已有 FT56 ckpt，只跑诚实 KPI replay
#   CKPT_FT=checkpoint/checkpoints_qqq_ft56_julw1/best.pth bash ...
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
SEED="${SEED:-42}"
export QQQ_BTC_SEED="$SEED"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="$HOME/train_data/lmdb"
CONFIG="qqq_btc/CONFIG/slow_feature_qqq_v4.json"
SYM="qqq_btc/CONFIG/symbol_map.json"
CKPT_V4="${CKPT_V4:-checkpoint/checkpoints_qqq_v4/best.pth}"
CKPT_OUT="${CKPT_OUT:-checkpoint/checkpoints_qqq_ft56_julw1}"
CKPT_FT="${CKPT_FT:-$CKPT_OUT/best.pth}"

# --- 微调特征（bak，与历史 FT56 一致）---
FEAT_BAK="${FEAT_BAK:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00}"

# --- 诚实 July W1（KPI）---
HONEST_ROOT="${HONEST_ROOT:-$HOME/train_data/july_w1_v4_honest_openwin}"
FEAT_JUL="${FEAT_JUL:-$HONEST_ROOT/quote_features_test}"
OPT1M_JUL="${OPT1M_JUL:-$HONEST_ROOT/options_1m}"
RAW1_JUL="${RAW1_JUL:-$HONEST_ROOT/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet}"

EVAL_FT="${EVAL_FT:-qqq_btc/results/ft56_julw1_honest_infer_fixed5m}"
EVAL_V4="${EVAL_V4:-qqq_btc/results/v4_jul_w1_fixed5m_infer}"
RESULTS="${RESULTS:-qqq_btc/results/ft56_julw1_honest_kpi_compare}"

TRAIN_FEAT="/tmp/quote_features_ft56_honest_kpi_train"
VAL_FEAT="/tmp/quote_features_ft56_honest_kpi_val"
EDGE_Q10_FLOOR="${EDGE_Q10_FLOOR:--0.2}"

setup_months() {
  local dest="$1"; shift
  rm -rf "$dest"
  mkdir -p "$dest/QQQ/regular/09:30-16:00/1min" "$dest/QQQ/regular/09:30-16:00/5min"
  for ym in "$@"; do
    ln -sf "$FEAT_BAK/1min/${ym}.parquet" "$dest/QQQ/regular/09:30-16:00/1min/${ym}.parquet"
    ln -sf "$FEAT_BAK/5min/${ym}.parquet" "$dest/QQQ/regular/09:30-16:00/5min/${ym}.parquet"
  done
}

echo "=== [0] check inputs ==="
for ym in 2026-05 2026-06; do
  [[ -f "$FEAT_BAK/1min/${ym}.parquet" ]] || { echo "missing bak $ym under $FEAT_BAK"; exit 1; }
done
[[ -f "$FEAT_JUL/QQQ/regular/09:30-16:00/1min/2026-07.parquet" ]] || { echo "missing honest july feat"; exit 1; }
[[ -d "$OPT1M_JUL/QQQ" ]] || { echo "missing honest july option 1m: $OPT1M_JUL/QQQ"; exit 1; }
[[ -f "$RAW1_JUL" ]] || { echo "missing raw1 for causal put_gate: $RAW1_JUL"; exit 1; }
[[ -f "$CKPT_V4" ]] || { echo "missing V4: $CKPT_V4"; exit 1; }

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== [1] feature dirs: train=5+6, val=6 ==="
  setup_months "$TRAIN_FEAT" 2026-05 2026-06
  setup_months "$VAL_FEAT" 2026-06

  echo "=== [2] build LMDB ==="
  "$PY" qqq_btc/tools/build_lmdb.py \
    --feature-root "$TRAIN_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
    --output "$DATA_ROOT/train_qqq_ft56_honest_kpi.lmdb" --symbols QQQ --window-step 1
  "$PY" qqq_btc/tools/build_lmdb.py \
    --feature-root "$VAL_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
    --output "$DATA_ROOT/val_qqq_ft56_honest_kpi.lmdb" --symbols QQQ --window-step 1

  echo "=== [3] finetune V4 on May-Jun (seed=$SEED) ==="
  rm -rf "$CKPT_OUT"
  mkdir -p "$CKPT_OUT"
  "$PY" -m qqq_btc.model.train \
    --mode finetune \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --train-lmdb train_qqq_ft56_honest_kpi.lmdb \
    --val-lmdbs val_qqq_ft56_honest_kpi.lmdb \
    --checkpoint-dir "$CKPT_OUT" \
    --init-checkpoint "$CKPT_V4" \
    --epochs 20 \
    --batch-size 512 \
    --num-workers 4 \
    --seed "$SEED" \
    --device "${DEVICE:-cuda}" 2>&1 | tee "$CKPT_OUT/train.log"
  CKPT_FT="$CKPT_OUT/best.pth"
else
  echo "=== [1-3] SKIP_TRAIN=1 reuse $CKPT_FT ==="
  [[ -f "$CKPT_FT" ]] || { echo "missing CKPT_FT=$CKPT_FT"; exit 1; }
fi

echo "=== [4] July W1 honest fixed5m infer (V4 + FT56) ==="
mkdir -p "$EVAL_FT" "$EVAL_V4" "$RESULTS"
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_V4" --config "$CONFIG" \
  --feature-root "$FEAT_JUL" --option-1m-root "$OPT1M_JUL" \
  --call-bucket 2 --put-bucket 0 \
  --output-dir "$EVAL_V4" --seed "$SEED" --device "${DEVICE:-cuda}" --live-replay

"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_FT" --config "$CONFIG" \
  --feature-root "$FEAT_JUL" --option-1m-root "$OPT1M_JUL" \
  --call-bucket 2 --put-bucket 0 \
  --output-dir "$EVAL_FT" --seed "$SEED" --device "${DEVICE:-cuda}" --live-replay

echo "=== [5] honest KPI replay: 1min causal put_gate + LIVE + q10=${EDGE_Q10_FLOOR} ==="
export EVAL_V4 EVAL_FT RESULTS CKPT_V4 CKPT_FT RAW1_JUL EDGE_Q10_FLOOR SEED
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
raw1_path = Path(os.environ["RAW1_JUL"])
q10 = float(os.environ["EDGE_Q10_FLOOR"])
seed = int(os.environ["SEED"])

def load_infer(path: str) -> pd.DataFrame:
    inf = pd.read_parquet(path).copy()
    inf["timestamp"] = pd.to_datetime(inf["timestamp"], utc=True)
    return inf.drop(columns=[c for c in inf.columns if c == "put_gate"], errors="ignore")

raw1 = pd.read_parquet(raw1_path, columns=["timestamp", "vix_level"])
raw1["timestamp"] = pd.to_datetime(raw1["timestamp"], utc=True)
raw1 = raw1.sort_values("timestamp").drop_duplicates("timestamp")

def attach_1m_causal(base: pd.DataFrame) -> pd.DataFrame:
    s = raw1.copy()
    s["timestamp"] = s["timestamp"] + pd.Timedelta(minutes=1)
    m = pd.merge_asof(
        base[["timestamp"]].reset_index(drop=True),
        s.rename(columns={"vix_level": "put_gate"}),
        on="timestamp",
        direction="backward",
    )
    out = base.copy()
    out["put_gate"] = m["put_gate"].to_numpy()
    return out

cfg = replace(qcfg.LIVE_REPLAY, edge_q10_floor=q10)
models = {
    "v4": f"{os.environ['EVAL_V4']}/test_infer.parquet",
    "ft56": f"{os.environ['EVAL_FT']}/test_infer.parquet",
}

rows = []
for name, path in models.items():
    p = Path(path)
    if not p.exists():
        rows.append({"model": name, "missing": True, "path": path})
        continue
    df = attach_1m_causal(load_infer(str(p)))
    res = run_strict_replay(
        df,
        qcfg.FILL_MODEL,
        cfg,
        qcfg.EXIT_RAILS,
        edge_col="net_edge",
        edge_q10_col="net_edge_q10",
        call_edge_col="call_net_edge",
        put_edge_col="put_net_edge",
        put_gate_col="put_gate",
    )
    s = res.summary(position_frac=0.25)
    trades = res.trades_frame()
    daily = []
    eq = 1.0
    if trades is not None and len(trades) and "entry_ts" in trades.columns:
        trades = trades.copy()
        trades["entry_ny"] = pd.to_datetime(trades["entry_ts"]).dt.tz_convert("America/New_York")
        trades["date"] = trades["entry_ny"].dt.strftime("%Y-%m-%d")
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
        if name == "ft56":
            trades.to_parquet(OUT / "ft56_honest_kpi_trades.parquet", index=False)
    rows.append(
        {
            "model": name,
            "acct25": float(s["total_net_return"]),
            "trades": int(s["trades"]),
            "hit": float(s.get("hit_rate") or 0),
            "pf": float(s.get("profit_factor") or 0) if s.get("profit_factor") is not None else None,
            "legs": s.get("trades_by_leg"),
            "daily": daily,
        }
    )

out = {
    "seed": seed,
    "recipe": "honest_features + 1min_causal(+1m) put_gate + LIVE_REPLAY + q10=-0.2",
    "closer_to_live": True,
    "finetune": {
        "train": "2026-05+06 bak",
        "val": "2026-06 bak",
        "init": os.environ["CKPT_V4"],
        "ckpt_ft56": os.environ["CKPT_FT"],
    },
    "paths": {
        "honest_feat": str(Path(os.environ["EVAL_FT"]).resolve()),
        "raw1_put_gate": str(raw1_path),
        "config": "qqq_btc/CONFIG/slow_feature_qqq_v4.json",
        "strategy": "qqq_btc.qqq.config LIVE_REPLAY / EXIT_RAILS",
    },
    "gates": {
        "edge_q10_floor": q10,
        "session_entry_start_bar": cfg.session_entry_start_bar,
        "session_entry_end_bar": cfg.session_entry_end_bar,
        "max_hold_bars": qcfg.EXIT_RAILS.max_hold_bars,
        "apply_put_entry_quantile": getattr(cfg, "apply_put_entry_quantile", None),
        "put_gate": "raw1 vix_level causal +1m",
        "note": "q10 floor 仅作用于 CALL（PUT 仍免 q10）；开仓默认 bar=15",
    },
    "results": rows,
    "note": (
        "本结果为离线 replay acct25，贴近实盘语义但仍非 Redis/OMS 流式。"
        "流式验收见 restart_ft56_july_w1_honest_live_parity.sh。"
    ),
}
(OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
print()
print(f"{'model':<8} {'acct25':>10} {'trades':>7} {'hit':>7}")
for r in rows:
    if r.get("missing"):
        print(f"{r['model']:<8} MISSING")
        continue
    print(f"{r['model']:<8} {r['acct25']*100:+9.2f}% {r['trades']:7d} {r['hit']*100:6.1f}%")
PY

echo "done"
echo "  ckpt_ft=$CKPT_FT"
echo "  summary=$RESULTS/summary.json"
echo "  docs=qqq_btc/docs/honest_live_kpi_finetune_replay.md"
