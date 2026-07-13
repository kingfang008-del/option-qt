#!/usr/bin/env bash
# V4 在 bak 4–6 月上 finetune（对照 FT56=5–6），再 replay 2026-07 W1
# train=Apr+May+Jun, val=Jun, init=V4, seed=42
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
SEED="${SEED:-42}"
export QQQ_BTC_SEED="$SEED"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="$HOME/train_data/lmdb"
CONFIG="qqq_btc/CONFIG/slow_feature_qqq_v4.json"
SYM="qqq_btc/CONFIG/symbol_map.json"
CKPT_V4="${CKPT_V4:-checkpoint/checkpoints_qqq_v4/best.pth}"
CKPT_OUT="${CKPT_OUT:-checkpoint/checkpoints_qqq_ft456_julw1}"
EVAL_FT="${EVAL_FT:-qqq_btc/results/ft456_julw1_fixed5m_infer}"
RESULTS="${RESULTS:-qqq_btc/results/ft456_julw1_compare}"

FEAT_BAK="${FEAT_BAK:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00}"
# July 评测用 honest openwin（与 FT56 +38.3% 同特征世界）
FEAT_JUL="${FEAT_JUL:-$HOME/train_data/july_w1_v4_honest_openwin/quote_features_test}"
OPT1M_JUL="${OPT1M_JUL:-$HOME/train_data/july_w1_v4_honest_openwin/options_1m}"

TRAIN_FEAT="/tmp/quote_features_ft456_train"
VAL_FEAT="/tmp/quote_features_ft456_val"
LMDB_TRAIN="${LMDB_TRAIN:-train_qqq_ft456.lmdb}"
LMDB_VAL="${LMDB_VAL:-val_qqq_ft456.lmdb}"

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
for ym in 2026-04 2026-05 2026-06; do
  [[ -f "$FEAT_BAK/1min/${ym}.parquet" ]] || { echo "missing bak $ym"; exit 1; }
done
[[ -f "$FEAT_JUL/QQQ/regular/09:30-16:00/1min/2026-07.parquet" ]] || { echo "missing july feat"; exit 1; }
[[ -d "$OPT1M_JUL/QQQ" ]] || { echo "missing july option 1m"; exit 1; }
[[ -f "$CKPT_V4" ]] || { echo "missing V4"; exit 1; }

echo "=== [1] feature dirs: train=4+5+6, val=6 ==="
setup_months "$TRAIN_FEAT" 2026-04 2026-05 2026-06
setup_months "$VAL_FEAT" 2026-06

echo "=== [2] build LMDB ==="
"$PY" qqq_btc/tools/build_lmdb.py \
  --feature-root "$TRAIN_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
  --output "$DATA_ROOT/$LMDB_TRAIN" --symbols QQQ --window-step 1
"$PY" qqq_btc/tools/build_lmdb.py \
  --feature-root "$VAL_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
  --output "$DATA_ROOT/$LMDB_VAL" --symbols QQQ --window-step 1

echo "=== [3] finetune V4 on Apr-May-Jun (seed=$SEED) ==="
rm -rf "$CKPT_OUT"
mkdir -p "$CKPT_OUT"
"$PY" -m qqq_btc.model.train \
  --mode finetune \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-lmdb "$LMDB_TRAIN" \
  --val-lmdbs "$LMDB_VAL" \
  --checkpoint-dir "$CKPT_OUT" \
  --init-checkpoint "$CKPT_V4" \
  --epochs 20 \
  --batch-size 512 \
  --num-workers 4 \
  --seed "$SEED" \
  --device "${DEVICE:-cuda}" 2>&1 | tee "$CKPT_OUT/train.log"

echo "=== [4] July W1 fixed5m infer ==="
mkdir -p "$EVAL_FT" "$RESULTS"
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_OUT/best.pth" --config "$CONFIG" \
  --feature-root "$FEAT_JUL" --option-1m-root "$OPT1M_JUL" \
  --call-bucket 2 --put-bucket 0 \
  --output-dir "$EVAL_FT" --seed "$SEED" --device "${DEVICE:-cuda}" --live-replay

echo "=== [5] same recipe as FT56 +38.3%: 1min causal gate + q10=-0.2 ==="
"$PY" - <<'PY'
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from qqq_btc.qqq import config as qcfg
from qqq_btc.common.replay_harness import run_strict_replay

OUT = Path("qqq_btc/results/ft456_julw1_compare")
OUT.mkdir(parents=True, exist_ok=True)

def load_infer(path: str) -> pd.DataFrame:
    inf = pd.read_parquet(path).copy()
    inf["timestamp"] = pd.to_datetime(inf["timestamp"], utc=True)
    return inf.drop(columns=[c for c in inf.columns if c == "put_gate"], errors="ignore")

raw1 = pd.read_parquet(
    Path.home() / "train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet",
    columns=["timestamp", "vix_level"],
)
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

cfg = replace(qcfg.LIVE_REPLAY, edge_q10_floor=-0.2)
models = {
    "ft456": "qqq_btc/results/ft456_julw1_fixed5m_infer/test_infer.parquet",
    "ft56": "qqq_btc/results/ft56_julw1_honest_infer_fixed5m/test_infer.parquet",
    "v4": "qqq_btc/results/v4_jul_w1_fixed5m_infer/test_infer.parquet",
}

rows = []
detail = {}
for name, path in models.items():
    p = Path(path)
    if not p.exists():
        rows.append({"model": name, "missing": True})
        continue
    df = attach_1m_causal(load_infer(str(p)))
    res = run_strict_replay(
        df, qcfg.FILL_MODEL, cfg, qcfg.EXIT_RAILS,
        edge_col="net_edge", edge_q10_col="net_edge_q10",
        call_edge_col="call_net_edge", put_edge_col="put_net_edge",
        put_gate_col="put_gate",
    )
    s = res.summary(position_frac=0.25)
    trades = res.trades_frame()
    trades["entry_ny"] = pd.to_datetime(trades["entry_ts"]).dt.tz_convert("America/New_York")
    trades["date"] = trades["entry_ny"].dt.strftime("%Y-%m-%d")
    daily = []
    eq = 1.0
    for date, g in trades.groupby("date", sort=True):
        day_eq = 1.0
        for r in g["net_return"]:
            day_eq *= 1 + 0.25 * float(r)
            eq *= 1 + 0.25 * float(r)
        daily.append({
            "date": date,
            "n": int(len(g)),
            "day_acct25": float(day_eq - 1),
            "cum_acct25": float(eq - 1),
            "sum_net": float(g["net_return"].sum()),
            "hit": float((g["net_return"] > 0).mean()),
            "legs": g["leg"].value_counts().to_dict() if "leg" in g.columns else {},
        })
    row = {
        "model": name,
        "acct25": float(s["total_net_return"]),
        "trades": int(s["trades"]),
        "hit": float(s.get("hit_rate") or 0),
        "pf": float(s.get("profit_factor") or 0) if s.get("profit_factor") is not None else None,
        "legs": s.get("trades_by_leg"),
        "daily": daily,
    }
    rows.append(row)
    if name == "ft456":
        trades.to_parquet(OUT / "ft456_replay_trades.parquet", index=False)
        detail = row

out = {
    "recipe": "fixed5m edge + 1min_causal(+1m) put_gate + LIVE + q10=-0.2",
    "ft456": {"train": "2026-04+05+06 bak", "val": "2026-06 bak", "init": "checkpoints_qqq_v4", "ckpt": "checkpoint/checkpoints_qqq_ft456_julw1/best.pth"},
    "ft56": {"train": "2026-05+06 bak", "val": "2026-06 bak"},
    "results": rows,
}
(OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print("\n=== compare @25% ===")
for r in rows:
    if r.get("missing"):
        print(r["model"], "MISSING")
        continue
    print(f"{r['model']:<6} acct={r['acct25']*100:+7.2f}%  n={r['trades']:2d}  hit={r['hit']:.1%}  pf={r['pf']:.2f}  legs={r['legs']}")
PY

echo "done ckpt=$CKPT_OUT/best.pth"
echo "summary=$RESULTS/summary.json"
