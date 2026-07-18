#!/usr/bin/env bash
# 假设验证：V4 原 val=1–3 → 4 月偏强；若改 val=2–4，5 月是否明显好转？
#
#   train : train_qqq_v4.lmdb（与 V4 同谱系历史窗）
#   val   : 2026-02 + 2026-03 + 2026-04（bak）
#   test  : 2026-05 old_lock 诚实 KPI（因果 put_gate + LIVE q10=-0.2）
#   init  : checkpoints_qqq_v4
#
# 用法:
#   bash qqq_btc/tools/train_ft_val234_test5.sh
#   SKIP_TRAIN=1 bash ...   # 只重跑 5 月对拍
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
CKPT_OUT="${CKPT_OUT:-checkpoint/checkpoints_qqq_ft_val234_test5}"
RESULTS="qqq_btc/results/ft_val234_test5_may"
EDGE_Q10_FLOOR="${EDGE_Q10_FLOOR:--0.2}"

FEAT_VAL="${FEAT_VAL:-$HOME/train_data/_bak_pre4c/quote_features_val_QQQ/regular/09:30-16:00}"
FEAT_TEST_BAK="${FEAT_TEST_BAK:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00}"
# May test：与近期 4–6 月对照同一 old_lock 链
FEAT_MAY="${FEAT_MAY:-$HOME/train_data/may_v4_old_lock/eval_feat_2026-05_only}"
# build script wrote eval_feat_${YM}_only
if [[ ! -d "$FEAT_MAY/QQQ" ]]; then
  FEAT_MAY="$HOME/train_data/may_v4_old_lock/quote_features_test"
fi
OPT1M_MAY="${OPT1M_MAY:-$HOME/train_data/may_v4_old_lock/options_1m_2026-05}"
RAW1_MAY="${RAW1_MAY:-$HOME/train_data/may_v4_old_lock/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-05.parquet}"
EVAL_V4="${EVAL_V4:-$RESULTS/infer_v4}"
EVAL_FT="${EVAL_FT:-$RESULTS/infer_ft}"

VAL_FEAT="/tmp/quote_features_val234"
TRAIN_LMDB="train_qqq_v4.lmdb"
VAL_LMDB_NAME="val_qqq_val234.lmdb"

setup_val_months() {
  rm -rf "$VAL_FEAT"
  mkdir -p "$VAL_FEAT/QQQ/regular/09:30-16:00/1min" "$VAL_FEAT/QQQ/regular/09:30-16:00/5min"
  for ym in 2026-02 2026-03; do
    ln -sf "$FEAT_VAL/1min/${ym}.parquet" "$VAL_FEAT/QQQ/regular/09:30-16:00/1min/${ym}.parquet"
    ln -sf "$FEAT_VAL/5min/${ym}.parquet" "$VAL_FEAT/QQQ/regular/09:30-16:00/5min/${ym}.parquet"
  done
  ln -sf "$FEAT_TEST_BAK/1min/2026-04.parquet" "$VAL_FEAT/QQQ/regular/09:30-16:00/1min/2026-04.parquet"
  ln -sf "$FEAT_TEST_BAK/5min/2026-04.parquet" "$VAL_FEAT/QQQ/regular/09:30-16:00/5min/2026-04.parquet"
}

echo "=== [0] check ==="
[[ -f "$DATA_ROOT/$TRAIN_LMDB/data.mdb" ]] || { echo "missing $DATA_ROOT/$TRAIN_LMDB"; exit 1; }
[[ -f "$CKPT_V4" ]] || { echo "missing V4"; exit 1; }
for ym in 2026-02 2026-03; do
  [[ -f "$FEAT_VAL/1min/${ym}.parquet" ]] || { echo "missing val $ym"; exit 1; }
done
[[ -f "$FEAT_TEST_BAK/1min/2026-04.parquet" ]] || { echo "missing bak Apr"; exit 1; }
[[ -f "$RAW1_MAY" ]] || { echo "missing May raw1 $RAW1_MAY"; exit 1; }
[[ -d "$OPT1M_MAY/QQQ" ]] || { echo "missing May opt1m $OPT1M_MAY"; exit 1; }

# May feat root: prefer month-only eval dir
MAY_1MIN=""
for cand in \
  "$HOME/train_data/may_v4_old_lock/eval_feat_2026-05_only/QQQ/regular/09:30-16:00/1min/2026-05.parquet" \
  "$HOME/train_data/may_v4_old_lock/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-05.parquet"
do
  if [[ -f "$cand" ]]; then MAY_1MIN="$cand"; break; fi
done
[[ -n "$MAY_1MIN" ]] || { echo "missing May feature parquet"; exit 1; }
FEAT_MAY_ROOT="$(cd "$(dirname "$MAY_1MIN")/../../../.." && pwd)"
# FEAT_MAY_ROOT should be .../eval_feat_... or quote_features_test parent of QQQ
# dirname 1min -> 09:30-16:00 -> regular -> QQQ -> root
FEAT_MAY_ROOT="$(python3 - <<PY
from pathlib import Path
p=Path("$MAY_1MIN").resolve()
# .../QQQ/regular/09:30-16:00/1min/2026-05.parquet -> root above QQQ
print(p.parents[4])
PY
)"

echo "May feat root=$FEAT_MAY_ROOT"
echo "May 1min=$MAY_1MIN"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== [1] val feature dir: 2026-02+03+04 ==="
  setup_val_months

  echo "=== [2] build val LMDB ==="
  "$PY" qqq_btc/tools/build_lmdb.py \
    --feature-root "$VAL_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
    --output "$DATA_ROOT/$VAL_LMDB_NAME" --symbols QQQ --window-step 1

  echo "=== [3] finetune V4 | train=$TRAIN_LMDB val=2+3+4 ==="
  rm -rf "$CKPT_OUT"
  mkdir -p "$CKPT_OUT"
  "$PY" -m qqq_btc.model.train \
    --mode finetune \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --train-lmdb "$TRAIN_LMDB" \
    --val-lmdbs "$VAL_LMDB_NAME" \
    --checkpoint-dir "$CKPT_OUT" \
    --init-checkpoint "$CKPT_V4" \
    --epochs "${EPOCHS:-20}" \
    --batch-size 512 \
    --num-workers 4 \
    --seed "$SEED" \
    --device "${DEVICE:-cuda}" 2>&1 | tee "$CKPT_OUT/train.log"
else
  echo "=== [1-3] SKIP_TRAIN reuse $CKPT_OUT/best.pth ==="
  [[ -f "$CKPT_OUT/best.pth" ]] || { echo "missing $CKPT_OUT/best.pth"; exit 1; }
fi

echo "=== [4] May infer V4 vs FT(val234) ==="
mkdir -p "$EVAL_V4" "$EVAL_FT" "$RESULTS"
# month-only feature root for clean infer
TMP_MAY="/tmp/quote_features_may_only_val234"
rm -rf "$TMP_MAY"
mkdir -p "$TMP_MAY/QQQ/regular/09:30-16:00/1min" "$TMP_MAY/QQQ/regular/09:30-16:00/5min"
ln -sf "$MAY_1MIN" "$TMP_MAY/QQQ/regular/09:30-16:00/1min/2026-05.parquet"
MAY_5="$(dirname "$(dirname "$MAY_1MIN")")/5min/2026-05.parquet"
if [[ -f "$MAY_5" ]]; then
  ln -sf "$MAY_5" "$TMP_MAY/QQQ/regular/09:30-16:00/5min/2026-05.parquet"
fi

"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_V4" --config "$CONFIG" \
  --feature-root "$TMP_MAY" --option-1m-root "$OPT1M_MAY" \
  --call-bucket 2 --put-bucket 0 \
  --output-dir "$EVAL_V4" --seed "$SEED" --device "${DEVICE:-cuda}" --live-replay

"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_OUT/best.pth" --config "$CONFIG" \
  --feature-root "$TMP_MAY" --option-1m-root "$OPT1M_MAY" \
  --call-bucket 2 --put-bucket 0 \
  --output-dir "$EVAL_FT" --seed "$SEED" --device "${DEVICE:-cuda}" --live-replay

echo "=== [5] honest KPI compare on May ==="
export EVAL_V4 EVAL_FT RESULTS CKPT_V4 CKPT_OUT RAW1_MAY EDGE_Q10_FLOOR SEED
"$PY" - <<'PY'
from __future__ import annotations
import json, os
from dataclasses import replace
from pathlib import Path
import pandas as pd
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

OUT = Path(os.environ["RESULTS"])
raw1 = pd.read_parquet(os.environ["RAW1_MAY"], columns=["timestamp", "vix_level"])
raw1["timestamp"] = pd.to_datetime(raw1["timestamp"], utc=True)
raw1 = raw1.sort_values("timestamp").drop_duplicates("timestamp")
q10 = float(os.environ["EDGE_Q10_FLOOR"])

def attach(path: str) -> pd.DataFrame:
    inf = pd.read_parquet(path).copy()
    inf["timestamp"] = pd.to_datetime(inf["timestamp"], utc=True)
    inf = inf.drop(columns=[c for c in inf.columns if c == "put_gate"], errors="ignore")
    s = raw1.copy()
    s["timestamp"] = s["timestamp"] + pd.Timedelta(minutes=1)
    m = pd.merge_asof(
        inf[["timestamp"]].reset_index(drop=True),
        s.rename(columns={"vix_level": "put_gate"}),
        on="timestamp",
        direction="backward",
    )
    inf["put_gate"] = m["put_gate"].to_numpy()
    return inf

def run(name: str, path: str) -> dict:
    df = attach(path)
    cfg = replace(qcfg.LIVE_REPLAY, edge_q10_floor=q10)
    res = run_strict_replay(
        df, qcfg.FILL_MODEL, cfg, qcfg.EXIT_RAILS,
        edge_col="net_edge", edge_q10_col="net_edge_q10",
        call_edge_col="call_net_edge", put_edge_col="put_net_edge",
        put_gate_col="put_gate",
    )
    s = res.summary(position_frac=0.25)
    trades = res.trades_frame()
    daily = []
    eq = 1.0
    if trades is not None and len(trades) and "entry_ts" in trades.columns:
        t = trades.copy()
        t["entry_ny"] = pd.to_datetime(t["entry_ts"]).dt.tz_convert("America/New_York")
        t["date"] = t["entry_ny"].dt.strftime("%Y-%m-%d")
        for date, g in t.groupby("date", sort=True):
            day_eq = 1.0
            for r in g["net_return"]:
                day_eq *= 1 + 0.25 * float(r)
                eq *= 1 + 0.25 * float(r)
            daily.append({
                "date": date, "n": int(len(g)),
                "day_acct25": float(day_eq - 1),
                "cum_acct25": float(eq - 1),
                "hit": float((g["net_return"] > 0).mean()),
                "legs": g["leg"].value_counts().to_dict() if "leg" in g.columns else {},
            })
        t.to_parquet(OUT / f"{name}_may_trades.parquet", index=False)
    # 连续亏损 streak（按日）
    max_lose_streak = 0
    cur = 0
    for d in daily:
        if d["day_acct25"] < 0:
            cur += 1
            max_lose_streak = max(max_lose_streak, cur)
        else:
            cur = 0
    return {
        "model": name,
        "acct25": float(s["total_net_return"]),
        "trades": int(s["trades"]),
        "hit": float(s.get("hit_rate") or 0),
        "pf": float(s["profit_factor"]) if s.get("profit_factor") is not None else None,
        "legs": s.get("trades_by_leg"),
        "max_lose_day_streak": max_lose_streak,
        "daily": daily,
    }

rows = [
    run("v4", f"{os.environ['EVAL_V4']}/test_infer.parquet"),
    run("ft_val234", f"{os.environ['EVAL_FT']}/test_infer.parquet"),
]
out = {
    "hypothesis": "V4 val=Jan-Mar 使 Apr 贴窗偏强；改 val=Feb-Apr 后 May 是否好转",
    "train": "train_qqq_v4.lmdb",
    "val": ["2026-02", "2026-03", "2026-04"],
    "test": "2026-05 old_lock honest KPI",
    "init": os.environ["CKPT_V4"],
    "ckpt_ft": f"{os.environ['CKPT_OUT']}/best.pth",
    "gates": {"edge_q10_floor": q10, "put_gate": "raw1 +1m causal", "live_replay": True},
    "results": rows,
}
(OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: ({kk: vv for kk, vv in r.items() if kk != "daily"} if isinstance(r, dict) else r) for k, r in [("meta", {m: out[m] for m in ("hypothesis","val","test")}), ("rows", rows)]}, indent=2, ensure_ascii=False, default=str))
print()
print(f"{'model':<12} {'acct25':>10} {'trades':>7} {'hit':>7} {'lose_streak':>12}")
for r in rows:
    print(f"{r['model']:<12} {r['acct25']*100:+9.2f}% {r['trades']:7d} {r['hit']*100:6.1f}% {r['max_lose_day_streak']:12d}")
print()
for r in rows:
    print(f"--- {r['model']} daily ---")
    for d in r["daily"]:
        legs = ",".join(f"{k}{v}" for k, v in (d.get("legs") or {}).items())
        print(f"  {d['date']} n={d['n']} day={d['day_acct25']*100:+6.2f}% cum={d['cum_acct25']*100:+7.2f}% hit={d['hit']*100:3.0f}% {legs}")
PY

echo "done summary=$RESULTS/summary.json ckpt=$CKPT_OUT/best.pth"
