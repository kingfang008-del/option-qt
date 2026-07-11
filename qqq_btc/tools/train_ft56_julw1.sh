#!/usr/bin/env bash
# V4 在 bak 5–6 月上 finetune，再 replay 2026-07 第一周
# train=May+Jun (bak 冻住特征), val=Jun, init=V4, seed=42
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
CKPT_OUT="${CKPT_OUT:-checkpoint/checkpoints_qqq_ft56_julw1}"
EVAL_FT="${EVAL_FT:-qqq_btc/results/ft56_julw1}"
EVAL_BASE="${EVAL_BASE:-qqq_btc/results/v4_base_julw1}"
RESULTS="${RESULTS:-qqq_btc/results/ft56_julw1_compare}"

FEAT_BAK="${FEAT_BAK:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00}"
FEAT_JUL="${FEAT_JUL:-$HOME/train_data/july_w1_v4_databento/quote_features_test}"
OPT1M_JUL="${OPT1M_JUL:-$HOME/train_data/july_w1_v4_databento/options_1m_july_w1}"

TRAIN_FEAT="/tmp/quote_features_ft56_train"
VAL_FEAT="/tmp/quote_features_ft56_val"

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
  [[ -f "$FEAT_BAK/1min/${ym}.parquet" ]] || { echo "missing bak $ym"; exit 1; }
done
[[ -f "$FEAT_JUL/QQQ/regular/09:30-16:00/1min/2026-07.parquet" ]] || { echo "missing july feat"; exit 1; }
[[ -d "$OPT1M_JUL/QQQ" ]] || { echo "missing july option 1m"; exit 1; }
[[ -f "$CKPT_V4" ]] || { echo "missing V4"; exit 1; }

echo "=== [1] feature dirs: train=5+6, val=6 ==="
setup_months "$TRAIN_FEAT" 2026-05 2026-06
setup_months "$VAL_FEAT" 2026-06

echo "=== [2] build LMDB ==="
"$PY" qqq_btc/tools/build_lmdb.py \
  --feature-root "$TRAIN_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
  --output "$DATA_ROOT/train_qqq_ft56.lmdb" --symbols QQQ --window-step 1
"$PY" qqq_btc/tools/build_lmdb.py \
  --feature-root "$VAL_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
  --output "$DATA_ROOT/val_qqq_ft56.lmdb" --symbols QQQ --window-step 1

echo "=== [3] finetune V4 on May-Jun (seed=$SEED) ==="
rm -rf "$CKPT_OUT"
mkdir -p "$CKPT_OUT"
"$PY" -m qqq_btc.model.train \
  --mode finetune \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-lmdb train_qqq_ft56.lmdb \
  --val-lmdbs val_qqq_ft56.lmdb \
  --checkpoint-dir "$CKPT_OUT" \
  --init-checkpoint "$CKPT_V4" \
  --epochs 20 \
  --batch-size 512 \
  --num-workers 4 \
  --seed "$SEED" \
  --device cuda 2>&1 | tee "$CKPT_OUT/train.log"

echo "=== [4] July W1 replay: baseline V4 vs ft56 ==="
mkdir -p "$EVAL_FT" "$EVAL_BASE" "$RESULTS"
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_V4" --config "$CONFIG" \
  --feature-root "$FEAT_JUL" --option-1m-root "$OPT1M_JUL" \
  --output-dir "$EVAL_BASE" --seed "$SEED" --device cuda

"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_OUT/best.pth" --config "$CONFIG" \
  --feature-root "$FEAT_JUL" --option-1m-root "$OPT1M_JUL" \
  --output-dir "$EVAL_FT" --seed "$SEED" --device cuda

echo "=== [5] compare ==="
"$PY" - <<PY
import json
from pathlib import Path
import pandas as pd
from qqq_btc.common.event_replay import prepare_minute_frame
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

f = qcfg.REPLAY.position_frac
kw = dict(
    edge_col="net_edge", edge_q10_col=qcfg.EDGE_Q10_COL,
    call_edge_col=qcfg.CALL_EDGE_COL, put_edge_col=qcfg.PUT_EDGE_COL,
    put_gate_col=qcfg.PUT_GATE_COL,
)
# July W1 trading days in this dataset
W1 = {"2026-07-01","2026-07-02","2026-07-06","2026-07-07","2026-07-08","2026-07-09"}

def summarize(path, label):
    df = prepare_minute_frame(pd.read_parquet(path))
    r = run_strict_replay(df, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS, **kw)
    trades = r.trades_frame()
    if len(trades) == 0:
        return {"label": label, "trades": 0, "acct25": 0.0, "hit": None, "legs": {}, "by_day": {}}
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"])
    if trades["entry_ts"].dt.tz is None:
        trades["entry_ts"] = trades["entry_ts"].dt.tz_localize("America/New_York")
    else:
        trades["entry_ts"] = trades["entry_ts"].dt.tz_convert("America/New_York")
    trades["day"] = trades["entry_ts"].dt.strftime("%Y-%m-%d")
    trades = trades[trades["day"].isin(W1)].sort_values("entry_ts")
    eq = 1.0
    for ret in trades["net_return"]:
        eq *= 1.0 + f * float(ret)
    by_day = {}
    for d, g in trades.groupby("day"):
        e = 1.0
        for ret in g.sort_values("entry_ts")["net_return"]:
            e *= 1.0 + f * float(ret)
        by_day[d] = {"n": int(len(g)), "acct25": float(e - 1.0), "sum_net": float(g["net_return"].sum())}
    legs = trades["leg"].value_counts().to_dict() if "leg" in trades.columns else {}
    return {
        "label": label,
        "trades": int(len(trades)),
        "acct25": float(eq - 1.0),
        "hit": float((trades["net_return"] > 0).mean()),
        "sum_net": float(trades["net_return"].sum()),
        "legs": {str(k): int(v) for k, v in legs.items()},
        "by_day": by_day,
        "ic": None,
    }

rows = [
    summarize(Path("$EVAL_BASE") / "test_infer.parquet", "v4_base"),
    summarize(Path("$EVAL_FT") / "test_infer.parquet", "v4_ft56"),
]
# attach IC from eval summaries if present
for row, sp in [
    (rows[0], Path("$EVAL_BASE") / "replay_summary.json"),
    (rows[1], Path("$EVAL_FT") / "replay_summary.json"),
]:
    if sp.exists():
        s = json.loads(sp.read_text())
        row["ic"] = (s.get("label_metrics") or {}).get("ic")
        row["replay_summary_acct"] = s.get("total_net_return")

out = {
    "seed": int("$SEED"),
    "finetune": {"train": "2026-05+06 bak", "val": "2026-06 bak", "init": "$CKPT_V4", "ckpt": "$CKPT_OUT/best.pth"},
    "july_feat": "$FEAT_JUL",
    "july_option_1m": "$OPT1M_JUL",
    "july_w1_days": sorted(W1),
    "results": rows,
}
Path("$RESULTS").mkdir(parents=True, exist_ok=True)
(Path("$RESULTS") / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
print()
print(f"{'model':<12} {'acct25':>10} {'trades':>7} {'hit':>7} {'IC':>8} legs")
for r in rows:
    hit = f"{r['hit']:.1%}" if r["hit"] is not None else "n/a"
    ic = f"{r['ic']:.4f}" if r["ic"] is not None else "n/a"
    print(f"{r['label']:<12} {r['acct25']*100:+9.2f}% {r['trades']:7d} {hit:>7} {ic:>8} {r['legs']}")
print("\nby day (ft56):")
for d, v in sorted(rows[1]["by_day"].items()):
    print(f"  {d}: n={v['n']} acct={v['acct25']*100:+.1f}% sum={v['sum_net']:+.2f}")
PY

echo "done ckpt=$CKPT_OUT/best.pth"
echo "summary=$RESULTS/summary.json"
