#!/usr/bin/env bash
# val=2026-04~05, test=2026-06 — 从 V4 finetune，对比 6 月基线
# 特征默认用 7/5 冻住 bak/归档（与 +97%/+143% 同谱系），不用当前被刷新的 quote_features_test
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
DATA_ROOT="$HOME/train_data/lmdb"
CONFIG="qqq_btc/CONFIG/slow_feature_qqq_v2.json"
SYM="qqq_btc/CONFIG/symbol_map.json"
CKPT_V4="${CKPT_V4:-checkpoint/checkpoints_qqq_v4/best.pth}"
CKPT_OUT="${CKPT_OUT:-checkpoint/checkpoints_qqq_val45_test6_rerun}"
EVAL_OUT="${EVAL_OUT:-/tmp/qqq_btc_eval_val45_test6_rerun}"
RESULTS_DIR="${RESULTS_DIR:-qqq_btc/results/val45_test6_rerun}"
# 冻住 7/5 test 特征（bak == archive）
FEAT_TEST_ROOT="${FEAT_TEST_ROOT:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ}"
FEAT_TEST="$FEAT_TEST_ROOT/regular/09:30-16:00"
OPTION_1M="${OPTION_1M:-/mnt/s990/data/raw_1m/options_databento}"
VAL45_ROOT="/tmp/quote_features_val45_rerun"
TEST_Q2_ROOT="/tmp/quote_features_test_q2_rerun"
ARCH_INFER="${ARCH_INFER:-/mnt/s990/data/v4_original_jul5/eval_v4/test_infer.parquet}"

setup_month_links() {
  local dest_root="$1"
  shift
  rm -rf "$dest_root"
  mkdir -p "$dest_root/QQQ/regular/09:30-16:00/1min"
  mkdir -p "$dest_root/QQQ/regular/09:30-16:00/5min"
  for ym in "$@"; do
    ln -sf "$FEAT_TEST/1min/${ym}.parquet" "$dest_root/QQQ/regular/09:30-16:00/1min/${ym}.parquet"
    ln -sf "$FEAT_TEST/5min/${ym}.parquet" "$dest_root/QQQ/regular/09:30-16:00/5min/${ym}.parquet"
  done
}

echo "=== [0] 特征源: $FEAT_TEST_ROOT ==="
for ym in 2026-04 2026-05 2026-06; do
  [[ -f "$FEAT_TEST/1min/${ym}.parquet" ]] || { echo "missing $FEAT_TEST/1min/${ym}.parquet"; exit 1; }
  [[ -f "$FEAT_TEST/5min/${ym}.parquet" ]] || { echo "missing $FEAT_TEST/5min/${ym}.parquet"; exit 1; }
done
[[ -f "$CKPT_V4" ]] || { echo "missing V4 ckpt: $CKPT_V4"; exit 1; }
[[ -f "$DATA_ROOT/train_qqq_v5.lmdb/data.mdb" ]] || { echo "missing train_qqq_v5.lmdb"; exit 1; }

echo "=== [0b] 准备 val(4-5) / Q2(4-6) 特征目录 ==="
setup_month_links "$VAL45_ROOT" 2026-04 2026-05
setup_month_links "$TEST_Q2_ROOT" 2026-04 2026-05 2026-06

echo "=== [1] 建 val LMDB (4-5月); train 复用 train_qqq_v5.lmdb ==="
"$PY" qqq_btc/tools/build_lmdb.py \
  --feature-root "$VAL45_ROOT" \
  --config "$CONFIG" --symbol-map "$SYM" \
  --output "$DATA_ROOT/val_qqq_val45_rerun.lmdb" --symbols QQQ

echo "=== [2] finetune: init=$CKPT_V4, val=4-5月 ==="
mkdir -p "$CKPT_OUT"
"$PY" -m qqq_btc.model.train \
  --mode finetune \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-lmdb train_qqq_v5.lmdb \
  --val-lmdbs val_qqq_val45_rerun.lmdb \
  --checkpoint-dir "$CKPT_OUT" \
  --init-checkpoint "$CKPT_V4" \
  --epochs 20 \
  --device auto 2>&1 | tee "$CKPT_OUT/train.log"

echo "=== [3] 全 Q2 infer (公平口径: 4-5 月分位预热) ==="
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_OUT/best.pth" \
  --feature-root "$TEST_Q2_ROOT" \
  --option-1m-root "$OPTION_1M" \
  --output-dir "$EVAL_OUT" \
  --device auto

echo "=== [4] 对比归档 V4 infer vs val45 微调 (拆 6 月 + Q2) ==="
mkdir -p "$RESULTS_DIR"
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
out_dir = Path("$RESULTS_DIR")

def metrics(path, label):
    df = prepare_minute_frame(pd.read_parquet(path))
    r = run_strict_replay(df, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS, **kw)
    monthly = {}
    for t in r.trades:
        m = int(pd.to_datetime(t.entry_ts).month)
        monthly.setdefault(m, []).append(t)
    monthly_out = {}
    for m, trades in sorted(monthly.items()):
        eq = 1.0
        for t in sorted(trades, key=lambda x: pd.to_datetime(x.entry_ts)):
            eq *= 1.0 + f * t.net_return
        monthly_out[m] = {
            "n_trades": len(trades),
            "acct25_pct": (eq - 1.0) * 100.0,
        }
    # Q2 compound across Apr-Jun in time order
    q2 = [t for t in r.trades if pd.to_datetime(t.entry_ts).month in (4, 5, 6)]
    eq = 1.0
    for t in sorted(q2, key=lambda x: pd.to_datetime(x.entry_ts)):
        eq *= 1.0 + f * t.net_return
    return {
        "label": label,
        "path": str(path),
        "n_trades_all": len(r.trades),
        "q2_acct25_pct": (eq - 1.0) * 100.0,
        "q2_n_trades": len(q2),
        "monthly": monthly_out,
        "position_frac": f,
    }

arch = Path("$ARCH_INFER")
newp = Path("$EVAL_OUT") / "test_infer.parquet"
rows = []
if arch.exists():
    rows.append(metrics(arch, "v4_archive_infer"))
else:
    print("WARN: missing archive infer", arch)
rows.append(metrics(newp, "val45_test6_rerun"))

summary = {
    "feat_root": "$FEAT_TEST_ROOT",
    "ckpt_v4": "$CKPT_V4",
    "ckpt_out": "$CKPT_OUT/best.pth",
    "option_1m": "$OPTION_1M",
    "results": rows,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
print()
print(f"{'模型':<28} {'6月%':>10} {'6月笔':>6} {'Q2%':>10} {'Q2笔':>6}")
for row in rows:
    jun = row["monthly"].get(6, {"acct25_pct": float("nan"), "n_trades": 0})
    print(
        f"{row['label']:<28} {jun['acct25_pct']:+9.2f}% {jun['n_trades']:>6} "
        f"{row['q2_acct25_pct']:+9.2f}% {row['q2_n_trades']:>6}"
    )
if len(rows) == 2:
    j0 = rows[0]["monthly"].get(6, {}).get("acct25_pct")
    j1 = rows[1]["monthly"].get(6, {}).get("acct25_pct")
    if j0 is not None and j1 is not None:
        print(f"{'Δ 6月':<28} {j1 - j0:+9.2f}pp")
PY

echo "done -> $CKPT_OUT/best.pth"
echo "summary -> $RESULTS_DIR/summary.json"
