#!/usr/bin/env bash
# 生产周更微调一键启动
#
# 规范:
#   - V4 底座 + 近月微调
#   - OOS 周门禁：候选不明显差于 baseline 才晋升 production link
#   - 实时反馈: status.json + pipeline.log
#
# 用法:
#   bash qqq_btc/tools/weekly_finetune.sh
#   bash qqq_btc/tools/weekly_finetune.sh --dry-run
#   bash qqq_btc/tools/weekly_finetune.sh --train-months 2026-05,2026-06 --val-months 2026-06
#   bash qqq_btc/tools/weekly_finetune.sh --watch   # 另开状态轮询（同终端分屏看 status）
#
# 另开终端看实时状态:
#   watch -n 2 cat ~/train_data/weekly_finetune_runs/<run_id>/status.json
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
POLICY="${POLICY:-qqq_btc/CONFIG/weekly_finetune_policy.json}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/train_data/weekly_finetune_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WATCH=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) WATCH=1; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$RUNS_ROOT/$RUN_ID"
STATUS="$RUNS_ROOT/$RUN_ID/status.json"
LOG="$RUNS_ROOT/$RUN_ID/launcher.log"

echo "[weekly_finetune] run_id=$RUN_ID"
echo "[weekly_finetune] status=$STATUS"
echo "[weekly_finetune] 另开终端: watch -n 2 cat $STATUS"

if [[ "$WATCH" -eq 1 ]]; then
  (
    for _ in $(seq 1 3600); do
      if [[ -f "$STATUS" ]]; then
        clear
        date
        python3 - <<PY
import json
from pathlib import Path
p=Path("$STATUS")
d=json.loads(p.read_text())
print(f"stage={d.get('stage')}  pct={d.get('pct')}%  ok={d.get('ok')}")
print(d.get('message'))
g=d.get('gate') or {}
if g:
    print('gate.passed=', g.get('passed'), 'reasons=', g.get('reasons'))
PY
        stage=$(python3 -c "import json;print(json.load(open('$STATUS')).get('stage',''))")
        [[ "$stage" == "done" || "$stage" == "dry_run" ]] && break
      else
        echo "waiting for status.json ..."
      fi
      sleep 2
    done
  ) &
  WATCH_PID=$!
fi

set +e
"$PY" qqq_btc/tools/weekly_finetune.py \
  --policy "$POLICY" \
  --run-id "$RUN_ID" \
  "${ARGS[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

if [[ "$WATCH" -eq 1 ]]; then
  wait "$WATCH_PID" 2>/dev/null || true
fi

echo
echo "[weekly_finetune] exit=$RC"
echo "[weekly_finetune] status=$STATUS"
echo "[weekly_finetune] summary=$RUNS_ROOT/$RUN_ID/summary.json"
exit "$RC"
