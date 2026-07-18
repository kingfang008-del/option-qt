#!/bin/bash
# 一天流式打数 → trade_log → 对 offline（production 同款思路）
# 有 Redis 走 S5；没有则进程内 1s 流。
set -euo pipefail

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SHELL_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATE="${1:-${MAG7_HARDEN_DATE:-2026-05-28}}"
shift || true

PY="${MAG7_PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -x "$HOME/anaconda3/envs/ibkr/bin/python" ]]; then
        PY="$HOME/anaconda3/envs/ibkr/bin/python"
    else
        PY="python3"
    fi
fi

PROFILE="${MAG7_PROFILE:-maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json}"

exec "$PY" -m maga7.tools.run_day_stream_check \
  --profile "$PROFILE" \
  --date "$DATE" \
  "$@"
