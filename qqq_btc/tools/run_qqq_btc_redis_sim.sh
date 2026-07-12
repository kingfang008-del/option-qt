#!/usr/bin/env bash
# qqq_btc Redis 秒级高仿真 — 自动使用 ibkr conda 环境
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
exec "$PY" "$REPO/qqq_btc/tools/run_qqq_btc_redis_sim.py" "$@"
