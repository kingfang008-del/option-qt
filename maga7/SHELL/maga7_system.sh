#!/bin/bash
# Mag7 统一启停入口（对齐 production/SHELL/quant_system.sh 用法）
# 管理：Redis / Dash 控制面 / Live session（dry|shadow|paper|live）
#
# 用法:
#   ./maga7_system.sh start all              # Redis + Dash + dry session
#   ./maga7_system.sh start dash
#   ./maga7_system.sh start dry              # 或 shadow|paper|live
#   ./maga7_system.sh stop all|dash|live
#   ./maga7_system.sh restart dash
#   ./maga7_system.sh status
#   ./maga7_system.sh sync|preopen|preflight
#
# 环境变量（可选）:
#   MAG7_PYTHON / MAG7_PROFILE / MAG7_ACCOUNT / MAG7_MODE
#   MAG7_DEFAULT_LIVE_MODE   start all 时默认 live 模式（默认 dry）
#   OPTION_QT_DASH_HOST / OPTION_QT_DASH_PORT

set -euo pipefail

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SHELL_DIR/../.." && pwd)"
LOG_DIR="${MAG7_LOG_DIR:-$PROJECT_ROOT/logs/maga7}"
PID_DIR="$LOG_DIR"
DASH_PID_FILE="$PID_DIR/dash.pid"
DASH_LOG="$LOG_DIR/dash.log"
MATCH_DASH="dash/app.py"
# Require "-m module" so interactive shells that merely mention the name do not match.
MATCH_LIVE="-m maga7.tools.run_live_session"
MATCH_PARITY="-m maga7.tools.run_live_tape_parity"
LIVE_LAUNCHER="$SHELL_DIR/start_maga7_live_session.sh"
FLOW_LAUNCHER="$SHELL_DIR/run_maga7_live_flow.sh"
PREFLIGHT="$SHELL_DIR/g4_shadow_preflight.sh"
DEFAULT_LIVE_MODE="${MAG7_DEFAULT_LIVE_MODE:-dry}"
PARITY_PID_FILE="$PID_DIR/tape_parity.pid"
PARITY_LOG="$LOG_DIR/tape_parity.log"
LIVE_SESSIONS_DIR="${MAG7_LIVE_SESSIONS_DIR:-/mnt/s990/data/maga7/live_sessions}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

log_info() { echo -e "${CYAN}$*${NC}"; }
log_ok() { echo -e "${GREEN}$*${NC}"; }
log_warn() { echo -e "${YELLOW}$*${NC}"; }
log_err() { echo -e "${RED}$*${NC}"; }

resolve_python() {
    if [ -n "${MAG7_PYTHON:-}" ] && [ -x "${MAG7_PYTHON}" ]; then
        echo "$MAG7_PYTHON"
        return
    fi
    if [ -x "$HOME/anaconda3/envs/ibkr/bin/python" ]; then
        echo "$HOME/anaconda3/envs/ibkr/bin/python"
        return
    fi
    if [ -x "$HOME/miniconda3/envs/ibkr/bin/python" ]; then
        echo "$HOME/miniconda3/envs/ibkr/bin/python"
        return
    fi
    command -v python3 2>/dev/null || command -v python 2>/dev/null || true
}

get_pids() {
    # Match real python -m workers only (exclude bash/cursor wrappers that embed the same text).
    local needle=$1
    ps -eo pid=,args= 2>/dev/null | awk -v n="$needle" '
        index($0, n) && $0 ~ /(^|\/)python[0-9.]* / && $0 !~ /extglob/ { print $1 }
    ' || true
}

check_redis() {
    local host="${MAG7_REDIS_HOST:-127.0.0.1}"
    local port="${MAG7_REDIS_PORT:-6379}"
    if redis-cli -h "$host" -p "$port" ping >/dev/null 2>&1; then
        log_ok "✅ Redis ready ($host:$port)"
        return 0
    fi
    log_warn "🔧 Redis not running. Starting redis-server --daemonize yes ..."
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes || true
        sleep 1
    fi
    if redis-cli -h "$host" -p "$port" ping >/dev/null 2>&1; then
        log_ok "✅ Redis ready"
        return 0
    fi
    log_err "❌ Redis unavailable"
    return 1
}

# ---------- Dash ----------

start_dash() {
    local pids
    pids="$(get_pids "$MATCH_DASH")"
    if [ -n "$pids" ]; then
        log_warn "⚠️  Dash already running (PID: $pids). Skipping."
        return 0
    fi
    local py
    py="$(resolve_python)"
    if [ -z "$py" ] || [ ! -x "$py" ]; then
        log_err "❌ Python not found"
        return 1
    fi
    if [ ! -f "$PROJECT_ROOT/dash/run.py" ]; then
        log_err "❌ dash/run.py not found"
        return 1
    fi
    local host="${OPTION_QT_DASH_HOST:-127.0.0.1}"
    local port="${OPTION_QT_DASH_PORT:-8501}"
    log_info "⏳ Starting Mag7 Dash ($host:$port)..."
    nohup env \
        PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        OPTION_QT_DASH_HOST="$host" \
        OPTION_QT_DASH_PORT="$port" \
        "$py" -u "$PROJECT_ROOT/dash/run.py" \
        >"$DASH_LOG" 2>&1 &
    local new_pid=$!
    echo "$new_pid" >"$DASH_PID_FILE"
    sleep 1.5
    pids="$(get_pids "$MATCH_DASH")"
    if [ -n "$pids" ]; then
        log_ok "✅ Dash STARTED. (PID: $pids)  http://${host}:${port}"
        log_info "   log: $DASH_LOG"
    else
        log_err "❌ Dash FAILED to start. Check: $DASH_LOG"
        return 1
    fi
}

stop_dash() {
    local pids
    pids="$(get_pids "$MATCH_DASH")"
    if [ -z "$pids" ]; then
        log_info "ℹ️  Dash is not running."
        rm -f "$DASH_PID_FILE"
        return 0
    fi
    log_warn "🛑 Stopping Dash (PID: $pids)..."
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(get_pids "$MATCH_DASH")"
    if [ -n "$pids" ]; then
        log_warn "force kill Dash..."
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
    fi
    rm -f "$DASH_PID_FILE"
    log_ok "✅ Dash Stopped."
}

# ---------- Live session (delegate) ----------

start_live() {
    local mode="${1:-$DEFAULT_LIVE_MODE}"
    shift || true
    if [ ! -x "$LIVE_LAUNCHER" ]; then
        log_err "❌ missing $LIVE_LAUNCHER"
        return 1
    fi
    local pids
    pids="$(get_pids "$MATCH_LIVE")"
    if [ -n "$pids" ]; then
        log_warn "⚠️  Live session already running (PID: $pids). Skipping."
        return 0
    fi
    log_info "⏳ Starting Mag7 live session mode=$mode ..."
    "$LIVE_LAUNCHER" start "$mode" "$@"
}

stop_live() {
    if [ ! -x "$LIVE_LAUNCHER" ]; then
        log_err "❌ missing $LIVE_LAUNCHER"
        return 1
    fi
    "$LIVE_LAUNCHER" stop
}

# ---------- Aggregates ----------

do_start() {
    local target=$1
    shift || true
    case "$target" in
        all)
            check_redis || return 1
            start_dash
            sleep 1
            start_live "$DEFAULT_LIVE_MODE" "$@"
            ;;
        redis)
            check_redis
            ;;
        dash|dashboard|ui)
            check_redis || true
            start_dash
            ;;
        live|session)
            check_redis || return 1
            start_live "$DEFAULT_LIVE_MODE" "$@"
            ;;
        dry|shadow|paper)
            check_redis || return 1
            start_live "$target" "$@"
            ;;
        g6|live-orders)
            check_redis || return 1
            start_live live "$@"
            ;;
        flow)
            check_redis || return 1
            "$FLOW_LAUNCHER" start "${1:-$DEFAULT_LIVE_MODE}" "${@:2}"
            ;;
        *)
            log_err "Unknown service: $target"
            show_help
            return 1
            ;;
    esac
}

do_stop() {
    local target=$1
    case "$target" in
        all)
            stop_dash
            stop_live
            ;;
        dash|dashboard|ui)
            stop_dash
            ;;
        live|session|dry|shadow|paper|g6|live-orders|flow)
            stop_live
            ;;
        redis)
            log_warn "ℹ️  Redis 不由本脚本停止（避免误伤其它服务）。"
            ;;
        *)
            log_err "Unknown service: $target"
            show_help
            return 1
            ;;
    esac
}

do_status() {
    echo "=================================================="
    echo "📊 Mag7 System Status"
    echo "Root: $PROJECT_ROOT"
    echo "NY now: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "=================================================="
    printf "%-16s %-10s %-30s\n" "Service" "Status" "PID"
    echo "--------------------------------------------------"

    check_one() {
        local name=$1
        local needle=$2
        local pid
        pid="$(get_pids "$needle")"
        if [ -n "$pid" ]; then
            printf "%-16s ${GREEN}%-10s${NC} %-30s\n" "$name" "Running" "$pid"
        else
            printf "%-16s ${RED}%-10s${NC} %-30s\n" "$name" "Stopped" "--"
        fi
    }

    local rh="${MAG7_REDIS_HOST:-127.0.0.1}"
    local rp="${MAG7_REDIS_PORT:-6379}"
    if redis-cli -h "$rh" -p "$rp" ping >/dev/null 2>&1; then
        printf "%-16s ${GREEN}%-10s${NC} %-30s\n" "Redis" "Ready" "$rh:$rp"
    else
        printf "%-16s ${RED}%-10s${NC} %-30s\n" "Redis" "Down" "$rh:$rp"
    fi

    check_one "Dash" "$MATCH_DASH"
    check_one "LiveSession" "$MATCH_LIVE"
    check_one "TapeParity" "$MATCH_PARITY"

    echo "=================================================="
    local host="${OPTION_QT_DASH_HOST:-127.0.0.1}"
    local port="${OPTION_QT_DASH_PORT:-8501}"
    echo "Dash URL : http://${host}:${port}"
    echo "Dash log : $DASH_LOG"
    echo "Live log : $LOG_DIR/live_session.log"
    echo "Launcher : $LIVE_LAUNCHER"
    if [ -x "$LIVE_LAUNCHER" ]; then
        echo ""
        "$LIVE_LAUNCHER" status 2>/dev/null | head -n 40 || true
    fi
}

do_oneshot() {
    local cmd=$1
    shift || true
    case "$cmd" in
        sync|sync-calendar)
            "$LIVE_LAUNCHER" sync-calendar "$@"
            ;;
        preopen)
            "$FLOW_LAUNCHER" preopen "$@"
            ;;
        preflight)
            "$PREFLIGHT" "$@"
            ;;
        resume)
            if [ $# -lt 1 ]; then
                log_err "resume 需要 session_id"
                echo "Example: $0 resume live_20260720_xxx dry"
                return 1
            fi
            "$LIVE_LAUNCHER" resume "$@"
            ;;
        parity)
            do_parity once "$@"
            ;;
        *)
            return 1
            ;;
    esac
}

do_parity() {
    local mode="${1:-once}"
    shift || true
    local py
    py="$(resolve_python)"
    if [ -z "$py" ]; then
        log_err "未找到 Python"
        return 1
    fi
    local session_arg=()
    if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
        session_arg=(--session-dir "$1")
        shift || true
    fi
    case "$mode" in
        once|parity)
            "$py" -u -m maga7.tools.run_live_tape_parity "${session_arg[@]}" "$@"
            ;;
        watch|loop)
            local loop_sec="${MAG7_PARITY_LOOP_SEC:-600}"
            local existing
            existing="$(get_pids "$MATCH_PARITY")"
            if [ -n "$existing" ]; then
                log_warn "tape_parity 已在运行: $existing"
                return 0
            fi
            mkdir -p "$LOG_DIR"
            nohup "$py" -u -m maga7.tools.run_live_tape_parity \
                --loop-seconds "$loop_sec" \
                "${session_arg[@]}" "$@" \
                >>"$PARITY_LOG" 2>&1 &
            echo $! >"$PARITY_PID_FILE"
            log_ok "tape_parity watch 已启动 PID=$(cat "$PARITY_PID_FILE") loop=${loop_sec}s"
            log_info "log: $PARITY_LOG"
            ;;
        stop)
            local pids
            pids="$(get_pids "$MATCH_PARITY")"
            if [ -z "$pids" ]; then
                log_warn "tape_parity 未运行"
                rm -f "$PARITY_PID_FILE"
                return 0
            fi
            # shellcheck disable=SC2086
            kill $pids 2>/dev/null || true
            sleep 1
            pids="$(get_pids "$MATCH_PARITY")"
            if [ -n "$pids" ]; then
                # shellcheck disable=SC2086
                kill -9 $pids 2>/dev/null || true
            fi
            rm -f "$PARITY_PID_FILE"
            log_ok "tape_parity 已停止"
            ;;
        *)
            log_err "parity 用法: parity [once|watch|stop] [session_dir]"
            return 1
            ;;
    esac
}

show_help() {
    cat <<EOF
Usage: $(basename "$0") {start|stop|restart|status|sync|preopen|preflight|resume|parity|watch} [service] [args...]

Commands:
  start all              Redis + Dash + Live(${DEFAULT_LIVE_MODE})
  stop all               Stop Dash + Live session
  restart <svc>          Stop then start
  status                 Show Redis / Dash / Live PIDs

Services (start/stop/restart):
  dash                   Mag7 Control Plane (Streamlit :8501)
  dry | shadow | paper   Live session mode（转发 start_maga7_live_session.sh）
  live | session         Live session（模式=\$MAG7_DEFAULT_LIVE_MODE，默认 dry）
  g6                     Live mode（真钱路径；仍需 launcher 的 live-orders 护栏）
  flow                   开盘前 sync + 摘要 + start（run_maga7_live_flow.sh）
  redis                  仅检查/拉起 Redis（不 stop）

One-shot:
  sync [START END]       同步 event calendar / news
  preopen                只做盘前 sync + 禁入摘要
  preflight              G4 盘前体检
  resume <sid> [mode]    恢复 live session
  parity [session_dir]   单次 tape↔Scanner 对拍（写 tape_parity.json）
  watch [session_dir]    后台每 10m 对拍（MAG7_PARITY_LOOP_SEC 可改）
  parity stop            停止后台对拍

Examples:
  $(basename "$0") start all
  $(basename "$0") start dash
  $(basename "$0") start dry
  $(basename "$0") start paper --account DUxxxxxx
  $(basename "$0") stop dash
  $(basename "$0") stop live
  $(basename "$0") restart all
  $(basename "$0") status
  $(basename "$0") preopen
  $(basename "$0") parity
  $(basename "$0") watch

Notes:
  - 底层 live 启停仍由 start_maga7_live_session.sh 负责（日志/PID/护栏不变）
  - 不要用 production/SHELL/quant_system.sh（旧截面排序栈）
  - 默认 start all 的 live 模式可用 MAG7_DEFAULT_LIVE_MODE=shadow 覆盖
  - Live 落盘: ${LIVE_SESSIONS_DIR}/<date>/<session_id>/
EOF
}

# ================= 主入口 =================

ACTION="${1:-}"
shift || true

case "$ACTION" in
    start)
        TARGET="${1:-all}"
        shift || true
        do_start "$TARGET" "$@"
        ;;
    stop)
        TARGET="${1:-all}"
        do_stop "$TARGET"
        ;;
    restart)
        TARGET="${1:-all}"
        shift || true
        do_stop "$TARGET"
        sleep 1
        do_start "$TARGET" "$@"
        ;;
    status)
        do_status
        ;;
    sync|sync-calendar|preopen|preflight|resume|parity)
        do_oneshot "$ACTION" "$@"
        ;;
    watch)
        do_parity watch "$@"
        ;;
    help|-h|--help|"")
        show_help
        [[ -n "$ACTION" ]] || exit 1
        ;;
    *)
        log_err "Unknown command: $ACTION"
        show_help
        exit 1
        ;;
esac
