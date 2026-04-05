#!/bin/bash

# ================= 配置区域 =================
# 项目根目录
PROJECT_ROOT=$(pwd)
# 自动探测 script 目录位置
if [ -d "$PROJECT_ROOT/script" ]; then
    SCRIPT_DIR="$PROJECT_ROOT/script"
else
    SCRIPT_DIR="$PROJECT_ROOT"
fi
LOG_DIR="$HOME/quant_project/logs"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 定义文件名 (方便统一管理)
FILE_SIGNAL="run_live_signal.py"
FILE_EXEC="run_live_exec.py"

# 定义文件名 (方便统一管理)
FILE_DB="data_persistence_service_v8_pg.py"
FILE_IB="ibkr_connector_v8.py"
FILE_ENG="feature_compute_service_v8.py"
FILE_DASH="dashboard_monitor_ultimate.py"

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================= 工具函数 =================

# 获取完整文件路径
get_script_path() {
    local name=$1
    if [ -f "$SCRIPT_DIR/$name" ]; then
        echo "$SCRIPT_DIR/$name"
    elif [ -f "$PROJECT_ROOT/$name" ]; then
        echo "$PROJECT_ROOT/$name"
    elif [ -f "$PROJECT_ROOT/baseline/$name" ]; then  # [New] 支持 baseline 目录
        echo "$PROJECT_ROOT/baseline/$name"
    elif [ -f "$PROJECT_ROOT/DB/$name" ]; then        # [New] 支持 DB 目录
        echo "$PROJECT_ROOT/DB/$name"
    else
        echo ""
    fi
}

# 检查进程是否运行
# 返回: PID 或 空
get_pid() {
    local script_name=$1
    # 排除 grep 自身，匹配 python 或 streamlit 运行的脚本名
    pgrep -f "$script_name"
}

# 启动服务
start_task() {
    local alias=$1
    local script_name=$2
    local log_file="${alias}.log"
    local full_log_path="$LOG_DIR/$log_file"
    local full_script_path=$(get_script_path "$script_name")

    # 1. 检查文件是否存在
    if [ -z "$full_script_path" ]; then
        echo -e "${RED}❌ [Error] File not found: $script_name${NC}"
        return
    fi

    # 2. 检查是否已运行 (防重)
    local existing_pid=$(get_pid "$script_name")
    if [ -n "$existing_pid" ]; then
        echo -e "${YELLOW}⚠️  $alias is already running (PID: $existing_pid). Skipping.${NC}"
        return
    fi

    echo -e "${YELLOW}⏳ Starting $alias ($script_name)...${NC}"

    # 3. 启动命令区分 (Streamlit vs Python)
    if [[ "$alias" == "Dashboard" ]]; then
        nohup streamlit run "$full_script_path" --server.port 8501 > "$full_log_path" 2>&1 &
    else
        # -u: 禁用缓存，实时写日志
        nohup python -u "$full_script_path" > "$full_log_path" 2>&1 &
    fi
    
    local new_pid=$!
    sleep 1

    # 4. 验证启动结果
    if ps -p $new_pid > /dev/null; then
        echo -e "${GREEN}✅ $alias STARTED. (PID: $new_pid)${NC}"
    else
        echo -e "${RED}❌ $alias FAILED to start! Check logs: $full_log_path${NC}"
    fi
}

# 停止服务
stop_task() {
    local alias=$1
    local script_name=$2
    
    local pid=$(get_pid "$script_name")
    
    if [ -z "$pid" ]; then
        echo -e "${BLUE}ℹ️  $alias is not running.${NC}"
    else
        echo -e "${YELLOW}🛑 Stopping $alias (PID: $pid)...${NC}"
        kill $pid
        sleep 1
        # 强制检查
        if ps -p $pid > /dev/null; then
            echo -e "${RED}⚠️  Process stuck. Force killing...${NC}"
            kill -9 $pid
        fi
        echo -e "${GREEN}✅ $alias Stopped.${NC}"
    fi
}

# ================= 核心控制逻辑 =================

# 检查 Redis
check_redis() {
    if ! redis-cli ping > /dev/null 2>&1; then
        echo -e "${YELLOW}🔧 Redis not running. Starting...${NC}"
        redis-server --daemonize yes
        sleep 1
    fi
}

# 路由：启动
do_start() {
    local target=$1
    check_redis

    # [自动模式检测] 检查是否处于重播锁定状态
    if [ -f "/tmp/replay_active.lock" ]; then
        echo -e "${CYAN}📡 [Auto-Detect] Replay Lock found. Setting MODE=LIVEREPLAY...${NC}"
        export RUN_MODE=LIVEREPLAY
    fi
    
    case "$target" in
        all)
            start_task "Persistence" "$FILE_DB"
            sleep 1
            start_task "Connector"   "$FILE_IB"
            sleep 1
            start_task "Engine"      "$FILE_ENG"
            sleep 1
            start_task "SignalEngine" "$FILE_SIGNAL"
            sleep 1
            start_task "ExecutionEngine" "$FILE_EXEC"
            sleep 1
            start_task "Dashboard"   "$FILE_DASH"
            ;;
        replay)
            echo -e "${CYAN}🎞️  Starting System in LIVEREPLAY Mode (Redis DB 1)...${NC}"
            export RUN_MODE=LIVEREPLAY   # 👈 这里把 MODE 改为 RUN_MODE
            start_task "Persistence" "$FILE_DB"
            start_task "Engine"      "$FILE_ENG"
            start_task "SignalEngine" "$FILE_SIGNAL"
            sleep 1
            start_task "ExecutionEngine" "$FILE_EXEC"
            start_task "Dashboard"   "$FILE_DASH"
            ;;
        db)    start_task "Persistence" "$FILE_DB" ;;
        ib)    start_task "Connector"   "$FILE_IB" ;;
        calc)  start_task "Engine"      "$FILE_ENG" ;;
        brain) start_task "SignalEngine" "$FILE_SIGNAL"
            sleep 1
            start_task "ExecutionEngine" "$FILE_EXEC" ;;
        dash)  start_task "Dashboard"   "$FILE_DASH" ;;
        *)     echo -e "${RED}Unknown service: $target${NC}"; show_help ;;
    esac
}

# 路由：停止
do_stop() {
    local target=$1
    case "$target" in
        all)
            stop_task "Dashboard"    "$FILE_DASH"
            stop_task "ExecutionEngine" "$FILE_EXEC"
            stop_task "SignalEngine" "$FILE_SIGNAL"
            stop_task "Engine"       "$FILE_ENG"
            stop_task "Connector"    "$FILE_IB"
            stop_task "Persistence"  "$FILE_DB"
            ;;
        db)    stop_task "Persistence" "$FILE_DB" ;;
        ib)    stop_task "Connector"   "$FILE_IB" ;;
        calc)  stop_task "Engine"      "$FILE_ENG" ;;
        brain) stop_task "ExecutionEngine" "$FILE_EXEC"
            stop_task "SignalEngine" "$FILE_SIGNAL" ;;
        dash)  stop_task "Dashboard"   "$FILE_DASH" ;;
        *)     echo -e "${RED}Unknown service: $target${NC}"; show_help ;;
    esac
}

# 显示状态
do_status() {
    echo "=================================================="
    echo "📊 System Status Monitor"
    echo "=================================================="
    printf "%-15s %-10s %-30s\n" "Service" "Status" "PID"
    echo "--------------------------------------------------"
    
    check_one_status() {
        local name=$1
        local file=$2
        local pid=$(get_pid "$file")
        if [ -n "$pid" ]; then
            printf "%-15s ${GREEN}%-10s${NC} %-30s\n" "$name" "Running" "$pid"
        else
            printf "%-15s ${RED}%-10s${NC} %-30s\n" "$name" "Stopped" "--"
        fi
    }

    check_one_status "Persistence" "$FILE_DB"
    check_one_status "Connector"   "$FILE_IB"
    check_one_status "Engine"      "$FILE_ENG"
    check_one_status "SignalEngine" "$FILE_SIGNAL"
    check_one_status "ExecutionEngine" "$FILE_EXEC"
    check_one_status "Dashboard"   "$FILE_DASH"
    echo "=================================================="
}

show_help() {
    echo "Usage: $0 {start|stop|restart|status} [service]"
    echo ""
    echo "Commands:"
    echo "  start all      : Start entire system (safe, skips if running)"
    echo "  stop all       : Stop entire system"
    echo "  status         : Show running PIDs"
    echo ""
    echo "Individual Services:"
    echo "  db    : Persistence (SQLite)"
    echo "  ib    : IBKR Connector"
    echo "  calc  : Feature Engine"
    echo "  brain : Signal Engine & Execution Engine"
    echo "  dash  : Streamlit Dashboard"
    echo "  replay: Start Backtest System (Persistence + Engine + Orch + Dash) in LIVEREPLAY mode"
    echo ""
    echo "Example:"
    echo "  $0 restart calc   (Restart only the feature engine)"
    echo "  $0 stop dash      (Stop the dashboard)"
}

# ================= 主入口 =================

ACTION=$1
TARGET=${2:-all} # 默认操作所有

case "$ACTION" in
    start)
        do_start "$TARGET"
        ;;
    stop)
        do_stop "$TARGET"
        ;;
    restart)
        do_stop "$TARGET"
        sleep 1
        do_start "$TARGET"
        ;;
    status)
        do_status
        ;;
    *)
        show_help
        exit 1
        ;;
esac