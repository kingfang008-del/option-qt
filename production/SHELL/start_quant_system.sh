#!/bin/bash

# ================= 配置区域 =================
# 项目根目录 (根据你的实际路径修改，或者直接在项目目录下运行)
PROJECT_ROOT=$(pwd)
SCRIPT_DIR="$PROJECT_ROOT/script"  # 假设脚本在 script 子目录下，如果都在根目录请改为 "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/logs"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 定义颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=================================================="
echo "🚀 V8 Quant System Startup (Debug Mode)"
echo "📂 Root: $PROJECT_ROOT"
echo "📂 Logs: $LOG_DIR"
echo "=================================================="

# ================= 工具函数 =================

# 启动 Python 服务的通用函数
# 用法: start_service "服务显示名" "脚本文件名" "日志文件名(可选)"
start_service() {
    local service_name=$1
    local script_name=$2
    # 如果没传第3个参数，自动生成日志名
    local log_file=${3:-"${script_name%.*}.log"}
    local full_log_path="$LOG_DIR/$log_file"

    echo -e "${YELLOW}⏳ Starting $service_name...${NC}"
    
    # 检查脚本是否存在
    # 注意：这里假设你的脚本在当前目录，或者在 script/ 目录下
    # 如果脚本找不到，尝试在 script/ 下找
    local script_path="$script_name"
    if [ ! -f "$script_path" ]; then
        if [ -f "script/$script_name" ]; then
            script_path="script/$script_name"
        elif [ -f "$PROJECT_ROOT/$script_name" ]; then
            script_path="$PROJECT_ROOT/$script_name"
        else
            echo -e "${RED}❌ Script not found: $script_name${NC}"
            return
        fi
    fi

    # 使用 nohup 后台启动
    # -u : 关键参数！禁用 Python 缓存，确保 print/logging 实时写入文件
    nohup python -u "$script_path" > "$full_log_path" 2>&1 &
    local pid=$!
    
    # 等待 2 秒进行预热和错误捕捉
    sleep 2
    
    # 检查进程是否还在运行
    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}✅ $service_name STARTED.${NC} (PID: $pid)"
        echo -e "   📄 Log: ${CYAN}$log_file${NC}"
    else
        echo -e "${RED}❌ $service_name FAILED to start!${NC}"
        echo -e "${RED}👇 Error Log Head ($log_file):${NC}"
        head -n 10 "$full_log_path"
        echo "..."
    fi
    echo "----------------------------------------"
}

# ================= 0. 环境检查 =================
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found. Please activate your conda environment first.${NC}"
    exit 1
fi

# ================= 1. 基础设施 (Redis) =================
echo -e "${YELLOW}🔧 Checking Infrastructure...${NC}"
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis not running. Attempting to start..."
    redis-server --daemonize yes
    sleep 1
fi

if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is ready.${NC}"
else
    echo -e "${RED}❌ Redis failed to start.${NC}"
    exit 1
fi
echo "----------------------------------------"

# ================= 2. 启动核心服务 (按依赖顺序) =================

# [Step 1] 持久化层 (SQLite) - 最先启动，防止漏数据
# 日志: persistence.log
start_service "Persistence (SQLite)" "data_persistence_service_v8_sqlite.py" "persistence.log"

# [Step 2] 数据源 (IBKR) - 必须在持久化之后
# 日志: connector.log
start_service "IBKR Connector" "ibkr_connector_v7.py" "connector.log"

# [Step 3] 计算引擎 (Feature Engine) - 你的关注重点！
# 这里将日志命名为 engine.log，里面会包含 realtime_feature_engine 的所有输出
start_service "Feature Engine" "feature_compute_service_v7.py" "engine.log"

# [Step 4] 策略编排 (Orchestrator) - 消费特征
# 日志: orchestrator.log
start_service "System Orchestrator" "system_orchestrator_v7.py" "orchestrator.log"

# ================= 3. 启动 Dashboard =================
echo -e "${YELLOW}⏳ Starting Dashboard...${NC}"
nohup streamlit run dashboard_monitor_ultimate.py --server.port 8501 > "$LOG_DIR/dashboard.log" 2>&1 &
dash_pid=$!
sleep 2

if ps -p $dash_pid > /dev/null; then
    echo -e "${GREEN}✅ Dashboard STARTED.${NC} (PID: $dash_pid)"
    echo -e "${GREEN}📊 Access: http://localhost:8501${NC}"
else
    echo -e "${RED}❌ Dashboard failed to start.${NC}"
fi

echo "=================================================="
echo -e "${GREEN}🎉 System Startup Sequence Complete.${NC}"
echo "=================================================="
echo -e "${YELLOW}🔎 DEBUGGING HINT:${NC}"
echo -e "To watch the Feature Engine calculation in real-time, run:"
echo -e "${CYAN}tail -f $LOG_DIR/engine.log${NC}"
echo "=================================================="