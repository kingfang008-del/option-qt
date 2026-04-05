#!/bin/bash

echo "🛑 Stopping Quant System..."

# 定义要杀死的脚本关键字列表
SCRIPTS=(
    "system_orchestrator_v7.py"
    "feature_compute_service_v7.py"
    "ibkr_connector_v7.py"
    "data_persistence_service_v8_sqlite.py"
    "dashboard_monitor_ultimate.py"
)

for script in "${SCRIPTS[@]}"; do
    # 查找进程 ID (排除 grep 自身)
    pids=$(ps -ef | grep "$script" | grep -v grep | awk '{print $2}')
    
    if [ -n "$pids" ]; then
        echo "Killing $script (PIDs: $pids)..."
        kill $pids
    else
        echo "$script is not running."
    fi
done

echo "✅ All services stopped."