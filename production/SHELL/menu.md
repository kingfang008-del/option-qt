如何使用这个脚本现在像一个命令行工具一样工作。1. 启动所有服务（最常用）如果服务已经在运行，它会跳过，不会重复启动。Bash./start_quant_system.sh start
# 或者
./start_quant_system.sh start all
2. 查看系统状态查看哪些服务活着，PID 是多少。Bash./start_quant_system.sh status
3. 单独重启某个服务这是调试最方便的功能。例如，你修改了特征计算代码 (feature_compute_service_v7.py)，只想重启它，不想断开 IBKR 连接：Bash./start_quant_system.sh restart calc
4. 停止 DashboardBash./start_quant_system.sh stop dash
5. 停止所有Bash./start_quant_system.sh stop all
映射关系表缩写指令对应服务名对应脚本文件dbPersistencedata_persistence_service_v8_sqlite.pyibConnectoribkr_connector_v7.pycalcEnginefeature_compute_service_v7.pybrainOrchestratorsystem_orchestrator_v7_new.pydashDashboarddashboard_monitor_ultimate.py