# production/SHELL 用法

> **注意**：本目录属于旧「截面排序」备用栈，**不是**当前 Mag7 生产路径。  
> Mag7 Shadow/Paper/Live 一键脚本在：`maga7/SHELL/start_maga7_live_session.sh`。

## V8 Quant System（旧栈）

1. 启动所有服务（最常用）如果服务已经在运行，它会跳过，不会重复启动。

```bash
./start_quant_system.sh start
# 或者
./start_quant_system.sh start all
```

2. 查看系统状态

```bash
./start_quant_system.sh status
```

3. 单独重启某个服务（例如只重启特征计算）

```bash
./start_quant_system.sh restart calc
```

4. 停止 Dashboard / 停止所有

```bash
./start_quant_system.sh stop dash
./start_quant_system.sh stop all
```

| 缩写 | 服务 | 脚本 |
|------|------|------|
| db | Persistence | data_persistence_service_v8_sqlite.py |
| ib | Connector | ibkr_connector_v7.py |
| calc | Engine | feature_compute_service_v7.py |
| brain | Orchestrator | system_orchestrator_v7_new.py |
| dash | Dashboard | dashboard_monitor_ultimate.py |

也可用 `quant_system.sh`（v8 新文件名映射）与 `stop_quant_system.sh`。
