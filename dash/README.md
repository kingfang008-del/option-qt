# Mag7 Control Plane（`dash/`）

对齐 `qqq_btc/dashboard` 的分层，避免逻辑分叉：

| Board | 用途 |
|---|---|
| **Download** | 补数据：可配起止日期 / 标的 → 扫缺数 → 一键日历同步与股票 1s 下载 → 页面看执行日志 |
| **Offline Replay** | 离线金标结果与复跑 |
| **Stream Parity** | 流式 / Redis S5 / day `trade_log` 对拍 |
| **Live** | Shadow/Paper/Live：数据流拓扑、订阅/锁约（TradingView K 线复用 production `futu_kline` + `locks.json`）、同时持仓、滑动窗口、session 证据；内含事件/新闻审核 tab |
| **Event News** | 开盘前审核 `event_calendar_live.json` + `event_news_audit.json`；驳回误杀 → `event_news_suppress.json` |

## 一致性契约

**对拍 ≈ 实盘，只换数据源与成交：**

- 同一 `strategy_profiles/*.json`
- 同一 Scanner / TopK / 选约 / 退出状态机
- 对拍：历史 1s 流式打入 + 模型 fill
- 实盘：IBKR 实时；Shadow=模型成交，Paper/Live=券商

## 启动

推荐用 Mag7 统一入口（可与 live session 一起/单独启停）：

```bash
./maga7/SHELL/maga7_system.sh start dash     # 只开 Dash
./maga7/SHELL/maga7_system.sh start all      # Redis + Dash + dry
./maga7/SHELL/maga7_system.sh stop dash
./maga7/SHELL/maga7_system.sh status
```

或手动：

```bash
export PYTHONPATH=$PWD
# 股票 1s 下载需要：
# export MASSIVE_API_KEY=...   # 或 POLYGON_API_KEY
python dash/run.py
```

默认：<http://127.0.0.1:8501>

侧栏切换 Board；Live Redis 默认 DB **0**（S5 研究用 DB1）。

盘前/盘后校验与盘中权威路径分开：

- Redis：`fused_market_stream:maga7:{sid}:pre|post`（允许 partial）vs 盘中权威 stream
- 磁盘：`/mnt/s990/data/maga7/live_sessions/<date>/<sid>/tape/{pre|rth|post}/`（不在 git 树内）
- 覆盖：`MAG7_LIVE_SESSIONS_DIR` / `MAG7_RESULTS_DIR`

## Download 板说明

- 任务状态写在 `{stock_1s_root}/logs/dash_mag7_backfill_job.json`
- 日志：`{stock_1s_root}/logs/mag7_*.log`
- 支持：`sync_calendar`、`stock_1s`；期权 quote / 锁约 map 本页只做路径检查
- Mag7 正股 1s **左标签**；勿套用 qqq_btc resampled 右标签纠正

## 安全边界

- Download：**可**后台启停补数进程（日历 / 股票 1s）
- Event News：**可**写 `event_news_suppress.json` 并改写 live 日历中的 `news_*` 行（防误杀）；不碰 Redis / 不发单
- 其余板：不写 Redis、不代启停 Live、不发单

## Event News 审核

```bash
# 1) 先同步（生成 audit + live）
./maga7/SHELL/start_maga7_live_session.sh sync-calendar
# 2) 打开 dash → Event News（或 Live → 事件/新闻审核）
python dash/run.py
```

政策：**新闻不定方向**。`hard_risk` 仅 CEO 可自动禁票；大单/合作只审计。重大标题可 **LLM 分析**（缓存 `event_news_llm.json`，不进黑名单）。Key：`OPENAI_API_KEY` / `DEEPSEEK_API_KEY`。

## 相关

- **操作手册**：[`maga7/docs/maga7_operations_guide.md`](../maga7/docs/maga7_operations_guide.md)
- 一天流式对拍：`maga7/SHELL/run_day_stream_check.sh`
- 实盘运维：`maga7/docs/live_session_operations.md`
- qqq_btc 原板：`qqq_btc/dashboard/README.md`
