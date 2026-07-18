# Mag7 SHELL 用法

本目录是 Mag7 实盘/Shadow 一键脚本所在处。  
**不要**使用 `production/SHELL`（旧截面排序备用栈，不合格）。

脚本：`start_maga7_live_session.sh`  
默认 profile：`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1`  
日志：`logs/maga7/live_session.log`（相对仓库根）  
产物：`maga7/results/live_sessions/<date>/<session_id>/`  
**操作手册（推荐先读）**：[`../docs/maga7_operations_guide.md`](../docs/maga7_operations_guide.md)  
运维细节（G4–G6）：[`../docs/live_session_operations.md`](../docs/live_session_operations.md)

## 全流程监控（分层同 qqq_btc）

```bash
# 仓库根目录
python dash/run.py
```

侧栏 Board：

1. **Download** — 补数据 / 锁约路径  
2. **Offline Replay** — 离线金标  
3. **Stream Parity** — 流式/S5/`trade_log` 对拍（模拟数据与成交）  
4. **Live** — 同时持仓、滑动窗口、Shadow/Paper session  

对拍与实盘共用同一 profile；只换数据源与成交方式。

## G4 Shadow（开盘前一键）

先开 IB Gateway Paper（API `4002`，行情 LIVE），再：

```bash
cd /home/kingfang007/文档/GitHub/option-qt/maga7/SHELL
./start_maga7_live_session.sh start shadow
```

等价简写：

```bash
./start_maga7_live_session.sh shadow
```

## 状态 / 停进程 / 看日志

```bash
./start_maga7_live_session.sh status
./start_maga7_live_session.sh stop
tail -f ../../logs/maga7/live_session.log
```

## 一天流式对拍（主路径，对齐 production trade_log）

打入一天 1s → 写 `trade_log.csv`（OPEN/CLOSE）→ 和 offline 比开平仓是否一致：

```bash
./run_day_stream_check.sh                  # 默认 2026-05-28；有 Redis 走 S5
./run_day_stream_check.sh 2026-06-02
./run_day_stream_check.sh 2026-05-28 --force-local   # 无 Redis 时进程内 1s 流
```

产物：`maga7/results/.../trade_log.csv` + `trade_log_offline.csv` + `day_stream_check.json`

## 盘前加固（可选加餐：故障注入单测）

```bash
./run_premarket_hardening.sh faults-only
./run_premarket_hardening.sh                 # 故障 + dry（一般不必，优先用上面 day stream）
```

说明：[`../docs/premarket_hardening.md`](../docs/premarket_hardening.md)


## 事件日历同步（盘前）

```bash
./start_maga7_live_session.sh sync-calendar
# 或指定区间
./start_maga7_live_session.sh sync-calendar 2026-07-01 2026-09-30
```


## G5 Paper / G6 Live

```bash
MAG7_ACCOUNT=DUxxxxxx ./start_maga7_live_session.sh start paper --account DUxxxxxx

# G6 需额外武装环境变量 + --live-orders
MAG7_LIVE_TRADING=1 MAG7_LIVE_CONFIRM=YYYY-MM-DD:hash12 \
  ./start_maga7_live_session.sh start live --account Uxxxxxx --live-orders
```

## 常用环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `MAG7_PROFILE` | 策略 profile | full_day peer3 |
| `MAG7_MODE` | shadow/paper/live | shadow |
| `MAG7_ACCOUNT` | IB 账户 | 空 |
| `MAG7_IB_PORT` | API 端口 | paper/shadow 4002，live 4001 |
| `MAG7_REDIS_DB` | Redis DB | 0 |
| `MAG7_EVENT_CALENDAR_PATH` | 事件禁入 JSON | `maga7/CONFIG/event_calendar_live.json` |
