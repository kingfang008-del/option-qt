# Mag7 SHELL 用法

本目录是 Mag7 实盘/Shadow 一键脚本所在处。  
**不要**使用 `production/SHELL`（旧截面排序备用栈，不合格）。

脚本：`start_maga7_live_session.sh`  
默认 profile：`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1`  
日志：`logs/maga7/live_session.log`（相对仓库根）  
产物：`maga7/results/live_sessions/<date>/<session_id>/`  
运维说明：[`../docs/live_session_operations.md`](../docs/live_session_operations.md)

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
