# Mag7 Scanner → OMS 接入草图

## 目标

把 **多标的 Rule-A TopK / m5_circuit** 接到现有 OMS / fill audit，**不经过** QQQ TFT / FCS 主信号链路。

当前整体分层、秒级股票事实源、分钟完成语义、实时恢复和 G0–G6 门禁统一见
[`current_architecture.md`](current_architecture.md)。本文聚焦 Scanner/OMS 与 Redis
对拍细节。

## 临时生产 Profile（默认）

`maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json`

- `open_ladder` OTM5 + `only_win` + concurrent p20 + `exit_mode=mf_flip`
- S3/S4/S5 默认挂此 profile，`--scheme m5_circuit`

## New_Pro / qqq_btc 复用地图（S5）

| 复用 | 路径 | Mag7 用法 |
|---|---|---|
| ✅ Redis 行情总线 | `fused_market_stream:maga7:<run_id>` + msgpack `batch` | payload 与 IBKR connector 同形，回放按 run 隔离 |
| ✅ 序列化 | `New_Pro/baseline_qqq/utils/serialization_utils.py` | pack/unpack |
| ✅ 发球机模式 | `qqq_btc/tools/redis_fused_pitcher_1s.py` | Mag7 重写数据源，不改总线 |
| ✅ replay 时钟 | run-scoped replay clock / sync ACK | ACK 必须精确匹配同一 `run_id:timestamp` |
| ❌ FCS / TFT | `feature_compute_service_v8` | Mag7 不用 |
| ❌ QQQ Signal/OMS | `run_live_*_qqq` / ExecutionEngineV8 | Mag7 用自有 scanner + stub |
| ❌ production S5 parquet | `unified_inference_stream` 预计算特征 | 那是 QQQ RL 路径 |

结论：**复用 New_Pro 的「行情进 Redis」层，不复用 QQQ 策略栈。**

## S5 拓扑（全真模拟）

```text
Mag7FusedPitcher (stock 1s + option_contracts delta)
        │ xadd msgpack batch
        ▼
 fused_market_stream:maga7:<run_id>   (Redis db=1)
        │ xreadgroup maga7_scanner_group:<run_id>
        ▼
Mag7RedisScannerLoop
  ① 完整帧校验 / 去重 / 倒序拒绝
  ② 全标的 options 入 book
  ③ 已有仓位因果退出
  ④ 全标的 stock 1s→1m
  ⑤ TopK 新入场
        │
        ▼  Mag7OmsStub / QuoteSimSession（只用已到达 Redis quote）
   fill_audit / trades
```

```bash
export PYTHONPATH=$PWD
# 默认：期权上总线 + OMS 优先 Redis，完全因果，无整日预热
python -m maga7.tools.run_maga7_redis_sim \
  --start-date 2026-05-01 --end-date 2026-05-01 \
  --options --compare-offline

# 墙钟 1s/sec
python -m maga7.tools.run_maga7_redis_sim \
  --start-date 2026-05-01 --end-date 2026-05-01 \
  --speed 1 --options
```

当前时钟与指纹口径验证（2026-05-01，run `382ee0d7`）：

- `pitcher_ticks = consumer_batches = unique_frames = stream_len = 23400`
- duplicate / foreign / rejected frame 均为 0
- Redis option updates 685729，`n_path_disk=0`
- S5 ↔ offline 3/3 笔匹配，`max_abs_ret_diff=0`
- strategy fingerprint：
  `4b2c565c2cc9a02ccc32f50aad9509f510ddad3d3eccd789260ea75608d85472`

该结果证明当前代码下的单日 run 隔离、秒级聚合和 Redis fill 一致性；它仍只是单日
冒烟，完整日期范围 G1–G3 必须另行冻结，且不能替代 G4–G6 真实 session。

每次运行使用独立 stream、consumer group、ACK 和 `<tag>/<run_id>` 结果目录；帧完整性失败时禁止继续做 parity。

多日：须 **sync**。股票事实源统一为 `/mnt/s990/data/raw_1s/stocks`；若 1s 聚合与
`spnq_train` 派生 1m 缓存有差异，应定位缺秒、聚合或缓存生成问题。S0 是快速规则
基准，S5 才验证秒级输入和 Redis I/O。

## 拓扑（S0–S4）

```text
正股 1s → MinuteAgg → Mag7 Scanner → OMS dry/stub
```

**两层时钟**

| 层 | 粒度 | 用途 |
|---|---|---|
| 信号 | 1m（由 1s 聚合） | mf10 / streak / TopK |
| 成交 | 1s 期权 quote | fill_frac=0.8 + mf_flip |

### 为什么 S0 对拍快？

S0 = 进程内规则对拍（无 Redis、无墙钟）。验证规则路径，不是 live I/O。

| 阶段 | 工具 | Redis 行情 | 墙钟 |
|---|---|---|---|
| S0 | `run_stream_parity` | 否 | 否 |
| S2–S4 | scanner / OMS stub | 否（读盘） | 否 |
| S4 `--redis` | 只发 **下单** BUY/SELL | 订单流 | 否 |
| **S5** | `run_maga7_redis_sim` | **是**（fused） | 可选 |

## 阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 | Offline ↔ stream 规则对拍 | ✅ |
| S1 | Shadow scanner 1m | ✅ |
| S2 | 1s→1m scanner audit | ✅ |
| S3 | OMS dry-run | ✅ |
| S4 | OMS stub + fill_audit | ✅ |
| **S5** | 1s → Redis fused → scanner → OMS | ✅ `run_maga7_redis_sim` |

## Shadow 用法

```bash
export PYTHONPATH=$PWD

# S3 / S4
python -m maga7.tools.run_oms_dry_run --start-date 2026-05-01 --end-date 2026-05-15 --ingest 1m --compare-offline
python -m maga7.tools.run_oms_live_stub --start-date 2026-05-01 --end-date 2026-05-15 --ingest 1m --compare-offline

# S5 全真 Redis
python -m maga7.tools.run_maga7_redis_sim --start-date 2026-05-01 --end-date 2026-05-15 --compare-offline
```

### S4 / S5 范围

- **做**：Mag7 stub + Redis 行情总线（S5）或下单 payload（S4 `--redis`）
- **不做**：挂整机 QQQ FCS/TFT/`ExecutionEngineV8`

| Env | 默认 | 含义 |
|---|---|---|
| Redis db | 1 | New_Pro replay 约定（0=live） |
| `MAG7_MAX_QTY` | 1 | stub 张数上限 |
| `MAG7_REDIS_PUBLISH` | 0 | S4 下单 xadd |

## OMS 对接约定

1. 读 `contract` / 1s quote；
2. 限价 = fill_model(bid,ask,0.8)；
3. `exit_mode=mf_flip`；
4. `scanner.record_fill(...)` 回写 only_win；
5. **禁止** QQQ TFT 选约。

## 全流程监控

根目录 `dash/` 提供只读统一面板：

```bash
python dash/run.py
```

按 G0 数据/锁约 → G1 Offline → G2 Stream parity → G3 Redis S5 →
G4 Shadow → G5 Broker Paper → G6 Live 展示证据与阻塞项。现有历史证据只通过到
G3；即使 G4–G6 代码已实现，在对应真实 session 门禁通过前也禁止宣称完整实盘。

G4–G6 的实现入口为：

```bash
python -m maga7.tools.run_live_session --help
```

部署、硬锁、恢复、盘后和回滚步骤见
`maga7/docs/live_session_operations.md`。实现存在不等于门禁通过；G4/G5/G6
仍必须分别产生真实 Shadow、Paper、Live session 证据。
