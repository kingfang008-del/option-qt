# 盘前加固（Premarket Hardening）

## 主路径（就用这个）

和 `production` 一样：**流式打入一天 → 写 trade_log → 对 offline 开平仓**。

```bash
cd maga7/SHELL
./run_day_stream_check.sh                 # 默认 2026-05-28
./run_day_stream_check.sh 2026-06-02
./run_day_stream_check.sh 2026-05-28 --force-local   # 无 Redis
```

- 有 Redis → `run_maga7_redis_sim`（S5 fused 总线，最接近实盘）
- 无 Redis → 进程内 1s→scanner→OMS
- 落盘：`trade_log.csv`（OPEN/CLOSE）+ `trade_log_offline.csv` + `day_stream_check.json`

> PASS ≠ G4/G5/G6；真实行情 Shadow/Paper 仍要跑。

## 可选加餐（故障单测 / 多阶段）

一般不必。改 risk/OMS 纯函数时可用：

```bash
./run_premarket_hardening.sh faults-only
./run_premarket_hardening.sh   # 故障 + dry；日常请优先 day_stream_check
```

默认 profile：`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1`  
默认 scheme：`recommended_scheme=single`  
默认 smoke 日：`2026-05-28`

## 阶段

| 阶段 | 内容 | 默认 |
|------|------|------|
| fault_tests | `test_live_fault_injection` + `test_risk_guards` | 开 |
| oms_dry | `run_oms_dry_run --ingest 1s --compare-offline` | 开 |
| stream_parity | G2 `run_stream_parity --stock-source stock_1s` | `--with-parity` |
| s5_redis | `run_maga7_redis_sim --options --compare-offline --sync` | `--with-s5` |

## 故障注入覆盖（pytest）

| 场景 | 断言 |
|------|------|
| 股票报价过期 | `stock_stale` / `is_fresh=False` |
| 价差过宽 / mid 跳空 | 拒入场 |
| 出场 mid 跳空 | `gap` → `gap_force` |
| 不利成交穿 ask | `fill_above_ask` |
| 事件黑名单 | `resolve_event_blackout` 命中日 |
| 日熔断 | `DAY_CIRCUIT` + 强平 + `day_halted` |

实现：`maga7/tests/test_live_fault_injection.py`。

## 建议节奏

| 何时 | 跑什么 |
|------|--------|
| 每次改 `broker_oms` / `scanner` / `risk_guards` / `requote` | `faults-only` + dry 单日 |
| 每个交易日前 | 完整 `./run_premarket_hardening.sh`（有 Redis 则加 `--with-s5`） |
| 策略/profile 变更 | 再加 `--with-parity` |
| 连续通过后 | 才上 G4 Shadow；有成交日再冲 G5 |

## 相关

- 运维门禁：[`live_session_operations.md`](live_session_operations.md)
- S5 拓扑：[`scanner_oms_integration.md`](scanner_oms_integration.md)
- 流式对拍：[`replay_stream_parity.md`](replay_stream_parity.md)
- freeze 基线：[`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md)
