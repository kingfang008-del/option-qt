# V4 July W1 流式对拍完整记录

> 日期：2026-07-14  
> 对拍目录：`qqq_btc/results/july_w1_v4_stream_final_aligned/`  
> 离线基线：`qqq_btc/results/v4_jul_w1_honest_kpi_replay/`  
> 目标配方：V4 + honest stream + `open30 + bounce_cut + lock45`  
> 结论：Jul1–9 交易路径与收益基本对齐；流式 `+63.66%`，离线 `+65.09%`。

---

## 1. 状态定义：成功到什么程度

这轮可以称为：

```text
V4 open30+bounce+lock45 的流式交易链基本对拍成功
```

不能称为：

```text
三闸门 Gate1/2/3 全部正式 PASS
```

原因是最终全周命令使用了：

```text
SKIP_GATES=1
```

因此产物中：

```json
{
  "parity_status": "UNGATED",
  "note": "ungated diagnostic PnL (FORCE_GATE3 or SKIP_GATES)"
}
```

`gate1_raw.pass=false`、`gate2_norm.pass=false` 在这次目录里表示
“没有执行 Gate1/2”，不是本次重新比较后确认失败。

本轮成功的具体含义：

1. 使用真实 FCS→Redis→signal→OMS→fill audit 历史流式链路
2. V4 checkpoint、策略门控和分钟退出与 `+65.09%` 离线基线统一
3. Jul1 的 `SPOT_THESIS`、Jul2 大 PUT、Jul6 bounce/午后 PUT 等关键路径恢复
4. Jul1–9 流式与离线仅剩 `-1.43` 个百分点
5. 剩余主要分歧收敛到 Jul7 一笔临界 edge CALL

---

## 2. 最终结果

### 2.1 可比区间 Jul1–9

| 版本 | acct25 | 笔数 | 差异 |
|---|---:|---:|---:|
| V4 离线冻结基线 | **+65.0874%** | 12 | — |
| V4 流式 | **+63.6575%** | 13 | **-1.4299pp** |

### 2.2 包含 Jul10 的流式全段

| 区间 | acct25 | 笔数 |
|---|---:|---:|
| Jul1–9 | +63.6575% | 13 |
| Jul10 | -3.6042% | 2 |
| Jul1–10 | +57.7590% | 15 |

Jul10 不属于 `+65.09%` 离线冻结产物的有效可比区间：

- `signal_diff_20260710.json` 中 offline replay/decision 均为空
- 流式有两笔实际成交
- 因此正式 parity 统计使用 Jul1–9，不能把 Jul10 的亏损算成
  `+65.09%` 对拍残差

---

## 3. 最终运行命令

历史执行命令：

```bash
mkdir -p "qqq_btc/results/july_w1_v4_stream_final_aligned"

HONEST_OUT_DIR="$PWD/qqq_btc/results/july_w1_v4_stream_final_aligned" \
DAYS="2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09 2026-07-10" \
SKIP_GATES=1 \
CKPT="$PWD/checkpoint/checkpoints_qqq_v4/best.pth" \
QQQ_BTC_RULE_PROFILE_SELECTOR=off \
QQQ_BTC_TICK_EXITS=off \
QQQ_BTC_EDGE_Q10_FLOOR=-0.2 \
bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
```

### 3.1 为什么这些覆盖项必须显式写出

| 参数 | 本轮值 | 原因 |
|---|---|---|
| `CKPT` | V4 `best.pth` | `+65.09%` 是 V4，不是 FT56 |
| `RULE_PROFILE_SELECTOR` | `off` | 离线 `+65.09%` 没有 VX/CHOP 日切 |
| `TICK_EXITS` | `off` | 离线金标使用分钟 rails；关闭秒级退出才可逐笔对拍 |
| `EDGE_Q10_FLOOR` | `-0.2` | 与 honest KPI 冻结配置一致 |
| `SKIP_GATES` | `1` | 本轮目标是修复后快速完成交易链/PnL 全周重跑 |
| `position_frac` | `0.25` | 所有日收益和累计收益使用账户 25% 仓位复利 |

---

## 4. 模型、数据与运行栈

### 4.1 模型

```text
checkpoint/checkpoints_qqq_v4/best.pth
```

不是：

```text
checkpoint/checkpoints_qqq_ft56_julw1/best.pth
```

### 4.2 期权秒级数据

```text
/mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s
```

### 4.3 股票与 VIXY 预热数据

```text
~/train_data/spnq_train
```

运行前将 2026 年 6 月 QQQ/VIXY 分钟数据写入 PG，执行 Deep Warmup；
每日运行前再注入前一个交易日，避免 rolling 特征冷启动。

### 4.4 honest 离线参照特征

Gate1 raw 参照：

```text
~/train_data/july_w1_v4_honest_openwin/
  quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet
```

Gate2 norm 参照：

```text
~/train_data/july_w1_v4_honest_openwin/
  quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet
```

### 4.5 归一化

```text
qqq_btc/CONFIG/frozen_norm_qqq_daily.npz
```

### 4.6 honest stream 约束

```text
OMS_MOCK_IBKR=1
REDIS_STREAM_SIM=1
QQQ_BTC_LIVE=1
QQQ_BTC_USE_LIVE_REPLAY=1
QQQ_BTC_PUT_GATE_MODE=vixy_z
QQQ_BTC_REGIME_GOLD_1M=0
RECALC_GREEKS=1
FCS_FORCE_RECALC_GREEKS=1
FCS_OPTION_T_LABEL=end
FCS_IV_PRICE_MODE=close
QQQ_BTC_LIVE_LABEL_SHIFT_SEC=60
EXECUTION_DELAY_BARS=0
OMS_SIGNAL_DELAY_BARS=0
BACKTEST_OPT_FILL_SPREAD_FRAC=0.775
```

明确关闭的“开卷拐杖”：

- greek parity / day_iv Greeks 注入
- `feature5m` put gate 金标
- regime gold 文件
- bak June rolling-norm seed

`quote_options_day_iv` 在该脚本中只用于补分钟 volume，不注入 Greeks。

---

## 5. 离线冻结目标

来源：

```text
qqq_btc/results/v4_jul_w1_honest_kpi_replay/summary.json
qqq_btc/results/v4_jul_w1_honest_kpi_replay/replay_trades.parquet
```

配方：

```text
V4
+ honest causal put_gate
+ LIVE_REPLAY
+ edge_q10_floor=-0.2
+ open30
+ bounce_cut / SPOT_THESIS
+ thesis_lock_leg_bars=45
```

日收益：

| 日期 | 离线 acct25 | 笔数 |
|---|---:|---:|
| Jul1 | -0.4248% | 2 |
| Jul2 | +16.2397% | 1 |
| Jul6 | +7.0778% | 2 |
| Jul7 | +25.6063% | 2 |
| Jul8 | +4.9200% | 3 |
| Jul9 | +1.0739% | 2 |
| **累计** | **+65.0874%** | **12** |

---

## 6. 初始为什么对不上

最初并不是一个 bug，而是“模型、策略、退出、字段接线”同时不一致。

### 6.1 模型不一致

错误比较：

```text
离线：V4 +65.09%
流式：FT56
```

V4 和 FT56 的 edge、腿选择、入场时刻不同，不能直接做收益 parity。

### 6.2 VX selector 不一致

流式曾启用：

```text
QQQ_BTC_RULE_PROFILE_SELECTOR=vx
```

离线 `+65.09%` 没有 VX selector。VX 会把 Jul8/Jul9 判为
`CHOP_NO_TRADE`，删除基线中 5 笔盈利交易。

### 6.3 tick exits 不一致

流式曾叠加秒级 `TICK_FAST_HARD` 等保护；离线金标只有分钟 rails。
秒级保护适合生产，但不适合要求逐笔收益完全相同的基线对拍。

### 6.4 `REPLAY` 与 `LIVE_REPLAY` 不一致

历史对拍工具曾使用：

```text
REPLAY: entry_delay_bars=1
```

实际流式使用：

```text
LIVE_REPLAY: immediate_entry=True, entry_delay_bars=0
```

一根 bar 的差异足以改变 0DTE/1DTE 期权成交价格和后续持仓路径。

### 6.5 q10 不一致

honest KPI 使用：

```text
edge_q10_floor=-0.2
```

文件默认或其他栈可能使用 `-0.25`，会改变 CALL 是否放行。

### 6.6 bounce-cut 实时字段未接通

最直接的代码故障：

```text
strategy_entry_bridge 需要 vwap_log_return
REGIME_CTX_KEYS 却没有传递 vwap_log_return
```

结果：

- 离线 Jul1 10:47 PUT 在数根后 `SPOT_THESIS`
- 流式 governor 的 `vwap_lrs` 为空
- 入场时没有挂上 bounce rails
- 最终拖到 `TICK_FAST_HARD`，约 `-18.36%`

### 6.7 bounce 输入记录时序错误

即使字段进入 context，如果只在 `decide_entry_via_replay()` 中记录，
被 V0 pre-condition 拒绝的分钟仍会断档。

离线 replay 每根分钟 close 都观察 spot/vwap；流式也必须在 V0 门控前
持续记录，不能只在最终准备开仓时补一根。

### 6.8 `spot_close` 标签不一致

退出 thesis 需要与 FCS/model 同一标签的分钟 close。
若退回 OMS 当前 tick/stock price，会领先或落后一拍，改变 bounce onset。

### 6.9 仓位状态没有跨分钟持久化

ExecutionEngine 会反复重建 holding context。若只把以下状态放在临时 dict：

```text
entry_spot
position spot_closes
```

下一分钟就会丢失，`SPOT_THESIS` 永远无法与离线相同。

### 6.10 入场前 momentum 历史不足

离线 thesis 使用入场前已有的日内 close 历史。流式若在开仓时把
`position_spot_closes` 初始化为空，会人为多等待一根或数根 bar。

最终修复是在开仓时复制：

```text
day_state.spot_closes[-(mom_window + 1):]
```

---

## 7. 修复过程

### 修复 1：传递 bounce 原始字段

文件：

```text
qqq_btc/live/regime_ctx.py
```

增加：

```text
vwap_log_return
spot_close
```

`spot_close` 使用 FCS enriched frame 的同标签 `close`，不退回领先一拍的
OMS tick price。

### 修复 2：bounce 输入按分钟去重

文件：

```text
qqq_btc/live/session_governor.py
```

分别记录：

```text
last_bounce_spot_minute_key
last_bounce_vwap_minute_key
```

同一分钟重复计算覆盖末值，不伪造额外 bar。

### 修复 3：在 V0 门控前记录输入

文件：

```text
qqq_btc/live/strategy_entry_bridge.py
```

把 `record_bounce_inputs()` 移到 `record_session_edges_from_ctx()`，
确保空仓但被 pre-condition 拦截的分钟仍进入历史。

### 修复 4：持久化仓位级 thesis 状态

文件：

```text
qqq_btc/live/fill_audit_writer.py
qqq_btc/live/oms_integration.py
qqq_btc/live/strategy_exit_bridge.py
```

在 OMS symbol state 保存并重新注入：

```text
qqq_btc_entry_spot
qqq_btc_position_spot_closes
```

平仓时清空。

### 修复 5：带进入场前 momentum 历史

开仓时不再初始化空数组，而是携带：

```text
mom_window + 1
```

根入场前 close，避免流式比离线晚触发 thesis。

### 修复 6：Jul13 低置信门控 fail-closed

文件：

```text
qqq_btc/common/entry_decision.py
```

当 early low-gap PUT 门控已启用，但：

```text
spot_ret_15bar / vix_ret_15bar
```

缺失或非有限值时，拒绝交易，避免流式缺列静默放行。

该项不是 `+63.66%` 的主要来源，但属于同一轮 offline/live parity 加固。

### 修复 7：统一对拍工具配置

文件：

```text
qqq_btc/common/signal_collect.py
qqq_btc/tools/compare_stream_replay_day.py
qqq_btc/tools/signal_diff_day.py
qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
```

统一使用：

```python
replace(qcfg.LIVE_REPLAY, edge_q10_floor=-0.2)
```

诊断环境默认关闭：

```text
QQQ_BTC_RULE_PROFILE_SELECTOR=off
QQQ_BTC_TICK_EXITS=off
```

### 修复 8：增加审计字段与短跑能力

文件：

```text
qqq_btc/live/signal_audit_writer.py
qqq_btc/live/fill_audit_writer.py
qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
```

增加审计：

```text
spot_close
vwap_log_return
spot_ret_15bar
vix_ret_15bar
vix_level
entry rails / spot tail / vwap tail
```

增加：

```text
MAX_SESSION_BARS
```

用于先跑到目标入场/退出附近，缩短调试周期。

---

## 8. Jul1 单日修复验证链

修复不是一次猜中，而是逐层缩小问题：

| 验证目录 | 目的 | 发现 |
|---|---|---|
| `july01_v4_bounce_transport_fix/` | 只补 vwap transport | 字段进入后仍未触发，说明时序/状态仍有问题 |
| `july01_v4_bounce_spot_align_fix/` | 对齐 spot 标签 | close 标签改善，但完整持仓状态仍丢失 |
| `july01_v4_bounce_pre_gate_fix/` | 门控前持续记录 | 入场历史完整，但 position state 仍需持久化 |
| `july01_v4_bounce_trace_short/` | 审计 spot/vwap/rails | 确认入场 rails 与历史尾部 |
| `july01_v4_bounce_exit_trace/` | 追踪退出判定 | 定位 `entry_spot`/position closes 未持久化 |
| `july01_v4_bounce_position_state_fix/` | 持久化仓位状态 | thesis 开始工作，但动量历史仍晚一拍 |
| `july01_v4_bounce_momentum_seed_fix/` | 携带入场前历史 | sb80 正确触发 `SPOT_THESIS` |

最终 Jul1 关键交易：

| 指标 | 流式 | 离线 |
|---|---:|---:|
| 入场 | sb77 PUT | sb77 PUT |
| 退出 | sb80 `SPOT_THESIS` | sb80 `SPOT_THESIS` |
| 权利金净收益 | -9.33% | -10.53% |

退出原因和分钟完全一致；净收益差来自成交价/盘口填价。

---

## 9. 测试与验证

新增/修改的关键测试覆盖：

1. `vwap_log_return` / `spot_close` 能进入 live context
2. bounce 输入同分钟去重
3. V0 门控前持续观察 bounce 历史
4. early low-gap PUT 缺 15m 确认时 fail-closed
5. strategy exit bridge 能读取持久化 `entry_spot` 和 spot history
6. signal collect / signal diff 使用统一 LIVE/q10 配置

针对性测试和 Python/bash 语法检查通过。

当时完整 `test_qqq_btc_path.py` 结果：

```text
121 passed
6 failed
```

6 个失败被隔离为既有配置/锚点断言，与本轮 bounce/parity 修改无关。
这意味着关键路径验证通过，但不能表述为“仓库全测试零失败”。

---

## 10. 每日收益对账

| 日期 | 流式笔数 | 流式 acct25 | 离线 acct25 | 日差异 |
|---|---:|---:|---:|---:|
| Jul1 | 2 | -0.1479% | -0.4248% | +0.2769pp |
| Jul2 | 1 | +16.2500% | +16.2397% | +0.0103pp |
| Jul6 | 2 | +7.0712% | +7.0778% | -0.0066pp |
| Jul7 | 3 | +23.3692% | +25.6063% | -2.2371pp |
| Jul8 | 3 | +5.0002% | +4.9200% | +0.0802pp |
| Jul9 | 2 | +1.6521% | +1.0739% | +0.5782pp |
| **Jul1–9** | **13** | **+63.6575%** | **+65.0874%** | **-1.4299pp** |

收益按账户 25% 仓位逐笔复利：

```python
day_return = product(1 + 0.25 * option_net_return) - 1
period_return = product(1 + day_return) - 1
```

---

## 11. 逐笔对账

| 日期 | 腿 | 入/出 sb | 流式退出/净收益 | 离线退出/净收益 | 结论 |
|---|---|---|---|---|---|
| Jul1 | PUT | 77/80 | SPOT_THESIS -9.33% | SPOT_THESIS -10.53% | 分钟和原因对齐，填价差 |
| Jul1 | PUT | 191/223 | STEP +8.95% | STEP +9.07% | 对齐 |
| Jul2 | PUT | 60/115 | MAX_HOLD +65.00% | MAX_HOLD +64.96% | 对齐 |
| Jul6 | PUT | 88/90 | SPOT_THESIS -9.04% | SPOT_THESIS -9.01% | 对齐 |
| Jul6 | PUT | 220/258 | STEP +38.19% | TRAILING +38.18% | 收益/分钟近似，原因族有差异 |
| Jul7 | PUT | 19/74 | MAX_HOLD +89.51% | MAX_HOLD +89.64% | 对齐 |
| Jul7 | CALL | 168/187 | STEP +10.66% | STEP +10.45% | 对齐 |
| Jul7 | CALL | 217/247 | TIME_STOP -7.23% | 无 | **流式额外交易** |
| Jul8 | PUT | 15/48 | TIME_STOP +0.86% | STEP +2.59% | 同腿/时段近似，退出原因与填价不同 |
| Jul8 | PUT | 59/85 | STEP +9.12% | STEP +7.55%（出84） | 1 bar/填价差 |
| Jul8 | PUT | 97/128 | STEP +9.76% | STEP +9.25%（入95） | 入场差2 bar |
| Jul9 | PUT | 43/58 | STEP +11.69% | STEP +9.06% | 填价差 |
| Jul9 | PUT | 179/209 | TIME_STOP -4.94% | TIME_STOP -4.66% | 对齐 |

这不是逐笔完全相等，但关键腿、主要入场窗口和风险退出已高度接近。

---

## 12. 剩余唯一主要分歧：Jul7 sb217 CALL

流式：

```text
call edge       = 0.199514552950859
dynamic threshold = 0.1992171615362167
结果：略高于阈值，开 CALL
```

离线对应 bar：

```text
call edge = 0.1945568174123764
结果：没有形成同一实际交易
```

edge 差：

```text
约 0.00496
```

该差异造成流式额外一笔：

```text
CALL sb217 → sb247 TIME_STOP，net -7.23%
```

这是全周 `-1.43pp` 累计差异的主要来源。

注意：

- `signal_diff_20260707.json` 的 decision replay/live_sim 部分可以完全匹配，
  因为两者读取同一 parquet
- 真正需要看的，是 parquet decision 与实际 dry-run FCS signal：
  `edge_offline≈0.194557` vs `edge_stream≈0.199515`
- 说明残差位于真实 FCS 特征/标签计算的边界值，而不是 replay/live
  共享决策函数本身

---

## 13. 为什么 Jul10 必须单列

`signal_diff_20260710.json` 显示：

```text
offline replay signals = 0
offline decision       = 0
stream dry-run signals = 7
stream actual trades   = 2
```

流式实际交易：

| 腿 | 入/出 sb | 退出 | net |
|---|---|---|---:|
| PUT | 55/70 | STEP_PROTECT | +3.05% |
| PUT | 87/102 | EARLY_STOP | -17.34% |

日账户收益：

```text
-3.6042%
```

由于离线冻结 infer 对 Jul10 没有可比信号，Jul10 不能用于评价
`+65.09%` 是否流式复现。

---

## 14. 产物索引

主目录：

```text
qqq_btc/results/july_w1_v4_stream_final_aligned/
```

### 汇总与配置

| 文件 | 用途 |
|---|---|
| `manifest.json` | checkpoint、数据根、norm、honest 开关 |
| `stream_summary_paired.json` | 15 笔全段汇总 |
| `summary.txt` | 人类可读汇总 |
| `gates_status.json` | Gate 状态；本轮为 skipped/UNGATED |
| `governor_quantile.pkl` | 跨日动态分位状态 |
| `run.log` | 总运行日志 |

### 每日审计

```text
fill_audit_20260701.csv
fill_audit_20260702.csv
fill_audit_20260706.csv
fill_audit_20260707.csv
fill_audit_20260708.csv
fill_audit_20260709.csv
fill_audit_20260710.csv
```

```text
signals_2026-07-01.csv
signals_2026-07-02.csv
signals_2026-07-06.csv
signals_2026-07-07.csv
signals_2026-07-08.csv
signals_2026-07-09.csv
signals_2026-07-10.csv
```

```text
stream_2026-07-01.log
...
stream_2026-07-10.log
```

### 首个分歧诊断

```text
signal_diff_20260707.json
signal_diff_20260707_live_sim.csv
signal_diff_20260707_replay.csv
signal_diff_20260710.json
signal_diff_20260710_live_sim.csv
signal_diff_20260710_replay.csv
```

---

## 15. 可复现性限制

### 15.1 当时工作区未冻结为干净 commit

该轮是在有未提交修改的工作区执行。结果目录的历史 `manifest.json`
记录了主要输入，但没有完整记录当时所有源码 diff。

因此：

- 产物本身可审计
- 命令、输入路径和结果已知
- 但不能只凭当前 HEAD 保证 bit-for-bit 重现

后续正式 run 必须保存：

```text
git commit
git dirty
git diff patch
strategy profile resolved snapshot + sha256
完整 env snapshot
checkpoint sha256
infer/raw/options 文件 hash
```

### 15.2 strategy profile 没有被该历史产物完整冻结

脚本名称和默认 profile 属于 FT56 production 系列，本轮通过环境变量切换到
V4 checkpoint，并覆盖 selector、tick exits 和 q10。历史结果目录没有保存
`strategy_profile.resolved.json` 或 profile SHA。

因此可以确认核心覆盖项和最终成交结果，但不能仅靠现有 manifest 证明
`ReplayConfig` 的每一个字段都与离线 V4 冻结配置完全相同。正式重跑必须保存
resolved profile，并对其 SHA 做离线/流式一致性检查。

### 15.3 当前代码已继续变化

Jul13 VIXY regime、cross-day quarantine 等代码随后继续修改。
直接在当前工作区重跑相同命令，不一定仍得到 `+63.66%`。

### 15.4 `SKIP_GATES=1`

该轮没有重新形成 Gate1/2 正式 PASS 证据。
若要提升为正式 release candidate，必须：

1. 在冻结代码和数据上重新运行 Gate1
2. Gate1 PASS 后运行 Gate2
3. Gate1+2 PASS 后再运行 Gate3
4. 不使用 `SKIP_GATES` 或 `FORCE_GATE3`

### 15.5 模拟成交不等于真实 IBKR

使用：

```text
OMS_MOCK_IBKR=1
BACKTEST_OPT_FILL_SPREAD_FRAC=0.775
```

没有完整模拟真实排队、部分成交、撤单延迟和流动性冲击。

---

## 16. 如何评价这轮结果

### 可以证明

- 之前的大幅离线/流式分叉主要来自可修复的接线与配置不一致
- V4 `open30+bounce+lock45` 能在 FCS/OMS 历史流式栈中大体复现
- `SPOT_THESIS` 和同腿锁机制已进入真实流式路径
- 主要日收益和大交易路径不是仅存在于纯离线引擎

### 不能证明

- 实盘一定获得 `+63.66%`
- Gate1/2 已在该最终目录正式 PASS
- Jul10/Jul13 已与 `+65.09%` 基线对齐
- 真实 IBKR 成交与 mock fill 一致
- 13 笔样本足以证明长期稳健性

---

## 17. 最终结论

```text
可比区间：2026-07-01 至 2026-07-09
模型：V4
策略：open30 + bounce_cut + lock45
流式：+63.6575%，13 笔
离线：+65.0874%，12 笔
差异：-1.4299 个百分点
主要残差：Jul7 sb217 临界 CALL edge
状态：交易链基本对拍成功；Gate1/2 未在本轮执行，产物标记 UNGATED
```

这轮应作为：

```text
V4 July W1 历史流式交易链对拍基准
```

而不是：

```text
当前 FT56+VX+tick 的生产收益预期
```

---

## 18. 已同步到统一 profile 主线（后续入口）

本记录中的历史 env 命令保留作审计，**后续不要再复制该命令重跑**，避免
checkpoint / selector / tick / q10 再次发生静默分叉。

当前唯一 V0 配方：

```text
qqq_btc/CONFIG/strategy_profiles/v4_honest_v0_parity_v1.json
```

该 profile 已用统一离线入口复现：

```text
acct25 = +65.09%
trades = 12
selector = off
cross-day defenses = off
tick exits = off
```

离线产物：

```text
qqq_btc/results/offline_live_aligned/v4_honest_v0_parity_v1_offline/
```

正式三闸门流式入口：

```bash
bash qqq_btc/tools/run_v4_v0_stream_parity.sh
```

该包装器限定 Jul1–9、清理 `QQQ_BTC_*` env override，并禁止
`SKIP_GATES/FORCE_GATE3`。新的正式结果必须同时满足：

```text
same profile SHA
+ Gate1 PASS
+ Gate2 PASS
+ Gate3 PASS
+ Jul1–9 independent summary
```

主评级与闭环方案见：

```text
qqq_btc/docs/replay_version_lineage_and_result_reconciliation.md §15
```
