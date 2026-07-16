# Mag7 开盘阶梯 + 实盘仓位/出场：当前结论（2026-07-17）

> 主证据窗：`2026-05-01` ~ `2026-07-13`（Mag7+GOOGL）  
> 成交：ATM，`fill_frac=0.8`，账户复利 MaxDD  
> **当前因果基线**：[Mag7+GOOGL + peer_min3 + T+30](causal_single_t30_rails_baseline.md)（**+374.8% / -16.5%**）  
> 旧 Mag7-only 基线对照：single + T+30（Jan–Jul +863.6% / -28.6%）  
> 旧方案对照：`m5_circuit` + only_win + `mf_flip`（pre-delay +3375% 已作废）  
> 选约：`open_ladder` OTM5（因果开盘阶梯）vs `day_lock`（前视对照）  
> 详表与消融链：[`jan_jul_replay_versions.md`](jan_jul_replay_versions.md)

> **实盘时钟更正**：股票事实源是 `/mnt/s990/data/raw_1s/stocks` 的秒级数据。
> 秒数据按 `[M, M+1min)` 聚合，分钟 `M` 只有在进入下一分钟后才完成；不再用
> “原始 1 分钟左/右标签”解释因果性。生产 profile 的 60 秒参数只用于预聚合
> 1 分钟研究表的可用时刻适配，原始 1 秒流使用实际 `available_ts`，不能重复延迟。
> mf_flip 也只读取决策时刻已经完成的 stock bar。
> 因此旧 `+3375% / -22%` 不能作为新 profile hash 的上线证据；见
> [`causal_single_t30_rails_baseline.md`](causal_single_t30_rails_baseline.md)。

---

## 1. 当前因果基线（优化起点）

| 项 | 值 |
|---|---|
| **基线 Profile** | [`single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json) |
| Universe / 过滤 | Mag7+GOOGL；`peer_align_min=3`（Mag7 mf10 同向） |
| Scheme / 出场 | `single` + T+30 + TP1.6/SL0.4（无 mf_flip、无复入） |
| 因果 May–Jul | **+374.8% / MaxDD -16.5%**（53 笔） |
| 文档 | [`causal_single_t30_rails_baseline.md`](causal_single_t30_rails_baseline.md) |

### 1b. 旧临时包（仅对照，非基线）

| 项 | 值 |
|---|---|
| **Legacy Profile** | [`m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1`](../CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json) |
| 复入 | `reentry_mode=only_win` |
| 仓位 | `position_sizing=concurrent`，`position_frac=0.20`（独处 20%；并发再开一笔 10%；最多 2 腿） |
| 出场 | `exit_mode=mf_flip`（mf10 翻向提前平，60s grace；仍保留 TP1.6 / SL0.4 / T+30） |
| 选约 | `contract_mode=open_ladder`，`ladder_otm_rungs=5` |
| 稳妥备选 | 同栈 `position_frac=0.15`（MaxDD 更低） |

```bash
# 临时生产：offline vs stream 对拍
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json \
  --scheme m5_circuit \
  --tag parity_open_ladder_otm5_mf_flip_p20_jan_jul
```

A/B 产物：`results/open_ladder_ab_1s_otm5_ow_conc_p20_mf_flip_jan_jul/`  
旧对拍产物：`results/parity_open_ladder_otm5_mf_flip_p20_jan_jul/`（旧时钟口径
247 笔一致；加入 60 秒 bar 可用性延迟后已失效，需重跑）

---

## 2. 1s 主结果表（day_lock vs open_ladder）

| 配置 | day 收益 | day MaxDD | lad 收益 | lad MaxDD | 结果目录 |
|---|---:|---:|---:|---:|---|
| only_win + topk p25（旧生产均分 12.5%） | +1471% | -22% | +718% | -26% | `open_ladder_ab_1s_otm5_jan_jul` |
| only_win + conc p25 + rails | +6849% | -33% | +1427% | **-58%** | `..._onlywin_conc25_...` |
| only_win + conc p15 + rails | +1468% | -23% | +714% | -33% | `..._ow_conc_p15_rails_...` |
| **only_win + conc p15 + mf_flip** | **+2007%** | **-18%** | **+1445%** | **-17%** | `..._ow_conc_p15_mf_flip_...` |
| **only_win + conc p20 + mf_flip** | **+5373%** | **-23%** | **+3375%** | **-22%** | `..._ow_conc_p20_mf_flip_...` |
| cooldown + conc p25（不可上线） | +2143% | **-52%** | +2732% | **-61%** | `..._cooldown_conc25_...` |

要点：

- **-58% / -61% 不可上线**；根因是满仓 25% + 允许亏后再开（或无窗口提前平）。
- **mf_flip**（同款滑动窗口：mf10 翻向）把 ladder 回撤压到约 **-17%～-22%**，且收益高于同仓位 rails。
- **p20** = `position_frac=0.20`（独处 20% / 并发 10%）。相对 p15 是仓位放大；相对同 p20 无 mf_flip（约 +994% / -42%）则是窗口出场带来的主增益。

出场构成（p15 mf_flip ladder 约）：`MF_FLIP` ~80%，`TP` ~16%，几乎无硬 SL。

---

## 3. 机制结论（短）

| 模块 | 结论 |
|---|---|
| 开盘阶梯 OTM5 | 因果可实盘；day_iv 已贴近日锁；1s 总收益差主要来自路径/复利，非缺 quote |
| `cooldown_only` | 集合对齐好，适合对拍；**不要做生产默认** |
| `concurrent` 仓位 | 实盘不知后面几笔：独处满袖套；最多再开 1 腿；禁止 25%+25%=50% |
| 完整反向 Rule-A 出场 | 30m 持仓内几乎触发不到 |
| **mf_flip 出场** | 有效：砍硬亏、降 MaxDD、抬总收益 |

---

## 4. 相关代码

| 路径 | 作用 |
|---|---|
| `maga7/common/open_lock.py` / `entry_contract.py` | 开盘锁 / 阶梯选约 |
| `maga7/common/reentry.py` | `reentry_mode` |
| `maga7/common/position_size.py` | concurrent / max 2 腿 |
| `maga7/common/replay.py` | `exit_mode=mf_flip` |
| `maga7/tools/run_open_ladder_ab.py` | A/B CLI（`--exit-mode` / `--position-frac`） |
| `maga7/tools/prepare_open_lock_quotes.py` | 开盘锁 1s 下载流水线 |

---

## 5. 生产状态（2026-07-16）

**临时生产**：`m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1`（only_win + conc p20 + mf_flip + open_ladder OTM5）。

旧稳定口径 `m5c_qqq_onlywin_stable_v1`（day_lock + topk）仍保留对照。上线前必须
以带 `bar_availability_delay_seconds=60` 的新 Offline / Stream / Redis 结果重新冻结。

当前统一的数据流、时钟、锁约、OMS、恢复和 G0–G6 门禁见
[`current_architecture.md`](current_architecture.md)。
