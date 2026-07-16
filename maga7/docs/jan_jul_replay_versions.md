# Mag7 Jan–Jul 回测版本与消融记录

> 窗口：`2026-01-02` ~ `2026-07-13`  
> 成交：ATM，`fill_frac=0.8`，账户复利 DD  
> 锁约：prefer 0DTE，允许 0/1/2（trading-DTE）；实际成交约半数为 0DTE  
> 已知空洞：正股约 `2026-03-19` ~ `2026-04-30`（交易日偏少）  
> 更新：2026-07-16  
> **当前因果包结论摘要**：[`open_ladder_live_package_results.md`](open_ladder_live_package_results.md)

---

## 1. Profile 管理

| role | profile_id | JSON | 正式结果目录 |
|---|---|---|---|
| **生产推荐 / 最平稳** | `m5c_qqq_onlywin_stable_v1` | [`CONFIG/strategy_profiles/m5c_qqq_onlywin_stable_v1.json`](../CONFIG/strategy_profiles/m5c_qqq_onlywin_stable_v1.json) | `results/jan_jul_m5c_qqq_onlywin` |
| **收益最高** | `m5c_qqq_align_maxret_v1` | [`CONFIG/strategy_profiles/m5c_qqq_align_maxret_v1.json`](../CONFIG/strategy_profiles/m5c_qqq_align_maxret_v1.json) | `results/jan_jul_m5c_qqq_maxret` |
| **研究推荐：开盘阶梯+mf_flip** | `m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p15_v1` | [`CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p15_v1.json`](../CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p15_v1.json) | `results/open_ladder_ab_1s_otm5_ow_conc_p15_mf_flip_jan_jul` |
| research mf=6 | `m5c_qqq_onlywin_mf6_research_v1` | [`CONFIG/strategy_profiles/m5c_qqq_onlywin_mf6_research_v1.json`](../CONFIG/strategy_profiles/m5c_qqq_onlywin_mf6_research_v1.json) | `results/jan_jul_m5c_qqq_onlywin_mf6` |
| research 信号时 ATM | `m5c_qqq_onlywin_signal_atm_research_v1` | [`CONFIG/strategy_profiles/m5c_qqq_onlywin_signal_atm_research_v1.json`](../CONFIG/strategy_profiles/m5c_qqq_onlywin_signal_atm_research_v1.json) | `results/signal_atm_ab_jan_jul/` |
| 仅赢复入 | `m5c_only_win_v1` | [`CONFIG/strategy_profiles/m5c_only_win_v1.json`](../CONFIG/strategy_profiles/m5c_only_win_v1.json) | 消融 `only_win` |
| 基线 | `m5c_baseline_v1` | [`CONFIG/strategy_profiles/m5c_baseline_v1.json`](../CONFIG/strategy_profiles/m5c_baseline_v1.json) | `results/jan_jul_m5_circuit` |

目录索引：[`CONFIG/strategy_profiles/catalog.json`](../CONFIG/strategy_profiles/catalog.json)

默认入口 [`CONFIG/mf10_top2_v1.json`](../CONFIG/mf10_top2_v1.json) **对齐** `m5c_qqq_onlywin_stable_v1`（`only_reenter_after_win` + 仅 QQQ `from_prev` 对齐）。

```bash
# 生产推荐
python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_stable_v1.json \
  --scheme m5_circuit --tag jan_jul_m5c_qqq_onlywin

# 收益最高（QQQ align，复入不限）
python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_align_maxret_v1.json \
  --scheme m5_circuit --tag jan_jul_m5c_qqq_maxret
```

---

## 2. 正式全样本结果（按版本）

### 2.1 Jan–Jun（旧基线，无 regime）

| Scheme | 收益 | MaxDD | 交易 | 日胜率 | 结果目录 |
|---|---:|---:|---:|---:|---|
| single | +501% | -26.5% | 154 | 51.5% | `results/jan_jun_single` |
| m5_circuit | +357% | -37.6% | 412 | 45.4% | `results/jan_jun_m5_circuit` |

### 2.2 Jan–Jul（无 regime 基线）

| Scheme | 收益 | MaxDD | 交易 | 日胜率 | E[r] | 结果目录 |
|---|---:|---:|---:|---:|---:|---|
| single | +867% | -26.5% | 168 | 53.3% | +11.7% | `results/jan_jul_single` |
| m5_circuit | +1057% | -37.6% | 456 | 46.7% | +5.1% | `results/jan_jul_m5_circuit` |

### 2.3 Jan–Jul 命名 profile（正式）

| 配置 | 收益 | MaxDD | 交易 | regime 拦截 | 结果目录 |
|---|---:|---:|---:|---:|---|
| **m5c_qqq_onlywin_stable_v1**（推荐） | **+1471%** | **-22.4%** | 261 | 521 | `results/jan_jul_m5c_qqq_onlywin` |
| **m5c_qqq_align_maxret_v1**（收益最高） | **+1998%** | **-26.5%** | 403 | 521 | `results/jan_jul_m5c_qqq_maxret` |
| m5c_baseline_v1 | +1057% | -37.6% | 456 | 0 | `results/jan_jul_m5_circuit` |

稳定版开关：`only_reenter_after_win=true` + 仅 QQQ align。  
最高收益版：复入不限 + 仅 QQQ align（7/7–9 仍约 -14%，不如稳定版）。

---

## 3. 消融（m5_circuit，1/2–7/13）

数据来源：`results/regime_ablation_jan_jul/scoreboard.csv`  
命令：`python -m maga7.tools.run_regime_ablation --scheme m5_circuit`

| 方案 | 收益 | MaxDD | 拦截次数 | 7月 | 7/7–9 | 7/8 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +1057% | -37.6% | 0 | +153% | -14.1% | -6.8% |
| only_win | +1707% | -27.9% | 0 | +134% | -6.1% | -0.2% |
| **regime_qqq_only** | **+1998%** | **-26.5%** | 521 | +111% | -14.1% | -6.8% |
| regime_put_only | +678% | -32.7% | 317 | +105% | -7.1% | +0.9% |
| regime_vix_only | +765% | -35.1% | 386 | +151% | -15.0% | -7.7% |
| regime_full（三闸全开） | +365% | -40.0% | 1127 | +69% | -8.0% | -0.1% |
| regime+only_win（三闸+only_win） | +238% | -41.6% | 1023 | +35% | -7.0% | -1.2% |
| single_no_regime（对照） | +867% | -26.5% | 0 | +61% | -3.5% | +2.5% |

### 读法

- **收益最高**：`regime_qqq_only` → profile `m5c_qqq_align_maxret_v1`（+1998% / DD -26.5%）。逆大盘票过滤有效，但 **不解** 7/7–9 复入连亏。
- **控 7/8 复入**：`only_win` 最直接（7/8 ≈ 0）。
- **最平稳（高收益里 MaxDD 最好）**：正式跑的 **QQQ align + only_win** → `m5c_qqq_onlywin_stable_v1`（+1471% / **-22.4%**）。注意：消融表里的 `regime+only_win` 是 **三闸全开**，过严，不是生产推荐。
- **三闸全开过猛**（拦 1000+），勿默认开启 VIX/put。

---

## 3.1 Regime v2（连亏诊断，mf=10 不变）

背景：稳定版仍有多段 **≥2/≥3 连亏日**。逐笔看，多数亏损首笔 **已经过** `qqq_from_prev` 对齐，说明「价格相对昨收」不够；部分日 `vix_reversal` 很高或 `qqq_mf10` 与个股方向相反。

新增能力（`maga7/common/regime.py` + replay）：

| 开关 | 含义 |
|---|---|
| `qqq_mf10_align` | Mag7 UP/DN 需与 QQQ `mf10` 同号 |
| `vix_reversal_max` | 对齐 QQQ：30m VIXY 反转次数过高则禁开 |
| `day_loss_streak_halt` | 连续亏损日 ≥N 后，**次日整日停开**（平日重置 streak） |

消融基准 = `m5c_qqq_onlywin_stable`；产物 `results/regime_v2_ablation_stable/`：

| 方案 | 收益 | MaxDD | ≥2连亏 | ≥3连亏 | 日停开次数 |
|---|---:|---:|---:|---:|---:|
| stable_base | +1471% | -22.4% | 15 | 6 | 0 |
| vix_rev6 | +595% | -28.9% | 14 | 6 | 0 |
| mf10_align | +980% | -21.2% | 12 | 4 | 0 |
| **day_halt2** | +649% | -30.8% | 16 | **0** | 16 |
| day_halt1 | +636% | -23.1% | **0** | **0** | 30 |
| day_halt3 | +1130% | -21.2% | 15 | 6 | 6 |
| mf10+halt2 | +429% | -25.2% | 13 | **0** | 13 |

### 结论（v2）

- **条级 QQQ VIX 反转** 对「日历连亏」帮助有限，还明显伤收益。
- **`qqq_mf10_align`** 略降连亏、DD 稍好，但收益从 +1471% → +980%。
- **真正砍掉 ≥3 连亏** 的是跨日 `day_loss_streak_halt=2`（或 =1 连 2 连亏也砍光）；代价是收益腰斩级。
- **生产默认暂不改**（仍 stable：仅 `from_prev` + only_win）。若优先「不许三连亏」，可开 research：`day_loss_streak_halt=2`。

---

## 4. 规则敏感性（single，同窗口）

`results/sensitivity_jan_jul_single/scoreboard.csv`  
基线 Rule-A：`streak=8 / from_prev=2% / vol_z=1` → +867% / DD -26.5%（中上，非尖峰）。

单维结论：`from_prev=2%` 优于 1.5%；`vol_z=0.5` 明显变差；网格尖峰（如 streak=6+vz=1.5）视为同样本过拟合，不改默认信号。

### mf_window 消融（稳定栈）

`results/mf_window_ablation_stable/`：8/12/15 均差于 10；**6 在同样本更好**。

| Profile | mf | 收益 | MaxDD | 角色 |
|---|---:|---:|---:|---|
| `m5c_qqq_onlywin_stable_v1` | **10** | +1471% | -22.4% | **生产** |
| `m5c_qqq_onlywin_mf6_research_v1` | 6 | +1788% | -20.9% | research only |

正式目录：`results/jan_jul_m5c_qqq_onlywin_mf6`。勿直接替换定稿，需 OOS。

### 信号时 ATM 重选（research A/B）

问题：日级/开盘预锁 → 信号时 K 可能已偏离 ATM。  
做法：**同 TopK / 同稳定闸门**，只改选约时钟 — `contract_mode=signal_atm`（day_iv，≤`sig_ts` 因果快照，prefer 0/1/2 DTE）。

公平对比用同一报价钟 `quote_source=day_iv`（close ±1% 半价差合成 bid/ask + fill 0.8）。**绝对值远低于 1s 生产 KPI**，只看相对差。

```bash
python -m maga7.tools.run_signal_atm_ab --scheme m5_circuit --quote-source day_iv \
  --tag signal_atm_ab_jan_jul --also-1s-baseline
```

| Arm（day_iv 钟） | 收益 | MaxDD | 交易 | E[r] | 与日锁同约 |
|---|---:|---:|---:|---:|---:|
| `day_lock` | +35% | -15.3% | 191 | +1.4% | 100% |
| **`signal_atm`** | **+122%** | -20.5% | 237 | **+3.0%** | ~22% |
| 参考：`day_lock` 1s 生产 | +1471% | -22.4% | 261 | +9.3% | — |

配对 187 笔：约 **73% 合约不同**；不同时 `E[r_sig]-E[r_lock]≈+2.9pp`，胜率 26%→49%。  
Profile：`m5c_qqq_onlywin_signal_atm_research_v1`。产物：`results/signal_atm_ab_jan_jul/`。

**结论**：机制上信号时重选更好（尤其合约已变时）；**暂不替换生产**——缺信号 ATM 的 1s quote。

补 1s 路径（已开）：

```bash
# 1) 按稳定栈事件导出信号 ATM 锁约表（含 day_lock companion）
python -m maga7.tools.export_signal_atm_lock_map --scheme m5_circuit

# 2) step2 只拉表内合约（非 S3 全链）
bash maga7/tools/prepare_signal_atm_quotes.sh
# map → ~/train_data/locked_targets_map_maga7_signal_atm_jan_jul.parquet
# out → /mnt/s990/data/raw_1s/maga7_mf10_signal_atm
```

下完后用 `quote_source=1s` 对拍 `day_lock` 生产目录。

### 1s 正式 A/B（已完成）

产物：`results/signal_atm_ab_1s_jan_jul/`（fill=0.8，稳定栈）

| Arm | 收益 | MaxDD | 交易 | E[r] | 胜率 |
|---|---:|---:|---:|---:|---:|
| **`day_lock` 1s（生产）** | **+1471%** | -22.4% | 261 | **+9.3%** | 52% |
| `signal_atm` 1s | +279% | **-18.0%** | 257 | +4.6% | 51% |

配对 242 笔：约 74% 合约不同；不同时 `E[r_sig]-E[r_lock]≈**-7.7pp**`。  
→ **生产保持日锁**；day_iv 研究里 signal_atm 的相对优势在真实 1s bid/ask 上**不成立**（DD 略好但收益/期望明显更差）。

---

### 开盘因果锁 + 明显 OTM 禁 0DTE（实盘口径）

离线 old 日锁（全天 `|delta|`）有前视，**不能直接上实盘**。实盘/可交易研究改为：

1. **开盘窗锁**（09:30；day_iv + option_1m 早盘成交补 0DTE）：同时锁 trading-DTE 0/1/2 的 ATM/OTM  
2. 信号触发时优先 0DTE ATM；若相对现价 **明显 OTM**（默认 ≥1%）：**禁止 0DTE**，改用 1DTE → 2DTE  

```bash
# 完整流水线说明：maga7/docs/open_lock_quote_pipeline.md
python -m maga7.tools.prepare_open_lock_quotes --step all
python -m maga7.tools.prepare_open_lock_quotes --step all --add-symbols GOOGL

python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json \
  --scheme m5_circuit --tag jan_jul_open_lock_clear_otm
```

## 开盘阶梯 ATM+2×OTM（逼近前视）

Rule-A 信号时开盘 ATM 多数已偏 ITM；当前 ATM 通常在开盘 **OTM 一侧**。  
因此锁 **ATM + OTM1 + OTM2**（**严格递增、禁止与 ATM 重复**），信号时选离现价最近的一档（`contract_mode=open_ladder`）。

```bash
# 导出阶梯锁约表
python -m maga7.tools.export_open_lock_map \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm2otm_v1.json \
  --otm-rungs 2

# day_iv A/B：day_lock vs open_lock vs open_ladder
python -m maga7.tools.run_open_ladder_ab --quote-source day_iv --scheme m5_circuit \
  --tag open_ladder_ab_dayiv_strict_jan_jul
```

day_iv 严格档结果（`results/open_ladder_ab_dayiv_strict_jan_jul/`）：

| Arm | 收益 | 交易 |
|---|---:|---:|
| day_lock | +35.1% | 191 |
| open_lock | -0.95% | 133 |
| **open_ladder 严格** | **+29.3%** | 159 |

1s 严格档（`results/open_ladder_ab_1s_strict_jan_jul/`，覆盖 ~100%）：

| Arm | 收益 | 交易 |
|---|---:|---:|
| day_lock | +1471% | 261 |
| open_lock | +126% | 257 |
| **open_ladder 严格** | **+412%** | 255 |

#### 宽度蒸馏 → ATM+OTM1..OTM4

前视日锁相对开盘 ATM 的 OTM 距离：p75≈$10 / ~4 档；**OTM≤4 覆盖 ~77%** 日锁合约。  
超出 OTM2 的日锁单 E[r] 明显更高（1s 上 ~0.145 vs 0.044）→ 加宽阶梯。

```bash
python -m maga7.tools.export_open_lock_map \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm4otm_v1.json --otm-rungs 4

python -m maga7.tools.run_open_ladder_ab \
  --ladder-profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm4otm_v1.json \
  --quote-source day_iv --tag open_ladder_ab_dayiv_otm4_jan_jul

# 1s 补数
python -m maga7.tools.prepare_open_lock_quotes \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm4otm_v1.json --step all
```

day_iv（`results/open_ladder_ab_dayiv_otm4_jan_jul/`）：day_lock **+35.1%** vs open_ladder OTM4 **+34.1%**（已基本贴齐）。

1s（覆盖 99.5%，`results/open_ladder_ab_1s_otm4_jan_jul/`）：

| Arm | 收益 | 交易 | E[r] |
|---|---:|---:|---:|
| day_lock | +1471% | 261 | 9.3% |
| open_lock | +126% | 257 | 2.8% |
| open_ladder OTM2 | +412% | 255 | 5.8% |
| **open_ladder OTM4** | **+665%** | 263 | **7.0%** |

#### only_day / only_lad 说明（不是缺数）

m5_circuit 下 `only_day`/`only_lad` 主要是 **复入路径分叉**：前几笔盈亏不同 → 后续 `n_in_day` 集合不一致。  
首笔（`n_in_day==1`）高度重叠；两边都有 1s quote。A/B 会写 `set_alignment.json`。

#### OTM5（覆盖 ~88%）

```bash
python -m maga7.tools.export_open_lock_map \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_v1.json --otm-rungs 5
python -m maga7.tools.run_open_ladder_ab \
  --ladder-profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_v1.json \
  --quote-source day_iv --tag open_ladder_ab_dayiv_otm5_jan_jul
python -m maga7.tools.prepare_open_lock_quotes \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_v1.json --step all
```

day_iv OTM5：`results/open_ladder_ab_dayiv_otm5_jan_jul/`（+32.9% vs day_lock +35.1%；与 OTM4 同级）。

1s OTM5（覆盖 98.7%，`results/open_ladder_ab_1s_otm5_jan_jul/`）：

| Arm | 收益 | 交易 | E[r] |
|---|---:|---:|---:|
| day_lock | +1471% | 261 | 9.3% |
| open_lock | +126% | 257 | 2.8% |
| open_ladder OTM2 | +412% | 255 | 5.8% |
| open_ladder OTM4 | +665% | 263 | 7.0% |
| **open_ladder OTM5** | **+718%** | 267 | **7.2%** |

#### 投影选约 α/H 网格（冻结进场、只换合约）

`python -m maga7.tools.run_ladder_project_grid`：在 OTM5 阶梯上扫 α 与涨速外推 H。

冻结日锁进场时刻的 1s 结果（`results/ladder_project_grid_1s_otm5_from_daylock/`）：

| 规则 | E[r] | 胜率 |
|---|---:|---:|
| nearest_spot | 10.39% | 54.8% |
| 最优 α0.25+speed_H10 | 10.56% | 52.9% |
| day_lock 合约 | 9.73% | 52.9% |
| α≥1 外推 | 明显变差 | — |

→ **投影相对 nearest 几乎无提升**；大 α 有害。日锁合约在同一进场时刻上甚至略逊于 nearest（scalp 目标 ≠ 日内 |delta|）。

#### `reentry_mode=cooldown_only`（解耦进场时钟）

`only_win` 下不同合约的早期盈亏会分叉后续复入集合。改为冷却可再进（不要求上一笔赢），保留 `m5_circuit`：

```bash
python -m maga7.tools.run_open_ladder_ab \
  --ladder-profile maga7/CONFIG/strategy_profiles/m5c_qqq_cooldown_open_ladder_atm5otm_v1.json \
  --quote-source 1s --scheme m5_circuit --reentry-mode cooldown_only \
  --tag open_ladder_ab_1s_otm5_cooldown_jan_jul
```

1s 结果（`results/open_ladder_ab_1s_otm5_cooldown_jan_jul/`）：

| Arm | 收益 | 交易 | E[r] | MaxDD |
|---|---:|---:|---:|---:|
| day_lock | +1998% | 403 | 6.9% | -26.5% |
| open_lock | +105% | 397 | 1.7% | -24.9% |
| **open_ladder OTM5** | **+1056%** | **403** | **5.7%** | -39.5% |

集合对齐（相对 only_win OTM5 的 242/261）：

| 指标 | only_win OTM5 | **cooldown_only** |
|---|---:|---:|
| n 两边 | 261 / 267 | **403 / 403** |
| key 交集 | 242 | **384** |
| only_day / only_lad | 19 / 25 | **19 / 19** |
| 配对 E[r] day / lad | — | **7.14% / 6.88%**（差 0.26pp） |

要点：配对口径下 ladder 已贴近日锁；总收益差主要来自复利与残差 only_lad 毒笔（19 笔 E[r]≈-17.7%）。Profile：`m5c_qqq_cooldown_open_ladder_atm5otm_v1`；共享解析：`maga7/common/reentry.py`。**生产默认仍是 only_win**（控回撤）；cooldown_only 用于锁约对拍 / 研究。


#### `position_sizing=concurrent`（实盘仓位：独处 25%）

旧默认 `topk`：一律 `0.25/top_k=12.5%`（先验预留 Top2 槽位）。实盘不知道后面还有没有第二笔，改为：

- 进场时若无其他持仓仍开着 → **整笔 25%**
- 若已有 1 个其他标的未平 → 再开一笔 **12.5%**（`position_frac / max_concurrent`）
- 若已有 2 个未平 → **拒开**（`max_concurrent_positions=2`，禁止第三腿，避免两笔满仓叠成 50%）
- 瞬时名义可到 25%+12.5%=37.5%，但不会出现 25%+25%=50%

```bash
python -m maga7.tools.run_open_ladder_ab \
  --ladder-profile maga7/CONFIG/strategy_profiles/m5c_qqq_cooldown_open_ladder_atm5otm_v1.json \
  --quote-source 1s --scheme m5_circuit --reentry-mode cooldown_only \
  --position-sizing concurrent \
  --tag open_ladder_ab_1s_otm5_cooldown_conc25_jan_jul
```

1s cooldown + concurrent（`results/open_ladder_ab_1s_otm5_cooldown_conc25_jan_jul/`）：

| Arm | 收益 | 交易 | MaxDD | 满仓25%笔数 |
|---|---:|---:|---:|---:|
| day_lock | +2143% | 345 | -51.6% | 251/345 |
| **open_ladder OTM5** | **+2732%** | 346 | -60.7% | 252/346 |

相对旧 `topk` 均分：仓位更大 → 收益与回撤都放大；circuit 更早触发故笔数略降。此口径下 ladder 总收益可超过 day_lock（路径/复利敏感，MaxDD 更差）。共享：`maga7/common/position_size.py`。


#### 回撤控制（concurrent 25% → MaxDD -52%）

根因：`cooldown_only` + 独处满仓 25% 时，2026-02-13→03-12 连亏窗口按约 2× 旧 topk 杠杆复利（peak≈269 → trough≈149）。

1s 消融（`results/dd_control_ablation_1s_jan_jul/`）要点（day_lock = 旧前视锁；open_ladder OTM5 = 因果新模式）：

| 配置 | day 收益 | day MaxDD | **lad 收益** | **lad MaxDD** | 备注 |
|---|---:|---:|---:|---:|---|
| cooldown + conc **p25** | +2143% | **-52%** | **+2732%** | **-61%** | 不可上线 |
| only_win + conc p25 | +6849% | -33% | **+1427%** | **-58%** | lad 仍不可忍 |
| only_win + conc **p20** | +4026% | **-27%** | **+994%** | -42% | day 可接受，lad 偏高 |
| only_win + conc **p15** | +1468% | **-23%** | **+714%** | **-33%** | concurrent 下较稳 |
| only_win + **topk** p25 | +1471% | **-22%** | **+718%** | **-26%** | 旧生产口径，DD 最好 |
| cooldown + conc p15 | +1301% | -28% | **+750%** | -50% | lad DD 仍差 |
| day_loss_streak_halt=2 | （收益↓） | 更差 | （收益↓） | 更差 | 本样本无效甚至有害 |

**建议（实盘）**

1. **必须保留 `only_win`**（不要用 cooldown_only 做生产）。
2. 若坚持 concurrent（独处满仓）：把 `position_frac` 降到 **0.15～0.20**（独处 15–20%，并发 7.5–10%），不要 25%。
3. 若优先控回撤：继续 **topk + only_win + 0.25**（每笔 12.5%），MaxDD 约 -22%/-26%。
4. `day_loss_streak_halt` 本窗口未改善 DD，暂不作为主杠杆。


#### `exit_mode=mf_flip`（滑动窗口失效平仓）

进场用 mf10 窗口确认趋势；持仓中若 **mf10 翻向**（UP 持仓且 mf10<0，对称 DN），在 TP/SL/T+30 之前提前平仓（默认入场后 60s grace）。

```bash
python -m maga7.tools.run_open_ladder_ab \
  --ladder-profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p15_v1.json \
  --quote-source 1s --scheme m5_circuit \
  --reentry-mode only_win --position-sizing concurrent --position-frac 0.15 --exit-mode mf_flip \
  --tag open_ladder_ab_1s_otm5_ow_conc_p15_mf_flip_jan_jul
```

1s only_win + concurrent（day_lock vs open_ladder OTM5）：

| 配置 | day 收益 | day MaxDD | lad 收益 | lad MaxDD |
|---|---:|---:|---:|---:|
| rails + conc p15 | +1468% | -23% | +714% | -33% |
| **mf_flip + conc p15** | **+2007%** | **-18%** | **+1445%** | **-17%** |
| **mf_flip + conc p20** | **+5373%** | **-23%** | **+3375%** | **-22%** |
| rails + conc p25（对照） | +6849% | -33% | +1427% | -58% |

→ **mf_flip 把 ladder 回撤从 -33%/-58% 压到约 -17%～-22%**，同时收益上升。推荐研究默认：`only_win + concurrent p15 + mf_flip`（profile `m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p15_v1`）。完整反向 Rule-A 出场无效；mf 翻向才是同窗口的可用平仓信号。

---


---

## 5. DTE 说明

锁约 `prefer_dte=0`，`allowed=[0,1,2]`。Jan–Jul single 日历 DTE 约：0DTE 54% / 1DTE 35% / 2DTE 11%。**多数为 0DTE**，不是「仅 1DTE+」。

---

## 6. 相关代码

| 模块 | 路径 |
|---|---|
| 开盘锁 + 1s 流水线 | `maga7/tools/prepare_open_lock_quotes.py` |
| 流水线说明 | `maga7/docs/open_lock_quote_pipeline.md` |
| 复入策略 | `maga7/common/reentry.py`（`reentry_mode`） |
| Regime 闸门 | `maga7/common/regime.py` |
| 信号时选约 | `maga7/common/contract_select.py` |
| Offline replay | `maga7/common/replay.py` |
| 消融 CLI | `maga7/tools/run_regime_ablation.py` |
| 信号 ATM A/B | `maga7/tools/run_signal_atm_ab.py` |
| 开盘阶梯 A/B | `maga7/tools/run_open_ladder_ab.py` |
| 敏感性 CLI | `maga7/tools/run_sensitivity_grid.py` |
| Scanner→OMS | `maga7/docs/scanner_oms_integration.md` |
