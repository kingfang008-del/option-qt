# AM v2 新路径（架构重开）

> 开工：2026-08-08  
> 背景：[`am_architecture_reassessment.md`](am_architecture_reassessment.md)  
> 原则：**新路径独立晋升；旧 AM 卫星只复用零件，不继承叙事。**

---

## 北星（锁定）

**北星 A — 可执行卫星**

```text
一条 AM 臂 · 一套 mark · 一个目标
mark 晋升唯一口径 = quote FillSpec（与 live OMS 同构）
trade-last          = 诊断 / 上界，不得单独晋级
density KPI         = 暂不设「几十笔/日」；先保证 quote 下均笔>0
CORE                = 继续 10:30+ 主账；本路径只负责 09:30–11:30 卫星
```

旧路径（Pulse KEEP、pocket retreat、launch multi、high_freq…）**冻结叙事**，代码可复用。

---

## 复用清单（零件）

| 零件 | 来源 | 用途 |
|------|------|------|
| Mag7 符号 / open-lock | peer3 profile | 合约 |
| `prepare_day_arrays` / `features_at` | `session_1s_features` | 正股 1s |
| `launch_slope` / `am_pulse_scout` | common | 候选信号（Step2+） |
| `FillSpec` + `simulate_quote_tpsl` | `fills` / `option_quote_tpsl` | **唯一晋级定价** |
| `simulate_trade_tpsl` | `option_trade_tpsl` | 诊断对照 |
| dual window | may_jul09 / jul10_23 | 验收 |
| scanner drain 模式 | `drain_am_pulse` | Step4+ 接线模板 |

---

## 分步门控（一步一测）

| Step | 内容 | PASS 标准 | 失败动作 |
|------|------|-----------|----------|
| **0** | 章程 + profile + 冻结旧 AM 叙事 | 本文档 + profile 落地 | — |
| **1** | Quote 可执行基线（时钟采样） | 报告开盘 ATM quote 覆盖/lag/spread；双窗有数 | 先修数据/锁约，不进信号 |
| **2** | 信号候选 bakeoff（≤3 个） | **quote 口径**双窗 econ；trade-last 仅附表 | 换信号，不放宽 mark |
| **3** | 退出固定一档 | 在 Step2 赢家上 TP/SL 网格；capture 次要 | 不叠 ride/scaleout |
| **4** | Shadow 接线 | scanner drain + `execute_mode=shadow` | 回 Step2 |
| **5** | Dry / paper | OMS dry 过关才谈 live | 停 |

任意 Step FAIL → **不进入下一步**，不平行开 densify/capture20。

---

## Profile / 产物

- Profile：`maga7/CONFIG/strategy_profiles/am_v2_executable_path_v1.json`
- Catalog：`am_v2_executable_path`（RESEARCH_ONLY，`enabled_on_spine=false`）
- Step1：`run_am_v2_step1_quote_baseline` → `…/research_am_v2_step1_quote_baseline/`
- Step2 / 2b：`run_am_v2_step2[_b]_signal_bakeoff` → `…/research_am_v2_step2[b]_signal_bakeoff/`
- Step3：`run_am_v2_step3_exit_grid` → `…/research_am_v2_step3_exit_grid/`
- Step4：`Mag7Scanner.drain_am_v2` · peer3 `am_v2` · `tests/test_am_v2_live.py`

---

## Step1 结果（PASS）

产物：`/mnt/s990/data/maga7/results/research_am_v2_step1_quote_baseline/`  
门：`lag≤5s · spread≤15% · mid≥0.05`

| 时段 | any_quote | gate_ok | lag p50 |
|------|----------:|--------:|--------:|
| **09:30–10:00** | **99%** | **26%** | **~600s** |
| 10:00–10:30 | 99% | **96%** | ~0s |
| 10:30–11:00 | 99% | **95%** | ~0s |
| 11:00–11:30 | 99% | **93%** | ~0s |
| ALL | 99% | 77% | — |

读法：开盘半小时可执行性差（lag 债）；**10:00 后** gate_ok>90%。

---

## Step2 结果（FAIL）

产物：`/mnt/s990/data/maga7/results/research_am_v2_step2_signal_bakeoff/`  
固定退出：TP15/SL20/h900 · FillSpec 0.75 · 同 Step1 门  
候选 3 个；**promote=NONE**（无一 quote 双窗 econ）

| signal | fill% | quote_n | tpd | mean | disc | blind | econ |
|--------|------:|--------:|----:|-----:|-----:|------:|:----:|
| pulse_fo08_causal_full | 35% | 147 | 2.5 | −2.3% | −23% | −2.7% | no |
| pulse_fo08_causal_post10 | **93%** | 324 | 5.4 | −2.7% | −36% | −11% | no |
| launch_s3_r002_cd120 | 22% | 452 | 7.5 | +0.9% | **+42%** | −13% | no |

读法：
- post10 pulse：时钟可用，edge 仍负 → **信号问题，不是门问题**。  
- launch：disc 正、blind 负 → 不过双窗；trade-last 也不能抬晋级。  
- **不进 Step3**；不放宽 mark。

**下一步 = Step2b**：换信号族/窗（仍 quote 唯一晋级）。

---

## Step2b 结果（PASS）

产物：`/mnt/s990/data/maga7/results/research_am_v2_step2b_signal_bakeoff/`  
同 Step2 退出/门；候选：launch r003 · launch r002/cd300/post10 · pulse fo12 post10

| signal | fill% | quote_n | tpd | mean | disc | blind | econ |
|--------|------:|--------:|----:|-----:|-----:|------:|:----:|
| launch_s3_r003_cd120 | 16% | 103 | 1.7 | +1.5% | ≈0 | +14% | no |
| **launch_s3_r002_cd300_post10** | **53%** | **292** | **4.9** | **+1.0%** | **+8.7%** | **+4.9%** | **yes** |
| pulse_fo12_causal_post10 | 92% | 263 | 4.4 | −1.4% | −32% | +4.9% | no |

**promote = `STEP2B_launch_s3_r002_cd300_post10`**

读法：
- 赢家 = 1s launch（|ret|≥0.2%、cd=300）+ **避开开盘债窗**。  
- quote 晋级、trade-last 对赢家 econ=false —— 若按旧口径会漏掉；门正确。  
- FO pulse 族再灭一次。

**Step2b PASS → 进入 Step3**（该信号上 TP/SL/h 网格；不叠 ride）。

---

## Step3 结果（PASS）

产物：`/mnt/s990/data/maga7/results/research_am_v2_step3_exit_grid/`  
冻结信号：`launch_s3_r002_cd300_post10`（n_sig=550）  
网格：TP∈{10,15,20}% × SL∈{15,20,25}% × h∈{600,900,1200} → 27 格

| exit | mean | disc | blind | econ |
|------|-----:|-----:|------:|:----:|
| **tp15_sl25_h900** | **+1.06%** | **+13.2%** | **+2.1%** | **yes** |
| tp15_sl20_h900（Step2b 基线） | +0.96% | +8.7% | +4.9% | yes |
| tp20_sl20_h600 | +0.90% | +4.4% | +0.5% | yes |
| tp15_sl20_h1200 | +0.85% | +1.5% | +4.9% | yes |
| tp15_sl25_h600 | +0.76% | +5.1% | +3.3% | yes |

**promote = `STEP3_tp15_sl25_h900`**（5/27 格 quote econ）

读法：TP15 带稍宽 SL25 优于基线；TP10 全灭；TP20 多数 blind 翻负。不叠 ride。

**冻结配方（研究，经 blind-lift 更新）**：
```text
signal: launch |ret|≥0.25% k=3 cd=300 · 10:00–11:30
exit:   TP15 / SL25 / h900 · FillSpec 0.75 · lag≤5 · spread≤15%
mark:   quote only
```

**下一步 = Step4**：scanner drain + `execute_mode=shadow`（未接线前不进 dry/live）。

---

## Step4 结果（PASS — shadow 接线）

接线（2026-08-09）：
- `maga7/common/am_v2_sleeve.py` — 配置 + 在线 launch tracker  
- `Mag7Scanner._feed_am_v2_second` / `drain_am_v2` — 1s 因果，不占 TopK  
- peer3 profile `am_v2.enabled=true` · `execute_mode=shadow`  
- OMS：`am_v2_sleeve` 在 paper/live 拒单（`am_v2_shadow_only`）  
- Catalog：`ACCEPT_RESEARCH` · `enabled_on_spine=false`（shadow 经 profile 打开）  
- 单测：`maga7/tests/test_am_v2_live.py`

约束：与 Pulse A/B 并行 shadow；同符号由 OMS `has_position` 互斥。不进 dry/live。

**下一步 = Step5**：OMS dry/paper 验收后才考虑 `execute_mode` 升格。

---

## Blind-lift（同频抬 blind，PASS）

产物：`/mnt/s990/data/maga7/results/research_am_v2_blind_lift/`  
固定 cd=300 / TP15/SL25/h900；禁 densify。

| variant | tpd | mean | disc | blind | econ |
|---------|----:|-----:|-----:|------:|:----:|
| baseline r002 | 4.8 | +1.06% | +13% | +2.1% | yes |
| up_only | 2.7 | +2.75% | +37% | +0.2% | yes |
| dn_only | 2.6 | −0.9% | −12% | −7% | no |
| **r0025** | **2.4** | **+1.52%** | **+4.9%** | **+11.9%** | **yes** |
| r003 | 1.0 | +3.0% | +6.6% | +9.5% | yes |

**promote = `BLIND_LIFT_r0025_cd300_1000_1130`**  
写回：`abs_ret_min=0.0025`（profile / peer3 / `am_v2_sleeve`）。  
取舍：blind 厚了，日频与 disc 变薄——用质量换稳定性，不是加仓加频。

---

## 开盘债诊断（09:30–10:00，不晋级）

产物：`/mnt/s990/data/maga7/results/research_am_v2_open_debt_diag/`  
主表 trade-last · 附表 quote · 退出对齐冻结 TP15/SL25/h900

| signal | trade mean | trade disc/blind | quote fill | quote econ |
|--------|-----------:|-----------------:|-----------:|:----------:|
| launch cd300 open | +0.3% | +17% / **−8%** | 12% | no |
| launch cd120 open | ≈0 | +15% / −10% | 11% | no |
| pulse fo08 open | −7% | −34% / +1% | 23% | no |

**verdict = OPEN_DEBT_TRADE_SOFT**：开盘有 disc 上界痕迹，但 **blind 不稳**；quote 几乎不可成交。  
→ 跳过 09:30–10:00 **不是漏金矿**；冻结 post10 袖套不变。

---

## 复现

```bash
# Step1
PYTHONPATH=. python -m maga7.tools.run_am_v2_step1_quote_baseline \
  --tag research_am_v2_step1_quote_baseline

# Step2
PYTHONPATH=. python -m maga7.tools.run_am_v2_step2_signal_bakeoff \
  --tag research_am_v2_step2_signal_bakeoff

# Step2b
PYTHONPATH=. python -m maga7.tools.run_am_v2_step2b_signal_bakeoff \
  --tag research_am_v2_step2b_signal_bakeoff

# Step3
PYTHONPATH=. python -m maga7.tools.run_am_v2_step3_exit_grid \
  --tag research_am_v2_step3_exit_grid

# Step4 smoke
PYTHONPATH=. python -m pytest maga7/tests/test_am_v2_live.py -q

# Open-debt diagnostic (not promotion)
PYTHONPATH=. python -m maga7.tools.run_am_v2_open_debt_diag \
  --tag research_am_v2_open_debt_diag
```
