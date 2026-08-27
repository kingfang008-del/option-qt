# Stock-flow → ATM put（正股流买期权）

> 假设：信号看**股价 1s**（短窗下跌 + down-tick volume share），车辆 = Mag7 ATM put。  
> 期权 tick 仅用于定价；正股仍无 aggressor（Δclose×volume 代理）。  
> 工具：`tools/scan_stock_flow_opt_foresight.py` · `tools/run_stock_flow_winrate_ablation.py` · `tools/run_stock_flow_exit_ablation.py`

## 冻结基线（2026-07-25）

| 项 | 值 |
|----|-----|
| 窗 | Jul10–23（10 个交易日，有 option tick） |
| fire | `rising`（关→开） |
| Champion | `stk_d0.003_f120_sh0.6_tp0.25_sl0.2` |
| 入场 | `ret_60s ≤ −0.3%` 且 `dn_vol_share_120s ≥ 0.60` |
| 出场 | tp25 / sl20 / max_hold 900s |
| 组合 | position_frac=0.10，max_concurrent=4，cooldown=1min |
| n | 164 |
| **单笔胜率** | **46.95%** |
| 单笔 mean | +2.30% |
| **日胜率** | **70%**（7/10） |
| **账本 add** | **+33.26%** |
| 产物 | `results/research_stock_flow_opt_jul10_23` |

状态：Jul 单窗口袋 PASS；**Feb–Apr / May–Jul9 OOS 双窗均 FAIL**（`SINGLE_WINDOW_ONLY`）。尚未 quote。  
Catalog：`stock_flow_opt`，`enabled_on_spine=false`。

## 研究序（约定）

1. **已记录**上表基线（勿再用 Jul 网格把 +33% 刷没）  
2. **抬胜率**：入场过滤 → SMC/VWAP → 出场网格（Jul 过 55%）  
3. **多窗 tick** → May–Jul9 与 Feb–Apr **均未过**（仅 Jul 口袋正）  
4. **其后**：不接线；勿 quote 升格；换假设前先停此臂  

## 抬单笔胜率消融（2026-07-25）

工具：`tools/run_stock_flow_winrate_ablation.py`  
产物：`results/research_stock_flow_winrate_ablation_jul10_23`  
约束：入场基线 + 额外过滤；**tp/sl 不动**；仅 Jul10–23。

| variant | n | 单笔胜率 | mean | day_win | 账本 add | vs 基线 |
|---------|---|----------|------|---------|----------|---------|
| **baseline（冻结）** | 164 | **46.95%** | +2.30% | 70% | **+33.3%** | — |
| **volz15**（推荐候选） | 114 | **50.88%** | +2.31% | 70% | **+35.1%** | 胜率+3.9pp，add 略升 |
| disp005 | 49 | 51.02% | +2.73% | 67% | +11.1% | 胜率最高之一，add 大砍 |
| volz20 | 101 | 49.50% | +1.98% | 60% | +22.1% | 小升胜率、add 降 |
| quality_pack / share65 / mf 堆叠 | ↓ | ≤44% | 常负 | 差 | 差 | **无效** |

裁决：`WINRATE_PARTIAL` — **未到 55% 目标**；当前最佳是 `volz15`（`vol_z≥1.5`），胜率约 **51%** 且不伤账本。  
硬抬到 55%+ 在这套 tp25/sl20 上需要改出场或换窗，不单靠入场过滤。

**工作候选（未升格）：** `stk_d0.003_f120_sh0.6` + `vol_z≥1.5` + tp25/sl20。  
基线仍冻结保留对照。

## SMC / ICT + VWAP 再抬胜率（2026-07-25）

工具：`tools/run_stock_flow_smc_vwap_ablation.py`  
产物：`results/research_stock_flow_smc_vwap_jul10_23`  
在冻结入场闸上叠加：会话 VWAP 下方 / 深度、BOS、sweep、与 volz15 组合。

| variant | n | 单笔胜率 | day_win | add | 评 |
|---------|---|----------|---------|-----|-----|
| baseline（本跑） | 151 | 45.0% | 70% | +17.4% | warm 含 swing300，略异于冻结表 |
| below_vwap | 130 | 46.9% | 60% | **+27.9%** | 胜率小升，账本更好 |
| below_vwap_volz15 | 91 | **49.5%** | 50% | +21.1% | 胜率较好但日胜率掉到 50% |
| bos / bos+vwap / ict_pack | ↓ | ≈44–45% | 差 | 差或平 | **无帮助** |
| sweep | 10 | 30% | 差 | 负 | **有害** |

裁决：仍 **`WINRATE_PARTIAL`**，**到不了 55%**。  
VWAP「折价区」对账本有用，对单笔胜率只是小幅抬升；硬套 BOS/sweep 在这 10 天**抬不起胜率**。  
若还要冲 55%+，优先改出场（更易触达的 tp / 时间止盈），而不是继续堆 ICT 结构过滤。

## 出场消融（更易打到的 TP / 时间止盈）（2026-07-25）

工具：`tools/run_stock_flow_exit_ablation.py`  
产物：`results/research_stock_flow_exit_ablation_jul10_23`  
固定入场：`baseline` / `volz15`；网格 `tp∈{8,10,12,15,20,25}%` × `sl∈{12,15,20,25}%` × `max_hold∈{120…900}s`。  
目标：`trade_win≥55%` 且 keep_edge（mean>0, add>0, day_win≥55%, n≥40）。

裁决：**`EXIT_WINRATE_LIFT`** — **59** 格同时 hit_target + keep_edge（几乎全在 `volz15`）。

| 候选 | n | 单笔胜率 | mean | day_win | add | 备注 |
|------|---|----------|------|---------|-----|------|
| 冻结对照 tp25/sl20/h900 | 164 | 46.95% | +2.30% | 70% | +33.3% | 出场未动 |
| **volz15 + tp10/sl25/h900（推荐）** | 120 | **73.3%** | +2.65% | **90%** | **+34.4%** | 胜率+账本双赢；tp 占比≈72% |
| volz15 + tp08/sl25/h900（最高胜率） | 121 | **73.6%** | +1.27% | 70% | +18.3% | 胜率顶，add 砍半 |
| volz15 + tp12/sl25/h900 | 119 | 71.4% | +3.20% | 90% | +37.5% | 略宽 TP，add 更好 |
| volz15 + tp25/sl25/h900（最高 add） | 112 | 58.9% | +4.54% | 80% | **+43.3%** | 仍过 55%，偏吃腿 |

要点：

- **小 TP（8–12%）+ 宽 SL（25%）** 把单笔胜率从 ~47–51% 抬到 **70%+**；多数靠打到 TP 离场（`frac_tp≈0.68–0.72`），不是靠时间止盈。
- 缩短 `max_hold`（600/450）也能抬胜率，但通常 **add 劣于同 TP 的 h900**；时间止盈是次要杠杆。
- 工作候选（未升格、未接线）：**`volz15` + `tp=0.10` / `sl=0.25` / `max_hold=900`**。  
  冻结基线 `tp25/sl20` 仍保留对照。
- Jul 单窗勿直接升格 → 见下方双窗。

## Tick 多窗（Feb–Apr × May–Jul9 × Jul10–23）（2026-07-25）

工具：`tools/run_stock_flow_tick_dual.py`  
产物：`results/research_stock_flow_tick_dual_feb_jul`（前序 May–Jul：`…_may_jul`）  
数据：tick n≈119（Feb19 / Mar22 / Apr21 / May20 / Jun21 / Jul16）。  
闸门：OOS = **Feb_Apr ∩ May1_Jul9** keep_edge；Jul10–23 仅作发现口袋对照。

| cell | FA n | FA win | FA add | MJ n | MJ win | MJ add | Jul win | Jul add | 裁决 |
|------|------|--------|--------|------|--------|--------|---------|---------|------|
| 冻结 baseline | 678 | 41.3% | **−108%** | 702 | 38.7% | **−101%** | 47.0% | +33.3% | 仅 Jul |
| **exit_cand tp10/sl25** | 499 | **61.5%** | **−89.6%** | 509 | **62.5%** | **−40.7%** | **73.3%** | **+34.4%** | 仅 Jul |
| volz15 tp08/sl25 | 504 | 66.7% | −74.3% | 521 | 67.0% | −43.3% | 73.6% | +18.3% | 仅 Jul |

裁决：**`SINGLE_WINDOW_ONLY`**（OOS 双窗未过）。

要点：

- Feb–Apr 与 May–Jul9 **同构失败**：高胜率 + 负账本；Jul 口袋是唯一正 add 窗。
- exit_cand：FA/MJ 胜率≈61–63%，add 分别 **−90% / −41%** → 不是缺数据，是 **Jul 过拟合 + 不对称出场**。
- **禁止**接线 / Shadow / quote 升格；勿再为抬 Jul 胜率拧闸。停臂或换假设后再开。

### 开盘 1 小时消融（09:35–10:30）

对照：此前多窗用的是 **09:35–15:30 四段**（AM/CORE/MID/PM），不是字面 9:30–16:00，但也**不是**只开盘。  
产物：`results/research_stock_flow_tick_dual_feb_jul_open1h`（`--sessions open1h`）。

| cell | FA win / add | MJ win / add | Jul win / add | 裁决 |
|------|--------------|--------------|---------------|------|
| exit_cand（全日四段） | 61.5% / **−90%** | 62.5% / **−41%** | 73.3% / +34% | 仅 Jul |
| **exit_cand（开盘1h）** | 64.1% / **−60%** | 62.2% / **−31%** | 73.2% / +22% | 仍仅 Jul |
| frozen（开盘1h） | 40.5% / −72% | 39.9% / −34% | 45.5% / +16% | 仍仅 Jul |

**缩到开盘 1 小时没有好转**：OOS 仍负；Jul 口袋 add 还从 +34% 降到 +22%。不是「全日噪音拖累开盘边」。

## 相关否定臂

- `option_flow_scout`：看期权 put 量 → REJECT / 前视无 distill  
- `smc_flow_scout`：SMC+正股量 → quote REJECT  
- put-flow multi-fire / rising：加笔杀边  
