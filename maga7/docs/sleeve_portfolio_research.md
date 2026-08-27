# Session sleeve research（1s 股价重扫）

**日期：** 2026-07-24  
**状态：** 旧规则（1m 左标签 / H120 / MF / MF_P2_FO）**全部作废**；May–Jul 已用 `/mnt/s990/data/raw_1s/stocks` 重扫  
**架构：** 无 peer3 / 无 T+30；期权定价 trades last ±1% slip；容量有限（≤2 并发），禁止把 additive 当无限复利。

## 已删除

带前视污染的 `research_session_*` / `session_h120_*` / `parity_session_*` / `parity_gate_*` / `sleeve_portfolio_session_*` 结果目录已从  
`/mnt/s990/data/maga7/results/` 清除。

## 干净重扫（2026-05-01 → 2026-07-22）

```bash
PYTHONPATH=. python -m maga7.tools.scan_session_horizon_foresight \
  --start-date 2026-05-01 --end-date 2026-07-22 \
  --stock-source 1s --sessions AM_0930_1000,MID_1230_1330 \
  --tag research_session_horizon_foresight_1s_may_jul
```

| 项 | 设定 |
|----|------|
| 股价 | **1s close** 因果：`last≤t` vs `last≤t−60s`（默认；禁 1m 左标签） |
| 信号 | stride 120s，`|stock_ret|≥5bps` |
| Hold 网格 | 30…900s foresight clock / oracle |
| 加固 | 1s→1m + **+1m 可用性平移** 后的 mf/peer（`stock-source 1s`） |

### H 网格（clock_mean，全样本）

| H | AM | MID |
|--:|---:|----:|
| 30 | **−2.1%** | **−1.8%** |
| 60 | −2.0% | −1.8% |
| 120 | −2.3% | −1.9% |
| … | 更差或同类 | 更差或同类 |

胜率 AM≈41%、MID≈30–37%。Oracle 仍为正（路径有期权性），但 **因果 clock 退出全负**。

### 加固消融（H=60 / 120，机会组合 ≤2 并发）

`research_session_{am,mid}_rein_1s_may_jul_h{60,120}`：  
BASE / MF / ST2 / P2 / P3 / FO / MF_P2_FO / VZ15 / ALL_SOFT **全部 mean&lt;0、recommend=[]**。

### 流动性

- 持仓窗 `n_prints` 高分位看似 +EV，但是 **前视**（入场时未知），**不可用**。  
- 因果代理：入场前 60s 同合约成交笔数 `prior_prints_60` → 过滤后仍为负。  
- 结论：流动性约束必须因果；且当前动量袖在有限容量下 **没有可推广 +EV 规则**。

## 裁决

| 旧主张 | 新结论 |
|--------|--------|
| H=120 + MF / MF_P2_FO 双窗 PROMOTE | **REJECT**（前视假象） |
| Session book 强窗 Σday +15 | **作废** |
| 可叠预算干跑 | **暂停**，直到干净数据上找到 +EV 规则 |

产物根：`/mnt/s990/data/maga7/results/research_session_horizon_foresight_1s_may_jul/`  
加固：`research_session_*_rein_1s_may_jul_h*`  
因果流动性探针：`causal_prior_prints_h60.csv`

## 1s 特征入口重扫（2026-07-24）

工具：`scan_session_1s_feature_entry`（正股特征全在 `/mnt/s990/data/raw_1s/stocks` 上因果计算，**不**用 1m）。  
特征对齐 `feature_merge_option_raw` / `slow_feature_qqq_v2` 可落地项：`ret_*`、`volume_ratio_60`、`vwap_diff`、`ret_div_*`、`range_*`、`vol_z`、`mf100`/`streak`、`from_open`；规则含 MOM/量价/VWAP 对齐/背离/MF/fade。

| 结果 | May–Jul AM/MID |
|------|----------------|
| 规则网格（H=30/60/90/120，≤2 并发） | **picks=0**，全部 mean&lt;0 |
| 特征五分位 → clock | 各袋约 −1.8%～−2.1%，无单调正袋 |
| 对照：同信号股票 60s 有向收益 | mean≈0、胜率≈48%（方向本身无边） |
| 期权相对股票更差 | slip ±1% 把零边打成 −2% |

产物：`research_session_1s_feat_may_jul/`、`…_v2/`（修 volume_ratio 后）。  
**结论：** 在「60s 动量 + 量价/VWAP/MF 过滤 + 短持期权」框架内，干净 1s **扫不出可推广入口**；问题在方向期望≈0，不是缺某一个过滤。

### 出场约束（硬规则）

**禁止**固定 clock 持仓作为主出场。必须 **止盈 + 止损**（期权 trade-last 路径 first-passage）；`max_hold` 仅安全强平。

工具：`scan_session_1s_tpsl_entry` + `common/option_trade_tpsl.py`  
May–Jul 网格：tp∈{5,10,15,20}% × sl∈{5,8,10,15}% × 同上入口规则，`max_hold=900s`。

| 结果 | |
|------|--|
| picks（mean&gt;0 ∧ day_win≥0.55 ∧ 多数经 TP/SL 出） | **0** |
| 典型路径 | 对称档位下 **SL 命中率 &gt; TP**（如 tp=sl=5% → ~67% SL） |
| 最不差 | AM `MOM60_VOLR15` tp20/sl8 ≈ −0.6%/笔，仍负 |

产物：`research_session_1s_tpsl_may_jul/`。  
**裁决：** 出场形态改对了，但当前入口仍无边；下一刀砍入口范式，不是再堆 clock H。

## Launch-slope + TP/SL（2026-07-24）

入口改回 1s launch-slope（非 MOM60），出场 trade-last TP/SL：  
`scan_launch_slope_tpsl` → `research_launch_slope_tpsl_may_jul/`。

May–Jul trade 过闸 **50** 组；Jan–Mar + quote 对拍后 **REJECT**。  
详见 [`launch_slope_research.md`](launch_slope_research.md)「Dual-window OOS」：quote Jan–Mar 全面负，trade 有流动性选择偏差。

## Morning sec-MF + quote TP/SL（2026-07-24）

正交入口：秒级 MF streak（`scan_morning_sec_edge` → `research_morn_sec_edge_{may_jul,jan_mar}`）。  
验收：`scan_morning_sec_quote_tpsl` → `research_morn_sec_quote_tpsl_dual/`。

| 项 | 结果 |
|----|------|
| 股票路径双窗弱正 | 有（exp≈+0.05–0.17%/笔） |
| quote TP/SL 双窗过闸 | **0 → REJECT** |
| Jan–Mar quote | 全面负（如 best May cell 在 JM add≈−0.12） |

## QQQ open_cont + quote TP/SL（2026-07-24）— 首个双窗 PASS

`scan_qqq_open_cont_quote_tpsl` → `research_qqq_open_cont_quote_tpsl_dual/`。

| 项 | 值 |
|----|-----|
| 入口 | QQQ 09:45 open_cont，`|from_open|≥0.2%` |
| 出场 | quote FillSpec TP/SL（tp10 / **sl25**） |
| 门 | spread≤15%、lag≤2s、mid≥0.05 |
| May–Jun* | n=24 mean **+3.6%** add +0.087 day_win 79% |
| Jan–Mar | n=26 mean **+6.8%** add +0.178 day_win 88% |
| 双窗过闸 | **72**（含 fo=0；**选择性 fo≥0.2%：30**） |

\*0DTE quote 文件目前只到 **2026-06-30**，May–Jul 实为 May–Jun。  
**谨慎未升格：** 全部硬过闸都靠 sl=25%；需补 Jul 数据、对冲「开盘漂移 null」、再写 profile。

## Micro-state scalp MVP（2026-07-24）

架构：**正股 \(f'/f''\)+SNR+双窗合流 → ATM quote → TP/SL/90–120s timer**。  
**搁置：** 期权曲线拟合 / `scan_smooth_regress_quote_tpsl`。  
**REJECT（松门）：** `scan_micro_state_quote_scalp` — Mag7/QQQ mean>0 = 0；根因是正股 fwd≈0 + 过密触发（~60/sym/day）。

### 正股诊断（`diagnose_maga7_micro_stock_edge`）

结果：`research_maga7_micro_stock_diag/`，`verdict=STOCK_EDGE_FOUND`。

| 门 | n | 双窗正股 | 期权 quote scalp（tp10/sl10/t90/sp8%） |
|---|---|---|---|
| 基线 s20_l60_snr2.5 | ~13k | win60≈50% | （此前全网格 REJECT） |
| **snr≥7 + vol_z≥1 + \|ret_s\|≥0.1%** | **53** | Jan–Mar win60 **61%** / May–Jul **72%** | fills **8+3**；mean 似正但 **n 不足过闸** |
| snr≥5 + vol_z≥1（放宽） | 803 | May–Jul **翻负** | 双窗 mean **−0.6% / −2.6%** |

### Follow-up（entry reject + QQQ/VWAP，2026-07-24）

| 检查 | 结果 |
|---|---|
| entry reject 根因 | **信号时 ATM 路径无报价**：lag 常为数百～1700s（非点差门） |
| 延迟入场 wait 15–60s | miss 仍 ~33/48，**救不回** |
| QQQ 20s/VWAP/from_open 合流扩样 | May–Jul 正股 **翻负**；期权双窗 mean 仍负 |
| 1DTE prefer | 稀 pocket mean 略好，fills 仍个位数 |

**裁决：`PARK_MAG7_MICRO_TO_OPTION`** — 正股稀 pocket 有边，但 **impulse 时刻期权放大器不可用**；合流无法在不毁 OOS 的前提下扩样。明细：`research_maga7_micro_stock_diag/FOLLOWUP_entry_confluence.json`。

## AM 09:30–10:00 × trades 双窗（收敛窗，2026-07-24）

只打这一时段：信号硬切 **[09:30, 10:00)**；定价 `/mnt/s990/new_option_data_s3_trades`；出场 trade-last TP/SL。

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_0930_1000_trades_dual \
  --tag research_am_0930_1000_trades_dual
```

| 项 | 结果 |
|----|------|
| 入口 | Mag7 launch-slope **open** cells（AM cut） |
| 双窗 PASS | **17**（`verdict=PASS`，trades 账本） |
| Champion | `open_s3_r002_p2` **tp15/sl25**：JM n=31 mean +2.6% dw 65%；MJ n=21 mean +9.2% dw 82% |
| 紧 SL（≤15%） | 仅 2 组过闸（`p2` / `fp003_p2_mf1` 的 **tp5/sl15**，mean 偏薄） |

**边界（未升格 live）：** 这是 **trades-book 研究过闸**，用来先攻克开盘半小时。

### Quote 可执行对拍（同窗，2026-07-24）

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_0930_1000_quote_dual \
  --tag research_am_0930_1000_quote_dual
```

| 项 | 结果 |
|----|------|
| 对象 | trades dual champions（p2/p3/fp003…） |
| 门 | spread≤{8,10,15}% · lag≤{2,3}s · mid≥0.05 · FillSpec 0.75 |
| 宽门 resolve | **31 / 119**（~26%；trades 同批可成交远更多） |
| Champion `p2` tp15/sl25 | Jan–Mar mean **负**（n≈16–21）；May–Jul mean 正但 **n=5–9** |
| 双窗 PASS | **0 → `verdict=REJECT`** |

**裁决：`AM_TRADES_PASS_QUOTE_REJECT`** — 开盘半小时在 trades 账本上能立住，但 **quote FillSpec 不可执行/不赚钱**；不写 profile、不 dry。根因仍是流动性选择偏差（有 print 的子集偏乐观）。

## QQQ open_cont × trades（AM，含 Jul，2026-07-24）

同窗收敛到 QQQ 开盘续作；定价 `new_option_data_s3_trades`（补齐 Jul）。

| 项 | 结果 |
|----|------|
| 工具 | `scan_qqq_open_cont_trades_tpsl` → `research_qqq_open_cont_trades_tpsl_dual/` |
| dual PASS | **15**（cont 10）；`verdict=PASS` |
| Champion | 09:45 \|from_open\|≥0.2% · tp10/sl25（JM +8.4%/dw90%，MJ +2.6%/dw75%） |
| Null | 09:45 fade **不过**；09:50 fade 有过闸（时钟别用 09:50） |

详见 [`qqq_open_cont_quote_tpsl.md`](qqq_open_cont_quote_tpsl.md)。

## Next（研究，非升格）

1. **QQQ open_cont champion**：在 trades 过闸基础上补 **quote FillSpec**（Jun 前已有；Jul 需 quote 数据或接受 trades 影子）。  
2. Mag7 AM launch→ATM：**停升格**。  
3. Mag7 micro→ATM / 松门 MOM clock / 午盘 sleeve **暂不碰**。  
4. **窄专家路由**：HF / CORE sync 袖只进 `CONFIG/narrow_experts/catalog_v1.json` 研究队列；quote PASS 前不进 Watchdog 默认臂。见 [`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md)。  
5. **CORE DN sync quote 覆盖（2026-07-24）**：文件齐；主因 next-quote **lag**（p50≈11s）+ FillSpec sync 与 trades 分歧；lag10/15 双窗仍 **mean&lt;0** → 维持 `QUOTE_REJECT`。探针 `tools/probe_core_dn_sync_quote_coverage.py`。  
6. **可执行卫星专家 `qqq_open_cont`（2026-07-24）**：quote 双窗 PASS → catalog `ACCEPT_RESEARCH`；profile `qqq_open_cont_0945_fo02_tp10_sl25_v1`；runner `run_qqq_open_cont_expert`。不改 Mag7 脊骨。  
7. **`core_up_sync`（2026-07-25）**：foresight Jul10+ 弱正口袋 → CORE UP-only sync/chase trades 双窗 **REJECT**（May–Jul9 尚可，Jul10–23 崩）。catalog `REJECT`；勿为 Jul 拧 thr。
## 相关

- [`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md)（脊骨）  
- [`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md)（Watchdog 下挂窄专家）
