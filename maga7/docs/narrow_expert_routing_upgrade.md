# 窄形态专家路由升级（2026-07-24）

> 配套：[`watchdog_stack_architecture.md`](watchdog_stack_architecture.md) · [`regime_router_research.md`](regime_router_research.md) · [`sleeve_portfolio_research.md`](sleeve_portfolio_research.md)  
> 目录：`CONFIG/narrow_experts/catalog_v1.json` · loader：`common/narrow_experts.py`

## 1. 升级目标

近期结构（尤其 Jul10 后）不等于「换一套 AM 高频主策略」。  
正确升级是：

1. **L0 10:30 research_baseline 仍是默认开仓脊骨**  
2. 在 Watchdog / Router 下挂 **窄专家**（一种失败/漏抓形态 = 一个专家）  
3. 风险专家可先挂；**开仓类专家**必须双窗 trades + **quote FillSpec PASS** 才进路由表  
4. 禁止为追近几日行情拧全局 Rule-A / streak / peer

## 2. 当前脊骨上已挂的专家

| expert_id | 层 | 状态 | 作用 |
|-----------|----|------|------|
| `rebound_trap_dn` | L1 Degrade | ACCEPT_RESEARCH | QQQ>open 时 DN×0.5 |
| `washout_gate_halt` | L1 Halt | ACCEPT_RESEARCH | 广度洗盘 → 当日禁 Rule-A |
| `hunt_washout_reclaim` | L2 Hunt | ACCEPT_RESEARCH | **第二条开仓**：单票洗盘收回偏 UP |

`regime_router` ML / `prevention_mixed_wash_up`：**未开**（因果未过 / 双窗 FAIL）。

Profile 指针：`narrow_experts.catalog_path` → 上表权威状态。

### 2.1 可执行卫星开仓专家（2026-07-24）

| expert_id | 状态 | 形态 | 与脊骨关系 |
|-----------|------|------|------------|
| **`qqq_open_cont`** | **ACCEPT_RESEARCH** | QQQ 0DTE 09:45 开盘续作，\|fo\|≥0.2%，tp10/sl25 | **独立袖**，不改 Mag7 Rule-A |

- Quote 双窗 PASS（Jan–Mar / May–Jun）；trades 含 Jul10–23 仍正。  
- Profile：`CONFIG/strategy_profiles/qqq_open_cont_0945_fo02_tp10_sl25_v1.json`  
- Replay：`python -m maga7.tools.run_qqq_open_cont_expert`  
- Jul dte0 quote 仅到 06-30 → expert runner `book=auto` 对 Jul 走 trades fallback。

## 3. 研究队列（未进脊骨）

| expert_id | trades | quote | 裁决 |
|-----------|--------|-------|------|
| `core_dn_sync` | CORE DN sync 硬闸 PASS（jul10 n≥7） | **QUOTE_REJECT**（见 §3.1） | **禁止**默认注入；放宽 lag 仍亏 |
| **`impulse_scout`** | DN impulse 独立侦查兵 **PASS**（champ `imp_t0.008_lb120_tp0.2_sl0.2`） | **QUOTE_REJECT**（dual_pass_n=0） | **禁止**接线；见 [`impulse_scout_research.md`](impulse_scout_research.md) |
| **`smc_flow_scout`** | BOS/sweep+位移；OF 代理（tick-vol share / mf）弱 **PASS**（flow=off） | **QUOTE_REJECT** | **禁止**接线；见 [`smc_flow_scout_research.md`](smc_flow_scout_research.md) |
| **`option_flow_scout`** | 现有期权 1s put/call 量份额 | —（trades **REJECT**） | **禁止**接线；见 [`option_flow_scout_research.md`](option_flow_scout_research.md) |
| **`stock_flow_opt`** | 正股跌+dn_vol_share → ATM put | 仅 Jul 口袋 PASS；**Feb–Apr / May–Jul9 均 FAIL**（win≈61–63%，add≈−90%/−41%） | **禁止**接线；停臂；见 [`stock_flow_opt_research.md`](stock_flow_opt_research.md) |
| **`stock_flow_up`** | 正股涨+up_vol_share → ATM call（2–6 安静牛假设） | 发现窗 **FORESIGHT_NO_DISTILL**（lift&lt;1）；因果软 PASS 不迁移 Jun/Jul | **禁止**接线；见 [`stock_flow_up_foresight_research.md`](stock_flow_up_foresight_research.md) |
| `am_delayed_confirm` | 软 PASS | **QUOTE_REJECT** | 研究旁路 |
| `am_hf_launch` / foresight AM | 偶有纸面边 | **反复 REJECT** | 勿升格 |

Champion（仅研究）：`sync_t0.003_ss30_so30_tp0.2_sl0.15` · 窗 10:30–11:30 · DN。  
产物：`research_certainty_morph_core_dn_sync_dual_n7` / `…_quote_dual`。

### 3.1 CORE DN sync — quote 覆盖诊断（2026-07-24）

工具：`python -m maga7.tools.probe_core_dn_sync_quote_coverage`  
产物：`results/research_core_dn_sync_quote_coverage*` · 放宽 lag 双窗 `…_quote_dual_lag{10,15}`

| 发现 | 数据 |
|------|------|
| 文件覆盖 | quote day / ticker **几乎不缺**（miss≈0） |
| lag=3（基线门） | DN 臂 130 → entry_probe_ok **37**（28%）；fail **全是 lag**；sync 后 quote fill **2** |
| 盘口质量 | spread p50≈4–5%（可过 8–15% 门）；问题是 **更新稀疏**：next-quote lag p50≈11s |
| quote vs trades sync | stock_sync 后 trades 标绿远多于 quote FillSpec 回看（`trades_only`≫`both`） |
| lag→10 / 15 | probe_ok 升到 67 / 83，fill 升到 ~15–20%；**但 quote 双窗 mean/day_win 仍为负** → 仍 REJECT |

**结论：** 不是「再拉几天 quote 文件」能修好；trades 纸面边在可执行 FillSpec 上消失。下一闸若重开，需要更密 NBBO 或换确认标记，**禁止**用 lag≥10 硬凑样本升格。

## 4. 升线闸门（开仓类）

```text
形态假设 → 双窗 trades scoreboard
         → quote FillSpec 双窗（fill 率与 PnL 同向）
         → 写入 catalog status=ACCEPT_RESEARCH
         → 独立研究 profile 接线（event_source=expert_* / hunt detector）
         → OOS / 邻域 / Live 对拍后再议 ACCEPT_LIVE
```

`common.narrow_experts.assert_entry_promotable` 用于防止把 `QUOTE_REJECT` 臂误接进 scanner。

## 5. 近期行情怎么「参与」

| 近日结构感 | 优先用 |
|------------|--------|
| 假跌反抽 / QQQ 红绿拧巴 | 已有 L1 degrade/halt（勿再开 AM 抢跑） |
| 早盘深洗后收回 | 已有 L2 Hunt washout_reclaim |
| CORE 确认式下跌（正股+期权同步） | `core_dn_sync` **仅研究**，勿升格 |
| 会话内急跌（\|ret_lb\|≥0.8%/120s） | `impulse_scout` **仅研究**（trades PASS / quote REJECT） |
| 开盘半小时指数方向清晰 | **`qqq_open_cont` 卫星袖**（可执行） |
| 基线 0 笔且信号被 path/regime 挡 | 先审计当日 block 原因；**不**放宽 L0 全局门 |

## 6. 明确不做

- 把 AM 秒级 sleeve 写成 research_baseline 默认窗  
- 在 quote REJECT 时用 trades 账本冒充可执行路由臂  
- 复活 `orb_fractal` / `early_mf` / 宽 washout prevention  
- 为单日故事加符号黑名单当默认专家

## 7. 下一刀（执行序）

1. **保持**脊骨 L0+L1+L2；Live 对齐 Hunt（P3 Shadow）  
2. **因子→动作**：用 Watchdog 四态（NORMAL/DEGRADE/HALT/HUNT），见 [`regime_factor_routing_playbook.md`](regime_factor_routing_playbook.md)；**勿**开 ML router  
3. **`qqq_open_cont`**：Shadow 旁路已接；ops 验收 + 补 Jul quote / IB 链  
4. 恢复 Mag7 **1m `stock_root`**（`spnq_train` 若缺失则脊骨 replay 0 日）后，强窗延到 Jul23 锁参复验 L1  
5. Jul 侦查兵（`stock_flow_*` / impulse / smc / core_*）：维持旁路 REJECT；新开仓臂须 quote 双窗 PASS