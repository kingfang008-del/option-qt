# Peer3 L3 causal soft exit（状态记录）

**Status (2026-07-22):** research / **shadow candidate** — **未**并入 `research_baseline`（`...peer3_v1`）。  
**Goal:** 在保留 peer3 入场与 T30→T45 **最大持仓时钟** 的前提下，用路径因果软出场降低「纯靠时钟磨到毒性」的比例（尤其 Jul20 类）。  
**尾损优先组合：** 本 L3 + `trade_toxic.cut_ret=0.20` → [`peer3_tail_loss_research.md`](peer3_tail_loss_research.md) · `..._peer3_tail_tox20_wash_m3_v1`。

## 出场分层（当前理解）

| Layer | 角色 | 基线 peer3_v1 | L3 候选 |
|-------|------|:-------------:|:-------:|
| L1 TP/SL | 价格轨 | ON | ON |
| L2 `trade_toxic` | 成交后毒性路径 | ON | ON |
| **L3 `STOCK_REV`** | 标的废证续持 → 时钟前可下 | **OFF** | **ON（候选）** |
| L4 T30→T45 | 最大持仓时钟 | ON | ON（仍为上限，非唯一裁判） |

要点：L3 **不是**「用因果逻辑取代 30 分钟」；而是 **因果可截断 + 时钟作上限**。

## Profiles

| Profile | Role | Arm | 参数摘要 |
|---------|------|-----|----------|
| `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_peer3_l3_wash_m3_v1.json` | **主推 shadow** `research_l3_causal` | `mixed_wash_up` | min_hold **10m**, `stock_max=−0.3%`, `opt_mtm_max=0`, breadth≥3 |
| `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_peer3_l3_uw_m3_h15_v1.json` | 备选 `research_l3_causal_alt` | `always` | min_hold **15m**, `stock_max=−0.3%`, `opt_mtm_max=0` |

Catalog 已登记（2026-07-22）。`frozen_at=null`。

## 晋级条（约定）与细网格结论

约定同时满足才可考虑升线：

1. May–Jul **ret retention ≳ 0.85**（相对 peer3 基线）
2. **clock_share 下降**（少靠 T+/TIME）
3. Jul20 fused **优于** 基线日亏，且尽量 **截 AMD、不误杀 GOOGL**

工具：`tools/run_peer3_l3_causal_exit_ablation.py`（`--mode fine`）  
产物：`/mnt/s990/data/maga7/results/peer3_l3_causal_exit_ablation_v1/`  
（`scoreboard_fine.csv` · `verdict_fine.json` · `jul20_fused_fine_tight.json`）

### 粗网格（否决/无效）

| 变体 | May–Jul | 结论 |
|------|--------:|------|
| 紧 `srev_10_10` / `srev_5_05` / `trail15` | retain ≪ 0.85 | 砍右尾，否决进基线 |
| `hold_watchdog` | ≈ 基线（0 触发） | 无效臂 |
| `srev_uw_m3` @ hold10 always | retain ~**0.77** | 方向对，未过 0.85 |

### 细网格（过晋级条的短名单）

基线对照（同窗 offline）：May–Jul **+18.56**（clock ~**46%**）；Jan–Mar **+0.86**（clock ~**62%**）；Jul20 fused **−8.3%**（AMD SL −55%，GOOGL T30 +29%）。

| 变体 | May–Jul ret / retain | clock | Jan–Mar retain | Jul20 fused | AMD / GOOGL |
|------|---------------------:|------:|---------------:|-----------:|-------------|
| **wash_m3_uw_h10**（主推） | +16.75 / **0.91** | **37%** | **1.01** | **−3.5%** | −32% REV / T30 留 |
| **uw_m3_h15**（备选） | +16.50 / **0.90** | **28%** | 0.93 | −4.1% | −35% REV / T30 留 |
| uw_m6_h20（保右尾） | +17.70 / **0.96** | 39% | 0.97 | −6.4% | −46% REV / T30 留 |

**结论（2026-07-22）：** 细网格后晋级条 **可以同时满足** → 已落 research profile；**尚未**改写 `...peer3_v1` research_baseline。

## 与相关路线的边界

- **path_hold_lit / wash_fast**：另一条「风险优先、可废 extend」研究线；见 [`path_adaptive_hold_research.md`](path_adaptive_hold_research.md)。L3 候选 **仍挂在 peer3 hold_extend 脚手架上**，改动更小。
- **VIXY / 成交量臂**：叙事分歧日可用作加严开关的研究想法；peer3 上 `vix_reversal_max` 仍为 null。未进本轮晋级。
- **勿**把 always-fast / 裸 wash / 全局 lit path-hold / 研究宇宙删 AMD 当成 L3 升线捷径（历史 ablation 已否决或伤强窗）。

## 升线门槛（何时才改 peer3_v1 / 开 peer3_v2）

全部满足再议：

1. Shadow / fused 再跑一轮：多日毒性样本（不仅 Jul20）+ OMS early-exit 确认含 `STOCK_REV`
2. 双窗复验：May–Jul retain ≳ 0.85 且 Jan–Mar 不显著劣于基线
3. 明确接受右尾代价（主推约 **~9%** May–Jul 复利保留缺口）与「时钟仍在」的产品语义
4. 文档与 catalog：新 `research_baseline` id（建议 **peer3_v2**），旧 v1 留对照

## Ops 用法（shadow）

```bash
# offline（示例）
python -m maga7.common.replay # 或既有 day_stream / offline 入口
# profile:
#   maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_peer3_l3_wash_m3_v1.json
```

Live：沿用 peer3 会话脚本，**profile 指向 L3 候选**；确认 broker/OMS 路径识别 `STOCK_REV`。

## Non-goals（当前）

- 不把 L3 默默写进 `...peer3_v1.json`
- 不宣称「30 分钟持仓已完全因果化」
- 不把 trail / 紧 srev 当默认 L3
- 不在未双窗验收前打开 VIX 硬门禁替代 wash∧STOCK_REV
