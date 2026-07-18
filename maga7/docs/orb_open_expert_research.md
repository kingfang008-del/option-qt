# ORB Open Expert：开盘洗盘 + 分形高点突破

## 动机

Regime Router 的 `reclaim_disp55` 只能在 10:30 缩 DN，盖不住「早盘单边洗盘后 V 反」这类机会。用户拆成两条早盘逻辑：

1. **ORB 分形高点突破**（本文件）— 1m 即可，先做  
2. **OFI tick 主动吃单** — 当前无稳定 aggressor tape，**暂不做**（配置占位见 `CONFIG/regime_router/orb_open_expert_v1.json` → `ofi_tick`）

原则与 Router 一致：**默认 freeze 不动**；ORB 是窄专家，验收前默认 `enabled=false`。

## 状态与信号（因果）

### `open_washout`

在 `09:30`–`wash_window_end`（默认 10:00）内，相对开盘价最低价回撤 ≥ `wash_drop_min`（默认 0.3%）。

### `orb_fractal_break`（v1 仅 UP）

1. 开盘后第一段**单边下杀**：连续刷新更低 `low` 且收盘仍低于开盘  
2. 下杀结束（不再创新低）时，锁定 **`fractal_high` = 最后一根下杀 K 的 high**  
3. 之后第一根 **收盘严格站上** `fractal_high` → 开火（可选 `hold_confirm_bars`）  
4. 截止 `signal_deadline`（默认 10:00）；错过则当日无 ORB

实现：`maga7/common/orb_open.py`。

## 扫描工具

```bash
/home/kingfang007/anaconda3/envs/ibkr/bin/python -m maga7.tools.scan_orb_open_expert \
  --start-date 2025-07-01 --end-date 2026-07-17 \
  --out maga7/results/orb_open_expert/scan_2025-07_2026-07-17
```

合约 / 出场对齐 freeze profile（open_ladder OTM5 + extend_mtm + 60s bar delay）。组合层：按日最早信号、`max_concurrent=2`、`position_frac=0.2` 并发分仓。

产出：`signals.csv` / `trades.csv` / `trades_raw.csv` / `scoreboard.csv` / `summary.json`。

## 首轮扫描结果（2026-07-18）

口径：freeze 合约 + `hold_extend` + 60s delay；组合 `pos=0.2` / concurrent≤2。  
期权报价覆盖主要在 2026 lock 窗；全样本大量 `underlying` proxy，**验收以 option-priced 为准**。

| 变体 | 窗 | n | win | cum (portfolio) | 备注 |
|------|----|---|-----|-----------------|------|
| 默认 wd≥0.3% | May–Jul option port | 44 | 32% | **−7.5%** | 基线扫描 `scan_2025-07_2026-07-17` |
| 默认 | Feb–Apr option port | 52 | 27% | **−48%** | 假突破多 |
| wd≥0.5% + hold1 | Feb–Jul | 86 | 34% | **−66%** | `scan_strict_wd5_hold1_feb_jul` |
| wd≥0.8% | Feb–Jul | 79 | 28% | **−91%** | 更深洗盘更差 |

wash_drop 三分位（raw option）：最浅档 win≈40% / avg≈−1%；更深两档更差 → **不是「洗得越深越好」**。

### 结论（当前定义）

**REJECT 作为独立开仓专家**（默认保持 `enabled=false`）。  
「收盘站上首波下杀分形高」在 Mag7 open_ladder 路径上胜率不足，双窗均为负；加严 wash / hold 未翻正。

### 门控第二刀（已做）

工具：`tools/run_washout_gate_scoreboard.py` / `_tight.py`  
明细：`results/regime_router/washout_gate_scoreboard(_tight)/`

| 变体 | May–Jul vs 基线 | Feb–Apr | 07-17 | 判定 |
|------|-----------------|---------|-------|------|
| `washout_b3@0.3%` 任意动作 | ≤59% / halt=0 | 水床 | — | **REJECT**（几乎天天触发） |
| `wd8_b5` DN/both/halt | 强窗尚可 | 弱窗 58–89% | 半仓/−3% | **REJECT**（误伤弱窗） |
| `wd8_b5_and_reclaim` + **halt** | **~108%** | **~100%** | **0**（跳过） | **候选**（窄门） |
| `reclaim_disp55`（对照） | ~104% | ~104% | −3.3% | 仍可用（软缩 DN） |

要点：

- 「有 washout」**不足以**禁开仓；0.3%/breadth3 会清空策略。  
- 与你的直觉对齐的可执行规则是：**多标的深洗盘（≥5×≥0.8%）且 QQQ 低开假收回** → 当日 Rule-A **硬停**。  
- 配置：`CONFIG/regime_router/router_rule_washout_and_reclaim_halt.json`（研究开关；**freeze 默认仍 off**）。

## 与 Router 的挂接（下一步，未默认开）

- ORB **不是**改 Rule-A `window_start`；当前更宜作状态标签，而非直接发单  
- 验收：强窗 ≥95% 基线 **且** 洗盘簇有增量；本轮未过  

## 验收门槛（写入 freeze 前）

| 项 | 门槛 |
|----|------|
| 默认开关 | `enabled=false` |
| 双窗 cum / MaxDD | 相对 freeze 不出现水床式腰斩 |
| 与 Rule-A 重叠 | 同日同向需有位移/并发规则（待定） |
| OFI | 有 tape 后再开第二专家 |

## 相关

- Freeze：`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`  
- Router：`docs/regime_router_research.md`  
- 配置：`CONFIG/regime_router/orb_open_expert_v1.json`
