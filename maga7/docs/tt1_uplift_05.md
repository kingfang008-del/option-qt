# TT1 uplift 05 → research baseline

**日期：** 2026-07-19  
**Profile：** `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`  
**消融：** `tools/run_tt1_uplift_ablation.py` → `results/tt1_uplift_ablation_peer3_may_jul/`  
**锁参产物：** `results/research_extend_mtm_full_day_peer3_l2_tt1_05_may_jul/`（gitignored）

## 问题

二四主仓多为 **1DTE**：股价大波动并不弱于一三五，但期权兑现偏弱。目标是**抬高二四收益**，不是砍掉二四。

## 写入基线的规则（臂 `05`）

| 旋钮 | 值 | 含义 |
|------|-----|------|
| `max_entry_abs_otm_pct` | `0.005` | 入场时行权价相对现价偏 OTM **>0.5%** 则跳过 |
| `entry_confirm_bars` | `2` | 信号后再等 **2** 根 1 分钟 K |
| `entry_confirm_mode` | `mf` | 确认期内 mf 仍同向 |
| `entry_confirm_weekdays` | `[1, 3]` | **仅周二、周四**启用 confirm（一三五不动） |

实现位置：offline `replay.py`；合约闸 `entry_contract.py`；stream / live scanner 对齐。

## 只看二四 1DTE（May–Jul）

| | n | 胜率 | 期望 | pnl_unit |
|--|--:|-----:|-----:|---------:|
| L2 对照（无 05） | 20 | 50% | +11.5% | +0.486 |
| **L2+05** | 18 | 56% | +14.5% | **+0.542（Δ+0.056）** |

## May–Jul 组合（复利）

| | total_ret | MaxDD | n |
|--|----------:|------:|--:|
| L2 对照 | ~+1282% | −12.2% | 60 |
| **L2+05（当前基线）** | **+1528%** | −12.2% | 57 |

仓位仍为 `position_frac=0.2`（相对**当时**权益），未突破 20% 上限。

## REJECT（二四本身也掉）

- 按 `|from_prev|` 重排 / displace / 11:00 commit TopK  
- 持有拖到 T+60 或硬 T+45  
- **全局** entry confirm（伤一三五 0DTE；必须加 `weekdays`）

## 同日基线一并冻结的相邻旋钮

（非 05 本体，但同日写入同一 research profile）

- `ladder_otm_rungs` / lock `otm_rungs`：**5 → 3**  
- `entry_frac` / `exit_frac`：**0.8 → 0.75**  
- entry **iceberg** MVP（`ask_size_frac=0.5`，`fallback_notional=8000`）

总览见 [`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md)。
