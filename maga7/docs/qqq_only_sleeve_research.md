# QQQ-only sleeve（短 DTE 可行性验证）

**日期：** 2026-07-18  
**角色：** `research_candidate` —— **不**并入 Mag7 `research_baseline`

## 目的

在期权结构干净（日频短 DTE）的 QQQ 上，验证同一套：

- L0：open_ladder OTM5 + Rule-A + T30→T45 `hold_extend` + 事件日历  
- L2：`washout_reclaim` Hunt  

是否仍可跑通。Mag7 peer3 / `qqq_align` / Mag7 广度 Halt **不照搬**。

## Profiles

| Profile | 内容 |
|---------|------|
| `qqq_only_open_ladder_atm5otm_extend_mtm_v1.json` | L0：Watchdog off · peer off · `qqq_align=false` · `mf_idio=off` |
| `qqq_only_open_ladder_atm5otm_extend_mtm_watchdog_hunter_v1.json` | L0+Hunt：`wash_drop_min=0.006`（ETF 标定）· Halt/Degrade off |

数据路径：`open_locked_map=…_maga7_googl_qqq_…` · `quote_1s_root=…/maga7_mf10_open_ladder_otm5/QQQ`

## 与 Mag7 基线的刻意差异

| 项 | Mag7 基线 | QQQ sleeve |
|----|-----------|------------|
| `peer_align_min` | 3 | **0** |
| `qqq_align` | true | **false**（自对齐无意义） |
| L1 Halt/Degrade | on | **off**（广度/假收回依赖 Mag7） |
| Hunt `wash_drop_min` | 1.5% | **0.6%**（1.5% 在 QQQ 上 5–7 月 0 触发） |
| 仓位 | pos 0.2 / concurrent 2 | pos 0.25 / concurrent 1 |

## 工程修复（Hunt-only 日）

Offline replay 原先只遍历 **有 Rule-A** 的日期；Hunt 日若无 Rule-A 则永不注入。  
已在 `replay.py` 对 `hunter.enabled` 补日历日。Mag7 May–Jul **无**此类日（不影响 +1255%）。

## May–Jul 结果（→07-13，锁约覆盖上限）

```bash
python -m maga7.tools.run_qqq_only_sleeve_scoreboard \
  --out maga7/results/qqq_only_sleeve_may_jul
```

| 档 | total_ret | vs L0 | n | Hunt | Hunt 笔均 | MaxDD |
|----|----------:|------:|--:|-----:|----------:|------:|
| L0 | **+66.5%** | 100% | 10 | 0 | — | −4.1% |
| L0+Hunt | **+91.7%** | **~115%** | 12 | **2** | **+30.7%** | −4.3% |

Hunt 成交：05-15（T+45，−4.1%）· 06-26（TP，+65%）。06-12 有候选未成交。  
**解读：** 短 DTE 上 L0 路径可成交；ETF 标定后的 Hunt 有增量（样本仅 2 笔，只作可行性，不作升线）。

## 数据缺口

| 窗 | 状态 |
|----|------|
| May–Jul（→07-13） | lock + 1s 有（49 日） |
| Feb–Apr | **无** QQQ open_lock / 1s → 双窗暂不可跑 |
| 07-14..17 | 锁约未覆盖 |

补 Feb–Apr 锁约+quote 后复跑同一 scoreboard。

## 非目标

- 不把 QQQ sleeve 并进 `peer3_v1`  
- 不把 Mag7 Hunt 阈值改成 0.6%  
- 不把「QQQ +66%/+92%」与 Mag7 +1255% 直接比绝对值  
