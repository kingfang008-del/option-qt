# event_calendar_block（事件日禁入）

> **full_day 机制说明 + 实盘/API 规划**见专项文档：  
> [`event_calendar_full_day.md`](event_calendar_full_day.md)  
> 07-07~09 大行情 vs 选标：[`jul7_9_mover_vs_pick_analysis.md`](jul7_9_mover_vs_pick_analysis.md)

在 `extend_mtm_only`（peer3、无 day_circuit）上按宏观/事件日历跳过整日入场。

## 配置

```json
"regime": {
  "event_calendar_block": true,
  "event_calendar": "core",
  "event_blackout_sessions": 0
}
```

| 键 | 含义 |
|---|---|
| `event_calendar_block` | 总开关 |
| `event_calendar` | `core` / `default`（或 `event_dates` 显式列表） |
| `event_blackout_sessions` | 事件日后额外禁入的交易日数（0=仅当日） |

`core`：`05-20` NVDA 财报、`06-12` SpaceX IPO、`06-17` FOMC 决议  
`default`：再加 `05-19` 长债冲击、`05-21` 财报次日、`06-16/18` FOMC 簇  

实现：`maga7/common/event_calendar.py`；replay / stream 已接线。

## 消融（对齐 archived extend_mtm_only）

```bash
python -m maga7.tools.run_event_calendar_ablation \
  --tag event_calendar_ablation_extend_mtm_peer3_may_jul
```

| 变体 | ret | MaxDD | n | 相对 extend |
|---|---|---|---|---|
| extend_mtm_only | +401.1% | -16.2% | 53 | — |
| core_day | +481.8% | -14.8% | 50 | +80pp |
| **core_plus1** | +542.8% | **-12.2%** | 46 | +142pp |
| **full_day** | **+673.3%** | **-12.2%** | 44 | **+272pp** |
| full_plus1 | +414.5% | -12.2% | 40 | 几乎打平（误伤 05-22/06-15/06-22 大胜） |

`full_day` 主要砍掉：05-20/21 连亏、06-12 SL、06-16/18 亏单；误伤较少。  
`full_plus1` 会连带禁掉大胜日，不推荐。

## 结论（研究态）

事件日禁入是目前相对 `extend_mtm_only` **少有的帕累托方向**（收益↑且 MaxDD↓）。  
**当前研究底仓：`extend_mtm_only` + `full_day`（+673% / -12.2%）**；勿用 full_plus1。  
日历 API 同步见 [`event_calendar_full_day.md`](event_calendar_full_day.md) §3；profile  
`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1`。  
未升格因果基线；日历需随样本外事件维护，存在前视标注风险（本消融日期来自已知宏观叙事）。

每日对照 CSV：  
`results/event_calendar_ablation_extend_mtm_peer3_may_jul/daily_compare_extend_vs_full_vs_core_plus1.csv`

## Live 开盘前接入

会话启动时 `run_live_session` 调用 `resolve_live_event_blackout`：

| 来源 | 用法 |
|---|---|
| profile `regime.event_calendar_block` | 回测同款 preset |
| 文件 | `MAG7_EVENT_CALENDAR_PATH` 或 `regime.event_calendar_path` → [`CONFIG/event_calendar_live.json`](../CONFIG/event_calendar_live.json) |
| Redis | key `maga7:event_blackout`（JSON 列表 / CSV / SET） |
| 强制当日 | `MAG7_EVENT_BLACKOUT_TODAY=1` |

命中则 Scanner 不发信号，OMS `day_halted` + `EVENT_BLACKOUT` 事件。
