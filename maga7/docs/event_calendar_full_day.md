# full_day 事件禁入：机制、实盘复现与 API 规划

## 1. full_day 是什么（机制）

`full_day` **不是**策略自动从行情里识别“今天有宏观冲击”。

它是一张**事先标注的交易日黑名单**：这些日期 **整日禁止开新仓**（`event_blackout_sessions=0`，不加次日）。

配置等价于：

```json
"regime": {
  "event_calendar_block": true,
  "event_calendar": "default",
  "event_blackout_sessions": 0
}
```

### 黑名单日期（May–Jul 2026 研究窗）

| 日期 | 标签 | 类型 |
|---|---|---|
| 2026-05-19 | 30Y 美债收益率冲高 / Mag7 杀估值 | 半突发（利率） |
| 2026-05-20 | NVDA 盘后财报 | **可预知** |
| 2026-05-21 | 财报次日消化 | **可预知**（+1） |
| 2026-06-12 | SpaceX IPO | **可预知** |
| 2026-06-16 | FOMC 会议日 | **可预知** |
| 2026-06-17 | FOMC 决议日 | **可预知** |
| 2026-06-18 | FOMC 次日 | **可预知**（+1） |

实现：`maga7/common/event_calendar.py` → replay / stream / live Scanner+OMS。  
命中则当日 `skip` 全部入场；**不读新闻正文，不做实时宏观打分**。

### 与 core / plus1 的区别

| 变体 | 集合 | +次日？ |
|---|---|---|
| `core_day` | 仅 05-20 / 06-12 / 06-17 | 否 |
| `core_plus1` | 同上 | 是（会误伤 06-15 大胜） |
| **`full_day`** | 上表 7 日 | 否 |
| `full_plus1` | 7 日 | 是（误伤过多，不推荐） |

消融成绩（`extend_mtm_only` 底仓）：见 [`event_calendar_block_research.md`](event_calendar_block_research.md)。  
`full_day` May–Jul：**+673% / MaxDD -12.2%**（底仓 +401% / -16.2%）。

---

## 2. 实盘能否用外部数据复现？

**可以复现“日历禁入”机制；不能无损复制“事后标注的完美名单”。**

| 事件类型 | 外部数据 | 开盘前可复现？ |
|---|---|---|
| FOMC | Fed / 经济日历 API | 高 |
| Mag7 财报 | Nasdaq / earnings calendar API | 高（AH 日建议禁当日，可选次日） |
| 巨型 IPO | 交易所 / 投行日历 | 中高 |
| 长债暴冲、地缘突发 | 新闻 / 利率 tick | **低**（需代理规则，允许漏挡） |

### 已接线的 live 入口

会话启动：`resolve_live_event_blackout`（`run_live_session`）

| 来源 | 用法 |
|---|---|
| profile | `regime.event_calendar_block` + preset |
| 文件 | `MAG7_EVENT_CALENDAR_PATH` → [`CONFIG/event_calendar_live.json`](../CONFIG/event_calendar_live.json) |
| Redis | `maga7:event_blackout`（JSON / CSV / SET） |
| 强制当日 | `MAG7_EVENT_BLACKOUT_TODAY=1` |

| 范围 | 行为 |
|---|---|
| **full-day**（FOMC / NFP / CPI / 无 symbol 的宏观） | OMS `day_halted`，全日不入场 |
| **symbol**（`earnings_*` / `news_*` + ticker） | **只禁该标的**；其它 Mag7 照常；OMS 不停全日 |

**2026-07-26：** live 补丁 `2026-07-22` TSLA/GOOGL `earnings_ah`（Finnhub 有、07-20 sync 漏）。下次 `sync_event_calendar` 请带 manual，或核对 `symbol_blackout` 未被冲掉。

Live 文件字段：`dates`=full-day；`symbol_blackout`=`{date:[SYM,…]}`；`events`=明细。

运维说明：[`live_session_operations.md`](live_session_operations.md)。

---

## 3. API / 日历同步（Phase A 已落地）

目标：开盘前自动生成接近 `full_day` 的黑名单，尽量少人工改 JSON。  
**研究底仓**现以本机制为准：`extend_mtm_only` + `full_day`（约 +673% / -12.2%）。  
Profile：[`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json)

### Phase A — 可预知日历（已实现）

| 组件 | 路径 |
|---|---|
| Providers | [`common/event_providers.py`](../common/event_providers.py) |
| Sync CLI | [`tools/sync_event_calendar.py`](../tools/sync_event_calendar.py) |
| 人工补漏 | [`CONFIG/event_calendar_manual.json`](../CONFIG/event_calendar_manual.json) |
| Live 输出 | [`CONFIG/event_calendar_live.json`](../CONFIG/event_calendar_live.json) |

```bash
# Key（二选一，代码会自动读，无需每次 export）
#   1) 本机文件：~/finnhub.txt  （或 ~/.config/maga7/finnhub.txt）
#   2) 环境变量：FINNHUB_API_KEY / FINNHUB_KEY
# 亦可：POLYGON_API_KEY / MASSIVE_API_KEY（Benzinga earnings）

python maga7/tools/sync_event_calendar.py \
  --start 2026-05-01 --end 2026-12-31 \
  --redis   # 可选：写入 maga7:event_blackout

export MAG7_EVENT_CALENDAR_PATH=maga7/CONFIG/event_calendar_live.json
```

规则摘要：

1. **FOMC**：抓 Fed 页面；失败则用内置 2025–2026 日程。默认会议日 + 决议日。  
2. **Earnings**：Finnhub / Polygon；AH → 当日 blackout（可选 `--earnings-ah-plus1-cal`）。  
3. **Manual**：IPO / 利率冲击 / AH+1 等 API 盖不住的日期。  

Live 合并：`resolve_live_event_blackout` = profile preset ∪ 文件 ∪ Redis ∪ `MAG7_EVENT_BLACKOUT_TODAY`。  
Offline 研究复现 +673%：仍用 profile `event_calendar=default`（7 日 curated），**不**自动读 live 文件。

### Phase B — 突发代理（待做）

- 隔夜 / 盘前 **US30Y** 或 **TNX** 跳升超阈 → 当日 blackout  
- 或 VIX 隔夜涨幅超阈  

### Phase C — 公司新闻（已接线，无 LLM）

开盘前 `sync_event_calendar` 默认拉取：

1. **Finnhub** `company-news`（Mag7 逐票，近 5 日）  
2. **Investing RSS**（与 `~/notebook/rss_feed/rss_feed_stable.py` 同源：`news_356.rss`）

规则（见 [`event_news_policy.py`](../common/event_news_policy.py)）：

| 层 | 行为 |
|---|---|
| 宏观 FOMC/NFP/CPI | **full-day** 禁入 |
| Mag7 财报 | **symbol** 禁入 |
| CEO 交接（`hard_risk`） | **symbol** 禁入 |
| 大单/合作/capex/fab | 只审计；可选 LLM 利好/空提示 |
| 交易方向 | **永不**由新闻/LLM 设定 |

- 默认 `MAG7_NEWS_MODE=hard_risk`（`blackout` 同义）；纯打分用 `audit`  
- Dash LLM → `event_news_llm.json`，**不进** live 黑名单  
- 路径亏损继续靠 tox / 仓位 / 时间止损

---

## 4. 未覆盖的亏损窗

`full_day` **不包含** 07-07~09。该三日跨标的连亏见专项分析：  
[`jul7_9_mover_vs_pick_analysis.md`](jul7_9_mover_vs_pick_analysis.md)。

---

## 5. 使用注意

- 回测成绩含**事后标注**成分；样本外必须维护日历或 API。  
- 勿把 `full_day` 名单直接写进冻结因果基线，除非配套自动日历管道。  
- `event_blackout_sessions≥1` 易误伤大胜日，默认保持 0，次日用显式日期表达。
