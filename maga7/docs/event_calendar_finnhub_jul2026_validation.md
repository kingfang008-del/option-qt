# Finnhub 七月验证（2026-07）

Key 落盘（推荐，无需每次 `export`）：

```bash
# 已支持自动读取：~/finnhub.txt 或 ~/.config/maga7/finnhub.txt
# 或环境变量 FINNHUB_API_KEY（勿提交进 git）
printf '%s\n' 'YOUR_TOKEN' > ~/finnhub.txt && chmod 600 ~/finnhub.txt
```

```bash
python maga7/tools/sync_event_calendar.py \
  --start 2026-07-01 --end 2026-07-31 \
  --out maga7/results/event_calendar_finnhub_jul2026.json \
  --no-manual
```

### May–Jun 验证（2026-05-01..06-30）

| 来源 | 结果 |
|---|---|
| Finnhub earnings | **0 条**（Free 档回不去 5–6 月历史） |
| FOMC builtin | 05-05/06、06-16/17 |
| manual | 05-19/20/21、06-12、06-18 |
| **合并 vs research full_day** | **7/7 全覆盖**；额外 05-05/06（5 月 FOMC） |

产物：`results/event_calendar_finnhub_may_jun2026.json`

产物：`results/event_calendar_finnhub_jul2026.json`（验证时已生成）。

## 结果（Free 档 REST）

| 日期 | 来源 | 事件 |
|---|---|---|
| 2026-07-22 | finnhub | **GOOGL / TSLA** earnings AH |
| 2026-07-28 | fomc_builtin | FOMC meeting |
| 2026-07-29 | finnhub + fomc | **META / MSFT** earnings AH + FOMC decision |
| 2026-07-30 | finnhub | **AAPL / AMZN** earnings AH |

- Finnhub 返回 Mag7 池内 **6** 条财报（皆 `amc` → `earnings_ah`）。  
- **NVDA / AMD** 在 7 月窗无财报行（可能在 8 月或未排期）。  
- 与研究 `full_day` 7 日名单：**无交集**（full_day 覆盖的是 5–6 月事件；7 月是向前 live 日历）。

## Free 档限制（已实测）

| 查询 | 结果 |
|---|---|
| Mag7 `2026-07-01..07-31` | 有数据 |
| NVDA / Mag7 `2026-05` 或 `2026-06` | **空**（无法回填研究窗 05-20 NVDA） |

→ 历史研究名单仍靠 `event_calendar=default` + `event_calendar_manual.json`；  
→ **向前 live** 用 Finnhub 拉「当前月～下周」即可。

## Webhook

本次只用 REST calendar。Finnhub Webhook 适合推送变更；若要接，可另做订阅写入 Redis `maga7:event_blackout`，不阻塞 Phase A。

## 安全

API key / webhook secret **不要**提交进 git 或写进 JSON profile。若曾粘贴到聊天，建议在 Finnhub 控制台 **轮换 key**。
