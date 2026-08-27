# 2026-02~03：大行情漏抓扫描（tt600d baseline）

> 工具：`python -m maga7.tools.scan_missed_movers --start-date 2026-02-01 --end-date 2026-03-31 --tag missed_movers_feb_mar`  
> 产物：`maga7/results/missed_movers_feb_mar/`  
> 对照成交：同窗 offline replay → total_ret **+35.3%** / MaxDD −21.7% / n=50

问题：弱窗只有 +35% 是否因为「没有大行情」？**不是。**

## 1. 总量

| 指标 | 值 |
|------|---:|
| 交易日 | 41 |
| 大行情 symbol-day（\|day_ret\|≥2%） | 85 |
| **同向抓住** | **20（23.5%）** |
| 漏抓 | 65 |
| 「当日最大行情」未成交天数 | **32 / 41** |
| \|day\|≥3% 抓住率 | 11/36（30.6%） |
| \|day\|≥4% 抓住率 | 6/17（35.3%） |
| \|day\|≥5% 抓住率 | 5/11（45.5%） |

## 2. 漏抓原因（symbol-day，\|day\|≥2%）

| miss_reason | n | 含义 |
|-------------|--:|------|
| `no_rule_a` | 20 | 信号窗内无 Rule-A |
| `topk_full_rank3+` | 31 | 有资格但来得晚，TopK=2 已满 |
| `eligible_topk_but_no_fill` | 9 | 进了 TopK / 扫描 eligible，实际无成交 |
| `regime:qqq_align_*` | 4 | QQQ 对齐挡掉 |
| `peer_fail` | 1 | peer&lt;3 |

主因仍是 **TopK 占坑 + 无 Rule-A**，与 Apr–Jul 漏抓结构同类；regime 挡相对较少。

## 3. 高价值漏抓（\|day\|≥4% 且未同向成交）

| 日 | 标的 | day_ret | 原因 |
|----|------|--------:|------|
| 02-05 | GOOGL | +7.3% | `no_rule_a` |
| 03-09 | AMD | +6.3% | `eligible_topk_but_no_fill`（rank1） |
| 03-19 | AMD | +5.5% | `no_rule_a` |
| 02-09 | AMD | +5.0% | `eligible_topk_but_no_fill`（rank1） |
| 02-12 | AMD | −5.2% | `topk_full_rank4` |
| 02-06 | AMD | +5.2% | `topk_full_rank3`（当日抓了 NVDA/TSLA） |
| 03-30 | AMD | −4.8% | `regime:qqq_align_dn` |
| 02-26 | NVDA | −4.4% | `topk_full_rank4` |
| 03-02 | AMD | +4.0% | `eligible_topk_but_no_fill`（Rule-A 却是 **DN**，与 EOD UP 反） |
| 03-31 | NVDA | +4.1% | `peer_fail` |

## 4. 为何抓住了正股日行情，账户仍不夸张

同窗「\|stock day\|≥3% 且同向成交」共 **11** 笔期权：

- 均 ret **+10%**，胜率仅 **45%**
- 反例：02-06 NVDA 正股 **+5.7%**，期权 T+30 **−9%**；03-31 GOOGL 正股 +3.3%，期权 **−26%**
- 正例：03-26 AMD/META DN 吃到 TP（正股 −7.6% / −5.4%）

→ 波动市里 **正股大波动 ≠ 期权窗口内可兑现**；+35% 不是「没有行情」，而是 **抓到率低 + 抓到后期权兑现差**。

## 5. `eligible_topk_but_no_fill` 解剖（9 笔）

> 产物：`results/missed_movers_feb_mar/nofill_autopsy.csv`  
> 对照：当日 offline replay + `open_lock` / `quote_1s` 存在性

| 日 | 标的 | day_ret | 根因 | 说明 |
|----|------|--------:|------|------|
| 02-02 | AMD | +3.1% | **结构性：无 0/1/2 DTE** | option_1m 有，但 `min_dte=4`；freeze 锁不出 |
| 02-09 | AMD | +5.0% | **结构性：无 0/1/2 DTE** | `min_dte=4`；仅 MSFT 成交 |
| 02-24 | AMD | +2.2% | **结构性 + Tue confirm** | `min_dte=3`；confirm 也会挡 |
| 03-02 | AMD | +4.0% | 结构缺口 + **方向反** | Rule-A=DN，EOD=UP |
| 03-09 | AMD | +6.3% | **结构性：无 0/1/2 DTE** | `min_dte=4`；弱窗最高价值缺口仍不可交易 |
| 03-23 | AMD | −2.1% | 结构缺口 + **方向反** | Rule-A=UP，EOD=DN |
| 02-20 | GOOGL | +3.1% | **仅缺 open_lock（已修）** | 见 §5.1：补 lock 后 **TP +65.8%** |
| 02-03 | NVDA | −3.0% | **扫描假阳性** | 与 AMZN 同刻；引擎按 `symbol` 字母序 AMZN 占 TopK#2 |
| 03-31 | TSLA | +2.6% | **Tue confirm 挡** | quote+lock 齐；`entry_confirm` mf 失败 |

**归类计数**

| 根因 | n | 动作 |
|------|--:|------|
| AMD 周一/周二无短到期（`min_dte≥3`） | 6 | **不是缺 quote 下载**；与 [`amd_1s_gap_backfill.md`](amd_1s_gap_backfill.md) 同族。勿放宽 `allowed_dte` |
| GOOGL Feb lock 未建（quote 已有） | 1 | **已重建并 merge**（2026-07-19） |
| TopK 平局（扫描误标 no-fill） | 1 | 改扫描 tie-break 与引擎一致（`sig_ts, symbol`） |
| TT1 周二 confirm | 1 | 故意挡，不改 |

### 5.1 GOOGL Feb lock 补齐（已做）

- 新建 18 个交易日 lock（360 行，`otm_rungs=5`）→ merge 进  
  `~/train_data/locked_targets_map_maga7_googl_open_ladder_atm5otm_jan_jul.parquet`  
- 备份：`…parquet.bak_pre_googl_feb_20260719_231527`  
- 02-20：24/24 锁约 quote 已齐；replay **GOOGL UP TP +65.8%**（另 AMZN TP）  
- Feb–Mar 弱窗复跑：`+35.3%` → **`+39.9%`**（MaxDD −21.7%→−23.9%；多出的 GOOGL 含 02-17 SL）  
- 产物：`results/research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_tt600d_feb_mar_googl_feb_lock/`

### 5.2 AMD 为何「补 quote」无效

Feb–Mar 缺 quote 的 16 个交易日几乎全是 **周一/周二**。上游 `option_1m` 在这些日 **最短到期 ≥3–4 个交易日**（周度链尚未进入 freeze 的 `{0,1,2}` 窗）。  
→ `build_open_lock_map` 行数 = 0 → 无合约可下 1s。记为 **数据/日历不可交易**，不是策略漏抓。

## 6. 结论

1. 2–3 月 **有** 个股大行情（单日 ±4–7% 多次），不是死水。  
2. 策略同向抓住率约 **1/4**；最大单日行情多数天没成交。  
3. 「eligible 无 fill」：AMD 侧是 **短 DTE 结构性缺失**；GOOGL 02-20 是 lock 地图滞后——**已修**，弱窗约 **+35%→+40%**。  
4. 再抬弱窗：勿再指望补 AMD Mon/Tue quote；可谈 TopK 回填，勿开 TCN / `all_first`。  
5. 抓到后仍要防「正股对了、期权窗口亏」的水床。
