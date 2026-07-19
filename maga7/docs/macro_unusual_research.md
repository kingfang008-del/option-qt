# Macro / Unusual-Activity 候选层（研究设计）

**日期：** 2026-07-18  
**状态：** research only · **默认 off** · **不进** `peer3_v1`  
**动机：** 人眼日 K 看得出的 META/NVDA 类大行情，earliest Rule-A + TopK 经常抓不住；探索「放量 + 结构」宏观视角能否**更早标出候选**（先不替换成交）。

## 设计原则

1. **只出候选**，不改 freeze TopK / 不改 QQQ 门。  
2. **因果 1m**：用截至当根的 cum$ 与同刻历史中位比，禁止用 EOD 信息。  
3. **与 Hunt 分流**：Hunt = 早盘洗盘收回；Macro = 趋势/缺口修复异动。  
4. 通过标准：先看「能否定位」，再谈「期权能否兑现」与双窗 scoreboard。

## 规则（v1）

| 条件 | 默认 |
|------|------|
| 窗 | 10:30–14:00 |
| 相对放量 | cum$ / 近 10 日同刻中位 ≥ **1.20** |
| 结构 | close ≥ day open 且 close ≥ session VWAP（DN 对称） |
| 动量 | \|from_prev\| ≥ **1%** |
| 排序分 | `vol_ratio × \|fp\|`（池内 rank） |

代码：`maga7/common/macro_unusual.py`  
扫描：`python -m maga7.tools.scan_macro_unusual_candidates`  
产物：`maga7/results/macro_unusual_scan/`

## Focus：07-08 NVDA / 07-09 META

默认参（vr≥1.2, fp≥1%）May–Jul 扫描：

| 日 | 标的 | 是否标出 | 首次 macro | vs Rule-A | 池内 rank | 备注 |
|----|------|:--------:|------------|-----------|----------:|------|
| 07-09 | **META UP** | **是** | **12:04** | Rule-A **12:21**（早 **17m**） | **#3** | regime ok；#1 仍是 **AMD**（缺口+放量） |
| 07-08 | **NVDA UP** | **否** | — | Rule-A 12:58 | — | 全日 cum$≈历史中位（ratio~1.0），**不是放量日** |

07-09 当日 macro top3：AMD UP@10:30 → NVDA DN@10:30（QQQ 挡）→ META UP@12:04。  
→ 宏观层能**更早看见 META**，但仍排在 AMD 后；**不能单独靠放量把名额从 AMD 抢走**。

### NVDA 07-08 敏感度

| 设定 | 最早标出 |
|------|----------|
| 默认 vr1.2 | 无 |
| vr≤1.0（放弃「放量」）+ fp≥1% | ~12:22（仍晚，且接近 Rule-A） |
| vr1.0 + fp≥0.5% | ~11:05（早，但阈值过松，易噪音） |

**结论：** 07-08 NVDA 是 **动量加速 + QQQ 不对齐**，不是「明显放量异动」；宏观放量层定位力弱，瓶颈仍在 regime / 信号定义。

## 全窗粗覆盖（May–Jul）

- \|EOD fp\|≥3% 的大行情日票：macro 同向命中率约 **55%**（扫描口径，非成交）。  
- 有候选日 ~45 / 候选行 127——密度不低，直接塞进 TopK 有灌噪音风险。

## 建议的架构接法（尚未实现）

```text
L0 earliest Rule-A TopK     （基线不动）
L-macro 候选板              （本层：只记录 / Dash 显示）
可选：与 topk_backfill 组合 —— 仅当 TopK#2 被 regime 挡时，
      允许 macro 池内高分同向票顺延（需双窗 ablation）
```

**不要：** 为这两天松 QQQ、降 Rule-A `from_prev`、或把 macro 默认 on。

## 下一步

1. Dash/日志：日终打印 macro 候选 vs 实际成交（观测一周）。  
2. 小 ablation：`topk_backfill` × macro 顺延（强约束：仅挡后回填）。  
3. 期权反事实：对 macro 首次时点强制入场（07-09 META@12:04）是否优于 12:21——已知 12:22 强制仍约 −8%，需单独再验。
