# 周状态相似检索（Regime Match）与微调窗选择

## 背景与动机

生产周更微调默认使用 **最近 N 个月**（见 `CONFIG/weekly_finetune_policy.json`，当前 `train_lookback_months=2`）。

这在市场结构稳定时够用；但若 **regime 突然切换**（波动台阶、趋势/震荡切换、VIX 结构变化），近两个月可能全是「另一种状态」，微调会把模型推向错误局部最优。

更合理的做法不是盲目加长 lookback，而是：

1. 刻画「当前一周」的市场状态向量（无标签、无未来信息）
2. 在历史滑窗中检索 **最相似的一段/几段**
3. 与「近 N 月中心」比较，判断是否像突变
4. 输出 **建议混合训练月**，并可由 `weekly_finetune.py --suggest-train-months` 自动填入 `--train-months`

定位：**诊断 / 训练窗选择器**，不是收益预测器。  
**相似 ≠ 可赚**；匹配结果服务于训练分布与压力测试，不要直接抄旧周交易或旧参数。  
**自动建议只改 train 窗**，不绕过 OOS 门禁、不自动晋升生产权重。

---

## 工具

| 项 | 路径 |
|---|---|
| 脚本 | `qqq_btc/tools/match_week_regime.py` |
| 默认行情根 | `~/train_data/spnq_train_resampled/{QQQ,VIXY}/regular/09:30-16:00/1min/` |
| 微调入口（下游） | `qqq_btc/tools/weekly_finetune.py` |
| 微调策略 | `qqq_btc/CONFIG/weekly_finetune_policy.json` |

### 特征原则

- 仅用 **QQQ + VIXY 1 分钟现货**，不含期权 ROI / 入场收益标签（避免泄漏）
- 先聚合成交易日统计，再聚合成「查询窗 / 历史窗」向量
- 默认检索模式：`rolling` = 与 query **同长度交易日**的滚动窗（比 ISO 自然周更公平）
- 距离：历史窗标准化后的 **cosine distance**（可选 euclidean）

### 周向量主要维度

- QQQ：日收益均值/波动、绝对收益、日内 range、隔夜 gap、上涨日占比、分钟路径趋势斜率与 R²、成交量 log
- VIXY：level 均值/波动/窗内变化、日收益波动、`vix_z` 均值/波动、约 30 分钟反转强度

完整键名见脚本内 `FEATURE_KEYS`，以及每次运行产物 `regime_match.json` → `settings.feature_keys`。

---

## 用法

```bash
# 查询一周（含端点），对比近 2 月，写 Top-K
python qqq_btc/tools/match_week_regime.py \
  --query 2026-07-01:2026-07-08 \
  --recent-months 2 \
  --top 15 \
  --out qqq_btc/results/regime_match_jul_w1_0701_0708

# 只打印建议 train months CSV（机器可读）
python qqq_btc/tools/match_week_regime.py \
  --query 2026-07-01:2026-07-08 \
  --emit-train-months --apply blend_on_shift

# 可选
# --mode calendar_week   # ISO 周（一般不如 rolling）
# --metric euclidean
# --apply always_blend   # 总是近月∪相似月
# --history-start 2024-01-01
# --spot-root ~/train_data/spnq_train_resampled
```

### 接到 `weekly_finetune.py`（自动 `--train-months`）

默认仍是近 N 月。打开建议后，**在未显式传 `--train-months` 时**由 regime match 填窗：

```bash
# dry-run：只解析窗 + 写 regime 报告，不训练
python qqq_btc/tools/weekly_finetune.py \
  --suggest-train-months \
  --regime-query 2026-07-01:2026-07-08 \
  --dry-run

# 正式跑（仍走 OOS 门禁；显式 --train-months 优先于 suggest）
python qqq_btc/tools/weekly_finetune.py \
  --suggest-train-months \
  --regime-query 2026-07-01:2026-07-08 \
  --regime-apply blend_on_shift
```

Policy 开关（`CONFIG/weekly_finetune_policy.json` → `regime_match`）：

| 字段 | 含义 |
|---|---|
| `auto_suggest` | `true` 时等价于总是开 `--suggest-train-months`（仍要有 `query`） |
| `query` | 默认查询窗；可被 CLI `--regime-query` 覆盖 |
| `apply` | `blend_on_shift`（默认）/ `always_blend` / `suggest_only` |
| `recent_months` / `top` / `mode` / `metric` / `spot_root` | 传给 `match_week_regime` |

`apply` 语义：

| 模式 | 行为 |
|---|---|
| `blend_on_shift` | NEAR_RECENT → 近 lookback 月；SHIFT/STRONG_SHIFT → 近月 ∪ Top 相似月 |
| `always_blend` | 总是用 `suggested_train_months_example` |
| `suggest_only` | 与 always_blend 选出同样 CSV，但语义上表示「仅建议」（weekly 仍会采用该列表当未手写 train-months） |

过滤：建议月会与 `feature_train_root/1min` **可用月求交**；缺失月写入 `dropped_unavailable`。  
产物：`<run_dir>/regime_match/regime_match.json`、`train_months_suggestion.json`，并写入 `manifest.json` / `summary.json` 的 `regime_suggestion`。

### 产物

| 文件 | 内容 |
|---|---|
| `regime_match.json` | query 特征、vs 近月距离、Top-K、recommendation |
| `top_matches.csv` | Top-K 简表（便于表格查看） |

### 如何读 `vs_recent` / `recommendation`

| 字段 | 含义 |
|---|---|
| `distance_to_recent_centroid` | query 到「近 N 月候选窗中心」的距离 |
| `distance_to_best_recent_window` | query 到近月内最近一窗的距离 |
| `hist_distance_median / p25 / p75` | 全历史距离分布，用来判断「近月中心是否偏远」 |
| `regime_shift_flag` | 近月中心距离 **> 历史中位距离** → 倾向认为更不像近月 |
| `strong_shift_flag` | 近月中心距离 **> 历史 p75** → 强切换提示 |
| `blend_similar_months` | Top-K 按距离加权投出的历史月 |
| `suggested_train_months_example` | 近月 ∪ 相似月 的示例混合窗（**建议，不自动晋升**） |
| `action` | `NEAR_RECENT` / `SHIFT` / `STRONG_SHIFT` 文案建议 |

### 建议工作流

```text
本周结束或开盘前
    → weekly_finetune.py --suggest-train-months --regime-query <本周> --dry-run
       （或先单独跑 match_week_regime.py）
    → 读 action / train_months_suggestion
    → 确认后去掉 --dry-run 正式微调
    → 仍走 OOS 门禁，不通过不晋升生产权重
    → 若要强制手写窗：--train-months 2025-12,2026-01,...（优先于 suggest）
```

手动指定窗（跳过自动建议）：

```bash
python qqq_btc/tools/weekly_finetune.py \
  --train-months 2025-12,2026-01,2026-05,2026-06 \
  --val-months 2026-06
```

---

## Jul W1 试跑快照（2026-07-01 .. 2026-07-08）

运行时间：2026-07-14；产物目录：

`qqq_btc/results/regime_match_jul_w1_0701_0708/`

### 结论摘要

| 项 | 值 |
|---|---|
| Query 交易日数 | 5 |
| 近月列表 | 2026-05, 2026-06 |
| dist(query, recent_centroid) | 0.585 |
| dist(query, best_recent_window) | 0.401（`2026-06-22..2026-06-26`） |
| 历史距离 median / p25 / p75 | 0.980 / 0.782 / 1.183 |
| `regime_shift_flag` | **false** |
| `strong_shift_flag` | **false** |
| Action | **NEAR_RECENT**：可维持近 2 月微调，并用 Top-K 做压力测试 |

含义：就现货状态向量而言，Jul1–8 **并不明显偏离** 近两个月中心；「固定近 2 月 FT」对这一周并非明显错误。  
但仍可参考更像的历史段做混合或压力测试。

### Top 匹配（节选）

| Rank | Distance | 窗口 | 月份 |
|---:|---:|---|---|
| 1 | 0.297 | 2025-12-29 .. 2026-01-05 | 2025-12, 2026-01 |
| 2 | 0.316 | 2024-02-29 .. 2024-03-06 | 2024-02, 2024-03 |
| 3 | 0.326 | 2025-12-31 .. 2026-01-07 | 2025-12, 2026-01 |
| 4 | 0.364 | 2024-07-11 .. 2024-07-17 | 2024-07 |
| 7 | 0.401 | 2026-06-22 .. 2026-06-26 | 2026-06 |

加权建议混合月：`2026-01`, `2025-12`, `2024-07`, `2026-06`  
示例训练月：`2025-12, 2026-01, 2026-05, 2026-06`

---

## 与「固定近 2 月微调」的关系

| 做法 | 评价 |
|---|---|
| 永远近 2 月 FT | 稳态 OK；突变时危险 |
| 永远更长历史 FT | 稀释近期，也不保证对 |
| **相似历史窗检索 + 混合 FT / 压力测试** | 推荐；本脚本的目标用法 |
| 只靠相似周直接复用 end240/hold55 等周特化参数 | **不建议**（易过拟合单周） |

Jul W1 离线 replay 调参拉到约 +49%（`ft56_julw1_end240_hold55_hardcap_20260713`）属于 **参数/路径优化结果**，与本脚本的「状态相似」是两层问题：  
先认清本周像谁，再决定训什么数据；收益参数仍要 **诚实 KPI**（诚实特征 + 因果 put_gate + LIVE）与流式三闸门验收。

详见：[`honest_live_kpi_finetune_replay.md`](./honest_live_kpi_finetune_replay.md)（含完整微调脚本与配置路径）。

---

## 已知局限与后续

1. **一周噪声大**：单周匹配不稳定；可滚动用「本周迄今 + 近 5～10 日」，或要求连续两周同簇再换窗。
2. **现货状态 ≠ 期权微观结构**：未纳入点差、合约 ladder、0DTE 流动性；必要时可在有长历史 `quote_features_raw` 时扩展可选特征。
3. **自动建议已接入 `weekly_finetune`**：`--suggest-train-months` / policy `regime_match`；默认 `auto_suggest=false`，需显式打开。建议月若特征根缺失会被丢弃。
4. **不替代诚实对拍**：流式 vs 离线分叉（特征/门控语义）仍按三闸门与共享门控库处理。

---

## 相关文档

- `qqq_btc/CONFIG/weekly_finetune_policy.json` — 近月微调与 OOS 门禁
- `qqq_btc/docs/honest_live_kpi_finetune_replay.md` — 诚实特征 + 因果 put_gate + LIVE 的微调/replay KPI
- `qqq_btc/docs/honest_3gate_live_parity_handoff.md` — 诚实流式对拍
- `qqq_btc/ARCHITECTURE.md` — 整体路径
