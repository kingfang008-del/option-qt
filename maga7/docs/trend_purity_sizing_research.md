# 趋势纯度打分 → 仓位缩放

想法：因果「趋势更纯」给满仓，混杂/弱信号自动缩仓。  
实现：`maga7/common/trend_purity.py`，开关 `trade.trend_purity_sizing=true`。

## 分数（开火时刻，无 EOD）

| 分量 | 权重 | 含义 |
|---|---:|---|
| \|from_prev\| / 2.5% | 35% | 隔夜/开盘动量强度 |
| peer 超额（相对 peer_min） | 25% | 同向宽度 |
| QQQ 同号 | 15% | 大盘对齐 |
| mf10 同号 | 10% | 资金流 |
| streak 超额 | 5% | 持续长度 |
| SI 同向强度 | 10% | 截面同步 |

`trend_purity_mode`：`continuous`（默认）/ `tier` / `skip_low` → 乘到 `size_frac`。

脚本：`maga7/tools/run_trend_purity_ablation.py`  
产物：`results/trend_purity_ablation_extend_mtm_peer3_may_jul/`

## 消融（May–Jul）

| 变体 | total_ret | MaxDD | scaled | 07-07~09 |
|---|---:|---:|---:|---:|
| extend_mtm_only | **+401.1%** | -16.2% | 0 | -0.50 |
| purity_cont | +380.5% | -16.2% | 7 | -0.50 |
| purity_cont_soft | +400.6% | -16.2% | 1 | -0.50 |
| purity_tier | +318.5% | -16.2% | 7 | -0.50 |
| confirm1+purity | +357.4% | **-12.7%** | 7 | -0.15（改善来自 confirm） |
| **full_day** | **+673.3%** | **-12.2%** | 0 | -0.50 |

07-07~09 **实际成交**的纯度（开火时）：

| 单 | pur | scale | 问题 |
|---|---:|---:|---|
| 07-07 NVDA DN | **0.73** | ×1.0 | 假 DN，分数仍高 |
| 07-08 META DN | **0.76** | ×1.0 | 未缩仓 |
| 07-09 AMD UP | **0.80** | ×1.0 | 晨间抢跑，\|fp\| 极大反而更高分 |

## 诊断：分数方向可能反了

同日未成交的「更像大行情」候选：

| 信号 | pur | 对比 |
|---|---:|---|
| 07-07 TSLA DN（真跌） | **0.64** | **低于** NVDA 假 DN 的 0.73 |
| 07-09 META UP（+8.9%） | **0.63** | **低于** AMD 抢跑的 0.80 |
| 07-09 AMD UP | 0.80 | fp=+7.8%、peer=5 → 看起来「更纯」 |

根因：当前纯度偏爱 **大 \|from_prev\| + 宽 peer**，而这正是早盘 Rule-A 过门的样子；午后真趋势常 **peer 刚好过线、\|fp\| 刚过 2%**，分数反而更低。  
缩仓触发不到这三日亏单（全是 ×1.0）；全期只缩了 7 笔，略伤收益、**MaxDD 不变**。

## 结论

1. **方向对、特征错**：按纯度调仓合理，但「隔夜动量+peer 宽」不等于「当日趋势纯」。  
2. **不升格**这版打分；勿指望它替代 confirm_1 / full_day。  
3. 效率/噪声版已测：[`trend_purity_efficiency_research.md`](trend_purity_efficiency_research.md)  
   （排序纠正，全期仍偏伤收益；可作降波旋钮，非默认升格）。

澄清命名：07-07 假信号是 **NVDA DN**（全日反涨）；真趋势 **TSLA DN**。07-09 真趋势是 **META UP**，不是 AMD。
