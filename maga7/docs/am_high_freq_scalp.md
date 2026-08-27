# AM 高频高胜率扫描（一天几十笔 × ~5%）

目标：提高交易次数与胜率，单笔可降到 ~5%。  
工具：`maga7/tools/scan_am_high_freq_scalp.py`  
结果：`/mnt/s990/data/maga7/results/research_am_high_freq_scalp/`

## 结论（硬）

**在现有因果 FO/VWAP 入场 + 快 TP/SL（trade-mark）下，做不到「一天几十笔 + 均笔~5% + 赚钱」。**

| 档位 | 笔/日 | 胜率 | 均笔 | 双窗 |
|---|---:|---:|---:|---|
| 密扫 mild_fo 等（v1） | 190–420 | 32–53% | **≈ −2.0%** | 全负 |
| top1\|FO\|/分钟 + 确认（v2） | 15–50 | ~48–53% | **≈ −1.3%～−1.8%** | 全负 |
| 现行稀 keep `vd_soft+ride` | ~1.1 | **60%** | **+7.7%** | 正 |

笔数上去后，均笔稳定为负；胜率最好也只到 ~53%，盖不住 SL/超时。  
`econ_dual` / `dense_ok`：**0**。

## 为什么「看起来该很多」却不行

1. 前视探针本身 ~900/日，但那是**诊断网格**（多标的×双向×每分钟），不是可交易边。  
2. 放宽到 mild FO 对齐后 raw 候选 ~400/日，快兑现 TP4–6% → **均笔约 −2%**。  
3. 每分钟只做 |FO| 最强 + 期权 15–20s 确认，能压到 20–50 笔/日，均笔仍负。  
4. 之前口袋 densify 已提示：ungated ~9/日就接近打平/亏损；再加密只会更差。

算术上「30 笔 × +5%」很香，但因果规则给不出 +5% 均笔；高频档实际是 **~−1.5%**。

## promote

```text
promote = NONE（高频目标未达成）
keep    = RETREAT_vd_soft_ride   # 稀、正期望
```

若仍要密度，只能接受：**更多笔 + 更低/负均笔**，或换全新特征/确认机制（本轮未找到）。

## 复现

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_high_freq_scalp \
  --tag research_am_high_freq_scalp
# v2 见同目录 v2_scoreboard.csv / v2_summary.json
```
