# AM 口袋：延迟入场 / Quote-ready

> 工具：`tools/scan_am_pocket_quote_ready.py`  
> 产物：`results/research_am_pocket_quote_ready/`  
> 前置：`docs/am_pocket_combo_opt.md`、`docs/am_pocket_flip_validation.md`

## 为什么做

冻结口袋 trade-last dual PASS，但 FillSpec quote dual 因开盘 NBBO 覆盖失败。  
回到 AM 策略本身，问两件可执行的事：

1. **能不能晚一点进？** 信号仍在口袋时刻，入场推迟 30–300s（等报价 / 等确认）  
2. **能不能等首笔可用 NBBO？** `max_wait` 内碰到合格 bid/ask 再 FillSpec TP8

## 裁决

**`TRADE_DELAY_OK_QUOTE_FAIL`**

| 族 | 结果 |
|----|------|
| trade delay=0（冻结） | dual PASS，disc +44% / blind +5% |
| trade delay≥30s | **全部 dual FAIL**；30s 已双窗转负 |
| quote wait 5–1800s | fill≤20%，disc≤0，**无 dual PASS** |

覆盖：champ 35 笔中位首笔 quote lag **≈1809s（~10:00）**；≤60s 仅 **20%**。

## 含义（对 AM 优化）

1. **口袋 edge 钉在开盘瞬间**——不能用“等报价 / 延迟确认”把策略挪到有历史 quote 的时段；一拖就没了。  
2. **历史 1s quote 库验不了这条袖**（与 flip 文档一致）；不是规则坏，是数据窗裁掉了 09:30–10:00。  
3. **Live/IB 实时 NBBO 与历史库不是一回事**——升 shadow 可以走实时报价路径，但离线 FillSpec dual 在补齐 morning quote 之前 **不能**当验收闸。  
4. 旧 FO@0.8% Pulse 因果已大亏；口袋是目前唯一 dual PASS 的 AM 候选，但 **尚未可 quote 验收**。

## 冻结建议（AM 主线）

**已采纳：离线用成交明细验收** → 见 [`am_pocket_trades_dual.md`](am_pocket_trades_dual.md)（`verdict=PASS`）。

```text
研究账本：trade-last 冻结口袋（delay=0 only）
历史 quote FillSpec：不作为本袖离线硬闸
不要：delay / quote-wait / FO 网格 / impact 叠加
```

研究 profile：`CONFIG/strategy_profiles/am_pocket_vd_multi_tp8_sl15_v1.json`  
（Scanner drain 尚未接线）

## 复现

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_pocket_quote_ready \
  --tag research_am_pocket_quote_ready
```
