# Mag7 正股全时段前视扫描（非 CORE）

> 工具：`python -m maga7.tools.scan_stock_session_foresight`  
> 数据：`/mnt/s990/data/raw_1s/stocks`（含盘前≈04:00、盘后至≈20:00）  
> 产物：`results/research_stock_session_foresight_jan_jul/`

## 设定

- 标的：NVDA/TSLA/AAPL/AMZN/META/MSFT/AMD/GOOGL  
- 时钟网格：PRE→AH（含 04:00…19:00）  
- 策略：`LONG` / `MOM(lb)` / `FADE(lb)`；前视 15/30/60/120m  
- 稳定：强窗与弱窗 **avg>0 且 win≥52%**

## 主结论

1. **裸多不稳。** `LONG` 仅 AM 强窗略正；弱窗/PM/AH/PRE 多数偏负或胜率<50%。  
2. **会话级最稳的是 AH·FADE**（lb 5–15m → 持 15–30m），双窗 win≈55%，但均收益仅 **~1.5bp**——扣 2bp 单边往返成本后基本没了。  
3. **时钟级更有意思（前视发现，未做成本账）：**

| 时钟 | 策略 | 持有 | 强 avg | 弱 avg | 备注 |
|------|------|-----:|-------:|-------:|------|
| 08:00 | FADE lb5 | 120m | +8.9bp | +30bp | 盘前反转 |
| 10:30 | MOM lb60 | 120m | +8.9bp | +7.7bp | 近 CORE 时段动量 |
| 10:00 | LONG | 60–120m | +8–16bp | +4–6bp | 早盘偏多 |
| 16:00 | FADE lb30–60 | 30–60m | +4–5bp | +9–13bp | 收盘后 fade，胜率~58% |

4. **相对期权：正股前视空间以 bp 计，不是 %。** 适合当「低杠杆卫星 / 对冲腿」，很难单独复制 CORE 期权复利。

## 复现

```bash
PYTHONPATH=. python -m maga7.tools.scan_stock_session_foresight \
  --tag research_stock_session_foresight_jan_jul \
  --stock-1s /mnt/s990/data/raw_1s/stocks
```
