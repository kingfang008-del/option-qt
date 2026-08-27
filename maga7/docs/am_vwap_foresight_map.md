# AM 前视利润地图（VWAP 规则发现前置）

> 工具：`tools/scan_am_vwap_foresight_map.py`  
> 产物：`results/research_am_vwap_foresight_map_may_jul/`  
> 协议：先标出有可达利润的时段，再看 10/20/30s VWAP 是否能抬升 edge（**不做网格入场搜参**）。

## 协议

| 步 | 内容 |
|----|------|
| 1 | 每 60s 网格，A `09:30–10:30` / B `10:30–11:30`，ATM call+put 各算一次 |
| 2 | edge = `oracle_ret@900s ≥ 15%`（路径最优卖；非固定持有） |
| 3 | 特征：因果 `fo_vwap{10,20,30}`、`from_open_px`、`vwap_diff`、`accel_10_30` |
| 4 | **对齐后分析**：只保留 `dir == sign(from_open_px)`（双边同时探会虚高 edge） |

日历：discover=`may_jul09`，blind=`jul10_23`。

## 主结论

1. **有边，但是路径边**：对齐后 discover edge_rate≈**58.7%**，mean oracle≈**+30%**；  
   同批 mean **clock@900s≈−0.3%**。固定持有吃不到，需要 TP/SL / 早切。
2. **VWAP 阈值几乎无 lift**：`|fo_vwap*|≥thr` / `from_open` 的 lift 仅 **1.00–1.06**。  
   说明「先网格 FO@0.8%/1.2%」方向错了——幅度门檻几乎不浓缩机会。
3. **TOD 口袋更有信息**（对齐 discover，edge≥35%、n≥30）：

| session | TOD | dir | n | edge | mean clock |
|---------|-----|-----|--:|-----:|-----------:|
| A | 09:30 | UP | 577 | 0.69 | −1.6% |
| B | 10:45 | UP | 449 | 0.69 | **+10.1%** |
| B | 10:30 | DN | 402 | 0.67 | +1.3% |
| B | 10:55 | UP | 436 | 0.67 | +1.5% |
| A | 09:30 | DN | 574 | 0.65 | −2.7% |

4. Blind：`fo_vwap*≥1.5%` 对齐后 edge≈0.60、clock≈+5%（样本可），但相对 base 的 lift 仍≈1.0，**不是可单独升格的入场规则**。

## 下一步（规则侧）

只在 TOD 口袋内做因果 distill，优先：

- 退出：TP/SL / confirm-abort / 时间衰减（因为 clock 差、oracle 好）
- 入场形态：VWAP **结构**（10 vs 30 加速、与 session VWAP 关系），而非单纯 `|fo|≥thr`
- 禁止：再对全窗做大网格 FO thr×tp×sl

## 命令

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_vwap_foresight_map \
  --stride-sec 60 --tag research_am_vwap_foresight_map_may_jul
```

对齐后重算：`summary_aligned.json` / `tod_pockets_aligned_discover.csv` / `feature_lift_aligned_discover.csv`。
