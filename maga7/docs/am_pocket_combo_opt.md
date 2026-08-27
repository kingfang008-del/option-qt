# AM 口袋：多指标入场 × 退出合订

> 工具：`tools/scan_am_pocket_combo_opt.py`  
> 产物：`results/research_am_pocket_combo_opt/`  
> 前置：`docs/am_pocket_multi_gate.md`、`docs/am_pocket_scaleout.md`  
> 组合：`@20% / 5`

## 协议

固定几组冠军入场 × 聚焦退出族（TP/SL、分档 trail、时间切），看能否同时抬捕获与压 DD。

## 关键对比

| 入场 | 退出 | 胜率 | maxDD | 复利 | 捕获 | 备注 |
|------|------|-----:|------:|-----:|-----:|------|
| `vd_soft` | TP8/SL15/h240 | 69% | −16% | +33% | 9.3% | 旧基线 |
| **`vd+cont60+mf100+volr12`** | **TP8/SL15/h240** | **74%** | **−12%** | **+44%** | **12.9%** | **综合首选** |
| 同上 | 分档 67%@8%+宽trail | 74% | −12% | +37% | 11.8% | 分档未再赢 |
| `vd+volr12` | TP8/SL15/h240 | 69% | **−8%** | +31% | 10.2% | 最稳 DD |
| `vd+streak3` | TP8/SL15 | 69% | −13% | +28% | **14.2%** | 捕获略高、日胜率弱 |

时间切（90/120/180）在冠军子集上与纯 TP8 **完全同结果**（子集本身已很快触达 TP/SL）。

## 结论

1. **合订增益主要来自入场多指标，不是再拧退出。**  
2. 在已筛硬的子集上，分档/trail **未能**把捕获从 ~13% 再抬向 20%+。  
3. 相对泄露幻觉仍差很远；相对因果 `vd_soft` 则是实打实的：胜率+5pp、DD −16%→−12%、复利+33%→+44%。

## 冻结候选（研究）

```text
pockets: no_b_up
entry:
  vd_soft (fo_vwap30∈[0.3%,1.5%], vwap_diff∈[0.2%,0.7%], accel∈[0,2.5e-4])
  AND sign(ret_60)==dir
  AND sign(mf100)==dir
  AND volume_ratio_60 ≥ 1.2
exit: TP 8% / SL 15% / max_hold 240s
portfolio: 20% / max 5 concurrent
```

若更嫌回撤：用 `vd+volr12` 同退出（maxDD≈−8%，牺牲一点复利与胜率）。

下一步路径状态退出已扫过：见 `docs/am_pocket_path_exit.md` —— **正股逆向/fail-fast/hybrid 未抬捕获**，冻结退出仍用 TP8。
