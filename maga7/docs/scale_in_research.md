# 分段建仓 + 回调二次确认（scale_in）

**日期：** 2026-07-19  
**状态：** 研究旋钮，**默认 off** · **Verdict: REJECT for research_baseline**  
**对照：** L2+TT1_05（`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1`）

## 规则

```json
"scale_in": {
  "enabled": false,
  "first_frac": 0.5,
  "add_frac": 0.5,
  "pullback_ret": 0.30,
  "confirm_mode": "mf",
  "min_hold_seconds": 120
}
```

- 信号成交只建 `first_frac`（账户贡献按 `size_frac * first_frac * r1` 编码进 `ret`）
- 持仓 ≥ `min_hold_seconds` 后，期权 MTM ≤ −`pullback_ret` 时做二次确认
- `confirm_mode=mf`：因果 mf10 仍同向才补 `add_frac`（更便宜的第二笔）
- 两笔共用同一退出轨（TP/SL/T+30/extend）
- `confirm_mode=never` = 永久半仓对照

## 双窗结果（L2+05）

产物：`results/scale_in_ablation_peer3_dual_window/`

| 窗 | 变体 | total_ret | vs base | MaxDD | n_add | worst_trade |
|----|------|----------:|--------:|------:|------:|------------:|
| May–Jul | baseline | **+1528%** | 100% | −12.2% | 0 | −60.8% |
| May–Jul | half_only | +323% | **21%** | −6.1% | 0 | −30.4% |
| May–Jul | pb20 always | +759% | 50% | −11.2% | 25 | −56.0% |
| May–Jul | pb30 always | +473% | 31% | −10.5% | 10 | −52.5% |
| May–Jul | pb20 mf | +521% | 34% | −11.2% | 16 | −56.0% |
| May–Jul | **pb30 mf** | +453% | **30%** | −6.1% | 5 | −47.5% |
| Feb–Apr | baseline | **+152%** | 100% | −26.4% | 0 | −61.9% |
| Feb–Apr | half_only | +65% | 43% | −14.0% | 0 | −30.9% |
| Feb–Apr | pb20 mf | +120% | 79% | −25.0% | 31 | −55.9% |
| Feb–Apr | pb30 mf | +89% | 59% | −20.8% | 17 | −52.2% |

## 解读

1. **直线赢家拿不到满仓**（TP 前从未回调 20–30%）是主因；半仓对照只剩强窗 21%。
2. **补仓常补进毒性路径**：05-11 TSLA / 06-24 GOOGL 在 `always` 下被加满，账户冲击仍大；`pb30+mf` 挡住了 TSLA 加仓，但挡不住整体收益塌陷。
3. 胜率可抬（部分臂），**账户复利仍远低于基线**——不是可升线的风控改进。

## 代码

- `maga7/common/scale_in.py`
- `simulate_trade(..., scale_in=)` → 混合 `ret` + `scale_in_added` 字段
- 工具：`python -m maga7.tools.run_scale_in_ablation`
- 单测：`maga7/tests/test_scale_in.py`

## 结论

**不升 research_baseline。** 灾难硬轨 `sl_mult=0.45` 已单独升线（[`sl55_hard_stop_research.md`](sl55_hard_stop_research.md)）；回调加仓仍 REJECT。
