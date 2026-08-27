# AM 口袋：正股路径状态退出

> 工具：`tools/scan_am_pocket_path_exit.py`  
> 产物：`results/research_am_pocket_path_exit/`  
> 前置：`docs/am_pocket_multi_gate.md`、`docs/am_pocket_combo_opt.md`  
> 组合：`@20% / 5`

## 协议

在多指标入场子集上，期权 mark 与正股 1s **联合因果行走**，测试：

| 族 | 含义 |
|----|------|
| `stock_adv` | 有向正股回撤 ≥ X bp，且期权未大涨时砍 |
| `fail_fast` | T 秒后正股+期权仍红 → 早砍 |
| `stk_gb` | 正股先武装再 giveback |
| `hybrid` | 正股逆向 + 期权 trail + 宽 TP |
| `scale_ref` | 既有 67%@6% 分档对照 |
| `tpsl` | 固定 TP/SL 基线 |

入场固定：`vd_soft` / **`vd+cont60+mf100+volr12`** / `vd+volr12`。

## 冠军入场结果（相对 TP8/SL15/h240）

| 退出 | 胜率 | maxDD | 复利 | 捕获 | 正股触发占比 |
|------|-----:|------:|-----:|-----:|-------------:|
| **TP8/SL15/h240（基线）** | **74%** | **−12%** | **+44%** | **12.9%** | 0% |
| TP10/SL15/h300 | 74% | −12% | +47%† | 12.2% | 0% |
| `sadv~15bp mh15`（会触发） | 68% | **−10%** | +38% | 11.4% | ~9% |
| hybrid（宽TP+trail） | 55–58% | −15~−18% | +22~+36% | 8–11% | 17–26% |
| scale 67%@6 | 74% | −12% | +36% | 10.6% | — |

† discovery 复利略高，但 **blind 转负**，不采纳。

## 结论

1. **正股路径退出能触发**（尤其 hybrid / 短 `min_hold` 的 adverse），不是对齐坏了。  
2. **没有一组在捕获上压过 TP8**（`BIG capture` 为空）；触发越多，越像在砍掉本会回到 TP 的路径。  
3. 唯一像样的 DD 小胜（−12%→−10%）以胜率 −6pp、复利 −6pp、捕获 −1.5pp 换来，**不值得换基线**。  
4. 与 combo / scaleout 同判：在已筛硬的因果子集上，**简单固定 TP8 仍是退出天花板附近**；捕获缺口（~13% vs foresight）不是“再加一层正股规则”能关掉的。

## 冻结建议（研究 → 可进 profile 草稿）

维持 `am_pocket_combo_opt.md` 冻结候选不变：

```text
pockets: no_b_up
entry: vd_soft ∩ sign(ret_60)==dir ∩ sign(mf100)==dir ∩ volr60≥1.2
exit:  TP 8% / SL 15% / max_hold 240s
size:  20% / max 5
```

正股路径 / fail-fast / hybrid **不进**默认退出栈。

## 若还要抬捕获

优先换问题，而不是再扫退出网格：

1. ~~**换口袋 / 时段**~~ → 已扫：`docs/am_pocket_universe.md`（B-UP 对冠军 0 成交，扩容无增益）  
2. ~~**合约选择**~~ → 已扫：`docs/am_pocket_contract.md`（无增益）  
3. **接受 ~13% 捕获**，把精力放在 live 与旧 FO shadow 对齐、以及 CORE 误归因排查
