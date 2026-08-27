# AM 口袋 × 稀有冲击硬门

> 工具：`tools/scan_am_pocket_impact_overlay.py`  
> 产物：`results/research_am_pocket_impact_overlay/`  
> 前置：`docs/buyer_impact_1s.md`、`docs/am_pocket_path_exit.md`

## 协议

| 项 | 值 |
|----|-----|
| 基线 | 冻结 champ：`no_b_up` + `vd_soft ∩ cont60 ∩ mf100+ ∩ volr12` |
| 出场 | TP8 / SL15 / h240 @20% / max5 |
| 叠加 | 用 `buyer_impact_1s` 的 AM 绝对切点：`impact_p90/95/98`、`volr≥1.5/2`、`volz≥2/3`、`|ret30|`、短涨跌∩量 |
| 问法 | 稀有冲击 AND 能否抬捕获 / dual，而不砍死 sleeve |

## 结果（相对 champ）

| gate | n | disc 复利 | blind 复利 | 捕获 | dual |
|------|--:|--------:|----------:|-----:|:----:|
| **champ** | **35** | **+44%** | **+5%** | **12.9%** | ✓ |
| +volr15 | 31 | +37% | +5% | 14.4% | ✓ |
| +volr20 | 28 | +27% | +5% | 12.5% | ✓ |
| +volz2/3 / +impact_p90–98 | 34 | +41% | +5% | 13.0% | ✓ |
| +abs_ret30_10bp | 22 | +29% | +5% | 15.1% | ✓ |
| +abs_ret30_20bp / +ret∩volr | 8 | +1.6% | +7% | 10.6% | ✓ |

`improved_vs_champ`（disc≥基线 且 捕获≥基线 且 blind>0）：**空**。

## 结论

1. **口袋 champ 已经坐在高冲击区。** `impact_p98` 几乎不筛（35→34）：开盘口袋 + `volr12` 时，冲击分大多已过 AM 全局 p98 切点。再 AND 全局稀有门是**冗余**。
2. **再收紧只换样本、不抬天花板。** `volr15` / `|ret30|≥10bp` 捕获略升（→14–15%），但 disc 复利明显掉；`|ret30|≥20bp` 把 n 砍到 8、复利≈0。
3. **与 `buyer_impact_1s` 不矛盾：** 全市场 stride 上稀有冲击有 lift；在**已筛硬的因果口袋**里，冲击信息已被 volr/MF/cont 吃掉，叠加无增量。
4. **冻结不变。** 不把 impact 百分位写进口袋默认入场。

## L2 / OBI 盘查

- `/mnt/s990` 下**无** order-book / depth / OBI 原始管线；现有期权是 trades / quotes / 1s 正股。
- 仓库里的 “L2” 指 **Hunter 看门狗层**（`washout_reclaim` 等），不是盘口 OBI。
- 若要再抬买方窗精度，需要**新数据源**（L2/OBI），不是再扫 1s MF/impact 网格。

## 复现

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_pocket_impact_overlay \
  --tag research_am_pocket_impact_overlay
```
