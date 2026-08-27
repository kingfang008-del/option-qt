# AM 口袋：宇宙（TOD cell）对照

> 工具：`tools/scan_am_pocket_universe.py`  
> 产物：`results/research_am_pocket_universe/`  
> 前置：`docs/am_pocket_path_exit.md`（退出拧不动）  
> 固定退出：`TP8/SL15/h240` @20%/5

## 协议

在 foresight 对齐探针上，对 `POCKET_SETS` + 逐格加回 B-UP，交叉入场：

- `vd_soft`
- **`vd+cont60+mf100+volr12`（冠军）**
- `vd+volr12`
- `accel0`（旧进攻对照）

## 冠军入场 × 口袋

| 口袋 | n | 胜率 | maxDD | 复利 | 捕获 | blind |
|------|--:|-----:|------:|-----:|-----:|------|
| **`no_b_up`** | **35** | **74%** | **−12%** | **+44%** | **12.9%** | +5% / 4笔 |
| `all`（含 B-UP） | 35 | 74% | −12% | +44% | 12.9% | 同左 |
| `a_only` | 34 | 73% | −12% | +41% | 13.0% | 同 blind |
| `dn_heavy` | 20 | 70% | **−8%** | +31% | **17.1%** | **0 笔** |
| `b_up_only` | **0** | — | — | — | — | — |

要点：

1. **冠军门控在 B-UP 格上 0 成交** —— 加回 10:40–10:55 UP 对冠军账本无影响（`all ≡ no_b_up`）。  
2. `a_only` 只少 1 笔 B-DN，几乎同构，略损复利。  
3. `dn_heavy` 捕获/DD 更好看，但样本更瘦、**无 blind、7 月复利为负**，不能替换主宇宙。  
4. `vd_soft` 下 B-UP 只有 5 笔且负捕获；`accel0` 在 `dn_heavy` 上复利很高（+119%）但 maxDD≈−23%，属旧风险档，不是因果多指标主线。

## 结论

- **口袋扩容不是抬捕获的杠杆**（至少在当前冠军门控下）。  
- 继续冻结：`no_b_up` + 多指标冠军 + TP8。  
- `dn_heavy` 仅作旁路观察（DN 袖子 / 更高捕获、更低容量），不进默认。

## 下一步

合约映射已扫：`docs/am_pocket_contract.md` —— **无增益，维持 open-ladder**。研究主线建议收束为 live/shadow 对齐。
