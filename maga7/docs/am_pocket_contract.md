# AM 口袋：合约映射（ATM/OTM/DTE）对照

> 工具：`tools/scan_am_pocket_contract.py`  
> 产物：`results/research_am_pocket_contract/`  
> 前置：`docs/am_pocket_universe.md`  
> 入场固定：`no_b_up` + `vd+cont60+mf100+volr12`；退出 `TP8/SL15/h240` @20%/5

## 协议

对同一批冠军入场时刻，用 open-lock 重选合约：

- 基线：foresight 探针自带 ticker（ladder closest-to-spot）
- ladder `otm_rungs` ∈ {0,1,2,3,5}
- prefer DTE1 / 仅允许 0DTE
- classic 固定 ATM / OTM1

## 结果

| 策略 | n | 同 ticker | 胜率 | maxDD | 复利 | 捕获 | mean oracle |
|------|--:|--------:|-----:|------:|-----:|-----:|------------:|
| **baseline（探针）** | **35** | 100% | **74%** | **−12%** | **+44%** | **12.9%** | 0.56 |
| ladder otm1 | 35 | 97% | 74% | −12% | +44% | 13.0% | 0.55 |
| ladder otm2/3/5 | 35 | 100% | ≡基线 | ≡ | ≡ | ≡ | ≡ |
| ATM-only / classic ATM | 35 | 74% | 71% | −15% | +30% | 12.0% | 0.53 |
| classic OTM1 | 35 | 23% | 71% | −12% | +22% | **7.7%** | 0.64 |
| prefer DTE1 | 34 | 88% | 73% | −12% | +32% | 11.6% | 0.52 |
| 仅 0DTE | 24 | 100% | 71% | **−8%** | +29% | 10.9% | 0.65 |

## 结论

1. **当前探针合约已接近 ladder 最优**：otm≥2 与基线完全相同；再宽无增益。  
2. **更贴钱（ATM-only）伤害复利与 DD**，不抬捕获。  
3. **固定 OTM1** oracle 更高但 **TP8 捕获更差**（杠杆路径更抖，固定止盈吃不满）。  
4. **强推 1DTE / 砍掉跳 0→1 的样本** 都不改善综合账本。  
5. 捕获缺口（~13% vs foresight）**主要不是合约错选**。

## 冻结

合约层维持现状：open-ladder（profile `otm_rungs`）+ prefer 0DTE + clear-OTM 跳期。不改。

## 研究收束

因果 AM 主线至此：

| 杠杆 | 结论 |
|------|------|
| 多指标入场 | **有用** → 冻结冠军门控 |
| 退出 / 分档 / 正股路径 | 拧不动 |
| 口袋扩容 | B-UP 对冠军 0 成交 |
| 合约重选 | 无增益 |

## 研究收束（更新）

用户纠正后的真假说是 **期权活跃 → 正股 MF 定边 → 短持**，不是 mark 换边。  
该假说首轮扫描见 `docs/am_activity_mf_scalp.md`：**双窗未通过**。

单边 TP8 研究冻结仍可作对照；换边/乐观 opp_first **否决**。
