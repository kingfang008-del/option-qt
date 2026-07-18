# inprove2 过滤器 × extend_mtm_only

底仓：与 [`hold_extend_ablation_.../extend_mtm_only`](../results/hold_extend_ablation_mag7_googl_peer3_may_jul/extend_mtm_only/) 同构  
（peer3 + T30→T45 MTM≥0 延长 + delay=60 rails）。  
来源文案：`preprocess/raw_data_deal/inprove2.txt`。

| 过滤器 | 状态 |
|---|---|
| 1 SI 板块共振 | 已有 `signal.si_min` |
| 2 PE 价格效率 | 已有 `signal.pe_min_ratio` |
| 3 分时 mf Z-Score | **新** `signal.tod_mf_z_min`（`signals.tod_mf_z_ok`） |
| 4 期权 Skew | **未做**（需分钟级 Greeks） |

```bash
python -m maga7.tools.run_inprove2_extend_ablation \
  --tag inprove2_extend_mtm_ablation_peer3_may_jul
```

## May–Jul 成绩（对齐 archived `extend_mtm_only`：`day_circuit=null`）

产物：`maga7/results/inprove2_extend_mtm_nocircuit_peer3_may_jul/`  
底仓 equity 与 archived daily.csv **逐点一致**（end=501.12）。

| 变体 | ret | MaxDD | n | 焦点簇 |
|---|---|---|---|---|
| **extend_mtm_only** | **+401.1%** | -16.2% | 53 | -2.84 |
| si_0.57 | +244.0% | **-12.2%** | 26 | -0.46 |
| pe_0.5 | +369.9% | -18.2% | 48 | -2.04 |
| tod_z_2 | +49.0% | -13.9% | 14 | -0.79 |
| tod_z_1.5 | +97.7% | -18.3% | 25 | -0.84 |
| si057 + tod_z2 | +30.9% | -12.2% | 6 | -0.61 |
| si + pe05 | +247.2% | -12.2% | 25 | -0.42 |

（另有带 `day_circuit=-0.05` 的对照：`inprove2_extend_mtm_ablation_peer3_may_jul/`，底仓变为 +388.5%/50 笔。）

## 结论

1. **文案优先的 SI + 分时 Z** 叠在一起过猛（只剩 6 笔）。  
2. **单独 TOD Z≥2** 同样过严（14 笔），不适合直接套在 Rule-A streak 体系上（streak 已隐含持续性，再要「该分钟历史 2σ」会双重过滤）。  
3. **SI≥0.57**：焦点簇与 MaxDD 改善，但收益腰斩——与此前「勿用 SI 替换 peer3」一致。  
4. **PE 单独**：略损收益且 MaxDD 变差，无明显价值。  
5. 相对 `extend_mtm_only`，inprove2 这套在本窗 **没有帕累托改进**；若只想压 DD，SI 与 `qqq_mf10_align` 同类（都到 -12.2% 地板，代价是收益）。
