# SMC / ICT-lite + 订单流代理侦查兵

> 定位：独立侦查兵（不滤「基线漏抓」）；`event_source=smc_flow_scout`。  
> 数据约束：stock 1s OHLCV **无 aggressor**；期权 trades/quote 只做定价。  
> 工具：`common/smc_flow.py` · `tools/scan_smc_flow_scout_tpsl.py` · `tools/scan_smc_flow_scout_quote_dual.py`

## 形态（代理）

| 层 | 实现 |
|----|------|
| Structure | `sweep_rev_dn`：扫 prior swing high 后 close 收回其下；`bos_disp_dn`：close 跌破 prior swing low |
| Displacement | 近 `disp_sec` 回报 ≤ −thr |
| Order-flow proxy | down-tick volume share；或 CLV `sec_mf<0` + `streak_dn` |

**不是**真 CVD / footprint / FVG / order block；那些需要 tick side 或另建结构库。

## 双窗（2026-07-25）

| 账本 | 产物 | 裁决 |
|------|------|------|
| trades | `research_smc_flow_scout_dn_tpsl_dual` | **PASS**（1 cell） |
| quote FillSpec | `research_smc_flow_scout_dn_quote_dual` | **QUOTE_REJECT** |

Trades champion：`bos_disp_dn_sw300_d0.005_foff_tp0.2_sl0.15`  
MJ09 n=72 mean≈+0.85% day_win≈55%；Jul10–23 n=24 mean≈+1.9% day_win≈78%。

### 订单流确认 ablation

- `share55` 抬高 MJ mean（≈+2.3% @ tp25/sl15），但 **day_win&lt;0.55** → 未 dual_pass。  
- `mf_st5` / `share60` 同：加严后稳定性不够。  
- 唯一过 trades 闸的是 **flow=off**（纯 BOS+位移）——说明当前 1s 代理未能把「可执行边」从结构里筛出来。

Quote：fill 后 Jul 样本极少且 mean 大幅为负（≈−16%）→ 与 impulse scout 同类失败。

## 结论

1. 先前 impulse **QUOTE_REJECT** 不是扫错窗；叠 SMC+vol 代理后 trades 仅弱 PASS，quote 仍拒。  
2. **禁止** Shadow / 升格。  
3. 下一刀若继续 ICT/OF：需要 **tick aggressor 或 denser NBBO**，或换「期权侧 imbalance」确认；勿在 1s OHLCV 上硬套完整 SMC 教材。
