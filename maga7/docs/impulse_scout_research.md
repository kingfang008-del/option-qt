# Impulse Scout（独立 DN 侦查兵）研究笔记

> 产品定位：窄专家 = **独立侦查兵**，与 Mag7 Rule-A / Hunt **隔离**；不做「基线漏抓」过滤；自有 `event_source=impulse_scout` 平行账本。  
> Catalog：`CONFIG/narrow_experts/catalog_v1.json` → `impulse_scout`  
> 工具：`tools/scan_impulse_scout_tpsl.py` · `tools/scan_impulse_scout_quote_dual.py`

## 形态

会话内（AM / CORE / MID / PM）首次 `|ret_lookback| ≥ thr` 触发 → 立即 ATM 期权（DN=put）tp/sl；每 ticker×session 最多 1 笔。

双向 trades 扫描：边在 **DN**；UP mean 为负 → 定稿 **DN-only**。

## 双窗裁决（2026-07-25）

| 账本 | 产物 | 裁决 |
|------|------|------|
| option trades last+slip | `research_impulse_scout_dn_tpsl_dual` | **PASS**（dual_pass_n=3） |
| quote FillSpec | `research_impulse_scout_dn_quote_dual` | **QUOTE_REJECT**（dual_pass_n=0） |

Trades champion：`imp_t0.008_lb120_tp0.2_sl0.2`

| 窗 | n | mean | day_win |
|----|---|------|---------|
| May–Jul9 | 39 | ≈+5.2% | ≈58% |
| Jul10–23 | 18 | ≈+6.8% | ≈60% |

Quote 最优细胞（`sp0.1_lag3`）May–Jul9 mean≈+5.4%（n=15）尚可，但 Jul10–23 mean≈+1.5%（n=6）且多数细胞 Jul mean&lt;0 → **不满足可执行双窗**。

## 结论

- **禁止** trades-only 升格 / Shadow 接线。  
- 与 `core_dn_sync` 同类：纸面边在 FillSpec 上消失。  
- 近期可执行参与仍优先 **Hunt + `qqq_open_cont`**。
