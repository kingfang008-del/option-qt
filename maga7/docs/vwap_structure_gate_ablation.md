# VWAP 结构门 ablation

May–Jul → 2026-07-17，freeze 底仓（`extend_mtm_full_day_peer3`）。

## 开关

| 开关 | 语义 |
|------|------|
| `signal.vwap_dir_lock` | 价 &lt; VWAP 禁 UP；价 &gt; VWAP 禁 DN |
| `signal.block_dn_if_vwap_lod` + `lod_bounce_min`（默认 0.02） | **仅 DN**：价 &gt; VWAP **且** 相对会话最低反弹 ≥X |

默认均 **false**。

## Scoreboard

| variant | total_ret | MaxDD | n | 07-17 | 挡住 |
|---------|-----------|-------|---|-------|------|
| baseline | **+810%** | −13.2% | 50 | −6.6%（NVDA+TSLA） | — |
| `vwap_dir_lock` | +707% | −13.2% | 48 | −7.0%（剩 TSLA） | 4 |
| `dn_vwap_lod_2pct` | +707% | −13.2% | 48 | −7.0%（剩 TSLA） | 2 |

明细：`maga7/results/vwap_structure_gate_ablation_may_jul_to_0717/`。

## 结论

- 两版都能挡 **NVDA**（价在 VWAP 上且离 LOD 反弹约 2.9%），**挡不住 TSLA**（信号时仍在 VWAP 下）。  
- 整段 May–Jul 从 +810% → +707%，MaxDD 不变 → **REJECT for freeze**。  
- 07-17 假空问题未靠 VWAP 族解决；需别的结构（或接受该日）/ 更长线的等量时钟研究。
