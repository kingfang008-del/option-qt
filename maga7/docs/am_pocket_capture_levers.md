# AM pocket：捕获杠杆逐项验收

目标：`mean_capture ≥ 0.20`，双窗 compound > 0。  
工具：`maga7/tools/scan_am_pocket_capture_levers.py`  
结果：`/mnt/s990/data/maga7/results/research_am_pocket_capture_levers/`

基线：`vd_soft` + ride_combo → capture **0.148**（未达 0.20）。

## 用户裁决（2026-08-02）

**退回稀疏 L1 过线方案。**  
`vd_acc0bp` 虽达 capture 0.210，但约 3 个月仅 ~20 笔（~0.33 笔/交易日），密度不可用。

```text
demote  = CAPTURE20_L1_vd_acc0bp__ride
promote = RETREAT_vd_soft_ride

entry: vd_soft / no_b_up
mark:  trade-last slip 1%
exit:  ride_combo_st15_f03_tp50
n≈65（~1.1 笔/交易日）
capture≈0.148  disc≈+58%  blind≈+24%
Scanner: not production-wired
```

## 逐项结果（存档）

### L1 更高 MFE 入场

| entry | n | oracle均值 | capture | disc | blind | 判定 |
|---|---:|---:|---:|---:|---:|---|
| **vd_soft + ride** | **65** | 0.52 | **0.148** | **+58%** | **+24%** | **现行 keep** |
| champ + ride | 35 | 0.56 | 0.163 | +55% | +9% | 备选（更稀一点） |
| vd_cont_fo80bp + ride | 25 | 0.53 | 0.172 | +26% | +9% | 备选 |
| vd_acc0bp + ride | 20 | 0.73 | 0.210 | +48% | +18% | **已 demote（太稀）** |

### L2 分批减仓 — 不用（本目标）

最佳 econ capture **0.156**；压捕获，不启用。

### L3 期权 peak/trail — 接近但未过

最佳 econ capture **0.188**（fo80+trail）；未替换现行 keep。

## 高频尝试

见 [`am_high_freq_scalp.md`](am_high_freq_scalp.md)：FO 网格一天几十笔 × ~5% **未达成**。

## 换信号源

见 [`am_new_signal_source.md`](am_new_signal_source.md)：  
Impact FAIL；**launch-slope 多火** `tp15/sl25/h300` → ~22 笔/日、胜率~55%、均笔~+0.9%、双窗 PASS。

## 复现

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_pocket_capture_levers \
  --tag research_am_pocket_capture_levers
```

## 高频尝试

见 [`am_high_freq_scalp.md`](am_high_freq_scalp.md)：一天几十笔×~5