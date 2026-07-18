# Option MAE early cut（持仓内早切）

在 freeze `exit_mode=hold_extend` 上叠加软早切：路径从未走出足够 MFE，却挖到 −`mae_cut_ret` 时平仓（`MAE_CUT`），避免拖到 SL（−60%）或到期。

**默认 OFF**（`early_exit_mode: null`）。不升 freeze。

## Config（`trade`）

```json
"early_exit_mode": "mae_cut",
"mae_cut_ret": 0.25,
"mae_cut_mfe_bypass": 0.20,
"mae_cut_min_hold_minutes": 5,
"mae_cut_only_dn": false,
"mae_cut_require_mf_against": false
```

| 参数 | 语义 |
|------|------|
| `mae_cut_ret` | 触阈：MTM ≤ −该值 |
| `mae_cut_mfe_bypass` | 若峰值 MFE 已 ≥ 该值，不早切（交给 SL/到期） |
| `mae_cut_only_dn` | 仅 DN |
| `mae_cut_require_mf_against` | 额外要求正股 mf10 逆势（delay 后） |

也可写进 `exit_mode`：`hold_extend+mae_cut`。

## 模块

| Path | Role |
|------|------|
| `maga7/common/replay.py` `simulate_trade` | `mae_cut` / `toxic_cut` |
| `maga7/tools/run_mae_cut_ablation.py` | 双窗 scoreboard |
| `maga7/results/mae_cut_ablation_dual_window/` | 结果 |

## Scoreboard（2026-07-18）

| window | variant | total_ret | vs base | MaxDD | n_MAE_CUT | 07-17 |
|--------|---------|-----------|---------|-------|-----------|-------|
| May–Jul | baseline | **+810%** | — | −13.2% | 0 | −6.6% |
| May–Jul | mae25 bypass20 | +502% | **62%** | −20.8% | 13 | −9.3% |
| May–Jul | mae25 DN-only | +583% | 72% | −20.3% | 9 | −9.3% |
| May–Jul | mae30 DN+MF | +640% | 79% | −22.3% | 7 | −9.3% |
| Feb–Apr | baseline | +140% | — | −28.9% | 0 | — |
| Feb–Apr | mae25 DN-only | +81% | **58%** | −30.4% | 12 | — |

### 07-17 单笔

| | NVDA | TSLA |
|--|------|------|
| baseline | −16.1%（T+45） | −34.8%（T+30） |
| MAE_CUT | **−31.8%** | −31.8% |

NVDA 先挖到 −32% 再反抽，早切**锁在谷底**；TSLA 略好一点，但日线与整段双窗都被拖累。

## 结论 → **REJECT for freeze / research_baseline**

与 entry 否决同类水床，且对「探底反抽」日更伤。`mfe_bypass` / DN-only / MF 确认均未能挽救。

### 复验：叠在 P1.1 L1 上（2026-07-18）

L1 已 Halt 07-17，仍挡不住整体回撤：

| window | L1 | L1+mae25 | vs L1 |
|--------|---:|---------:|------:|
| May–Jul | +875% | +564% | **64%** |
| Feb–Apr | +146% | +54% | **37%** |

个案（05-06/11、07-07）改善，见 [`loss_days_0511_0506_0707.md`](loss_days_0511_0506_0707.md)。  
产物：`results/mae_cut_on_l1_dual_window/`。

**保持 `early_exit_mode: null`。**

更稳的方向（若再试）：

1. **确认延迟**：首次触 −thr 后等 N 秒，仍 ≤ −thr 才切  
2. **只砍未延伸单**：与 `hold_extend` 互斥（已 extend 的不 mae_cut）  
3. 接受毒性尾部，继续用现有 SL/仓位管理，不再加路径硬切
