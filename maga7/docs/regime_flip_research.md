# Regime：隔日 QQQ 翻面 / Put–VIXY / mf10 对齐

相对因果基线（仅 `qqq_align`），研究「连亏是否来自 regime 翻转」的门禁，**不改信号窗、不改出场 rails**。

## 机制

| 键 | 作用 |
|---|---|
| `qqq_day_flip_mode=block\|scale` | 今日 `sign(qqq_from_prev)` ≠ 昨收盘同量 → 停手或降仓 |
| `qqq_day_flip_scale` | scale 模式仓位乘数（默认 0.5） |
| `put_vixy_z_min` | DN 要求 `vixy_z ≥` 阈值（压波动时禁 put） |
| `qqq_mf10_align` | 方向还须与 `qqq_mf10` 同号 |

实现：`maga7/common/regime.py`（`RegimeDecision.size_scale`）；replay / stream 已接线。

## 消融（May–Jul peer3，delay=60）

```bash
python -m maga7.tools.run_regime_flip_ablation --with-hold-extend \
  --tag regime_flip_ablation_mag7_googl_peer3_may_jul
```

产物：`maga7/results/regime_flip_ablation_mag7_googl_peer3_may_jul/`

| 变体 | ret | MaxDD | n | 焦点簇 ret 和 |
|---|---|---|---|---|
| baseline | **+356.6%** | -16.5% | 50 | -2.63 |
| put_vixy_z0 | +189.3% | -16.5% | 40 | -2.30 |
| **qqq_mf10_align** | +289.8% | **-12.2%** | 43 | **-1.51** |
| flip_block | +176.5% | -12.2% | 25 | -1.05 |
| flip_scale50 | +233.0% | -12.2% | 51 | -2.90 |
| flip_block_put0 | +88.1% | -12.2% | 19 | -1.05 |
| extend_mtm_only + baseline | **+388.5%** | -16.2% | 50 | — |
| extend_mtm_only + mf10_align | +326.1% | **-12.2%** | 43 | — |

焦点簇：05-20/21、06-12/16/18、07-07/08/09。

## 结论（研究态，未升格）

1. **隔日 flip 全天停手过猛**：挡住不少大胜日（翻向本身常是趋势日），ret 掉到 +176%。
2. **`put_vixy_z≥0` 同样过猛**：会砍掉 VIXY 未抬升时的 DN 大胜（如 05-18、07-02）。
3. **相对可取：`qqq_mf10_align`**：用「价方向 vs 10m 资金流」背离过滤  
   - 焦点簇挡住：05-21 NVDA DN、06-18 NVDA UP、07-08 META DN  
   - 07-07 NVDA DN / 06-12 GOOGL SL **仍在**（入场时 QQQ mf10 已与方向同号）  
   - MaxDD 从 -16.5% → **-12.2%**  
   - 代价：ret +357% → +290%（硬 T30）或 +389% → +326%（叠 `extend_mtm_only`）
4. 因果基线默认 **仍不写** 这些键；若要压回撤，优先试 `qqq_mf10_align`，不要用裸 `flip_block`。

## 细组合：背离禁入 + 翻面降仓

产物：`maga7/results/regime_flip_fine_ablation_mag7_googl_peer3_may_jul/`

| 变体 | ret | MaxDD | 相对 mf10_align |
|---|---|---|---|
| mf10_align | +289.8% | -12.2% | 基准 |
| mf10_align + flip_scale50 | +210.5% | -12.2% | 同 MaxDD，ret 更差 |
| mf10_align + flip_scale25 | +173.9% | -12.2% | 更差 |
| extend + mf10_align | **+326.1%** | -12.2% | 叠延长更好 |
| extend + mf10 + flip_scale50 | +235.5% | -12.2% | 叠翻面降仓仍伤收益 |

结论：MaxDD 已被 `qqq_mf10_align`（+ 06-12 单笔 SL 地板）钉住；再叠翻面降仓只砍赢家、不改善焦点簇。**细组合不成立**；研究候选仍是单独 `qqq_mf10_align`（可选叠 `extend_mtm_only`）。
