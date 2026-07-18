# 更快认错：mtm_floor / mf_flip（叠 extend_mtm_only）

针对 07-07~09 假信号持仓拖满 T+30 的止血实验。

## 机制

`exit_mode` 现支持叠加（`+` 连接），例如：

- `hold_extend+mtm_floor`
- `hold_extend+mf_flip`
- `hold_extend+mf_reversal`

也可 `trade.early_exit_mode`。软出场在 TP/SL 之后、硬超时之前检查。

| 模式 | 触发 | 默认宽限 |
|---|---|---|
| `mtm_floor` | 期权 MTM ≤ `mtm_floor_ret`（默认 0） | 10m |
| `mf_flip` | mf10 与持仓方向相反 | 60s |
| `mf_reversal` | 同 mf_flip | 10m |

Offline：`simulate_trade`；Live OMS / stream 已同步解析叠加。

脚本：`maga7/tools/run_early_cut_ablation.py`  
产物：`results/early_cut_ablation_extend_mtm_peer3_may_jul/`  
软阈值：`results/early_cut_soft_threshold_may_jul/`

## 主消融（May–Jul，peer3）

| 变体 | total_ret | MaxDD | floor/flip 次数 | 07-07~09 |
|---|---:|---:|---:|---:|
| **extend_mtm_only** | **+401.1%** | **-16.2%** | 0/0 | -0.50 |
| ext_mtm_floor_h10 | +79.1% | -25.4% | 39/0 | -0.28 |
| ext_mtm_floor_h5 | +51.2% | -20.3% | 42/0 | -0.24 |
| ext_mf_flip_g60 | +71.0% | -23.3% | 0/42 | **-0.14** |
| ext_mf_reversal_h10 | +67.1% | -25.8% | 0/41 | -0.26 |
| **full_day** | **+673.3%** | **-12.2%** | 0/0 | -0.50 |
| full_day+floor_h5 | +88.5% | -11.1% | 33/0 | -0.24 |
| full_day+mf_flip | +116.0% | -10.7% | 0/33 | -0.14 |

07-07~09 个例（`mf_flip`）：NVDA **-5.1%**（原 -30.6%）、AMD **-8.3%**（原 -18.4%）；但 TSLA 赢家被截成 +11.9%（原 +24.4%）。

## 软阈值（少砍）

| 变体 | total_ret | MaxDD | 触发 | 07-07~09 |
|---|---:|---:|---:|---:|
| floor≤-10% @10m | +130% | -26.4% | 31 | **-0.07**（保住 TSLA T+45） |
| floor≤-20% @5m | +176% | -24.3% | 24 | -0.38 |
| floor≤-25% @5m | +234% | -27.8% | 19 | -0.52 |
| mf_reversal @20m | +226% | **-14.1%** | 31 | -0.29 |

`flip_h20` 是唯一 MaxDD 略好于底仓的点，但收益仍少约 **175pp**。

## 结论

1. **局部有效**：连亏窗单笔亏幅下降（尤其 `mf_flip` / `floor≤-10%`）。
2. **全局失败**：53 笔里 30–40 笔被软出场截断，大量赢家（含 T+45 延长）被误杀；全期收益腰斩甚至更多。
3. **勿叠 full_day**：日历优势被软出场吃掉（+673% → ~+100%）。
4. **不升格**。07-07~09 若还要挖，优先入场质量 / 事件之外的日级过滤，而不是持仓中途认错。
