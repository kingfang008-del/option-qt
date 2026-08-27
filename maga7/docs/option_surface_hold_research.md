# Option surface / Greeks → hold & day-type（研究）

**Status:** research direction — does **not** mutate peer3 freeze.  
**Motivation:** MAGA7 用股票 flow 选方向、用期权当车辆，却几乎不用希腊值 / IV 曲面做「是否快进快出 / 是否续持」。  
**Feature sources:**

| 管线 | 路径 | 产出 |
|------|------|------|
| Greeks 日算 | `preprocess/raw_data_deal/option_cac_day_vectorized.py` | `quote_options_monthly_iv/{SYM}/standard`：逐 bucket `iv/delta/gamma/vega/theta/vanna/charm` + spread/imbalance |
| 锁定展平 | `preprocess/ask_bid/options_locked_feature.py` | `quote_options_bucketed_v7/{SYM}`：`options_vw_*` 曲面聚合 |

## 数据现实（2026-07-21）

| 资产 | bucketed_v7（完整曲面） | day_iv `*_high_features` |
|------|------------------------:|--------------------------|
| QQQ | 有（至 **2026-06**，缺 7 月） | 有 |
| Mag7 单票 | **无** | 仅 4 列结构：`atm_iv/skew/pcr/term` |

→ 第一阶段用 **QQQ 曲面作 Mag7 的期权体制先验**（指数 vol 微观）。  
→ 单票 Greeks 决策需先对 Mag7 跑 `option_cac` + `options_locked_feature`（与 QQQ 同管线）。

## 特征 → 决策层（对齐 MAG7 架构）

禁止再用「VIXY 红了就 scalp」单因子；曲面特征进 **软先验 / Failure / 动态持仓**：

| Layer | 特征（优先） | 作用 |
|-------|--------------|------|
| L1 Regime prior | `options_vw_iv`, `options_struc_term`, `options_flow_skew`, `options_vw_spread` @10:30 | 缩放仓位 / **武装 path_fast_pack**（不发方向） |
| L3 Entry | `options_vw_imbalance` 与信号同向；`spread` 不过热 | 拒明显假突破（软） |
| L4 Failure | 持仓后 `options_iv_divergence` 恶化、`gamma_accel` 反向、spread 跳升 | 快速 `EXIT`（比纯股价更早） |
| L5 Hold | 合约自身 Δ 停滞（已有 delta_time_stop）+ 曲面 `iv_momentum` 与方向一致才续持 | 废止无条件 T30 |

### 快模式武装（替换纯 wash / 纯 VIXY）

建议研究门（因果 asof 10:30，QQQ 曲面）：

```text
opt_chop = rank(|iv_divergence|) + rank(vw_spread) + rank(|gamma_accel|)
arm_fast if opt_chop high OR (mixed_wash_up AND iv_momentum hostile to UP sleeve)
```

直觉：

- **高 spread + IV–价格背离 + gamma 急变** → 期权定价在「来回/恐慌」，适合快进快出。  
- **IV 动量与股票信号同向、term 陡、skew 稳定** → 允许 lit runner。

## 与现有模块衔接

- `path_fast_pack.when` 扩展：`mixed_wash_up` | `qqq_opt_chop` | `wash_or_opt_chop`  
- Loader：`maga7/common/option_surface.py`（读 bucketed_v7，asof 快照）  
- Live：需 QQQ（或标的）分钟 Greeks 流；短期可 shadow 只读盘后特征回放。

## 验收

1. May–Jun（有 QQQ bucketed）上：`opt_chop` 武装 vs 纯 wash —— 强窗收益回撤是否小于 always-fast，7 月需补特征后再验 7/20。  
2. 单票曲面就绪后：持仓合约 `delta/vega/iv` 路径进 L4/L5，对照 lit_wash_fast。  
3. 不接受：仅 ATM IV level 硬阈值（与 VIXY level 同类偏置）。

## 实现（已接）

- `path_fast_pack.when`: `qqq_opt_chop` | `wash_or_opt_chop` | `wash_and_opt_chop`  
- 规则：`imbalance <= opt_imbalance_max` **或** 当日 chop score 相对近 `opt_lookback_days` 个 10:30 快照分位 ≥ `opt_chop_pctile_min`  
- Profile: `path_hold_lit_opt_chop_v1`（默认 **`wash_and_opt_chop`**）  
- Ablation: `tools/run_path_hold_opt_chop_ablation.py`  
- Build: `SHELL/run_maga7_option_surface_build.sh`（Stage A=day_iv；Stage B=bucketed_v7，需 monthly_iv/standard）

## May–Jun 消融（2026-05-01…06-30，peer3 entries）

| variant | total_ret | maxdd | fast_days | 解读 |
|---------|-----------|-------|-----------|------|
| lit_always | +63.7% | −16.0% | 0 | 强窗基准（无 scalp） |
| **wash_and_opt** | **+27.7%** | −16.0% | **11** | **最优条件 pack** |
| wash_fast | +18.0% | −16.9% | 17 | 纯 wash 略宽 |
| opt_chop | +6.2% | −14.9% | 23 | 单用曲面偏宽 |
| wash_or_opt | −2.1% | −20.7% | 29 | **拒**：过度武装 |
| baseline_t30 | +847%* | −5.4% | 0 | 固定 hold；*与 lit 非同口径收益定义，勿直接比绝对数 |

结论：**OR 否决**；**AND 优于纯 wash**（少武装、更高收益）；仍低于 lit_always（强窗牺牲），用途是砍 false-hold 日而非全年默认 scalp。

## 数据进度

- QQQ `bucketed_v7/2026-07.parquet`：**已补**（day_iv 7/1–7/20 → monthly → locked）  
- Jul 20 @10:30：`imbalance≈+0.21`（非卖压），但 chop score 分位仍触发 `qqq_opt_chop`；**wash=True 且 AND=True**  
- Mag7：仍无 `quote_options_day_iv/.../standard`（需 open-lock sniper；QQQ 管线配置不直接复用）

## Jul 20 fused m5（disable_prevention，对齐 live 入场）

| variant | day_ret | trade_sum | 武装 |
|---------|---------|-----------|------|
| clock T30 | −8.3% | −48% | — |
| lit | −4.6% | −22% | — |
| wash_fast | **+1.1%** | **+6%** | wash |
| wash_and_opt | **+1.1%** | **+6%** | wash∧opt（同 pack） |

当日 AND 与纯 wash 同出口；曲面在 Jul 20 **没有额外增量**，但确认了「能打分且与 wash 一致武装」。

## L1/L2 离线验证（2026-07-21）

工具：`tools/verify_vol_surface_layers.py` · 结果：`/mnt/s990/data/maga7/results/vol_surface_layer_verify/`

- **P0 SVI**：AMD/NVDA/QQQ × 7/14–16 @10:30 → 9/9 拟合，median IV RMSE≈0.15pt，蝶式负曲率≈0  
- **P1 路径 Greeks（Jul20 clock）**：Redis `localSymbol` mid 反推 IV/Δ；L2 规则（giveback / iv_shock / delta_fade）把 clock trade-sum **−48.5% → +0.7%**（MSFT iv_shock 贡献最大；GOOGL giveback 切赢家）  
- 同日 `wash_and_opt` 仍更好（+6%）；L2 证明路径 Greeks **有增量信号**，需防切赢家

## L2 调参消融（Jul20，叠 wash_and_opt）

工具：`tools/run_path_greeks_exit_ablation.py` · `common/path_greeks_exit.py`

| book | preset | trade-sum | L2 触发 |
|------|--------|----------:|--------|
| clock | off | −48.5% | 0 |
| clock | naive | −37.5% | 2（切 GOOGL） |
| clock | winner_safe* | （见 scoreboard） | giveback 收紧后应保住赢家 |
| clock | **toxic_only** | **−24.4%** | 1（MSFT IV，**保留 GOOGL**） |
| wash_and_opt | 任一 L2 | **+6.2%** | **0**（pack 已更早出场） |

\* giveback 改为必须回到 flat/red（`giveback_ret_max=0`）。

结论：
1. **控亏主力仍是 path_fast_pack（wash∧opt）**；Jul20 上再叠 L2 **无增量**。  
2. L2 `toxic_only` 适合作为 **非 fast 日 / T30 残留** 的安全网（clock −48→−24，且不切赢家）。  
3. **多日可 offline replay**：`tools/run_path_greeks_offline_replay.py` 用 `maga7_mf10_open_ladder_otm5` 1s mid + stock 1s（不依赖 Redis）。  
4. Profile 已挂 `path_greeks_exit.enabled=false`（待接入 OMS）。

## 日级武装诊断（May–Jul，lit vs wash_fast）

工具：`tools/run_fast_pack_arm_diagnosis.py`  
结果：`/mnt/s990/data/maga7/results/fast_pack_arm_diagnosis_v1/`  
标签：`help_delta = wash_fast − lit`；`|Δ|>0.5%` 为 helps/hurts。

| 规则 | 武装天数 | 抓住 helps | 漏网 | 误杀 hurts | 武装日 Δ 合计 |
|------|--------:|----------:|-----:|----------:|-------------:|
| wash | 28 | 4/7 | 3 | **8/9** | **−0.54** |
| opt | 30 | 6/7 | 1 | 4/9 | −0.34 |
| **AND** | 16 | 3/7 | 4 | 4/9 | −0.39 |
| OR | 42 | 7/7 | 0 | 8/9 | −0.49 |

误杀日（AND）：5/1、5/15、6/9、7/2（单日伤害大，尤 7/2 −17pt）。  
漏网日：5/8、5/13、5/27、7/14（单日增益小，~+0.5–2pt）。  
**结论：优先降误杀，不是追漏网。**  
阈值敏感：`wash ∧ (imb≤−0.05)` → FA 4→**2**（去掉 5/1、6/9 正 imb 误杀）；漏网仍 4（增益小）。  
`opt_gate=imb_only` 可降 +imb 误杀，但 **Jul20 imb=+0.21 → AND 会漏**；该日靠 **wash** 武装。

### wash_refine（在已 wash 日上再过滤）

因果特征（@10:30）：`chop<1.85` ∧ `med_stock_ret<0.3%` ∧ `pcr<2` ∧ `iv_mom<3%` ∧ `n_down≤4`

| | wash 裸 | wash+refine |
|--|--------:|------------:|
| 误杀 FA | 8 | **1**（仅 6/3 −2.3pt） |
| 抓住 HELP | 4(+Jul20) | **全保留含 Jul20** |
| 武装日 Δ | −0.54 | **+0.01** |

Profile：`when=mixed_wash_up` + `wash_refine=true`。

Jul20 不在 may_jul daily（止于 7/17）；fused：wash 武装，help≈+5.7pt。

## May–Jun offline L2 replay（1s quotes，无 Redis）

`run_path_greeks_offline_replay.py` · miss=0

| book | off | toxic_only | lift |
|------|----:|----------:|-----:|
| baseline_t30 | +13.45 | +8.28 | **−5.17**（切强窗） |
| lit_always | +3.08 | +2.13 | −0.95 |
| wash_and_opt | +2.11 | +1.68 | −0.42 |

Jul20 clock 上 L2 救命；**May–Jun 强窗上 L2 净负**。因此 L2 只能作「fast pack 未武装时的毒日网」，不能全年挂。

## Next

1. OMS：`toxic_only` 仅当 `path_fast_pack` 未武装  
2. Mag7 单票 open-lock → day_iv → bucketed  
