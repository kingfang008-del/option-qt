# Path-adaptive hold（研究）

**Status:** research only — does **not** mutate peer3 freeze.  
**Primary profile:** `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_path_hold_lit_wash_fast_v1.json`  
**Lit-only (no day gate):** `..._path_hold_lit_v1.json`  
**Prior (rev-only on extend scaffold):** `..._path_hold_cond_rev_v1.json`  
**Goal:** **风险优先**（防 7/20 假持仓 / giveback），不是最高收益。  
**Ablation:** `tools/run_path_hold_wash_fast_ablation.py` → `path_hold_wash_fast_v1`

## Literature rails（当前默认）

对齐 trailing / drawdown optimal-stopping 文献（Leung–Zhang trailing；Rodosthenous–Zhang drawdown anxiety；Kaminski–Lo regime stops）：

| Rail | 机制 | 配置 |
|------|------|------|
| 1 路径止盈 | MTM trail：峰值 ≥+20% 后回撤 ≥12% → `TRAIL` | `exit_mode=mtm_trail`, `trail_activate=0.20`, `trail_dd=0.12` |
| 2 废证续持 | 入场 ≥10m，标的 signed≤0 且 opt MTM≤+10% → `STOCK_REV` | `stock_rev_exit.when=always` |
| 3 时钟安全阀 | 未发展持仓上限 **T+45** | `hold_minutes=45`，**`hold_extend_minutes=null`**（废除无条件续持） |

外轨仍保留：`tp_mult=1.6` / `sl_mult=0.45` / `trade_toxic`。

### Wash-day fast pack（推荐）

干净日用 lit；`mixed_wash_up` 早晨武装快包（`trade.path_fast_pack`）：

| 参数 | lit（干净） | fast pack（wash） |
|------|------------:|------------------:|
| trail_activate / dd | 0.20 / 0.12 | **0.15 / 0.08** |
| STOCK_REV min_hold / opt_mtm | 10m / +10% | **5m / +5%** |
| undeveloped clock | T+45 | **T+20** |

实现：`common/path_fast_pack.py`；接线 replay / OMS fill / broker_oms。  

| 样本 | clock | lit | **wash_fast** | always-fast |
|------|------:|----:|--------------:|------------:|
| 7/20 fused day | −8.3% | −4.6% | **+1.1%** | +1.1% |
| May–Jul total | +1856% | +129% | +40%（23 wash 日） | **−6%**（否决） |
| Jan–Mar MaxDD | −17.8% | −16.3% | **−14.6%** | −13.3% |

`always-fast` 强窗翻负 → 否决。条件武装保留正收益，并修好 7/20。

## Rejected: `ext_stock15`

仍绑在 T30 续仓门上（「到点再看股价」），**防不住**中途反转磨到 SL/T30。双窗收益好看，但不符合风险目标。

## Thesis

固定 T30 **不是**边。May–Jul 基线按出场原因拆（交易收益加总）：

| reason | n | sum_ret |
|--------|--:|--------:|
| TP | 28 | **+17.28** |
| T+45 | 15 | +2.04 |
| T+30 | 11 | **−1.14** |
| TRADE_TOX | 3 | −0.78 |

高收益来自打到 TP 的趋势腿；T+30 是时钟磨亏桶。Layer 5：持仓必须回答「现在还有什么路径证据继续拿」。

## 7/20 实盘动机

5 笔 UP，sum trade-ret ≈ **−168%**（离线最差单日约 −59%）。典型 giveback：MSFT +38%→T+30 −19%；GOOGL +40%→EOD −11%。  
归因：`A_dominates_false_hold_not_signal`。宏观/震荡可解释入场脏，**亏损放大是无条件时钟持有**。

Fused m5（4/5 笔对齐）反事实：

| variant | sum trade-ret | 要点 |
|---------|--------------:|------|
| live_clock | −48.5% | T+30/SL |
| rev10 | −44.6% | AMD 早砍；MSFT 锁绿 |
| trail20 | −40.7% | GOOGL/MSFT TRAIL |
| **rev10+trail20** | **−22.3%** | 双轨最佳 |

## Offline scoreboard（`path_hold_lit_v1`）

| window | variant | total_ret | MaxDD | worst_day | loser med hold | TP/TRAIL/REV |
|--------|---------|----------:|------:|----------:|---------------:|--------------|
| May–Jul | baseline_t30 | **+1856%** | **−5.4%** | −5.4% | 1800s | 28/0/0 |
| May–Jul | lit_trail_rev | +129% | −16.0% | −7.6% | **600s** | 7/24/21 |
| Jan–Mar | baseline_t30 | +86% | −17.8% | −11.1% | 1800s | 15/0/0 |
| Jan–Mar | lit_trail_rev | +22% | **−16.3%** | **−7.7%** | **600s** | 3/20/30 |

解读：强窗收益大幅让步（trail 误杀部分 TP）；弱窗 **worst-day / MaxDD 改善**，亏损持仓中位从 30m→10m。风险优先 promote research，**非**收益 freeze。

Fused 7/20 m5：day −8.3%→**−4.6%**，sum trade −48%→**−22%**（TRAIL×2 + STOCK_REV×2）。

## 历史消融摘要

| 方案 | 结果 |
|------|------|
| 废除 T30 → 激进 delta/trail | 强窗腰斩（误杀 TP） |
| `STOCK_REV` always@20 + 严阈值 | Jan–Mar MaxDD 恶化 |
| `ext_stock15` | 收益好看，worst-day/MaxDD 无改善 |
| **lit trail+rev+T45 safety** | 上表；`best_risk=lit_trail_rev` |

## 实现注意

- `STOCK_REV` / `TRAIL` 必须在 OMS stub **early-exit** 集合里。
- `broker_oms.evaluate_exits`：`STOCK_REV` 与经典 `TRAIL`（`mtm_trail`）均在硬 SL / 时钟前。
- Fused/live：`Mag7OmsStub._sync_live_stock_into_session` 同步当日股票棒。
- 7/20 全簿对齐用 `scheme=m5` + `disable_prevention`（peer3 `single` 往往只复现 AMD）。

## 运行

```bash
python maga7/tools/run_path_hold_lit_ablation.py \
  --windows may_jul,jan_mar \
  --out /mnt/s990/data/maga7/results/path_hold_lit_v1
```

## 下一步

1. Shadow `path_hold_lit_v1`；对比 `cond_ladder_loose_wash` 是否毒日叠用  
2. 网格：`trail_activate` ∈ {0.15,0.20} × `trail_dd` ∈ {0.10,0.12}（风险优先，非收益优先）  
3. 入场侧 state gate（减少 wash 假分散）与出场三轨正交  
