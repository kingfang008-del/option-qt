# 条件化持仓延长：T30 → T45（hold_extend）

> 研究日期：**2026-07-17**  
> 对照基线：[`single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json)  
> 时钟：`bar_availability_delay_seconds=60` · rails TP×1.6 / SL×0.4 · Mag7+GOOGL · `peer_align_min=3`

## 1. 动机

硬 T+30 在因果窗上已经很强，但 oracle / hold 消融显示赢家右尾常落在 30–47 分钟。  
无条件拉到 T45 在 May–Jul 反而更差；因此试 **条件化延长**：多数单仍 30 分钟离场，仅浮盈单多拿一截。

## 2. 规则

`exit_mode=hold_extend`（实现：`maga7/common/replay.py` → `simulate_trade`）

| 项 | 值 |
|---|---|
| 基础超时 | `hold_minutes=30` |
| 延长上限 | `hold_extend_minutes=45` |
| 触发门 | 触及 T+30 时，期权 MTM ≥ `hold_extend_mtm_min`（默认 0） |
| 可选确认 | `hold_extend_require_mf=true` 时还要求 mf10 仍与开仓方向同向 |
| rails | 全程保留；延长后仍可 TP / SL |

研究 profile：[`single_qqq_open_ladder_atm5otm_t30_hold_extend45_peer3_v1`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_hold_extend45_peer3_v1.json)（默认 `require_mf=true`；消融另含 MTM-only）。

复现：

```bash
python -m maga7.tools.run_hold_extend_ablation \
  --start-date 2026-05-01 --end-date 2026-07-13 \
  --tag hold_extend_ablation_mag7_googl_peer3_may_jul

python -m maga7.tools.run_hold_extend_ablation \
  --start-date 2026-01-02 --end-date 2026-07-13 \
  --tag hold_extend_ablation_mag7_googl_peer3_jan_jul
```

## 3. Scoreboard

### 3.1 May–Jul（2026-05-01 ~ 2026-07-13，n=53）

| 变体 | Ret | MaxDD | T+45 出场 | vs 基线 |
|---|---:|---:|---:|---|
| baseline T30 rails | +374.8% | -16.5% | 0 | — |
| 无条件 T45 rails | +340.3% | -20.2% | 23 | −35pp |
| **extend MTM+mf** | **+398.6%** | **-16.2%** | 9 | **+24pp** |
| **extend MTM only** | **+401.1%** | **-16.2%** | 11 | **+26pp** |
| extend MTM≥5%+mf | +345.2% | -16.2% | 9 | −30pp |

产物：`maga7/results/hold_extend_ablation_mag7_googl_peer3_may_jul/`

### 3.2 Jan–Jul（2026-01-02 ~ 2026-07-13，n=124）

| 变体 | Ret | MaxDD | T+45 出场 | vs 基线 |
|---|---:|---:|---:|---|
| baseline T30 rails | +1299% | -32.2% | 0 | — |
| 无条件 T45 rails | +1390% | -35.7% | 68 | +91pp（DD 变差） |
| **extend MTM+mf** | **+1483%** | **-32.2%** | 20 | **+184pp** |
| **extend MTM only** | **+1550%** | **-32.2%** | 25 | **+251pp** |
| extend MTM≥5%+mf | +1414% | -32.2% | 18 | +115pp |

产物：`maga7/results/hold_extend_ablation_mag7_googl_peer3_jan_jul/`

## 4. 健康度（连亏）

相对 **旧 Mag7-only T+30**（May 有 **4 日**连亏 05-06~05-11，复利约 -21%），peer3 之后 May–Jul **已无 ≥4 日连亏**。

`extend_mtm_only` May–Jul：

| 最长连亏 | 窗口 | 复利 |
|---|---|---:|
| 3 日 | 05-19~05-21 | -13.4% |
| 3 日 | 07-07~07-09 | -12.2% |

与 peer3 硬 T30 同级（仍有 3 日窗，但不再出现旧版 4 日连亏）。  
Jan–Jul 仍有更长连亏（如 04-20~04-24 共 5 日 / -23.9%），与 peer3 基线相同——延长 hold **未放大**该窗 MaxDD（仍 -32.2%）。

日明细：`.../extend_mtm_only/daily.csv`。

## 5. 机制观察

- 抬升主要来自少数浮盈单在延长段打到 **TP**，不是普遍多拿 15 分钟。
- 无条件 T45 在 May 窗更差 → **条件化**是关键。
- `MTM≥5%` 过严，挡掉后续 TP。
- 胜率相对硬 T30 略降（May：64%→55%），因部分浮盈回吐；账户收益与 MaxDD 仍改善或持平。

## 6. 日熔断 A/B（2026-07-17）

此前 `scheme=single` **忽略** `trade.day_circuit`（只有名字含 `circuit` 的 scheme 才开）。  
已改为：profile 里只要设置了 `day_circuit` 即对 single 生效（`null`/`false` 关闭）。

`extend_mtm_only` May–Jul 对照（产物：`results/hold_extend_circuit_ab_may_jul/`）：

| 变体 | Ret | MaxDD | n | 最差单日 | 说明 |
|---|---:|---:|---:|---:|---|
| 无熔断 | +401.1% | **-16.2%** | 53 | **-12.2%** | 原结果 |
| `day_circuit=-5%` | +388.5% | **-16.2%** | 50 | **-12.2%** | 熔断 8 次 |

关键日：

- **05-21**：拦掉第二笔 AMD → 日亏 **-11.6% → -6.5%**（有效）。
- **06-12**：单笔 GOOGL SL，熔断来不及 → 仍 **-12.2%**（MaxDD 主因未动）。
- **05-11**：TSLA SL 后熔断，**砍掉同日 NVDA TP** → 日亏 **-6.9% → -12.2%**（副作用）。

硬 T30 同样：MaxDD 不变（-16.5%），总收益略降（+374.8%→+356.6%）。

结论：`-5%` 日熔断能压「连亏两笔 full sleeve」，**压不住**「一枪 SL×0.4 × 20%≈12%」，且会误伤同日修复单。若要单日 <10%，还需降仓或收紧 SL，不能只靠熔断。

## 7. 大胜后约束（修订）

账户级「大胜日后全局 skip/scale」**方向不对**：会改整体规则、误伤续赢日。  
更合理的是 **标的×方向**：昨日本标的同向大赚（TP / ret≥阈值）→ 今日禁止该标的再开同向。

配置：`block_same_dir_after_win=true`、`block_same_dir_after_win_ret=0.50`  
（实现：`position_size.is_symbol_dir_big_win` + offline/stream）

`extend_mtm_only` May–Jul 事实：23 笔大胜里，次日同标的同向续开 **仅 2 次** ——

| 前日 | 次日 | 若拦住 |
|---|---|---|
| 05-13 NVDA UP TP | 05-14 NVDA UP **-9.6%** | 躲掉一笔亏 |
| 06-09 TSLA DN TP | 06-10 TSLA DN **+60% TP** | 也会砍掉赢家 |

因此该规则 **对症但命中少**，且 **解释不了** 05-19~21 / 07-07~09 连亏（那两段是跨标的/翻向，不是同标的追涨）。  
账户级 `post_win_cooldown_*` 仅作研究遗留，默认 off，不进基线。

## 8. 候选与下一步

| 候选 | 角色 | 备注 |
|---|---|---|
| `extend_mtm_mf` | 更保守 | MTM≥0 **且** mf10 同向；May +398.6% |
| `extend_mtm_only` | 收益更高 | 仅 MTM≥0；May +401.1% / Jan +1550% |

**尚未升格基线。** 下一步在两者间做最终取舍（可再补 live/stream 对齐与 Jan–Jul 冻结叙述），再决定是否替换 catalog 中的 `causal_baseline`。

软出场（mf_flip / flow_die / mtm_trail）结论不变：勿替换硬超时；本方案是 **硬超时上的条件延长**，不是软出场。
