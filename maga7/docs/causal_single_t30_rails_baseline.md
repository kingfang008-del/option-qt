# Mag7 因果基线：single + T+30 + TP/SL

> 冻结日期：2026-07-16  
> 时钟：`bar_availability_delay_seconds=60`（预聚合 1m 股票表 → 分钟完成后才可交易）  
> 成交：open_ladder OTM5，`fill_frac=0.8`，QQQ align regime，concurrent `position_frac=0.20`

## 1. 当前基线（以此为准，后续优化都相对它）

| 项 | 值 |
|---|---|
| **Profile** | [`single_qqq_open_ladder_atm5otm_t30_rails_p20_v1`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_v1.json) |
| Scheme | **`single`**（每标的每日至多首笔 Rule-A，不做 only_win 复入） |
| 出场 | **`exit_mode=none`**：TP×1.6 / SL×0.4 / **T+30**（无 `mf_flip`） |
| 选约 | `open_ladder`，`ladder_otm_rungs=5` |
| 仓位 | concurrent p20（独处 20%；并发第二腿 10%；最多 2） |

```bash
python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_v1.json \
  --scheme single \
  --tag replay_single_t30_rails_p20_jan_jul_delay60
```

冻结产物：`maga7/results/replay_single_t30_rails_p20_jan_jul_delay60/`  
出场/复入消融：`maga7/results/exit_reentry_ablation_{may,jan}_jul/`

---

## 2. 为什么旧「约 33 倍 / -22%」是虚幻的

旧 headline（`+3375% / MaxDD -22%`，约 247 笔）来自 **未加分钟可用性延迟** 的研究口径，不能当作可交易业绩。

根因（预聚合 1 分钟股票路径）：

1. 分钟 bar 的 `feature_ts = M` 表示区间 `[M, M+1min)`。  
2. 该分钟的 OHLC / mf / Rule-A 特征，要到 **`M+1min` 之后** 才因果可见。  
3. 旧 replay 在 `decision_ts ≈ M`（等价 `bar_availability_delay_seconds=0`）就入场，等于用了整分钟信息却提前约 **60 秒** 成交。  
4. 策略对入场早晚极度敏感：把 delay 从 0→30→60 时，收益从数千百分比掉到数百再掉到约 +50%（旧 mf_flip 栈）。  
5. 临时把 delay 设回 0，可精确复现旧笔数/合约/收益 → 证明崩塌来自 **前视时钟**，不是数据偶然坏了。

因此：

- **+3375% / -22%**：pre-delay 研究记录，已作废为上线证据。  
- **可当真的旧栈因果口径**（仍用 mf_flip + only_win）：Jan–Jul 约 **+50% / MaxDD -52%**。  
- 架构说明见 [`current_architecture.md`](current_architecture.md)。

---

## 3. 因果基线成绩

### 3.1 Jan–Jul（2026-01-02 ~ 2026-07-13）

| 指标 | single + T+30 + TP/SL | 旧因果 mf_flip+only_win |
|---|---|---|
| 总收益 | **+863.6%** | +49.8% |
| MaxDD | **-28.6%** | -51.7% |
| 笔数 | 134 | 207 |
| 胜率 | 55.2% | 36.7% |
| 单笔期望 | +11.4% | +2.2% |

分月账户（权益从 100）：  
1 月 +108% → 2 月 +3.6% → 3 月 -8.2% → 4 月 +27% → 5 月 +79% → 6 月 +63% → 7 月（至 13 日）+32%。

### 3.2 May–Jul（2026-05-01 ~ 2026-07-13）— 健康度核对

| 指标 | 值 |
|---|---|
| 账户总收益 / MaxDD | **+252.8% / -25.1%** |
| 笔数 / 胜率 | 64 / **57.8%** |
| 平均盈利 / 平均亏损 | +42.1% / -25.5% |
| 盈亏比（均盈/\|均亏\|） | **1.65** |
| 盈利因子（毛利/\|毛亏\|） | **2.26** |
| 有交易日胜率 | 54.8%（42/49 个交易日有单） |

分月：5 月 **+64.5%**，6 月 **+62.5%**，7 月（至 13 日）**+32.0%**。  
日路径少见长连亏；明细见  
`results/exit_reentry_ablation_may_jul/single_t30_rails/daily_pnl_may_jul.csv`。

---

## 4. 为何相对旧因果栈能抬这么多

同 delay=60 消融（`run_exit_reentry_ablation`）表明边主要来自规则，不是扫描器突然变准：

1. **`mf_flip` 过早下车**：对齐首笔时，flip 出场均值显著低于死拿/T+30+轨道。  
2. **only_win 复入期望为负**：首笔尚可，复入拖累账户。  
3. **`single + T+30 + TP/SL`** 在 May–Jul 与 Jan–Jul 均为收益/回撤最优稳健点；纯 T+60 无轨道收益偶发更高但 MaxDD 明显恶化。

扫描侧补充：大波动日 Rule-A 召回高，但仍有约 25–35% 偏弱/假突破噪声；TopK 精确率（\|day\|≥2%）约 75%。信号边存在，完整旧栈被出场与复入吃掉。

---

## 5. 后续优化起点

后续一切对比默认相对本基线（delay=60 + single + T+30 rails）：

- 质量过滤（慎用 `vol_z≥2`，May–Jul 曾伤收益）  
- 缺口日 `prev_close` 清理  
- 秒级完成分钟后的真实 `available_ts`（勿再叠 60s）  
- stream parity / live 对齐本 profile + `scheme=single`

旧临时生产 [`m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1`](../CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json) 仅作对照，不再作为因果优化基准。
