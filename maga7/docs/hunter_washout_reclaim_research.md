# Hunter：washout → reclaim-open（研究线）

> 总览：[`watchdog_stack_architecture.md`](watchdog_stack_architecture.md)

## 动机

07-17 MU 类行情：开盘深洗后 V 反，期权弹性在 **09:45–10:00**。  
现有 Hunter 原型不适用：

| 检测器 | 问题 |
|--------|------|
| `orb_fractal` | 独立开仓双窗负（已 REJECT） |
| `early_mf` | 信号偏晚（MU 10:10 追高）；强窗腰斩（已 REJECT） |
| Halt `washout_and_reclaim` | **极性相反**：多标的深洗+假收回 → **停手**，不是开仓 |

本线：单票（或窄门）**深洗盘后收盘收回开盘价** → 短窗 Hunt UP。

## 信号（因果，1m）

1. `09:30`–`wash_window_end` 内相对开盘回撤 ≥ `wash_drop_min`  
2. 洗盘低点之后，第一根 **收盘 > 开盘**（`reclaim_level=open`）→ 开火  
3. 截止 `signal_deadline`（默认 10:15）；`top_k=1` / 日预算 1  

实现：`maga7/common/orb_open.py` → `detect_washout_reclaim` / `scan_washout_reclaim_day`  
接入：`watchdog.hunter.detector = "washout_reclaim"`

与 Halt 的关系：默认 `block_when_halt=true`（架构优先级不变）。Mag7 全市场 Halt 日不 Hunt；单票研究 case 可关 Halt。

## 配置（研究默认）

- Profile：`…_watchdog_hunter_washout_reclaim_v1.json`  
- Watchdog：`CONFIG/watchdog/degrade_halt_hunter_washout_reclaim_v1.json`  
- **推荐 `wd15 + opp`**：`wash_drop_min=0.015`，`mutex_scope=symbol_dir`，`allow_baseline_opposite=true`  
- Freeze：`hunter.enabled=false`（本线不进默认）

```bash
python -m maga7.tools.run_washout_reclaim_hunter_scoreboard \
  --out maga7/results/watchdog/hunter_washout_reclaim_v2_opp
```

## 验收（2026-07-18）

门槛：双窗相对 freeze ≥95%。

### v2（反向基线放行，当前默认）

| window | baseline (L0) | L1 only | +hunter v2 opp (L2) | L2 vs L0 | n_hunt |
|--------|---------------|---------|---------------------|----------|--------|
| May–Jul | 810% | **875%** (~108%) | **1255%** | **155%** | 12 |
| Feb–Apr | 140% | — | **152%** | **108%** | 5 |

> **防忘**：口中「约 1200% / 1255%」= 本表 **L2**，不是 research_baseline 默认的 L1（+875%）。  
> L2 **未**并入 `peer3_v1`；基线对照见 [`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md)。

明细：`results/watchdog/hunter_washout_reclaim_v2_opp/`、消融 `…/hunter_washout_reclaim_v2/ablation_v2.csv`。  
**每日盈亏表（May–Jul）**：[`L2_strong_daily_pnl.md`](../results/watchdog/hunter_washout_reclaim_v2_opp/L2_strong_daily_pnl.md)（含 vs L1/L0）。

**Verdict: `ACCEPT_RESEARCH`**（相对 v1 再抬强窗；弱窗持平；仍不进 research_baseline / 生产）。

### 升线闸门复验（2026-07-18）

完整表：[`l2_hunter_validation_gates.md`](l2_hunter_validation_gates.md)。

| 闸门 | 结果 |
|------|------|
| 流式对拍（stock_1s，May–Jul，60 笔含 12 Hunt） | **PASS** |
| 邻域 wd×opp | **PASS_NEIGHBORHOOD**（勿降到 1.2%） |
| OOS 2025H2 | vs L0 OK，但 **L2 172% < L1 191%** |
| 并入 research_baseline | **仍 NO** |

### v1 → v2 关键修复

拖累根因：Hunt UP 后 **同标的 mutex + 日单笔上限** 挡掉后面 Rule-A DN（06-26 / 07-02）。  
v2：`mutex_scope=symbol_dir` + `allow_baseline_opposite=true` → 允许反向基线。

| 日期 | v1 day_ret | v2 day_ret | 说明 |
|------|------------|------------|------|
| 06-26 | −0.4% | **+12.0%** | Hunt 小亏后仍吃到 AMD DN TP |
| 07-02 | −2.5% | **+5.0%** | Hunt −53% + AMD DN + TSLA DN（仍弱于纯基线 +17%，因 Hunt 亏） |

### 旋钮消融

| 变体 | 强窗 vs freeze | 弱窗 vs freeze | 备注 |
|------|----------------|----------------|------|
| wd10_h0 | 104% | 77% | REJECT |
| wd15_h0（v1） | 123% | 108% | 初版候选 |
| **opp_only（v2）** | **155%** | **108%** | **当前默认** |
| opp_h1 | 118% | 143% | 弱更强、强略逊 |
| opp_buf3 | 101% | 125% | 缓冲偏紧 |
| opp_buf3_h1 | 86% | 127% | REJECT |

### MU 07-17 case（池外）

Halt 关闭、MU-only：`09:51` Hunt ATM call → TP **+61%** 期权 / 日账户 **+12.3%**  
（对比 `early_mf` 10:11 入场 −35%）。  
结果：`results/mu_0717/mu_washout_reclaim_wd15/`。

## 纪律

- 不改 freeze 信号窗 / Rule-A。  
- 默认 `hunter.enabled=false`；升线需双窗复验 + live 纸面。  
- 与 Mag7 Halt 窄门并存：Halt 保坏日，本 Hunter 只在非 Halt 日抢单票 V 反。
