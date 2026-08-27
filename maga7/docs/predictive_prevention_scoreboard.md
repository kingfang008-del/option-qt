# Predictive Prevention 双窗 Scoreboard

> 日期：**2026-07-20**  
> 工具：`python -m maga7.tools.run_prevention_scoreboard`  
> 产物：`maga7/results/prevention_scoreboard_dual_window/`  
> 基线：research peer3（L1 degrade/halt + L2 hunter 保持），只拧 `watchdog.prevention`

## 变体

| tag | 含义 |
|-----|------|
| `00_off` | prevention 关 |
| `01_soft` | `prefer_risk_off=false` → `up_toxic`（UP×0.5） |
| `02_hard` | `prefer_risk_off=true` → `up_toxic_block`（禁 UP） |

规则：`mixed_wash_up`（washout breadth≥3 @0.8%，`frac_above∈[0.35,0.70]`）。

闸门：强窗 vs off ≥90%（研究）/ ≥95%（freeze）；弱窗 ≥85%。

## 结果

| window | variant | total_ret | vs_off | trades | trigger_days | bad≤−3% sum |
|--------|---------|----------:|-------:|-------:|-------------:|------------:|
| May–Jul | off | **+1856%** | 100% | 57 | 0 | −8.9% |
| May–Jul | soft | +1003% | **54%** | 54 | 22 | −8.9% |
| May–Jul | hard | +628% | **34%** | 42 | 22 | −8.9% |
| Feb–Apr | off | +296% | 100% | 68 | 0 | −71.2% |
| Feb–Apr | soft | +249% | 84% | 68 | 15 | −65.0% |
| Feb–Apr | hard | +203% | 69% | 57 | 15 | −63.3% |

## Verdict

**FAIL — 默认保持 `prevention.enabled=false`。**

原因：

1. 触发过密：强窗 22 日 / 弱窗 15 日被标成 mixed-wash（约占交易日 1/3–1/2）。  
2. 强窗收益被腰斩以下（soft 54%、hard 34%），远低于 90%/95% 闸。  
3. 强窗 bad-day 合计几乎不变；弱窗 bad 略改善，但代价是砍掉大量盈利 UP。

今日（2026-07-20）硬防能空仓避亏，但**不能**用当前阈值换全样本。

## 下一步（收窄规则，再开 scoreboard）

按触发日数目标 ≈ 强窗 ≤5–8 日 / 季 试邻域：

| 旋钮 | 候选 |
|------|------|
| `washout_breadth_min` | 4 或 5（今日=4） |
| `wash_drop_min` | 0.010–0.012 |
| `frac_above_max` | 0.60（收窄「混合」） |
| 附加 | QQQ 仍绿 / bounce&lt;X% 才防（对齐今日形态） |

软臂只有在 hard 仍过猛但 soft 能过 90% 时再考虑。
