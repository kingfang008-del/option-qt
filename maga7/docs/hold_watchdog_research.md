# Hold Watchdog（持仓期极端反向）

**日期：** 2026-07-19  
**状态：** 研究旋钮，**research_baseline 默认 `enabled=false`**  
**动机：** Entry Watchdog 只管开仓；持仓 T+30 只有 TP/SL / hold_extend。突发指数级反向（「一句话」式跳水）需要独立的持仓闸。

## 规则（窄触发）

配置：`trade.hold_watchdog`

| 字段 | 默认 | 含义 |
|------|------|------|
| `enabled` | `false` | 总开关 |
| `qqq_adverse_from_entry` | `0.008` | 入场后 QQQ 逆向 ≥ **0.8%** → 平仓 |
| `min_hold_seconds` | `60` | 最短持有后再看 |
| `require_option_mtm_max` | `null` | 若设为 `0`，仅当期权 MTM≤0 时才因冲击平仓 |

极性：

- **UP 仓**：QQQ 相对入场价跌 ≥ 阈值 → `HOLD_SHOCK`
- **DN 仓**：QQQ 相对入场价涨 ≥ 阈值 → `HOLD_SHOCK`

时钟：与基线一致，用 1m QQQ + `bar_availability_delay_seconds`（默认 60s）因果可见。

## 与现有层的关系

| 层 | 角色 |
|----|------|
| L1 Degrade/Halt | 日初/早盘 **禁开/缩仓** |
| L2 Hunt | 早盘短窗 **加开** |
| 期权 TP/SL / extend | 持仓内 **合约轨** |
| **Hold Watchdog** | 持仓内 **指数冲击轨**（本文件） |

不做：个股 mf_flip（已 REJECT）、宽 VIXY 禁开（过猛）。

## 消融

```bash
python -m maga7.tools.run_hold_watchdog_ablation \
  --start-date 2026-05-01 --end-date 2026-07-17
```

产出：`results/hold_watchdog_ablation_peer3_may_jul/`  
升线门槛建议：**MaxDD 改善或毒性日减轻**，且 total_ret ≥ 关闸臂的 **95%**。

### May–Jul（L2+05）消融结果（2026-07-19）

| arm | total_ret | MaxDD | n_HOLD_SHOCK | 备注 |
|-----|----------:|------:|-------------:|------|
| off | **+1528%** | −12.2% | 0 | 基线 |
| qqq **0.8%** | +1504% | −12.2% | 1 | 仅 06-11 TSLA UP；早砍 −24.5% vs 原 T+30 −17.6%（**更差**） |
| qqq **1.0% / 1.5%** | +1528% | −12.2% | 0 | 本窗未触发 |
| 0.8% + MTM≤0 | +1504% | −12.2% | 1 | 同 0.8% |

**解读：** 强窗几乎遇不到「指数级冲击」样本；0.8% 偏敏感，偶发误砍会锁更深亏损。  
**默认保持 off**；若要当保险，倾向 **≥1.0%**（本窗无差分，专防更极端事件）。Live 真·秒级冲击另需 OMS 帧级接线。

## 代码

- `common/hold_watchdog.py`
- `simulate_trade(..., qqq_day=, hold_watchdog=)` → reason `HOLD_SHOCK`
- offline / stream / OMS dry fill session 已接线
- 单测：`tests/test_hold_watchdog.py`

## Live 备注

Shadow/dry 走 `simulate_trade` 即生效。Paper/Live 真实 IB 持仓需在 OMS 帧循环里复用同一 QQQ 判定（后续工程项）；先以 offline 消融定阈值。
