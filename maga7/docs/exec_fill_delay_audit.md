# 执行审计：fill × delay（freeze 栈）

## 目的

信号层已接近天花板后，先回答：**基线边在更差成交 / 不同 bar delay 下还剩多少**。  
不改 freeze；纯 `entry_frac` / `exit_frac` / `bar_availability_delay_seconds` 网格。

## 跑法

```bash
/home/kingfang007/anaconda3/envs/ibkr/bin/python -m maga7.tools.run_fill_delay_audit \
  --entry-fracs 0.6,0.8,1.0 --delays 30,60 \
  --out maga7/results/exec_audit/fill_delay_grid
```

Freeze 锚点：`entry_frac=exit_frac=0.8`，`delay=60`。

## 判读

| 现象 | 含义 |
|------|------|
| `entry_frac=1.0` 仍 ≥80% freeze | 边对「贴 ask 买」相对耐打 |
| `entry_frac=1.0` 腰斩 | replay 偏乐观；live 需 requote/更严选约 |
| `delay=30` ≫ `delay=60` | 边依赖「晚进」；加速有 uplift 但实盘未必跟得上 |
| `delay=30` ≈ 或更差 | 60s 因果延迟不是主要拖累 |

产出：`results/exec_audit/fill_delay_grid/scoreboard.csv`。

## 首轮结果（2026-07-18，`exit_frac=entry_frac`）

Freeze 锚点：`ef=0.8 / d=60` → May–Jul **+810%** / Feb–Apr **+140%**。

| entry_frac | delay | May–Jul vs freeze | Feb–Apr vs freeze | 备注 |
|------------|-------|-------------------|-------------------|------|
| 0.6 | 60 | **110%** | **147%** | 更优成交假设显著抬升弱窗 |
| 0.6 | 30 | 105% | 124% | |
| **0.8** | **60** | **100%** | **100%** | freeze |
| 0.8 | 30 | 95% | 104% | 加速几乎持平，强窗略损 |
| 1.0 | 30 | 92% | 82% | 贴 ask 仍大体存活 |
| 1.0 | 60 | **75%** | **50%** | 弱窗腰斩级敏感 |

### 结论

1. **边对成交质量敏感，尤其弱窗**：`entry_frac=1.0`（约对手价买）在 d=60 下弱窗只剩约一半；强窗仍有 ~75%。  
2. **delay=30 不是免费午餐**：相对 freeze（d=60）强窗略差或持平，说明 60s 因果延迟并非主要拖累，也未证明「更早进」稳定更好。  
3. **replay 默认 0.8 偏乐观于最差成交，但未到「全假」**：恶劣 fill 下边还在，live 若经常摸到 ask，预期应向 `ef=1.0` 一档靠拢做压力测试。  
4. **下一刀（执行）**：requote 仿真 vs 静态 fill；OTM rung / DTE 轻消融——看能否在不改信号下抬升「坏成交」分位。
