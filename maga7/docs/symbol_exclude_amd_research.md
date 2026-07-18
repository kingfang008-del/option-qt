# 简单处理：剔除 AMD / 加重 NVDA·META

不做复杂纯度分，只改交易宇宙或标的权重。

## 机制

| 键 | 含义 |
|---|---|
| `trade.symbol_exclude` | 不交易的标的（可仍留在 `peer_symbols` 里算宽度） |
| `trade.symbol_size_scale` | 按标的乘仓位，如 `{"NVDA":1.25,"META":1.25}` |

脚本：`maga7/tools/run_symbol_exclude_ablation.py`  
产物：`results/symbol_exclude_ablation_extend_mtm_peer3_may_jul/`

## 消融（May–Jul，extend_mtm_only）

| 变体 | total_ret | MaxDD | n | AMD笔数 |
|---|---:|---:|---:|---:|
| **extend_mtm_only** | **+401.1%** | **-16.2%** | 53 | 8 |
| **no_amd** | +299.4% | -24.0% | 54 | 0 |
| no_amd（peer 也去掉） | +256.4% | -24.0% | 53 | 0 |
| nvda_meta ×1.25 | +415.0% | -16.8% | 53 | 8 |
| no_amd + nvda_meta×1.25 | +306.2% | -28.8% | 54 | 0 |
| amd ×0.5 | +369.9% | -15.4% | 53 | 8 |
| **full_day** | **+673.3%** | **-12.2%** | 44 | 5 |
| full_day_no_amd | +523.4% | -24.0% | 46 | 0 |

## 为何剔 AMD 反而更差

1. **AMD 全期并非净拖累**：底仓 8 笔 AMD 里有赢家；一刀切掉会少赚。  
2. **占坑被释放后换上更差的票**：  
   - 07-07：去掉 AMD 后多进 **TSLA -14.7%**（NVDA -30.6% 仍在）→ 更亏  
   - 07-09：META 替上，期权仍 **-8%**（好于 AMD -18%，但补不回全期）  
3. **加重 NVDA/META**：收益略升到 +415%，但把 07-07 NVDA、07-08 META 亏单放大，MaxDD 略差。

## 结论

- **不要**用「剔除 AMD」当默认优化——全期收益与回撤都变差。  
- **不要**裸加重 NVDA/META——会放大同标的假信号。  
- 简单规则里更干净的仍是：**事件日历 `full_day`**，以及可选 **`confirm_1_mf`**（挡 07-07 NVDA），而不是静态剔票。
