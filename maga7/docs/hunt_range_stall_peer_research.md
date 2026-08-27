# Hunt × range_stall peer（06-24 AMD）

**Status: DUAL_FAIL — 不接线**（2026-07-27）

## 问题

06-24 AMD UP **Hunt** `washout_reclaim` → T+30 **−26%**。  
若在 **10:00 成交钟** 上算 peer，则 `peer≤3 ∧ pre5≈0` 触发已接线的 `range_stall` peer_pre5 臂。  
但 Hunt 默认 `skip_peer` → `peer_n=None` → RS 臂全部空转。

## 陷阱（已踩）

1. **把 peer_n 喂给全体 morph**：`peer_gap` 会误杀 07-01 META Hunt TP（+63%）。peer 只能给 RS。  
2. **决策钟 ≠ 成交钟**：信号 09:51、delay→09:52 时 pre5 仍热；期权 quote 从 **10:00** 才有，fill 在 10:00，此时才 stall。RS 必须 `hunt_asof=signal_deadline`。

## 双窗 v2

| arm | weak | strong | 06-24 AMD Hunt | 备注 |
|-----|-----:|-------:|:--------------:|------|
| OFF | 1.000 | 1.000 | 仍在 | spine |
| HUNT_RS_PEER | 1.000 | 1.000 | 仍在 | 决策钟 no-op |
| **HUNT_RS_PEER_T10** | **1.000** | **0.918** | **清除** | 顺带砍 07-01 META Hunt TP |

产物：`/mnt/s990/data/maga7/results/hunt_range_stall_peer_dual_v2/`  
工具：`python -m maga7.tools.run_hunt_range_stall_peer_dual`

## 裁决

强窗 keep **0.92 &lt; 0.95**，且误伤 Hunt 大赢家 → **不晋级**。  
06-24 日线因去掉 AMD 从 +3.7%→+9.4%，但账户总收益被 META 07-01 拖回去。

代码能力保留（默认关）：`range_stall_gate.hunt_peer_align` / `hunt_asof`。

## 下一步（可选）

更窄的 Hunt 专用臂（例如仅 `washout_reclaim` + peer_pre5 + T10），或接受 tox/T+30 路径处理 AMD，而不是扩 RS。
