# AM TOD 口袋退出 Distill

> 前置：[`am_vwap_foresight_map.md`](am_vwap_foresight_map.md)  
> 工具：`tools/scan_am_pocket_exit_distill.py`  
> 产物：`results/research_am_pocket_exit_distill/`

## 协议

1. 只用对齐前视探针（`dir == sign(from_open)`）落在 discover 选出的 TOD 口袋。
2. 每标的×日×session 只取最早一笔（避免 60s 网格重复）。
3. 入场结构门：`pocket` / `accel0` / `vd*` / `fo30_*` 等（非全窗 FO thr 网格）。
4. 退出：TP/SL ± confirm-abort；定价 = option trade last±1%。
5. **选参**：discover (`may_jul09`) PASS；blind (`jul10_23`) 要求 mean>0、day_win≥0.50、n≥5。

## 结果（trades）

`DISTILL_PASS`，**25** 格 dual。

| 候选 | MJ n / mean / day_win | J10 n / mean / day_win | 评 |
|------|----------------------:|-----------------------:|----|
| **`accel0 \| tp0.2/sl0.25/h300`** | 188 / **+5.1%** / 0.64 | 39 / **+1.0%** / 0.56 | **稳健首选**（样本够） |
| `vd10 \| tp0.2/sl0.25/h900` | 35 / +8.0% / 0.77 | 14 / +0.5% / 0.71 | 均值最高但 n 薄 |
| `vd8 \| tp0.2/sl0.25/h900` | 64 / +4.3% / 0.59 | 18 / +2.8% / 0.63 | 折中 |
| `vd5 \| tp0.2/sl0.25/h900` | 135 / +3.4% / 0.60 | 33 / +2.7% / 0.56 | 可备份 |
| plain `pocket` | 多数弱 | — | 几乎靠退出；结构门必要 |

Confirm-abort 仅 1 格弱 PASS，未胜过对应纯 TP/SL。

## 规则草稿（稳健）

```
session/TOD ∈ foresight pockets
dir = sign(from_open)          # 已对齐
accel_10_30 ≥ 0                # 10s VWAP 相对 30s 同向不减速
ATM option, TP 20% / SL 25% / max_hold 300s
```

**尚未**做 quote FillSpec dual；升 shadow 前必须补 lag5/sp15。

## 命令

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_pocket_exit_distill \
  --with-ca --tag research_am_pocket_exit_distill
```
