# TFT 协议：Base 冻结 → 近月微调 → July 前向报告

**日期：** 2026-07-22  
**状态：** 已采纳（勿再把 val 扩到 1–6、把 7 月当干净 test）

相关：[`july_tft_lock_vehicle_correction.md`](july_tft_lock_vehicle_correction.md) · [`honest_live_kpi_finetune_replay.md`](honest_live_kpi_finetune_replay.md)

---

## 1. 原则

| 做 | 不做 |
|----|------|
| V4 **test=4–6 永久冻结**，只作基线外推证据 | 把 4–6 塞进 val 再选模 |
| 近月 **微调**（默认 5–6）适应 regime | 默认 val=1–6 / test=7 |
| 锁约纠正后 **7 月一次性前向报告** | 用已看过的 Jul W1 金标回灌选轨 |
| 标签/锁约定义变了 → **重训 Base** | 在错误锁约特征上微调充数 |

真正未偷看的 OOS 优先留给 **8 月**；7 月报告标为 `FORWARD_REPORT`（非干净 holdout）。

---

## 2. 三层切分

| 层 | 模型角色 | 数据 | 产出 |
|----|----------|------|------|
| **Base** | V4 pretrain | train→2025-12；val=2026-01–03；**test=2026-04–06 冻结** | `checkpoint/checkpoints_qqq_v4/` |
| **Adapt** | FT（FT56 谱系） | FT 窗=2026-05+06；val≈2026-06；init=V4 | `checkpoint/checkpoints_qqq_ft56_julw1/` |
| **Forward** | V4 vs FT 对照 | **old_style 前视锁约** 的 2026-07（勿用 open_4bucket） | `results/offline_live_aligned/*_july_forward_*` |

对照车辆（非训练一致）：maga7 OTM3/OTM5 开盘阶梯，见锁约纠正文档。

---

## 3. 何时重训 / 何时微调

| 条件 | 动作 |
|------|------|
| 训练史标签/锁约/特征定义变更 | **重训 Base**（`rebuild_train_v4*.sh` / `train_v4_old_v2_retrain.sh` 谱系） |
| 定义不变，只吃近月 IV/波动 | **微调 Adapt**（`train_ft56_julw1_honest_kpi.sh`） |
| 仅补齐 July 前视 1s 做报告 | **不重训**；下载 → `build_v4_old_lock_month.sh` → frozen+VX replay |

当前默认：Base LMDB/ckpt 已存在 → **先不重训**；优先打通 Forward 前视链，Adapt 可 `SKIP_TRAIN=1` 复用 FT56。

---

## 4. 一键入口

```bash
# 就绪检查（默认）
bash qqq_btc/tools/run_tft_base_ft_july_forward.sh check

# 下载 July old_style 1s（需 MASSIVE/POLYGON key）
bash qqq_btc/tools/run_tft_base_ft_july_forward.sh download_july

# 建 July 特征月包
bash qqq_btc/tools/run_tft_base_ft_july_forward.sh build_july

# 近月微调（可选 SKIP_TRAIN=1）
bash qqq_btc/tools/run_tft_base_ft_july_forward.sh adapt

# July 前向报告（V4 + FT56，frozen + VX）
bash qqq_btc/tools/run_tft_base_ft_july_forward.sh forward

# 全流程（check → download → build → adapt → forward）；重训需显式开
FORCE_BASE_RETRAIN=1 bash qqq_btc/tools/run_tft_base_ft_july_forward.sh all
```

---

## 5. 门禁

Forward 报告写入前必须满足：

1. 锁约 map 为 **old_style foresight**（非 open_4bucket）  
2. 1s 根目录与 map 一致，且 **未** 混入开盘锁污染树  
3. 特征用 **frozen_norm**；门控与 live-aligned 配方一致  
4. summary 标注 `protocol=base_ft_july_forward`、`lock=old_style_foresight`

不满足则只允许车辆对照（OTM ladder），不得宣称训练一致 KPI。
