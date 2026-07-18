# 诚实 KPI：微调 + 离线 Replay（贴近实盘）

> 用途：统一「能否当作实盘预期」的离线验收口径，避免与 databento / 默认 REPLAY 开卷对照数字横比。  
> 配套脚本：`qqq_btc/tools/train_ft56_julw1_honest_kpi.sh`  
> 流式三闸门：`qqq_btc/docs/honest_3gate_live_parity_handoff.md`  
> 微调窗检索：`qqq_btc/docs/week_regime_match_finetune.md`

---

## 1. 结论先说

**「诚实特征 + 因果 put_gate + LIVE 门控」相对更贴近实盘**，应作为离线主 KPI。

| 组件 | 含义 | 为何更贴近实盘 |
|---|---|---|
| 诚实特征 | `july_w1_v4_honest_openwin` | 无 greek-parity / 无开卷金标注入 |
| 因果 put_gate | raw 1min `vix_level`，`ts+1m` 后 `merge_asof(backward)` | 无 5min asof 桶内前视 |
| LIVE 门控 | `LIVE_REPLAY` + `edge_q10_floor=-0.2` | 与线上 immediate entry / 门控语义对齐 |

仍不是实盘本身：分钟 infer + 填价模型 ≠ Redis tick / OMS；特征仍可能与 FCS 流式有缝。最终上线仍要诚实三闸门流式对拍。

**反例（不要当实盘预期）**  
曾用 `july_w1_v4_databento` + 默认 `REPLAY` + 特征列 `vix_level` 得到类似：

| 模型 | acct25 |
|---|---:|
| V4 base | +60.18% |
| FT56 | +26.67% |
| FT 12/01/05/06 | +25.78% |

这是**另一条评测链**，不能与诚实 KPI / `ft56_julw1_end240_hold55_hardcap_20260713`（约 **+49%**）直接比高低。

---

## 2. 路径与配置清单

### 2.1 模型与训练配置

| 项 | 路径 |
|---|---|
| V4 底座 | `checkpoint/checkpoints_qqq_v4/best.pth` |
| FT56 输出（默认） | `checkpoint/checkpoints_qqq_ft56_julw1/best.pth` |
| 特征配置 | `qqq_btc/CONFIG/slow_feature_qqq_v4.json` |
| symbol map | `qqq_btc/CONFIG/symbol_map.json` |
| 策略 / 门控 | `qqq_btc/qqq/config.py` → `LIVE_REPLAY` / `EXIT_RAILS` |
| frozen norm（流式） | `qqq_btc/CONFIG/frozen_norm_qqq_daily.npz` |

### 2.2 微调数据（bak，与历史 FT56 一致）

| 项 | 路径 |
|---|---|
| bak 特征根 | `~/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00` |
| train 月 | `2026-05` + `2026-06` |
| val 月 | `2026-06` |

### 2.3 诚实 July W1（KPI 评测）

| 项 | 路径 |
|---|---|
| 诚实根 | `~/train_data/july_w1_v4_honest_openwin` |
| norm 特征 | `.../quote_features_test` |
| raw 1min（因果 gate） | `.../quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet` |
| 期权 1m | `.../options_1m` |
| 重建脚本 | `qqq_btc/tools/rebuild_july_w1_honest_openwin_features.sh` |

### 2.4 LIVE 门控关键字段（当前仓库默认）

定义见 `qqq_btc/qqq/config.py`：

| 字段 | 典型 KPI 取值 | 说明 |
|---|---|---|
| `LIVE_REPLAY` | `immediate_entry=True`, `entry_delay_bars=0` | 贴近实盘同 bar 决策 |
| `edge_q10_floor` | **KPI 覆盖为 `-0.2`**（文件默认可能是 `-0.25`） | hardcap/+49% 配方用 `-0.2` |
| `session_entry_end_bar` | `240` | 14:30 后禁新开 |
| `EXIT_RAILS.max_hold_bars` | `55` | hardcap 同款 |
| `apply_put_entry_quantile` | `False`（CALL-only 分位） | 避免 put_dyn 挡大 PUT |
| put_gate 数据 | raw1 `vix_level` **+1 分钟**再 asof | **不要**直接用 infer 里的 5min `vix_level` 当因果门控 |

流式侧对应：

| 项 | 值 |
|---|---|
| 脚本 | `qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh` |
| `QQQ_BTC_PUT_GATE_MODE` | `vixy_z`（默认因果） |
| `QQQ_BTC_USE_LIVE_REPLAY` | `1` |
| `QQQ_BTC_APPLY_PUT_ENTRY_QUANTILE` | `0`（CALL-only） |

---

## 3. 完整微调 + 诚实 KPI 脚本

一键入口（已落盘）：

```bash
# 微调 V4→FT56（5–6 月）+ 诚实 KPI replay（V4 vs FT56）
bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh

# 已有 FT56 ckpt，只跑诚实 KPI
SKIP_TRAIN=1 bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh

# 指定 ckpt
SKIP_TRAIN=1 CKPT_FT=checkpoint/checkpoints_qqq_ft56_julw1/best.pth \
  bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh
```

脚本做的事：

1. 用 bak `2026-05/06` 建 LMDB，从 V4 finetune 出 FT56  
2. 在诚实 `quote_features_test` 上 `--live-replay` 推断 V4 / FT56  
3. 给 infer 挂上 **1min 因果 put_gate**  
4. 用 `replace(LIVE_REPLAY, edge_q10_floor=-0.2)` + `EXIT_RAILS` 跑 `run_strict_replay`  
5. 写 `qqq_btc/results/ft56_julw1_honest_kpi_compare/summary.json`

### 3.1 脚本全文位置

源文件：[`qqq_btc/tools/train_ft56_julw1_honest_kpi.sh`](../tools/train_ft56_julw1_honest_kpi.sh)

核心因果 gate + LIVE replay 片段（与脚本内嵌 Python 一致）：

```python
from dataclasses import replace
import pandas as pd
from qqq_btc.qqq import config as qcfg
from qqq_btc.common.replay_harness import run_strict_replay

raw1 = pd.read_parquet(
    ".../july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet",
    columns=["timestamp", "vix_level"],
)
raw1["timestamp"] = pd.to_datetime(raw1["timestamp"], utc=True)
raw1 = raw1.sort_values("timestamp").drop_duplicates("timestamp")

def attach_1m_causal(base: pd.DataFrame) -> pd.DataFrame:
    s = raw1.copy()
    s["timestamp"] = s["timestamp"] + pd.Timedelta(minutes=1)  # 因果：下一分钟才可见
    m = pd.merge_asof(
        base[["timestamp"]].reset_index(drop=True),
        s.rename(columns={"vix_level": "put_gate"}),
        on="timestamp",
        direction="backward",
    )
    out = base.copy()
    out["put_gate"] = m["put_gate"].to_numpy()
    return out

cfg = replace(qcfg.LIVE_REPLAY, edge_q10_floor=-0.2)
df = attach_1m_causal(infer_df)
res = run_strict_replay(
    df, qcfg.FILL_MODEL, cfg, qcfg.EXIT_RAILS,
    edge_col="net_edge", edge_q10_col="net_edge_q10",
    call_edge_col="call_net_edge", put_edge_col="put_net_edge",
    put_gate_col="put_gate",
)
print(res.summary(position_frac=0.25)["total_net_return"])  # acct25
```

### 3.2 仅微调、不评测（旧 FT56 训练脚本）

若只要 ckpt、评测另跑诚实 KPI：

```bash
bash qqq_btc/tools/train_ft56_julw1.sh
# 注意：该脚本默认 FEAT_JUL=databento，末段对照不是诚实 KPI
```

---

## 4. 参考结果与口径对照

### 4.1 诚实 KPI（应对齐这条）

| 产物 | 说明 |
|---|---|
| `qqq_btc/results/ft56_julw1_end240_hold55_hardcap_20260713/` | FT56 + 诚实配方，全周约 **+49.00%** acct25（14 笔） |
| `qqq_btc/results/ft56_jul_w1_fixed5m_1mingate_q10m20/` | 更早的 fixed5m+1m gate，Jul1–8 约 **+38.3%** |

hardcap 目录内记录的关键配置：

```json
{
  "session_entry_end_bar": 240,
  "max_hold_bars": 55,
  "edge_q10_floor": -0.2
}
```

复现确认：同一 FT56 honest infer + `LIVE` + `q10=-0.2` + 1m causal gate → **acct25 = +49.00%**。

### 4.2 非诚实对照（仅诊断）

| 产物 | 说明 |
|---|---|
| `qqq_btc/results/v4_vs_ft56_vs_1201_0506_julw1/summary.json` | databento + 默认 REPLAY：V4 +60% / FT56 +27% / 12·01 混合 +26% |

差异来源一览：

| 项 | 诚实 KPI | 非诚实对照表 |
|---|---|---|
| 特征根 | `july_w1_v4_honest_openwin` | `july_w1_v4_databento` |
| put_gate | raw1 **+1m 因果** | infer 内 `vix_level` |
| 配置 | `LIVE_REPLAY` + `q10=-0.2` | 默认 `REPLAY`（常 `q10=-0.25`） |
| 典型 FT56 | ~+49%（hardcap） | ~+27% |

---

## 5. 混合月微调（12/01/05/06）注意

regime match 曾建议 `2025-12,2026-01,2026-05,2026-06`。按 **FT56 同款流程**（V4 init + bak 特征）微调后，在 **databento 对照链**上约 +25.8%，弱于裸 V4 与 FT56。

要点：

1. 必须 `init=checkpoints_qqq_v4`，不是换别的底座  
2. Jul W1 是否「更强」必须以**诚实 KPI**重评，不能只看 databento 表  
3. Jul W1 对 match 是 `NEAR_RECENT` 时，默认应维持近 2 月 FT；相似月优先做压力测试

---

## 6. 推荐工作流

```text
1. （可选）match_week_regime / weekly_finetune --suggest-train-months
2. bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh
3. 读 results/.../summary.json 的 acct25（诚实 KPI）
4. 流式：bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
5. Gate1→2→3 通过后再谈晋升
```

---

## 7. 相关文档

- `qqq_btc/docs/honest_3gate_live_parity_handoff.md` — 流式三闸门  
- `qqq_btc/docs/week_regime_match_finetune.md` — 周状态相似与 train-months 建议  
- `qqq_btc/tools/train_ft56_julw1.sh` — 历史 FT56 训练（默认 databento 评测段）  
- `qqq_btc/tools/train_ft56_julw1_honest_kpi.sh` — **本文件配套一键脚本**
