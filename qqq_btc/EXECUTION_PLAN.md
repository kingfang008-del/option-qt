# qqq_btc 执行步骤 —— 从模型训练到双引擎上线

> **路线图与集群里程碑**见 [CLUSTER_ROADMAP.md](./CLUSTER_ROADMAP.md)(Phase 0–5、G0–G5、L1/L2/L3 分工)。

顺序不可调换:每个阶段末尾有**验收门**,不通过不进入下一阶段。
状态:✅ 已完成 / 🔧 待数据验证 / ⏳ 依赖外部资源(数据/训练机)。

---

## 阶段 0:训练数据准备(P0,本机 + 数据盘)

> 目标:产出带主标签(及可选双腿标签)的 LMDB,通过 strict 质量验收。
>
> **完整命令、顺序与排错见 [DATA_PIPELINE.md](./DATA_PIPELINE.md)**（以该文档为准）。

| # | 步骤 | 命令/工具 | 状态 |
|---|---|---|---|
| 0.1 | 选约锁定 | `step1_build_target_map.py` (qqq_0dte) | ✅ |
| 0.2 | 报价下载 | `step2_databento_second_sniper_v1.py` → `step3_databento_aggregate_1s_to_1m.py` | ✅ |
| 0.3 | IV / 特征 merge | `option_cac_day_vectorized_day` → `feature_merge_option_raw` | ✅ |
| 0.4 | **主标签** | `write_main_labels.py` / `process_labels_file`（必需） | ✅ |
| 0.4b | 双腿标签 | `label_pipeline.py`（可选） | 可选 |
| 0.5 | **先** split | `split_raw_features.py` → train/val/test | ✅ |
| 0.6 | **再** rolling_norm | `FEATURE_CONFIG=.../slow_feature_qqq_v2.json` + `apply_rolling_norm_standalone.py` | ✅ |
| 0.7 | **LMDB** | `build_lmdb.py`（勿用旧 `s0_create_slow_channel_lmdb_alpha.py`） | ✅ |
| 0.8 | 数据集验收 | `LMDBAlphaDataset.sanity_check()` | ✅ |

```bash
# 主标签 → 切分 → 归一化 → LMDB（顺序不可调换 split/norm）
python qqq_btc/tools/write_main_labels.py \
  --feature-roots ~/train_data/quote_features_raw \
  --config qqq_btc/CONFIG/slow_feature_qqq_v2.json

python preprocess/ask_bid/split_raw_features.py

export FEATURE_CONFIG="/home/kingfang007/文档/GitHub/option-qt/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
python preprocess/ask_bid/apply_rolling_norm_standalone.py

python qqq_btc/tools/build_lmdb.py \
  --feature-root ~/train_data/quote_features_train \
  --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
  --symbol-map qqq_btc/CONFIG/symbol_map.json \
  --output ~/train_data/lmdb/train_qqq.lmdb \
  --symbols QQQ
```

**验收门 G0**: 主标签列齐全且 net_std>0; train/val/test 按月切分; sanity_check strict 通过。

---

## 阶段 1:两阶段训练(P1,训练机)

```bash
python qqq_btc/tools/make_feature_config.py

python -m qqq_btc.model.train --mode pretrain \
  --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
  --data-root ~/train_data/lmdb \
  --train-lmdb train_qqq_spy.lmdb --val-lmdbs val_qqq.lmdb \
  --checkpoint-dir checkpoints_index_etf_v2

python -m qqq_btc.model.train --mode finetune \
  --init-checkpoint checkpoints_index_etf_v2/best.pth \
  --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
  --data-root ~/train_data/lmdb \
  --train-lmdb train_qqq_only.lmdb --val-lmdbs val_qqq.lmdb \
  --checkpoint-dir checkpoints_qqq_net_edge_v2
```

**验收门 G1**: val IC>0; q10_coverage≈0.10; call/put/straddle 头各自有效。

---

## 阶段 2:strict replay 验收(P1,本机)

```bash
# 推理
python qqq_btc/tools/run_inference.py \
  --checkpoint checkpoints_qqq_net_edge_v2/best.pth \
  --input ~/train_data/quote_features_qqq_v2_norm/QQQ/.../1min/QQQ_2026-01.parquet \
  --output /tmp/qqq_infer.parquet

# 回放
python qqq_btc/tools/run_replay.py --input /tmp/qqq_infer.parquet --output /tmp/replay.json

# L2 事件回放(分钟信号 + 1s tick;非向量化)
python qqq_btc/tools/run_event_replay.py \
  --input /tmp/qqq_infer.parquet \
  --ticks ~/train_data/sniper_1s/QQQ_2026-01.parquet \
  --fill-timing first_tick \
  --output /tmp/replay_event.json \
  --trades /tmp/replay_trades.parquet

# L1 vs L2 对拍
python qqq_btc/tools/run_event_replay.py \
  --input /tmp/qqq_infer.parquet --ticks ~/train_data/sniper_1s/QQQ_2026-01.parquet --compare

# 双腿/跨式对照
python qqq_btc/tools/run_replay.py --input /tmp/qqq_infer.parquet --dual-leg --output /tmp/replay_dual.json

# rails 标定
python qqq_btc/tools/calibrate_rails.py --parquet /tmp/qqq_infer.parquet
```

**验收门 G2**: fill 口径 PnL>0; 去 best-2-day 仍为正; 增量腿才开启。

---

## 阶段 3:双引擎接线(P1 末)

| # | 组件 | 路径 | 状态 |
|---|---|---|---|
| 3.1 | Live 信号引擎 | `qqq_btc/live/signal_engine.py` | ✅ |
| 3.2 | FCS 特征补算 | `qqq_btc/live/fcs_adapter.py` | ✅ |
| 3.3 | OMS fill 适配 | `qqq_btc/live/oms_adapter.py` | ✅ |
| 3.4 | 入场决策(与 replay 同实现) | `qqq_btc/common/entry_decision.py` | ✅ |
| 3.5 | Parity 审计 | `qqq_btc/tools/parity_audit.py` | ✅ |
| 3.6 | 嵌入旧 execution_engine | `run_live_exec_qqq.py` + `oms_integration` patch | ✅ |
| 3.6b | 分钟 exit 对齐 | `strategy_exit_bridge` → `exit_rails` | ✅ |
| 3.7 | fill 审计 shadow CSV | `fill_audit_writer` patch accounting | ✅ |
| 3.8 | exit 分布对拍 | `parity_audit exits` + `export_replay_trades` | ✅ 工具 |
| 3.9 | 影子模式 ≥2 周 | 跑 REALTIME_DRY 积累 CSV 后 G3 | ⏳ |

### 双引擎启动(qqq_btc 路径,不修改 New_Pro 源码)

```bash
# 终端 1 — FCS(沿用原启动)
# 终端 2 — Signal(v2 checkpoint + Redis ALPHA_FRAME 外壳)
cd New_Pro/baseline_qqq
QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_signal_qqq.py \\
  --checkpoint ~/quant_project/checkpoints_qqq_net_edge_v2/best.pth

# 终端 3 — OMS(fill 0.775 + disaster-only tick exit)
QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_exec_qqq.py
# 恢复 legacy 秒级阶梯: QQQ_BTC_TICK_EXITS=legacy
```

### execution_engine 接线(实现方式)

通过 `qqq_btc.live.oms_integration.apply_oms_patches()` monkey-patch,无需改 `orchestrator_execution.py`:

```python
from qqq_btc.live.bootstrap import bootstrap_qqq_btc_live
bootstrap_qqq_btc_live()  # entry/exit limit → oms_adapter; tick → disaster_only
```

### Live 嵌入示例

```python
from qqq_btc.live.signal_engine import LiveSignalEngine

engine = LiveSignalEngine(checkpoint="checkpoints_qqq_net_edge_v2/best.pth")
action = engine.on_bar_close(fcs_history_df, quotes={
    "exec_call_bid": ..., "exec_call_ask": ..., "exec_call_spread_pct": ...,
    "exec_put_bid": ..., "exec_put_ask": ..., "exec_put_spread_pct": ...,
})
```

### Parity 影子对账

```bash
# fill 点差位 median ≈ 0.775
python qqq_btc/tools/parity_audit.py fill \
  --audit-log ~/quant_project/shadow/fill_audit.csv

# strict replay 基准
python qqq_btc/tools/export_replay_trades.py \
  --parquet /tmp/qqq_infer.parquet \
  --output /tmp/replay_trades.csv

# exit_reason 分布 L1 对拍
python qqq_btc/tools/parity_audit.py exits \
  --audit-log ~/quant_project/shadow/fill_audit.csv \
  --replay-trades /tmp/replay_trades.csv
```

**验收门 G3**: feature pass_rate>0.95; fill median≈0.775; exit distribution L1≤0.35。

> **逐日核对表** → [PARITY_CHECKLIST.md](./PARITY_CHECKLIST.md)(含每日/每周打勾项与挂掉排查)

---

## 文件索引(新增)

| 文件 | 作用 |
|---|---|
| `tools/label_pipeline.py` | 双腿 quote merge + time/trend + fill 标签 |
| `tools/build_lmdb.py` | 无 PG 的 LMDB 建库 |
| `tools/run_inference.py` | checkpoint → edge 列 parquet |
| `tools/run_replay.py` | edge parquet → strict replay summary |
| `tools/parity_audit.py` | 特征/fill 影子对账 |
| `common/entry_decision.py` | replay/live 共用入场决策 |
| `live/signal_engine.py` | 实盘 bar-close 信号引擎 |
| `live/oms_adapter.py` | OMS 限价 + fill 反推 |
| `live/fill_audit_writer.py` | 成交 → ~/quant_project/shadow/fill_audit.csv |
| `live/fcs_adapter.py` | FCS bar 特征补算 |
| `CONFIG/symbol_map.json` | stock_id 映射(无 Postgres) |
