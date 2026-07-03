# QQQ 0DTE 专版：数据管线 + 双引擎 + 策略

本文档描述 **New_Pro/baseline_qqq** 从 legacy 多标的截面排序迁移到 **QQQ 单标的 + 绝对 net_edge + 0DTE/1DTE 合约** 的完整操作手册。

---

## 1. 架构总览

```text
FCS (feature_compute_service_v8.py)
  → SignalEngineV8 (run_live_signal.py)     # 推理 net_edge，发布 AlphaFrame
  → ExecutionEngineV8 (run_live_exec.py)   # OMS：StrategyCoreV0 决策 + 下单
```

| 维度 | Legacy (production) | QQQ 专版 (baseline_qqq) |
|------|---------------------|-------------------------|
| 标的 | 多标的截面 | QQQ 交易 + VIXY regime |
| Alpha | 截面 rank / z-score | **绝对 net_edge** |
| 合约 DTE | Front ~9d + Next ~37d (6 bucket) | **0/1/2 DTE front (4 bucket)** |
| 开仓门控 | OMS 需 ≥10 标的同帧 | **OMS_ENTRY_MIN_BATCH_SYMBOLS=1** |
| 持仓 | 多仓 | **MAX_POSITIONS=1** |

> `system_orchestrator_v8.py` 为 legacy 单进程编排，**不在此路径维护**。

---

## 2. 配置文件索引

| 文件 | 用途 |
|------|------|
| `New_Pro/CONFIG/anchor_qqq_0dte.json` | 0DTE 合约锚点（DTE、bucket、路径） |
| `New_Pro/CONFIG/slow_feature.json` | 慢通道特征 + `option_exec_label` + `anchor_profile` |
| `New_Pro/CONFIG/fast_feature_qqq.json` | 快通道 gate（spread / IV 动量，不训 WaveTCN） |
| `New_Pro/baseline_qqq/config.py` | 运行时：TARGET_SYMBOLS、ALPHA_ZSCORE_MODE、OMS 门控 |
| `New_Pro/baseline_qqq/strategy_config0.py` | V0 策略阈值（net_edge 量级） |

### 2.1 环境变量（实盘默认）

```bash
export OPTION_ANCHOR_PROFILE=qqq_0dte
export ANCHOR_CONFIG_PATH=New_Pro/CONFIG/anchor_qqq_0dte.json

export ALPHA_ZSCORE_MODE=absolute
export USE_NET_EDGE_ALPHA=1
export OMS_ENTRY_MIN_BATCH_SYMBOLS=1
export NET_EDGE_CLIP=0.25

export FAST_GATE_SPREAD_MAX=0.12
export VOL_Z_USE_PRICE_PROXY=0

# Exec profile (Path A / Path C 对比)
export EXEC_PROFILE=auto_hybrid          # scalp_0dte | swing_1dte | auto_hybrid
export EXEC_PROFILE_SHADOW_COMPARE=1     # 回放时并行记账 SCALP vs SWING
export EXEC_PROFILE_SHADOW_OUTPUT=~/train_data/exec_profile_shadow.json
```

---

## 3. 合约锚点：6 bucket → 4 bucket

### 3.1 问题

Legacy `step1_build_target_map.py` 选 **Front DTE≈9** + **Next DTE≈37**，与 0DTE 交易的 theta/gamma/spread 物理量级不一致，造成 train/serve skew。

### 3.2 新 bucket 定义

| bucket | 含义 | 实盘默认下单 |
|--------|------|-------------|
| 0 | 0DTE PUT ATM | 做空 |
| 1 | 0DTE PUT OTM | skew |
| 2 | 0DTE CALL ATM | **默认** (`TRADE_OPTION_MONEYNESS=ATM`) |
| 3 | 0DTE CALL OTM | 进攻型 |

bucket 4/5（次月）在 0DTE profile 下**不再订阅**。

### 3.3 共享代码

- `New_Pro/preprocess/anchor_contract_utils.py` — 离线/标签/实盘共用选约
- `New_Pro/preprocess/step1_build_target_map_0dte.py` — 生成锁定清单
- `New_Pro/baseline_qqq/anchor_config.py` — IBKR 实盘加载同一 JSON

---

## 4. 数据管线重跑顺序

改 features 或锚点后 **必须全链路重跑**，旧 checkpoint 特征维度不兼容。

```text
① step1_build_target_map_0dte.py
     输出: ~/train_data/locked_targets_map_0dte.parquet

② step2_polygon_sniper_v7.py  (或 thetadata sniper)
     输入 map 改为 locked_targets_map_0dte.parquet
     输出: ~/train_data/option_quote_sniper/QQQ/

③ option_cac_day_vectorized_day.py
     Greeks + bucket_id 透传
     输出: ~/train_data/quote_options_day_iv/

④ options_locked_feature.py
     pivot 4 bucket → VW 微观特征
     输出: ~/train_data/quote_options_bucketed_v7/

⑤ feature_merge_option_raw.py
     merge 期权特征 + 标签（0DTE bucket CALL ATM PnL）

⑥ apply_rolling_norm_standalone_res_cost.py

⑦ s0_create_slow_channel_lmdb_alpha.py

⑧ trading_tft_stock_embed.py  # 全量重训
     checkpoint: checkpoints_qqq_net_edge/
```

### 4.1 Step 1 示例

```bash
cd New_Pro/preprocess
OPTION_ANCHOR_PROFILE=qqq_0dte \
python step1_build_target_map_0dte.py \
  --start-date 2022-09-01 \
  --end-date 2026-03-01
```

### 4.2 Step 2 修改要点

在 `production/preprocess/download/step2_polygon_sniper_v7.py` 中：

```python
TARGET_MAP_FILE = "/home/kingfang007/train_data/locked_targets_map_0dte.parquet"
```

每天仅下载 **4 个**锁定合约（较 legacy 6 个更少）。

### 4.3 标签对齐（P1）

`slow_feature.json` → `option_exec_label`:

- `primary_entry_delay_seconds`: 60
- `primary_seconds`: 300

`process_labels_file` 在 `anchor_profile=qqq_0dte` 时：

- 从 bucket **2 (CALL ATM)** 加载 sniper/day_iv 分钟 mid
- `label_return_fwd` / `label_return_fwd_net` = 60s 入场 + 300s 持有的 **期权收益率**
- 不再使用 DTE 7–60 链上搜索

---

## 5. 双引擎：SignalEngineV8

**职责**：特征 → TFT 推理 → 发布 AlphaFrame（**不做策略决策**）。

### 5.1 关键行为

| 项 | 说明 |
|----|------|
| 推理输出 | `net_edge`（`USE_NET_EDGE_ALPHA=1`） |
| 归一化 | `ALPHA_ZSCORE_MODE=absolute`，**无截面 z-score** |
| VIXY | 仅 regime，`_prep_symbol_metrics` 返回 None，不进 AlphaFrame 交易项 |
| AlphaFrame | 含 `net_edge_raw`、`options_vw_spread`、`options_iv_momentum` |

### 5.2 启动

```bash
cd New_Pro/baseline_qqq
RUN_MODE=REALTIME_DRY python run_live_signal.py
```

Checkpoint 默认：`checkpoints_qqq_net_edge/advanced_alpha_best.pth`

---

## 6. 双引擎：ExecutionEngineV8 (OMS)

**职责**：消费 AlphaFrame → `StrategyCoreV0` → 下单。

### 6.1 单标的优化（相对 legacy）

| Legacy | QQQ 专版 |
|--------|----------|
| 截面排序选 Top-N | **仅 QQQ 候选**，按 `|net_edge|` 决策 |
| `max_entries=3` 硬编码 | **`MAX_POSITIONS=1`** |
| CALL/PUT 分池抢名额 | **`ENTRY_DIRECTION_SPLIT_POOL_ENABLED=False`** |
| 需 ≥10 symbol 同帧 | **`OMS_ENTRY_MIN_BATCH_SYMBOLS=1`** |

单候选时跳过 direction-split / priority-slot 重排，直接执行策略放行结果。

### 6.2 启动

```bash
cd New_Pro/baseline_qqq
RUN_MODE=REALTIME_DRY python run_live_exec.py
```

---

## 7. 策略层：StrategyCoreV0

### 7.1 信号语义变更

| 字段 | Legacy 含义 | QQQ 专版 |
|------|-------------|----------|
| `ctx['alpha_z']` | 截面 z-score | **net_edge 别名**（absolute 模式） |
| `ctx['net_edge_raw']` | 无 | TFT 原始 net_edge |
| `MIN_CS_ALPHA_Z` | 截面门槛 | **0（禁用）** |

### 7.2 阈值量级（net_edge）

| 参数 | Legacy (z) | QQQ 0DTE |
|------|------------|----------|
| `ALPHA_ENTRY_THRESHOLD` | ~0.5–1.0 | **0.015** |
| `ALPHA_ENTRY_STRICT` | — | **0.030** |
| `ALPHA_FLIP_THRESHOLD` | 0.8 | **0.012** |
| `HIGH_CONFIDENCE_THRESHOLD` | 1.2 | **0.025** |
| `SLOW_BULL_ALPHA_THRESHOLD` | z 量级 | **0.012** |

动态门槛 `_calculate_dynamic_alpha_threshold` 上限改为 `ALPHA_ENTRY_STRICT`，不再 cap 到 3.0（z-score 量级）。

### 7.3 快通道门控（非 alpha）

`strategy_core_v0._check_fast_regime_gate`:

- `options_vw_spread` ≤ `FAST_GATE_SPREAD_MAX` (0.12)
- `|options_iv_momentum|` ≤ `FAST_GATE_IV_MOMENTUM_ABS_MAX` (0.50)

### 7.4 0DTE 时段建议

| 时段 | 建议 |
|------|------|
| 09:45–15:30 | 正常入场窗口（`strategy_config0`） |
| 15:30+ | 禁新开仓 |
| 15:50 | 强平 |

训练样本建议剔除 **15:30 后**（pin / 流动性失真），与 `max_spread_pct` 对齐。

---

## 8. IBKR 实盘选约 parity

`ibkr_connector_v8._find_contracts`:

- `dte >= 0`（允许 0DTE/1DTE）
- `front_prefer_dte = 0`
- 仅订阅 `BUCKET_SPECS` 中 front 4 档（无 NEXT）

与离线 `step1_build_target_map_0dte.py` 使用同一 `anchor_qqq_0dte.json`。

---

## 9. 验证清单

- [ ] `step1` 输出 DTE 分布：`front_dte=0` 占主导（2022-09 后）
- [ ] sniper 日文件含 `bucket_id` 0–3
- [ ] `options_locked_feature` 产出 `options_vw_theta` 等列
- [ ] LMDB 标签 `label_return_fwd_net` 非全 0（需 sniper 数据）
- [ ] SignalEngine 日志 `[NetEdge-Audit] QQQ | raw=...`
- [ ] OMS 无 `min_symbols_gate` 拒绝（单 QQQ 候选）
- [ ] 策略 trace：`E10.ch_a_alpha_dyn` 阈值在 0.01–0.03 量级

---

## 10. 常见问题

**Q: 能否复用旧 9DTE checkpoint？**  
A: 不能。特征分布与标签合约均已变更，需重训。

**Q: VIXY 还在 TARGET_SYMBOLS 里？**  
A: 是。仅用于 FCS regime，不参与交易、不进 OMS 开仓候选。

**Q: 1DTE 怎么办？**  
A: `anchor_qqq_0dte.json` 中 `front_allowed_dte: [0,1,2]`，`front_prefer_dte: 0`。无 0DTE 日 fallback 到 1DTE。可另开 `1DTE` profile 单独训练。

**Q: legacy production 路径会受影响吗？**  
A: 不会。0DTE 脚本与配置均在 `New_Pro/` 下；`production/` 保持 6-bucket / 9DTE 逻辑。

---

## 11. 相关文件速查

```
New_Pro/
├── CONFIG/
│   ├── anchor_qqq_0dte.json
│   ├── slow_feature.json
│   └── fast_feature_qqq.json
├── preprocess/
│   ├── anchor_contract_utils.py
│   ├── step1_build_target_map_0dte.py
│   ├── options_locked_feature.py
│   └── feature_merge_option_raw.py
└── baseline_qqq/
    ├── config.py
    ├── strategy_config0.py
    ├── strategy_core_v0.py
    ├── signal_engine_v8.py
    ├── execution_engine_v8.py
    ├── anchor_config.py
    ├── run_live_signal.py
    └── run_live_exec.py
```
