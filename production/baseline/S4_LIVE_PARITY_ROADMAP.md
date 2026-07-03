# S4 回放 ↔ 实盘 严格对齐路线图

目标：在**同一策略配置**与**同一套点差等执行约束**（均以 `strategy_config0.StrategyConfig` 为准，不在本路线图中单独改点差）下，使实盘决策与 `s4_run_historical_replay_s2_1s.py` 路径产生的 `StrategyCore` 输入语义一致，差异仅来自不可消除项（真实撮合、断线、延迟），且这些项可被单独计量。

---

## 阶段 0：范围与成功标准（1–2 天）

| 项目 | 内容 |
|------|------|
| 参考链路 | `production/preprocess/backtest/second/s4_run_historical_replay_s2_1s.py`：分钟窗 `ExecutionWindow` + `SignalEngineV8.process_batch` + `ExecutionEngineV8.execute_window` |
| 对标链路 | 实盘：`system_orchestrator_v8` / `signal_engine_v8` → `ExecutionEngineV8._build_strategy_ctx` → `StrategyCoreV0.decide_entry` / `check_exit` |
| 成功标准 | （1）`strategy_ctx_contract` 校验在回放与单测中全通过；（2）同一 `item + opt_data + frame` 快照，回放 OMS 与实盘 OMS 构建的 `ctx` 在约定字段上数值一致（容差写入测试）；（3）点差规则与配置在 S4 使用的 `StrategyConfig` 与实盘 `STRATEGY_CORE_VERSION=V0` 时使用同一份模块默认值或可证明同源 |

**显式非目标（记录在案，避免误判「不一致」）**：真实 IB 部分成交、行情源切换、与 MockIBKR 的撮合差异；这些归入「执行层差异报表」，不与「策略 ctx 语义」混谈。

---

## 阶段 1：数据与时钟契约（3–5 天）

**状态（本轮已收口语义 / 工具，实盘张量仍须训练侧保证同一公式）：**

1. **Alpha 可见延迟**  
   - 已文档化：左对齐分钟标签 + 防 lookahead；S4 为 ``ts+60`` 再 merge；FCS 为 ``alpha_label_ts = current_minute_ts - 60`` 详见 ``replay_live_parity_utils.py``；S4 脚本顶部注释已指向该模块。

2. **指数 ROC**  
   - S4 定义已固化为工具函数；并有单测与 pandas 对齐。  
   - **实盘**：FCS 在满足 301 个 1s 样本后，若张量与 S4 缓冲 ROC 不一致则**自动用缓冲覆盖**并打 ``[FCS-IndexROC-S4]`` 日志；跨日 premarket flush 时清空缓冲。

3. **`spread_divergence` 与空仓**  
   - 已选 **A**：空仓散度恒 0；入场以绝对点差与 bid/ask 为主，与回放一致；``execution_engine_v8._build_strategy_ctx`` 内注释已写明。

4. **`resolve_db_path` 可移植**  
   - 已完成：``HISTORY_SQLITE_1S_DIR`` + 本地目录回退。

---

## 阶段 2：OMS `ctx` 单一真相源（5–10 天）

1. **模块**  
   - `strategy_ctx_contract.py`：canonical key 集合、`validate_strategy_ctx_for_v0()`、字段说明与 S4 `item` / `frame` 来源注释。

2. **重构方向（渐进）**  
   - 将 `_build_strategy_ctx` 中「纯函数」部分抽到 `build_strategy_ctx_dict(...)`，回放与实盘只传不同 `opt_data` / `frame` 来源，**禁止**在 orchestrator 再拼半套 `ctx` 再合并。

3. **Golden 快照测试**  
   - 从真实 replay 中间导出一条 `(item, opt_data, frame, state)` → 单测断言 `ctx` 与归档 JSON 一致（容忍 float）。

---

## 阶段 3：执行与风控外环对齐（并行）

1. **点差**：**不单独修改默认值**；S4 与实盘共用同一份 `StrategyConfig`（如 `MAX_SPREAD_PCT_ENTRY_*`），对齐即保证两侧读同一配置源、同一计算口径（相对 bid–ask 等）。

2. **分钟 vs 秒**：确认 `EXIT_SIGNAL_MINUTE_ONLY`、建仓保护期、stale quote guard 在「对齐实验」中有开关矩阵（回放可复制实盘开关跑对照）。

3. **`ExecutionWindow`**：实盘若尚未走与 S4 相同的 `execute_window` 编排，列出差异清单，逐项「要么统一、要么文档化为已知偏差」。

---

## 阶段 4：持续保障

- PR 改 `_build_strategy_ctx` 或 `strategy_core_v0` 的 `ctx` 读字段 → 必须更新 `strategy_ctx_contract` + `test_v0_ctx_contract.py`。  
- 周期性：同一日 S4 全量 replay vs 实盘 shadow（只记录 signal，不下单）对比 gate trace 直方图。

---

## 当前已执行（本提交/近期）

- `strategy_ctx_contract.py`：V0 契约校验入口。  
- `test_v0_ctx_contract.py`：接入校验。  
- `replay_live_parity_utils.py`：阶段 1 语义说明 + 指数 5m ROC（300×1s）工具函数。  
- `test_index_roc_s4_parity.py`：指数 ROC 与 pandas/S4 定义对齐单测。  
- `s4_run_historical_replay_s2_1s.py`：`HISTORY_SQLITE_1S_DIR` + 本地目录回退；`ALPHA_AVAILABLE_DELAY_SECONDS` 注释链到 parity utils。  
- `feature_compute_service_v8.py`：SPY/QQQ 1s 缓冲 + 张量不一致时 S4 ROC 回算 + 日志；premarket flush 清缓冲。  
- `fcs_realtime_pipeline.py`：每条 tick 写入 1s 缓冲；flush 时清空。  
- 本文档：总路线图。  

点差阈值沿用 `strategy_config0.py` 现有默认，与实盘保持一致，未在本路线图单独收紧或放宽。

后续可按阶段 1→2 拆解为 issue / 小 PR 逐步合并。
