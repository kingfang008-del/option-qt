# Mag7 开盘锁约 + 1s Quote 下载流水线

实盘因果口径：09:30 开盘锁 0/1/2 DTE（ATM/OTM），再按需补 1s bid/ask。  
统一入口：`maga7.tools.prepare_open_lock_quotes`（shell：`maga7/tools/prepare_open_lock_quotes.sh`）。

> 与前视日锁 `prepare_jan_jul_data`（`step1_build_target_map_old`）分开；本流水线只服务 `contract_mode=open_lock`。

## 前置

```bash
export MASSIVE_API_KEY=...    # 或 POLYGON_API_KEY；status/lock/seed/miss/merge 可不设
cd /path/to/option-qt
export PYTHONPATH=$PWD
```

默认 profile：

`maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json`

依赖数据（锁约阶段）：

| 输入 | 路径（profile） |
|---|---|
| 正股 1s 事实源 | `paths.stock_1s_root` → `/mnt/s990/data/raw_1s/stocks` |
| 正股派生 1m 缓存 | `paths.stock_root` → `~/train_data/spnq_train` |
| day_iv | `paths.day_iv_root` → `~/train_data/nq_options_day_iv` |
| option_1m（早盘补 0DTE） | `paths.option_1m_root` → `/mnt/s990/new_option_data_s3` |

锁约工具当前可读取派生 1m 缓存取得开盘 spot，但该缓存的上游权威仍是股票 1s
事实源；其时间戳只表示分钟归属，不单独决定策略可用时刻。

## 流水线步骤

```text
lock → seed → miss → quotes → merge → status
```

| step | 作用 |
|---|---|
| `lock` | 09:30 开盘多 DTE 锁约 → `open_locked_map` |
| `seed` | 从旧 1s 目录抽取已有合约，写入主 quote 目录（避免重复下） |
| `miss` | 对比锁约表 vs 主目录，写出 miss 锁约表 |
| `quotes` | step2 只下 miss → **旁路** `quote_1s_root_miss`（避免 seed 日文件被跳过） |
| `merge` | 旁路合并进主 `quote_1s_root` |
| `status` | 按标的打印合约覆盖率 |
| `all` | 上述全跑（可 `--skip-seed` / `--skip-download`） |

## 默认路径

| 产物 | 默认 |
|---|---|
| 开盘锁约表 | `~/train_data/locked_targets_map_maga7_open_multidte_jan_jul.parquet` |
| miss 锁约表 | `~/train_data/locked_targets_map_maga7_open_miss_1s.parquet` |
| 主 1s 目录 | `/mnt/s990/data/raw_1s/maga7_mf10_open_lock` |
| 旁路 miss 1s | `/mnt/s990/data/raw_1s/maga7_mf10_open_lock_miss` |
| 正股 1s 事实源 | `/mnt/s990/data/raw_1s/stocks` |

seed 来源（存在才用）：

- `/mnt/s990/data/raw_1s/maga7_mf10_old_lock`
- `/mnt/s990/data/raw_1s/maga7_mf10_signal_atm`
- `/mnt/s990/data/raw_1s/mag7_short_dte_old_lock`

可用 `--seed-roots a,b,c` 覆盖。

## 常用命令

```bash
# 全流程（Mag7 默认 7 标的）
python -m maga7.tools.prepare_open_lock_quotes --step all

# 或 shell
bash maga7/tools/prepare_open_lock_quotes.sh

# 只看覆盖率（不需要 API key）
python -m maga7.tools.prepare_open_lock_quotes --step status

# 已有锁约/seed，只下 miss + 合并
python -m maga7.tools.prepare_open_lock_quotes --step quotes \
  --max-workers 12 --contract-workers 4
python -m maga7.tools.prepare_open_lock_quotes --step merge

# 改日期窗
python -m maga7.tools.prepare_open_lock_quotes --step all \
  --start-date 2026-05-01 --end-date 2026-07-13
```

### 添加股票标的

```bash
# 在 profile 标的列表上追加
python -m maga7.tools.prepare_open_lock_quotes --step all --add-symbols GOOGL,GOOG

# 或完全替换列表
python -m maga7.tools.prepare_open_lock_quotes --step all \
  --symbols NVDA,TSLA,AAPL,AMZN,META,MSFT,AMD,GOOGL

# 长期固定：改 profile 的 "symbols"，再跑 --step all
```

注意：新标的需已有对应区间的 **正股 1m + day_iv（或 option_1m）**，否则 `lock` 会缺行。

### 并行参数

| 参数 | 含义 | 建议 |
|---|---|---|
| `--max-workers` | 外层 symbol-day 线程数 | 12–20 |
| `--contract-workers` | 日内合约并行 quote 流 | 4（miss 约 3 合约/日） |
| `--window-start` / `--window-end` | 下载时间窗（ET） | 默认 `10:00`–`15:00` |

瓶颈是 Polygon `list_quotes` 分页，不是 CPU；**不要改成一股一进程**，用全局 day pool + 合约并行即可。

`quotes` 日志示例：`/tmp/open_lock_miss_1s.log`（若自行 redirect）。

## 与 replay / 对拍

```bash
# offline
python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json \
  --scheme m5_circuit --tag jan_jul_open_lock_clear_otm

# stream ↔ offline
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json \
  --scheme m5_circuit --tag parity_open_lock_clear_otm_jan_jul
```

Profile 要点：

- `trade.contract_mode = open_lock`
- `trade.clear_otm_ban_0dte_pct = 0.01`（信号时 0DTE 明显 OTM → 改用 1DTE+）
- `paths.quote_1s_root` / `paths.open_locked_map` 与本流水线一致

仅锁约、不下载：

```bash
python -m maga7.tools.export_open_lock_map
# 或
python -m maga7.tools.prepare_open_lock_quotes --step lock --skip-download
```

## 开盘阶梯（ATM+2×OTM）

`contract_mode=open_ladder`：开盘锁 ATM/OTM1/OTM2，信号时选离现价最近。

```bash
python -m maga7.tools.export_open_lock_map \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm2otm_v1.json --otm-rungs 2

python -m maga7.tools.run_open_ladder_ab --quote-source day_iv

# 1s（OTM2 需补数）
python -m maga7.tools.prepare_open_lock_quotes \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm2otm_v1.json --step all
```

详见 `open_lock_quote_pipeline.md` 与 `jan_jul_replay_versions.md`。

| 模块 | 路径 |
|---|---|
| 流水线 CLI | `maga7/tools/prepare_open_lock_quotes.py` |
| shell 包装 | `maga7/tools/prepare_open_lock_quotes.sh` |
| 开盘锁逻辑 | `maga7/common/open_lock.py` |
| 选约（offline/stream/live） | `maga7/common/entry_contract.py` |
| step2 1s sniper | `preprocess/download/step2_polygon_second_sniper_v1.py` |
| 前视日锁数据准备（对照） | `maga7/tools/prepare_jan_jul_data.py` |
