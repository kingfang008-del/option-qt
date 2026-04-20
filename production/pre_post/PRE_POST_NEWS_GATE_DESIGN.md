# Pre/Post 新闻门控设计（隔离版）

本目录实现一个与主交易链路隔离的 `pre_post` 模块，用于在盘前/盘后构建“新闻驱动 + 流动性约束”的动态标的池。

目标：
- 不改动 `baseline` 双引擎/FCS 现有逻辑；
- 可复用现有 Redis 与配置；
- 支持渐进式接入（先 shadow，再生产）。

---

## 1. 目录与职责

- `news_universe_gate.py`
  - 从 Redis 新闻源读取候选；
  - 可选从 PostgreSQL `stocks_us` 按行业加载扫描池（默认 Biotechnology）；
  - 在新闻候选不足时，用扫描池轮转补位（适配 300 订阅额度）；
  - 按规则筛选可交易标的；
  - 发布 `active_symbols` 到 Redis。

---

## 2. Redis 输入/输出协议

### 输入（新闻侧写入）

- `ZSET prepost:news_scores`
  - `member`: `SYMBOL`（如 `NVDA`）
  - `score`: 新闻分（建议 0~1）

- `HASH prepost:news_meta:<SYMBOL>`
  - `last_ts`: 最新新闻时间戳（秒）
  - `news_score`: 可选，覆盖或补充 zset score
  - `dollar_volume`: 盘前/盘后美元成交额
  - `spread_pct`: 当前点差百分比（0.02=2%）
  - `quote_stability`: 报价稳定度（0~1）
  - `headline/source/event_type`: 文本审计字段

### 输出（本模块写入）

- `STRING prepost:active_symbols`
  - JSON 数组，例如 `["NVDA","TSLA","AAPL"]`

- `HASH prepost:active_symbols_meta`
  - `generated_ts`
  - `selected_count`
  - `reason` (`disabled` / `rth` / `prepost_news_gate`)
  - `selected_extra`
  - `cap`
  - `pg_universe_size`
  - `scan_filled`

---

## 3. 规则流程

1. 判断是否处于 pre/post 时段（美东时间）：
   - 盘前 `04:00-09:29`
   - 盘后 `16:00-19:59`
2. 读取新闻候选（zset 降序）。
3. 过滤：
   - 新闻时效（`lookback_minutes`）
   - 最低新闻分（`min_news_score`）
   - 最低美元成交额（`min_dollar_volume`）
   - 最大点差（`max_spread_pct`）
   - 最低报价稳定度（`min_quote_stability`）
4. 与基础池（`base_symbols`）合并，限制 `max_symbols`。
5. 当新闻候选不足时，从 PG 行业扫描池按 round-robin 补位。
6. 应用最小驻留时间（`min_hold_seconds`）防抖。
7. 发布到 Redis。

---

## 4. 环境变量

核心参数：
- `PREPOST_NEWS_GATE_ENABLED`（默认 `1`）
- `PREPOST_MAX_SYMBOLS`（默认 `300`）
- `PREPOST_REFRESH_SECONDS`（默认 `30`）
- `PREPOST_MIN_HOLD_SECONDS`（默认 `180`）
- `PREPOST_NEWS_LOOKBACK_MINUTES`（默认 `120`）
- `PREPOST_MIN_NEWS_SCORE`（默认 `0.55`）
- `PREPOST_MIN_DOLLAR_VOLUME`（默认 `1500000`）
- `PREPOST_MAX_SPREAD_PCT`（默认 `0.025`）
- `PREPOST_MIN_QUOTE_STABILITY`（默认 `0.50`）
- `PREPOST_SCAN_FILL_ENABLED`（默认 `1`，新闻不足时启用扫描池补位）

PG 扫描池参数：
- `PREPOST_PG_UNIVERSE_ENABLED`（默认 `1`）
- `PREPOST_PG_DB_URL`（默认优先读 `PG_DB_URL`）
- `PREPOST_PG_TABLE`（默认 `stocks_us`）
- `PREPOST_PG_SYMBOL_COL`（默认 `symbol`）
- `PREPOST_PG_INDUSTRY_COL`（默认 `industry`）
- `PREPOST_PG_INDUSTRY_VALUE`（默认 `Biotechnology`）
- `PREPOST_PG_REFRESH_SECONDS`（默认 `900`）

Redis key 参数：
- `PREPOST_NEWS_SCORE_ZSET_KEY`（默认 `prepost:news_scores`）
- `PREPOST_NEWS_META_KEY_PREFIX`（默认 `prepost:news_meta:`）
- `PREPOST_ACTIVE_SYMBOLS_KEY`（默认 `prepost:active_symbols`）
- `PREPOST_ACTIVE_META_KEY`（默认 `prepost:active_symbols_meta`）

Base symbols：
- `PREPOST_BASE_SYMBOLS=NVDA,TSLA,...`
  - 未配置时，会尝试复用 `baseline/config.py` 的 `TARGET_SYMBOLS`。

---

## 5. 运行方式

```bash
cd production/pre_post
python news_universe_gate.py
```

示例（生物科技 300 订阅池）：

```bash
export PREPOST_MAX_SYMBOLS=300
export PREPOST_PG_UNIVERSE_ENABLED=1
export PREPOST_PG_TABLE=stocks_us
export PREPOST_PG_INDUSTRY_COL=industry
export PREPOST_PG_INDUSTRY_VALUE=Biotechnology
python news_universe_gate.py
```

---

## 6. 升级路线

1. **Shadow 阶段**：仅写 Redis，不接订阅。
2. **Connector 接入**：`ibkr_connector_v8.py` 增加可选读取 `prepost:active_symbols`，动态增减订阅。
3. **回放验证**：复用历史新闻事件样本做离线回放，对比“固定池 vs 动态池”。
4. **风险扩展**：加入事件类型权重（FDA/M&A/财报）与 blacklist（低流动性陷阱标的）。

