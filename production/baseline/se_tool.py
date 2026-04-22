#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Legacy SignalEngine strategy/trading helpers archived during AlphaEngine split.

SignalEngineV8 is now an AlphaEngine and no longer owns StrategyCore or emits
BUY/SELL. These methods are kept here temporarily for debugging/reference. If
new OMS-side AlphaFrame flow stays stable, this module can be deleted.
"""


class LegacySignalEngineStrategyTools:
    """Archived mixin; not imported by production SignalEngineV8."""

    # ---- archived from signal_engine_v8.py: _process_fast_fused_tick ----
    async def _process_fast_fused_tick(self, payload: dict):
        """[高频通道] 处理来自 fused_market_stream 的混合 Tick，执行毫秒级平仓保护"""
        # AlphaEngine boundary: fast ticks must not call StrategyCore or emit
        # SELL. OMS is the single strategy owner.
        return
        # 策略开关: 分钟级信号模式下，禁止秒级链路直接触发平仓。
        if bool(getattr(self.cfg, 'EXIT_SIGNAL_MINUTE_ONLY', False)):
            return

        # 🚀 [核心对齐修复: 屏蔽高频插针平仓]
        # 如果处于回测/回放模式并且开启了降频 (converge_to_single)，则直接屏蔽此流入口！
        converge_to_single = os.environ.get('DUAL_CONVERGE_TO_SINGLE') == '1'
        is_backtest_env = (self.mode == 'backtest' or os.environ.get('RUN_MODE') in ['BACKTEST', 'LIVEREPLAY'])
        if converge_to_single and is_backtest_env:
            return

        from config import HASH_OPTION_SNAPSHOT, TARGET_SYMBOLS
        is_live_replay = IS_LIVEREPLAY

        sym = payload.get('symbol')
        if not sym: return

        # [🔥 核心增强] 在 LIVEREPLAY 模式下，将回放的期权数据“镜像”到 Redis Hash 中
        # 彻底解决 process_batch 对 live_option_snapshot 的依赖（使其不再处于断流状态）
        if IS_LIVEREPLAY and sym in TARGET_SYMBOLS:
            opt_buckets = payload.get('option_buckets')
            opt_contracts = payload.get('opt_contracts')
            if opt_buckets and opt_contracts:
                try:
                    snap_payload = {
                        'real_history_len': elapsed_minutes,
                        'symbol': sym,
                        'ts': payload.get('ts'),
                        'buckets': opt_buckets,
                        'contracts': opt_contracts
                    }
                    self.r.hset(HASH_OPTION_SNAPSHOT, sym, ser.pack(snap_payload))
                except Exception as e:
                    logger.error(f"❌ Failed to mirror option snapshot for {sym}: {e}")

        if sym not in self.states: return

        st = self.states[sym]
        if st.position == 0 or st.is_pending: return

       # 👇 [🔥 终极修复 1：拔除 time.time() 毒药，强制使用逻辑时钟]
        curr_ts = payload.get('ts')
        if not curr_ts:
            if os.environ.get('RUN_MODE') == 'LIVEREPLAY' and hasattr(self, 'last_curr_ts'):
                curr_ts = self.last_curr_ts
            else:
                curr_ts = time.time()
        # 👆

        # 👇 [🔥 终极修复 2：防止 60.0 秒整点浮点数死锁，改为 59.0]
        if curr_ts - st.entry_ts < 59.0:
            return
        # 👆

        # =================================================================
        # ⏱️ [终极修复 2] 实盘风控节流阀 (Throttle Lock)
        # =================================================================
        last_fast_check = getattr(st, 'last_fast_check', 0)
        if curr_ts - last_fast_check < 15.0: # 高频扫描间隔设为 15 秒
            return



        st.last_fast_check = curr_ts

        # =================================================================
        # 🔍 [终极修复 3] 取消原先错误的注释，恢复真正的数据解析逻辑
        # =================================================================
        stock_data = payload.get('stock', {})
        stock_price = float(stock_data.get('close', 0.0))
        if stock_price <= 0: return

        opt_buckets = payload.get('option_buckets', [])
        if not opt_buckets or len(opt_buckets) < 2: return

        from config import TAG_TO_INDEX
        idx_c = TAG_TO_INDEX.get('CALL_ATM', 2)
        idx_p = TAG_TO_INDEX.get('PUT_ATM', 0)

        # 根据当前持仓方向选择对应的期权 Bucket
        idx = idx_c if st.position == 1 else idx_p

        try:
            opt_data = opt_buckets[idx]
            base_price = float(opt_data[0])
            bid = float(opt_data[8]) if len(opt_data) > 8 else 0.0
            ask = float(opt_data[9]) if len(opt_data) > 9 else 0.0

            # 使用系统统一的公允价计算逻辑
            market_opt_price = self._get_fair_market_price(base_price, bid, ask)

        except Exception as e:
            # 捕获异常而不是静默吞掉，方便未来排错
            logger.debug(f"⚠️ Fast tick parse error for {sym}: {e}")
            return

        # 极度干涸或错误盘口，交给 1min 主循环的断流估值法去处理，高频这里不接锅
        if market_opt_price <= 0.01:
            logger.debug(f"⚠️ [Fast Tick 忽略] {sym} 高频市价过低 ({market_opt_price})，跳过本次高频检查。")
            return

        # [🎯 靶向日志 A1] 确认高频流已捕捉到持仓，并准备送审
        st_held_mins = (curr_ts - st.entry_ts) / 60.0
        st_current_roi = (market_opt_price - st.entry_price) / st.entry_price if st.entry_price > 0 else 0
        logger.debug(f"⚡ [Fast Tick 嗅探] {sym} | 现价: {market_opt_price:.2f} | 成本: {st.entry_price:.2f} | 当前 ROI: {st_current_roi*100:.2f}% | 已持仓: {st_held_mins:.1f}分钟")

        from datetime import datetime
        from pytz import timezone
        ny_now = datetime.fromtimestamp(curr_ts, tz=timezone('America/New_York'))

        # 组装 check_exit 所需的上下文 (补齐了 V15 策略需要的全部字段)
        ctx = {
            'symbol': sym, 'time': ny_now, 'curr_ts': curr_ts, 'price': stock_price,
            'alpha_z': getattr(st, 'last_alpha_z', 0.0),
            'stock_roc': (stock_price - st.entry_stock) / st.entry_stock if st.entry_stock > 0 else 0.0,
            'macd_hist': getattr(st, 'last_macd_hist', 0.0),
            'macd_hist_slope': getattr(st, 'last_macd_hist_slope', 0.0),
            'spy_roc': getattr(st, 'last_spy_roc', 0.0),
            'qqq_roc': getattr(st, 'last_qqq_roc', 0.0),
            'position': st.position,
            'cooldown_until': st.cooldown_until,
            'is_ready': st.warmup_complete,
            'is_banned': curr_ts < self.global_cooldown_until,
            'held_mins': self._calc_trading_minutes(st.entry_ts, curr_ts),
            'stock_iv': st.last_valid_iv,
            'holding': {
                'entry_price': st.entry_price, 'entry_stock': st.entry_stock,
                'entry_ts': st.entry_ts, 'dir': st.position,
                'max_roi': st.max_roi, 'entry_spy_roc': getattr(st, 'entry_spy_roc', 0.0),
                'entry_index_trend': getattr(st, 'entry_index_trend', 0)
            },
            'curr_price': market_opt_price, 'curr_stock': stock_price,
            'bid': bid,
            'ask': ask,
            'snap_roc': getattr(st, 'last_snap_roc', 0.0),
            'global_regime_reversal_cnt': getattr(self, 'last_global_regime_reversal_cnt', 0),
            'regime_reversal_count': getattr(self, 'last_global_regime_reversal_cnt', 0),
            'is_volatile_regime': getattr(self, 'last_global_is_volatile_regime', False),
        }

        # 高频更新 max_roi (让 Protect 和 Trailing Stop 的追踪更加灵敏)
        if st.entry_price > 0:
            current_roi = (market_opt_price - st.entry_price) / st.entry_price
            if current_roi > st.max_roi:
                st.max_roi = current_roi
            ctx['holding']['max_roi'] = st.max_roi

        # 调用核心策略进行裁决
        exit_sig = self.strategy.check_exit(ctx)
        # [Gate Trace] 高频 fast-tick 路径同样 publish, 让 Dashboard 看到秒级 exit 决策
        self._publish_gate_trace(sym, 'exit', exit_sig, event_ts=curr_ts)

        if exit_sig:
            exit_sig['price'] = market_opt_price
            exit_sig['market_price'] = market_opt_price
            exit_sig['bid'] = bid
            exit_sig['ask'] = ask

            logger.info(f"⚡ [Fast Tick 触发] {sym} 捕捉到高频平仓信号: {exit_sig['reason']}")

            # 执行离场
            await self._execute_exit(sym, exit_sig, stock_price, curr_ts, -1)


    # ---- archived from signal_engine_v8.py: _evaluate_symbol_signals ----
    async def _evaluate_symbol_signals(self, i, sym, metrics, opt_data, ny_now, curr_ts, spy_roc, qqq_roc, is_zombie_market, index_trend=0, global_regime_reversal_cnt=0, global_is_volatile_regime=None, global_regime_band='calm', global_regime_score=0.0, should_update_full=True):
        """[Refactor] 核心策略评价 logic (平仓与开仓信号收集) - 修复空仓不交易BUG"""
        from config import TAG_TO_INDEX
        st = metrics['st']
        if global_is_volatile_regime is None:
            global_is_volatile_regime = getattr(self, 'last_global_is_volatile_regime', False)

        # 🚀 [终极期权对位]
        # 如果当前是分钟边界 (:00 秒触发的决策)，为了对齐 1m 基准表，必须使用上一秒缓存的期权快照 (:59)
        real_opt_data = opt_data
        if int(curr_ts) % 60 == 0 and st.last_tick_opt_data is not None:
            real_opt_data = st.last_tick_opt_data

        price = metrics['price']
        final_alpha = metrics['final_alpha']
        alpha_log_iv = st.last_valid_iv

        # Alpha 日志固定记录“当前 batch 自带的分钟快照 IV”，
        # 不跟随 real_opt_data 的 :59 回退逻辑，避免表里看起来慢一拍。
        if opt_data['has_feed']:
            if st.position == 1:
                log_iv_candidate = opt_data.get('call_iv', 0.0)
            elif st.position == -1:
                log_iv_candidate = opt_data.get('put_iv', 0.0)
            else:
                log_call_iv = opt_data.get('call_iv', 0.0)
                log_put_iv = opt_data.get('put_iv', 0.0)
                if log_call_iv > 0.01 and log_put_iv > 0.01:
                    log_iv_candidate = (log_call_iv + log_put_iv) / 2.0
                elif log_call_iv > 0.01:
                    log_iv_candidate = log_call_iv
                else:
                    log_iv_candidate = log_put_iv

            if log_iv_candidate > 0.01:
                alpha_log_iv = log_iv_candidate

         # 1. 更新 IV 状态：消除均值污染，采用流动性择优法则
        if real_opt_data['has_feed']:
            if st.position == 1:
                curr_iv = real_opt_data['call_iv']
            elif st.position == -1:
                curr_iv = real_opt_data['put_iv']
            else:
                # 🚀 [对齐修复] 强制使用均值 IV，确保与 1m 基准 100% 对位
                if real_opt_data.get('call_iv', 0) > 0 and real_opt_data.get('put_iv', 0) > 0:
                    curr_iv = (real_opt_data['call_iv'] + real_opt_data['put_iv']) / 2.0
                elif real_opt_data.get('call_iv', 0) > 0:
                    curr_iv = real_opt_data['call_iv']
                else:
                    curr_iv = real_opt_data['put_iv']

            if curr_iv > 0.01:
                st.last_valid_iv = curr_iv

        # 🚀 [终极修复]：动态方向推断 (Dynamic Direction Inference)
        # 如果空仓，我们利用 Alpha 的方向预判策略想看哪个盘口，避免传入 0.0 导致策略的风险校验拒单！
        eval_dir = st.position if st.position != 0 else (1 if final_alpha > 0 else -1)

        ctx_bid = real_opt_data['call_bid'] if eval_dir == 1 else real_opt_data['put_bid']
        ctx_ask = real_opt_data['call_ask'] if eval_dir == 1 else real_opt_data['put_ask']
        market_opt_price = real_opt_data['call_price'] if eval_dir == 1 else real_opt_data['put_price']

        # 计算 Context 中的公允价
        ctx_curr_price = 0.0
        if real_opt_data['has_feed']:
            # 🚀 [诊断日志] 打印 Signal Engine 接收到的期权数据细节
            if sym == 'NVDA' and getattr(self, '_iv_se_loud_count', 0) < 3:
                logger.info(f"📢 [SE_LOUD_TRACE] {sym} | IV_from_OptData: {opt_data.get('call_iv', -1):.4f} | Has_Feed: {opt_data['has_feed']}")
                self._iv_se_loud_count = getattr(self, '_iv_se_loud_count', 0) + 1

            ctx_curr_price = self._get_fair_market_price(market_opt_price, ctx_bid, ctx_ask, getattr(st, 'last_opt_price', 0.0))
        elif st.position != 0:
            ctx_curr_price = max(st.entry_price, 0.01)

        # 2. 构建 Context
        ctx = {
            'symbol': sym, 'time': ny_now, 'curr_ts': curr_ts, 'price': price,
            'alpha': final_alpha, # 🚀 [Critical Fix] 补充策略接口必需的 alpha 键
            'alpha_z': final_alpha, 'cs_alpha_z': metrics.get('cs_alpha_z', final_alpha), # [🆕 新增]
            'vol_z': metrics['vol_z'], 'stock_roc': metrics['roc_5m'],
            'event_prob': self.cached_event_probs.get(sym, 0.0),
            'macd_hist': metrics['macd'], 'macd_hist_slope': metrics['macd_slope'],
            'spy_roc': spy_roc, 'qqq_roc': qqq_roc,
            'index_trend': index_trend,
            'position': st.position, 'cooldown_until': st.cooldown_until,
            'is_ready': st.warmup_complete,
            'is_banned': curr_ts < self.global_cooldown_until,
            'held_mins': self._calc_trading_minutes(st.entry_ts, curr_ts) if st.position != 0 else 0.0,
            'stock_iv': st.last_valid_iv,
            'holding': {'entry_price': st.entry_price, 'entry_stock': st.entry_stock, 'entry_ts': st.entry_ts, 'dir': st.position, 'max_roi': st.max_roi, 'entry_spy_roc': st.entry_spy_roc, 'entry_index_trend': getattr(st, 'entry_index_trend', 0)} if st.position != 0 else None,
            'curr_price': ctx_curr_price, 'curr_stock': price,
            'bid': ctx_bid,
            'ask': ctx_ask,
            'spread_divergence': 0.0,
            'snap_roc': metrics['snap_roc'],
            'global_regime_reversal_cnt': global_regime_reversal_cnt, # 🚀 [NEW]
            'regime_reversal_count': global_regime_reversal_cnt,     # For V0 compatibility
            'is_volatile_regime': bool(global_is_volatile_regime),
            'regime_band': str(global_regime_band or 'calm'),
            'regime_score': float(global_regime_score or 0.0),
            'state': st
        }

        # 🚨 [IMPORTANT] 在整个信号处理流的末尾，缓存当前价格与期权
        # 给下一秒 (:00) 的分钟指标计算锁定物理截面
        st.last_tick_price = price
        st.last_tick_opt_data = opt_data


        # 补齐 Spread Divergence 与 ROI 更新 (仅持仓时进行计算，防止污染空仓环境)
        if st.position != 0:
            # 🛡️ [价格防投毒] 只有当期权价格合理时才更新缓存
            # 若 ctx_curr_price 超过股票价格的 50%，说明是数据污染，拒绝覆盖
            _stock_ref = float(ctx.get('curr_stock', price))
            _is_valid_opt_price = (ctx_curr_price > 0.01 and
                                   (_stock_ref <= 0 or ctx_curr_price < _stock_ref * 0.5))
            if _is_valid_opt_price:
                st.last_opt_price = ctx_curr_price
            if ctx_bid > 0 and ctx_ask > 0 and ctx_curr_price > 0.01:
                curr_s = (ctx_ask - ctx_bid) / ctx_curr_price
                if st.last_spread_pct > 0:
                    ctx['spread_divergence'] = curr_s - st.last_spread_pct
                st.last_spread_pct = curr_s

            if st.entry_price > 0:
                current_roi = (ctx_curr_price - st.entry_price) / st.entry_price
                st.max_roi = max(st.max_roi, current_roi)
                ctx['holding']['max_roi'] = st.max_roi

        # Alpha Log (零干扰：仅在 full update 窗口写入)
        if should_update_full:
            self._emit_trade_log({
                'action': 'ALPHA',
                'ts': getattr(self, 'current_log_ts', curr_ts), # Use logical log_ts for DB storage
                'symbol': sym,
                'alpha': final_alpha, 'iv': alpha_log_iv, 'price': price, 'vol_z': metrics['vol_z'],
                'event_prob': self.cached_event_probs.get(sym, 0.0),
                'index_trend': index_trend
            })

        # 3. 执行平仓
        if st.position != 0:
            # if curr_ts - st.entry_ts < 59.0:
            #     return None

            current_roi = (ctx_curr_price - st.entry_price) / st.entry_price if st.entry_price > 0 else 0

            exit_sig = self.strategy.check_exit(ctx)
            # [Gate Trace] 回测/备用平仓路径也 publish, 保持 Dashboard 同一数据源一致性
            self._publish_gate_trace(sym, 'exit', exit_sig, event_ts=curr_ts)
            if exit_sig:
                exit_sig['price'] = ctx_curr_price
                exit_sig['market_price'] = market_opt_price
                if opt_data['has_feed']:
                    exit_sig['bid'] = ctx_bid
                    exit_sig['ask'] = ctx_ask
                    exit_sig['bid_size'] = opt_data['call_bid_size'] if st.position == 1 else opt_data['put_bid_size']
                    exit_sig['ask_size'] = opt_data['call_ask_size'] if st.position == 1 else opt_data['put_ask_size']
                else:
                    exit_sig['bid'] = ctx_curr_price
                    exit_sig['ask'] = ctx_curr_price
                    exit_sig['bid_size'] = 999.0
                    exit_sig['ask_size'] = 999.0

                if not getattr(self, 'only_log_alpha', False):
                    await self._execute_exit(sym, exit_sig, price, curr_ts, i)
            return None

        # 4. 开仓决策
        if not should_update_full:
            return None  # 🚀 [核心门控] 高频增量帧仅允许平仓，坚决禁止开仓检查（信息性，不计为拒绝）

        # [Cross-Process Consistency Gate]
        # OMS 快照未就绪前禁止发新 BUY，避免重启窗口的重复开单/错配。
        if (
            self.mode == 'realtime'
            and (not IS_SIMULATED)
            and (not getattr(self, 'use_shared_mem', False))
            and (not getattr(self, '_oms_snapshot_ready', False))
        ):
            self._bump_entry_reject('oms_snapshot_not_ready', sym)
            return None

        self._entry_attempt_count += 1

        if not ctx['is_ready']:
            self._bump_entry_reject('warmup_incomplete', sym, {'alpha_hist': len(self.states[sym].alpha_history)})
            if getattr(self, '_warmup_log_count', 0) < 5:
                logger.info(f"⏳ [SE-Gate] {sym} not ready (Warmup: {len(self.states[sym].alpha_history)})")
                self._warmup_log_count = getattr(self, '_warmup_log_count', 0) + 1
            return None

        if curr_ts < self.global_cooldown_until:
            self._bump_entry_reject('global_cooldown', sym, {'until': self.global_cooldown_until})
            logger.info(f"🛡️ [SE-Gate] Global Cooldown active until {self.global_cooldown_until}")
            return None

        if is_zombie_market:
            self._bump_entry_reject('zombie_market', sym)
            return None

        no_entry_h = self.strategy.cfg.NO_ENTRY_HOUR
        no_entry_m = self.strategy.cfg.NO_ENTRY_MINUTE
        if ny_now.time() >= dt_time(no_entry_h, no_entry_m):
            self._bump_entry_reject('no_entry_window', sym, {'ny': ny_now.strftime('%H:%M:%S')})
            return None

        entry_sig = self.strategy.decide_entry(ctx)
        # [Gate Trace] 不论成败都 publish, publisher 内部节流。放在这里而不是 return 前,
        # 是为了让 entry_sig 既可能是 None (REJECT) 也可能是 dict (BUY) 都经过同一条通路。
        self._publish_gate_trace(sym, 'entry', entry_sig, event_ts=curr_ts)
        if not entry_sig:
            try:
                sub = self.strategy.get_last_reject_reason() or 'strategy_unspecified'
            except Exception:
                sub = getattr(self.strategy, '_last_reject_reason', 'strategy_unspecified') or 'strategy_unspecified'
            self._bump_entry_reject(f'strategy:{sub}', sym, {
                'alpha': ctx.get('alpha_z', 0.0),
                'cs_alpha_z': ctx.get('cs_alpha_z', 0.0),
                'vol_z': ctx.get('vol_z', 0.0),
                'event_prob': ctx.get('event_prob', 0.0),
                'macd_hist': ctx.get('macd_hist', 0.0),
                'spy_roc': ctx.get('spy_roc', 0.0),
            })
            # [Layer A 拒单节流日志] 直接从 trace 抽取 last block gate 打印一次,
            # 方便运维在没有 dashboard 时也能肉眼看到"策略卡在哪条规则"
            try:
                try:
                    tr = self.strategy.get_last_gate_trace()
                except Exception:
                    tr = getattr(self.strategy, '_last_gate_trace', []) or []
                last_block = next((g for g in reversed(tr) if g.get('status') == 'block'), None)
                if last_block:
                    key = (sym, 'entry', last_block.get('gate'))
                    if getattr(self, '_last_reject_log_key', None) != key:
                        self._last_reject_log_key = key
                        logger.info(f"⛔ [SE-Reject] {sym} entry blocked @ {last_block.get('gate')} | {last_block.get('detail')}")
            except Exception:
                pass
            return None

        # [严格守卫]：策略决定好方向后，精准提取对应方向的真实参数提交订单！
        if opt_data['has_feed']:
            is_call = (entry_sig['dir'] == 1)
            t_price  = opt_data['call_price'] if is_call else opt_data['put_price']
            t_id     = opt_data['call_id']    if is_call else opt_data['put_id']
            t_k      = opt_data['call_k']     if is_call else opt_data['put_k']
            t_iv     = opt_data['call_iv']    if is_call else opt_data['put_iv']
            t_vol    = opt_data['call_vol']   if is_call else opt_data['put_vol']
            t_bid    = opt_data['call_bid']   if is_call else opt_data['put_bid']
            t_ask    = opt_data['call_ask']   if is_call else opt_data['put_ask']
            t_bs     = opt_data['call_bid_size'] if is_call else opt_data['put_bid_size']
            t_as     = opt_data['call_ask_size'] if is_call else opt_data['put_ask_size']

            # 🚀 [放宽实盘与回测限制] 不再要求 Size > 0。如果 Bid/Ask 为 0，统一使用 Last Price 兜底放行
            if t_bid <= 0 or t_ask <= 0:
                t_bid = t_price
                t_ask = t_price

            fair_p = self._get_fair_market_price(t_price, t_bid, t_ask)
            if fair_p < 0.05:
                self._bump_entry_reject('opt_fair_price_low', sym, {'fair_p': fair_p, 'bid': t_bid, 'ask': t_ask})
                logger.info(f"🚫 [SE-Gate] {sym} Fair Price {fair_p:.4f} too low")
                return None
            if not t_id:
                self._bump_entry_reject('opt_missing_contract_id', sym, {'strike': t_k})
                logger.info(f"🚫 [SE-Gate] {sym} missing Option ID")
                return None

            strike_valid = (t_k > 1.0 and abs(t_k - price) / max(price, 1.0) < 0.80)
            if strike_valid:
                intrinsic = max(0.0, price - t_k) if is_call else max(0.0, t_k - price)
                if fair_p < intrinsic * 0.9:
                    self._bump_entry_reject('opt_below_intrinsic', sym, {'fair_p': fair_p, 'intrinsic': intrinsic, 'strike': t_k, 'stock': price})
                    return None

            entry_sig.update({
                'price': fair_p, 'contract_id': t_id,
                'meta': {
                    'strike': t_k, 'iv': t_iv, 'contract_id': t_id, 'volume': t_vol,
                    'bid': t_bid, 'ask': t_ask, 'bid_size': t_bs, 'ask_size': t_as,
                    'spy_roc': spy_roc, 'alpha_z': final_alpha,
                    'index_trend': index_trend,
                    'alpha_label_ts': metrics.get('alpha_label_ts', 0.0),
                    'alpha_available_ts': metrics.get('alpha_available_ts', curr_ts),
                }
            })
            self._entry_pass_count += 1
            return entry_sig
        self._bump_entry_reject('no_option_feed', sym)
        return None


    # ---- archived from signal_engine_v8.py: _evaluate_symbol_signals_back ----
    async def _evaluate_symbol_signals_back(self, i, sym, metrics, opt_data, ny_now, curr_ts, spy_roc, qqq_roc, is_zombie_market, index_trend=0):
        """[Refactor] 核心策略评价 logic (平仓与开仓信号收集)"""
        st = metrics['st']
        price = metrics['price']
        final_alpha = metrics['final_alpha']
        alpha_log_iv = st.last_valid_iv

        if opt_data['has_feed']:
            if st.position == 1:
                log_iv_candidate = opt_data.get('call_iv', 0.0)
            elif st.position == -1:
                log_iv_candidate = opt_data.get('put_iv', 0.0)
            else:
                log_call_iv = opt_data.get('call_iv', 0.0)
                log_put_iv = opt_data.get('put_iv', 0.0)
                if log_call_iv > 0.01 and log_put_iv > 0.01:
                    log_iv_candidate = (log_call_iv + log_put_iv) / 2.0
                elif log_call_iv > 0.01:
                    log_iv_candidate = log_call_iv
                else:
                    log_iv_candidate = log_put_iv

            if log_iv_candidate > 0.01:
                alpha_log_iv = log_iv_candidate

        # 1. 更新 IV 状态
        if opt_data['has_feed']:
            if st.position == 1: curr_iv = opt_data['call_iv']
            elif st.position == -1: curr_iv = opt_data['put_iv']
            else:
                cv = opt_data['call_iv']
                pv = opt_data['put_iv']
                if cv > 0.01 and pv > 0.01: curr_iv = (cv + pv) / 2.0
                elif cv > 0.01: curr_iv = cv
                elif pv > 0.01: curr_iv = pv
                else: curr_iv = 0.0

            # 🚀 [Debug] 最终阶段 TRACE
            if sym == 'NVDA' and getattr(self, '_iv_se_count', 0) < 5:
                # 检查 opt_data 原始值
                c_data = opt_data.get('call_iv', -1)
                p_data = opt_data.get('put_iv', -1)
                logger.info(f"🧪 [IV_TRACE_3] {sym} | SE Raw OptData | Call_IV_Data: {c_data:.4f} | Put_IV_Data: {p_data:.4f} | Final_Curr_IV: {curr_iv:.4f}")
                self._iv_se_count = getattr(self, '_iv_se_count', 0) + 1

            if curr_iv > 0.01:
                st.last_valid_iv = curr_iv



        # 2. 构建 Context
        ctx = {
            'symbol': sym, 'time': ny_now, 'curr_ts': curr_ts, 'price': price,
            'alpha_z': final_alpha, 'vol_z': metrics['vol_z'], 'stock_roc': metrics['roc_5m'],
            'macd_hist': metrics['macd'], 'macd_hist_slope': metrics['macd_slope'],
            'spy_roc': spy_roc, 'qqq_roc': qqq_roc,
            'index_trend': index_trend, # [NEW] 传导当日大盘趋势
            'position': st.position, 'cooldown_until': st.cooldown_until,
            'is_ready': st.warmup_complete,
            'is_banned': curr_ts < self.global_cooldown_until,
            'held_mins': self._calc_trading_minutes(st.entry_ts, curr_ts) if st.position != 0 else 0.0,
            'stock_iv': st.last_valid_iv,
            'holding': {'entry_price': st.entry_price, 'entry_stock': st.entry_stock, 'entry_ts': st.entry_ts, 'dir': st.position, 'max_roi': st.max_roi, 'entry_spy_roc': st.entry_spy_roc, 'entry_index_trend': getattr(st, 'entry_index_trend', 0)} if st.position != 0 else None,
            'curr_price': 0.0, 'curr_stock': price,
            # [🔥 核心新增] 传递 Bid/Ask 供策略进行流动性（Spread）校验
            'bid': opt_data.get('call_bid' if st.position >= 0 else 'put_bid', 0.0) if opt_data['has_feed'] else 0.0,
            'ask': opt_data.get('call_ask' if st.position >= 0 else 'put_ask', 0.0) if opt_data['has_feed'] else 0.0,
            'spread_divergence': 0.0, # 默认为 0
            'snap_roc': metrics['snap_roc'] # [🔥 新增] 传导最后一分钟/Snap的价格Delta
        }

        # 计算价差变化率 (Spread Divergence)
        if ctx['bid'] > 0 and ctx['ask'] > 0 and ctx['curr_price'] > 0.01:
             curr_s = (ctx['ask'] - ctx['bid']) / ctx['curr_price']
             if st.last_spread_pct > 0:
                 ctx['spread_divergence'] = curr_s - st.last_spread_pct
             st.last_spread_pct = curr_s

        # Alpha Log
        self._emit_trade_log({
            'action': 'ALPHA',
            'ts': getattr(self, 'current_log_ts', curr_ts), # Use logical log_ts for DB storage
            'symbol': sym,
            'alpha': final_alpha, 'iv': alpha_log_iv, 'price': price, 'vol_z': metrics['vol_z'],
            'index_trend': index_trend # [NEW] 记录入场时的趋势背景
        })

        # 3. 计算期权公允价
        market_opt_price = 0.0
        if opt_data['has_feed']:
            market_opt_price = opt_data['call_price'] if st.position >= 0 else opt_data['put_price'] # st.position=0 时默认 call
            if st.position != 0:
                bid = opt_data['call_bid'] if st.position == 1 else opt_data['put_bid']
                ask = opt_data['call_ask'] if st.position == 1 else opt_data['put_ask']

                # 在回测模式或盘口正常时，使用公允价 (Mid-price from Feature Service)
                # 🚀 [终极修正] 彻底删除 Delta 0.5 投影估值逻辑，始终信任数据源提供的公允价。
                ctx['curr_price'] = self._get_fair_market_price(market_opt_price, bid, ask)
                st.last_opt_price = ctx['curr_price'] # 👈 [新增] 缓存正常公允价

                if st.entry_price > 0:
                    current_roi = (ctx['curr_price'] - st.entry_price) / st.entry_price
                    st.max_roi = max(st.max_roi, current_roi)
                    ctx['holding']['max_roi'] = st.max_roi

            else:
                ctx['curr_price'] = market_opt_price
                st.last_opt_price = ctx['curr_price'] # 👈 [新增]
        else:
            if st.position != 0:
                # 🛡️ [防断流核心 3] 彻底缺失数据 (has_feed=False)，直接用开仓价兜底
                effective_price = st.entry_price
                if effective_price <= 0.01: effective_price = 0.01

                # 👇 [🔥 把 Debug 改为 Error，强制暴露问题]
                ctx['curr_price'] = effective_price
                st.last_opt_price = ctx['curr_price'] # 👈 [新增] 缓存兜底价
                logger.error(f"🚨 [致命盲区] {sym} 期权行情彻底丢失(has_feed=False)！系统被蒙住双眼，当前 ROI 强制归 0.0！")
            else:
                ctx['curr_price'] = 0.0

        # 4. 执行平仓 (必须在 is_zombie_market 之前，确保 EOD 和止损能随时触发)
        if st.position != 0:
            high_freq_tick = self._is_high_freq_tick(st, curr_ts)
            # 🛡️ [终极修复] 建仓绝对保护期 (Entry Breathing Room)
            # 绝对禁止在建仓后的最初 60 秒内通过常规逻辑平仓
            if curr_ts - st.entry_ts < 59.0:
                logger.debug(f"🔒 [平仓屏蔽] {sym} 处于 59 秒建仓保护期内，暂不进行平仓评估。")
                return None

            # [🎯 靶向日志 B2] 送入 check_exit 前的最终状态切片
            current_roi = (ctx['curr_price'] - st.entry_price) / st.entry_price if st.entry_price > 0 else 0
            logger.info(f"🔍 [主循环平仓送审] {sym} | 模式: {self.mode} | 当前价: {ctx['curr_price']:.2f} | 成本: {st.entry_price:.2f} | ROI: {current_roi*100:.2f}% | Max ROI: {st.max_roi*100:.2f}%")


            exit_sig = self.strategy.check_exit(ctx)
            # [Gate Trace] publish 决策链。注意: high-freq confirm 之后可能把 exit_sig 变 None,
            # 但那是"延迟确认", 不是策略拒绝平仓, 所以 publish 用 confirm 前的值更能反映意图。
            self._publish_gate_trace(sym, 'exit', exit_sig, event_ts=curr_ts)
            if exit_sig:
                if high_freq_tick:
                    exit_sig = self._confirm_high_freq_exit(st, exit_sig, curr_ts)
                    if not exit_sig:
                        return None
                exit_sig['price'] = ctx['curr_price']
                exit_sig['market_price'] = market_opt_price
                if opt_data['has_feed']:
                    is_call = (st.position == 1)
                    exit_sig['bid'] = opt_data['call_bid'] if is_call else opt_data['put_bid']
                    exit_sig['ask'] = opt_data['call_ask'] if is_call else opt_data['put_ask']
                    # 🚀 [新增] 穿透传输深度数据，支持秒级分段批量成交
                    exit_sig['bid_size'] = opt_data['call_bid_size'] if is_call else opt_data['put_bid_size']
                    exit_sig['ask_size'] = opt_data['call_ask_size'] if is_call else opt_data['put_ask_size']
                else:
                    exit_sig['bid'] = ctx['curr_price']
                    exit_sig['ask'] = ctx['curr_price']
                    exit_sig['bid_size'] = 999.0
                    exit_sig['ask_size'] = 999.0
                if not self.only_log_alpha:
                    await self._execute_exit(sym, exit_sig, price, curr_ts, i)
                    return None # 非 Parity 模式，等待 OMS 异步清算回调后再充当空仓
                else:
                    logger.info(f"📝 [Shadow] {sym} Exit signal detected, but skipping execution (Alpha-Only).")
                    return None
            else:
                if high_freq_tick:
                    self._reset_exit_confirmation(st)
                # [🎯 靶向日志 B3] 明确是被策略内部拒绝
                logger.debug(f"🛡️ [策略拒平] {sym} 的 check_exit 返回 None，当前无满足条件的平仓信号。")
                return None

        # 5. 开仓决策 (受僵尸市场和冷却时间保护)
        if curr_ts < self.global_cooldown_until or is_zombie_market: return None

        # [🔥 修复] 移除硬编码的 15:30，动态读取策略配置的停止开仓时间
        no_entry_h = self.strategy.cfg.NO_ENTRY_HOUR
        no_entry_m = self.strategy.cfg.NO_ENTRY_MINUTE
        if ny_now.time() >= dt_time(no_entry_h, no_entry_m): return None

        entry_sig = self.strategy.decide_entry(ctx)
        # [Gate Trace] 备用 decide_entry 路径亦 publish, Dashboard 不区分路径
        self._publish_gate_trace(sym, 'entry', entry_sig, event_ts=curr_ts)
        if not entry_sig:
            # logger.info(f"⚪ [SE-Gate] {sym} strategy rejected entry")
            return None

        # [严格守卫]
        if opt_data['has_feed']:
            is_call = (entry_sig['dir'] == 1)
            t_price  = opt_data['call_price'] if is_call else opt_data['put_price']
            t_id     = opt_data['call_id']    if is_call else opt_data['put_id']
            t_k      = opt_data['call_k']     if is_call else opt_data['put_k']
            t_iv     = opt_data['call_iv']    if is_call else opt_data['put_iv']
            t_vol    = opt_data['call_vol']   if is_call else opt_data['put_vol']
            t_bid    = opt_data['call_bid']   if is_call else opt_data['put_bid']
            t_ask    = opt_data['call_ask']   if is_call else opt_data['put_ask']
            t_bs     = opt_data['call_bid_size'] if is_call else opt_data['put_bid_size']
            t_as     = opt_data['call_ask_size'] if is_call else opt_data['put_ask_size']

            # 🛡️ [New] 流动性过滤门槛 (仅在 STRICT_LIQUIDITY_MODE=1 时启用)
            from config import STRICT_LIQUIDITY_MODE
            if STRICT_LIQUIDITY_MODE == 1 and (t_bs + t_as) <= 50:
                logger.info(f"🚫 [SE-Gate] {sym} Liquidity Filtered: bid_size+ask_size = {t_bs+t_as} <= 100")
                return None

            # 🚀 [放宽实盘与回测限制] 不再要求 Size > 0。如果 Bid/Ask 为 0，统一使用 Last Price 兜底放行
            if t_bid <= 0 or t_ask <= 0:
                t_bid = t_price
                t_ask = t_price

            fair_p = self._get_fair_market_price(t_price, t_bid, t_ask)
            if fair_p < 0.05:
                logger.info(f"🚫 [SE-Gate] {sym} Fair Price {fair_p:.4f} too low")
                return None
            if not t_id:
                logger.info(f"🚫 [SE-Gate] {sym} missing Option ID")
                return None

            # 内在价值校验
            strike_valid = (t_k > 1.0 and abs(t_k - price) / max(price, 1.0) < 0.80)
            if strike_valid:
                intrinsic = max(0.0, price - t_k) if is_call else max(0.0, t_k - price)
                if fair_p < intrinsic * 0.9: return None

            # 记录空跑截影
            spread_pct = (t_ask - t_bid) / fair_p if fair_p > 0 else 0
            logger.info(f"📸 [实盘空跑截影] Alpha: {final_alpha:.2f} | {sym} | Bid:{t_bid:.2f} Ask:{t_ask:.2f} | Fair:{fair_p:.2f}")

            entry_sig.update({
                'price': fair_p, 'contract_id': t_id,
                'meta': {
                    'strike': t_k, 'iv': t_iv, 'contract_id': t_id, 'volume': t_vol,
                    'bid': t_bid, 'ask': t_ask, 'bid_size': t_bs, 'ask_size': t_as,
                    'spy_roc': spy_roc, 'alpha_z': final_alpha, 'ask_size': t_as, # [Fix] 显式传导 ask_size 供流动性评估
                    'index_trend': index_trend,
                    'alpha_label_ts': metrics.get('alpha_label_ts', 0.0),
                    'alpha_available_ts': metrics.get('alpha_available_ts', curr_ts),
                }
            })
            return entry_sig
        return None

    # ---- archived from signal_engine_v8.py: _process_exits ----
    async def _process_exits(self, batch: dict):
        """[高频风控] 每一秒执行一次，检查持仓标的是否达到策略止损/止盈阈值"""
        symbols = batch['symbols']
        prices = batch['stock_price']
        curr_ts = getattr(self, 'last_curr_ts', time.time())
        from pytz import timezone
        ny_now = datetime.fromtimestamp(curr_ts, timezone('America/New_York'))

        # 1. 预判断是否有持仓，减少无谓计算
        has_any_pos = any(st.position != 0 for st in self.states.values())
        if not has_any_pos: return

        # 2. 扫描所有持仓标的 (改为“持仓驱动”，确保即便这一秒行情缺失也会进行风控扫荡)
        # 先建立当前 batch 的快速索引
        sym_to_idx = {s: idx for idx, s in enumerate(symbols)}

        active_positions = [st for st in self.states.values() if st.position != 0]

        for st in active_positions:
            sym = st.symbol
            idx_in_batch = sym_to_idx.get(sym)

            # 3. 确定行情数据源
            if idx_in_batch is not None:
                # 这一秒有新数据，使用最新的
                curr_price = float(prices[idx_in_batch])
                if self.mode == 'realtime' or os.environ.get('RUN_MODE') == 'LIVEREPLAY':
                    opt_data = self._get_opt_data_realtime(sym, st, ny_now, curr_price, batch)
                else:
                    opt_data = self._get_opt_data_backtest(batch, idx_in_batch, sym, st)
            else:
                # 这一秒没新数据，使用上一秒的缓存
                curr_price = getattr(st, 'last_tick_price', 0.0)
                opt_data = getattr(st, 'last_tick_opt_data', None)

                # 如果连缓存都没有（刚恢复还没收到第一笔），只能跳过
                if not opt_data or curr_price <= 0.01:
                    continue

            # 🚀 [核心对齐] 出场检查使用的是：
            # 1. 当前秒级的价格 (curr_price)
            # 2. 上一分钟结算好的稳健指标快照 (get_strategy_metrics)
            roc_5m, macd, macd_slope, snap_roc = st.get_strategy_metrics()

            market_opt_price = opt_data['call_price'] if st.position == 1 else opt_data['put_price']
            ctx_bid = opt_data['call_bid'] if st.position == 1 else opt_data['put_bid']
            ctx_ask = opt_data['call_ask'] if st.position == 1 else opt_data['put_ask']

            raw_price = self._get_fair_market_price(market_opt_price, ctx_bid, ctx_ask, getattr(st, 'last_opt_price', 0.0))
            st.last_opt_price = raw_price

            metrics = {
                'price': float(curr_price),
                'roc_5m': roc_5m, 'macd': macd, 'macd_slope': macd_slope,
                'snap_roc': snap_roc, 'st': st,
                'final_alpha': st.last_alpha_z, 'vol_z': st.last_vol_z
            }

            # 4. 提交风控评价 (should_update_full=False 确保只执行平仓逻辑，不触动开仓评价)
            await self._evaluate_symbol_signals(
                0, sym, metrics, opt_data, ny_now, curr_ts,
                self.last_spy_roc_val, self.last_qqq_roc_val,
                getattr(self, 'last_is_zombie', False), self.last_index_trend,
                global_regime_reversal_cnt=getattr(self, 'last_global_regime_reversal_cnt', 0),
                global_is_volatile_regime=getattr(self, 'last_global_is_volatile_regime', False),
                global_regime_band=getattr(self, 'last_global_regime_band', 'calm'),
                global_regime_score=getattr(self, 'last_global_regime_score', 0.0),
                should_update_full=False
            )


    # ---- archived from signal_engine_v8.py: _emit_trade_signal ----
    async def _emit_trade_signal(self, action, sym, sig, stock_price, curr_ts, batch_idx):
        st = self.states[sym]
        original_position = st.position

        # 🚨 [真正的无状态设计]
        # SE 只负责发出意图，绝对不允许越权修改本地的 st.position 等真实资金状态。
        # 所有的状态变更必须由 OMS 撮合后通过 Redis 反向同步过来。

        if action == 'BUY':
            st.is_pending = True
            st.pending_action = 'BUY'  # 👈 [新增] 标记为正在等待买入确认
            # [⏱️ BUY Emit Cooldown] 打上 emit 时间戳, 给 process_batch 的 cooldown 和
            # Batch-Limit 的 active-count 精确窗口做支撑, 避免同一 symbol 每秒重复 emit。
            st.last_buy_emit_ts = float(curr_ts) if curr_ts else time.time()

        elif action == 'SELL':
            st.is_pending = True
            st.pending_action = 'SELL'
            sig['original_position'] = original_position

            # 🛑 绝对禁止在这里写 st.position = 0！
            # 在共享内存 (S4) 下，如果 SE 提前清零，执行侧 (EE) 会因为找不到持仓而拒绝执行 SELL。
            # 必须留给 EE 在真正处理完平仓后再亲自清零。

            # 🛑 绝对禁止在这里写 st.entry_price = 0 或 st.qty = 0！
            # 必须把这些带着成本记忆的原始数据，原封不动地留在共享内存里。
            # 留给 OMS (_execute_exit) 算完精准的利润后，由 OMS 亲自去清空！

            # 🛑 绝对禁止在这里写 st.cooldown_until = ...！
            # 止盈单不需要冷却！把冷却的判断权交还给 OMS 的止损记账模块！


        payload = {
            'ts': curr_ts,
            'symbol': sym,
            'action': action,
            'sig': sig,
            'stock_price': stock_price,
            'batch_idx': batch_idx,
            'logical_roi': getattr(st, 'max_roi', 0.0) if action == 'SELL' else 0.0,
            'prices': {sym: getattr(st, 'last_opt_price', 0.0)}
        }

        if getattr(self, 'use_shared_mem', False):
            # 🚀 压入内存队列 (耗时 1 微秒)
            await self.signal_queue.put(payload)
        else:
            # 原有的 Redis 打包发送逻辑
            self.r.xadd('orch_trade_signals', {'data': ser.pack(payload)}, maxlen=5000)
            logger.info(f"🚀 [SIGNAL_ENGINE] Published {action} signal for {sym} to Redis.")


    # ================= 新增：从 OMS 同步真实账本的方法 =================

    # ---- archived from signal_engine_v8.py: _publish_gate_trace ----
    def _publish_gate_trace(self, sym: str, kind: str, result_sig, event_ts=None):
        try:
            try:
                trace = self.strategy.get_last_gate_trace()
            except Exception:
                trace = list(getattr(self.strategy, '_last_gate_trace', []) or [])
            if not trace:
                return
            event_ts_safe = None
            if event_ts is not None:
                try:
                    event_ts_safe = float(event_ts)
                    if event_ts_safe <= 0:
                        event_ts_safe = None
                except (TypeError, ValueError):
                    event_ts_safe = None
            # 推导 result_label
            if result_sig:
                act = result_sig.get('action')
                if act == 'BUY':
                    result_label = 'BUY'
                elif act == 'SELL':
                    # 从返回信号里带 reason 简写
                    reason = (result_sig.get('reason') or 'SELL').split('|')[0][:32]
                    result_label = f"SELL:{reason}"
                else:
                    result_label = 'PASS'
            else:
                # 找最后一个 block 作为 reject 理由
                last_block = None
                for g in reversed(trace):
                    if g.get('status') == 'block':
                        last_block = g.get('gate')
                        break
                result_label = f"REJECT:{last_block}" if last_block else "PASS"

            last_block_gate = None
            for g in reversed(trace):
                if g.get('status') == 'block':
                    last_block_gate = g.get('gate')
                    break

            # 节流判定: 相同 (sym,kind) 上次的 result 与 last_block 无变化则跳过写 Redis
            key = (sym, kind)
            prev = self._gate_trace_pub_state.get(key)
            curr = (result_label, last_block_gate)
            # 计数去抖: 仅在 gate 变化或日期切换时 INCR，避免持续 block 期间每 tick 写一次。
            if last_block_gate:
                try:
                    ts_for_day = event_ts_safe if event_ts_safe is not None else time.time()
                    ny_date = datetime.fromtimestamp(ts_for_day, timezone('America/New_York')).strftime('%Y%m%d')
                    counter_key = f"meta:gate_counter:{ny_date}"
                    prev_cnt = self._gate_counter_pub_state.get(key)
                    curr_cnt = (ny_date, last_block_gate)
                    if prev_cnt != curr_cnt:
                        self.r.hincrby(counter_key, last_block_gate, 1)
                        # 48h TTL, 日终 day-boundary 钩子也会显式 DEL 掉, 这里是双保险
                        self.r.expire(counter_key, 48 * 3600)
                        self._gate_counter_pub_state[key] = curr_cnt
                except Exception:
                    pass

            if prev == curr:
                # 刷新 TTL, 保证 Dashboard 端不会把仍然 active 的 trace 当过期条目
                try:
                    self.r.expire(f"meta:gate_trace:{sym}", self._gate_trace_ttl)
                except Exception:
                    pass
                return
            self._gate_trace_pub_state[key] = curr

            # 序列化: trace 列表 JSON (<= 40 项, 每项 ~80B, 总包体 < 5KB, 单次 HSET 完全可接受)
            import json as _json
            payload = {
                'kind': kind,
                'ts': f"{event_ts_safe:.3f}" if event_ts_safe is not None else f"{time.time():.3f}",
                'result': result_label,
                'last_block': last_block_gate or '',
                'trace_json': _json.dumps(trace, ensure_ascii=False),
            }
            pipe = self.r.pipeline()
            pipe.hset(f"meta:gate_trace:{sym}", mapping=payload)
            pipe.expire(f"meta:gate_trace:{sym}", self._gate_trace_ttl)
            pipe.execute()
        except Exception as e:
            # 首次报错打一行, 避免刷屏
            if not getattr(self, '_gate_pub_err_logged', False):
                logger.warning(f"⚠️ [Gate Trace Pub] failed: {e}")
                self._gate_pub_err_logged = True


    # ---- archived from signal_engine_v8.py: _bump_entry_reject ----
    def _bump_entry_reject(self, reason, sym=None, extra=None):
        try:
            self._entry_reject_counts[reason] = self._entry_reject_counts.get(reason, 0) + 1
            if sym and reason not in self._entry_reject_samples:
                snap = {'sym': sym}
                if isinstance(extra, dict):
                    for k, v in list(extra.items())[:6]:
                        try:
                            snap[k] = float(v) if isinstance(v, (int, float)) else str(v)[:40]
                        except Exception:
                            pass
                self._entry_reject_samples[reason] = snap
        except Exception:
            pass


    # ---- archived from signal_engine_v8.py: _emit_entry_reject_stats ----
    def _emit_entry_reject_stats(self):
        try:
            if not self._entry_reject_counts and self._entry_attempt_count == 0:
                return
            total_rejects = sum(self._entry_reject_counts.values())
            top = sorted(self._entry_reject_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            top_str = " | ".join([f"{k}={v}" for k, v in top])
            logger.info(
                f"🛂 [Entry-Reject-Stats] 60s attempts={self._entry_attempt_count} passes={self._entry_pass_count} "
                f"rejects={total_rejects} | {top_str}"
            )
            # 每 5 分钟额外输出一次典型样本，便于直接定位
            if int(time.time()) // 300 != getattr(self, '_entry_reject_sample_bucket', -1):
                self._entry_reject_sample_bucket = int(time.time()) // 300
                if self._entry_reject_samples:
                    for reason, snap in list(self._entry_reject_samples.items())[:6]:
                        logger.info(f"🔎 [Entry-Reject-Sample] {reason}: {snap}")
            self._entry_reject_counts.clear()
            self._entry_reject_samples.clear()
            self._entry_attempt_count = 0
            self._entry_pass_count = 0
        except Exception as e:
            logger.warning(f"[Entry-Reject-Stats] emit failed: {e}")

    # === Accounting Logic ===

    # ---- archived from signal_engine_v8.py: _sync_state_from_oms ----
    def _sync_state_from_oms(self):
        """[核心解耦] 每次评估前，从 OMS 获取最真实的仓位与成本"""
        is_simulated_runtime = bool(IS_SIMULATED)
        is_backtest_runtime = bool(IS_BACKTEST)
        if getattr(self, 'use_shared_mem', False):
            # 🚀 [Parity Fix] 在 S4 单进程回测或秒级同步模式下，绝对禁止主动清空 is_pending
            # 该标志位必须由 ExecutionEngine 在处理完信号后亲自清空。
            # 如果在这里清零，会导致同一分钟内的后续秒级 Tick 重复发单，造成严重的风控拒单风暴。
            if is_backtest_runtime or is_simulated_runtime or os.environ.get('RUN_MODE') == 'LIVEREPLAY':
                # [🛡️ Ghost Pending Watchdog]
                # 第二道防御：即便 OMS 因 bug 漏清了 is_pending (历史上出现过:
                # _execute_entry 的早退分支未释放锁、IBKR 断线、check_exit TypeError 等),
                # 也要避免 pending BUY 无限占用 MAX_POSITIONS 名额。
                # 规则: position==0 且 pending_action=='BUY' 的 state, 每 tick 记一帧,
                # 超过 GHOST_PENDING_FRAMES 帧 (默认 10 秒级 tick ≈ 10s) 自动回收。
                try:
                    ghost_ttl = int(os.environ.get('GHOST_PENDING_FRAMES', '10'))
                except (TypeError, ValueError):
                    ghost_ttl = 10
                for _sym, _st in self.states.items():
                    if not getattr(_st, 'is_pending', False):
                        continue
                    if getattr(_st, 'position', 0) != 0:
                        # 已经真的有仓位，pending 等 OMS 自己释放
                        continue
                    if getattr(_st, 'pending_action', '') != 'BUY':
                        continue
                    _frames = getattr(_st, '_pending_frames', 0) + 1
                    _st._pending_frames = _frames
                    if _frames > ghost_ttl:
                        logger.warning(
                            f"👻 [Ghost Pending GC] {_sym} 幽灵 BUY pending 超时 ({_frames} 帧)，"
                            f"强制释放锁以避免堆满 MAX_POSITIONS。"
                        )
                        _st.is_pending = False
                        _st.pending_action = ''
                        _st._pending_frames = 0
                return

            # 只有在非回放的实时共享内存模式下，才保留这种“强制解锁”自救机制。
            for sym, st in self.states.items():
                if getattr(st, 'is_pending', False):
                    st.is_pending = False
                    st.pending_action = ''
            return

        raw_states = self.r.hgetall("oms:live_positions")
        has_system_cash = False
        if raw_states:
            has_system_cash = (
                (b"____SYSTEM_CASH____" in raw_states) or ("____SYSTEM_CASH____" in raw_states)
            )

        # [Startup/Outage Guard]
        # 双进程实盘下，OMS 未就绪时会出现空快照；此时绝不能把 SE 本地状态清零。
        if (not raw_states) or (not has_system_cash):
            if not is_simulated_runtime:
                self._oms_snapshot_ready = False
                now_ts = float(getattr(self, 'last_curr_ts', 0.0) or time.time())
                last_warn = float(getattr(self, '_last_oms_snapshot_warn_ts', 0.0) or 0.0)
                if now_ts - last_warn >= 15.0:
                    logger.warning(
                        "⏳ [SE_SYNC] OMS snapshot not ready (empty/missing SYSTEM_CASH); "
                        "keep local state and block new entries temporarily."
                    )
                    self._last_oms_snapshot_warn_ts = now_ts
            return
        self._oms_snapshot_ready = True
        active_syms = set()

        # 1. 恢复 OMS 确认的持仓
        for sym_b, data_b in raw_states.items():
            sym = sym_b.decode('utf-8')

            # 特殊项：系统资金
            if sym == "____SYSTEM_CASH____":
                try:
                    cash_data = json.loads(data_b.decode('utf-8'))
                    self.mock_cash = cash_data['cash']
                except: pass
                continue

            if sym in self.states:
                import json
                data = json.loads(data_b.decode('utf-8'))
                st = self.states[sym]

                # 👇 [核心修复 1: 保护最大收益率不被 OMS 的陈旧数据覆盖]
                old_pos = st.position
                new_pos = data['pos']

                st.position = new_pos
                st.qty = data.get('qty', 0)
                st.entry_price = data.get('price', 0.0)
                st.entry_stock = data.get('stock', 0.0)
                st.entry_ts = data.get('ts', 0.0)

                # 🛑 记忆护城河：如果是新开仓，初始化 max_roi；如果已经在持仓，坚决保留 SE 本地追踪的 max_roi！
                if old_pos == 0 and new_pos != 0:
                    st.max_roi = 0.0
                elif new_pos == 0:
                    st.max_roi = -1.0

                # 恢复其他的元数据 (这些在入场时固定，可以安全覆盖)
                # [🛡️ Defensive Coerce] 与 from_dict 同样防御: 上游 OMS 通过
                # holding payload 回传过来时曾把 meta.spy_roc 等字段错当 dict 透传.
                def _sc_f(v, d=0.0):
                    if isinstance(v, dict): return d
                    try: return float(v)
                    except (TypeError, ValueError): return d
                def _sc_i(v, d=0):
                    if isinstance(v, dict): return d
                    try: return int(v)
                    except (TypeError, ValueError): return d
                st.entry_spy_roc = _sc_f(data.get('entry_spy_roc', 0.0))
                st.entry_index_trend = _sc_i(data.get('entry_index_trend', 0))
                st.entry_alpha_z = _sc_f(data.get('entry_alpha_z', 0.0))
                st.entry_iv = _sc_f(data.get('entry_iv', getattr(st, 'last_valid_iv', 0.0)))

                # [🎯 核心修复] 释放锁
                st.is_pending = False
                st._pending_frames = 0
                active_syms.add(sym)

        # 2. 清理幻觉：如果 SE 以为有仓位，但 OMS 账本里没有，强制清零！
        for sym, st in self.states.items():
            if sym not in active_syms:
                # 如果是刚刚发出的 BUY 指令，OMS 还没来得及确认，我们容忍几帧
                if getattr(st, 'is_pending', False) and getattr(st, 'pending_action', '') == 'BUY':
                    # [🛡️ Pending 清理门槛]
                    # 原设计 threshold=3 容忍 OMS 3~4 秒才确认, 但这直接导致 MAX_POSITIONS
                    # 被顽固 pending 堆满 (SE 每秒重新 emit, frames 到 4 清一次立刻又被打上)。
                    # 现在改为 1 (≈ 2s): 给 OMS 一个 round-trip 的容忍度即可。配合
                    # `last_buy_emit_ts` 的 emit cooldown, 清锁后也不会立即重发。
                    threshold = 0 if is_simulated_runtime else 1
                    st._pending_frames = getattr(st, '_pending_frames', 0) + 1
                    if st._pending_frames > threshold:
                        logger.info(
                            f"🚨 [SE_SYNC] {sym} pending BUY 被 OMS 拒绝或超时, 释放锁 (frames={st._pending_frames})。"
                        )
                        st.is_pending = False
                        st._pending_frames = 0
                        st.pending_action = ''
                        # 种下 emit cooldown, 让下一个 tick 不会立刻又对同一 symbol 发 BUY,
                        # 给系统一个恢复/重评估窗口 (cooldown 实际长度由 SE_BUY_EMIT_COOLDOWN_SEC 决定)。
                        try:
                            _now_ts = getattr(self, 'last_curr_ts', 0.0) or time.time()
                        except Exception:
                            _now_ts = time.time()
                        st.last_buy_emit_ts = float(_now_ts)
                else:
                    # 🚀 [核心修复：光速垃圾回收]
                    # 如果是 SELL 之后的确认，或者压根没发单 OMS 也不承认，立刻果断清空！
                    # 绝不允许幽灵状态占用 active_count！
                    st.position = 0
                    st.is_pending = False
                    st.entry_price = 0.0
                    st._pending_frames = 0
                    st.pending_action = ''

        # 3. 🔥 [Circuit Breaker Cross-Process Sync]
        #    OMS 进程在 orchestrator_accounting._process_exit_accounting 触发连败熔断后,
        #    会把 global_cooldown_until 写到 Redis hash `meta:circuit_breaker`。
        #    SE 这里读取, 让独立进程的 SE gate (curr_ts < self.global_cooldown_until)
        #    能够真正生效; 否则 SE 本地 global_cooldown_until 永远是 0, BUY 信号会被
        #    源源不断地 emit, 被 OMS 静默拒单, 用户看上去就是"熔断消失了"。
        self._sync_circuit_breaker_from_redis()

        # 4. ⏳ [Per-Symbol Cooldown Cross-Process Sync]
        #    OMS _process_exit_accounting 在止损/反转后会写 st.cooldown_until,
        #    用来抑制同标的 60min 内再次开仓。但双引擎架构下该字段从未广播,
        #    SE 端 st.cooldown_until 永远是 0, 策略层 `ctx['cooldown_until']` 失效。
        #    修复方式: OMS 广播 `meta:symbol_cooldowns` hash, SE 同步进 st.cooldown_until。
        self._sync_symbol_cooldowns_from_redis()


    # ---- archived from signal_engine_v8.py: _sync_circuit_breaker_from_redis ----
    def _sync_circuit_breaker_from_redis(self):
        """从 Redis 读取 OMS 广播的熔断状态, 同步到 SE 本地 global_cooldown_until。

        策略:
          - 仅当 Redis 中的 cb_until 晚于 SE 当前节拍 (last_curr_ts / wall clock), 才认为熔断仍生效。
          - 用 max(local, redis) 避免 Redis 残留的旧值覆盖掉 day-boundary 的清零 (本地 0 与过期值比较后,
            若 Redis 值已在过去, 不会写入; 若仍在未来, SE 应当感知)。
          - 任何异常都不能影响交易主流程; 捕获后静默返回即可, 下一 tick 再试。
        """
        try:
            r = getattr(self, 'r', None)
            if r is None:
                return
            data = r.hgetall("meta:circuit_breaker")
            if not data:
                return
            raw = data.get(b'global_cooldown_until') or data.get('global_cooldown_until')
            if raw is None:
                return
            try:
                cb_until = float(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
            except (TypeError, ValueError):
                return

            # 判断是否过期: 优先用 SE 自己的节拍时间 (回放/模拟时 wall clock 不可靠)
            try:
                now_ts = float(getattr(self, 'last_curr_ts', 0.0) or 0.0)
            except Exception:
                now_ts = 0.0
            if now_ts <= 0.0:
                now_ts = time.time()
            if cb_until <= now_ts:
                return  # 熔断窗口已过, 忽略

            prev_local = float(getattr(self, 'global_cooldown_until', 0.0) or 0.0)
            if cb_until > prev_local:
                self.global_cooldown_until = cb_until
                # 首次感知熔断或窗口被延长时, 打一条警示日志
                try:
                    ny = datetime.fromtimestamp(cb_until, tz=timezone('America/New_York'))
                    reason_raw = data.get(b'last_trip_reason') or data.get('last_trip_reason') or b''
                    reason_str = reason_raw.decode('utf-8') if isinstance(reason_raw, bytes) else str(reason_raw)
                    logger.warning(
                        f"🔥 [SE] 收到 OMS 熔断广播 → 暂停至 {ny.strftime('%H:%M:%S')} "
                        f"(reason={reason_str or 'N/A'})"
                    )
                except Exception:
                    pass
        except Exception as e:
            # 不让熔断同步失败阻塞信号引擎
            if getattr(self, '_cb_sync_err_logged', False) is False:
                logger.warning(f"⚠️ [CB Sync] read meta:circuit_breaker failed: {e}")
                self._cb_sync_err_logged = True


    # ---- archived from signal_engine_v8.py: _sync_symbol_cooldowns_from_redis ----
    def _sync_symbol_cooldowns_from_redis(self):
        """从 Redis 读取 OMS 广播的 per-symbol cooldown_until, 同步到 SE 本地 st.cooldown_until。

        策略:
          - OMS 在 _broadcast_state_to_redis 里只广播仍在未来的冷却窗口 (cd > now),
            所以读到什么就直接 merge 进去。
          - 采用 max(local, remote): 若 SE 因为 from_dict 从 PG 恢复过, 本地可能已有值;
            若本地已过期, remote 是未来的, 取 remote; 若本地是未来的更远, 保留 local。
          - 已不在 remote 里但本地值已过去的 cooldown 不做清理 (反正 curr_ts > cd 时策略自然跳过),
            避免误抹掉刚刚本地生成的冷却。
          - 任何异常不阻塞主流程, 下一 tick 再试。
        """
        try:
            r = getattr(self, 'r', None)
            if r is None:
                return
            data = r.hgetall("meta:symbol_cooldowns")
            if not data:
                return
            updated = 0
            for sym_raw, val_raw in data.items():
                sym = sym_raw.decode('utf-8') if isinstance(sym_raw, bytes) else sym_raw
                if sym not in self.states:
                    continue
                try:
                    remote_cd = float(val_raw.decode('utf-8') if isinstance(val_raw, bytes) else val_raw)
                except (TypeError, ValueError):
                    continue
                st = self.states[sym]
                local_cd = float(getattr(st, 'cooldown_until', 0.0) or 0.0)
                if remote_cd > local_cd:
                    st.cooldown_until = remote_cd
                    updated += 1
            if updated > 0 and not getattr(self, '_cd_sync_first_logged', False):
                logger.info(f"⏳ [SE] 同步 OMS per-symbol cooldown: 更新 {updated} 个标的")
                self._cd_sync_first_logged = True
        except Exception as e:
            if getattr(self, '_cd_sync_err_logged', False) is False:
                logger.warning(f"⚠️ [Cooldown Sync] read meta:symbol_cooldowns failed: {e}")
                self._cd_sync_err_logged = True

    # ------------------------------------------------------------------
    # [Gate Trace Publisher]
    # 将 strategy_core_v0._last_gate_trace 落 Redis 的统一出口。
    # Dashboard 新 tab "🧬 策略门禁" 消费的全部数据均由此方法产出。
    # 约束:
    #   1. 节流 -- 只在 (result_label, last_block_gate) 发生变化时才 HSET,
    #      否则一天 15 标的 × 1Hz 会产生 1M+ 次无意义写入;
    #   2. 永不阻断 -- 任何异常吞掉, Dashboard 空就空, 不能影响策略主循环;
    #   3. 计数器 -- 去抖 INCR: 仅在 (ny_date, last_block_gate) 发生变化时计一次,
    #      避免持续 block 期间每 tick 写放大; 日终自动由 day-boundary 钩子清掉;
    #   4. kind ∈ {"entry", "exit"}, Dashboard 据此分开渲染。
    # ------------------------------------------------------------------


    # ---- archived from signal_engine_v8.py: _maybe_publish_global_gates ----
    def _maybe_publish_global_gates(self, ny_now, curr_ts: float):
        """把全局门禁 G1~G5 聚合成一个快照写到 Redis, Dashboard 顶部横条消费。

        节流: 默认 5s 一次; 熔断切换 / MAX_POSITIONS 翻位等关键变化会立即 force 一次 (TODO, 当前只做节流)。
        写 key: meta:global_gates (hash)
        字段:
          now_ny            - '15:32:10'
          session           - 'pre_open' | 'entry_open' | 'no_entry' | 'close_forced'
          cb_active         - '0' | '1'
          cb_until          - epoch_s
          consecutive_losses/threshold
          positions_used/limit
          exposure_used/limit  (float 0~1)
          updated_at
        """
        now = time.time()
        if now - float(getattr(self, '_last_global_gate_pub_ts', 0.0)) < 5.0:
            return
        self._last_global_gate_pub_ts = now

        # G3/G4 session 判定 (与 strategy_core_v0 _check_entry_pre_conditions 对齐)
        cfg = self.cfg
        t = ny_now.time()
        if t.hour < cfg.START_HOUR or (t.hour == cfg.START_HOUR and t.minute < cfg.START_MINUTE):
            session = 'pre_open'
        elif (t.hour == cfg.CLOSE_HOUR and t.minute >= cfg.CLOSE_MINUTE) or t.hour > cfg.CLOSE_HOUR:
            session = 'close_forced'
        elif (t.hour == cfg.NO_ENTRY_HOUR and t.minute >= cfg.NO_ENTRY_MINUTE) or t.hour > cfg.NO_ENTRY_HOUR:
            session = 'no_entry'
        else:
            session = 'entry_open'

        # G1 熔断: 直接从本地状态读 (与 meta:circuit_breaker 同源)
        cb_until = float(getattr(self, 'global_cooldown_until', 0.0) or 0.0)
        cb_active = 1 if cb_until > curr_ts else 0
        streak = int(getattr(self, 'consecutive_stop_losses', 0) or 0)
        cb_threshold = int(getattr(self, 'CIRCUIT_BREAKER_THRESHOLD',
                                   getattr(cfg, 'CIRCUIT_BREAKER_THRESHOLD', 3)))

        # G2 持仓占用
        positions_used = 0
        try:
            for st in self.states.values():
                if int(getattr(st, 'position', 0) or 0) != 0 or bool(getattr(st, 'is_pending', False)):
                    positions_used += 1
        except Exception:
            pass
        positions_limit = int(getattr(cfg, 'MAX_POSITIONS', 0))

        # G5 exposure: 已用资金 / mock_cash 作为近似 (精确值在 OMS, SE 侧只展示容量层面)
        exposure_used = 0.0
        try:
            mock_cash = float(getattr(self, 'mock_cash', 0.0) or 0.0)
            used = 0.0
            for st in self.states.values():
                ep = float(getattr(st, 'entry_price', 0.0) or 0.0)
                qty = float(getattr(st, 'qty', 0.0) or 0.0)
                if ep > 0 and qty > 0:
                    used += ep * qty * 100.0
            if mock_cash + used > 1.0:
                exposure_used = used / max(1.0, (mock_cash + used))
        except Exception:
            pass
        exposure_limit = float(getattr(cfg, 'GLOBAL_EXPOSURE_LIMIT', 0.0))

        mapping = {
            'now_ny': ny_now.strftime('%H:%M:%S'),
            'session': session,
            'session_open_at': f"{cfg.START_HOUR:02d}:{cfg.START_MINUTE:02d}",
            'session_no_entry_at': f"{cfg.NO_ENTRY_HOUR:02d}:{cfg.NO_ENTRY_MINUTE:02d}",
            'session_close_at': f"{cfg.CLOSE_HOUR:02d}:{cfg.CLOSE_MINUTE:02d}",
            'cb_active': str(cb_active),
            'cb_until': f"{cb_until:.3f}",
            'consecutive_losses': str(streak),
            'cb_threshold': str(cb_threshold),
            'positions_used': str(positions_used),
            'positions_limit': str(positions_limit),
            'exposure_used': f"{exposure_used:.4f}",
            'exposure_limit': f"{exposure_limit:.4f}",
            'updated_at': f"{now:.3f}",
        }
        try:
            pipe = self.r.pipeline()
            pipe.delete("meta:global_gates")
            pipe.hset("meta:global_gates", mapping=mapping)
            pipe.expire("meta:global_gates", 120)
            pipe.execute()
        except Exception as e:
            if not getattr(self, '_gg_err_logged', False):
                logger.warning(f"⚠️ [Global Gates] HSET failed: {e}")
                self._gg_err_logged = True

