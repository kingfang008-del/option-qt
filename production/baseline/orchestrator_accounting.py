import json
import time
import logging
import os
import pickle
import asyncio
from datetime import datetime
from pytz import timezone
from config import COMMISSION_PER_CONTRACT, COOLDOWN_MINUTES, STREAM_TRADE_LOG, TRADING_ENABLED
import sys
# [NEW] Add project root to sys.path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import serialization_utils as ser

logger = logging.getLogger("V8_Orchestrator.Accounting")

class OrchestratorAccounting:
    def __init__(self, orchestrator):
        self.orch = orchestrator

    @staticmethod
    def _resolve_mode_label(mode_override=None):
        if mode_override:
            return mode_override
        return "BACKTEST" if getattr(os.environ, 'get', None) and os.environ.get('RUN_MODE') == 'LIVEREPLAY' else None

    def _emit_trade_log(self, payload):
        """发送交易日志到 Redis Stream供 Dashboard 展示"""
        try:
            # [NEW] 注入当前账户可用资金
            payload['account_cash'] = round(self.orch.mock_cash, 2)
            
             
            # [🔥 影子交易支持] 如果全局交易开关关闭，且当前处于实盘驱动模式，强行标记为 LIVEREPLAY
            # 这样持久化服务会自动将其路由到 backtest 表，而 Dashboard 也能识别
            if not TRADING_ENABLED and payload.get('mode') != 'BACKTEST':
                payload['mode'] = 'LIVEREPLAY'
            
            # 必须用 ser.pack 序列化以确保全链路兼容 (Msgpack/Pickle 自动识别)
            self.orch.r.xadd(STREAM_TRADE_LOG, {'data': ser.pack(payload)}, maxlen=10000)
        except Exception as e:
            logger.error(f"Failed to emit trade log: {e}")

    def _process_open_accounting(
        self,
        sym,
        st,
        filled_qty,
        fill_price,
        stock_price,
        entry_ts,
        sig,
        duration,
        ratio,
        timing_fields=None,
        mode_override=None,
        note_suffix="",
    ):
        """统一处理 OPEN 成交后的状态回写与日志落库。"""
        if filled_qty <= 0:
            return

        st.position = sig['dir']
        st.qty = filled_qty
        st.entry_stock = stock_price
        st.entry_price = fill_price
        st.entry_ts = entry_ts
        st.opt_type = 'call' if st.position == 1 else 'put'
        st.entry_spy_roc = sig.get('meta', {}).get('spy_roc', 0.0)
        st.entry_index_trend = sig.get('meta', {}).get('index_trend', 0)
        st.entry_alpha_z = sig.get('meta', {}).get('alpha_z', 0.0)
        st.entry_iv = sig.get('meta', {}).get('iv', st.last_valid_iv)
        st.max_roi = 0.0

        timing_fields = timing_fields or {}
        reason = sig.get('reason', '')
        if note_suffix:
            reason = f"{reason}{note_suffix}"

        self._emit_trade_log({
            'ts': entry_ts,
            'symbol': sym,
            'action': 'OPEN',
            'side': 'BUY',
            'qty': filled_qty,
            'price': fill_price,
            'stock_price': stock_price,
            **timing_fields,
            'strategy_note': json.dumps({
                'tag': sig.get('tag'),
                'reason': reason,
                'iv': getattr(st, 'last_valid_iv', 0.0),
                **timing_fields,
            }),
            'fill_duration': round(duration, 1),
            'fill_ratio': round(ratio, 2),
            'mode': mode_override or self.orch.mode.upper(),
        })

    def _process_exit_accounting(self, sym, st, filled_qty, fill_price, stock_price, curr_ts, reason, duration, ratio, original_position=None):
        """[核心新增] 统一财务清算中心：只有发生物理成交才碰钱、算盈亏、写日志"""
        if filled_qty <= 0: return
        
        # 🚀 [Bug2 修复] 优先使用传入的 original_position，因为共享内存下 st.position 可能已被 SE 清零
        pos_for_accounting = original_position if original_position is not None else st.position
        
        commission = filled_qty * COMMISSION_PER_CONTRACT
        proceeds = fill_price * filled_qty * 100 - commission
        self.orch.mock_cash += proceeds
        
        # [Fix] PnL 必须扣除建仓时产生的对应部分手续费，否则 Realized PnL 总是比账户真实回款多！
        open_commission = filled_qty * COMMISSION_PER_CONTRACT
        pnl = proceeds - (st.entry_price * filled_qty * 100) - open_commission
        
        # 逆势统计 - 使用 pos_for_accounting 替代 st.position
        cached_index_trend = getattr(st, 'entry_index_trend', 0)
        is_ct_long = (cached_index_trend == -1) or (st.entry_spy_roc < -0.0001 and pos_for_accounting == 1)
        is_ct_short = (cached_index_trend == 1) or (st.entry_spy_roc > 0.0001 and pos_for_accounting == -1)
        
        if is_ct_long and pos_for_accounting == 1:
            self.orch.stats_counter_trend_long_count += 1
            self.orch.stats_counter_trend_long_pnl += pnl
            if pnl > 0: self.orch.stats_counter_trend_long_win_count += 1
        elif is_ct_short and pos_for_accounting == -1:
            self.orch.stats_counter_trend_short_count += 1
            self.orch.stats_counter_trend_short_pnl += pnl
            if pnl > 0: self.orch.stats_counter_trend_short_win_count += 1
            
        roi = (fill_price - st.entry_price) / st.entry_price if st.entry_price > 0 else 0.0

        # 👇 [🔥 核心路由：获取安全的时间戳]
        is_simulated = self.orch.mode == 'backtest' or os.environ.get('RUN_MODE') == 'LIVEREPLAY'
        safe_now_ts = curr_ts if is_simulated else time.time()
        
        # 冷却与熔断逻辑
        if roi < 0 or "STOP" in reason or "FLIP" in reason:
             st.cooldown_until = safe_now_ts + (COOLDOWN_MINUTES * 60)
        if 'HARD_STOP' in reason or 'COND_STOP' in reason or 'STOCK_STOP' in reason:
            self.orch.consecutive_stop_losses += 1
            if self.orch.consecutive_stop_losses >= self.orch.CIRCUIT_BREAKER_THRESHOLD:
                self.orch.global_cooldown_until = safe_now_ts + (self.orch.CIRCUIT_BREAKER_MINUTES * 60)
                ny_cool = datetime.fromtimestamp(self.orch.global_cooldown_until, tz=timezone('America/New_York'))
                logger.warning(f"🔥 连败熔断触发! 暂停至 {ny_cool.strftime('%H:%M')}")
                self.orch.consecutive_stop_losses = 0
        elif roi >= 0:
            self.orch.consecutive_stop_losses = 0
            
        # 写日志
        self._emit_trade_log({
            'ts': safe_now_ts ,
            'symbol': sym, 'action': 'CLOSE', 'side': 'SELL',
            'qty': filled_qty, 'price': fill_price, 'stock_price': stock_price,
            'entry_stock': st.entry_stock, 'pnl': pnl, 'roi': roi,
            'strategy_note': json.dumps({'reason': reason, 'duration': (safe_now_ts - st.entry_ts)/60}),
            'fill_duration': round(duration, 1), 'fill_ratio': round(ratio, 2), 'mode': self.orch.mode.upper()
        })
        
        logger.info(f"🛑 CLOSE {sym} {reason} | PnL: ${pnl:.0f} ({roi:.1%}) | 成交: {filled_qty}手")
        
        # [Fix] gross_pnl 同样需要扣除买卖双向手续费以反应真实情况
        gross_pnl = (fill_price - st.entry_price) * filled_qty * 100 - commission - open_commission
        gross_roi = gross_pnl / ((st.entry_price * filled_qty * 100) + open_commission) if st.entry_price > 0 else 0

        # 🔍 [AUDIT LOG] 精准审计日志，用于对齐 MockIBKR
        logger.info(f"📊 [PnL AUDIT] {sym} | Entry: {st.entry_price:.2f} | Exit: {fill_price:.2f} | Qty: {filled_qty} | PnL: ${gross_pnl:.2f} | Net Accum: ${self.orch.realized_pnl + gross_pnl:.2f}")
        
        # 📈 [更新 Ground Truth 统计]
        self.orch.realized_pnl += gross_pnl
        self.orch.total_commission += (commission + open_commission)
        self.orch.trade_count += 1
        if gross_pnl > 0: self.orch.win_count += 1
        elif gross_pnl < 0: self.orch.loss_count += 1

        self.orch.daily_trades.append({
            'symbol': sym, 'opt_type': st.opt_type, 'position': pos_for_accounting,
            'entry_ts': st.entry_ts, 'exit_ts': safe_now_ts,
            'duration_min': (safe_now_ts - st.entry_ts) / 60.0,
            'qty': filled_qty, 'entry_price': st.entry_price, 'exit_price': fill_price,
            'entry_stock': st.entry_stock, 'exit_stock': stock_price, 'pnl': gross_pnl, 'roi': gross_roi,
            'spy_roc': st.entry_spy_roc, 'alpha_z': getattr(st, 'entry_alpha_z', 0.0),
            'entry_iv': getattr(st, 'entry_iv', st.last_valid_iv), 'reason': reason
        })
        
        # 扣减仓位
        st.qty -= filled_qty
        if st.qty <= 0:
            st.position = 0
            st.qty = 0
            st.entry_price = 0.0
            st.entry_ts = 0.0
            st.max_roi = 0.0

    def _generate_daily_analysis_report(self, report_date_str: str = None):
        """盘后诊断报告"""
        if not self.orch.daily_trades:
            logger.info("ℹ️ 今日无交易记录，跳过盘后诊断报告生成。")
            return
            
        if report_date_str:
            today_str = report_date_str
        else:
            try:
                from config import NY_TZ
                today_str = datetime.now(NY_TZ).strftime('%Y%m%d')
            except:
                today_str = datetime.now().strftime('%Y%m%d')
            
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        report_path = os.path.join(log_dir, f"daily_analysis_{today_str}.md")
        
        total_trades = len(self.orch.daily_trades)
        winning_trades = [t for t in self.orch.daily_trades if t['pnl'] > 0]
        losing_trades = [t for t in self.orch.daily_trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t['pnl'] for t in self.orch.daily_trades)
        
        best_trade = max(self.orch.daily_trades, key=lambda x: x['pnl']) if winning_trades else None
        worst_trade = min(self.orch.daily_trades, key=lambda x: x['pnl']) if losing_trades else None
        
        recommendations = []
        if losing_trades:
            loss_count = len(losing_trades)
            # 1. Counter-Trend
            counter_trend_losses = [t for t in losing_trades if (t['spy_roc'] < -0.0001 and t['position'] == 1) or (t['spy_roc'] > 0.0001 and t['position'] == -1)]
            if len(counter_trend_losses) / loss_count >= 0.5:
                recommendations.append("📈 **病因：逆势摸顶/抄底被打爆**\n...")
            
            # 2. Fast Stop Loss
            fast_stop_losses = [t for t in losing_trades if "FAST_STOP_LOSS" in t['reason']]
            if len(fast_stop_losses) / loss_count >= 0.4:
                recommendations.append("📉 **病因：高频极速止损锯齿割肉**\n...")
                                    
            # 3. Liquidity Drought
            bsm_liquidations = [t for t in losing_trades if t['exit_price'] <= 0.01 or self.orch.stats_liquidity_drought_liquidations > 0]
            if len(bsm_liquidations) > 0 and (len(bsm_liquidations) / loss_count >= 0.2):
                recommendations.append("⚠️ **病因：高频“假盘口”踩踏 (BSM Liquidity Drought)**\n...")
                                    
            # 4. EOD Timeout
            eod_timeouts = [t for t in losing_trades if "EOD_CLEAR" in t['reason']]
            if len(eod_timeouts) / loss_count >= 0.4:
                recommendations.append("🐢 **病因：到期前自然亏损/超时熬盘 (Time Decay)**\n...")

        if not recommendations:
            if win_rate > 0.6 and total_pnl > 0:
                recommendations.append("✅ **资金运转健康**\n...")
            else:
                recommendations.append("🔍 **无明显单一病因**\n...")

        lines = [
            f"# 📊 每日交易盘后诊断报告 | {today_str}",
            "",
            "## 1. 业绩摘要 (Performance Summary)",
            f"- **总交易笔数**: {total_trades}",
            f"- **胜率 (Win Rate)**: {win_rate:.1%}",
            f"- **全天净盈亏 (Net PnL)**: **${total_pnl:,.2f}**",
            "",
            "### 高光/低谷时刻",
            f"- **今日最佳交易**: {best_trade['symbol']} ({best_trade['opt_type'].upper()}) | 盈亏: +${best_trade['pnl']:.2f} | 耗时: {best_trade['duration_min']:.1f} 分钟 | 原因: {best_trade['reason']}" if best_trade else "- **今日最佳交易**: 无",
            f"- **今日最惨交易**: {worst_trade['symbol']} ({worst_trade['opt_type'].upper()}) | 盈亏: -${abs(worst_trade['pnl']):.2f} | 耗时: {worst_trade['duration_min']:.1f} 分钟 | 原因: {worst_trade['reason']}" if worst_trade else "- **今日最惨交易**: 无",
            "",
            "## 2. 诊断建议 (Algorithmic Recommendations)",
        ]
        
        for r in recommendations:
            lines.append(f"> {r}")
            lines.append("")
            
        lines.extend([
            "---",
            "## 3. 亏损明细账单 (Losing Trades Breakdown)"
        ])
        
        if not losing_trades:
            lines.append("*今日无亏损单，完胜！*")
        else:
            lines.append("| 标的 | 方向 | 盈亏 ($) | 耗时 (Min) | 强平原因 | 进场大盘势能 (SPY ROC) |")
            lines.append("|---|---|---|---|---|---|")
            for t in sorted(losing_trades, key=lambda x: x['pnl']):
                lines.append(f"| {t['symbol']} | {t['opt_type'].upper()} | <span style='color:red'>{t['pnl']:.2f}</span> | {t['duration_min']:.1f} | {t['reason']} | {t['spy_roc']:.4f} |")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info(f"✅ 每日盘后诊断报告已生成: {report_path}")
        except Exception as e:
            logger.error(f"❌ 生成每日盘后报告失败: {e}")
            
        self.orch.daily_trades.clear()

    async def _pnl_monitor_loop(self):
        """[🔥 新增] 物理时间监视器 (仅限实盘或空闲状态)"""
        logger.info("📡 [PnL Monitor] 物理时间监视器已启动。")
        while True:
            try:
                await asyncio.sleep(60) # 每物理分钟核对一次
                # 如果是实盘模式，且距离上次逻辑报告已经超过 2 分钟物理时间，则补充一个物理报告
                if self.orch.mode == 'realtime' and time.time() - self.orch.last_save_time > 120:
                     await self._report_pnl_status_logic(time.time(), "WALL-CLOCK")
            except Exception as e:
                logger.error(f"❌ PnL Physical Monitor Error: {e}")

    async def _report_pnl_status_logic(self, timestamp, label="Summary"):
        """[🔥 核心] 统一盈亏汇总打印逻辑"""
        try:
            unrealized = 0.0
            active_count = 0
            for sym, st in self.orch.symbol_states.items():
                if st.position != 0 and st.last_opt_price > 0:
                    curr_price = st.last_opt_price
                    unrealized += (curr_price - st.entry_price) * st.qty * 100
                    active_count += 1
            
            total_net = self.orch.realized_pnl + unrealized
            win_rate = (self.orch.win_count / self.orch.trade_count) if self.orch.trade_count > 0 else 0.0
            
            ny_tz = timezone('America/New_York')
            ny_time = datetime.fromtimestamp(timestamp, ny_tz).strftime('%H:%M:%S')
            
            logger.info(f"📈 [PnL {label}] {ny_time} | Net: ${total_net:+,.2f} | Realized: ${self.orch.realized_pnl:+,.2f} | Unreal: ${unrealized:+,.2f} | WR: {win_rate:.1%} ({self.orch.win_count}W/{self.orch.loss_count}L) | Pos: {active_count}")
        except Exception as e:
            logger.error(f"❌ Report PnL Failed: {e}")

    def print_counter_trend_summary(self):
        long_wr = (self.orch.stats_counter_trend_long_win_count / self.orch.stats_counter_trend_long_count) * 100 if self.orch.stats_counter_trend_long_count > 0 else 0
        short_wr = (self.orch.stats_counter_trend_short_win_count / self.orch.stats_counter_trend_short_count) * 100 if self.orch.stats_counter_trend_short_count > 0 else 0
        
        print("\n" + "="*50)
        print("📊 [Counter-Trend Stats] 逆势交易统计结果")
        print("="*50)
        print(f"🔸 下跌趋势中强制做多 (CALL) 拦截失效逃逸数: {self.orch.stats_counter_trend_long_count} 笔 (胜率: {long_wr:.1f}%)")
        print(f"   => 累计无厘头做多盈亏: ${self.orch.stats_counter_trend_long_pnl:.2f}")
        print(f"🔹 上升趋势中违背大盘做空 (PUT) 次数: {self.orch.stats_counter_trend_short_count} 笔 (胜率: {short_wr:.1f}%)")
        print(f"   => 累计顶风做空盈亏: ${self.orch.stats_counter_trend_short_pnl:.2f}")
        print("-" * 50)
        print(f"⚠️ [极端情况监控] 盘口完全断流被迫使用 BSM 强平次数: {self.orch.stats_liquidity_drought_liquidations} 次")
        print("="*50 + "\n")

    def print_backtest_summary(self):
        """[NEW] 最终回测财务汇总报告 (Orchestrator Internal View)"""
        print("\n" + "🏆" * 30)
        print("     V8 ORCHESTRATOR FINAL BACKTEST SUMMARY")
        print("🏆" * 30)
        
        win_rate = (self.orch.win_count / self.orch.trade_count * 100) if self.orch.trade_count > 0 else 0
        
        # 计算平均盈亏
        avg_pnl = (self.orch.realized_pnl / self.orch.trade_count) if self.orch.trade_count > 0 else 0
        
        print(f"💰 总实现盈亏 (Net realized):  ${self.orch.realized_pnl:+,.2f}")
        print(f"🧾 总手续费覆盖 (Total Comm):  ${self.orch.total_commission:,.2f}")
        print(f"📈 交易总笔数 (Total Trades):  {self.orch.trade_count}")
        print(f"🎯 胜率 (Win Rate):           {win_rate:.2f}%")
        print(f"秤️  平均每笔盈亏 (Avg PnL):    $${avg_pnl:+,.2f}")
        print(f"📅 交易记录已保存至:           logs/replay_trades_v8.csv")
        print("=" * 60 + "\n")
