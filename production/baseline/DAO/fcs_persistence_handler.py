from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import NY_TZ
from utils import serialization_utils as ser


logger = logging.getLogger("FeatService")


class FCSPersistenceHandler:
    """分钟级落库与同步 ACK 处理器。"""

    def __init__(self, service):
        self.service = service

    def finalize_1min_bar(self, sym, dt, cleanup=True):
        svc = self.service
        bars = svc.current_bars_5s.get(sym, [])

        def gvx(b, k, alt, default=0):
            s_data = b.get('stock', b)
            return float(s_data.get(k, s_data.get(alt, default)))

        valid_ticks = [b for b in bars if gvx(b, 'close', 'c') > 0]
        bar_time = dt.replace(second=0, microsecond=0)

        if not valid_ticks:
            c_raw = svc.last_tick_price.get(sym, 0.0)
            if c_raw == 0:
                c_raw = svc.latest_prices.get(sym, 0.0)
            if c_raw == 0:
                return
            o = h = l = c_raw
            v = 0.0
            vwap = c_raw
        else:
            last_tick = valid_ticks[-1]
            o = gvx(valid_ticks[0], 'open', 'o')
            c_raw = gvx(last_tick, 'close', 'c')
            h = max(max(gvx(b, 'high', 'h', c_raw), gvx(b, 'close', 'c', c_raw)) for b in valid_ticks)
            l = min(min(gvx(b, 'low', 'l', c_raw), gvx(b, 'close', 'c', c_raw)) for b in valid_ticks)
            v = sum(max(0.0, gvx(b, 'volume', 'v', 0.0)) for b in valid_ticks)
            pv_sum = sum(gvx(b, 'close', 'c', c_raw) * max(0.0, gvx(b, 'volume', 'v', 0.0)) for b in valid_ticks)
            vwap = pv_sum / (v + 1e-10) if v > 0 else c_raw

        raw_buckets = getattr(svc, 'frozen_option_snapshot', svc.option_snapshot).get(sym)
        enriched_buckets = getattr(svc, 'frozen_latest_opt_buckets', svc.latest_opt_buckets).get(sym)
        o_contracts = None

        if raw_buckets is None:
            last_opt_tick = next((b for b in reversed(bars) if b.get('buckets') or b.get('option_buckets')), None)
            if last_opt_tick:
                raw_buckets = last_opt_tick.get('buckets', last_opt_tick.get('option_buckets', []))
                if o_contracts is None:
                    o_contracts = last_opt_tick.get('contracts', last_opt_tick.get('option_contracts', []))
        if o_contracts is None:
            o_contracts = getattr(svc, 'frozen_latest_opt_contracts', svc.latest_opt_contracts).get(sym, [])

        merged_buckets = svc._merge_option_snapshot_with_greeks(raw_buckets, enriched_buckets)
        atm_c_iv, atm_p_iv = svc._extract_tagged_atm_iv(merged_buckets)

        ts_key = int(bar_time.timestamp())
        o_buckets = merged_buckets.tolist() if hasattr(merged_buckets, 'tolist') else merged_buckets
        if isinstance(o_contracts, np.ndarray):
            o_contracts = o_contracts.tolist()

        snap_arr = merged_buckets if isinstance(merged_buckets, np.ndarray) else np.array(merged_buckets, dtype=np.float32)
        if isinstance(snap_arr, np.ndarray) and snap_arr.shape == (6, 12):
            has_price_content = np.sum(np.abs(snap_arr[:, [0, 5]])) > 1e-6
            if has_price_content:
                g_sum = np.sum(np.abs(snap_arr[:, 1:5]))
                if g_sum > 0:
                    logger.debug(f"✅ [Greeks-Check] {sym} snapshot contains Greeks (sum={g_sum:.4f})")
                    if sym == 'NVDA' or not hasattr(svc, '_last_sampled_ts') or getattr(svc, '_last_sampled_ts') != ts_key:
                        svc._last_sampled_ts = ts_key
                        p_row = snap_arr[0]
                        logger.info(
                            f"📊 [Data-Sample] {sym} @ {bar_time.strftime('%H:%M:%S')} | "
                            f"ATM_P_K: {p_row[5]:.2f} | P_Price: {p_row[0]:.4f} | Delta: {p_row[1]:.4f} | IV: {p_row[7]:.4f}"
                        )
                else:
                    logger.warning(f"⚠️ [Greeks-Check] {sym} snapshot has DATA but ZERO Greeks. Calculation failed?")

        bar_payload = {'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c_raw), 'volume': float(v), 'vwap': float(vwap)}
        opt_payload = {'buckets': o_buckets, 'contracts': o_contracts, 'atm_c_iv': atm_c_iv, 'atm_p_iv': atm_p_iv}

        success = False
        try:
            svc.r.hset(f"BAR:1M:{sym}", str(ts_key), json.dumps(bar_payload))
            svc.r.hset(f"BAR_OPT:1M:{sym}", str(ts_key), json.dumps(opt_payload))
            success = True
        except Exception:
            pass

        should_full_sync = not success or svc.history_1min.get(sym, pd.DataFrame()).empty
        if not should_full_sync:
            last_ts_loc = svc.history_1min[sym].index[-1].timestamp()
            if ts_key - last_ts_loc > 65:
                should_full_sync = True

        if should_full_sync:
            svc._sync_history_from_redis(sym)
            svc._sync_option_history_from_redis(sym)
        else:
            new_ts = pd.Timestamp(ts_key, unit='s', tz=NY_TZ)
            for k, val in bar_payload.items():
                svc.history_1min[sym].loc[new_ts, k] = val
            if len(svc.history_1min[sym]) > 500:
                svc.history_1min[sym] = svc.history_1min[sym].iloc[-500:]

        df_1m = svc.history_1min[sym]
        if not df_1m.empty:
            df_5m = df_1m.resample('5min', closed='left', label='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            })
            if 'vwap' in df_1m.columns:
                pv = df_1m['vwap'] * df_1m['volume']
                vwap_5m = pv.resample('5min', closed='left', label='left').sum() / (df_5m['volume'] + 1e-10)
                df_5m['vwap'] = np.where(df_5m['volume'] > 0, vwap_5m, df_5m['close'])
            else:
                df_5m['vwap'] = df_5m['close']
            df_5m = df_5m.dropna(subset=['open', 'close'])
            svc.history_5min[sym] = df_5m.iloc[-100:]
            if not df_5m.empty:
                last_5m = df_5m.iloc[-1]
                ts_5m_key = int(df_5m.index[-1].timestamp())
                bar_5m_payload = {
                    'open': float(last_5m['open']), 'high': float(last_5m['high']),
                    'low': float(last_5m['low']), 'close': float(last_5m['close']),
                    'volume': float(last_5m['volume']), 'vwap': float(last_5m['vwap'])
                }
                try:
                    svc.r.hset(f"BAR:5M:{sym}", str(ts_5m_key), json.dumps(bar_5m_payload))
                except Exception:
                    pass

        if cleanup:
            svc.current_bars_5s[sym] = []
        return True

    def finalization_barrier(self):
        svc = self.service
        if not hasattr(svc, 'pending_finalization') or not svc.pending_finalization:
            return
        pending_items = list(svc.pending_finalization.items())
        for sym, dt in pending_items:
            self.finalize_1min_bar(sym, dt, cleanup=True)
            svc.pending_finalization.pop(sym, None)

    def atomic_commit_minute_payload(self, payload: dict) -> bool:
        svc = self.service
        try:
            ts_val = float(payload.get('ts', 0.0))
            frame_id = str(payload.get('frame_id') or str(int(ts_val)))
            svc.publish_frame_seq = int(svc.publish_frame_seq) + 1
            frame_seq = int(payload.get('frame_seq') or svc.publish_frame_seq)
            payload['frame_id'] = frame_id
            payload['frame_seq'] = frame_seq
            payload['frame_complete'] = True
            ts_field = str(int(ts_val))
            symbols = payload.get('symbols', []) or []
            live_options = payload.get('live_options', {}) or {}
            missing_option_syms = []

            packed_payload = ser.pack(payload)
            pipe = svc.r.pipeline(transaction=True)

            for sym in symbols:
                opt = live_options.get(sym, {}) or {}
                buckets = opt.get('buckets', [])
                contracts = opt.get('contracts', [])
                if not buckets:
                    missing_option_syms.append(sym)
                c_iv, p_iv = svc._extract_tagged_atm_iv(buckets)
                opt_payload = {
                    'buckets': buckets.tolist() if isinstance(buckets, np.ndarray) else buckets,
                    'contracts': contracts.tolist() if isinstance(contracts, np.ndarray) else contracts,
                    'atm_c_iv': float(c_iv),
                    'atm_p_iv': float(p_iv)
                }
                pipe.hset(f"BAR_OPT:1M:{sym}", ts_field, json.dumps(opt_payload))

            pipe.xadd(svc.redis_cfg['output_stream'], {'data': packed_payload}, maxlen=100)
            pipe.set("sync:feature_calc_done", str(ts_val))
            pipe.set("sync:feature_calc_done_frame_id", frame_id)
            pipe.expire("sync:feature_calc_done", 120)
            pipe.expire("sync:feature_calc_done_frame_id", 120)
            pipe.hincrby("diag:fcs:counters", "minute_commits", 1)
            pipe.hset("monitor:feature_commit:last", mapping={
                'ts': str(ts_val), 'symbols': str(len(symbols)), 'mode': 'atomic_minute_commit',
                'frame_id': frame_id, 'frame_seq': str(frame_seq)
            })
            pipe.expire("monitor:feature_commit:last", 3600)
            if missing_option_syms:
                pipe.hset("monitor:option_1m:missing_last", mapping={
                    'ts': str(ts_val), 'missing_count': str(len(missing_option_syms)),
                    'missing_sample': json.dumps(missing_option_syms[:8])
                })
                pipe.expire("monitor:option_1m:missing_last", 3600)
            pipe.execute()
            if missing_option_syms:
                logger.warning(
                    f"⚠️ [Option1M-Missing] ts={int(ts_val)} missing={len(missing_option_syms)}/{len(symbols)} sample={missing_option_syms[:5]}"
                )
            return True
        except Exception as e:
            logger.error(f"❌ [Atomic Commit] Failed at ts={payload.get('ts')}, frame_id={payload.get('frame_id')}: {e}")
            logger.warning("⚠️ [Redis SPOF] Atomic commit failed (single-node Redis); downstream consistency may be affected.")
            return False

    def set_feature_sync_ack(self, ts_val: Optional[float], frame_id: Optional[str] = None):
        svc = self.service
        if ts_val is None:
            return
        try:
            svc.r.set("sync:feature_calc_done", str(float(ts_val)))
            svc.r.expire("sync:feature_calc_done", 120)
            if frame_id:
                svc.r.set("sync:feature_calc_done_frame_id", str(frame_id))
                svc.r.expire("sync:feature_calc_done_frame_id", 120)
            svc.r.hincrby("diag:fcs:counters", "sync_ack", 1)
        except Exception:
            pass
