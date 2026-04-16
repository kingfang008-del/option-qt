from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from config import NY_TZ, USE_5M_OPTION_DATA


logger = logging.getLogger("FeatService")


class FCSRealtimePipeline:
    """实时计算 -> 推理发布流程处理器。"""

    def __init__(self, service):
        self.service = service

    @staticmethod
    def _coerce_payload_ts(raw_ts, _depth=0):
        """容错解析 payload.ts，避免上游异常结构导致流程中断。"""
        if _depth > 8:
            raise TypeError("invalid payload ts type: max recursion depth exceeded")
        if isinstance(raw_ts, (int, float)):
            return float(raw_ts)
        if isinstance(raw_ts, str):
            return float(raw_ts)
        if isinstance(raw_ts, (list, tuple)) and raw_ts:
            # 常见包装: [ts] / [ {"ts": ...} ]
            return FCSRealtimePipeline._coerce_payload_ts(raw_ts[0], _depth=_depth + 1)
        if isinstance(raw_ts, dict):
            # 先匹配常见键
            for k in ("ts", "value", "v", "time", "timestamp"):
                if k in raw_ts:
                    return FCSRealtimePipeline._coerce_payload_ts(raw_ts[k], _depth=_depth + 1)
            # 再匹配常见序列化键（如 Mongo Extended JSON）
            for k in ("$numberLong", "$numberInt", "$numberDouble"):
                if k in raw_ts:
                    return FCSRealtimePipeline._coerce_payload_ts(raw_ts[k], _depth=_depth + 1)
            # 单键 dict 兜底
            if len(raw_ts) == 1:
                return FCSRealtimePipeline._coerce_payload_ts(next(iter(raw_ts.values())), _depth=_depth + 1)
            # 遍历 values 深度兜底
            for v in raw_ts.values():
                try:
                    return FCSRealtimePipeline._coerce_payload_ts(v, _depth=_depth + 1)
                except Exception:
                    continue
        raise TypeError(f"invalid payload ts type: {type(raw_ts).__name__}")

    async def process_market_data(self, batch, current_replay_ts=None, return_payload=False):
        svc = self.service
        svc.msg_count += 1
        try:
            for payload in batch:
                sym = payload.get('symbol')
                if not sym:
                    continue

                raw_ts = payload.get('ts')
                try:
                    ts = self._coerce_payload_ts(raw_ts)
                except Exception:
                    # 容错降级：优先用 replay 主时钟，其次用当前墙钟，避免整批中断。
                    ts = float(current_replay_ts) if current_replay_ts is not None else time.time()
                    bad_ts_log_count = getattr(svc, "_bad_ts_log_count", 0)
                    if bad_ts_log_count < 20:
                        logger.warning(
                            "⚠️ [FCS-TS-Coerce] invalid payload.ts type=%s; fallback_ts=%.3f; sample=%r",
                            type(raw_ts).__name__,
                            float(ts),
                            raw_ts,
                        )
                    svc._bad_ts_log_count = bad_ts_log_count + 1
                dt_utc = datetime.fromtimestamp(ts, timezone.utc)
                dt_ny = dt_utc.astimezone(NY_TZ)
                curr_minute = dt_ny.replace(second=0, microsecond=0)
                is_preaggregated_1m = bool(payload.get('bar_preaggregated_1m', False))
                if is_preaggregated_1m:
                    svc.preaggregated_1m_mode = True
                    if not svc._preagg_mode_logged:
                        logger.info("🧭 [Mode] Pre-aggregated 1m feed detected: bypass tick->bar finalize path.")
                        svc._preagg_mode_logged = True
                elif bool(getattr(svc, 'preaggregated_1m_mode', False)):
                    # 防止一次误标记把服务永久锁到 preagg 模式，造成分钟标签提前与断档。
                    svc.preaggregated_1m_mode = False
                    svc._preagg_mode_logged = False
                    logger.warning("⚠️ [Mode] Switched back to tick/minute finalize mode (bar_preaggregated_1m absent).")

                if sym in svc.symbols:
                    svc._update_option_frame_state(sym, payload, ts)

                if not is_preaggregated_1m:
                    if not hasattr(svc, 'global_last_minute'):
                        svc.global_last_minute = curr_minute
                    if curr_minute > svc.global_last_minute:
                        while svc.global_last_minute < curr_minute:
                            for s in svc.symbols:
                                if not hasattr(svc, 'frozen_option_snapshot'):
                                    svc.frozen_option_snapshot = {}
                                    svc.frozen_option_snapshot_5m = {}
                                    svc.frozen_latest_opt_buckets = {}
                                    svc.frozen_latest_opt_contracts = {}
                                agg_snap, agg_contracts, agg_update_ts = svc.option_minute_agg.finalize(s, svc.global_last_minute)
                                current_snap = agg_snap if agg_snap is not None else svc.option_snapshot.get(s)
                                svc.frozen_option_snapshot[s] = current_snap.copy() if current_snap is not None else np.zeros((6, 12), dtype=np.float32)
                                if USE_5M_OPTION_DATA:
                                    current_snap_5m = getattr(svc, 'option_snapshot_5m', {}).get(s, current_snap)
                                    svc.frozen_option_snapshot_5m[s] = current_snap_5m.copy() if current_snap_5m is not None else np.zeros((6, 12), dtype=np.float32)
                                svc.frozen_latest_opt_buckets[s] = current_snap.copy() if current_snap is not None else svc.latest_opt_buckets.get(s, [])
                                svc.frozen_latest_opt_contracts[s] = agg_contracts if agg_contracts else svc.latest_opt_contracts.get(s, [])
                                if agg_update_ts is not None:
                                    svc.last_option_update_ts[s] = float(agg_update_ts)
                                svc.pending_finalization[s] = svc.global_last_minute
                            svc.global_last_minute += timedelta(minutes=1)

                b_raw = payload.get('buckets', payload.get('option_buckets'))
                b_data = b_raw.get('buckets', []) if isinstance(b_raw, dict) else b_raw
                # 仅在“尚无补算结果”时接受原始快照，避免每秒原始 payload 覆盖 minute-end Greeks 结果。
                has_cached_enriched = sym in svc.latest_opt_buckets and svc.latest_opt_buckets.get(sym) is not None
                if not has_cached_enriched:
                    if isinstance(b_data, np.ndarray):
                        if b_data.size > 0:
                            svc.latest_opt_buckets[sym] = b_data
                    elif isinstance(b_data, list) and len(b_data) > 0:
                        svc.latest_opt_buckets[sym] = b_data

                c_raw = payload.get('contracts', payload.get('option_contracts'))
                c_data = c_raw.get('contracts', []) if isinstance(c_raw, dict) else c_raw
                if isinstance(c_data, np.ndarray):
                    c_data = c_data.tolist()
                if isinstance(c_data, list) and len(c_data) > 0:
                    svc.latest_opt_contracts[sym] = c_data

                if sym not in svc.symbols:
                    continue
                svc._last_ingest_meta = {
                    'symbol': sym,
                    'payload_ts': float(ts),
                    'frame_id': payload.get('frame_id'),
                    'seq': payload.get('seq'),
                    'preaggregated_1m': bool(payload.get('bar_preaggregated_1m', False)),
                    'stream': payload.get('_fcs_stream_name'),
                    'stream_msg_id': payload.get('_fcs_stream_msg_id'),
                    'recv_wall_ts': payload.get('_fcs_recv_wall_ts'),
                }

                profile = getattr(svc, "market_profile", None)
                if profile is not None and not profile.accept_realtime_tick(dt_ny):
                    continue
                if profile is not None and profile.should_flush_premarket(dt_ny, getattr(svc, '_premarket_flushed_date', None)):
                        for s in svc.symbols:
                            df = svc.history_1min.get(s, pd.DataFrame())
                            if not df.empty:
                                idx = df.index
                                if getattr(idx, "tz", None) is None:
                                    idx = idx.tz_localize(NY_TZ)
                                mask = profile.history_keep_mask(idx, dt_ny)
                                svc.history_1min[s] = df[mask].sort_index()
                        svc._premarket_flushed_date = dt_ny.date()
                        logger.info("🧹 Session pre-open cleanup completed.")

                stock = payload.get('stock', {})
                if stock:
                    c_val = float(stock.get('close', stock.get('c', 0.0)))
                    if c_val > 0:
                        svc.latest_prices[sym] = c_val
                        svc.last_tick_price[sym] = c_val
                        svc.last_stock_update_ts[sym] = float(ts)
                    if is_preaggregated_1m:
                        o_val = float(stock.get('open', stock.get('o', c_val)))
                        h_val = float(stock.get('high', stock.get('h', c_val)))
                        l_val = float(stock.get('low', stock.get('l', c_val)))
                        v_val = float(stock.get('volume', stock.get('v', 0.0)))
                        vw_val = float(stock.get('vwap', c_val))
                        svc.history_1min[sym].loc[curr_minute, ['open', 'high', 'low', 'close', 'volume', 'vwap']] = [o_val, h_val, l_val, c_val, v_val, vw_val]
                        if len(svc.history_1min[sym]) > svc.HISTORY_LEN:
                            svc.history_1min[sym] = svc.history_1min[sym].iloc[-svc.HISTORY_LEN:]

                stock_5m = payload.get('stock_5m')
                if stock_5m and dt_ny.minute % 5 == 0:
                    o, h, l, c, v = stock_5m['open'], stock_5m['high'], stock_5m['low'], stock_5m['close'], stock_5m['volume']
                    vw = stock_5m.get('vwap', c)
                    svc.history_5min[sym].loc[curr_minute, ['open', 'high', 'low', 'close', 'volume', 'vwap']] = [o, h, l, c, v, vw]
                    if len(svc.history_5min[sym]) > 100:
                        svc.history_5min[sym] = svc.history_5min[sym].iloc[-100:]

                buckets = payload.get('buckets', payload.get('option_buckets'))
                has_buckets = isinstance(buckets, np.ndarray) and buckets.size > 0 or isinstance(buckets, list) and len(buckets) > 0
                if has_buckets:
                    arr = np.array(buckets, dtype=np.float32)
                    if arr.shape[1] < 12:
                        arr = np.hstack([arr, np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)])
                    if arr.shape[0] < 6:
                        arr = np.vstack([arr, np.zeros((6 - arr.shape[0], 12), dtype=np.float32)])
                    valid_quote_mask = (arr[:, 8] > 0) & (arr[:, 9] > 0)
                    arr[valid_quote_mask, 0] = (arr[valid_quote_mask, 8] + arr[valid_quote_mask, 9]) / 2.0
                    if np.sum(arr[:, 0]) > 0.001 or np.sum(arr[:, 6]) > 0.001:
                        arr[:, 6] = np.maximum(arr[:, 6], 0.0)
                        svc.option_snapshot[sym] = arr
                        svc.last_option_update_ts[sym] = float(ts)
                        if not is_preaggregated_1m:
                            svc.option_minute_agg.update(sym, curr_minute, arr, c_data, update_ts=ts)
                        if USE_5M_OPTION_DATA:
                            if not hasattr(svc, 'option_snapshot_5m'):
                                svc.option_snapshot_5m = {}
                            svc.option_snapshot_5m[sym] = arr.copy()
                        if is_preaggregated_1m:
                            if not hasattr(svc, 'frozen_option_snapshot'):
                                svc.frozen_option_snapshot = {}
                                svc.frozen_option_snapshot_5m = {}
                                svc.frozen_latest_opt_buckets = {}
                                svc.frozen_latest_opt_contracts = {}
                            svc.frozen_option_snapshot[sym] = arr.copy()
                            svc.frozen_latest_opt_buckets[sym] = arr.copy()
                            svc.frozen_latest_opt_contracts[sym] = list(c_data) if c_data else []
                        svc.warmup_needed[sym] = False

                if USE_5M_OPTION_DATA:
                    buckets_5m = payload.get('buckets_5m', payload.get('option_buckets_5m'))
                    if (not buckets_5m or len(buckets_5m) == 0) and dt_ny.minute % 5 == 0:
                        buckets_5m = payload.get('buckets', payload.get('option_buckets'))
                    if buckets_5m and len(buckets_5m) > 0 and dt_ny.minute % 5 == 0:
                        arr = np.array(buckets_5m, dtype=np.float32)
                        if arr.shape[1] < 12:
                            arr = np.hstack([arr, np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)])
                        if arr.shape[0] < 6:
                            arr = np.vstack([arr, np.zeros((6 - arr.shape[0], 12), dtype=np.float32)])
                        valid_quote_mask_5m = (arr[:, 8] > 0) & (arr[:, 9] > 0)
                        arr[valid_quote_mask_5m, 0] = (arr[valid_quote_mask_5m, 8] + arr[valid_quote_mask_5m, 9]) / 2.0
                        if np.sum(arr[:, 6]) > 0.0001:
                            svc.last_cum_volume_5m[sym] = arr[:, 6].copy()
                            if not hasattr(svc, 'option_snapshot_5m'):
                                svc.option_snapshot_5m = {}
                            svc.option_snapshot_5m[sym] = arr
                            if is_preaggregated_1m and hasattr(svc, 'frozen_option_snapshot_5m'):
                                svc.frozen_option_snapshot_5m[sym] = arr.copy()

                stock_close = float(stock.get('close', stock.get('c', 0.0))) if stock else 0.0
                if (not is_preaggregated_1m) and stock_close > 0:
                    svc.current_bars_5s[sym].append(stock)
                if (not is_preaggregated_1m) and dt_ny.second >= 55:
                    svc._finalize_1min_bar(sym, curr_minute, cleanup=False)

        except Exception as e:
            logger.error(f"Process Error: {e}", exc_info=True)

    async def run_compute_cycle(self, ts_from_payload=None, return_payload=False):
        svc = self.service
        current_replay_ts = ts_from_payload if ts_from_payload else svc.r.get("replay:current_ts")
        sync_ts = float(current_replay_ts) if current_replay_ts else time.time()
        run_mode = os.environ.get("RUN_MODE", "").strip().upper()
        if ts_from_payload is not None and run_mode in {"REALTIME", "REALTIME_DRY"}:
            wall_ts = time.time()
            max_lead = float(os.environ.get("FCS_MAX_TS_LEAD_SEC", "1.5"))
            if (sync_ts - wall_ts) > max_lead:
                ingest_meta = getattr(svc, '_last_ingest_meta', {}) or {}
                logger.warning(
                    f"⚠️ [FCS-TimeGuard] payload ts leads wall-clock by {sync_ts - wall_ts:.3f}s; "
                    f"clamp sync_ts from {sync_ts:.3f} to {wall_ts:.3f} | "
                    f"source=sym={ingest_meta.get('symbol')} frame_id={ingest_meta.get('frame_id')} "
                    f"seq={ingest_meta.get('seq')} preagg={ingest_meta.get('preaggregated_1m')} "
                    f"stream={ingest_meta.get('stream')} msg_id={ingest_meta.get('stream_msg_id')} "
                    f"payload_ts={ingest_meta.get('payload_ts')} recv_wall_ts={ingest_meta.get('recv_wall_ts')}"
                )
                sync_ts = wall_ts
        target_time = datetime.fromtimestamp(sync_ts, NY_TZ)
        current_minute_ts = int(sync_ts // 60) * 60
        curr_minute_dt = target_time.replace(second=0, microsecond=0)

        if not bool(getattr(svc, 'preaggregated_1m_mode', False)):
            if not hasattr(svc, 'global_last_minute'):
                svc.global_last_minute = curr_minute_dt
            if svc.global_last_minute < curr_minute_dt:
                while svc.global_last_minute < curr_minute_dt:
                    for s in svc.symbols:
                        if not hasattr(svc, 'frozen_option_snapshot'):
                            svc.frozen_option_snapshot = {}
                            svc.frozen_option_snapshot_5m = {}
                            svc.frozen_latest_opt_buckets = {}
                            svc.frozen_latest_opt_contracts = {}
                        current_snap = svc.option_snapshot.get(s)
                        svc.frozen_option_snapshot[s] = current_snap.copy() if current_snap is not None else np.zeros((6, 12), dtype=np.float32)
                        if USE_5M_OPTION_DATA:
                            current_snap_5m = getattr(svc, 'option_snapshot_5m', {}).get(s, current_snap)
                            svc.frozen_option_snapshot_5m[s] = current_snap_5m.copy() if current_snap_5m is not None else np.zeros((6, 12), dtype=np.float32)
                        svc.frozen_latest_opt_buckets[s] = svc.latest_opt_buckets.get(s, [])
                        svc.frozen_latest_opt_contracts[s] = svc.latest_opt_contracts.get(s, [])
                        svc.pending_finalization[s] = svc.global_last_minute
                    svc.global_last_minute += timedelta(minutes=1)
        else:
            svc.global_last_minute = curr_minute_dt

        ready_symbols = [s for s in svc.symbols if not svc.history_1min.get(s, pd.DataFrame()).empty]
        if not ready_symbols:
            svc._set_feature_sync_ack(sync_ts, frame_id=str(int(sync_ts)))
            return None
        sample_s = ready_symbols[0]
        if len(svc.history_1min[sample_s]) < 1:
            svc._set_feature_sync_ack(sync_ts, frame_id=str(int(sync_ts)))
            return None

        data_ts = float(current_replay_ts) if current_replay_ts else target_time.timestamp()
        last_minute_ts = getattr(svc, 'last_model_minute_ts', 0)
        preagg_mode = bool(getattr(svc, 'preaggregated_1m_mode', False))
        is_new_minute = False
        if preagg_mode:
            if last_minute_ts == 0:
                # 首帧仅建锚点，避免 preagg 模式冷启动时把“当前分钟”提前写入标签。
                svc.last_model_minute_ts = current_minute_ts
                svc._set_feature_sync_ack(data_ts, frame_id=str(int(data_ts)))
                return None
            elif current_minute_ts > last_minute_ts:
                is_new_minute = True
                svc.last_model_minute_ts = current_minute_ts
        else:
            if last_minute_ts == 0:
                svc.last_model_minute_ts = current_minute_ts
                svc._set_feature_sync_ack(data_ts, frame_id=str(int(data_ts)))
                return None
            if current_minute_ts > last_minute_ts:
                is_new_minute = True
                svc.last_model_minute_ts = current_minute_ts

        alpha_label_ts = float(current_minute_ts) if preagg_mode and is_new_minute else ((float(current_minute_ts) - 60.0) if is_new_minute else data_ts)
        step_res = svc._step_engine_compute(data_ts=data_ts, is_new_minute=is_new_minute, alpha_label_ts=alpha_label_ts)
        if not step_res:
            svc._set_feature_sync_ack(data_ts, frame_id=str(int(data_ts)))
            return None
        batch_raw, valid_mask, results_map = step_res
        if batch_raw is None:
            svc._set_feature_sync_ack(data_ts, frame_id=str(int(data_ts)))
            return None

        norm_seq_30 = svc._apply_normalization_sequence(batch_raw=batch_raw, valid_mask=valid_mask, data_ts=data_ts, is_new_minute=is_new_minute)
        payload = svc._assemble_compute_payload(
            norm_seq_30=norm_seq_30, batch_raw=batch_raw, valid_mask=valid_mask, results_map=results_map,
            alpha_label_ts=alpha_label_ts, data_ts=data_ts, is_new_minute=is_new_minute, ready_symbols=ready_symbols
        )
        if not payload:
            svc._set_feature_sync_ack(data_ts, frame_id=str(int(data_ts)))
            return None

        if payload.get('is_new_minute') and (not bool(payload.get('is_warmed_up', False))):
            if getattr(svc, '_warmup_diag_log_count', 0) < 20:
                logger.info(
                    f"⏳ [Warmup-Diag] ts={int(payload.get('ts', 0))} "
                    f"| real_history_len={payload.get('real_history_len', 0)}/{payload.get('warmup_required_len', 31)} "
                    f"| total_history_len={payload.get('total_history_len', 0)} "
                    f"| real_norm_history_len={payload.get('real_norm_history_len', 0)} "
                    f"| cross_day={payload.get('has_cross_day_warmup', False)} "
                    f"| symbols={len(payload.get('symbols', []))}"
                )
                svc._warmup_diag_log_count = getattr(svc, '_warmup_diag_log_count', 0) + 1

        if payload.get('is_new_minute'):
            ok = svc._atomic_commit_minute_payload(payload)
            if not ok:
                svc._set_feature_sync_ack(data_ts, frame_id=str(int(data_ts)))
        else:
            svc._set_feature_sync_ack(data_ts, frame_id=str(int(data_ts)))

        if return_payload:
            return payload
        return None
