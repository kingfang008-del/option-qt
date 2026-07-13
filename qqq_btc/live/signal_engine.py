#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实盘信号引擎 —— FCS 1min bar close → 模型推理 → 与 replay 同口径决策。

状态机与 strict/event replay 共用 ReplaySession,保证实盘 = 回放同一实现。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from qqq_btc.common.replay_session import ReplaySession, SessionQuotes, SessionSignal
from qqq_btc.common.session_history import (
    DEFAULT_CARRYOVER_BARS,
    prepend_carryover,
    session_tail,
)
from qqq_btc.common.time_features import session_minute
from qqq_btc.live.fcs_adapter import enrich_fcs_bars
from qqq_btc.model.backbone import DualStreamAlphaNet, resolve_embedding_caps
from qqq_btc.qqq import config as qcfg
from qqq_btc.tools.run_inference import build_feature_maps, row_to_tensors

logger = logging.getLogger("qqq_btc.live.signal")


class LiveSignalEngine:
    def __init__(
        self,
        checkpoint: str | Path,
        config_path: str | Path = qcfg.FEATURE_CONFIG_PATH,
        symbol: str = "QQQ",
        stock_id: int = 1,
        sector_id: int = 1,
        device: str = "auto",
        carryover_bars: int = DEFAULT_CARRYOVER_BARS,
    ):
        self.symbol = symbol
        self.stock_id = stock_id
        self.sector_id = sector_id
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        caps = resolve_embedding_caps(self.config)
        self.model = DualStreamAlphaNet(self.config, caps).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict"):
                if key in ckpt:
                    ckpt = ckpt[key]
                    break
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        self.stock_map, self.option_map, self.n_stock, self.n_opt = build_feature_maps(self.config)

        self.replay_cfg = qcfg.LIVE_REPLAY
        self.rails_cfg = qcfg.EXIT_RAILS
        self.fill_model = qcfg.FILL_MODEL
        self.dual_mode = True  # 推理始终产出全部头;是否启用由 replay_cfg.long_only 控制

        self.session = ReplaySession(
            self.replay_cfg,
            self.rails_cfg,
            self.fill_model,
            dual_mode=True,
            default_leg="CALL",
            is_option=True,
        )
        self.bar_index = 0
        self.carryover_bars = int(carryover_bars)
        self._carryover_buffer: Optional[pd.DataFrame] = None

    def set_session_carryover(self, prior_session_df: pd.DataFrame) -> None:
        """注入前日 RTH tail(通常昨日最后 29 根 1min bar),供次日 09:30 满 seq_len。"""
        self._carryover_buffer = session_tail(prior_session_df, self.carryover_bars)

    def snapshot_carryover_from_history(self, history_df: pd.DataFrame) -> None:
        """收盘后调用:从当日 history 提取 tail 供下一交易日。"""
        self.set_session_carryover(history_df)

    def _prepare_history(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """当日 history 前拼 carryover(若已设置);每次 bar 调用,避免仅首 bar 拼接。"""
        df = history_df.sort_values("timestamp").reset_index(drop=True)
        if self._carryover_buffer is not None and not self._carryover_buffer.empty:
            df = prepend_carryover(df, self._carryover_buffer, carryover_bars=self.carryover_bars)
        return enrich_fcs_bars(df)

    def _infer_row(self, df: pd.DataFrame, i: int) -> dict:
        x_stk, x_opt = row_to_tensors(df, i, self.stock_map, self.option_map, self.n_stock, self.n_opt)
        ts = df["timestamp"].iloc[i]
        try:
            dow = pd.to_datetime(ts).dayofweek
        except (ValueError, TypeError):
            dow = 0
        with torch.no_grad():
            out = self.model(
                torch.from_numpy(x_stk[None]).to(self.device),
                torch.from_numpy(x_opt[None]).to(self.device),
                {
                    "stock_id": torch.tensor([self.stock_id], device=self.device),
                    "sector_id": torch.tensor([self.sector_id], device=self.device),
                    "day_of_week": torch.tensor([int(dow)], device=self.device),
                },
            )
        return {k: float(out[k].squeeze().cpu().item()) for k in out if k != "logits_dir"}

    @staticmethod
    def _quotes_from_dict(quotes: Dict[str, float]) -> SessionQuotes:
        return SessionQuotes(
            call_bid=float(quotes.get("exec_call_bid", np.nan)),
            call_ask=float(quotes.get("exec_call_ask", np.nan)),
            call_spread_pct=float(quotes.get("exec_call_spread_pct", quotes.get("exec_call_spread", 0.0)) or 0.0),
            put_bid=quotes.get("exec_put_bid"),
            put_ask=quotes.get("exec_put_ask"),
            put_spread_pct=quotes.get("exec_put_spread_pct"),
        )

    def on_tick_disaster(self, tick_quotes: Dict[str, float]) -> Optional[dict]:
        """tick 级灾难止损(3-5s 平滑后喂入)。"""
        tq = self._quotes_from_dict(tick_quotes)
        evs = self.session.on_tick(self.bar_index, tick_quotes.get("timestamp"), tq)
        if not evs:
            return None
        ev = evs[0]
        return {
            "action": "EXIT",
            "reason": ev.reason,
            "leg": ev.leg,
            "price": ev.price,
            "net_return": ev.net_return,
        }

    def on_bar_close(
        self,
        history_df: pd.DataFrame,
        quotes: Dict[str, float],
    ) -> Optional[dict]:
        """
        history_df: 含 FCS 特征 + 历史 bar,按 timestamp 升序。
        首根 bar 即可推理;可选 `set_session_carryover(昨日 tail)` 满 seq_len。
        quotes 必需键: exec_call_bid/ask/spread_pct; dual 时 exec_put_*。
        """
        if history_df.empty:
            return None
        df = self._prepare_history(history_df)
        i = len(df) - 1

        ts = df["timestamp"].iloc[i]
        day_key = pd.to_datetime(ts).date()
        sess = int(session_minute(pd.Series([ts])).iloc[0])
        sq = self._quotes_from_dict(quotes)

        if self.session.position is not None:
            evs = self.session.on_minute_bar(
                self.bar_index, ts, sess, sq, SessionSignal(), day_key=day_key
            )
            self.bar_index += 1
            if evs and evs[0].kind == "EXIT":
                ev = evs[0]
                return {
                    "action": "EXIT",
                    "reason": ev.reason,
                    "leg": ev.leg,
                    "price": ev.price,
                    "net_return": ev.net_return,
                }
            return None

        if (
            self.session.pending_entry_bar is not None
            and self.bar_index >= self.session.pending_entry_bar
        ):
            evs = self.session.on_minute_bar(
                self.bar_index, ts, sess, sq, SessionSignal(), day_key=day_key
            )
            self.bar_index += 1
            if evs and evs[0].kind == "ENTER":
                ev = evs[0]
                return {"action": "ENTER", "leg": ev.leg, "limit_price": ev.price, "edge": ev.edge}
            return None

        preds = self._infer_row(df, i)
        row = df.iloc[i]
        signal = SessionSignal(
            edge=preds.get("net_edge"),
            call_edge=preds.get(qcfg.CALL_EDGE_COL),
            put_edge=preds.get(qcfg.PUT_EDGE_COL),
            straddle_edge=preds.get(qcfg.STRADDLE_EDGE_COL),
            edge_q10=preds.get(qcfg.EDGE_Q10_COL),
            put_gate=preds.get(qcfg.PUT_GATE_COL),
            open30_max_ret=preds.get("open30_max_ret") or row.get("open30_max_ret"),
            open30_peak_dd=preds.get("open30_peak_dd") or row.get("open30_peak_dd"),
            spot_ret_5bar=preds.get("spot_ret_5bar") or row.get("spot_ret_5bar"),
            trend_ret_30m=preds.get("trend_fit_ret_30m") or row.get("trend_fit_ret_30m"),
            trend_r2_30m=preds.get("trend_fit_r2_30m") or row.get("trend_fit_r2_30m"),
            vix_reversal_count_30m=row.get("vix_reversal_count_30m"),
            spot_day_ret=row.get("spot_day_ret"),
            spot_range_30m=row.get("spot_range_30m"),
            day_range_pos=row.get("day_range_pos"),
            bb_width=row.get("bb_width"),
            best_side_put_prob=preds.get("best_side_put_prob") or row.get("best_side_put_prob"),
            best_side_none_prob=preds.get("best_side_none_prob") or row.get("best_side_none_prob"),
            best_side_call_prob=preds.get("best_side_call_prob") or row.get("best_side_call_prob"),
            spot_down_prob=preds.get("spot_down_prob") or row.get("spot_down_prob"),
            spot_flat_prob=preds.get("spot_flat_prob") or row.get("spot_flat_prob"),
            spot_up_prob=preds.get("spot_up_prob") or row.get("spot_up_prob"),
            spot_close=(
                float(row["close"])
                if "close" in row.index and row.get("close") is not None and np.isfinite(row.get("close"))
                else None
            ),
        )
        evs = self.session.on_minute_bar(
            self.bar_index, ts, sess, sq, signal, day_key=day_key
        )
        self.bar_index += 1
        if not evs:
            return None
        ev = evs[0]
        if ev.kind == "SIGNAL":
            return {
                "action": "SIGNAL",
                "leg": ev.leg,
                "edge": ev.edge,
                "pending_bar": ev.extra.get("pending_bar"),
                "preds": preds,
            }
        if ev.kind == "ENTER":
            return {"action": "ENTER", "leg": ev.leg, "limit_price": ev.price, "edge": ev.edge, "preds": preds}
        return None

    @property
    def position(self):
        return self.session.position

    @property
    def replay_result(self):
        return self.session.result
