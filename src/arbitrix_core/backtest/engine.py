from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import arbitrix_core.costs as costs
from arbitrix_core.portfolio import Portfolio
from arbitrix_core.strategies.base import BaseStrategy, invoke_strategy_on_bar
from arbitrix_core.trading import Order, Signal, Trade, Position
from arbitrix_core.types import InstrumentConfig


@dataclass
class BTConfig:
    commission_per_lot: float = 3.0
    default_slippage_points: float = 0.5
    slippage_atr_multiplier: float = 0.0
    apply_spread_cost: bool = True
    apply_swap_cost: bool = True
    apply_stop_take: bool = True
    market_fill_price: str = "close"  # "open" or "close"
    exit_fill_price: str = "close"  # "open" or "close"
    intra_bar_model: str = "sl_first"  # "sl_first", "tp_first", "none"
    trailing_mode: str = "none"  # placeholder for future engine modes
    trailing_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class BTResult:
    trades: List[Trade]
    daily_equity: pd.Series
    gross_equity: pd.Series
    equity_marked: pd.Series
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    positions: List["Position"] = field(default_factory=list)
    prepared: Optional[pd.DataFrame] = None


class Backtester:
    def __init__(self, cfg: BTConfig, instruments: Optional[Dict[str, InstrumentConfig]] = None):
        self.cfg = cfg
        self.instruments = instruments or {}
        self._order_id = 0
        costs.set_commission_per_lot(cfg.commission_per_lot)

    def _next_order_id(self) -> int:
        self._order_id += 1
        return self._order_id

    @staticmethod
    def _preserve_prepared_columns(
        raw_frame: pd.DataFrame,
        prepared_frame: pd.DataFrame,
        *,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Re-attach market columns that strategies may drop during prepare()."""
        if prepared_frame is None or prepared_frame.empty:
            return prepared_frame
        if raw_frame is None or raw_frame.empty:
            return prepared_frame
        requested = [str(col) for col in (columns or ("spread", "__regime_output__"))]
        missing = [
            col
            for col in requested
            if col in raw_frame.columns and col not in prepared_frame.columns
        ]
        if not missing:
            return prepared_frame
        source = raw_frame[missing].copy()
        source.index = pd.to_datetime(source.index, utc=True)
        if not source.index.is_monotonic_increasing:
            source = source.sort_index(kind="mergesort")
        if source.index.has_duplicates:
            source = source[~source.index.duplicated(keep="last")]
        target = prepared_frame.copy()
        target_index_utc = pd.to_datetime(target.index, utc=True)
        aligned = source.reindex(target_index_utc)
        for col in missing:
            series = aligned[col]
            if col == "__regime_output__" and not source.empty:
                # Preserve causal semantics with as-of alignment (latest source ts <= target ts).
                source_values = source[col]
                source_ns = source_values.index.view("int64")
                target_ns = target_index_utc.view("int64")
                positions = np.searchsorted(source_ns, target_ns, side="right") - 1
                asof_values: List[Any] = []
                for position in positions:
                    if position < 0:
                        asof_values.append(None)
                        continue
                    value = source_values.iat[position]
                    asof_values.append(value.copy() if hasattr(value, "copy") else value)
                asof_series = pd.Series(asof_values, index=series.index, dtype="object")
                series = series.where(series.notna(), asof_series)
            if col == "spread":
                series = series.fillna(0.0)
            target[col] = series.to_numpy()
        return target

    def run_single(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        risk_perc: float,
        initial_equity: float,
        swap_override: Optional[dict] = None,
        *,
        cancel_callback: Optional[Callable[[], None]] = None,
        early_stop_conditions: Optional[Dict[str, Any]] = None,
        window_start: Optional[datetime] = None,
        capture_prepared: bool = False,
        collect_diagnostics: bool = True,
    ) -> BTResult:
        run_started = time.monotonic()
        prepare_elapsed = 0.0
        loop_elapsed = 0.0
        finalize_elapsed = 0.0
        runtime_breakdown_enabled = not collect_diagnostics
        prepare_breakdown: Dict[str, float] = {
            "strategy_prepare_s": 0.0,
            "preserve_columns_s": 0.0,
            "warmup_trim_s": 0.0,
            "start_filter_s": 0.0,
        }
        loop_breakdown: Dict[str, float] = {
            "control_s": 0.0,
            "pre_trade_s": 0.0,
            "stop_check_s": 0.0,
            "portfolio_sync_s": 0.0,
            "on_bar_s": 0.0,
            "apply_signals_s": 0.0,
            "expire_orders_s": 0.0,
            "fill_orders_s": 0.0,
            "mark_to_market_s": 0.0,
            "early_stop_check_s": 0.0,
        }
        post_loop_breakdown: Dict[str, float] = {
            "closeout_s": 0.0,
            "final_portfolio_sync_s": 0.0,
        }
        finalize_breakdown: Dict[str, float] = {
            "metrics_aggregation_s": 0.0,
            "guardrails_s": 0.0,
            "metadata_s": 0.0,
        }

        def _maybe_cancel() -> None:
            if cancel_callback:
                cancel_callback()

        _maybe_cancel()
        if df.empty:
            raise ValueError("DataFrame is empty; cannot run backtest.")

        start_filter: Optional[pd.Timestamp] = None
        if window_start is not None:
            start_filter = pd.Timestamp(window_start)
            if start_filter.tzinfo is None:
                start_filter = start_filter.tz_localize("UTC")
            else:
                start_filter = start_filter.tz_convert("UTC")

        portfolio = Portfolio(initial_equity=initial_equity, equity_source="backtest")
        strategy.portfolio = portfolio

        # Log input data range for standard backtest
        strategy_name = getattr(strategy, 'name', '') or strategy.__class__.__name__
        df_start = df.index[0].isoformat() if not df.empty else 'N/A'
        df_end = df.index[-1].isoformat() if not df.empty else 'N/A'
        if collect_diagnostics:
            logger.info(
                f"[STANDARD] {strategy_name} - Input data range: {df_start} to {df_end} ({len(df)} bars), "
                f"start_filter: {start_filter.isoformat() if start_filter else 'None'}"
            )

        prepare_started = time.monotonic()
        section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
        prepared = strategy.prepare(df)
        if runtime_breakdown_enabled:
            prepare_breakdown["strategy_prepare_s"] += max(
                0.0,
                time.monotonic() - section_started,
            )
        _maybe_cancel()

        if prepared is None or not isinstance(prepared, pd.DataFrame):
            raise ValueError("Strategy.prepare() must return a DataFrame.")
        section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
        prepared = self._preserve_prepared_columns(df, prepared, columns=("spread", "__regime_output__"))
        if runtime_breakdown_enabled:
            prepare_breakdown["preserve_columns_s"] += max(
                0.0,
                time.monotonic() - section_started,
            )

        if prepared.empty:
            strategy_name = getattr(strategy, 'name', '') or strategy.__class__.__name__
            symbol = getattr(strategy, 'symbol', 'unknown')
            timeframe = getattr(strategy, 'timeframe', 'unknown')
            raise ValueError(
                f"Strategy '{strategy_name}' produced no data for {symbol}/{timeframe}. "
                f"This usually means insufficient input data for the strategy's warmup period. "
                f"Input had {len(df)} bars. Check logs for strategy-specific requirements."
            )

        warmup_bars = int(getattr(strategy, "warmup_bars", lambda: 0)() or 0)
        if warmup_bars > 0 and start_filter is None:
            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            if len(prepared) < warmup_bars:
                strategy_name = getattr(strategy, 'name', '') or strategy.__class__.__name__
                symbol = getattr(strategy, 'symbol', 'unknown')
                timeframe = getattr(strategy, 'timeframe', 'unknown')
                raise ValueError(
                    f"Strategy '{strategy_name}' produced no data after warmup for {symbol}/{timeframe}. "
                    f"Warmup required {warmup_bars} bars but prepared had {len(prepared)}. "
                    f"Input had {len(df)} bars."
                )
            drop_bars = max(0, warmup_bars - 1)
            if drop_bars:
                prepared = prepared.iloc[drop_bars:]
            if collect_diagnostics:
                logger.info(
                    f"[STANDARD] {strategy_name} - Applied warmup_bars={warmup_bars} (drop={drop_bars}); "
                    f"prepared trimmed to {len(prepared)} bars"
                )
            if runtime_breakdown_enabled:
                prepare_breakdown["warmup_trim_s"] += max(
                    0.0,
                    time.monotonic() - section_started,
                )

        # Log prepared data range before filtering
        prep_start = prepared.index[0].isoformat() if not prepared.empty else 'N/A'
        prep_end = prepared.index[-1].isoformat() if not prepared.empty else 'N/A'
        if collect_diagnostics:
            logger.info(
                f"[STANDARD] {strategy_name} - Prepared data range: {prep_start} to {prep_end} ({len(prepared)} bars)"
            )

        if start_filter is not None and not prepared.empty:
            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            index = prepared.index
            if not isinstance(index, pd.DatetimeIndex):
                raise ValueError("Strategy output must use a datetime index for windowed runs.")
            if index.tz is None:
                cutoff = start_filter.tz_localize(None)
            else:
                cutoff = start_filter.tz_convert(index.tz)
            prepared = prepared.loc[prepared.index >= cutoff]
            if prepared.empty:
                raise ValueError(
                    f"No data available at or after {start_filter.isoformat()} "
                    f"for strategy {getattr(strategy, 'name', '') or 'unknown'}."
                )
            # Log after filtering
            filt_start = prepared.index[0].isoformat() if not prepared.empty else 'N/A'
            filt_end = prepared.index[-1].isoformat() if not prepared.empty else 'N/A'
            if collect_diagnostics:
                logger.info(
                    f"[STANDARD] {strategy_name} - After start_filter: {filt_start} to {filt_end} ({len(prepared)} bars)"
                )
            if runtime_breakdown_enabled:
                prepare_breakdown["start_filter_s"] += max(
                    0.0,
                    time.monotonic() - section_started,
                )
        prepare_elapsed = max(0.0, time.monotonic() - prepare_started)
        prepared_snapshot = prepared.copy() if capture_prepared else None

        equity = float(initial_equity)
        gross_equity = float(initial_equity)
        symbol = strategy.symbol or "SYMBOL"
        open_trades: List[Trade] = []
        closed_trades: List[Trade] = []
        signal_intents: Optional[List[Dict[str, Any]]] = [] if collect_diagnostics else None
        working_orders: List[Order] = []
        all_orders: List[Order] = []
        equity_by_day: Dict[pd.Timestamp, float] = {}
        equity_marked_by_day: Dict[pd.Timestamp, float] = {}
        gross_by_day: Dict[pd.Timestamp, float] = {}
        position_snapshots: List[Position] = []

        early_stop_flag = bool(
            early_stop_conditions.get("enabled", True)
            if isinstance(early_stop_conditions, dict)
            else False
        )
        max_dd_threshold = (
            early_stop_conditions.get("max_drawdown")
            if early_stop_conditions and early_stop_flag
            else None
        )
        min_trades_threshold = (
            early_stop_conditions.get("min_trades")
            if early_stop_conditions and early_stop_flag
            else None
        )
        check_interval = (
            early_stop_conditions.get("check_interval", 50)
            if early_stop_conditions and early_stop_flag
            else 50
        )
        early_stop_enabled = bool(
            early_stop_flag
            and (
                (max_dd_threshold is not None)
                or (min_trades_threshold is not None and min_trades_threshold > 0)
            )
        )
        running_peak_equity = float(initial_equity)
        running_max_drawdown = 0.0
        prepared_index = prepared.index
        if not isinstance(prepared_index, pd.DatetimeIndex):
            raise ValueError("Strategy output must use a datetime index.")
        if prepared_index.tz is None:
            loop_index = prepared_index.tz_localize("UTC")
        else:
            loop_index = prepared_index.tz_convert("UTC")
        loop_days = loop_index.normalize()
        cancel_check_interval = 1 if collect_diagnostics else 64
        bar_count = 0
        early_stopped = False
        early_stop_reason = None

        loop_started = time.monotonic()
        prepared_len = len(prepared)
        prepared_iloc = prepared.iloc
        unrealized_pnl_fn = self._unrealized_pnl
        for row_idx in range(prepared_len):
            row = prepared_iloc[row_idx]
            control_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            if row_idx % cancel_check_interval == 0:
                _maybe_cancel()
            bar_count += 1
            ts = loop_index[row_idx]
            day = loop_days[row_idx]
            if runtime_breakdown_enabled:
                loop_breakdown["control_s"] += max(0.0, time.monotonic() - control_started)

            # Apply swap logic before evaluating stops/signals.
            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            pre_sl_trades: List[Trade] = []
            for trade in open_trades:
                swap_delta = self._apply_overnight_swap(symbol, trade, day, swap_override)
                if swap_delta:
                    equity += swap_delta
                pre_sl_trades.append(trade)
            if runtime_breakdown_enabled:
                loop_breakdown["pre_trade_s"] += max(0.0, time.monotonic() - section_started)

            # Vectorised SL/TP check (numpy hot-path when many trades are open)
            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            open_trades, equity, gross_equity = self._check_stops_vectorized(
                symbol, pre_sl_trades, row, ts, equity, gross_equity, closed_trades
            )
            if runtime_breakdown_enabled:
                loop_breakdown["stop_check_s"] += max(0.0, time.monotonic() - section_started)

            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            unrealized_before_signals = unrealized_pnl_fn(symbol, open_trades, row)
            equity_marked = equity + unrealized_before_signals
            portfolio.sync(
                timestamp=ts,
                equity=equity,
                gross_equity=gross_equity,
                equity_marked=equity_marked,
                open_trades=open_trades,
                closed_trades=closed_trades,
                pending_orders=working_orders,
                orders=all_orders,
            )
            if runtime_breakdown_enabled:
                loop_breakdown["portfolio_sync_s"] += max(0.0, time.monotonic() - section_started)

            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            regime_output = None
            if isinstance(row, pd.Series):
                regime_output = row.get("__regime_output__")
            bar_signals = invoke_strategy_on_bar(strategy, row, portfolio, regime_output)
            if runtime_breakdown_enabled:
                loop_breakdown["on_bar_s"] += max(0.0, time.monotonic() - section_started)
            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            equity, gross_equity, open_trades, working_orders = self._apply_bar_signals(
                strategy=strategy,
                signals=bar_signals,
                row=row,
                ts=ts,
                symbol=symbol,
                risk_perc=risk_perc,
                open_trades=open_trades,
                closed_trades=closed_trades,
                working_orders=working_orders,
                all_orders=all_orders,
                equity=equity,
                gross_equity=gross_equity,
                signal_intents=signal_intents,
            )
            if runtime_breakdown_enabled:
                loop_breakdown["apply_signals_s"] += max(0.0, time.monotonic() - section_started)

            # Drop expired orders
            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            still_working: List[Order] = []
            for order in working_orders:
                if order.valid_until is not None and ts > order.valid_until:
                    order.status = "expired"
                    continue
                still_working.append(order)
            working_orders = still_working
            if runtime_breakdown_enabled:
                loop_breakdown["expire_orders_s"] += max(0.0, time.monotonic() - section_started)

            # Attempt fills
            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            newly_filled: List[Trade] = []
            remaining_orders: List[Order] = []
            for order in working_orders:
                filled = self._try_fill_order(order, row)
                if filled is None:
                    remaining_orders.append(order)
                    continue
                trade, equity = self._open_trade_from_order(symbol, order, filled, row, equity)
                if trade:
                    newly_filled.append(trade)
            working_orders = remaining_orders
            open_trades.extend(newly_filled)
            if runtime_breakdown_enabled:
                loop_breakdown["fill_orders_s"] += max(0.0, time.monotonic() - section_started)

            # Record equity (realized and marked-to-market)
            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            if equity > running_peak_equity:
                running_peak_equity = equity
            if running_peak_equity > 0.0:
                dd_now = (equity - running_peak_equity) / running_peak_equity
                if dd_now < running_max_drawdown:
                    running_max_drawdown = dd_now
            equity_by_day[day] = equity
            gross_by_day[day] = gross_equity
            unrealized_after_signals = unrealized_pnl_fn(symbol, open_trades, row)
            equity_marked_by_day[day] = equity + unrealized_after_signals
            if runtime_breakdown_enabled:
                loop_breakdown["mark_to_market_s"] += max(0.0, time.monotonic() - section_started)

            section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
            if early_stop_enabled and bar_count % check_interval == 0:
                if max_dd_threshold is not None and equity_by_day:
                    current_dd = running_max_drawdown
                    if abs(current_dd) > max_dd_threshold:
                        early_stopped = True
                        early_stop_reason = f"max_drawdown exceeded: {abs(current_dd):.4f} > {max_dd_threshold}"
                        break

                if min_trades_threshold is not None and bar_count > len(prepared) * 0.3:
                    if len(closed_trades) < min_trades_threshold:
                        early_stopped = True
                        early_stop_reason = f"insufficient trades: {len(closed_trades)} < {min_trades_threshold}"
                        break
            if runtime_breakdown_enabled:
                loop_breakdown["early_stop_check_s"] += max(0.0, time.monotonic() - section_started)
        loop_elapsed = max(0.0, time.monotonic() - loop_started)

        last_ts = prepared.index[-1].tz_localize("UTC") if prepared.index[-1].tzinfo is None else prepared.index[-1].tz_convert("UTC")
        last_row = prepared.iloc[-1]
        day = last_ts.normalize()
        section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
        if early_stopped:
            for trade in open_trades:
                equity, gross_equity, _ = self._close_trade(
                    symbol, trade, last_row, last_ts, equity, gross_equity, closed_trades, reason="early_stop"
                )
            equity_by_day[day] = equity
            gross_by_day[day] = gross_equity
            equity_marked_by_day[day] = equity
        else:
            # Force close remaining trades at last bar
            for trade in list(open_trades):
                swap_delta = self._apply_overnight_swap(symbol, trade, day, swap_override)
                if swap_delta:
                    equity += swap_delta
                equity, gross_equity, _ = self._close_trade(
                    symbol, trade, last_row, last_ts, equity, gross_equity, closed_trades, reason="eod"
                )
            equity_by_day[day] = equity
            gross_by_day[day] = gross_equity
            equity_marked_by_day[day] = equity
        if runtime_breakdown_enabled:
            post_loop_breakdown["closeout_s"] += max(0.0, time.monotonic() - section_started)

        section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
        portfolio.sync(
            timestamp=last_ts,
            equity=equity,
            gross_equity=gross_equity,
            equity_marked=equity + unrealized_pnl_fn(symbol, open_trades, last_row),
            open_trades=open_trades,
            closed_trades=closed_trades,
            pending_orders=working_orders,
            orders=all_orders,
        )
        if runtime_breakdown_enabled:
            post_loop_breakdown["final_portfolio_sync_s"] += max(
                0.0,
                time.monotonic() - section_started,
            )

        daily_equity = pd.Series(equity_by_day).sort_index().ffill()
        gross_equity_series = pd.Series(gross_by_day).sort_index().ffill()
        if gross_equity_series.empty and not daily_equity.empty:
            gross_equity_series = daily_equity.copy()
        equity_marked_series = pd.Series(equity_marked_by_day).sort_index().ffill()

        finalize_started = time.monotonic()
        _maybe_cancel()
        section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
        metrics = self._compute_metrics(daily_equity, initial_equity=initial_equity)
        total_commission = sum(t.commission_paid for t in closed_trades)
        total_spread = sum(t.spread_cost for t in closed_trades)
        total_slippage = sum(t.slippage_cost for t in closed_trades)
        total_swap = sum(t.swap_pnl for t in closed_trades)
        total_gross = sum(t.gross_pnl for t in closed_trades)
        total_net = sum(t.net_pnl for t in closed_trades)
        total_fees = total_commission + total_spread + total_slippage
        winning_net = sum(t.net_pnl for t in closed_trades if t.net_pnl > 0)
        losing_net = -sum(t.net_pnl for t in closed_trades if t.net_pnl < 0)
        gross_abs = abs(total_gross)
        if gross_abs <= 1e-9:
            fees_to_gross = 1.0 if total_fees > 0 else 0.0
        else:
            fees_to_gross = float(total_fees) / gross_abs

        returns_series, _, _ = self._returns_for_metrics(daily_equity, initial_equity=initial_equity)
        returns_count = int(len(returns_series))
        trade_count = int(len(closed_trades))
        expectancy = float(total_net) / float(trade_count) if trade_count > 0 else 0.0
        if losing_net <= 1e-9:
            profit_factor = 10.0 if winning_net > 1e-9 else 0.0
        else:
            profit_factor = float(winning_net) / float(losing_net)
        if runtime_breakdown_enabled:
            finalize_breakdown["metrics_aggregation_s"] += max(
                0.0,
                time.monotonic() - section_started,
            )

        # Robust-score disqualification guardrails are only applied when explicitly provided.
        # This avoids hidden PSR sample gating coming from unrelated early-stop defaults.
        default_thresholds = {
            "sharpe_target": 3.0,
            "sortino_target": 4.0,
            "tail_ratio_target": 3.0,
        }
        psr_min_trades: Optional[int] = None
        psr_min_returns: Optional[int] = None
        psr_min_observations: Optional[int] = None
        psr_threshold: Optional[float] = None
        fees_to_gross_limit: Optional[float] = None
        drawdown_limit: Optional[float] = None
        turnover_limit: Optional[float] = None
        sharpe_target = 3.0
        sortino_target = 4.0
        tail_ratio_target = 3.0
        ignore_disqualification = False

        if early_stop_conditions and early_stop_flag:
            def _get_optional_int(key: str) -> Optional[int]:
                if key not in early_stop_conditions:
                    return None
                try:
                    return max(0, int(early_stop_conditions.get(key)))
                except (TypeError, ValueError):
                    return None

            def _get_optional_float(key: str) -> Optional[float]:
                if key not in early_stop_conditions:
                    return None
                try:
                    return max(0.0, float(early_stop_conditions.get(key)))
                except (TypeError, ValueError):
                    return None

            def _get_setting(key: str, default: Any) -> Any:
                return early_stop_conditions[key] if key in early_stop_conditions else default

            psr_min_trades = _get_optional_int("psr_min_trades")
            psr_min_returns = _get_optional_int("psr_min_returns")
            psr_min_observations = _get_optional_int("psr_min_observations")
            psr_threshold = _get_optional_float("psr_threshold")
            fees_to_gross_limit = _get_optional_float("fees_to_gross")
            drawdown_limit = _get_optional_float("drawdown_threshold")
            turnover_limit = _get_optional_float("turnover_threshold")
            sharpe_target = float(
                _get_setting("sharpe_target", default_thresholds["sharpe_target"]) or default_thresholds["sharpe_target"]
            )
            sortino_target = float(
                _get_setting("sortino_target", default_thresholds["sortino_target"]) or default_thresholds["sortino_target"]
            )
            tail_ratio_target = float(
                _get_setting("tail_ratio_target", default_thresholds["tail_ratio_target"]) or default_thresholds["tail_ratio_target"]
            )
            ignore_disqualification = bool(_get_setting("ignore_disqualification", False))

        section_started = time.monotonic() if runtime_breakdown_enabled else 0.0
        effective_psr_min_observations = int(psr_min_observations) if psr_min_observations is not None else 2
        metrics["PSR"] = self._probabilistic_sharpe(
            returns_series,
            min_observations=effective_psr_min_observations,
        )

        (
            robust_score,
            score_components,
            disqualified,
            reasons,
            psr_guardrail,
        ) = self._evaluate_robust_score(
            metrics,
            fees_to_gross,
            returns_count=returns_count,
            trade_count=trade_count,
            thresholds={
                "psr_min_trades": psr_min_trades,
                "psr_min_returns": psr_min_returns,
                "psr_min_observations": psr_min_observations,
                "psr_threshold": psr_threshold,
                "fees_to_gross": fees_to_gross_limit,
                "drawdown_threshold": drawdown_limit,
                "turnover_threshold": turnover_limit,
                "sharpe_target": sharpe_target,
                "sortino_target": sortino_target,
                "tail_ratio_target": tail_ratio_target,
                "ignore_disqualification": ignore_disqualification,
            },
        )

        metrics.update(
            {
                "gross_pnl": float(total_gross),
                "net_pnl": float(total_net),
                "total_commission": float(total_commission),
                "total_spread_cost": float(total_spread),
                "total_slippage_cost": float(total_slippage),
                "total_swap_pnl": float(total_swap),
                "FeesToGross": float(fees_to_gross),
                "RobustScore": float(robust_score),
                "Qualified": 0.0 if disqualified else 1.0,
                "ReturnCount": float(returns_count),
                "TradeCount": float(trade_count),
                "Expectancy": float(expectancy),
                "expectancy": float(expectancy),
                "ProfitFactor": float(profit_factor),
                "profit_factor": float(profit_factor),
            }
        )

        psr_min_returns_meta: Optional[int] = int(psr_guardrail["min_returns"]) if psr_guardrail else (
            int(psr_min_returns) if psr_min_returns is not None else None
        )
        psr_min_trades_meta: Optional[int] = int(psr_guardrail["min_trades"]) if psr_guardrail else (
            int(psr_min_trades) if psr_min_trades is not None else None
        )
        psr_min_observations_meta: Optional[int] = int(psr_guardrail["min_observations"]) if psr_guardrail else (
            int(psr_min_observations) if psr_min_observations is not None else None
        )

        metadata = {
            "initial_equity": float(initial_equity),
            "disqualified": disqualified,
            "disqualify_reasons": reasons,
            "robust_score_components": score_components,
            "robust_score_thresholds": {
                "drawdown": drawdown_limit,
                "psr": psr_threshold,
                "psr_min_returns": psr_min_returns_meta,
                "psr_min_trades": psr_min_trades_meta,
                "psr_min_observations": psr_min_observations_meta,
                "fees_to_gross": fees_to_gross_limit,
                "turnover": turnover_limit,
                "sharpe": sharpe_target,
                "sortino": sortino_target,
                "tail_ratio": tail_ratio_target,
            },
            "sample_counts": {
                "returns": returns_count,
                "trades": trade_count,
            },
        }
        if signal_intents is not None:
            metadata["signal_intents"] = signal_intents
        if psr_guardrail:
            metadata["psr_guardrail"] = psr_guardrail
        if early_stopped:
            metadata.update(
                {
                    "disqualified": True,
                    "disqualify_reasons": [f"early_stop: {early_stop_reason}"],
                    "early_stopped": True,
                    "early_stop_reason": early_stop_reason,
                    "bars_processed": bar_count,
                }
            )
        finalize_elapsed = max(0.0, time.monotonic() - finalize_started)
        metadata["runtime_timing"] = {
            "prepare_s": float(prepare_elapsed),
            "loop_s": float(loop_elapsed),
            "finalize_s": float(finalize_elapsed),
            "total_s": float(max(0.0, time.monotonic() - run_started)),
        }
        if runtime_breakdown_enabled:
            prepare_accounted = float(sum(prepare_breakdown.values()))
            loop_accounted = float(sum(loop_breakdown.values()))
            finalize_accounted = float(sum(finalize_breakdown.values()))
            prepare_residual = max(0.0, float(prepare_elapsed) - prepare_accounted)
            loop_residual = max(0.0, float(loop_elapsed) - loop_accounted)
            finalize_residual = max(0.0, float(finalize_elapsed) - finalize_accounted)
            metadata["runtime_timing"]["prepare_breakdown"] = {
                key: float(value) for key, value in prepare_breakdown.items()
            }
            metadata["runtime_timing"]["loop_breakdown"] = {
                key: float(value) for key, value in loop_breakdown.items()
            }
            metadata["runtime_timing"]["post_loop_breakdown"] = {
                key: float(value) for key, value in post_loop_breakdown.items()
            }
            metadata["runtime_timing"]["finalize_breakdown"] = {
                key: float(value) for key, value in finalize_breakdown.items()
            }
            metadata["runtime_timing"]["prepare_breakdown_residual_s"] = float(prepare_residual)
            metadata["runtime_timing"]["loop_breakdown_residual_s"] = float(loop_residual)
            metadata["runtime_timing"]["finalize_breakdown_residual_s"] = float(finalize_residual)
            metadata["runtime_timing"]["loop_bar_count"] = int(bar_count)
            metadata["runtime_timing"]["loop_per_bar_ms"] = float(
                (loop_elapsed / float(bar_count) * 1000.0) if bar_count > 0 else 0.0
            )

        return BTResult(
            trades=closed_trades,
            daily_equity=daily_equity,
            gross_equity=gross_equity_series,
            equity_marked=equity_marked_series,
            metrics=metrics,
            metadata=metadata,
            orders=all_orders if collect_diagnostics else [],
            positions=self._final_positions(closed_trades) if collect_diagnostics else [],
            prepared=prepared_snapshot,
        )

    def _unrealized_pnl(self, symbol: str, trades: List[Trade], row: pd.Series) -> float:
        pv = costs.get_point_value(symbol)
        if pv == 0:
            return 0.0
        pnl = 0.0
        for trade in trades:
            if trade.side == "long":
                pnl += (row["close"] - trade.entry_price) * pv * trade.volume
            else:
                pnl += (trade.entry_price - row["close"]) * pv * trade.volume
        return pnl

    def _create_order_from_signal(
        self,
        strategy: BaseStrategy,
        signal: Signal,
        row: pd.Series,
            risk_perc: float,
            equity: float,
        ) -> Optional[Order]:
        symbol = strategy.symbol or "SYMBOL"
        stop_points = float(strategy.stop_distance_points(row))
        if stop_points <= 0:
            return None
        take_points = float(strategy.take_distance_points(row))
        point_value = costs.get_point_value(symbol)
        if point_value <= 0:
            return None


        # here we set the volume based on the Signal produced by the strategy
        volume = signal.volume
        # if the volume is not set, we calculate it based on the risk percentage (on the equity)
        # the strategy can also set a risk_multiplier
        # that will be multiplied by the risk_perc
        # it's useful when the strategy want to module the risk but not calculating the position size
        if volume is None:
            risk_dollars = equity * risk_perc * signal.risk_multiplier
            volume = round(risk_dollars / (point_value * stop_points), 2)
        if volume <= 0:
            return None

        price: Optional[float] = None
        if signal.order_type == "limit":
            price = signal.limit_price if signal.limit_price is not None else signal.price
        elif signal.order_type == "stop":
            price = signal.stop_price if signal.stop_price is not None else signal.price

        valid_until = self._resolve_valid_until(strategy, signal)

        return Order(
            id=str(self._next_order_id()),
            symbol=symbol,
            side="buy" if signal.action == "buy" else "sell",
            type=signal.order_type,
            volume=float(volume),
            price=float(price) if price is not None else None,
            created_at=signal.when,
            stop_points=stop_points,
            take_points=take_points,
            valid_until=valid_until,
            tif=signal.tif,
            strategy=getattr(strategy, "name", strategy.__class__.__name__),
            magic=signal.magic,
        )

    def _try_fill_order(self, order: Order, row: pd.Series) -> Optional[float]:
        if order.type == "market":
            fill = row["open"] if self.cfg.market_fill_price == "open" else row["close"]
            order.status = "filled"
            return float(fill)

        if order.type == "limit":
            if order.side == "buy" and row["low"] <= float(order.price):
                order.status = "filled"
                return float(order.price)
            if order.side == "sell" and row["high"] >= float(order.price):
                order.status = "filled"
                return float(order.price)
            order.status = "working"
            return None

        if order.type == "stop":
            if order.side == "buy" and row["high"] >= float(order.price):
                order.status = "filled"
                return float(order.price)
            if order.side == "sell" and row["low"] <= float(order.price):
                order.status = "filled"
                return float(order.price)
            order.status = "working"
            return None
        return None

    def _open_trade_from_order(
        self,
        symbol: str,
        order: Order,
        fill_price: float,
        row: pd.Series,
        equity: float,
    ) -> tuple[Optional[Trade], float]:
        commission = costs.commission_one_side(symbol, float(fill_price), order.volume)
        spread_points = float(row.get("spread", 0.0)) if self.cfg.apply_spread_cost else 0.0
        if pd.isna(spread_points):
            spread_points = 0.0
        spread_cost = costs.spread_cost(symbol, spread_points / 2.0, order.volume) if self.cfg.apply_spread_cost else 0.0
        slippage_points = self._slippage_points(symbol, row)
        slippage_cost_val = costs.slippage_cost(symbol, slippage_points, order.volume)
        equity -= commission + spread_cost + slippage_cost_val

        trade = Trade(
            symbol=symbol,
            side="long" if order.side == "buy" else "short",
            entry_time=order.created_at,
            entry_price=float(fill_price),
            volume=float(order.volume),
            stop_points=float(order.stop_points),
            take_points=float(order.take_points),
            commission_paid=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost_val,
            order_id=order.id,
            strategy=order.strategy,
            magic=order.magic,
        )
        trade._last_swap_day = order.created_at.normalize() if order.created_at is not None else None
        return trade, equity

    def _maybe_close_trade(
        self,
        symbol: str,
        trade: Trade,
        row: pd.Series,
        ts: pd.Timestamp,
        equity: float,
        gross_equity: float,
        trades: List[Trade],
    ) -> tuple[float, float, Optional[Trade]]:
        pv = costs.get_point_value(symbol)
        if pv == 0:
            return equity, gross_equity, trade
        if not self.cfg.apply_stop_take:
            return equity, gross_equity, trade

        stop_hit = False
        take_hit = False
        if trade.side == "long":
            stop_price = trade.entry_price - trade.stop_points
            take_price = trade.entry_price + trade.take_points if trade.take_points > 0 else None
            stop_hit = row["low"] <= stop_price
            take_hit = take_price is not None and row["high"] >= take_price
        else:
            stop_price = trade.entry_price + trade.stop_points
            take_price = trade.entry_price - trade.take_points if trade.take_points > 0 else None
            stop_hit = row["high"] >= stop_price
            take_hit = take_price is not None and row["low"] <= take_price

        if not stop_hit and not take_hit:
            return equity, gross_equity, trade

        if stop_hit and take_hit:
            model = self.cfg.intra_bar_model
            if model == "tp_first":
                stop_hit = False
            elif model == "none":
                stop_hit = True
                take_hit = False
            else:
                take_hit = False

        if trade.side == "long":
            if stop_hit:
                fill = stop_price
                pnl = (fill - trade.entry_price) * pv * trade.volume
            else:
                fill = take_price if take_price is not None else row["close"]
                pnl = (fill - trade.entry_price) * pv * trade.volume
        else:
            if stop_hit:
                fill = stop_price
                pnl = (trade.entry_price - fill) * pv * trade.volume
            else:
                fill = take_price if take_price is not None else row["close"]
                pnl = (trade.entry_price - fill) * pv * trade.volume

        commission = costs.commission_one_side(symbol, float(fill), trade.volume)
        slippage_points = self._slippage_points(symbol, row)
        slippage_cost_val = costs.slippage_cost(symbol, slippage_points, trade.volume)
        trade.exit_time = ts
        trade.exit_price = float(fill)
        trade.gross_pnl = pnl
        trade.commission_paid += commission
        trade.slippage_cost += slippage_cost_val
        trade.pnl = pnl - commission - slippage_cost_val
        total_costs = trade.commission_paid + trade.spread_cost + trade.slippage_cost
        trade.net_pnl = trade.gross_pnl - total_costs + trade.swap_pnl
        trade.notes["exit_stop"] = 1.0 if stop_hit else 0.0
        trade.notes["exit_take"] = 1.0 if take_hit else 0.0
        equity += trade.pnl
        gross_equity += trade.gross_pnl
        trades.append(trade)
        return equity, gross_equity, None

    def _check_stops_vectorized(
        self,
        symbol: str,
        trades_to_check: List[Trade],
        row,
        ts: pd.Timestamp,
        equity: float,
        gross_equity: float,
        closed_trades: List[Trade],
    ) -> tuple[List[Trade], float, float]:
        """Check SL/TP using vectorised numpy when many trades are open.

        Falls back to the scalar ``_maybe_close_trade`` path for small counts
        to avoid array-creation overhead.
        """
        if not self.cfg.apply_stop_take or not trades_to_check:
            return trades_to_check, equity, gross_equity

        pv = costs.get_point_value(symbol)
        if pv == 0:
            return trades_to_check, equity, gross_equity

        n = len(trades_to_check)

        # Scalar path for small trade counts (array overhead dominates)
        if n <= 3:
            updated: List[Trade] = []
            for trade in trades_to_check:
                equity, gross_equity, maybe_open = self._maybe_close_trade(
                    symbol, trade, row, ts, equity, gross_equity, closed_trades,
                )
                if maybe_open:
                    updated.append(maybe_open)
            return updated, equity, gross_equity

        # Vectorised path
        from arbitrix_core.backtest.fast_loop import check_stops

        entry_prices = np.array([t.entry_price for t in trades_to_check], dtype=np.float64)
        stop_points = np.array([t.stop_points for t in trades_to_check], dtype=np.float64)
        take_points = np.array([t.take_points for t in trades_to_check], dtype=np.float64)
        sides = np.array([0 if t.side == "long" else 1 for t in trades_to_check], dtype=np.int8)

        model_map = {"sl_first": 0, "tp_first": 1, "none": 2}
        model = model_map.get(self.cfg.intra_bar_model, 0)

        closed_mask, exit_prices, is_stop, is_take = check_stops(
            entry_prices, stop_points, take_points, sides,
            float(row["high"]), float(row["low"]), model,
        )

        updated_trades: List[Trade] = []
        for i, trade in enumerate(trades_to_check):
            if not closed_mask[i]:
                updated_trades.append(trade)
                continue

            fill = float(exit_prices[i])
            if trade.side == "long":
                pnl = (fill - trade.entry_price) * pv * trade.volume
            else:
                pnl = (trade.entry_price - fill) * pv * trade.volume

            commission = costs.commission_one_side(symbol, fill, trade.volume)
            slippage_pts = self._slippage_points(symbol, row)
            slippage_cost_val = costs.slippage_cost(symbol, slippage_pts, trade.volume)

            trade.exit_time = ts
            trade.exit_price = fill
            trade.gross_pnl = pnl
            trade.commission_paid += commission
            trade.slippage_cost += slippage_cost_val
            trade.pnl = pnl - commission - slippage_cost_val
            total_costs = trade.commission_paid + trade.spread_cost + trade.slippage_cost
            trade.net_pnl = trade.gross_pnl - total_costs + trade.swap_pnl
            trade.notes["exit_stop"] = 1.0 if is_stop[i] else 0.0
            trade.notes["exit_take"] = 1.0 if is_take[i] else 0.0

            equity += trade.pnl
            gross_equity += trade.gross_pnl
            closed_trades.append(trade)

        return updated_trades, equity, gross_equity

    def _apply_bar_signals(
        self,
        *,
        strategy: BaseStrategy,
        signals: Iterable[Signal] | Signal | None,
        row: pd.Series,
        ts: pd.Timestamp,
        symbol: str,
        risk_perc: float,
        open_trades: List[Trade],
        closed_trades: List[Trade],
        working_orders: List[Order],
        all_orders: List[Order],
        equity: float,
        gross_equity: float,
        signal_intents: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[float, float, List[Trade], List[Order]]:
        for next_sig in self._normalize_signals(signals):
            filtered_sig = next_sig
            if signal_intents is not None and filtered_sig.is_entry():
                signal_intents.append(self._serialize_signal_intent(filtered_sig, row, ts))

            if filtered_sig.action == "exit":
                for trade in open_trades:
                    equity, gross_equity, _ = self._close_trade(
                        symbol, trade, row, ts, equity, gross_equity, closed_trades, reason="signal_exit"
                    )
                for order in working_orders:
                    order.status = "cancelled"
                working_orders = []
                open_trades = []
                continue

            if filtered_sig.action in ("close", "partial_close", "modify_sl", "modify_tp", "cancel_order"):
                equity, gross_equity, open_trades, working_orders = self._apply_management_signal(
                    filtered_sig,
                    row,
                    ts,
                    open_trades,
                    closed_trades,
                    working_orders,
                    equity,
                    gross_equity,
                )
                continue

            if not filtered_sig.is_entry():
                continue

            order = self._create_order_from_signal(strategy, filtered_sig, row, risk_perc, equity)
            if order:
                all_orders.append(order)
                working_orders.append(order)
        return equity, gross_equity, open_trades, working_orders

    @staticmethod
    def _serialize_signal_intent(
        signal: Signal,
        row: pd.Series,
        ts: pd.Timestamp,
    ) -> Dict[str, Any]:
        def _float_or_none(value: Any) -> Optional[float]:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if math.isnan(numeric) or math.isinf(numeric):
                return None
            return float(numeric)

        provider_spread = _float_or_none(row.get("spread")) if isinstance(row, pd.Series) else None
        price = (
            _float_or_none(signal.price)
            or _float_or_none(signal.limit_price)
            or _float_or_none(signal.stop_price)
            or _float_or_none(row.get("close") if isinstance(row, pd.Series) else None)
            or 0.0
        )
        volume = abs(_float_or_none(signal.volume) or 1.0)
        payload: Dict[str, Any] = {
            "time": ts,
            "action": signal.action,
            "price": price,
            "volume": volume,
            "order_type": signal.order_type,
        }
        if signal.reason:
            payload["reason"] = signal.reason
        if signal.risk_multiplier and signal.risk_multiplier != 1.0:
            payload["risk_multiplier"] = float(signal.risk_multiplier)
        if provider_spread is not None:
            payload["provider_spread"] = max(provider_spread, 0.0)
        return payload

    def _close_trade(
        self,
        symbol: str,
        trade: Trade,
        row: pd.Series,
        ts: pd.Timestamp,
        equity: float,
        gross_equity: float,
        trades: List[Trade],
        reason: str = "force",
    ) -> tuple[float, float, Optional[Trade]]:
        pv = costs.get_point_value(symbol)
        if pv == 0:
            return equity, gross_equity, trade
        fill_price = row["open"] if self.cfg.exit_fill_price == "open" else row["close"]
        if trade.side == "long":
            pnl = (fill_price - trade.entry_price) * pv * trade.volume
        else:
            pnl = (trade.entry_price - fill_price) * pv * trade.volume
        commission = costs.commission_one_side(symbol, float(fill_price), trade.volume)
        slippage_points = self._slippage_points(symbol, row)
        slippage_cost_val = costs.slippage_cost(symbol, slippage_points, trade.volume)
        trade.exit_time = ts
        trade.exit_price = float(fill_price)
        trade.gross_pnl = pnl
        trade.commission_paid += commission
        trade.slippage_cost += slippage_cost_val
        trade.pnl = pnl - commission - slippage_cost_val
        total_costs = trade.commission_paid + trade.spread_cost + trade.slippage_cost
        trade.net_pnl = trade.gross_pnl - total_costs + trade.swap_pnl
        trade.notes[f"exit_{reason}"] = 1.0
        equity += trade.pnl
        gross_equity += trade.gross_pnl
        trades.append(trade)
        return equity, gross_equity, None

    def _apply_management_signal(
        self,
        signal: Signal,
        row: pd.Series,
        ts: pd.Timestamp,
        open_trades: List[Trade],
        closed_trades: List[Trade],
        working_orders: List[Order],
        equity: float,
        gross_equity: float,
    ) -> tuple[float, float, List[Trade], List[Order]]:
        action = signal.action
        if action in ("close", "partial_close", "modify_sl", "modify_tp"):
            trade_id = signal.target_trade_id
            if not trade_id:
                if action in ("modify_sl", "modify_tp") and signal.target_order_id:
                    order_id = signal.target_order_id
                    order = next((o for o in working_orders if o.id == order_id), None)
                    if order is None:
                        logger.debug("Backtest signal %s order not found: %s", action, order_id)
                        return equity, gross_equity, open_trades, working_orders
                    if order.price is None:
                        logger.debug("Backtest signal %s missing order price: %s", action, order_id)
                        return equity, gross_equity, open_trades, working_orders
                    if action == "modify_sl" and signal.new_sl is not None:
                        if order.side == "buy":
                            order.stop_points = max(float(order.price) - float(signal.new_sl), 0.0)
                        else:
                            order.stop_points = max(float(signal.new_sl) - float(order.price), 0.0)
                    if action == "modify_tp" and signal.new_tp is not None:
                        if order.side == "buy":
                            order.take_points = max(float(signal.new_tp) - float(order.price), 0.0)
                        else:
                            order.take_points = max(float(order.price) - float(signal.new_tp), 0.0)
                    return equity, gross_equity, open_trades, working_orders
                logger.debug("Backtest signal %s missing target_trade_id", action)
                return equity, gross_equity, open_trades, working_orders
            trade = next((t for t in open_trades if t.id == trade_id), None)
            if trade is None:
                logger.debug("Backtest signal %s trade not found: %s", action, trade_id)
                return equity, gross_equity, open_trades, working_orders

            if action in ("close", "partial_close"):
                close_volume = signal.close_volume if action == "partial_close" else None
                equity, gross_equity, updated_trade = self._partial_close_trade(
                    trade,
                    row,
                    ts,
                    equity,
                    gross_equity,
                    closed_trades,
                    close_volume,
                    reason=f"signal_{action}",
                )
                if updated_trade is None:
                    open_trades = [t for t in open_trades if t.id != trade_id]
                return equity, gross_equity, open_trades, working_orders

            if action == "modify_sl" and signal.new_sl is not None:
                trade.stop_points = self._points_from_price(trade, signal.new_sl, kind="sl")
            if action == "modify_tp" and signal.new_tp is not None:
                trade.take_points = self._points_from_price(trade, signal.new_tp, kind="tp")
            return equity, gross_equity, open_trades, working_orders

        if action == "cancel_order":
            order_id = signal.target_order_id
            if not order_id:
                logger.debug("Backtest cancel_order missing target_order_id")
                return equity, gross_equity, open_trades, working_orders
            remaining: List[Order] = []
            for order in working_orders:
                if order.id == order_id:
                    order.status = "cancelled"
                    continue
                remaining.append(order)
            working_orders = remaining
            return equity, gross_equity, open_trades, working_orders

        return equity, gross_equity, open_trades, working_orders

    def _partial_close_trade(
        self,
        trade: Trade,
        row: pd.Series,
        ts: pd.Timestamp,
        equity: float,
        gross_equity: float,
        trades: List[Trade],
        close_volume: Optional[float],
        *,
        reason: str,
    ) -> tuple[float, float, Optional[Trade]]:
        symbol = trade.symbol
        pv = costs.get_point_value(symbol)
        if pv == 0:
            return equity, gross_equity, trade
        fill_price = row["open"] if self.cfg.exit_fill_price == "open" else row["close"]
        volume = float(trade.volume)
        target = volume if close_volume is None else min(float(close_volume), volume)
        if target <= 0:
            return equity, gross_equity, trade

        pnl = (
            (fill_price - trade.entry_price) * pv * target
            if trade.side == "long"
            else (trade.entry_price - fill_price) * pv * target
        )
        entry_ratio = target / volume if volume else 1.0
        entry_commission = trade.commission_paid * entry_ratio
        entry_spread = trade.spread_cost * entry_ratio
        entry_slippage = trade.slippage_cost * entry_ratio
        entry_swap = trade.swap_pnl * entry_ratio

        close_commission = costs.commission_one_side(symbol, float(fill_price), target)
        slippage_points = self._slippage_points(symbol, row)
        close_slippage = costs.slippage_cost(symbol, slippage_points, target)

        closed_trade = Trade(
            symbol=trade.symbol,
            side=trade.side,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            volume=target,
            stop_points=trade.stop_points,
            take_points=trade.take_points,
            commission_paid=entry_commission + close_commission,
            spread_cost=entry_spread,
            slippage_cost=entry_slippage + close_slippage,
            swap_pnl=entry_swap,
            order_id=trade.order_id,
            strategy=trade.strategy,
            magic=trade.magic,
        )
        closed_trade.exit_time = ts
        closed_trade.exit_price = float(fill_price)
        closed_trade.gross_pnl = pnl
        closed_trade.pnl = pnl - closed_trade.commission_paid - closed_trade.slippage_cost - closed_trade.spread_cost
        closed_trade.net_pnl = closed_trade.gross_pnl - (closed_trade.commission_paid + closed_trade.spread_cost + closed_trade.slippage_cost) + closed_trade.swap_pnl
        closed_trade.notes[f"exit_{reason}"] = 1.0

        trade.volume = volume - target
        trade.commission_paid -= entry_commission
        trade.spread_cost -= entry_spread
        trade.slippage_cost -= entry_slippage
        trade.swap_pnl -= entry_swap

        equity += closed_trade.pnl
        gross_equity += closed_trade.gross_pnl
        trades.append(closed_trade)
        if trade.volume <= 0:
            return equity, gross_equity, None
        return equity, gross_equity, trade

    @staticmethod
    def _points_from_price(trade: Trade, price: float, *, kind: str) -> float:
        if trade.side == "long":
            return (trade.entry_price - price) if kind == "sl" else (price - trade.entry_price)
        return (price - trade.entry_price) if kind == "sl" else (trade.entry_price - price)

    def _slippage_points(self, symbol: str, row: pd.Series) -> float:
        tick = self._tick_size(symbol)
        base = float(self.cfg.default_slippage_points) * tick
        if self.cfg.slippage_atr_multiplier > 0 and "atr" in row and not pd.isna(row["atr"]):
            base += float(row["atr"]) * self.cfg.slippage_atr_multiplier
        return base if base else 0.0

    def _tick_size(self, symbol: str) -> float:
        inst = self.instruments.get(symbol)
        if inst and inst.tick_size:
            return float(inst.tick_size)
        return 1.0

    def _apply_overnight_swap(
        self,
        symbol: str,
        trade: Trade,
        current_day: pd.Timestamp,
        swap_override: Optional[dict],
    ) -> float:
        if not self.cfg.apply_swap_cost:
            return 0.0
        if trade._last_swap_day is None:
            trade._last_swap_day = current_day
            return 0.0
        delta_total = 0.0
        while trade._last_swap_day < current_day:
            trade._last_swap_day += pd.Timedelta(days=1)
            direction = "long" if trade.side == "long" else "short"
            weekend = trade._last_swap_day.dayofweek >= 5
            override = dict(swap_override or {})
            if weekend and "weekend" not in override:
                override["weekend"] = True
            swap_delta = costs.swap_cost_per_day(symbol, trade.volume, direction, static_override=override)
            trade.swap_pnl += swap_delta
            delta_total += swap_delta
        return delta_total

    def _final_positions(self, trades: List[Trade]) -> List[Position]:
        """Aggregate closed trades into position summaries (post-run snapshot)."""

        aggregates: Dict[tuple[str, str], Dict[str, float]] = {}
        buckets: Dict[tuple[str, str], List[Trade]] = {}
        for t in trades:
            key = (t.symbol, t.side)
            if key not in aggregates:
                aggregates[key] = {"volume": 0.0, "notional": 0.0}
                buckets[key] = []
            aggregates[key]["volume"] += float(t.volume)
            aggregates[key]["notional"] += float(t.entry_price * t.volume)
            buckets[key].append(t)

        positions: List[Position] = []
        for key, agg in aggregates.items():
            symbol, side = key
            volume = agg["volume"]
            avg_price = agg["notional"] / volume if volume else 0.0
            positions.append(Position(symbol=symbol, side=side, volume=volume, avg_price=avg_price, trades=buckets[key]))
        return positions

    def _resolve_valid_until(self, strategy: BaseStrategy, signal: Signal) -> Optional[pd.Timestamp]:
        if signal.valid_until is not None:
            ts = pd.Timestamp(signal.valid_until)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if signal.tif == "GTD":
            delta = self._timeframe_to_timedelta(getattr(strategy, "timeframe", ""))
            if delta is None:
                return None
            ts = pd.Timestamp(signal.when)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            return ts + delta
        return None

    @staticmethod
    def _normalize_ts(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
        if ts is None:
            return None
        timestamp = pd.Timestamp(ts)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    @staticmethod
    def _normalize_signals(raw: Any) -> List[Signal]:
        if raw is None:
            return []
        if isinstance(raw, Signal):
            return [raw]
        return list(raw)

    @staticmethod
    def _timeframe_to_timedelta(timeframe: str) -> Optional[pd.Timedelta]:
        if not timeframe:
            return None
        try:
            if isinstance(timeframe, (int, float)):
                return pd.Timedelta(minutes=int(timeframe))
        except Exception:
            pass
        tf_str = str(timeframe)
        match = re.match(r"^\s*([0-9]+)\s*([smhdwSMHDW])\s*$", tf_str)
        if not match:
            match = re.match(r"^\s*([smhdwSMHDW])\s*([0-9]+)\s*$", tf_str)
            if match:
                match = re.match(r"^\s*([smhdwSMHDW])\s*([0-9]+)\s*$", tf_str)
        if not match:
            return None
        if len(match.groups()) == 2 and match.group(1).isdigit():
            value = int(match.group(1))
            unit = match.group(2).lower()
        else:
            unit = match.group(1).lower()
            value = int(match.group(2))
        if unit == "s":
            return pd.Timedelta(seconds=value)
        if unit == "m":
            return pd.Timedelta(minutes=value)
        if unit == "h":
            return pd.Timedelta(hours=value)
        if unit == "d":
            return pd.Timedelta(days=value)
        if unit == "w":
            return pd.Timedelta(weeks=value)
        return None

    def _compute_metrics(self, daily_equity: pd.Series, initial_equity: Optional[float] = None) -> Dict[str, float]:
        default_metrics = {
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "MaxDD": 0.0,
            "PSR": 0.0,
            "DSR": 0.0,
            "Calmar": 0.0,
            "TailRatio": 0.0,
            "ExpectedShortfall": 0.0,
            "ReturnAutocorr": 0.0,
            "Stability": 0.0,
            "Turnover": 0.0,
            "Expectancy": 0.0,
            "expectancy": 0.0,
            "ProfitFactor": 0.0,
            "profit_factor": 0.0,
            # API-facing aliases used by UI and downstream consumers.
            "NetReturn": 0.0,
            "NetReturnPct": 0.0,
            "net_return": 0.0,
            "net_return_pct": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
        }
        if daily_equity.empty or len(daily_equity) < 2:
            return default_metrics.copy()

        eq = daily_equity.sort_index().astype(float)
        returns, start_equity, include_initial_return = self._returns_for_metrics(
            eq, initial_equity=initial_equity
        )
        if returns.empty:
            return default_metrics.copy()

        ann_factor = 252
        mu = returns.mean() * ann_factor
        sigma = returns.std(ddof=1) * math.sqrt(ann_factor)
        downside = self._downside_deviation(returns, mar=0.0, annualization=ann_factor)
        sharpe = mu / sigma if sigma > 0 else 0.0
        sortino = mu / downside if downside > 0 else 0.0
        if include_initial_return:
            eq_for_drawdown = pd.concat(
                [pd.Series([start_equity], dtype=float), eq.astype(float)],
                ignore_index=True,
            )
        else:
            eq_for_drawdown = eq.astype(float)
        max_dd = self._max_drawdown(eq_for_drawdown)
        psr = self._probabilistic_sharpe(returns)
        dsr = self._deflated_sharpe(returns, sharpe)
        end_equity = float(eq.iloc[-1])
        days = (eq.index[-1] - eq.index[0]).days if len(eq.index) >= 2 else 0
        if days <= 0:
            days = max(int(len(eq_for_drawdown) - 1), 1)
        years = days / 365.25
        cagr = self._safe_cagr(start_equity, end_equity, years)
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
        tail_ratio = self._tail_ratio(returns)
        expected_shortfall = self._expected_shortfall(returns)
        autocorr = returns.autocorr(lag=1)
        if pd.isna(autocorr):
            autocorr = 0.0
        autocorr = float(autocorr)
        stability = max(0.0, min(1.0, 1.0 - abs(autocorr)))
        turnover = float(returns.abs().mean())
        net_return = (
            float((end_equity - start_equity) / abs(start_equity))
            if abs(start_equity) > 1e-12
            else 0.0
        )
        max_dd_pct = float(max_dd * 100.0)

        return {
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": max_dd,
            "PSR": psr,
            "DSR": dsr,
            "Calmar": calmar,
            "TailRatio": tail_ratio,
            "ExpectedShortfall": expected_shortfall,
            "ReturnAutocorr": autocorr,
            "Stability": stability,
            "Turnover": turnover,
            "NetReturn": net_return,
            "NetReturnPct": net_return * 100.0,
            "net_return": net_return,
            "net_return_pct": net_return * 100.0,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
        }

    @staticmethod
    def _safe_cagr(start: float, end: float, years: float) -> float:
        """Compute CAGR while avoiding complex/overflow values for pathological equity paths."""
        try:
            start_f = float(start)
            end_f = float(end)
            years_f = float(years)
        except (TypeError, ValueError):
            return 0.0
        if not (np.isfinite(start_f) and np.isfinite(end_f) and np.isfinite(years_f)):
            return 0.0
        if start_f <= 0.0 or years_f <= 0.0:
            return 0.0
        growth = end_f / start_f
        if not np.isfinite(growth):
            return 0.0
        if growth <= 0.0:
            # Non-positive terminal equity makes CAGR mathematically undefined.
            # Represent this as complete annualized loss instead of raising/complex values.
            return -1.0
        try:
            annualized_log_growth = math.log(growth) / years_f
        except (OverflowError, ValueError, ZeroDivisionError):
            return 0.0
        annualized_log_growth = float(np.clip(annualized_log_growth, -50.0, 50.0))
        return float(math.expm1(annualized_log_growth))

    @staticmethod
    def _returns_for_metrics(
        daily_equity: pd.Series,
        initial_equity: Optional[float] = None,
    ) -> tuple[pd.Series, float, bool]:
        eq = daily_equity.sort_index().astype(float)
        start_equity = float(eq.iloc[0])
        include_initial_return = False
        if initial_equity is not None:
            try:
                candidate_start = float(initial_equity)
                if np.isfinite(candidate_start) and abs(candidate_start) > 1e-12:
                    if abs(candidate_start - start_equity) > 1e-12:
                        start_equity = candidate_start
                        include_initial_return = True
            except Exception:
                pass

        returns = eq.pct_change().dropna().astype(float)
        if include_initial_return and abs(start_equity) > 1e-12:
            first_return = float((float(eq.iloc[0]) - start_equity) / start_equity)
            returns = pd.concat(
                [pd.Series([first_return], dtype=float), returns],
                ignore_index=True,
            )
        return returns, start_equity, include_initial_return

    @staticmethod
    def _downside_deviation(returns: pd.Series, mar: float = 0.0, annualization: float = 252.0) -> float:
        """Downside deviation used by Sortino ratio.

        Uses the standard semideviation definition:
            sqrt(mean(min(r - mar, 0)^2)) * sqrt(annualization)
        """
        if returns is None:
            return 0.0
        clean = pd.Series(returns, copy=False).replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        downside = np.minimum(clean.to_numpy(dtype=float) - float(mar), 0.0)
        if downside.size == 0:
            return 0.0
        semivariance = float(np.mean(np.square(downside)))
        if not np.isfinite(semivariance) or semivariance <= 0.0:
            return 0.0
        scale = math.sqrt(float(annualization)) if annualization and annualization > 0 else 1.0
        return float(math.sqrt(semivariance) * scale)

    @staticmethod
    def _tail_ratio(returns: pd.Series, upper_q: float = 0.95, lower_q: float = 0.05) -> float:
        clean = pd.Series(returns, copy=False).replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        upper = float(clean.quantile(upper_q))
        lower = float(clean.quantile(lower_q))
        if not np.isfinite(upper) or not np.isfinite(lower):
            return 0.0
        if upper <= 0.0 or lower >= 0.0:
            return 0.0
        denom = abs(lower)
        if denom <= 0:
            return 0.0
        return float(upper) / denom

    @staticmethod
    def _expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
        if returns.empty:
            return 0.0
        threshold = returns.quantile(alpha)
        tail = returns[returns <= threshold]
        if tail.empty:
            return 0.0
        return float(-tail.mean())

    @staticmethod
    def _evaluate_robust_score(
        metrics: Dict[str, float],
        fees_to_gross: float,
        *,
        returns_count: int = 0,
        trade_count: int = 0,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, Dict[str, float], bool, List[str], Optional[Dict[str, Any]]]:
        sharpe = float(metrics.get("Sharpe") or 0.0)
        sortino = float(metrics.get("Sortino") or 0.0)
        max_dd = float(metrics.get("MaxDD") or 0.0)
        tail_ratio = float(metrics.get("TailRatio") or 0.0)
        stability = float(metrics.get("Stability") or 0.0)
        turnover = float(metrics.get("Turnover") or 0.0)
        psr = float(metrics.get("PSR") or 0.0)

        thresholds = thresholds or {}
        drawdown_limit = thresholds.get("drawdown_threshold")
        psr_threshold = thresholds.get("psr_threshold")
        fees_limit = thresholds.get("fees_to_gross")
        turnover_threshold = thresholds.get("turnover_threshold", 0.15) or 0.15
        sharpe_target = float(thresholds.get("sharpe_target", 3.0) or 3.0)
        sortino_target = float(thresholds.get("sortino_target", 4.0) or 4.0)
        tail_ratio_target = float(thresholds.get("tail_ratio_target", 3.0) or 3.0)
        min_psr_trades = thresholds.get("psr_min_trades")
        min_psr_returns = thresholds.get("psr_min_returns")
        min_psr_observations = thresholds.get("psr_min_observations")
        ignore_disqualification = bool(thresholds.get("ignore_disqualification", False))
        drawdown_target = drawdown_limit if drawdown_limit is not None else 0.5

        def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
            return max(lower, min(upper, value))

        def _normalize_positive(value: float, upper: float) -> float:
            if upper <= 0:
                return 0.0
            if value <= 0:
                return 0.0
            return _clamp(value / upper)

        sharpe_score = _normalize_positive(sharpe, sharpe_target)
        sortino_score = _normalize_positive(sortino, sortino_target)
        drawdown_penalty = _clamp(abs(max_dd) / drawdown_target) if drawdown_target > 0 else 1.0
        drawdown_score = _clamp(1.0 - drawdown_penalty)
        tail_score = _normalize_positive(tail_ratio, tail_ratio_target)
        stability_score = _clamp(stability)
        turnover_penalty = _clamp(turnover / turnover_threshold) if turnover_threshold > 0 else 1.0
        turnover_score = _clamp(1.0 - turnover_penalty)

        robust_score = (
            0.35 * sharpe_score
            + 0.15 * sortino_score
            + 0.15 * drawdown_score
            + 0.10 * tail_score
            + 0.10 * stability_score
            + 0.15 * turnover_score
        )
        robust_score = _clamp(robust_score)

        psr_guardrail: Optional[Dict[str, Any]] = None
        if (
            min_psr_returns is not None
            or min_psr_trades is not None
            or min_psr_observations is not None
        ):
            guardrail_min_returns = int(min_psr_returns or 0)
            guardrail_min_trades = int(min_psr_trades or 0)
            guardrail_min_observations = int(min_psr_observations or 0)
            psr_guardrail = {
                "eligible": bool(
                    returns_count >= guardrail_min_returns
                    and trade_count >= guardrail_min_trades
                    and returns_count >= guardrail_min_observations
                ),
                "min_returns": guardrail_min_returns,
                "min_trades": guardrail_min_trades,
                "min_observations": guardrail_min_observations,
                "actual_returns": int(returns_count),
                "actual_trades": int(trade_count),
                "actual_observations": int(returns_count),
            }

        disqualify_reasons: List[str] = []
        if drawdown_limit is not None and abs(max_dd) > 1e-9 and abs(max_dd) > drawdown_limit:
            disqualify_reasons.append("drawdown")
        psr_eligible = psr_guardrail["eligible"] if psr_guardrail is not None else True
        if psr_threshold is not None and psr < psr_threshold and psr_eligible:
            disqualify_reasons.append("psr")
        if fees_limit is not None and fees_to_gross > fees_limit:
            disqualify_reasons.append("fees")

        if psr_guardrail is not None and not psr_eligible:
            psr_guardrail["reason"] = "insufficient_samples"
            disqualify_reasons.append("psr_insufficient_samples")

        disqualified = bool(disqualify_reasons)
        if disqualified and not ignore_disqualification:
            robust_score = 0.0

        components = {
            "sharpe": sharpe_score,
            "sortino": sortino_score,
            "drawdown": drawdown_score,
            "tail_ratio": tail_score,
            "stability": stability_score,
            "turnover": turnover_score,
        }

        return robust_score, components, disqualified, disqualify_reasons, psr_guardrail

    @staticmethod
    def _cagr(eq: pd.Series) -> float:
        if eq is None or eq.empty:
            return 0.0
        start = eq.iloc[0]
        end = eq.iloc[-1]
        days = (eq.index[-1] - eq.index[0]).days or 1
        years = days / 365.25
        return Backtester._safe_cagr(float(start), float(end), float(years))

    @staticmethod
    def _max_drawdown(eq: pd.Series) -> float:
        if eq is None:
            return 0.0
        clean = (
            pd.Series(eq, copy=False)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if clean.empty:
            return 0.0
        running_max = clean.cummax()
        denominator = running_max.where(running_max.abs() > 1e-12, np.nan)
        drawdown = (clean - running_max) / denominator
        if drawdown.empty:
            return 0.0
        minimum = float(drawdown.min(skipna=True))
        if not np.isfinite(minimum):
            return 0.0
        return minimum if minimum < 0 else 0.0

    @staticmethod
    def _probabilistic_sharpe(
        returns: pd.Series,
        benchmark: float = 0.0,
        min_observations: int = 20,
    ) -> float:
        values = returns.values
        n = len(values)
        try:
            min_required = int(min_observations)
        except (TypeError, ValueError):
            min_required = 20
        min_required = max(2, min_required)
        if n < min_required:
            return 0.0
        mean = values.mean()
        sd = values.std(ddof=1)
        if sd <= 0:
            return 0.0
        sr_hat = (mean / sd) * math.sqrt(252)
        series = pd.Series(values)
        skew = series.skew()
        kurt = series.kurtosis()
        se_term = (1 + 0.5 * sr_hat**2 - skew * sr_hat + (kurt / 4) * sr_hat**2) / (n - 1)
        if not np.isfinite(se_term) or se_term <= 0:
            return 0.0
        se = math.sqrt(se_term)
        if se <= 0:
            return 0.0
        z = (sr_hat - benchmark) / se
        from math import erf, sqrt

        return 0.5 * (1 + erf(z / sqrt(2)))

    @staticmethod
    def _deflated_sharpe(returns: pd.Series, sharpe: float) -> float:
        values = returns.values
        n = len(values)
        if n < 20 or sharpe == 0:
            return 0.0
        series = pd.Series(values)
        skew = series.skew()
        kurt = series.kurtosis()
        term = math.sqrt(max(1e-9, 1 - skew * sharpe + (kurt - 1) * (sharpe ** 2) / 4))
        z = sharpe * math.sqrt(n - 1) / term
        from math import erf, sqrt

        return 0.5 * (1 + erf(z / sqrt(2)))
