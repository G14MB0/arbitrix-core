from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from arbitrix_core.backtest.bar_view import BarViewSource
from arbitrix_core.data.market import FeedKey, MarketBars, MarketFrames, PreparedMarket
from arbitrix_core.portfolio import Portfolio
from arbitrix_core.strategies.base import invoke_strategy_prepare
from arbitrix_core.trading import Order, Trade

if TYPE_CHECKING:
    from arbitrix_core.backtest.engine import BTResult, Backtester
    from arbitrix_core.strategies.base import BaseStrategy


_MARKET_COLUMNS = (
    "spread",
    "__regime_output__",
    "__account_point_value__",
)
_OHLC_COLUMNS = frozenset({"open", "high", "low", "close"})
logger = logging.getLogger(__name__)


def _validate_utc_frame(
    key: FeedKey,
    frame: pd.DataFrame,
    *,
    require_ohlc: bool,
    stage: str,
) -> pd.DatetimeIndex:
    if frame.empty:
        raise ValueError(f"{stage} feed {key!r} is empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{stage} feed {key!r} must use a DatetimeIndex")
    if frame.index.tz is None or str(frame.index.tz).upper() != "UTC":
        raise ValueError(f"{stage} feed {key!r} must use a UTC index")
    if frame.index.has_duplicates:
        raise ValueError(f"{stage} feed {key!r} contains duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{stage} feed {key!r} must be sorted by timestamp")
    if require_ohlc:
        missing = sorted(_OHLC_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{stage} feed {key!r} is missing OHLC columns: {missing}")
    return frame.index


def _validate_source_market(market: MarketFrames, strategy: BaseStrategy) -> None:
    if not isinstance(market, MarketFrames):
        raise TypeError("Backtester.run_market() requires a MarketFrames value")
    if market.primary not in market.execution_keys:
        raise ValueError("the primary feed must be an execution feed")
    if any(key.timeframe != market.primary.timeframe for key in market):
        raise ValueError("all V1 market feeds must use the same timeframe")
    if getattr(strategy, "timeframe", None) not in (None, "", market.primary.timeframe):
        raise ValueError("strategy timeframe must match the MarketFrames primary timeframe")
    if getattr(strategy, "symbol", None) not in (None, "", market.primary.symbol):
        raise ValueError("strategy symbol must match the MarketFrames primary symbol")

    primary_index = _validate_utc_frame(
        market.primary,
        market.primary_frame,
        require_ohlc=True,
        stage="source",
    )
    for key, frame in market.items():
        index = _validate_utc_frame(
            key,
            frame,
            require_ohlc=True,
            stage="source",
        )
        if not index.equals(primary_index):
            raise ValueError("all V1 source feeds must have identical decision timestamps")


def _validate_prepared_market(
    market: MarketFrames,
    prepared: Any,
) -> PreparedMarket:
    if not isinstance(prepared, PreparedMarket):
        raise ValueError("Strategy.prepare() must return PreparedMarket when run_market() is used")
    if prepared.primary != market.primary:
        raise ValueError("PreparedMarket.primary must match MarketFrames.primary")
    if set(prepared) != set(market):
        raise ValueError("PreparedMarket must preserve the exact MarketFrames feed keys")

    primary_index = _validate_utc_frame(
        prepared.primary,
        prepared.primary_frame,
        require_ohlc=True,
        stage="prepared",
    )
    for key, frame in prepared.items():
        index = _validate_utc_frame(
            key,
            frame,
            require_ohlc=key in market.execution_keys,
            stage="prepared",
        )
        if not index.equals(primary_index):
            raise ValueError("all V1 prepared feeds must have identical decision timestamps")
        if not index.isin(market[key].index).all():
            raise ValueError(f"prepared feed {key!r} contains decision timestamps absent from source")
    return prepared


def _preserve_execution_columns(
    backtester: Backtester,
    market: MarketFrames,
    prepared: PreparedMarket,
) -> PreparedMarket:
    frames: dict[FeedKey, pd.DataFrame] = {}
    for key, frame in prepared.items():
        columns = _MARKET_COLUMNS if key in market.execution_keys else ("__regime_output__",)
        frames[key] = backtester._preserve_prepared_columns(
            market[key],
            frame,
            columns=columns,
        )
    return PreparedMarket(frames, primary=prepared.primary)


def _apply_warmup_and_window(
    prepared: PreparedMarket,
    *,
    warmup_bars: int,
    start_filter: pd.Timestamp | None,
) -> PreparedMarket:
    frames = dict(prepared)
    if warmup_bars > 0 and start_filter is None:
        if len(prepared.primary_frame) < warmup_bars:
            raise ValueError(
                f"Strategy produced insufficient PreparedMarket data for warmup_bars={warmup_bars}"
            )
        drop_bars = max(0, warmup_bars - 1)
        if drop_bars:
            frames = {key: frame.iloc[drop_bars:] for key, frame in frames.items()}
    if start_filter is not None:
        frames = {key: frame.loc[frame.index >= start_filter] for key, frame in frames.items()}
        if frames[prepared.primary].empty:
            raise ValueError(f"No prepared market data available at or after {start_filter.isoformat()}")
    return PreparedMarket(frames, primary=prepared.primary)


def _rows_for_position(
    prepared: PreparedMarket,
    position: int,
    *,
    row_mode: str,
    bar_view_sources: Mapping[FeedKey, BarViewSource],
) -> dict[FeedKey, Any]:
    if row_mode == "bar_view":
        return {key: bar_view_sources[key].row_at(position) for key in prepared}
    return {key: frame.iloc[position] for key, frame in prepared.items()}


def _unrealized_for_market(
    backtester: Backtester,
    trades: Iterable[Trade],
    rows_by_symbol: Mapping[str, Any],
) -> float:
    grouped: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.symbol].append(trade)
    return sum(
        backtester._unrealized_pnl(symbol, symbol_trades, rows_by_symbol[symbol])
        for symbol, symbol_trades in grouped.items()
    )


def _fill_ready_market_orders_at_open(
    backtester: Backtester,
    *,
    rows_by_symbol: Mapping[str, Any],
    ts: pd.Timestamp,
    open_trades: list[Trade],
    working_orders: list[Order],
    equity: float,
) -> tuple[float, list[Trade], list[Order], list[Trade], list[Order]]:
    if backtester.cfg.market_fill_price != "next_open" or not working_orders:
        return equity, open_trades, working_orders, [], []
    current_ts = backtester._normalize_ts(ts)
    if current_ts is None:
        return equity, open_trades, working_orders, [], []

    remaining_orders: list[Order] = []
    newly_filled: list[Trade] = []
    filled_orders: list[Order] = []
    for order in working_orders:
        if order.type != "market":
            remaining_orders.append(order)
            continue
        created_at = backtester._order_created_at(order)
        if created_at is not None and created_at > current_ts:
            order.status = "working"
            remaining_orders.append(order)
            continue
        row = rows_by_symbol[order.symbol]
        filled = backtester._try_fill_order(order, row)
        if filled is None:
            remaining_orders.append(order)
            continue
        fill_time = created_at if created_at is not None and created_at == current_ts else current_ts
        trade, equity = backtester._open_trade_from_order(
            order.symbol,
            order,
            filled,
            row,
            fill_time,
            equity,
        )
        if trade is not None:
            open_trades.append(trade)
            newly_filled.append(trade)
            filled_orders.append(order)
    return equity, open_trades, remaining_orders, newly_filled, filled_orders


def _check_market_stops(
    backtester: Backtester,
    *,
    rows_by_symbol: Mapping[str, Any],
    ts: pd.Timestamp,
    open_trades: list[Trade],
    equity: float,
    gross_equity: float,
    closed_trades: list[Trade],
) -> tuple[list[Trade], float, float]:
    grouped: dict[str, list[Trade]] = defaultdict(list)
    for trade in open_trades:
        grouped[trade.symbol].append(trade)
    surviving_ids: set[str] = set()
    for symbol, symbol_trades in grouped.items():
        survivors, equity, gross_equity = backtester._check_stops_vectorized(
            symbol,
            symbol_trades,
            rows_by_symbol[symbol],
            ts,
            equity,
            gross_equity,
            closed_trades,
        )
        surviving_ids.update(trade.id for trade in survivors)
    return (
        [trade for trade in open_trades if trade.id in surviving_ids],
        equity,
        gross_equity,
    )


def _fill_working_orders(
    backtester: Backtester,
    *,
    rows_by_symbol: Mapping[str, Any],
    ts: pd.Timestamp,
    working_orders: list[Order],
    equity: float,
) -> tuple[float, list[Order], list[Trade], list[Order]]:
    newly_filled: list[Trade] = []
    filled_orders: list[Order] = []
    remaining_orders: list[Order] = []
    for order in working_orders:
        created_at = backtester._order_created_at(order)
        if (
            (order.type != "market" or backtester.cfg.market_fill_price == "next_open")
            and created_at is not None
            and created_at > ts
        ):
            order.status = "working"
            remaining_orders.append(order)
            continue
        row = rows_by_symbol[order.symbol]
        filled = backtester._try_fill_order(order, row)
        if filled is None:
            remaining_orders.append(order)
            continue
        if order.type == "market" and created_at is not None:
            if backtester.cfg.market_fill_price == "close":
                fill_time = created_at
            else:
                fill_time = created_at if created_at == ts else ts
        else:
            fill_time = ts
        trade, equity = backtester._open_trade_from_order(
            order.symbol,
            order,
            filled,
            row,
            fill_time,
            equity,
        )
        if trade is not None:
            newly_filled.append(trade)
            filled_orders.append(order)
    return equity, remaining_orders, newly_filled, filled_orders


def _build_result(
    backtester: Backtester,
    *,
    closed_trades: list[Trade],
    all_orders: list[Order],
    margin_call_events: list[Any],
    daily_equity: pd.Series,
    gross_equity: pd.Series,
    equity_marked: pd.Series,
    initial_equity: float,
    early_stop_conditions: Mapping[str, Any] | None,
    signal_intents: list[dict[str, Any]] | None,
    early_stopped: bool,
    early_stop_reason: str | None,
    bar_count: int,
    prepared_snapshot: PreparedMarket | None,
    primary_symbol: str,
    execution_symbols: Iterable[str],
    feed_inventory: Iterable[Mapping[str, Any]],
    collect_diagnostics: bool,
    prepare_elapsed: float,
    loop_elapsed: float,
    run_started: float,
) -> BTResult:
    from arbitrix_core.backtest.engine import BTResult

    finalize_started = time.monotonic()
    early_stop_flag = bool(
        early_stop_conditions.get("enabled", True) if isinstance(early_stop_conditions, dict) else False
    )
    metrics, metadata = backtester._build_result_payload(
        daily_equity=daily_equity,
        initial_equity=initial_equity,
        closed_trades=closed_trades,
        early_stop_conditions=early_stop_conditions,
        early_stop_flag=early_stop_flag,
        signal_intents=signal_intents,
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        bar_count=bar_count,
    )
    metadata["market_data_mode"] = "provider_symbol"
    metadata["primary_symbol"] = primary_symbol
    metadata["execution_symbols"] = [str(symbol) for symbol in execution_symbols]
    metadata["market_feeds"] = [dict(feed) for feed in feed_inventory]
    finalize_elapsed = max(0.0, time.monotonic() - finalize_started)
    metadata["runtime_timing"] = {
        "prepare_s": float(prepare_elapsed),
        "loop_s": float(loop_elapsed),
        "finalize_s": float(finalize_elapsed),
        "total_s": float(max(0.0, time.monotonic() - run_started)),
        "loop_bar_count": int(bar_count),
        "loop_per_bar_ms": float(loop_elapsed / bar_count * 1000.0 if bar_count else 0.0),
    }

    return BTResult(
        trades=closed_trades,
        daily_equity=daily_equity,
        gross_equity=gross_equity,
        equity_marked=equity_marked,
        metrics=metrics,
        metadata=metadata,
        orders=all_orders if collect_diagnostics else [],
        positions=(backtester._final_positions(closed_trades) if collect_diagnostics else []),
        prepared=(prepared_snapshot.primary_frame if prepared_snapshot is not None else None),
        margin_call_events=margin_call_events,
        prepared_market=prepared_snapshot,
    )


def run_market(
    backtester: Backtester,
    market: MarketFrames,
    strategy: BaseStrategy,
    risk_perc: float,
    initial_equity: float,
    swap_override: Optional[dict] = None,
    *,
    cancel_callback: Optional[Callable[[], None]] = None,
    early_stop_conditions: Optional[dict[str, Any]] = None,
    window_start: Optional[datetime] = None,
    capture_prepared: bool = False,
    collect_diagnostics: bool = True,
    bar_observer: Optional[Callable[[dict[str, Any]], None]] = None,
) -> BTResult:
    """Run one strategy against an exact provider-symbol candle barrier.

    This is an additive path. ``Backtester.run_single`` remains the unchanged
    mono-provider engine; callers opt into this contract by supplying
    :class:`MarketFrames` and a strategy returning :class:`PreparedMarket`.
    """
    from arbitrix_core.backtest.engine import (
        _audit_row_payload,
        _audit_signal_payload,
        _backtest_audit_writer,
        _invoke_strategy_on_bar,
        _invoke_strategy_should_exit_trade,
    )

    run_started = time.monotonic()

    def maybe_cancel() -> None:
        if cancel_callback is not None:
            cancel_callback()

    maybe_cancel()
    _validate_source_market(market, strategy)
    start_filter: pd.Timestamp | None = None
    if window_start is not None:
        start_filter = pd.Timestamp(window_start)
        start_filter = (
            start_filter.tz_localize("UTC") if start_filter.tzinfo is None else start_filter.tz_convert("UTC")
        )

    portfolio = Portfolio(initial_equity=initial_equity, equity_source="backtest")
    portfolio.max_margin_utilization = backtester.cfg.max_margin_utilization
    strategy.portfolio = portfolio
    strategy_name = getattr(strategy, "name", "") or strategy.__class__.__name__
    audit_write = _backtest_audit_writer(strategy_name)

    def write_audit(payload: Mapping[str, Any]) -> None:
        if audit_write is None:
            return
        try:
            audit_write(dict(payload))
        except Exception:
            logger.debug(
                "Failed to write multiprovider backtest audit for %s",
                strategy_name,
                exc_info=True,
            )

    warmup_bars = int(getattr(strategy, "warmup_bars", lambda: 0)() or 0)
    strict_prepare_window = str(
        os.getenv("ARBITRIX_BACKTEST_STRICT_PREPARE_WINDOW", "") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    rolling_prepare = warmup_bars > 1 and strict_prepare_window

    prepare_started = time.monotonic()
    prepared_market: PreparedMarket | None = None
    rolling_source_positions: list[int] = []
    if rolling_prepare:
        for source_position, ts in enumerate(market.primary_frame.index):
            if source_position + 1 < warmup_bars:
                continue
            if start_filter is not None and ts < start_filter:
                continue
            rolling_source_positions.append(source_position)
        if not rolling_source_positions:
            raise ValueError(f"No market data available with warmup_bars={warmup_bars}")
        loop_index = pd.DatetimeIndex(
            [market.primary_frame.index[position] for position in rolling_source_positions]
        )
    else:
        prepared = invoke_strategy_prepare(strategy, market.primary_frame, data=market)
        maybe_cancel()
        prepared_market = _validate_prepared_market(market, prepared)
        prepared_market = _preserve_execution_columns(
            backtester,
            market,
            prepared_market,
        )
        prepared_market = _apply_warmup_and_window(
            prepared_market,
            warmup_bars=warmup_bars,
            start_filter=start_filter,
        )
        _validate_prepared_market(market, prepared_market)
        loop_index = prepared_market.primary_frame.index
    prepare_elapsed = max(0.0, time.monotonic() - prepare_started)
    prepared_snapshot = (
        PreparedMarket(
            {key: frame.copy() for key, frame in prepared_market.items()},
            primary=prepared_market.primary,
        )
        if capture_prepared and prepared_market is not None
        else None
    )
    rolling_snapshot_rows: dict[FeedKey, list[pd.Series]] = {key: [] for key in market}

    if audit_write is not None:
        write_audit(
            {
                "event": "backtest_prepare",
                "wall_ts": datetime.now(timezone.utc),
                "strategy": strategy.__class__.__name__,
                "strategy_name": strategy_name,
                "symbol": market.primary.symbol,
                "timeframe": market.primary.timeframe,
                "feeds": [
                    {
                        "provider": key.provider,
                        "symbol": key.symbol,
                        "timeframe": key.timeframe,
                    }
                    for key in market
                ],
                "source_rows": len(market.primary_frame),
                "prepared_rows": len(loop_index),
                "prepare_window_mode": "warmup" if rolling_prepare else "vectorized",
            }
        )

    execution_keys_by_symbol = {key.symbol: key for key in market.execution_keys}
    execution_symbols = tuple(execution_keys_by_symbol)
    for symbol, key in execution_keys_by_symbol.items():
        startup_frame = market[key] if prepared_market is None else prepared_market[key]
        backtester._validate_target_spread_at_startup(symbol, startup_frame)

    equity = float(initial_equity)
    gross_equity_value = float(initial_equity)
    open_trades: list[Trade] = []
    closed_trades: list[Trade] = []
    working_orders: list[Order] = []
    all_orders: list[Order] = []
    margin_call_events: list[Any] = []
    signal_intents: list[dict[str, Any]] | None = [] if collect_diagnostics else None
    equity_by_day: dict[pd.Timestamp, float] = {}
    gross_by_day: dict[pd.Timestamp, float] = {}
    marked_by_day: dict[pd.Timestamp, float] = {}

    early_settings = early_stop_conditions or {}
    early_flag = bool(early_settings.get("enabled", True)) if early_stop_conditions else False
    max_dd_threshold = early_settings.get("max_drawdown") if early_flag else None
    min_trades_threshold = early_settings.get("min_trades") if early_flag else None
    check_interval = int(early_settings.get("check_interval", 50)) if early_flag else 50
    early_stop_enabled = bool(
        early_flag
        and (max_dd_threshold is not None or (min_trades_threshold is not None and min_trades_threshold > 0))
    )
    running_peak_equity = float(initial_equity)
    running_max_drawdown = 0.0
    early_stopped = False
    early_stop_reason: str | None = None
    bar_count = 0

    row_mode = str(getattr(backtester.cfg, "row_mode", "series") or "series").lower()
    if row_mode not in {"series", "bar_view"}:
        raise ValueError("BTConfig.row_mode must be 'series' or 'bar_view'.")
    bar_view_sources = (
        {key: BarViewSource(frame) for key, frame in prepared_market.items()}
        if row_mode == "bar_view" and prepared_market is not None
        else {}
    )
    loop_days = loop_index.normalize()
    cancel_interval = 1 if collect_diagnostics else 64
    last_rows_by_symbol: dict[str, Any] | None = None
    last_ts: pd.Timestamp | None = None

    loop_started = time.monotonic()
    for loop_position, ts in enumerate(loop_index):
        if loop_position % cancel_interval == 0:
            maybe_cancel()
        bar_count += 1
        day = loop_days[loop_position]
        if rolling_prepare:
            source_position = rolling_source_positions[loop_position]
            window_frames = {
                key: frame.iloc[source_position - warmup_bars + 1 : source_position + 1]
                for key, frame in market.items()
            }
            window_market = MarketFrames(
                window_frames,
                primary=market.primary,
                execution_keys=market.execution_keys,
            )
            current_prepared = invoke_strategy_prepare(
                strategy,
                window_market.primary_frame,
                data=window_market,
            )
            current_prepared = _validate_prepared_market(
                window_market,
                current_prepared,
            )
            current_prepared = _preserve_execution_columns(
                backtester,
                window_market,
                current_prepared,
            )
            _validate_prepared_market(window_market, current_prepared)
            if current_prepared.primary_frame.index[-1] != ts:
                raise ValueError("rolling PreparedMarket must end at the current decision timestamp")
            rows_by_key = {key: frame.iloc[-1] for key, frame in current_prepared.items()}
            if capture_prepared:
                for key, row in rows_by_key.items():
                    rolling_snapshot_rows[key].append(row.copy())
        else:
            assert prepared_market is not None
            rows_by_key = _rows_for_position(
                prepared_market,
                loop_position,
                row_mode=row_mode,
                bar_view_sources=bar_view_sources,
            )
        bars = MarketBars(
            rows_by_key,
            primary=market.primary,
            decision_time=ts,
        )
        rows_by_symbol = {symbol: rows_by_key[key] for symbol, key in execution_keys_by_symbol.items()}
        last_rows_by_symbol = rows_by_symbol
        last_ts = ts

        working_orders = backtester._expire_orders_before_bar(working_orders, ts)
        (
            equity,
            open_trades,
            working_orders,
            filled_at_open,
            filled_orders_at_open,
        ) = _fill_ready_market_orders_at_open(
            backtester,
            rows_by_symbol=rows_by_symbol,
            ts=ts,
            open_trades=open_trades,
            working_orders=working_orders,
            equity=equity,
        )

        for trade in open_trades:
            swap_delta = backtester._apply_overnight_swap(
                trade.symbol,
                trade,
                rows_by_symbol[trade.symbol],
                day,
                swap_override,
            )
            equity += swap_delta

        open_trades, equity, gross_equity_value = _check_market_stops(
            backtester,
            rows_by_symbol=rows_by_symbol,
            ts=ts,
            open_trades=open_trades,
            equity=equity,
            gross_equity=gross_equity_value,
            closed_trades=closed_trades,
        )
        conditional_survivors: list[Trade] = []
        for trade in open_trades:
            trade_row = rows_by_symbol[trade.symbol]
            if _invoke_strategy_should_exit_trade(
                strategy,
                trade,
                trade_row,
                symbol=trade.symbol,
                data=bars,
            ):
                equity, gross_equity_value, _ = backtester._close_trade(
                    trade.symbol,
                    trade,
                    trade_row,
                    ts,
                    equity,
                    gross_equity_value,
                    closed_trades,
                    reason="conditional_exit",
                )
            else:
                conditional_survivors.append(trade)
        open_trades = conditional_survivors
        unrealized_before = _unrealized_for_market(
            backtester,
            open_trades,
            rows_by_symbol,
        )
        portfolio.sync(
            timestamp=ts,
            equity=equity,
            gross_equity=gross_equity_value,
            equity_marked=equity + unrealized_before,
            open_trades=open_trades,
            closed_trades=closed_trades,
            pending_orders=working_orders,
            orders=all_orders,
        )
        margin_call_events.extend(portfolio.check_maintenance_margin(ts=ts))

        primary_row = bars.primary_bar
        regime_output = primary_row.get("__regime_output__")
        raw_signals = _invoke_strategy_on_bar(
            strategy,
            primary_row,
            portfolio,
            regime_output,
            symbol=market.primary.symbol,
            data=bars,
        )
        bar_signals = backtester._normalize_signals(raw_signals)
        if audit_write is not None:
            write_audit(
                {
                    "event": "bar_decision",
                    "wall_ts": datetime.now(timezone.utc),
                    "run_phase": "backtest",
                    "strategy": strategy.__class__.__name__,
                    "strategy_name": strategy_name,
                    "symbol": market.primary.symbol,
                    "timeframe": market.primary.timeframe,
                    "bar_ts": ts,
                    "signals_count": len(bar_signals),
                    "signals": [_audit_signal_payload(signal) for signal in bar_signals],
                    "row": _audit_row_payload(primary_row),
                }
            )

        order_count_before = len(all_orders)
        equity, gross_equity_value, open_trades, working_orders = backtester._apply_bar_signals(
            strategy=strategy,
            signals=bar_signals,
            row=primary_row,
            ts=ts,
            symbol=market.primary.symbol,
            risk_perc=risk_perc,
            open_trades=open_trades,
            closed_trades=closed_trades,
            working_orders=working_orders,
            all_orders=all_orders,
            equity=equity,
            gross_equity=gross_equity_value,
            signal_intents=signal_intents,
            rows_by_symbol=rows_by_symbol,
            execution_symbols=execution_symbols,
            data=bars,
        )
        newly_created_orders = list(all_orders[order_count_before:])
        working_orders = backtester._expire_orders_before_bar(working_orders, ts)
        equity, remaining_orders, newly_filled, newly_filled_orders = _fill_working_orders(
            backtester,
            rows_by_symbol=rows_by_symbol,
            ts=ts,
            working_orders=working_orders,
            equity=equity,
        )
        working_orders = backtester._expire_orders_after_bar(remaining_orders, ts)
        open_trades.extend(newly_filled)

        running_peak_equity = max(running_peak_equity, equity)
        if running_peak_equity > 0.0:
            running_max_drawdown = min(
                running_max_drawdown,
                (equity - running_peak_equity) / running_peak_equity,
            )
        equity_by_day[day] = equity
        gross_by_day[day] = gross_equity_value
        marked_by_day[day] = equity + _unrealized_for_market(
            backtester,
            open_trades,
            rows_by_symbol,
        )

        if bar_observer is not None:
            bar_observer(
                {
                    "bar_ts": ts,
                    "row": primary_row,
                    "data": bars,
                    "rows_by_symbol": dict(rows_by_symbol),
                    "portfolio": portfolio,
                    "open_trades": list(open_trades),
                    "closed_trades": list(closed_trades),
                    "working_orders": list(working_orders),
                    "equity": float(equity),
                    "gross_equity": float(gross_equity_value),
                    "bar_signals": list(bar_signals),
                    "newly_created_orders": newly_created_orders,
                    "newly_filled": list(filled_at_open) + list(newly_filled),
                    "newly_filled_orders": (list(filled_orders_at_open) + list(newly_filled_orders)),
                }
            )

        if early_stop_enabled and bar_count % check_interval == 0:
            if max_dd_threshold is not None and abs(running_max_drawdown) > max_dd_threshold:
                early_stopped = True
                early_stop_reason = (
                    f"max_drawdown exceeded: {abs(running_max_drawdown):.4f} > {max_dd_threshold}"
                )
                break
            if (
                min_trades_threshold is not None
                and bar_count > len(loop_index) * 0.3
                and len(closed_trades) < min_trades_threshold
            ):
                early_stopped = True
                early_stop_reason = f"insufficient trades: {len(closed_trades)} < {min_trades_threshold}"
                break
    loop_elapsed = max(0.0, time.monotonic() - loop_started)

    if rolling_prepare and capture_prepared:
        snapshot_frames: dict[FeedKey, pd.DataFrame] = {}
        for key, rows in rolling_snapshot_rows.items():
            snapshot_frame = pd.DataFrame(rows)
            snapshot_frame.index = pd.DatetimeIndex([row.name for row in rows])
            snapshot_frames[key] = snapshot_frame
        prepared_snapshot = PreparedMarket(
            snapshot_frames,
            primary=market.primary,
        )

    if last_ts is None or last_rows_by_symbol is None:
        raise ValueError("PreparedMarket produced no decision bars")
    last_day = last_ts.normalize()
    working_orders = backtester._expire_orders_after_bar(
        working_orders,
        last_ts,
        expire_unfilled_market=True,
    )
    exit_reason = "early_stop" if early_stopped else "eod"
    for trade in list(open_trades):
        row = last_rows_by_symbol[trade.symbol]
        if not early_stopped:
            equity += backtester._apply_overnight_swap(
                trade.symbol,
                trade,
                row,
                last_day,
                swap_override,
            )
        equity, gross_equity_value, _ = backtester._close_trade(
            trade.symbol,
            trade,
            row,
            last_ts,
            equity,
            gross_equity_value,
            closed_trades,
            reason=exit_reason,
        )
    open_trades = []
    equity_by_day[last_day] = equity
    gross_by_day[last_day] = gross_equity_value
    marked_by_day[last_day] = equity
    portfolio.sync(
        timestamp=last_ts,
        equity=equity,
        gross_equity=gross_equity_value,
        equity_marked=equity,
        open_trades=open_trades,
        closed_trades=closed_trades,
        pending_orders=working_orders,
        orders=all_orders,
    )

    maybe_cancel()
    daily_equity = pd.Series(equity_by_day).sort_index().ffill()
    gross_equity = pd.Series(gross_by_day).sort_index().ffill()
    if gross_equity.empty and not daily_equity.empty:
        gross_equity = daily_equity.copy()
    equity_marked = pd.Series(marked_by_day).sort_index().ffill()
    return _build_result(
        backtester,
        closed_trades=closed_trades,
        all_orders=all_orders,
        margin_call_events=margin_call_events,
        daily_equity=daily_equity,
        gross_equity=gross_equity,
        equity_marked=equity_marked,
        initial_equity=initial_equity,
        early_stop_conditions=early_stop_conditions,
        signal_intents=signal_intents,
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        bar_count=bar_count,
        prepared_snapshot=prepared_snapshot,
        primary_symbol=market.primary.symbol,
        execution_symbols=(key.symbol for key in market.execution_keys),
        feed_inventory=(
            {
                "provider": key.provider,
                "symbol": key.symbol,
                "timeframe": key.timeframe,
                "primary": key == market.primary,
                "execution": key in market.execution_keys,
            }
            for key in market
        ),
        collect_diagnostics=collect_diagnostics,
        prepare_elapsed=prepare_elapsed,
        loop_elapsed=loop_elapsed,
        run_started=run_started,
    )


__all__ = ["run_market"]
