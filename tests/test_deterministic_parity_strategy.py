from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytest

from arbitrix_core.backtest.engine import BTConfig, Backtester
from arbitrix_core.data.market import FeedKey, MarketFrames
from arbitrix_core.strategies.deterministic_parity import (
    DeterministicMultiMarketParityStrategy,
)
from arbitrix_core.types import InstrumentConfig

START = pd.Timestamp("2026-08-30T12:00:00Z")
PRIMARY = FeedKey("main", "EURUSD", "M1")
SECONDARY = FeedKey("main", "GBPUSD", "M1")
AUXILIARY = FeedKey("confirmation", "DXY", "M1")


def _frame(index: pd.DatetimeIndex, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": 100.0,
            "spread": 0.0,
            "__account_point_value__": 1.0,
        },
        index=index,
    )


def _market(start: str, periods: int) -> MarketFrames:
    index = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    return MarketFrames(
        {
            PRIMARY: _frame(index, 1.10),
            SECONDARY: _frame(index, 1.30),
            AUXILIARY: _frame(index, 100.0),
        },
        primary=PRIMARY,
        execution_keys=(PRIMARY, SECONDARY),
    )


def _strategy() -> DeterministicMultiMarketParityStrategy:
    return DeterministicMultiMarketParityStrategy(
        scenario_start_utc=START.isoformat(),
        primary_symbol=PRIMARY.symbol,
        secondary_symbol=SECONDARY.symbol,
        auxiliary_provider=AUXILIARY.provider,
        auxiliary_symbol=AUXILIARY.symbol,
        primary_volume=0.02,
        secondary_volume=0.02,
        distance_fraction=0.20,
        base_magic=826_000,
    )


def _backtester() -> Backtester:
    instruments = {
        symbol: InstrumentConfig(
            ib_symbol=symbol,
            point_value=1.0,
            contract_size=1.0,
            tick_size=0.0001,
            min_order_size=0.01,
        )
        for symbol in (PRIMARY.symbol, SECONDARY.symbol)
    }
    return Backtester(
        BTConfig(
            commission_per_lot=0.0,
            default_slippage_points=0.0,
            apply_spread_cost=False,
            apply_swap_cost=False,
            apply_stop_take=False,
            market_fill_price="close",
        ),
        instruments=instruments,
    )


def _signature(signals: Iterable[Any]) -> list[tuple[Any, ...]]:
    return [
        (
            pd.Timestamp(signal.when),
            signal.action,
            signal.symbol,
            signal.magic,
            signal.volume,
            signal.close_volume,
            signal.new_sl,
            signal.new_tp,
            signal.new_price,
            signal.reason,
        )
        for signal in signals
    ]


def _run(market: MarketFrames):
    emitted: list[Any] = []
    result = _backtester().run_market(
        market,
        _strategy(),
        risk_perc=0.01,
        initial_equity=10_000.0,
        capture_prepared=True,
        bar_observer=lambda state: emitted.extend(state["bar_signals"]),
    )
    return result, emitted


def test_scenario_is_absolute_utc_and_independent_of_backtest_prefix() -> None:
    exact_result, exact_signals = _run(_market(START.isoformat(), 20))
    prefixed_result, prefixed_signals = _run(_market("2026-08-30T11:58:00Z", 22))

    assert _signature(prefixed_signals) == _signature(exact_signals)
    assert len(exact_signals) == 66
    assert [sum(signal.when == START + pd.Timedelta(minutes=minute) for signal in exact_signals) for minute in range(20)] == [
        2,
        4,
        2,
        4,
        2,
        4,
        4,
        2,
        4,
        4,
        4,
        4,
        2,
        4,
        2,
        4,
        4,
        4,
        4,
        2,
    ]
    assert all(signal.reason.startswith(f"DMP1:{pd.Timestamp(signal.when).strftime('%Y%m%d%H%M')}:") for signal in exact_signals)
    assert {signal.symbol for signal in exact_signals} == {PRIMARY.symbol, SECONDARY.symbol}
    assert {signal.action for signal in exact_signals} >= {
        "buy",
        "sell",
        "close",
        "partial_close",
        "modify_sl",
        "modify_tp",
        "modify_price",
        "cancel_order",
    }
    assert len(exact_result.trades) == len(prefixed_result.trades) == 16
    assert len(exact_result.orders) == len(prefixed_result.orders) == 12
    assert exact_result.prepared is not None
    assert "parity_aux_close" in exact_result.prepared


def test_scenario_rejects_ambiguous_start_and_missing_auxiliary_feed() -> None:
    with pytest.raises(ValueError, match="explicit timezone"):
        DeterministicMultiMarketParityStrategy(scenario_start_utc="2026-08-30 12:00:00")
    with pytest.raises(ValueError, match="exact minute"):
        DeterministicMultiMarketParityStrategy(scenario_start_utc="2026-08-30T12:00:01Z")

    market = _market(START.isoformat(), 2)
    incomplete = MarketFrames(
        {PRIMARY: market[PRIMARY], SECONDARY: market[SECONDARY]},
        primary=PRIMARY,
        execution_keys=(PRIMARY, SECONDARY),
    )
    with pytest.raises(ValueError, match="missing feeds"):
        _strategy().prepare(incomplete.primary_frame, data=incomplete)
