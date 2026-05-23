from __future__ import annotations

import pandas as pd
import pytest

from arbitrix_core.backtest.engine import Backtester, BTConfig
from arbitrix_core.strategies.base import BaseStrategy
from arbitrix_core.trading import Signal, SignalAction


class _NoopStrategy(BaseStrategy):
    name = "noop"
    symbol = "X"
    timeframe = "M1"

    def prepare(self, df):
        return df

    def on_bar(self, row, portfolio, regime_output=None):
        return []


def _df():
    idx = pd.date_range("2025-06-02", periods=10, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0, "spread": 0.0},
        index=idx,
    )


def test_bar_observer_called_once_per_bar():
    seen: list[pd.Timestamp] = []

    def observer(ctx):
        seen.append(ctx["bar_ts"])

    bt = Backtester(BTConfig(commission_per_lot=0.0, apply_swap_cost=False))
    bt.run_single(_df(), _NoopStrategy(), risk_perc=0.005, initial_equity=10_000.0,
                  bar_observer=observer)

    assert len(seen) == 10
    assert seen[0] == pd.Timestamp("2025-06-02T00:00:00Z")
    assert seen[-1] == pd.Timestamp("2025-06-02T00:09:00Z")


def test_bar_observer_default_none_is_no_op():
    bt = Backtester(BTConfig(commission_per_lot=0.0, apply_swap_cost=False))
    res = bt.run_single(_df(), _NoopStrategy(), risk_perc=0.005, initial_equity=10_000.0)
    assert res is not None


def test_bar_observer_receives_full_context():
    captured: list[dict] = []

    def observer(ctx):
        captured.append(dict(ctx))

    bt = Backtester(BTConfig(commission_per_lot=0.0, apply_swap_cost=False))
    bt.run_single(_df(), _NoopStrategy(), risk_perc=0.005, initial_equity=10_000.0,
                  bar_observer=observer)

    keys = set(captured[0].keys())
    assert {"bar_ts", "row", "portfolio", "open_trades", "closed_trades",
            "working_orders", "equity", "gross_equity",
            "bar_signals", "newly_filled"} <= keys
