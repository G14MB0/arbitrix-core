from __future__ import annotations

import pandas as pd

from arbitrix_core import costs
from arbitrix_core.backtest.engine import Backtester, BTConfig
from arbitrix_core.strategies.base import BaseStrategy
from arbitrix_core.trading import Signal


class _NoopStrategy(BaseStrategy):
    name = "noop"
    symbol = "X"
    timeframe = "M1"

    def prepare(self, df):
        return df

    def on_bar(self, row, portfolio, regime_output=None):
        return []


class _BuyOnBarTwoStrategy(BaseStrategy):
    """Emits exactly one market BUY on bar index 2; otherwise silent."""

    name = "buybar2"
    symbol = "X"
    timeframe = "M1"

    def __init__(self) -> None:
        super().__init__()
        self._emitted = False
        self._target_ts: pd.Timestamp | None = None

    def prepare(self, df):
        # Lock in the third timestamp (bar index 2) as the emission target.
        if len(df) >= 3:
            self._target_ts = df.index[2]
        return df

    def stop_distance_points(self, row):
        # Must be > 0 for `_create_order_from_signal` to produce an Order.
        return 1.0

    def take_distance_points(self, row):
        return 2.0

    def on_bar(self, row, portfolio, regime_output=None):
        if self._emitted or self._target_ts is None:
            return []
        ts = row.name if hasattr(row, "name") else None
        if ts == self._target_ts:
            self._emitted = True
            return [
                Signal(
                    when=ts,
                    action="buy",
                    price=float(row["open"]),
                    reason="test_entry",
                )
            ]
        return []


def _df():
    idx = pd.date_range("2025-06-02", periods=10, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0, "spread": 0.0},
        index=idx,
    )


def _df_signal():
    """Deterministic 20-bar OHLCV frame for signals+fills test."""

    idx = pd.date_range("2025-06-02", periods=20, freq="1min", tz="UTC")
    # Slight upward drift so a long fill has well-defined PnL.
    closes = [1.0 + 0.01 * i for i in range(20)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.05 for c in closes],
            "low": [c - 0.05 for c in closes],
            "close": closes,
            "volume": 1.0,
            "spread": 0.0,
        },
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
    """Prove the bar_observer=None branch is byte-identical to a no-op lambda.

    Runs the same backtest twice on identical inputs, comparing every field of
    the resulting :class:`BTResult` that can be observed externally. If the
    observer's None branch had any side effect on the production path, this
    would diverge.
    """

    costs.configure(commission_per_lot=0.0, point_overrides={"X": 1.0})
    df = _df_signal()
    bt_none = Backtester(BTConfig(commission_per_lot=0.0, apply_swap_cost=False))
    res_none = bt_none.run_single(
        df.copy(),
        _BuyOnBarTwoStrategy(),
        risk_perc=0.005,
        initial_equity=10_000.0,
    )

    bt_noop = Backtester(BTConfig(commission_per_lot=0.0, apply_swap_cost=False))
    res_noop = bt_noop.run_single(
        df.copy(),
        _BuyOnBarTwoStrategy(),
        risk_perc=0.005,
        initial_equity=10_000.0,
        bar_observer=lambda ctx: None,
    )

    # Trade count and identity must match.
    assert len(res_none.trades) == len(res_noop.trades)

    # Equity series must be element-wise identical.
    pd.testing.assert_series_equal(
        res_none.daily_equity,
        res_noop.daily_equity,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        res_none.gross_equity,
        res_noop.gross_equity,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        res_none.equity_marked,
        res_noop.equity_marked,
        check_names=False,
    )

    # Metrics dict must be identical.
    assert res_none.metrics.keys() == res_noop.metrics.keys()
    for key in res_none.metrics:
        a, b = res_none.metrics[key], res_noop.metrics[key]
        # Two NaNs should be treated as equal here.
        if isinstance(a, float) and isinstance(b, float):
            if pd.isna(a) and pd.isna(b):
                continue
        assert a == b, f"metric {key} differs: {a} != {b}"


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


def test_bar_observer_captures_signals_and_fills():
    """A market BUY emitted on bar 2 should show up in both `bar_signals`
    (on the emission bar) and `newly_filled` (same bar — market orders fill
    in the same loop iteration via ``_try_fill_order``)."""

    costs.configure(commission_per_lot=0.0, point_overrides={"X": 1.0})
    df = _df_signal()
    captured: list[dict] = []

    def observer(ctx):
        # Snapshot the lists at observation time (they are already copies in
        # the engine, but defensively rebuild to be sure).
        captured.append(
            {
                "bar_ts": ctx["bar_ts"],
                "bar_signals": list(ctx["bar_signals"]),
                "newly_filled": list(ctx["newly_filled"]),
            }
        )

    bt = Backtester(BTConfig(commission_per_lot=0.0, apply_swap_cost=False))
    bt.run_single(
        df,
        _BuyOnBarTwoStrategy(),
        risk_perc=0.005,
        initial_equity=10_000.0,
        bar_observer=observer,
    )

    # Observer fires once per bar.
    assert len(captured) == len(df)

    # Find the bar where the signal was emitted.
    signal_bars = [c for c in captured if c["bar_signals"]]
    assert len(signal_bars) == 1, (
        f"expected exactly one bar with bar_signals, got {len(signal_bars)}"
    )
    emit = signal_bars[0]
    assert emit["bar_ts"] == df.index[2]
    assert len(emit["bar_signals"]) == 1
    # Market BUY fills in the same loop iteration as the signal.
    assert len(emit["newly_filled"]) == 1, (
        f"expected market BUY to fill same bar, got newly_filled={emit['newly_filled']!r}"
    )

    # Every other bar has empty signals and empty fills.
    for ctx in captured:
        if ctx["bar_ts"] == df.index[2]:
            continue
        assert ctx["bar_signals"] == []
        assert ctx["newly_filled"] == []
