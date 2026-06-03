from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arbitrix_core import costs
from arbitrix_core.backtest.bar_view import BarViewSource
from arbitrix_core.backtest.engine import Backtester, BTConfig
from arbitrix_core.strategies.base import BaseStrategy
from arbitrix_core.trading import Signal


def _frame(periods: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=periods, freq="min", tz="UTC")
    close = [100.0 + i * 0.5 for i in range(periods)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.75 for value in close],
            "low": [value - 0.75 for value in close],
            "close": close,
            "volume": [1000.0] * periods,
            "spread": [0.0] * periods,
            "entry": [i == 2 for i in range(periods)],
            "payload": [{"i": i} for i in range(periods)],
        },
        index=idx,
    )


def test_bar_view_matches_series_scalar_access() -> None:
    frame = _frame()
    source = BarViewSource(frame)
    row = source.row_at(2)
    series = frame.iloc[2]

    assert row.name == series.name
    assert row["close"] == series["close"]
    assert row.get("entry", False) == series.get("entry", False)
    assert row.get("missing", "fallback") == "fallback"
    assert "close" in row
    assert "missing" not in row
    assert list(row.keys()) == list(series.keys())
    assert row.to_dict() == series.to_dict()
    assert row.close == series.close
    assert row[["open", "close"]].to_dict() == series[["open", "close"]].to_dict()


class _ParityStrategy(BaseStrategy):
    name = "bar-view-parity"
    symbol = "TEST"
    timeframe = "M1"

    def __init__(self) -> None:
        super().__init__()
        self._entered = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["entry"] = work["entry"].astype(bool)
        work["atr"] = 1.0
        return work

    def on_bar(self, row, portfolio):
        _ = portfolio
        assert not math.isnan(float(row["atr"]))
        if self._entered or not bool(row.get("entry", False)):
            return []
        self._entered = True
        return [
            Signal(
                when=row.name,
                action="buy",
                price=float(row.get("close")),
                reason="bar-view-entry",
            )
        ]

    def stop_distance_points(self, row) -> float:
        return float(row.get("atr", 1.0)) * 2.0

    def take_distance_points(self, row) -> float:
        return float(row["atr"]) * 3.0


def test_backtester_bar_view_mode_matches_series_mode() -> None:
    costs.configure(commission_per_lot=0.0, point_overrides={"TEST": 1.0})
    frame = _frame()

    series_result = Backtester(
        BTConfig(commission_per_lot=0.0, apply_swap_cost=False)
    ).run_single(
        frame.copy(),
        _ParityStrategy(),
        risk_perc=0.01,
        initial_equity=10_000.0,
    )
    bar_view_result = Backtester(
        BTConfig(commission_per_lot=0.0, apply_swap_cost=False, row_mode="bar_view")
    ).run_single(
        frame.copy(),
        _ParityStrategy(),
        risk_perc=0.01,
        initial_equity=10_000.0,
    )

    assert len(bar_view_result.trades) == len(series_result.trades)
    assert [trade.net_pnl for trade in bar_view_result.trades] == [
        trade.net_pnl for trade in series_result.trades
    ]
    pd.testing.assert_series_equal(
        bar_view_result.daily_equity,
        series_result.daily_equity,
        check_names=False,
    )
    assert bar_view_result.metrics == series_result.metrics


def test_prev_day_breakout_advanced_bar_view_matches_series_mode() -> None:
    strategy_root = (
        Path(__file__).resolve().parents[4]
        / "runtime"
        / "strategies"
        / ".connections"
        / "b46a14ccc06172ff"
    )
    sys.path.insert(0, str(strategy_root))
    try:
        from PrevDayBreakoutAdvanced.strategy import (  # type: ignore[import-not-found]
            PrevDayBreakoutAdvanced,
            StrategyConfig,
        )
    finally:
        try:
            sys.path.remove(str(strategy_root))
        except ValueError:
            pass

    periods = 5_000
    idx = pd.date_range("2025-01-01", periods=periods, freq="min", tz="UTC")
    x = np.linspace(0.0, 20.0 * np.pi, periods)
    close = 100.0 + np.linspace(0.0, 12.0, periods) + 3.0 * np.sin(x)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.8,
            "low": np.minimum(open_, close) - 0.8,
            "close": close,
            "volume": np.full(periods, 1000.0),
            "spread": np.zeros(periods),
        },
        index=idx,
    )
    cfg_kwargs = {
        "symbol": "Usa500",
        "session_timezone": "America/New_York",
        "ret_min": 0.0001,
        "buf": 0.0,
        "max_holding_bars": 30,
    }

    costs.configure(commission_per_lot=0.0, point_overrides={"Usa500": 1.0})
    series_result = Backtester(
        BTConfig(commission_per_lot=0.0, apply_swap_cost=False, apply_spread_cost=False)
    ).run_single(
        frame.copy(),
        PrevDayBreakoutAdvanced(StrategyConfig(**cfg_kwargs)),
        risk_perc=0.005,
        initial_equity=10_000.0,
    )

    costs.configure(commission_per_lot=0.0, point_overrides={"Usa500": 1.0})
    bar_view_result = Backtester(
        BTConfig(
            commission_per_lot=0.0,
            apply_swap_cost=False,
            apply_spread_cost=False,
            row_mode="bar_view",
        )
    ).run_single(
        frame.copy(),
        PrevDayBreakoutAdvanced(StrategyConfig(**cfg_kwargs)),
        risk_perc=0.005,
        initial_equity=10_000.0,
    )

    assert len(bar_view_result.trades) == len(series_result.trades)
    assert [trade.net_pnl for trade in bar_view_result.trades] == [
        trade.net_pnl for trade in series_result.trades
    ]
    pd.testing.assert_series_equal(
        bar_view_result.daily_equity,
        series_result.daily_equity,
        check_names=False,
    )
    assert bar_view_result.metrics == series_result.metrics
