from __future__ import annotations

from collections.abc import Mapping
import inspect

import pandas as pd
import pytest

from arbitrix_core.backtest.engine import BTConfig, Backtester
from arbitrix_core.data.market import FeedKey, MarketBars, MarketFrames, PreparedMarket
from arbitrix_core.strategies.base import BaseStrategy
from arbitrix_core.trading import Signal
from arbitrix_core.types import InstrumentConfig


def _frame(
    closes: list[float],
    *,
    lows: list[float] | None = None,
    point_value: float = 1.0,
) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=len(closes), freq="h")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": lows or [value - 1.0 for value in closes],
            "close": closes,
            "volume": 100.0,
            "spread": 0.0,
            "__account_point_value__": point_value,
        },
        index=index,
    )


PRIMARY = FeedKey("main", "EURUSD", "H1")
SECONDARY = FeedKey("main", "GBPUSD", "H1")
AUXILIARY = FeedKey("macro", "DXY", "H1")


def _market(*, auxiliary_frame: pd.DataFrame | None = None) -> MarketFrames:
    frames = {
        PRIMARY: _frame([10.0, 10.0, 10.0, 10.0]),
        SECONDARY: _frame(
            [100.0, 101.0, 102.0, 103.0],
            lows=[99.0, 94.0, 101.0, 102.0],
            point_value=2.0,
        ),
        AUXILIARY: (auxiliary_frame if auxiliary_frame is not None else _frame([1.0, 2.0, 3.0, 4.0])),
    }
    return MarketFrames(
        frames,
        primary=PRIMARY,
        execution_keys=(PRIMARY, SECONDARY),
    )


def _backtester(
    *,
    row_mode: str = "series",
    market_fill_price: str = "close",
) -> Backtester:
    instruments = {
        symbol: InstrumentConfig(
            ib_symbol=symbol,
            point_value=1.0,
            contract_size=1.0,
            tick_size=1.0,
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
            market_fill_price=market_fill_price,
            row_mode=row_mode,
        ),
        instruments=instruments,
    )


def test_run_market_matches_run_single_option_surface() -> None:
    single = inspect.signature(Backtester.run_single).parameters
    market = inspect.signature(Backtester.run_market).parameters

    assert tuple(single)[2:] == tuple(market)[2:]
    assert single["df"].kind == market["market"].kind


class MultiMarketStrategy(BaseStrategy):
    symbol = PRIMARY.symbol
    timeframe = PRIMARY.timeframe
    name = "multi-market"

    def __init__(self, *, target_symbol: str | None = SECONDARY.symbol) -> None:
        self.target_symbol = target_symbol
        self.prepare_data: MarketFrames | None = None
        self.seen_bars: list[MarketBars] = []
        self.stop_calls: list[tuple[str | None, float, MarketBars | None]] = []
        self.take_calls: list[tuple[str | None, float, MarketBars | None]] = []

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        data: MarketFrames | None = None,
    ) -> PreparedMarket:
        assert data is not None
        assert df is data.primary_frame
        self.prepare_data = data
        prepared = {
            key: frame.assign(aux_close=data[AUXILIARY]["close"].to_numpy()) for key, frame in data.items()
        }
        return PreparedMarket(prepared, primary=data.primary)

    def on_bar(
        self,
        row: pd.Series,
        portfolio,
        *,
        data: MarketBars | None = None,
    ) -> list[Signal]:
        del portfolio
        assert data is not None
        assert row is data.primary_bar
        self.seen_bars.append(data)
        if len(self.seen_bars) != 1:
            return []
        return [
            Signal(
                when=row.name,
                action="buy",
                price=float(data[SECONDARY]["close"]),
                volume=1.0,
                symbol=self.target_symbol,
            )
        ]

    def stop_distance_points(
        self,
        row: pd.Series,
        *,
        symbol: str | None = None,
        data: MarketBars | None = None,
    ) -> float:
        self.stop_calls.append((symbol, float(row["close"]), data))
        return 5.0

    def take_distance_points(
        self,
        row: pd.Series,
        *,
        symbol: str | None = None,
        data: MarketBars | None = None,
    ) -> float:
        self.take_calls.append((symbol, float(row["close"]), data))
        return 0.0


@pytest.mark.parametrize("row_mode", ["series", "bar_view"])
def test_run_market_injects_synchronized_data_and_routes_target_symbol(
    row_mode: str,
) -> None:
    market = _market()
    strategy = MultiMarketStrategy()

    result = _backtester(row_mode=row_mode).run_market(
        market,
        strategy,
        risk_perc=0.01,
        initial_equity=10_000.0,
        capture_prepared=True,
    )

    assert strategy.prepare_data is market
    assert len(strategy.seen_bars) == len(market.primary_frame)
    assert all(isinstance(bars, MarketBars) for bars in strategy.seen_bars)
    assert all(set(bars) == set(market) for bars in strategy.seen_bars)
    assert [bars.decision_time for bars in strategy.seen_bars] == list(market.primary_frame.index)
    assert strategy.seen_bars[2][AUXILIARY]["aux_close"] == pytest.approx(3.0)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.symbol == SECONDARY.symbol
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(95.0)
    assert trade.gross_pnl == pytest.approx(-10.0)
    assert trade.notes["exit_stop"] == pytest.approx(1.0)
    assert result.orders[0].symbol == SECONDARY.symbol
    assert result.metadata["primary_symbol"] == PRIMARY.symbol
    assert result.metadata["execution_symbols"] == [
        PRIMARY.symbol,
        SECONDARY.symbol,
    ]
    assert result.metadata["market_feeds"] == [
        {
            "provider": "main",
            "symbol": PRIMARY.symbol,
            "timeframe": "H1",
            "primary": True,
            "execution": True,
        },
        {
            "provider": "main",
            "symbol": SECONDARY.symbol,
            "timeframe": "H1",
            "primary": False,
            "execution": True,
        },
        {
            "provider": "macro",
            "symbol": AUXILIARY.symbol,
            "timeframe": "H1",
            "primary": False,
            "execution": False,
        },
    ]

    assert strategy.stop_calls == [(SECONDARY.symbol, 100.0, strategy.seen_bars[0])]
    assert strategy.take_calls == [(SECONDARY.symbol, 100.0, strategy.seen_bars[0])]
    assert isinstance(result.prepared_market, PreparedMarket)
    assert result.prepared_market.primary == PRIMARY
    pd.testing.assert_frame_equal(result.prepared, result.prepared_market.primary_frame)
    assert {
        "Sharpe",
        "PSR",
        "RobustScore",
        "TradeCount",
        "gross_pnl",
        "net_pnl",
    }.issubset(result.metrics)


def test_run_market_audit_failures_are_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arbitrix_core.backtest.engine as engine_module

    def broken_audit_write(payload) -> None:
        del payload
        raise OSError("audit sink unavailable")

    monkeypatch.setattr(
        engine_module,
        "_backtest_audit_writer",
        lambda strategy_name: broken_audit_write,
    )

    result = _backtester().run_market(
        _market(),
        MultiMarketStrategy(),
        risk_perc=0.01,
        initial_equity=10_000.0,
    )

    assert len(result.trades) == 1


def test_run_market_next_open_fill_uses_target_symbol_bar() -> None:
    result = _backtester(market_fill_price="next_open").run_market(
        _market(),
        MultiMarketStrategy(),
        risk_perc=0.01,
        initial_equity=10_000.0,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.symbol == SECONDARY.symbol
    assert trade.entry_price == pytest.approx(101.0)
    # Protection remains anchored to the strategy's 100.0 reference price,
    # matching run_single semantics even though execution occurs next-open.
    assert trade.exit_price == pytest.approx(95.0)
    assert trade.gross_pnl == pytest.approx(-12.0)


def test_run_market_rejects_data_only_execution_target() -> None:
    with pytest.raises(ValueError, match="non-executable symbol"):
        _backtester().run_market(
            _market(),
            MultiMarketStrategy(target_symbol=AUXILIARY.symbol),
            risk_perc=0.01,
            initial_equity=10_000.0,
        )


class LegacyPrimarySignalStrategy(MultiMarketStrategy):
    def on_bar(
        self,
        row: pd.Series,
        portfolio,
        *,
        data: MarketBars | None = None,
    ) -> list[Signal]:
        del portfolio
        assert data is not None
        self.seen_bars.append(data)
        if len(self.seen_bars) != 1:
            return []
        return [
            Signal(
                when=row.name,
                action="buy",
                price=float(row["close"]),
                volume=1.0,
            )
        ]


def test_run_market_legacy_signal_without_symbol_targets_primary() -> None:
    result = _backtester().run_market(
        _market(),
        LegacyPrimarySignalStrategy(target_symbol=None),
        risk_perc=0.01,
        initial_equity=10_000.0,
    )

    assert len(result.trades) == 1
    assert result.trades[0].symbol == PRIMARY.symbol
    assert result.orders[0].symbol == PRIMARY.symbol


def test_run_market_blank_signal_symbol_targets_primary() -> None:
    result = _backtester().run_market(
        _market(),
        MultiMarketStrategy(target_symbol=""),
        risk_perc=0.01,
        initial_equity=10_000.0,
    )

    assert len(result.trades) == 1
    assert result.trades[0].symbol == PRIMARY.symbol
    assert result.orders[0].symbol == PRIMARY.symbol


class InvalidPrepareStrategy(MultiMarketStrategy):
    def __init__(self, result) -> None:
        super().__init__()
        self.result = result

    def prepare(self, df: pd.DataFrame, *, data: MarketFrames | None = None):
        del df, data
        return self.result


def test_run_market_requires_prepared_market() -> None:
    with pytest.raises(ValueError, match="PreparedMarket"):
        _backtester().run_market(
            _market(),
            InvalidPrepareStrategy(_frame([1.0, 2.0, 3.0, 4.0])),
            risk_perc=0.01,
            initial_equity=10_000.0,
        )


class PreparedGapStrategy(MultiMarketStrategy):
    def prepare(
        self,
        df: pd.DataFrame,
        *,
        data: MarketFrames | None = None,
    ) -> PreparedMarket:
        del df
        assert data is not None
        frames = dict(data)
        frames[AUXILIARY] = frames[AUXILIARY].iloc[:-1]
        return PreparedMarket(frames, primary=data.primary)


def test_run_market_rejects_prepared_feed_gap() -> None:
    with pytest.raises(ValueError, match="identical decision timestamps"):
        _backtester().run_market(
            _market(),
            PreparedGapStrategy(),
            risk_perc=0.01,
            initial_equity=10_000.0,
        )


def test_run_market_rejects_raw_feed_gap_before_prepare() -> None:
    auxiliary = _frame([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="identical decision timestamps"):
        _backtester().run_market(
            _market(auxiliary_frame=auxiliary),
            MultiMarketStrategy(),
            risk_perc=0.01,
            initial_equity=10_000.0,
        )


@pytest.mark.parametrize("invalid_index", ["naive", "duplicate"])
def test_run_market_rejects_non_utc_or_duplicate_source_index(
    invalid_index: str,
) -> None:
    frames: Mapping[FeedKey, pd.DataFrame] = dict(_market())
    auxiliary = frames[AUXILIARY].copy()
    if invalid_index == "naive":
        auxiliary.index = auxiliary.index.tz_localize(None)
        message = "UTC"
    else:
        auxiliary.index = pd.DatetimeIndex([auxiliary.index[0], auxiliary.index[0], *auxiliary.index[2:]])
        message = "duplicate"
    frames = {**frames, AUXILIARY: auxiliary}

    with pytest.raises(ValueError, match=message):
        _backtester().run_market(
            MarketFrames(
                frames,
                primary=PRIMARY,
                execution_keys=(PRIMARY, SECONDARY),
            ),
            MultiMarketStrategy(),
            risk_perc=0.01,
            initial_equity=10_000.0,
        )


def test_run_market_rejects_mixed_timeframes() -> None:
    wrong_timeframe = FeedKey("macro", "DXY", "M5")
    market = _market()
    frames = {wrong_timeframe if key == AUXILIARY else key: frame for key, frame in market.items()}

    with pytest.raises(ValueError, match="same timeframe"):
        _backtester().run_market(
            MarketFrames(
                frames,
                primary=PRIMARY,
                execution_keys=(PRIMARY, SECONDARY),
            ),
            MultiMarketStrategy(),
            risk_perc=0.01,
            initial_equity=10_000.0,
        )


class SingletonParityStrategy(BaseStrategy):
    symbol = PRIMARY.symbol
    timeframe = PRIMARY.timeframe
    name = "singleton-market-parity"

    def __init__(self) -> None:
        self.entered = False

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        data: MarketFrames | None = None,
    ):
        prepared = df.copy()
        if data is None:
            return prepared
        return PreparedMarket({data.primary: prepared}, primary=data.primary)

    def on_bar(
        self,
        row: pd.Series,
        portfolio,
        *,
        data: MarketBars | None = None,
    ) -> list[Signal]:
        del portfolio, data
        if self.entered:
            return []
        self.entered = True
        return [
            Signal(
                when=row.name,
                action="buy",
                price=float(row["close"]),
                volume=1.0,
                symbol=self.symbol,
            )
        ]

    def stop_distance_points(self, row: pd.Series, **kwargs) -> float:
        del row, kwargs
        return 100.0


def test_singleton_market_path_matches_run_single_result_contract() -> None:
    frame = _frame([10.0, 11.0, 12.0, 13.0])
    market = MarketFrames({PRIMARY: frame}, primary=PRIMARY)
    mono = _backtester().run_single(
        frame,
        SingletonParityStrategy(),
        risk_perc=0.01,
        initial_equity=10_000.0,
        collect_diagnostics=True,
    )
    bundled = _backtester().run_market(
        market,
        SingletonParityStrategy(),
        risk_perc=0.01,
        initial_equity=10_000.0,
        collect_diagnostics=True,
    )

    pd.testing.assert_series_equal(bundled.daily_equity, mono.daily_equity)
    pd.testing.assert_series_equal(bundled.gross_equity, mono.gross_equity)
    pd.testing.assert_series_equal(bundled.equity_marked, mono.equity_marked)
    assert bundled.metrics == pytest.approx(mono.metrics, nan_ok=True)
    assert len(bundled.trades) == len(mono.trades) == 1
    assert mono.metadata["primary_symbol"] == PRIMARY.symbol
    assert bundled.metadata["primary_symbol"] == PRIMARY.symbol
    for attribute in (
        "symbol",
        "side",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "volume",
        "gross_pnl",
        "net_pnl",
    ):
        bundled_value = getattr(bundled.trades[0], attribute)
        mono_value = getattr(mono.trades[0], attribute)
        if isinstance(mono_value, float):
            assert bundled_value == pytest.approx(mono_value)
        else:
            assert bundled_value == mono_value
    assert set(bundled.metadata) == set(mono.metadata) | {
        "market_data_mode",
        "execution_symbols",
        "market_feeds",
    }
    for key in set(mono.metadata) - {"runtime_timing"}:
        assert bundled.metadata[key] == mono.metadata[key]


class RollingMarketStrategy(MultiMarketStrategy):
    def __init__(self) -> None:
        super().__init__(target_symbol=SECONDARY.symbol)
        self.prepare_windows: list[dict[FeedKey, int]] = []

    def warmup_bars(self) -> int:
        return 2

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        data: MarketFrames | None = None,
    ) -> PreparedMarket:
        assert data is not None
        assert df is data.primary_frame
        self.prepare_windows.append({key: len(frame) for key, frame in data.items()})
        return PreparedMarket(
            {key: frame.assign(window_size=len(frame)) for key, frame in data.items()},
            primary=data.primary,
        )


def test_run_market_honors_strict_rolling_prepare_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARBITRIX_BACKTEST_STRICT_PREPARE_WINDOW", "true")
    strategy = RollingMarketStrategy()

    result = _backtester().run_market(
        _market(),
        strategy,
        risk_perc=0.01,
        initial_equity=10_000.0,
        capture_prepared=True,
    )

    assert strategy.prepare_windows == [
        {PRIMARY: 2, SECONDARY: 2, AUXILIARY: 2},
        {PRIMARY: 2, SECONDARY: 2, AUXILIARY: 2},
        {PRIMARY: 2, SECONDARY: 2, AUXILIARY: 2},
    ]
    assert [bars.decision_time for bars in strategy.seen_bars] == list(_market().primary_frame.index[1:])
    assert result.prepared_market is not None
    assert all(len(frame) == 3 for frame in result.prepared_market.values())
