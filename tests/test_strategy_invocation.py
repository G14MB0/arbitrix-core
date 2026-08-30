from __future__ import annotations

from dataclasses import fields

import pandas as pd

from arbitrix_core.data.market import FeedKey, MarketBars, MarketFrames, PreparedMarket
from arbitrix_core.strategies.base import (
    BaseStrategy,
    invoke_strategy_on_bar,
    invoke_strategy_prepare,
    invoke_strategy_stop_distance_points,
    invoke_strategy_take_distance_points,
    strategy_supports_parameter,
)
from arbitrix_core.trading import Signal


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.2, 2.2],
            "low": [0.8, 1.8],
            "close": [1.1, 2.1],
            "volume": [10.0, 20.0],
        },
        index=index,
    )


class _LegacyStrategy(BaseStrategy):
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(("prepare", df))
        return df

    def on_bar(self, row: pd.Series, portfolio: object):
        self.calls.append(("on_bar", row, portfolio))
        return ["legacy"]

    def stop_distance_points(self, row: pd.Series) -> float:
        self.calls.append(("stop", row))
        return 11.0

    def take_distance_points(self, row: pd.Series) -> float:
        self.calls.append(("take", row))
        return 22.0


class _BundleAwareStrategy(BaseStrategy):
    def __init__(self) -> None:
        self.received: dict[str, object] = {}

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        data: MarketFrames,
    ) -> PreparedMarket:
        self.received["prepare"] = (df, data)
        return PreparedMarket(data, primary=data.primary)

    def on_bar(
        self,
        row: pd.Series,
        portfolio: object,
        *,
        regime_output: object,
        ctx: object,
        data: MarketBars,
    ):
        self.received["on_bar"] = (row, portfolio, regime_output, ctx, data)
        return ["aware"]

    def stop_distance_points(
        self,
        row: pd.Series,
        *,
        symbol: str,
        ctx: object,
        data: MarketBars,
    ) -> float:
        self.received["stop"] = (row, symbol, ctx, data)
        return 33.0

    def take_distance_points(
        self,
        row: pd.Series,
        *,
        symbol: str,
        ctx: object,
        data: MarketBars,
    ) -> float:
        self.received["take"] = (row, symbol, ctx, data)
        return 44.0


def _bundles() -> tuple[pd.DataFrame, MarketFrames, MarketBars]:
    key = FeedKey("main", "EURUSD", "M5")
    frame = _frame()
    frames = MarketFrames({key: frame}, primary=key)
    bars = MarketBars(
        {key: frame.iloc[0]},
        primary=key,
        decision_time=frame.index[0],
    )
    return frame, frames, bars


def test_invokers_preserve_all_legacy_hook_signatures() -> None:
    strategy = _LegacyStrategy()
    frame, frames, bars = _bundles()
    row = frame.iloc[0]
    portfolio = object()

    assert invoke_strategy_prepare(strategy, frame, data=frames) is frame
    assert invoke_strategy_on_bar(
        strategy,
        row,
        portfolio,
        {"regime": 1},
        ctx=object(),
        data=bars,
    ) == ["legacy"]
    assert invoke_strategy_stop_distance_points(
        strategy,
        row,
        symbol="EURUSD",
        ctx=object(),
        data=bars,
    ) == 11.0
    assert invoke_strategy_take_distance_points(
        strategy,
        row,
        symbol="EURUSD",
        ctx=object(),
        data=bars,
    ) == 22.0

    assert [call[0] for call in strategy.calls] == ["prepare", "on_bar", "stop", "take"]


def test_invokers_deliver_new_inputs_by_exact_keyword_name() -> None:
    strategy = _BundleAwareStrategy()
    frame, frames, bars = _bundles()
    row = frame.iloc[0]
    portfolio = object()
    ctx = object()
    regime = {"regime": 2}

    prepared = invoke_strategy_prepare(strategy, frame, data=frames)
    signals = invoke_strategy_on_bar(
        strategy,
        row,
        portfolio,
        regime,
        ctx=ctx,
        data=bars,
    )
    stop = invoke_strategy_stop_distance_points(
        strategy,
        row,
        symbol="EURUSD",
        ctx=ctx,
        data=bars,
    )
    take = invoke_strategy_take_distance_points(
        strategy,
        row,
        symbol="EURUSD",
        ctx=ctx,
        data=bars,
    )

    assert isinstance(prepared, PreparedMarket)
    assert signals == ["aware"]
    assert stop == 33.0
    assert take == 44.0
    assert strategy.received["prepare"] == (frame, frames)
    assert strategy.received["on_bar"] == (row, portfolio, regime, ctx, bars)
    assert strategy.received["stop"] == (row, "EURUSD", ctx, bars)
    assert strategy.received["take"] == (row, "EURUSD", ctx, bars)


def test_parameter_detection_does_not_guess_from_positional_count() -> None:
    class _DifferentThirdParameter(BaseStrategy):
        def on_bar(self, row, portfolio, unrelated=None):
            return unrelated

    strategy = _DifferentThirdParameter()
    row = _frame().iloc[0]

    assert strategy_supports_parameter(strategy, "on_bar", "regime_output") is False
    assert invoke_strategy_on_bar(
        strategy,
        row,
        object(),
        {"must_not": "leak"},
    ) is None


def test_ctx_only_signature_never_receives_regime_as_ctx() -> None:
    class _CtxOnly(BaseStrategy):
        def on_bar(self, row, portfolio, ctx=None):
            return ctx

    ctx = object()
    result = invoke_strategy_on_bar(
        _CtxOnly(),
        _frame().iloc[0],
        object(),
        {"regime": "ignored"},
        ctx=ctx,
    )

    assert result is ctx


def test_var_positional_on_bar_keeps_legacy_regime_delivery() -> None:
    class _VarPositional(BaseStrategy):
        def on_bar(self, row, portfolio, *args):
            return args

    regime = {"regime": "legacy-positional"}

    assert invoke_strategy_on_bar(
        _VarPositional(),
        _frame().iloc[0],
        object(),
        regime,
        ctx=object(),
    ) == (regime,)


def test_positional_only_named_regime_output_keeps_legacy_delivery() -> None:
    class _PositionalOnlyRegime(BaseStrategy):
        def on_bar(self, row, portfolio, regime_output, /):
            return regime_output

    regime = {"regime": "positional-only"}

    assert invoke_strategy_on_bar(
        _PositionalOnlyRegime(),
        _frame().iloc[0],
        object(),
        regime,
        ctx=object(),
    ) is regime


def test_bare_var_keyword_does_not_opt_into_engine_injections() -> None:
    class _VarKeywordOnly(BaseStrategy):
        def __init__(self) -> None:
            self.received: dict[str, dict[str, object]] = {}

        def prepare(self, df, **kwargs):
            self.received["prepare"] = kwargs
            return df

        def on_bar(self, row, portfolio, **kwargs):
            self.received["on_bar"] = kwargs
            return []

        def stop_distance_points(self, row, **kwargs):
            self.received["stop"] = kwargs
            return 1.0

        def take_distance_points(self, row, **kwargs):
            self.received["take"] = kwargs
            return 2.0

    strategy = _VarKeywordOnly()
    frame, frames, bars = _bundles()
    row = frame.iloc[0]

    assert invoke_strategy_prepare(strategy, frame, data=frames) is frame
    assert invoke_strategy_on_bar(
        strategy,
        row,
        object(),
        {"regime": "must-not-leak"},
        ctx=object(),
        data=bars,
    ) == []
    assert invoke_strategy_stop_distance_points(
        strategy,
        row,
        symbol="EURUSD",
        ctx=object(),
        data=bars,
    ) == 1.0
    assert invoke_strategy_take_distance_points(
        strategy,
        row,
        symbol="EURUSD",
        ctx=object(),
        data=bars,
    ) == 2.0

    assert strategy.received == {
        "prepare": {},
        "on_bar": {},
        "stop": {},
        "take": {},
    }


def test_signal_symbol_is_optional_last_field_for_positional_compatibility() -> None:
    when = pd.Timestamp("2026-01-01T00:00:00Z")
    legacy = Signal(
        when,
        "buy",
        1.25,
        "reason",
        "market",
        None,
        None,
        1.0,
        "GTC",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1.0,
        42,
    )

    assert fields(Signal)[-1].name == "symbol"
    assert legacy.magic == 42
    assert legacy.symbol is None
    assert Signal(when=when, action="buy", price=1.25, symbol="GBPUSD").symbol == "GBPUSD"
