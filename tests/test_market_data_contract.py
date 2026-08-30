from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from arbitrix_core.backtest.bar_view import BarViewSource
from arbitrix_core.data.market import FeedKey, MarketBars, MarketFrames, PreparedMarket


def _frame(*, start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=2, freq="5min", tz="UTC")
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


def test_feed_key_is_immutable_and_preserves_exact_identity() -> None:
    key = FeedKey(provider="execution", symbol="EURUSD.a", timeframe="M5")

    assert (key.provider, key.symbol, key.timeframe) == (
        "execution",
        "EURUSD.a",
        "M5",
    )
    with pytest.raises(FrozenInstanceError):
        key.symbol = "GBPUSD"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", ""),
        ("provider", "  "),
        ("provider", " main"),
        ("symbol", "EURUSD "),
        ("symbol", ""),
        ("timeframe", ""),
    ],
)
def test_feed_key_rejects_empty_identity_parts(field: str, value: str) -> None:
    values = {"provider": "main", "symbol": "EURUSD", "timeframe": "M5"}
    values[field] = value

    with pytest.raises(ValueError, match=field):
        FeedKey(**values)


def test_market_frames_is_a_shallow_immutable_snapshot() -> None:
    primary = FeedKey("main", "EURUSD", "M5")
    second = FeedKey("main", "GBPUSD", "M5")
    auxiliary = FeedKey("macro", "DXY", "M5")
    primary_frame = _frame()
    source = {
        primary: primary_frame,
        second: _frame(),
        auxiliary: _frame(),
    }

    bundle = MarketFrames(
        source,
        primary=primary,
        execution_keys=(primary, second),
    )
    source.pop(auxiliary)

    assert set(bundle) == {primary, second, auxiliary}
    assert bundle.primary == primary
    assert bundle.primary_frame is primary_frame
    assert bundle.execution_keys == (primary, second)
    with pytest.raises(TypeError):
        bundle[primary] = _frame()  # type: ignore[index]


def test_market_frames_rejects_missing_primary_or_execution_feed() -> None:
    primary = FeedKey("main", "EURUSD", "M5")
    missing = FeedKey("main", "GBPUSD", "M5")

    with pytest.raises(ValueError, match="primary"):
        MarketFrames({missing: _frame()}, primary=primary)

    with pytest.raises(ValueError, match="execution_keys"):
        MarketFrames(
            {primary: _frame()},
            primary=primary,
            execution_keys=(primary, missing),
        )

    auxiliary = FeedKey("auxiliary", "DXY", "M5")
    with pytest.raises(ValueError, match="same provider"):
        MarketFrames(
            {primary: _frame(), auxiliary: _frame()},
            primary=primary,
            execution_keys=(primary, auxiliary),
        )

    with pytest.raises(TypeError, match="FeedKey"):
        MarketFrames(  # type: ignore[arg-type]
            {None: _frame()},
            primary=primary,
        )


def test_market_frames_requires_unique_primary_first_execution_order() -> None:
    primary = FeedKey("main", "EURUSD", "M5")
    second = FeedKey("main", "GBPUSD", "M5")
    frames = {primary: _frame(), second: _frame()}

    with pytest.raises(TypeError, match="ordered iterable"):
        MarketFrames(frames, primary=primary, execution_keys={primary, second})
    with pytest.raises(ValueError, match="first execution key"):
        MarketFrames(frames, primary=primary, execution_keys=(second, primary))
    with pytest.raises(ValueError, match="unique"):
        MarketFrames(
            frames,
            primary=primary,
            execution_keys=(primary, second, second),
        )


def test_prepared_market_keeps_provider_symbol_timeframe_keys() -> None:
    primary = FeedKey("main", "EURUSD", "M5")
    auxiliary = FeedKey("macro", "DXY", "M5")
    prepared_primary = _frame().assign(momentum=[0.0, 1.0])
    prepared_auxiliary = _frame().assign(regime=[1, 2])

    prepared = PreparedMarket(
        {primary: prepared_primary, auxiliary: prepared_auxiliary},
        primary=primary,
    )

    assert prepared.primary_frame is prepared_primary
    assert prepared[auxiliary] is prepared_auxiliary


def test_market_bars_preserves_series_and_bar_view_without_materializing() -> None:
    primary = FeedKey("main", "EURUSD", "M5")
    auxiliary = FeedKey("macro", "DXY", "M5")
    primary_frame = _frame()
    auxiliary_frame = _frame()
    primary_row = primary_frame.iloc[0]
    auxiliary_row = BarViewSource(auxiliary_frame).row_at(0)

    bars = MarketBars(
        {primary: primary_row, auxiliary: auxiliary_row},
        primary=primary,
        decision_time=pd.Timestamp("2026-01-01T01:00:00+01:00"),
    )

    assert bars.decision_time == pd.Timestamp("2026-01-01T00:00:00Z")
    assert bars.primary_bar is primary_row
    assert bars[auxiliary] is auxiliary_row
    assert bars[auxiliary]["close"] == pytest.approx(1.1)


def test_market_bars_requires_utc_aware_matching_timestamps() -> None:
    primary = FeedKey("main", "EURUSD", "M5")
    row = _frame().iloc[0]

    with pytest.raises(ValueError, match="timezone-aware"):
        MarketBars(
            {primary: row},
            primary=primary,
            decision_time=pd.Timestamp("2026-01-01T00:00:00"),
        )

    with pytest.raises(ValueError, match="does not match decision_time"):
        MarketBars(
            {primary: row},
            primary=primary,
            decision_time=pd.Timestamp("2026-01-01T00:05:00Z"),
        )


def test_market_contracts_survive_worker_process_serialization() -> None:
    primary = FeedKey("main", "EURUSD", "M5")
    frame = _frame()
    frames = MarketFrames({primary: frame}, primary=primary)
    prepared = PreparedMarket({primary: frame}, primary=primary)
    bars = MarketBars(
        {primary: frame.iloc[0]},
        primary=primary,
        decision_time=frame.index[0],
    )

    restored_frames = pickle.loads(pickle.dumps(frames))
    restored_prepared = pickle.loads(pickle.dumps(prepared))
    restored_bars = pickle.loads(pickle.dumps(bars))

    assert restored_frames.primary == primary
    assert restored_frames.execution_keys == (primary,)
    pd.testing.assert_frame_equal(restored_frames.primary_frame, frame)
    pd.testing.assert_frame_equal(restored_prepared.primary_frame, frame)
    pd.testing.assert_series_equal(restored_bars.primary_bar, frame.iloc[0])
