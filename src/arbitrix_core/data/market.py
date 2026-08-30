from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import pandas as pd


@dataclass(frozen=True, slots=True)
class FeedKey:
    """Provider-agnostic identity of one candle stream.

    ``provider`` is a logical binding name. It deliberately does not contain a
    provider object, database identifier, or credentials, so the same strategy
    contract can be used by local, worker, and live engines.
    """

    provider: str
    symbol: str
    timeframe: str

    def __post_init__(self) -> None:
        for field_name in ("provider", "symbol", "timeframe"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"FeedKey.{field_name} must be a non-empty string")
            if value != value.strip():
                raise ValueError(
                    f"FeedKey.{field_name} must not contain surrounding whitespace"
                )


_ValueT = TypeVar("_ValueT")


class _FeedMapping(Mapping[FeedKey, _ValueT], Generic[_ValueT]):
    """Shallow immutable mapping used by the public market-data contracts."""

    __slots__ = ("_items", "_primary")

    def __init__(
        self,
        items: Mapping[FeedKey, _ValueT],
        *,
        primary: FeedKey,
    ) -> None:
        copied = dict(items)
        for key in copied:
            if not isinstance(key, FeedKey):
                raise TypeError("market-data mappings require FeedKey keys")
        if primary not in copied:
            raise ValueError("primary feed must exist in the market-data mapping")
        object.__setattr__(self, "_items", MappingProxyType(copied))
        object.__setattr__(self, "_primary", primary)

    def __getitem__(self, key: FeedKey) -> _ValueT:
        return self._items[key]

    def __iter__(self) -> Iterator[FeedKey]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def primary(self) -> FeedKey:
        return self._primary

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __repr__(self) -> str:
        keys = ", ".join(repr(key) for key in self)
        return f"{type(self).__name__}(primary={self.primary!r}, keys=[{keys}])"


def _validate_frames(frames: Mapping[FeedKey, Any]) -> None:
    for key, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"feed {key!r} must contain a pandas DataFrame")


class MarketFrames(_FeedMapping[pd.DataFrame]):
    """Raw frames injected into ``prepare``.

    The mapping structure is immutable and snapshots the caller's mapping, but
    frames are intentionally not copied. ``execution_keys`` is an ordered,
    primary-first tuple identifying the main-provider feeds on which emitted
    signals may execute. Its order is stable for deterministic engine work.
    """

    __slots__ = ("_execution_keys",)

    def __init__(
        self,
        frames: Mapping[FeedKey, pd.DataFrame],
        *,
        primary: FeedKey,
        execution_keys: Iterable[FeedKey] | None = None,
    ) -> None:
        _validate_frames(frames)
        super().__init__(frames, primary=primary)
        if isinstance(execution_keys, (set, frozenset)):
            raise TypeError("execution_keys must be an ordered iterable")
        resolved = tuple((primary,) if execution_keys is None else execution_keys)
        if not resolved:
            raise ValueError("execution_keys must contain the primary feed")
        if any(not isinstance(key, FeedKey) for key in resolved):
            raise TypeError("execution_keys must contain FeedKey values")
        if len(set(resolved)) != len(resolved):
            raise ValueError("execution_keys must be unique")
        if resolved[0] != primary:
            raise ValueError("the primary feed must be the first execution key")
        if not set(resolved).issubset(self._items):
            raise ValueError("execution_keys must all exist in the market-data mapping")
        if any(key.provider != primary.provider for key in resolved):
            raise ValueError("execution_keys must use the same provider as the primary feed")
        object.__setattr__(self, "_execution_keys", resolved)

    @property
    def primary_frame(self) -> pd.DataFrame:
        return self[self.primary]

    @property
    def execution_keys(self) -> tuple[FeedKey, ...]:
        return self._execution_keys

    def __reduce__(self):
        return (
            _restore_market_frames,
            (dict(self), self.primary, tuple(self.execution_keys)),
        )


class PreparedMarket(_FeedMapping[pd.DataFrame]):
    """Provider-keyed frames returned by a multiprovider ``prepare`` call."""

    __slots__ = ()

    def __init__(
        self,
        frames: Mapping[FeedKey, pd.DataFrame],
        *,
        primary: FeedKey,
    ) -> None:
        _validate_frames(frames)
        super().__init__(frames, primary=primary)

    @property
    def primary_frame(self) -> pd.DataFrame:
        return self[self.primary]

    def __reduce__(self):
        return (_restore_prepared_market, (dict(self), self.primary))


class MarketBars(_FeedMapping[Any]):
    """Synchronized provider-keyed bar views injected into ``on_bar``.

    Values remain the original :class:`pandas.Series` or lightweight
    :class:`~arbitrix_core.backtest.bar_view.BarView`; no row is materialized or
    copied by this wrapper.
    """

    __slots__ = ("_decision_time",)

    def __init__(
        self,
        bars: Mapping[FeedKey, Any],
        *,
        primary: FeedKey,
        decision_time: Any,
    ) -> None:
        timestamp = pd.Timestamp(decision_time)
        if timestamp.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        timestamp = timestamp.tz_convert("UTC")

        # Local import avoids making the generic data namespace initialize the
        # backtest engine while still preserving BarView in streaming hot paths.
        from arbitrix_core.backtest.bar_view import BarView

        for key, bar in bars.items():
            if not isinstance(bar, (pd.Series, BarView)):
                raise TypeError(f"feed {key!r} must contain a pandas Series or BarView")
            bar_time = pd.Timestamp(bar.name)
            if bar_time.tzinfo is None:
                raise ValueError(f"bar for feed {key!r} must have a timezone-aware name")
            if bar_time.tz_convert("UTC") != timestamp:
                raise ValueError(
                    f"bar for feed {key!r} does not match decision_time {timestamp!s}"
                )

        super().__init__(bars, primary=primary)
        object.__setattr__(self, "_decision_time", timestamp)

    @property
    def primary_bar(self) -> Any:
        return self[self.primary]

    @property
    def decision_time(self) -> pd.Timestamp:
        return self._decision_time

    def __reduce__(self):
        return (_restore_market_bars, (dict(self), self.primary, self.decision_time))


def _restore_market_frames(
    frames: Mapping[FeedKey, pd.DataFrame],
    primary: FeedKey,
    execution_keys: Iterable[FeedKey],
) -> MarketFrames:
    return MarketFrames(frames, primary=primary, execution_keys=execution_keys)


def _restore_prepared_market(
    frames: Mapping[FeedKey, pd.DataFrame],
    primary: FeedKey,
) -> PreparedMarket:
    return PreparedMarket(frames, primary=primary)


def _restore_market_bars(
    bars: Mapping[FeedKey, Any],
    primary: FeedKey,
    decision_time: pd.Timestamp,
) -> MarketBars:
    return MarketBars(bars, primary=primary, decision_time=decision_time)


__all__ = ["FeedKey", "MarketFrames", "PreparedMarket", "MarketBars"]
