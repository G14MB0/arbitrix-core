from __future__ import annotations

from collections.abc import Iterable, Iterator, KeysView
from typing import Any

import pandas as pd


class BarViewSource:
    """Array-backed source for lightweight streaming bar rows."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.columns = list(frame.columns)
        self._column_set = set(self.columns)
        self._arrays = {
            column: frame[column].to_numpy(copy=False) for column in self.columns
        }
        self._index = frame.index

    def row_at(self, index: int) -> "BarView":
        return BarView(self, index)


class BarView:
    """Small pd.Series-like row facade for backtest streaming hot paths."""

    __slots__ = ("_source", "_index", "_series")

    def __init__(self, source: BarViewSource, index: int) -> None:
        self._source = source
        self._index = int(index)
        self._series: pd.Series | None = None

    @property
    def name(self) -> Any:
        return self._source._index[self._index]

    @property
    def index(self) -> pd.Index:
        return pd.Index(self._source.columns)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return self._source._arrays[key][self._index]
            except KeyError:
                raise KeyError(key) from None
        return self._materialized()[key]

    def __contains__(self, key: object) -> bool:
        return key in self._source._column_set

    def __len__(self) -> int:
        return len(self._source.columns)

    def __iter__(self) -> Iterator[Any]:
        for column in self._source.columns:
            yield self._source._arrays[column][self._index]

    def __getattr__(self, name: str) -> Any:
        if name in self._source._column_set:
            return self[name]
        return getattr(self._materialized(), name)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._source._column_set:
            return default
        return self._source._arrays[key][self._index]

    def keys(self) -> KeysView[str]:
        return self._materialized().keys()

    def items(self) -> Iterable[tuple[Any, Any]]:
        for column in self._source.columns:
            yield column, self._source._arrays[column][self._index]

    def to_dict(self) -> dict[str, Any]:
        return {
            column: self._source._arrays[column][self._index]
            for column in self._source.columns
        }

    def copy(self) -> pd.Series:
        return self._materialized().copy()

    def _materialized(self) -> pd.Series:
        if self._series is None:
            self._series = pd.Series(
                [self._source._arrays[column][self._index] for column in self._source.columns],
                index=self._source.columns,
                name=self.name,
            )
        return self._series
