"""Data loader for arbitrix_core.

Defines the input DataFrame contract for the open-core backtest engine and
provides a CSV/parquet loader plus a typing-only ``DataProvider`` Protocol
that closed arbitrix uses to inject live broker symbol-info sources.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union, runtime_checkable

import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close", "volume")
OPTIONAL_COLS = ("spread",)

PathLike = Union[str, Path]


@runtime_checkable
class DataProvider(Protocol):
    """Symbol-info provider injected by closed arbitrix. Open-core never instantiates one."""

    def get_symbol_info(self, symbol: str) -> dict | None:
        ...


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise ValueError if df does not satisfy the open-core OHLCV schema.

    Schema:
      - ``DatetimeIndex`` with ``tz='UTC'``
      - Monotonic increasing, no duplicates
      - Required columns: open, high, low, close, volume (lowercase, float64-coercible)
      - Optional column: spread (float64-coercible)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("DataFrame index must be tz-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonic increasing")
    if df.index.has_duplicates:
        raise ValueError("DataFrame index has duplicates")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {missing}")


def load_ohlcv(path: PathLike, *, time_col: str = "time") -> pd.DataFrame:
    """Load CSV or parquet into a validated UTC-indexed OHLCV DataFrame.

    Parameters
    ----------
    path : str or Path
        File to load. Suffix ``.parquet`` selects parquet; everything else CSV.
    time_col : str
        Column name containing timestamps when input is flat (not yet indexed).
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if not isinstance(df.index, pd.DatetimeIndex):
        if time_col not in df.columns:
            raise ValueError(
                f"Input has no time column named '{time_col}' and index is not a DatetimeIndex"
            )
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col)
        df.index.name = None
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df = df.sort_index(kind="mergesort")
    df = df[~df.index.duplicated(keep="last")]

    validate_ohlcv(df)
    return df
