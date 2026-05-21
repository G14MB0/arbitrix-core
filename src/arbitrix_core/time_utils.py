from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


def coerce_utc_datetime(value: Any) -> Optional[datetime]:
    """Coerce a datetime/Timestamp to a UTC-aware python ``datetime``.

    ``None``, ``NaT``, or ``NaN`` return ``None``. Naive input is assumed to be
    UTC and localized; tz-aware input is converted to UTC. The result is a plain
    python :class:`datetime.datetime` (not a :class:`pandas.Timestamp`).

    This is the single shared UTC coercer for the order/fill/reconciliation and
    cache paths — every site delegates here so the NaT guard and the
    naive-as-UTC assumption stay identical everywhere (ARB-90).
    """
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _validate_tz(tz_name: str) -> str:
    name = str(tz_name or "").strip()
    if not name:
        raise ValueError("timezone name must not be empty")
    try:
        ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone '{name}'") from exc
    return name


def normalize_ohlcv_index_to_utc(index: Any) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(index, errors="coerce")
    dt_index = pd.DatetimeIndex(parsed)
    if len(dt_index) == 0:
        return dt_index.tz_localize("UTC")
    if dt_index.tz is None:
        raise ValueError(
            "OHLCV index must be timezone-aware (UTC). "
            "Naive timestamps are not accepted — ensure the data source returns UTC-aware data."
        )
    return dt_index.tz_convert("UTC").sort_values()


def normalize_ohlcv_frame_to_utc(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return frame
    normalized = frame.copy()
    normalized.index = normalize_ohlcv_index_to_utc(normalized.index)
    normalized = normalized[~normalized.index.isna()]
    return normalized.sort_index()


def to_market_time(ts: Any, tz: str) -> pd.Timestamp:
    market_tz = _validate_tz(tz)
    value = to_utc_time(ts)
    return value.tz_convert(market_tz)


def to_utc_time(ts: Any, tz: str | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp(ts)
    if tz is not None:
        local_tz = _validate_tz(tz)
    else:
        local_tz = None
    if timestamp.tzinfo is None:
        source_tz = local_tz or "UTC"
        localized = pd.DatetimeIndex([timestamp]).tz_localize(
            source_tz,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
        timestamp = pd.Timestamp(localized[0])
    return timestamp.tz_convert("UTC")


def session_day(ts: Any, tz: str) -> date:
    return to_market_time(ts, tz).date()


def session_hour(ts: Any, tz: str) -> float:
    local = to_market_time(ts, tz)
    return float(local.hour) + (float(local.minute) / 60.0)


def is_in_session(
    ts: Any,
    tz: str,
    windows: Iterable[Sequence[str]],
) -> bool:
    local = to_market_time(ts, tz)
    minute_of_day = int(local.hour) * 60 + int(local.minute)
    for window in windows:
        if not isinstance(window, Sequence) or len(window) != 2:
            continue
        start_min = _parse_hhmm(window[0])
        end_min = _parse_hhmm(window[1])
        if start_min is None or end_min is None:
            continue
        if start_min <= end_min:
            if start_min <= minute_of_day <= end_min:
                return True
        else:
            # Overnight window (e.g. 22:00 -> 02:00)
            if minute_of_day >= start_min or minute_of_day <= end_min:
                return True
    return False


def _parse_hhmm(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text or ":" not in text:
        return None
    hour_text, minute_text = text.split(":", 1)
    try:
        hour = int(hour_text)
        minute = int(minute_text)
    except Exception:
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour * 60 + minute


__all__ = [
    "coerce_utc_datetime",
    "is_in_session",
    "normalize_ohlcv_frame_to_utc",
    "normalize_ohlcv_index_to_utc",
    "session_day",
    "session_hour",
    "to_market_time",
    "to_utc_time",
]
