from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

DEFAULT_SYMBOL_TIMEZONE = "UTC"

# Curated list shared across API validation and UI selector.
ALLOWED_SYMBOL_TIMEZONES: Tuple[str, ...] = (
    "UTC",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Sao_Paulo",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Europe/Rome",
    "Europe/Madrid",
    "Europe/Zurich",
    "Europe/Athens",
    "Europe/Istanbul",
    "Asia/Dubai",
    "Asia/Kolkata",
    "Asia/Singapore",
    "Asia/Hong_Kong",
    "Asia/Shanghai",
    "Asia/Tokyo",
    "Australia/Sydney",
    "Pacific/Auckland",
)

PROVIDER_TIME_SEMANTIC_DEFAULT = "default"
PROVIDER_TIME_SEMANTIC_MT5_WALL_CLOCK = "mt5_wall_clock"
_SUPPORTED_PROVIDER_TIME_SEMANTICS = {
    PROVIDER_TIME_SEMANTIC_DEFAULT,
    PROVIDER_TIME_SEMANTIC_MT5_WALL_CLOCK,
}


def list_supported_symbol_timezones() -> list[str]:
    return list(ALLOWED_SYMBOL_TIMEZONES)


def normalize_symbol_timezone(value: Any, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError("timezone is required")
    tz_name = str(value).strip()
    if not tz_name:
        if allow_none:
            return None
        raise ValueError("timezone is required")
    if tz_name not in ALLOWED_SYMBOL_TIMEZONES:
        raise ValueError(f"Unsupported timezone '{tz_name}'")
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:  # pragma: no cover - depends on host tzdata
        raise ValueError(f"Timezone '{tz_name}' is not available on this runtime") from exc
    return tz_name


def normalize_provider_time_semantic(value: Any) -> str:
    mode = str(value or PROVIDER_TIME_SEMANTIC_DEFAULT).strip().lower()
    if mode not in _SUPPORTED_PROVIDER_TIME_SEMANTICS:
        raise ValueError(f"Unsupported provider time semantic '{mode}'")
    return mode


def normalize_ohlcv_index_to_utc(
    index: Any,
    *,
    symbol_timezone: str,
    provider_semantic_mode: str = PROVIDER_TIME_SEMANTIC_DEFAULT,
) -> pd.DatetimeIndex:
    tz_name = normalize_symbol_timezone(symbol_timezone)
    mode = normalize_provider_time_semantic(provider_semantic_mode)
    parsed = pd.to_datetime(index, errors="coerce")
    dt_index = pd.DatetimeIndex(parsed)
    if len(dt_index) == 0:
        return dt_index.tz_localize("UTC")
    if mode == PROVIDER_TIME_SEMANTIC_MT5_WALL_CLOCK:
        # MT5 feeds are interpreted as wall-clock bars in provider/local market timezone.
        naive = dt_index.tz_localize(None) if dt_index.tz is not None else dt_index
        localized = naive.tz_localize(tz_name, ambiguous="infer", nonexistent="shift_forward")
        return localized.tz_convert("UTC").sort_values()
    if dt_index.tz is None:
        localized = dt_index.tz_localize(tz_name, ambiguous="infer", nonexistent="shift_forward")
        return localized.tz_convert("UTC").sort_values()
    return dt_index.tz_convert("UTC").sort_values()


def normalize_ohlcv_frame_to_utc(
    frame: pd.DataFrame,
    *,
    symbol_timezone: str,
    provider_semantic_mode: str = PROVIDER_TIME_SEMANTIC_DEFAULT,
) -> pd.DataFrame:
    if frame is None:
        return frame
    normalized = frame.copy()
    normalized.index = normalize_ohlcv_index_to_utc(
        normalized.index,
        symbol_timezone=symbol_timezone,
        provider_semantic_mode=provider_semantic_mode,
    )
    normalized = normalized[~normalized.index.isna()]
    return normalized.sort_index()


def to_market_time(ts: Any, tz: str) -> pd.Timestamp:
    market_tz = normalize_symbol_timezone(tz)
    value = to_utc_time(ts)
    return value.tz_convert(market_tz)


def to_utc_time(ts: Any, tz: str | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp(ts)
    local_tz = normalize_symbol_timezone(tz, allow_none=True)
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
    "ALLOWED_SYMBOL_TIMEZONES",
    "DEFAULT_SYMBOL_TIMEZONE",
    "PROVIDER_TIME_SEMANTIC_DEFAULT",
    "PROVIDER_TIME_SEMANTIC_MT5_WALL_CLOCK",
    "is_in_session",
    "list_supported_symbol_timezones",
    "normalize_ohlcv_frame_to_utc",
    "normalize_ohlcv_index_to_utc",
    "normalize_provider_time_semantic",
    "normalize_symbol_timezone",
    "session_day",
    "session_hour",
    "to_market_time",
    "to_utc_time",
]
