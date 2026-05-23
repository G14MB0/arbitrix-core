from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List, Tuple

SUPPORTED_STRATEGY_TIMEFRAMES: Tuple[str, ...] = (
    "M1",
    "M5",
    "M10",
    "M12",
    "M15",
    "M20",
    "M30",
    "H1",
    "H2",
    "H3",
    "H4",
    "H6",
    "H8",
    "H12",
    "D1",
    "W1",
    "MN1",
)
DEFAULT_STRATEGY_TIMEFRAME = "M5"
DEFAULT_STRATEGY_SESSION_TIMEZONE = "UTC"

_SUPPORTED_SESSION_TIMEZONES: Tuple[str, ...] = (
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


def list_supported_strategy_timeframes() -> List[str]:
    return list(SUPPORTED_STRATEGY_TIMEFRAMES)


def list_supported_strategy_sessions() -> List[str]:
    return list(_SUPPORTED_SESSION_TIMEZONES)


def list_strategy_parameter_allowed_value_presets() -> Dict[str, List[Any]]:
    sessions = list_supported_strategy_sessions()
    return {
        "timeframes": list_supported_strategy_timeframes(),
        "sessions": list(sessions),
        "timezones": list(sessions),
        "booleans": [True, False],
    }


def resolve_strategy_parameter_allowed_values_preset(name: Any) -> List[Any] | None:
    preset_name = str(name or "").strip().lower()
    if not preset_name:
        return None
    presets = list_strategy_parameter_allowed_value_presets()
    values = presets.get(preset_name)
    if values is None:
        return None
    return list(values)


def _build_metadata(
    *,
    group: str,
    optimizable: bool,
    allowed_values: List[Any],
    allowed_values_preset: str,
    description: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "group": str(group),
        "optimizable": bool(optimizable),
        "allowed_values": list(allowed_values),
        "allowed_values_preset": str(allowed_values_preset),
    }
    if description:
        payload["description"] = str(description)
    if isinstance(metadata, dict):
        payload.update(metadata)
    return payload


def strategy_timeframe_field(
    *,
    default: str = DEFAULT_STRATEGY_TIMEFRAME,
    group: str = "identity",
    optimizable: bool = True,
    description: str | None = None,
    metadata: Dict[str, Any] | None = None,
):
    return field(
        default=str(default),
        metadata=_build_metadata(
            group=group,
            optimizable=optimizable,
            allowed_values=list_supported_strategy_timeframes(),
            allowed_values_preset="timeframes",
            description=description,
            metadata=metadata,
        ),
    )


def strategy_session_timezone_field(
    *,
    default: str = DEFAULT_STRATEGY_SESSION_TIMEZONE,
    group: str = "session",
    optimizable: bool = True,
    description: str | None = None,
    metadata: Dict[str, Any] | None = None,
):
    return field(
        default=str(default),
        metadata=_build_metadata(
            group=group,
            optimizable=optimizable,
            allowed_values=list_supported_strategy_sessions(),
            allowed_values_preset="sessions",
            description=description,
            metadata=metadata,
        ),
    )


def strategy_boolean_flag_field(
    *,
    default: bool,
    group: str = "default",
    optimizable: bool = True,
    description: str | None = None,
    metadata: Dict[str, Any] | None = None,
):
    return field(
        default=bool(default),
        metadata=_build_metadata(
            group=group,
            optimizable=optimizable,
            allowed_values=[True, False],
            allowed_values_preset="booleans",
            description=description,
            metadata=metadata,
        ),
    )


__all__ = [
    "DEFAULT_STRATEGY_SESSION_TIMEZONE",
    "DEFAULT_STRATEGY_TIMEFRAME",
    "SUPPORTED_STRATEGY_TIMEFRAMES",
    "list_strategy_parameter_allowed_value_presets",
    "list_supported_strategy_sessions",
    "list_supported_strategy_timeframes",
    "resolve_strategy_parameter_allowed_values_preset",
    "strategy_boolean_flag_field",
    "strategy_session_timezone_field",
    "strategy_timeframe_field",
]
