from __future__ import annotations

from typing import Any, Dict, Optional

import arbitrix_core.costs as costs
from .. import base

__all__ = [
    "configure",
    "commission_one_side",
    "commission_round_turn",
    "spread_cost",
    "slippage_cost",
    "swap_points",
    "swap_cost_per_day",
]

MODULE_NAME = __name__


def configure(context: Optional[Dict[str, Any]] = None) -> None:  # pragma: no cover - simple hook
    """Parameterized model does not require extra setup beyond provided parameters."""
    return None


def _params(symbol: Optional[str] = None) -> Dict[str, Any]:
    return costs.model_parameters(symbol, module_name=MODULE_NAME) or {}


def _numeric(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _param_value(params: Dict[str, Any], key: str) -> Optional[float]:
    if not params:
        return None
    if key not in params:
        return None
    return _numeric(params.get(key))


def _spread_mode(params: Dict[str, Any]) -> str:
    spread_model = params.get("spread_model") if isinstance(params.get("spread_model"), dict) else {}
    mode = spread_model.get("mode", params.get("spread_mode"))
    normalized = str(mode or "").strip().lower()
    if normalized in {"static", "provider_only", "stochastic_only", "provider_plus_stochastic"}:
        return normalized
    return ""


def commission_one_side(symbol: str, price: float, volume_lot: float) -> float:
    params = _params(symbol)
    fixed_fee = _param_value(params, "commission_per_lot")
    if fixed_fee is not None:
        commission = abs(volume_lot) * fixed_fee
        return max(commission, base.commission_minimum(volume_lot))
    notional = base.trade_notional(symbol, price, volume_lot)
    return base.commission_from_notional(
        symbol=symbol,
        price=price,
        volume_lot=volume_lot,
        notional=notional,
    )


def commission_round_turn(symbol: str, price: float, volume_lot: float) -> float:
    return commission_one_side(symbol, price, volume_lot) * 2.0


def _resolve_points(configured: Optional[float], provided: Optional[float]) -> float:
    if provided is not None:
        try:
            provided_val = float(provided)
        except (TypeError, ValueError):
            provided_val = 0.0
        else:
            if provided_val != 0:
                return provided_val
    return float(configured or 0.0)


def spread_cost(symbol: str, spread_points: float, volume_lot: float) -> float:
    params = _params(symbol)
    configured_points = _param_value(params, "spread_points")
    mode = _spread_mode(params)
    provided_points = _numeric(spread_points)
    if mode in {"provider_only", "stochastic_only", "provider_plus_stochastic"}:
        points = float(provided_points or 0.0)
    elif mode == "static":
        spread_model = params.get("spread_model") if isinstance(params.get("spread_model"), dict) else {}
        static_points = _numeric(spread_model.get("static_points"))
        if static_points is not None:
            points = float(static_points)
        else:
            points = _resolve_points(configured_points, spread_points)
    else:
        points = _resolve_points(configured_points, spread_points)
    points = max(float(points), 0.0)
    pv = base.get_point_value(symbol)
    ts = base.tick_size(symbol)
    cost = float(points) * float(ts) * float(pv) * float(volume_lot)
    # try:
    #     import logging
    #     logging.getLogger(__name__).debug(
    #         "spread_cost | symbol=%s points=%s tick_size=%s point_value=%s cost=%s configured=%s provided=%s volume=%s params=%s",
    #         symbol,
    #         points,
    #         ts,
    #         pv,
    #         cost,
    #         configured_points,
    #         spread_points,
    #         volume_lot,
    #         sorted(params.keys()),
    #     )
    # except Exception:
    #     pass
    if points == 0:
        return 0.0
    return cost


def slippage_cost(symbol: str, slippage_points: float, volume_lot: float) -> float:
    params = _params(symbol)
    configured_points = _param_value(params, "slippage_points")
    points = _resolve_points(configured_points, slippage_points)
    pv = base.get_point_value(symbol)
    ts = base.tick_size(symbol)
    cost = float(points) * float(ts) * float(pv) * float(volume_lot)
    # try:
    #     import logging
    #     logging.getLogger(__name__).debug(
    #         "slippage_cost | symbol=%s points=%s tick_size=%s point_value=%s cost=%s configured=%s provided=%s volume=%s params=%s",
    #         symbol,
    #         points,
    #         ts,
    #         pv,
    #         cost,
    #         configured_points,
    #         slippage_points,
    #         volume_lot,
    #         sorted(params.keys()),
    #     )
    # except Exception:
    #     pass
    if points == 0:
        return 0.0
    return cost


def swap_points(symbol: str, direction: str, static_override: Optional[dict] = None) -> float:
    override = static_override or {}
    side_key = "long" if direction == "long" else "short"
    # Explicit override wins first
    if isinstance(override, dict) and side_key in override:
        value = _numeric(override.get(side_key))
        if value is not None:
            return value * base.tick_size(symbol)

    params = _params(symbol)
    configured = _param_value(params, f"swap_points_{side_key}")
    if configured is not None:
        return float(configured) * base.tick_size(symbol)

    fallback = base.swap_points_static(symbol, direction, static_override)
    if fallback is not None:
        return fallback
    return base.swap_points_from_cache(symbol, direction)


def _swap_cost_from_points(symbol: str, volume_lot: float, points: float) -> float:
    pv = base.get_point_value(symbol)
    return float(points) * pv * float(volume_lot)


def swap_cost_per_day(
    symbol: str,
    volume_lot: float,
    direction: str,
    static_override: Optional[dict] = None,
) -> float:
    params = _params(symbol)
    side_key = "long" if direction == "long" else "short"
    override = static_override or {}
    is_weekend = bool(override.get("weekend"))

    # 1) Explicit per-day currency override
    if isinstance(override, dict) and side_key in override:
        override_cost = _numeric(override.get(side_key))
        if override_cost is not None:
            return float(override_cost) * float(volume_lot)

    # 2) Weekend-specific override if provided
    weekend_key = f"swap_cost_weekend_{side_key}"
    if is_weekend:
        weekend_cost = _param_value(params, weekend_key)
        if weekend_cost is not None:
            return float(weekend_cost) * float(volume_lot)

    # 3) Standard per-day currency override
    daily_cost = _param_value(params, f"swap_cost_per_day_{side_key}")
    if daily_cost is not None:
        return float(daily_cost) * float(volume_lot)

    # 4) Fallback to configured swap points (or provider values)
    points_override = _param_value(params, f"swap_points_{side_key}")
    if points_override is not None:
        return _swap_cost_from_points(symbol, volume_lot, points_override * base.tick_size(symbol))

    points = swap_points(symbol, direction, static_override)
    return _swap_cost_from_points(symbol, volume_lot, points)
