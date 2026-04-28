"""User-defined cost model: flat commission, no slippage, no swap."""
from __future__ import annotations

MODULE_NAME = __name__

_FLAT_COMMISSION = 1.5


def commission_one_side(symbol: str, price: float, volume_lot: float) -> float:
    return max(_FLAT_COMMISSION * volume_lot, 0.01)


def commission_round_turn(symbol: str, price: float, volume_lot: float) -> float:
    return 2.0 * commission_one_side(symbol, price, volume_lot)


def spread_cost(symbol: str, spread_points: float, volume_lot: float) -> float:
    return 0.5 * spread_points * volume_lot


def slippage_cost(symbol: str, slippage_points: float, volume_lot: float) -> float:
    return 0.0


def swap_points(symbol: str, direction: str, static_override=None) -> float:
    return 0.0


def swap_cost_per_day(
    symbol: str, volume_lot: float, direction: str, static_override=None
) -> float:
    return 0.0
