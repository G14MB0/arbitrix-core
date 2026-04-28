"""
Backward-compatible entrypoint that now delegates to the parameterized model.

The default cost model accepts user-provided parameters (commission, spread,
slippage, swap) and falls back to provider/instrument data when absent.
"""

from __future__ import annotations

from .parameterized import (  # noqa: F401
    commission_one_side,
    commission_round_turn,
    configure,
    slippage_cost,
    spread_cost,
    swap_cost_per_day,
    swap_points,
)

MODULE_NAME = __name__

__all__ = [
    "configure",
    "commission_one_side",
    "commission_round_turn",
    "spread_cost",
    "slippage_cost",
    "swap_points",
    "swap_cost_per_day",
]
