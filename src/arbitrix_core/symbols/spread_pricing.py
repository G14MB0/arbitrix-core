"""Quote-side translation for reference-priced pending orders."""

from __future__ import annotations

import math
from typing import Optional


MAX_SPREAD_PRICE_RATIO = 0.05


class SpreadPriceError(ValueError):
    """Structured validation failure for a configured or provider spread."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


def validate_spread_price(
    spread_price: float,
    *,
    tick_size: float,
    reference_price: float,
    max_ratio: float = MAX_SPREAD_PRICE_RATIO,
) -> float:
    """Validate and return a full bid/ask spread in quote-price units."""

    try:
        spread = float(spread_price)
    except (TypeError, ValueError) as exc:
        raise SpreadPriceError("not_numeric", "spread must be numeric") from exc
    if not math.isfinite(spread):
        raise SpreadPriceError("not_finite", "spread must be finite")
    if spread <= 0.0:
        raise SpreadPriceError("non_positive", "spread must be greater than zero")

    try:
        tick = float(tick_size)
    except (TypeError, ValueError) as exc:
        raise SpreadPriceError("tick_invalid", "tick_size must be numeric") from exc
    if not math.isfinite(tick) or tick <= 0.0:
        raise SpreadPriceError("tick_invalid", "tick_size must be finite and positive")
    if spread < tick and not math.isclose(spread, tick, rel_tol=1e-9, abs_tol=1e-15):
        raise SpreadPriceError("below_tick", "spread must be at least one tick")

    ticks = spread / tick
    if not math.isclose(ticks, round(ticks), rel_tol=1e-9, abs_tol=1e-8):
        raise SpreadPriceError(
            "not_tick_aligned",
            f"spread {spread} is not aligned to tick_size {tick}",
        )

    try:
        reference = float(reference_price)
    except (TypeError, ValueError) as exc:
        raise SpreadPriceError(
            "reference_invalid", "reference_price must be numeric"
        ) from exc
    if not math.isfinite(reference) or reference <= 0.0:
        raise SpreadPriceError(
            "reference_invalid", "reference_price must be finite and positive"
        )
    if spread / reference >= float(max_ratio):
        raise SpreadPriceError(
            "ratio_too_high",
            f"spread {spread} must be below {float(max_ratio):.2%} of reference price {reference}",
        )
    return spread


def resolve_effective_spread(
    *,
    target_spread: Optional[float],
    current_spread: Optional[float],
    tick_size: float,
    reference_price: float,
) -> float:
    """Prefer the configured target spread, otherwise validate current spread."""

    selected = target_spread if target_spread is not None else current_spread
    if selected is None:
        raise SpreadPriceError(
            "missing",
            "target_spread is unset and current provider spread is unavailable",
        )
    return validate_spread_price(
        selected,
        tick_size=tick_size,
        reference_price=reference_price,
    )


def translate_reference_price(
    *,
    reference_price: float,
    action: str,
    spread_price: float,
    reference_basis: str,
) -> float:
    """Translate a reference price to the quote side executing *action*."""

    price = float(reference_price)
    spread = float(spread_price)
    side = str(action).strip().lower()
    if side not in {"buy", "sell"}:
        raise ValueError(f"Unsupported execution action: {action!r}")

    basis = str(reference_basis or "").strip().upper()
    if basis == "BID":
        offset = spread if side == "buy" else 0.0
    elif basis in {"MIDPOINT", "TRADES"}:
        offset = spread / 2.0 if side == "buy" else -spread / 2.0
    elif basis == "ASK":
        offset = 0.0 if side == "buy" else -spread
    else:
        raise ValueError(f"Unsupported reference quote basis: {reference_basis!r}")
    return price + offset


__all__ = [
    "MAX_SPREAD_PRICE_RATIO",
    "SpreadPriceError",
    "resolve_effective_spread",
    "translate_reference_price",
    "validate_spread_price",
]
