"""Single read surface for per-symbol metadata.

Every engine consumer (backtest, live runtime, costs, portfolio, microstructure,
risk sizing) reads from `SymbolContext` rather than scattered `getattr` chains
on `InstrumentConfig`. The registry is populated from `InstrumentConfig` rows.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

from arbitrix_core.symbols.asset_class import AssetClass, classify_asset_class, validate_asset_class
from arbitrix_core.types import InstrumentConfig
from arbitrix_core.valuation import MissingExecutionMetadata


@dataclass(frozen=True)
class SymbolContext:
    symbol: str
    asset_class: AssetClass
    multiplier: float
    point_value: float
    tick_size: float
    currency: str
    commission_scheme: Optional[str]
    fee_per_contract: float
    fee_min_per_order: float
    min_order_size: float
    target_spread: Optional[float] = None


_REGISTRY: dict[str, SymbolContext] = {}
_LOCK = threading.RLock()


def register_symbol_context(ctx: SymbolContext) -> None:
    with _LOCK:
        _REGISTRY[ctx.symbol.lower()] = ctx


def get_symbol_context(symbol: str) -> SymbolContext:
    key = str(symbol).lower()
    with _LOCK:
        ctx = _REGISTRY.get(key)
    if ctx is None:
        raise KeyError(f"SymbolContext for {symbol!r} not registered")
    return ctx


def clear_symbol_context_registry() -> None:
    with _LOCK:
        _REGISTRY.clear()


def build_symbol_context_from_instrument(
    inst: InstrumentConfig,
    *,
    symbol: str,
) -> SymbolContext:
    """Build a fully-populated :class:`SymbolContext` from an :class:`InstrumentConfig`.

    Auto-classifies ``asset_class`` from ``security_type`` when
    ``inst.asset_class`` is ``None``; validates it against the known taxonomy
    when explicitly set.
    """
    asset_class = (
        validate_asset_class(inst.asset_class)
        if inst.asset_class is not None
        else classify_asset_class(inst.security_type)
    )
    missing = [
        field
        for field in ("tick_size", "point_value", "min_order_size")
        if getattr(inst, field, None) is None
    ]
    if inst.multiplier is None and inst.contract_size is None:
        missing.append("contract_size")
    if missing:
        raise MissingExecutionMetadata(
            f"{symbol}: unresolved execution metadata: {', '.join(missing)}"
        )
    multiplier = float(
        inst.multiplier if inst.multiplier is not None else inst.contract_size
    )
    point_value = float(inst.point_value)
    tick_size = float(inst.tick_size)
    min_order_size = float(inst.min_order_size)
    invalid = [
        name
        for name, value in (
            ("contract_size", multiplier),
            ("point_value", point_value),
            ("tick_size", tick_size),
            ("min_order_size", min_order_size),
        )
        if not math.isfinite(value) or value <= 0
    ]
    if invalid:
        raise MissingExecutionMetadata(
            f"{symbol}: invalid execution metadata: {', '.join(invalid)}"
        )
    return SymbolContext(
        symbol=symbol,
        asset_class=asset_class,
        multiplier=multiplier,
        point_value=point_value,
        tick_size=tick_size,
        currency=str(inst.currency or "USD"),
        commission_scheme=inst.commission_scheme,
        fee_per_contract=float(inst.fee_per_contract or 0.0),
        fee_min_per_order=float(inst.fee_min_per_order or 0.0),
        min_order_size=min_order_size,
        target_spread=(
            None if inst.target_spread is None else float(inst.target_spread)
        ),
    )
