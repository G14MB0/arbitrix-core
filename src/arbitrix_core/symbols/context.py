"""Single read surface for per-symbol metadata.

Every engine consumer (backtest, live runtime, costs, portfolio, microstructure,
risk sizing) reads from `SymbolContext` rather than scattered `getattr` chains
on `InstrumentConfig`. The registry is populated from `InstrumentConfig` rows.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from arbitrix_core.symbols.asset_class import AssetClass, classify_asset_class, validate_asset_class
from arbitrix_core.types import InstrumentConfig


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
    margin_per_contract: Optional[float]  # populated by Sub-spec 2


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
    multiplier = float(
        inst.multiplier
        if inst.multiplier is not None
        else (inst.contract_size if inst.contract_size is not None else 1.0)
    )
    point_value = float(
        inst.point_value if inst.point_value is not None else multiplier
    )
    return SymbolContext(
        symbol=symbol,
        asset_class=asset_class,
        multiplier=multiplier,
        point_value=point_value,
        tick_size=float(inst.tick_size or 1.0),
        currency=str(inst.currency or "USD"),
        commission_scheme=inst.commission_scheme,
        fee_per_contract=float(inst.fee_per_contract or 0.0),
        fee_min_per_order=float(inst.fee_min_per_order or 0.0),
        min_order_size=float(inst.min_order_size or 1.0),
        margin_per_contract=None,  # Sub-spec 2 populates
    )
