"""Single read surface for per-symbol metadata.

Every engine consumer (backtest, live runtime, costs, portfolio, microstructure,
risk sizing) reads from `SymbolContext` rather than scattered `getattr` chains
on `InstrumentConfig`. The registry is populated from `InstrumentConfig` rows.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from arbitrix_core.symbols.asset_class import AssetClass


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
