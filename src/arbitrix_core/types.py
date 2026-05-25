from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InstrumentConfig:
    """Lightweight metadata describing a tradable instrument.

    Extracted into the core namespace so that importing backtesting helpers
    does not implicitly require the heavier configuration loader machinery.
    """

    ib_symbol: str
    security_type: str = "CFD"
    exchange: str = "SMART"
    currency: str = "USD"
    conid: Optional[int] = None
    trading_exchange: Optional[str] = None
    trading_security_type: Optional[str] = None
    trading_local_symbol: Optional[str] = None
    trading_primary_exchange: Optional[str] = None
    trading_multiplier: Optional[float] = None
    trading_expiry: Optional[str] = None
    local_symbol: Optional[str] = None
    primary_exchange: Optional[str] = None
    multiplier: Optional[float] = None
    expiry: Optional[str] = None
    what_to_show: str = "TRADES"
    history_conid: Optional[int] = None
    history_symbol: Optional[str] = None
    history_security_type: Optional[str] = None
    history_exchange: Optional[str] = None
    history_currency: Optional[str] = None
    history_local_symbol: Optional[str] = None
    history_primary_exchange: Optional[str] = None
    history_multiplier: Optional[float] = None
    history_expiry: Optional[str] = None
    history_what_to_show: Optional[str] = None
    point_value: Optional[float] = None
    contract_size: Optional[float] = None
    tick_size: Optional[float] = None
    commission_rate: Optional[float] = None
    commission_min: Optional[float] = None
    cost_model: Optional[str] = None
    # Smallest order quantity the venue accepts (e.g. 1 contract for a future,
    # 1 share for a stock). Used by the live executor to decide whether a
    # partial-close residual is too small to resize a protective leg to (ARB-96).
    min_order_size: Optional[float] = None
    # ARB / Sub-spec 1: per-symbol classification used by every engine consumer.
    # asset_class is the canonical taxonomy; None on load means "auto-classify
    # from security_type via symbols.asset_class.classify_asset_class".
    asset_class: Optional[str] = None
    # commission_scheme + fee fields unlock the `per_contract` path in
    # costs/base.py:_resolve_commission_scheme. None preserves legacy behavior.
    commission_scheme: Optional[str] = None
    fee_per_contract: float = 0.0
    fee_min_per_order: float = 0.0
