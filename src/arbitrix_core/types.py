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
    timezone: Optional[str] = None
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
