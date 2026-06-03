"""Concrete MarginModel implementations.

* :class:`NoMargin` — for cash FX / crypto / anything not margined.
* :class:`FuturesUSDMargin` — flat per-contract requirement (CME-style).
* :class:`RegTMargin` — initial 50%, maintenance 25% of notional (US equities).
* :class:`CFDMargin` — notional / leverage; leverage defaults to 20.

All return :class:`Money` with currency ``"USD"`` for now — Sub-spec 2 ships
USD-only; multi-currency settlement is a Sub-spec 3 concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from arbitrix_core.margin.protocol import Money


@dataclass(frozen=True)
class NoMargin:
    """Zero margin for every position. Used for cash FX, crypto, etc."""

    def initial(self, symbol: str, qty: float, price: float) -> Money:
        return Money(amount=0.0, currency="USD")

    def maintenance(self, symbol: str, qty: float, price: float) -> Money:
        return Money(amount=0.0, currency="USD")


@dataclass(frozen=True)
class FuturesUSDMargin:
    """Flat per-contract margin (CME-style).

    Params:
        initial_per_contract: required to open one contract (intraday default).
        maintenance_per_contract: required to keep one contract.
        overnight_initial_per_contract: optional; used when ``overnight=True``.
        overnight_maintenance_per_contract: optional; used when ``overnight=True``.

    Quantity sign is ignored — both long and short use ``abs(qty)``.
    """

    initial_per_contract: float
    maintenance_per_contract: float
    overnight_initial_per_contract: Optional[float] = None
    overnight_maintenance_per_contract: Optional[float] = None

    def initial(
        self, symbol: str, qty: float, price: float, *, overnight: bool = False
    ) -> Money:
        per = (
            self.overnight_initial_per_contract
            if overnight and self.overnight_initial_per_contract is not None
            else self.initial_per_contract
        )
        return Money(amount=abs(qty) * per, currency="USD")

    def maintenance(
        self, symbol: str, qty: float, price: float, *, overnight: bool = False
    ) -> Money:
        per = (
            self.overnight_maintenance_per_contract
            if overnight and self.overnight_maintenance_per_contract is not None
            else self.maintenance_per_contract
        )
        return Money(amount=abs(qty) * per, currency="USD")


@dataclass(frozen=True)
class RegTMargin:
    """Reg-T-style equity margin: % of notional.

    Defaults: 50% initial, 25% maintenance (US Reg-T overnight). Operators
    can override via ``initial_ratio`` / ``maintenance_ratio`` (e.g., a
    portfolio-margin account would use lower ratios).
    """

    initial_ratio: float = 0.50
    maintenance_ratio: float = 0.25

    def initial(self, symbol: str, qty: float, price: float) -> Money:
        notional = abs(qty) * price
        return Money(amount=notional * self.initial_ratio, currency="USD")

    def maintenance(self, symbol: str, qty: float, price: float) -> Money:
        notional = abs(qty) * price
        return Money(amount=notional * self.maintenance_ratio, currency="USD")


@dataclass(frozen=True)
class CFDMargin:
    """CFD margin: notional / leverage. Defaults to 20x.

    Initial == maintenance (CFDs typically don't have a separate maintenance
    tier; brokers margin-call when equity hits the same level).

    Note on microstructure leverage: ``_default_leverage_for`` in
    ``arbitrix/services/lib/microstructure_engine.py`` already defaults
    CFD leverage to 20.0 for cost/microstructure simulation — a separate
    concern from portfolio enforcement. Both default to 20.0 today so
    behavior is unchanged. See ``docs/architecture/cost-vs-margin.md``.
    """

    leverage: float = 20.0

    def __post_init__(self) -> None:
        if self.leverage <= 0:
            raise ValueError(f"CFDMargin leverage must be > 0, got {self.leverage}")

    def initial(self, symbol: str, qty: float, price: float) -> Money:
        notional = abs(qty) * price
        return Money(amount=notional / self.leverage, currency="USD")

    def maintenance(self, symbol: str, qty: float, price: float) -> Money:
        return self.initial(symbol, qty, price)
