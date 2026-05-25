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
