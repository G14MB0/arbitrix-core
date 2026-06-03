"""MarginModel Protocol + supporting value types.

A MarginModel computes the cash margin a position requires given the symbol,
quantity, and a reference price. Returns Money so callers can carry currency
through to portfolio-level enforcement; FX conversion is the microstructure
layer's responsibility, not ours.

Two operations:

* ``initial(symbol, qty, price)`` — capital required to open the position.
  Checked by ``Portfolio.can_open`` before order submission.
* ``maintenance(symbol, qty, price)`` — capital required to hold it.
  Checked per-bar by ``Portfolio.check_maintenance_margin`` after sync.

Concrete implementations live in :mod:`arbitrix_core.margin.models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

import pandas as pd


class Money(NamedTuple):
    """Amount + currency. Immutable, tuple-iterable."""

    amount: float
    currency: str


@runtime_checkable
class MarginModel(Protocol):
    """Per-symbol margin requirement contract."""

    def initial(self, symbol: str, qty: float, price: float) -> Money:
        """Capital required to open a position of ``qty`` units at ``price``."""

    def maintenance(self, symbol: str, qty: float, price: float) -> Money:
        """Capital required to keep a position of ``qty`` units at ``price``."""


@dataclass(frozen=True)
class MarginCallEvent:
    """Emitted by ``Portfolio.check_maintenance_margin`` when equity drops
    below the per-symbol maintenance requirement. Sub-spec 4 consumes these
    to materialize a forced-flat order at the next close.
    """

    symbol: str
    equity: float
    maintenance_required: float
    ts: pd.Timestamp
