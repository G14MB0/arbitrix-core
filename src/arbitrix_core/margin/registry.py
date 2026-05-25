"""Per-symbol margin params registry + asset-class → model-id resolver.

Mirrors the pattern in :mod:`arbitrix_core.symbols.context`:

* Module-level ``_PARAMS`` dict, lower-cased symbol keys.
* ``threading.RLock`` guards reads + writes for live-runtime safety.
* ``register_margin_params`` and ``get_margin_params`` both lowercase
  the symbol so callers don't have to remember the convention.
* Last-write-wins (no merge), matching the SymbolContext registry.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional


# Asset-class → margin-model-id default mapping. Single source of truth used
# by the alembic backfill (mirrored inline there because alembic can't always
# import app code) and by runtime resolvers.
_ASSET_CLASS_DEFAULTS: Dict[str, str] = {
    "futures": "futures_usd",
    "futures_continuous": "futures_usd",
    "stock": "regt",
    "cfd": "cfd_20x",
    "fx": "nomargin",
    "crypto": "nomargin",
}


def default_margin_model_for(asset_class: Optional[str]) -> str:
    """Return the canonical model id for an asset class.

    Falls back to ``"nomargin"`` for ``None``, empty string, or any
    asset class not in the default table (operators can still override
    by setting ``margin_model_id`` explicitly on the symbol row).
    """
    if not asset_class:
        return "nomargin"
    return _ASSET_CLASS_DEFAULTS.get(asset_class, "nomargin")


@dataclass(frozen=True)
class MarginParams:
    """Per-symbol payload describing which MarginModel to instantiate and how.

    The registry stores these; engine code resolves them to a concrete
    :class:`MarginModel` via :func:`resolve_margin_model`.
    """

    model_id: str
    initial_per_contract: Optional[float] = None
    maintenance_per_contract: Optional[float] = None
    overnight_initial_per_contract: Optional[float] = None
    overnight_maintenance_per_contract: Optional[float] = None
    leverage: Optional[float] = None


_PARAMS: Dict[str, MarginParams] = {}
_LOCK = threading.RLock()


def register_margin_params(symbol: str, params: MarginParams) -> None:
    """Register or overwrite the margin params for ``symbol``.

    Symbol key is lower-cased internally; callers can pass any casing.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    with _LOCK:
        _PARAMS[symbol.strip().lower()] = params


def get_margin_params(symbol: str) -> MarginParams:
    """Return the registered params for ``symbol`` or raise ``KeyError``.

    Lookup is case-insensitive.
    """
    with _LOCK:
        return _PARAMS[str(symbol).lower()]


def clear_margin_params_registry() -> None:
    """Wipe the registry (test fixture helper, not a runtime API)."""
    with _LOCK:
        _PARAMS.clear()


from arbitrix_core.margin.models import (
    NoMargin, FuturesUSDMargin, RegTMargin, CFDMargin,
)
from arbitrix_core.margin.protocol import MarginModel


def resolve_margin_model(params: MarginParams) -> MarginModel:
    """Build a concrete MarginModel from a MarginParams payload.

    Dispatches on ``params.model_id``. Defaults are applied per-model:

    * ``nomargin`` → :class:`NoMargin` (no params used).
    * ``futures_usd`` → :class:`FuturesUSDMargin`. ``initial_per_contract`` and
      ``maintenance_per_contract`` are REQUIRED; raises ``ValueError`` if missing.
    * ``regt`` → :class:`RegTMargin` (50%/25% defaults; params currently ignored
      — operator overrides via custom ratios are a Sub-spec 3 concern).
    * ``cfd_20x`` → :class:`CFDMargin`. Honors ``params.leverage`` if set,
      else defaults to 20x.

    Raises ``ValueError`` on unknown ``model_id``.
    """
    mid = params.model_id
    if mid == "nomargin":
        return NoMargin()
    if mid == "futures_usd":
        if params.initial_per_contract is None or params.maintenance_per_contract is None:
            raise ValueError(
                f"futures_usd MarginParams must include initial_per_contract "
                f"and maintenance_per_contract; got {params}"
            )
        return FuturesUSDMargin(
            initial_per_contract=params.initial_per_contract,
            maintenance_per_contract=params.maintenance_per_contract,
            overnight_initial_per_contract=params.overnight_initial_per_contract,
            overnight_maintenance_per_contract=params.overnight_maintenance_per_contract,
        )
    if mid == "regt":
        return RegTMargin()
    if mid == "cfd_20x":
        leverage = params.leverage if params.leverage is not None else 20.0
        return CFDMargin(leverage=leverage)
    raise ValueError(f"unknown margin model id: {mid!r}")
