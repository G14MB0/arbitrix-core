"""Auto-classify `InstrumentConfig.security_type` to a canonical `asset_class`."""

from __future__ import annotations

from typing import Literal

AssetClass = Literal["stock", "cfd", "fx", "futures", "futures_continuous"]

_SECURITY_TYPE_TO_ASSET_CLASS: dict[str, AssetClass] = {
    "STK": "stock",
    "CFD": "cfd",
    "CASH": "fx",
    "FUT": "futures",
    "CONTFUT": "futures_continuous",
}


def classify_asset_class(security_type: str) -> AssetClass:
    if not security_type:
        raise ValueError("security_type is required to classify asset_class")
    key = str(security_type).upper().strip()
    if key not in _SECURITY_TYPE_TO_ASSET_CLASS:
        raise ValueError(
            f"Unknown security_type {security_type!r}. "
            f"Expected one of {sorted(_SECURITY_TYPE_TO_ASSET_CLASS)}."
        )
    return _SECURITY_TYPE_TO_ASSET_CLASS[key]


def validate_asset_class(value: str) -> AssetClass:
    """Validate a free-string ``asset_class`` against the known taxonomy.

    Returns the value typed as :data:`AssetClass`. Raises :class:`ValueError`
    on unknown — protects :class:`SymbolContext` from typos at builder time.
    """
    valid = set(_SECURITY_TYPE_TO_ASSET_CLASS.values())
    if value not in valid:
        raise ValueError(
            f"Unknown asset_class {value!r}. Expected one of {sorted(valid)}."
        )
    return value  # type: ignore[return-value]
