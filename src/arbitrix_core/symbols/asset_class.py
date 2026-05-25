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
