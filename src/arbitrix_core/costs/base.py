from __future__ import annotations

import logging
import threading
import math
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from arbitrix_core.types import InstrumentConfig

if TYPE_CHECKING:
    from arbitrix_core.data import DataProvider

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_COMMISSION",
    "configure_environment",
    "set_commission_per_lot",
    "get_commission_per_lot",
    "get_provider",
    "get_instruments",
    "get_point_overrides",
    "get_point_value",
    "get_instrument",
    "trade_notional",
    "tick_size",
    "commission_from_notional",
    "commission_minimum",
    "swap_points_from_cache",
    "swap_points_static",
    "warmup_from_provider",
]


_POINT_OVERRIDES: Dict[str, float] = {}
_POINT_VALUE_CACHE: Dict[str, float] = {}
_INSTRUMENTS: Dict[str, InstrumentConfig] = {}
_SWAP_POINTS_CACHE: Dict[tuple[str, str], float] = {}
_DATA_PROVIDER: DataProvider | None = None
_ALLOW_PROVIDER_LOOKUPS = True
_COMMISSION_PER_LOT = 3.0
MIN_COMMISSION = 1e-6
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Environment / config
# ---------------------------------------------------------------------------

def configure_environment(
    *,
    provider: DataProvider | None = None,
    commission_per_lot: Optional[float] = None,
    point_overrides: Optional[Dict[str, float]] = None,
    instruments: Optional[Dict[str, InstrumentConfig]] = None,
    allow_provider_lookups: bool = True,
    clear_provider: bool = False,
) -> None:
    """Configure shared cost infrastructure for the active model."""
    global _DATA_PROVIDER, _COMMISSION_PER_LOT, _POINT_VALUE_CACHE, _POINT_OVERRIDES, _INSTRUMENTS, _ALLOW_PROVIDER_LOOKUPS
    if clear_provider:
        _DATA_PROVIDER = None
        _POINT_VALUE_CACHE = {}
        _SWAP_POINTS_CACHE.clear()
    elif provider is not None:
        _DATA_PROVIDER = provider
        _POINT_VALUE_CACHE = {}
        _SWAP_POINTS_CACHE.clear()
    if commission_per_lot is not None:
        set_commission_per_lot(commission_per_lot)
    if point_overrides is not None:
        normalized: Dict[str, float] = {}
        for symbol, value in point_overrides.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            normalized[str(symbol)] = numeric
            normalized[str(symbol).lower()] = numeric
        _POINT_OVERRIDES = normalized
        _POINT_VALUE_CACHE = {}
    if instruments is not None:
        normalized_instruments: Dict[str, InstrumentConfig] = {}
        for symbol, inst in instruments.items():
            normalized_instruments[str(symbol)] = inst
            normalized_instruments[str(symbol).lower()] = inst
        _INSTRUMENTS = normalized_instruments
        _POINT_VALUE_CACHE = {}
    _ALLOW_PROVIDER_LOOKUPS = bool(allow_provider_lookups)
    try:
        logger.debug(
            "costs.configure_environment | provider=%s clear_provider=%s allow_provider_lookups=%s commission_per_lot=%s overrides=%s instruments=%s cache_keys=%s",
            getattr(provider, "provider_id", None) if provider else None,
            clear_provider,
            _ALLOW_PROVIDER_LOOKUPS,
            _COMMISSION_PER_LOT,
            list(_POINT_OVERRIDES.keys()),
            list(_INSTRUMENTS.keys()),
            list(_POINT_VALUE_CACHE.keys()),
        )
    except Exception:
        pass


def set_commission_per_lot(value: float) -> None:
    numeric = float(value)
    if numeric < 0:
        raise ValueError("commission_per_lot must be non-negative")
    global _COMMISSION_PER_LOT
    if numeric <= 0:
        logger.debug("commission_per_lot=0 supplied; using minimum %.6f", MIN_COMMISSION)
    _COMMISSION_PER_LOT = max(numeric, MIN_COMMISSION)


def get_commission_per_lot() -> float:
    return _COMMISSION_PER_LOT


def commission_minimum(volume_lot: float) -> float:
    # Keep a tiny positive min to avoid zeros downstream.
    return max(abs(volume_lot) * MIN_COMMISSION, MIN_COMMISSION)


def get_provider() -> DataProvider | None:
    return _DATA_PROVIDER


def get_instruments() -> Dict[str, InstrumentConfig]:
    return _INSTRUMENTS


def get_point_overrides() -> Dict[str, float]:
    return _POINT_OVERRIDES


def get_instrument(symbol: str) -> Optional[InstrumentConfig]:
    if not _INSTRUMENTS:
        return None
    return _INSTRUMENTS.get(symbol) or _INSTRUMENTS.get(str(symbol).lower())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _point_value_from_symbol_info(info: Dict[str, Any]) -> Optional[float]:
    """
    Derive point (tick) value (currency per point per 1 lot) from typical provider fields.
    """
    try:
        logger.debug("point_value_from_symbol_info: keys=%s", sorted(info.keys()))
    except Exception:
        pass

    def first_numeric(*keys: str) -> Optional[float]:
        for key in keys:
            if key in info:
                numeric = _as_float(info.get(key))
                if numeric is not None and numeric != 0:
                    return abs(numeric)
        return None

    # Prefer an explicit point_value field when providers supply it.
    for explicit_key in ("point_value", "pointValue"):
        explicit_point_value = first_numeric(explicit_key)
        if explicit_point_value is not None and explicit_point_value > 0:
            try:
                logger.debug(
                    "point_value_from_symbol_info: using explicit %s=%s",
                    explicit_key,
                    explicit_point_value,
                )
            except Exception:
                pass
            return explicit_point_value

    contract_size = first_numeric("trade_contract_size", "contract_size", "lot_size")
    if contract_size is not None and contract_size > 0:
        try:
            logger.debug("point_value_from_symbol_info: contract_size=%s", contract_size)
        except Exception:
            pass
        return contract_size

    point_size = first_numeric("point", "trade_tick_size", "tick_size")

    volume_min = first_numeric("volume_min", "min_volume")

    tick_keys = (
        "trade_tick_value_profit",
        "trade_tick_value",
        "tick_value",
        "tickValue",
    )
    for key in tick_keys:
        raw = _as_float(info.get(key)) if key in info else None
        if raw is None or raw <= 0:
            continue
        candidate = abs(raw)
        # Some brokers report tick value for min volume < 1; scale up to 1.0 lot.
        if volume_min is not None and 0 < volume_min < 1.0 and candidate < 1.0:
            scaled = candidate / volume_min
            if scaled > candidate:
                candidate = scaled
        
        # If we have tick value and tick size, we can calculate multiplier (point value)
        if point_size is not None and point_size > 0:
            try:
                logger.debug(
                    "point_value_from_symbol_info: tick_value=%s tick_size=%s -> %s",
                    candidate,
                    point_size,
                    candidate / point_size,
                )
            except Exception:
                pass
            return candidate / point_size

        if candidate > 0:
            # Fallback: if we only have tick value but no tick size, we return the tick value.
            # This assumes tick_size=1.0 or that the caller handles it (which they don't).
            # But for backward compatibility with tests/providers that rely on this:
            return candidate

    try:
        logger.debug("point_value_from_symbol_info: unable to derive point value")
    except Exception:
        pass
    return None


def _resolve_point_value(symbol: str) -> Optional[float]:
    logger.debug("Resolving point value for symbol: %s", symbol)
    symbol_key = str(symbol)
    override = _POINT_OVERRIDES.get(symbol_key) or _POINT_OVERRIDES.get(symbol_key.lower())
    if override is not None:
        logger.debug("Found point value override: %s", override)
        return float(override)
    if _ALLOW_PROVIDER_LOOKUPS and _DATA_PROVIDER is not None:
        info = _DATA_PROVIDER.get_symbol_info(symbol)
        if info:
            derived = _point_value_from_symbol_info(info)
            if derived is not None:
                logger.debug("Point value from provider info for %s: %s", symbol, derived)
                return derived
    inst = get_instrument(symbol)
    logger.debug("InstrumentConfig for %s: %s", symbol, inst)
    if inst:
        logger.debug("Checking InstrumentConfig point_value: %s", inst.point_value)
        if inst.point_value:
            return float(inst.point_value)
        logger.debug("No InstrumentConfig point_value; checking contract_size: %s", inst.contract_size)
        if inst.contract_size:
            return float(inst.contract_size)
    default = _POINT_OVERRIDES.get("default")
    if default is not None:
        logger.debug("Using default point value override for %s", symbol)
        return float(default)
    return None


def get_point_value(symbol: str) -> float:
    cache_key = str(symbol).lower()
    with _LOCK:
        cached = _POINT_VALUE_CACHE.get(cache_key)
        if cached is not None:
            return cached
    value = _resolve_point_value(symbol)
    if value is None:
        raise RuntimeError(
            f"Point value not available for {symbol}. Provide broker.point_value_overrides, instrument.point_value or provider symbol info."
        )
    try:
        import logging
        logging.getLogger(__name__).debug("Point value resolved | symbol=%s value=%s", symbol, value)
    except Exception:
        pass
    with _LOCK:
        _POINT_VALUE_CACHE[cache_key] = value
    return value


def trade_notional(symbol: str, price: float, volume_lot: float) -> float:
    pv = get_point_value(symbol)
    return abs(price) * pv * abs(volume_lot)


def tick_size(symbol: str) -> float:
    inst = get_instrument(symbol)
    if inst and inst.tick_size:
        return float(inst.tick_size)
    if _ALLOW_PROVIDER_LOOKUPS and _DATA_PROVIDER is not None:
        info = _DATA_PROVIDER.get_symbol_info(symbol) or {}
        for key in ("point", "trade_tick_size", "tick_size"):
            candidate = _as_float(info.get(key))
            if candidate and candidate > 0:
                return float(candidate)
    # If no explicit tick size, fall back to 1 "point".
    return 1.0


# ---------------------------------------------------------------------------
# Commission scheme resolution (provider-agnostic)
# ---------------------------------------------------------------------------

def _symbol_info(symbol: str) -> Dict[str, Any]:
    if not (_ALLOW_PROVIDER_LOOKUPS and _DATA_PROVIDER is not None):
        return {}
    return _DATA_PROVIDER.get_symbol_info(symbol) or {}


def _contracts_per_lot_from_info(info: Dict[str, Any]) -> Optional[float]:
    """
    Try to interpret how many *contracts* 1 lot represents.
    Many brokers expose a 'contract_size' or 'lot_size'. For index CFDs it is often 1.
    """
    # Prefer explicit "contracts_per_lot" if present
    cpl = _as_float(info.get("contracts_per_lot"))
    if cpl and cpl > 0:
        return cpl

    # If they expose "contract_size" as units per lot, treat that as contracts_per_lot when sensible.
    contract_size = _as_float(info.get("trade_contract_size") or info.get("contract_size") or info.get("lot_size"))
    if contract_size and contract_size > 0:
        # For indices: contract_size is often 1 (i.e., 1 index contract per lot)
        return contract_size

    # If nothing, return None; callers will default to 1.0
    return None


def _detect_spread_only(info: Dict[str, Any], inst: Optional[InstrumentConfig]) -> bool:
    # Explicit instrument flag
    if inst and getattr(inst, "spread_only", False):
        return True
    # Provider hints: commission=0, spread_only flags, or explicit type
    flags = {str(k).lower(): info[k] for k in info.keys()}
    # Common patterns across brokers/platforms
    zero_like = {0, 0.0, "0", "0.0", None}
    if info.get("commission") in zero_like:
        return True
    if str(info.get("commission_type", "")).lower() in {"spread_only", "included_in_spread"}:
        return True
    if str(info.get("pricing_model", "")).lower() in {"spread-only", "spread_only"}:
        return True
    return False


def _resolve_commission_scheme(symbol: str) -> Tuple[str, Dict[str, float]]:
    """
    Returns a pair (scheme, params) where:
      scheme in {"spread_only","bps","per_contract","per_lot_fixed"}
      params: dict with fields depending on scheme:
        - "bps": {"rate_bps": float, "min_commission": float}
        - "per_contract": {"fee_per_contract": float, "contracts_per_lot": float, "per_block": float|None, "fee_per_block": float|None}
        - "per_lot_fixed": {"fee_per_lot": float}
        - "spread_only": {}
    Priority: InstrumentConfig explicit fields > provider info hints > security_type defaults > fallback.
    """
    inst = get_instrument(symbol)
    info = _symbol_info(symbol)

    # 1) InstrumentConfig explicit overrides (add these optional fields to your InstrumentConfig if you like):
    #    commission_scheme: "spread_only" | "bps" | "per_contract" | "per_lot_fixed"
    scheme = getattr(inst, "commission_scheme", None) if inst else None
    if scheme:
        scheme = str(scheme).lower().strip()

    if scheme == "spread_only":
        return "spread_only", {}

    if scheme == "bps":
        rate_bps = float(getattr(inst, "commission_rate_bps", None) or (getattr(inst, "commission_rate", 0.0) * 10_000.0))
        min_c = float(getattr(inst, "commission_min", 0.0) or 0.0)
        return "bps", {"rate_bps": max(rate_bps, 0.0), "min_commission": max(min_c, 0.0)}

    if scheme == "per_contract":
        fee_per_contract = float(getattr(inst, "fee_per_contract", 0.0) or 0.0)
        per_block = _as_float(getattr(inst, "per_contract_block", None))
        fee_per_block = _as_float(getattr(inst, "fee_per_block", None))
        cpl = _as_float(getattr(inst, "contracts_per_lot", None)) or _contracts_per_lot_from_info(info) or 1.0
        return "per_contract", {
            "fee_per_contract": max(fee_per_contract, 0.0),
            "contracts_per_lot": max(cpl, 0.0) or 1.0,
            "per_block": per_block or 0.0,
            "fee_per_block": fee_per_block or 0.0,
        }

    if scheme == "per_lot_fixed":
        fee_per_lot = _as_float(getattr(inst, "fee_per_lot", None)) or get_commission_per_lot()
        return "per_lot_fixed", {"fee_per_lot": max(float(fee_per_lot), 0.0)}

    # 2) Provider info hints
    if _detect_spread_only(info, inst):
        return "spread_only", {}

    rate_bps = _as_float(info.get("commission_rate_bps"))
    if rate_bps is None:
        # Some providers give decimal rate (e.g., 0.00005); convert to bps if present
        rate_decimal = _as_float(info.get("commission_rate") or info.get("commissionRate"))
        if rate_decimal is not None and rate_decimal > 0:
            rate_bps = rate_decimal * 10_000.0
    min_commission = _as_float(info.get("min_commission") or info.get("minimum_commission"))

    if rate_bps and rate_bps > 0:
        return "bps", {"rate_bps": float(rate_bps), "min_commission": float(min_commission or 0.0)}

    # Look for per-contract hints
    fee_per_contract = _as_float(info.get("fee_per_contract") or info.get("commission_per_contract"))
    if fee_per_contract and fee_per_contract > 0:
        cpl = _contracts_per_lot_from_info(info) or 1.0
        per_block = _as_float(info.get("commission_block_size"))
        fee_per_block = _as_float(info.get("commission_fee_per_block"))
        return "per_contract", {
            "fee_per_contract": float(fee_per_contract),
            "contracts_per_lot": float(cpl),
            "per_block": float(per_block or 0.0),
            "fee_per_block": float(fee_per_block or 0.0),
        }

    # 3) security_type defaults (backwards compatible with your previous IBKR logic)
    sec_type = getattr(inst, "security_type", None) if inst else None
    if sec_type == "CASH":  # FX
        return "bps", {"rate_bps": 0.20, "min_commission": 2.0}  # 0.00002 → 0.20 bps per side
    if sec_type == "CFD":
        # Many index CFDs are spread-only; if not explicit, use a conservative tiny bps with $1 min.
        return "bps", {"rate_bps": 0.50, "min_commission": 1.0}

    # 4) Fallback to legacy per-lot fixed fee
    return "per_lot_fixed", {"fee_per_lot": get_commission_per_lot()}


# ---------------------------------------------------------------------------
# Commission computation
# ---------------------------------------------------------------------------

def commission_from_notional(
    *,
    symbol: str,
    price: float,
    volume_lot: float,
    notional: Optional[float] = None,
) -> float:
    """
    Compute ONE-SIDE commission for the given symbol/price/volume.
    Scheme is auto-resolved from InstrumentConfig and provider info.
    Always enforces commission_minimum(volume_lot) > 0 to avoid zeros.
    """
    inst = get_instrument(symbol)
    notional_value = notional if notional is not None else trade_notional(symbol, price, volume_lot)
    scheme, p = _resolve_commission_scheme(symbol)

    if scheme == "spread_only":
        # Commission = 0; spread/slippage will hit PnL separately.
        commission = 0.0

    elif scheme == "bps":
        rate_bps = float(p.get("rate_bps", 0.0))
        min_c = float(p.get("min_commission", 0.0))
        commission = max(notional_value * (rate_bps / 10_000.0), min_c)

    elif scheme == "per_contract":
        fee_per_contract = float(p.get("fee_per_contract", 0.0))
        cpl = float(p.get("contracts_per_lot", 1.0)) or 1.0
        per_block = float(p.get("per_block", 0.0))
        fee_per_block = float(p.get("fee_per_block", 0.0))

        contracts = abs(volume_lot) * cpl
        if per_block and fee_per_block:
            # Some brokers charge e.g. $X per 100 contracts.
            blocks = contracts / per_block
            commission = blocks * fee_per_block
        else:
            commission = contracts * fee_per_contract

    elif scheme == "per_lot_fixed":
        fee_per_lot = float(p.get("fee_per_lot", get_commission_per_lot()))
        commission = fee_per_lot * abs(volume_lot)

    else:
        # Shouldn't happen; default to tiny min.
        commission = 0.0

    # Enforce tiny positive minimum to keep downstream math safe
    commission = max(commission, commission_minimum(volume_lot))
    return commission


# ---------------------------------------------------------------------------
# Swaps (unchanged)
# ---------------------------------------------------------------------------

def swap_points_static(symbol: str, direction: str, static_override: Optional[dict] = None) -> Optional[float]:
    if not static_override:
        return None
    key = "long" if direction == "long" else "short"
    return float(static_override.get(key, 0.0)) * tick_size(symbol)


def swap_points_from_cache(symbol: str, direction: str) -> float:
    cache_key = str(symbol).lower()
    cached = _SWAP_POINTS_CACHE.get((cache_key, "long" if direction == "long" else "short"))
    if cached is not None:
        return cached
    if not _ALLOW_PROVIDER_LOOKUPS or _DATA_PROVIDER is None:
        return 0.0
    info = _DATA_PROVIDER.get_symbol_info(symbol) or {}
    field = "swap_long" if direction == "long" else "swap_short"
    return float(info.get(field, 0.0)) * tick_size(symbol)


def warmup_from_provider(symbols: list[str]) -> None:
    """Prime caches for use by worker threads."""
    for sym in symbols:
        cache_key = str(sym).lower()
        if cache_key not in _POINT_VALUE_CACHE:
            try:
                # Keep warmup precedence aligned with runtime resolution:
                # point overrides > provider symbol-info > instrument config.
                pv = _resolve_point_value(sym)
            except Exception:
                pv = None
            if pv is not None:
                _POINT_VALUE_CACHE[cache_key] = float(pv)

        if _DATA_PROVIDER is None:
            continue
        info = _DATA_PROVIDER.get_symbol_info(sym) or {}
        long_p = float(info.get("swap_long", 0.0))
        short_p = float(info.get("swap_short", 0.0))
        _SWAP_POINTS_CACHE[(cache_key, "long")] = long_p * tick_size(sym)
        _SWAP_POINTS_CACHE[(cache_key, "short")] = short_p * tick_size(sym)


def inject_point_value_cache(cache: Dict[str, float]) -> None:
    """
    Inject pre-computed point values into the cache.
    Used by optimization workers to avoid provider lookups entirely.
    
    Args:
        cache: Dictionary mapping lowercase symbol names to point values.
    """
    global _POINT_VALUE_CACHE
    for key, value in cache.items():
        _POINT_VALUE_CACHE[key.lower()] = float(value)


def inject_swap_cache(cache: Dict[tuple[str, str], float]) -> None:
    """
    Inject pre-computed swap points into the cache.
    Used by optimization workers to avoid provider lookups entirely.
    
    Args:
        cache: Dictionary mapping (symbol, direction) to swap points.
    """
    global _SWAP_POINTS_CACHE
    for key, value in cache.items():
        sym, direction = key
        _SWAP_POINTS_CACHE[(sym.lower(), direction)] = float(value)


def export_caches() -> Dict[str, Any]:
    """
    Export the current point value and swap caches for transfer to workers.
    
    Returns:
        Dictionary containing 'point_values' and 'swap_points' caches.
    """
    return {
        "point_values": dict(_POINT_VALUE_CACHE),
        "swap_points": dict(_SWAP_POINTS_CACHE),
    }


def import_caches(data: Dict[str, Any]) -> None:
    """
    Import point value and swap caches from exported data.
    Used by optimization workers to have full cache without provider access.
    
    Args:
        data: Dictionary from export_caches() containing cached values.
    """
    pv_cache = data.get("point_values") or {}
    swap_cache = data.get("swap_points") or {}
    inject_point_value_cache(pv_cache)
    for key, value in swap_cache.items():
        if isinstance(key, (list, tuple)) and len(key) == 2:
            _SWAP_POINTS_CACHE[(str(key[0]).lower(), key[1])] = float(value)

