from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Optional


QuantityUnit = Literal["lot", "base_currency_unit", "share", "contract", "unit"]


class MissingConversionRate(ValueError):
    """Raised when profit currency cannot be converted to account currency."""


class MissingAccountCurrency(ValueError):
    """Raised when an online valuation has no provider account currency."""


def _currency(value: str, *, field: str) -> str:
    normalized = str(value or "").strip().upper()
    if len(normalized) != 3 or not normalized.isalpha():
        raise ValueError(f"{field} must be a three-letter currency code")
    return normalized


def _positive(value: float, *, field: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{field} must be finite and greater than zero")
    return numeric


@dataclass(frozen=True)
class ContractSpec:
    """Stable inputs for linear PnL valuation.

    ``quantity_multiplier`` converts one execution quantity into the units used
    by the instrument PnL formula. It is 100,000 for a standard MT5 FX lot, 1
    for an IBKR CASH currency unit/share, and the contract multiplier for a
    future.
    """

    base_currency: str
    profit_currency: str
    quantity_multiplier: float
    quantity_unit: QuantityUnit

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_currency",
            _currency(self.base_currency, field="base_currency"),
        )
        object.__setattr__(
            self,
            "profit_currency",
            _currency(self.profit_currency, field="profit_currency"),
        )
        object.__setattr__(
            self,
            "quantity_multiplier",
            _positive(self.quantity_multiplier, field="quantity_multiplier"),
        )


def account_point_value(
    contract: ContractSpec,
    *,
    account_currency: str,
    instrument_price: Optional[float] = None,
    profit_to_account_rate: Optional[float] = None,
) -> float:
    """Return account-currency PnL for a 1.0 price move and quantity 1.0."""

    account = _currency(account_currency, field="account_currency")
    if contract.profit_currency == account:
        conversion_rate = 1.0
    elif profit_to_account_rate is not None:
        conversion_rate = _positive(
            profit_to_account_rate,
            field="profit_to_account_rate",
        )
    elif contract.base_currency == account:
        price = _positive(instrument_price, field="instrument_price")
        conversion_rate = 1.0 / price
    else:
        raise MissingConversionRate(
            f"Missing conversion rate for {contract.profit_currency}->{account}"
        )
    return contract.quantity_multiplier * conversion_rate


def _read(source: Any, *keys: str) -> Any:
    if source is None:
        return None
    for key in keys:
        if isinstance(source, Mapping) and key in source:
            value = source.get(key)
        else:
            value = getattr(source, key, None)
        if value is not None and str(value).strip() != "":
            return value
    return None


def configured_point_value(
    symbol: str,
    *,
    instrument: Any = None,
    point_overrides: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    """Return an explicit user value, never a provider-derived fallback."""

    overrides = point_overrides or {}
    marker = object()
    raw: Any = marker
    for key in (str(symbol), str(symbol).lower()):
        if key in overrides:
            raw = overrides[key]
            break
    if raw is marker:
        raw = getattr(instrument, "point_value", None)
    if raw is None:
        return None
    return _positive(raw, field=f"point_value override for {symbol}")


def current_point_value_from_metadata(
    symbol: str,
    *,
    provider_type: str,
    symbol_info: Optional[Mapping[str, Any]] = None,
    instrument: Any = None,
    account_currency: Optional[str] = None,
    instrument_price: Optional[float] = None,
    point_overrides: Optional[Mapping[str, Any]] = None,
) -> float:
    """Resolve current point value with user metadata taking precedence."""

    override = configured_point_value(
        symbol,
        instrument=instrument,
        point_overrides=point_overrides,
    )
    if override is not None:
        return override

    provider = str(provider_type or "").strip().lower()
    if provider not in {"mt5", "local-mt5", "local_mt5", "localmt5", "portable-mt5"}:
        raise ValueError(f"Dynamic point value is unsupported for provider {provider_type!r}")

    info = symbol_info or {}
    tick_size = _read(info, "trade_tick_size", "tick_size", "point")
    tick_value = _read(
        info,
        "trade_tick_value_profit",
        "trade_tick_value",
        "tick_value",
    )
    if tick_size is not None and tick_value is not None:
        return _positive(tick_value, field="provider tick value") / _positive(
            tick_size,
            field="provider tick size",
        )

    account = str(account_currency or "").strip().upper()
    if len(account) != 3:
        raise MissingAccountCurrency(
            f"Provider account currency unavailable while valuing {symbol}"
        )
    contract = contract_spec_from_metadata(
        symbol,
        provider_type=provider,
        symbol_info=info,
        instrument=instrument,
    )
    return account_point_value(
        contract,
        account_currency=account,
        instrument_price=instrument_price,
    )


def _symbol_pair(symbol: str) -> tuple[Optional[str], Optional[str]]:
    normalized = "".join(ch for ch in str(symbol or "").upper() if ch.isalpha())
    if len(normalized) < 6:
        return None, None
    return normalized[:3], normalized[3:6]


def contract_spec_from_metadata(
    symbol: str,
    *,
    provider_type: str,
    symbol_info: Optional[Mapping[str, Any]] = None,
    instrument: Any = None,
) -> ContractSpec:
    """Normalize provider metadata into stable linear contract semantics."""

    provider = str(provider_type or "").strip().lower()
    info = symbol_info or {}
    security_type = str(
        _read(info, "security_type", "secType")
        or _read(instrument, "security_type", "trading_security_type")
        or ""
    ).upper()
    pair_base, pair_quote = _symbol_pair(symbol)

    if provider in {"mt5", "local-mt5", "local_mt5", "localmt5", "portable-mt5"}:
        base_currency = _read(info, "currency_base", "base_currency") or _read(
            instrument, "base_currency"
        ) or pair_base
        profit_currency = _read(info, "currency_profit", "profit_currency") or _read(
            instrument, "profit_currency", "currency"
        ) or pair_quote
        multiplier = _read(instrument, "contract_size", "multiplier") or _read(
            info,
            "trade_contract_size",
            "contract_size",
            "lot_size",
        )
        return ContractSpec(
            base_currency=str(base_currency or ""),
            profit_currency=str(profit_currency or ""),
            quantity_multiplier=float(multiplier),
            quantity_unit="lot",
        )

    if provider in {"ib", "ibkr", "interactive_brokers", "interactive-brokers"}:
        contract_currency = _read(info, "currency") or _read(instrument, "currency")
        if security_type == "CASH":
            base_currency = _read(info, "symbol") or _read(instrument, "ib_symbol") or pair_base
            profit_currency = contract_currency or pair_quote
            return ContractSpec(
                base_currency=str(base_currency or ""),
                profit_currency=str(profit_currency or ""),
                quantity_multiplier=1.0,
                quantity_unit="base_currency_unit",
            )

        multiplier = _read(info, "multiplier", "contract_size") or _read(
            instrument,
            "trading_multiplier",
            "multiplier",
            "contract_size",
        ) or 1.0
        currency = str(contract_currency or "")
        quantity_unit: QuantityUnit = "contract" if security_type in {"FUT", "FOP", "OPT"} else "share"
        return ContractSpec(
            base_currency=currency,
            profit_currency=currency,
            quantity_multiplier=float(multiplier),
            quantity_unit=quantity_unit,
        )

    raise ValueError(f"Unsupported provider type for valuation: {provider_type!r}")


__all__ = [
    "ContractSpec",
    "MissingAccountCurrency",
    "MissingConversionRate",
    "QuantityUnit",
    "account_point_value",
    "configured_point_value",
    "contract_spec_from_metadata",
    "current_point_value_from_metadata",
]
