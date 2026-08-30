from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Optional, TYPE_CHECKING
from weakref import WeakKeyDictionary

import pandas as pd

from arbitrix_core.data.market import MarketBars, MarketFrames, PreparedMarket
from arbitrix_core.time_utils import (
    is_in_session as _is_in_session,
    session_day as _session_day,
    session_hour as _session_hour,
    to_market_time as _to_market_time,
    to_utc_time as _to_utc_time,
)
from arbitrix_core.trading import Signal

if TYPE_CHECKING:
    from arbitrix_core.portfolio import Portfolio
    from arbitrix_core.symbols.context import SymbolContext


@dataclass(frozen=True)
class _InvocationSpec:
    keyword_names: frozenset[str]
    positional_only_names: frozenset[str] = frozenset()
    accepts_var_positional: bool = False

    def accepts(self, name: str) -> bool:
        # Optional engine inputs are an explicit-name contract. A bare
        # ``**kwargs`` did not receive ctx/regime inputs in the legacy engine,
        # and treating it as opt-in would silently change mono strategies.
        return name in self.keyword_names

    def accepts_positional_only(self, name: str) -> bool:
        return name in self.positional_only_names


_SIGNATURE_CACHE: "WeakKeyDictionary[type, Dict[str, _InvocationSpec]]" = WeakKeyDictionary()
_SIGNATURE_CACHE_LOCK = RLock()
# Kept as an internal compatibility alias for existing cache-reset tooling.
_ON_BAR_SIGNATURE_CACHE = _SIGNATURE_CACHE
_UNSET = object()


def _method_invocation_spec(strategy: "BaseStrategy", method_name: str) -> _InvocationSpec:
    cls = strategy.__class__
    with _SIGNATURE_CACHE_LOCK:
        class_specs = _SIGNATURE_CACHE.get(cls)
        if class_specs is not None and method_name in class_specs:
            return class_specs[method_name]

    try:
        signature = inspect.signature(getattr(strategy, method_name))
    except (AttributeError, TypeError, ValueError):
        spec = _InvocationSpec(frozenset())
    else:
        keyword_names = frozenset(
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
        spec = _InvocationSpec(
            keyword_names=keyword_names,
            positional_only_names=frozenset(
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
            ),
            accepts_var_positional=any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            ),
        )

    with _SIGNATURE_CACHE_LOCK:
        class_specs = _SIGNATURE_CACHE.setdefault(cls, {})
        return class_specs.setdefault(method_name, spec)


def strategy_supports_parameter(
    strategy: "BaseStrategy",
    method_name: str,
    parameter_name: str,
) -> bool:
    """Return whether a strategy hook accepts an optional named input."""

    return _method_invocation_spec(strategy, method_name).accepts(parameter_name)


def _invoke_with_optional_keywords(
    strategy: "BaseStrategy",
    method_name: str,
    positional_args: tuple[Any, ...],
    optional_values: Dict[str, Any],
):
    spec = _method_invocation_spec(strategy, method_name)
    kwargs = {
        name: value
        for name, value in optional_values.items()
        if value is not _UNSET and spec.accepts(name)
    }
    return getattr(strategy, method_name)(*positional_args, **kwargs)


def strategy_supports_regime_output(strategy: "BaseStrategy") -> bool:
    spec = _method_invocation_spec(strategy, "on_bar")
    return (
        spec.accepts("regime_output")
        or spec.accepts_positional_only("regime_output")
        or spec.accepts_var_positional
    )


def invoke_strategy_prepare(
    strategy: "BaseStrategy",
    df: pd.DataFrame,
    *,
    data: Any = _UNSET,
):
    return _invoke_with_optional_keywords(
        strategy,
        "prepare",
        (df,),
        {"data": data},
    )


def invoke_strategy_on_bar(
    strategy: "BaseStrategy",
    row: pd.Series,
    portfolio: "Portfolio",
    regime_output: Any = None,
    *,
    ctx: Any = _UNSET,
    data: Any = _UNSET,
):
    spec = _method_invocation_spec(strategy, "on_bar")
    positional_args: tuple[Any, ...] = (row, portfolio)
    named_regime: Any = regime_output
    if not spec.accepts("regime_output") and (
        spec.accepts_positional_only("regime_output")
        or spec.accepts_var_positional
    ):
        # Preserve intentional legacy signatures that consumed regime output
        # positionally, without reviving arbitrary third-parameter guessing.
        positional_args += (regime_output,)
        named_regime = _UNSET
    return _invoke_with_optional_keywords(
        strategy,
        "on_bar",
        positional_args,
        {
            "regime_output": named_regime,
            "ctx": ctx,
            "data": data,
        },
    )


def invoke_strategy_stop_distance_points(
    strategy: "BaseStrategy",
    row: pd.Series,
    *,
    symbol: Any = _UNSET,
    ctx: Any = _UNSET,
    data: Any = _UNSET,
) -> float:
    return _invoke_with_optional_keywords(
        strategy,
        "stop_distance_points",
        (row,),
        {"symbol": symbol, "ctx": ctx, "data": data},
    )


def invoke_strategy_take_distance_points(
    strategy: "BaseStrategy",
    row: pd.Series,
    *,
    symbol: Any = _UNSET,
    ctx: Any = _UNSET,
    data: Any = _UNSET,
) -> float:
    return _invoke_with_optional_keywords(
        strategy,
        "take_distance_points",
        (row,),
        {"symbol": symbol, "ctx": ctx, "data": data},
    )


def invoke_strategy_should_exit_trade(
    strategy: "BaseStrategy",
    trade: Any,
    row: pd.Series,
    *,
    symbol: Any = _UNSET,
    ctx: Any = _UNSET,
    data: Any = _UNSET,
) -> bool:
    """Invoke the optional legacy conditional-exit hook safely."""

    if not callable(getattr(strategy, "should_exit_trade", None)):
        return False
    return bool(
        _invoke_with_optional_keywords(
            strategy,
            "should_exit_trade",
            (trade, row),
            {"symbol": symbol, "ctx": ctx, "data": data},
        )
    )


class BaseStrategy:
    name: str
    symbol: str = ""
    timeframe: str = "M5"
    requires_portfolio: bool = False
    portfolio: Optional["Portfolio"] = None
    # Set to True by live runtime to indicate live/dispatcher mode.
    _live_mode: bool = False

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        data: "MarketFrames | None" = None,
    ) -> "pd.DataFrame | PreparedMarket":  # pragma: no cover - to override
        return df

    def stop_distance_points(
        self,
        row: pd.Series,
        *,
        symbol: str | None = None,
        ctx: "SymbolContext | None" = None,
        data: "MarketBars | None" = None,
    ) -> float:  # pragma: no cover - to override
        return 0.0

    def take_distance_points(
        self,
        row: pd.Series,
        *,
        symbol: str | None = None,
        ctx: "SymbolContext | None" = None,
        data: "MarketBars | None" = None,
    ) -> float:
        return 0.0

    def warmup_bars(self) -> int:  # pragma: no cover - override when strategy needs extra history
        """Return the number of additional bars required before evaluation."""

        return 0

    def on_bar(
        self,
        row: pd.Series,
        portfolio: "Portfolio",
        regime_output: Any = None,
        ctx: "SymbolContext | None" = None,
        *,
        data: "MarketBars | None" = None,
    ) -> list[Signal]:
        """Called once per prepared bar in backtest and live modes.

        ARB / Sub-spec 1: ``ctx`` is the :class:`SymbolContext` for
        ``self.symbol``. Engines pass it when the override accepts it; legacy
        strategies (no ``ctx`` parameter) keep working via ``inspect``-based
        dispatch in ``_invoke_strategy_on_bar``.
        """
        return []

    @staticmethod
    def to_market_time(ts: Any, tz: str) -> pd.Timestamp:
        return _to_market_time(ts, tz)

    @staticmethod
    def to_utc_time(ts: Any, tz: Optional[str] = None) -> pd.Timestamp:
        return _to_utc_time(ts, tz=tz)

    @staticmethod
    def session_day(ts: Any, tz: str):
        return _session_day(ts, tz)

    @staticmethod
    def session_hour(ts: Any, tz: str) -> float:
        return _session_hour(ts, tz)

    @staticmethod
    def is_in_session(
        ts: Any,
        tz: str,
        windows: Any,
    ) -> bool:
        return _is_in_session(ts, tz, windows=windows)

    def clone(self) -> "BaseStrategy":
        """Create a fresh strategy instance for parallel backtest modes."""
        cfg = getattr(self, "cfg", None)
        cls = self.__class__
        try:
            if cfg is not None:
                cloned = cls(copy.deepcopy(cfg))
            else:
                cloned = cls()
        except Exception:
            cloned = copy.deepcopy(self)
        cloned.portfolio = None
        if getattr(cloned, "symbol", None) in (None, ""):
            cloned.symbol = getattr(self, "symbol", "")
        if getattr(cloned, "timeframe", None) in (None, ""):
            cloned.timeframe = getattr(self, "timeframe", "M5")
        return cloned

__all__ = [
    "BaseStrategy",
    "invoke_strategy_prepare",
    "invoke_strategy_on_bar",
    "invoke_strategy_stop_distance_points",
    "invoke_strategy_take_distance_points",
    "strategy_supports_parameter",
    "strategy_supports_regime_output",
]
