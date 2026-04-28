from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Optional, TYPE_CHECKING

import pandas as pd

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


_ON_BAR_SIGNATURE_CACHE: Dict[type, bool] = {}


def strategy_supports_regime_output(strategy: "BaseStrategy") -> bool:
    cls = strategy.__class__
    cached = _ON_BAR_SIGNATURE_CACHE.get(cls)
    if cached is not None:
        return cached
    try:
        signature = inspect.signature(strategy.on_bar)
    except (TypeError, ValueError):
        _ON_BAR_SIGNATURE_CACHE[cls] = False
        return False

    supports = False
    positional = 0
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            supports = True
            break
        if param.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            positional += 1
        if param.kind == inspect.Parameter.KEYWORD_ONLY and param.name == "regime_output":
            supports = True
            break
    if not supports:
        supports = positional >= 3
    _ON_BAR_SIGNATURE_CACHE[cls] = supports
    return supports


def invoke_strategy_on_bar(
    strategy: "BaseStrategy",
    row: pd.Series,
    portfolio: "Portfolio",
    regime_output: Any = None,
):
    if strategy_supports_regime_output(strategy):
        return strategy.on_bar(row, portfolio, regime_output)
    return strategy.on_bar(row, portfolio)


class BaseStrategy:
    name: str
    symbol: str = ""
    timeframe: str = "M5"
    requires_portfolio: bool = False
    portfolio: Optional["Portfolio"] = None
    # Set to True by live runtime to indicate live/dispatcher mode.
    _live_mode: bool = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - to override
        return df

    def stop_distance_points(self, row: pd.Series) -> float:  # pragma: no cover - to override
        return 0.0

    def take_distance_points(self, row: pd.Series) -> float:
        return 0.0

    def warmup_bars(self) -> int:  # pragma: no cover - override when strategy needs extra history
        """Return the number of additional bars required before evaluation."""

        return 0

    def on_bar(
        self,
        row: pd.Series,
        portfolio: "Portfolio",
        regime_output: Any = None,
    ) -> list[Signal]:
        """Called once per prepared bar in backtest and live modes."""
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
    "invoke_strategy_on_bar",
    "strategy_supports_regime_output",
]
