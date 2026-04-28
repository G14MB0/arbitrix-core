"""arbitrix-core — open-source backtest engine + cost model.

MIT-licensed subset of the Arbitrix trading toolkit. Provides:
  - Backtester / BTConfig / BTResult — the bar-by-bar batch engine
  - BaseStrategy — the strategy interface with optional portfolio-aware hooks
  - Signal / Trade / Order / Position — trading domain types
  - InstrumentConfig — instrument metadata
  - load_ohlcv / validate_ohlcv — DataFrame loader + schema validator
  - costs — namespace for cost-model configuration and registration
"""
from arbitrix_core import costs
from arbitrix_core.backtest import Backtester, BTConfig, BTResult
from arbitrix_core.data import DataProvider, load_ohlcv, validate_ohlcv
from arbitrix_core.strategies import BaseStrategy
from arbitrix_core.trading import Order, Position, Signal, Trade
from arbitrix_core.types import InstrumentConfig

try:
    from arbitrix_core._version import __version__
except ImportError:
    __version__ = "0.1.0.dev0"

__all__ = [
    "Backtester",
    "BTConfig",
    "BTResult",
    "BaseStrategy",
    "Signal",
    "Trade",
    "Order",
    "Position",
    "InstrumentConfig",
    "DataProvider",
    "load_ohlcv",
    "validate_ohlcv",
    "costs",
    "__version__",
]
