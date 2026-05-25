"""Re-export shim for :class:`BTConfig`.

``BTConfig`` is defined inline in :mod:`arbitrix_core.backtest.engine` for
historical reasons (engine + config grew together). This module exposes the
same class under the more intuitive ``arbitrix_core.backtest.config`` path so
callers and tests can import config without pulling the whole engine module.

Both import paths return the identical dataclass — there is one canonical
``BTConfig`` definition.
"""

from arbitrix_core.backtest.engine import BTConfig

__all__ = ["BTConfig"]
