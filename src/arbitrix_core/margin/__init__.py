"""Per-symbol margin model package.

Public surface:

* :class:`MarginModel` — protocol
* :class:`Money` — amount + currency value type
* :class:`MarginCallEvent` — emitted by Portfolio.check_maintenance_margin
* :class:`NoMargin`, :class:`FuturesUSDMargin`, :class:`RegTMargin`,
  :class:`CFDMargin` — four shipped implementations
* :class:`MarginParams` — per-symbol params payload
* :func:`default_margin_model_for` — asset-class → model resolver
* :func:`register_margin_params`, :func:`get_margin_params`,
  :func:`clear_margin_params_registry` — process-global registry
"""

from arbitrix_core.margin.protocol import MarginModel, Money, MarginCallEvent
from arbitrix_core.margin.models import NoMargin, FuturesUSDMargin, RegTMargin, CFDMargin

__all__ = [
    "MarginModel",
    "Money",
    "MarginCallEvent",
    "NoMargin",
    "FuturesUSDMargin",
    "RegTMargin",
    "CFDMargin",
]
