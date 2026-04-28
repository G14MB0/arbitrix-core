from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from arbitrix_core.costs import base

_pre_import_hooks: List[Callable[[], None]] = []


def register_pre_import_hook(hook: Callable[[], None]) -> None:
    """Register a callable invoked before each cost-model module import.

    Allows hosts (e.g. arbitrix) to inject sys.path adjustments so
    filesystem-discovered cost models resolve through importlib.
    """
    if hook not in _pre_import_hooks:
        _pre_import_hooks.append(hook)

_COST_FUNCTIONS = (
    "commission_one_side",
    "commission_round_turn",
    "spread_cost",
    "slippage_cost",
    "swap_points",
    "swap_cost_per_day",
)
_DEFAULT_MODEL_MODULE = "arbitrix_core.costs.models.parameterized"


@dataclass
class _CostModel:
    name: str
    module_name: str
    module: ModuleType
    functions: Dict[str, Callable[..., Any]]
    parameters: Dict[str, Any]
    configure_hook: Optional[Callable[[Dict[str, Any]], None]] = None

    def call(self, func_name: str, *args, **kwargs):
        func = self.functions.get(func_name)
        if func is None:
            raise AttributeError(f"Cost model '{self.name}' does not implement {func_name}")
        return func(*args, **kwargs)


_active_model: _CostModel
_active_model_id: str
_default_model: _CostModel
_symbol_models: Dict[str, _CostModel] = {}
_MODEL_PARAMETERS: Dict[str, Dict[str, Any]] = {}
_SYMBOL_MODEL_PARAMETERS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_module(module_name: str) -> ModuleType:
    for hook in list(_pre_import_hooks):
        try:
            hook()
        except Exception:
            pass
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def _build_model(name: str, module_name: str, module: ModuleType) -> _CostModel:
    functions: Dict[str, Callable[..., Any]] = {}
    for func_name in _COST_FUNCTIONS:
        func = getattr(module, func_name, None)
        if callable(func):
            functions[func_name] = func
    configure_hook = getattr(module, "configure", None)
    if configure_hook is not None and not callable(configure_hook):
        configure_hook = None
    return _CostModel(
        name=name,
        module_name=module_name,
        module=module,
        functions=functions,
        parameters={},
        configure_hook=configure_hook,
    )


def _normalize_identifier(identifier: Optional[str]) -> tuple[str, str]:
    if not identifier:
        return ("default", _DEFAULT_MODEL_MODULE)
    lowered = identifier.strip()
    if lowered in {"default", "builtin", "standard"}:
        return ("default", _DEFAULT_MODEL_MODULE)
    if "." in lowered:
        return (lowered, lowered)
    return (lowered, lowered)


def _activate_model(identifier: Optional[str]) -> _CostModel:
    name, module_name = _normalize_identifier(identifier)
    module = _load_module(module_name)
    canonical = getattr(module, "MODULE_NAME", None)
    if isinstance(canonical, str) and canonical:
        module_name = canonical
    return _build_model(name, module_name, module)


def set_cost_model(identifier: Optional[str]) -> str:
    global _active_model, _active_model_id
    model = _activate_model(identifier)
    _active_model = model
    _active_model_id = model.module_name
    return model.name


def get_active_cost_model() -> Dict[str, str]:
    return {"name": _active_model.name, "module": _active_model.module_name}


def _model_for_symbol(symbol: Optional[str]) -> _CostModel:
    try:
        key = str(symbol).lower() if symbol is not None else ""
    except Exception:
        key = ""
    if key and key in _symbol_models:
        return _symbol_models[key]
    try:
        return _active_model
    except NameError:  # pragma: no cover - module not yet initialised
        _bootstrap_default_model()
        return _active_model


def _call_cost_function(name: str, *args, **kwargs):
    symbol = kwargs.get("symbol")
    if symbol is None and args:
        symbol = args[0]
    model = _model_for_symbol(symbol)
    try:
        func = model.functions.get(name)
    except NameError:  # pragma: no cover - module not yet initialised
        _bootstrap_default_model()
        model = _model_for_symbol(symbol)
        func = model.functions.get(name)
    if func is None:
        func = _default_model.functions[name]
    result = func(*args, **kwargs)
    if name in {"commission_one_side", "commission_round_turn"}:
        if result is None:
            raise ValueError(f"Cost model '{model.name}' returned None for {name}")
        if float(result) <= 0:
            raise ValueError(f"Cost model '{model.name}' produced non-positive commission: {result}")
    return result


def _build_context() -> Dict[str, Any]:
    return {
        "model": get_active_cost_model(),
        "symbol_models": {key: value.module_name for key, value in _symbol_models.items()},
        "provider": base.get_provider(),
        "instruments": base.get_instruments(),
        "point_overrides": base.get_point_overrides(),
        "commission_per_lot": base.get_commission_per_lot(),
        "model_parameters": dict(_MODEL_PARAMETERS),
        "symbol_model_parameters": dict(_SYMBOL_MODEL_PARAMETERS),
        "base": base,
    }


def configure(
    *,
    provider=None,
    commission_per_lot: Optional[float] = None,
    point_overrides: Optional[Dict[str, float]] = None,
    instruments: Optional[Dict[str, Any]] = None,
    allow_provider_lookups: bool = True,
    model_identifier: Optional[str] = None,
    symbol_models: Optional[Dict[str, str]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    symbol_model_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    clear_provider: bool = False,
) -> None:
    global _symbol_models, _MODEL_PARAMETERS, _SYMBOL_MODEL_PARAMETERS
    _symbol_models = {}
    _MODEL_PARAMETERS = {}
    _SYMBOL_MODEL_PARAMETERS = {}
    if symbol_models:
        for symbol, identifier in symbol_models.items():
            if symbol is None or identifier is None:
                continue
            try:
                model_name, module_name = _normalize_identifier(identifier)
                module = _load_module(module_name)
                canonical = getattr(module, "MODULE_NAME", None)
                if isinstance(canonical, str) and canonical:
                    module_name = canonical
                model = _build_model(model_name, module_name, module)
                model.parameters = (symbol_model_parameters or {}).get(str(symbol).lower(), {}) if symbol_model_parameters else {}
                _symbol_models[str(symbol).lower()] = model
            except Exception:
                continue
    if model_identifier is not None:
        set_cost_model(model_identifier)
    # Persist parameters for active model and symbol overrides
    try:
        _active_model.parameters = model_parameters or {}
        _MODEL_PARAMETERS[_active_model.module_name] = dict(model_parameters or {})
    except NameError:
        pass
    if symbol_model_parameters:
        for key, params in symbol_model_parameters.items():
            if key is None:
                continue
            _SYMBOL_MODEL_PARAMETERS[str(key).lower()] = dict(params or {})
    base.configure_environment(
        provider=provider,
        commission_per_lot=commission_per_lot,
        point_overrides=point_overrides,
        instruments=instruments,
        allow_provider_lookups=allow_provider_lookups,
        clear_provider=clear_provider,
    )
    hook = getattr(_active_model, "configure_hook", None)
    if callable(hook):
        context = _build_context()
        try:
            hook(context)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                f"Cost model '{_active_model.name}' failed during configure: {exc}"
            ) from exc


def commission_round_turn(symbol: str, price: float, volume_lot: float) -> float:
    return float(_call_cost_function("commission_round_turn", symbol, price, volume_lot))


def commission_one_side(symbol: str, price: float, volume_lot: float) -> float:
    return float(_call_cost_function("commission_one_side", symbol, price, volume_lot))


def spread_cost(symbol: str, spread_points: float, volume_lot: float) -> float:
    return float(_call_cost_function("spread_cost", symbol, spread_points, volume_lot))


def slippage_cost(symbol: str, slippage_points: float, volume_lot: float) -> float:
    return float(_call_cost_function("slippage_cost", symbol, slippage_points, volume_lot))


def swap_points(symbol: str, direction: str, static_override: Optional[dict] = None) -> float:
    return float(_call_cost_function("swap_points", symbol, direction, static_override))


def swap_cost_per_day(symbol: str, volume_lot: float, direction: str, static_override: Optional[dict] = None) -> float:
    return float(
        _call_cost_function("swap_cost_per_day", symbol, volume_lot, direction, static_override)
    )


def warmup_from_provider(symbols: list[str]) -> None:
    base.warmup_from_provider(symbols)


def get_point_value(symbol: str) -> float:
    return base.get_point_value(symbol)


def set_commission_per_lot(value: float) -> None:
    base.set_commission_per_lot(value)


def get_commission_per_lot() -> float:
    return base.get_commission_per_lot()


def commission_minimum(volume_lot: float) -> float:
    return base.commission_minimum(volume_lot)


def get_provider():
    return base.get_provider()


def get_instruments():
    return base.get_instruments()


def get_point_overrides():
    return base.get_point_overrides()


def model_parameters(symbol: Optional[str] = None, module_name: Optional[str] = None) -> Dict[str, Any]:
    """Return configured parameters for the active or symbol-specific cost model."""
    base_params: Dict[str, Any] = {}
    if module_name:
        base_params = dict(_MODEL_PARAMETERS.get(module_name, {}))
    else:
        try:
            base_params = dict(_MODEL_PARAMETERS.get(_active_model.module_name, {}))
        except NameError:
            base_params = {}
    if symbol is not None:
        params = _SYMBOL_MODEL_PARAMETERS.get(str(symbol).lower())
        if params:
            merged = dict(base_params)
            merged.update(params)
            return merged
    return dict(base_params)


def trade_notional(symbol: str, price: float, volume_lot: float) -> float:
    return base.trade_notional(symbol, price, volume_lot)


def tick_size(symbol: str) -> float:
    return base.tick_size(symbol)


# ---------------------------------------------------------------------------
# Bootstrap default model on module import
# ---------------------------------------------------------------------------

def _bootstrap_default_model() -> None:
    global _default_model, _active_model, _active_model_id, _symbol_models
    module = _load_module(_DEFAULT_MODEL_MODULE)
    _default_model = _build_model("default", _DEFAULT_MODEL_MODULE, module)
    _active_model = _default_model
    _active_model_id = _default_model.module_name
    _symbol_models = {}


_bootstrap_default_model()


# Cache management for optimization workers
def export_caches() -> Dict[str, Any]:
    """Export point value and swap caches for transfer to workers."""
    return base.export_caches()


def import_caches(data: Dict[str, Any]) -> None:
    """Import cached values into worker process."""
    base.import_caches(data)


__all__ = [
    "configure",
    "set_cost_model",
    "get_active_cost_model",
    "commission_round_turn",
    "commission_one_side",
    "spread_cost",
    "slippage_cost",
    "swap_points",
    "swap_cost_per_day",
    "warmup_from_provider",
    "get_point_value",
    "set_commission_per_lot",
    "get_commission_per_lot",
    "commission_minimum",
    "trade_notional",
    "tick_size",
    "get_provider",
    "get_instruments",
    "get_point_overrides",
    "model_parameters",
    "export_caches",
    "import_caches",
]
