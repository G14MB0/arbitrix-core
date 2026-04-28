# Cost models

`arbitrix-core` separates **what costs to charge** (the cost model — a Python
module) from **what parameters drive them** (commission per lot, spread points,
slippage points, swap rates). You configure both through a single entry point:
`arbitrix_core.costs.configure(...)`.

## The default model

By default the engine uses
`arbitrix_core.costs.models.parameterized` — a six-function module that
applies user-provided parameters with sensible defaults:

| Function | Meaning |
|----------|---------|
| `commission_one_side(symbol, price, volume_lot)` | Fee per fill, scaled by volume |
| `commission_round_turn(symbol, price, volume_lot)` | Sum of both fills |
| `spread_cost(symbol, spread_points, volume_lot)` | Spread paid in account currency |
| `slippage_cost(symbol, slippage_points, volume_lot)` | Slippage paid in account currency |
| `swap_points(symbol, direction, static_override=None)` | Daily swap in points |
| `swap_cost_per_day(symbol, volume_lot, direction, static_override=None)` | Daily swap in account currency |

## Configuring costs

```python
from arbitrix_core import costs

costs.configure(
    commission_per_lot=3.0,
    point_overrides={"EURUSD": 10.0},   # 1 point = $10 per 1.0 lot
    allow_provider_lookups=False,        # open-core has no provider by default
    model_identifier="arbitrix_core.costs.models.parameterized",  # default
    model_parameters={
        "commission_per_lot": 3.0,
        "spread_points": 1.0,
        "slippage_points": 0.5,
    },
    symbol_model_parameters={
        "eurusd": {"commission_per_lot": 2.5},   # per-symbol override
    },
)
```

| Argument | Purpose |
|----------|---------|
| `commission_per_lot` | Per-fill commission baseline |
| `point_overrides` | `{symbol: point_value}` — required when no provider is wired |
| `allow_provider_lookups` | Set `False` for offline / open-core use |
| `model_identifier` | Module path of the active cost model |
| `model_parameters` | Dict passed to the active model's functions |
| `symbol_model_parameters` | Per-symbol overrides; lookup key is lowercased symbol |
| `symbol_models` | `{symbol: identifier}` — different model per symbol |
| `instruments` | `{symbol: InstrumentConfig}` — for advanced cost structures |
| `provider` | Live broker symbol-info source (closed Arbitrix only) |

After configuration:

```python
costs.get_active_cost_model()  # -> {"name": "...", "module": "..."}
costs.commission_one_side("EURUSD", price=1.10, volume_lot=1.0)
costs.spread_cost("EURUSD", spread_points=1.5, volume_lot=1.0)
```

## Disabling cost components

Two layers of switches control how costs flow into the equity curve:

- `BTConfig.apply_spread_cost` (default `True`) — whether the engine adds
  spread cost to each trade
- `BTConfig.apply_swap_cost` (default `True`) — whether daily swap accrues on
  open trades
- Set `model_parameters={"spread_points": 0, "slippage_points": 0}` to zero
  out at the model layer

## Custom cost modules

Any importable Python module exposing the six required functions can be a
cost model. Convention: define `MODULE_NAME = __name__` at the top so the
registry stores the canonical import path.

Skeleton:

```python
# my_costs.py
MODULE_NAME = __name__

def commission_one_side(symbol, price, volume_lot):
    return 1.5 * volume_lot

def commission_round_turn(symbol, price, volume_lot):
    return 2.0 * commission_one_side(symbol, price, volume_lot)

def spread_cost(symbol, spread_points, volume_lot):
    return 0.5 * spread_points * volume_lot

def slippage_cost(symbol, slippage_points, volume_lot):
    return 0.0

def swap_points(symbol, direction, static_override=None):
    return 0.0

def swap_cost_per_day(symbol, volume_lot, direction, static_override=None):
    return 0.0
```

Registering it:

```python
costs.configure(
    commission_per_lot=1.5,
    point_overrides={"EURUSD": 10.0},
    allow_provider_lookups=False,
    model_identifier="my_costs",            # any importable name
)
```

The registry calls `importlib.import_module(model_identifier)` — there is no
filesystem magic in open-core. Make sure your module is on `sys.path`.

A complete worked example, including a custom `cost_models/zero_slippage.py`
sibling to a runnable backtest, lives at `examples/03_custom_cost_model.py`
in the source tree.

## Per-symbol models

Pass `symbol_models={"EURUSD": "module.a", "AAPL": "module.b"}` to
`configure()` to use a different model per symbol. The active default model
applies to every other symbol. Use `symbol_model_parameters` to give each
symbol its own parameter set.
