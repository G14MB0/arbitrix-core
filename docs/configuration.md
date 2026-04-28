# Configuration reference

This page documents every public field on the dataclasses you'll touch in
day-to-day use: `BTConfig`, `BTResult`, `Trade`, `Order`, and `Position`.

## BTConfig

Controls global engine behaviour. Pass an instance to `Backtester(cfg)`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `commission_per_lot` | `float` | `3.0` | Baseline commission per fill, also seeded into `costs.set_commission_per_lot()` at construction time |
| `default_slippage_points` | `float` | `0.5` | Slippage in points used when no per-symbol override applies |
| `slippage_atr_multiplier` | `float` | `0.0` | Multiplier on ATR-based slippage (`0.0` disables) |
| `apply_spread_cost` | `bool` | `True` | Whether to charge spread on every trade |
| `apply_swap_cost` | `bool` | `True` | Whether to accrue daily swap on open trades |
| `apply_stop_take` | `bool` | `True` | Whether the SL/TP loop runs |
| `market_fill_price` | `str` | `"close"` | `"open"` or `"close"` — bar reference for entry fills |
| `exit_fill_price` | `str` | `"close"` | `"open"` or `"close"` — bar reference for exit fills |
| `intra_bar_model` | `str` | `"sl_first"` | `"sl_first"`, `"tp_first"`, or `"none"` — order of SL/TP checks intra-bar |
| `trailing_mode` | `str` | `"none"` | Reserved for future trailing-stop modes |
| `trailing_params` | `dict[str, float]` | `{}` | Reserved for future trailing-stop parameters |

Note: there is **no** `initial_equity` field on `BTConfig`. Initial equity is
passed at call time to `run_single(...)`.

## Backtester

```python
Backtester(cfg: BTConfig, instruments: dict[str, InstrumentConfig] | None = None)
```

Methods:

- `run_single(df, strategy, risk_perc, initial_equity, swap_override=None, *, ...)`
  → `BTResult`

`risk_perc` is the fraction of equity risked per trade (e.g. `0.01` for 1%).
The engine sizes lot volume from this and the strategy's
`stop_distance_points(row)`.

## BTResult

Returned by `Backtester.run_single(...)`.

| Field | Type | Notes |
|-------|------|-------|
| `trades` | `list[Trade]` | Every closed trade, in chronological order |
| `daily_equity` | `pd.Series` | End-of-day equity, indexed by date |
| `gross_equity` | `pd.Series` | Equity before cost adjustments |
| `equity_marked` | `pd.Series` | Bar-level marked-to-market equity |
| `metrics` | `dict[str, float]` | Summary statistics — see below |
| `metadata` | `dict[str, Any]` | Engine internals (warmup index, prepared schema, etc.) |
| `orders` | `list[Order]` | Every order created during the run |
| `positions` | `list[Position]` | Aggregated view of trades by symbol |
| `prepared` | `pd.DataFrame \| None` | The post-`prepare()` DataFrame for diagnostics |

### Common `metrics` keys

- `net_return_pct` — total net return as %
- `Sharpe`, `Sortino` — risk-adjusted ratios
- `MaxDD`, `max_drawdown_pct` — drawdown depth (absolute and %)
- `ProfitFactor`, `Expectancy`, `TradeCount`
- `gross_pnl`, `net_pnl`
- `total_commission`, `total_spread_cost`, `total_slippage_cost`

Use `result.metrics.get("net_return_pct", 0.0)` rather than direct attribute
access — keys are populated only when relevant.

## Trade

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | UUID4 |
| `symbol` | `str` | |
| `side` | `Literal["long", "short"]` | |
| `entry_time` | `pd.Timestamp` | UTC |
| `entry_price` | `float` | |
| `exit_time` | `pd.Timestamp \| None` | `None` while open |
| `exit_price` | `float \| None` | |
| `volume` | `float` | Lots |
| `stop_points` / `take_points` | `float` | Distances in points |
| `pnl`, `gross_pnl`, `net_pnl` | `float` | Account currency |
| `commission_paid`, `spread_cost`, `slippage_cost`, `swap_pnl` | `float` | Cost components |
| `notes` | `dict[str, float]` | Strategy-supplied annotations |
| `order_id`, `exit_order_id` | `str \| None` | Linked order ids |
| `broker_ticket` | `int \| None` | Reserved for live mirroring |
| `strategy`, `magic` | `str \| int` | Provenance tags |

## Order

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | UUID4 |
| `symbol`, `side`, `type`, `volume` | various | Required |
| `status` | `Literal["new", "working", "filled", "cancelled", "expired"]` | |
| `tif` | `Literal["GTC", "GTD"]` | |
| `price`, `created_at`, `valid_until` | optional | |
| `stop_points`, `take_points` | `float` | SL/TP distances |
| `trail_params` | `dict[str, float]` | |
| `parent_id` | `str \| None` | For partial-close children |
| `broker_ticket`, `strategy`, `magic` | optional | |

## Position

Aggregated view across multiple `Trade`s for a single `(symbol, side)`:

| Field | Type | Notes |
|-------|------|-------|
| `symbol` | `str` | |
| `side` | `Literal["long", "short"]` | |
| `volume` | `float` | Net volume |
| `avg_price` | `float` | Volume-weighted average entry |
| `trades` | `list[Trade]` | Member trades |
