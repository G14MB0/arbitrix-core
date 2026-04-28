# Strategies

A strategy is a subclass of `arbitrix_core.BaseStrategy`. The engine drives it
through a fixed lifecycle on every bar.

## Lifecycle

```
1. backtester.run_single(df, strategy, ...) called
2. strategy.prepare(df)              -> prepared DataFrame
3. strategy.warmup_bars()             -> int N; engine skips first N bars
4. for each bar past the warmup:
       strategy.on_bar(row, portfolio, regime_output=None) -> list[Signal]
       engine fills orders, applies costs, marks equity, checks SL/TP
5. result = BTResult(...)
```

`prepare(df)` runs once on the full DataFrame. Compute indicators, drop nulls,
return the augmented frame. Anything you set up here is available as columns on
`row` inside `on_bar()`.

## Class attributes

| Attribute | Default | Purpose |
|-----------|---------|---------|
| `name` | (required) | Identifier used in result metadata |
| `symbol` | `""` | Single-symbol strategies (the open-core default) |
| `timeframe` | `"M5"` | Informational; informs no engine behaviour |
| `requires_portfolio` | `False` | Reserved for advanced multi-symbol portfolio hooks |

## Required hooks

```python
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    symbol = "EURUSD"

    def prepare(self, df):
        # Compute and return augmented DataFrame
        return df

    def warmup_bars(self) -> int:
        # Bars to skip before on_bar() starts firing
        return 24

    def stop_distance_points(self, row) -> float:
        # SL distance in points from entry
        return 50.0

    def take_distance_points(self, row) -> float:
        # TP distance in points from entry
        return 100.0

    def on_bar(self, row, portfolio, regime_output=None) -> list[Signal]:
        # Return zero or more Signals to act this bar
        return []
```

## Signals

`Signal` is the open-core trading-intent dataclass. The minimum required
fields:

```python
from arbitrix_core import Signal

Signal(when=row.name, action="buy",  price=float(row["close"]))
Signal(when=row.name, action="sell", price=float(row["close"]))
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `when` | `pd.Timestamp` | yes | Always pass `row.name` |
| `action` | `Literal` | yes | `"buy"`, `"sell"`, `"exit"`, `"close"`, `"partial_close"`, `"modify_sl"`, `"modify_tp"`, `"cancel_order"` |
| `price` | `float` | yes | Reference price; the engine may fill at open or close per `BTConfig.market_fill_price` |
| `volume` | `float` | no | If `None`, engine sizes from `risk_perc + initial_equity + stop_distance_points` |
| `order_type` | `Literal` | no | `"market"` (default), `"limit"`, `"stop"` |
| `limit_price` / `stop_price` | `float` | no | Required when `order_type` ≠ `"market"` |
| `tif` | `Literal` | no | `"GTC"` (default) or `"GTD"` |
| `valid_until` | `pd.Timestamp` | no | Required when `tif="GTD"` |
| `target_trade_id` / `target_order_id` | `str` | no | For partial closes / modifications |
| `close_volume` | `float` | no | Volume to close in `partial_close` actions |
| `new_sl` / `new_tp` | `float` | no | New SL / TP for `modify_sl` / `modify_tp` actions |
| `risk_multiplier` | `float` | no | Default `1.0`; multiplies sized volume |
| `magic` | `int` | no | Strategy-defined tag forwarded to the resulting Order/Trade |
| `reason` | `str` | no | Free-form annotation copied into the Trade `notes` |

`Signal` carries no `symbol` field — the symbol is implied by the strategy
that emitted it.

## Position-aware hooks

Open-core ships the building blocks for portfolio-aware strategies:

- `BaseStrategy.requires_portfolio = True` — opt in to the portfolio object
  being injected before `on_bar()` fires.
- `portfolio` argument inside `on_bar()` — exposes current open trades and
  marked-to-market equity so you can gate entries or emit exits reactively.

Use `portfolio.open_trades()` to iterate live trades and emit `Signal(action="exit", target_trade_id=...)` to flatten them.

## Helpers

`BaseStrategy` exposes a few static helpers for time-of-day logic:

- `to_market_time(ts, tz)` — convert a UTC timestamp to a market timezone
- `to_utc_time(ts, tz=None)` — convert any timestamp back to UTC
- `session_day(ts, tz)` / `session_hour(ts, tz)` — date/hour in a given tz
- `is_in_session(ts, tz, windows)` — gate signals to specific intraday windows
