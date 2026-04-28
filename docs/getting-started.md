# Getting started

## Install

From PyPI:

```bash
pip install arbitrix-core
```

Optional extras:

| Extra | Adds |
|-------|------|
| `arbitrix-core[fast]` | numba JIT for the SL/TP vectorized loop |
| `arbitrix-core[docs]` | mkdocs build dependencies |
| `arbitrix-core[dev]` | pytest, pytest-cov, ruff |

For local development (after cloning):

```bash
pip install -e ".[dev]"
pytest
```

## Your first backtest

The package ships three runnable examples under `examples/`. The simplest is
`01_quickstart.py`, reproduced here in full:

```python
"""Quickstart: load CSV, run a simple mean-reversion backtest."""
from pathlib import Path

import pandas as pd
from arbitrix_core import (
    BTConfig, Backtester, BaseStrategy, Signal, costs, load_ohlcv,
)


class SimpleMeanReversion(BaseStrategy):
    name = "simple_mr"
    symbol = "EURUSD"
    timeframe = "H1"

    def prepare(self, df):
        df = df.copy()
        df["sma_24"] = df["close"].rolling(24).mean()
        df["std_24"] = df["close"].rolling(24).std()
        df["zscore"] = (df["close"] - df["sma_24"]) / df["std_24"]
        return df

    def stop_distance_points(self, row): return 50.0
    def take_distance_points(self, row): return 100.0
    def warmup_bars(self): return 24

    def on_bar(self, row, portfolio, regime_output=None):
        z = row.get("zscore")
        if z is None or pd.isna(z):
            return []
        if z < -1.5:
            return [Signal(when=row.name, action="buy",  price=float(row["close"]))]
        if z > 1.5:
            return [Signal(when=row.name, action="sell", price=float(row["close"]))]
        return []


csv = Path(__file__).parent / "data" / "eurusd_h1_sample.csv"
df = load_ohlcv(csv, time_col="datetime")

costs.configure(
    commission_per_lot=3.0,
    point_overrides={"EURUSD": 10.0},
    allow_provider_lookups=False,
)

cfg = BTConfig(commission_per_lot=3.0, apply_swap_cost=False)
result = Backtester(cfg).run_single(
    df, SimpleMeanReversion(), risk_perc=0.01, initial_equity=10_000.0,
)

print(f"Final equity:   {float(result.daily_equity.iloc[-1]):,.2f}")
print(f"Net return:     {result.metrics.get('net_return_pct', 0):.2f}%")
print(f"Trades:         {len(result.trades)}")
print(f"Sharpe:         {result.metrics.get('Sharpe', 0):.2f}")
print(f"Max drawdown:   {result.metrics.get('max_drawdown_pct', 0):.2f}%")
```

Run it:

```bash
python examples/01_quickstart.py
```

## What you get back

`Backtester.run_single(...)` returns a [`BTResult`](configuration.md#btresult)
dataclass containing:

- `trades` — `list[Trade]` of every closed trade
- `daily_equity` — `pd.Series` of end-of-day equity (use `.iloc[-1]` for final)
- `gross_equity` — equity series before cost adjustments
- `equity_marked` — bar-level mark-to-market equity
- `metrics` — dict of summary statistics (`net_return_pct`, `Sharpe`, `Sortino`,
  `MaxDD`, `max_drawdown_pct`, `ProfitFactor`, `Expectancy`, `TradeCount`,
  `gross_pnl`, `net_pnl`, `total_commission`, `total_spread_cost`,
  `total_slippage_cost`)
- `metadata` — engine metadata (warmup index, prepared schema, etc.)
- `orders`, `positions`, `prepared` — internals exposed for diagnostics

## Where to next

- [Data format](data-format.md) — DataFrame schema your CSV must satisfy
- [Strategies](strategies.md) — `BaseStrategy` lifecycle and hooks
- [Cost models](costs.md) — parameterized model + custom modules
- [Configuration](configuration.md) — every field on `BTConfig` and `BTResult`
