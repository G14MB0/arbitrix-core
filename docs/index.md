# arbitrix-core

MIT-licensed open-source backtest engine and cost model from the
[Arbitrix](https://github.com/G14MB0/arbitrix-core) trading toolkit.

## What it is

A minimal, batch-only backtester plus a configurable cost model.

- **Backtester** — bar-by-bar batch execution over an OHLCV `DataFrame`.
- **Cost model** — user-supplied parameters or a registered Python module.
- **Data loader** — strict OHLCV schema validation.

## What it isn't

- No live execution, no broker connectivity, no order routing.
- No optimization, walk-forward, or parameter search.
- No streaming engine — batch only.

For the full proprietary stack, see the closed Arbitrix product.

## Install

```bash
pip install arbitrix-core
```

Numba acceleration:

```bash
pip install arbitrix-core[fast]
```

## 30-line quickstart

```python
import pandas as pd
from arbitrix_core import Backtester, BTConfig, BaseStrategy, Signal, costs, load_ohlcv

class Strat(BaseStrategy):
    name = "demo"
    symbol = "EURUSD"
    timeframe = "H1"

    def prepare(self, df):
        df = df.copy()
        df["sma"] = df["close"].rolling(24).mean()
        return df

    def stop_distance_points(self, row): return 50.0
    def take_distance_points(self, row): return 100.0
    def warmup_bars(self): return 24

    def on_bar(self, row, portfolio, regime_output=None):
        if row["close"] < row["sma"]:
            return [Signal(when=row.name, action="buy", price=float(row["close"]))]
        return []

costs.configure(point_overrides={"EURUSD": 10.0}, allow_provider_lookups=False)
df = load_ohlcv("eurusd_h1.csv", time_col="datetime")
res = Backtester(BTConfig(apply_swap_cost=False)).run_single(
    df, Strat(), risk_perc=0.01, initial_equity=10_000.0,
)
print(float(res.daily_equity.iloc[-1]), res.metrics.get("net_return_pct"))
```

See [Getting started](getting-started.md) for the full walkthrough.
