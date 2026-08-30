# arbitrix-core

[![PyPI](https://img.shields.io/pypi/v/arbitrix-core.svg)](https://pypi.org/project/arbitrix-core/)
[![Python](https://img.shields.io/pypi/pyversions/arbitrix-core.svg)](https://pypi.org/project/arbitrix-core/)
[![License](https://img.shields.io/pypi/l/arbitrix-core.svg)](https://github.com/G14MB0/arbitrix-core/blob/main/LICENSE)
[![CI](https://github.com/G14MB0/arbitrix-core/actions/workflows/ci.yml/badge.svg)](https://github.com/G14MB0/arbitrix-core/actions/workflows/ci.yml)
[![Docs](https://github.com/G14MB0/arbitrix-core/actions/workflows/docs.yml/badge.svg)](https://g14mb0.github.io/arbitrix-core/)

MIT-licensed open-source backtest engine and cost model from the Arbitrix trading toolkit.

## Install

```bash
pip install arbitrix-core
```

Optional extras:
- `arbitrix-core[fast]` — enables numba JIT for the SL/TP vectorized loop.
- `arbitrix-core[docs]` — mkdocs build dependencies.
- `arbitrix-core[dev]` — pytest + ruff.

## Quickstart

```python
import pandas as pd

from arbitrix_core import Backtester, BTConfig, Signal, BaseStrategy, InstrumentConfig
from arbitrix_core import costs
from arbitrix_core import load_ohlcv


class SmaCross(BaseStrategy):
    symbol = "EURUSD"
    timeframe = "H1"

    def prepare(self, df):
        prepared = df.copy()
        fast = df["close"].rolling(10).mean()
        slow = df["close"].rolling(30).mean()
        prepared["cross_up"] = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
        prepared["cross_down"] = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
        return prepared.dropna()

    def on_bar(self, row, portfolio):
        if row["cross_up"]:
            return [Signal(when=row.name, action="buy", price=float(row["close"]))]
        if row["cross_down"]:
            return [Signal(when=row.name, action="sell", price=float(row["close"]))]
        return []

    def stop_distance_points(self, row):
        return 0.002


df = load_ohlcv("eurusd_h1.csv", time_col="datetime")
costs.configure(commission_per_lot=3.0, point_overrides={"EURUSD": 10.0}, allow_provider_lookups=False)
instrument = InstrumentConfig(
    ib_symbol="EURUSD",
    point_value=10.0,
    contract_size=1.0,
    tick_size=0.0001,
    min_order_size=0.01,
)
result = Backtester(BTConfig(), instruments={"EURUSD": instrument}).run_single(
    df,
    SmaCross(),
    risk_perc=0.01,
    initial_equity=10_000.0,
)
print(result.metrics)
```

`run_single` remains the mono-provider API. Multiprovider and multisymbol code
uses the additive `run_market` API with the `FeedKey` / `MarketFrames` ->
`PreparedMarket` -> `MarketBars` contract.

Full documentation at https://g14mb0.github.io/arbitrix-core/

## Sync to public repo

`src/arbitrix_core` is the canonical source and is published from the upstream
Arbitrix monorepo via subtree split. The private workflow
`.github/workflows/arbitrix-core-sync.yml` force-pushes that subtree to
https://github.com/G14MB0/arbitrix-core
on every push to `main` or `development` that touches the subtree, and
on `workflow_dispatch`.

Requirements:
- Repo secret `PUBLIC_REPO_TOKEN` — fine-grained PAT with `Contents: write`
  scope on `G14MB0/arbitrix-core`.
- Public repo `development` branch is overwritten on every sync. Do not
  commit directly to that branch.
