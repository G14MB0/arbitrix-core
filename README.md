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

from arbitrix_core import Backtester, BTConfig, Signal, BaseStrategy
from arbitrix_core import costs
from arbitrix_core import load_ohlcv


class SmaCross(BaseStrategy):
    def generate_signals(self, df):
        fast = df["close"].rolling(10).mean()
        slow = df["close"].rolling(30).mean()
        signals = []
        for i in range(1, len(df)):
            if fast.iloc[i - 1] < slow.iloc[i - 1] and fast.iloc[i] > slow.iloc[i]:
                signals.append(Signal(when=df.index[i], action="buy", price=df["close"].iloc[i]))
            elif fast.iloc[i - 1] > slow.iloc[i - 1] and fast.iloc[i] < slow.iloc[i]:
                signals.append(Signal(when=df.index[i], action="sell", price=df["close"].iloc[i]))
        return signals


df = load_ohlcv("eurusd_h1.csv", time_col="datetime")
costs.configure(commission_per_lot=3.0, point_overrides={"EURUSD": 10.0}, allow_provider_lookups=False)
result = Backtester(BTConfig()).run_single(df, SmaCross(), risk_perc=0.01, initial_equity=10_000.0)
print(result.metrics)
```

Full documentation at https://g14mb0.github.io/arbitrix-core/

## Sync to public repo

`arbitrix-core` is published from the upstream Arbitrix monorepo via subtree split. The
private workflow `.github/workflows/arbitrix-core-sync.yml` force-pushes
`arbitrix/src/arbitrix_core` to https://github.com/G14MB0/arbitrix-core
on every push to `main` or `development` that touches the subtree, and
on `workflow_dispatch`.

Requirements:
- Repo secret `PUBLIC_REPO_TOKEN` — fine-grained PAT with `Contents: write`
  scope on `G14MB0/arbitrix-core`.
- Public repo `development` branch is overwritten on every sync. Do not
  commit directly to that branch.
