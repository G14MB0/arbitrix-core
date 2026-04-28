"""Custom cost model: register a user module and run a backtest against it."""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from arbitrix_core import BTConfig, Backtester, costs, load_ohlcv

EXAMPLES = Path(__file__).parent
sys.path.insert(0, str(EXAMPLES))

quickstart = import_module("01_quickstart")
SimpleMeanReversion = quickstart.SimpleMeanReversion


def main() -> None:
    df = load_ohlcv(EXAMPLES / "data" / "eurusd_h1_sample.csv", time_col="datetime")

    costs.configure(
        commission_per_lot=1.5,
        point_overrides={"EURUSD": 10.0},
        allow_provider_lookups=False,
        model_identifier="cost_models.zero_slippage",
    )
    active = costs.get_active_cost_model()
    print(f"Active cost model: {active['name']}  ({active['module']})")

    bt = Backtester(BTConfig(commission_per_lot=1.5, apply_swap_cost=False))
    res = bt.run_single(
        df, SimpleMeanReversion(), risk_perc=0.01, initial_equity=10_000.0
    )
    final_equity = (
        float(res.daily_equity.iloc[-1]) if len(res.daily_equity) else 10_000.0
    )
    ret_pct = res.metrics.get("net_return_pct", 0.0)
    print(
        f"return={ret_pct:.2f}%  trades={len(res.trades)}  "
        f"final_equity={final_equity:,.2f}"
    )


if __name__ == "__main__":
    main()
