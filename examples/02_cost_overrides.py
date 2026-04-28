"""Cost overrides: same strategy run thrice under different cost regimes."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd

from arbitrix_core import BTConfig, Backtester, costs, load_ohlcv

quickstart = import_module("01_quickstart")
SimpleMeanReversion = quickstart.SimpleMeanReversion


def run_with(
    commission: float,
    spread_points: float,
    slippage_points: float,
    label: str,
    df: pd.DataFrame,
) -> None:
    costs.configure(
        commission_per_lot=commission,
        point_overrides={"EURUSD": 10.0},
        allow_provider_lookups=False,
        model_identifier="arbitrix_core.costs.models.parameterized",
        model_parameters={
            "commission_per_lot": commission,
            "spread_points": spread_points,
            "slippage_points": slippage_points,
        },
    )
    cfg = BTConfig(commission_per_lot=commission, apply_swap_cost=False)
    bt = Backtester(cfg)
    res = bt.run_single(
        df, SimpleMeanReversion(), risk_perc=0.01, initial_equity=10_000.0
    )
    final_equity = (
        float(res.daily_equity.iloc[-1]) if len(res.daily_equity) else 10_000.0
    )
    ret_pct = res.metrics.get("net_return_pct", 0.0)
    print(
        f"[{label:>10}]  return={ret_pct:6.2f}%  trades={len(res.trades):3d}  "
        f"final_equity={final_equity:10,.2f}"
    )


def main() -> None:
    csv = Path(__file__).parent / "data" / "eurusd_h1_sample.csv"
    df = load_ohlcv(csv, time_col="datetime")

    run_with(0.0, 0.0, 0.0, "free", df)
    run_with(3.0, 1.0, 0.5, "retail", df)
    run_with(7.0, 3.0, 2.0, "punitive", df)


if __name__ == "__main__":
    main()
