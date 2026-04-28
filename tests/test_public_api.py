"""Smoke test confirming the documented public API is importable from the package root."""

import arbitrix_core


def test_public_api_exports():
    expected = set(arbitrix_core.__all__)
    available = set(dir(arbitrix_core))
    missing = expected - available
    assert not missing, f"Missing public exports declared in __all__: {missing}"
    for name in expected:
        assert getattr(arbitrix_core, name) is not None, (
            f"Public export {name!r} resolves to None"
        )


def test_costs_namespace_callable():
    arbitrix_core.costs.set_cost_model("default")
    info = arbitrix_core.costs.get_active_cost_model()
    assert info["module"] == "arbitrix_core.costs.models.parameterized"


def test_run_simple_backtest():
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(100.0, 100.5, 20),
            "high": np.linspace(100.1, 100.6, 20),
            "low": np.linspace(99.9, 100.4, 20),
            "close": np.linspace(100.0, 100.5, 20),
            "volume": np.full(20, 1000.0),
        },
        index=idx,
    )

    class _BuyAndHold(arbitrix_core.BaseStrategy):
        name = "buyhold"
        symbol = "TEST"
        timeframe = "M5"

        def prepare(self, df):
            return df

        def stop_distance_points(self, row):
            return 0.0

        def take_distance_points(self, row):
            return 0.0

        def generate_signals(self, df):
            return [
                arbitrix_core.Signal(
                    when=df.index[0],
                    action="buy",
                    price=float(df.iloc[0]["open"]),
                    reason="entry",
                )
            ]

    arbitrix_core.costs.configure(commission_per_lot=0.0)
    bt = arbitrix_core.Backtester(
        arbitrix_core.BTConfig(
            commission_per_lot=0.0,
            apply_spread_cost=False,
            apply_swap_cost=False,
        )
    )
    result = bt.run_single(
        df,
        _BuyAndHold(),
        risk_perc=1.0,
        initial_equity=10_000.0,
    )

    assert isinstance(result, arbitrix_core.BTResult)
    assert isinstance(result.daily_equity, pd.Series)
    assert isinstance(result.metrics, dict)
