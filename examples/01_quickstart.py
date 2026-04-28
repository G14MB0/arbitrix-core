"""Quickstart: load CSV, run a simple mean-reversion backtest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from arbitrix_core import (
    BTConfig,
    Backtester,
    BaseStrategy,
    Signal,
    costs,
    load_ohlcv,
)


class SimpleMeanReversion(BaseStrategy):
    name = "simple_mr"
    symbol = "EURUSD"
    timeframe = "H1"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma_24"] = df["close"].rolling(24).mean()
        df["std_24"] = df["close"].rolling(24).std()
        df["zscore"] = (df["close"] - df["sma_24"]) / df["std_24"]
        return df

    def stop_distance_points(self, row: pd.Series) -> float:
        return 50.0

    def take_distance_points(self, row: pd.Series) -> float:
        return 100.0

    def warmup_bars(self) -> int:
        return 24

    def on_bar(self, row, portfolio, regime_output=None):
        z = row.get("zscore")
        if z is None or pd.isna(z):
            return []
        ts = row.name if hasattr(row, "name") else None
        price = float(row["close"])
        if z < -1.5:
            return [Signal(when=ts, action="buy", price=price)]
        if z > 1.5:
            return [Signal(when=ts, action="sell", price=price)]
        return []


def main() -> None:
    csv_path = Path(__file__).parent / "data" / "eurusd_h1_sample.csv"
    df = load_ohlcv(csv_path, time_col="datetime")

    costs.configure(
        commission_per_lot=3.0,
        point_overrides={"EURUSD": 10.0},
        allow_provider_lookups=False,
    )

    cfg = BTConfig(commission_per_lot=3.0, apply_swap_cost=False)
    bt = Backtester(cfg)
    result = bt.run_single(
        df, SimpleMeanReversion(), risk_perc=0.01, initial_equity=10_000.0
    )

    final_equity = (
        float(result.daily_equity.iloc[-1])
        if len(result.daily_equity)
        else 10_000.0
    )
    m = result.metrics
    print(f"Final equity:    {final_equity:,.2f}")
    print(f"Net return:      {m.get('net_return_pct', 0.0):.2f}%")
    print(f"Trades:          {len(result.trades)}")
    print(f"Sharpe:          {m.get('Sharpe', 0.0):.2f}")
    print(f"Max drawdown:    {m.get('max_drawdown_pct', 0.0):.2f}%")


if __name__ == "__main__":
    main()
