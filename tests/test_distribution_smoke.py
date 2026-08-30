from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_built_wheel_runs_multifeed_multisymbol_strategy(tmp_path: Path) -> None:
    """Prove the public artifact works without importing the source checkout."""

    project_root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheel"
    install_dir = tmp_path / "installed"
    wheel_dir.mkdir()

    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(project_root),
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next(wheel_dir.glob("arbitrix_core-*.whl"))
    installed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(wheel),
            "--no-deps",
            "--target",
            str(install_dir),
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr

    smoke = tmp_path / "smoke.py"
    smoke.write_text(
        """
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

import pandas as pd
import arbitrix_core
from arbitrix_core import (
    BTConfig,
    Backtester,
    BaseStrategy,
    FeedKey,
    InstrumentConfig,
    MarketFrames,
    PreparedMarket,
    Signal,
)

installed = Path(sys.argv[1]).resolve()
assert Path(arbitrix_core.__file__).resolve().is_relative_to(installed)

primary = FeedKey("main", "EURUSD", "M1")
target = FeedKey("main", "GBPUSD", "M1")
auxiliary = FeedKey("ibkr-data", "DXY", "M1")
index = pd.date_range("2026-08-30T10:00:00Z", periods=3, freq="min")

def frame(close):
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.01 for value in close],
            "low": [value - 0.01 for value in close],
            "close": close,
            "volume": 100.0,
            "spread": 0.0,
            "__account_point_value__": 1.0,
        },
        index=index,
    )

market = MarketFrames(
    {
        primary: frame([1.10, 1.11, 1.12]),
        target: frame([1.30, 1.31, 1.32]),
        auxiliary: frame([98.0, 98.1, 98.2]),
    },
    primary=primary,
    execution_keys=(primary, target),
)

class WheelStrategy(BaseStrategy):
    symbol = "EURUSD"
    timeframe = "M1"

    def prepare(self, df, *, data=None):
        assert data is market
        return PreparedMarket(
            {
                key: value.assign(aux_close=data[auxiliary]["close"].to_numpy())
                for key, value in data.items()
            },
            primary=data.primary,
        )

    def on_bar(self, row, portfolio, *, data=None):
        assert float(data[auxiliary]["aux_close"]) == float(data[auxiliary]["close"])
        if row.name != index[0]:
            return []
        return [
            Signal(
                when=row.name,
                action="buy",
                price=float(data[target]["close"]),
                volume=1.0,
                symbol="GBPUSD",
            )
        ]

    def stop_distance_points(self, row, *, symbol=None, data=None):
        assert symbol == "GBPUSD"
        return 0.01

    def take_distance_points(self, row, *, symbol=None, data=None):
        return 0.02

instruments = {
    symbol: InstrumentConfig(
        ib_symbol=symbol,
        point_value=1.0,
        contract_size=1.0,
        tick_size=0.0001,
        min_order_size=0.01,
    )
    for symbol in ("EURUSD", "GBPUSD")
}
result = Backtester(
    BTConfig(
        commission_per_lot=0.0,
        default_slippage_points=0.0,
        apply_spread_cost=False,
        apply_swap_cost=False,
    ),
    instruments=instruments,
).run_market(market, WheelStrategy(), risk_perc=0.01, initial_equity=10_000.0)

assert result.orders[0].symbol == "GBPUSD"
assert result.trades[0].symbol == "GBPUSD"
assert result.metadata["primary_symbol"] == "EURUSD"
assert len(result.metadata["market_feeds"]) == 3
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(smoke), str(install_dir)],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
