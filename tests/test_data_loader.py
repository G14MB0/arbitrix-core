import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arbitrix_core.data import load_ohlcv, validate_ohlcv, DataProvider


def _good_df(rows: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.linspace(1.10, 1.11, rows, dtype="float64"),
            "high": np.linspace(1.105, 1.115, rows, dtype="float64"),
            "low": np.linspace(1.095, 1.105, rows, dtype="float64"),
            "close": np.linspace(1.10, 1.11, rows, dtype="float64"),
            "volume": np.full(rows, 1000.0, dtype="float64"),
        },
        index=idx,
    )


def test_validate_ohlcv_accepts_well_formed_df():
    validate_ohlcv(_good_df())


def test_validate_ohlcv_accepts_optional_spread():
    df = _good_df()
    df["spread"] = 0.5
    validate_ohlcv(df)


def test_validate_ohlcv_rejects_non_datetime_index():
    df = _good_df().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_naive_index():
    df = _good_df()
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_non_utc_index():
    df = _good_df()
    df.index = df.index.tz_convert("Europe/London")
    with pytest.raises(ValueError, match="UTC"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_descending_index():
    df = _good_df()
    df = df.iloc[::-1]
    with pytest.raises(ValueError, match="monotonic"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_duplicate_timestamps():
    df = _good_df()
    df = pd.concat([df, df.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError, match="duplicates"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_missing_required_column():
    df = _good_df().drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        validate_ohlcv(df)


def test_load_ohlcv_csv_roundtrip(tmp_path: Path):
    df = _good_df()
    csv_path = tmp_path / "data.csv"
    df.reset_index().rename(columns={"index": "time"}).to_csv(csv_path, index=False)

    loaded = load_ohlcv(csv_path)
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_load_ohlcv_parquet_roundtrip(tmp_path: Path):
    pytest.importorskip("pyarrow")
    df = _good_df()
    pq_path = tmp_path / "data.parquet"
    df.to_parquet(pq_path)

    loaded = load_ohlcv(pq_path)
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_load_ohlcv_dedupes_keeping_last(tmp_path: Path):
    df = _good_df()
    duplicated = pd.concat([df, df.iloc[[0]].assign(close=999.0)])
    csv_path = tmp_path / "data.csv"
    duplicated.reset_index().rename(columns={"index": "time"}).to_csv(csv_path, index=False)

    loaded = load_ohlcv(csv_path)
    assert loaded.iloc[0]["close"] == 999.0


def test_data_provider_protocol_runtime_check():
    class FakeProvider:
        def get_symbol_info(self, symbol: str):
            return {"point_value": 1.0}

    assert isinstance(FakeProvider(), DataProvider)
