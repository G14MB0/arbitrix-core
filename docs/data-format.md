# Data format

`arbitrix-core` accepts a single, strict OHLCV schema. Anything that doesn't
satisfy it is rejected up front by `validate_ohlcv()`, before the backtester
sees a single bar.

## Schema

| Component | Requirement |
|-----------|-------------|
| Index | `pd.DatetimeIndex`, tz-aware, **UTC** |
| Index ordering | monotonic increasing, no duplicates |
| Columns (required) | `open`, `high`, `low`, `close`, `volume` (lowercase, numeric) |
| Columns (optional) | `spread` (numeric, in points) |

Any extra columns survive untouched — they're available to your strategy in
`prepare()` and `on_bar()`. Useful for indicators you've precomputed offline.

## Loading from disk

`load_ohlcv()` handles CSV and parquet:

```python
from arbitrix_core import load_ohlcv

df = load_ohlcv("eurusd_h1.csv", time_col="datetime")
# or
df = load_ohlcv("eurusd_h1.parquet")  # parquet detected by suffix
```

Behaviour:

- File suffix `.parquet` → `pd.read_parquet`; everything else → `pd.read_csv`.
- If the loaded frame has no `DatetimeIndex`, the column named by `time_col`
  (default `"time"`) is parsed with `pd.to_datetime(..., utc=True)` and set as
  the index.
- If the index already has `tz=None`, it's localised to UTC. Otherwise it's
  converted to UTC.
- Rows are sorted by index (mergesort, stable). Duplicate timestamps are
  collapsed keeping the last row.
- The frame is then passed through `validate_ohlcv()` before being returned.

## Validating an in-memory DataFrame

If you've built the DataFrame yourself, validate it explicitly:

```python
from arbitrix_core import validate_ohlcv

validate_ohlcv(df)  # raises ValueError on schema problems
```

## Common errors and fixes

| Error message | Cause | Fix |
|---------------|-------|-----|
| `DataFrame index must be a DatetimeIndex` | Index is `RangeIndex` / int / object | `df = df.set_index(pd.to_datetime(df["time"], utc=True))` |
| `DataFrame index must be tz-aware UTC` | Naive timestamps or non-UTC tz | `df.index = df.index.tz_localize("UTC")` or `tz_convert("UTC")` |
| `DataFrame index must be monotonic increasing` | Rows out of order | `df = df.sort_index(kind="mergesort")` |
| `DataFrame index has duplicates` | Repeated timestamps | `df = df[~df.index.duplicated(keep="last")]` |
| `DataFrame is missing required column(s): [...]` | Schema mismatch | Rename columns to lowercase `open/high/low/close/volume` |

## DataProvider (advanced)

`arbitrix_core.DataProvider` is a `runtime_checkable` `Protocol` with one
method:

```python
def get_symbol_info(self, symbol: str) -> dict | None: ...
```

Open-core never instantiates one — it exists so that closed Arbitrix can inject
a live broker symbol-info source (point value, contract size, swap rates) into
the cost model. As an open-core user you typically don't need it; pass
`point_overrides={"EURUSD": 10.0}` to `costs.configure()` instead.
