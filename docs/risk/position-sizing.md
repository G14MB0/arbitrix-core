# Position sizing

Integer-floor rule for futures
-----------------------------

Futures contracts are indivisible. Three engine call sites compute volume from a risk budget:

- `arbitrix.risk.RiskManager.calc_volume_by_risk`
- `arbitrix.live.runtime.LiveRuntime._resolve_quantity`
- `arbitrix_core.backtest.engine.Backtester._create_order_from_signal`

For symbols with `ctx.asset_class in {"futures", "futures_continuous"}`, the raw volume is floored. If the floored result is below `ctx.min_order_size`, the signal is skipped (returns `0.0` or `None`). Non-FUT call sites keep the legacy `round(volume, 2)` behavior so CFDs / FX / stocks behave identically to before.

`arbitrix.execution.ib.IBExecutor.place_order` enforces a defense-in-depth check: a fractional FUT quantity reaching the executor raises `ValueError` rather than being rounded silently.
