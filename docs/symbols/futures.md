# Futures support — foundations

Futures instruments differ from CFDs/stocks in three first-class ways:

1. **Multiplier**: a futures bar quotes an index *point*; one contract = `price × multiplier` notional. MES = `$5`, ES = `$50`.
2. **Integer-only sizing**: contracts are indivisible. Engines floor for FUT and skip when the floored size is below `min_order_size`.
3. **Asset class**: `InstrumentConfig.asset_class` auto-classifies from `security_type` (`FUT` → `futures`, `CONTFUT` → `futures_continuous`), overridable per symbol.

## SymbolContext

Per-symbol metadata is read from `SymbolContext` (immutable dataclass) via `arbitrix_core.symbols.context.get_symbol_context(symbol)`. Every consumer — backtest, live runtime, costs, portfolio, microstructure — reads from this single surface.

::: arbitrix_core.symbols.context.SymbolContext

## Auto-classification

::: arbitrix_core.symbols.asset_class.classify_asset_class

## Commission scheme

`InstrumentConfig.commission_scheme="per_contract"` together with `fee_per_contract` activates the per-contract path in `arbitrix_core.costs.base._resolve_commission_scheme`. See [Cost models](../costs.md) for the full taxonomy.
