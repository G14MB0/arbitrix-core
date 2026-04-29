# Changelog

All notable changes to **arbitrix-core** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-29

### Fixed
- Re-release under 0.1.1 due to PyPI filename reuse restriction on 0.1.0.

### Added
- Initial open-source release extracted from the Arbitrix monorepo.
- `Backtester` engine with batch-mode `run_single` API.
- `BaseStrategy` with `generate_signals`, `filter_signal`, `on_bar` hooks.
- Default cost model with commission, spread, slippage, swap components,
  configurable via `costs.configure(...)`.
- User-pluggable cost models via `costs.configure(model_identifier=...)`.
- `load_ohlcv` / `validate_ohlcv` for CSV/Parquet ingestion with strict UTC
  datetime index validation.
- mkdocs-material documentation site published at
  https://g14mb0.github.io/arbitrix-core/.
- Three runnable examples: quickstart, cost overrides, custom cost model.
- CI matrix on Linux/macOS/Windows × Python 3.10/3.11/3.12.

[Unreleased]: https://github.com/G14MB0/arbitrix-core/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/G14MB0/arbitrix-core/releases/tag/v0.1.1
