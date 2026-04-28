# Contributing

## Repository layout

This public `arbitrix-core` repo is **a mirror**. The authoritative source lives
in the upstream private Arbitrix monorepo at `arbitrix/src/arbitrix_core/`. A
GitHub Action runs `git subtree split` on every push and force-pushes the
result to `development` here.

That has two consequences:

- **Direct commits to `development` are overwritten on the next sync.** Don't
  push there.
- **Pull requests against this repo can't be merged here directly** — they
  need to land upstream, and the next sync brings them across.

## How to contribute

- **Bug reports and feature requests** — open an issue on
  [G14MB0/arbitrix-core/issues](https://github.com/G14MB0/arbitrix-core/issues).
- **Code patches** — open the issue first; the maintainer will guide whether
  the patch lands here as a backport or upstream first.
- **Documentation fixes** — pull requests welcome. The maintainer will replay
  the change upstream so it survives the next sync.

## Local development

```bash
git clone https://github.com/G14MB0/arbitrix-core
cd arbitrix-core
pip install -e ".[dev]"
pytest
ruff check .
```

To rebuild docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then browse to <http://127.0.0.1:8000>.

## Code style

- Python ≥ 3.10
- `ruff` enforces formatting + lint (line length 110, target Python 3.10)
- Strict OHLCV schema — see [Data format](data-format.md). Validation lives in
  `arbitrix_core.data.loader.validate_ohlcv` and runs before any backtest.
- Cost models are plain Python modules with six functions — see
  [Cost models](costs.md).

## Releasing

Tag-driven via the `release.yml` workflow on `main`. The workflow uses PyPI
trusted publishing (OIDC), so no API tokens are stored in the repo.

```bash
git tag v0.1.0
git push origin v0.1.0
```
