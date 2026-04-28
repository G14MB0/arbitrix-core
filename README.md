# arbitrix-core

MIT-licensed open-source backtest engine and cost model from the Arbitrix trading toolkit.

## Install

```bash
pip install arbitrix-core
```

Optional extras:
- `arbitrix-core[fast]` — enables numba JIT for the SL/TP vectorized loop.
- `arbitrix-core[docs]` — mkdocs build dependencies.
- `arbitrix-core[dev]` — pytest + ruff.

## Quickstart

See full documentation at https://g14mb0.github.io/arbitrix-core/

## Sync to public repo

`arbitrix-core` is published from the upstream Arbitrix monorepo via subtree split. The
private workflow `.github/workflows/arbitrix-core-sync.yml` force-pushes
`arbitrix/src/arbitrix_core` to https://github.com/G14MB0/arbitrix-core
on every push to `main` or `development` that touches the subtree, and
on `workflow_dispatch`.

Requirements:
- Repo secret `PUBLIC_REPO_TOKEN` — fine-grained PAT with `Contents: write`
  scope on `G14MB0/arbitrix-core`.
- Public repo `development` branch is overwritten on every sync. Do not
  commit directly to that branch.
