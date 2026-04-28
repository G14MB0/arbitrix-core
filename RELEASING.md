# Releasing arbitrix-core

`arbitrix-core` is published from the Arbitrix monorepo via subtree split. Releases happen on the public repo `G14MB0/arbitrix-core`.

## Cutting a release

1. Make sure the latest changes are synced to `development` on the public repo (the sync workflow does this automatically on push to monorepo `main`/`development`).
2. On the public repo `main` branch, bump `CHANGELOG.md` and add a new section under `[Unreleased]` for the new version.
3. Fast-forward `main` from `development`:
   ```bash
   git checkout main && git merge --ff-only origin/development && git push origin main
   ```
4. Tag and push:
   ```bash
   git tag -a vX.Y.Z -m "arbitrix-core X.Y.Z"
   git push origin vX.Y.Z
   ```
5. The `release.yml` workflow will build and publish to PyPI via OIDC trusted publishing.

## Pre-release dry-run

To dry-run a release on TestPyPI, push a pre-release tag matching `v*a*`, `v*b*`, or `v*rc*`:

```bash
git tag vX.Y.Z-rc1   # or vX.Y.Za1
git push origin vX.Y.Z-rc1
```

The `release-test.yml` workflow publishes to TestPyPI.

## Trusted publisher prerequisites

Both PyPI and TestPyPI must have a *Pending Publisher* registered against this repo before the first release:

- PyPI: https://pypi.org/manage/account/publishing/
  - Owner: `G14MB0`, repo: `arbitrix-core`, workflow: `release.yml`, environment: `pypi`.
- TestPyPI: https://test.pypi.org/manage/account/publishing/
  - Same owner/repo, workflow: `release-test.yml`, environment: `testpypi`.

GitHub repo environments `pypi` and `testpypi` must exist (Settings → Environments). No reviewers required.
