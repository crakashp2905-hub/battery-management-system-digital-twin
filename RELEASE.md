# Release checklist

The package is publish-ready (`python -m build` produces a valid sdist + wheel).
Publishing to PyPI needs **your** PyPI credentials, so the final upload is a step
you run — everything up to it is prepared here.

## 1. Pre-flight (all already green in CI)

```bash
pip install -e ".[dev]"
pytest                       # 404 passed, 1 skipped
ruff check .                 # clean
pytest --cov=bms --cov-fail-under=85
```

## 2. Bump the version

Edit `__version__` in `bms/__init__.py` (the package version is read from it),
add a `CHANGELOG.md` entry, and commit on a branch → PR → merge.

## 3. Build the artifacts

```bash
pip install build twine
python -m build              # → dist/bms_digital_twin-<version>.tar.gz + .whl
twine check dist/*           # metadata sanity check
```

## 4. (Optional) test on TestPyPI first

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ bms-digital-twin
```

## 5. Publish to PyPI  ← you run this, with your token

```bash
twine upload dist/*          # prompts for your PyPI API token
```

Use a scoped **PyPI API token** (Account → API tokens), not your password.
Never commit the token; pass it via `~/.pypirc` or the `TWINE_PASSWORD` env var.

## 6. Tag the release

```bash
git tag -a v<version> -m "v<version>"
git push origin v<version>
gh release create v<version> --notes-file CHANGELOG.md dist/*
```

## 7. Publish the docs site (optional)

```bash
pip install -e ".[docs]"
mkdocs build --strict        # → ./site
mkdocs gh-deploy             # publishes to GitHub Pages
```

## Notes

- `bms-digital-twin` is the distribution name; the import package is `bms`.
- The core install is lightweight; heavy features are extras
  (`[app]`, `[api]`, `[agent]`, `[onnx]`, `[can]`, `[docs]`, `[dev]`).
- This is research/simulation software — see `docs/safety_case.md` and
  `docs/firmware.md` for the simulation-only vs safety-critical boundary.
