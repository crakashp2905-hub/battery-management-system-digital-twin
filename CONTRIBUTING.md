# Contributing

Thanks for your interest in the BMS Digital Twin.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,app,notebook]"
```

This installs the `bms` package in editable mode plus the test, dashboard, and
notebook dependencies.

## Running the tests

```bash
pytest -q
```

All tests should pass (currently **164**). Please add a test for any bug fix or
new behaviour — the `tests/test_bms.py::TestFixes` class shows the expected
style for regression tests.

## Linting

```bash
ruff check .
```

Ruff runs in CI as an advisory (non-blocking) job; the test matrix is the
gate. Configuration lives in `pyproject.toml`.

## Conventions

- Keep public functions typed and documented with NumPy-style docstrings.
- **Every randomness source must be seedable and reproducible** — thread a
  `seed`/RNG through rather than touching the global `np.random` state.
- Prefer adding tunables to the relevant config dataclass over hard-coding.

## Pull requests

1. Branch from `main`.
2. Keep changes focused and add an entry to `CHANGELOG.md`.
3. Ensure `pytest -q` passes locally before opening the PR.
