"""Tests: performance benchmark smoke + packaging sanity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import bms

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))


class TestPerfBenchmark:
    def test_benchmark_runs_and_reports_positive_throughput(self):
        import benchmark_perf
        res = benchmark_perf.benchmark(n=150, n_sup=30)   # tiny, fast
        expected = {"ecm_simulate", "estimator_ekf", "estimator_pf",
                    "spm_simulate", "control_core_step", "supervisor_step"}
        assert expected <= set(res)
        assert all(v > 0 for v in res.values())
        # The dependency-light control core is far cheaper than the full supervisor.
        assert res["control_core_step"] > res["supervisor_step"]


class TestPackagingMetadata:
    def _pyproject(self):
        # tomllib is stdlib only on 3.11+; skip these on 3.10 rather than add a dep.
        tomllib = pytest.importorskip("tomllib")
        with open(_ROOT / "pyproject.toml", "rb") as fh:
            return tomllib.load(fh)

    def test_core_metadata_present(self):
        proj = self._pyproject()["project"]
        assert proj["name"] == "bms-digital-twin"
        assert proj["dynamic"] == ["version"]
        assert proj["requires-python"].startswith(">=3.10")
        assert "readme" in proj and "license" in proj

    def test_version_is_pep440_and_matches_package(self):
        proj = self._pyproject()
        # Dynamic version is read from bms.__version__.
        assert proj["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "bms.__version__"
        parts = bms.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)

    def test_optional_extras_declared(self):
        extras = self._pyproject()["project"]["optional-dependencies"]
        for name in ("app", "api", "dev", "docs", "agent", "onnx", "can"):
            assert name in extras, name

    def test_release_checklist_exists(self):
        assert (_ROOT / "RELEASE.md").exists()
        assert (_ROOT / "mkdocs.yml").exists()
