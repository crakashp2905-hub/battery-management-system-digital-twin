"""Tests: HTML health-report generator."""

from __future__ import annotations

import bms


class TestHealthReport:
    def test_report_is_self_contained_html_with_all_sections(self):
        html = bms.build_health_report("nmc", duration_s=600, seed=1)
        assert html.lstrip().startswith("<!doctype html>")
        assert "</html>" in html
        for section in ("Estimator leaderboard", "Safety case",
                        "Optimal charging", "Degradation-mode diagnosis"):
            assert section in html
        # No external assets — CSS is inline, nothing to fetch.
        assert "<style>" in html
        assert "http://" not in html and "src=" not in html

    def test_report_embeds_the_version_and_chemistry(self):
        html = bms.build_health_report("lfp", duration_s=600, seed=2)
        assert bms.__version__ in html
        assert "LFP" in html

    def test_save_report_writes_a_file(self, tmp_path):
        out = tmp_path / "report.html"
        path = bms.save_report(out, chemistry="nmc", duration_s=400, seed=0)
        assert out.exists()
        assert out.read_text(encoding="utf-8").strip().endswith("</html>")
        assert str(out) == str(path)
