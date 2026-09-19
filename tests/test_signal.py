"""Tests: signal conditioning (despiking, smoothing) for raw BMS data."""

from __future__ import annotations

import numpy as np

import bms


class TestHampel:
    def test_removes_spikes_and_flags_them(self):
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 6, 300)) + rng.normal(0, 0.01, 300)
        x[50] += 5.0; x[120] -= 4.0; x[200] += 3.0          # inject 3 spikes
        out, mask = bms.hampel(x, window=7, n_sigma=3.0)
        assert mask[[50, 120, 200]].all()                   # all spikes flagged
        assert abs(out[50]) < 1.5 and abs(out[120]) < 1.5   # and repaired
        assert mask.sum() < 15                              # clean points not over-flagged

    def test_preserves_a_clean_signal(self):
        x = np.linspace(0, 1, 200)
        out, mask = bms.hampel(x, window=7)
        assert mask.sum() == 0 and np.allclose(out, x)


class TestSmoothers:
    def test_savgol_reduces_noise_but_keeps_shape(self):
        t = np.linspace(0, 1, 400)
        clean = np.sin(2 * np.pi * t)
        noisy = clean + np.random.default_rng(1).normal(0, 0.1, t.size)
        sm = bms.savitzky_golay(noisy, window=21, poly=2)
        assert np.sqrt(np.mean((sm - clean) ** 2)) < np.sqrt(np.mean((noisy - clean) ** 2))

    def test_median_filter_kills_impulses(self):
        x = np.zeros(100); x[40] = 10.0
        out = bms.median_filter(x, window=5)
        assert out[40] == 0.0

    def test_ewma_and_moving_average_smooth(self):
        x = np.random.default_rng(2).normal(0, 1, 300)
        assert np.std(bms.ewma(x, alpha=0.1)) < np.std(x)
        assert np.std(bms.moving_average(x, window=9)) < np.std(x)


class TestCleanPipeline:
    def test_clean_signal_despikes_and_reports(self):
        rng = np.random.default_rng(3)
        x = np.cos(np.linspace(0, 4, 250)) + rng.normal(0, 0.01, 250)
        x[80] += 6.0; x[160] -= 5.0
        res = bms.clean_signal(x)
        assert res.n_outliers >= 2
        assert abs(res.signal[80]) < 1.5 and abs(res.signal[160]) < 1.5
        assert "n_outliers" in res.to_dict()
