"""
tests/unit/test_profile_plotter.py
====================================
TDD tests for nominal_drift.viz.profile_plotter.

Tests verify:
  1. PNG file is created at the requested path.
  2. Returned path string matches the save_path argument.
  3. File size is non-trivial (> 1 kB — a real PNG, not empty).
  4. Function works with a fully valid DiffusionOutput.
  5. `threshold_wt_pct` argument does not raise or break plotting.
  6. `title` argument does not raise or break plotting.
  7. Minimal DiffusionOutput (2 stored time steps) is handled gracefully.
  8. Returned value is a plain str even when save_path is a Path object.

The fixture constructs a realistic DiffusionOutput using synthetic erf-shaped
profiles rather than running the full diffusion engine, so the tests are fast
and isolated from engine behaviour.

Run with:
    pytest tests/unit/test_profile_plotter.py -v
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # force non-interactive backend before any pyplot import

import pytest

from nominal_drift.schemas.diffusion_output import DiffusionOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _erf_profile(
    x_m: list[float],
    t_s: float,
    C_sink: float,
    C_bulk: float,
    D: float = 1.5e-19,
) -> list[float]:
    """Approximate Cr-depletion profile using the analytical erf solution."""
    profile = []
    for x in x_m:
        if t_s <= 0.0 or x == 0.0:
            c = C_sink if x == 0.0 else C_bulk
        else:
            arg = x / (2.0 * math.sqrt(D * t_s))
            c = C_sink + (C_bulk - C_sink) * math.erf(arg)
            c = max(C_sink, min(C_bulk, c))
        profile.append(c)
    # Enforce Dirichlet BCs exactly
    profile[0] = C_sink
    profile[-1] = C_bulk
    return profile


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_output() -> DiffusionOutput:
    """
    Synthetic DiffusionOutput with 10 stored time steps and 60 spatial nodes.

    Uses the analytical erf solution at D = 1.5e-19 m²/s (Cr in austenite
    at 650°C) over a 2-hour hold in a 5 µm domain.  Realistic enough to
    exercise all plotter features without running the full diffusion engine.
    """
    n_x = 60
    n_t = 10
    x_max_m = 5e-6
    C_bulk = 16.5
    C_sink = 12.0
    total_s = 7200.0   # 120 min

    x_m = [x_max_m * i / (n_x - 1) for i in range(n_x)]
    t_vals = [total_s * i / (n_t - 1) for i in range(n_t)]

    profiles = [_erf_profile(x_m, t, C_sink, C_bulk) for t in t_vals]

    return DiffusionOutput(
        element="Cr",
        matrix="austenite_FeCrNi",
        x_m=x_m,
        t_s=t_vals,
        concentration_profiles=profiles,
        C_bulk_wt_pct=C_bulk,
        C_sink_wt_pct=C_sink,
        min_concentration_wt_pct=C_sink,
        depletion_depth_nm=45.0,
        warnings=[],
        metadata={"arrhenius": {"Cr": {"D0": 3.6e-4, "Qd": 272000}},
                  "solver": {"scheme": "crank_nicolson"}},
    )


@pytest.fixture(scope="module")
def minimal_output() -> DiffusionOutput:
    """
    Minimal DiffusionOutput with only 2 stored time steps (t=0 and t=final).
    Tests that the plotter handles the degenerate case gracefully.
    """
    n_x = 20
    x_max_m = 5e-6
    C_bulk = 18.0
    C_sink = 12.0
    total_s = 3600.0

    x_m = [x_max_m * i / (n_x - 1) for i in range(n_x)]
    t_vals = [0.0, total_s]
    profiles = [
        _erf_profile(x_m, 0.0, C_sink, C_bulk),
        _erf_profile(x_m, total_s, C_sink, C_bulk),
    ]

    return DiffusionOutput(
        element="N",
        matrix="austenite_FeCrNi",
        x_m=x_m,
        t_s=t_vals,
        concentration_profiles=profiles,
        C_bulk_wt_pct=C_bulk,
        C_sink_wt_pct=C_sink,
        min_concentration_wt_pct=C_sink,
        depletion_depth_nm=None,
        warnings=[],
        metadata={"arrhenius": {}, "solver": {}},
    )


# ---------------------------------------------------------------------------
# Import helper — deferred so import errors surface as test FAILURES not
# collection errors, giving a clear message if the module is missing.
# ---------------------------------------------------------------------------

def _import_plotter():
    from nominal_drift.viz import profile_plotter
    return profile_plotter


# ---------------------------------------------------------------------------
# 1. PNG file creation
# ---------------------------------------------------------------------------

class TestPngFileCreation:
    """Basic smoke test: does the function produce a file?"""

    def test_png_file_is_created(self, sample_output, tmp_path):
        mod = _import_plotter()
        save_path = tmp_path / "test_profile.png"
        mod.plot_concentration_profile(sample_output, save_path)
        assert save_path.exists(), (
            f"Expected PNG at {save_path} but file was not created."
        )

    def test_file_has_png_extension(self, sample_output, tmp_path):
        mod = _import_plotter()
        save_path = tmp_path / "profile.png"
        mod.plot_concentration_profile(sample_output, save_path)
        assert save_path.suffix.lower() == ".png"


# ---------------------------------------------------------------------------
# 2. Return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    """Function must return the saved path as a plain str."""

    def test_returns_str(self, sample_output, tmp_path):
        mod = _import_plotter()
        result = mod.plot_concentration_profile(
            sample_output, tmp_path / "out.png"
        )
        assert isinstance(result, str), (
            f"Expected str return type, got {type(result).__name__}"
        )

    def test_returned_path_matches_save_path(self, sample_output, tmp_path):
        mod = _import_plotter()
        save_path = tmp_path / "profile.png"
        result = mod.plot_concentration_profile(sample_output, save_path)
        assert Path(result) == save_path, (
            f"Return value '{result}' does not match save_path '{save_path}'"
        )

    def test_returned_path_is_str_when_given_path_object(
        self, sample_output, tmp_path
    ):
        mod = _import_plotter()
        save_path = tmp_path / "path_type.png"   # pathlib.Path
        result = mod.plot_concentration_profile(sample_output, save_path)
        assert isinstance(result, str), (
            "Return value must be str even when save_path is a Path object."
        )

    def test_returned_path_is_str_when_given_str(self, sample_output, tmp_path):
        mod = _import_plotter()
        save_path = str(tmp_path / "str_type.png")   # plain str
        result = mod.plot_concentration_profile(sample_output, save_path)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. File content — non-trivial size
# ---------------------------------------------------------------------------

class TestFileSize:
    """Saved PNG must be a real image, not an empty or near-empty file."""

    MIN_BYTES = 1024   # 1 kB floor; a real matplotlib PNG is typically 50–300 kB

    def test_file_size_is_nontrivial(self, sample_output, tmp_path):
        mod = _import_plotter()
        save_path = tmp_path / "size_check.png"
        mod.plot_concentration_profile(sample_output, save_path)
        size = os.path.getsize(save_path)
        assert size > self.MIN_BYTES, (
            f"PNG file is suspiciously small: {size} bytes.  "
            f"Expected > {self.MIN_BYTES} bytes for a real plot."
        )


# ---------------------------------------------------------------------------
# 4. Correct function signature and basic plot content
# ---------------------------------------------------------------------------

class TestFunctionSignature:
    """Function accepts the documented keyword arguments without raising."""

    def test_default_call_succeeds(self, sample_output, tmp_path):
        """plot_concentration_profile(output, save_path) — minimum call."""
        mod = _import_plotter()
        mod.plot_concentration_profile(sample_output, tmp_path / "default.png")

    def test_with_title(self, sample_output, tmp_path):
        mod = _import_plotter()
        mod.plot_concentration_profile(
            sample_output,
            tmp_path / "titled.png",
            title="316L sensitization at 650°C",
        )

    def test_with_none_title(self, sample_output, tmp_path):
        """title=None must use a sensible auto-generated title, not crash."""
        mod = _import_plotter()
        mod.plot_concentration_profile(
            sample_output,
            tmp_path / "no_title.png",
            title=None,
        )

    def test_with_threshold_line(self, sample_output, tmp_path):
        """threshold_wt_pct draws a horizontal dashed reference line."""
        mod = _import_plotter()
        mod.plot_concentration_profile(
            sample_output,
            tmp_path / "threshold.png",
            threshold_wt_pct=12.0,
        )

    def test_with_none_threshold(self, sample_output, tmp_path):
        """threshold_wt_pct=None must not draw any line and must not crash."""
        mod = _import_plotter()
        mod.plot_concentration_profile(
            sample_output,
            tmp_path / "no_threshold.png",
            threshold_wt_pct=None,
        )

    def test_all_kwargs_combined(self, sample_output, tmp_path):
        """All optional arguments used together must not conflict."""
        mod = _import_plotter()
        mod.plot_concentration_profile(
            sample_output,
            tmp_path / "all_kwargs.png",
            title="Full kwargs test",
            threshold_wt_pct=13.5,
        )


# ---------------------------------------------------------------------------
# 5. Minimal output (2 time steps) — robustness
# ---------------------------------------------------------------------------

class TestMinimalOutput:
    """Plotter must handle a DiffusionOutput with only 2 stored profiles."""

    def test_minimal_output_creates_file(self, minimal_output, tmp_path):
        mod = _import_plotter()
        save_path = tmp_path / "minimal.png"
        mod.plot_concentration_profile(minimal_output, save_path)
        assert save_path.exists()

    def test_minimal_output_file_nontrivial(self, minimal_output, tmp_path):
        mod = _import_plotter()
        save_path = tmp_path / "minimal_size.png"
        mod.plot_concentration_profile(minimal_output, save_path)
        assert os.path.getsize(save_path) > 1024

    def test_minimal_output_with_threshold(self, minimal_output, tmp_path):
        mod = _import_plotter()
        mod.plot_concentration_profile(
            minimal_output,
            tmp_path / "minimal_threshold.png",
            threshold_wt_pct=14.0,
        )

    def test_minimal_output_with_title(self, minimal_output, tmp_path):
        mod = _import_plotter()
        mod.plot_concentration_profile(
            minimal_output,
            tmp_path / "minimal_title.png",
            title="Minimal N profile",
        )


# ---------------------------------------------------------------------------
# 6. Different elements — generic interface
# ---------------------------------------------------------------------------

class TestMultiElementInterface:
    """Plotter must accept any element in DiffusionOutput without hardcoding Cr."""

    @pytest.mark.parametrize("element,C_bulk,C_sink", [
        ("Cr", 16.5,  12.0),
        ("N",  0.07,   0.0),
        ("C",  0.02,   0.0),
    ])
    def test_plot_for_element(self, element, C_bulk, C_sink, tmp_path):
        n_x = 40
        x_max_m = 5e-6
        x_m = [x_max_m * i / (n_x - 1) for i in range(n_x)]
        n_t = 5
        total_s = 3600.0
        t_vals = [total_s * i / (n_t - 1) for i in range(n_t)]
        profiles = [_erf_profile(x_m, t, C_sink, C_bulk) for t in t_vals]

        output = DiffusionOutput(
            element=element,
            matrix="austenite_FeCrNi",
            x_m=x_m,
            t_s=t_vals,
            concentration_profiles=profiles,
            C_bulk_wt_pct=C_bulk,
            C_sink_wt_pct=C_sink,
            min_concentration_wt_pct=C_sink,
            depletion_depth_nm=None,
            warnings=[],
            metadata={"arrhenius": {}, "solver": {}},
        )
        mod = _import_plotter()
        save_path = tmp_path / f"profile_{element}.png"
        result = mod.plot_concentration_profile(output, save_path)
        assert Path(result).exists()
        assert os.path.getsize(save_path) > 1024
