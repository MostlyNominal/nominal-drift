"""
tests/unit/test_mechanism_animator.py
=======================================
Unit tests for ``nominal_drift.viz.mechanism_animator``.

All tests use a tiny DiffusionOutput (20 spatial nodes, 5 stored profiles)
so rendering is fast (< 2 s per animation call).  Animation is rendered with
``max_frames=3, fps=5`` to keep GIF files small.

Test coverage:
  - Animation file is created at the expected path
  - Returned path matches actual saved file
  - File size is non-trivial (> 10 kB for a 3-frame GIF)
  - Works with valid DiffusionOutput (Cr baseline)
  - Multi-species compatibility: N and C elements
  - title argument does not raise or corrupt output
  - Custom scheme (dict) does not raise or corrupt output
  - Custom scheme (MechanismScheme instance) does not raise
  - max_frames subsampling: max_frames < n_stored works
  - max_frames=1 (edge case) works
  - Extension is always normalised to .gif
  - Deterministic file size across two calls with the same input
  - DISCLAIMER module constant is non-empty
  - DISCLAIMER is surfaced through MechanismScheme default
  - No-driving-force case (C_sink == C_bulk) does not raise
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.viz.mechanism_animator import (
    DISCLAIMER,
    MechanismScheme,
    animate_mechanism,
)

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

_FAST_KWARGS: dict = dict(max_frames=3, fps=5)   # keep tests quick
_MIN_GIF_BYTES: int = 10_000                       # sanity floor for a 3-frame GIF


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_output(element: str = "Cr") -> DiffusionOutput:
    """Return a minimal but valid DiffusionOutput for *element*."""
    import numpy as np

    n_x = 20
    n_t = 5
    x_m = np.linspace(0.0, 5e-6, n_x).tolist()
    t_s = np.linspace(0.0, 3600.0, n_t).tolist()

    c_bulk = 16.5
    c_sink = 12.0

    # Profiles: t=0 → uniform; later frames → depletion from x=0
    profiles: list[list[float]] = []
    for i in range(n_t):
        row = [c_bulk] * n_x
        if i > 0:
            row[0] = c_sink
            for j in range(1, min(6, n_x)):
                frac = j / 6.0
                row[j] = c_sink + (c_bulk - c_sink) * frac
        profiles.append(row)

    return DiffusionOutput(
        element=element,
        matrix="austenite_FeCrNi",
        x_m=x_m,
        t_s=t_s,
        concentration_profiles=profiles,
        C_bulk_wt_pct=c_bulk,
        C_sink_wt_pct=c_sink,
        min_concentration_wt_pct=c_sink,
        depletion_depth_nm=50.0,
        warnings=[],
        metadata={},
    )


@pytest.fixture(scope="module")
def cr_output() -> DiffusionOutput:
    """Reusable Cr DiffusionOutput (immutable, module-scoped)."""
    return _make_output("Cr")


# ---------------------------------------------------------------------------
# TestAnimationFileCreation
# ---------------------------------------------------------------------------

class TestAnimationFileCreation:

    def test_file_is_created(self, tmp_path, cr_output):
        dest = tmp_path / "mech.gif"
        result = animate_mechanism(cr_output, dest, **_FAST_KWARGS)
        assert Path(result).exists(), "GIF file was not created"

    def test_returned_path_matches_saved_file(self, tmp_path, cr_output):
        dest = tmp_path / "mech.gif"
        result = animate_mechanism(cr_output, dest, **_FAST_KWARGS)
        assert Path(result).exists()
        assert os.path.abspath(result) == os.path.abspath(dest)

    def test_file_size_non_trivial(self, tmp_path, cr_output):
        dest = tmp_path / "mech.gif"
        animate_mechanism(cr_output, dest, **_FAST_KWARGS)
        size = dest.stat().st_size
        assert size > _MIN_GIF_BYTES, (
            f"GIF is suspiciously small: {size} bytes "
            f"(expected > {_MIN_GIF_BYTES})"
        )

    def test_extension_normalised_to_gif(self, tmp_path, cr_output):
        # Request .mp4 — must be silently re-routed to .gif
        dest = tmp_path / "mech.mp4"
        result = animate_mechanism(cr_output, dest, **_FAST_KWARGS)
        assert result.endswith(".gif"), f"Expected .gif extension, got: {result}"
        assert Path(result).exists()

    def test_returned_path_is_string(self, tmp_path, cr_output):
        dest = tmp_path / "mech.gif"
        result = animate_mechanism(cr_output, dest, **_FAST_KWARGS)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestMultiSpecies
# ---------------------------------------------------------------------------

class TestMultiSpecies:

    def test_nitrogen_element(self, tmp_path):
        out = _make_output("N")
        dest = tmp_path / "mech_N.gif"
        result = animate_mechanism(out, dest, **_FAST_KWARGS)
        assert Path(result).exists()

    def test_carbon_element(self, tmp_path):
        out = _make_output("C")
        dest = tmp_path / "mech_C.gif"
        result = animate_mechanism(out, dest, **_FAST_KWARGS)
        assert Path(result).exists()

    def test_generic_element_label(self, tmp_path):
        # A future element not in the arrhenius database can still be visualised
        out = _make_output("Mo")
        dest = tmp_path / "mech_Mo.gif"
        result = animate_mechanism(out, dest, **_FAST_KWARGS)
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# TestTitleArgument
# ---------------------------------------------------------------------------

class TestTitleArgument:

    def test_title_none_does_not_raise(self, tmp_path, cr_output):
        dest = tmp_path / "mech.gif"
        result = animate_mechanism(cr_output, dest, title=None, **_FAST_KWARGS)
        assert Path(result).exists()

    def test_title_string_does_not_raise(self, tmp_path, cr_output):
        dest = tmp_path / "mech_titled.gif"
        result = animate_mechanism(
            cr_output, dest,
            title="316L Cr sensitization at 700 °C",
            **_FAST_KWARGS,
        )
        assert Path(result).exists()

    def test_title_unicode_does_not_raise(self, tmp_path, cr_output):
        dest = tmp_path / "mech_unicode.gif"
        result = animate_mechanism(
            cr_output, dest,
            title="Cr depletion — Δt = 60 min",
            **_FAST_KWARGS,
        )
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# TestSchemeArgument
# ---------------------------------------------------------------------------

class TestSchemeArgument:

    def test_scheme_none_uses_defaults(self, tmp_path, cr_output):
        dest = tmp_path / "mech_default.gif"
        result = animate_mechanism(cr_output, dest, scheme=None, **_FAST_KWARGS)
        assert Path(result).exists()

    def test_scheme_dict_is_accepted(self, tmp_path, cr_output):
        dest = tmp_path / "mech_dict_scheme.gif"
        custom = {
            "species_colour":  "#7C3AED",   # purple
            "depleted_colour": "#FDE68A",   # amber
            "domain_label":    "Grain matrix",
        }
        result = animate_mechanism(cr_output, dest, scheme=custom, **_FAST_KWARGS)
        assert Path(result).exists()

    def test_scheme_instance_is_accepted(self, tmp_path, cr_output):
        dest = tmp_path / "mech_scheme_obj.gif"
        scheme = MechanismScheme(
            species_colour="#10B981",
            boundary_label="HAZ boundary",
        )
        result = animate_mechanism(cr_output, dest, scheme=scheme, **_FAST_KWARGS)
        assert Path(result).exists()

    def test_scheme_custom_disclaimer(self, tmp_path, cr_output):
        dest = tmp_path / "mech_custom_disc.gif"
        extended = DISCLAIMER + "  Extended for Al alloy context."
        scheme = MechanismScheme(disclaimer=extended)
        result = animate_mechanism(cr_output, dest, scheme=scheme, **_FAST_KWARGS)
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# TestFrameSubsampling
# ---------------------------------------------------------------------------

class TestFrameSubsampling:

    def test_max_frames_less_than_stored(self, tmp_path, cr_output):
        # cr_output has 5 stored profiles; request only 3
        dest = tmp_path / "mech_sub.gif"
        result = animate_mechanism(cr_output, dest, max_frames=3, fps=5)
        assert Path(result).exists()

    def test_max_frames_one_edge_case(self, tmp_path, cr_output):
        dest = tmp_path / "mech_1frame.gif"
        result = animate_mechanism(cr_output, dest, max_frames=1, fps=5)
        assert Path(result).exists()

    def test_max_frames_larger_than_stored(self, tmp_path, cr_output):
        # cr_output has 5 profiles; requesting 200 must not crash
        dest = tmp_path / "mech_over.gif"
        result = animate_mechanism(cr_output, dest, max_frames=200, fps=5)
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_fixed_rng_seed_gives_stable_file_size(self, tmp_path, cr_output):
        """Two calls with identical inputs must produce same-size GIF."""
        dest_a = tmp_path / "mech_a.gif"
        dest_b = tmp_path / "mech_b.gif"
        animate_mechanism(cr_output, dest_a, **_FAST_KWARGS)
        animate_mechanism(cr_output, dest_b, **_FAST_KWARGS)
        assert dest_a.stat().st_size == dest_b.stat().st_size, (
            "Mechanism animation is not deterministic — check RNG seeding"
        )


# ---------------------------------------------------------------------------
# TestNoDrivingForce
# ---------------------------------------------------------------------------

class TestNoDrivingForce:

    def test_c_sink_equals_c_bulk_does_not_raise(self, tmp_path):
        """When C_sink == C_bulk there is no driving force; must not crash."""
        import numpy as np

        n_x, n_t = 10, 3
        x_m = np.linspace(0.0, 1e-6, n_x).tolist()
        t_s = [0.0, 1800.0, 3600.0]
        c_uniform = 16.5
        profiles = [[c_uniform] * n_x for _ in range(n_t)]

        out = DiffusionOutput(
            element="Cr",
            matrix="austenite_FeCrNi",
            x_m=x_m,
            t_s=t_s,
            concentration_profiles=profiles,
            C_bulk_wt_pct=c_uniform,
            C_sink_wt_pct=c_uniform,   # no driving force
            min_concentration_wt_pct=c_uniform,
            depletion_depth_nm=None,
            warnings=["C_sink >= C_bulk; no depletion expected."],
            metadata={},
        )
        dest = tmp_path / "mech_nodrive.gif"
        result = animate_mechanism(out, dest, **_FAST_KWARGS)
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# TestDisclaimerModule
# ---------------------------------------------------------------------------

class TestDisclaimerModule:

    def test_disclaimer_constant_is_non_empty(self):
        assert isinstance(DISCLAIMER, str)
        assert len(DISCLAIMER) > 20

    def test_disclaimer_mentions_continuum(self):
        assert "Continuum" in DISCLAIMER or "continuum" in DISCLAIMER

    def test_disclaimer_mentions_not_atomistic(self):
        assert "atomistic" in DISCLAIMER.lower()

    def test_mechanism_scheme_default_disclaimer_matches_module_constant(self):
        scheme = MechanismScheme()
        assert scheme.disclaimer == DISCLAIMER

    def test_mechanism_scheme_disclaimer_overridable(self):
        custom = "Custom disclaimer for test purposes."
        scheme = MechanismScheme(disclaimer=custom)
        assert scheme.disclaimer == custom
