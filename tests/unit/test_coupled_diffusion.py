"""
tests/unit/test_coupled_diffusion.py
========================================
Unit tests for ``nominal_drift.science.coupled_diffusion``.

All tests use small synthetic DiffusionOutput objects — no real solver runs.
The test coverage verifies:

  - Cr-only case: effective Cr == base Cr (no coupling applied)
  - Cr + C: effective Cr < base Cr; contribution is positive
  - Cr + N: effective Cr < base Cr; contribution is positive
  - Cr + C + N: both contributions applied, additive
  - Sink floor clamping: effective Cr never goes below c_sink
  - lambda_c = 0: C output has no effect on result
  - lambda_n = 0: N output has no effect on result
  - Custom lambda changes result proportionally
  - Contributions are always >= 0
  - cr_min_effective <= cr_min_base when C/N present
  - cr_sink_floor echoes cr_output.C_sink_wt_pct
  - Mechanism components reflect active species
  - lambda_c_used / lambda_n_used echoed in result
  - species_inputs reflects provided outputs
  - Assumptions always present and non-empty
  - Warnings propagated from cr_output, c_output, n_output
  - Unusually high lambda triggers a warning
  - Saturation at sink floor triggers a warning
  - Large coupling margin triggers a warning
  - Result is frozen (immutable)
  - Result is deterministic (same inputs → same output)
  - Result is a CoupledDiffusionResult instance
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.science.coupled_diffusion import (
    LAMBDA_C,
    LAMBDA_N,
    CoupledDiffusionResult,
    evaluate_coupled_depletion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output(
    element: str = "Cr",
    c_bulk: float = 16.5,
    c_sink: float = 12.0,
    min_conc: float = 11.0,
    depth: float | None = 60.0,
    solver_warnings: list[str] | None = None,
) -> DiffusionOutput:
    """Return a minimal, valid DiffusionOutput for testing."""
    import numpy as np

    n_x, n_t = 10, 2
    x_m = np.linspace(0.0, 5e-6, n_x).tolist()
    t_s = [0.0, 3600.0]

    row0 = [c_bulk] * n_x
    row1 = list(row0)
    row1[0] = min_conc
    profiles = [row0, row1]

    return DiffusionOutput(
        element=element,
        matrix="austenite_FeCrNi",
        x_m=x_m,
        t_s=t_s,
        concentration_profiles=profiles,
        C_bulk_wt_pct=c_bulk,
        C_sink_wt_pct=c_sink,
        min_concentration_wt_pct=min_conc,
        depletion_depth_nm=depth,
        warnings=solver_warnings or [],
        metadata={},
    )


@pytest.fixture(scope="module")
def cr_out():
    return _make_output("Cr", c_bulk=16.5, c_sink=12.0, min_conc=11.0, depth=60.0)


@pytest.fixture(scope="module")
def c_out():
    return _make_output("C", c_bulk=0.04, c_sink=0.001, min_conc=0.001, depth=None)


@pytest.fixture(scope="module")
def n_out():
    return _make_output("N", c_bulk=0.07, c_sink=0.001, min_conc=0.001, depth=None)


# ===========================================================================
# TestCrOnly
# ===========================================================================

class TestCrOnly:

    def test_cr_only_effective_equals_base(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert result.cr_min_effective_wt_pct == pytest.approx(result.cr_min_base_wt_pct)

    def test_cr_only_contributions_are_zero(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert result.c_contribution_wt_pct == pytest.approx(0.0)
        assert result.n_contribution_wt_pct == pytest.approx(0.0)

    def test_cr_only_mechanism_has_one_component(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert len(result.mechanism_components) == 1
        assert "Dirichlet" in result.mechanism_components[0]

    def test_cr_only_species_inputs_is_cr(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert result.species_inputs == ["Cr"]

    def test_cr_only_base_echoes_c_sink(self, cr_out):
        # cr_min_base is derived from C_sink_wt_pct (the Dirichlet BC),
        # not from min_concentration_wt_pct.
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert result.cr_min_base_wt_pct == pytest.approx(cr_out.C_sink_wt_pct)

    def test_sink_floor_is_zero(self, cr_out):
        # cr_sink_floor is the absolute physical floor (0.0 wt% Cr),
        # not C_sink_wt_pct.
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert result.cr_sink_floor_wt_pct == pytest.approx(0.0)


# ===========================================================================
# TestCarbonCoupling
# ===========================================================================

class TestCarbonCoupling:

    def test_c_contribution_positive(self, cr_out, c_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        assert result.c_contribution_wt_pct > 0.0

    def test_c_contribution_formula(self, cr_out, c_out):
        # c_contribution = lambda_c * c_out.C_bulk_wt_pct
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        expected = LAMBDA_C * c_out.C_bulk_wt_pct
        assert result.c_contribution_wt_pct == pytest.approx(expected)

    def test_effective_cr_less_than_base_with_c(self, cr_out, c_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        assert result.cr_min_effective_wt_pct <= result.cr_min_base_wt_pct

    def test_n_contribution_zero_when_no_n(self, cr_out, c_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        assert result.n_contribution_wt_pct == pytest.approx(0.0)

    def test_c_mechanism_present(self, cr_out, c_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        assert any("M₂₃C₆" in m or "carbide" in m.lower()
                   for m in result.mechanism_components)

    def test_c_in_species_inputs(self, cr_out, c_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        assert "C" in result.species_inputs

    def test_lambda_c_zero_means_no_c_effect(self, cr_out, c_out):
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, lambda_c=0.0
        )
        assert result.c_contribution_wt_pct == pytest.approx(0.0)
        assert result.cr_min_effective_wt_pct == pytest.approx(result.cr_min_base_wt_pct)

    def test_custom_lambda_c_scales_contribution(self, cr_out, c_out):
        r1 = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out, lambda_c=2.0)
        r2 = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out, lambda_c=4.0)
        assert r2.c_contribution_wt_pct == pytest.approx(2.0 * r1.c_contribution_wt_pct)

    def test_lambda_c_used_echoed(self, cr_out, c_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out, lambda_c=3.7)
        assert result.lambda_c_used == pytest.approx(3.7)


# ===========================================================================
# TestNitrogenCoupling
# ===========================================================================

class TestNitrogenCoupling:

    def test_n_contribution_positive(self, cr_out, n_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_out)
        assert result.n_contribution_wt_pct > 0.0

    def test_n_contribution_formula(self, cr_out, n_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_out)
        expected = LAMBDA_N * n_out.C_bulk_wt_pct
        assert result.n_contribution_wt_pct == pytest.approx(expected)

    def test_effective_cr_less_than_base_with_n(self, cr_out, n_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_out)
        assert result.cr_min_effective_wt_pct <= result.cr_min_base_wt_pct

    def test_c_contribution_zero_when_no_c(self, cr_out, n_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_out)
        assert result.c_contribution_wt_pct == pytest.approx(0.0)

    def test_n_mechanism_present(self, cr_out, n_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_out)
        assert any("Cr₂N" in m or "nitride" in m.lower()
                   for m in result.mechanism_components)

    def test_n_in_species_inputs(self, cr_out, n_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_out)
        assert "N" in result.species_inputs

    def test_lambda_n_zero_means_no_n_effect(self, cr_out, n_out):
        result = evaluate_coupled_depletion(
            cr_output=cr_out, n_output=n_out, lambda_n=0.0
        )
        assert result.n_contribution_wt_pct == pytest.approx(0.0)

    def test_lambda_n_used_echoed(self, cr_out, n_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_out, lambda_n=1.8)
        assert result.lambda_n_used == pytest.approx(1.8)


# ===========================================================================
# TestCombinedCoupling
# ===========================================================================

class TestCombinedCoupling:

    def test_both_c_and_n_reduce_effective_cr(self, cr_out, c_out, n_out):
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, n_output=n_out
        )
        assert result.cr_min_effective_wt_pct < result.cr_min_base_wt_pct

    def test_combined_contribution_additive(self, cr_out, c_out, n_out):
        combined = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, n_output=n_out
        )
        c_only = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        n_only = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_out)
        expected_total = c_only.c_contribution_wt_pct + n_only.n_contribution_wt_pct
        actual_total = combined.c_contribution_wt_pct + combined.n_contribution_wt_pct
        assert actual_total == pytest.approx(expected_total)

    def test_combined_has_three_mechanism_components(self, cr_out, c_out, n_out):
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, n_output=n_out
        )
        assert len(result.mechanism_components) == 3

    def test_all_three_species_in_inputs(self, cr_out, c_out, n_out):
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, n_output=n_out
        )
        assert set(result.species_inputs) == {"Cr", "C", "N"}


# ===========================================================================
# TestSinkFloorClamping
# ===========================================================================

class TestSinkFloorClamping:

    def test_effective_cr_never_negative(self):
        # Very high bulk C (2 wt%) with high lambda_c → raw_effective goes negative.
        # Model must clamp to 0.0 (absolute physical floor).
        cr_out = _make_output("Cr", c_bulk=16.5, c_sink=12.0, min_conc=12.0)
        c_out  = _make_output("C",  c_bulk=2.0,  c_sink=0.001, min_conc=0.001)
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, lambda_c=10.0
        )
        # 12.0 - 10.0×2.0 = −8.0 → clamped to 0.0
        assert result.cr_min_effective_wt_pct >= 0.0

    def test_clamped_effective_is_below_base(self):
        cr_out = _make_output("Cr", c_bulk=16.5, c_sink=12.0, min_conc=12.0)
        c_out  = _make_output("C",  c_bulk=2.0,  c_sink=0.001, min_conc=0.001)
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, lambda_c=10.0
        )
        assert result.cr_min_effective_wt_pct < result.cr_min_base_wt_pct

    def test_saturation_warning_issued_when_raw_goes_negative(self):
        cr_out = _make_output("Cr", c_bulk=16.5, c_sink=12.0, min_conc=12.0)
        c_out  = _make_output("C",  c_bulk=2.0,  c_sink=0.001, min_conc=0.001)
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, lambda_c=10.0
        )
        assert any("negative" in w.lower() or "saturated" in w.lower()
                   for w in result.warnings)


# ===========================================================================
# TestWarnings
# ===========================================================================

class TestWarnings:

    def test_clean_run_has_no_warnings(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert result.warnings == []

    def test_cr_solver_warnings_propagated(self):
        cr_w = _make_output("Cr", solver_warnings=["Domain boundary warning."])
        result = evaluate_coupled_depletion(cr_output=cr_w)
        assert any("Domain boundary warning" in w for w in result.warnings)

    def test_c_solver_warnings_propagated_with_prefix(self):
        cr_out = _make_output("Cr")
        c_warn = _make_output("C", c_bulk=0.04, c_sink=0.001,
                              min_conc=0.001, solver_warnings=["C domain small."])
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_warn)
        assert any("[C]" in w for w in result.warnings)

    def test_n_solver_warnings_propagated_with_prefix(self):
        cr_out = _make_output("Cr")
        n_warn = _make_output("N", c_bulk=0.07, c_sink=0.001,
                              min_conc=0.001, solver_warnings=["N domain small."])
        result = evaluate_coupled_depletion(cr_output=cr_out, n_output=n_warn)
        assert any("[N]" in w for w in result.warnings)

    def test_high_lambda_c_triggers_warning(self):
        cr_out = _make_output("Cr")
        c_out  = _make_output("C", c_bulk=0.04, c_sink=0.001, min_conc=0.001)
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, lambda_c=15.0
        )
        assert any("lambda_c" in w for w in result.warnings)

    def test_high_lambda_n_triggers_warning(self):
        cr_out = _make_output("Cr")
        n_out  = _make_output("N", c_bulk=0.07, c_sink=0.001, min_conc=0.001)
        result = evaluate_coupled_depletion(
            cr_output=cr_out, n_output=n_out, lambda_n=8.0
        )
        assert any("lambda_n" in w for w in result.warnings)


# ===========================================================================
# TestAssumptions
# ===========================================================================

class TestAssumptions:

    def test_assumptions_always_present(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert len(result.assumptions) >= 4

    def test_assumptions_mention_first_order(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        combined = " ".join(result.assumptions).lower()
        assert "first-order" in combined

    def test_assumptions_mention_not_calphad(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        combined = " ".join(result.assumptions).lower()
        assert "calphad" in combined


# ===========================================================================
# TestResultIntegrity
# ===========================================================================

class TestResultIntegrity:

    def test_contributions_always_non_negative(self, cr_out, c_out, n_out):
        result = evaluate_coupled_depletion(
            cr_output=cr_out, c_output=c_out, n_output=n_out
        )
        assert result.c_contribution_wt_pct >= 0.0
        assert result.n_contribution_wt_pct >= 0.0

    def test_effective_cr_at_most_base_cr_when_coupling_present(self, cr_out, c_out):
        result = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        assert result.cr_min_effective_wt_pct <= result.cr_min_base_wt_pct

    def test_result_is_frozen(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        with pytest.raises(ValidationError):
            result.cr_min_effective_wt_pct = 99.0  # type: ignore[misc]

    def test_result_is_deterministic(self, cr_out, c_out):
        r1 = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        r2 = evaluate_coupled_depletion(cr_output=cr_out, c_output=c_out)
        assert r1 == r2

    def test_result_is_coupled_diffusion_result_instance(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert isinstance(result, CoupledDiffusionResult)

    def test_default_lambda_values_echoed(self, cr_out):
        result = evaluate_coupled_depletion(cr_output=cr_out)
        assert result.lambda_c_used == pytest.approx(LAMBDA_C)
        assert result.lambda_n_used == pytest.approx(LAMBDA_N)
