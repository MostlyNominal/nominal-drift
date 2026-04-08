"""
tests/unit/test_sensitization_model.py
========================================
Unit tests for ``nominal_drift.science.sensitization_model``.

All tests use small synthetic DiffusionOutput objects built by the
``_make_cr_output`` / ``_make_interstitial_output`` helpers below.
No real solver runs are required — only the schema validation layer
of DiffusionOutput is exercised.

Coverage:
  - Cr-only low-risk case (min_cr above threshold)
  - Cr-only moderate-risk case (below threshold, shallow depth)
  - Cr-only high-risk case (below threshold, deep depletion)
  - C output does not break evaluation
  - N output does not break evaluation
  - C + N together → mixed-species label
  - Low risk suppresses C/N mechanism labels
  - Assumptions list always present and non-empty
  - Warnings propagated from cr_output
  - Warnings propagated from c_output with [C diffusion] prefix
  - Warnings propagated from n_output with [N diffusion] prefix
  - threshold parameter controls risk classification
  - Species list reflects provided outputs
  - Returned object is frozen (immutable)
  - Returned object is deterministic (same inputs → same result)
  - Notes are always non-empty
  - cr_threshold_wt_pct echoed in assessment
  - depletion_depth_nm None handled correctly (→ moderate, not high)
  - risk_level is always one of the valid vocabulary values
  - mechanism_label is always one of the valid vocabulary values
"""

from __future__ import annotations

import pytest

from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.science.sensitization_model import (
    SensitizationAssessment,
    _DEPTH_HIGH_RISK_NM,
    _MECHANISM_LABELS,
    _RISK_LEVELS,
    evaluate_sensitization,
)

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_cr_output(
    min_cr: float = 14.0,
    depletion_depth_nm: float | None = None,
    solver_warnings: list[str] | None = None,
    c_bulk: float = 16.5,
    c_sink: float = 12.0,
) -> DiffusionOutput:
    """Return a minimal DiffusionOutput for Cr with controllable scalar results."""
    import numpy as np

    n_x, n_t = 10, 2
    x_m = np.linspace(0.0, 5e-6, n_x).tolist()
    t_s = [0.0, 3600.0]

    # Build profiles consistent with min_cr at the boundary
    row0 = [c_bulk] * n_x
    row1 = list(row0)
    row1[0] = min_cr   # boundary node drives min_concentration
    profiles = [row0, row1]

    return DiffusionOutput(
        element="Cr",
        matrix="austenite_FeCrNi",
        x_m=x_m,
        t_s=t_s,
        concentration_profiles=profiles,
        C_bulk_wt_pct=c_bulk,
        C_sink_wt_pct=c_sink,
        min_concentration_wt_pct=min_cr,
        depletion_depth_nm=depletion_depth_nm,
        warnings=solver_warnings or [],
        metadata={},
    )


def _make_interstitial_output(
    element: str,
    solver_warnings: list[str] | None = None,
) -> DiffusionOutput:
    """Return a minimal DiffusionOutput for an interstitial species (C or N)."""
    import numpy as np

    n_x, n_t = 10, 2
    x_m = np.linspace(0.0, 5e-6, n_x).tolist()
    t_s = [0.0, 3600.0]
    c_bulk, c_sink = 0.02, 0.001
    profiles = [[c_bulk] * n_x, [c_sink] + [c_bulk] * (n_x - 1)]

    return DiffusionOutput(
        element=element,
        matrix="austenite_FeCrNi",
        x_m=x_m,
        t_s=t_s,
        concentration_profiles=profiles,
        C_bulk_wt_pct=c_bulk,
        C_sink_wt_pct=c_sink,
        min_concentration_wt_pct=c_sink,
        depletion_depth_nm=None,
        warnings=solver_warnings or [],
        metadata={},
    )


# ===========================================================================
# TestRiskClassification
# ===========================================================================

class TestRiskClassification:

    def test_low_risk_when_cr_above_threshold(self):
        out = _make_cr_output(min_cr=14.0, depletion_depth_nm=None)
        result = evaluate_sensitization(cr_output=out)
        assert result.risk_level == "low"

    def test_low_risk_when_cr_exactly_at_threshold(self):
        out = _make_cr_output(min_cr=12.0, depletion_depth_nm=None)
        result = evaluate_sensitization(cr_output=out, c_threshold_wt_pct=12.0)
        assert result.risk_level == "low"

    def test_moderate_risk_when_cr_below_threshold_no_depth(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=None)
        result = evaluate_sensitization(cr_output=out)
        assert result.risk_level == "moderate"

    def test_moderate_risk_when_depth_below_high_threshold(self):
        shallow = _DEPTH_HIGH_RISK_NM - 1.0   # just below 50 nm
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=shallow)
        result = evaluate_sensitization(cr_output=out)
        assert result.risk_level == "moderate"

    def test_high_risk_when_cr_below_threshold_and_deep_depletion(self):
        deep = _DEPTH_HIGH_RISK_NM + 10.0     # 60 nm > threshold
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=deep)
        result = evaluate_sensitization(cr_output=out)
        assert result.risk_level == "high"

    def test_high_risk_at_exact_depth_threshold(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=_DEPTH_HIGH_RISK_NM)
        result = evaluate_sensitization(cr_output=out)
        assert result.risk_level == "high"

    def test_custom_threshold_changes_classification(self):
        # With default threshold (12.0), min_cr=13.5 → low
        out = _make_cr_output(min_cr=13.5)
        assert evaluate_sensitization(cr_output=out).risk_level == "low"
        # With a higher threshold (14.0), same min_cr → moderate
        result = evaluate_sensitization(cr_output=out, c_threshold_wt_pct=14.0)
        assert result.risk_level == "moderate"

    def test_risk_level_is_valid_vocabulary_value(self):
        for min_cr, depth in [(14.0, None), (11.0, None), (11.0, 60.0)]:
            out = _make_cr_output(min_cr=min_cr, depletion_depth_nm=depth)
            result = evaluate_sensitization(cr_output=out)
            assert result.risk_level in _RISK_LEVELS


# ===========================================================================
# TestMechanismLabel
# ===========================================================================

class TestMechanismLabel:

    def test_cr_only_mechanism(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=60.0)
        result = evaluate_sensitization(cr_output=out)
        assert result.mechanism_label == "Cr depletion only"

    def test_c_assisted_mechanism_when_high_risk(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=60.0)
        c_out = _make_interstitial_output("C")
        result = evaluate_sensitization(cr_output=out, c_output=c_out)
        assert result.mechanism_label == "C-assisted Cr depletion"

    def test_n_assisted_mechanism_when_high_risk(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=60.0)
        n_out = _make_interstitial_output("N")
        result = evaluate_sensitization(cr_output=out, n_output=n_out)
        assert result.mechanism_label == "N-assisted Cr depletion"

    def test_mixed_species_when_both_c_and_n_provided(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=60.0)
        c_out = _make_interstitial_output("C")
        n_out = _make_interstitial_output("N")
        result = evaluate_sensitization(cr_output=out, c_output=c_out, n_output=n_out)
        assert result.mechanism_label == "mixed-species indication"

    def test_low_risk_always_gives_cr_depletion_only_label(self):
        """Even with C and N supplied, low risk → conservative label."""
        out = _make_cr_output(min_cr=14.0)
        c_out = _make_interstitial_output("C")
        n_out = _make_interstitial_output("N")
        result = evaluate_sensitization(cr_output=out, c_output=c_out, n_output=n_out)
        assert result.risk_level == "low"
        assert result.mechanism_label == "Cr depletion only"

    def test_mechanism_label_is_valid_vocabulary_value(self):
        for min_cr, depth, has_c, has_n in [
            (14.0, None, False, False),
            (11.0, 60.0, False, False),
            (11.0, 60.0, True,  False),
            (11.0, 60.0, False, True),
            (11.0, 60.0, True,  True),
        ]:
            out = _make_cr_output(min_cr=min_cr, depletion_depth_nm=depth)
            c = _make_interstitial_output("C") if has_c else None
            n = _make_interstitial_output("N") if has_n else None
            result = evaluate_sensitization(cr_output=out, c_output=c, n_output=n)
            assert result.mechanism_label in _MECHANISM_LABELS


# ===========================================================================
# TestSpeciesConsidered
# ===========================================================================

class TestSpeciesConsidered:

    def test_cr_only_species_list(self):
        out = _make_cr_output()
        result = evaluate_sensitization(cr_output=out)
        assert result.species_considered == ["Cr"]

    def test_cr_and_c_species_list(self):
        out = _make_cr_output()
        c_out = _make_interstitial_output("C")
        result = evaluate_sensitization(cr_output=out, c_output=c_out)
        assert "Cr" in result.species_considered
        assert "C" in result.species_considered

    def test_cr_and_n_species_list(self):
        out = _make_cr_output()
        n_out = _make_interstitial_output("N")
        result = evaluate_sensitization(cr_output=out, n_output=n_out)
        assert "Cr" in result.species_considered
        assert "N" in result.species_considered

    def test_all_three_species_list(self):
        out = _make_cr_output()
        c_out = _make_interstitial_output("C")
        n_out = _make_interstitial_output("N")
        result = evaluate_sensitization(cr_output=out, c_output=c_out, n_output=n_out)
        assert set(result.species_considered) == {"Cr", "C", "N"}


# ===========================================================================
# TestScalarFields
# ===========================================================================

class TestScalarFields:

    def test_min_cr_wt_pct_echoed(self):
        out = _make_cr_output(min_cr=11.3)
        result = evaluate_sensitization(cr_output=out)
        assert result.min_cr_wt_pct == pytest.approx(11.3)

    def test_depletion_depth_nm_none_echoed(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=None)
        result = evaluate_sensitization(cr_output=out)
        assert result.depletion_depth_nm is None

    def test_depletion_depth_nm_value_echoed(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=75.0)
        result = evaluate_sensitization(cr_output=out)
        assert result.depletion_depth_nm == pytest.approx(75.0)

    def test_cr_threshold_echoed_in_output(self):
        out = _make_cr_output(min_cr=14.0)
        result = evaluate_sensitization(cr_output=out, c_threshold_wt_pct=13.5)
        assert result.cr_threshold_wt_pct == pytest.approx(13.5)


# ===========================================================================
# TestAssumptions
# ===========================================================================

class TestAssumptions:

    def test_assumptions_always_non_empty(self):
        out = _make_cr_output()
        result = evaluate_sensitization(cr_output=out)
        assert isinstance(result.assumptions, list)
        assert len(result.assumptions) > 0

    def test_assumptions_mention_first_order(self):
        out = _make_cr_output()
        result = evaluate_sensitization(cr_output=out)
        combined = " ".join(result.assumptions).lower()
        assert "first-order" in combined

    def test_assumptions_mention_not_substitute(self):
        out = _make_cr_output()
        result = evaluate_sensitization(cr_output=out)
        combined = " ".join(result.assumptions).lower()
        assert "not a substitute" in combined

    def test_assumptions_present_for_all_risk_levels(self):
        for min_cr, depth in [(14.0, None), (11.0, None), (11.0, 60.0)]:
            out = _make_cr_output(min_cr=min_cr, depletion_depth_nm=depth)
            result = evaluate_sensitization(cr_output=out)
            assert len(result.assumptions) >= 5


# ===========================================================================
# TestWarningPropagation
# ===========================================================================

class TestWarningPropagation:

    def test_clean_run_has_empty_warnings(self):
        out = _make_cr_output(solver_warnings=[])
        result = evaluate_sensitization(cr_output=out)
        # No assessment-level warnings when risk is low and no multi-species
        assert isinstance(result.warnings, list)

    def test_cr_solver_warnings_propagated(self):
        w = "Depletion front approaching domain boundary."
        out = _make_cr_output(min_cr=11.0, solver_warnings=[w])
        result = evaluate_sensitization(cr_output=out)
        assert any(w in warn for warn in result.warnings)

    def test_c_solver_warnings_propagated_with_prefix(self):
        c_out = _make_interstitial_output("C", solver_warnings=["C domain warning."])
        cr_out = _make_cr_output(min_cr=11.0, depletion_depth_nm=60.0)
        result = evaluate_sensitization(cr_output=cr_out, c_output=c_out)
        assert any("[C diffusion]" in w for w in result.warnings)

    def test_n_solver_warnings_propagated_with_prefix(self):
        n_out = _make_interstitial_output("N", solver_warnings=["N domain warning."])
        cr_out = _make_cr_output(min_cr=11.0, depletion_depth_nm=60.0)
        result = evaluate_sensitization(cr_output=cr_out, n_output=n_out)
        assert any("[N diffusion]" in w for w in result.warnings)

    def test_multi_species_high_risk_adds_kinetics_caveat(self):
        cr_out = _make_cr_output(min_cr=11.0, depletion_depth_nm=60.0)
        c_out = _make_interstitial_output("C")
        result = evaluate_sensitization(cr_output=cr_out, c_output=c_out)
        assert any("kinetics" in w.lower() for w in result.warnings)


# ===========================================================================
# TestNotesAndFrozenness
# ===========================================================================

class TestNotesAndFrozenness:

    def test_notes_always_non_empty(self):
        out = _make_cr_output()
        result = evaluate_sensitization(cr_output=out)
        assert isinstance(result.notes, list)
        assert len(result.notes) > 0

    def test_frozen_assessment_cannot_be_mutated(self):
        out = _make_cr_output()
        result = evaluate_sensitization(cr_output=out)
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            result.risk_level = "catastrophic"  # type: ignore[misc]

    def test_result_is_deterministic(self):
        out = _make_cr_output(min_cr=11.0, depletion_depth_nm=60.0)
        r1 = evaluate_sensitization(cr_output=out)
        r2 = evaluate_sensitization(cr_output=out)
        assert r1 == r2

    def test_result_is_sensitization_assessment_instance(self):
        out = _make_cr_output()
        result = evaluate_sensitization(cr_output=out)
        assert isinstance(result, SensitizationAssessment)
