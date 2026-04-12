"""
tests/unit/test_schemas.py
==========================
Unit tests for nominal_drift.schemas.composition and
nominal_drift.schemas.ht_schedule.

All tests are fully isolated — no filesystem, database, LLM, or network
access.  pytest-mock is not required here; Pydantic validation errors are
raised synchronously during model construction.

Test naming convention:
  test_<model>_<scenario>  where scenario describes what is expected.

Run with:
    pytest tests/unit/test_schemas.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nominal_drift.schemas.composition import (
    COMPOSITION_SUM_TOLERANCE_WTP,
    AlloyComposition,
)
from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep


# =============================================================================
# Fixtures — reusable valid data
# =============================================================================

@pytest.fixture()
def valid_316L_composition() -> dict:
    """Nominal 316L composition that satisfies all validation rules."""
    return dict(
        alloy_designation="316L",
        alloy_matrix="austenite",
        composition_wt_pct={
            "Fe": 65.88,
            "Cr": 16.50,
            "Ni": 10.50,
            "Mo": 2.10,
            "Mn": 1.80,
            "Si": 0.50,
            "C":  0.02,
            "N":  0.07,
            "P":  0.03,
            "S":  0.003,
            # sum ≈ 97.433 — within ±2 wt% tolerance
            # (P + S + trace elements excluded from this fixture intentionally
            #  to exercise the tolerance window; adjust Fe to reach ≈100)
        },
        elemental_analysis_method="EDS",
    )


# Adjust Fe so the sum is ~100 wt%
_316L_COMP = {
    "Fe": 65.88 + (100.0 - 97.433),  # pad Fe to reach 100
    "Cr": 16.50,
    "Ni": 10.50,
    "Mo": 2.10,
    "Mn": 1.80,
    "Si": 0.50,
    "C":  0.02,
    "N":  0.07,
    "P":  0.03,
    "S":  0.003,
}

@pytest.fixture()
def valid_316L_comp_dict() -> dict[str, float]:
    return _316L_COMP.copy()


@pytest.fixture()
def valid_single_step() -> HTStep:
    return HTStep(
        step=1,
        type="sensitization_soak",
        T_hold_C=650.0,
        hold_min=120.0,
        cooling_method="air_cool",
        atmosphere="air",
    )


@pytest.fixture()
def valid_two_step_schedule(valid_single_step: HTStep) -> HTSchedule:
    solution_anneal = HTStep(
        step=1,
        type="solution_anneal",
        T_hold_C=1080.0,
        hold_min=30.0,
        ramp_rate_C_min=10.0,
        cooling_method="water_quench",
        atmosphere="argon",
    )
    sensitization = HTStep(
        step=2,
        type="sensitization_soak",
        T_hold_C=650.0,
        hold_min=120.0,
        cooling_method="air_cool",
    )
    return HTSchedule(steps=[solution_anneal, sensitization])


# =============================================================================
# AlloyComposition — valid cases
# =============================================================================

class TestAlloyCompositionValid:

    def test_valid_316L_minimal(self, valid_316L_comp_dict: dict[str, float]) -> None:
        """Standard 316L composition with mandatory fields only."""
        comp = AlloyComposition(
            alloy_designation="316L",
            alloy_matrix="austenite",
            composition_wt_pct=valid_316L_comp_dict,
        )
        assert comp.alloy_designation == "316L"
        assert comp.alloy_matrix == "austenite"
        assert comp.bulk_Cr_wt_pct == pytest.approx(16.50)
        assert comp.elemental_analysis_method is None
        assert comp.uncertainty_wt_pct is None

    def test_valid_composition_with_nitrogen(
        self, valid_316L_comp_dict: dict[str, float]
    ) -> None:
        """
        Nitrogen must be accepted as an optional element.
        316LN and high-N duplex grades include N as a deliberate alloying
        element and it must flow through to the diffusion engine (N-driven
        sensitization pathway).
        """
        comp = AlloyComposition(
            alloy_designation="316LN",
            alloy_matrix="austenite",
            composition_wt_pct=valid_316L_comp_dict,  # already contains "N"
        )
        assert comp.has_nitrogen is True
        assert comp.composition_wt_pct["N"] == pytest.approx(0.07)

    def test_valid_composition_without_carbon_or_nitrogen(self) -> None:
        """
        Carbon and nitrogen are optional.  A composition with only Fe and Cr
        (and balance elements summing to ~100) must be accepted.
        """
        comp = AlloyComposition(
            alloy_designation="FeCr-binary",
            alloy_matrix="ferrite",
            composition_wt_pct={"Fe": 83.5, "Cr": 16.5},
        )
        assert comp.has_carbon is False
        assert comp.has_nitrogen is False

    def test_valid_composition_with_full_optional_fields(
        self, valid_316L_comp_dict: dict[str, float]
    ) -> None:
        """All optional fields accepted when valid."""
        comp = AlloyComposition(
            alloy_designation="316L",
            alloy_matrix="austenite",
            composition_wt_pct=valid_316L_comp_dict,
            elemental_analysis_method="EPMA",
            uncertainty_wt_pct={"Cr": 0.30, "Ni": 0.25, "Mo": 0.10},
        )
        assert comp.elemental_analysis_method == "EPMA"
        assert comp.uncertainty_wt_pct["Cr"] == pytest.approx(0.30)

    def test_valid_duplex_matrix(self) -> None:
        """Duplex alloy_matrix is accepted.

        2205 nominal composition (wt%):
          Fe 66.5, Cr 22.0, Ni 5.5, Mo 3.0, N 0.17, Mn 2.0, Si 0.80, C 0.03
          sum = 100.0 wt%
        """
        comp = AlloyComposition(
            alloy_designation="2205",
            alloy_matrix="duplex",
            composition_wt_pct={
                "Fe": 66.5, "Cr": 22.0, "Ni": 5.5,
                "Mo": 3.0,  "N":  0.17, "Mn": 2.0,
                "Si": 0.80, "C":  0.03,
            },
        )
        assert comp.alloy_matrix == "duplex"

    def test_valid_unknown_matrix(self) -> None:
        """'unknown' matrix is accepted for partially characterised specimens."""
        comp = AlloyComposition(
            alloy_designation="UNKNOWN-SAMPLE-01",
            alloy_matrix="unknown",
            composition_wt_pct={"Fe": 70.0, "Cr": 18.0, "Ni": 12.0},
        )
        assert comp.alloy_matrix == "unknown"

    def test_composition_sum_at_upper_tolerance_boundary(self) -> None:
        """Sum of exactly 100 + tolerance is accepted."""
        hi = 100.0 + COMPOSITION_SUM_TOLERANCE_WTP
        comp = AlloyComposition(
            alloy_designation="BOUNDARY",
            alloy_matrix="ferrite",
            composition_wt_pct={"Fe": hi - 16.5, "Cr": 16.5},
        )
        assert comp.composition_sum == pytest.approx(hi)

    def test_composition_sum_at_lower_tolerance_boundary(self) -> None:
        """Sum of exactly 100 - tolerance is accepted."""
        lo = 100.0 - COMPOSITION_SUM_TOLERANCE_WTP
        comp = AlloyComposition(
            alloy_designation="BOUNDARY",
            alloy_matrix="ferrite",
            composition_wt_pct={"Fe": lo - 16.5, "Cr": 16.5},
        )
        assert comp.composition_sum == pytest.approx(lo)

    def test_convenience_property_T_hold_K_not_on_composition(
        self, valid_316L_comp_dict: dict[str, float]
    ) -> None:
        """bulk_Cr_wt_pct convenience property returns correct value."""
        comp = AlloyComposition(
            alloy_designation="316L",
            alloy_matrix="austenite",
            composition_wt_pct=valid_316L_comp_dict,
        )
        assert comp.bulk_Cr_wt_pct == pytest.approx(
            valid_316L_comp_dict["Cr"]
        )


# =============================================================================
# AlloyComposition — invalid cases
# =============================================================================

class TestAlloyCompositionInvalid:

    def test_missing_Fe_raises(self) -> None:
        """
        Iron (Fe) is a mandatory element for Track 1 metallurgy workflows.
        A composition without Fe must be rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            AlloyComposition(
                alloy_designation="CrNi-binary",
                alloy_matrix="austenite",
                composition_wt_pct={"Cr": 20.0, "Ni": 80.0},
            )
        errors = exc_info.value.errors()
        assert any("Fe" in str(e["msg"]) for e in errors), (
            "ValidationError should mention missing 'Fe'"
        )

    def test_missing_Cr_raises(self) -> None:
        """
        Chromium (Cr) is mandatory for depletion and sensitization
        calculations.  A composition without Cr must be rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            AlloyComposition(
                alloy_designation="FeMn-alloy",
                alloy_matrix="ferrite",
                composition_wt_pct={"Fe": 95.0, "Mn": 5.0},
            )
        errors = exc_info.value.errors()
        assert any("Cr" in str(e["msg"]) for e in errors), (
            "ValidationError should mention missing 'Cr'"
        )

    def test_negative_composition_value_raises(self) -> None:
        """Negative wt% is physically impossible and must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AlloyComposition(
                alloy_designation="BAD",
                alloy_matrix="austenite",
                composition_wt_pct={"Fe": 70.0, "Cr": 16.5, "Ni": -5.0},
            )
        errors = exc_info.value.errors()
        assert any("non-negative" in str(e["msg"]) or "Ni" in str(e["msg"])
                   for e in errors)

    def test_composition_sum_too_low_raises(self) -> None:
        """
        A composition summing to 50 wt% is clearly erroneous (missing
        elements) and must be rejected regardless of tolerance.
        """
        with pytest.raises(ValidationError) as exc_info:
            AlloyComposition(
                alloy_designation="INCOMPLETE",
                alloy_matrix="austenite",
                composition_wt_pct={"Fe": 33.5, "Cr": 16.5},  # sum = 50
            )
        errors = exc_info.value.errors()
        assert any(
            "50" in str(e["msg"]) or "outside" in str(e["msg"])
            for e in errors
        )

    def test_composition_sum_too_high_raises(self) -> None:
        """A composition summing to 150 wt% is clearly erroneous."""
        with pytest.raises(ValidationError) as exc_info:
            AlloyComposition(
                alloy_designation="INFLATED",
                alloy_matrix="ferrite",
                composition_wt_pct={"Fe": 133.5, "Cr": 16.5},  # sum = 150
            )
        errors = exc_info.value.errors()
        assert any(
            "150" in str(e["msg"]) or "outside" in str(e["msg"])
            for e in errors
        )

    def test_empty_composition_raises(self) -> None:
        """An empty composition dict must be rejected."""
        with pytest.raises(ValidationError):
            AlloyComposition(
                alloy_designation="EMPTY",
                alloy_matrix="austenite",
                composition_wt_pct={},
            )

    def test_invalid_alloy_matrix_raises(self) -> None:
        """An unrecognised alloy_matrix value must be rejected."""
        with pytest.raises(ValidationError):
            AlloyComposition(
                alloy_designation="316L",
                alloy_matrix="amorphous",  # not in AlloyMatrix Literal
                composition_wt_pct={"Fe": 83.5, "Cr": 16.5},
            )

    def test_uncertainty_keys_not_subset_raises(
        self, valid_316L_comp_dict: dict[str, float]
    ) -> None:
        """
        Uncertainty may only be specified for elements that appear in
        composition_wt_pct.  An extra key must be rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            AlloyComposition(
                alloy_designation="316L",
                alloy_matrix="austenite",
                composition_wt_pct=valid_316L_comp_dict,
                uncertainty_wt_pct={"Cr": 0.30, "W": 0.05},  # W not in comp
            )
        errors = exc_info.value.errors()
        assert any("W" in str(e["msg"]) for e in errors)

    def test_negative_uncertainty_raises(
        self, valid_316L_comp_dict: dict[str, float]
    ) -> None:
        """Negative uncertainty is physically meaningless and must be rejected."""
        with pytest.raises(ValidationError):
            AlloyComposition(
                alloy_designation="316L",
                alloy_matrix="austenite",
                composition_wt_pct=valid_316L_comp_dict,
                uncertainty_wt_pct={"Cr": -0.10},
            )

    def test_blank_alloy_designation_raises(
        self, valid_316L_comp_dict: dict[str, float]
    ) -> None:
        """A blank or whitespace-only alloy_designation must be rejected."""
        with pytest.raises(ValidationError):
            AlloyComposition(
                alloy_designation="   ",
                alloy_matrix="austenite",
                composition_wt_pct=valid_316L_comp_dict,
            )


# =============================================================================
# HTStep — valid cases
# =============================================================================

class TestHTStepValid:

    def test_minimal_step(self) -> None:
        """A step with only mandatory fields is valid."""
        step = HTStep(step=1, type="solution_anneal", T_hold_C=1080.0, hold_min=30.0)
        assert step.step == 1
        assert step.T_hold_C == pytest.approx(1080.0)
        assert step.hold_min == pytest.approx(30.0)
        assert step.ramp_rate_C_min is None
        assert step.cooling_method is None
        assert step.atmosphere is None

    def test_full_step(self) -> None:
        """A step with all optional fields is valid."""
        step = HTStep(
            step=2,
            type="sensitization_soak",
            T_hold_C=650.0,
            hold_min=120.0,
            ramp_rate_C_min=5.0,
            cooling_method="air_cool",
            atmosphere="air",
        )
        assert step.ramp_rate_C_min == pytest.approx(5.0)
        assert step.cooling_method == "air_cool"

    def test_T_hold_K_convenience_property(self) -> None:
        """T_hold_K should equal T_hold_C + 273.15."""
        step = HTStep(step=1, type="test", T_hold_C=700.0, hold_min=60.0)
        assert step.T_hold_K == pytest.approx(973.15)

    def test_hold_s_convenience_property(self) -> None:
        """hold_s should equal hold_min × 60."""
        step = HTStep(step=1, type="test", T_hold_C=700.0, hold_min=90.0)
        assert step.hold_s == pytest.approx(5400.0)


# =============================================================================
# HTStep — invalid cases
# =============================================================================

class TestHTStepInvalid:

    def test_negative_temperature_raises(self) -> None:
        """A negative hold temperature is unphysical and must be rejected."""
        with pytest.raises(ValidationError):
            HTStep(step=1, type="test", T_hold_C=-100.0, hold_min=30.0)

    def test_zero_temperature_raises(self) -> None:
        """0°C is not a valid isothermal hold temperature (must be > 0)."""
        with pytest.raises(ValidationError):
            HTStep(step=1, type="test", T_hold_C=0.0, hold_min=30.0)

    def test_zero_hold_time_raises(self) -> None:
        """A zero-duration hold has no physical meaning and must be rejected."""
        with pytest.raises(ValidationError):
            HTStep(step=1, type="test", T_hold_C=650.0, hold_min=0.0)

    def test_negative_hold_time_raises(self) -> None:
        """Negative hold times are unphysical."""
        with pytest.raises(ValidationError):
            HTStep(step=1, type="test", T_hold_C=650.0, hold_min=-10.0)

    def test_zero_ramp_rate_raises(self) -> None:
        """A ramp rate of 0 °C/min is meaningless (infinite time to reach T)."""
        with pytest.raises(ValidationError):
            HTStep(
                step=1, type="test", T_hold_C=650.0, hold_min=60.0,
                ramp_rate_C_min=0.0,
            )

    def test_negative_ramp_rate_raises(self) -> None:
        """A negative ramp rate is unphysical."""
        with pytest.raises(ValidationError):
            HTStep(
                step=1, type="test", T_hold_C=650.0, hold_min=60.0,
                ramp_rate_C_min=-5.0,
            )

    def test_non_positive_step_number_raises(self) -> None:
        """Step numbers must be positive integers."""
        with pytest.raises(ValidationError):
            HTStep(step=0, type="test", T_hold_C=650.0, hold_min=60.0)

    def test_negative_step_number_raises(self) -> None:
        with pytest.raises(ValidationError):
            HTStep(step=-1, type="test", T_hold_C=650.0, hold_min=60.0)

    def test_blank_type_raises(self) -> None:
        """A blank or whitespace-only type label must be rejected."""
        with pytest.raises(ValidationError):
            HTStep(step=1, type="  ", T_hold_C=650.0, hold_min=60.0)


# =============================================================================
# HTSchedule — valid cases
# =============================================================================

class TestHTScheduleValid:

    def test_single_step_schedule(self, valid_single_step: HTStep) -> None:
        """A schedule with exactly one step is valid."""
        schedule = HTSchedule(steps=[valid_single_step])
        assert schedule.n_steps == 1
        assert schedule.total_hold_min == pytest.approx(120.0)
        assert schedule.total_hold_s == pytest.approx(7200.0)

    def test_two_step_schedule(self, valid_two_step_schedule: HTSchedule) -> None:
        """A two-step schedule with ascending step numbers is valid."""
        schedule = valid_two_step_schedule
        assert schedule.n_steps == 2
        assert schedule.T_min_C == pytest.approx(650.0)
        assert schedule.T_max_C == pytest.approx(1080.0)
        assert schedule.total_hold_min == pytest.approx(150.0)  # 30 + 120

    def test_non_contiguous_step_numbers_accepted(self) -> None:
        """
        Step numbers 1, 3, 7 are non-contiguous but strictly ascending —
        they must be accepted.  Gaps may represent omitted sub-steps.
        """
        steps = [
            HTStep(step=1, type="solution_anneal",    T_hold_C=1080.0, hold_min=30.0),
            HTStep(step=3, type="intermediate_hold",  T_hold_C=800.0,  hold_min=10.0),
            HTStep(step=7, type="sensitization_soak", T_hold_C=650.0,  hold_min=120.0),
        ]
        schedule = HTSchedule(steps=steps)
        assert schedule.n_steps == 3

    def test_convenience_properties(
        self, valid_two_step_schedule: HTSchedule
    ) -> None:
        """Verify all HTSchedule convenience properties."""
        s = valid_two_step_schedule
        assert s.n_steps == 2
        assert s.T_min_C == pytest.approx(650.0)
        assert s.T_max_C == pytest.approx(1080.0)
        assert s.total_hold_s == pytest.approx(s.total_hold_min * 60.0)


# =============================================================================
# HTSchedule — invalid cases
# =============================================================================

class TestHTScheduleInvalid:

    def test_empty_step_list_raises(self) -> None:
        """
        An empty step list provides no thermal history to simulate and
        must be rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            HTSchedule(steps=[])
        errors = exc_info.value.errors()
        # Pydantic raises a min_length error on the list OR our model_validator
        # raises — either way ValidationError must be raised.
        assert exc_info.value is not None

    def test_duplicate_step_numbers_raises(self) -> None:
        """
        Two steps with the same step number create an ambiguous schedule
        and must be rejected.
        """
        steps = [
            HTStep(step=1, type="solution_anneal",    T_hold_C=1080.0, hold_min=30.0),
            HTStep(step=1, type="sensitization_soak", T_hold_C=650.0,  hold_min=120.0),
        ]
        with pytest.raises(ValidationError) as exc_info:
            HTSchedule(steps=steps)
        errors = exc_info.value.errors()
        assert any("duplicate" in str(e["msg"]).lower() for e in errors)

    def test_non_ascending_step_order_raises(self) -> None:
        """
        Step numbers that are not strictly ascending indicate an out-of-order
        schedule and must be rejected.
        """
        steps = [
            HTStep(step=2, type="sensitization_soak", T_hold_C=650.0,  hold_min=120.0),
            HTStep(step=1, type="solution_anneal",    T_hold_C=1080.0, hold_min=30.0),
        ]
        with pytest.raises(ValidationError) as exc_info:
            HTSchedule(steps=steps)
        errors = exc_info.value.errors()
        assert any("ascending" in str(e["msg"]).lower() for e in errors)

    def test_equal_step_numbers_treated_as_non_ascending(self) -> None:
        """
        Equal consecutive step numbers (1, 1) are caught by both the
        duplicate check and the ascending check.  Confirm ValidationError
        is raised regardless of which check fires first.
        """
        steps = [
            HTStep(step=5, type="step_a", T_hold_C=900.0, hold_min=60.0),
            HTStep(step=5, type="step_b", T_hold_C=700.0, hold_min=30.0),
        ]
        with pytest.raises(ValidationError):
            HTSchedule(steps=steps)

    def test_individual_invalid_step_propagates_to_schedule(self) -> None:
        """
        An invalid HTStep (zero hold time) embedded in a schedule must
        cause a ValidationError when the schedule is constructed.
        """
        with pytest.raises(ValidationError):
            HTSchedule(steps=[
                HTStep(step=1, type="bad_step", T_hold_C=650.0, hold_min=0.0),
            ])
