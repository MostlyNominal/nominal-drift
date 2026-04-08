"""
tests/unit/test_doe_planner.py
================================
Unit tests for nominal_drift.science.doe_planner.

Coverage:
  - full_factorial with 3 T × 2 t = 6 base points (+center if enabled)
  - n_total_runs = n_replicates * n_points
  - all points have purpose in {"factorial", "center_point", "validation", "repeatability"}
  - temperature_range_C echoes min/max of input
  - minimum_validation always has 5 base points (4 corners + 1 center)
  - repeatability_plan has n_total_runs == n_replicates
  - assumptions non-empty
  - warnings non-empty when n_replicates < 2
  - result is frozen
  - full_factorial without center is correct
  - full_factorial with center is correct
  - DOEPlan can be serialized
  - ExperimentPoint is frozen
"""

from __future__ import annotations

import pytest

from nominal_drift.science.doe_planner import (
    DOEPlan,
    ExperimentPoint,
    generate_full_factorial,
    generate_minimum_validation,
    generate_repeatability_plan,
)


# ===========================================================================
# Tests: ExperimentPoint
# ===========================================================================

class TestExperimentPoint:
    """Test ExperimentPoint creation and validation."""

    def test_create_experiment_point(self):
        """Can create an ExperimentPoint."""
        point = ExperimentPoint(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=2,
            label="T=700C_t=60min",
            purpose="factorial",
        )
        assert point.temperature_C == 700.0
        assert point.hold_min == 60.0
        assert point.n_replicates == 2

    def test_experiment_point_is_frozen(self):
        """ExperimentPoint is frozen (immutable)."""
        from pydantic import ValidationError
        point = ExperimentPoint(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=2,
            label="Test",
            purpose="factorial",
        )
        with pytest.raises(ValidationError):
            point.temperature_C = 800.0

    def test_rejects_invalid_purpose(self):
        """Rejects invalid purpose value."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ExperimentPoint(
                temperature_C=700.0,
                hold_min=60.0,
                n_replicates=2,
                label="Test",
                purpose="invalid",
            )

    def test_rejects_zero_replicates(self):
        """Rejects n_replicates < 1."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ExperimentPoint(
                temperature_C=700.0,
                hold_min=60.0,
                n_replicates=0,
                label="Test",
                purpose="factorial",
            )


# ===========================================================================
# Tests: DOEPlan
# ===========================================================================

class TestDOEPlan:
    """Test DOEPlan creation and validation."""

    def test_doeplan_is_frozen(self):
        """DOEPlan is frozen (immutable)."""
        from pydantic import ValidationError
        point = ExperimentPoint(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=2,
            label="Test",
            purpose="factorial",
        )
        plan = DOEPlan(
            plan_type="full_factorial",
            experiment_points=[point],
            n_total_runs=2,
            temperature_range_C=(700.0, 700.0),
            time_range_min=(60.0, 60.0),
            alloy_label="Test",
            rationale="Test plan",
            assumptions=[],
            warnings=[],
            notes=[],
        )
        with pytest.raises(ValidationError):
            plan.plan_type = "minimum_validation"

    def test_rejects_invalid_plan_type(self):
        """Rejects invalid plan_type."""
        from pydantic import ValidationError
        point = ExperimentPoint(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=2,
            label="Test",
            purpose="factorial",
        )
        with pytest.raises(ValidationError):
            DOEPlan(
                plan_type="invalid_type",
                experiment_points=[point],
                n_total_runs=2,
                temperature_range_C=(700.0, 700.0),
                time_range_min=(60.0, 60.0),
                alloy_label="Test",
                rationale="Test",
                assumptions=[],
                warnings=[],
                notes=[],
            )


# ===========================================================================
# Tests: Full Factorial
# ===========================================================================

class TestFullFactorial:
    """Test generate_full_factorial."""

    def test_full_factorial_basic(self):
        """Full factorial with 2 T × 2 t = 4 points + center (by default)."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
            include_center=True,  # explicitly enable to match default
        )
        assert result.plan_type == "full_factorial"
        # 4 factorial points + 1 center
        assert len(result.experiment_points) == 5
        assert result.n_total_runs == 5 * 2  # 5 points × 2 replicates

    def test_full_factorial_with_center(self):
        """Full factorial with center point adds 5th point."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
            include_center=True,
        )
        assert len(result.experiment_points) == 5  # 4 + 1 center
        assert result.n_total_runs == 5 * 2  # 5 points × 2 replicates

    def test_full_factorial_without_center(self):
        """Full factorial with include_center=False omits center."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
            include_center=False,
        )
        # All points should be "factorial", no "center_point"
        purposes = {p.purpose for p in result.experiment_points}
        assert "center_point" not in purposes
        assert "factorial" in purposes

    def test_full_factorial_3x2(self):
        """Full factorial 3 T × 2 t = 6 points (+ center)."""
        result = generate_full_factorial(
            temperatures_C=[600.0, 700.0, 800.0],
            hold_times_min=[30.0, 90.0],
            n_replicates=3,
            include_center=True,
        )
        assert len(result.experiment_points) == 7  # 6 + 1 center
        assert result.n_total_runs == 7 * 3  # 7 points × 3 replicates

    def test_full_factorial_single_temperature(self):
        """Works with single temperature."""
        result = generate_full_factorial(
            temperatures_C=[700.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        assert len(result.experiment_points) == 2  # 2 times, 1 temp
        # No center point added if less than 2 temps
        purposes = {p.purpose for p in result.experiment_points}
        assert purposes == {"factorial"}

    def test_full_factorial_single_time(self):
        """Works with single hold time."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[60.0],
            n_replicates=2,
        )
        assert len(result.experiment_points) == 2  # 2 temps, 1 time

    def test_full_factorial_temperature_range(self):
        """temperature_range_C echoes min/max of input."""
        result = generate_full_factorial(
            temperatures_C=[600.0, 700.0, 900.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        assert result.temperature_range_C == (600.0, 900.0)

    def test_full_factorial_time_range(self):
        """time_range_min echoes min/max of input."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[15.0, 120.0],
            n_replicates=2,
        )
        assert result.time_range_min == (15.0, 120.0)

    def test_full_factorial_all_purposes_valid(self):
        """All points have valid purpose values."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
            include_center=True,
        )
        valid_purposes = {"factorial", "center_point"}
        for point in result.experiment_points:
            assert point.purpose in valid_purposes

    def test_full_factorial_has_rationale(self):
        """Result has non-empty rationale."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        assert len(result.rationale) > 0

    def test_full_factorial_assumptions_non_empty(self):
        """assumptions list is non-empty."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        assert len(result.assumptions) > 0

    def test_full_factorial_large_grid(self):
        """Works with large temperature/time grids."""
        result = generate_full_factorial(
            temperatures_C=[500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0],
            hold_times_min=[10.0, 30.0, 60.0, 120.0],
            n_replicates=2,
            include_center=False,  # avoid center point to keep count predictable
        )
        assert len(result.experiment_points) == 7 * 4  # 7 temps × 4 times
        assert result.n_total_runs == (7 * 4) * 2

    def test_full_factorial_with_1_replicate(self):
        """Works with n_replicates=1."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=1,
            include_center=False,  # avoid center to keep count predictable
        )
        assert result.n_total_runs == 4  # 4 points × 1

    def test_full_factorial_warns_on_low_replicates(self):
        """Warns if n_replicates < 2."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=1,
        )
        assert len(result.warnings) > 0
        assert "less than 2" in result.warnings[0].lower()

    def test_full_factorial_empty_temps_raises(self):
        """Raises ValueError if temperatures_C is empty."""
        with pytest.raises(ValueError, match="must be non-empty"):
            generate_full_factorial(
                temperatures_C=[],
                hold_times_min=[30.0, 60.0],
                n_replicates=2,
            )

    def test_full_factorial_empty_times_raises(self):
        """Raises ValueError if hold_times_min is empty."""
        with pytest.raises(ValueError, match="must be non-empty"):
            generate_full_factorial(
                temperatures_C=[650.0, 750.0],
                hold_times_min=[],
                n_replicates=2,
            )


# ===========================================================================
# Tests: Minimum Validation
# ===========================================================================

class TestMinimumValidation:
    """Test generate_minimum_validation."""

    def test_minimum_validation_basic(self):
        """Minimum validation with 2 T × 2 t has 5 points (4 corners + 1 center)."""
        result = generate_minimum_validation(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=3,
        )
        assert result.plan_type == "minimum_validation"
        assert len(result.experiment_points) == 5  # 4 corners + 1 center
        assert result.n_total_runs == 5 * 3  # 5 points × 3 replicates

    def test_minimum_validation_single_temperature(self):
        """With single temperature, uses that as both min and max."""
        result = generate_minimum_validation(
            temperatures_C=[700.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        # Corners: (700, 30), (700, 60) + center
        assert len(result.experiment_points) == 3

    def test_minimum_validation_single_time(self):
        """With single hold time, uses that as both min and max."""
        result = generate_minimum_validation(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[60.0],
            n_replicates=2,
        )
        # Corners: (650, 60), (750, 60) + center
        assert len(result.experiment_points) == 3

    def test_minimum_validation_always_has_center(self):
        """Center point always included in minimum_validation."""
        result = generate_minimum_validation(
            temperatures_C=[600.0, 700.0],
            hold_times_min=[30.0, 90.0],
            n_replicates=2,
        )
        purposes = {p.purpose for p in result.experiment_points}
        assert "center_point" in purposes

    def test_minimum_validation_temperature_range(self):
        """temperature_range_C from input min/max."""
        result = generate_minimum_validation(
            temperatures_C=[550.0, 600.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        # Should use only min and max
        assert result.temperature_range_C == (550.0, 750.0)

    def test_minimum_validation_time_range(self):
        """time_range_min from input min/max."""
        result = generate_minimum_validation(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[15.0, 60.0, 120.0],
            n_replicates=2,
        )
        # Should use only min and max
        assert result.time_range_min == (15.0, 120.0)

    def test_minimum_validation_assumptions_non_empty(self):
        """assumptions list is non-empty."""
        result = generate_minimum_validation(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        assert len(result.assumptions) > 0

    def test_minimum_validation_warns_on_low_replicates(self):
        """Warns if n_replicates < 2."""
        result = generate_minimum_validation(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=1,
        )
        assert len(result.warnings) > 0

    def test_minimum_validation_no_warnings_with_adequate_replicates(self):
        """No warnings if n_replicates >= 2."""
        result = generate_minimum_validation(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=3,
        )
        # Should be empty or not contain low-replicate warnings
        low_rep_warnings = [w for w in result.warnings if "less than 2" in w.lower()]
        assert len(low_rep_warnings) == 0

    def test_minimum_validation_large_temperature_range(self):
        """Works with many temperature values (uses only min/max)."""
        temps = [500.0, 600.0, 700.0, 800.0, 900.0]
        result = generate_minimum_validation(
            temperatures_C=temps,
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        # Should still be 5 points (4 corners + 1 center)
        assert len(result.experiment_points) == 5
        assert result.temperature_range_C == (500.0, 900.0)


# ===========================================================================
# Tests: Repeatability Plan
# ===========================================================================

class TestRepeatabilityPlan:
    """Test generate_repeatability_plan."""

    def test_repeatability_plan_basic(self):
        """Repeatability plan has 1 point repeated n_replicates times."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=6,
        )
        assert result.plan_type == "repeatability"
        assert len(result.experiment_points) == 1
        assert result.n_total_runs == 6

    def test_repeatability_plan_purpose(self):
        """All points in repeatability plan have purpose="repeatability"."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=6,
        )
        for point in result.experiment_points:
            assert point.purpose == "repeatability"

    def test_repeatability_plan_temperature_range(self):
        """temperature_range_C is constant (same T)."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=6,
        )
        assert result.temperature_range_C == (700.0, 700.0)

    def test_repeatability_plan_time_range(self):
        """time_range_min is constant (same t)."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=6,
        )
        assert result.time_range_min == (60.0, 60.0)

    def test_repeatability_plan_assumptions_non_empty(self):
        """assumptions list is non-empty."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=6,
        )
        assert len(result.assumptions) > 0

    def test_repeatability_plan_notes_non_empty(self):
        """notes list is non-empty."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=6,
        )
        assert len(result.notes) > 0

    def test_repeatability_plan_warns_on_low_replicates(self):
        """Warns if n_replicates < 2."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=1,
        )
        assert len(result.warnings) > 0

    def test_repeatability_plan_default_replicates(self):
        """Default n_replicates is 6."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
        )
        assert result.n_total_runs == 6

    def test_repeatability_plan_custom_replicates(self):
        """Custom n_replicates is respected."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=10,
        )
        assert result.n_total_runs == 10


# ===========================================================================
# Tests: Alloy Label
# ===========================================================================

class TestAlloyLabel:
    """Test alloy_label parameter."""

    def test_full_factorial_with_alloy_label(self):
        """alloy_label is stored in result."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
            alloy_label="316L",
        )
        assert result.alloy_label == "316L"

    def test_minimum_validation_with_alloy_label(self):
        """alloy_label is stored in result."""
        result = generate_minimum_validation(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
            alloy_label="304 SS",
        )
        assert result.alloy_label == "304 SS"

    def test_repeatability_plan_with_alloy_label(self):
        """alloy_label is stored in result."""
        result = generate_repeatability_plan(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=6,
            alloy_label="2205 Duplex",
        )
        assert result.alloy_label == "2205 Duplex"


# ===========================================================================
# Tests: Serialization
# ===========================================================================

class TestSerialization:
    """Test that plans can be serialized (e.g., for JSON output)."""

    def test_doeplan_model_dump(self):
        """DOEPlan can be dumped to dict via model_dump."""
        result = generate_full_factorial(
            temperatures_C=[650.0, 750.0],
            hold_times_min=[30.0, 60.0],
            n_replicates=2,
        )
        dumped = result.model_dump()
        assert isinstance(dumped, dict)
        assert "plan_type" in dumped
        assert "experiment_points" in dumped

    def test_experiment_point_model_dump(self):
        """ExperimentPoint can be dumped to dict."""
        point = ExperimentPoint(
            temperature_C=700.0,
            hold_min=60.0,
            n_replicates=2,
            label="Test",
            purpose="factorial",
        )
        dumped = point.model_dump()
        assert dumped["temperature_C"] == 700.0
        assert dumped["hold_min"] == 60.0
