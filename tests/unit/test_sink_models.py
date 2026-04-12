"""
tests/unit/test_sink_models.py
==============================
Unit tests for ``nominal_drift.science.sink_models``.

Coverage:
  - SinkLookupTable construction and validation
  - build_sink_table factory
  - DEFAULT_SINK_TABLE_316L and DEFAULT_SINK_TABLE_304 structure
  - evaluate_sink: interpolated (in-range) queries
  - evaluate_sink: exact calibration points match table values
  - evaluate_sink: linear interpolation midpoint arithmetic
  - evaluate_sink: extrapolation low — clamps to first point, adds warning
  - evaluate_sink: extrapolation high — clamps to last point, adds warning
  - evaluate_sink: large extrapolation gap triggers second warning
  - evaluate_sink: result is a SinkEvaluationResult instance
  - evaluate_sink: result is frozen (immutable)
  - evaluate_sink: result is deterministic
  - evaluate_sink: C_sink_wt_pct is always >= 0
  - evaluate_sink: assumptions always non-empty and mention first-order
  - evaluate_sink: notes always non-empty
  - evaluate_sink: interpolation_mode controlled vocabulary
  - evaluate_sink: T_table_min/max echoed from table
  - evaluate_sink: alloy_label echoed from table
  - Custom table produces correct interpolated values
  - 304 table has lower C_sink at sensitisation nose vs 316L
  - SinkLookupTable rejects non-monotonic temperatures
  - SinkLookupTable rejects length mismatch
  - SinkLookupTable rejects negative c_sink values
  - SinkLookupTable rejects fewer than 2 points
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nominal_drift.science.sink_models import (
    DEFAULT_SINK_TABLE_304,
    DEFAULT_SINK_TABLE_316L,
    SinkEvaluationResult,
    SinkLookupTable,
    build_sink_table,
    evaluate_sink,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def simple_table() -> SinkLookupTable:
    """A minimal 3-point table for arithmetic testing."""
    return build_sink_table(
        temperatures_C=[600.0, 700.0, 800.0],
        c_sink_wt_pct=[14.0, 12.0, 13.5],
        alloy_label="Test alloy",
        source_notes=["Synthetic table for unit tests."],
    )


# ===========================================================================
# TestSinkLookupTableConstruction
# ===========================================================================

class TestSinkLookupTableConstruction:

    def test_valid_table_builds_without_error(self):
        t = build_sink_table([600.0, 700.0, 800.0], [14.0, 12.0, 13.5], "Test")
        assert isinstance(t, SinkLookupTable)

    def test_T_min_property(self):
        t = build_sink_table([600.0, 700.0, 800.0], [14.0, 12.0, 13.5], "Test")
        assert t.T_min == pytest.approx(600.0)

    def test_T_max_property(self):
        t = build_sink_table([600.0, 700.0, 800.0], [14.0, 12.0, 13.5], "Test")
        assert t.T_max == pytest.approx(800.0)

    def test_table_is_frozen(self):
        t = build_sink_table([600.0, 700.0, 800.0], [14.0, 12.0, 13.5], "Test")
        with pytest.raises((AttributeError, TypeError)):
            t.alloy_label = "changed"  # type: ignore[misc]

    def test_source_notes_stored(self):
        notes = ["Note A", "Note B"]
        t = build_sink_table([600.0, 700.0], [14.0, 12.0], "Test", source_notes=notes)
        assert "Note A" in t.source_notes

    def test_no_source_notes_defaults_empty(self):
        t = build_sink_table([600.0, 700.0], [14.0, 12.0], "Test")
        assert t.source_notes == ()

    def test_rejects_fewer_than_two_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            build_sink_table([700.0], [12.0], "Bad table")

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            build_sink_table([600.0, 700.0, 800.0], [14.0, 12.0], "Bad table")

    def test_rejects_non_monotonic_temperatures(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            build_sink_table([600.0, 600.0, 800.0], [14.0, 12.0, 13.5], "Bad table")

    def test_rejects_decreasing_temperatures(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            build_sink_table([800.0, 700.0, 600.0], [13.5, 12.0, 14.0], "Bad table")

    def test_rejects_negative_c_sink(self):
        with pytest.raises(ValueError, match="non-negative"):
            build_sink_table([600.0, 700.0], [14.0, -1.0], "Bad table")


# ===========================================================================
# TestDefaultTables
# ===========================================================================

class TestDefaultTables:

    def test_316L_table_has_multiple_points(self):
        assert len(DEFAULT_SINK_TABLE_316L.temperatures_C) >= 5

    def test_304_table_has_multiple_points(self):
        assert len(DEFAULT_SINK_TABLE_304.temperatures_C) >= 5

    def test_316L_T_min_below_600(self):
        assert DEFAULT_SINK_TABLE_316L.T_min < 600.0

    def test_316L_T_max_above_900(self):
        assert DEFAULT_SINK_TABLE_316L.T_max > 900.0

    def test_304_sensitisation_nose_lower_than_316L(self):
        """304 should have a lower minimum C_sink (deeper sensitisation)."""
        min_316L = min(DEFAULT_SINK_TABLE_316L.c_sink_wt_pct)
        min_304  = min(DEFAULT_SINK_TABLE_304.c_sink_wt_pct)
        assert min_304 < min_316L

    def test_316L_all_c_sink_values_positive(self):
        assert all(v > 0 for v in DEFAULT_SINK_TABLE_316L.c_sink_wt_pct)

    def test_304_all_c_sink_values_positive(self):
        assert all(v > 0 for v in DEFAULT_SINK_TABLE_304.c_sink_wt_pct)

    def test_316L_label_contains_316(self):
        assert "316" in DEFAULT_SINK_TABLE_316L.alloy_label

    def test_304_label_contains_304(self):
        assert "304" in DEFAULT_SINK_TABLE_304.alloy_label

    def test_316L_source_notes_non_empty(self):
        assert len(DEFAULT_SINK_TABLE_316L.source_notes) > 0

    def test_304_source_notes_non_empty(self):
        assert len(DEFAULT_SINK_TABLE_304.source_notes) > 0


# ===========================================================================
# TestEvaluateSinkInterpolation
# ===========================================================================

class TestEvaluateSinkInterpolation:

    def test_exact_first_point_returns_table_value(self, simple_table):
        result = evaluate_sink(600.0, simple_table)
        assert result.C_sink_wt_pct == pytest.approx(14.0)

    def test_exact_last_point_returns_table_value(self, simple_table):
        result = evaluate_sink(800.0, simple_table)
        assert result.C_sink_wt_pct == pytest.approx(13.5)

    def test_exact_middle_point_returns_table_value(self, simple_table):
        result = evaluate_sink(700.0, simple_table)
        assert result.C_sink_wt_pct == pytest.approx(12.0)

    def test_midpoint_linear_interpolation(self, simple_table):
        # Between 600°C (14.0) and 700°C (12.0): midpoint at 650°C → 13.0
        result = evaluate_sink(650.0, simple_table)
        assert result.C_sink_wt_pct == pytest.approx(13.0)

    def test_three_quarter_interpolation(self, simple_table):
        # Between 700°C (12.0) and 800°C (13.5): 3/4 point at 775°C
        # 12.0 + 0.75 * (13.5 - 12.0) = 12.0 + 1.125 = 13.125
        result = evaluate_sink(775.0, simple_table)
        assert result.C_sink_wt_pct == pytest.approx(13.125)

    def test_in_range_mode_is_interpolated(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert result.interpolation_mode == "interpolated"

    def test_in_range_has_no_warnings(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert result.warnings == []

    def test_default_table_is_316L(self):
        result = evaluate_sink(700.0)
        assert "316" in result.alloy_label

    def test_default_316L_at_700C_near_sensitisation_nose(self):
        result = evaluate_sink(700.0)
        # 700°C is the sensitisation nose for 316L — C_sink should be low
        assert result.C_sink_wt_pct < 13.5

    def test_304_at_700C_lower_than_316L_at_700C(self):
        r316 = evaluate_sink(700.0, DEFAULT_SINK_TABLE_316L)
        r304 = evaluate_sink(700.0, DEFAULT_SINK_TABLE_304)
        assert r304.C_sink_wt_pct < r316.C_sink_wt_pct


# ===========================================================================
# TestEvaluateSinkExtrapolation
# ===========================================================================

class TestEvaluateSinkExtrapolation:

    def test_below_range_mode_is_extrapolated_low(self, simple_table):
        result = evaluate_sink(500.0, simple_table)
        assert result.interpolation_mode == "extrapolated_low"

    def test_above_range_mode_is_extrapolated_high(self, simple_table):
        result = evaluate_sink(900.0, simple_table)
        assert result.interpolation_mode == "extrapolated_high"

    def test_below_range_clamped_to_first_value(self, simple_table):
        result = evaluate_sink(500.0, simple_table)
        # table[0] = 14.0 at 600°C — clamp to that
        assert result.C_sink_wt_pct == pytest.approx(14.0)

    def test_above_range_clamped_to_last_value(self, simple_table):
        result = evaluate_sink(900.0, simple_table)
        # table[-1] = 13.5 at 800°C — clamp to that
        assert result.C_sink_wt_pct == pytest.approx(13.5)

    def test_below_range_has_warning(self, simple_table):
        result = evaluate_sink(500.0, simple_table)
        assert len(result.warnings) >= 1
        assert any("below" in w or "minimum" in w for w in result.warnings)

    def test_above_range_has_warning(self, simple_table):
        result = evaluate_sink(900.0, simple_table)
        assert len(result.warnings) >= 1
        assert any("above" in w or "maximum" in w for w in result.warnings)

    def test_large_extrapolation_gap_adds_second_warning(self, simple_table):
        # 500 - 600 = 100°C gap > _EXTRAPOLATION_WARN_K (50°C) → second warning
        result = evaluate_sink(400.0, simple_table)
        assert len(result.warnings) >= 2

    def test_small_extrapolation_gap_only_one_warning(self, simple_table):
        # 590 is 10°C below table minimum 600 — gap < 50°C threshold
        result = evaluate_sink(590.0, simple_table)
        assert len(result.warnings) == 1


# ===========================================================================
# TestResultIntegrity
# ===========================================================================

class TestResultIntegrity:

    def test_result_is_sink_evaluation_result_instance(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert isinstance(result, SinkEvaluationResult)

    def test_result_is_frozen(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        with pytest.raises((ValidationError, TypeError)):
            result.C_sink_wt_pct = 99.0  # type: ignore[misc]

    def test_result_is_deterministic(self, simple_table):
        r1 = evaluate_sink(650.0, simple_table)
        r2 = evaluate_sink(650.0, simple_table)
        assert r1 == r2

    def test_c_sink_always_non_negative(self, simple_table):
        # Test at multiple temperatures including extremes
        for T in [400.0, 600.0, 650.0, 700.0, 800.0, 1000.0]:
            result = evaluate_sink(T, simple_table)
            assert result.C_sink_wt_pct >= 0.0

    def test_T_C_echoed_in_result(self, simple_table):
        result = evaluate_sink(672.5, simple_table)
        assert result.T_C == pytest.approx(672.5)

    def test_T_table_min_echoed(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert result.T_table_min_C == pytest.approx(simple_table.T_min)

    def test_T_table_max_echoed(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert result.T_table_max_C == pytest.approx(simple_table.T_max)

    def test_alloy_label_echoed(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert result.alloy_label == "Test alloy"


# ===========================================================================
# TestAssumptionsAndNotes
# ===========================================================================

class TestAssumptionsAndNotes:

    def test_assumptions_always_non_empty(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert len(result.assumptions) >= 3

    def test_assumptions_mention_first_order(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        combined = " ".join(result.assumptions).lower()
        assert "first-order" in combined

    def test_assumptions_mention_not_calphad(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        combined = " ".join(result.assumptions).lower()
        assert "calphad" in combined

    def test_notes_always_non_empty(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert len(result.notes) >= 1

    def test_source_notes_reflected_in_result_notes(self):
        table = build_sink_table(
            [600.0, 700.0], [14.0, 12.0], "Custom",
            source_notes=["Custom provenance note."]
        )
        result = evaluate_sink(650.0, table)
        assert any("Custom provenance note" in n for n in result.notes)


# ===========================================================================
# TestInterpolationModeVocabulary
# ===========================================================================

class TestInterpolationModeVocabulary:

    VALID_MODES = {"interpolated", "extrapolated_low", "extrapolated_high"}

    def test_in_range_mode_is_valid(self, simple_table):
        result = evaluate_sink(650.0, simple_table)
        assert result.interpolation_mode in self.VALID_MODES

    def test_extrapolated_low_mode_is_valid(self, simple_table):
        result = evaluate_sink(400.0, simple_table)
        assert result.interpolation_mode in self.VALID_MODES

    def test_extrapolated_high_mode_is_valid(self, simple_table):
        result = evaluate_sink(1000.0, simple_table)
        assert result.interpolation_mode in self.VALID_MODES

    def test_invalid_mode_raises_pydantic_error(self):
        with pytest.raises(ValidationError):
            SinkEvaluationResult(
                T_C=700.0,
                C_sink_wt_pct=12.5,
                interpolation_mode="bogus_mode",   # invalid
                T_table_min_C=400.0,
                T_table_max_C=1100.0,
                alloy_label="Test",
                assumptions=["A"],
                warnings=[],
                notes=["N"],
            )

    def test_negative_c_sink_raises_pydantic_error(self):
        with pytest.raises(ValidationError):
            SinkEvaluationResult(
                T_C=700.0,
                C_sink_wt_pct=-1.0,               # invalid
                interpolation_mode="interpolated",
                T_table_min_C=400.0,
                T_table_max_C=1100.0,
                alloy_label="Test",
                assumptions=["A"],
                warnings=[],
                notes=["N"],
            )


# ===========================================================================
# TestCustomTable
# ===========================================================================

class TestCustomTable:

    def test_custom_two_point_table(self):
        table = build_sink_table([500.0, 900.0], [15.0, 14.0], "Two-point test")
        result = evaluate_sink(700.0, table)
        # Linear interp: 15.0 + (700-500)/(900-500) * (14.0-15.0) = 15.0 - 0.5 = 14.5
        assert result.C_sink_wt_pct == pytest.approx(14.5)

    def test_flat_table_returns_constant_value(self):
        table = build_sink_table([600.0, 700.0, 800.0], [13.0, 13.0, 13.0], "Flat")
        for T in [600.0, 650.0, 700.0, 750.0, 800.0]:
            result = evaluate_sink(T, table)
            assert result.C_sink_wt_pct == pytest.approx(13.0)

    def test_custom_table_label_propagated(self):
        table = build_sink_table([600.0, 800.0], [14.0, 13.0], "My special alloy")
        result = evaluate_sink(700.0, table)
        assert result.alloy_label == "My special alloy"
