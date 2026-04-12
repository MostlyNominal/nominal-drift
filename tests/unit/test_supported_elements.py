"""
tests/unit/test_supported_elements.py
======================================
Unit tests for nominal_drift.science.supported_elements.

These tests verify:
  - The module correctly reads the Arrhenius JSON.
  - DIFFUSION_SUPPORTED contains exactly the elements in that JSON.
  - The helper functions behave correctly at boundary conditions.
  - No element is silently added or removed without a corresponding test.

The tests are intentionally strict about which elements are supported so
that adding a new element to arrhenius.json is a conscious, tested change.
"""
from __future__ import annotations

import pytest

from nominal_drift.science.supported_elements import (
    DIFFUSION_SUPPORTED,
    VISUALISATION_ONLY,
    filter_to_supported,
    get_supported_for_matrix,
    is_diffusion_supported,
    known_matrices,
    unsupported_explanation,
)


# ---------------------------------------------------------------------------
# DIFFUSION_SUPPORTED set
# ---------------------------------------------------------------------------

class TestDiffusionSupported:

    def test_is_frozenset(self):
        assert isinstance(DIFFUSION_SUPPORTED, frozenset)

    def test_contains_cr(self):
        assert "Cr" in DIFFUSION_SUPPORTED

    def test_contains_c(self):
        assert "C" in DIFFUSION_SUPPORTED

    def test_contains_n(self):
        assert "N" in DIFFUSION_SUPPORTED

    def test_exactly_three_elements_in_current_db(self):
        """
        This test pins the current Arrhenius database contents.
        If a new element is added (D0 + Qd + literature ref), update this count.
        Currently: Cr, C, N — all in austenite_FeCrNi.
        """
        assert len(DIFFUSION_SUPPORTED) == 3, (
            f"Expected 3 elements in Arrhenius database, got {len(DIFFUSION_SUPPORTED)}: "
            f"{sorted(DIFFUSION_SUPPORTED)}. "
            f"If you added a new element, update this test to the new count."
        )

    def test_does_not_contain_cu(self):
        """Cu has no Arrhenius constants — must not be in supported set."""
        assert "Cu" not in DIFFUSION_SUPPORTED

    def test_does_not_contain_al(self):
        """Al has no Arrhenius constants."""
        assert "Al" not in DIFFUSION_SUPPORTED

    def test_does_not_contain_ni(self):
        """Ni is a composition element but has no Arrhenius constants."""
        assert "Ni" not in DIFFUSION_SUPPORTED

    def test_does_not_contain_fe(self):
        """Fe (matrix base) has no Arrhenius constants in this DB."""
        assert "Fe" not in DIFFUSION_SUPPORTED

    def test_immutable(self):
        with pytest.raises((TypeError, AttributeError)):
            DIFFUSION_SUPPORTED.add("Fake")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# VISUALISATION_ONLY set
# ---------------------------------------------------------------------------

class TestVisualisationOnly:

    def test_is_frozenset(self):
        assert isinstance(VISUALISATION_ONLY, frozenset)

    def test_contains_ni(self):
        assert "Ni" in VISUALISATION_ONLY

    def test_contains_al(self):
        assert "Al" in VISUALISATION_ONLY

    def test_no_overlap_with_diffusion_supported(self):
        """
        VISUALISATION_ONLY and DIFFUSION_SUPPORTED must be disjoint.
        An element cannot be both 'visualisation-only' and 'diffusion-supported'.
        """
        overlap = DIFFUSION_SUPPORTED & VISUALISATION_ONLY
        assert not overlap, (
            f"Elements in both sets: {overlap}. "
            f"Move diffusion-capable elements out of VISUALISATION_ONLY."
        )


# ---------------------------------------------------------------------------
# is_diffusion_supported
# ---------------------------------------------------------------------------

class TestIsDiffusionSupported:

    @pytest.mark.parametrize("element", ["Cr", "C", "N"])
    def test_returns_true_for_supported(self, element):
        assert is_diffusion_supported(element) is True

    @pytest.mark.parametrize("element", ["Cu", "Al", "Ni", "Ti", "Mg", "Zn", "Fe"])
    def test_returns_false_for_unsupported(self, element):
        assert is_diffusion_supported(element) is False

    def test_case_sensitive_lowercase_cr(self):
        """Element symbols are case-sensitive: 'cr' != 'Cr'."""
        assert is_diffusion_supported("cr") is False

    def test_case_sensitive_lowercase_c(self):
        assert is_diffusion_supported("c") is False

    def test_empty_string(self):
        assert is_diffusion_supported("") is False

    def test_unknown_element(self):
        assert is_diffusion_supported("Xx") is False


# ---------------------------------------------------------------------------
# get_supported_for_matrix
# ---------------------------------------------------------------------------

class TestGetSupportedForMatrix:

    def test_known_matrix_returns_subset(self):
        result = get_supported_for_matrix("austenite_FeCrNi")
        assert isinstance(result, frozenset)
        assert result <= DIFFUSION_SUPPORTED  # must be a subset
        assert "Cr" in result
        assert "C" in result
        assert "N" in result

    def test_unknown_matrix_returns_empty_set(self):
        """No-fake-fallback contract: an unknown matrix yields an EMPTY set.

        Returning the full ``DIFFUSION_SUPPORTED`` here would let an
        aluminium or nickel-base preset silently inherit
        Cr-in-austenite constants, which is physically meaningless.
        """
        result = get_supported_for_matrix("totally_unknown_matrix")
        assert result == frozenset()

    def test_empty_matrix_string_returns_empty_set(self):
        result = get_supported_for_matrix("")
        assert result == frozenset()

    def test_al_fcc_matrix_returns_empty_set(self):
        """Aluminium FCC has no validated Arrhenius constants — empty set."""
        result = get_supported_for_matrix("Al_FCC")
        assert result == frozenset()

    def test_ni_base_matrix_returns_empty_set(self):
        """Ni-base superalloy has no validated Arrhenius constants — empty set."""
        result = get_supported_for_matrix("Ni_base_superalloy")
        assert result == frozenset()


# ---------------------------------------------------------------------------
# unsupported_explanation
# ---------------------------------------------------------------------------

class TestUnsupportedExplanation:

    def test_mentions_element_name(self):
        msg = unsupported_explanation("Cu")
        assert "Cu" in msg

    def test_mentions_supported_elements(self):
        msg = unsupported_explanation("Cu")
        # Should list current supported elements
        assert "Cr" in msg

    def test_mentions_arrhenius(self):
        msg = unsupported_explanation("Al")
        assert "Arrhenius" in msg or "arrhenius" in msg.lower()

    def test_supported_element_returns_positive(self):
        msg = unsupported_explanation("Cr")
        assert "supported" in msg.lower()
        # Should not say "not supported" for a supported element
        assert "not yet supported" not in msg

    def test_returns_string(self):
        assert isinstance(unsupported_explanation("Ti"), str)


# ---------------------------------------------------------------------------
# filter_to_supported
# ---------------------------------------------------------------------------

class TestFilterToSupported:

    def test_all_supported(self):
        sup, unsup = filter_to_supported(["Cr", "C", "N"])
        assert sup == ["Cr", "C", "N"]
        assert unsup == []

    def test_all_unsupported(self):
        sup, unsup = filter_to_supported(["Al", "Cu", "Ti"])
        assert sup == []
        assert set(unsup) == {"Al", "Cu", "Ti"}

    def test_mixed(self):
        sup, unsup = filter_to_supported(["Cr", "Al", "C", "Ti", "N"])
        assert sup == ["Cr", "C", "N"]
        assert set(unsup) == {"Al", "Ti"}

    def test_preserves_order_of_supported(self):
        sup, _ = filter_to_supported(["N", "C", "Cr"])
        assert sup == ["N", "C", "Cr"]  # original order preserved

    def test_empty_input(self):
        sup, unsup = filter_to_supported([])
        assert sup == []
        assert unsup == []

    def test_inconel_preset_elements_mostly_unsupported(self):
        """
        Inconel 718 preset lists ['Al', 'Ti', 'Cr', 'Nb', 'C'].
        Only Cr and C should pass through.
        """
        inconel_elements = ["Al", "Ti", "Cr", "Nb", "C"]
        sup, unsup = filter_to_supported(inconel_elements)
        assert "Cr" in sup
        assert "C" in sup
        assert set(unsup) == {"Al", "Ti", "Nb"}

    def test_al_alloy_preset_elements_all_unsupported(self):
        """
        6061-T6 lists ['Mg', 'Si', 'Cu'] — none are in Arrhenius DB.
        This scenario caused the original Cu runtime error.
        """
        al_elements = ["Cu", "Mg"]
        sup, unsup = filter_to_supported(al_elements)
        assert sup == []
        assert set(unsup) == {"Cu", "Mg"}


# ---------------------------------------------------------------------------
# known_matrices — used by GUI/status layer to distinguish "matrix exists
# but element missing" from "matrix entirely unsupported".
# ---------------------------------------------------------------------------

class TestKnownMatrices:

    def test_returns_frozenset(self):
        assert isinstance(known_matrices(), frozenset)

    def test_contains_austenite(self):
        assert "austenite_FeCrNi" in known_matrices()

    def test_does_not_contain_al_fcc(self):
        assert "Al_FCC" not in known_matrices()

    def test_does_not_contain_ni_base(self):
        assert "Ni_base_superalloy" not in known_matrices()


# ---------------------------------------------------------------------------
# Material-aware diffusion selection — the regression behaviour the GUI
# now relies on. These tests pin the no-fake-fallback contract: a non-steel
# matrix MUST yield zero solver-supported species, never a Cr default.
# ---------------------------------------------------------------------------

class TestMaterialAwareSelection:
    """End-to-end checks for the form's element-selection contract."""

    def _intersect_preset(self, preset_elements, matrix):
        """Reproduce the form's intersection logic."""
        supported_for_matrix = get_supported_for_matrix(matrix)
        return [el for el in preset_elements if el in supported_for_matrix]

    def test_steel_316l_returns_cr_c_n(self):
        """Steel 316L preset: full overlap with austenite_FeCrNi."""
        result = self._intersect_preset(
            ["Cr", "C", "N"], "austenite_FeCrNi"
        )
        assert result == ["Cr", "C", "N"]

    def test_aluminium_2024_returns_empty(self):
        """2024-T3 lists Cu, Mg — neither has validated constants in Al matrix."""
        result = self._intersect_preset(["Cu", "Mg"], "Al_FCC")
        assert result == []

    def test_aluminium_6061_returns_empty(self):
        """6061-T6 lists Mg, Si, Cu — none supported in Al matrix."""
        result = self._intersect_preset(["Mg", "Si", "Cu"], "Al_FCC")
        assert result == []

    def test_aluminium_7075_returns_empty(self):
        """7075-T6 lists Zn, Mg, Cu — none supported in Al matrix."""
        result = self._intersect_preset(["Zn", "Mg", "Cu"], "Al_FCC")
        assert result == []

    def test_inconel_718_returns_empty(self):
        """Inconel 718's Ni-base matrix has no validated constants for any species.

        Critically: even though Cr and C *are* in DIFFUSION_SUPPORTED for the
        steel matrix, they MUST NOT leak into Ni_base_superalloy via a fake
        fallback. Cr-in-austenite is not a valid proxy for Cr-in-Ni-base.
        """
        result = self._intersect_preset(
            ["Al", "Ti", "Cr", "Nb", "C"], "Ni_base_superalloy"
        )
        assert result == []

    def test_waspaloy_returns_empty(self):
        """Waspaloy: Ni-base matrix → no supported species."""
        result = self._intersect_preset(
            ["Al", "Ti", "Cr", "C"], "Ni_base_superalloy"
        )
        assert result == []

    def test_hastelloy_x_returns_empty(self):
        """Hastelloy X: Ni-base matrix → no supported species."""
        result = self._intersect_preset(
            ["Cr", "Mo", "C"], "Ni_base_superalloy"
        )
        assert result == []

    def test_unknown_matrix_returns_empty_not_default(self):
        """Custom future matrix: empty, not a Cr fallback."""
        result = self._intersect_preset(["Cr", "C", "N"], "exotic_xyz_matrix")
        assert result == []
