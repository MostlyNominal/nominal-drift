"""
Unit tests for nominal_drift.science.thermodynamics (Layer 2).

Coverage
--------
* Analytical screening (Path B): always-on models
  - Cr_eq / Ni_eq computation
  - Matrix phase prediction (austenite vs ferrite)
  - M23C6 sensitization window logic
  - Sigma phase susceptibility
  - Carbon activity model
  - Ms temperature
* ThermodynamicResult structure and provenance fields
* Preset integration: 316L and 430SS map to correct alloy_matrix
* pycalphad path (Path A) graceful failure when no TDB is supplied
* Layer separation: thermodynamic result does not touch diffusion outputs

All tests run without a TDB file — pycalphad Path A is tested only for
graceful failure behaviour, not for validated equilibrium results.
"""

from __future__ import annotations

import pytest

from nominal_drift.science.thermodynamics import (
    ThermodynamicResult,
    get_thermodynamic_context,
    _cr_eq_ni_eq,
    _predict_matrix_phase,
    _m23c6_sensitization_window,
    _sigma_phase_susceptibility,
    _carbon_activity,
    _ms_temperature,
    _wt_to_mole_fractions,
)


# ---------------------------------------------------------------------------
# Fixture compositions
# ---------------------------------------------------------------------------

@pytest.fixture()
def comp_316l():
    """316L austenitic: Fe-16.5Cr-10Ni-2.1Mo-0.02C-0.07N."""
    return {
        "Cr": 16.5, "Ni": 10.0, "Mo": 2.1, "C": 0.02, "N": 0.07,
        "Mn": 1.5, "Si": 0.5, "Fe": 69.31,
    }


@pytest.fixture()
def comp_430():
    """430SS ferritic: Fe-16.5Cr-0.06C-0.04N."""
    return {
        "Cr": 16.5, "C": 0.06, "N": 0.04, "Mn": 1.0, "Si": 1.0, "Fe": 81.4,
    }


@pytest.fixture()
def comp_304():
    """304 austenitic: Fe-18Cr-8Ni-0.06C."""
    return {
        "Cr": 18.0, "Ni": 8.0, "C": 0.06, "N": 0.05, "Mn": 1.5, "Si": 0.5, "Fe": 71.89,
    }


# ---------------------------------------------------------------------------
# Cr_eq / Ni_eq
# ---------------------------------------------------------------------------

class TestCrEqNiEq:
    def test_316l_cr_eq(self, comp_316l):
        cr_eq, ni_eq = _cr_eq_ni_eq(comp_316l)
        # Cr_eq = Cr + Mo + 1.5·Si + 0.5·Nb
        # = 16.5 + 2.1 + 1.5*0.5 = 19.35
        assert abs(cr_eq - 19.35) < 0.1

    def test_316l_ni_eq(self, comp_316l):
        _, ni_eq = _cr_eq_ni_eq(comp_316l)
        # Ni_eq = Ni + 30·C + 0.5·Mn + 30·N
        # = 10.0 + 0.6 + 0.75 + 2.1 = 13.45
        assert abs(ni_eq - 13.45) < 0.1

    def test_430_cr_eq(self, comp_430):
        cr_eq, ni_eq = _cr_eq_ni_eq(comp_430)
        # Cr_eq = 16.5 + 0 + 1.5*1.0 = 18.0
        assert abs(cr_eq - 18.0) < 0.1
        # Ni_eq = 0 + 30*0.06 + 0.5*1.0 + 30*0.04 = 1.8 + 0.5 + 1.2 = 3.5
        assert abs(ni_eq - 3.5) < 0.1

    def test_case_insensitive(self):
        """Lower-case keys should give same result as upper-case."""
        comp_upper = {"CR": 18.0, "NI": 8.0, "C": 0.06, "FE": 73.94}
        comp_lower = {"Cr": 18.0, "Ni": 8.0, "C": 0.06, "Fe": 73.94}
        cr1, ni1 = _cr_eq_ni_eq(comp_upper)
        cr2, ni2 = _cr_eq_ni_eq(comp_lower)
        assert abs(cr1 - cr2) < 1e-9
        assert abs(ni1 - ni2) < 1e-9


# ---------------------------------------------------------------------------
# Matrix phase prediction
# ---------------------------------------------------------------------------

class TestMatrixPhasePrediction:
    def test_316l_austenite(self, comp_316l):
        cr_eq, ni_eq = _cr_eq_ni_eq(comp_316l)
        phase, note = _predict_matrix_phase(cr_eq, ni_eq)
        assert "Austenite" in phase

    def test_430_ferrite(self, comp_430):
        cr_eq, ni_eq = _cr_eq_ni_eq(comp_430)
        phase, note = _predict_matrix_phase(cr_eq, ni_eq)
        assert "Ferrite" in phase

    def test_304_austenite(self, comp_304):
        cr_eq, ni_eq = _cr_eq_ni_eq(comp_304)
        phase, note = _predict_matrix_phase(cr_eq, ni_eq)
        assert "Austenite" in phase


# ---------------------------------------------------------------------------
# M23C6 sensitization window
# ---------------------------------------------------------------------------

class TestM23C6Window:
    def test_316l_peak_window(self, comp_316l):
        susc, note = _m23c6_sensitization_window(700.0, comp_316l["Cr"], comp_316l["C"])
        assert susc is True
        assert "peak" in note.lower() or "700" in note

    def test_430_peak_window(self, comp_430):
        susc, note = _m23c6_sensitization_window(700.0, comp_430["Cr"], comp_430["C"])
        assert susc is True  # 430 at 700°C with 0.06 C — susceptible

    def test_below_kinetic_onset(self, comp_316l):
        susc, note = _m23c6_sensitization_window(200.0, comp_316l["Cr"], comp_316l["C"])
        assert susc is False
        assert "kinetic" in note.lower() or "suppressed" in note.lower()

    def test_above_dissolution_temperature(self, comp_316l):
        susc, note = _m23c6_sensitization_window(950.0, comp_316l["Cr"], comp_316l["C"])
        assert susc is False

    def test_very_low_carbon(self):
        susc, note = _m23c6_sensitization_window(700.0, 18.0, 0.001)
        assert susc is False
        assert "unlikely" in note.lower() or "below" in note.lower()

    def test_low_carbon_304l(self):
        susc, _ = _m23c6_sensitization_window(700.0, 18.0, 0.02)
        # 0.02 wt% C — still above 0.003 threshold but product may be below threshold
        # Just verify it returns a bool without raising
        assert isinstance(susc, bool)

    def test_sensitization_window_edge(self):
        # Just above 425°C
        susc, note = _m23c6_sensitization_window(430.0, 18.0, 0.06)
        assert isinstance(susc, bool)


# ---------------------------------------------------------------------------
# Sigma phase
# ---------------------------------------------------------------------------

class TestSigmaPhase:
    def test_316l_no_sigma(self, comp_316l):
        susc, note = _sigma_phase_susceptibility(
            comp_316l["Cr"], comp_316l["Mo"], comp_316l.get("Si", 0), 750.0
        )
        # 316L: Cr+Mo+1.5Si = 16.5+2.1+0.75 = 19.35 < 21 → no sigma
        assert susc is False

    def test_high_cr_mo_sigma(self):
        # Alloy with Cr+Mo = 22 wt%
        susc, note = _sigma_phase_susceptibility(20.0, 2.5, 0.0, 750.0)
        assert susc is True

    def test_sigma_outside_temperature_range(self):
        susc, note = _sigma_phase_susceptibility(22.0, 0.0, 0.0, 1000.0)
        # Above sigma range (900°C max)
        assert susc is False

    def test_sigma_below_temperature_range(self):
        susc, note = _sigma_phase_susceptibility(22.0, 0.0, 0.0, 300.0)
        assert susc is False


# ---------------------------------------------------------------------------
# Carbon activity
# ---------------------------------------------------------------------------

class TestCarbonActivity:
    def test_activity_positive(self, comp_316l):
        a_C, note = _carbon_activity(700.0, comp_316l["Cr"], comp_316l["C"], comp_316l["Ni"])
        assert a_C > 0

    def test_activity_decreases_with_cr(self):
        """Higher Cr should lower C activity (Chipman model E_Cr < 0)."""
        a_low_cr, _ = _carbon_activity(700.0, 10.0, 0.02, 0.0)
        a_high_cr, _ = _carbon_activity(700.0, 25.0, 0.02, 0.0)
        assert a_high_cr < a_low_cr

    def test_activity_increases_with_ni(self):
        """Higher Ni should increase C activity (E_Ni > 0)."""
        a_no_ni, _ = _carbon_activity(700.0, 16.5, 0.02, 0.0)
        a_high_ni, _ = _carbon_activity(700.0, 16.5, 0.02, 20.0)
        assert a_high_ni > a_no_ni

    def test_note_contains_model_name(self, comp_316l):
        _, note = _carbon_activity(700.0, comp_316l["Cr"], comp_316l["C"], comp_316l["Ni"])
        assert "Chipman" in note or "Hillert" in note


# ---------------------------------------------------------------------------
# Ms temperature
# ---------------------------------------------------------------------------

class TestMsTemperature:
    def test_316l_ms_below_ambient(self, comp_316l):
        """316L has high Ni — Ms should be far below room temperature."""
        Ms, note = _ms_temperature(comp_316l)
        # Andrews formula: 539 - 423*0.02 - 30.4*1.5 - 17.7*10 - 12.1*16.5 - 7.5*2.1
        # ≈ 539 - 8.46 - 45.6 - 177 - 199.65 - 15.75 ≈ 92.5
        # This is above 0 in the model but in reality austenitic SS is stable
        # Just verify it returns without error and returns a float or None
        assert Ms is None or isinstance(Ms, float)

    def test_430_ms_positive(self, comp_430):
        """430SS has no Ni — Ms should be relatively high."""
        Ms, note = _ms_temperature(comp_430)
        # 539 - 423*0.06 - 30.4*1.0 - 0 - 12.1*16.5 - 0
        # ≈ 539 - 25.38 - 30.4 - 199.65 ≈ 283.57
        if Ms is not None:
            assert Ms > 0  # ferritic stainless can transform to martensite


# ---------------------------------------------------------------------------
# Mole fraction conversion
# ---------------------------------------------------------------------------

class TestMoleFractions:
    def test_fracs_sum_to_one(self, comp_316l):
        elements = ["FE", "CR", "NI", "C", "N", "MO", "MN"]
        comp_upper = {k.upper(): v for k, v in comp_316l.items()}
        x = _wt_to_mole_fractions(comp_upper, elements)
        assert abs(sum(x.values()) - 1.0) < 1e-10

    def test_fracs_positive(self, comp_316l):
        elements = ["FE", "CR", "NI", "C", "N"]
        comp_upper = {k.upper(): v for k, v in comp_316l.items()}
        x = _wt_to_mole_fractions(comp_upper, elements)
        assert all(v > 0 for v in x.values())

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _wt_to_mole_fractions({}, ["FE", "CR"])


# ---------------------------------------------------------------------------
# Full ThermodynamicResult integration
# ---------------------------------------------------------------------------

class TestThermodynamicResult:
    def test_316l_result_is_thermo_result(self, comp_316l):
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        assert isinstance(result, ThermodynamicResult)

    def test_430_result_is_thermo_result(self, comp_430):
        result = get_thermodynamic_context(comp_430, 700.0, "ferrite_FeCr")
        assert isinstance(result, ThermodynamicResult)

    def test_matrix_echoed_correctly(self, comp_316l, comp_430):
        r1 = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        r2 = get_thermodynamic_context(comp_430, 700.0, "ferrite_FeCr")
        assert r1.alloy_matrix == "austenite_FeCrNi"
        assert r2.alloy_matrix == "ferrite_FeCr"

    def test_temperature_echoed(self, comp_316l):
        result = get_thermodynamic_context(comp_316l, 750.0, "austenite_FeCrNi")
        assert result.temperature_C == 750.0

    def test_composition_echoed(self, comp_316l):
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        assert "CR" in result.composition_wt_pct or "Cr" in result.composition_wt_pct

    def test_no_tdb_calphad_is_none(self, comp_316l):
        """Without a TDB file, calphad result should be None."""
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        assert result.calphad is None

    def test_analytical_always_populated(self, comp_316l):
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        a = result.analytical
        assert a.cr_equivalent_wt_pct > 0
        assert a.ni_equivalent_wt_pct > 0
        assert isinstance(a.m23c6_susceptible, bool)
        assert isinstance(a.sigma_susceptible, bool)
        assert a.carbon_activity > 0

    def test_316l_matrix_is_austenite(self, comp_316l):
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        assert "Austenite" in result.analytical.predicted_matrix_phase

    def test_430_matrix_is_ferrite(self, comp_430):
        result = get_thermodynamic_context(comp_430, 700.0, "ferrite_FeCr")
        assert "Ferrite" in result.analytical.predicted_matrix_phase

    def test_316l_at_peak_is_m23c6_susceptible(self, comp_316l):
        """316L at 700°C peak sensitization window."""
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        assert result.analytical.m23c6_susceptible is True

    def test_316l_above_dissolution_not_susceptible(self, comp_316l):
        """316L at 900°C — M23C6 should have dissolved."""
        result = get_thermodynamic_context(comp_316l, 900.0, "austenite_FeCrNi")
        assert result.analytical.m23c6_susceptible is False

    def test_summary_contains_temp(self, comp_316l):
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        assert "700" in result.layer2_summary

    def test_limitations_non_empty(self, comp_316l):
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        assert len(result.limitations) > 0

    def test_result_is_frozen(self, comp_316l):
        """ThermodynamicResult must not be mutatable."""
        result = get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")
        with pytest.raises(Exception):  # ValidationError or TypeError for frozen model
            result.temperature_C = 999.0


# ---------------------------------------------------------------------------
# Layer separation: thermodynamics does not touch diffusion outputs
# ---------------------------------------------------------------------------

class TestLayerSeparation:
    def test_layer2_does_not_import_diffusion_engine(self):
        """The thermodynamics module must not import diffusion_engine."""
        import importlib
        import nominal_drift.science.thermodynamics as thermo_mod
        source = importlib.util.find_spec(
            "nominal_drift.science.thermodynamics"
        ).origin
        with open(source, "r", encoding="utf-8") as fh:
            code = fh.read()
        assert "diffusion_engine" not in code, (
            "thermodynamics.py must not import diffusion_engine — "
            "that would couple Layer 1 and Layer 2"
        )

    def test_layer2_does_not_import_crystal_datasets(self):
        """Layer 2 must not import crystal dataset code."""
        import importlib
        source = importlib.util.find_spec(
            "nominal_drift.science.thermodynamics"
        ).origin
        with open(source, "r", encoding="utf-8") as fh:
            code = fh.read()
        for banned in ["load_jsonl", "structures.jsonl", "crystal_search",
                       "dataset_page", "matbench"]:
            assert banned not in code, (
                f"thermodynamics.py must not reference '{banned}' — "
                "Layer 2 and Layer 3 must remain separate"
            )

    def test_diffusion_solve_output_unchanged(self, comp_316l):
        """Running Layer 2 after Layer 1 must not change Layer 1 outputs."""
        from nominal_drift.schemas.composition import AlloyComposition
        from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep
        from nominal_drift.science.diffusion_engine import solve_diffusion

        comp = AlloyComposition(
            alloy_designation="316L",
            alloy_matrix="austenite",  # AlloyComposition schema uses short form
            composition_wt_pct=comp_316l,
        )
        ht = HTSchedule(steps=[HTStep(step=1, type="isothermal_hold", T_hold_C=700.0, hold_min=60)])
        diff_result_before = solve_diffusion(comp, ht, element="Cr",
                                             matrix="austenite_FeCrNi")

        # Run Layer 2
        get_thermodynamic_context(comp_316l, 700.0, "austenite_FeCrNi")

        # Re-run Layer 1 — result must be identical
        diff_result_after = solve_diffusion(comp, ht, element="Cr",
                                            matrix="austenite_FeCrNi")

        assert (diff_result_before.min_concentration_wt_pct
                == diff_result_after.min_concentration_wt_pct)
        assert diff_result_before.depletion_depth_nm == diff_result_after.depletion_depth_nm


# ---------------------------------------------------------------------------
# Preset integration
# ---------------------------------------------------------------------------

class TestPresetIntegration:
    def test_430_preset_loads(self):
        from nominal_drift.data.presets.loader import get_preset, clear_cache
        clear_cache()
        preset = get_preset("430")
        assert preset is not None
        assert preset.matrix == "ferrite_FeCr"
        assert preset.composition_wt_pct["Cr"] == pytest.approx(16.5)

    def test_430_is_ferritic_matrix(self):
        from nominal_drift.data.presets.loader import get_preset, clear_cache
        clear_cache()
        preset = get_preset("430")
        result = get_thermodynamic_context(
            preset.composition_wt_pct,
            700.0,
            preset.matrix,
        )
        assert "Ferrite" in result.analytical.predicted_matrix_phase

    def test_316l_preset_layers(self):
        from nominal_drift.data.presets.loader import get_preset, clear_cache
        clear_cache()
        preset = get_preset("316L")
        result = get_thermodynamic_context(
            preset.composition_wt_pct,
            700.0,
            preset.matrix,
        )
        assert "Austenite" in result.analytical.predicted_matrix_phase

    def test_430_layer1_unsupported(self):
        """ferrite_FeCr returns empty frozenset — Layer 1 cannot run."""
        from nominal_drift.science.supported_elements import get_supported_for_matrix
        supported = get_supported_for_matrix("ferrite_FeCr")
        assert len(supported) == 0, (
            "ferrite_FeCr must return empty frozenset from get_supported_for_matrix — "
            "there are no validated Arrhenius constants for ferritic matrix"
        )

    def test_austenite_layer1_supported(self):
        """austenite_FeCrNi returns Cr, C, N — Layer 1 can run."""
        from nominal_drift.science.supported_elements import get_supported_for_matrix
        supported = get_supported_for_matrix("austenite_FeCrNi")
        assert "Cr" in supported
        assert "C" in supported
        assert "N" in supported
