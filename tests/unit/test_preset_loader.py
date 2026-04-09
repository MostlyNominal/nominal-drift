"""Tests for nominal_drift.data.presets.loader."""

import json
import tempfile
from pathlib import Path

import pytest

from nominal_drift.data.presets.loader import (
    AlloyPreset,
    _parse_preset_file,
    clear_cache,
    get_preset,
    list_designations,
    list_material_systems,
    load_all_presets,
    validate_presets,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_loader_cache():
    """Clear the preset cache before each test."""
    clear_cache()
    yield
    clear_cache()


def _write_preset_file(path: Path, system: str, alloys: list[dict]) -> Path:
    """Write a temporary preset JSON file."""
    data = {
        "_schema_version": "1.0",
        "_material_system": system,
        "alloys": alloys,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_alloy(designation: str = "TestAlloy", **overrides) -> dict:
    """Return a minimal alloy entry dict."""
    base = {
        "designation": designation,
        "matrix": "test_matrix",
        "composition_wt_pct": {"Fe": 70.0, "Cr": 18.0, "Ni": 12.0},
        "diffusion_elements": ["Cr"],
        "default_diffusing_element": "Cr",
        "default_sink_wt_pct": 12.0,
        "default_depletion_threshold_wt_pct": 12.0,
        "ht_ranges": {
            "T_min_C": 400,
            "T_max_C": 1100,
            "T_default_C": 700,
            "hold_min_min": 1,
            "hold_min_max": 1440,
            "hold_min_default": 60,
        },
        "mechanism_families": ["sensitization_depletion"],
        "notes": "Test alloy.",
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------
# AlloyPreset dataclass
# -----------------------------------------------------------------------


class TestAlloyPreset:
    def test_construction(self):
        p = AlloyPreset(
            designation="304",
            material_system="austenitic_stainless_steels",
            matrix="austenite_FeCrNi",
            composition_wt_pct={"Cr": 18.0, "Ni": 8.0, "Fe": 74.0},
            diffusion_elements=["Cr", "C"],
            default_diffusing_element="Cr",
            default_sink_wt_pct=12.0,
            default_depletion_threshold_wt_pct=12.0,
            ht_ranges={"T_default_C": 700},
            mechanism_families=["sensitization_depletion"],
        )
        assert p.designation == "304"
        assert p.material_system == "austenitic_stainless_steels"

    def test_frozen(self):
        p = AlloyPreset(
            designation="X",
            material_system="X",
            matrix="X",
            composition_wt_pct={},
            diffusion_elements=[],
            default_diffusing_element="",
            default_sink_wt_pct=0.0,
            default_depletion_threshold_wt_pct=0.0,
            ht_ranges={},
            mechanism_families=[],
        )
        with pytest.raises(AttributeError):
            p.designation = "Y"

    def test_notes_default_empty(self):
        p = AlloyPreset(
            designation="X",
            material_system="X",
            matrix="X",
            composition_wt_pct={},
            diffusion_elements=[],
            default_diffusing_element="",
            default_sink_wt_pct=0.0,
            default_depletion_threshold_wt_pct=0.0,
            ht_ranges={},
            mechanism_families=[],
        )
        assert p.notes == ""


# -----------------------------------------------------------------------
# _parse_preset_file
# -----------------------------------------------------------------------


class TestParsePresetFile:
    def test_parses_valid_file(self, tmp_path):
        f = _write_preset_file(
            tmp_path / "test.json", "test_system",
            [_minimal_alloy("A"), _minimal_alloy("B")],
        )
        result = _parse_preset_file(f)
        assert len(result) == 2
        assert result[0].designation == "A"
        assert result[1].designation == "B"

    def test_material_system_from_file(self, tmp_path):
        f = _write_preset_file(
            tmp_path / "test.json", "my_system", [_minimal_alloy()],
        )
        result = _parse_preset_file(f)
        assert result[0].material_system == "my_system"

    def test_skips_malformed_entries(self, tmp_path):
        data = {
            "_material_system": "test",
            "alloys": [
                {"designation": "Good", "matrix": "m",
                 "composition_wt_pct": {}, "diffusion_elements": [],
                 "default_diffusing_element": "", "default_sink_wt_pct": 0,
                 "default_depletion_threshold_wt_pct": 0,
                 "ht_ranges": {}, "mechanism_families": []},
                {"bad_key": "no designation"},
            ],
        }
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))
        result = _parse_preset_file(f)
        assert len(result) == 1
        assert result[0].designation == "Good"

    def test_returns_empty_for_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        result = _parse_preset_file(f)
        assert result == []

    def test_returns_empty_for_missing_file(self, tmp_path):
        result = _parse_preset_file(tmp_path / "nonexistent.json")
        assert result == []

    def test_returns_empty_for_no_alloys_key(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"_material_system": "test"}')
        result = _parse_preset_file(f)
        assert result == []


# -----------------------------------------------------------------------
# load_all_presets — uses real preset files on disk
# -----------------------------------------------------------------------


class TestLoadAllPresets:
    def test_returns_list(self):
        result = load_all_presets()
        assert isinstance(result, list)

    def test_has_at_least_10_presets(self):
        # 3 files × 3-4 alloys each = at least 10
        result = load_all_presets()
        assert len(result) >= 10

    def test_all_are_alloy_preset(self):
        for p in load_all_presets():
            assert isinstance(p, AlloyPreset)

    def test_all_have_designation(self):
        for p in load_all_presets():
            assert p.designation

    def test_all_have_composition(self):
        for p in load_all_presets():
            assert p.composition_wt_pct

    def test_all_have_ht_ranges(self):
        for p in load_all_presets():
            assert "T_default_C" in p.ht_ranges

    def test_all_have_mechanism_families(self):
        for p in load_all_presets():
            assert len(p.mechanism_families) > 0


# -----------------------------------------------------------------------
# get_preset
# -----------------------------------------------------------------------


class TestGetPreset:
    def test_finds_known_preset(self):
        p = get_preset("316L")
        assert p is not None
        assert p.designation == "316L"

    def test_returns_none_for_unknown(self):
        assert get_preset("NonexistentAlloy999") is None

    def test_316l_has_correct_system(self):
        p = get_preset("316L")
        assert p.material_system == "austenitic_stainless_steels"

    def test_inconel_718_has_correct_system(self):
        p = get_preset("Inconel 718")
        assert p is not None
        assert p.material_system == "nickel_superalloys"

    def test_7075_has_correct_system(self):
        p = get_preset("7075-T6")
        assert p is not None
        assert p.material_system == "aluminium_alloys"

    def test_preset_composition_nonempty(self):
        p = get_preset("304L")
        assert len(p.composition_wt_pct) > 3

    def test_preset_diffusion_elements_nonempty(self):
        p = get_preset("304")
        assert len(p.diffusion_elements) >= 1


# -----------------------------------------------------------------------
# list_designations
# -----------------------------------------------------------------------


class TestListDesignations:
    def test_returns_sorted_list(self):
        result = list_designations()
        assert result == sorted(result)

    def test_contains_known_alloys(self):
        result = list_designations()
        assert "316L" in result
        assert "Inconel 718" in result
        assert "7075-T6" in result

    def test_no_duplicates(self):
        result = list_designations()
        assert len(result) == len(set(result))


# -----------------------------------------------------------------------
# list_material_systems
# -----------------------------------------------------------------------


class TestListMaterialSystems:
    def test_returns_sorted_list(self):
        result = list_material_systems()
        assert result == sorted(result)

    def test_contains_three_systems(self):
        result = list_material_systems()
        assert "austenitic_stainless_steels" in result
        assert "nickel_superalloys" in result
        assert "aluminium_alloys" in result

    def test_no_duplicates(self):
        result = list_material_systems()
        assert len(result) == len(set(result))


# -----------------------------------------------------------------------
# Cross-system checks — ensures generality
# -----------------------------------------------------------------------


class TestCrossSystemGenerality:
    """Verify that the preset system is not steel-only."""

    def test_has_non_cr_diffusion_elements(self):
        """At least one preset uses a non-Cr default diffusing element."""
        presets = load_all_presets()
        non_cr = [p for p in presets if p.default_diffusing_element != "Cr"]
        assert len(non_cr) > 0, "All presets default to Cr — not general enough"

    def test_has_al_based_alloy(self):
        presets = load_all_presets()
        al = [p for p in presets if "Al" in p.composition_wt_pct
              and p.composition_wt_pct.get("Al", 0) > 50]
        assert len(al) > 0

    def test_has_ni_based_alloy(self):
        presets = load_all_presets()
        ni = [p for p in presets if "Ni" in p.composition_wt_pct
              and p.composition_wt_pct.get("Ni", 0) > 40]
        assert len(ni) > 0

    def test_sink_values_differ_across_systems(self):
        """Different material systems should have different sink defaults."""
        presets = load_all_presets()
        sinks = {p.default_sink_wt_pct for p in presets}
        assert len(sinks) > 1, "All presets have the same sink — not general"

    def test_ht_ranges_differ_across_systems(self):
        """Al alloys have lower HT temps than steels."""
        steel = get_preset("316L")
        al = get_preset("7075-T6")
        assert al.ht_ranges["T_max_C"] < steel.ht_ranges["T_max_C"]

    def test_multiple_mechanism_families_present(self):
        presets = load_all_presets()
        all_families = set()
        for p in presets:
            all_families.update(p.mechanism_families)
        assert len(all_families) >= 2


# -----------------------------------------------------------------------
# balance_element field
# -----------------------------------------------------------------------


class TestBalanceElement:
    """Verify the explicit balance_element field on presets."""

    def test_all_disk_presets_have_balance_element(self):
        """Every preset loaded from real JSON files must declare a balance element."""
        for p in load_all_presets():
            assert p.balance_element, (
                f"Preset '{p.designation}' ({p.material_system}) "
                f"has no balance_element"
            )

    def test_balance_element_in_composition(self):
        """The declared balance element must be a key in composition_wt_pct."""
        for p in load_all_presets():
            assert p.balance_element in p.composition_wt_pct, (
                f"Preset '{p.designation}': balance_element "
                f"'{p.balance_element}' not in composition"
            )

    def test_steel_presets_balance_is_fe(self):
        for name in ("304", "304L", "316L", "321"):
            p = get_preset(name)
            assert p.balance_element == "Fe"

    def test_ni_presets_balance_is_ni(self):
        for name in ("Inconel 718", "Waspaloy", "Hastelloy X"):
            p = get_preset(name)
            assert p.balance_element == "Ni"

    def test_al_presets_balance_is_al(self):
        for name in ("2024-T3", "6061-T6", "7075-T6"):
            p = get_preset(name)
            assert p.balance_element == "Al"

    def test_balance_element_defaults_empty_when_not_provided(self):
        """AlloyPreset constructed without balance_element gets '' default."""
        p = AlloyPreset(
            designation="X", material_system="X", matrix="X",
            composition_wt_pct={}, diffusion_elements=[],
            default_diffusing_element="", default_sink_wt_pct=0.0,
            default_depletion_threshold_wt_pct=0.0,
            ht_ranges={}, mechanism_families=[],
        )
        assert p.balance_element == ""

    def test_parser_infers_balance_from_max_element(self, tmp_path):
        """When JSON has no balance_element, the loader infers from max wt%."""
        alloy = _minimal_alloy("InferTest")
        # No balance_element in the dict; Fe=70 is the max
        f = _write_preset_file(
            tmp_path / "test.json", "test", [alloy],
        )
        result = _parse_preset_file(f)
        assert result[0].balance_element == "Fe"

    def test_parser_uses_explicit_balance_element(self, tmp_path):
        """When JSON has explicit balance_element, the loader uses it."""
        alloy = _minimal_alloy("ExplicitTest")
        alloy["balance_element"] = "Ni"
        f = _write_preset_file(
            tmp_path / "test.json", "test", [alloy],
        )
        result = _parse_preset_file(f)
        assert result[0].balance_element == "Ni"


# -----------------------------------------------------------------------
# validate_presets
# -----------------------------------------------------------------------


class TestValidatePresets:
    """Tests for the validate_presets() helper."""

    def test_all_real_presets_pass_validation(self):
        errors = validate_presets()
        assert errors == [], f"Preset validation errors: {errors}"

    def test_catches_bad_sum(self, tmp_path, monkeypatch):
        """A preset whose composition sums to 50 wt% is flagged."""
        alloy = _minimal_alloy("BadSum")
        alloy["balance_element"] = "Fe"
        alloy["composition_wt_pct"] = {"Fe": 30.0, "Cr": 20.0}  # sum=50
        f = _write_preset_file(tmp_path / "bad.json", "test", [alloy])

        # Monkeypatch the presets dir to use our tmp dir
        import nominal_drift.data.presets.loader as loader_mod
        monkeypatch.setattr(loader_mod, "_PRESETS_DIR", tmp_path)
        clear_cache()

        errors = validate_presets()
        assert len(errors) == 1
        assert "50.00" in errors[0]

    def test_catches_missing_balance_element(self, tmp_path, monkeypatch):
        """A preset with no balance_element (and empty comp) is flagged."""
        alloy = _minimal_alloy("NoBalance")
        alloy["composition_wt_pct"] = {}
        f = _write_preset_file(tmp_path / "nb.json", "test", [alloy])

        import nominal_drift.data.presets.loader as loader_mod
        monkeypatch.setattr(loader_mod, "_PRESETS_DIR", tmp_path)
        clear_cache()

        errors = validate_presets()
        assert any("missing balance_element" in e for e in errors)

    def test_catches_balance_not_in_composition(self, tmp_path, monkeypatch):
        """balance_element that doesn't appear in composition is flagged."""
        alloy = _minimal_alloy("MismatchBalance")
        alloy["balance_element"] = "Zr"
        f = _write_preset_file(tmp_path / "mis.json", "test", [alloy])

        import nominal_drift.data.presets.loader as loader_mod
        monkeypatch.setattr(loader_mod, "_PRESETS_DIR", tmp_path)
        clear_cache()

        errors = validate_presets()
        assert any("Zr" in e and "not found" in e for e in errors)
