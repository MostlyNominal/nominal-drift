"""Tests for nominal_drift.viz.species_styles."""

import pytest

from nominal_drift.viz.species_styles import (
    ELEMENT_STYLES,
    SpeciesStyle,
    _fallback_style,
    build_style_map,
    get_species_style,
)


# -----------------------------------------------------------------------
# SpeciesStyle dataclass
# -----------------------------------------------------------------------


class TestSpeciesStyleDataclass:
    """Basic SpeciesStyle frozen dataclass tests."""

    def test_default_construction(self):
        s = SpeciesStyle(colour="#FF0000", radius=1.0)
        assert s.colour == "#FF0000"
        assert s.radius == 1.0
        assert s.priority == 50
        assert s.label == ""

    def test_full_construction(self):
        s = SpeciesStyle(colour="#00FF00", radius=0.5, priority=10, label="Mg")
        assert s.priority == 10
        assert s.label == "Mg"

    def test_frozen(self):
        s = SpeciesStyle(colour="#AABBCC", radius=1.0)
        with pytest.raises(AttributeError):
            s.colour = "#000000"

    def test_equality(self):
        a = SpeciesStyle(colour="#123456", radius=0.8, priority=20, label="X")
        b = SpeciesStyle(colour="#123456", radius=0.8, priority=20, label="X")
        assert a == b


# -----------------------------------------------------------------------
# ELEMENT_STYLES registry
# -----------------------------------------------------------------------


class TestElementStylesRegistry:
    """Tests for the pre-defined ELEMENT_STYLES dict."""

    def test_is_dict(self):
        assert isinstance(ELEMENT_STYLES, dict)

    def test_has_common_engineering_metals(self):
        for el in ["Fe", "Cr", "Ni", "Mo", "Al", "Ti", "Cu"]:
            assert el in ELEMENT_STYLES, f"Missing engineering metal: {el}"

    def test_has_common_nonmetals(self):
        for el in ["C", "N", "O", "H", "S", "P"]:
            assert el in ELEMENT_STYLES, f"Missing nonmetal: {el}"

    def test_has_perovskite_species(self):
        for el in ["Ba", "Sr", "La", "Ca", "Pb"]:
            assert el in ELEMENT_STYLES, f"Missing perovskite species: {el}"

    def test_has_lanthanides(self):
        for el in ["La", "Ce", "Nd", "Gd", "Yb"]:
            assert el in ELEMENT_STYLES, f"Missing lanthanide: {el}"

    def test_all_values_are_species_style(self):
        for symbol, style in ELEMENT_STYLES.items():
            assert isinstance(style, SpeciesStyle), f"{symbol} not a SpeciesStyle"

    def test_all_colours_are_hex_strings(self):
        for symbol, style in ELEMENT_STYLES.items():
            assert style.colour.startswith("#"), f"{symbol} colour not hex"
            assert len(style.colour) == 7, f"{symbol} colour wrong length"

    def test_all_radii_positive(self):
        for symbol, style in ELEMENT_STYLES.items():
            assert style.radius > 0, f"{symbol} radius not positive"

    def test_all_labels_nonempty(self):
        for symbol, style in ELEMENT_STYLES.items():
            assert style.label, f"{symbol} missing label"

    def test_registry_has_at_least_50_elements(self):
        assert len(ELEMENT_STYLES) >= 50


# -----------------------------------------------------------------------
# get_species_style
# -----------------------------------------------------------------------


class TestGetSpeciesStyle:
    """Tests for the get_species_style() lookup function."""

    def test_known_element_returns_registry_entry(self):
        s = get_species_style("Fe")
        assert s is ELEMENT_STYLES["Fe"]

    def test_unknown_element_returns_fallback(self):
        s = get_species_style("Uue")  # not in registry
        assert isinstance(s, SpeciesStyle)
        assert s.label == "Uue"

    def test_unknown_element_fallback_is_deterministic(self):
        a = get_species_style("Zz")
        b = get_species_style("Zz")
        assert a == b

    def test_different_unknowns_may_differ(self):
        a = get_species_style("Xx")
        b = get_species_style("Qq")
        # They might collide in colour but should each have correct label
        assert a.label == "Xx"
        assert b.label == "Qq"

    def test_fallback_has_valid_colour(self):
        s = get_species_style("Mystery")
        assert s.colour.startswith("#")

    def test_fallback_has_positive_radius(self):
        s = get_species_style("Custom")
        assert s.radius > 0

    def test_known_elements_have_correct_labels(self):
        for el in ["Cr", "Ni", "O", "La", "Al"]:
            assert get_species_style(el).label == el


# -----------------------------------------------------------------------
# build_style_map
# -----------------------------------------------------------------------


class TestBuildStyleMap:
    """Tests for the build_style_map() convenience function."""

    def test_returns_dict(self):
        m = build_style_map(["Fe", "Cr"])
        assert isinstance(m, dict)

    def test_all_requested_symbols_present(self):
        syms = ["Fe", "Cr", "Ni", "C", "O"]
        m = build_style_map(syms)
        for s in syms:
            assert s in m

    def test_known_symbols_use_registry(self):
        m = build_style_map(["Fe"])
        assert m["Fe"] is ELEMENT_STYLES["Fe"]

    def test_unknown_symbols_get_fallback(self):
        m = build_style_map(["Fe", "Zz"])
        assert isinstance(m["Zz"], SpeciesStyle)
        assert m["Zz"].label == "Zz"

    def test_overrides_take_precedence(self):
        custom = SpeciesStyle(colour="#000000", radius=99.0, label="Custom")
        m = build_style_map(["Fe"], overrides={"Fe": custom})
        assert m["Fe"] is custom

    def test_empty_input(self):
        m = build_style_map([])
        assert m == {}

    def test_overrides_only_affect_specified(self):
        custom = SpeciesStyle(colour="#111111", radius=1.0, label="X")
        m = build_style_map(["Fe", "Cr"], overrides={"Fe": custom})
        assert m["Fe"] is custom
        assert m["Cr"] is ELEMENT_STYLES["Cr"]

    def test_duplicate_symbols_handled(self):
        m = build_style_map(["Fe", "Fe", "Cr"])
        assert "Fe" in m
        assert "Cr" in m


# -----------------------------------------------------------------------
# _fallback_style (internal, but important for robustness)
# -----------------------------------------------------------------------


class TestFallbackStyle:
    """Tests for the _fallback_style internal function."""

    def test_returns_species_style(self):
        s = _fallback_style("Unk")
        assert isinstance(s, SpeciesStyle)

    def test_label_matches_input(self):
        s = _fallback_style("Foo")
        assert s.label == "Foo"

    def test_deterministic(self):
        a = _fallback_style("Bar")
        b = _fallback_style("Bar")
        assert a == b

    def test_radius_is_one(self):
        s = _fallback_style("Baz")
        assert s.radius == 1.0

    def test_priority_is_60(self):
        s = _fallback_style("Qux")
        assert s.priority == 60
