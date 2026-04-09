"""
nominal_drift.viz.species_styles
================================
Generic element / species styling for visualisation.

This module provides a reusable, periodic-table-inspired mapping of chemical
element symbols to visual properties (colour, radius, display priority).
It is intentionally material-agnostic: steels, aluminium alloys, nickel
superalloys, oxides, perovskites, carbon structures, and arbitrary CIF-based
materials are all supported.

Unknown or custom species (e.g. ``"Vacancy"``, ``"Interstitial"``) receive
sensible fallback defaults so visualisations never crash.

Public API
----------
``SpeciesStyle``
    Frozen dataclass holding colour, radius, and priority for one species.

``get_species_style(symbol)``
    Look up or generate a style for any element symbol.

``build_style_map(symbols)``
    Return ``{symbol: SpeciesStyle}`` for an iterable of symbols.

``ELEMENT_STYLES``
    Module-level dict of pre-defined styles (subset of the periodic table
    covering the most common engineering and crystal-data elements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


# ---------------------------------------------------------------------------
# SpeciesStyle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeciesStyle:
    """Visual styling for a single chemical species or element.

    Attributes
    ----------
    colour : str
        CSS hex colour string (e.g. ``"#FF6347"``).
    radius : float
        Relative display radius.  1.0 is the reference (approximately the
        metallic / covalent radius of iron, normalised).
    priority : int
        Display priority.  Lower numbers are drawn later (on top).
        Useful for ensuring minor species remain visible.
    label : str
        Human-readable label (defaults to the element symbol).
    """
    colour: str
    radius: float
    priority: int = 50
    label: str = ""


# ---------------------------------------------------------------------------
# Pre-defined element styles — periodic-table-inspired
# ---------------------------------------------------------------------------
# Colours are chosen for perceptual distinguishability and accessibility.
# Radii are approximate relative metallic / covalent radii (normalised to
# Fe = 1.0) — used as particle size in scatter-based visualisations.
#
# Coverage:  transition metals (3d, 4d, 5d series), alkali / alkaline earth,
# common non-metals, lanthanides commonly found in crystal datasets, and
# selected metalloids.  This list covers >95% of species encountered in
# Perov-5, MP-20, Carbon-24, and MPTS-52 datasets.

ELEMENT_STYLES: Dict[str, SpeciesStyle] = {
    # ------- 3d transition metals (steels, Ni-base, Ti alloys) -------
    "Fe": SpeciesStyle(colour="#C0392B", radius=1.00, priority=40, label="Fe"),
    "Cr": SpeciesStyle(colour="#2E86C1", radius=1.01, priority=30, label="Cr"),
    "Ni": SpeciesStyle(colour="#27AE60", radius=0.98, priority=35, label="Ni"),
    "Mo": SpeciesStyle(colour="#8E44AD", radius=1.10, priority=45, label="Mo"),
    "Mn": SpeciesStyle(colour="#E67E22", radius=1.05, priority=50, label="Mn"),
    "Ti": SpeciesStyle(colour="#1ABC9C", radius=1.16, priority=45, label="Ti"),
    "V":  SpeciesStyle(colour="#D4AC0D", radius=1.06, priority=50, label="V"),
    "Co": SpeciesStyle(colour="#5DADE2", radius=0.99, priority=40, label="Co"),
    "Cu": SpeciesStyle(colour="#CA6F1E", radius=1.00, priority=45, label="Cu"),
    "Zn": SpeciesStyle(colour="#839192", radius=1.06, priority=55, label="Zn"),
    "Sc": SpeciesStyle(colour="#76D7C4", radius=1.30, priority=55, label="Sc"),
    "W":  SpeciesStyle(colour="#5B2C6F", radius=1.10, priority=45, label="W"),

    # ------- Aluminium alloys -------
    "Al": SpeciesStyle(colour="#7FB3D8", radius=1.13, priority=40, label="Al"),
    "Si": SpeciesStyle(colour="#F5B041", radius=0.93, priority=50, label="Si"),
    "Mg": SpeciesStyle(colour="#58D68D", radius=1.28, priority=50, label="Mg"),

    # ------- Common non-metals / interstitials -------
    "C":  SpeciesStyle(colour="#2C3E50", radius=0.60, priority=20, label="C"),
    "N":  SpeciesStyle(colour="#3498DB", radius=0.56, priority=20, label="N"),
    "O":  SpeciesStyle(colour="#E74C3C", radius=0.52, priority=15, label="O"),
    "H":  SpeciesStyle(colour="#ECF0F1", radius=0.25, priority=10, label="H"),
    "S":  SpeciesStyle(colour="#F1C40F", radius=0.82, priority=50, label="S"),
    "P":  SpeciesStyle(colour="#E59866", radius=0.87, priority=50, label="P"),
    "F":  SpeciesStyle(colour="#76D7C4", radius=0.50, priority=25, label="F"),
    "Cl": SpeciesStyle(colour="#45B39D", radius=0.79, priority=30, label="Cl"),
    "Br": SpeciesStyle(colour="#A93226", radius=0.94, priority=35, label="Br"),
    "I":  SpeciesStyle(colour="#6C3483", radius=1.10, priority=40, label="I"),
    "B":  SpeciesStyle(colour="#F0B27A", radius=0.68, priority=45, label="B"),
    "Se": SpeciesStyle(colour="#D4E6F1", radius=0.92, priority=50, label="Se"),
    "Te": SpeciesStyle(colour="#BB8FCE", radius=1.11, priority=50, label="Te"),

    # ------- Alkali / alkaline earth -------
    "Li": SpeciesStyle(colour="#82E0AA", radius=1.23, priority=55, label="Li"),
    "Na": SpeciesStyle(colour="#AED6F1", radius=1.54, priority=55, label="Na"),
    "K":  SpeciesStyle(colour="#D7BDE2", radius=1.96, priority=55, label="K"),
    "Rb": SpeciesStyle(colour="#F9E79F", radius=2.11, priority=55, label="Rb"),
    "Cs": SpeciesStyle(colour="#FADBD8", radius=2.35, priority=55, label="Cs"),
    "Ca": SpeciesStyle(colour="#A3E4D7", radius=1.57, priority=50, label="Ca"),
    "Sr": SpeciesStyle(colour="#D5F5E3", radius=1.74, priority=50, label="Sr"),
    "Ba": SpeciesStyle(colour="#FCF3CF", radius=1.78, priority=50, label="Ba"),
    "Be": SpeciesStyle(colour="#D4EFDF", radius=0.90, priority=55, label="Be"),

    # ------- 4d / 5d transition metals -------
    "Zr": SpeciesStyle(colour="#85C1E9", radius=1.25, priority=45, label="Zr"),
    "Nb": SpeciesStyle(colour="#73C6B6", radius=1.17, priority=45, label="Nb"),
    "Ru": SpeciesStyle(colour="#CD6155", radius=1.06, priority=45, label="Ru"),
    "Rh": SpeciesStyle(colour="#AF7AC5", radius=1.06, priority=45, label="Rh"),
    "Pd": SpeciesStyle(colour="#5499C7", radius=1.08, priority=45, label="Pd"),
    "Ag": SpeciesStyle(colour="#BDC3C7", radius=1.15, priority=45, label="Ag"),
    "Cd": SpeciesStyle(colour="#ABEBC6", radius=1.20, priority=55, label="Cd"),
    "Hf": SpeciesStyle(colour="#A9CCE3", radius=1.23, priority=45, label="Hf"),
    "Ta": SpeciesStyle(colour="#D2B4DE", radius=1.17, priority=45, label="Ta"),
    "Re": SpeciesStyle(colour="#F5CBA7", radius=1.10, priority=45, label="Re"),
    "Os": SpeciesStyle(colour="#ABB2B9", radius=1.08, priority=45, label="Os"),
    "Ir": SpeciesStyle(colour="#F1948A", radius=1.08, priority=45, label="Ir"),
    "Pt": SpeciesStyle(colour="#D5DBDB", radius=1.09, priority=45, label="Pt"),
    "Au": SpeciesStyle(colour="#F4D03F", radius=1.10, priority=35, label="Au"),
    "Sn": SpeciesStyle(colour="#D7DBDD", radius=1.17, priority=50, label="Sn"),
    "In": SpeciesStyle(colour="#E8DAEF", radius=1.24, priority=50, label="In"),
    "Ga": SpeciesStyle(colour="#A2D9CE", radius=1.08, priority=50, label="Ga"),
    "Ge": SpeciesStyle(colour="#FAD7A0", radius=0.99, priority=50, label="Ge"),
    "As": SpeciesStyle(colour="#D5D8DC", radius=0.96, priority=50, label="As"),
    "Bi": SpeciesStyle(colour="#C39BD3", radius=1.20, priority=50, label="Bi"),
    "Pb": SpeciesStyle(colour="#566573", radius=1.22, priority=50, label="Pb"),
    "Tl": SpeciesStyle(colour="#CACFD2", radius=1.30, priority=55, label="Tl"),

    # ------- Lanthanides (common in perovskite/oxide datasets) -------
    "La": SpeciesStyle(colour="#F7DC6F", radius=1.52, priority=50, label="La"),
    "Ce": SpeciesStyle(colour="#F8C471", radius=1.48, priority=50, label="Ce"),
    "Pr": SpeciesStyle(colour="#EB984E", radius=1.46, priority=50, label="Pr"),
    "Nd": SpeciesStyle(colour="#E59866", radius=1.44, priority=50, label="Nd"),
    "Sm": SpeciesStyle(colour="#DC7633", radius=1.40, priority=50, label="Sm"),
    "Eu": SpeciesStyle(colour="#CA6F1E", radius=1.39, priority=50, label="Eu"),
    "Gd": SpeciesStyle(colour="#BA4A00", radius=1.38, priority=50, label="Gd"),
    "Dy": SpeciesStyle(colour="#A04000", radius=1.36, priority=50, label="Dy"),
    "Er": SpeciesStyle(colour="#873600", radius=1.34, priority=50, label="Er"),
    "Yb": SpeciesStyle(colour="#6E2C00", radius=1.32, priority=50, label="Yb"),
    "Y":  SpeciesStyle(colour="#EDBB99", radius=1.42, priority=50, label="Y"),
    "Lu": SpeciesStyle(colour="#784212", radius=1.32, priority=50, label="Lu"),

    # ------- Actinides (occasionally in materials datasets) -------
    "U":  SpeciesStyle(colour="#1B4F72", radius=1.38, priority=50, label="U"),
    "Th": SpeciesStyle(colour="#1A5276", radius=1.42, priority=50, label="Th"),
}

# ---------------------------------------------------------------------------
# Fallback generator
# ---------------------------------------------------------------------------

# Deterministic fallback colours for unknown elements (hash-based)
_FALLBACK_COLOURS = [
    "#95A5A6", "#7DCEA0", "#85929E", "#F0B27A",
    "#BB8FCE", "#73C6B6", "#F1948A", "#82E0AA",
    "#D4AC0D", "#5DADE2", "#CD6155", "#A3E4D7",
]


def _fallback_style(symbol: str) -> SpeciesStyle:
    """Generate a deterministic fallback style for an unknown element."""
    idx = sum(ord(c) for c in symbol) % len(_FALLBACK_COLOURS)
    return SpeciesStyle(
        colour=_FALLBACK_COLOURS[idx],
        radius=1.0,
        priority=60,
        label=symbol,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_species_style(symbol: str) -> SpeciesStyle:
    """Return the visual style for a chemical species.

    Looks up ``ELEMENT_STYLES`` first; generates a deterministic fallback
    for unknown elements.  Never raises.

    Parameters
    ----------
    symbol : str
        Chemical element symbol (e.g. ``"Cr"``, ``"O"``, ``"La"``).
        Case-sensitive — use standard chemical notation.

    Returns
    -------
    SpeciesStyle
        Visual styling for the species.
    """
    if symbol in ELEMENT_STYLES:
        return ELEMENT_STYLES[symbol]
    return _fallback_style(symbol)


def build_style_map(
    symbols: Iterable[str],
    overrides: Optional[Dict[str, SpeciesStyle]] = None,
) -> Dict[str, SpeciesStyle]:
    """Build a ``{symbol: SpeciesStyle}`` map for a set of species.

    Parameters
    ----------
    symbols : iterable of str
        Element symbols to include.
    overrides : dict, optional
        Custom ``{symbol: SpeciesStyle}`` entries that take precedence
        over the default lookup.

    Returns
    -------
    dict
        ``{symbol: SpeciesStyle}`` for every symbol in *symbols*.
    """
    overrides = overrides or {}
    return {
        s: overrides[s] if s in overrides else get_species_style(s)
        for s in symbols
    }
