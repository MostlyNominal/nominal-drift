"""
nominal_drift.science.supported_elements
=========================================
Single source of truth for which elements are supported by the diffusion
engine and the Arrhenius constants database.

The constants are read once from ``science/constants/arrhenius.json`` at
module load time so this file never needs to be updated by hand — adding
a new element to the JSON is sufficient to expose it in the GUI.

Public API
----------
``DIFFUSION_SUPPORTED``
    ``frozenset[str]`` of element symbols supported for diffusion simulation
    (i.e. present in the Arrhenius database for at least one matrix).

``VISUALISATION_ONLY``
    ``frozenset[str]`` of element symbols valid for visualisation and CIF
    workflows but **not** for diffusion (no Arrhenius constants).
    This set is informational only — it is not exhaustive.

``is_diffusion_supported(element)``
    ``bool`` — True if the element can be simulated by the diffusion engine.

``get_supported_for_matrix(matrix)``
    ``frozenset[str]`` — elements supported for a specific Arrhenius matrix.
    Falls back to the full supported set if matrix not found.

``unsupported_explanation(element)``
    Human-readable string explaining *why* an element is not yet supported
    and what would be needed to add it.

Notes
-----
* Adding a new element to the diffusion engine requires:
  1. A validated D₀ and Qd entry in ``science/constants/arrhenius.json``
     (with literature references in ``docs/arrhenius_sources.md``).
  2. At minimum one integration test verifying the Crank–Nicolson solver
     converges for the new element at a representative temperature.
  Anything short of this is out-of-scope for the current sprint.

* Elements present in alloy *compositions* (e.g. Mo, Ti, Al, Nb, Zn) are
  NOT diffusion-supported unless they have Arrhenius constants.  They are
  valid composition keys and valid visualisation targets.
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Load constants
# ---------------------------------------------------------------------------

_ARRHENIUS_PATH: Path = Path(__file__).parent / "constants" / "arrhenius.json"


def _load_arrhenius() -> dict:
    """Read the Arrhenius JSON, stripping metadata keys (underscore-prefixed)."""
    try:
        raw = json.loads(_ARRHENIUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read Arrhenius constants from {_ARRHENIUS_PATH}: {exc}"
        ) from exc
    return {k: v for k, v in raw.items() if not k.startswith("_")}


_ARRHENIUS: dict = _load_arrhenius()

# ---------------------------------------------------------------------------
# Derived sets — built once at import time
# ---------------------------------------------------------------------------

#: Elements that have validated Arrhenius constants and can be simulated.
DIFFUSION_SUPPORTED: frozenset[str] = frozenset(_ARRHENIUS.keys())

#: Per-matrix supported sets. Key = matrix string from Arrhenius JSON.
_PER_MATRIX: dict[str, frozenset[str]] = {}
for _el, _entry in _ARRHENIUS.items():
    _mat = _entry.get("matrix", "unknown")
    _PER_MATRIX.setdefault(_mat, set()).add(_el)
_PER_MATRIX = {m: frozenset(s) for m, s in _PER_MATRIX.items()}

#: Elements that appear in alloy compositions / presets but have no Arrhenius
#: constants.  Useful for visualisation, CIF workflows, and composition display.
#: This is not exhaustive — it covers elements present in current presets.
VISUALISATION_ONLY: frozenset[str] = frozenset({
    # Ni-base superalloy elements
    "Ni", "Co", "Mo", "Al", "Ti", "Nb", "W", "Ta", "Re", "Hf",
    # Al-alloy elements
    "Mg", "Zn", "Cu", "Mn", "Si", "Fe",
    # Steel alloying elements (not Cr/C/N which are diffusion-supported)
    "P", "S", "V", "B",
    # Ceramics / functional materials
    "Ba", "Sr", "La", "Ca", "O",
    # Common structural
    "Zr", "Li", "Na", "K",
})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_diffusion_supported(element: str) -> bool:
    """Return True if *element* has Arrhenius constants in the database."""
    return element in DIFFUSION_SUPPORTED


def get_supported_for_matrix(matrix: str) -> frozenset[str]:
    """Return the set of diffusion-supported elements for *matrix*.

    If *matrix* is not in the Arrhenius database (e.g. a future non-steel
    matrix), returns the full ``DIFFUSION_SUPPORTED`` set as a safe default
    so the caller can still validate.

    Parameters
    ----------
    matrix : str
        Arrhenius matrix string, e.g. ``"austenite_FeCrNi"``.
    """
    return _PER_MATRIX.get(matrix, DIFFUSION_SUPPORTED)


def unsupported_explanation(element: str) -> str:
    """Return a concise human-readable explanation for an unsupported element.

    Used in GUI warnings and error messages so users understand *why* an
    element cannot be simulated and what is needed to add support.

    Parameters
    ----------
    element : str
        Element symbol that is not in ``DIFFUSION_SUPPORTED``.
    """
    if element in DIFFUSION_SUPPORTED:
        return f"'{element}' is supported for diffusion simulation."
    return (
        f"'{element}' is not yet supported for diffusion simulation. "
        f"Currently supported: {sorted(DIFFUSION_SUPPORTED)}. "
        f"To add '{element}', validated D₀ and Qd Arrhenius constants "
        f"(with literature references) must be added to "
        f"nominal_drift/science/constants/arrhenius.json."
    )


def filter_to_supported(elements: list[str]) -> tuple[list[str], list[str]]:
    """Partition *elements* into (supported, unsupported) lists.

    Parameters
    ----------
    elements : list[str]
        Input element symbols.

    Returns
    -------
    supported : list[str]
        Elements present in ``DIFFUSION_SUPPORTED``, preserving input order.
    unsupported : list[str]
        Elements not in ``DIFFUSION_SUPPORTED``.
    """
    supported = [el for el in elements if el in DIFFUSION_SUPPORTED]
    unsupported = [el for el in elements if el not in DIFFUSION_SUPPORTED]
    return supported, unsupported
