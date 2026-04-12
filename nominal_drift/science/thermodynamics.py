"""
nominal_drift.science.thermodynamics
=====================================
Layer 2 — Thermodynamic context for Nominal Drift.

PURPOSE
-------
Provides equilibrium and semi-analytical thermodynamic context alongside the
Layer 1 Crank–Nicolson diffusion solver.  Two computation paths:

  Path A — pycalphad equilibrium (optional, user-supplied TDB)
      Full CALPHAD equilibrium via pycalphad.  Requires a thermodynamic
      database file (TDB) covering Fe-Cr-Ni-C-N (austenitic) or Fe-Cr-C-N
      (ferritic).  Not bundled due to licensing; drop a TDB into
      ``data/thermodynamics/`` and set the path in config.

  Path B — analytical screening (always computed, no TDB needed)
      Implements published open-literature thermodynamic rules and
      semi-analytical models:
        - Schaeffler–DeLong Cr_eq / Ni_eq → predicted matrix phase
        - M23C6 sensitization window (Hull 1973, Tedmon–Vermilyea–Rosolowski)
        - Sigma phase susceptibility (>21 wt% Cr+Mo+W rule, Sedriks 1996)
        - Carbon activity in austenite (Chipman analytical model)
        - Empirical martensite-start temperature (Andrews 1965)

SCOPE / LIMITATIONS
--------------------
Path A (pycalphad):
  * Only as accurate as the supplied TDB and pycalphad convergence.
  * Multi-element stainless TDB files with validated Fe-Cr-Ni-C-N parameters
    are NOT open-source; commercial TDBs (TCFE, TCSTEEL) or academic releases
    are required.
  * If TDB is missing or the system is not fully covered, Path A is skipped
    and only Path B results are returned.

Path B (analytical):
  * Based on empirical and semi-analytical correlations, NOT full CALPHAD.
  * Validity ranges are specified for each correlation.
  * Outputs are screening-level; do not use for design-critical decisions.

Public API
----------
``ThermodynamicResult``
    Frozen Pydantic model holding both Path A and Path B results + provenance.

``get_thermodynamic_context(composition_wt_pct, T_C, alloy_matrix, tdb_path)``
    Main entry point.  Returns a ThermodynamicResult.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Thermodynamic constants
# ---------------------------------------------------------------------------

R_J_MOL_K: float = 8.31446  # J mol⁻¹ K⁻¹

# Molar masses (g mol⁻¹)
MOLAR_MASS: Dict[str, float] = {
    "FE": 55.847, "CR": 51.996, "NI": 58.69,
    "MO": 95.94,  "MN": 54.938, "SI": 28.086,
    "C":  12.011, "N":  14.007, "TI": 47.88,
    "NB": 92.906, "CU": 63.546, "CO": 58.933,
}

# ---------------------------------------------------------------------------
# Semi-analytical screening models (Path B)
# ---------------------------------------------------------------------------

def _cr_eq_ni_eq(comp: Dict[str, float]) -> Tuple[float, float]:
    """
    Delong (1974) Cr-equivalent and Ni-equivalent for weld metal / wrought alloys.

    Cr_eq = Cr + Mo + 1.5·Si + 0.5·Nb
    Ni_eq = Ni + 30·C + 0.5·Mn + 30·N

    Returns (Cr_eq, Ni_eq) in wt-% equivalents.
    Validity: stainless steels, 10–30 wt% Cr, 0–25 wt% Ni.
    """
    c = {k.upper(): v for k, v in comp.items()}
    cr_eq = (c.get("CR", 0) + c.get("MO", 0)
             + 1.5 * c.get("SI", 0)
             + 0.5 * c.get("NB", 0))
    ni_eq = (c.get("NI", 0)
             + 30.0 * c.get("C", 0)
             + 0.5  * c.get("MN", 0)
             + 30.0 * c.get("N", 0))
    return cr_eq, ni_eq


def _predict_matrix_phase(cr_eq: float, ni_eq: float) -> Tuple[str, str]:
    """
    Predict equilibrium matrix phase from DeLong / Schaeffler diagram.

    Returns (phase_label, confidence_note).
    Boundaries are approximate; martensite region not fully represented.
    """
    # Schaeffler boundaries (approximate, not a replacement for full CALPHAD)
    if ni_eq >= cr_eq * 0.7 - 1.0 and cr_eq < 18:
        phase, note = "Martensite", "Near austenite-martensite boundary"
    elif ni_eq >= cr_eq * 0.2 + 4.0:
        phase, note = "Austenite (FCC_A1)", "Well within austenite field"
    elif cr_eq > 22 and ni_eq < 5:
        phase, note = "Ferrite (BCC_A2)", "High-Cr ferritic region"
    elif ni_eq < cr_eq * 0.2 + 2.0:
        phase, note = "Ferrite (BCC_A2)", "Low Ni-eq, ferritic tendency"
    else:
        phase, note = "Austenite + Ferrite (duplex)", "Duplex field"
    return phase, note


def _m23c6_sensitization_window(
    T_C: float,
    cr_wt_pct: float,
    c_wt_pct: float,
) -> Tuple[bool, str]:
    """
    Empirical M23C6 precipitation susceptibility window.

    Based on:
    - Hull (1973) solubility product approach for M23C6 in austenite
    - Tedmon, Vermilyea & Rosolowski (1971) sensitization kinetics
    - ASTM A262 sensitization temperature range (425–870°C, peak 650–750°C)

    The solubility product for M23C6 in austenite is approximately:
      log[Cr]^23 · [C]^6 = A - B/T  (simplified Hillert form)
    Here we use empirical bounds.

    Returns (is_susceptible: bool, explanation: str).
    Validity: 12–26 wt% Cr, 0.01–0.10 wt% C, 400–900°C.
    """
    # Solubility limit of Cr in M23C6 (empirical)
    # C content check: below ~0.003 wt% C, M23C6 is unlikely
    LOWER_T_C = 425.0    # onset of precipitation kinetics
    UPPER_T_C = 870.0    # dissolution temperature (approx)
    PEAK_LOW  = 620.0
    PEAK_HIGH = 780.0

    if c_wt_pct < 0.003:
        return False, (
            f"C = {c_wt_pct:.4f} wt% is below ~0.003 wt% — M23C6 precipitation "
            "unlikely regardless of temperature."
        )

    if T_C < LOWER_T_C:
        return False, (
            f"T = {T_C:.0f} °C is below the M23C6 kinetic onset (~425 °C). "
            "Precipitation kinetically suppressed."
        )

    if T_C > UPPER_T_C:
        return False, (
            f"T = {T_C:.0f} °C is above the M23C6 dissolution temperature (~870 °C). "
            "M23C6 dissolves at this temperature."
        )

    # Approximate solubility product: log10([Cr_23][C_6]) > threshold
    # Simplified: precipitation requires Cr·C^(6/23) > empirical constant
    # This is a screening test — not a substitute for CALPHAD.
    cr_c_product = cr_wt_pct * (c_wt_pct ** (6.0 / 23.0))
    threshold = 1.5  # empirical, approximate

    if cr_c_product < threshold:
        return False, (
            f"Cr × C^(6/23) = {cr_c_product:.3f} < {threshold} "
            "— below approximate M23C6 solubility threshold."
        )

    if PEAK_LOW <= T_C <= PEAK_HIGH:
        return True, (
            f"T = {T_C:.0f} °C is in the peak sensitization window ({PEAK_LOW:.0f}–"
            f"{PEAK_HIGH:.0f} °C). M23C6 precipitation is thermodynamically "
            f"favoured and kinetically fast. Cr = {cr_wt_pct:.1f} wt%, "
            f"C = {c_wt_pct:.4f} wt%."
        )

    return True, (
        f"T = {T_C:.0f} °C is within the sensitization window ({LOWER_T_C:.0f}–"
        f"{UPPER_T_C:.0f} °C). M23C6 thermodynamically accessible. Precipitation "
        f"rate depends on prior thermal history and grain-boundary density."
    )


def _sigma_phase_susceptibility(
    cr_wt_pct: float,
    mo_wt_pct: float,
    si_wt_pct: float,
    T_C: float,
) -> Tuple[bool, str]:
    """
    Sigma phase formation susceptibility based on Sedriks (1996).

    Rule of thumb: sigma phase forms when Cr + Mo + W > ~21 wt%,
    typically in the 600–900 °C range for austenitic steels.

    Returns (susceptible: bool, note: str).
    """
    cr_mo_equiv = cr_wt_pct + mo_wt_pct + 1.5 * si_wt_pct
    sigma_T_low  = 600.0
    sigma_T_high = 900.0

    in_T_range = sigma_T_low <= T_C <= sigma_T_high

    if cr_mo_equiv >= 21.0 and in_T_range:
        return True, (
            f"Cr + Mo + 1.5·Si = {cr_mo_equiv:.1f} wt% > 21 wt% threshold. "
            f"Sigma phase precipitation is thermodynamically accessible at "
            f"{T_C:.0f} °C. Sigma formation is slow in austenite; long hold times "
            f"or cold-work accelerate it."
        )
    elif cr_mo_equiv >= 21.0 and not in_T_range:
        return False, (
            f"Cr + Mo + 1.5·Si = {cr_mo_equiv:.1f} wt% > 21 wt% but T = {T_C:.0f} °C "
            f"is outside sigma formation window ({sigma_T_low:.0f}–{sigma_T_high:.0f} °C)."
        )
    else:
        return False, (
            f"Cr + Mo + 1.5·Si = {cr_mo_equiv:.1f} wt% < 21 wt% — "
            "sigma phase not expected for this composition."
        )


def _carbon_activity(
    T_C: float,
    cr_wt_pct: float,
    c_wt_pct: float,
    ni_wt_pct: float = 0.0,
) -> Tuple[float, str]:
    """
    Estimate carbon activity relative to graphite in Fe-Cr-Ni-C austenite.

    Uses the Chipman (1952) model extended by Hillert & Staffansson (1970)
    for Cr effect on C activity in austenite:

      ln(a_C) = ln(x_C) + (E_Cr / RT) · x_Cr + (E_Ni / RT) · x_Ni

    Interaction parameters (kJ/mol):
      E_Cr = -68.0  (Cr strongly lowers C activity — stabilises M23C6)
      E_Ni = +8.0   (Ni slightly raises C activity)

    Returns (a_C, note).
    NOTE: This is a first-order estimate. Accuracy ±30% for typical SS.
    """
    T_K = T_C + 273.15
    mm = MOLAR_MASS

    # Convert wt% to mole fraction (approximate, Fe balance)
    # Only key elements for this calculation
    wt = {
        "C": c_wt_pct, "CR": cr_wt_pct, "NI": ni_wt_pct,
        "FE": max(0.0, 100.0 - cr_wt_pct - ni_wt_pct - c_wt_pct),
    }
    moles = {el: wt[el] / mm[el] for el in wt if wt[el] > 0}
    tot = sum(moles.values())
    x = {el: moles[el] / tot for el in moles}

    x_C  = x.get("C", 1e-9)
    x_Cr = x.get("CR", 0.0)
    x_Ni = x.get("NI", 0.0)

    E_Cr = -68000.0  # J mol⁻¹
    E_Ni =   8000.0  # J mol⁻¹

    ln_aC = math.log(x_C) + (E_Cr / (R_J_MOL_K * T_K)) * x_Cr + \
            (E_Ni / (R_J_MOL_K * T_K)) * x_Ni
    a_C = math.exp(ln_aC)

    note = (
        f"x_C = {x_C:.5f}, x_Cr = {x_Cr:.4f}, x_Ni = {x_Ni:.4f}. "
        f"Chipman–Hillert model (±30% accuracy). "
        f"Cr lowers C activity (E_Cr = −68 kJ/mol); "
        f"this drives C toward M23C6 at grain boundaries."
    )
    return a_C, note


def _ms_temperature(comp: Dict[str, float]) -> Tuple[Optional[float], str]:
    """
    Andrews (1965) martensite-start temperature for steels.

    Ms(°C) = 539 − 423·C − 30.4·Mn − 17.7·Ni − 12.1·Cr − 7.5·Mo

    Valid for: C < 0.6 wt%, Cr < 18 wt%, Ni < 10 wt%.
    Returns (Ms_C, note).
    """
    c = {k.upper(): v for k, v in comp.items()}
    Ms = (539.0
          - 423.0 * c.get("C", 0)
          - 30.4  * c.get("MN", 0)
          - 17.7  * c.get("NI", 0)
          - 12.1  * c.get("CR", 0)
          - 7.5   * c.get("MO", 0))
    note = "Andrews (1965) empirical formula. Valid for C < 0.6 wt%, Cr < 18 wt%."
    if Ms < -200:
        return None, f"Ms = {Ms:.0f} °C (far below room temperature — austenite thermally stable)."
    return Ms, note


# ---------------------------------------------------------------------------
# pycalphad Path A (optional)
# ---------------------------------------------------------------------------

def _wt_to_mole_fractions(
    comp_wt: Dict[str, float],
    elements_to_include: List[str],
) -> Dict[str, float]:
    """Convert wt% dict to mole fractions for the listed elements (plus VA=0)."""
    mm = MOLAR_MASS
    moles = {}
    for el in elements_to_include:
        key = el.upper()
        wt = comp_wt.get(el, comp_wt.get(key, 0.0))
        if wt > 0 and key in mm:
            moles[key] = wt / mm[key]
    total = sum(moles.values())
    if total <= 0:
        raise ValueError("No valid elements for mole fraction conversion.")
    return {el: mol / total for el, mol in moles.items()}


def _run_pycalphad(
    comp_wt: Dict[str, float],
    T_C: float,
    tdb_path: str,
    phases_to_check: Optional[List[str]] = None,
) -> Dict:
    """
    Run pycalphad equilibrium.  Returns a dict with:
      phases: list of (phase_name, NP) tuples for phases with NP > 0.001
      GM_J_mol: Gibbs energy of the system
      tdb_used: path used
      elements: list of elements in calculation
      error: None or error string
    """
    try:
        from pycalphad import Database, equilibrium, variables as v
        import numpy as np
    except ImportError:
        return {"error": "pycalphad not installed", "phases": [], "tdb_used": tdb_path}

    if not os.path.isfile(tdb_path):
        return {"error": f"TDB file not found: {tdb_path}", "phases": [], "tdb_used": tdb_path}

    try:
        db = Database(tdb_path)
        db_elements = set(db.elements)

        # Determine which elements we can use from the composition
        available = []
        for el in ["FE", "CR", "NI", "C", "N", "MO", "MN", "SI", "VA"]:
            if el in db_elements and (
                el == "VA"
                or comp_wt.get(el, comp_wt.get(el.lower(), 0)) > 0
                or el == "FE"
            ):
                available.append(el)

        # Must have at least Fe + Cr
        if "FE" not in available or "CR" not in available:
            return {
                "error": "TDB does not contain Fe and Cr — not a steel database.",
                "phases": [], "tdb_used": tdb_path,
            }

        # Elements for equilibrium (exclude VA from X conditions)
        calc_elements = [e for e in available if e != "VA"]

        # Build conditions
        x_fracs = _wt_to_mole_fractions(comp_wt, calc_elements)
        conditions = {v.T: T_C + 273.15, v.P: 101325}
        for el in calc_elements:
            if el != "FE":  # Fe is balance
                xval = x_fracs.get(el, 0.0)
                if xval > 1e-6:
                    conditions[v.X(el)] = xval

        # Choose phases to check
        if phases_to_check is None:
            candidate = ["FCC_A1", "BCC_A2", "M23C6", "M7C3", "SIGMA",
                         "CEMENTITE", "HCP_A3", "LIQUID"]
        else:
            candidate = phases_to_check

        phases_in_db = [p for p in candidate if p in db.phases]
        if not phases_in_db:
            return {
                "error": "None of the requested phases found in TDB.",
                "phases": [], "tdb_used": tdb_path,
            }

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = equilibrium(db, available, phases_in_db, conditions)

        # Extract phase fractions
        phase_arr = result.Phase.values.flatten()
        np_arr    = result.NP.values.flatten()
        gm_val    = float(result.GM.values.flatten()[0])

        phase_fracs = [
            (str(ph), float(np))
            for ph, np in zip(phase_arr, np_arr)
            if str(ph) and str(np) != "nan" and float(np) > 0.001
        ]

        return {
            "phases": phase_fracs,
            "GM_J_mol": gm_val,
            "tdb_used": tdb_path,
            "elements": available,
            "error": None,
        }

    except Exception as exc:
        return {
            "error": f"pycalphad error: {exc}",
            "phases": [], "tdb_used": tdb_path,
        }


# ---------------------------------------------------------------------------
# Public result model
# ---------------------------------------------------------------------------

class AnalyticalScreening(BaseModel):
    """Results from the analytical thermodynamic screening (Path B)."""

    model_config = {"frozen": True}

    # Matrix phase prediction
    cr_equivalent_wt_pct: float
    ni_equivalent_wt_pct: float
    predicted_matrix_phase: str
    matrix_phase_note: str

    # M23C6 sensitization
    m23c6_susceptible: bool
    m23c6_note: str

    # Sigma phase
    sigma_susceptible: bool
    sigma_note: str

    # Carbon activity
    carbon_activity: float
    carbon_activity_note: str

    # Martensite start (may be None for fully austenitic alloys)
    ms_temperature_C: Optional[float]
    ms_temperature_note: str

    # Sources cited
    references: List[str] = [
        "DeLong (1974) — Cr_eq / Ni_eq",
        "Hull (1973) — M23C6 sensitization window",
        "Tedmon, Vermilyea & Rosolowski (1971) — sensitization kinetics",
        "Sedriks (1996) — sigma phase formation rule",
        "Chipman (1952) / Hillert & Staffansson (1970) — C activity model",
        "Andrews (1965) — Ms temperature",
    ]


class CalchadResult(BaseModel):
    """Results from the pycalphad equilibrium calculation (Path A)."""

    model_config = {"frozen": True}

    phases_present: List[Tuple[str, float]]   # (name, mole_fraction)
    gm_j_mol: Optional[float]
    tdb_path: str
    elements_used: List[str]
    error: Optional[str]      # None = success, str = error message
    available: bool           # True if calculation completed successfully


class ThermodynamicResult(BaseModel):
    """Complete Layer 2 thermodynamic context for one composition + temperature."""

    model_config = {"frozen": True}

    # Inputs (echoed for provenance)
    composition_wt_pct: Dict[str, float]
    temperature_C: float
    alloy_matrix: str           # 'austenite_FeCrNi' or 'ferrite_FeCr'

    # Path A — pycalphad (may be unavailable)
    calphad: Optional[CalchadResult]

    # Path B — always computed
    analytical: AnalyticalScreening

    # Human-readable summary
    layer2_summary: str

    # Limitations block — always shown
    limitations: List[str] = [
        "Analytical models are screening-level — not design-critical tools.",
        "pycalphad results depend entirely on the TDB file supplied; "
        "no TDB is bundled due to licensing.",
        "C/N interstitial effects on phase stability are first-order estimates.",
        "Multi-element interaction parameters (Mo, Mn, Si, Ti) are approximated.",
        "Kinetic effects (precipitation incubation, coarsening) are NOT modelled.",
        "This module does NOT replace a qualified CALPHAD assessment.",
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_thermodynamic_context(
    composition_wt_pct: Dict[str, float],
    T_C: float,
    alloy_matrix: str = "austenite_FeCrNi",
    tdb_path: Optional[str] = None,
) -> ThermodynamicResult:
    """
    Compute Layer 2 thermodynamic context for a given alloy composition and
    heat-treatment temperature.

    Parameters
    ----------
    composition_wt_pct : dict
        Elemental composition in wt%.  Keys: any subset of Fe, Cr, Ni, C, N,
        Mo, Mn, Si, Ti, Nb.
    T_C : float
        Temperature in °C.
    alloy_matrix : str
        ``'austenite_FeCrNi'`` or ``'ferrite_FeCr'``.
    tdb_path : str or None
        Path to a TDB file for pycalphad.  If None, the default search path
        (``data/thermodynamics/<matrix>.tdb``) is tried.  If not found, the
        pycalphad path is skipped.

    Returns
    -------
    ThermodynamicResult
    """
    comp = {k.upper(): float(v) for k, v in composition_wt_pct.items()}

    # -----------------------------------------------------------------------
    # Path B — analytical screening (always runs)
    # -----------------------------------------------------------------------
    cr_eq, ni_eq = _cr_eq_ni_eq(comp)
    matrix_phase, matrix_note = _predict_matrix_phase(cr_eq, ni_eq)

    m23_susc, m23_note = _m23c6_sensitization_window(
        T_C,
        comp.get("CR", 0.0),
        comp.get("C", 0.0),
    )

    sigma_susc, sigma_note = _sigma_phase_susceptibility(
        comp.get("CR", 0.0),
        comp.get("MO", 0.0),
        comp.get("SI", 0.0),
        T_C,
    )

    a_C, a_C_note = _carbon_activity(
        T_C,
        comp.get("CR", 0.0),
        comp.get("C", 0.0),
        comp.get("NI", 0.0),
    )

    Ms, Ms_note = _ms_temperature(comp)

    analytical = AnalyticalScreening(
        cr_equivalent_wt_pct=round(cr_eq, 2),
        ni_equivalent_wt_pct=round(ni_eq, 2),
        predicted_matrix_phase=matrix_phase,
        matrix_phase_note=matrix_note,
        m23c6_susceptible=m23_susc,
        m23c6_note=m23_note,
        sigma_susceptible=sigma_susc,
        sigma_note=sigma_note,
        carbon_activity=round(a_C, 6),
        carbon_activity_note=a_C_note,
        ms_temperature_C=round(Ms, 1) if Ms is not None else None,
        ms_temperature_note=Ms_note,
    )

    # -----------------------------------------------------------------------
    # Path A — pycalphad (optional)
    # -----------------------------------------------------------------------
    calphad: Optional[CalchadResult] = None
    resolved_tdb = _resolve_tdb_path(tdb_path, alloy_matrix)

    if resolved_tdb is not None:
        raw = _run_pycalphad(comp, T_C, resolved_tdb)
        calphad = CalchadResult(
            phases_present=raw.get("phases", []),
            gm_j_mol=raw.get("GM_J_mol"),
            tdb_path=resolved_tdb,
            elements_used=raw.get("elements", []),
            error=raw.get("error"),
            available=raw.get("error") is None,
        )

    # -----------------------------------------------------------------------
    # Build summary
    # -----------------------------------------------------------------------
    flags = []
    if m23_susc:
        flags.append("⚠ M23C6 sensitization thermodynamically accessible")
    if sigma_susc:
        flags.append("⚠ Sigma phase accessible (slow kinetics)")
    if not flags:
        flags.append("✓ No major precipitation phases predicted at this condition")

    summary = (
        f"T = {T_C:.0f} °C | "
        f"Cr_eq = {cr_eq:.1f} wt%, Ni_eq = {ni_eq:.1f} wt% | "
        f"Predicted matrix: {matrix_phase}. "
        + "  ".join(flags)
    )

    return ThermodynamicResult(
        composition_wt_pct={k: float(v) for k, v in comp.items()},
        temperature_C=float(T_C),
        alloy_matrix=alloy_matrix,
        calphad=calphad,
        analytical=analytical,
        layer2_summary=summary,
    )


def _resolve_tdb_path(
    tdb_path: Optional[str],
    alloy_matrix: str,
) -> Optional[str]:
    """
    Resolve a TDB file path:
      1. Use tdb_path if provided and exists.
      2. Check project-local data/thermodynamics/<matrix>.tdb
      3. Check data/thermodynamics/steel.tdb (generic fallback)
      4. Return None if nothing found.
    """
    if tdb_path is not None:
        if os.path.isfile(tdb_path):
            return tdb_path
        # Path provided but not found — warn, don't silently use another
        return tdb_path  # _run_pycalphad will report file-not-found

    # Auto-discover in project data/ directory
    project_root = Path(__file__).parent.parent.parent
    search_dirs = [
        project_root / "data" / "thermodynamics",
        project_root / "data",
    ]
    candidates = []
    if "ferritic" in alloy_matrix or "ferrite" in alloy_matrix:
        candidates = ["ferritic.tdb", "ferrite.tdb", "steel.tdb", "fecrcn.tdb"]
    else:
        candidates = ["austenitic.tdb", "steel.tdb", "fecrnin.tdb", "fecrcni.tdb"]

    for d in search_dirs:
        for fname in candidates:
            p = d / fname
            if p.is_file():
                return str(p)

    return None  # no TDB found — pycalphad path skipped
