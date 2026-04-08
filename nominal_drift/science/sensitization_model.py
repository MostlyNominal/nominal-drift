"""
nominal_drift.science.sensitization_model
==========================================
First-order engineering assessment of sensitization-related depletion
behaviour in austenitic alloys.

PURPOSE
-------
This module bridges the raw diffusion solver outputs (``DiffusionOutput``)
and higher-level metallurgical interpretation.  It is the first step toward
a coupled C–N–Cr sensitization reasoning layer.

Sprint 2B scope:
  * Cr-only assessment (required)
  * Optional C-output (mechanism labelling, not kinetics)
  * Optional N-output (mechanism labelling, not kinetics)
  * Risk classification: low / moderate / high
  * Transparent assumptions and caveats on every output

WHAT THIS MODULE IS NOT
-----------------------
* Not a CALPHAD engine
* Not a carbide nucleation or growth model
* Not a validated EPR / DLEPR electrochemical predictor
* Not a grain-growth or recrystallisation model
* Not a TTT / CCT calculator

All outputs include explicit assumptions lists so downstream users
(LLM narration, reports, dashboards) know exactly what was and was not
modelled.

DESIGN PRINCIPLES
-----------------
* Rule-based, transparent, easy to audit
* All thresholds are named module constants — no magic numbers in logic
* C and N outputs inform mechanism labelling only; they do NOT directly
  alter the risk level in this first-order model (honest uncertainty)
* Solver warnings are propagated from all input DiffusionOutputs
* Deterministic: same inputs → same output, always

Public API
----------
``SensitizationAssessment``
    Pydantic v2 frozen model representing a complete assessment result.

``evaluate_sensitization(cr_output, c_output, n_output, c_threshold_wt_pct)``
    Consume one or more DiffusionOutput objects and return a
    ``SensitizationAssessment``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from nominal_drift.schemas.diffusion_output import DiffusionOutput

# ---------------------------------------------------------------------------
# Module-level engineering constants
# ---------------------------------------------------------------------------

#: Depletion depth (nm) at or above which risk is classified as "high".
#: Below this threshold, depletion is "moderate" when Cr < c_threshold.
_DEPTH_HIGH_RISK_NM: float = 50.0

#: Controlled vocabulary for risk levels.
_RISK_LEVELS: frozenset[str] = frozenset({"low", "moderate", "high"})

#: Controlled vocabulary for mechanism labels.
_MECHANISM_LABELS: frozenset[str] = frozenset({
    "Cr depletion only",
    "C-assisted Cr depletion",
    "N-assisted Cr depletion",
    "mixed-species indication",
    "undetermined",
})

# ---------------------------------------------------------------------------
# Fixed assumption strings (always included on every assessment output)
# ---------------------------------------------------------------------------

_BASE_ASSUMPTIONS: list[str] = [
    (
        "Cr depletion below c_threshold_wt_pct is the primary sensitization "
        "indicator (first-order engineering criterion only)."
    ),
    (
        "c_threshold_wt_pct is a simplified engineering parameter — it is not "
        "a validated electrochemical sensitization criterion and varies with "
        "alloy system, grain boundary character, and test method."
    ),
    (
        "Depletion depth is taken from the final stored simulation profile; "
        "intermediate profiles are not re-evaluated here."
    ),
    (
        "C and N DiffusionOutput objects, when provided, are used solely for "
        "mechanism labelling.  No carbide, nitride, or interstitial-precipitation "
        "kinetics are modelled in this first-order assessment."
    ),
    (
        "Risk level is derived from Cr criterion alone.  C and N species "
        "do not directly alter risk level — this is a conservative choice "
        "pending Sprint 2C+ coupled-species models."
    ),
    (
        "This assessment does not account for grain boundary chemistry, "
        "grain size, prior cold-work, texture, or sensitisation-recovery "
        "heat treatments."
    ),
    (
        "This output is a first-order engineering interpretation and is not a "
        "substitute for experimental measurement (EPR, DLEPR, ASTM A262, "
        "or equivalent electrochemical / metallographic testing)."
    ),
]

_BASE_NOTES: list[str] = [
    (
        "For higher-fidelity assessment, consider coupled C–Cr–N modelling "
        "with precipitation kinetics (planned for Sprint 2C+)."
    ),
    (
        "Risk classification is intentionally conservative: the 'high' band "
        "encompasses a wide range of practical sensitivities and should not "
        "be interpreted as a guaranteed failure prediction."
    ),
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class SensitizationAssessment(BaseModel, frozen=True):
    """Result of a first-order sensitization assessment.

    Attributes
    ----------
    mechanism_label : str
        Controlled-vocabulary label describing the dominant depletion
        mechanism based on the species considered.
        One of: ``"Cr depletion only"``, ``"C-assisted Cr depletion"``,
        ``"N-assisted Cr depletion"``, ``"mixed-species indication"``,
        ``"undetermined"``.
    risk_level : str
        Qualitative engineering risk classification.
        One of: ``"low"``, ``"moderate"``, ``"high"``.
    min_cr_wt_pct : float
        Minimum Cr concentration anywhere in the simulated domain [wt%].
        Taken directly from ``cr_output.min_concentration_wt_pct``.
    depletion_depth_nm : float | None
        Cr depletion depth [nm] from the final profile.
        ``None`` if no depletion front was detected by the solver.
    species_considered : list[str]
        Element symbols for which DiffusionOutput objects were supplied,
        e.g. ``["Cr"]`` or ``["Cr", "C", "N"]``.
    cr_threshold_wt_pct : float
        The Cr threshold used for this assessment (echoes
        ``c_threshold_wt_pct`` argument for reproducibility).
    assumptions : list[str]
        Explicit model assumptions.  Always non-empty.
    warnings : list[str]
        Solver-level warnings propagated from all input DiffusionOutputs,
        plus any assessment-level warnings.  May be empty if all inputs
        ran cleanly and no edge cases were detected.
    notes : list[str]
        Contextual notes about the assessment.  Always non-empty.
    """

    mechanism_label:     str
    risk_level:          str
    min_cr_wt_pct:       float
    depletion_depth_nm:  Optional[float]
    species_considered:  list[str]
    cr_threshold_wt_pct: float
    assumptions:         list[str]
    warnings:            list[str]
    notes:               list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_risk(
    min_cr_wt_pct: float,
    depletion_depth_nm: Optional[float],
    threshold: float,
) -> str:
    """Return 'low', 'moderate', or 'high' based on Cr criterion.

    Rules (transparent, easy to audit):

    1. If min Cr is at or above the threshold → no significant depletion
       → **low**.
    2. If min Cr is below the threshold AND depletion depth is absent or
       shallower than ``_DEPTH_HIGH_RISK_NM`` → depletion is present but
       limited in extent → **moderate**.
    3. If min Cr is below the threshold AND depletion depth reaches or
       exceeds ``_DEPTH_HIGH_RISK_NM`` nm → depletion is significant in
       both magnitude and extent → **high**.
    """
    if min_cr_wt_pct >= threshold:
        return "low"

    # Cr is below threshold — check depletion depth
    if depletion_depth_nm is None or depletion_depth_nm < _DEPTH_HIGH_RISK_NM:
        return "moderate"

    return "high"


def _assign_mechanism(
    risk_level: str,
    has_c: bool,
    has_n: bool,
) -> str:
    """Return a controlled-vocabulary mechanism label.

    Rules:
    * When risk is 'low', label is always "Cr depletion only" — we do
      not claim C or N assistance when no significant depletion is present.
    * For moderate/high risk, label depends on which species were supplied:
        - Cr only          → "Cr depletion only"
        - Cr + C           → "C-assisted Cr depletion"
        - Cr + N           → "N-assisted Cr depletion"
        - Cr + C + N       → "mixed-species indication"
    """
    if risk_level == "low":
        return "Cr depletion only"

    if has_c and has_n:
        return "mixed-species indication"
    if has_c:
        return "C-assisted Cr depletion"
    if has_n:
        return "N-assisted Cr depletion"
    return "Cr depletion only"


def _collect_warnings(
    cr_output: DiffusionOutput,
    c_output: Optional[DiffusionOutput],
    n_output: Optional[DiffusionOutput],
    has_c: bool,
    has_n: bool,
    risk_level: str,
) -> list[str]:
    """Build the combined warnings list from all inputs."""
    warnings: list[str] = []

    # Propagate solver-level warnings
    warnings.extend(cr_output.warnings)

    if c_output is not None:
        warnings.extend(f"[C diffusion] {w}" for w in c_output.warnings)

    if n_output is not None:
        warnings.extend(f"[N diffusion] {w}" for w in n_output.warnings)

    # Assessment-level caveats when multi-species data is present
    if (has_c or has_n) and risk_level != "low":
        warnings.append(
            "C and/or N outputs are present but carbide/nitride precipitation "
            "kinetics are NOT modelled — mechanism label may under-represent "
            "actual microstructural complexity."
        )

    return warnings


def _build_notes(
    has_c: bool,
    has_n: bool,
    risk_level: str,
) -> list[str]:
    """Return contextual notes specific to this assessment."""
    notes: list[str] = list(_BASE_NOTES)

    if not has_c and not has_n and risk_level != "low":
        notes.append(
            "Only Cr output was supplied.  If carbon or nitrogen content "
            "is significant for this alloy, consider re-running with C and/or "
            "N DiffusionOutput objects to refine the mechanism label."
        )

    if has_c and not has_n and risk_level != "low":
        notes.append(
            "C output supplied.  N was not supplied — potential N-driven "
            "Cr₂N precipitation effects are not considered."
        )

    if has_n and not has_c and risk_level != "low":
        notes.append(
            "N output supplied.  C was not supplied — potential M₂₃C₆ "
            "carbide-driven depletion effects are not considered."
        )

    return notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_sensitization(
    *,
    cr_output: DiffusionOutput,
    c_output: Optional[DiffusionOutput] = None,
    n_output: Optional[DiffusionOutput] = None,
    c_threshold_wt_pct: float = 12.0,
) -> SensitizationAssessment:
    """Evaluate the sensitization risk from one or more diffusion outputs.

    This is a **first-order engineering assessment** only.  See the module
    docstring and ``SensitizationAssessment.assumptions`` for a full list
    of what is and is not modelled.

    Parameters
    ----------
    cr_output : DiffusionOutput
        Diffusion result for chromium (required).  Must have
        ``element == "Cr"`` is recommended but not enforced — the function
        will use whatever element is labelled in the output.
    c_output : DiffusionOutput | None
        Diffusion result for carbon (optional).  Used for mechanism
        labelling only.
    n_output : DiffusionOutput | None
        Diffusion result for nitrogen (optional).  Used for mechanism
        labelling only.
    c_threshold_wt_pct : float
        Cr concentration below which sensitization risk is considered
        present.  Default: 12.0 wt% (typical engineering criterion for
        304/316L in the 550–850 °C sensitization range).

    Returns
    -------
    SensitizationAssessment
        Frozen Pydantic model containing risk level, mechanism label,
        species considered, assumptions, warnings, and notes.
    """
    has_c = c_output is not None
    has_n = n_output is not None

    # ------------------------------------------------------------------ #
    # 1.  Core Cr metrics                                                  #
    # ------------------------------------------------------------------ #
    min_cr  = cr_output.min_concentration_wt_pct
    depth   = cr_output.depletion_depth_nm

    # ------------------------------------------------------------------ #
    # 2.  Risk classification (Cr criterion only)                          #
    # ------------------------------------------------------------------ #
    risk = _classify_risk(min_cr, depth, c_threshold_wt_pct)

    # ------------------------------------------------------------------ #
    # 3.  Mechanism label                                                  #
    # ------------------------------------------------------------------ #
    mechanism = _assign_mechanism(risk, has_c, has_n)

    # ------------------------------------------------------------------ #
    # 4.  Species list                                                     #
    # ------------------------------------------------------------------ #
    species: list[str] = [cr_output.element]
    if has_c:
        species.append(c_output.element)  # type: ignore[union-attr]
    if has_n:
        species.append(n_output.element)  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # 5.  Warnings and notes                                               #
    # ------------------------------------------------------------------ #
    warnings = _collect_warnings(cr_output, c_output, n_output, has_c, has_n, risk)
    notes    = _build_notes(has_c, has_n, risk)

    return SensitizationAssessment(
        mechanism_label=mechanism,
        risk_level=risk,
        min_cr_wt_pct=min_cr,
        depletion_depth_nm=depth,
        species_considered=species,
        cr_threshold_wt_pct=c_threshold_wt_pct,
        assumptions=list(_BASE_ASSUMPTIONS),
        warnings=warnings,
        notes=notes,
    )
