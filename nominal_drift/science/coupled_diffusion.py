"""
nominal_drift.science.coupled_diffusion
=========================================
First-order engineering layer for coupled multi-species diffusion effects.

PURPOSE
-------
Extends the Sprint 1 single-species solver toward a first-order interaction
model between Cr, C, and N.  The core scientific observation is:

  * Carbon forms Cr-rich M₂₃C₆ carbides at grain boundaries, consuming
    chromium and deepening the Cr-depleted zone beyond what pure Cr
    diffusion alone predicts.

  * Nitrogen forms Cr₂N nitrides by an analogous mechanism, also consuming
    chromium at the boundary.

  * In alloys containing both C and N, both effects are present and can be
    treated as additive at this first-order level (no C–N cross-coupling
    beyond the shared Cr pool).

COUPLING APPROACH
-----------------
The coupling is implemented as an **adjusted minimum Cr concentration**
rather than re-running the diffusion solver.  This is deliberate: Sprint 3A
is intentionally additive — it interprets existing solver outputs, not
replaces them.

The coupling coefficients λ_C and λ_N encode the stoichiometric + kinetic
efficiency of precipitation:

    additional_Cr_from_C = λ_C × C_bulk_C   [wt%]
    additional_Cr_from_N = λ_N × C_bulk_N   [wt%]

Default values are engineering estimates derived from M₂₃C₆ and Cr₂N
stoichiometry with a conservative 30% precipitation efficiency:

    M₂₃C₆ stoichiometric Cr/C ratio ≈ 16.6  →  λ_C = 5.0 (30% efficiency)
    Cr₂N  stoichiometric Cr/N ratio ≈  7.4  →  λ_N = 2.5 (34% efficiency)

The effective minimum Cr is bounded below by the physical Cr sink BC — we
cannot deplete below the grain-boundary equilibrium set by the Dirichlet
condition.

WHAT THIS MODULE IS NOT
-----------------------
* Not a CALPHAD solver
* Not a full carbide/nitride nucleation and growth model
* Not a kinetic Monte-Carlo or DICTRA replacement
* Not a validated CCT / TTT calculator

Every output carries an explicit ``assumptions`` list so downstream modules
(narration, reports, sensitization model) know exactly what was and was not
modelled.

Public API
----------
``LAMBDA_C``
    Default coupling coefficient for C → Cr depletion [wt% Cr / wt% C].

``LAMBDA_N``
    Default coupling coefficient for N → Cr depletion [wt% Cr / wt% N].

``CoupledDiffusionResult``
    Pydantic v2 frozen result model.

``evaluate_coupled_depletion(cr_output, c_output, n_output,
                              lambda_c, lambda_n)``
    Compute the first-order coupled depletion adjustment.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from nominal_drift.schemas.diffusion_output import DiffusionOutput

# ---------------------------------------------------------------------------
# Default coupling coefficients
# ---------------------------------------------------------------------------

#: Wt% of Cr depleted per wt% of C in the alloy.
#: Derived from M₂₃C₆ stoichiometry (Cr/C mass ratio ≈ 16.6) with a
#: conservative 30% precipitation efficiency: 16.6 × 0.30 ≈ 5.0.
#: Calibrate against experiment for specific alloy–temperature combinations.
LAMBDA_C: float = 5.0

#: Wt% of Cr depleted per wt% of N in the alloy.
#: Derived from Cr₂N stoichiometry (Cr/N mass ratio ≈ 7.4) with a
#: conservative 34% precipitation efficiency: 7.4 × 0.34 ≈ 2.5.
LAMBDA_N: float = 2.5

#: Warning threshold: unusually high coupling coefficient for C.
_LAMBDA_C_WARN: float = 12.0

#: Warning threshold: unusually high coupling coefficient for N.
_LAMBDA_N_WARN: float = 6.0

# ---------------------------------------------------------------------------
# Fixed assumption strings
# ---------------------------------------------------------------------------

_BASE_ASSUMPTIONS: list[str] = [
    (
        "C and N contributions to Cr depletion are estimated via first-order "
        "coupling coefficients (lambda_c, lambda_n) — not kinetic precipitation "
        "calculations."
    ),
    (
        "Default coefficients (LAMBDA_C=5.0, LAMBDA_N=2.5) are derived from "
        "M₂₃C₆ and Cr₂N stoichiometry at ~30% precipitation efficiency. "
        "Calibrate against experiment for specific alloy–temperature systems."
    ),
    (
        "C and N depletion contributions are treated as independent and additive. "
        "Cross-coupling between M₂₃C₆ and Cr₂N formation is not modelled."
    ),
    (
        "The effective minimum Cr is bounded below by the Cr sink boundary "
        "condition (C_sink_wt_pct from the Cr DiffusionOutput) — depletion "
        "cannot exceed the physical grain-boundary equilibrium."
    ),
    (
        "Temperature dependence of precipitation efficiency is not modelled in "
        "this first-order layer. Lambda coefficients are treated as constants."
    ),
    (
        "This is a first-order engineering estimate — not a CALPHAD, DICTRA, "
        "or full thermodynamic equilibrium calculation."
    ),
]

_BASE_NOTES: list[str] = [
    (
        "To re-calibrate lambda coefficients for a specific alloy system, "
        "fit against measured Cr depletion profiles with known C/N contents."
    ),
    (
        "For higher-fidelity coupled modelling, consider DICTRA-style "
        "multi-component diffusion (planned for a future sprint)."
    ),
]


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class CoupledDiffusionResult(BaseModel, frozen=True):
    """First-order coupled depletion assessment for a multi-species system.

    Attributes
    ----------
    cr_min_base_wt_pct : float
        Minimum Cr from the input ``cr_output`` alone (no coupling applied).
        Equals ``cr_output.min_concentration_wt_pct``.
    c_contribution_wt_pct : float
        Additional Cr depletion attributed to C-driven M₂₃C₆ formation.
        Zero when no C output is supplied.
    n_contribution_wt_pct : float
        Additional Cr depletion attributed to N-driven Cr₂N formation.
        Zero when no N output is supplied.
    cr_min_effective_wt_pct : float
        Adjusted minimum Cr after applying C and N contributions.
        Bounded below by ``cr_sink_floor_wt_pct`` (physical limit).
    cr_sink_floor_wt_pct : float
        Physical lower bound for Cr concentration — echoes
        ``cr_output.C_sink_wt_pct``.  Effective Cr cannot go below this.
    mechanism_components : list[str]
        Active coupling mechanisms in this assessment (controlled vocabulary).
    lambda_c_used : float
        The C coupling coefficient actually used for this computation.
    lambda_n_used : float
        The N coupling coefficient actually used for this computation.
    species_inputs : list[str]
        Element symbols for which DiffusionOutput objects were provided.
    assumptions : list[str]
        Explicit model assumptions.  Always non-empty.
    warnings : list[str]
        Propagated solver warnings + model-level caveats.  May be empty.
    notes : list[str]
        Contextual notes for downstream interpretation.
    """

    cr_min_base_wt_pct:   float
    c_contribution_wt_pct: float
    n_contribution_wt_pct: float
    cr_min_effective_wt_pct: float
    cr_sink_floor_wt_pct: float
    mechanism_components: list[str]
    lambda_c_used:        float
    lambda_n_used:        float
    species_inputs:       list[str]
    assumptions:          list[str]
    warnings:             list[str]
    notes:                list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_contribution(
    output: Optional[DiffusionOutput],
    lambda_coeff: float,
) -> float:
    """Return wt% Cr additionally depleted by one interstitial species.

    Uses the bulk concentration of the interstitial as the maximum available
    driver for precipitation.  Returns 0.0 when *output* is None.

    Parameters
    ----------
    output : DiffusionOutput | None
        Diffusion result for the interstitial species (C or N).
    lambda_coeff : float
        Coupling coefficient: wt% Cr depleted per wt% of species.
    """
    if output is None or lambda_coeff <= 0.0:
        return 0.0
    return lambda_coeff * output.C_bulk_wt_pct


#: Absolute physical floor for Cr concentration — can never go negative.
_CR_ABSOLUTE_FLOOR: float = 0.0


def _build_mechanism_components(
    has_c: bool,
    has_n: bool,
) -> list[str]:
    """Return the list of active coupling mechanism strings."""
    components = ["Cr boundary sink depletion (Dirichlet BC)"]
    if has_c:
        components.append(
            "C-driven M₂₃C₆ carbide precipitation (first-order λ_C estimate)"
        )
    if has_n:
        components.append(
            "N-driven Cr₂N nitride precipitation (first-order λ_N estimate)"
        )
    return components


def _collect_warnings(
    cr_output: DiffusionOutput,
    c_output: Optional[DiffusionOutput],
    n_output: Optional[DiffusionOutput],
    c_contribution: float,
    n_contribution: float,
    raw_effective: float,
    lambda_c: float,
    lambda_n: float,
) -> list[str]:
    """Build the combined warnings list."""
    warnings: list[str] = []

    # Propagate solver warnings
    warnings.extend(cr_output.warnings)
    if c_output is not None:
        warnings.extend(f"[C] {w}" for w in c_output.warnings)
    if n_output is not None:
        warnings.extend(f"[N] {w}" for w in n_output.warnings)

    # Unusually high coupling coefficients
    if lambda_c > _LAMBDA_C_WARN:
        warnings.append(
            f"lambda_c={lambda_c:.2f} exceeds the typical engineering range "
            f"(> {_LAMBDA_C_WARN}).  Verify the coefficient for this alloy system."
        )
    if lambda_n > _LAMBDA_N_WARN:
        warnings.append(
            f"lambda_n={lambda_n:.2f} exceeds the typical engineering range "
            f"(> {_LAMBDA_N_WARN}).  Verify the coefficient for this alloy system."
        )

    # Coupling saturated: contributions push below the physical floor
    if raw_effective < _CR_ABSOLUTE_FLOOR:
        warnings.append(
            "C/N coupling contributions exceed the base Cr sink concentration — "
            "effective Cr would be negative without physical clamping to zero. "
            "The lambda coefficients may be too high for this alloy system."
        )

    # Large total coupling relative to (C_bulk - C_sink) margin
    total = c_contribution + n_contribution
    margin = cr_output.C_bulk_wt_pct - cr_output.C_sink_wt_pct
    if total > 0.0 and margin > 0.0 and total > 0.5 * margin:
        warnings.append(
            "C + N coupling contributes more than 50% of the available "
            "(C_bulk − C_sink) depletion margin.  This estimate is highly "
            "sensitive to lambda coefficient calibration."
        )

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_coupled_depletion(
    *,
    cr_output: DiffusionOutput,
    c_output: Optional[DiffusionOutput] = None,
    n_output: Optional[DiffusionOutput] = None,
    lambda_c: float = LAMBDA_C,
    lambda_n: float = LAMBDA_N,
) -> CoupledDiffusionResult:
    """Compute the first-order coupled depletion adjustment for Cr, C, and N.

    This function interprets existing ``DiffusionOutput`` objects — it does
    **not** re-run the diffusion solver.  The coupling is implemented as an
    adjusted minimum Cr concentration that accounts for additional Cr consumed
    by carbide and nitride precipitation at the grain boundary.

    Parameters
    ----------
    cr_output : DiffusionOutput
        Diffusion result for chromium (required).
    c_output : DiffusionOutput | None
        Diffusion result for carbon (optional).  When provided, carbide-
        driven Cr depletion is estimated using ``lambda_c``.
    n_output : DiffusionOutput | None
        Diffusion result for nitrogen (optional).  When provided, nitride-
        driven Cr depletion is estimated using ``lambda_n``.
    lambda_c : float
        Coupling coefficient for C → Cr depletion [wt% Cr / wt% C].
        Default: ``LAMBDA_C`` (= 5.0).
    lambda_n : float
        Coupling coefficient for N → Cr depletion [wt% Cr / wt% N].
        Default: ``LAMBDA_N`` (= 2.5).

    Returns
    -------
    CoupledDiffusionResult
        Frozen result model with adjusted Cr minimum, species contributions,
        coupling mechanism list, assumptions, and warnings.
    """
    has_c = c_output is not None
    has_n = n_output is not None

    # ------------------------------------------------------------------ #
    # 1.  Base Cr metric                                                   #
    #     Use C_sink_wt_pct as the base: this is the Cr held at the grain #
    #     boundary by the Dirichlet BC — the physical minimum in the       #
    #     decoupled model.  C/N coupling adjusts this downward.            #
    # ------------------------------------------------------------------ #
    cr_min_base   = cr_output.C_sink_wt_pct
    cr_sink_floor = _CR_ABSOLUTE_FLOOR  # absolute physical floor (0.0)

    # ------------------------------------------------------------------ #
    # 2.  Individual species contributions                                  #
    # ------------------------------------------------------------------ #
    c_contrib = _compute_contribution(c_output, lambda_c)
    n_contrib = _compute_contribution(n_output, lambda_n)

    # ------------------------------------------------------------------ #
    # 3.  Effective minimum Cr (bounded below by absolute physical floor)  #
    # ------------------------------------------------------------------ #
    raw_effective    = cr_min_base - c_contrib - n_contrib
    cr_min_effective = max(raw_effective, cr_sink_floor)

    # ------------------------------------------------------------------ #
    # 4.  Mechanism components                                             #
    # ------------------------------------------------------------------ #
    mechanisms = _build_mechanism_components(has_c, has_n)

    # ------------------------------------------------------------------ #
    # 5.  Species inputs list                                              #
    # ------------------------------------------------------------------ #
    species: list[str] = [cr_output.element]
    if has_c:
        species.append(c_output.element)   # type: ignore[union-attr]
    if has_n:
        species.append(n_output.element)   # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # 6.  Warnings                                                         #
    # ------------------------------------------------------------------ #
    warnings = _collect_warnings(
        cr_output, c_output, n_output,
        c_contrib, n_contrib,
        raw_effective,
        lambda_c, lambda_n,
    )

    return CoupledDiffusionResult(
        cr_min_base_wt_pct=cr_min_base,
        c_contribution_wt_pct=c_contrib,
        n_contribution_wt_pct=n_contrib,
        cr_min_effective_wt_pct=cr_min_effective,
        cr_sink_floor_wt_pct=cr_sink_floor,
        mechanism_components=mechanisms,
        lambda_c_used=lambda_c,
        lambda_n_used=lambda_n,
        species_inputs=species,
        assumptions=list(_BASE_ASSUMPTIONS),
        warnings=warnings,
        notes=list(_BASE_NOTES),
    )
