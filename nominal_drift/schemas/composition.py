"""
nominal_drift.schemas.composition
==============================
Pydantic v2 schema for alloy composition input.

This schema is the entry point for Track 1 metallurgy workflows.  It
validates that the composition is physically plausible before it is
forwarded to any scientific engine module.

Design notes
------------
- All element symbols use standard two-character capitalisation (e.g. "Cr",
  "Fe", "Ni").  Keys are case-sensitive: "cr" != "Cr".
- The sum-to-100 check uses a generous tolerance (±COMPOSITION_SUM_TOLERANCE_WTP
  wt%) because real EDS / EPMA / XRF measurements frequently exclude trace
  elements that were not in the scan list, resulting in sums slightly below
  100.  Values well above 100 indicate a data-entry error and are rejected.
- Multi-species support: compositions may include any element.  Elements
  relevant to sensitization and diffusion workflows (Cr, C, N) receive
  special treatment in downstream modules but are not mandatory in this
  schema beyond Cr (required for sensitization) and Fe (required for
  ferritic / austenitic / duplex / martensitic stainless matrices).
- The schema is intentionally NOT restricted to stainless steels.  Future
  workflows (Ni-base superalloys, tool steels, crystal-generation inputs)
  may use different element sets.  Only the two mandatory keys (Fe, Cr) are
  domain-specific constraints for Sprint 1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Absolute tolerance on the wt% sum check.
#: Rationale: EDS/EPMA measurements exclude unmeasured trace elements;
#: sums between 98 and 102 wt% are considered physically plausible.
#: Sums outside [100 - tol, 100 + tol] indicate a likely data-entry error.
COMPOSITION_SUM_TOLERANCE_WTP: float = 2.0

#: Elements that must be present in steel / ferrous compositions.
#: For non-ferrous material systems the required-element check is
#: relaxed — only a non-empty composition is enforced.
REQUIRED_ELEMENTS_FERROUS: frozenset[str] = frozenset({"Fe", "Cr"})

#: Backward-compatible alias (used in existing tests).
REQUIRED_ELEMENTS: frozenset[str] = REQUIRED_ELEMENTS_FERROUS

#: Matrix identifiers that are considered ferrous (require Fe + Cr).
_FERROUS_MATRICES: frozenset[str] = frozenset({
    "austenite", "ferrite", "duplex", "martensite",
})

#: Recognised alloy-matrix identifiers.
#:
#: The matrix field serves two purposes:
#:   1. Crystallographic structure class (fcc, bcc, hcp, etc.)
#:   2. Material-family hint for downstream modules
#:
#: A cleaner split (separate crystal_system + material_family fields)
#: is a future refactor target.  For now, a single flat Literal covers
#: both use-cases and avoids breaking the frozen-model contract.
#:
#: Backward-compatible steel values are preserved.  New values use
#: lowercase for consistency.
AlloyMatrix = Literal[
    # --- Ferrous / steel (backward-compatible) ---
    "austenite", "ferrite", "duplex", "martensite",
    # --- Generic crystallographic classes ---
    "fcc", "bcc", "hcp",
    # --- Material-family specific ---
    "al_fcc", "ni_fcc", "cu_fcc", "co_hcp", "ti_hcp", "ti_bcc",
    # --- Ceramics / functional materials ---
    "perovskite", "oxide", "carbon",
    # --- Catch-all ---
    "generic", "unknown",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AlloyComposition(BaseModel):
    """
    Validated alloy composition for scientific workflow input.

    Parameters
    ----------
    alloy_designation : str
        Standard or internal designation, e.g. "316L", "2205", "430", "IN718".
    alloy_matrix : AlloyMatrix
        Crystal / microstructural matrix type.  Governs which Arrhenius
        constants and phase-stability data are loaded by scientific modules.
        Ferrous: "austenite", "ferrite", "duplex", "martensite".
        Non-ferrous: "ni_fcc", "al_fcc", "ti_hcp", "cu_fcc", etc.
        General: "generic", "unknown".
    composition_wt_pct : dict[str, float]
        Elemental composition in weight percent.  Keys are element symbols
        (case-sensitive, e.g. "Cr" not "cr").  Values must be non-negative.
        Ferrous matrices require "Fe" and "Cr"; non-ferrous only require
        a non-empty dict.
        The sum must lie within COMPOSITION_SUM_TOLERANCE_WTP of 100 wt%.
    elemental_analysis_method : str | None
        Technique used to measure the composition, e.g. "EDS", "WDS",
        "EPMA", "XRF", "ICP-OES".  None if nominal / certificate value.
    uncertainty_wt_pct : dict[str, float] | None
        Per-element 1σ measurement uncertainty in weight percent.
        Keys must be a subset of composition_wt_pct keys.
        All values must be non-negative.

    Examples
    --------
    >>> comp = AlloyComposition(
    ...     alloy_designation="316L",
    ...     alloy_matrix="austenite",
    ...     composition_wt_pct={
    ...         "Fe": 65.88, "Cr": 16.50, "Ni": 10.50,
    ...         "Mo": 2.10,  "Mn": 1.80,  "Si": 0.50,
    ...         "C":  0.02,  "N":  0.07,  "P":  0.03,
    ...         "S":  0.003,
    ...     },
    ...     elemental_analysis_method="EDS",
    ...     uncertainty_wt_pct={"Cr": 0.30, "Ni": 0.30, "Mo": 0.10},
    ... )
    """

    model_config = ConfigDict(
        # Forbid extra keys so that typos in field names raise an error
        # rather than being silently ignored.
        extra="forbid",
        # Freeze instances after creation to prevent accidental mutation
        # of composition data mid-workflow.
        frozen=True,
    )

    alloy_designation: str = Field(
        ...,
        min_length=1,
        description="Standard or internal alloy designation (e.g. '316L', '2205').",
    )

    alloy_matrix: AlloyMatrix = Field(
        ...,
        description=(
            "Crystal / microstructural matrix type.  Governs Arrhenius "
            "constant selection and phase-stability lookups.  "
            "Ferrous, Ni-base, Al, Ti, oxide, and custom matrices supported."
        ),
    )

    composition_wt_pct: dict[str, float] = Field(
        ...,
        description=(
            "Elemental composition in weight percent.  Keys are element "
            "symbols (case-sensitive).  Ferrous matrices require 'Fe' and "
            "'Cr'; non-ferrous only require a non-empty dict."
        ),
    )

    elemental_analysis_method: str | None = Field(
        default=None,
        description=(
            "Measurement technique: 'EDS', 'WDS', 'EPMA', 'XRF', 'ICP-OES', "
            "etc.  None if composition is nominal or from a material certificate."
        ),
    )

    uncertainty_wt_pct: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-element 1σ measurement uncertainty in weight percent.  "
            "Keys must be a subset of composition_wt_pct keys."
        ),
    )

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("alloy_designation")
    @classmethod
    def designation_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("alloy_designation must not be blank or whitespace.")
        return v.strip()

    @field_validator("composition_wt_pct")
    @classmethod
    def validate_composition_values(cls, v: dict[str, float]) -> dict[str, float]:
        """
        Reject empty dicts and negative concentrations.
        Element-presence and sum-range checks are in the model validator
        (which runs after all field validators and has access to all fields).
        """
        if not v:
            raise ValueError("composition_wt_pct must not be empty.")

        negative = {el: wt for el, wt in v.items() if wt < 0.0}
        if negative:
            raise ValueError(
                f"Composition values must be non-negative.  "
                f"Negative values found: {negative}"
            )
        return v

    @field_validator("uncertainty_wt_pct")
    @classmethod
    def validate_uncertainty_values(
        cls, v: dict[str, float] | None
    ) -> dict[str, float] | None:
        """Uncertainty values must be non-negative if provided."""
        if v is None:
            return v
        negative = {el: u for el, u in v.items() if u < 0.0}
        if negative:
            raise ValueError(
                f"Uncertainty values must be non-negative.  "
                f"Negative values found: {negative}"
            )
        return v

    # ------------------------------------------------------------------
    # Model-level validators (run after all fields are populated)
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_composition_constraints(self) -> "AlloyComposition":
        """
        Cross-field checks that require access to the full model:

        1. Required elements (Fe, Cr) must be present.
        2. Composition sum must be within tolerance of 100 wt%.
        3. Uncertainty keys must be a subset of composition keys.
        """
        comp = self.composition_wt_pct

        # 1 — Required elements (ferrous matrices require Fe + Cr;
        #     non-ferrous only require a non-empty composition)
        if self.alloy_matrix in _FERROUS_MATRICES:
            missing = REQUIRED_ELEMENTS_FERROUS - comp.keys()
            if missing:
                raise ValueError(
                    f"composition_wt_pct is missing required element(s): "
                    f"{sorted(missing)}.  Both 'Fe' and 'Cr' must be present "
                    f"for ferrous matrix '{self.alloy_matrix}'."
                )

        # 2 — Sum-to-100 check
        total = sum(comp.values())
        lo = 100.0 - COMPOSITION_SUM_TOLERANCE_WTP
        hi = 100.0 + COMPOSITION_SUM_TOLERANCE_WTP
        if not (lo <= total <= hi):
            raise ValueError(
                f"composition_wt_pct sums to {total:.3f} wt%, which is "
                f"outside the accepted range [{lo:.1f}, {hi:.1f}] wt%.  "
                f"Check for missing elements or data-entry errors.  "
                f"(Tolerance = ±{COMPOSITION_SUM_TOLERANCE_WTP} wt%; "
                f"rationale: EDS/EPMA measurements may exclude unmeasured "
                f"trace elements, but deviations beyond ±{COMPOSITION_SUM_TOLERANCE_WTP} "
                f"wt% indicate a likely data error.)"
            )

        # 3 — Uncertainty keys subset check
        if self.uncertainty_wt_pct is not None:
            extra_keys = set(self.uncertainty_wt_pct.keys()) - set(comp.keys())
            if extra_keys:
                raise ValueError(
                    f"uncertainty_wt_pct contains element(s) not in "
                    f"composition_wt_pct: {sorted(extra_keys)}.  "
                    f"Uncertainty can only be specified for measured elements."
                )

        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def bulk_Cr_wt_pct(self) -> float:
        """Nominal chromium content in wt%.  Used as C_bulk in diffusion solver."""
        return self.composition_wt_pct["Cr"]

    @property
    def has_carbon(self) -> bool:
        """True if carbon is explicitly listed in the composition."""
        return "C" in self.composition_wt_pct

    @property
    def has_nitrogen(self) -> bool:
        """True if nitrogen is explicitly listed in the composition."""
        return "N" in self.composition_wt_pct

    @property
    def composition_sum(self) -> float:
        """Sum of all elemental concentrations in wt%."""
        return sum(self.composition_wt_pct.values())
