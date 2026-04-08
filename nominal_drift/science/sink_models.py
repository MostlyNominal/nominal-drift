"""
nominal_drift.science.sink_models
==================================
Temperature-dependent grain-boundary sink model for Cr in austenitic alloys.

PURPOSE
-------
The diffusion engine (``diffusion_engine.py``) uses a fixed Dirichlet boundary
condition at x = 0 to represent the Cr concentration in the matrix immediately
adjacent to a M₂₃C₆ carbide precipitate at the grain boundary.  In Sprint 1
this was a scalar constant (default: 12.0 wt%).

This module replaces that fixed assumption with a **first-order temperature-
aware model**.  The equilibrium matrix Cr content in contact with M₂₃C₆ is
temperature-dependent:

  * At low temperatures (< ~600 °C) precipitation is kinetically slow;
    the effective sink is shallow (high Cr, close to bulk).
  * At the "nose" of sensitisation (~700–750 °C for 304/316L) the sink
    is deepest (minimum matrix Cr).
  * At high temperatures (> ~900 °C) M₂₃C₆ dissolves back into solution;
    the sink approaches the bulk Cr content.

MODELLING APPROACH
------------------
The model is implemented as a **configurable lookup table** with linear
interpolation between calibration points.  This is deliberately simple:

  1. The table ``(T_C, C_sink_wt_pct)`` is the sole model input.
  2. Interpolation is performed with ``numpy.interp`` (linear, 1-D).
  3. Extrapolation outside the table range is clamped to the nearest endpoint
     and flagged with a warning.
  4. Default tables for 316L and 304 stainless steels are provided, built from
     literature-informed engineering estimates (see ``source_notes`` on each
     table for references and caveats).

DEFAULT TABLES
--------------
``DEFAULT_SINK_TABLE_316L``
    Engineering sink curve for 316L (low-carbon, Mo-bearing austenitic SS).
    Calibrated against published Cr-depletion depth data in the 550–900 °C range.
    More resistant to sensitisation than 304 due to lower C content and Mo.

``DEFAULT_SINK_TABLE_304``
    Engineering sink curve for 304 austenitic SS.
    Shallower sink floor (lower minimum Cr) reflects higher C content driving
    deeper grain-boundary depletion.

WHAT THIS MODULE IS NOT
-----------------------
* Not a CALPHAD or Thermo-Calc calculation.
* Not a validated CCT/TTT predictor.
* Not a grain-boundary segregation or nucleation model.
* The default tables are engineering estimates — not fitted to a specific heat
  of material.  Calibrate against measured profiles for quantitative work.

Public API
----------
``SinkLookupTable``
    Dataclass holding the T → C_sink interpolation table and metadata.

``SinkEvaluationResult``
    Pydantic v2 frozen result model returned by ``evaluate_sink``.

``DEFAULT_SINK_TABLE_316L``
    Default sink table for 316L stainless steel.

``DEFAULT_SINK_TABLE_304``
    Default sink table for 304 stainless steel.

``build_sink_table(temperatures_C, c_sink_wt_pct, alloy_label, source_notes)``
    Constructor for custom ``SinkLookupTable`` objects.

``evaluate_sink(T_C, table)``
    Evaluate the grain-boundary Cr sink concentration at temperature *T_C*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Module-level engineering constants
# ---------------------------------------------------------------------------

#: Minimum physically meaningful Cr concentration [wt%].
#: The model will never return a value below this floor.
_CR_FLOOR_WT_PCT: float = 0.0

#: Maximum temperature extrapolation tolerance before issuing a warning.
#: If the requested T_C is more than this many degrees outside the table range,
#: a warning is issued in addition to clamping.
_EXTRAPOLATION_WARN_K: float = 50.0

# ---------------------------------------------------------------------------
# Assumption strings
# ---------------------------------------------------------------------------

_BASE_ASSUMPTIONS: list[str] = [
    (
        "C_sink(T) is modelled as a piecewise-linear interpolation over a "
        "calibration table.  Linear interpolation may under- or over-estimate "
        "the true equilibrium value between calibration points."
    ),
    (
        "Default calibration tables (DEFAULT_SINK_TABLE_316L, "
        "DEFAULT_SINK_TABLE_304) are first-order engineering estimates "
        "derived from published Cr-depletion data.  They are NOT fitted to a "
        "specific heat of material."
    ),
    (
        "Temperature dependence of C_sink captures the equilibrium matrix Cr "
        "adjacent to M₂₃C₆ as a function of T only.  Kinetic effects (time "
        "to reach equilibrium, carbide coarsening, dissolution kinetics) are "
        "NOT modelled."
    ),
    (
        "Extrapolation outside the table range clamps to the nearest endpoint "
        "value.  Results outside the table range are unreliable and flagged "
        "with a warning."
    ),
    (
        "This is a first-order engineering model — not a CALPHAD calculation, "
        "Thermo-Calc result, or validated electrochemical sensitisation predictor."
    ),
]

_BASE_NOTES: list[str] = [
    (
        "To calibrate for a specific alloy heat, fit the lookup table against "
        "measured Cr-depletion profiles at multiple temperatures."
    ),
    (
        "For higher-fidelity modelling, replace this module with a "
        "CALPHAD-driven or DICTRA-style thermodynamic equilibrium calculation."
    ),
]


# ---------------------------------------------------------------------------
# Lookup table dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SinkLookupTable:
    """An immutable T → C_sink interpolation table with metadata.

    Attributes
    ----------
    temperatures_C : tuple[float, ...]
        Monotonically increasing temperature values [°C].
    c_sink_wt_pct : tuple[float, ...]
        Corresponding grain-boundary sink concentrations [wt%].
        Must be the same length as *temperatures_C*.
    alloy_label : str
        Human-readable label for this alloy/condition (e.g. ``"316L SS"``).
    source_notes : tuple[str, ...]
        Provenance notes describing data source, assumptions, and limitations.
    """

    temperatures_C: tuple[float, ...]
    c_sink_wt_pct:  tuple[float, ...]
    alloy_label:    str
    source_notes:   tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.temperatures_C) < 2:
            raise ValueError(
                "SinkLookupTable requires at least 2 calibration points; "
                f"got {len(self.temperatures_C)}."
            )
        if len(self.temperatures_C) != len(self.c_sink_wt_pct):
            raise ValueError(
                f"temperatures_C length ({len(self.temperatures_C)}) must match "
                f"c_sink_wt_pct length ({len(self.c_sink_wt_pct)})."
            )
        for i in range(1, len(self.temperatures_C)):
            if self.temperatures_C[i] <= self.temperatures_C[i - 1]:
                raise ValueError(
                    f"temperatures_C must be strictly increasing; "
                    f"violation at index {i}: "
                    f"{self.temperatures_C[i-1]} >= {self.temperatures_C[i]}."
                )
        for v in self.c_sink_wt_pct:
            if v < 0.0:
                raise ValueError(
                    f"c_sink_wt_pct values must be non-negative; got {v}."
                )

    @property
    def T_min(self) -> float:
        """Lower bound of the calibrated temperature range [°C]."""
        return self.temperatures_C[0]

    @property
    def T_max(self) -> float:
        """Upper bound of the calibrated temperature range [°C]."""
        return self.temperatures_C[-1]


def build_sink_table(
    temperatures_C: list[float],
    c_sink_wt_pct: list[float],
    alloy_label: str,
    source_notes: Optional[list[str]] = None,
) -> SinkLookupTable:
    """Construct a validated :class:`SinkLookupTable` from list inputs.

    Parameters
    ----------
    temperatures_C : list[float]
        Calibration temperatures in degrees Celsius.  Must be strictly
        increasing and contain at least 2 points.
    c_sink_wt_pct : list[float]
        Grain-boundary Cr sink concentrations [wt%] at each temperature.
        Must be non-negative and the same length as *temperatures_C*.
    alloy_label : str
        Human-readable identifier for this alloy / table (e.g. ``"316L SS"``).
    source_notes : list[str] | None
        Optional provenance notes.  Defaults to an empty tuple if not given.

    Returns
    -------
    SinkLookupTable
        Validated, immutable lookup table ready for use with
        :func:`evaluate_sink`.

    Raises
    ------
    ValueError
        If the inputs violate any of the table invariants (length mismatch,
        non-monotonic temperatures, negative sink values, fewer than 2 points).
    """
    return SinkLookupTable(
        temperatures_C=tuple(temperatures_C),
        c_sink_wt_pct=tuple(c_sink_wt_pct),
        alloy_label=alloy_label,
        source_notes=tuple(source_notes or []),
    )


# ---------------------------------------------------------------------------
# Default literature-informed sink tables
# ---------------------------------------------------------------------------

#: Default engineering sink curve for **316L stainless steel**.
#:
#: Sources / basis:
#:   * Lo, K. H. et al., "Recent developments in stainless steels",
#:     Materials Science and Engineering R (2009) — sensitization temperature
#:     range and Cr-depletion behaviour.
#:   * Was, G. S., "Fundamentals of Radiation Materials Science" (2007) —
#:     grain-boundary Cr depletion curves for austenitic alloys.
#:   * Engineering estimates for 316L (C ≤ 0.03 wt%) with Mo addition
#:     providing some carbide retardation.
#:
#: Notes:
#:   * Values represent approximate equilibrium matrix Cr adjacent to M₂₃C₆
#:     at each temperature.  Not fitted to a specific commercial heat.
#:   * 316L has a higher minimum C_sink (~12.5 wt%) than 304 at the sensitisation
#:     nose due to its lower bulk C content.
DEFAULT_SINK_TABLE_316L: SinkLookupTable = build_sink_table(
    temperatures_C=[400.0, 500.0, 550.0, 600.0, 650.0, 700.0,
                    750.0, 800.0, 850.0, 900.0, 1000.0, 1100.0],
    c_sink_wt_pct= [16.0,  15.5,  15.0,  14.0,  13.0,  12.5,
                    12.8,  13.5,  14.5,  15.5,  16.0,   16.5],
    alloy_label="316L SS (literature estimate)",
    source_notes=[
        "Engineering estimate for 316L (C ≤ 0.03 wt%, Mo 2–3 wt%).",
        "T_min=400°C: precipitation kinetically negligible; sink ≈ bulk Cr.",
        "T=700°C: sensitisation nose; C_sink minimum (~12.5 wt%).",
        "T>900°C: M₂₃C₆ dissolution accelerates; C_sink returns toward bulk.",
        "Reference: Lo et al. (2009), Was (2007). Not fitted to a specific heat.",
    ],
)

#: Default engineering sink curve for **304 stainless steel**.
#:
#: Sources / basis:
#:   * Same literature sources as 316L table above.
#:   * 304 has higher C content (0.08 wt% max) → deeper sensitisation trough.
#:   * No Mo addition means the sensitisation nose is more pronounced.
#:
#: Notes:
#:   * Lower minimum C_sink (~11.5 wt%) relative to 316L reflects the higher
#:     bulk C content and absence of Mo.
DEFAULT_SINK_TABLE_304: SinkLookupTable = build_sink_table(
    temperatures_C=[400.0, 500.0, 550.0, 600.0, 650.0, 700.0,
                    750.0, 800.0, 850.0, 900.0, 1000.0, 1100.0],
    c_sink_wt_pct= [18.0,  17.0,  15.5,  14.0,  12.5,  11.5,
                    12.0,  13.5,  15.0,  16.5,  18.0,   18.5],
    alloy_label="304 SS (literature estimate)",
    source_notes=[
        "Engineering estimate for 304 (C ≤ 0.08 wt%, no Mo).",
        "T=700°C: sensitisation nose; C_sink minimum (~11.5 wt%).",
        "Deeper sensitisation trough vs 316L due to higher C and no Mo.",
        "Reference: Lo et al. (2009), Was (2007). Not fitted to a specific heat.",
    ],
)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class SinkEvaluationResult(BaseModel, frozen=True):
    """Result of evaluating C_sink at a given temperature.

    Attributes
    ----------
    T_C : float
        Temperature at which the sink was evaluated [°C].
    C_sink_wt_pct : float
        Interpolated grain-boundary Cr sink concentration [wt%].
    interpolation_mode : str
        One of: ``"interpolated"`` (T within table range),
        ``"extrapolated_low"`` (T below table minimum),
        ``"extrapolated_high"`` (T above table maximum).
    T_table_min_C : float
        Lower bound of the calibrated temperature range [°C].
    T_table_max_C : float
        Upper bound of the calibrated temperature range [°C].
    alloy_label : str
        Label from the :class:`SinkLookupTable` used for evaluation.
    assumptions : list[str]
        Explicit model assumptions.  Always non-empty.
    warnings : list[str]
        Extrapolation warnings or other caveats.  Empty for in-range queries.
    notes : list[str]
        Contextual notes for downstream use.  Always non-empty.
    """

    T_C:                float
    C_sink_wt_pct:      float
    interpolation_mode: str
    T_table_min_C:      float
    T_table_max_C:      float
    alloy_label:        str
    assumptions:        list[str]
    warnings:           list[str]
    notes:              list[str]

    @field_validator("interpolation_mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        valid = {"interpolated", "extrapolated_low", "extrapolated_high"}
        if v not in valid:
            raise ValueError(
                f"interpolation_mode must be one of {valid}; got {v!r}."
            )
        return v

    @field_validator("C_sink_wt_pct")
    @classmethod
    def _check_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(
                f"C_sink_wt_pct must be non-negative; got {v}."
            )
        return v


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _determine_mode_and_warnings(
    T_C: float,
    table: SinkLookupTable,
) -> tuple[str, list[str]]:
    """Return (interpolation_mode, warnings) for a temperature query.

    Parameters
    ----------
    T_C : float
        Requested temperature [°C].
    table : SinkLookupTable
        The sink lookup table being queried.

    Returns
    -------
    tuple[str, list[str]]
        Interpolation mode string and list of warning strings (may be empty).
    """
    warnings: list[str] = []

    if T_C < table.T_min:
        mode = "extrapolated_low"
        delta = table.T_min - T_C
        warnings.append(
            f"Requested T_C={T_C:.1f}°C is below the table minimum "
            f"({table.T_min:.1f}°C) by {delta:.1f}°C.  "
            f"C_sink is clamped to the table's lowest value "
            f"({table.c_sink_wt_pct[0]:.2f} wt%).  "
            f"Extrapolation outside the calibrated range is unreliable."
        )
        if delta > _EXTRAPOLATION_WARN_K:
            warnings.append(
                f"Extrapolation gap of {delta:.1f}°C exceeds the recommended "
                f"tolerance ({_EXTRAPOLATION_WARN_K:.0f}°C).  "
                f"Consider extending the lookup table to cover this temperature."
            )
    elif T_C > table.T_max:
        mode = "extrapolated_high"
        delta = T_C - table.T_max
        warnings.append(
            f"Requested T_C={T_C:.1f}°C is above the table maximum "
            f"({table.T_max:.1f}°C) by {delta:.1f}°C.  "
            f"C_sink is clamped to the table's highest value "
            f"({table.c_sink_wt_pct[-1]:.2f} wt%).  "
            f"Extrapolation outside the calibrated range is unreliable."
        )
        if delta > _EXTRAPOLATION_WARN_K:
            warnings.append(
                f"Extrapolation gap of {delta:.1f}°C exceeds the recommended "
                f"tolerance ({_EXTRAPOLATION_WARN_K:.0f}°C).  "
                f"Consider extending the lookup table to cover this temperature."
            )
    else:
        mode = "interpolated"

    return mode, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_sink(
    T_C: float,
    table: SinkLookupTable = DEFAULT_SINK_TABLE_316L,
) -> SinkEvaluationResult:
    """Evaluate the grain-boundary Cr sink concentration at temperature *T_C*.

    Uses linear interpolation over the calibration points in *table*.
    Queries outside the table range are clamped (not extrapolated) and a
    warning is added to the result.

    Parameters
    ----------
    T_C : float
        Temperature of the isothermal hold [°C].
    table : SinkLookupTable
        Lookup table to use.  Defaults to :data:`DEFAULT_SINK_TABLE_316L`.

    Returns
    -------
    SinkEvaluationResult
        Frozen Pydantic result containing the interpolated C_sink value,
        interpolation mode, warnings, assumptions, and notes.

    Examples
    --------
    >>> from nominal_drift.science.sink_models import evaluate_sink
    >>> result = evaluate_sink(700.0)
    >>> result.C_sink_wt_pct          # interpolated at 700°C for 316L
    12.5
    >>> result.interpolation_mode
    'interpolated'
    >>> result.warnings               # empty — 700°C is within the table range
    []
    """
    # ------------------------------------------------------------------ #
    # 1.  Determine interpolation mode and collect extrapolation warnings  #
    # ------------------------------------------------------------------ #
    mode, warnings = _determine_mode_and_warnings(T_C, table)

    # ------------------------------------------------------------------ #
    # 2.  Interpolate (numpy.interp clamps at boundaries automatically)   #
    # ------------------------------------------------------------------ #
    raw_c_sink = float(
        np.interp(
            T_C,
            table.temperatures_C,
            table.c_sink_wt_pct,
        )
    )

    # ------------------------------------------------------------------ #
    # 3.  Apply absolute physical floor                                    #
    # ------------------------------------------------------------------ #
    c_sink = max(raw_c_sink, _CR_FLOOR_WT_PCT)

    # ------------------------------------------------------------------ #
    # 4.  Append source notes from the table as additional notes           #
    # ------------------------------------------------------------------ #
    notes = list(_BASE_NOTES) + [f"[table] {n}" for n in table.source_notes]

    return SinkEvaluationResult(
        T_C=T_C,
        C_sink_wt_pct=c_sink,
        interpolation_mode=mode,
        T_table_min_C=table.T_min,
        T_table_max_C=table.T_max,
        alloy_label=table.alloy_label,
        assumptions=list(_BASE_ASSUMPTIONS),
        warnings=warnings,
        notes=notes,
    )
