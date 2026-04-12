"""
nominal_drift.science.doe_planner
==================================
Design-of-Experiments (DOE) planner for heat treatment studies.

PURPOSE: Generate experiment plans (full factorial, minimum validation, repeatability)
for systematic exploration of sensitisation parameter space (temperature × hold time).

Philosophy: Minimal but complete — no black-box optimisation, no sequential strategies.
All plans are deterministic and independent of data.
"""

from __future__ import annotations

import statistics
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ExperimentPoint(BaseModel, frozen=True):
    """A single experiment condition with replicates.

    Attributes
    ----------
    temperature_C : float
        Isothermal hold temperature [°C].
    hold_min : float
        Isothermal hold duration [min].
    n_replicates : int
        Number of replicates at this condition (≥ 1).
    label : str
        Human-readable label (e.g. "T=700C_t=60min").
    purpose : str
        Classification: "factorial", "center_point", "validation", "repeatability".
    """

    model_config = {"frozen": True}

    temperature_C: float
    hold_min: float
    n_replicates: int
    label: str
    purpose: str

    @field_validator("n_replicates")
    @classmethod
    def _check_n_replicates(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_replicates must be >= 1; got {v}")
        return v

    @field_validator("purpose")
    @classmethod
    def _check_purpose(cls, v: str) -> str:
        valid = {"factorial", "center_point", "validation", "repeatability"}
        if v not in valid:
            raise ValueError(
                f"purpose must be one of {valid}; got {v!r}"
            )
        return v


class DOEPlan(BaseModel, frozen=True):
    """A complete design-of-experiments plan.

    Attributes
    ----------
    plan_type : str
        Type of plan: "full_factorial", "minimum_validation", "center_augmented",
        or "repeatability".
    experiment_points : list[ExperimentPoint]
        List of experiment conditions (unordered).
    n_total_runs : int
        Total number of runs = sum of n_replicates across all points.
    temperature_range_C : tuple[float, float]
        (T_min, T_max) of the parameter space.
    time_range_min : tuple[float, float]
        (t_min, t_max) of the parameter space.
    alloy_label : str
        Alloy designation or label.
    rationale : str
        Brief explanation of the plan and its purpose.
    assumptions : list[str]
        Model and design assumptions.
    warnings : list[str]
        Caveats (e.g. n_replicates < 2 for statistics).
    notes : list[str]
        Guidance for execution or calibration.
    """

    model_config = {"frozen": True}

    plan_type: str
    experiment_points: list[ExperimentPoint]
    n_total_runs: int
    temperature_range_C: tuple[float, float]
    time_range_min: tuple[float, float]
    alloy_label: str
    rationale: str
    assumptions: list[str]
    warnings: list[str]
    notes: list[str]

    @field_validator("plan_type")
    @classmethod
    def _check_plan_type(cls, v: str) -> str:
        valid = {
            "full_factorial",
            "minimum_validation",
            "center_augmented",
            "repeatability",
        }
        if v not in valid:
            raise ValueError(
                f"plan_type must be one of {valid}; got {v!r}"
            )
        return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_full_factorial(
    temperatures_C: list[float],
    hold_times_min: list[float],
    n_replicates: int = 2,
    alloy_label: str = "",
    include_center: bool = True,
) -> DOEPlan:
    """Generate a full factorial DOE plan.

    Creates all combinations of temperature × hold_time, optionally augmented
    with a center point at (mean_T, mean_t).

    Parameters
    ----------
    temperatures_C : list[float]
        Temperature levels [°C].  Must have at least 1 element.
    hold_times_min : list[float]
        Hold time levels [min].  Must have at least 1 element.
    n_replicates : int
        Replicates per condition (default 2).
    alloy_label : str
        Alloy designation (default "").
    include_center : bool
        If True and both lists have 2+ elements, add a center point
        (default True).

    Returns
    -------
    DOEPlan
        Complete factorial plan.
    """
    # --------- Validate inputs ---------
    temperatures_C = list(temperatures_C)
    hold_times_min = list(hold_times_min)

    if not temperatures_C or not hold_times_min:
        raise ValueError(
            "Both temperatures_C and hold_times_min must be non-empty."
        )

    # --------- Generate factorial points ---------
    points: list[ExperimentPoint] = []

    for T_C in temperatures_C:
        for t_min in hold_times_min:
            label = f"T={T_C:.0f}C_t={t_min:.0f}min"
            points.append(
                ExperimentPoint(
                    temperature_C=T_C,
                    hold_min=t_min,
                    n_replicates=n_replicates,
                    label=label,
                    purpose="factorial",
                )
            )

    # --------- Add center point if requested ---------
    if include_center and len(temperatures_C) >= 2 and len(hold_times_min) >= 2:
        T_center = statistics.mean(temperatures_C)
        t_center = statistics.mean(hold_times_min)
        center_label = f"T={T_center:.0f}C_t={t_center:.0f}min_center"
        points.append(
            ExperimentPoint(
                temperature_C=T_center,
                hold_min=t_center,
                n_replicates=n_replicates,
                label=center_label,
                purpose="center_point",
            )
        )

    # --------- Compute plan metadata ---------
    n_total = sum(p.n_replicates for p in points)
    T_min, T_max = min(temperatures_C), max(temperatures_C)
    t_min, t_max = min(hold_times_min), max(hold_times_min)

    assumptions = [
        "Factorial plans assume independent, randomised experiments.",
        "Center points improve estimation of curvature.",
        "Response surface is assumed approximately quadratic in the local region.",
    ]

    warnings: list[str] = []
    if n_replicates < 2:
        warnings.append(
            f"n_replicates={n_replicates} is less than 2; "
            "statistical inference will be severely limited."
        )

    notes = [
        "Randomise the order of runs to reduce systematic drift.",
        "Use the center point to estimate pure quadratic curvature.",
        "Consider screening first with a fractional factorial to reduce cost.",
    ]

    rationale = (
        f"Full factorial plan with {len(temperatures_C)} temperature(s) × "
        f"{len(hold_times_min)} time(s) = {len(temperatures_C) * len(hold_times_min)} "
        f"base conditions, "
        f"{'plus 1 center point ' if include_center else ''}"
        f"replicated {n_replicates} time(s)."
    )

    return DOEPlan(
        plan_type="full_factorial",
        experiment_points=points,
        n_total_runs=n_total,
        temperature_range_C=(T_min, T_max),
        time_range_min=(t_min, t_max),
        alloy_label=alloy_label,
        rationale=rationale,
        assumptions=assumptions,
        warnings=warnings,
        notes=notes,
    )


def generate_minimum_validation(
    temperatures_C: list[float],
    hold_times_min: list[float],
    n_replicates: int = 3,
    alloy_label: str = "",
) -> DOEPlan:
    """Generate a minimum-validation corner-point design.

    Selects the 4 corner points of the parameter space (min/max T × min/max t),
    plus a center point. Efficient for initial screening or model validation.

    Parameters
    ----------
    temperatures_C : list[float]
        Temperature levels [°C].  If < 2 elements, all are used as corners.
    hold_times_min : list[float]
        Hold time levels [min].  If < 2 elements, all are used as corners.
    n_replicates : int
        Replicates per condition (default 3).
    alloy_label : str
        Alloy designation (default "").

    Returns
    -------
    DOEPlan
        Minimum validation plan (always 5 base points: 4 corners + 1 center).
    """
    temperatures_C = list(temperatures_C)
    hold_times_min = list(hold_times_min)

    if not temperatures_C or not hold_times_min:
        raise ValueError(
            "Both temperatures_C and hold_times_min must be non-empty."
        )

    # --------- Determine corner points ---------
    if len(temperatures_C) >= 2:
        T_corner = [min(temperatures_C), max(temperatures_C)]
    else:
        T_corner = temperatures_C

    if len(hold_times_min) >= 2:
        t_corner = [min(hold_times_min), max(hold_times_min)]
    else:
        t_corner = hold_times_min

    # --------- Generate corner points ---------
    points: list[ExperimentPoint] = []

    for T_C in T_corner:
        for t_min in t_corner:
            label = f"T={T_C:.0f}C_t={t_min:.0f}min"
            points.append(
                ExperimentPoint(
                    temperature_C=T_C,
                    hold_min=t_min,
                    n_replicates=n_replicates,
                    label=label,
                    purpose="validation",
                )
            )

    # --------- Add center point ---------
    T_center = statistics.mean(temperatures_C)
    t_center = statistics.mean(hold_times_min)
    center_label = f"T={T_center:.0f}C_t={t_center:.0f}min_center"
    points.append(
        ExperimentPoint(
            temperature_C=T_center,
            hold_min=t_center,
            n_replicates=n_replicates,
            label=center_label,
            purpose="center_point",
        )
    )

    # --------- Compute plan metadata ---------
    n_total = sum(p.n_replicates for p in points)
    T_min, T_max = min(temperatures_C), max(temperatures_C)
    t_min, t_max = min(hold_times_min), max(hold_times_min)

    assumptions = [
        "Corner-point design assumes main effects dominate; interaction terms are secondary.",
        "Center point provides a model validation check.",
    ]

    warnings: list[str] = []
    if n_replicates < 2:
        warnings.append(
            f"n_replicates={n_replicates} is less than 2; "
            "statistical inference will be severely limited."
        )

    notes = [
        "Efficient screening plan — only 5 unique conditions.",
        "Use center-point replicates to estimate pure experimental error.",
        "Suitable for initial validation or calibration studies.",
    ]

    rationale = (
        f"Minimum validation (corner + center) with {len(T_corner)} temperature(s) × "
        f"{len(t_corner)} time(s) corner points + 1 center point, "
        f"replicated {n_replicates} time(s) = {n_total} total runs."
    )

    return DOEPlan(
        plan_type="minimum_validation",
        experiment_points=points,
        n_total_runs=n_total,
        temperature_range_C=(T_min, T_max),
        time_range_min=(t_min, t_max),
        alloy_label=alloy_label,
        rationale=rationale,
        assumptions=assumptions,
        warnings=warnings,
        notes=notes,
    )


def generate_repeatability_plan(
    temperature_C: float,
    hold_min: float,
    n_replicates: int = 6,
    alloy_label: str = "",
) -> DOEPlan:
    """Generate a single-condition repeatability plan.

    Repeats one (T, t) condition multiple times to estimate experimental
    variability and measurement noise.

    Parameters
    ----------
    temperature_C : float
        Isothermal temperature [°C].
    hold_min : float
        Hold duration [min].
    n_replicates : int
        Number of replicates (default 6).
    alloy_label : str
        Alloy designation (default "").

    Returns
    -------
    DOEPlan
        Repeatability plan (always 1 unique condition × n_replicates runs).
    """
    label = f"T={temperature_C:.0f}C_t={hold_min:.0f}min_repeatability"
    point = ExperimentPoint(
        temperature_C=temperature_C,
        hold_min=hold_min,
        n_replicates=n_replicates,
        label=label,
        purpose="repeatability",
    )

    assumptions = [
        "All replicates are independent and randomised.",
        "Experimental conditions (temperature, time, sample prep) are held constant.",
    ]

    warnings: list[str] = []
    if n_replicates < 2:
        warnings.append(
            f"n_replicates={n_replicates} is less than 2; "
            "statistical inference will be severely limited."
        )

    notes = [
        "Use repeatability data to estimate measurement uncertainty.",
        "A standard deviation from these replicates helps quantify instrumental noise.",
        "Typically 6–10 replicates is sufficient for uncertainty estimation.",
    ]

    rationale = (
        f"Repeatability study at single condition "
        f"(T={temperature_C:.0f}°C, t={hold_min:.0f} min) "
        f"with {n_replicates} replicates."
    )

    return DOEPlan(
        plan_type="repeatability",
        experiment_points=[point],
        n_total_runs=n_replicates,
        temperature_range_C=(temperature_C, temperature_C),
        time_range_min=(hold_min, hold_min),
        alloy_label=alloy_label,
        rationale=rationale,
        assumptions=assumptions,
        warnings=warnings,
        notes=notes,
    )
