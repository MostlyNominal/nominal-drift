"""
nominal_drift.viz.risk_map
===========================
Generate TTT/CCT-style risk maps for sensitisation using analytical (erfc) approximation.

PURPOSE: Uses the analytical Fickian solution for a semi-infinite domain with Dirichlet BC
to compute the full T×t grid quickly (no CN solver needed).

PHYSICS: The analytical solution gives:
  depth_nm = 2e9 * sqrt(D(T) * t) * erfinv(depth_threshold / (C_bulk - C_sink(T)))
  where depth_threshold = 0.5 wt% (matches diffusion_engine._compute_depletion_depth_nm)

RISK CLASSIFICATION (same thresholds as sensitization_model.py):
  low      — C_sink(T) >= cr_threshold_wt_pct (no driving force regardless of time)
  moderate — C_sink(T) < cr_threshold AND depth_nm < depth_high_risk_nm (50.0 nm)
  high     — C_sink(T) < cr_threshold AND depth_nm >= depth_high_risk_nm (50.0 nm)
  Special case: if C_bulk - C_sink(T) <= depth_threshold (no depletion front detectable),
    depth_nm = 0.0 and apply same rules.
"""

from __future__ import annotations

from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pydantic import BaseModel, field_validator
from scipy.special import erfinv

from nominal_drift.science.diffusion_engine import arrhenius_D, load_arrhenius_constants
from nominal_drift.science.sink_models import evaluate_sink, DEFAULT_SINK_TABLE_316L, SinkLookupTable
from nominal_drift.schemas.composition import AlloyComposition


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Depth threshold for depletion detection [wt%]
_DEPTH_THRESHOLD_WT_PCT: float = 0.5

#: Factor to convert from m to nm
_M_TO_NM: float = 1e9


# ---------------------------------------------------------------------------
# Result Schema
# ---------------------------------------------------------------------------

class RiskMapResult(BaseModel, frozen=True):
    """Result of generating a sensitisation risk map.

    Attributes
    ----------
    temperatures_C : list[float]
        Temperature grid values [°C].
    times_s : list[float]
        Time grid values [s].
    risk_matrix : list[list[str]]
        2D risk classification grid [i_T][i_t] → "low"|"moderate"|"high".
    c_sink_at_T : list[float]
        C_sink(T) for each temperature [wt%].
    depletion_depth_matrix : list[list[float]]
        Depletion depth [nm] for each (T, t).
    cr_threshold_wt_pct : float
        Critical Cr threshold used for risk classification [wt%].
    depth_high_risk_nm : float
        Depth threshold above which risk is "high" [nm].
    alloy_label : str
        Label from the sink table.
    element : str
        Element being analysed (e.g. "Cr").
    assumptions : list[str]
        Model assumptions.
    warnings : list[str]
        Warnings (deduplicated).
    notes : list[str]
        Contextual notes.
    """

    temperatures_C: list[float]
    times_s: list[float]
    risk_matrix: list[list[str]]
    c_sink_at_T: list[float]
    depletion_depth_matrix: list[list[float]]
    cr_threshold_wt_pct: float
    depth_high_risk_nm: float
    alloy_label: str
    element: str
    assumptions: list[str]
    warnings: list[str]
    notes: list[str]

    @field_validator("risk_matrix")
    @classmethod
    def _validate_risk_matrix(cls, v: list[list[str]]) -> list[list[str]]:
        """All risk values must be in {"low", "moderate", "high"}."""
        valid = {"low", "moderate", "high"}
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if val not in valid:
                    raise ValueError(
                        f"risk_matrix[{i}][{j}] = {val!r} not in {valid}"
                    )
        return v

    @field_validator("c_sink_at_T")
    @classmethod
    def _validate_c_sink(cls, v: list[float]) -> list[float]:
        """All C_sink values must be non-negative."""
        for i, val in enumerate(v):
            if val < 0.0:
                raise ValueError(f"c_sink_at_T[{i}] = {val} is negative")
        return v

    @field_validator("depletion_depth_matrix")
    @classmethod
    def _validate_depth_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        """All depths must be non-negative."""
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if val < 0.0:
                    raise ValueError(
                        f"depletion_depth_matrix[{i}][{j}] = {val} is negative"
                    )
        return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_risk_map(
    composition: AlloyComposition,
    temperatures_C: list[float],
    times_s: list[float],
    element: str = "Cr",
    matrix: str = "austenite_FeCrNi",
    sink_table: SinkLookupTable = DEFAULT_SINK_TABLE_316L,
    cr_threshold_wt_pct: float = 12.0,
    depth_high_risk_nm: float = 50.0,
) -> RiskMapResult:
    """Generate a TTT/CCT-style sensitisation risk map.

    Uses the analytical erfc approximation (semi-infinite domain) to compute
    depletion depth at each (T, t) point, then classifies risk based on
    C_sink and depth thresholds.

    Parameters
    ----------
    composition : AlloyComposition
        Alloy composition; C_bulk is extracted from composition_wt_pct[element].
    temperatures_C : list[float]
        Temperature grid [°C].  Must have at least 1 element.
    times_s : list[float]
        Time grid [s].  Must have at least 1 element.  Can be log-spaced.
    element : str
        Element symbol (default "Cr").
    matrix : str
        Matrix identifier (default "austenite_FeCrNi").
    sink_table : SinkLookupTable
        Temperature-dependent sink model.
    cr_threshold_wt_pct : float
        Critical Cr threshold [wt%] above which no risk is incurred
        (default 12.0).
    depth_high_risk_nm : float
        Depth threshold [nm] above which risk is classified as "high"
        (default 50.0).

    Returns
    -------
    RiskMapResult
        Frozen result containing risk matrix, depletion depths, and metadata.

    Raises
    ------
    ValueError
        If element is not in composition.
    KeyError
        If element is not in Arrhenius constants database.
    """
    if element not in composition.composition_wt_pct:
        raise ValueError(
            f"Element '{element}' not found in composition. "
            f"Available: {list(composition.composition_wt_pct.keys())}"
        )

    c_bulk = composition.composition_wt_pct[element]
    constants = load_arrhenius_constants()

    # --------- Evaluate C_sink at each temperature ---------
    c_sink_list: list[float] = []
    sink_warnings: list[str] = []

    for T_C in temperatures_C:
        sink_result = evaluate_sink(T_C, sink_table)
        c_sink_list.append(sink_result.C_sink_wt_pct)
        sink_warnings.extend(sink_result.warnings)

    # Deduplicate warnings
    sink_warnings = list(dict.fromkeys(sink_warnings))

    # --------- Compute depletion depth matrix ---------
    risk_matrix: list[list[str]] = []
    depth_matrix: list[list[float]] = []

    for i_T, T_C in enumerate(temperatures_C):
        risk_row: list[str] = []
        depth_row: list[float] = []

        c_sink = c_sink_list[i_T]
        D_T = arrhenius_D(T_C, element, matrix, constants)

        for t_s in times_s:
            # -------- Compute depletion depth --------
            driving_force = c_bulk - c_sink
            if driving_force <= _DEPTH_THRESHOLD_WT_PCT:
                depth_nm = 0.0
            else:
                try:
                    # depth = 2e9 * sqrt(D*t) * erfinv(threshold / driving_force)
                    arg = _DEPTH_THRESHOLD_WT_PCT / driving_force
                    # Clamp arg to avoid domain errors in erfinv
                    arg = min(arg, 0.9999)
                    depth_nm = (
                        2.0 * _M_TO_NM * np.sqrt(D_T * t_s) * erfinv(arg)
                    )
                except (ValueError, ZeroDivisionError):
                    depth_nm = 0.0

            depth_row.append(max(0.0, depth_nm))

            # -------- Classify risk --------
            if c_sink >= cr_threshold_wt_pct:
                risk = "low"
            elif depth_nm < depth_high_risk_nm:
                risk = "moderate"
            else:
                risk = "high"

            risk_row.append(risk)

        risk_matrix.append(risk_row)
        depth_matrix.append(depth_row)

    # --------- Build result --------
    assumptions = [
        "Risk map uses analytical erfc approximation (semi-infinite domain) — NOT the CN solver.",
        "C_sink(T) is taken from a configurable lookup table (temperature-dependent sink model).",
        "Depletion depth threshold: 0.5 wt% above C_sink (matches diffusion_engine definition).",
        "This is a first-order engineering estimate — not a CALPHAD or validated CCT calculator.",
    ]

    notes = [
        "Use this risk map to identify sensitisation-prone process windows.",
        "Low-risk region (green): no significant Cr depletion regardless of time.",
        "Moderate-risk region (yellow): shallow depletion, but nucleation may be suppressed by kinetics.",
        "High-risk region (red): significant Cr depletion combined with long hold times.",
    ]

    return RiskMapResult(
        temperatures_C=list(temperatures_C),
        times_s=list(times_s),
        risk_matrix=risk_matrix,
        c_sink_at_T=c_sink_list,
        depletion_depth_matrix=depth_matrix,
        cr_threshold_wt_pct=cr_threshold_wt_pct,
        depth_high_risk_nm=depth_high_risk_nm,
        alloy_label=sink_table.alloy_label,
        element=element,
        assumptions=assumptions,
        warnings=sink_warnings,
        notes=notes,
    )


def plot_risk_map(
    result: RiskMapResult,
    use_log_time: bool = True,
    time_unit: str = "h",
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a risk map as a heatmap with contour boundary.

    Parameters
    ----------
    result : RiskMapResult
        Output from generate_risk_map.
    use_log_time : bool
        If True, x-axis is log-spaced (default True).
    time_unit : str
        Time unit for x-axis labels: "s", "min", "h" (default "h").
    save_path : str | None
        If provided, save the figure to this path.
    show : bool
        If True, call plt.show() (default False).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    # --------- Convert time to requested unit ---------
    unit_factor = {"s": 1.0, "min": 60.0, "h": 3600.0}
    if time_unit not in unit_factor:
        raise ValueError(
            f"time_unit must be one of {list(unit_factor.keys())}; got {time_unit!r}"
        )
    factor = unit_factor[time_unit]
    times_display = [t / factor for t in result.times_s]

    # --------- Map risk labels to numeric codes ---------
    risk_to_code = {"low": 1, "moderate": 2, "high": 3}
    risk_numeric = [
        [risk_to_code[cell] for cell in row]
        for row in result.risk_matrix
    ]

    # --------- Create figure ---------
    fig, ax = plt.subplots(figsize=(12, 8))

    # --------- Prepare grids for pcolormesh ---------
    T_mesh, t_mesh = np.meshgrid(result.temperatures_C, times_display, indexing="ij")
    risk_array = np.array(risk_numeric, dtype=float)

    # --------- Plot heatmap ---------
    cmap = mcolors.ListedColormap(["green", "yellow", "red"])
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    pcm = ax.pcolormesh(t_mesh, T_mesh, risk_array, cmap=cmap, norm=norm, shading="auto")

    # --------- Add contour line at moderate→high boundary ---------
    depth_array = np.array(result.depletion_depth_matrix, dtype=float)
    contours = ax.contour(
        t_mesh, T_mesh, depth_array,
        levels=[result.depth_high_risk_nm],
        colors="black",
        linewidths=2.5,
        linestyles="--",
    )
    ax.clabel(contours, inline=True, fontsize=9, fmt=f"{result.depth_high_risk_nm:.0f} nm")

    # --------- Labels and formatting ---------
    ax.set_xlabel(f"Time ({time_unit})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Sensitisation Risk Map — {result.element} / {result.alloy_label}",
        fontsize=14,
        fontweight="bold",
    )

    # --------- Log time axis if requested ---------
    if use_log_time:
        ax.set_xscale("log")

    # --------- Colorbar ---------
    cbar = plt.colorbar(pcm, ax=ax, ticks=[1, 2, 3], label="Risk Level")
    cbar.ax.set_yticklabels(["Low", "Moderate", "High"])

    # --------- Grid ---------
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    # --------- Layout ---------
    fig.tight_layout()

    # --------- Save or show ---------
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
