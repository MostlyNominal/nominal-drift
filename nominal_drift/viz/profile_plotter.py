"""
nominal_drift.viz.profile_plotter
==================================
Static concentration-profile plotter for 1D diffusion simulation results.

Produces a single engineering-style PNG showing how the elemental concentration
evolves from the grain boundary (x = 0) into the grain interior (x = x_max)
at a set of representative time snapshots.

Design principles
-----------------
- Element-agnostic: works for Cr, C, N, or any future diffusing species.
  All axis labels, legends, and annotations use the ``element`` field from
  the DiffusionOutput directly — nothing is hard-coded around chromium.
- Self-contained: reads everything it needs from the validated DiffusionOutput
  object; no extra parameters required beyond the save path.
- Engineering style: clean white background, fine grid, SI-prefixed axes,
  publication-ready font sizes.
- Non-interactive: uses the Agg backend explicitly so the function works in
  headless CI, HPC, and Jupyter environments without a display.

Public API
----------
``plot_concentration_profile(output, save_path, title, threshold_wt_pct)``
    Create and save the plot.  Returns the saved path as a str.

Extending for future species
-----------------------------
Because all labels are derived from ``output.element``, ``output.matrix``,
``output.C_bulk_wt_pct``, and ``output.C_sink_wt_pct``, this plotter
requires no modification to support:
    - C and N depletion (interstitial sensitization tracks)
    - Future DFT-informed mobility datasets with different matrix labels
    - Multi-element outputs (call this function once per DiffusionOutput in
      the list returned by the multi-species orchestrator)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")   # enforce non-interactive backend; must precede pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from nominal_drift.schemas.diffusion_output import DiffusionOutput

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

#: Figure dimensions in inches (width × height)
_FIG_SIZE: tuple[float, float] = (10.0, 6.0)

#: Resolution for saved PNG
_DPI: int = 150

#: Base font size for axis labels and tick labels
_FONT_SIZE: int = 11

#: Font size for the legend
_LEGEND_FONT_SIZE: int = 9

#: Colours for the five representative time curves.
#: Index 0 = initial (t = 0), index 4 = final.  Interior snapshots use
#: a sequential ramp from mid-blue to deep blue so that temporal progression
#: reads left-to-right on the colour axis.
_CURVE_COLOURS: list[str] = [
    "#b0bec5",   # 0 — light blue-grey  (initial / t = 0)
    "#64b5f6",   # 1 — sky blue         (~25 % of hold)
    "#1e88e5",   # 2 — medium blue      (~50 % of hold)
    "#1565c0",   # 3 — royal blue       (~75 % of hold)
    "#0d1b2a",   # 4 — near-black navy  (final)
]

#: Line styles matching the colour list
_CURVE_STYLES: list[str] = ["--", "-", "-", "-", "-"]

#: Line widths matching the colour list
_CURVE_WIDTHS: list[float] = [1.2, 1.4, 1.6, 1.8, 2.2]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_snapshot_indices(n_stored: int) -> list[int]:
    """Return up to 5 de-duplicated indices spanning the stored time axis.

    Always includes the first (t = 0) and last (t = final) index.  The three
    intermediate indices target the 25 %, 50 %, and 75 % marks.

    Parameters
    ----------
    n_stored : int
        Total number of stored time steps, including t = 0.

    Returns
    -------
    list[int]
        Sorted, de-duplicated list of at most 5 integer indices.
    """
    if n_stored <= 1:
        return [0]

    targets = [
        0,
        int(round(0.25 * (n_stored - 1))),
        int(round(0.50 * (n_stored - 1))),
        int(round(0.75 * (n_stored - 1))),
        n_stored - 1,
    ]
    # De-duplicate while preserving order
    seen: set[int] = set()
    indices: list[int] = []
    for idx in targets:
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _format_time_label(t_s: float) -> str:
    """Format a time value in seconds as a human-readable label.

    Uses minutes for values ≥ 1 min, seconds for shorter durations.

    Parameters
    ----------
    t_s : float
        Elapsed time in seconds.

    Returns
    -------
    str
        Formatted label, e.g. ``"t = 0.0 min"``, ``"t = 30 s"``.
    """
    if t_s < 60.0:
        return f"t = {t_s:.0f} s"
    minutes = t_s / 60.0
    if minutes < 10.0:
        return f"t = {minutes:.1f} min"
    return f"t = {minutes:.0f} min"


def _make_footer_text(output: DiffusionOutput) -> str:
    """Build the footer metadata string shown below the plot area.

    Parameters
    ----------
    output : DiffusionOutput
        Simulation result whose provenance fields are used.

    Returns
    -------
    str
        Single-line string with element, matrix, bulk and sink concentrations.
    """
    return (
        f"Element: {output.element}  |  "
        f"Matrix: {output.matrix}  |  "
        f"C\u2080 (bulk) = {output.C_bulk_wt_pct:.2f} wt%  |  "
        f"C\u209B (sink) = {output.C_sink_wt_pct:.2f} wt%"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_concentration_profile(
    output: DiffusionOutput,
    save_path: Union[str, Path],
    title: str | None = None,
    threshold_wt_pct: float | None = None,
) -> str:
    """Create and save a static concentration-profile plot.

    Plots the elemental concentration as a function of distance from the
    grain boundary at up to five representative time snapshots: the initial
    profile (t = 0), snapshots at approximately 25 %, 50 %, and 75 % of the
    total hold time, and the final profile.  Curves are colour-coded from
    light (early) to dark (late) so that the temporal evolution reads
    intuitively.

    Parameters
    ----------
    output : DiffusionOutput
        Validated simulation result produced by ``solve_diffusion()``.
        All plot content (axis scales, labels, annotations) is derived from
        this object — no element-specific logic is hard-coded.
    save_path : str or Path
        Destination for the PNG file.  Parent directory must exist.
    title : str | None
        Optional main title for the figure.  If ``None``, an auto-generated
        title is used: ``"<Element> Concentration Profile — <matrix>"``.
    threshold_wt_pct : float | None
        If provided, a horizontal dashed red line is drawn at this
        concentration value, labelled as a reference threshold.  Useful for
        marking the sensitization criterion (e.g. 12 wt% for Cr depletion)
        or any other engineering limit.

    Returns
    -------
    str
        Absolute path to the saved PNG file as a plain Python ``str``.

    Notes
    -----
    The function uses the Agg (non-interactive) matplotlib backend so it can
    run safely in headless environments (CI, HPC, Docker).  It closes the
    figure after saving to release memory — call it multiple times without
    fear of resource leaks.

    Examples
    --------
    >>> from nominal_drift.viz.profile_plotter import plot_concentration_profile
    >>> path = plot_concentration_profile(
    ...     result,
    ...     "outputs/cr_profile_650C_120min.png",
    ...     title="316L — sensitization at 650 °C for 120 min",
    ...     threshold_wt_pct=12.0,
    ... )
    >>> print(path)
    outputs/cr_profile_650C_120min.png
    """
    save_path = Path(save_path)

    # ------------------------------------------------------------------
    # 1. Select representative time indices
    # ------------------------------------------------------------------
    n_stored = len(output.t_s)
    snap_indices = _pick_snapshot_indices(n_stored)

    # Assign colours / styles cycling through the palette if we have fewer
    # than 5 distinct curves (handles the 2-step edge case).
    n_curves = len(snap_indices)
    if n_curves >= 5:
        colours = _CURVE_COLOURS
        styles  = _CURVE_STYLES
        widths  = _CURVE_WIDTHS
    else:
        # Subsample palette: always use first colour for t=0 and last for final
        palette_indices = [int(round(4 * k / (n_curves - 1))) for k in range(n_curves)]
        colours = [_CURVE_COLOURS[i] for i in palette_indices]
        styles  = [_CURVE_STYLES[i]  for i in palette_indices]
        widths  = [_CURVE_WIDTHS[i]  for i in palette_indices]

    # ------------------------------------------------------------------
    # 2. Prepare spatial axis in nanometres
    # ------------------------------------------------------------------
    x_nm = [x * 1e9 for x in output.x_m]

    # ------------------------------------------------------------------
    # 3. Build figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=_FIG_SIZE, dpi=_DPI)
    fig.subplots_adjust(bottom=0.18)   # reserve space for footer text

    # ------------------------------------------------------------------
    # 4. Plot concentration curves
    # ------------------------------------------------------------------
    for curve_idx, snap_idx in enumerate(snap_indices):
        t_seconds = output.t_s[snap_idx]
        profile   = output.concentration_profiles[snap_idx]
        label     = _format_time_label(t_seconds)

        # Distinguish initial profile visually
        zorder = 2 + curve_idx
        ax.plot(
            x_nm,
            profile,
            color     = colours[curve_idx],
            linestyle = styles[curve_idx],
            linewidth = widths[curve_idx],
            label     = label,
            zorder    = zorder,
        )

    # ------------------------------------------------------------------
    # 5. Optional threshold reference line
    # ------------------------------------------------------------------
    if threshold_wt_pct is not None:
        ax.axhline(
            y          = threshold_wt_pct,
            color      = "#e53935",   # Material Design red 600
            linestyle  = "--",
            linewidth  = 1.4,
            alpha      = 0.85,
            label      = f"Threshold: {threshold_wt_pct:.2f} wt%",
            zorder     = 10,
        )

    # ------------------------------------------------------------------
    # 6. Annotate minimum concentration at the grain boundary (x = 0)
    #    on the final profile
    # ------------------------------------------------------------------
    final_profile = output.concentration_profiles[snap_indices[-1]]
    c_min_gb      = final_profile[0]

    ax.annotate(
        f"GB min\n{c_min_gb:.2f} wt%",
        xy       = (x_nm[0], c_min_gb),
        xytext   = (max(x_nm) * 0.05, c_min_gb + 0.05 * (output.C_bulk_wt_pct - output.C_sink_wt_pct + 1e-9)),
        fontsize = _FONT_SIZE - 1,
        color    = _CURVE_COLOURS[-1],
        arrowprops = dict(
            arrowstyle = "->",
            color      = _CURVE_COLOURS[-1],
            lw         = 1.0,
        ),
        zorder = 20,
    )

    # ------------------------------------------------------------------
    # 7. Axis labels, limits, grid
    # ------------------------------------------------------------------
    ax.set_xlabel("Distance from grain boundary (nm)", fontsize=_FONT_SIZE)
    ax.set_ylabel(f"{output.element} concentration (wt%)", fontsize=_FONT_SIZE)

    # Y-axis: add 5 % padding above C_bulk and below C_sink
    driving_force = output.C_bulk_wt_pct - output.C_sink_wt_pct
    pad = max(0.05 * driving_force, 0.05)
    y_lo = output.C_sink_wt_pct - pad
    y_hi = output.C_bulk_wt_pct + pad
    # Extend upper bound if threshold is above C_bulk
    if threshold_wt_pct is not None:
        y_lo = min(y_lo, threshold_wt_pct - pad)
        y_hi = max(y_hi, threshold_wt_pct + pad)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(left=0.0, right=max(x_nm))

    # Minor ticks and grid
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6, color="#cccccc")
    ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.4, color="#e0e0e0")
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="major", labelsize=_FONT_SIZE - 1)

    # ------------------------------------------------------------------
    # 8. Title
    # ------------------------------------------------------------------
    plot_title = title if title is not None else (
        f"{output.element} Concentration Profile — {output.matrix}"
    )
    ax.set_title(plot_title, fontsize=_FONT_SIZE + 2, fontweight="bold", pad=10)

    # ------------------------------------------------------------------
    # 9. Legend
    # ------------------------------------------------------------------
    legend = ax.legend(
        loc            = "lower right",
        fontsize       = _LEGEND_FONT_SIZE,
        framealpha     = 0.92,
        edgecolor      = "#aaaaaa",
        title          = "Time snapshot",
        title_fontsize = _LEGEND_FONT_SIZE,
    )

    # ------------------------------------------------------------------
    # 10. Footer metadata text
    # ------------------------------------------------------------------
    footer = _make_footer_text(output)
    fig.text(
        0.5, 0.03,
        footer,
        ha         = "center",
        va         = "bottom",
        fontsize   = _FONT_SIZE - 2,
        color      = "#555555",
        style      = "italic",
    )

    # Thin border around the axes for a clean engineering look
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")

    # ------------------------------------------------------------------
    # 11. Save and clean up
    # ------------------------------------------------------------------
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)
