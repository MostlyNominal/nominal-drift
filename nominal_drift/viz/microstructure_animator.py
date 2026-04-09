"""
nominal_drift.viz.microstructure_animator
==========================================
Material-agnostic, microstructure-inspired animation driven by continuum
diffusion results.

PURPOSE
-------
Produces a grain-boundary / interface scene with particles whose size, colour,
density, and clustering evolve with the local concentration field.  The goal is
to visualise "what the mechanism looks like" — depletion, enrichment,
precipitate formation — in a way that is intuitive to engineers, students, and
non-specialists.

IMPORTANT SCIENTIFIC FRAMING
------------------------------
This is a **mechanism-inspired, continuum-driven, illustrative** visualisation.
Particle positions are seeded reproducibly and do NOT represent literal atomic
coordinates.  Colour and opacity derive from the solved concentration field
C(x, t).  This is NOT a molecular-dynamics or kinetic-Monte-Carlo trajectory.
A non-removable disclaimer is rendered on every frame.

MATERIAL GENERALITY
-------------------
The animator is material-agnostic:

* Particle colours and radii come from ``species_styles.py``, which provides
  periodic-table-inspired styling for arbitrary elements.
* The interface stripe label and background can be overridden for any material
  class (grain boundary, phase boundary, heterointerface, surface, etc.).
* An optional ``structure_record`` (CrystalRecord) or ``cif_path`` may be
  supplied for future structure-aware extensions.  In the current MVP these
  are logged in metadata only.

Public API
----------
``MicrostructureConfig``
    Frozen dataclass controlling scene layout, particle count, and labels.

``animate_microstructure(output, save_path, *, config, extra_species,
                         structure_record, cif_path, species_style_map, ...)``
    Build and save a microstructure-inspired animation as GIF.
    Returns the saved path as str.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.viz.species_styles import (
    SpeciesStyle,
    build_style_map,
    get_species_style,
)


# ---------------------------------------------------------------------------
# Scientific disclaimer (rendered on every frame — non-removable)
# ---------------------------------------------------------------------------

DISCLAIMER: str = (
    "Continuum-driven illustrative animation — NOT an atomistic simulation.  "
    "Particle positions are schematic; colours encode the solved C(x,t) field."
)


# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

_RNG_SEED: int = 42
_DPI: int = 90
_Y_MIN: float = -0.5
_Y_MAX: float = 0.5
_BOUNDARY_FRACTION: float = 0.07


# ---------------------------------------------------------------------------
# MicrostructureConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MicrostructureConfig:
    """Scene configuration for the microstructure animation.

    All parameters have sensible defaults.  Override individual fields to
    customise the look for different material classes.

    Attributes
    ----------
    n_primary : int
        Number of primary-species particles (the diffusing element).
    n_matrix : int
        Number of background matrix particles (e.g. Fe for steels, O for
        oxides, C for carbon structures).
    n_secondary : int
        Number of secondary-species particles (e.g. precipitate formers).
    boundary_colour : str
        Colour of the interface / grain-boundary stripe.
    boundary_label : str
        Label rendered inside the interface stripe.  Set to
        ``"Phase\\nBoundary"`` or ``"Surface"`` for non-steel materials.
    matrix_bg : str
        Canvas background colour.
    domain_label : str
        Label for the grain interior / bulk region.
    precipitate_zone_fraction : float
        Fraction of the domain (from x=0) where precipitate clusters appear
        at high depletion.  Set to 0.0 to disable clustering.
    precipitate_threshold : float
        Depletion-index threshold [0–1] above which precipitate markers appear.
    show_legend : bool
        Whether to render a legend with species labels.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.
    """
    n_primary: int = 500
    n_matrix: int = 300
    n_secondary: int = 100
    boundary_colour: str = "#374151"
    boundary_label: str = "Grain\nBoundary"
    matrix_bg: str = "#F8FAFC"
    domain_label: str = "Grain Interior"
    precipitate_zone_fraction: float = 0.25
    precipitate_threshold: float = 0.6
    show_legend: bool = True
    figsize: tuple = (11, 5.5)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_colour: str) -> np.ndarray:
    """Convert CSS hex to (3,) float64 array in [0, 1]."""
    h = hex_colour.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)],
                    dtype=np.float64)


def _make_rgba(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Append alpha channel to an (N, 3) or (3,) RGB array."""
    if rgb.ndim == 1:
        return np.append(rgb, alpha)
    a = np.full((rgb.shape[0], 1), alpha)
    return np.hstack([rgb, a])


# ---------------------------------------------------------------------------
# Internal: particle population
# ---------------------------------------------------------------------------

@dataclass
class _ParticlePopulation:
    """One set of scatter-plot particles for a single species role."""
    x: np.ndarray          # x positions (nm)
    y: np.ndarray          # y positions (canvas units)
    grid_idx: np.ndarray   # index into the spatial grid for each particle
    base_size: float       # matplotlib scatter size
    style: SpeciesStyle    # colour and label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_microstructure(
    output: DiffusionOutput,
    save_path: Union[str, Path],
    *,
    config: Union[MicrostructureConfig, dict, None] = None,
    extra_species: Optional[Dict[str, float]] = None,
    structure_record: Optional[Any] = None,
    cif_path: Optional[Union[str, Path]] = None,
    species_style_map: Optional[Dict[str, SpeciesStyle]] = None,
    title: Optional[str] = None,
    fps: int = 12,
    max_frames: int = 100,
    matrix_element: Optional[str] = None,
    secondary_element: Optional[str] = None,
) -> str:
    """Build and save a microstructure-inspired animation as a GIF.

    Parameters
    ----------
    output : DiffusionOutput
        Continuum diffusion result.  Provides C(x, t), element, boundaries.
    save_path : str or Path
        Destination path (extension normalised to ``.gif``).
    config : MicrostructureConfig | dict | None
        Scene configuration.  Dict → converted via ``MicrostructureConfig(**d)``.
    extra_species : dict, optional
        ``{element: bulk_wt_pct}`` for secondary species to visualise
        alongside the primary diffusing element.  Their concentration is
        treated as uniform (no coupled field in MVP).
    structure_record : any, optional
        Future extension point.  A ``CrystalRecord`` or similar structure
        object.  Currently logged in metadata only.
    cif_path : str or Path, optional
        Future extension point.  Path to a CIF file describing the material.
        Currently logged in metadata only.
    species_style_map : dict, optional
        ``{symbol: SpeciesStyle}`` overrides for element colours / radii.
    title : str, optional
        Title displayed above the animation canvas.
    fps : int
        Frames per second (default: 12).
    max_frames : int
        Maximum frames to render (default: 100).
    matrix_element : str, optional
        Element to use for background matrix particles (e.g. ``"Fe"``).
        If None, inferred from the output metadata or defaults to ``"Fe"``.
    secondary_element : str, optional
        Element for secondary / precipitate particles.  If None, ``"C"`` is
        used when the primary is ``"Cr"``, otherwise disabled.

    Returns
    -------
    str
        Absolute path to the saved GIF.
    """
    # ------------------------------------------------------------------
    # 0. Resolve config
    # ------------------------------------------------------------------
    if config is None:
        config = MicrostructureConfig()
    elif isinstance(config, dict):
        config = MicrostructureConfig(**config)

    # ------------------------------------------------------------------
    # 1. Resolve species and styles
    # ------------------------------------------------------------------
    primary_el = output.element
    if matrix_element is None:
        matrix_element = _infer_matrix_element(output)
    if secondary_element is None:
        secondary_element = _infer_secondary_element(primary_el)

    all_species = [primary_el]
    if matrix_element:
        all_species.append(matrix_element)
    if secondary_element:
        all_species.append(secondary_element)

    style_map = build_style_map(all_species, overrides=species_style_map)

    # ------------------------------------------------------------------
    # 2. Prepare concentration arrays
    # ------------------------------------------------------------------
    profiles = np.array(output.concentration_profiles, dtype=np.float64)
    x_nm = np.array(output.x_nm, dtype=np.float64)
    t_s = np.array(output.t_s, dtype=np.float64)
    n_stored = profiles.shape[0]

    n_frames = max(1, min(max_frames, n_stored))
    frame_idx = np.linspace(0, n_stored - 1, n_frames, dtype=int)

    denom = output.C_bulk_wt_pct - output.C_sink_wt_pct

    def _depletion(C: np.ndarray) -> np.ndarray:
        if denom <= 0.0:
            return np.zeros_like(C)
        return np.clip((output.C_bulk_wt_pct - C) / denom, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 3. Seed particle populations
    # ------------------------------------------------------------------
    rng = np.random.default_rng(_RNG_SEED)
    x_max_nm = float(x_nm[-1])
    boundary_nm = max(_BOUNDARY_FRACTION * x_max_nm, 1.0)

    def _seed_pop(n: int, element: str, size_scale: float = 1.0) -> _ParticlePopulation:
        px = rng.uniform(boundary_nm, x_max_nm, n)
        py = rng.uniform(_Y_MIN, _Y_MAX, n)
        idx = np.searchsorted(x_nm, px).clip(0, len(x_nm) - 1)
        s = style_map.get(element, get_species_style(element))
        base_size = max(3.0, 12.0 * s.radius * size_scale)
        return _ParticlePopulation(
            x=px, y=py, grid_idx=idx,
            base_size=base_size, style=s,
        )

    pops = []
    # Primary (diffusing) species
    pop_primary = _seed_pop(config.n_primary, primary_el, size_scale=1.0)
    pops.append(("primary", pop_primary))

    # Matrix (background) species
    if matrix_element and config.n_matrix > 0:
        pop_matrix = _seed_pop(config.n_matrix, matrix_element, size_scale=0.9)
        pops.append(("matrix", pop_matrix))

    # Secondary (precipitate) species
    if secondary_element and config.n_secondary > 0:
        pop_sec = _seed_pop(config.n_secondary, secondary_element, size_scale=0.5)
        pops.append(("secondary", pop_sec))

    # ------------------------------------------------------------------
    # 4. Build figure
    # ------------------------------------------------------------------
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=config.figsize)
    fig.patch.set_facecolor(config.matrix_bg)
    ax.set_facecolor(config.matrix_bg)

    # Interface / grain boundary stripe
    ax.axvspan(0.0, boundary_nm,
               color=config.boundary_colour, alpha=0.92, zorder=1)
    ax.text(
        boundary_nm * 0.5, 0.0,
        config.boundary_label,
        ha="center", va="center", fontsize=7,
        color="white", fontweight="bold", zorder=5,
    )

    # Domain label
    ax.text(
        x_max_nm * 0.75, _Y_MAX * 0.90,
        config.domain_label,
        ha="center", va="top", fontsize=8,
        color="#9CA3AF", fontstyle="italic", zorder=5,
    )

    # Precipitate zone indicator (faint shading near boundary)
    if config.precipitate_zone_fraction > 0:
        pz_end = boundary_nm + config.precipitate_zone_fraction * (x_max_nm - boundary_nm)
        ax.axvspan(boundary_nm, pz_end,
                   color="#FEF3C7", alpha=0.25, zorder=0)
        ax.text(
            (boundary_nm + pz_end) / 2, _Y_MIN * 0.88,
            "Precipitate\nZone",
            ha="center", va="bottom", fontsize=6,
            color="#D97706", fontstyle="italic", alpha=0.7, zorder=5,
        )

    # ------------------------------------------------------------------
    # 5. Initial scatter artists
    # ------------------------------------------------------------------
    scatters = {}
    primary_rgb = _hex_to_rgb(pop_primary.style.colour)
    depleted_rgb = _hex_to_rgb("#D1D5DB")  # universal depleted colour

    for role, pop in pops:
        rgb = _hex_to_rgb(pop.style.colour)
        if role == "primary":
            # Primary particles: coloured by depletion
            d0 = _depletion(profiles[frame_idx[0]])
            d0_p = d0[pop.grid_idx]
            rgba = _blend_primary(d0_p, primary_rgb, depleted_rgb)
            s = ax.scatter(pop.x, pop.y, c=rgba, s=pop.base_size,
                           zorder=3, linewidths=0)
        elif role == "matrix":
            # Matrix particles: static, low-opacity
            rgba = _make_rgba(rgb, 0.20)
            s = ax.scatter(pop.x, pop.y, c=[rgba] * len(pop.x),
                           s=pop.base_size, zorder=2, linewidths=0)
        elif role == "secondary":
            # Secondary particles: appear in precipitate zone at high depletion
            rgba = _make_rgba(rgb, 0.0)  # initially invisible
            s = ax.scatter(pop.x, pop.y, c=[rgba] * len(pop.x),
                           s=pop.base_size, zorder=4, linewidths=0,
                           marker="D")
        scatters[role] = s

    # ------------------------------------------------------------------
    # 6. Annotations (updated per frame)
    # ------------------------------------------------------------------
    # Temperature annotation
    T_C_str = ""
    if output.metadata and "ht_schedule_summary" in output.metadata:
        ht_sum = output.metadata["ht_schedule_summary"]
        if ht_sum:
            T_C_str = f"  |  T = {ht_sum[-1].get('T_hold_C', '?')} °C"

    t_min_0 = t_s[frame_idx[0]] / 60.0
    time_text = ax.text(
        x_max_nm * 0.97, _Y_MAX * 0.90,
        f"t = {t_min_0:.1f} min{T_C_str}",
        ha="right", va="top", fontsize=9,
        color="#1F2937", fontweight="bold", zorder=6,
    )

    # Element label
    ax.text(
        x_max_nm * 0.97, _Y_MIN * 0.90,
        f"[{primary_el}]",
        ha="right", va="bottom", fontsize=10,
        color=pop_primary.style.colour, fontweight="bold", zorder=6,
    )

    # Legend
    if config.show_legend:
        handles = []
        for role, pop in pops:
            handles.append(mpatches.Patch(
                color=pop.style.colour,
                label=pop.style.label or role.capitalize(),
            ))
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=7,
            framealpha=0.7,
            edgecolor="#D1D5DB",
        )

    # Axes cosmetics
    ax.set_xlim(0.0, x_max_nm)
    ax.set_ylim(_Y_MIN, _Y_MAX)
    ax.set_xlabel("Distance from interface (nm)", fontsize=9, color="#374151")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#6B7280", labelsize=8)

    if title:
        ax.set_title(title, fontsize=11, pad=12, color="#1F2937")

    # Non-removable disclaimer
    fig.text(
        0.5, 0.005,
        DISCLAIMER,
        ha="center", va="bottom", fontsize=6,
        color="#9CA3AF", fontstyle="italic",
    )

    fig.tight_layout(rect=[0.0, 0.045, 1.0, 1.0])

    # ------------------------------------------------------------------
    # 7. Animation update
    # ------------------------------------------------------------------
    pz_end_nm = boundary_nm + config.precipitate_zone_fraction * (x_max_nm - boundary_nm)

    def _update(frame_num: int):
        fi = frame_idx[frame_num]
        C = profiles[fi]
        d = _depletion(C)

        # Update primary particles
        d_p = d[pop_primary.grid_idx]
        rgba_p = _blend_primary(d_p, primary_rgb, depleted_rgb)
        scatters["primary"].set_facecolor(rgba_p)

        # Update secondary (precipitate) particles if present
        if "secondary" in scatters:
            pop_s = dict(pops)["secondary"]
            d_s = d[pop_s.grid_idx]
            sec_rgb = _hex_to_rgb(pop_s.style.colour)
            # Alpha scales with local depletion AND proximity to boundary
            in_zone = pop_s.x < pz_end_nm
            above_thresh = d_s > config.precipitate_threshold
            visible = in_zone & above_thresh
            alpha = np.where(visible, np.clip(d_s * 0.9, 0.0, 0.85), 0.0)
            rgba_s = np.zeros((len(pop_s.x), 4))
            rgba_s[:, :3] = sec_rgb
            rgba_s[:, 3] = alpha
            scatters["secondary"].set_facecolor(rgba_s)

        # Update time text
        t_min = t_s[fi] / 60.0
        time_text.set_text(f"t = {t_min:.1f} min{T_C_str}")

        return tuple(scatters.values()) + (time_text,)

    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        blit=False,
        interval=max(1, 1000 // fps),
    )

    # ------------------------------------------------------------------
    # 8. Save as GIF
    # ------------------------------------------------------------------
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".gif":
        save_path = save_path.with_suffix(".gif")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=fps)
    anim.save(str(save_path), writer=writer, dpi=_DPI)
    plt.close(fig)

    return str(save_path.resolve())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _blend_primary(
    depletion: np.ndarray,
    colour_full: np.ndarray,
    colour_depleted: np.ndarray,
) -> np.ndarray:
    """Blend primary particle RGBA based on local depletion index.

    Parameters
    ----------
    depletion : (N,) array in [0, 1]
        0 = full (bulk) concentration, 1 = fully depleted.
    colour_full, colour_depleted : (3,) RGB arrays.

    Returns
    -------
    (N, 4) RGBA array.
    """
    t = depletion[:, None]
    rgb = (1.0 - t) * colour_full + t * colour_depleted
    alpha = 0.88 - 0.55 * depletion  # fade as depleted
    return np.hstack([rgb, alpha[:, None]])


def _infer_matrix_element(output: DiffusionOutput) -> str:
    """Infer the dominant matrix element from metadata or defaults."""
    matrix_str = output.matrix.lower()
    # Check compound / structural keywords first (before element substrings)
    if "oxide" in matrix_str or "perovskite" in matrix_str:
        return "O"
    if "fe" in matrix_str:
        return "Fe"
    if "ni" in matrix_str:
        return "Ni"
    if "al" in matrix_str:
        return "Al"
    if "ti" in matrix_str:
        return "Ti"
    if "cu" in matrix_str:
        return "Cu"
    if "co" in matrix_str:
        return "Co"
    if "si" in matrix_str:
        return "Si"
    if "zr" in matrix_str:
        return "Zr"
    return "Fe"  # safe default


def _infer_secondary_element(primary: str) -> Optional[str]:
    """Infer a secondary species relevant to the primary diffusing element."""
    # Common metallurgical pairings
    pairings = {
        "Cr": "C",    # Cr depletion → carbide precipitation
        "C":  "Cr",   # Carbon diffusion → Cr-carbide
        "N":  "Cr",   # Nitrogen → Cr-nitride
        "Ni": "Al",   # Ni diffusion → gamma-prime (Al)
        "Al": "Cu",   # Al alloys → Cu precipitates
        "Cu": "Sn",   # Cu → tin precipitates
        "O":  "Ti",   # Oxygen diffusion → TiO2
        "Ti": "O",    # Ti → oxide scale
    }
    return pairings.get(primary)
