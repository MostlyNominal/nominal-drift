"""
nominal_drift.viz.mechanism_animator
======================================
Mechanism-inspired schematic animation for 1D continuum diffusion results.

PURPOSE
-------
This module produces a **second, separate visualisation mode** alongside the
quantitative engineering animation in ``animator.py``.  Where ``animator.py``
shows a precise scientific concentration-vs-distance plot, this module
produces a stylised **material scene** that communicates the physical
mechanism to a broader audience:

  * a grain-boundary stripe on the left
  * a grain interior on the right
  * coloured particles whose appearance encodes local species concentration
  * the visual evolving frame-by-frame as diffusion proceeds

IMPORTANT SCIENTIFIC FRAMING
------------------------------
The particle positions and colours are a **visual encoding of the
continuum concentration field C(x, t)** — they are NOT literal atom
trajectories.  The depletion index

    depletion_index(x, t) = (C_bulk − C(x, t)) / (C_bulk − C_sink)

drives particle colour and opacity.  No kinetic Monte-Carlo or molecular
dynamics is performed.  This distinction must be communicated to viewers;
the module enforces a non-removable disclaimer on every animation frame.

DESIGN
------
* Particles are seeded once with a fixed RNG seed (42) for reproducibility.
* Only their RGBA colour updates each frame — positions never change.
* No ``ffmpeg`` dependency: GIF output via ``PillowWriter`` only.
* Headless: uses the Agg matplotlib backend.
* Element-agnostic: all labels derive from ``DiffusionOutput.element``.

Public API
----------
``MechanismScheme``
    Lightweight dataclass controlling colours, labels, and disclaimer text.

``animate_mechanism(output, save_path, *, scheme, title, fps, max_frames)``
    Build and save the mechanism animation.  Returns the saved path as str.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from nominal_drift.schemas.diffusion_output import DiffusionOutput

# ---------------------------------------------------------------------------
# Non-removable scientific disclaimer (must appear on every frame)
# ---------------------------------------------------------------------------

DISCLAIMER: str = (
    "Continuum model — mechanism-inspired illustration.  "
    "Not an atomistic simulation."
)

# ---------------------------------------------------------------------------
# Internal rendering constants
# ---------------------------------------------------------------------------

_RNG_SEED: int = 42          # fixed seed → reproducible particle layout
_N_PARTICLES: int = 600      # total dot count across the grain interior
_Y_MIN: float = -0.5         # canvas y-extent (arbitrary units)
_Y_MAX: float = 0.5
_BOUNDARY_FRACTION: float = 0.06   # boundary stripe width / x_max
_DPI: int = 80               # GIF raster resolution


# ---------------------------------------------------------------------------
# MechanismScheme
# ---------------------------------------------------------------------------

@dataclass
class MechanismScheme:
    """Visual scheme for the mechanism animation.

    All colour values are CSS hex strings (e.g. ``"#2563EB"``).

    Attributes
    ----------
    species_colour : str
        Colour for particles at full (bulk) concentration.
    depleted_colour : str
        Colour for fully depleted particles (at sink concentration).
    matrix_colour : str
        Background colour for the grain-interior region.
    boundary_colour : str
        Fill colour for the grain-boundary stripe.
    boundary_label : str
        Short label printed inside the boundary stripe.
    domain_label : str
        Italic label printed in the grain-interior region.
    disclaimer : str
        Footer text printed on every frame.  Default is the module-level
        ``DISCLAIMER`` constant.  Callers *may* extend it, but should not
        remove the core scientific caveat.
    """

    species_colour:   str = "#2563EB"   # steel-blue → enriched
    depleted_colour:  str = "#D1D5DB"   # light gray  → depleted
    matrix_colour:    str = "#F8FAFC"   # near-white background
    boundary_colour:  str = "#374151"   # dark slate  → boundary stripe
    boundary_label:   str = "Grain\nBoundary"
    domain_label:     str = "Grain Interior"
    disclaimer:       str = field(default_factory=lambda: DISCLAIMER)


# ---------------------------------------------------------------------------
# Internal colour utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_colour: str) -> np.ndarray:
    """Convert a CSS hex colour string to a (3,) float64 RGB array in [0, 1]."""
    h = hex_colour.lstrip("#")
    return np.array([int(h[i: i + 2], 16) / 255.0 for i in (0, 2, 4)],
                    dtype=np.float64)


def _blend_rgba(
    t: np.ndarray,
    colour_a: np.ndarray,
    colour_b: np.ndarray,
    alpha_max: float = 0.88,
    alpha_min: float = 0.18,
) -> np.ndarray:
    """Return an (N, 4) RGBA array blending colour_a→colour_b by parameter t.

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Blend parameter in [0, 1].  t=0 → colour_a, t=1 → colour_b.
    colour_a, colour_b : np.ndarray, shape (3,)
        RGB endpoints in [0, 1].
    alpha_max, alpha_min : float
        Alpha at t=0 and t=1 respectively (particles fade as they deplete).
    """
    rgb = (1.0 - t[:, None]) * colour_a + t[:, None] * colour_b  # (N, 3)
    alpha = alpha_max - (alpha_max - alpha_min) * t               # (N,)
    return np.hstack([rgb, alpha[:, None]])                       # (N, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_mechanism(
    output: DiffusionOutput,
    save_path: Union[str, Path],
    scheme: Union[MechanismScheme, dict, None] = None,
    title: str | None = None,
    fps: int = 15,
    max_frames: int = 120,
) -> str:
    """Build and save a mechanism-inspired diffusion animation as a GIF.

    The animation visualises the continuum concentration field C(x, t) as a
    material scene: a grain-boundary stripe on the left and a grain interior
    on the right, with coloured particles whose appearance encodes local
    species depletion.

    .. note::
        Particle positions are driven by a **continuum field**, not literal
        atomic trajectories.  A scientific disclaimer is printed on every
        frame and cannot be removed.

    Parameters
    ----------
    output : DiffusionOutput
        Result of ``solve_diffusion()``.  Provides the concentration field,
        spatial grid, time axis, element symbol, and boundary conditions.
    save_path : str or Path
        Destination file path.  The extension is always normalised to
        ``.gif`` — GIF is the only format supported by this MVP function.
    scheme : MechanismScheme | dict | None
        Visual scheme.  Pass a ``MechanismScheme`` instance, a dict of
        keyword arguments to ``MechanismScheme``, or ``None`` for defaults.
    title : str | None
        Optional title displayed above the animation canvas.
    fps : int
        Frames per second in the output GIF.  Default: 15.
    max_frames : int
        Maximum number of animation frames.  If the solver stored more
        profiles than ``max_frames``, they are subsampled uniformly.
        Default: 120.

    Returns
    -------
    str
        Absolute path to the saved GIF file (extension always ``.gif``).
    """
    # ------------------------------------------------------------------
    # 0.  Resolve scheme
    # ------------------------------------------------------------------
    if scheme is None:
        scheme = MechanismScheme()
    elif isinstance(scheme, dict):
        scheme = MechanismScheme(**scheme)

    # ------------------------------------------------------------------
    # 1.  Prepare arrays
    # ------------------------------------------------------------------
    profiles = np.array(output.concentration_profiles, dtype=np.float64)
    # profiles shape: (n_stored, n_spatial)

    x_nm_arr = np.array(output.x_nm, dtype=np.float64)      # nm
    t_s_arr  = np.array(output.t_s,  dtype=np.float64)       # seconds
    n_stored = len(output.t_s)

    # Subsample frame indices
    n_frames = max(1, min(max_frames, n_stored))
    frame_indices = np.linspace(0, n_stored - 1, n_frames, dtype=int)

    # Depletion-index denominator (guard for no driving force)
    denom = output.C_bulk_wt_pct - output.C_sink_wt_pct

    def _depletion_index(profile: np.ndarray) -> np.ndarray:
        if denom <= 0.0:
            return np.zeros_like(profile)
        return np.clip((output.C_bulk_wt_pct - profile) / denom, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2.  Seed particles (positions fixed for entire animation)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(_RNG_SEED)
    x_max_nm        = float(x_nm_arr[-1])
    boundary_nm     = max(_BOUNDARY_FRACTION * x_max_nm, 0.5)

    px = rng.uniform(boundary_nm, x_max_nm, _N_PARTICLES)  # x in grain interior
    py = rng.uniform(_Y_MIN, _Y_MAX, _N_PARTICLES)

    # Map each particle to its nearest x-grid index
    particle_idx = np.searchsorted(x_nm_arr, px).clip(0, len(x_nm_arr) - 1)

    # Pre-compute colour endpoints
    species_rgb  = _hex_to_rgb(scheme.species_colour)
    depleted_rgb = _hex_to_rgb(scheme.depleted_colour)

    # ------------------------------------------------------------------
    # 3.  Build figure
    # ------------------------------------------------------------------
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(scheme.matrix_colour)
    ax.set_facecolor(scheme.matrix_colour)

    # Grain-boundary stripe
    ax.axvspan(0.0, boundary_nm,
               color=scheme.boundary_colour, alpha=0.92, zorder=1)
    ax.text(
        boundary_nm * 0.5, 0.0,
        scheme.boundary_label,
        ha="center", va="center", fontsize=7,
        color="white", fontweight="bold", zorder=3,
    )

    # Domain label (grain interior)
    ax.text(
        x_max_nm * 0.72, _Y_MAX * 0.88,
        scheme.domain_label,
        ha="center", va="top", fontsize=8,
        color="#9CA3AF", fontstyle="italic", zorder=3,
    )

    # Species concentration label
    ax.text(
        x_max_nm * 0.72, _Y_MIN * 0.82,
        f"[{output.element}]",
        ha="center", va="bottom", fontsize=10,
        color=scheme.species_colour, fontweight="bold", zorder=3,
    )

    # Initial particle colours
    d0       = _depletion_index(profiles[frame_indices[0]])
    d0_p     = d0[particle_idx]
    init_rgba = _blend_rgba(d0_p, species_rgb, depleted_rgb)

    scatter = ax.scatter(
        px, py,
        c=init_rgba,
        s=9,
        zorder=2,
        linewidths=0,
    )

    # Time annotation (updates each frame)
    t_min_0   = t_s_arr[frame_indices[0]] / 60.0
    time_text = ax.text(
        x_max_nm * 0.97, _Y_MAX * 0.88,
        f"t = {t_min_0:.1f} min",
        ha="right", va="top", fontsize=9,
        color="#1F2937", fontweight="bold", zorder=4,
    )

    # Total-time reference (static, bottom-right of axes)
    ax.text(
        x_max_nm * 0.97, _Y_MIN * 0.88,
        f"Total: {output.total_time_min:.1f} min",
        ha="right", va="bottom", fontsize=7.5,
        color="#9CA3AF", zorder=4,
    )

    # Axes cosmetics
    ax.set_xlim(0.0, x_max_nm)
    ax.set_ylim(_Y_MIN, _Y_MAX)
    ax.set_xlabel("Distance from grain boundary (nm)", fontsize=9, color="#374151")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#6B7280", labelsize=8)

    if title:
        ax.set_title(title, fontsize=10, pad=10, color="#1F2937")

    # Non-removable disclaimer footer (figure-level, below axes)
    fig.text(
        0.5, 0.005,
        scheme.disclaimer,
        ha="center", va="bottom", fontsize=6.5,
        color="#9CA3AF", fontstyle="italic",
    )

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])

    # ------------------------------------------------------------------
    # 4.  Animation update function
    # ------------------------------------------------------------------
    def _update(frame_num: int):
        fi  = frame_indices[frame_num]
        C   = profiles[fi]
        d   = _depletion_index(C)
        d_p = d[particle_idx]

        rgba = _blend_rgba(d_p, species_rgb, depleted_rgb)
        scatter.set_facecolor(rgba)

        t_min = t_s_arr[fi] / 60.0
        time_text.set_text(f"t = {t_min:.1f} min")

        return scatter, time_text

    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        blit=False,
        interval=max(1, 1000 // fps),
    )

    # ------------------------------------------------------------------
    # 5.  Save as GIF (no ffmpeg required)
    # ------------------------------------------------------------------
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".gif":
        save_path = save_path.with_suffix(".gif")

    writer = PillowWriter(fps=fps)
    anim.save(str(save_path), writer=writer, dpi=_DPI)
    plt.close(fig)

    return str(save_path)
