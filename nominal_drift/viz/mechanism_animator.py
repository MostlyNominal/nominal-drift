"""
nominal_drift.viz.mechanism_animator
======================================
Mechanism-inspired schematic animation for 1D continuum diffusion results.

PURPOSE
-------
Produces a dual-panel animation:

  Top panel  — concentration profile C(x, t) plotted as a line against
               distance.  This panel moves visibly every frame because the
               solver outputs 200+ concentration snapshots.  It is the same
               data shown by ``animator.py``'s ``animate_diffusion()``,
               but cropped to the depletion region for legibility.

  Bottom panel — grain-boundary / grain-interior schematic with coloured
               particles whose RGBA encodes local depletion from the solver
               concentration field.  Particles are concentrated in the
               depletion region (x < ``zoom_nm``) so colour changes are
               visually prominent.

SCIENTIFIC CONSTRAINTS
-----------------------
* Both panels derive entirely from the Crank–Nicolson C(x,t) field.
* No molecular dynamics, kinetic Monte-Carlo, DFT, CALPHAD, precipitation
  or phase kinetics are modelled.
* A non-removable disclaimer is rendered on every frame.
* Particle positions are FIXED for the entire animation.
  Only RGBA colour updates frame-by-frame.

Public API
----------
``MechanismScheme``
    Lightweight dataclass controlling colours, labels, and disclaimer text.

``animate_mechanism(output, save_path, scheme, title, fps, max_frames)``
    Build and save a dual-panel GIF.  Returns the saved path as str.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from nominal_drift.schemas.diffusion_output import DiffusionOutput

# ---------------------------------------------------------------------------
# Non-removable scientific disclaimer (rendered on every frame)
# ---------------------------------------------------------------------------

DISCLAIMER: str = (
    "Continuum model — particle colours encode Crank–Nicolson C(x,t) field.  "
    "No atomistic simulation.  No precipitation / phase kinetics modelled."
)

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

_RNG_SEED: int = 42
_N_PARTICLES: int = 500   # particles concentrated inside the zoom window
_DPI: int = 90
_Y_MIN: float = -0.5
_Y_MAX: float = 0.5
_BOUNDARY_FRACTION: float = 0.06  # boundary stripe width / zoom_nm
# How many multiples of the final-profile depletion depth to show
_ZOOM_MULTIPLIER: float = 5.0
_ZOOM_MIN_NM: float = 200.0   # minimum zoom window even if depletion is tiny


# ---------------------------------------------------------------------------
# MechanismScheme
# ---------------------------------------------------------------------------

@dataclass
class MechanismScheme:
    """Visual scheme for the mechanism animation.

    Attributes
    ----------
    species_colour : str
        CSS hex colour for particles at full bulk concentration.
    depleted_colour : str
        CSS hex colour for fully depleted particles.
    matrix_colour : str
        Background colour for grain-interior region.
    boundary_colour : str
        Fill colour for the grain-boundary stripe.
    boundary_label : str
        Label printed inside the boundary stripe.
    domain_label : str
        Italic label for the grain-interior region.
    profile_colour : str
        Line colour for the concentration profile in the top panel.
    disclaimer : str
        Footer text on every frame.
    """
    species_colour:   str = "#2563EB"   # steel-blue → enriched
    depleted_colour:  str = "#D1D5DB"   # light gray  → depleted
    matrix_colour:    str = "#F8FAFC"   # near-white
    boundary_colour:  str = "#374151"   # dark slate
    boundary_label:   str = "Grain\nBoundary"
    domain_label:     str = "Grain Interior"
    profile_colour:   str = "#1D4ED8"   # darker blue for profile line
    disclaimer:       str = field(default_factory=lambda: DISCLAIMER)


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_colour: str) -> np.ndarray:
    h = hex_colour.lstrip("#")
    return np.array([int(h[i: i + 2], 16) / 255.0 for i in (0, 2, 4)],
                    dtype=np.float64)


def _blend_rgba(
    t: np.ndarray,
    colour_a: np.ndarray,
    colour_b: np.ndarray,
    alpha_max: float = 0.90,
    alpha_min: float = 0.15,
) -> np.ndarray:
    """Return (N, 4) RGBA: blend from colour_a (t=0) to colour_b (t=1)."""
    rgb = (1.0 - t[:, None]) * colour_a + t[:, None] * colour_b
    alpha = alpha_max - (alpha_max - alpha_min) * t
    return np.hstack([rgb, alpha[:, None]])


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
    """Build and save a dual-panel mechanism animation as a GIF.

    Top panel: concentration profile C(x, t) vs distance — evolves visibly
    every frame from the real solver output.

    Bottom panel: grain-boundary / grain-interior schematic with particles
    whose colours are updated every frame from the depletion field derived
    from the solver output.  Particles are concentrated in the zoom window
    (depletion region) so colour changes are visually prominent.

    Parameters
    ----------
    output : DiffusionOutput
        Result of ``solve_diffusion()``.
    save_path : str or Path
        Destination.  Extension always normalised to ``.gif``.
    scheme : MechanismScheme | dict | None
        Visual scheme.
    title : str | None
        Optional title.
    fps : int
        Frames per second (default 15).
    max_frames : int
        Maximum frames rendered (default 120).

    Returns
    -------
    str
        Absolute path to the saved GIF.
    """
    # ------------------------------------------------------------------
    # 0. Resolve scheme
    # ------------------------------------------------------------------
    if scheme is None:
        scheme = MechanismScheme()
    elif isinstance(scheme, dict):
        scheme = MechanismScheme(**scheme)

    # ------------------------------------------------------------------
    # 1. Prepare arrays
    # ------------------------------------------------------------------
    profiles = np.array(output.concentration_profiles, dtype=np.float64)
    x_nm_arr = np.array(output.x_nm, dtype=np.float64)   # property: x_m * 1e9
    t_s_arr  = np.array(output.t_s,  dtype=np.float64)
    n_stored = profiles.shape[0]

    n_frames     = max(1, min(max_frames, n_stored))
    frame_indices = np.linspace(0, n_stored - 1, n_frames, dtype=int)

    denom = output.C_bulk_wt_pct - output.C_sink_wt_pct

    def _depletion(profile: np.ndarray) -> np.ndarray:
        if denom <= 0.0:
            return np.zeros_like(profile)
        return np.clip((output.C_bulk_wt_pct - profile) / denom, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2. Determine zoom window
    #    Use the FINAL profile depletion depth to size the x-axis so that
    #    the most depleted region always fills most of the frame.
    # ------------------------------------------------------------------
    final_depletion = _depletion(profiles[-1])
    x_max_nm = float(x_nm_arr[-1])

    # Find how far depletion > 0.01 reaches in final profile
    depleted_mask = final_depletion > 0.01
    if depleted_mask.any():
        depth_nm = float(x_nm_arr[depleted_mask].max())
    else:
        depth_nm = x_max_nm * 0.05  # fallback: 5% of domain

    zoom_nm = max(_ZOOM_MIN_NM, depth_nm * _ZOOM_MULTIPLIER)
    zoom_nm = min(zoom_nm, x_max_nm)

    boundary_nm = max(_BOUNDARY_FRACTION * zoom_nm, 2.0)

    # Grid indices within zoom window
    zoom_mask = x_nm_arr <= zoom_nm
    x_zoom    = x_nm_arr[zoom_mask]
    profiles_zoom = profiles[:, zoom_mask]   # (n_stored, n_zoom)

    # Concentration axis bounds — fixed across all frames
    c_min_plot = max(0.0, output.C_sink_wt_pct - 0.5)
    c_max_plot = output.C_bulk_wt_pct + 0.5

    # ------------------------------------------------------------------
    # 3. Seed particles inside the zoom window
    # ------------------------------------------------------------------
    rng = np.random.default_rng(_RNG_SEED)
    px = rng.uniform(boundary_nm, zoom_nm, _N_PARTICLES)
    py = rng.uniform(_Y_MIN, _Y_MAX, _N_PARTICLES)
    # Map each particle to its nearest x-grid index WITHIN the zoom window
    particle_idx = np.searchsorted(x_zoom, px).clip(0, len(x_zoom) - 1)

    species_rgb  = _hex_to_rgb(scheme.species_colour)
    depleted_rgb = _hex_to_rgb(scheme.depleted_colour)

    # ------------------------------------------------------------------
    # 4. Build dual-panel figure
    # ------------------------------------------------------------------
    matplotlib.use("Agg")

    fig = plt.figure(figsize=(10, 7))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.35)

    ax_profile  = fig.add_subplot(gs[0])
    ax_particle = fig.add_subplot(gs[1])

    fig.patch.set_facecolor(scheme.matrix_colour)

    # ---- Top panel: concentration profile ----
    ax_profile.set_facecolor(scheme.matrix_colour)
    ax_profile.set_xlim(0.0, zoom_nm)
    ax_profile.set_ylim(c_min_plot, c_max_plot)
    ax_profile.set_xlabel("Distance from grain boundary (nm)", fontsize=8, color="#374151")
    ax_profile.set_ylabel(f"[{output.element}] (wt%)", fontsize=8, color="#374151")

    # Reference lines
    ax_profile.axhline(output.C_bulk_wt_pct, color="#9CA3AF", lw=1.0,
                       ls="--", zorder=1, label=f"Bulk ({output.C_bulk_wt_pct:.1f} wt%)")
    ax_profile.axhline(output.C_sink_wt_pct, color="#EF4444", lw=0.8,
                       ls=":", zorder=1, label=f"Sink ({output.C_sink_wt_pct:.1f} wt%)")

    # Boundary stripe (profile panel)
    ax_profile.axvspan(0.0, boundary_nm, color=scheme.boundary_colour, alpha=0.6, zorder=2)

    # Initial profile line + fill
    init_C_zoom = profiles_zoom[frame_indices[0]]
    (profile_line,) = ax_profile.plot(
        x_zoom, init_C_zoom,
        color=scheme.profile_colour, lw=2.0, zorder=4,
    )
    fill_poly = ax_profile.fill_between(
        x_zoom, output.C_sink_wt_pct, init_C_zoom,
        alpha=0.15, color=scheme.profile_colour, zorder=3,
    )

    ax_profile.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax_profile.spines["top"].set_visible(False)
    ax_profile.spines["right"].set_visible(False)
    ax_profile.spines["bottom"].set_color("#D1D5DB")
    ax_profile.spines["left"].set_color("#D1D5DB")
    ax_profile.tick_params(colors="#6B7280", labelsize=7)

    profile_title = (
        title
        or f"{output.element} diffusion — {output.matrix} — Crank–Nicolson solver"
    )
    ax_profile.set_title(profile_title, fontsize=9, pad=6, color="#1F2937")

    # Time annotation (top panel, top-right)
    t_min_0   = t_s_arr[frame_indices[0]] / 60.0
    time_text_top = ax_profile.text(
        zoom_nm * 0.98, c_max_plot * 0.97,
        f"t = {t_min_0:.1f} min",
        ha="right", va="top", fontsize=9, color="#1F2937",
        fontweight="bold", zorder=5,
    )

    # ---- Bottom panel: particle schematic ----
    ax_particle.set_facecolor(scheme.matrix_colour)

    # Grain-boundary stripe
    ax_particle.axvspan(0.0, boundary_nm,
                        color=scheme.boundary_colour, alpha=0.92, zorder=1)
    ax_particle.text(
        boundary_nm * 0.5, 0.0,
        scheme.boundary_label,
        ha="center", va="center", fontsize=7,
        color="white", fontweight="bold", zorder=5,
    )

    # Domain label
    ax_particle.text(
        zoom_nm * 0.72, _Y_MAX * 0.88,
        scheme.domain_label,
        ha="center", va="top", fontsize=8,
        color="#9CA3AF", fontstyle="italic", zorder=5,
    )

    # Note on x-axis zoom
    ax_particle.text(
        zoom_nm * 0.98, _Y_MIN * 0.88,
        f"Zoomed to {zoom_nm:.0f} nm (depletion zone)",
        ha="right", va="bottom", fontsize=6.5,
        color="#9CA3AF", fontstyle="italic", zorder=5,
    )

    # Species label
    ax_particle.text(
        zoom_nm * 0.98, _Y_MAX * 0.88,
        f"[{output.element}]",
        ha="right", va="top", fontsize=10,
        color=scheme.species_colour, fontweight="bold", zorder=5,
    )

    # Initial particle colours
    d0_p   = _depletion(profiles_zoom[frame_indices[0]])[particle_idx]
    init_rgba = _blend_rgba(d0_p, species_rgb, depleted_rgb)
    scatter = ax_particle.scatter(
        px, py, c=init_rgba, s=8, zorder=2, linewidths=0,
    )

    # Axes cosmetics (particle panel)
    ax_particle.set_xlim(0.0, zoom_nm)
    ax_particle.set_ylim(_Y_MIN, _Y_MAX)
    ax_particle.set_xlabel("Distance from grain boundary (nm)", fontsize=8, color="#374151")
    ax_particle.set_yticks([])
    ax_particle.spines["top"].set_visible(False)
    ax_particle.spines["left"].set_visible(False)
    ax_particle.spines["right"].set_visible(False)
    ax_particle.spines["bottom"].set_color("#D1D5DB")
    ax_particle.tick_params(colors="#6B7280", labelsize=7)

    ax_particle.set_title(
        "Mechanism schematic — particle colour = depletion index from C(x,t)",
        fontsize=8, pad=4, color="#6B7280",
    )

    # Non-removable disclaimer (figure footer)
    fig.text(
        0.5, 0.005,
        scheme.disclaimer,
        ha="center", va="bottom", fontsize=6.0,
        color="#9CA3AF", fontstyle="italic",
    )

    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.10, hspace=0.45)

    # ------------------------------------------------------------------
    # 5. Animation update
    # ------------------------------------------------------------------
    def _update(frame_num: int):
        fi = frame_indices[frame_num]
        C_zoom = profiles_zoom[fi]

        # Update profile line
        profile_line.set_ydata(C_zoom)

        # Update fill (recreate poly each frame — simpler than PathCollection update)
        nonlocal fill_poly
        fill_poly.remove()
        fill_poly = ax_profile.fill_between(
            x_zoom, output.C_sink_wt_pct, C_zoom,
            alpha=0.15, color=scheme.profile_colour, zorder=3,
        )

        # Update particle colours
        d_p  = _depletion(C_zoom)[particle_idx]
        rgba = _blend_rgba(d_p, species_rgb, depleted_rgb)
        scatter.set_facecolor(rgba)

        # Update time annotation
        t_min = t_s_arr[fi] / 60.0
        time_text_top.set_text(f"t = {t_min:.1f} min")

        return profile_line, scatter, time_text_top

    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        blit=False,
        interval=max(1, 1000 // fps),
    )

    # ------------------------------------------------------------------
    # 6. Save as GIF
    # ------------------------------------------------------------------
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".gif":
        save_path = save_path.with_suffix(".gif")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=fps)
    anim.save(str(save_path), writer=writer, dpi=_DPI)
    plt.close(fig)

    return str(save_path.resolve())
