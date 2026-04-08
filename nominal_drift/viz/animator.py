"""
nominal_drift.viz.animator
===========================
Time-evolving diffusion animation for 1D concentration profiles.

Produces an MP4 or GIF animation showing how the elemental concentration
field evolves from the grain boundary (x = 0) into the grain interior
(x = x_max) over the full heat-treatment schedule.

Design principles
-----------------
- Element-agnostic: all labels, annotations, and on-screen text are derived
  from the ``DiffusionOutput`` fields (``element``, ``matrix``,
  ``C_bulk_wt_pct``, ``C_sink_wt_pct``).  Nothing is hard-coded for
  chromium.  The same function produces correct output for Cr, C, N, or
  any future interstitial / substitutional species added to arrhenius.json.
- Writer selection with graceful fallback:
    1. If the requested extension is ``.mp4`` and FFMpeg is available on
       the system, use ``FFMpegWriter`` (H.264, compressed, small files).
    2. Otherwise fall back to ``PillowWriter`` (GIF), which has no external
       dependency beyond Pillow.  The saved path is returned with the
       ``.gif`` extension so the caller always gets a valid, openable file.
    3. An explicit ``.gif`` request always goes directly to ``PillowWriter``.
- Non-interactive: uses the Agg matplotlib backend so the function runs
  safely in headless CI, HPC, and Docker environments.
- Consistent styling: mirrors the engineering aesthetic of
  ``profile_plotter.py`` (same colour palette, grid style, font sizes).

Public API
----------
``animate_diffusion(output, save_path, title, threshold_wt_pct, fps, max_frames)``
    Build and save the animation.  Returns the actual saved path as a str.

Internal helpers (not exported)
--------------------------------
``_select_frame_indices(n_stored, max_frames)``
    Return the list of stored-profile indices to render, capped at
    ``max_frames`` with first and last always included.
``_choose_writer(save_path, fps)``
    Select the animation writer and resolve the actual output path.
``_format_time_label(t_s)``
    Convert a time in seconds to a human-readable string.
``_make_info_text(element, t_s, c_min)``
    Compose the per-frame on-screen annotation string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from nominal_drift.schemas.diffusion_output import DiffusionOutput

# ---------------------------------------------------------------------------
# Styling — mirrors profile_plotter.py for visual consistency
# ---------------------------------------------------------------------------

_FIG_SIZE: tuple[float, float] = (10.0, 6.0)
_DPI: int = 100                    # slightly lower than static plot for file size
_FONT_SIZE: int = 11
_PROFILE_COLOUR: str = "#1565c0"   # royal blue — the "final" colour from Day 5
_INITIAL_COLOUR: str = "#b0bec5"   # light blue-grey — reference ghost line
_THRESHOLD_COLOUR: str = "#e53935" # Material Design red 600

# Position of the on-frame annotation box (axes-fraction coordinates)
_INFO_X: float = 0.97
_INFO_Y: float = 0.97


# ---------------------------------------------------------------------------
# Public helper exposed for testing
# ---------------------------------------------------------------------------

def _select_frame_indices(n_stored: int, max_frames: int) -> list[int]:
    """Return the profile indices to render, capped at *max_frames*.

    Rules
    -----
    - If ``n_stored <= max_frames`` every stored profile is used.
    - Otherwise the indices are evenly spaced from 0 to ``n_stored − 1``
      with exactly ``max_frames`` points selected (before de-duplication).
    - The first index (0) and the last index (n_stored − 1) are always
      present in the returned list.
    - Duplicate indices that arise from integer rounding are removed while
      preserving order.

    Parameters
    ----------
    n_stored : int
        Total number of stored concentration profiles, including t = 0.
    max_frames : int
        Maximum number of frames to render.

    Returns
    -------
    list[int]
        Sorted, de-duplicated list of integer indices into
        ``DiffusionOutput.concentration_profiles``.
    """
    if n_stored <= 1:
        return [0]
    if n_stored <= max_frames:
        return list(range(n_stored))

    # Evenly space max_frames points across [0, n_stored − 1]
    raw = [
        int(round(i * (n_stored - 1) / (max_frames - 1)))
        for i in range(max_frames)
    ]

    # De-duplicate while preserving order
    seen: set[int] = set()
    indices: list[int] = []
    for idx in raw:
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)

    # Guarantee first and last are included (rounding may push them out)
    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != n_stored - 1:
        indices.append(n_stored - 1)

    return indices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _choose_writer(
    save_path: Path,
    fps: int,
) -> tuple[object, Path]:
    """Select the animation writer and resolve the actual output path.

    Priority order
    --------------
    1. ``.gif`` extension → always ``PillowWriter`` (no external dependency).
    2. ``.mp4`` extension + FFMpeg available → ``FFMpegWriter`` (H.264).
    3. ``.mp4`` extension + FFMpeg unavailable → fall back to ``PillowWriter``
       and change the extension to ``.gif``.
    4. Any other extension → ``PillowWriter`` with ``.gif`` extension.

    Parameters
    ----------
    save_path : Path
        Intended output path (may be ``.mp4`` or ``.gif``).
    fps : int
        Frames per second passed to the writer constructor.

    Returns
    -------
    tuple[writer, actual_path]
        The instantiated writer object and the path that will be written.
        The caller must use ``actual_path`` (not ``save_path``) for saving
        and for the return value of ``animate_diffusion``.
    """
    ext = save_path.suffix.lower()

    if ext == ".gif":
        return PillowWriter(fps=fps), save_path

    if ext == ".mp4" and FFMpegWriter.isAvailable():
        # H.264 with sensible defaults; -pix_fmt yuv420p ensures broad
        # compatibility (QuickTime, Windows Media Player, most browsers).
        writer = FFMpegWriter(
            fps=fps,
            bitrate=1800,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        return writer, save_path

    # Fallback: convert to GIF regardless of requested extension
    gif_path = save_path.with_suffix(".gif")
    return PillowWriter(fps=fps), gif_path


def _format_time_label(t_s: float) -> str:
    """Format elapsed time as a compact human-readable string.

    Parameters
    ----------
    t_s : float
        Elapsed time in seconds.

    Returns
    -------
    str
        E.g. ``"t = 0 s"``, ``"t = 1.5 min"``, ``"t = 120 min"``.
    """
    if t_s < 60.0:
        return f"t = {t_s:.0f} s"
    minutes = t_s / 60.0
    if minutes < 10.0:
        return f"t = {minutes:.1f} min"
    return f"t = {minutes:.0f} min"


def _make_info_text(element: str, t_s: float, c_min: float) -> str:
    """Compose the per-frame on-screen annotation string.

    The string is intentionally generic — it uses ``element`` directly so
    it works identically for Cr, C, N, or any future species.

    Parameters
    ----------
    element : str
        Chemical symbol of the diffusing species.
    t_s : float
        Elapsed simulation time at this frame [s].
    c_min : float
        Minimum concentration anywhere in the domain at this frame [wt%].

    Returns
    -------
    str
        Multi-line annotation text.
    """
    return (
        f"{_format_time_label(t_s)}\n"
        f"Species: {element}\n"
        f"Min [{element}] = {c_min:.3f} wt%"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_diffusion(
    output: DiffusionOutput,
    save_path: Union[str, Path],
    title: str | None = None,
    threshold_wt_pct: float | None = None,
    fps: int = 15,
    max_frames: int = 300,
) -> str:
    """Build and save a time-evolving concentration-profile animation.

    Each frame shows the elemental concentration as a function of distance
    from the grain boundary at one stored simulation time step.  A faint
    ghost of the initial profile is shown throughout for reference.

    Writer selection
    ----------------
    - Request ``.mp4`` and FFMpeg is present → H.264 MP4 (small, high quality).
    - Request ``.mp4`` but FFMpeg absent → silently falls back to GIF; the
      returned path ends in ``.gif`` so the caller always gets a valid file.
    - Request ``.gif`` → GIF via PillowWriter regardless of FFMpeg status.

    Parameters
    ----------
    output : DiffusionOutput
        Validated simulation result produced by ``solve_diffusion()``.
    save_path : str or Path
        Destination file.  Extension determines the intended format
        (``.mp4`` or ``.gif``).  Parent directory must already exist.
    title : str | None
        Figure title.  If ``None``, an auto-generated title is used:
        ``"<Element> Diffusion — <matrix>"``.
    threshold_wt_pct : float | None
        If provided, a static horizontal dashed reference line is drawn
        at this concentration value across all frames.
    fps : int
        Frames per second in the saved animation.  Affects playback speed;
        does not change the number of frames rendered.
    max_frames : int
        Maximum number of frames to render.  If ``len(output.t_s) >
        max_frames`` the stored profiles are sub-sampled evenly while
        always preserving the first (t = 0) and last (t = final) frames.

    Returns
    -------
    str
        Absolute path to the saved file as a plain ``str``.  May differ
        from ``save_path`` if the GIF fallback was activated (extension
        changed from ``.mp4`` to ``.gif``).

    Notes
    -----
    The figure is closed after saving to prevent memory leaks when the
    function is called repeatedly from an orchestrator loop.
    """
    save_path = Path(save_path)

    # ------------------------------------------------------------------
    # 1. Frame selection
    # ------------------------------------------------------------------
    n_stored = len(output.t_s)
    frame_indices = _select_frame_indices(n_stored, max_frames)
    n_frames = len(frame_indices)

    # ------------------------------------------------------------------
    # 2. Spatial axis in nanometres (static for all frames)
    # ------------------------------------------------------------------
    x_nm = [x * 1e9 for x in output.x_m]

    # ------------------------------------------------------------------
    # 3. Axis bounds (fixed for all frames so the plot does not jump)
    # ------------------------------------------------------------------
    driving_force = output.C_bulk_wt_pct - output.C_sink_wt_pct
    pad = max(0.05 * driving_force, 0.05)
    y_lo = output.C_sink_wt_pct - pad
    y_hi = output.C_bulk_wt_pct + pad
    if threshold_wt_pct is not None:
        y_lo = min(y_lo, threshold_wt_pct - pad)
        y_hi = max(y_hi, threshold_wt_pct + pad)

    # ------------------------------------------------------------------
    # 4. Build figure and static elements
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=_FIG_SIZE, dpi=_DPI)
    fig.subplots_adjust(bottom=0.14)

    # Auto-title
    plot_title = title if title is not None else (
        f"{output.element} Diffusion — {output.matrix}"
    )
    ax.set_title(plot_title, fontsize=_FONT_SIZE + 2, fontweight="bold", pad=10)

    # Axis labels and limits
    ax.set_xlabel("Distance from grain boundary (nm)", fontsize=_FONT_SIZE)
    ax.set_ylabel(
        f"{output.element} concentration (wt%)", fontsize=_FONT_SIZE
    )
    ax.set_xlim(0.0, max(x_nm))
    ax.set_ylim(y_lo, y_hi)

    # Grid
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6, color="#cccccc")
    ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.4, color="#e0e0e0")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=_FONT_SIZE - 1)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")

    # Ghost initial profile (static reference; drawn once)
    initial_profile = output.concentration_profiles[0]
    ax.plot(
        x_nm, initial_profile,
        color=_INITIAL_COLOUR, linestyle="--", linewidth=1.0,
        label=f"Initial ({_format_time_label(output.t_s[0])})",
        zorder=2,
    )

    # Static threshold reference line
    if threshold_wt_pct is not None:
        ax.axhline(
            y=threshold_wt_pct,
            color=_THRESHOLD_COLOUR, linestyle="--", linewidth=1.4,
            alpha=0.85, label=f"Threshold: {threshold_wt_pct:.2f} wt%",
            zorder=3,
        )

    # Footer metadata
    footer = (
        f"Element: {output.element}  |  "
        f"Matrix: {output.matrix}  |  "
        f"C\u2080 (bulk) = {output.C_bulk_wt_pct:.2f} wt%  |  "
        f"C\u209B (sink) = {output.C_sink_wt_pct:.2f} wt%"
    )
    fig.text(
        0.5, 0.02, footer,
        ha="center", va="bottom",
        fontsize=_FONT_SIZE - 2, color="#555555", style="italic",
    )

    # ------------------------------------------------------------------
    # 5. Dynamic elements (updated each frame)
    # ------------------------------------------------------------------
    (profile_line,) = ax.plot(
        [], [],
        color=_PROFILE_COLOUR, linewidth=2.2, zorder=5,
        label="Current profile",
    )

    # On-frame annotation box (upper-right, inside axes)
    info_box = ax.text(
        _INFO_X, _INFO_Y, "",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=_FONT_SIZE - 1,
        color="#1a1a2e",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#aaaaaa",
            alpha=0.88,
        ),
        zorder=10,
    )

    ax.legend(
        loc="lower right",
        fontsize=_FONT_SIZE - 2,
        framealpha=0.90,
        edgecolor="#aaaaaa",
    )

    # ------------------------------------------------------------------
    # 6. Animation callbacks
    # ------------------------------------------------------------------

    def _init():
        """Initialise dynamic artists to empty / blank state."""
        profile_line.set_data([], [])
        info_box.set_text("")
        return profile_line, info_box

    def _update(frame_num: int):
        """Update dynamic artists for frame *frame_num*."""
        snap_idx = frame_indices[frame_num]
        profile  = output.concentration_profiles[snap_idx]
        t_s_val  = output.t_s[snap_idx]
        c_min    = min(profile)

        profile_line.set_data(x_nm, profile)
        info_box.set_text(_make_info_text(output.element, t_s_val, c_min))
        return profile_line, info_box

    anim = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=n_frames,
        interval=max(1, int(1000 / fps)),
        blit=False,   # blit=False is safer for saving and avoids artist
                      # restoration issues with text boxes on some backends
        repeat=False,
    )

    # ------------------------------------------------------------------
    # 7. Select writer and save
    # ------------------------------------------------------------------
    writer, actual_path = _choose_writer(save_path, fps)
    anim.save(str(actual_path), writer=writer)
    plt.close(fig)

    return str(actual_path)
