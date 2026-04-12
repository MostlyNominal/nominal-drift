"""
nominal_drift/demo/run_diffusion_demo.py
=========================================
End-to-end local demo: 316L Cr diffusion → static plot + animation.

This is a TEMPORARY verification script, NOT the production orchestrator.
It wires together the Sprint 1 physics and visualisation modules in the
simplest possible way so the pipeline can be confirmed working before Day 7
(SQLite store) and Day 8+ (orchestrator / CLI integration) are built.

Run with:
    python -m nominal_drift.demo.run_diffusion_demo

Outputs are written to:
    outputs/demo/cr_profile_316L_700C_60min.png
    outputs/demo/cr_animation_316L_700C_60min.gif   (or .mp4 if ffmpeg present)
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the project root so the script works whether invoked with
#   python -m nominal_drift.demo.run_diffusion_demo
# or directly from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _REPO_ROOT / "outputs" / "demo"


def _hr(char: str = "─", width: int = 60) -> str:
    return char * width


def _banner(text: str) -> None:
    print(_hr())
    print(f"  {text}")
    print(_hr())


def main() -> None:
    # ------------------------------------------------------------------
    # 0. Imports (deferred so import errors surface with a clear message)
    # ------------------------------------------------------------------
    try:
        from nominal_drift.cli.main import _DEMO_COMPOSITION, _DEMO_SCHEDULE
        from nominal_drift.science.diffusion_engine import solve_diffusion
        from nominal_drift.viz.profile_plotter import plot_concentration_profile
        from nominal_drift.viz.animator import animate_diffusion
    except ImportError as exc:
        print(f"\n[ERROR] Import failed — is the package installed?\n  {exc}")
        print("  Hint: run  pip install -e '.[dev]'  from the repo root.\n")
        sys.exit(1)

    _banner("Nominal Drift — Sprint 1 Demo")
    print()

    # ------------------------------------------------------------------
    # 1. Alloy composition — 316L austenitic stainless steel
    #    (single source of truth: cli.main._DEMO_COMPOSITION)
    # ------------------------------------------------------------------
    print("► Building alloy composition …")
    composition = _DEMO_COMPOSITION
    print(f"   Alloy : {composition.alloy_designation}")
    print(f"   Matrix: {composition.alloy_matrix}")
    print(f"   Cr    : {composition.bulk_Cr_wt_pct:.2f} wt%")
    print(f"   Sum   : {composition.composition_sum:.2f} wt%")
    print()

    # ------------------------------------------------------------------
    # 2. Heat-treatment schedule — single sensitisation soak at 700 °C
    #    (single source of truth: cli.main._DEMO_SCHEDULE)
    # ------------------------------------------------------------------
    print("► Building HT schedule …")
    schedule = _DEMO_SCHEDULE
    step = schedule.steps[0]
    print(f"   Step 1 : {step.T_hold_C:.0f} °C  for  {step.hold_min:.0f} min"
          f"  ({step.hold_s:.0f} s)  — {step.cooling_method}")
    print()

    # ------------------------------------------------------------------
    # 3. Solve diffusion
    # ------------------------------------------------------------------
    print("► Running Crank–Nicolson diffusion solver …")
    t0 = time.perf_counter()

    result = solve_diffusion(
        composition   = composition,
        ht_schedule   = schedule,
        element       = "Cr",
        matrix        = "austenite_FeCrNi",
        n_spatial     = 300,
        x_max_m       = 5e-6,
        C_sink_wt_pct = 12.0,
    )

    elapsed = time.perf_counter() - t0
    print(f"   Solver finished in {elapsed:.3f} s")
    print(f"   Stored profiles : {result.n_timesteps_stored}")
    print(f"   Spatial nodes   : {result.n_spatial}")
    print(f"   Total time      : {result.total_time_min:.1f} min")
    print(f"   Min [Cr] at GB  : {result.min_concentration_wt_pct:.4f} wt%")
    if result.depletion_depth_nm is not None:
        print(f"   Depletion depth : {result.depletion_depth_nm:.1f} nm")
    if result.warnings:
        for w in result.warnings:
            print(f"   ⚠  {w}")
    print()

    # ------------------------------------------------------------------
    # 4. Create output directory
    # ------------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"cr_316L_{step.T_hold_C:.0f}C_{step.hold_min:.0f}min"

    # ------------------------------------------------------------------
    # 5. Static concentration-profile plot
    # ------------------------------------------------------------------
    print("► Saving static profile plot …")
    png_path = plot_concentration_profile(
        output            = result,
        save_path         = _OUTPUT_DIR / f"{stem}.png",
        title             = (
            f"316L — Cr depletion at {step.T_hold_C:.0f} °C "
            f"for {step.hold_min:.0f} min"
        ),
        threshold_wt_pct  = 12.0,   # sensitisation criterion
    )
    print(f"   PNG  → {png_path}")
    print()

    # ------------------------------------------------------------------
    # 6. Animation
    # ------------------------------------------------------------------
    print("► Saving diffusion animation …")
    t1 = time.perf_counter()

    anim_path = animate_diffusion(
        output           = result,
        save_path        = _OUTPUT_DIR / f"{stem}.mp4",
        title            = (
            f"316L — Cr diffusion at {step.T_hold_C:.0f} °C "
            f"for {step.hold_min:.0f} min"
        ),
        threshold_wt_pct = 12.0,
        fps              = 15,
        max_frames       = 120,
    )
    elapsed_anim = time.perf_counter() - t1
    print(f"   Animation → {anim_path}  ({elapsed_anim:.1f} s)")
    print()

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    _banner("Demo complete")
    summary = textwrap.dedent(f"""
      Alloy         : {composition.alloy_designation}  ({composition.alloy_matrix})
      Schedule      : Step 1 — {step.T_hold_C:.0f} °C × {step.hold_min:.0f} min
      Element       : {result.element}  in  {result.matrix}
      C_bulk        : {result.C_bulk_wt_pct:.2f} wt%
      C_sink        : {result.C_sink_wt_pct:.2f} wt%
      Min [Cr] @ GB : {result.min_concentration_wt_pct:.4f} wt%

      PNG saved     : {png_path}
      Anim saved    : {anim_path}
    """).strip()
    print(summary)
    print(_hr())
    print()


if __name__ == "__main__":
    main()
