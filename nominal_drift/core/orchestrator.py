"""
nominal_drift.core.orchestrator
=================================
Thin orchestration layer for NominalDrift Sprint 1.

Connects the already-built components into one callable showcase workflow:
1. Diffusion solve   (``science.diffusion_engine``)
2. Static plot       (``viz.profile_plotter``)
3. Animation         (``viz.animator``)
4. Experiment store  (``knowledge.experiment_store``)
5. LLM narration     (``llm.narration``)

Design principles
-----------------
- Thin: no business logic is duplicated from the underlying modules.
- Resilient: narration failure (Ollama unreachable) does not abort the run;
  a placeholder message is stored instead.
- Extensible: the return dict uses generic key names so multi-species and
  multi-step orchestration can add entries without breaking callers.
- Isolated: all I/O paths are derived from ``base_output_dir`` so multiple
  runs never collide.

Public API
----------
``run_showcase_workflow(composition, ht_schedule, *, ...)``
    Run the full Sprint 1 workflow and return a plain result dict.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional

from nominal_drift.core.session import NominalDriftSession
from nominal_drift.knowledge.experiment_store import write_experiment
from nominal_drift.llm.client import OllamaConnectionError
from nominal_drift.llm.narration import narrate_diffusion_result
from nominal_drift.science.diffusion_engine import solve_diffusion
from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.ht_schedule import HTSchedule
from nominal_drift.viz.animator import animate_diffusion
from nominal_drift.viz.profile_plotter import plot_concentration_profile

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "outputs")

# ---------------------------------------------------------------------------
# Narration fallback message
# ---------------------------------------------------------------------------

_NARRATION_UNAVAILABLE = (
    "[narration unavailable: Ollama server not reachable — "
    "start it with 'ollama serve' and re-run to generate narration]"
)


# ---------------------------------------------------------------------------
# Public workflow
# ---------------------------------------------------------------------------

def run_showcase_workflow(
    composition: AlloyComposition,
    ht_schedule: HTSchedule,
    *,
    element: str = "Cr",
    matrix: str = "austenite_FeCrNi",
    c_sink_wt_pct: float = 12.0,
    user_label: Optional[str] = None,
    user_notes: Optional[str] = None,
    base_output_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Run the full Sprint 1 showcase workflow and return a result dict.

    Steps
    -----
    1. Solve the 1-D Crank–Nicolson diffusion problem.
    2. Create a run-specific output directory.
    3. Generate and save a static concentration-profile PNG.
    4. Generate and save a diffusion-evolution animation (MP4 or GIF).
    5. Persist the experiment record to the SQLite store.
    6. Generate a short engineering narration via the local Ollama client
       (falls back gracefully if the server is unreachable).

    Parameters
    ----------
    composition : AlloyComposition
        Validated alloy composition.
    ht_schedule : HTSchedule
        Validated heat-treatment schedule.
    element : str
        Diffusing species symbol (e.g. ``"Cr"``, ``"N"``, ``"C"``).
    matrix : str
        Arrhenius diffusivity matrix key (e.g. ``"austenite_FeCrNi"``).
    c_sink_wt_pct : float
        Dirichlet sink concentration at the domain boundary [wt%].
    user_label : str | None
        Short human-readable label for this run (stored and shown in narration).
    user_notes : str | None
        Free-form notes stored with the experiment record.
    base_output_dir : str | None
        Root directory under which the run subdirectory is created.
        Defaults to ``<repo_root>/outputs``.
    db_path : str | None
        Path to the SQLite experiment database.  ``None`` uses the
        experiment-store default (``data/experiments.db``).

    Returns
    -------
    dict
        Plain result dict with the following keys:

        * ``experiment_id``          — UUID string
        * ``output_dir``             — run-specific output directory path
        * ``plot_path``              — absolute path to the saved PNG
        * ``animation_path``         — absolute path to the saved MP4/GIF
        * ``narration``              — LLM narration text (or fallback message)
        * ``min_concentration_wt_pct`` — minimum element concentration [wt%]
        * ``depletion_depth_nm``     — depletion depth [nm] or ``None``
        * ``warnings``               — list of solver warning strings
        * ``element``                — diffusing species used
        * ``alloy_designation``      — alloy label from composition
    """

    # ------------------------------------------------------------------
    # 1. Solve diffusion
    # ------------------------------------------------------------------
    diffusion_output = solve_diffusion(
        composition,
        ht_schedule,
        element=element,
        matrix=matrix,
        C_sink_wt_pct=c_sink_wt_pct,
    )

    # ------------------------------------------------------------------
    # 2. Create run-specific output directory
    # ------------------------------------------------------------------
    exp_id = str(uuid.uuid4())
    safe_alloy = re.sub(r"[^a-zA-Z0-9_-]", "_", composition.alloy_designation.lower())[:12]
    run_name = f"{element.lower()}_{safe_alloy}_{exp_id[:8]}"
    effective_output_base = base_output_dir or _DEFAULT_OUTPUT_DIR
    run_dir = Path(effective_output_base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Static plot
    # ------------------------------------------------------------------
    plot_path = str(run_dir / f"{element.lower()}_profile.png")
    actual_plot = plot_concentration_profile(
        diffusion_output,
        plot_path,
        threshold_wt_pct=c_sink_wt_pct,
    )

    # ------------------------------------------------------------------
    # 4. Animation
    # ------------------------------------------------------------------
    anim_path = str(run_dir / f"{element.lower()}_diffusion.mp4")
    actual_anim = animate_diffusion(
        diffusion_output,
        anim_path,
        threshold_wt_pct=c_sink_wt_pct,
    )

    # ------------------------------------------------------------------
    # 5. Persist experiment record
    # ------------------------------------------------------------------
    record: dict = {
        "experiment_id":            exp_id,
        "alloy_designation":        composition.alloy_designation,
        "alloy_matrix":             composition.alloy_matrix,
        "composition_json":         dict(composition.composition_wt_pct),
        "ht_schedule_json":         ht_schedule.model_dump(),
        "element":                  element,
        "matrix":                   matrix,
        "c_bulk_wt_pct":            diffusion_output.C_bulk_wt_pct,
        "c_sink_wt_pct":            diffusion_output.C_sink_wt_pct,
        "min_concentration_wt_pct": diffusion_output.min_concentration_wt_pct,
        "depletion_depth_nm":       diffusion_output.depletion_depth_nm,
        "warnings_json":            list(diffusion_output.warnings),
        "plot_path":                actual_plot,
        "animation_path":           actual_anim,
        "user_label":               user_label,
        "user_notes":               user_notes,
    }
    if db_path is not None:
        stored_id = write_experiment(record, db_path=db_path)
    else:
        stored_id = write_experiment(record)

    # ------------------------------------------------------------------
    # 6. LLM narration (graceful fallback on connection failure)
    # ------------------------------------------------------------------
    db_context: Optional[dict] = (
        {"user_label": user_label, "user_notes": user_notes}
        if (user_label is not None or user_notes is not None)
        else None
    )
    try:
        narration_text = narrate_diffusion_result(
            diffusion_output,
            composition,
            ht_schedule,
            db_record=db_context,
        )
    except OllamaConnectionError:
        narration_text = _NARRATION_UNAVAILABLE

    # ------------------------------------------------------------------
    # 7. Build session (internal use — not returned)
    # ------------------------------------------------------------------
    _session = NominalDriftSession(
        composition=composition,
        ht_schedule=ht_schedule,
        experiment_id=stored_id,
        output_dir=str(run_dir),
        db_path=db_path,
    )

    # ------------------------------------------------------------------
    # 8. Return result dict
    # ------------------------------------------------------------------
    return {
        "experiment_id":            stored_id,
        "output_dir":               str(run_dir),
        "plot_path":                actual_plot,
        "animation_path":           actual_anim,
        "narration":                narration_text,
        "min_concentration_wt_pct": diffusion_output.min_concentration_wt_pct,
        "depletion_depth_nm":       diffusion_output.depletion_depth_nm,
        "warnings":                 list(diffusion_output.warnings),
        "element":                  element,
        "alloy_designation":        composition.alloy_designation,
    }
