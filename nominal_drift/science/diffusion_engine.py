"""
nominal_drift.science.diffusion_engine
=======================================
1-D Crank–Nicolson diffusion solver for sensitization modelling.

Physics
-------
Governing PDE: Fick's 2nd law, 1D
    ∂C/∂t = D(T) · ∂²C/∂x²

Domain: x ∈ [0, x_max], half-space from grain boundary into grain interior.

Boundary conditions:
    Left  (x = 0):     Dirichlet sink  — C(0, t) = C_sink
                        (fast-precipitation approximation for M₂₃C₆)
    Right (x = x_max): Dirichlet bulk  — C(L, t) = C_bulk

Initial condition:
    C(x, 0) = C_bulk  (uniform, fully annealed)

Solver: Crank–Nicolson (unconditionally stable, 2nd-order in space and time).
The tridiagonal system is solved per time step via ``scipy.linalg.solve_banded``.

Analytical validation reference (constant D):
    C(x, t) = C_sink + (C_bulk − C_sink) · erf(x / (2√(D·t)))

This is valid for a semi-infinite domain; the numerical domain is finite so
the comparison is only meaningful when the depletion front is well inside the
domain (i.e. √(D·t) << x_max).

Public API
----------
- ``load_arrhenius_constants()``    → dict
- ``arrhenius_D(T_C, element, matrix, constants)``  → float
- ``solve_diffusion(composition, ht_schedule, ...)`` → DiffusionOutput

Internal helpers (not exported):
- ``_apply_dirichlet_bcs(C, C_sink, C_bulk)``
- ``_crank_nicolson_step(C, D, dt, dx, C_sink, C_bulk)``
- ``_compute_depletion_depth_nm(final_profile, x_m, C_sink, C_bulk)``
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_banded

from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.schemas.ht_schedule import HTSchedule

# ---------------------------------------------------------------------------
# Physical / numerical constants
# ---------------------------------------------------------------------------

#: Universal gas constant [J / (mol · K)]
_R: float = 8.314

#: Path to the Arrhenius constants JSON database
_ARRHENIUS_PATH: Path = Path(__file__).parent / "constants" / "arrhenius.json"

#: Maximum number of concentration-profile snapshots stored in a result
#: (including the t = 0 initial condition).  Keeps memory and serialisation
#: size bounded for long simulations.
_MAX_FRAMES: int = 300

#: Minimum number of Crank–Nicolson time steps per HT schedule step.
#: Using fewer steps reduces temporal accuracy even though CN is
#: unconditionally stable.
_MIN_STEPS: int = 200

#: Maximum number of Crank–Nicolson time steps per HT schedule step.
#: Caps wall-clock time for very fine grids at high temperature.
_MAX_STEPS: int = 5000


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------

def load_arrhenius_constants() -> dict:
    """Load the Arrhenius diffusion constant database from the JSON file.

    Returns
    -------
    dict
        Full parsed contents of ``science/constants/arrhenius.json``.
        Keys starting with ``_`` are metadata fields (schema version,
        description, units, references); element keys (e.g. ``"Cr"``,
        ``"C"``, ``"N"``) hold dicts with at minimum ``"D0"`` and ``"Qd"``.
    """
    with open(_ARRHENIUS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def arrhenius_D(
    T_C: float,
    element: str,
    matrix: str,
    constants: dict,
) -> float:
    """Compute diffusivity D(T) = D₀ · exp(−Qd / (R · T)).

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    element : str
        Element symbol (e.g. ``"Cr"``, ``"C"``, ``"N"``).  Must be a key
        in *constants* (case-sensitive).
    matrix : str
        Matrix identifier (e.g. ``"austenite_FeCrNi"``).  Checked for
        consistency against the ``"matrix"`` field stored in the element
        record; a mismatch raises ``ValueError``.
    constants : dict
        Arrhenius constants dict as returned by
        :func:`load_arrhenius_constants`.

    Returns
    -------
    float
        Diffusivity in m²/s at temperature *T_C*.

    Raises
    ------
    KeyError
        If *element* is not present in *constants*.
    ValueError
        If the stored matrix label does not match *matrix*.
    """
    if element not in constants:
        available = [k for k in constants if not k.startswith("_")]
        raise KeyError(
            f"Element '{element}' not found in the Arrhenius constants "
            f"database.  Available elements: {available}"
        )

    record = constants[element]
    stored_matrix = record.get("matrix", "")
    if stored_matrix and stored_matrix != matrix:
        raise ValueError(
            f"Matrix mismatch for element '{element}': "
            f"requested '{matrix}', stored in database '{stored_matrix}'.  "
            f"Check that the correct matrix identifier is being used."
        )

    D0: float = record["D0"]   # pre-exponential factor [m²/s]
    Qd: float = record["Qd"]   # activation energy [J/mol]
    T_K: float = T_C + 273.15  # convert to Kelvin
    return D0 * math.exp(-Qd / (_R * T_K))


# ---------------------------------------------------------------------------
# Internal numerical helpers
# ---------------------------------------------------------------------------

def _apply_dirichlet_bcs(
    C: np.ndarray,
    C_sink: float,
    C_bulk: float,
) -> None:
    """Overwrite the two boundary nodes with their Dirichlet values (in-place).

    Parameters
    ----------
    C : np.ndarray
        Concentration array, shape ``(N,)``.  Modified in-place.
    C_sink : float
        Value to assign to ``C[0]`` (grain-boundary sink) [wt%].
    C_bulk : float
        Value to assign to ``C[-1]`` (far-field bulk) [wt%].
    """
    C[0] = C_sink
    C[-1] = C_bulk


def _crank_nicolson_step(
    C: np.ndarray,
    D: float,
    dt: float,
    dx: float,
    C_sink: float,
    C_bulk: float,
) -> np.ndarray:
    """Advance the concentration field by one Crank–Nicolson time step.

    Discretisation
    --------------
    With ``r = D · dt / (2 · dx²)``, the interior equation for node *i* is::

        −r · C[i−1]ⁿ⁺¹ + (1+2r) · C[i]ⁿ⁺¹ − r · C[i+1]ⁿ⁺¹
            = r · C[i−1]ⁿ + (1−2r) · C[i]ⁿ + r · C[i+1]ⁿ

    Boundary rows (0 and N−1) are replaced by identity equations that
    enforce the Dirichlet values directly.

    The resulting tridiagonal system is solved with
    ``scipy.linalg.solve_banded`` using the (l=1, u=1) band storage
    format::

        ab[0, j] = a[j−1, j]   (super-diagonal)
        ab[1, j] = a[j,   j]   (main diagonal)
        ab[2, j] = a[j+1, j]   (sub-diagonal)

    Parameters
    ----------
    C : np.ndarray
        Current concentration profile [wt%], shape ``(N,)``.
    D : float
        Diffusivity [m²/s] at the current hold temperature.
    dt : float
        Time step [s].
    dx : float
        Uniform spatial step [m].
    C_sink : float
        Left Dirichlet boundary value [wt%].
    C_bulk : float
        Right Dirichlet boundary value [wt%].

    Returns
    -------
    np.ndarray
        Updated concentration profile [wt%], shape ``(N,)``.
    """
    N = len(C)
    r = D * dt / (2.0 * dx ** 2)

    # Build banded matrix (3 × N)
    ab = np.zeros((3, N))

    # Main diagonal
    ab[1, 0] = 1.0             # BC row 0: identity
    ab[1, 1:-1] = 1.0 + 2 * r  # CN interior rows
    ab[1, -1] = 1.0            # BC row N−1: identity

    # Super-diagonal: ab[0, j] = a[j−1, j]
    # CN row i has a[i, i+1] = −r  →  ab[0, i+1] = −r  for i = 1..N−2
    # i.e. ab[0, 2:N] = −r
    # ab[0, 1] = a[0, 1] = 0 because row 0 is a BC identity (already 0).
    ab[0, 2:] = -r

    # Sub-diagonal: ab[2, j] = a[j+1, j]
    # CN row i has a[i, i−1] = −r  →  ab[2, i−1] = −r  for i = 1..N−2
    # i.e. ab[2, 0:N−2] = −r
    # ab[2, N−2] = a[N−1, N−2] = 0 because row N−1 is a BC identity.
    ab[2, : N - 2] = -r

    # Right-hand side
    rhs = np.empty(N)
    rhs[0] = C_sink
    rhs[1:-1] = r * C[:-2] + (1.0 - 2 * r) * C[1:-1] + r * C[2:]
    rhs[-1] = C_bulk

    return solve_banded((1, 1), ab, rhs)


def _compute_depletion_depth_nm(
    final_profile: np.ndarray,
    x_m: np.ndarray,
    C_sink: float,
    C_bulk: float,
) -> float | None:
    """Return the depletion depth at the final time step in nanometres.

    The depletion depth is defined as the distance from the grain boundary
    (x = 0) at which the concentration first exceeds ``C_sink + 0.5 wt%``.
    This threshold is chosen to mark the edge of the depletion trough.

    Parameters
    ----------
    final_profile : np.ndarray
        Concentration profile at the last stored time step [wt%].
    x_m : np.ndarray
        Spatial grid positions [m].
    C_sink : float
        Grain-boundary Dirichlet value [wt%].
    C_bulk : float
        Far-field concentration [wt%].

    Returns
    -------
    float | None
        Depletion depth in nanometres, or ``None`` if there is no driving
        force (``C_sink == C_bulk``) or the depletion front has not entered
        the domain.
    """
    if abs(C_bulk - C_sink) < 1e-9:
        return None  # no driving force → no meaningful depletion depth

    threshold = C_sink + 0.5
    for i, c in enumerate(final_profile):
        if c > threshold:
            return float(x_m[i] * 1e9)
    return None  # depletion front has not yet penetrated domain


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_diffusion(
    composition: AlloyComposition,
    ht_schedule: HTSchedule,
    element: str = "Cr",
    matrix: str = "austenite_FeCrNi",
    n_spatial: int = 200,
    x_max_m: float = 5e-6,
    C_sink_wt_pct: float = 12.0,
) -> DiffusionOutput:
    """Solve 1D Fickian diffusion for a single element over a multi-step HT schedule.

    The solver processes each :class:`~nominal_drift.schemas.ht_schedule.HTStep`
    in ascending step-number order, advancing the concentration field
    isothermally through the hold at ``T_hold_C``.  The concentration array
    is carried forward between steps so that a multi-step schedule (e.g.
    solution anneal → sensitization soak) is handled automatically.

    Parameters
    ----------
    composition : AlloyComposition
        Validated alloy composition.  The bulk concentration of *element* is
        read from ``composition.composition_wt_pct[element]`` and used as
        the right-boundary Dirichlet value and the initial condition.
    ht_schedule : HTSchedule
        One or more isothermal hold steps.  Steps are sorted by step number
        before processing.
    element : str
        Chemical symbol of the diffusing species (e.g. ``"Cr"``, ``"N"``).
        Must be a key in the Arrhenius constants database.
    matrix : str
        Matrix identifier for Arrhenius look-up (e.g. ``"austenite_FeCrNi"``).
    n_spatial : int
        Number of spatial grid nodes, including both boundary nodes.
        Higher values give better spatial resolution at the cost of
        computation time.
    x_max_m : float
        Half-width of the simulation domain in metres.  Should be large
        enough that the depletion front does not reach the right boundary
        during the simulation (a warning is issued if it does).
    C_sink_wt_pct : float
        Dirichlet sink concentration at x = 0 [wt%].  Represents the
        equilibrium Cr content in the matrix at the grain boundary under
        the fast-precipitation approximation.  Default: 12.0 wt% for Cr
        in 304/316L.

    Returns
    -------
    DiffusionOutput
        Fully validated simulation result containing the concentration
        profiles, derived scalar outputs, warnings, and provenance metadata.

    Raises
    ------
    KeyError
        If *element* is not present in the Arrhenius constants database.
    """
    # ------------------------------------------------------------------
    # 1. Load constants and validate element availability
    # ------------------------------------------------------------------
    constants = load_arrhenius_constants()
    # Element availability is validated inside arrhenius_D() — no duplicate check here.

    # ------------------------------------------------------------------
    # 2. Read bulk concentration from the alloy composition
    # ------------------------------------------------------------------
    if element not in composition.composition_wt_pct:
        available = sorted(composition.composition_wt_pct.keys())
        raise KeyError(
            f"Element '{element}' not found in composition for "
            f"'{composition.alloy_designation}'.  "
            f"Available elements: {available}"
        )
    C_bulk: float = composition.composition_wt_pct[element]

    # ------------------------------------------------------------------
    # 2b. Early validation: C_sink must not exceed C_bulk
    # ------------------------------------------------------------------
    if C_sink_wt_pct > C_bulk:
        raise ValueError(
            f"C_sink_wt_pct ({C_sink_wt_pct}) must be <= C_bulk_wt_pct "
            f"({C_bulk}) for element '{element}'.  The sink model represents "
            f"depletion (C_sink < C_bulk), not enrichment."
        )

    # ------------------------------------------------------------------
    # 3. Build uniform spatial grid
    # ------------------------------------------------------------------
    x_m: np.ndarray = np.linspace(0.0, x_max_m, n_spatial)
    dx: float = float(x_m[1] - x_m[0])

    # ------------------------------------------------------------------
    # 4. Set initial condition (uniform at C_bulk; BCs applied immediately)
    # ------------------------------------------------------------------
    C: np.ndarray = np.full(n_spatial, C_bulk, dtype=float)
    _apply_dirichlet_bcs(C, C_sink_wt_pct, C_bulk)

    # ------------------------------------------------------------------
    # 5. Accumulate snapshots — always start with the t = 0 profile
    # ------------------------------------------------------------------
    stored_profiles: list[list[float]] = [C.tolist()]
    stored_t_s: list[float] = [0.0]

    # ------------------------------------------------------------------
    # 6. Pre-compute per-HT-step solver parameters
    # ------------------------------------------------------------------
    sorted_steps = sorted(ht_schedule.steps, key=lambda s: s.step)

    step_params: list[tuple] = []
    for ht_step in sorted_steps:
        T_C = ht_step.T_hold_C
        total_step_s = ht_step.hold_s
        D = arrhenius_D(T_C, element, matrix, constants)

        # Number of CN steps: bounded in [_MIN_STEPS, _MAX_STEPS].
        # Lower bound ensures temporal accuracy; upper bound caps wall time.
        # The stability limit for an explicit scheme (dx²/(2D)) is used as
        # a guide; CN is unconditionally stable so we can exceed it.
        if D > 0:
            explicit_limit_steps = math.ceil(total_step_s / (dx ** 2 / (2.0 * D)))
        else:
            explicit_limit_steps = _MIN_STEPS

        n_steps = max(_MIN_STEPS, min(_MAX_STEPS, explicit_limit_steps))
        dt = total_step_s / n_steps  # exact so cumulative time has no drift

        step_params.append((ht_step, D, n_steps, dt, total_step_s))

    # ------------------------------------------------------------------
    # 7. Snapshot frequency: spread ≤ _MAX_FRAMES across all HT steps
    # ------------------------------------------------------------------
    total_n_cn_steps = sum(p[2] for p in step_params)
    # We already have 1 frame (t=0); budget (_MAX_FRAMES − 1) more.
    budget = _MAX_FRAMES - 1
    snap_every = max(1, total_n_cn_steps // budget)

    # ------------------------------------------------------------------
    # 8. Time-marching loop
    # ------------------------------------------------------------------
    arrhenius_meta: dict[str, Any] = {}
    ht_step_summaries: list[dict] = []
    global_cn_step: int = 0
    t_step_start: float = 0.0

    for ht_step, D, n_steps, dt, total_step_s in step_params:
        arrhenius_meta[element] = {
            "D0": constants[element]["D0"],
            "Qd": constants[element]["Qd"],
            "D_at_T": D,
            "T_C": ht_step.T_hold_C,
        }
        ht_step_summaries.append({
            "step": ht_step.step,
            "T_hold_C": ht_step.T_hold_C,
            "hold_min": ht_step.hold_min,
        })

        for step_i in range(n_steps):
            C = _crank_nicolson_step(C, D, dt, dx, C_sink_wt_pct, C_bulk)
            # Reinforce Dirichlet BCs after each solve (prevents floating-point
            # drift at boundary nodes in long simulations).
            _apply_dirichlet_bcs(C, C_sink_wt_pct, C_bulk)

            global_cn_step += 1
            is_last_cn_step = (global_cn_step == total_n_cn_steps)
            t_now = t_step_start + (step_i + 1) * dt

            if global_cn_step % snap_every == 0 or is_last_cn_step:
                stored_profiles.append(C.tolist())
                stored_t_s.append(t_now)

        t_step_start += total_step_s

    # Snap the very last frame if not already stored (guards against
    # snap_every not dividing total_n_cn_steps evenly).
    total_hold_s: float = ht_schedule.total_hold_s
    if not stored_t_s or abs(stored_t_s[-1] - total_hold_s) > 1.0:
        stored_profiles.append(C.tolist())
        stored_t_s.append(total_hold_s)

    # Correct any floating-point drift in the final timestamp.
    stored_t_s[-1] = total_hold_s

    # ------------------------------------------------------------------
    # 9. Post-processing: scalar derived outputs
    # ------------------------------------------------------------------
    final_profile = np.array(stored_profiles[-1])
    min_conc = float(min(min(row) for row in stored_profiles))

    depletion_depth_nm = _compute_depletion_depth_nm(
        final_profile, x_m, C_sink_wt_pct, C_bulk
    )

    # ------------------------------------------------------------------
    # 10. Warnings
    # ------------------------------------------------------------------
    sim_warnings: list[str] = []

    # Warn when C_sink default (12.0 wt%) is used for a non-Cr element —
    # this default is only physically meaningful for Cr in austenitic SS.
    if element != "Cr" and C_sink_wt_pct == 12.0:
        sim_warnings.append(
            f"C_sink_wt_pct=12.0 is the default for Cr in austenitic "
            f"stainless steel.  For element '{element}', this value is "
            f"likely not physically meaningful.  Specify an appropriate "
            f"C_sink_wt_pct for this species."
        )

    # Warn when the depletion zone has penetrated to at least 80% of the
    # domain.  This is detected by checking whether the concentration at
    # the node closest to 80 % of x_max has been pulled more than 5 % of
    # the driving force (C_bulk − C_sink) below C_bulk.
    #
    # This criterion handles both the "moving front" case (where the front
    # reaches 80 % of x_max) and the "saturated domain" case (where the
    # entire domain has been depleted and the profile is nearly linear),
    # because in both cases the concentration at x = 0.8 · x_max is
    # significantly below C_bulk.
    if abs(C_bulk - C_sink_wt_pct) > 1e-9:
        idx_80 = int(round(0.8 * (n_spatial - 1)))
        c_at_80 = float(final_profile[idx_80])
        warn_threshold = C_bulk - 0.05 * (C_bulk - C_sink_wt_pct)
        if c_at_80 < warn_threshold:
            domain_nm = x_max_m * 1e9
            sim_warnings.append(
                f"Depletion zone has penetrated to at least 80 % of the "
                f"domain boundary ({domain_nm:.1f} nm): concentration at "
                f"80 % position = {c_at_80:.3f} wt% (threshold "
                f"{warn_threshold:.3f} wt%).  Results near the right "
                f"boundary may be affected.  Consider increasing x_max_m."
            )

    # ------------------------------------------------------------------
    # 11. Provenance metadata
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "element": element,
        "matrix": matrix,
        "arrhenius": arrhenius_meta,
        "solver": {
            "scheme": "crank_nicolson",
            "n_spatial": n_spatial,
            "x_max_m": x_max_m,
            "dx_m": dx,
            "min_steps_per_ht_step": _MIN_STEPS,
            "max_steps_per_ht_step": _MAX_STEPS,
            "snap_every": snap_every,
            "total_cn_steps": total_n_cn_steps,
        },
        "ht_schedule_summary": ht_step_summaries,
    }

    # ------------------------------------------------------------------
    # 12. Construct and return validated DiffusionOutput
    # ------------------------------------------------------------------
    return DiffusionOutput(
        element=element,
        matrix=matrix,
        x_m=x_m.tolist(),
        t_s=stored_t_s,
        concentration_profiles=stored_profiles,
        C_bulk_wt_pct=C_bulk,
        C_sink_wt_pct=C_sink_wt_pct,
        min_concentration_wt_pct=min_conc,
        depletion_depth_nm=depletion_depth_nm,
        warnings=sim_warnings,
        metadata=metadata,
    )
