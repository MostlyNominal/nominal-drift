"""
nominal_drift.schemas.diffusion_output
=======================================
Pydantic v2 output schema for a completed 1D diffusion simulation.

Design principles
-----------------
- Element-agnostic: the ``element`` field identifies which species was
  simulated (Cr, C, N, or any future element).  All concentration values
  are in wt% regardless of element, keeping the interface uniform across
  species.
- Multi-species extensible: the schema stores one simulation result per
  instance.  Coupled multi-species runs (e.g. simultaneous Cr + C) are
  represented as a list of DiffusionOutput objects, one per element, at
  the orchestrator level.  This keeps each output self-describing and
  serialisable independently.
- Matrix-aware: the ``matrix`` field records which Arrhenius dataset was
  used.  Future DFT-informed mobility databases will supply different
  matrix labels (e.g. "austenite_DFT", "ferrite_CALPHAD") while the
  schema contract remains identical.
- Immutable: frozen=True prevents accidental mutation of simulation results
  mid-workflow.

Field layout
------------
Spatial / temporal axes:
    x_m                  — spatial grid positions [m]
    t_s                  — time stamps corresponding to each stored profile [s]

Concentration data:
    concentration_profiles   — outer list: time steps, inner list: spatial nodes
                               shape: (n_timesteps_stored, n_spatial)
                               units: wt%

Derived scalar results:
    min_concentration_wt_pct — minimum concentration reached anywhere in the
                               domain over the entire simulation [wt%]
    depletion_depth_nm       — distance from x=0 at which concentration first
                               exceeds C_sink + depletion_threshold_wt_pct [nm]
                               None if the depletion front never enters the domain

Boundary / initial conditions (preserved for reproducibility):
    C_bulk_wt_pct            — far-field / initial concentration [wt%]
    C_sink_wt_pct            — Dirichlet value at x=0 (grain boundary) [wt%]

Quality / warning flags:
    warnings                 — list of human-readable warning strings;
                               empty if simulation ran without issues

Provenance metadata:
    metadata                 — free-form dict; populated by solve_diffusion()
                               with solver parameters, Arrhenius constants used,
                               HT schedule summary, etc.  Keys are strings.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiffusionOutput(BaseModel):
    """
    Result of a completed 1D diffusion simulation for a single element.

    Parameters
    ----------
    element : str
        Chemical symbol of the diffusing species, e.g. ``"Cr"``, ``"C"``,
        ``"N"``.  Used by downstream modules to select the correct
        post-processing logic (e.g. sensitization threshold for Cr is
        different from the one used for N-driven Cr₂N depletion).
    matrix : str
        Matrix identifier used to look up Arrhenius constants, e.g.
        ``"austenite_FeCrNi"``.  Preserved here so results are
        self-describing and can be reproduced without the original config.
    x_m : list[float]
        Spatial grid node positions in metres.  Length = n_spatial.
        x_m[0] = 0 (grain boundary), x_m[-1] = x_max.
    t_s : list[float]
        Elapsed time stamps at which concentration profiles were stored,
        in seconds.  Length = n_timesteps_stored.
        t_s[0] = 0 (initial condition).
    concentration_profiles : list[list[float]]
        Concentration field at each stored time step.
        Shape: (n_timesteps_stored, n_spatial).  Units: wt%.
        concentration_profiles[i][j] = C(x_m[j], t_s[i]).
    C_bulk_wt_pct : float
        Nominal / far-field concentration of the element [wt%].
        Used as the initial condition and right-boundary Dirichlet value.
    C_sink_wt_pct : float
        Grain-boundary sink concentration [wt%].
        Left-boundary Dirichlet value under the fast-precipitation
        approximation.  For Cr in 304/316L: default 12.0 wt%.
    min_concentration_wt_pct : float
        Minimum concentration value reached anywhere in the domain over
        the entire simulation [wt%].  For Cr depletion this is the lowest
        recorded Cr content at the grain boundary.
    depletion_depth_nm : float | None
        Distance from x=0 (grain boundary) at which concentration first
        exceeds ``C_sink_wt_pct + 0.5`` wt% at the final time step [nm].
        None if the depletion front has not penetrated the domain or if
        ``C_sink == C_bulk`` (no driving force).
    warnings : list[str]
        Human-readable warning messages generated during the simulation.
        Empty list = no warnings.  Typical warnings:
        - depletion front approaching domain boundary (>80% of x_max)
        - C_sink >= C_bulk (no driving force; flat profile expected)
        - time step stability flag (should not fire for Crank-Nicolson)
    metadata : dict[str, Any]
        Provenance record.  Populated by solve_diffusion() with:
        - ``arrhenius``: {D0, Qd} values used
        - ``solver``: {n_spatial, x_max_m, dt_s, n_steps}
        - ``ht_schedule_summary``: list of {T_hold_C, hold_min} dicts
        - ``element``: element symbol (redundant but convenient)
        - ``matrix``: matrix label (redundant but convenient)
        Downstream report generators and the LLM narration template read
        from this dict so the narration is always tied to the exact
        parameters used.

    Examples
    --------
    >>> result = DiffusionOutput(
    ...     element="Cr",
    ...     matrix="austenite_FeCrNi",
    ...     x_m=[0.0, 2.5e-8, 5.0e-8],
    ...     t_s=[0.0, 3600.0],
    ...     concentration_profiles=[[16.5, 16.5, 16.5], [12.0, 14.2, 16.5]],
    ...     C_bulk_wt_pct=16.5,
    ...     C_sink_wt_pct=12.0,
    ...     min_concentration_wt_pct=12.0,
    ...     depletion_depth_nm=12.3,
    ...     warnings=[],
    ...     metadata={"element": "Cr", "matrix": "austenite_FeCrNi"},
    ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ------------------------------------------------------------------
    # Species and matrix identity
    # ------------------------------------------------------------------

    element: str = Field(
        ...,
        min_length=1,
        description=(
            "Chemical symbol of the diffusing species "
            "(e.g. 'Cr', 'C', 'N').  Case-sensitive."
        ),
    )

    matrix: str = Field(
        ...,
        min_length=1,
        description=(
            "Matrix identifier used to look up Arrhenius constants, "
            "e.g. 'austenite_FeCrNi'.  Must match a key in the "
            "Arrhenius constants database."
        ),
    )

    # ------------------------------------------------------------------
    # Spatial / temporal axes
    # ------------------------------------------------------------------

    x_m: list[float] = Field(
        ...,
        description="Spatial grid positions in metres.  x_m[0] = 0 (grain boundary).",
    )

    t_s: list[float] = Field(
        ...,
        description=(
            "Elapsed time stamps in seconds at which profiles were stored.  "
            "t_s[0] = 0 (initial condition)."
        ),
    )

    # ------------------------------------------------------------------
    # Concentration data
    # ------------------------------------------------------------------

    concentration_profiles: list[list[float]] = Field(
        ...,
        description=(
            "Concentration field [wt%] at each stored time step.  "
            "Shape: (len(t_s), len(x_m)).  "
            "concentration_profiles[i][j] = C(x_m[j], t_s[i])."
        ),
    )

    C_bulk_wt_pct: float = Field(
        ...,
        gt=0.0,
        description="Nominal far-field / initial concentration [wt%].  Must be > 0.",
    )

    C_sink_wt_pct: float = Field(
        ...,
        ge=0.0,
        description=(
            "Grain-boundary Dirichlet sink concentration [wt%].  "
            "Must be >= 0.  For Cr sensitization: typically 12.0 wt%."
        ),
    )

    # ------------------------------------------------------------------
    # Derived scalar results
    # ------------------------------------------------------------------

    min_concentration_wt_pct: float = Field(
        ...,
        description=(
            "Minimum concentration reached anywhere in the domain "
            "over all stored time steps [wt%]."
        ),
    )

    depletion_depth_nm: float | None = Field(
        default=None,
        description=(
            "Distance from grain boundary (x=0) at which concentration "
            "exceeds C_sink + 0.5 wt% at the final time step [nm].  "
            "None if no depletion penetration or no driving force."
        ),
    )

    # ------------------------------------------------------------------
    # Quality flags
    # ------------------------------------------------------------------

    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable warning strings.  Empty list = clean run."
        ),
    )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Solver provenance: Arrhenius constants used, grid parameters, "
            "HT schedule summary.  Read by narration templates and report generators."
        ),
    )

    # ------------------------------------------------------------------
    # Cross-field validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_shape_consistency(self) -> "DiffusionOutput":
        """
        Validate that the axis lengths are consistent with the profile array.

        1. concentration_profiles must have len(t_s) rows.
        2. Each row must have len(x_m) columns.
        3. t_s must be non-decreasing.
        4. x_m must be strictly increasing and start at or near 0.
        5. C_sink_wt_pct must be <= C_bulk_wt_pct (depletion, not enrichment).
        """
        n_t = len(self.t_s)
        n_x = len(self.x_m)
        profiles = self.concentration_profiles

        # 1 — row count
        if len(profiles) != n_t:
            raise ValueError(
                f"concentration_profiles has {len(profiles)} rows but "
                f"t_s has {n_t} entries.  They must match."
            )

        # 2 — column count
        for i, row in enumerate(profiles):
            if len(row) != n_x:
                raise ValueError(
                    f"concentration_profiles[{i}] has {len(row)} values but "
                    f"x_m has {n_x} nodes.  Every profile row must match x_m length."
                )

        # 3 — t_s non-decreasing
        for i in range(len(self.t_s) - 1):
            if self.t_s[i] > self.t_s[i + 1]:
                raise ValueError(
                    f"t_s must be non-decreasing: t_s[{i}]={self.t_s[i]} "
                    f"> t_s[{i+1}]={self.t_s[i+1]}."
                )

        # 4 — x_m strictly increasing
        for i in range(len(self.x_m) - 1):
            if self.x_m[i] >= self.x_m[i + 1]:
                raise ValueError(
                    f"x_m must be strictly increasing: x_m[{i}]={self.x_m[i]} "
                    f">= x_m[{i+1}]={self.x_m[i+1]}."
                )

        # 5 — sink <= bulk (depletion direction)
        if self.C_sink_wt_pct > self.C_bulk_wt_pct:
            raise ValueError(
                f"C_sink_wt_pct ({self.C_sink_wt_pct}) must be <= "
                f"C_bulk_wt_pct ({self.C_bulk_wt_pct}).  The sink model "
                f"represents depletion (C_sink < C_bulk), not enrichment."
            )

        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def x_nm(self) -> list[float]:
        """Spatial grid positions in nanometres (convenience for plotting)."""
        return [x * 1e9 for x in self.x_m]

    @property
    def final_profile(self) -> list[float]:
        """Concentration profile [wt%] at the final stored time step."""
        return self.concentration_profiles[-1]

    @property
    def initial_profile(self) -> list[float]:
        """Concentration profile [wt%] at t=0 (initial condition)."""
        return self.concentration_profiles[0]

    @property
    def n_spatial(self) -> int:
        """Number of spatial grid nodes."""
        return len(self.x_m)

    @property
    def n_timesteps_stored(self) -> int:
        """Number of stored time snapshots (including t=0)."""
        return len(self.t_s)

    @property
    def total_time_s(self) -> float:
        """Total simulated time in seconds."""
        return self.t_s[-1] if self.t_s else 0.0

    @property
    def total_time_min(self) -> float:
        """Total simulated time in minutes."""
        return self.total_time_s / 60.0
