"""
nominal_drift.schemas.ht_schedule
================================
Pydantic v2 schemas for heat-treatment (HT) schedule input.

A schedule is an ordered sequence of HTStep records describing a complete
thermal history applied to a material specimen.  The orchestrator forwards
a validated HTSchedule to scientific engine modules (diffusion solver, HT
interpreter, sensitization model, etc.).

Design notes
------------
- Each step carries enough physical information for the diffusion engine
  to reconstruct the temperature–time history: T_hold_C, hold_min, and
  optionally ramp_rate_C_min.
- Step numbers must be positive, unique, and strictly ascending so the
  schedule can be reconstructed unambiguously from any serialised form.
- cooling_method and atmosphere are optional strings rather than enumerations
  in Sprint 1.  Downstream modules validate that the value is supported for
  the requested calculation and raise a typed exception if not.
- The schema is not limited to sensitization-relevant schedules.  It can
  represent homogenisation anneals, solution treatments, age-hardening
  cycles, stress-relief treatments, or any other thermal workflow.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# HTStep — single step in a thermal schedule
# ---------------------------------------------------------------------------

class HTStep(BaseModel):
    """
    One discrete step in a heat-treatment schedule.

    Parameters
    ----------
    step : int
        Step number.  Must be a positive integer.  Used to enforce ordering
        and detect duplicates within an HTSchedule.
    type : str
        Human-readable step type label, e.g. "solution_anneal",
        "sensitization_soak", "stress_relief", "homogenisation",
        "age_hardening".  Not constrained to an enum in Sprint 1 — the
        HT interpreter module (Day 3+) normalises and maps this field.
    T_hold_C : float
        Isothermal hold temperature in degrees Celsius.  Must be > 0°C.
        Note: 0°C is rejected because no metallurgically meaningful
        isothermal hold occurs at the freezing point of water.  A future
        validator may add an alloy-specific upper-limit guard (liquidus).
    hold_min : float
        Duration of the isothermal hold in minutes.  Must be > 0.
    ramp_rate_C_min : float | None
        Heating rate to T_hold_C in °C/min.  None implies the ramp is
        either negligible (fast furnace transfer) or unknown.
        Must be > 0 if provided.
    cooling_method : str | None
        Post-hold cooling description, e.g. "water_quench", "air_cool",
        "furnace_cool", "oil_quench".  None if unknown or not applicable.
    atmosphere : str | None
        Furnace atmosphere during this step, e.g. "argon", "vacuum",
        "air", "N2", "H2".  None if unknown.

    Examples
    --------
    >>> step = HTStep(
    ...     step=1,
    ...     type="solution_anneal",
    ...     T_hold_C=1080.0,
    ...     hold_min=30.0,
    ...     ramp_rate_C_min=10.0,
    ...     cooling_method="water_quench",
    ...     atmosphere="argon",
    ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: Annotated[int, Field(gt=0, description="Step number; must be a positive integer.")]

    type: str = Field(
        ...,
        min_length=1,
        description=(
            "Step type label, e.g. 'solution_anneal', 'sensitization_soak', "
            "'homogenisation'.  Normalised by the HT interpreter module."
        ),
    )

    T_hold_C: float = Field(
        ...,
        gt=0.0,
        description=(
            "Isothermal hold temperature in degrees Celsius.  Must be > 0°C."
        ),
    )

    hold_min: float = Field(
        ...,
        gt=0.0,
        description="Duration of the isothermal hold in minutes.  Must be > 0.",
    )

    ramp_rate_C_min: float | None = Field(
        default=None,
        description=(
            "Heating rate in °C/min.  None = ramp negligible or unknown.  "
            "Must be > 0 if provided."
        ),
    )

    cooling_method: str | None = Field(
        default=None,
        description=(
            "Post-hold cooling method, e.g. 'water_quench', 'air_cool', "
            "'furnace_cool', 'oil_quench'.  None if unknown."
        ),
    )

    atmosphere: str | None = Field(
        default=None,
        description=(
            "Furnace atmosphere during this step, e.g. 'argon', 'vacuum', "
            "'air', 'N2'.  None if unknown."
        ),
    )

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("type")
    @classmethod
    def type_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("type must not be blank or whitespace.")
        return v.strip()

    @field_validator("ramp_rate_C_min")
    @classmethod
    def ramp_rate_positive(cls, v: float | None) -> float | None:
        """Ramp rate must be strictly positive when supplied."""
        if v is not None and v <= 0.0:
            raise ValueError(
                f"ramp_rate_C_min must be > 0 when provided; got {v}."
            )
        return v

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def T_hold_K(self) -> float:
        """Hold temperature in Kelvin.  Used by Arrhenius calculations."""
        return self.T_hold_C + 273.15

    @property
    def hold_s(self) -> float:
        """Hold duration in seconds.  Used by the diffusion engine."""
        return self.hold_min * 60.0


# ---------------------------------------------------------------------------
# HTSchedule — ordered sequence of steps
# ---------------------------------------------------------------------------

class HTSchedule(BaseModel):
    """
    Ordered sequence of heat-treatment steps describing a complete thermal
    history.

    Parameters
    ----------
    steps : list[HTStep]
        At least one step is required.  Steps must have unique step numbers
        in strictly ascending order.

    Validation rules
    ----------------
    - At least one step must be present.
    - step numbers must be unique across all steps.
    - step numbers must be in strictly ascending order (enforces that the
      physical sequence is unambiguous regardless of serialisation order).

    Examples
    --------
    >>> schedule = HTSchedule(steps=[
    ...     HTStep(step=1, type="solution_anneal",
    ...            T_hold_C=1080.0, hold_min=30.0,
    ...            cooling_method="water_quench"),
    ...     HTStep(step=2, type="sensitization_soak",
    ...            T_hold_C=650.0, hold_min=120.0,
    ...            cooling_method="air_cool"),
    ... ])
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: list[HTStep] = Field(
        ...,
        description=(
            "Ordered list of heat-treatment steps.  At least one step is "
            "required.  Step numbers must be unique and strictly ascending."
        ),
    )

    # ------------------------------------------------------------------
    # Model-level validator
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_step_ordering(self) -> "HTSchedule":
        """
        Enforce:
          1. Schedule is non-empty.
          2. step numbers are unique.
          3. step numbers are strictly ascending.
        """
        steps = self.steps

        # 1 — Non-empty
        if not steps:
            raise ValueError(
                "HTSchedule must contain at least one step."
            )

        step_numbers = [s.step for s in steps]

        # 2 — Uniqueness
        seen: set[int] = set()
        duplicates: list[int] = []
        for n in step_numbers:
            if n in seen:
                duplicates.append(n)
            seen.add(n)
        if duplicates:
            raise ValueError(
                f"HTSchedule contains duplicate step number(s): {duplicates}.  "
                f"Each step must have a unique step number."
            )

        # 3 — Strictly ascending
        for i in range(len(step_numbers) - 1):
            if step_numbers[i] >= step_numbers[i + 1]:
                raise ValueError(
                    f"HTSchedule step numbers must be strictly ascending.  "
                    f"Step {step_numbers[i]} is followed by step "
                    f"{step_numbers[i + 1]} at position {i + 1}."
                )

        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_hold_min(self) -> float:
        """Sum of all isothermal hold durations in minutes."""
        return sum(s.hold_min for s in self.steps)

    @property
    def total_hold_s(self) -> float:
        """Sum of all isothermal hold durations in seconds."""
        return self.total_hold_min * 60.0

    @property
    def T_min_C(self) -> float:
        """Lowest hold temperature across all steps (°C)."""
        return min(s.T_hold_C for s in self.steps)

    @property
    def T_max_C(self) -> float:
        """Highest hold temperature across all steps (°C)."""
        return max(s.T_hold_C for s in self.steps)

    @property
    def n_steps(self) -> int:
        """Number of steps in the schedule."""
        return len(self.steps)
