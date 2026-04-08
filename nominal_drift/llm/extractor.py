"""
nominal_drift.llm.extractor
============================
Minimal prompt-rendering helpers for the LLM narration layer.

This module is intentionally small.  It contains only the utilities that
``narration.py`` needs to transform structured Pydantic objects into the
plain-string fields expected by the Jinja2 prompt templates.

Do NOT add full NLP extraction logic here (that belongs in a future
``extractors/`` sub-package).  The only public symbol is
``summarise_ht_schedule``.

Public API
----------
``summarise_ht_schedule(ht_schedule) -> str``
    Return a compact, human-readable one-line summary of an HTSchedule
    for insertion into the ``ht_summary`` template variable.
"""

from __future__ import annotations

from nominal_drift.schemas.ht_schedule import HTSchedule


def summarise_ht_schedule(ht_schedule: HTSchedule) -> str:
    """Return a compact human-readable summary string for an HTSchedule.

    Each step is rendered as::

        Step N: <T> °C × <duration> (<type>)[, <cooling_method>]

    Steps are separated by " → " and listed in step-number order.
    Duration is shown in minutes for holds under one hour, and in hours
    (decimal if needed) for longer holds.

    Parameters
    ----------
    ht_schedule : HTSchedule
        A validated HTSchedule instance.

    Returns
    -------
    str
        A single-line summary string, e.g.::

            "Step 1: 1080 °C × 30 min (solution_anneal), water_quench →
             Step 2: 700 °C × 2.0 h (sensitization_soak), air_cool"

    Examples
    --------
    >>> from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep
    >>> s = HTSchedule(steps=[
    ...     HTStep(step=1, type="sensitization_soak",
    ...            T_hold_C=700.0, hold_min=60.0,
    ...            cooling_method="air_cool"),
    ... ])
    >>> summarise_ht_schedule(s)
    'Step 1: 700 °C × 60 min (sensitization_soak), air_cool'
    """
    parts: list[str] = []
    for step in sorted(ht_schedule.steps, key=lambda s: s.step):
        # Duration: use hours for holds ≥ 60 min
        if step.hold_min < 60.0:
            duration = f"{step.hold_min:.3g} min"
        else:
            hours = step.hold_min / 60.0
            # Show as integer if exact (within float tolerance), otherwise 1 d.p.
            duration = (
                f"{int(round(hours))} h"
                if abs(hours - round(hours)) < 1e-9
                else f"{hours:.1f} h"
            )

        part = (
            f"Step {step.step}: {step.T_hold_C:.4g} °C × {duration}"
            f" ({step.type})"
        )
        if step.cooling_method:
            part += f", {step.cooling_method}"
        parts.append(part)

    return " → ".join(parts)
