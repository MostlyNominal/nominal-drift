"""
nominal_drift.core.session
===========================
Minimal session container for NominalDrift Sprint 1.

A ``NominalDriftSession`` captures the inputs and outputs of a single
workflow invocation so that the orchestrator and CLI can pass state
between steps without threading individual variables through every
function call.

Design notes
------------
- This is a plain ``dataclass``, not a Pydantic model.  Session objects
  live only in memory for the duration of one run and are not persisted
  directly (the experiment store handles persistence).
- Fields are intentionally minimal for Sprint 1.  Future phases may add
  ``cached_outputs: dict``, ``rag_context: list``, and ``config: dict``
  without breaking existing callers.
- ``output_dir`` and ``db_path`` are plain strings (not ``Path``) to
  match the conventions used by the experiment store and viz layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.ht_schedule import HTSchedule


def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string (no microseconds)."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class NominalDriftSession:
    """Lightweight container holding the context of one workflow run.

    Parameters
    ----------
    composition : AlloyComposition
        Validated alloy composition used in this session.
    ht_schedule : HTSchedule
        Validated heat-treatment schedule applied in this session.
    experiment_id : str
        UUID string assigned to this run by the experiment store.
    output_dir : str
        Absolute path to the run-specific output directory where plots
        and animations are saved.
    db_path : str | None
        Path to the SQLite experiment database.  ``None`` means the
        experiment store will use its own default path.
    created_at : str
        ISO-8601 UTC timestamp marking when the session was constructed.
        Auto-generated if not supplied.

    Notes
    -----
    The session object is created by ``orchestrator.run_showcase_workflow``
    and is not returned in the public result dict — it is used internally
    to organise state during the workflow.  External callers receive a
    plain ``dict`` from the orchestrator.
    """

    composition:   AlloyComposition
    ht_schedule:   HTSchedule
    experiment_id: str
    output_dir:    str
    db_path:       Optional[str] = None
    created_at:    str = field(default_factory=_utc_now)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NominalDriftSession("
            f"experiment_id={self.experiment_id!r}, "
            f"alloy={self.composition.alloy_designation!r}, "
            f"output_dir={self.output_dir!r})"
        )
