"""
nominal_drift.knowledge.experiment_store
=========================================
Minimal SQLite-backed experiment store for NominalDrift Sprint 1.

Provides four public functions for persisting and retrieving diffusion
simulation results.  All I/O is done through plain Python dicts — no
SQLAlchemy ORM objects are ever returned to the caller.

Compound fields (``composition_json``, ``ht_schedule_json``,
``warnings_json``) may be supplied as either a Python dict/list or a
pre-serialised JSON string; they are always deserialised back to
dict/list on read.

Public API
----------
``init_store(db_path)``
    Create the database and tables if they do not yet exist.

``write_experiment(record, db_path)``
    Persist one experiment record; return the experiment_id string.

``read_experiment(experiment_id, db_path)``
    Return a single record as a plain dict; raise ``KeyError`` if not found.

``list_experiments(alloy_designation, limit, db_path)``
    Return a list of plain-dict records, newest first, with optional
    alloy filter and row-count limit (default 20).

Default DB path
---------------
``data/experiments.db`` relative to the repository root (resolved via
``__file__``).  All public functions accept an optional ``db_path``
keyword argument that overrides this default, which is used extensively
in the test suite for per-test isolation.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from nominal_drift.knowledge.schema_db import (
    ExperimentRecord,
    init_db,
    session_scope,
)

# ---------------------------------------------------------------------------
# Default database location
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = str(_REPO_ROOT / "data" / "experiments.db")


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------

def _to_json_str(value: Any) -> str:
    """Serialise *value* to a JSON string if it is not already one.

    Accepts a dict, list, or a pre-serialised JSON string.  Raises
    ``TypeError`` for any other type.
    """
    if isinstance(value, str):
        # Validate that it is actually valid JSON and normalise
        return json.dumps(json.loads(value))
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    raise TypeError(
        f"Expected dict, list, or JSON str; got {type(value).__name__!r}"
    )


def _from_json_str(raw: Optional[str], fallback: Any = None) -> Any:
    """Deserialise *raw* JSON string; return *fallback* if *raw* is None."""
    if raw is None:
        return fallback
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_store(db_path: str = _DEFAULT_DB) -> None:
    """Create the database file and ``experiments`` table if absent.

    Safe to call multiple times on the same database (idempotent).

    Parameters
    ----------
    db_path : str
        Path to the SQLite ``.db`` file.  Parent directories are created
        automatically.
    """
    init_db(db_path)


def write_experiment(
    record: dict,
    db_path: str = _DEFAULT_DB,
) -> str:
    """Persist one experiment record and return its ID.

    The store is automatically initialised if the database file does not yet
    exist, so callers do not need to call ``init_store`` first.

    Parameters
    ----------
    record : dict
        Flat dict of experiment fields.  Required keys:

        * ``alloy_designation`` — e.g. ``"316L"``
        * ``alloy_matrix``      — e.g. ``"austenite"``
        * ``composition_json``  — dict or JSON string
        * ``ht_schedule_json``  — dict or JSON string
        * ``element``           — e.g. ``"Cr"``
        * ``matrix``            — e.g. ``"austenite_FeCrNi"``
        * ``c_bulk_wt_pct``     — float
        * ``c_sink_wt_pct``     — float
        * ``min_concentration_wt_pct`` — float

        Optional keys: ``experiment_id``, ``created_at``,
        ``depletion_depth_nm``, ``warnings_json``, ``plot_path``,
        ``animation_path``, ``user_label``, ``user_notes``.

    db_path : str
        Path to the SQLite ``.db`` file.

    Returns
    -------
    str
        The ``experiment_id`` of the persisted record (UUID string if
        none was supplied in *record*).
    """
    # ---- auto-initialise ----
    init_db(db_path)

    # ---- identity / audit ----
    exp_id     = record.get("experiment_id") or str(uuid.uuid4())
    created_at = record.get("created_at") or (
        datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    )

    # ---- JSON fields ----
    composition_json = _to_json_str(record["composition_json"])
    ht_schedule_json = _to_json_str(record["ht_schedule_json"])
    warnings_raw = record.get("warnings_json", [])
    warnings_json = _to_json_str(warnings_raw)

    orm_row = ExperimentRecord(
        experiment_id            = exp_id,
        created_at               = created_at,
        alloy_designation        = record["alloy_designation"],
        alloy_matrix             = record["alloy_matrix"],
        composition_json         = composition_json,
        ht_schedule_json         = ht_schedule_json,
        element                  = record["element"],
        matrix                   = record["matrix"],
        c_bulk_wt_pct            = float(record["c_bulk_wt_pct"]),
        c_sink_wt_pct            = float(record["c_sink_wt_pct"]),
        min_concentration_wt_pct = float(record["min_concentration_wt_pct"]),
        depletion_depth_nm       = record.get("depletion_depth_nm"),
        warnings_json            = warnings_json,
        plot_path                = record.get("plot_path"),
        animation_path           = record.get("animation_path"),
        user_label               = record.get("user_label"),
        user_notes               = record.get("user_notes"),
    )

    with session_scope(db_path) as session:
        session.add(orm_row)

    return exp_id


def read_experiment(
    experiment_id: str,
    db_path: str = _DEFAULT_DB,
) -> dict:
    """Return a single experiment record as a plain dict.

    JSON text columns are deserialised back to Python objects.

    Parameters
    ----------
    experiment_id : str
        The ID to look up.
    db_path : str
        Path to the SQLite ``.db`` file.

    Returns
    -------
    dict
        All columns as a plain Python dict.  JSON fields are deserialised
        to dict/list.

    Raises
    ------
    KeyError
        If *experiment_id* does not exist in the database.
    """
    with session_scope(db_path) as session:
        row = session.get(ExperimentRecord, experiment_id)
        if row is None:
            raise KeyError(
                f"No experiment with id {experiment_id!r} found in {db_path!r}"
            )
        result = _row_to_dict(row)

    return result


def list_experiments(
    alloy_designation: Optional[str] = None,
    limit: int = 20,
    db_path: str = _DEFAULT_DB,
) -> list[dict]:
    """Return a list of experiment records, newest first.

    Parameters
    ----------
    alloy_designation : str | None
        If provided, only records matching this alloy are returned.
    limit : int
        Maximum number of records to return (default 20).
    db_path : str
        Path to the SQLite ``.db`` file.  If the file does not exist the
        store is initialised first and an empty list is returned.

    Returns
    -------
    list[dict]
        Plain-dict records, ordered by ``created_at`` descending.
    """
    # Auto-initialise so an empty DB returns [] rather than raising
    init_db(db_path)

    with session_scope(db_path) as session:
        stmt = select(ExperimentRecord).order_by(
            ExperimentRecord.created_at.desc()
        )
        if alloy_designation is not None:
            stmt = stmt.where(
                ExperimentRecord.alloy_designation == alloy_designation
            )
        stmt = stmt.limit(limit)
        rows = session.scalars(stmt).all()
        results = [_row_to_dict(row) for row in rows]

    return results


# ---------------------------------------------------------------------------
# Internal row-to-dict converter
# ---------------------------------------------------------------------------

def _row_to_dict(row: ExperimentRecord) -> dict:
    """Convert an ORM row to a plain dict with JSON fields deserialised."""
    return {
        "experiment_id":            row.experiment_id,
        "created_at":               row.created_at,
        "alloy_designation":        row.alloy_designation,
        "alloy_matrix":             row.alloy_matrix,
        "composition_json":         _from_json_str(row.composition_json, {}),
        "ht_schedule_json":         _from_json_str(row.ht_schedule_json, {}),
        "element":                  row.element,
        "matrix":                   row.matrix,
        "c_bulk_wt_pct":            row.c_bulk_wt_pct,
        "c_sink_wt_pct":            row.c_sink_wt_pct,
        "min_concentration_wt_pct": row.min_concentration_wt_pct,
        "depletion_depth_nm":       row.depletion_depth_nm,
        "warnings_json":            _from_json_str(row.warnings_json, []),
        "plot_path":                row.plot_path,
        "animation_path":           row.animation_path,
        "user_label":               row.user_label,
        "user_notes":               row.user_notes,
    }
