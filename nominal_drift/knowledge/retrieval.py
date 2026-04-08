"""
nominal_drift.knowledge.retrieval
===================================
First useful retrieval layer on top of the SQLite experiment store.

PURPOSE
-------
Extends the experiment store from simple CRUD toward structured experimental
memory.  All queries are SQL-backed, rule-based, and deterministic — no
embeddings or semantic search in Sprint 2C.  The functions here answer
questions such as:

  * "show me all 316L experiments"
  * "find runs for Cr in this alloy system"
  * "retrieve experiments with depletion depth between 30 and 80 nm"
  * "find experiments most similar to this run"
  * "what did we observe at a similar c_sink concentration?"

DESIGN PRINCIPLES
-----------------
* Zero changes to existing experiment_store.py or schema_db.py.
* All public functions return plain Python dicts — identical contract
  to the experiment_store layer.  Callers can drop in either API.
* Deterministic: same DB state + same arguments → same result, always.
* Ordering rules are explicit and documented so users understand ranking.
* Limit is always respected — no accidental full-table scans at the call site.

SIMILARITY RANKING (find_similar_experiments)
---------------------------------------------
Similarity is structured and rule-based, using a priority ordering:

  1. Exact alloy_designation match (highest priority)
  2. Exact element match
  3. Exact matrix match
  4. Closest c_sink_wt_pct by absolute difference
  5. Most recent created_at (recency tiebreaker)

Each criterion is optional — omit a field by passing ``None`` and that
dimension is excluded from the ranking.

FUTURE EXTENSION SURFACE
-------------------------
This module is designed to be extended without breaking existing callers:

  * Semantic search → add ``find_by_notes(query_text)`` using FTS5
  * Embedding-based similarity → add ``find_by_vector(embedding)``
  * Literature retrieval → a separate ``literature_retrieval.py`` module
  * Multi-modal memory → extend ``find_similar_experiments`` with new dims

Public API
----------
``find_by_alloy(alloy_designation, *, limit, db_path)``
``find_by_element(element, *, limit, db_path)``
``find_by_depletion_depth(min_depth_nm, max_depth_nm, *, limit, db_path)``
``find_similar_experiments(*, alloy_designation, element, matrix,
                           c_sink_wt_pct, limit, db_path)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import case, func, select

from nominal_drift.knowledge.experiment_store import (
    _DEFAULT_DB,       # default DB path constant
    _row_to_dict,      # ORM row → plain dict (no JSON leakage)
)
from nominal_drift.knowledge.schema_db import (
    ExperimentRecord,
    init_db,
    session_scope,
)


# ---------------------------------------------------------------------------
# Public retrieval functions
# ---------------------------------------------------------------------------

def find_by_alloy(
    alloy_designation: str,
    *,
    limit: int = 20,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Return all experiment records for a given alloy, newest first.

    Parameters
    ----------
    alloy_designation : str
        Exact alloy name to filter on, e.g. ``"316L"``.  Case-sensitive.
    limit : int
        Maximum number of records to return.  Default: 20.
    db_path : str | None
        Path to the SQLite ``.db`` file.  Uses the default store path
        when ``None``.

    Returns
    -------
    list[dict]
        Plain-dict records ordered by ``created_at`` descending.
        Empty list if no matches exist or the database is empty.
    """
    db = _resolve_db(db_path)
    init_db(db)

    with session_scope(db) as session:
        stmt = (
            select(ExperimentRecord)
            .where(ExperimentRecord.alloy_designation == alloy_designation)
            .order_by(ExperimentRecord.created_at.desc())
            .limit(limit)
        )
        rows = session.scalars(stmt).all()
        return [_row_to_dict(r) for r in rows]


def find_by_element(
    element: str,
    *,
    limit: int = 20,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Return all experiment records for a given diffusing element, newest first.

    Parameters
    ----------
    element : str
        Element symbol to filter on, e.g. ``"Cr"``, ``"C"``, ``"N"``.
        Case-sensitive.
    limit : int
        Maximum number of records to return.  Default: 20.
    db_path : str | None
        Path to the SQLite ``.db`` file.  Uses the default store path
        when ``None``.

    Returns
    -------
    list[dict]
        Plain-dict records ordered by ``created_at`` descending.
        Empty list if no matches exist or the database is empty.
    """
    db = _resolve_db(db_path)
    init_db(db)

    with session_scope(db) as session:
        stmt = (
            select(ExperimentRecord)
            .where(ExperimentRecord.element == element)
            .order_by(ExperimentRecord.created_at.desc())
            .limit(limit)
        )
        rows = session.scalars(stmt).all()
        return [_row_to_dict(r) for r in rows]


def find_by_depletion_depth(
    min_depth_nm: Optional[float] = None,
    max_depth_nm: Optional[float] = None,
    *,
    limit: int = 20,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Return experiments whose depletion depth falls within a range.

    Only records with a non-null ``depletion_depth_nm`` are ever returned.
    Passing both ``min_depth_nm=None`` and ``max_depth_nm=None`` returns
    all records that have a recorded depletion depth (useful for "show me
    every run where a depletion front was detected").

    Parameters
    ----------
    min_depth_nm : float | None
        Lower bound (inclusive) in nanometres.  ``None`` = no lower bound.
    max_depth_nm : float | None
        Upper bound (inclusive) in nanometres.  ``None`` = no upper bound.
    limit : int
        Maximum number of records to return.  Default: 20.
    db_path : str | None
        Path to the SQLite ``.db`` file.  Uses the default store path
        when ``None``.

    Returns
    -------
    list[dict]
        Plain-dict records ordered by ``depletion_depth_nm`` ascending
        (shallowest first), then by ``created_at`` descending for ties.
        Empty list if no matches exist.
    """
    db = _resolve_db(db_path)
    init_db(db)

    with session_scope(db) as session:
        # Base: only rows with a recorded depletion depth
        stmt = select(ExperimentRecord).where(
            ExperimentRecord.depletion_depth_nm.isnot(None)
        )

        if min_depth_nm is not None:
            stmt = stmt.where(
                ExperimentRecord.depletion_depth_nm >= min_depth_nm
            )

        if max_depth_nm is not None:
            stmt = stmt.where(
                ExperimentRecord.depletion_depth_nm <= max_depth_nm
            )

        stmt = stmt.order_by(
            ExperimentRecord.depletion_depth_nm.asc(),
            ExperimentRecord.created_at.desc(),
        ).limit(limit)

        rows = session.scalars(stmt).all()
        return [_row_to_dict(r) for r in rows]


def find_similar_experiments(
    *,
    alloy_designation: Optional[str] = None,
    element: Optional[str] = None,
    matrix: Optional[str] = None,
    c_sink_wt_pct: Optional[float] = None,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Return experiments ranked by structural similarity to a query.

    Similarity is computed using an explicit priority ordering of criteria
    (see module docstring).  Each criterion is optional — omit with ``None``
    to exclude it from ranking.

    Ranking rules (in priority order):
      1. Exact ``alloy_designation`` match preferred (if provided)
      2. Exact ``element`` match preferred (if provided)
      3. Exact ``matrix`` match preferred (if provided)
      4. Closest ``c_sink_wt_pct`` by absolute difference (if provided)
      5. Most recent ``created_at`` as final tiebreaker

    Parameters
    ----------
    alloy_designation : str | None
        Alloy to match, e.g. ``"316L"``.
    element : str | None
        Diffusing species to match, e.g. ``"Cr"``.
    matrix : str | None
        Matrix key to match, e.g. ``"austenite_FeCrNi"``.
    c_sink_wt_pct : float | None
        Grain-boundary sink concentration to match.  Closest value wins.
    limit : int
        Maximum number of records to return.  Default: 10.
    db_path : str | None
        Path to the SQLite ``.db`` file.  Uses the default store path
        when ``None``.

    Returns
    -------
    list[dict]
        Plain-dict records ranked from most-similar to least-similar.
        Empty list if the database is empty.
    """
    db = _resolve_db(db_path)
    init_db(db)

    with session_scope(db) as session:
        order_clauses = []

        if alloy_designation is not None:
            order_clauses.append(
                case(
                    (ExperimentRecord.alloy_designation == alloy_designation, 0),
                    else_=1,
                ).asc()
            )

        if element is not None:
            order_clauses.append(
                case(
                    (ExperimentRecord.element == element, 0),
                    else_=1,
                ).asc()
            )

        if matrix is not None:
            order_clauses.append(
                case(
                    (ExperimentRecord.matrix == matrix, 0),
                    else_=1,
                ).asc()
            )

        if c_sink_wt_pct is not None:
            order_clauses.append(
                func.abs(
                    ExperimentRecord.c_sink_wt_pct - c_sink_wt_pct
                ).asc()
            )

        # Always break ties by recency
        order_clauses.append(ExperimentRecord.created_at.desc())

        stmt = (
            select(ExperimentRecord)
            .order_by(*order_clauses)
            .limit(limit)
        )
        rows = session.scalars(stmt).all()
        return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_db(db_path: Optional[str]) -> str:
    """Return *db_path* if given, else the package default."""
    return db_path if db_path is not None else _DEFAULT_DB
