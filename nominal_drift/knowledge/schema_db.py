"""
nominal_drift.knowledge.schema_db
===================================
SQLAlchemy 2.0 ORM model and database-engine helpers for the local
experiment store.

The single table ``experiments`` holds every field that
``experiment_store.write_experiment`` can persist.  Compound Python objects
(composition dict, HT-schedule dict, warnings list) are serialised to JSON
TEXT columns; the experiment_store layer handles serialisation /
deserialisation so this module stays focused on table structure.

Public API
----------
``ExperimentRecord``
    SQLAlchemy ``DeclarativeBase`` mapped class — one row per simulation run.

``init_db(db_path)``
    Create all tables (idempotent — safe to call on an existing database).

``make_engine(db_path)``
    Return a ``create_engine`` instance for *db_path*.

``session_scope(db_path)``
    Context-manager yielding a ``Session``; commits on clean exit, rolls back
    on exception.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    Float,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
)


# ---------------------------------------------------------------------------
# ORM base
# ---------------------------------------------------------------------------

class _Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class ExperimentRecord(_Base):
    """One row per diffusion-simulation experiment.

    Columns
    -------
    experiment_id         : str (PK) — UUID or caller-supplied string.
    created_at            : str — ISO-8601 timestamp string.
    alloy_designation     : str — e.g. "316L".
    alloy_matrix          : str — e.g. "austenite".
    composition_json      : TEXT — JSON-serialised composition dict.
    ht_schedule_json      : TEXT — JSON-serialised HT-schedule dict.
    element               : str — diffusing species symbol, e.g. "Cr".
    matrix                : str — diffusion-coefficient matrix key.
    c_bulk_wt_pct         : float — bulk concentration [wt%].
    c_sink_wt_pct         : float — grain-boundary sink concentration [wt%].
    min_concentration_wt_pct : float — minimum [element] anywhere in domain.
    depletion_depth_nm    : float | None — depth of depletion zone [nm].
    warnings_json         : TEXT — JSON-serialised list of warning strings.
    plot_path             : str | None — absolute path to static PNG.
    animation_path        : str | None — absolute path to MP4/GIF animation.
    user_label            : str | None — short human-readable run label.
    user_notes            : TEXT | None — free-form notes.
    """

    __tablename__ = "experiments"

    # ---- identity / audit ----
    experiment_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at:    Mapped[str] = mapped_column(String, nullable=False)

    # ---- alloy ----
    alloy_designation: Mapped[str] = mapped_column(String, nullable=False, index=True)
    alloy_matrix:      Mapped[str] = mapped_column(String, nullable=False)
    composition_json:  Mapped[str] = mapped_column(Text,   nullable=False)
    ht_schedule_json:  Mapped[str] = mapped_column(Text,   nullable=False)

    # ---- diffusion ----
    element:  Mapped[str] = mapped_column(String, nullable=False)
    matrix:   Mapped[str] = mapped_column(String, nullable=False)

    # ---- results ----
    c_bulk_wt_pct:            Mapped[float] = mapped_column(Float, nullable=False)
    c_sink_wt_pct:            Mapped[float] = mapped_column(Float, nullable=False)
    min_concentration_wt_pct: Mapped[float] = mapped_column(Float, nullable=False)
    depletion_depth_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ---- diagnostics ----
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # ---- artefact paths ----
    plot_path:      Mapped[Optional[str]] = mapped_column(String, nullable=True)
    animation_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # ---- user metadata ----
    user_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    user_notes: Mapped[Optional[str]] = mapped_column(Text,   nullable=True)


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def make_engine(db_path: str) -> Engine:
    """Return a SQLAlchemy engine for the SQLite file at *db_path*.

    Parameters
    ----------
    db_path : str
        Absolute or relative path to the ``.db`` file.  The parent directory
        must already exist (call ``init_db`` which handles directory creation).

    Returns
    -------
    Engine
        A ``create_engine`` instance configured for SQLite.
    """
    return create_engine(f"sqlite:///{db_path}", echo=False, future=True)


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Create all tables in *db_path* (idempotent).

    Creates the parent directory tree if it does not already exist, then
    calls ``metadata.create_all`` with ``checkfirst=True`` so calling this
    function on an existing database is safe.

    Parameters
    ----------
    db_path : str
        Path to the SQLite ``.db`` file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(db_path)
    _Base.metadata.create_all(engine, checkfirst=True)
    engine.dispose()


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def session_scope(db_path: str) -> Generator[Session, None, None]:
    """Yield a ``Session`` that commits on clean exit and rolls back on error.

    The engine is created fresh for each call so this helper is safe to use
    from multiple test threads / processes without shared state.

    Parameters
    ----------
    db_path : str
        Path to the SQLite ``.db`` file (must already be initialised).

    Yields
    ------
    Session
        An open SQLAlchemy ``Session`` bound to *db_path*.
    """
    engine = make_engine(db_path)
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            engine.dispose()
