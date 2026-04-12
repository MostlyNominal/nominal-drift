"""
tests/unit/test_experiment_store.py
=====================================
TDD tests for nominal_drift.knowledge.experiment_store.

Test categories:
  1.  DB initialisation — file created, re-initialise is idempotent.
  2.  write_experiment — returns an ID, accepts minimal and full records.
  3.  read_experiment — returns stored values correctly.
  4.  list_experiments — ordering, alloy filter, limit.
  5.  JSON round-trips — composition, schedule, warnings fields.
  6.  File paths — plot_path and animation_path preserved faithfully.
  7.  Optional fields — None / missing values handled without error.
  8.  ID and timestamp generation — UUID generated when absent; timestamp
      generated when absent; caller-supplied values are respected.

Every test receives its own isolated DB via the `db` fixture, which creates
a fresh SQLite file under pytest's tmp_path.  No test depends on the
state of any other test or on the default data/experiments.db file.

Run with:
    pytest tests/unit/test_experiment_store.py -v
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Import helpers (deferred so errors surface as clear test failures)
# ---------------------------------------------------------------------------

def _store():
    from nominal_drift.knowledge import experiment_store
    return experiment_store


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _minimal_record(**overrides) -> dict:
    """Return the smallest valid record dict accepted by write_experiment."""
    base = {
        "alloy_designation": "316L",
        "alloy_matrix":      "austenite",
        "composition_json":  {"Fe": 68.88, "Cr": 16.50, "Ni": 10.10,
                               "Mo": 2.10,  "Mn": 1.80,  "Si": 0.50,
                               "C": 0.02,   "N": 0.07,   "P": 0.03},
        "ht_schedule_json":  {"steps": [{"step": 1, "T_hold_C": 700.0,
                                         "hold_min": 60.0}]},
        "element":                 "Cr",
        "matrix":                  "austenite_FeCrNi",
        "c_bulk_wt_pct":           16.50,
        "c_sink_wt_pct":           12.0,
        "min_concentration_wt_pct": 12.0,
    }
    base.update(overrides)
    return base


def _full_record(**overrides) -> dict:
    """Return a record with every optional field populated."""
    base = _minimal_record(
        user_label       = "sprint1-test",
        depletion_depth_nm = 42.5,
        warnings_json    = ["domain boundary approaching"],
        plot_path        = "/outputs/demo/cr_profile.png",
        animation_path   = "/outputs/demo/cr_animation.mp4",
        user_notes       = "Reference run for 316L at 700 °C.",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> str:
    """Return a fresh, isolated DB path for each test."""
    return str(tmp_path / "test_experiments.db")


# ---------------------------------------------------------------------------
# 1. DB initialisation
# ---------------------------------------------------------------------------

class TestInitStore:

    def test_init_creates_db_file(self, db):
        s = _store()
        s.init_store(db_path=db)
        assert Path(db).exists(), "DB file was not created by init_store()"

    def test_init_is_idempotent(self, db):
        """Calling init_store twice must not raise."""
        s = _store()
        s.init_store(db_path=db)
        s.init_store(db_path=db)   # second call — tables already exist

    def test_init_creates_parent_dirs(self, tmp_path):
        """init_store must create missing parent directories."""
        s = _store()
        nested = str(tmp_path / "a" / "b" / "c" / "experiments.db")
        s.init_store(db_path=nested)
        assert Path(nested).exists()

    def test_write_auto_inits_db(self, db):
        """write_experiment must initialise the DB if it does not exist yet."""
        s = _store()
        # Do NOT call init_store — write_experiment must handle this
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        assert exp_id  # write succeeded → DB was auto-initialised


# ---------------------------------------------------------------------------
# 2. write_experiment
# ---------------------------------------------------------------------------

class TestWriteExperiment:

    def test_write_returns_string_id(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0

    def test_write_returns_valid_uuid_when_no_id_supplied(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        # Must be parseable as a UUID
        parsed = uuid.UUID(exp_id)
        assert str(parsed) == exp_id

    def test_write_respects_caller_supplied_id(self, db):
        s = _store()
        custom_id = "my-custom-run-001"
        exp_id = s.write_experiment(
            _minimal_record(experiment_id=custom_id), db_path=db
        )
        assert exp_id == custom_id

    def test_write_minimal_record_succeeds(self, db):
        s = _store()
        s.write_experiment(_minimal_record(), db_path=db)

    def test_write_full_record_succeeds(self, db):
        s = _store()
        s.write_experiment(_full_record(), db_path=db)

    def test_write_multiple_records(self, db):
        s = _store()
        id1 = s.write_experiment(_minimal_record(), db_path=db)
        id2 = s.write_experiment(_minimal_record(), db_path=db)
        assert id1 != id2


# ---------------------------------------------------------------------------
# 3. read_experiment
# ---------------------------------------------------------------------------

class TestReadExperiment:

    def test_read_returns_dict(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert isinstance(result, dict)

    def test_read_returns_correct_id(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert result["experiment_id"] == exp_id

    def test_read_returns_alloy_designation(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert result["alloy_designation"] == "316L"

    def test_read_returns_numeric_fields(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert result["c_bulk_wt_pct"] == pytest.approx(16.50)
        assert result["c_sink_wt_pct"] == pytest.approx(12.0)
        assert result["min_concentration_wt_pct"] == pytest.approx(12.0)

    def test_read_missing_id_raises(self, db):
        s = _store()
        s.init_store(db_path=db)
        with pytest.raises((KeyError, ValueError, LookupError)):
            s.read_experiment("does-not-exist", db_path=db)

    def test_read_has_created_at_field(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert "created_at" in result
        assert result["created_at"]  # non-empty

    def test_read_respects_caller_supplied_timestamp(self, db):
        s = _store()
        ts = "2025-01-15T12:34:56"
        exp_id = s.write_experiment(
            _minimal_record(created_at=ts), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["created_at"] == ts


# ---------------------------------------------------------------------------
# 4. list_experiments
# ---------------------------------------------------------------------------

class TestListExperiments:

    def test_list_returns_list(self, db):
        s = _store()
        s.write_experiment(_minimal_record(), db_path=db)
        results = s.list_experiments(db_path=db)
        assert isinstance(results, list)

    def test_list_returns_written_record(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        results = s.list_experiments(db_path=db)
        ids = [r["experiment_id"] for r in results]
        assert exp_id in ids

    def test_list_returns_multiple_records(self, db):
        s = _store()
        for _ in range(3):
            s.write_experiment(_minimal_record(), db_path=db)
        results = s.list_experiments(db_path=db)
        assert len(results) >= 3

    def test_list_empty_db_returns_empty_list(self, db):
        s = _store()
        s.init_store(db_path=db)
        results = s.list_experiments(db_path=db)
        assert results == []

    def test_list_alloy_filter_includes_match(self, db):
        s = _store()
        s.write_experiment(_minimal_record(alloy_designation="316L"), db_path=db)
        s.write_experiment(_minimal_record(alloy_designation="304"),   db_path=db)
        results = s.list_experiments(alloy_designation="316L", db_path=db)
        assert all(r["alloy_designation"] == "316L" for r in results)

    def test_list_alloy_filter_excludes_others(self, db):
        s = _store()
        s.write_experiment(_minimal_record(alloy_designation="316L"), db_path=db)
        s.write_experiment(_minimal_record(alloy_designation="304"),   db_path=db)
        results = s.list_experiments(alloy_designation="316L", db_path=db)
        assert not any(r["alloy_designation"] == "304" for r in results)

    def test_list_limit_respected(self, db):
        s = _store()
        for _ in range(10):
            s.write_experiment(_minimal_record(), db_path=db)
        results = s.list_experiments(limit=3, db_path=db)
        assert len(results) <= 3

    def test_list_default_limit_applied(self, db):
        s = _store()
        for _ in range(25):
            s.write_experiment(_minimal_record(), db_path=db)
        results = s.list_experiments(db_path=db)
        assert len(results) <= 20   # default limit is 20

    def test_list_each_item_is_dict(self, db):
        s = _store()
        s.write_experiment(_minimal_record(), db_path=db)
        results = s.list_experiments(db_path=db)
        for item in results:
            assert isinstance(item, dict)


# ---------------------------------------------------------------------------
# 5. JSON round-trips
# ---------------------------------------------------------------------------

class TestJsonRoundTrips:

    def test_composition_dict_roundtrips(self, db):
        s = _store()
        comp = {"Fe": 68.88, "Cr": 16.50, "Ni": 10.10}
        exp_id = s.write_experiment(
            _minimal_record(composition_json=comp), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["composition_json"] == comp

    def test_composition_string_roundtrips(self, db):
        """write_experiment must also accept a pre-serialised JSON string."""
        s = _store()
        comp = {"Fe": 68.88, "Cr": 16.50}
        exp_id = s.write_experiment(
            _minimal_record(composition_json=json.dumps(comp)), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["composition_json"] == comp

    def test_ht_schedule_dict_roundtrips(self, db):
        s = _store()
        sched = {"steps": [{"step": 1, "T_hold_C": 650.0, "hold_min": 120.0}]}
        exp_id = s.write_experiment(
            _minimal_record(ht_schedule_json=sched), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["ht_schedule_json"] == sched

    def test_warnings_list_roundtrips(self, db):
        s = _store()
        warnings = ["domain boundary approaching", "check x_max_m"]
        exp_id = s.write_experiment(
            _minimal_record(warnings_json=warnings), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["warnings_json"] == warnings

    def test_empty_warnings_roundtrips(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(warnings_json=[]), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["warnings_json"] == []

    def test_warnings_absent_returns_empty_list(self, db):
        """If warnings_json is not supplied, read must return [] not None."""
        s = _store()
        record = _minimal_record()
        record.pop("warnings_json", None)   # ensure key is absent
        exp_id = s.write_experiment(record, db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert result["warnings_json"] == []


# ---------------------------------------------------------------------------
# 6. File paths
# ---------------------------------------------------------------------------

class TestFilePaths:

    def test_plot_path_preserved(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(plot_path="/outputs/demo/cr_profile.png"),
            db_path=db,
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["plot_path"] == "/outputs/demo/cr_profile.png"

    def test_animation_path_preserved(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(animation_path="/outputs/demo/cr_anim.mp4"),
            db_path=db,
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["animation_path"] == "/outputs/demo/cr_anim.mp4"

    def test_both_paths_preserved_together(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(
                plot_path="/outputs/plot.png",
                animation_path="/outputs/anim.gif",
            ),
            db_path=db,
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["plot_path"]      == "/outputs/plot.png"
        assert result["animation_path"] == "/outputs/anim.gif"

    def test_missing_paths_return_none(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert result["plot_path"]      is None
        assert result["animation_path"] is None


# ---------------------------------------------------------------------------
# 7. Optional fields
# ---------------------------------------------------------------------------

class TestOptionalFields:

    def test_depletion_depth_none(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(depletion_depth_nm=None), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["depletion_depth_nm"] is None

    def test_depletion_depth_float(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(depletion_depth_nm=42.5), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["depletion_depth_nm"] == pytest.approx(42.5)

    def test_user_label_preserved(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(user_label="my-run-01"), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["user_label"] == "my-run-01"

    def test_user_label_absent_returns_none(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert result["user_label"] is None

    def test_user_notes_preserved(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(user_notes="Reference baseline run."), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert result["user_notes"] == "Reference baseline run."

    def test_user_notes_absent_returns_none(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert result["user_notes"] is None


# ---------------------------------------------------------------------------
# 8. No ORM objects leaked
# ---------------------------------------------------------------------------

class TestNoOrmLeak:
    """Public functions must return plain dicts and lists, never ORM rows."""

    def test_read_returns_plain_dict(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        # Must be a plain dict — not a SQLAlchemy Row or mapped instance
        assert type(result) is dict

    def test_list_returns_plain_dicts(self, db):
        s = _store()
        s.write_experiment(_minimal_record(), db_path=db)
        results = s.list_experiments(db_path=db)
        for item in results:
            assert type(item) is dict

    def test_read_composition_json_is_dict_not_string(self, db):
        s = _store()
        exp_id = s.write_experiment(_minimal_record(), db_path=db)
        result = s.read_experiment(exp_id, db_path=db)
        assert isinstance(result["composition_json"], dict), (
            "composition_json must be deserialised to a dict on read, "
            f"not left as a {type(result['composition_json']).__name__}"
        )

    def test_read_warnings_json_is_list_not_string(self, db):
        s = _store()
        exp_id = s.write_experiment(
            _minimal_record(warnings_json=["w1"]), db_path=db
        )
        result = s.read_experiment(exp_id, db_path=db)
        assert isinstance(result["warnings_json"], list), (
            "warnings_json must be deserialised to a list on read"
        )
