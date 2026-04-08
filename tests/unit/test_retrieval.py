"""
tests/unit/test_retrieval.py
=============================
Unit tests for ``nominal_drift.knowledge.retrieval``.

Each test class uses a fresh temporary SQLite database populated with
a small fixture set of experiment records.  No shared state between
tests — all databases are created/destroyed per test via ``tmp_path``.

Fixture record set (5 records, inserted with controlled timestamps):
  R1  316L  Cr  austenite_FeCrNi  c_sink=12.0  min_cr=11.0  depth=60.0  t=T+1
  R2  316L  Cr  austenite_FeCrNi  c_sink=12.0  min_cr=14.0  depth=None  t=T+2
  R3  304   Cr  austenite_FeCrNi  c_sink=12.0  min_cr=11.5  depth=30.0  t=T+3
  R4  316L  N   austenite_FeCrNi  c_sink=0.001 min_cr=0.015 depth=None  t=T+4
  R5  Inco  Cr  nickel_base       c_sink=20.0  min_cr=18.0  depth=None  t=T+5

Coverage:
  - find_by_alloy: returns expected records, excludes others
  - find_by_element: returns expected records, excludes others
  - find_by_depletion_depth: min/max filtering, null exclusion
  - find_similar_experiments: priority ordering (alloy > element > matrix > c_sink)
  - limit parameter is respected by all functions
  - empty DB returns empty lists cleanly (no crash)
  - returned records are plain dicts (not ORM objects)
  - retrieval does not mutate stored data
  - results are deterministic (same inputs → same output)
  - all records include comparison-ready key fields
"""

from __future__ import annotations

import pytest

from nominal_drift.knowledge.experiment_store import (
    read_experiment,
    write_experiment,
)
from nominal_drift.knowledge.retrieval import (
    find_by_alloy,
    find_by_depletion_depth,
    find_by_element,
    find_similar_experiments,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_COMP_JSON = {"Fe": 68.88, "Cr": 16.5, "Ni": 10.1}
_HT_JSON   = {"steps": [{"step": 1, "T_hold_C": 700, "hold_min": 60}]}


def _rec(
    alloy: str,
    element: str,
    matrix: str,
    c_sink: float,
    min_cr: float,
    depth: float | None,
    timestamp: str,
) -> dict:
    """Return a minimal write_experiment-compatible record dict."""
    return {
        "alloy_designation":        alloy,
        "alloy_matrix":             "austenite" if "nickel" not in matrix else "nickel",
        "composition_json":         _COMP_JSON,
        "ht_schedule_json":         _HT_JSON,
        "element":                  element,
        "matrix":                   matrix,
        "c_bulk_wt_pct":            float(min_cr + 3.0),
        "c_sink_wt_pct":            c_sink,
        "min_concentration_wt_pct": min_cr,
        "depletion_depth_nm":       depth,
        "warnings_json":            [],
        "created_at":               timestamp,
    }


def _populate(db: str) -> dict[str, str]:
    """Write fixture records; return mapping {label: experiment_id}."""
    ids = {}
    ids["R1"] = write_experiment(
        _rec("316L", "Cr", "austenite_FeCrNi", 12.0, 11.0, 60.0, "2026-01-01T00:00:01"),
        db_path=db,
    )
    ids["R2"] = write_experiment(
        _rec("316L", "Cr", "austenite_FeCrNi", 12.0, 14.0, None, "2026-01-01T00:00:02"),
        db_path=db,
    )
    ids["R3"] = write_experiment(
        _rec("304",  "Cr", "austenite_FeCrNi", 12.0, 11.5, 30.0, "2026-01-01T00:00:03"),
        db_path=db,
    )
    ids["R4"] = write_experiment(
        _rec("316L", "N",  "austenite_FeCrNi", 0.001, 0.015, None, "2026-01-01T00:00:04"),
        db_path=db,
    )
    ids["R5"] = write_experiment(
        _rec("Inco", "Cr", "nickel_base",      20.0, 18.0, None, "2026-01-01T00:00:05"),
        db_path=db,
    )
    return ids


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path):
    """Yield path to a populated temporary SQLite database."""
    path = str(tmp_path / "test_retrieval.db")
    _populate(path)
    return path


@pytest.fixture()
def empty_db(tmp_path):
    """Yield path to a completely empty (freshly initialised) database."""
    from nominal_drift.knowledge.experiment_store import init_store
    path = str(tmp_path / "empty.db")
    init_store(path)
    return path


# ===========================================================================
# TestFindByAlloy
# ===========================================================================

class TestFindByAlloy:

    def test_returns_all_matching_alloy_records(self, db):
        results = find_by_alloy("316L", db_path=db)
        alloys = [r["alloy_designation"] for r in results]
        assert all(a == "316L" for a in alloys)
        assert len(results) == 3   # R1, R2, R4

    def test_excludes_non_matching_alloy(self, db):
        results = find_by_alloy("316L", db_path=db)
        returned_ids_elements = {r["element"] for r in results}
        # Inco (R5) and 304 (R3) must NOT appear
        alloys_found = {r["alloy_designation"] for r in results}
        assert "304"  not in alloys_found
        assert "Inco" not in alloys_found

    def test_returns_newest_first(self, db):
        results = find_by_alloy("316L", db_path=db)
        timestamps = [r["created_at"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_returns_only_304_when_queried(self, db):
        results = find_by_alloy("304", db_path=db)
        assert len(results) == 1
        assert results[0]["alloy_designation"] == "304"

    def test_limit_respected(self, db):
        results = find_by_alloy("316L", limit=2, db_path=db)
        assert len(results) <= 2

    def test_no_match_returns_empty_list(self, db):
        results = find_by_alloy("Hastelloy-C276", db_path=db)
        assert results == []

    def test_empty_db_returns_empty_list(self, empty_db):
        results = find_by_alloy("316L", db_path=empty_db)
        assert results == []


# ===========================================================================
# TestFindByElement
# ===========================================================================

class TestFindByElement:

    def test_returns_all_cr_records(self, db):
        results = find_by_element("Cr", db_path=db)
        elements = [r["element"] for r in results]
        assert all(e == "Cr" for e in elements)
        assert len(results) == 4   # R1, R2, R3, R5

    def test_returns_n_records(self, db):
        results = find_by_element("N", db_path=db)
        assert len(results) == 1
        assert results[0]["element"] == "N"

    def test_excludes_other_elements(self, db):
        results = find_by_element("Cr", db_path=db)
        assert all(r["element"] == "Cr" for r in results)

    def test_returns_newest_first(self, db):
        results = find_by_element("Cr", db_path=db)
        timestamps = [r["created_at"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_limit_respected(self, db):
        results = find_by_element("Cr", limit=2, db_path=db)
        assert len(results) == 2

    def test_unknown_element_returns_empty(self, db):
        results = find_by_element("Mo", db_path=db)
        assert results == []

    def test_empty_db_returns_empty_list(self, empty_db):
        results = find_by_element("Cr", db_path=empty_db)
        assert results == []


# ===========================================================================
# TestFindByDepletionDepth
# ===========================================================================

class TestFindByDepletionDepth:

    def test_excludes_null_depth_records(self, db):
        # R2, R4, R5 have depth=None → must never appear
        results = find_by_depletion_depth(db_path=db)
        for r in results:
            assert r["depletion_depth_nm"] is not None

    def test_no_filter_returns_all_with_depth(self, db):
        results = find_by_depletion_depth(db_path=db)
        assert len(results) == 2   # R1 (60 nm) and R3 (30 nm)

    def test_min_depth_filter(self, db):
        results = find_by_depletion_depth(min_depth_nm=50.0, db_path=db)
        assert len(results) == 1
        assert results[0]["depletion_depth_nm"] >= 50.0

    def test_max_depth_filter(self, db):
        results = find_by_depletion_depth(max_depth_nm=35.0, db_path=db)
        assert len(results) == 1
        assert results[0]["depletion_depth_nm"] <= 35.0

    def test_min_and_max_together(self, db):
        results = find_by_depletion_depth(
            min_depth_nm=20.0, max_depth_nm=65.0, db_path=db
        )
        for r in results:
            assert 20.0 <= r["depletion_depth_nm"] <= 65.0

    def test_exclusive_range_returns_empty(self, db):
        # No record has depth between 35 and 55 nm
        results = find_by_depletion_depth(
            min_depth_nm=35.0, max_depth_nm=55.0, db_path=db
        )
        assert results == []

    def test_ordered_shallowest_first(self, db):
        results = find_by_depletion_depth(db_path=db)
        depths = [r["depletion_depth_nm"] for r in results]
        assert depths == sorted(depths)

    def test_limit_respected(self, db):
        results = find_by_depletion_depth(limit=1, db_path=db)
        assert len(results) == 1

    def test_empty_db_returns_empty_list(self, empty_db):
        results = find_by_depletion_depth(db_path=empty_db)
        assert results == []


# ===========================================================================
# TestFindSimilarExperiments
# ===========================================================================

class TestFindSimilarExperiments:

    def test_alloy_match_ranked_first(self, db, tmp_path):
        """Records matching the queried alloy must appear before non-matches."""
        results = find_similar_experiments(
            alloy_designation="316L",
            db_path=db,
        )
        alloys = [r["alloy_designation"] for r in results]
        # All 316L records must appear before non-316L
        i_last_316l = max(i for i, a in enumerate(alloys) if a == "316L")
        i_first_other = min(
            (i for i, a in enumerate(alloys) if a != "316L"),
            default=len(alloys),
        )
        assert i_last_316l < i_first_other

    def test_element_match_ranked_higher_than_non_match(self, db):
        """Within same alloy, exact element match should appear first."""
        results = find_similar_experiments(
            alloy_designation="316L",
            element="Cr",
            db_path=db,
        )
        # R1 and R2 (316L + Cr) must come before R4 (316L + N)
        ids_316l_cr = []
        ids_316l_n  = []
        for r in results:
            if r["alloy_designation"] == "316L" and r["element"] == "Cr":
                ids_316l_cr.append(results.index(r))
            elif r["alloy_designation"] == "316L" and r["element"] == "N":
                ids_316l_n.append(results.index(r))
        if ids_316l_n:
            assert max(ids_316l_cr) < min(ids_316l_n)

    def test_closest_c_sink_ranked_higher(self, db):
        """Record with c_sink nearest to query value should be ranked first."""
        # R1 and R2 share c_sink=12.0; R4 has c_sink=0.001; R5 has c_sink=20.0
        # Query c_sink=12.5 → R1/R2 are closest (diff=0.5) vs R5 (diff=7.5)
        results = find_similar_experiments(c_sink_wt_pct=12.5, db_path=db)
        # The first results should have c_sink close to 12.5
        top = results[0]
        assert abs(top["c_sink_wt_pct"] - 12.5) < abs(
            results[-1]["c_sink_wt_pct"] - 12.5
        )

    def test_no_criteria_returns_all_records(self, db):
        """Calling with all-None criteria returns all records (recency order)."""
        results = find_similar_experiments(db_path=db)
        assert len(results) == 5

    def test_limit_respected(self, db):
        results = find_similar_experiments(limit=2, db_path=db)
        assert len(results) == 2

    def test_empty_db_returns_empty_list(self, empty_db):
        results = find_similar_experiments(
            alloy_designation="316L", db_path=empty_db
        )
        assert results == []

    def test_deterministic_same_query_same_result(self, db):
        r1 = find_similar_experiments(
            alloy_designation="316L", element="Cr", db_path=db
        )
        r2 = find_similar_experiments(
            alloy_designation="316L", element="Cr", db_path=db
        )
        assert [r["experiment_id"] for r in r1] == [r["experiment_id"] for r in r2]


# ===========================================================================
# TestReturnFormat
# ===========================================================================

class TestReturnFormat:

    _COMPARISON_KEYS = {
        "experiment_id",
        "alloy_designation",
        "element",
        "matrix",
        "min_concentration_wt_pct",
        "depletion_depth_nm",
        "created_at",
        "plot_path",
        "animation_path",
    }

    def test_records_are_plain_dicts(self, db):
        results = find_by_alloy("316L", db_path=db)
        for r in results:
            assert isinstance(r, dict), f"Expected dict, got {type(r)}"

    def test_records_contain_comparison_ready_keys(self, db):
        results = find_by_element("Cr", db_path=db)
        for r in results:
            for key in self._COMPARISON_KEYS:
                assert key in r, f"Missing key '{key}' in record"

    def test_retrieval_does_not_mutate_stored_data(self, db, tmp_path):
        """Modifying the returned dict must not affect the stored record."""
        results = find_by_alloy("316L", db_path=db)
        first_id = results[0]["experiment_id"]
        original_alloy = results[0]["alloy_designation"]

        # Mutate the returned dict
        results[0]["alloy_designation"] = "MUTATED"

        # Re-read from store — should be unchanged
        fresh = read_experiment(first_id, db_path=db)
        assert fresh["alloy_designation"] == original_alloy

    def test_all_retrieval_functions_return_list(self, db):
        assert isinstance(find_by_alloy("316L", db_path=db), list)
        assert isinstance(find_by_element("Cr", db_path=db), list)
        assert isinstance(find_by_depletion_depth(db_path=db), list)
        assert isinstance(find_similar_experiments(db_path=db), list)
